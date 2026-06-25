// NVIDIA Tensor-Core GEMM via the nvcuda::wmma API (16x16x16, fp16-in/fp32-out).
// One warp computes one 16x16 output tile; a 256-thread block packs 8 warps along N.
// Requires M, N, K multiples of 16. Row-major.
#pragma once
#include <mma.h>
#include "bench/backend.hpp"
#include "bench/gpu_types.hpp"

namespace bench {
using namespace nvcuda;

__global__ void gemm_wmma_kernel(const half* A, const half* B, float* C,
                                 int M, int N, int K) {
  const int WS = 32;
  const int wavesPerBlock = blockDim.x / WS;
  const int warp = threadIdx.x / WS;
  const int row = blockIdx.y * 16;
  const int col = (blockIdx.x * wavesPerBlock + warp) * 16;
  if (row >= M || col >= N) return;

  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (int k0 = 0; k0 < K; k0 += 16) {
    wmma::load_matrix_sync(a_frag, A + (std::size_t)row * K + k0, K);  // lda=K
    wmma::load_matrix_sync(b_frag, B + (std::size_t)k0 * N + col, N);  // ldb=N
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }
  wmma::store_matrix_sync(C + (std::size_t)row * N + col, c_frag, N, wmma::mem_row_major);
}

inline void launch_gemm_matrix(const half_t* A, const half_t* B, float* C,
                               int M, int N, int K) {
  const int WS = 32, wavesPerBlock = 8, block = WS * wavesPerBlock;  // 256 threads
  dim3 grid((N + wavesPerBlock * 16 - 1) / (wavesPerBlock * 16), (M + 15) / 16);
  gemm_wmma_kernel<<<grid, dim3(block)>>>(reinterpret_cast<const half*>(A),
                                          reinterpret_cast<const half*>(B), C, M, N, K);
  check_launch();
}

// ===================================================================
//  Optimized WMMA: shared-memory staged + warp-tiled, bank-conflict padded.
//  Brings the Tensor-Core kernel to the SAME optimization level as the
//  regular-core tiled kernel (block tile in shared memory + register/warp
//  tiling). Block computes BM x BN; each warp owns a WM x WN sub-tile made
//  of (WM/16)x(WN/16) accumulator fragments held in registers.
//  Requires M,N,K multiples of BM,BN,BK. Tile sizes overridable.
// ===================================================================
#ifndef BENCH_MM_BM
#define BENCH_MM_BM 64
#endif
#ifndef BENCH_MM_BN
#define BENCH_MM_BN 64
#endif
#ifndef BENCH_MM_BK
#define BENCH_MM_BK 32
#endif
#ifndef BENCH_MM_WM
#define BENCH_MM_WM 32
#endif
#ifndef BENCH_MM_WN
#define BENCH_MM_WN 32
#endif
// Pad the shared-memory leading dimension to break LDS/shared bank conflicts
// (and keep each fragment row 16-byte aligned: BK+8 and BN+8 are multiples of 8).
#ifndef BENCH_MM_PAD
#define BENCH_MM_PAD 8
#endif

__global__ void gemm_wmma_opt_kernel(const half* A, const half* B, float* C,
                                     int M, int N, int K) {
  constexpr int BM = BENCH_MM_BM, BN = BENCH_MM_BN, BK = BENCH_MM_BK;
  constexpr int WM = BENCH_MM_WM, WN = BENCH_MM_WN;
  constexpr int WMITER = WM / 16, WNITER = WN / 16;     // 16x16 frags per warp
  constexpr int ldA = BK + BENCH_MM_PAD, ldB = BN + BENCH_MM_PAD;
  __shared__ half As[BM * ldA];   // A block tile, row-major [BM][ldA]
  __shared__ half Bs[BK * ldB];   // B block tile, row-major [BK][ldB]

  const int warpsN  = BN / WN;
  const int tid     = threadIdx.x;
  const int nThr    = blockDim.x;
  const int warpId  = tid / 32;
  const int warpRow = warpId / warpsN;
  const int warpCol = warpId % warpsN;
  const int blockRow = blockIdx.y, blockCol = blockIdx.x;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[WMITER][WNITER];
  for (int i = 0; i < WMITER; ++i)
    for (int j = 0; j < WNITER; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

  for (int k0 = 0; k0 < K; k0 += BK) {
    for (int idx = tid; idx < BM * BK; idx += nThr) {     // stage A tile
      int r = idx / BK, c = idx % BK;
      As[r * ldA + c] = A[(std::size_t)(blockRow * BM + r) * K + (k0 + c)];
    }
    for (int idx = tid; idx < BK * BN; idx += nThr) {     // stage B tile
      int r = idx / BN, c = idx % BN;
      Bs[r * ldB + c] = B[(std::size_t)(k0 + r) * N + (blockCol * BN + c)];
    }
    __syncthreads();
    for (int ks = 0; ks < BK; ks += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[WMITER];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[WNITER];
      for (int i = 0; i < WMITER; ++i)
        wmma::load_matrix_sync(a_frag[i], &As[(warpRow * WM + i * 16) * ldA + ks], ldA);
      for (int j = 0; j < WNITER; ++j)
        wmma::load_matrix_sync(b_frag[j], &Bs[ks * ldB + (warpCol * WN + j * 16)], ldB);
      for (int i = 0; i < WMITER; ++i)
        for (int j = 0; j < WNITER; ++j)
          wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
    }
    __syncthreads();
  }
  for (int i = 0; i < WMITER; ++i)
    for (int j = 0; j < WNITER; ++j) {
      int row = blockRow * BM + warpRow * WM + i * 16;
      int col = blockCol * BN + warpCol * WN + j * 16;
      wmma::store_matrix_sync(&C[(std::size_t)row * N + col], acc[i][j], N, wmma::mem_row_major);
    }
}

inline void launch_gemm_matrix_opt(const half_t* A, const half_t* B, float* C,
                                   int M, int N, int K) {
  constexpr int BM = BENCH_MM_BM, BN = BENCH_MM_BN, WM = BENCH_MM_WM, WN = BENCH_MM_WN;
  const int warps = (BM / WM) * (BN / WN);
  dim3 block(warps * 32);
  dim3 grid(N / BN, M / BM);
  gemm_wmma_opt_kernel<<<grid, block>>>(reinterpret_cast<const half*>(A),
                                        reinterpret_cast<const half*>(B), C, M, N, K);
  check_launch();
}
}  // namespace bench
