// Regular-core tiled GEMM (row-major C[MxN] = A[MxK]*B[KxN]).
// Shared-memory tiling + register blocking. Accumulate in float for both
// FP32 and FP16 inputs. Tile sizes are compile-time (override with -DBENCH_BM=..).
// The 256-thread block is a multiple of both warp(32) and wavefront(64), so the
// same kernel maps cleanly onto NVIDIA and AMD; tune the tile sizes per arch later.
#pragma once
#include "bench/backend.hpp"
#include "bench/gpu_types.hpp"
#include "bench/reference.hpp"

#ifndef BENCH_BM
#define BENCH_BM 128
#endif
#ifndef BENCH_BN
#define BENCH_BN 128
#endif
#ifndef BENCH_BK
#define BENCH_BK 8
#endif
#ifndef BENCH_TM
#define BENCH_TM 8
#endif
#ifndef BENCH_TN
#define BENCH_TN 8
#endif

namespace bench {

#if BENCH_DEVICE
__device__ inline float dload(float x) { return x; }
__device__ inline void  dstore(float* p, float v) { *p = v; }
template <typename T> __device__ inline T dzero();
template <> __device__ inline float dzero<float>() { return 0.f; }
#if defined(BENCH_BACKEND_CUDA) || defined(BENCH_BACKEND_HIP)
__device__ inline float dload(half_t x) { return __half2float(x); }
__device__ inline void  dstore(half_t* p, float v) { *p = __float2half(v); }
template <> __device__ inline half_t dzero<half_t>() { return __float2half(0.f); }
#endif

template <typename TIn, int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_tiled_kernel(const TIn* A, const TIn* B, TIn* C,
                                  int M, int N, int K) {
  __shared__ TIn As[BM * BK];
  __shared__ TIn Bs[BK * BN];
  const int txN = BN / TN, tyM = BM / TM;        // threads in x, y
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int tid = ty * txN + tx;
  const int nThreads = txN * tyM;
  const int blockRow = blockIdx.y, blockCol = blockIdx.x;

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

  for (int k0 = 0; k0 < K; k0 += BK) {
    for (int idx = tid; idx < BM * BK; idx += nThreads) {
      int r = idx / BK, c = idx % BK;
      int gr = blockRow * BM + r, gc = k0 + c;
      As[r * BK + c] = (gr < M && gc < K) ? A[(std::size_t)gr * K + gc] : dzero<TIn>();
    }
    for (int idx = tid; idx < BK * BN; idx += nThreads) {
      int r = idx / BN, c = idx % BN;
      int gr = k0 + r, gc = blockCol * BN + c;
      Bs[r * BN + c] = (gr < K && gc < N) ? B[(std::size_t)gr * N + gc] : dzero<TIn>();
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      float aReg[TM], bReg[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) aReg[i] = dload(As[(ty * TM + i) * BK + kk]);
#pragma unroll
      for (int j = 0; j < TN; ++j) bReg[j] = dload(Bs[kk * BN + (tx * TN + j)]);
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] += aReg[i] * bReg[j];
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      int r = blockRow * BM + ty * TM + i, c = blockCol * BN + tx * TN + j;
      if (r < M && c < N) dstore(&C[(std::size_t)r * N + c], acc[i][j]);
    }
}

template <typename TIn>
void launch_gemm_tiled(const TIn* A, const TIn* B, TIn* C, int M, int N, int K) {
  constexpr int BM = BENCH_BM, BN = BENCH_BN, BK = BENCH_BK, TM = BENCH_TM, TN = BENCH_TN;
  dim3 block(BN / TN, BM / TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  gemm_tiled_kernel<TIn, BM, BN, BK, TM, TN><<<grid, block>>>(A, B, C, M, N, K);
  check_launch();
}
#else
// CPU backend: the reference GEMM IS the kernel (fp32 only).
template <typename TIn>
void launch_gemm_tiled(const TIn* A, const TIn* B, TIn* C, int M, int N, int K) {
  reference_gemm<TIn, double>(A, B, C, M, N, K);
}
#endif

}  // namespace bench
