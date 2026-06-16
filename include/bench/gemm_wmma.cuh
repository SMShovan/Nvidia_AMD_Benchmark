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
}  // namespace bench
