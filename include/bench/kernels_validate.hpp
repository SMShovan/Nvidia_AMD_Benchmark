// Phase 0 validation kernel: out = alpha*A + B.
// Triple-chevron launch works under both nvcc and hipcc; CPU path is a loop.
#pragma once
#include <cstddef>
#include "bench/backend.hpp"

namespace bench {

#if BENCH_DEVICE
template <typename T>
__global__ void k_vadd(const T* A, const T* B, T* C, std::size_t n, double alpha) {
  std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) C[i] = static_cast<T>(alpha * static_cast<double>(A[i]) +
                                   static_cast<double>(B[i]));
}

template <typename T>
void launch_vadd(const T* A, const T* B, T* C, std::size_t n, double alpha) {
  const int block = 256;
  const std::size_t grid = (n + block - 1) / block;
  k_vadd<T><<<dim3(static_cast<unsigned>(grid)), dim3(block)>>>(A, B, C, n, alpha);
  check_launch();
}
#else
template <typename T>
void launch_vadd(const T* A, const T* B, T* C, std::size_t n, double alpha) {
  for (std::size_t i = 0; i < n; ++i)
    C[i] = static_cast<T>(alpha * static_cast<double>(A[i]) +
                          static_cast<double>(B[i]));
}
#endif

}  // namespace bench
