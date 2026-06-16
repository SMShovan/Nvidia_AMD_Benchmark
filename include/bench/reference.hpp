// CPU reference implementations (accumulate in double) for verification.
#pragma once
#include <cstddef>

namespace bench {

// out[i] = alpha*A[i] + B[i]   (Phase 0 validation kernel)
template <typename T>
void reference_vadd(const T* A, const T* B, T* out, std::size_t n, double alpha) {
  for (std::size_t i = 0; i < n; ++i)
    out[i] = static_cast<T>(alpha * static_cast<double>(A[i]) + static_cast<double>(B[i]));
}

// Row-major C[MxN] = A[MxK] * B[KxN]   (used as the CPU-backend GEMM "kernel")
template <typename T, typename Acc = double>
void reference_gemm(const T* A, const T* B, T* C, int M, int N, int K) {
  for (int m = 0; m < M; ++m)
    for (int nn = 0; nn < N; ++nn) {
      Acc acc = Acc(0);
      for (int k = 0; k < K; ++k)
        acc += static_cast<Acc>(A[(std::size_t)m * K + k]) *
               static_cast<Acc>(B[(std::size_t)k * N + nn]);
      C[(std::size_t)m * N + nn] = static_cast<T>(acc);
    }
}

// One output entry C[r,c] from float master data (for cheap sampled verification
// of large GEMMs without computing the whole reference).
inline double gemm_ref_entry(const float* A, const float* B, int N, int K, int r, int c) {
  double s = 0.0;
  for (int k = 0; k < K; ++k)
    s += static_cast<double>(A[(std::size_t)r * K + k]) *
         static_cast<double>(B[(std::size_t)k * N + c]);
  return s;
}

}  // namespace bench
