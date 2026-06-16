// Selects the matrix-unit GEMM for the active backend. Both backends expose
//   void launch_gemm_matrix(const half_t* A, const half_t* B, float* C, M,N,K)
// (fp16 inputs, fp32 output). No matrix path on the CPU backend.
#pragma once
#include "bench/backend.hpp"
#include "bench/gpu_types.hpp"
#if defined(BENCH_BACKEND_CUDA)
  #include "bench/gemm_wmma.cuh"
#elif defined(BENCH_BACKEND_HIP)
  #include "bench/gemm_mfma.hiph"
#endif
