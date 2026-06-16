// Data-type helpers. fp16 is device-only (CUDA/HIP); CPU backend is fp32-only.
#pragma once
#include "bench/backend.hpp"
#if BENCH_DEVICE
  #if defined(BENCH_BACKEND_CUDA)
    #include <cuda_fp16.h>
  #elif defined(BENCH_BACKEND_HIP)
    #include <hip/hip_fp16.h>
  #endif
#endif

namespace bench {
#if BENCH_DEVICE
using half_t = __half;
#endif

// host-side converters: master data is float; device buffers are TIn
inline void  set_in(float& dst, float v) { dst = v; }
inline float get_out(float v)            { return v; }
#if BENCH_DEVICE
inline void  set_in(half_t& dst, float v) { dst = __float2half(v); }
inline float get_out(half_t v)            { return __half2float(v); }
#endif
}  // namespace bench
