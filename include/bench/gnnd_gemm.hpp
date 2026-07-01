// GNND-style batched tiny GEMM microbenchmark.
//   One block per point computes OLD (list x dim) . NEW (list x dim)^T = list x list,
//   then reduces to per-OLD (row) minima and per-NEW (col) minima -> 2*list values/point.
//   Regular core = fp32 hand-tiled; tensor core = fp16-in / fp32-out (WMMA / MFMA).
//   Neighbor indices and dataset values are generated deterministically from hashes,
//   so no host transfer and both kernels see identical inputs.
#pragma once
#include "bench/backend.hpp"
#include "bench/gpu_types.hpp"
#include <vector>
#include <limits>

#if !BENCH_DEVICE
  #ifndef __host__
  #define __host__
  #endif
  #ifndef __device__
  #define __device__
  #endif
  #ifndef __forceinline__
  #define __forceinline__ inline
  #endif
#endif

#ifndef GNND_LIST
#define GNND_LIST 64          // OLD and NEW list length (2p = 2k = 64 at sample rate 1)
#endif
#ifndef GNND_BK
#define GNND_BK 32            // K-tile (dim looped in chunks of GNND_BK)
#endif

namespace bench {

// splitmix64 finalizer: cheap high-quality integer hash (host + device identical).
__host__ __device__ __forceinline__ unsigned long long gnnd_mix(unsigned long long z){
  z += 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}
constexpr unsigned long long GNND_SALT_OLD = 0xA5A5A5A5ULL;
constexpr unsigned long long GNND_SALT_NEW = 0x5A5A5A5AULL;

// Neighbor point index for (point p, slot s, list salt) into [0, npts). Deterministic,
// so both the regular and tensor kernels gather exactly the same rows (fair comparison).
__host__ __device__ __forceinline__ long long gnnd_neighbor(long long p, int s,
                                                            unsigned long long salt, long long npts){
  unsigned long long h = gnnd_mix((((unsigned long long)p) * 0x9E3779B97F4A7C15ULL)
                                  ^ (((unsigned long long)(s + 1)) * 2654435761ULL) ^ salt);
  return (long long)(h % (unsigned long long)npts);
}
// Deterministic dataset value at flat index i, in [-1, 1).
__host__ __device__ __forceinline__ float gnnd_dataval(long long i){
  unsigned long long h = gnnd_mix((unsigned long long)i * 0x9E3779B97F4A7C15ULL + 1ULL);
  return ((float)(h & 0xFFFFFFu) / 8388608.0f) - 1.0f;   // 2^23 -> [-1,1)
}

// Host reference for one point (used by --check): full fp32 dot products + row/col minima.
inline void gnnd_reference_point(long long p, const float* data, long long npts, int dim, int list,
                                 std::vector<float>& oldMin, std::vector<float>& newMin){
  const float INF = std::numeric_limits<float>::infinity();
  oldMin.assign(list, INF); newMin.assign(list, INF);
  std::vector<long long> oi(list), ni(list);
  for (int s = 0; s < list; ++s){
    oi[s] = gnnd_neighbor(p, s, GNND_SALT_OLD, npts);
    ni[s] = gnnd_neighbor(p, s, GNND_SALT_NEW, npts);
  }
  for (int i = 0; i < list; ++i)
    for (int j = 0; j < list; ++j){
      double acc = 0.0;
      for (int k = 0; k < dim; ++k) acc += (double)data[oi[i]*dim + k] * (double)data[ni[j]*dim + k];
      float c = (float)acc;
      if (c < oldMin[i]) oldMin[i] = c;    // per-OLD (row) min over NEW
      if (c < newMin[j]) newMin[j] = c;    // per-NEW (col) min over OLD
    }
}

}  // namespace bench

#if defined(BENCH_BACKEND_CUDA)
  #include "bench/gnnd_gemm.cuh"
#elif defined(BENCH_BACKEND_HIP)
  #include "bench/gnnd_gemm.hiph"
#endif
