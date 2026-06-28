// Backend portability shim: maps gpu* names to CUDA, HIP, or CPU(no-op).
// The build defines exactly one of BENCH_BACKEND_{CUDA,HIP,CPU}.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>

#if defined(BENCH_BACKEND_CUDA)
  #include <cuda_runtime.h>
  #define BENCH_DEVICE 1
  #define BENCH_BACKEND_NAME "CUDA"
  using gpuError_t      = cudaError_t;
  using gpuEvent_t      = cudaEvent_t;
  using gpuDeviceProp_t = cudaDeviceProp;
  static const gpuError_t gpuSuccess = cudaSuccess;
  #define gpuMalloc              cudaMalloc
  #define gpuFree                cudaFree
  #define gpuMemcpy              cudaMemcpy
  #define gpuMemset              cudaMemset
  #define gpuMemcpyHostToDevice  cudaMemcpyHostToDevice
  #define gpuMemcpyDeviceToHost  cudaMemcpyDeviceToHost
  #define gpuDeviceSynchronize   cudaDeviceSynchronize
  #define gpuGetLastError        cudaGetLastError
  #define gpuGetErrorString      cudaGetErrorString
  #define gpuEventCreate         cudaEventCreate
  #define gpuEventRecord         cudaEventRecord
  #define gpuEventSynchronize    cudaEventSynchronize
  #define gpuEventElapsedTime    cudaEventElapsedTime
  #define gpuEventDestroy        cudaEventDestroy
  #define gpuGetDeviceProperties cudaGetDeviceProperties
  #define gpuSetDevice           cudaSetDevice
#elif defined(BENCH_BACKEND_HIP)
  #include <hip/hip_runtime.h>
  #define BENCH_DEVICE 1
  #define BENCH_BACKEND_NAME "HIP"
  using gpuError_t      = hipError_t;
  using gpuEvent_t      = hipEvent_t;
  using gpuDeviceProp_t = hipDeviceProp_t;
  static const gpuError_t gpuSuccess = hipSuccess;
  #define gpuMalloc              hipMalloc
  #define gpuFree                hipFree
  #define gpuMemcpy              hipMemcpy
  #define gpuMemset              hipMemset
  #define gpuMemcpyHostToDevice  hipMemcpyHostToDevice
  #define gpuMemcpyDeviceToHost  hipMemcpyDeviceToHost
  #define gpuDeviceSynchronize   hipDeviceSynchronize
  #define gpuGetLastError        hipGetLastError
  #define gpuGetErrorString      hipGetErrorString
  #define gpuEventCreate         hipEventCreate
  #define gpuEventRecord         hipEventRecord
  #define gpuEventSynchronize    hipEventSynchronize
  #define gpuEventElapsedTime    hipEventElapsedTime
  #define gpuEventDestroy        hipEventDestroy
  #define gpuGetDeviceProperties hipGetDeviceProperties
  #define gpuSetDevice           hipSetDevice
#else
  #ifndef BENCH_BACKEND_CPU
  #define BENCH_BACKEND_CPU 1
  #endif
  #define BENCH_DEVICE 0
  #define BENCH_BACKEND_NAME "CPU"
#endif

#if BENCH_DEVICE
  #define GPU_CHECK(call)                                                     \
    do {                                                                      \
      gpuError_t err__ = (call);                                             \
      if (err__ != gpuSuccess) {                                             \
        std::fprintf(stderr, "GPU error %s:%d: %s\n", __FILE__, __LINE__,    \
                     gpuGetErrorString(err__));                             \
        std::exit(1);                                                        \
      }                                                                       \
    } while (0)
#endif

namespace bench {

inline void device_sync() {
#if BENCH_DEVICE
  GPU_CHECK(gpuDeviceSynchronize());
#endif
}

// Zero/set device memory in one call: a real GPU memset on CUDA/HIP, std::memset on
// CPU. Lets the spinlock reset huge (multi-GB) lock/counter arrays cheaply instead of
// looping on the host.
inline void device_memset(void* p, int v, std::size_t bytes) {
#if BENCH_DEVICE
  GPU_CHECK(gpuMemset(p, v, bytes));
#else
  std::memset(p, v, bytes);
#endif
}

inline void check_launch() {
#if BENCH_DEVICE
  gpuError_t e = gpuGetLastError();
  if (e != gpuSuccess) {
    std::fprintf(stderr, "kernel launch error: %s\n", gpuGetErrorString(e));
    std::exit(1);
  }
#endif
}

// Short human-readable device name (gfx942 / sm_90 / host CPU).
inline const char* device_name() {
#if BENCH_DEVICE
  static char buf[256];
  gpuDeviceProp_t p{};
  if (gpuGetDeviceProperties(&p, 0) == gpuSuccess) {
  #if defined(BENCH_BACKEND_HIP)
    std::snprintf(buf, sizeof(buf), "%s (%s)", p.name, p.gcnArchName);
  #else
    std::snprintf(buf, sizeof(buf), "%s (sm_%d%d)", p.name, p.major, p.minor);
  #endif
    return buf;
  }
  return "unknown-device";
#else
  return "host-cpu";
#endif
}

}  // namespace bench
