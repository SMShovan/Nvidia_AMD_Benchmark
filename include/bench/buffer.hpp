// Buffer<T>: the one place the memory model differs across architectures.
//   CUDA : separate host + device allocations, explicit memcpy.
//   HIP  : single hipMalloc on the unified-memory APU; host writes in place,
//          host_ptr == device_ptr, copies are no-ops (advisor's zero-copy way).
//   CPU  : single malloc; copies are no-ops.
#pragma once
#include "bench/backend.hpp"
#include <cstdlib>

namespace bench {

template <typename T>
class Buffer {
 public:
  explicit Buffer(std::size_t n) : n_(n) {
#if defined(BENCH_BACKEND_CUDA)
    h_ = static_cast<T*>(std::malloc(n * sizeof(T)));
    GPU_CHECK(gpuMalloc(&d_, n * sizeof(T)));
#elif defined(BENCH_BACKEND_HIP)
    GPU_CHECK(gpuMalloc(&d_, n * sizeof(T)));  // unified HBM: host-accessible
    h_ = d_;                                   // same pointer, zero copy
#else
    h_ = static_cast<T*>(std::malloc(n * sizeof(T)));
    d_ = h_;
#endif
  }

  ~Buffer() {
#if defined(BENCH_BACKEND_CUDA)
    std::free(h_);
    if (d_) gpuFree(d_);
#elif defined(BENCH_BACKEND_HIP)
    if (d_) gpuFree(d_);
#else
    std::free(h_);
#endif
  }

  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  T* host() { return h_; }
  T* device() { return d_; }
  std::size_t size() const { return n_; }
  std::size_t bytes() const { return n_ * sizeof(T); }

  void to_device() {
#if defined(BENCH_BACKEND_CUDA)
    GPU_CHECK(gpuMemcpy(d_, h_, n_ * sizeof(T), gpuMemcpyHostToDevice));
#endif
  }
  void to_host() {
#if defined(BENCH_BACKEND_CUDA)
    GPU_CHECK(gpuMemcpy(h_, d_, n_ * sizeof(T), gpuMemcpyDeviceToHost));
#endif
  }

 private:
  std::size_t n_ = 0;
  T* h_ = nullptr;
  T* d_ = nullptr;
};

}  // namespace bench
