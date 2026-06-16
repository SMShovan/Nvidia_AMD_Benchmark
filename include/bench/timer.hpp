// Timers: CPU wall-clock and GPU-event (falls back to CPU on the CPU backend).
#pragma once
#include "bench/backend.hpp"
#include <chrono>

namespace bench {

struct CpuTimer {
  std::chrono::high_resolution_clock::time_point t0;
  void start() { t0 = std::chrono::high_resolution_clock::now(); }
  double stop_ms() {
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
  }
};

#if BENCH_DEVICE
struct DeviceTimer {
  gpuEvent_t a_, b_;
  DeviceTimer() { GPU_CHECK(gpuEventCreate(&a_)); GPU_CHECK(gpuEventCreate(&b_)); }
  ~DeviceTimer() { gpuEventDestroy(a_); gpuEventDestroy(b_); }
  void start() { GPU_CHECK(gpuEventRecord(a_, 0)); }
  double stop_ms() {
    GPU_CHECK(gpuEventRecord(b_, 0));
    GPU_CHECK(gpuEventSynchronize(b_));
    float ms = 0.f;
    GPU_CHECK(gpuEventElapsedTime(&ms, a_, b_));
    return static_cast<double>(ms);
  }
};
#else
using DeviceTimer = CpuTimer;
#endif

}  // namespace bench
