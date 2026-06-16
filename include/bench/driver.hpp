// Driver: dispatches on --kernel (vadd | tiled | matrix) and --dtype (fp32 | fp16).
// Reports kernel-only and end-to-end time, verifies, and appends a CSV row.
#pragma once
#include <cstdio>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "bench/args.hpp"
#include "bench/backend.hpp"
#include "bench/buffer.hpp"
#include "bench/config.hpp"
#include "bench/gemm_matrix.hpp"
#include "bench/gemm_tiled.hpp"
#include "bench/gpu_types.hpp"
#include "bench/kernels_validate.hpp"
#include "bench/metrics.hpp"
#include "bench/reference.hpp"
#include "bench/timer.hpp"
#include "bench/verify.hpp"

namespace bench {

inline const char* CSV_HEADER =
    "backend,device,kernel,dtype,M,N,K,kernel_ms,e2e_ms,gflops,gbps,verified,max_rel";

inline void write_row(const char* kernel, const char* dtype, const Args& a,
                      double kernel_ms, double e2e_ms, double gflops, double gbps,
                      const char* verified, double max_rel) {
  std::ostringstream r;
  r << BENCH_BACKEND_NAME << "," << device_name() << "," << kernel << "," << dtype << ","
    << a.M << "," << a.N << "," << a.K << "," << kernel_ms << "," << e2e_ms << ","
    << gflops << "," << gbps << "," << verified << "," << max_rel;
  append_csv(a.csv, CSV_HEADER, r.str());
}

// Sampled verification of a GEMM: compares S random C entries to a double reference.
template <typename GetC>
inline bool verify_gemm(GetC getC, const std::vector<float>& mA,
                        const std::vector<float>& mB, int M, int N, int K,
                        double tol, double& max_rel) {
  const int S = (int)(((std::size_t)M * N < 1024) ? (std::size_t)M * N : 1024);
  std::uint64_t rng = 0x9e3779b97f4a7c15ull;
  max_rel = 0.0;
  for (int s = 0; s < S; ++s) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    int r = (int)(rng % (std::uint64_t)M);
    int c = (int)((rng / M) % (std::uint64_t)N);
    double ref = gemm_ref_entry(mA.data(), mB.data(), N, K, r, c);
    double got = getC((std::size_t)r * N + c);
    double denom = (ref < 0 ? -ref : ref); if (denom < 1e-9) denom = 1.0;
    double rel = (got - ref); if (rel < 0) rel = -rel; rel /= denom;
    if (rel > max_rel) max_rel = rel;
  }
  return max_rel <= tol;
}

inline void fill_master(std::vector<float>& mA, std::vector<float>& mB) {
  for (std::size_t i = 0; i < mA.size(); ++i) mA[i] = (float)((i % 17) * 0.03125);
  for (std::size_t i = 0; i < mB.size(); ++i) mB[i] = (float)((i % 13) * 0.0625);
}

// ----- Phase 0 validation kernel: out = alpha*A + B -----
inline int run_vadd(const Args& args) {
  const std::size_t n = (std::size_t)args.M * (std::size_t)args.N;
  const double alpha = 2.0;
  using T = float;
  Buffer<T> A(n), B(n), C(n);
  for (std::size_t i = 0; i < n; ++i) { A.host()[i] = (T)((i % 97) * 0.01 + 1.0);
                                        B.host()[i] = (T)((i % 89) * 0.02); }
  A.to_device(); B.to_device();
  for (int w = 0; w < args.warmup; ++w) launch_vadd<T>(A.device(), B.device(), C.device(), n, alpha);
  device_sync();
  std::vector<double> kt;
  for (int r = 0; r < args.reps; ++r) { DeviceTimer t; t.start();
    launch_vadd<T>(A.device(), B.device(), C.device(), n, alpha); kt.push_back(t.stop_ms()); }
  double kernel_ms = median(kt);
  std::vector<double> et;
  for (int r = 0; r < args.reps; ++r) { CpuTimer t; t.start();
    A.to_device(); B.to_device(); launch_vadd<T>(A.device(), B.device(), C.device(), n, alpha);
    device_sync(); C.to_host(); et.push_back(t.stop_ms()); }
  double e2e_ms = median(et);
  C.to_host();
  bool ok = true; double max_rel = 0.0;
  if (args.check) { std::vector<T> ref(n); reference_vadd<T>(A.host(), B.host(), ref.data(), n, alpha);
    auto v = verify<T>(C.host(), ref.data(), n, 1e-4); ok = v.ok; max_rel = v.max_rel; }
  double gbps = 3.0 * (double)n * sizeof(T) / 1e9 / (kernel_ms / 1e3);
  std::printf("[vadd] kernel %.4f ms  e2e %.4f ms  ~%.1f GB/s%s\n", kernel_ms, e2e_ms, gbps,
              args.check ? (ok ? "  [OK]" : "  [FAIL]") : "");
  write_row("vadd", "fp32", args, kernel_ms, e2e_ms, 0.0, gbps,
            args.check ? (ok ? "yes" : "no") : "skip", max_rel);
  return (args.check && !ok) ? 1 : 0;
}

// ----- Phase 1: regular-core tiled GEMM (fp32 or fp16-in/fp32-acc) -----
template <typename TIn>
int run_gemm(const Args& args) {
  const int M = (int)args.M, N = (int)args.N, K = (int)args.K;
  const std::size_t aN = (std::size_t)M * K, bN = (std::size_t)K * N, cN = (std::size_t)M * N;
  std::vector<float> mA(aN), mB(bN); fill_master(mA, mB);
  Buffer<TIn> dA(aN), dB(bN), dC(cN);
  for (std::size_t i = 0; i < aN; ++i) set_in(dA.host()[i], mA[i]);
  for (std::size_t i = 0; i < bN; ++i) set_in(dB.host()[i], mB[i]);
  dA.to_device(); dB.to_device();
  for (int w = 0; w < args.warmup; ++w) launch_gemm_tiled<TIn>(dA.device(), dB.device(), dC.device(), M, N, K);
  device_sync();
  std::vector<double> kt;
  for (int r = 0; r < args.reps; ++r) { DeviceTimer t; t.start();
    launch_gemm_tiled<TIn>(dA.device(), dB.device(), dC.device(), M, N, K); kt.push_back(t.stop_ms()); }
  double kernel_ms = median(kt);
  std::vector<double> et;
  for (int r = 0; r < args.reps; ++r) { CpuTimer t; t.start();
    dA.to_device(); dB.to_device(); launch_gemm_tiled<TIn>(dA.device(), dB.device(), dC.device(), M, N, K);
    device_sync(); dC.to_host(); et.push_back(t.stop_ms()); }
  double e2e_ms = median(et);
  dC.to_host();
  const char* verified = "skip"; double max_rel = 0.0;
  if (args.check) {
    double tol = (args.dtype == DType::f32) ? 5e-3 : 5e-2;
    bool ok = verify_gemm([&](std::size_t idx){ return (double)get_out(dC.host()[idx]); },
                          mA, mB, M, N, K, tol, max_rel);
    verified = ok ? "yes" : "no";
  }
  double gflops = 2.0 * (double)M * (double)N * (double)K / (kernel_ms / 1e3) / 1e9;
  std::printf("[tiled %s] %dx%dx%d  kernel %.4f ms  e2e %.4f ms  %.1f GFLOP/s  %s\n",
              dtype_name(args.dtype), M, N, K, kernel_ms, e2e_ms, gflops,
              args.check ? verified : "");
  write_row("tiled", dtype_name(args.dtype), args, kernel_ms, e2e_ms, gflops, 0.0, verified, max_rel);
  return (args.check && std::string(verified) == "no") ? 1 : 0;
}

#if BENCH_DEVICE
// ----- Phase 2: matrix-unit GEMM (Tensor/Matrix cores), fp16-in / fp32-out -----
inline int run_matrix(const Args& args) {
  const int M = (int)args.M, N = (int)args.N, K = (int)args.K;
  if (M % 16 || N % 16 || K % 16) {
    std::fprintf(stderr, "matrix kernel requires M, N, K multiples of 16\n");
    return 2;
  }
  const std::size_t aN = (std::size_t)M * K, bN = (std::size_t)K * N, cN = (std::size_t)M * N;
  std::vector<float> mA(aN), mB(bN); fill_master(mA, mB);
  Buffer<half_t> dA(aN), dB(bN);
  Buffer<float>  dC(cN);
  for (std::size_t i = 0; i < aN; ++i) set_in(dA.host()[i], mA[i]);
  for (std::size_t i = 0; i < bN; ++i) set_in(dB.host()[i], mB[i]);
  dA.to_device(); dB.to_device();
  for (int w = 0; w < args.warmup; ++w) launch_gemm_matrix(dA.device(), dB.device(), dC.device(), M, N, K);
  device_sync();
  std::vector<double> kt;
  for (int r = 0; r < args.reps; ++r) { DeviceTimer t; t.start();
    launch_gemm_matrix(dA.device(), dB.device(), dC.device(), M, N, K); kt.push_back(t.stop_ms()); }
  double kernel_ms = median(kt);
  std::vector<double> et;
  for (int r = 0; r < args.reps; ++r) { CpuTimer t; t.start();
    dA.to_device(); dB.to_device(); launch_gemm_matrix(dA.device(), dB.device(), dC.device(), M, N, K);
    device_sync(); dC.to_host(); et.push_back(t.stop_ms()); }
  double e2e_ms = median(et);
  dC.to_host();
  const char* verified = "skip"; double max_rel = 0.0;
  if (args.check) {
    bool ok = verify_gemm([&](std::size_t idx){ return (double)dC.host()[idx]; },
                          mA, mB, M, N, K, 5e-2, max_rel);
    verified = ok ? "yes" : "no";
  }
  double gflops = 2.0 * (double)M * (double)N * (double)K / (kernel_ms / 1e3) / 1e9;
  std::printf("[matrix fp16] %dx%dx%d  kernel %.4f ms  e2e %.4f ms  %.1f GFLOP/s  %s\n",
              M, N, K, kernel_ms, e2e_ms, gflops, args.check ? verified : "");
  write_row("matrix", "fp16", args, kernel_ms, e2e_ms, gflops, 0.0, verified, max_rel);
  return (args.check && std::string(verified) == "no") ? 1 : 0;
}
#endif

inline int run(int argc, char** argv) {
  Args args = parse_args(argc, argv);
  std::printf("backend=%s  device=%s\n", BENCH_BACKEND_NAME, device_name());
  if (args.kernel == "vadd") return run_vadd(args);
  if (args.kernel == "tiled") {
    if (args.dtype == DType::f32) return run_gemm<float>(args);
#if BENCH_DEVICE
    if (args.dtype == DType::f16) return run_gemm<half_t>(args);
#endif
    std::fprintf(stderr, "dtype '%s' not supported on backend %s for 'tiled'\n",
                 dtype_name(args.dtype), BENCH_BACKEND_NAME);
    return 2;
  }
  if (args.kernel == "matrix") {
#if BENCH_DEVICE
    return run_matrix(args);
#else
    std::fprintf(stderr, "matrix kernel requires a GPU backend (CUDA/HIP)\n");
    return 2;
#endif
  }
  std::fprintf(stderr, "unknown --kernel '%s' (have: vadd, tiled, matrix)\n", args.kernel.c_str());
  return 2;
}

}  // namespace bench
