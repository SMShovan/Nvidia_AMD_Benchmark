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
#include "bench/gnnd_gemm.hpp"
#include "bench/gpu_types.hpp"
#include "bench/kernels_spinlock.hpp"
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

// ----- Optimized matrix-unit GEMM (shared/LDS staged + warp/wavefront tiled) -----
inline int run_matrix_opt(const Args& args) {
  const int M = (int)args.M, N = (int)args.N, K = (int)args.K;
  if (M % BENCH_MM_BM || N % BENCH_MM_BN || K % BENCH_MM_BK) {
    std::fprintf(stderr, "matrix_opt requires M%%%d==0, N%%%d==0, K%%%d==0\n",
                 BENCH_MM_BM, BENCH_MM_BN, BENCH_MM_BK);
    return 2;
  }
  const std::size_t aN = (std::size_t)M * K, bN = (std::size_t)K * N, cN = (std::size_t)M * N;
  std::vector<float> mA(aN), mB(bN); fill_master(mA, mB);
  Buffer<half_t> dA(aN), dB(bN);
  Buffer<float>  dC(cN);
  for (std::size_t i = 0; i < aN; ++i) set_in(dA.host()[i], mA[i]);
  for (std::size_t i = 0; i < bN; ++i) set_in(dB.host()[i], mB[i]);
  dA.to_device(); dB.to_device();
  for (int w = 0; w < args.warmup; ++w) launch_gemm_matrix_opt(dA.device(), dB.device(), dC.device(), M, N, K);
  device_sync();
  std::vector<double> kt;
  for (int r = 0; r < args.reps; ++r) { DeviceTimer t; t.start();
    launch_gemm_matrix_opt(dA.device(), dB.device(), dC.device(), M, N, K); kt.push_back(t.stop_ms()); }
  double kernel_ms = median(kt);
  std::vector<double> et;
  for (int r = 0; r < args.reps; ++r) { CpuTimer t; t.start();
    dA.to_device(); dB.to_device(); launch_gemm_matrix_opt(dA.device(), dB.device(), dC.device(), M, N, K);
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
  std::printf("[matrix_opt fp16] %dx%dx%d  kernel %.4f ms  e2e %.4f ms  %.1f GFLOP/s  %s\n",
              M, N, K, kernel_ms, e2e_ms, gflops, args.check ? verified : "");
  write_row("matrix_opt", "fp16", args, kernel_ms, e2e_ms, gflops, 0.0, verified, max_rel);
  return (args.check && std::string(verified) == "no") ? 1 : 0;
}
#endif

// ----- Spinlock microbenchmark: H100 (ITS, deadlock-free) vs MI300A (lock-step) -----
inline const char* SPIN_CSV_HEADER =
    "backend,device,variant,map,threads,locks,work,critsize,maxspin,"
    "kernel_ms,mops_per_s,correct,abandoned,nodes,segs,contention,seed,private";

inline int run_spinlock(const Args& a) {
  const int variant = parse_variant(a.variant);
  const int map = parse_map(a.map);
  const int W = a.work;
  // Node mode (GNND-scale): L is derived from N nodes x S segments, and each update
  // targets a scattered (node,segment) lock with G wavefronts sharing it (contention).
  const bool node_mode = (a.nodes > 0);
  const int S = (a.segs < 1) ? 1 : a.segs;
  const int G = (a.contention < 1) ? 1 : a.contention;
  // Round total threads up to a multiple of the block size so every warp/wavefront is
  // full. That makes the leader counting model (adds warpSize per acquired lock) exact
  // on both 32-lane (NVIDIA) and 64-lane (AMD) hardware: (T/warpSize)*warpSize*W = T*W.
  const int tpb = 256;
  long long Teff = ((a.threads + tpb - 1) / tpb) * tpb;
  if (Teff < tpb) Teff = tpb;
  long long L = node_mode ? (a.nodes * (long long)S) : a.locks;
  if (L < 1) L = 1;

  Buffer<int> lock((std::size_t)L);
  Buffer<u64> counter((std::size_t)L);
  Buffer<u64> abandoned(1), sink(1);

  // Zero the device buffers directly (fast even when L is hundreds of millions, i.e.
  // multi-GB arrays): a host loop here would dominate the run.
  auto reset = [&]() {
    device_memset(counter.device(), 0, (std::size_t)L * sizeof(u64));
    device_memset(lock.device(),    0, (std::size_t)L * sizeof(int));
    device_memset(abandoned.device(), 0, sizeof(u64));
    device_memset(sink.device(),      0, sizeof(u64));
    device_sync();
  };

  auto do_launch = [&]() {
    if (node_mode)
      launch_spinlock_node(variant, lock.device(), counter.device(), abandoned.device(),
                           sink.device(), Teff, W, L, S, G, a.critsize, a.maxspin, a.seed, a.priv);
    else
      launch_spinlock(variant, lock.device(), counter.device(), abandoned.device(),
                      sink.device(), Teff, W, L, map, a.critsize, a.maxspin);
  };

  reset();
  for (int w = 0; w < a.warmup; ++w) do_launch();
  device_sync();

  std::vector<double> kt;
  for (int r = 0; r < a.reps; ++r) {
    reset();
    DeviceTimer t; t.start();
    do_launch();
    kt.push_back(t.stop_ms());
  }
  double kernel_ms = median(kt);

  counter.to_host(); abandoned.to_host();
  u64 sum = 0; for (long long i = 0; i < L; ++i) sum += counter.host()[i];
  u64 expected = (u64)Teff * (u64)W;
  u64 aband = abandoned.host()[0];
  // "correct" = every critical section that should have run did run exactly once and
  // nobody bailed. On MI300A the naive/perwarp variant can livelock-bail, so abandoned>0
  // (or sum<expected) is itself the headline result, not a harness bug.
  bool correct = (sum == expected) && (aband == 0ULL);

  double total_ops = (double)Teff * (double)W;
  double mops = (kernel_ms > 0.0) ? total_ops / (kernel_ms / 1e3) / 1e6 : 0.0;

  if (node_mode)
    std::printf("[spinlock %s node%s] T=%lld nodes=%lld segs=%d L=%lld %s W=%d  "
                "kernel %.4f ms  %.1f Mops/s  %s  abandoned=%llu  (sum=%llu expected=%llu)\n",
                a.variant.c_str(), a.priv ? "/private" : "", Teff, a.nodes, S, L,
                a.priv ? "no-contention" : (a.seed ? "contention=2warp,random" : "contention=2warp"),
                W, kernel_ms, mops,
                correct ? "[OK]" : "[INCOMPLETE]", (unsigned long long)aband,
                (unsigned long long)sum, (unsigned long long)expected);
  else
    std::printf("[spinlock %s/%s] T=%lld L=%lld W=%d crit=%d  kernel %.4f ms  %.1f Mops/s  "
                "%s  abandoned=%llu  (sum=%llu expected=%llu)\n",
                a.variant.c_str(), a.map.c_str(), Teff, L, W, a.critsize, kernel_ms, mops,
                correct ? "[OK]" : "[INCOMPLETE]", (unsigned long long)aband,
                (unsigned long long)sum, (unsigned long long)expected);

  std::ostringstream row;
  row << BENCH_BACKEND_NAME << "," << device_name() << "," << a.variant << "," << a.map << ","
      << Teff << "," << L << "," << W << "," << a.critsize << "," << a.maxspin << ","
      << kernel_ms << "," << mops << "," << (correct ? "yes" : "no") << ","
      << (unsigned long long)aband << ","
      << (node_mode ? a.nodes : 0LL) << "," << (node_mode ? S : 0) << ","
      << (node_mode ? (a.priv ? 0 : G) : 0) << "," << (node_mode ? a.seed : 0ULL) << ","
      << (node_mode ? a.priv : 0);
  append_csv(a.csv, SPIN_CSV_HEADER, row.str());
  // Abandonment is a measured outcome, not a failure: always exit 0 so a sweep can
  // record the deadlock-bail row instead of aborting the run.
  return 0;
}

// ----- GNND-style batched tiny GEMM: OLD . NEW^T per point, regular fp32 vs tensor fp16 -----
inline const char* GNND_GEMM_CSV_HEADER =
    "backend,device,variant,points,dim,list,kernel_ms,gflops,verified,max_abs";

inline int run_gnnd_gemm(const Args& a) {
  const long long npts = a.points;
  const int dim = a.dim, list = a.listsz;
  const bool tensor = (a.variant == "tensor");
  if (list != GNND_LIST) {
    std::fprintf(stderr, "gnnd_gemm built for list=%d (rebuild with -DGNND_LIST=%d)\n", GNND_LIST, list);
    return 2;
  }
  if (dim % GNND_BK) { std::fprintf(stderr, "gnnd_gemm requires dim %% %d == 0\n", GNND_BK); return 2; }
  if (tensor && (dim % 16)) { std::fprintf(stderr, "tensor path requires dim %% 16 == 0\n"); return 2; }
  const std::size_t ndata = (std::size_t)npts * dim;
  const std::size_t nout  = (std::size_t)npts * 2 * list;
  double gflops = 0.0, kernel_ms = 0.0; const char* verified = "skip"; double max_abs = 0.0;

#if BENCH_DEVICE
  Buffer<float>  data(ndata);
  Buffer<half_t> dataH(tensor ? ndata : 1);
  Buffer<float>  out(nout);
  launch_gnnd_fill(data.device(), (long long)ndata); device_sync();
  if (tensor) { launch_gnnd_to_f16(data.device(), dataH.device(), (long long)ndata); device_sync(); }
  auto go = [&](){ launch_gnnd_gemm(tensor, data.device(), dataH.device(), out.device(), npts, dim, list); };
  for (int w = 0; w < a.warmup; ++w) go();
  device_sync();
  std::vector<double> kt;
  for (int r = 0; r < a.reps; ++r) { DeviceTimer t; t.start(); go(); kt.push_back(t.stop_ms()); }
  kernel_ms = median(kt);
  if (a.check) {
    data.to_host(); out.to_host();
    std::vector<float> om, nm; gnnd_reference_point(0, data.host(), npts, dim, list, om, nm);
    for (int r = 0; r < list; ++r) max_abs = std::max(max_abs, (double)std::fabs(out.host()[r]        - om[r]));
    for (int c = 0; c < list; ++c) max_abs = std::max(max_abs, (double)std::fabs(out.host()[list + c] - nm[c]));
    double tol = tensor ? 1.0 : 1e-2;
    verified = (max_abs <= tol) ? "yes" : "no";
  }
#else
  // CPU reference path (serial; for smoke-testing the harness at small --points).
  std::vector<float> data(ndata);
  for (std::size_t i = 0; i < ndata; ++i) data[i] = gnnd_dataval((long long)i);
  std::vector<float> out(nout);
  CpuTimer t; t.start();
  std::vector<float> om, nm;
  for (long long p = 0; p < npts; ++p) {
    gnnd_reference_point(p, data.data(), npts, dim, list, om, nm);
    for (int r = 0; r < list; ++r) out[p*2*list + r]        = om[r];
    for (int c = 0; c < list; ++c) out[p*2*list + list + c] = nm[c];
  }
  kernel_ms = t.stop_ms();
  verified = a.check ? "yes" : "skip";   // CPU path IS the reference
#endif

  gflops = 2.0 * (double)npts * (double)list * (double)list * (double)dim / (kernel_ms / 1e3) / 1e9;
  std::printf("[gnnd_gemm %s] points=%lld dim=%d list=%d  kernel %.3f ms  %.1f GFLOP/s  %s%s\n",
              tensor ? "tensor(fp16)" : "regular(fp32)", npts, dim, list, kernel_ms, gflops,
              a.check ? verified : "", (a.check && max_abs > 0) ? "" : "");
  std::ostringstream row;
  row << BENCH_BACKEND_NAME << "," << device_name() << "," << (tensor ? "tensor" : "regular") << ","
      << npts << "," << dim << "," << list << "," << kernel_ms << "," << gflops << ","
      << verified << "," << max_abs;
  append_csv(a.csv, GNND_GEMM_CSV_HEADER, row.str());
  return (a.check && std::string(verified) == "no") ? 1 : 0;
}

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
  if (args.kernel == "matrix_opt") {
#if BENCH_DEVICE
    return run_matrix_opt(args);
#else
    std::fprintf(stderr, "matrix_opt kernel requires a GPU backend (CUDA/HIP)\n");
    return 2;
#endif
  }
  if (args.kernel == "spinlock") return run_spinlock(args);
  if (args.kernel == "gnnd_gemm") return run_gnnd_gemm(args);
  std::fprintf(stderr, "unknown --kernel '%s' (have: vadd, tiled, matrix, matrix_opt, spinlock, gnnd_gemm)\n",
               args.kernel.c_str());
  return 2;
}

}  // namespace bench
