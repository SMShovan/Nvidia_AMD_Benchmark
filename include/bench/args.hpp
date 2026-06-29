// Minimal CLI parser (GEMM + spinlock benchmarks).
#pragma once
#include <cstdlib>
#include <cstring>
#include <string>
#include "bench/config.hpp"

namespace bench {

struct Args {
  // --- GEMM ---
  long long M = 4096, N = 4096, K = 4096;
  DType dtype = DType::f32;
  // --- common ---
  std::string kernel = "vadd";    // vadd | tiled | matrix | spinlock
  int reps = 20;
  int warmup = 5;
  bool check = false;
  std::string csv = "";
  double peak_gflops = 0.0;
  // --- spinlock ---
  std::string variant = "atomic"; // naive | leader | atomic
  std::string map = "striped";    // striped | perwarp | perthread
  long long threads = 65536;      // total threads T
  long long locks = 1024;         // number of locks L
  int work = 64;                  // ops per thread W
  int critsize = 0;               // dummy work inside the critical section
  int maxspin = 100000;           // bounded spin cap (safety; detects deadlock)
  // --- spinlock node mode (GNND-scale): when nodes>0, L is derived as nodes*segs and
  //     each update targets a (node,segment) lock, scattered across the whole array, with
  //     `contention` wavefronts concurrently hitting the same lock. ---
  long long nodes = 0;            // graph nodes N (0 => flat --locks mode)
  int segs = 4;                   // lock segments per node S (k/32 in GNND, e.g. 4)
  int contention = 1;             // wavefronts (warps) sharing one lock G (>=1)
};

inline void print_usage(const char* prog) {
  std::printf(
      "Usage: %s --kernel vadd|tiled|matrix|spinlock [options]\n"
      "  GEMM:     --M --N --K --dtype fp32|fp16|bf16\n"
      "  spinlock: --variant naive|leader|atomic  --map striped|perwarp|perthread\n"
      "            --threads T --locks L --work W --critsize C --maxspin S\n"
      "  spinlock node mode (GNND-scale): --nodes N --segs S --contention G\n"
      "            (L = N*S; G wavefronts share each lock; --locks ignored when --nodes>0)\n"
      "  common:   --reps --warmup --check --csv path --peak gflops\n", prog);
}

inline Args parse_args(int argc, char** argv) {
  Args a;
  auto need = [&](int i){ if (i + 1 >= argc){ print_usage(argv[0]); std::exit(2);} };
  for (int i = 1; i < argc; ++i) {
    const char* s = argv[i];
    if      (!std::strcmp(s,"--M"))       { need(i); a.M = std::atoll(argv[++i]); }
    else if (!std::strcmp(s,"--N"))       { need(i); a.N = std::atoll(argv[++i]); }
    else if (!std::strcmp(s,"--K"))       { need(i); a.K = std::atoll(argv[++i]); }
    else if (!std::strcmp(s,"--dtype"))   { need(i); a.dtype = parse_dtype(argv[++i]); }
    else if (!std::strcmp(s,"--kernel"))  { need(i); a.kernel = argv[++i]; }
    else if (!std::strcmp(s,"--reps"))    { need(i); a.reps = std::atoi(argv[++i]); }
    else if (!std::strcmp(s,"--warmup"))  { need(i); a.warmup = std::atoi(argv[++i]); }
    else if (!std::strcmp(s,"--check"))   { a.check = true; }
    else if (!std::strcmp(s,"--csv"))     { need(i); a.csv = argv[++i]; }
    else if (!std::strcmp(s,"--peak"))    { need(i); a.peak_gflops = std::atof(argv[++i]); }
    else if (!std::strcmp(s,"--variant")) { need(i); a.variant = argv[++i]; }
    else if (!std::strcmp(s,"--map"))     { need(i); a.map = argv[++i]; }
    else if (!std::strcmp(s,"--threads")) { need(i); a.threads = std::atoll(argv[++i]); }
    else if (!std::strcmp(s,"--locks"))   { need(i); a.locks = std::atoll(argv[++i]); }
    else if (!std::strcmp(s,"--work"))    { need(i); a.work = std::atoi(argv[++i]); }
    else if (!std::strcmp(s,"--critsize")){ need(i); a.critsize = std::atoi(argv[++i]); }
    else if (!std::strcmp(s,"--maxspin")) { need(i); a.maxspin = std::atoi(argv[++i]); }
    else if (!std::strcmp(s,"--nodes"))   { need(i); a.nodes = std::atoll(argv[++i]); }
    else if (!std::strcmp(s,"--segs"))    { need(i); a.segs = std::atoi(argv[++i]); }
    else if (!std::strcmp(s,"--contention")){ need(i); a.contention = std::atoi(argv[++i]); }
    else if (!std::strcmp(s,"-h")||!std::strcmp(s,"--help")) { print_usage(argv[0]); std::exit(0); }
    else { std::fprintf(stderr,"unknown arg: %s\n", s); print_usage(argv[0]); std::exit(2); }
  }
  return a;
}

}  // namespace bench
