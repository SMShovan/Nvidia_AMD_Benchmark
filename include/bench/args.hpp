// Minimal CLI parser.
#pragma once
#include <cstdlib>
#include <cstring>
#include <string>
#include "bench/config.hpp"

namespace bench {

struct Args {
  long long M = 4096, N = 4096, K = 4096;  // Phase 0 uses M*N elements; K used in GEMM phases
  DType dtype = DType::f32;
  std::string kernel = "vadd";   // Phase 0 validation kernel
  int reps = 20;
  int warmup = 5;
  bool check = false;
  std::string csv = "";
  double peak_gflops = 0.0;      // optional, for %-of-peak in later phases
};

inline void print_usage(const char* prog) {
  std::printf(
      "Usage: %s [--M n] [--N n] [--K n] [--dtype fp32|fp16|bf16]\n"
      "          [--kernel vadd|tiled|matrix] [--reps n] [--warmup n]\n"
      "          [--check] [--csv path] [--peak gflops]\n",
      prog);
}

inline Args parse_args(int argc, char** argv) {
  Args a;
  auto need = [&](int i) { if (i + 1 >= argc) { print_usage(argv[0]); std::exit(2); } };
  for (int i = 1; i < argc; ++i) {
    const char* s = argv[i];
    if      (!std::strcmp(s, "--M"))     { need(i); a.M = std::atoll(argv[++i]); }
    else if (!std::strcmp(s, "--N"))     { need(i); a.N = std::atoll(argv[++i]); }
    else if (!std::strcmp(s, "--K"))     { need(i); a.K = std::atoll(argv[++i]); }
    else if (!std::strcmp(s, "--dtype")) { need(i); a.dtype = parse_dtype(argv[++i]); }
    else if (!std::strcmp(s, "--kernel")){ need(i); a.kernel = argv[++i]; }
    else if (!std::strcmp(s, "--reps"))  { need(i); a.reps = std::atoi(argv[++i]); }
    else if (!std::strcmp(s, "--warmup")){ need(i); a.warmup = std::atoi(argv[++i]); }
    else if (!std::strcmp(s, "--check")) { a.check = true; }
    else if (!std::strcmp(s, "--csv"))   { need(i); a.csv = argv[++i]; }
    else if (!std::strcmp(s, "--peak"))  { need(i); a.peak_gflops = std::atof(argv[++i]); }
    else if (!std::strcmp(s, "-h") || !std::strcmp(s, "--help")) { print_usage(argv[0]); std::exit(0); }
    else { std::fprintf(stderr, "unknown arg: %s\n", s); print_usage(argv[0]); std::exit(2); }
  }
  return a;
}

}  // namespace bench
