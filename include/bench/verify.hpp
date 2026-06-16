// Tolerance-based comparison of computed vs reference results.
#pragma once
#include <cmath>
#include <cstddef>
#include "bench/config.hpp"

namespace bench {

struct VerifyResult { bool ok; double max_rel; };

template <typename T>
VerifyResult verify(const T* got, const T* ref, std::size_t n, double tol) {
  double max_rel = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    double g = static_cast<double>(got[i]);
    double r = static_cast<double>(ref[i]);
    double denom = std::fabs(r) > 1e-12 ? std::fabs(r) : 1.0;
    double rel = std::fabs(g - r) / denom;
    if (rel > max_rel) max_rel = rel;
  }
  return {max_rel <= tol, max_rel};
}

// Reasonable default tolerance per dtype.
inline double default_tol(DType d) {
  switch (d) {
    case DType::f16:  return 2e-2;
    case DType::bf16: return 4e-2;
    default:          return 1e-4;
  }
}

}  // namespace bench
