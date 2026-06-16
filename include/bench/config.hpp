// Data-type selection (used for CSV labels now; full templating in Phase 1/2).
#pragma once
#include <cstddef>
#include <string>

namespace bench {
enum class DType { f32, f16, bf16 };

inline DType parse_dtype(const std::string& s) {
  if (s == "fp16" || s == "f16" || s == "half") return DType::f16;
  if (s == "bf16") return DType::bf16;
  return DType::f32;
}
inline const char* dtype_name(DType d) {
  switch (d) {
    case DType::f16:  return "fp16";
    case DType::bf16: return "bf16";
    default:          return "fp32";
  }
}
inline std::size_t dtype_bytes(DType d) {
  return d == DType::f32 ? 4u : 2u;
}
}  // namespace bench
