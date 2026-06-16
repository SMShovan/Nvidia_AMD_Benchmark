// Metrics + CSV output helpers.
#pragma once
#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

namespace bench {

inline double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  std::size_t m = v.size() / 2;
  return (v.size() % 2) ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

inline void append_csv(const std::string& path, const std::string& header,
                       const std::string& row) {
  if (path.empty()) return;
  bool exists = false;
  { std::ifstream f(path); exists = f.good(); }
  std::ofstream f(path, std::ios::app);
  if (!exists) f << header << "\n";
  f << row << "\n";
}

}  // namespace bench
