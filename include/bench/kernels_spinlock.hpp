// Spinlock microbenchmark kernels.
//   variants: naive  = per-lane atomicCAS spinlock (deadlock-prone intra-group)
//             leader = one lane per warp/wavefront locks (warp-aggregated, safe)
//             atomic = lock-free atomicAdd (the ceiling; SOLANET-style)
//   maps:     striped   idx = tid % L
//             perwarp   idx = (tid / warpSize) % L   (intra-group contention)
//             perthread idx = tid % L  (caller sets L >= T for no contention)
// Bounded spin (maxspin) guarantees termination; a lane that gives up increments
// 'abandoned', so deadlock/livelock is measurable without hanging the GPU.
#pragma once
#include <string>
#include "bench/backend.hpp"

namespace bench {
using u64 = unsigned long long;
enum { V_NAIVE = 0, V_LEADER = 1, V_ATOMIC = 2 };
enum { MAP_STRIPED = 0, MAP_PERWARP = 1, MAP_PERTHREAD = 2 };

inline int parse_variant(const std::string& s){
  if (s=="naive")  return V_NAIVE;
  if (s=="leader") return V_LEADER;
  return V_ATOMIC;
}
inline int parse_map(const std::string& s){
  if (s=="perwarp")   return MAP_PERWARP;
  if (s=="perthread") return MAP_PERTHREAD;
  return MAP_STRIPED;
}

#if BENCH_DEVICE
__device__ __forceinline__ long long lock_index(long long tid, long long L, int map, int warp){
  if (map == MAP_PERWARP) return (tid / warp) % L;
  return tid % L;  // striped and perthread
}
__device__ __forceinline__ u64 crit_work(int critsize, long long idx){
  u64 s = 0; for (int c = 0; c < critsize; ++c) s += (u64)(idx + 1) * (u64)(c + 1); return s;
}

__global__ void k_spin_naive(int* lock, u64* counter, u64* abandoned, u64* sink,
                             long long T, int W, long long L, int map, int critsize, int maxspin){
  long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (tid >= T) return;
  u64 mywork = 0;
  for (int j = 0; j < W; ++j){
    long long idx = lock_index(tid, L, map, warpSize);
    int spins = 0; bool got = false;
    while (spins < maxspin){ if (atomicCAS(&lock[idx], 0, 1) == 0){ got = true; break; } ++spins; }
    if (got){ __threadfence(); counter[idx] += 1ULL; mywork += crit_work(critsize, idx);
              __threadfence(); atomicExch(&lock[idx], 0); }
    else { atomicAdd(abandoned, 1ULL); }
  }
  if (critsize > 0) atomicAdd(sink, mywork);
}

__global__ void k_spin_leader(int* lock, u64* counter, u64* abandoned, u64* sink,
                              long long T, int W, long long L, int map, int critsize, int maxspin){
  long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (tid >= T) return;
  int lane = threadIdx.x % warpSize;
  u64 mywork = 0;
  for (int j = 0; j < W; ++j){
    long long idx = (tid / warpSize) % L;   // whole warp shares one lock
    if (lane == 0){
      int spins = 0; bool got = false;
      while (spins < maxspin){ if (atomicCAS(&lock[idx], 0, 1) == 0){ got = true; break; } ++spins; }
      if (got){ __threadfence(); counter[idx] += (u64)warpSize; mywork += crit_work(critsize, idx);
                __threadfence(); atomicExch(&lock[idx], 0); }
      else { atomicAdd(abandoned, (u64)warpSize); }
    }
  }
  if (critsize > 0 && lane == 0) atomicAdd(sink, mywork);
}

__global__ void k_spin_atomic(u64* counter, u64* abandoned, u64* sink,
                              long long T, int W, long long L, int map, int critsize){
  long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (tid >= T) return;
  (void)abandoned;
  u64 mywork = 0;
  for (int j = 0; j < W; ++j){
    long long idx = lock_index(tid, L, map, warpSize);
    atomicAdd(&counter[idx], 1ULL); mywork += crit_work(critsize, idx);
  }
  if (critsize > 0) atomicAdd(sink, mywork);
}

inline void launch_spinlock(int variant, int* lock, u64* counter, u64* abandoned, u64* sink,
                            long long T, int W, long long L, int map, int critsize, int maxspin){
  const int tpb = 256;
  dim3 block(tpb);
  dim3 grid((unsigned)((T + tpb - 1) / tpb));
  if      (variant == V_NAIVE)  k_spin_naive <<<grid, block>>>(lock, counter, abandoned, sink, T, W, L, map, critsize, maxspin);
  else if (variant == V_LEADER) k_spin_leader<<<grid, block>>>(lock, counter, abandoned, sink, T, W, L, map, critsize, maxspin);
  else                          k_spin_atomic<<<grid, block>>>(counter, abandoned, sink, T, W, L, map, critsize);
  check_launch();
}

// ----------------------------------------------------------------------------
// Node mode (GNND-scale): the lock array models N graph nodes x S segments each,
// so L = N*S. Each thread does W "candidate insert" steps; on step j a thread's
// whole wavefront targets one (node,segment) lock chosen by a 64-bit scatter hash,
// so accesses spread across the entire multi-GB array (realistic cache behavior).
// `G` wavefronts share each lock => contention is G warp-leaders (leader) or
// G*warpSize threads (atomic/naive) hammering the same lock concurrently.
// ----------------------------------------------------------------------------
__device__ __forceinline__ long long node_lock_index(long long gwarp, long long numGroups,
                                                     long long L, int j, int G){
  long long g = (G < 1 ? 1 : G);
  long long groupId = (gwarp / g) % numGroups;            // which lock-group this warp is in
  // splitmix64 avalanche so the % L below scatters uniformly over the WHOLE array for any
  // L (a plain multiply would alias to multiples of gcd(mult,L) and miss most locks).
  unsigned long long z = (unsigned long long)(groupId + (long long)j * numGroups) + 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  z =  z ^ (z >> 31);
  return (long long)(z % (unsigned long long)L);
}

__global__ void k_spin_atomic_node(u64* counter, u64* sink,
                                   long long T, int W, long long L, int S, int G, int critsize){
  long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (tid >= T) return;
  (void)S;
  long long ws = warpSize, gwarp = tid / ws;
  long long numWarps = (T + ws - 1) / ws;
  long long numGroups = numWarps / (G < 1 ? 1 : G); if (numGroups < 1) numGroups = 1;
  u64 mywork = 0;
  for (int j = 0; j < W; ++j){
    long long idx = node_lock_index(gwarp, numGroups, L, j, G);  // whole warp -> same lock
    atomicAdd(&counter[idx], 1ULL); mywork += crit_work(critsize, idx);
  }
  if (critsize > 0) atomicAdd(sink, mywork);
}

__global__ void k_spin_leader_node(int* lock, u64* counter, u64* abandoned, u64* sink,
                                   long long T, int W, long long L, int S, int G,
                                   int critsize, int maxspin){
  long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (tid >= T) return;
  (void)S;
  int lane = threadIdx.x % warpSize;
  long long ws = warpSize, gwarp = tid / ws;
  long long numWarps = (T + ws - 1) / ws;
  long long numGroups = numWarps / (G < 1 ? 1 : G); if (numGroups < 1) numGroups = 1;
  u64 mywork = 0;
  for (int j = 0; j < W; ++j){
    long long idx = node_lock_index(gwarp, numGroups, L, j, G);  // G warps share this lock
    if (lane == 0){
      int spins = 0; bool got = false;
      while (spins < maxspin){ if (atomicCAS(&lock[idx], 0, 1) == 0){ got = true; break; } ++spins; }
      if (got){ __threadfence(); counter[idx] += (u64)warpSize; mywork += crit_work(critsize, idx);
                __threadfence(); atomicExch(&lock[idx], 0); }
      else { atomicAdd(abandoned, (u64)warpSize); }
    }
  }
  if (critsize > 0 && lane == 0) atomicAdd(sink, mywork);
}

__global__ void k_spin_naive_node(int* lock, u64* counter, u64* abandoned, u64* sink,
                                  long long T, int W, long long L, int S, int G,
                                  int critsize, int maxspin){
  long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (tid >= T) return;
  (void)S;
  long long ws = warpSize, gwarp = tid / ws;
  long long numWarps = (T + ws - 1) / ws;
  long long numGroups = numWarps / (G < 1 ? 1 : G); if (numGroups < 1) numGroups = 1;
  u64 mywork = 0;
  for (int j = 0; j < W; ++j){
    long long idx = node_lock_index(gwarp, numGroups, L, j, G);  // every lane CAS-locks idx
    int spins = 0; bool got = false;
    while (spins < maxspin){ if (atomicCAS(&lock[idx], 0, 1) == 0){ got = true; break; } ++spins; }
    if (got){ __threadfence(); counter[idx] += 1ULL; mywork += crit_work(critsize, idx);
              __threadfence(); atomicExch(&lock[idx], 0); }
    else { atomicAdd(abandoned, 1ULL); }
  }
  if (critsize > 0) atomicAdd(sink, mywork);
}

inline void launch_spinlock_node(int variant, int* lock, u64* counter, u64* abandoned, u64* sink,
                                 long long T, int W, long long L, int S, int G,
                                 int critsize, int maxspin){
  const int tpb = 256;
  dim3 block(tpb);
  dim3 grid((unsigned)((T + tpb - 1) / tpb));
  if      (variant == V_NAIVE)  k_spin_naive_node <<<grid, block>>>(lock, counter, abandoned, sink, T, W, L, S, G, critsize, maxspin);
  else if (variant == V_LEADER) k_spin_leader_node<<<grid, block>>>(lock, counter, abandoned, sink, T, W, L, S, G, critsize, maxspin);
  else                          k_spin_atomic_node<<<grid, block>>>(counter, sink, T, W, L, S, G, critsize);
  check_launch();
}
#else
// CPU smoke backend: serial (no concurrency), validates counting + harness only.
inline long long lock_index_cpu(long long tid, long long L, int map, int warp){
  if (map == MAP_PERWARP) return (tid / warp) % L;
  return tid % L;
}
inline void launch_spinlock(int variant, int* lock, u64* counter, u64* abandoned, u64* sink,
                            long long T, int W, long long L, int map, int critsize, int maxspin){
  (void)variant;(void)lock;(void)abandoned;(void)sink;(void)critsize;(void)maxspin;
  for (long long tid = 0; tid < T; ++tid)
    for (int j = 0; j < W; ++j) counter[lock_index_cpu(tid, L, map, 32)] += 1ULL;
}

// CPU smoke for node mode: emulates the device mapping with a fixed 32-lane "warp" so
// the count invariant (sum == T*W) and the scatter/grouping logic can be validated
// without a GPU. Leader adds warpSize per warp-step; atomic/naive add 1 per thread-step.
inline long long node_lock_index_cpu(long long gwarp, long long numGroups, long long L, int j, int G){
  long long g = (G < 1 ? 1 : G);
  long long groupId = (gwarp / g) % numGroups;
  unsigned long long z = (unsigned long long)(groupId + (long long)j * numGroups) + 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  z =  z ^ (z >> 31);
  return (long long)(z % (unsigned long long)L);
}
inline void launch_spinlock_node(int variant, int* lock, u64* counter, u64* abandoned, u64* sink,
                                 long long T, int W, long long L, int S, int G,
                                 int critsize, int maxspin){
  (void)lock;(void)abandoned;(void)sink;(void)S;(void)critsize;(void)maxspin;
  const long long ws = 32;
  const long long numWarps = (T + ws - 1) / ws;
  long long numGroups = numWarps / (G < 1 ? 1 : G); if (numGroups < 1) numGroups = 1;
  if (variant == V_LEADER) {
    for (long long w = 0; w < numWarps; ++w)
      for (int j = 0; j < W; ++j) counter[node_lock_index_cpu(w, numGroups, L, j, G)] += (u64)ws;
  } else {
    for (long long tid = 0; tid < T; ++tid){
      long long gwarp = tid / ws;
      for (int j = 0; j < W; ++j) counter[node_lock_index_cpu(gwarp, numGroups, L, j, G)] += 1ULL;
    }
  }
}
#endif

}  // namespace bench
