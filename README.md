# Nvidia_AMD_Benchmark

Microbenchmark comparing **NVIDIA H100** and **AMD MI300A (APU)** on the *same*
hand-written tiled GEMM, to find which architecture is faster and exactly where.
One source tree builds on either machine via a CMake backend flag; a runtime flag
picks the kernel variant. Motivated by GNND's hand-written tiled kernel: no BLAS
library, classic shared-memory tiling, adapted per architecture (NVIDIA warp = 32,
AMD wavefront = 64).

## Status

- **Phase 0 (done):** harness, memory model, validation kernel (`out = a*A + B`).
- **Phase 1 (done):** regular-core **tiled GEMM** (`--kernel tiled`), shared-memory
  tiling + 8x8 register blocking, **fp32** and **fp16-in/fp32-acc**. 256-thread block
  (multiple of both 32 and 64) so one kernel maps to both GPUs. Tile sizes overridable.
- **Phase 2 (done):** matrix-unit **GEMM** (`--kernel matrix`, fp16-in / fp32-out,
  16x16x16): NVIDIA Tensor Cores via `nvcuda::wmma`, AMD Matrix Cores via
  `__builtin_amdgcn_mfma_f32_16x16x16f16` (layout per the ROCm/salykova reference).
  Requires M,N,K multiples of 16. **Validate on hardware with `--check`** (esp. the
  MFMA path; it could not be run in the dev sandbox).
- Phase 3: sweeps + profiling.  Phase 4: analysis.  Phase 5: the explainer book.

All GEMM kernels verify against a CPU reference (sampled for large sizes).

## The memory model (the key architectural difference)

`include/bench/buffer.hpp` is the one place architectures diverge:

- **CUDA (H100):** separate host + device allocations, explicit `cudaMemcpy`.
- **HIP (MI300A):** a single `hipMalloc` on unified HBM; host writes in place and
  the GPU reads the same pointer, so **no `memcpy`** (the advisor's zero-copy way).
  Requires unified memory (`HSA_XNACK=1`).
- **CPU:** a single `malloc`; portable smoke-test backend (fp32 only).

Timing reports **kernel-only** and **end-to-end** (H2D + kernel + D2H). On the APU
the copies are no-ops, so end-to-end equals kernel time; that gap versus the H100
is the unified-memory advantage we want to measure.

## Build and run

### CPU (portable smoke test)
```bash
cmake -S . -B build -DBACKEND=CPU -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/gemm_bench --kernel tiled --dtype fp32 --M 1024 --N 1024 --K 1024 --check
```

### NVIDIA H100
```bash
cmake -S . -B build -DBACKEND=CUDA -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/gemm_bench --kernel tiled  --dtype fp16 --M 4096 --N 4096 --K 4096 --check --csv results/h100.csv
./build/gemm_bench --kernel matrix             --M 4096 --N 4096 --K 4096 --check --csv results/h100.csv
```

### AMD MI300A (Tuolumne)
```bash
ml rocmcc/6.4.3-magic rocm/6.4.3 cmake/3.29.2 ninja
cmake -S . -B build -G Ninja -DBACKEND=HIP -DCMAKE_HIP_ARCHITECTURES=gfx942 \
      -DCMAKE_CXX_COMPILER=amdclang++ -DCMAKE_BUILD_TYPE=Release
cmake --build build
export HSA_XNACK=1
flux run -N1 -g1 ./build/gemm_bench --kernel tiled  --dtype fp16 --M 4096 --N 4096 --K 4096 --check --csv results/mi300a.csv
flux run -N1 -g1 ./build/gemm_bench --kernel matrix             --M 4096 --N 4096 --K 4096 --check --csv results/mi300a.csv
```

Sweep:  `BIN=build/gemm_bench DTYPES="fp32 fp16" scripts/run_sweep.sh`
(use `DTYPES=fp32` for the CPU backend).

## CLI

`--kernel vadd|tiled|matrix`, `--dtype fp32|fp16`, `--M --N --K`,
`--reps --warmup`, `--check`, `--csv path`, `--peak gflops` (optional).

## Tuning the tiled kernel

Tile sizes are compile-time and overridable per architecture, e.g.:
```
-DBENCH_BM=128 -DBENCH_BN=128 -DBENCH_BK=8 -DBENCH_TM=8 -DBENCH_TN=8
```
(block threads = (BN/TN)*(BM/TM); keep it a multiple of the warp/wavefront size).
