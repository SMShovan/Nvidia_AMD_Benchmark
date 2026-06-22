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
- **Spinlock study (done):** `--kernel spinlock` — a separate microbenchmark for
  GNND's spinlock idea, measuring how a busy-wait lock behaves on NVIDIA (Volta+
  Independent Thread Scheduling) vs AMD CDNA (lock-step wavefronts). See below.
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

## Spinlock microbenchmark

GNND protects each neighbor list with a busy-wait **spinlock**. The interesting
question is architectural: an intra-warp spinlock can **deadlock/livelock on AMD
CDNA** (lanes run lock-step, so a masked-off lock holder can never release while its
peers spin) but **completes on NVIDIA Volta+** (per-thread program counters guarantee
forward progress). This kernel makes that difference measurable, safely.

Three lock variants, all doing the same work (`W` critical sections per thread, each
incrementing a shared counter):

- `naive` — every lane takes a per-index `atomicCAS` spinlock. With `--map perwarp`
  (a whole warp/wavefront contends for one lock) this is the deadlock-prone case.
- `leader` — one lane per warp/wavefront acquires on behalf of the group
  (warp-aggregated). Deadlock-safe on both architectures.
- `atomic` — lock-free `atomicAdd` (no lock at all): the SOLANET-style ceiling.

Safety: a **bounded spin** (`--maxspin`) guarantees the kernel always terminates. A
lane that gives up bumps an `abandoned` counter instead of hanging the GPU, so a
livelock shows up as a finished run with `correct=no` / `abandoned>0` rather than a
wedged device. Correctness check: total counted increments must equal `threads*work`.

```bash
# headline + contention sweeps -> one CSV
BIN=build/gemm_bench CSV=results/spin_h100.csv   scripts/run_spinlock.sh   # on H100
BIN=build/gemm_bench CSV=results/spin_mi300a.csv scripts/run_spinlock.sh   # on MI300A
#   (MI300A: prefix with `flux run -N1 -g1` and `export HSA_XNACK=1`)

# one-off
./build/gemm_bench --kernel spinlock --variant naive --map perwarp \
    --threads 262144 --locks 1024 --work 64 --reps 20 --csv results/spin.csv

# tables + plots (throughput vs contention, headline bars, livelock signal)
python3 scripts/analyze_spinlock.py results/spin_*.csv --out results
```

The expected story: `naive`+`perwarp` completes on H100 but **bails on MI300A**
(`abandoned>0`); `leader` fixes it on both; `atomic` is the fastest everywhere. The
contention sweep (throughput vs `--locks`) compares the two vendors' atomics from
maximum contention (1 lock) to none.

## Full experiment study (one runner + seaborn plots)

`scripts/run_experiments.sh` runs the whole study on one machine, repeating every
configuration **M times** so the plots can use an interquartile-trimmed mean and std
(keep the 25th–75th percentile, drop the extremes). Raw results go to a **separate
tree**, `results/experiments/`, so they never mix with the ad-hoc `results/*.csv`.

```bash
scripts/run_experiments.sh <small|large> <h100|mi300a|cpu>
#   small = quick smoke (tiny sizes, few reps); large = the real run (M=15)
```

Covers correctness (`--check` gate), GEMM tiled kernel-only and end-to-end time,
matrix-core time, the regular-vs-matrix speedup, and the spinlock headline + contention
sweep. Output layout:

```
results/experiments/raw_csv/    <phase>_gemm_<gpu>.csv, <phase>_spin_<gpu>.csv  (M rows/config)
results/experiments/raw_text/   <phase>_<gpu>_console.txt  (full console log)
```

Typical flow (run once per cluster, plot once locally):

```bash
# Matrix (H100), inside an salloc shell:
cmake -S . -B build -DBACKEND=CUDA -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
scripts/run_experiments.sh small h100   # smoke first
scripts/run_experiments.sh large h100

# Tuolumne (MI300A): build with HIP (see above), then:
scripts/run_experiments.sh small mi300a   # auto: HSA_XNACK=1 + flux run -N1 -g1
scripts/run_experiments.sh large mi300a

# download the results/experiments/ tree to your own machine, then:
pip install --user pandas seaborn matplotlib
python3 scripts/plot_experiments.py results/experiments --out results/experiments
```

`plot_experiments.py` writes one figure per experiment (kernel time, transfer
overhead, matrix-core time, regular-vs-matrix speedup, the NVIDIA/AMD gap ratio vs
size, and the two spinlock figures) plus `gemm_summary.csv` / `spin_summary.csv` with
the trimmed mean/std per configuration. The GEMM x-axis is matrix size, so a widening
gap is visible directly. Note: on MI300A the `naive`+`perwarp` headline case spins to
`--maxspin` before bailing, so that row runs slower than the others (expected).

## CLI

`--kernel vadd|tiled|matrix|spinlock`

- GEMM: `--dtype fp32|fp16`, `--M --N --K`, `--peak gflops` (optional).
- spinlock: `--variant naive|leader|atomic`, `--map striped|perwarp|perthread`,
  `--threads T`, `--locks L`, `--work W`, `--critsize C`, `--maxspin S`.
- common: `--reps --warmup`, `--check`, `--csv path`.

## Tuning the tiled kernel

Tile sizes are compile-time and overridable per architecture, e.g.:
```
-DBENCH_BM=128 -DBENCH_BN=128 -DBENCH_BK=8 -DBENCH_TM=8 -DBENCH_TN=8
```
(block threads = (BN/TN)*(BM/TM); keep it a multiple of the warp/wavefront size).
