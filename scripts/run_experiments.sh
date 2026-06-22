#!/usr/bin/env bash
# =============================================================================
# One-shot experiment runner for the H100 (NVIDIA) vs MI300A (AMD) study.
#
# Runs the full set of experiments on ONE machine, repeating each configuration
# M times so the plot script can take an interquartile (25-75 pct) trimmed mean
# and std. Raw results go to a separate tree so they never mix with the existing
# results/*.csv files.
#
#   Experiments covered (see the codebase book, Part V):
#     0  correctness gate (--check; not timed, not recorded)
#     2  GEMM tiled kernel-only time        (kernel_ms)
#     3  GEMM end-to-end time (memcpy story) (e2e_ms)
#     4  GEMM matrix-core time               (kernel_ms, --kernel matrix)
#     5  regular-vs-matrix speedup           (derived in the plot script)
#     6  spinlock: naive/leader/atomic x perwarp (headline) + contention sweep
#
# Usage:
#   scripts/run_experiments.sh <small|large> <h100|mi300a|cpu>
#
#   small  = quick smoke (tiny sizes, few reps) -> confirm everything runs
#   large  = the real measurement (big sizes, M repeats)
#
# Env overrides (all optional):
#   BIN         path to the binary            (default build/gemm_bench)
#   OUTDIR      output root                   (default results/experiments)
#   LAUNCH      prefix for each run, e.g. "flux run -N1 -g1" or "srun -n1 --gpus=1"
#   M           repeats per timed config      (large:15  small:3)
#   M_HEAD      repeats for the spinlock headline (large:5  small:2)
#   REPS WARMUP per-invocation reps/warmups   (large:5/5  small:3/1)
#   SIZES DTYPES                              GEMM sweep lists
#   THREADS WORK MAXSPIN HEAD_LOCKS LOCKS_SWEEP   spinlock knobs
#
# After running on BOTH clusters, download the whole results/experiments/ tree to
# your own machine and run:  python3 scripts/plot_experiments.py
# =============================================================================
set -uo pipefail

PHASE="${1:-}"; GPU="${2:-}"
usage(){ echo "usage: $0 <small|large> <h100|mi300a|cpu>"; exit 2; }
case "$PHASE" in small|large) ;; *) usage;; esac
case "$GPU"   in h100|mi300a|cpu) ;; *) usage;; esac

BIN="${BIN:-build/gemm_bench}"
OUTDIR="${OUTDIR:-results/experiments}"
LAUNCH="${LAUNCH:-}"

# ---- machine specifics ----
if [ "$GPU" = "mi300a" ]; then
  export HSA_XNACK=1                                  # unified-memory zero copy
  [ -z "$LAUNCH" ] && LAUNCH="flux run -N1 -g1"       # Tuolumne uses Flux
fi

# ---- phase defaults ----
if [ "$PHASE" = "small" ]; then
  M="${M:-3}"; M_HEAD="${M_HEAD:-2}"; REPS="${REPS:-3}"; WARMUP="${WARMUP:-1}"
  SIZES="${SIZES:-256 512 1024}"; DTYPES="${DTYPES:-fp32 fp16}"
  THREADS="${THREADS:-8192}"; WORK="${WORK:-8}"; MAXSPIN="${MAXSPIN:-10000}"
  HEAD_LOCKS="${HEAD_LOCKS:-128}"; LOCKS_SWEEP="${LOCKS_SWEEP:-1 64 4096}"
else
  M="${M:-15}"; M_HEAD="${M_HEAD:-5}"; REPS="${REPS:-5}"; WARMUP="${WARMUP:-5}"
  SIZES="${SIZES:-1024 2048 4096 8192}"; DTYPES="${DTYPES:-fp32 fp16}"
  THREADS="${THREADS:-262144}"; WORK="${WORK:-64}"; MAXSPIN="${MAXSPIN:-100000}"
  HEAD_LOCKS="${HEAD_LOCKS:-1024}"; LOCKS_SWEEP="${LOCKS_SWEEP:-1 16 256 4096 65536 1048576}"
fi
# CPU backend is fp32-only and has no matrix cores.
[ "$GPU" = "cpu" ] && DTYPES="fp32"

mkdir -p "$OUTDIR/raw_csv" "$OUTDIR/raw_text"
GEMM_CSV="$OUTDIR/raw_csv/${PHASE}_gemm_${GPU}.csv"
SPIN_CSV="$OUTDIR/raw_csv/${PHASE}_spin_${GPU}.csv"
TXT="$OUTDIR/raw_text/${PHASE}_${GPU}_console.txt"
: > "$TXT"
# Move any previous same-named CSVs aside so M-row accumulation starts clean.
ts="$(date +%Y%m%d_%H%M%S)"
for f in "$GEMM_CSV" "$SPIN_CSV"; do [ -f "$f" ] && mv "$f" "${f%.csv}.$ts.bak.csv"; done

say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$TXT"; }
run(){ echo "+ $LAUNCH $BIN $*" | tee -a "$TXT"; $LAUNCH "$BIN" "$@" 2>&1 | tee -a "$TXT"; }

say "phase=$PHASE gpu=$GPU bin=$BIN launch='${LAUNCH:-<none>}' M=$M M_HEAD=$M_HEAD reps=$REPS warmup=$WARMUP"
say "sizes='$SIZES' dtypes='$DTYPES'  spin: threads=$THREADS work=$WORK maxspin=$MAXSPIN locks='$LOCKS_SWEEP'"

# ---------- 0. correctness gate (not timed, not written to the timing CSVs) ----------
say "=== (0) correctness check ==="
CHK_FAIL=0
chk(){ echo "+ check: $*" | tee -a "$TXT"; $LAUNCH "$BIN" "$@" --check 2>&1 | tee -a "$TXT";
       [ "${PIPESTATUS[0]}" -ne 0 ] && { CHK_FAIL=1; say "CHECK FAILED: $*"; }; return 0; }
chk --kernel tiled --dtype fp32 --M 512 --N 512 --K 512
if [ "$GPU" != "cpu" ]; then
  chk --kernel tiled --dtype fp16 --M 512 --N 512 --K 512
  chk --kernel matrix             --M 512 --N 512 --K 512
fi
run --kernel spinlock --variant atomic --map striped --threads 4096 --locks 64 --work 8   # informational
[ "$CHK_FAIL" -ne 0 ] && { say "Aborting: a correctness check failed. Fix before timing."; exit 1; }
say "checks passed."

# ---------- GEMM: tiled sweep (experiments 2, 3, and the regular side of 5) ----------
say "=== GEMM tiled sweep -> $GEMM_CSV  (M=$M) ==="
for ((m=1; m<=M; m++)); do
  for S in $SIZES; do for dt in $DTYPES; do
    run --kernel tiled --dtype "$dt" --M "$S" --N "$S" --K "$S" \
        --reps "$REPS" --warmup "$WARMUP" --csv "$GEMM_CSV"
  done; done
done

# ---------- GEMM: matrix-core sweep (experiments 4 and the matrix side of 5) ----------
if [ "$GPU" != "cpu" ]; then
  say "=== GEMM matrix-core sweep -> $GEMM_CSV  (M=$M) ==="
  for ((m=1; m<=M; m++)); do
    for S in $SIZES; do
      run --kernel matrix --M "$S" --N "$S" --K "$S" \
          --reps "$REPS" --warmup "$WARMUP" --csv "$GEMM_CSV"
    done
  done
fi

# ---------- Spinlock (experiment 6) ----------
say "=== spinlock headline: {naive,leader,atomic} x perwarp -> $SPIN_CSV  (M=$M_HEAD) ==="
say "    (on MI300A the naive+perwarp case is expected to bail; it may run slowly)"
for ((m=1; m<=M_HEAD; m++)); do
  for v in naive leader atomic; do
    run --kernel spinlock --variant "$v" --map perwarp --threads "$THREADS" \
        --locks "$HEAD_LOCKS" --work "$WORK" --reps "$REPS" --warmup "$WARMUP" \
        --maxspin "$MAXSPIN" --csv "$SPIN_CSV"
  done
done
say "=== spinlock contention sweep: {leader,atomic} x striped -> $SPIN_CSV  (M=$M) ==="
for ((m=1; m<=M; m++)); do
  for v in leader atomic; do for L in $LOCKS_SWEEP; do
    run --kernel spinlock --variant "$v" --map striped --threads "$THREADS" \
        --locks "$L" --work "$WORK" --reps "$REPS" --warmup "$WARMUP" \
        --maxspin "$MAXSPIN" --csv "$SPIN_CSV"
  done; done
done

say "DONE."
say "  raw GEMM csv: $GEMM_CSV"
say "  raw spin csv: $SPIN_CSV"
say "  console log : $TXT"
say "Download the '$OUTDIR' tree to your machine, then: python3 scripts/plot_experiments.py"
