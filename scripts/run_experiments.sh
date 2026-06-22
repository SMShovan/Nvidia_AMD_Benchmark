#!/usr/bin/env bash
# =============================================================================
# One-shot experiment runner for the H100 (NVIDIA) vs MI300A (AMD) study.
#
# Runs the experiments on ONE machine, repeating each configuration M times so the
# plot script can take an interquartile (25-75 pct) trimmed mean and std. Raw
# results go to a separate tree so they never mix with the existing results/*.csv.
#
#   Experiments (see the codebase book, Part V):
#     0  correctness gate (--check; not timed, not recorded)
#     2  GEMM tiled kernel-only time        (kernel_ms)
#     3  GEMM end-to-end time (memcpy story) (e2e_ms)
#     4  GEMM matrix-core time               (kernel_ms, --kernel matrix)
#     5  regular-vs-matrix speedup           (derived in the plot script)
#     6  spinlock: naive/leader/atomic x perwarp (headline) + contention sweep
#
# Usage:
#   scripts/run_experiments.sh <small|large|huge> <h100|mi300a|cpu> [all|gemm|spin]
#
#   small = quick smoke (tiny sizes)          large = main run (1024..8192, M=15)
#   huge  = big matrices (12288..24576, fewer repeats; GEMM only by default)
#   3rd arg picks what to run (default: all, except 'huge' defaults to 'gemm').
#     -> use 'spin' to redo ONLY the spinlock without re-running GEMM.
#
# Env overrides (all optional):
#   BIN OUTDIR LAUNCH                          paths / per-run launcher
#   M M_HEAD REPS WARMUP                       repeat counts
#   SIZES DTYPES                               GEMM sweep lists
#   THREADS WORK LOCKS_SWEEP                   spinlock knobs
#   HEAD_LOCKS HEAD_MAXSPIN MAXSPIN            spinlock headline / sweep spin caps
#
# After running on BOTH clusters, download the results/experiments/ tree to your
# machine and run:  python3 scripts/plot_experiments.py
# =============================================================================
set -uo pipefail

PHASE="${1:-}"; GPU="${2:-}"; PART="${3:-}"
usage(){ echo "usage: $0 <small|large|huge> <h100|mi300a|cpu> [all|gemm|spin]"; exit 2; }
case "$PHASE" in small|large|huge) ;; *) usage;; esac
case "$GPU"   in h100|mi300a|cpu) ;; *) usage;; esac
if [ -z "$PART" ]; then [ "$PHASE" = "huge" ] && PART=gemm || PART=all; fi
case "$PART" in all|gemm|spin) ;; *) usage;; esac
RUN_GEMM=0; RUN_SPIN=0
case "$PART" in
  all)  RUN_GEMM=1; RUN_SPIN=1;;
  gemm) RUN_GEMM=1;;
  spin) RUN_SPIN=1;;
esac

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
  THREADS="${THREADS:-8192}"; WORK="${WORK:-8}"
  HEAD_LOCKS="${HEAD_LOCKS:-128}"; HEAD_MAXSPIN="${HEAD_MAXSPIN:-10000}"
  MAXSPIN="${MAXSPIN:-10000}"; LOCKS_SWEEP="${LOCKS_SWEEP:-1 64 4096}"
elif [ "$PHASE" = "huge" ]; then
  # Big matrices: fewer repeats (kernels are long and stable). GEMM-focused.
  M="${M:-5}"; M_HEAD="${M_HEAD:-5}"; REPS="${REPS:-3}"; WARMUP="${WARMUP:-2}"
  SIZES="${SIZES:-12288 16384 24576}"; DTYPES="${DTYPES:-fp32 fp16}"
  THREADS="${THREADS:-262144}"; WORK="${WORK:-64}"
  HEAD_LOCKS="${HEAD_LOCKS:-8192}"; HEAD_MAXSPIN="${HEAD_MAXSPIN:-20000}"
  MAXSPIN="${MAXSPIN:-100000}"; LOCKS_SWEEP="${LOCKS_SWEEP:-1 16 256 4096 65536 1048576}"
else  # large
  M="${M:-15}"; M_HEAD="${M_HEAD:-5}"; REPS="${REPS:-5}"; WARMUP="${WARMUP:-5}"
  SIZES="${SIZES:-1024 2048 4096 8192}"; DTYPES="${DTYPES:-fp32 fp16}"
  THREADS="${THREADS:-262144}"; WORK="${WORK:-64}"
  # headline: each warp/wavefront gets its own lock (8192 >= threads/32) so NVIDIA
  # completes cleanly and AMD bails FAST (low contention) at a small spin cap.
  HEAD_LOCKS="${HEAD_LOCKS:-8192}"; HEAD_MAXSPIN="${HEAD_MAXSPIN:-20000}"
  MAXSPIN="${MAXSPIN:-100000}"; LOCKS_SWEEP="${LOCKS_SWEEP:-1 16 256 4096 65536 1048576}"
fi
# CPU backend is fp32-only and has no matrix cores.
[ "$GPU" = "cpu" ] && DTYPES="fp32"

mkdir -p "$OUTDIR/raw_csv" "$OUTDIR/raw_text"
GEMM_CSV="$OUTDIR/raw_csv/${PHASE}_gemm_${GPU}.csv"
SPIN_CSV="$OUTDIR/raw_csv/${PHASE}_spin_${GPU}.csv"
TXTNAME="${PHASE}_${GPU}"; [ "$PART" != "all" ] && TXTNAME="${TXTNAME}_${PART}"
TXT="$OUTDIR/raw_text/${TXTNAME}_console.txt"
: > "$TXT"
# Move aside only the CSV(s) this run will (re)write, so a spin-only rerun keeps GEMM.
ts="$(date +%Y%m%d_%H%M%S)"
[ "$RUN_GEMM" = 1 ] && [ -f "$GEMM_CSV" ] && mv "$GEMM_CSV" "${GEMM_CSV%.csv}.$ts.bak.csv"
[ "$RUN_SPIN" = 1 ] && [ -f "$SPIN_CSV" ] && mv "$SPIN_CSV" "${SPIN_CSV%.csv}.$ts.bak.csv"

say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$TXT"; }
run(){ echo "+ $LAUNCH $BIN $*" | tee -a "$TXT"; $LAUNCH "$BIN" "$@" 2>&1 | tee -a "$TXT"; }

say "phase=$PHASE gpu=$GPU part=$PART bin=$BIN launch='${LAUNCH:-<none>}'"
say "M=$M M_HEAD=$M_HEAD reps=$REPS warmup=$WARMUP sizes='$SIZES' dtypes='$DTYPES'"
say "spin: threads=$THREADS work=$WORK head_locks=$HEAD_LOCKS head_maxspin=$HEAD_MAXSPIN sweep_maxspin=$MAXSPIN locks='$LOCKS_SWEEP'"

# ---------- 0. correctness gate (not timed, not written to the timing CSVs) ----------
say "=== (0) correctness check ==="
CHK_FAIL=0
chk(){ echo "+ check: $*" | tee -a "$TXT"; $LAUNCH "$BIN" "$@" --check 2>&1 | tee -a "$TXT";
       [ "${PIPESTATUS[0]}" -ne 0 ] && { CHK_FAIL=1; say "CHECK FAILED: $*"; }; return 0; }
if [ "$RUN_GEMM" = 1 ]; then
  chk --kernel tiled --dtype fp32 --M 512 --N 512 --K 512
  if [ "$GPU" != "cpu" ]; then
    chk --kernel tiled --dtype fp16 --M 512 --N 512 --K 512
    chk --kernel matrix             --M 512 --N 512 --K 512
  fi
fi
[ "$RUN_SPIN" = 1 ] && run --kernel spinlock --variant atomic --map striped --threads 4096 --locks 64 --work 8
[ "$CHK_FAIL" -ne 0 ] && { say "Aborting: a correctness check failed. Fix before timing."; exit 1; }
say "checks passed."

# ---------- GEMM: tiled + matrix sweeps (experiments 2, 3, 4, 5) ----------
if [ "$RUN_GEMM" = 1 ]; then
  say "=== GEMM tiled sweep -> $GEMM_CSV  (M=$M) ==="
  for ((m=1; m<=M; m++)); do
    for S in $SIZES; do for dt in $DTYPES; do
      run --kernel tiled --dtype "$dt" --M "$S" --N "$S" --K "$S" \
          --reps "$REPS" --warmup "$WARMUP" --csv "$GEMM_CSV"
    done; done
  done
  if [ "$GPU" != "cpu" ]; then
    say "=== GEMM matrix-core sweep -> $GEMM_CSV  (M=$M) ==="
    for ((m=1; m<=M; m++)); do
      for S in $SIZES; do
        run --kernel matrix --M "$S" --N "$S" --K "$S" \
            --reps "$REPS" --warmup "$WARMUP" --csv "$GEMM_CSV"
      done
    done
  fi
fi

# ---------- Spinlock (experiment 6) ----------
if [ "$RUN_SPIN" = 1 ]; then
  say "=== spinlock headline: {naive,leader,atomic} x perwarp -> $SPIN_CSV  (M=$M_HEAD) ==="
  say "    (MI300A naive+perwarp bails by design; head_locks=$HEAD_LOCKS keeps it fast)"
  for ((m=1; m<=M_HEAD; m++)); do
    for v in naive leader atomic; do
      run --kernel spinlock --variant "$v" --map perwarp --threads "$THREADS" \
          --locks "$HEAD_LOCKS" --work "$WORK" --reps "$REPS" --warmup "$WARMUP" \
          --maxspin "$HEAD_MAXSPIN" --csv "$SPIN_CSV"
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
fi

say "DONE."
[ "$RUN_GEMM" = 1 ] && say "  raw GEMM csv: $GEMM_CSV"
[ "$RUN_SPIN" = 1 ] && say "  raw spin csv: $SPIN_CSV"
say "  console log : $TXT"
say "Download the '$OUTDIR' tree to your machine, then: python3 scripts/plot_experiments.py"
