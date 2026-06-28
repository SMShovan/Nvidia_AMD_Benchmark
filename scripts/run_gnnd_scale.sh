#!/usr/bin/env bash
# =============================================================================
# GNND-scale lock-vs-lock-free microbenchmark: leader (lock) vs atomic (lock-free),
# AMD vs NVIDIA, sized like a real NN-Descent graph.
#
#   nodes  N  (default 128,000,000; fallback 64,000,000 if memory is tight)
#   locks  L = 4 * N   (4 lock segments per node, as in GNND for k=128)
#   work   = lightweight counter increment only (--critsize 0)
#
# Memory: L * (4 + 8) bytes  ->  ~6 GB at N=128M, ~3 GB at N=64M (fits 80/128 GB).
# Only the two safe variants are run (naive deadlocks on AMD by design).
#
# Usage:
#   scripts/run_gnnd_scale.sh <h100|mi300a|cpu> [nodes]
# Env overrides: BIN OUTDIR LAUNCH THREADS WORK M REPS WARMUP MAXSPIN
# =============================================================================
set -uo pipefail

GPU="${1:-}"; NODES="${2:-128000000}"
case "$GPU" in h100|mi300a|cpu) ;; *) echo "usage: $0 <h100|mi300a|cpu> [nodes]"; exit 2;; esac

BIN="${BIN:-build/gemm_bench}"
OUTDIR="${OUTDIR:-results/experiments}"
LAUNCH="${LAUNCH:-}"
if [ "$GPU" = "mi300a" ]; then export HSA_XNACK=1; [ -z "$LAUNCH" ] && LAUNCH="flux run -N1 -g1"; fi

LOCKS=$(( NODES * 4 ))                       # 4 lock segments per node
THREADS="${THREADS:-16777216}"              # ~16M threads (saturate the GPU)
WORK="${WORK:-16}"                          # critical sections per thread
M="${M:-5}"                                 # outer repeats (for avg +/- std)
REPS="${REPS:-3}"; WARMUP="${WARMUP:-2}"
MAXSPIN="${MAXSPIN:-100000}"

mkdir -p "$OUTDIR/raw_csv" "$OUTDIR/raw_text"
CSV="$OUTDIR/raw_csv/gnnd_spin_${GPU}.csv"
TXT="$OUTDIR/raw_text/gnnd_${GPU}_console.txt"
: > "$TXT"
[ -f "$CSV" ] && mv "$CSV" "${CSV%.csv}.$(date +%Y%m%d_%H%M%S).bak.csv"

say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$TXT"; }
run(){ echo "+ $LAUNCH $BIN $*" | tee -a "$TXT"; $LAUNCH "$BIN" "$@" 2>&1 | tee -a "$TXT"; }

say "GNND-scale lock vs lock-free: gpu=$GPU nodes=$NODES locks=$LOCKS threads=$THREADS work=$WORK"
say "approx device memory: $(( LOCKS * 12 / 1000000000 )) GB   (int lock + u64 counter)"
say "M=$M reps=$REPS warmup=$WARMUP  (lightweight increment, --critsize 0, map striped)"

# correctness gate (small) before the big run
run --kernel spinlock --variant atomic --map striped --threads 4096 --locks 64 --work 8

for ((m=1; m<=M; m++)); do
  for v in leader atomic; do
    run --kernel spinlock --variant "$v" --map striped --threads "$THREADS" \
        --locks "$LOCKS" --work "$WORK" --critsize 0 --reps "$REPS" --warmup "$WARMUP" \
        --maxspin "$MAXSPIN" --csv "$CSV"
  done
done

say "DONE -> $CSV"
say "If a run aborts with a GPU allocation error, retry with fewer nodes:"
say "  scripts/run_gnnd_scale.sh $GPU 64000000"
say "Download '$OUTDIR' and plot with scripts/plot_gnnd.py"
