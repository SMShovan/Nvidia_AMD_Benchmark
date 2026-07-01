#!/usr/bin/env bash
# =============================================================================
# GNND-style batched tiny GEMM: one block per point computes OLD . NEW^T
# (list x list, K=dim), reduced to per-OLD and per-NEW minima. Compares the
# hand-tiled regular-core fp32 kernel against the fp16-in/fp32-out tensor-core
# kernel (WMMA on NVIDIA, MFMA on AMD), at GNND scale.
#
#   points  P  graph points / blocks   (default 32,000,000)
#   dim     D  feature-vector length K (default 96; must be a multiple of 32)
#   list    L  OLD/NEW list length     (default 64; must match the build GNND_LIST)
#
# Memory: fp32 dataset P*D*4 (~12 GB) + fp16 copy P*D*2 (~6 GB, tensor only)
#         + output P*2*L*4 (~16 GB). Fits 80/128 GB. Reduce --points if tight.
#
# Usage:   scripts/run_gnnd_gemm.sh <h100|mi300a|cpu> [points]
# Env:     BIN OUTDIR LAUNCH POINTS DIM LIST M REPS WARMUP VARIANTS
# =============================================================================
set -uo pipefail

GPU="${1:-}"; POINTS="${2:-32000000}"
case "$GPU" in h100|mi300a|cpu) ;; *) echo "usage: $0 <h100|mi300a|cpu> [points]"; exit 2;; esac

BIN="${BIN:-build/gemm_bench}"
OUTDIR="${OUTDIR:-results/experiments}"
LAUNCH="${LAUNCH:-}"
if [ "$GPU" = "mi300a" ]; then export HSA_XNACK=1; [ -z "$LAUNCH" ] && LAUNCH="flux run -N1 -g1"; fi

DIM="${DIM:-96}"; LIST="${LIST:-64}"
M="${M:-5}"; REPS="${REPS:-10}"; WARMUP="${WARMUP:-3}"
VARIANTS="${VARIANTS:-regular tensor}"

mkdir -p "$OUTDIR/raw_csv" "$OUTDIR/raw_text"
CSV="$OUTDIR/raw_csv/gnnd_gemm_${GPU}.csv"
TXT="$OUTDIR/raw_text/gnnd_gemm_${GPU}_console.txt"
: > "$TXT"
[ -f "$CSV" ] && mv "$CSV" "${CSV%.csv}.$(date +%Y%m%d_%H%M%S).bak.csv"

say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$TXT"; }
run(){ echo "+ $LAUNCH $BIN $*" | tee -a "$TXT"; $LAUNCH "$BIN" "$@" 2>&1 | tee -a "$TXT"; }

say "GNND batched GEMM: gpu=$GPU points=$POINTS dim=$DIM list=$LIST variants='$VARIANTS'"
say "approx device memory: dataset $(( POINTS * DIM * 4 / 1000000000 )) GB (fp32) + $(( POINTS * DIM * 2 / 1000000000 )) GB (fp16) + out $(( POINTS * 2 * LIST * 4 / 1000000000 )) GB"
say "M=$M reps=$REPS warmup=$WARMUP"

# correctness gate (small, --check) for each variant before the big run
for v in $VARIANTS; do
  run --kernel gnnd_gemm --variant "$v" --points 4096 --dim "$DIM" --list "$LIST" --reps 3 --warmup 1 --check
done

for ((m=1; m<=M; m++)); do
  for v in $VARIANTS; do
    run --kernel gnnd_gemm --variant "$v" --points "$POINTS" --dim "$DIM" --list "$LIST" \
        --reps "$REPS" --warmup "$WARMUP" --csv "$CSV"
  done
done

say "DONE -> $CSV"
say "If a run aborts with a GPU allocation error, retry with fewer points:"
say "  scripts/run_gnnd_gemm.sh $GPU 16000000"
say "Download '$OUTDIR' and plot with scripts/plot_gnnd_gemm.py"
