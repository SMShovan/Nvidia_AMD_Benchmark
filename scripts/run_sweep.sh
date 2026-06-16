#!/usr/bin/env bash
# Sweep a kernel over square sizes and dtypes; append to a CSV.
#   BIN=build/gemm_bench KERNEL=tiled DTYPES="fp32 fp16" scripts/run_sweep.sh
#   BIN=build/gemm_bench KERNEL=matrix scripts/run_sweep.sh        # fp16-in/fp32-out
set -euo pipefail
BIN="${BIN:-build/gemm_bench}"
CSV="${CSV:-results/sweep.csv}"
KERNEL="${KERNEL:-tiled}"          # tiled | matrix
SIZES="${SIZES:-512 1024 2048 4096 8192}"
DTYPES="${DTYPES:-fp32 fp16}"      # ignored for matrix (always fp16-in/fp32-out)
mkdir -p "$(dirname "$CSV")"
if [ "$KERNEL" = "matrix" ]; then DTYPES="fp16"; fi
for dt in $DTYPES; do
  for S in $SIZES; do
    "$BIN" --kernel "$KERNEL" --dtype "$dt" --M "$S" --N "$S" --K "$S" \
           --reps 20 --warmup 5 --check --csv "$CSV"
  done
done
echo "Wrote $CSV"
