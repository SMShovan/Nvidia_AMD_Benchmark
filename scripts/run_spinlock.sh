#!/usr/bin/env bash
# Spinlock microbenchmark sweeps. Two experiments, both append to one CSV:
#
#   (A) HEADLINE / safety  -- the H100-vs-MI300A behaviour difference.
#       For each variant {naive, leader, atomic} with map=perwarp (a whole
#       warp/wavefront contends for one lock). On NVIDIA (Volta+ ITS) the naive
#       lock completes; on AMD CDNA lock-step it livelocks and bails, so its row
#       shows correct=no / abandoned>0. leader and atomic complete on both.
#
#   (B) CONTENTION  -- throughput vs number of locks (1 = all threads on one
#       lock ... up to no contention). Run for the variants that finish on both
#       GPUs (leader, atomic) so the curves are comparable.
#
# Usage:
#   BIN=build/gemm_bench scripts/run_spinlock.sh
#   BIN=build/gemm_bench CSV=results/spin_h100.csv scripts/run_spinlock.sh
#   THREADS=262144 WORK=64 scripts/run_spinlock.sh
set -euo pipefail
BIN="${BIN:-build/gemm_bench}"
CSV="${CSV:-results/spinlock.csv}"
THREADS="${THREADS:-262144}"     # total threads (rounded up to a multiple of 256)
WORK="${WORK:-64}"               # critical sections per thread
CRIT="${CRIT:-0}"                # dummy work inside the critical section
REPS="${REPS:-20}"
WARMUP="${WARMUP:-5}"
MAXSPIN="${MAXSPIN:-1000000}"    # bounded-spin cap (safety net; detects livelock)
LOCKS_SWEEP="${LOCKS_SWEEP:-1 16 256 4096 65536 1048576}"
mkdir -p "$(dirname "$CSV")"

run() { "$BIN" --kernel spinlock --threads "$THREADS" --work "$WORK" --critsize "$CRIT" \
                --reps "$REPS" --warmup "$WARMUP" --maxspin "$MAXSPIN" --csv "$CSV" "$@"; }

echo "== (A) headline: variant x map=perwarp (one lock per warp/wavefront) =="
# perwarp deliberately forces intra-group contention -> exposes the lock-step deadlock.
HEAD_LOCKS="${HEAD_LOCKS:-1024}"
for v in naive leader atomic; do
  run --variant "$v" --map perwarp --locks "$HEAD_LOCKS"
done

echo "== (B) contention: throughput vs #locks (leader, atomic; map=striped) =="
for v in leader atomic; do
  for L in $LOCKS_SWEEP; do
    run --variant "$v" --map striped --locks "$L"
  done
done

echo "Wrote $CSV"
