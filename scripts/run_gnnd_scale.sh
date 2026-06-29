#!/usr/bin/env bash
# =============================================================================
# GNND-scale lock-vs-lock-free microbenchmark: leader (lock) vs atomic (lock-free),
# AMD vs NVIDIA, sized like a real NN-Descent graph.
#
#   nodes   N   graph points            (default 128,000,000; fallback 64,000,000)
#   segs    S   lock segments per node  (default 4  => k/32 for k=128)
#   locks   L = N * S                    (derived inside the binary)
#   contention G wavefronts share each lock concurrently (realistic insert contention)
#   work    W   candidate inserts per thread (controls coverage + runtime)
#
# Each candidate insert targets a scattered (node,segment) lock, so accesses span the
# whole multi-GB array. G wavefronts hammer the same lock => leader sees G-way contention,
# atomic/naive see G*warpSize-way contention. Memory: L*(4+8) bytes ~ 6 GB at N=128M.
#
# Coverage note: accesses are uniformly scattered over the WHOLE lock array (full address
# range), so the default (fast) W already exercises the full multi-GB footprint while
# touching a uniform subset of locks. For ~95% distinct-lock coverage (heavier, ~minutes on
# AMD) use the "high-coverage W" value printed at startup.
#
# Only the two safe variants run by default (naive deadlocks on AMD by design); set
# VARIANTS="naive leader atomic" to include it.
#
# Usage:
#   scripts/run_gnnd_scale.sh <h100|mi300a|cpu> [nodes]
# Env overrides: BIN OUTDIR LAUNCH THREADS SEGS CONTENTION WORK M REPS WARMUP MAXSPIN VARIANTS
# =============================================================================
set -uo pipefail

GPU="${1:-}"; NODES="${2:-128000000}"
case "$GPU" in h100|mi300a|cpu) ;; *) echo "usage: $0 <h100|mi300a|cpu> [nodes]"; exit 2;; esac

BIN="${BIN:-build/gemm_bench}"
OUTDIR="${OUTDIR:-results/experiments}"
LAUNCH="${LAUNCH:-}"
if [ "$GPU" = "mi300a" ]; then export HSA_XNACK=1; [ -z "$LAUNCH" ] && LAUNCH="flux run -N1 -g1"; fi

SEGS="${SEGS:-4}"                           # lock segments per node (k/32)
# Equal-leaders contention: the SAME number of warps/wavefronts (lock contenders) per lock on
# BOTH GPUs, so the leader (lock) bars are directly comparable. Consequence: atomic threads/
# lock differ by warp width -> AMD 2x64=128, NVIDIA 2x32=64 threads/lock (atomic favors NV).
# Override CONTENTION (warps/lock) to change it; e.g. CONTENTION via TPL: set 4 on NVIDIA for
# equal-threads instead.
WS=32; [ "$GPU" = "mi300a" ] && WS=64
CONTENTION="${CONTENTION:-2}"               # warps/wavefronts sharing each lock (leaders/lock)
[ "$CONTENTION" -lt 1 ] && CONTENTION=1
THREADS="${THREADS:-16777216}"             # ~16M threads (saturate the GPU)
WORK="${WORK:-64}"                          # candidate inserts per thread (coverage/runtime)
M="${M:-5}"                                 # outer repeats (for avg +/- std)
REPS="${REPS:-3}"; WARMUP="${WARMUP:-2}"
MAXSPIN="${MAXSPIN:-100000}"
VARIANTS="${VARIANTS:-leader atomic}"
LOCKS=$(( NODES * SEGS ))

# Accesses are uniformly scattered over the WHOLE lock array. Coverage of distinct locks
# grows as 1-exp(-(numWarps/G)*W / L); ~3x oversampling gives ~95%.
NUMWARPS=$(( THREADS / WS )); [ "$NUMWARPS" -lt 1 ] && NUMWARPS=1
HIGH_W=$(( (3 * LOCKS * CONTENTION + NUMWARPS - 1) / NUMWARPS ))

mkdir -p "$OUTDIR/raw_csv" "$OUTDIR/raw_text"
CSV="$OUTDIR/raw_csv/gnnd_spin_${GPU}.csv"
TXT="$OUTDIR/raw_text/gnnd_${GPU}_console.txt"
: > "$TXT"
[ -f "$CSV" ] && mv "$CSV" "${CSV%.csv}.$(date +%Y%m%d_%H%M%S).bak.csv"

say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$TXT"; }
run(){ echo "+ $LAUNCH $BIN $*" | tee -a "$TXT"; $LAUNCH "$BIN" "$@" 2>&1 | tee -a "$TXT"; }

say "GNND-scale lock vs lock-free: gpu=$GPU nodes=$NODES segs=$SEGS locks=$LOCKS"
say "threads=$THREADS  contention=${CONTENTION} warps/lock (leaders/lock; atomic sees $(( CONTENTION * WS )) threads/lock, ws=$WS)  work=$WORK  variants='$VARIANTS'"
say "approx device memory: $(( LOCKS * 12 / 1000000000 )) GB   (int lock + u64 counter)"
say "M=$M reps=$REPS warmup=$WARMUP maxspin=$MAXSPIN  (lightweight increment, --critsize 0)"
say "default W scatters across all $((LOCKS/1000000))M locks (touches a uniform subset, fast)"
say "for ~95% lock coverage use WORK=$HIGH_W (heavier, minutes on AMD):  WORK=$HIGH_W scripts/run_gnnd_scale.sh $GPU $NODES"

# correctness gate (small node-mode run) before the big run
run --kernel spinlock --variant atomic --nodes 1024 --segs "$SEGS" --contention "$CONTENTION" \
    --threads 4096 --work 8

for ((m=1; m<=M; m++)); do
  for v in $VARIANTS; do
    run --kernel spinlock --variant "$v" --nodes "$NODES" --segs "$SEGS" \
        --contention "$CONTENTION" --threads "$THREADS" --work "$WORK" --critsize 0 \
        --reps "$REPS" --warmup "$WARMUP" --maxspin "$MAXSPIN" --csv "$CSV"
  done
done

say "DONE -> $CSV"
say "If a run aborts with a GPU allocation error, retry with fewer nodes:"
say "  scripts/run_gnnd_scale.sh $GPU 64000000"
say "Download '$OUTDIR' and plot with scripts/plot_gnnd.py"
