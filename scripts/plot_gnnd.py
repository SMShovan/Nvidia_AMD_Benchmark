#!/usr/bin/env python3
"""
Plot the GNND-scale lock-vs-lock-free result: leader (lock) vs atomic (lock-free),
AMD vs NVIDIA, as a grouped bar with interquartile-trimmed mean and std error bars.

Usage:
  python3 scripts/plot_gnnd.py results/experiments/raw_csv/gnnd_spin_*.csv --out results/experiments
"""
import argparse, glob, os, sys
import numpy as np, pandas as pd
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt, seaborn as sns
except Exception as e:
    sys.exit("Install plotting libs: pip install --user pandas seaborn matplotlib  (%s)" % e)

sns.set_theme(style="whitegrid", context="talk")
VC = {"AMD (MI300A)": "#E2231A", "NVIDIA (H100)": "#76B900"}
def vend(b): return {"CUDA": "NVIDIA (H100)", "HIP": "AMD (MI300A)", "CPU": "CPU"}.get(str(b).upper(), b)
def trim(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0: return np.nan, np.nan
    q1, q3 = s.quantile(.25), s.quantile(.75); t = s[(s >= q1) & (s <= q3)]
    t = t if len(t) else s
    return float(t.mean()), (float(t.std(ddof=1)) if len(t) > 1 else 0.0)

def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*", default=["results/experiments/raw_csv/gnnd_spin_*.csv"])
    ap.add_argument("--out", default="results/experiments")
    a = ap.parse_args(argv)
    files = []
    for p in (a.inputs or ["results/experiments/raw_csv/gnnd_spin_*.csv"]):
        files += glob.glob(p) or ([p] if os.path.isfile(p) else [])
    rows = [pd.read_csv(f) for f in sorted(set(files)) if os.path.isfile(f)]
    if not rows: sys.exit("no gnnd CSVs found in: %s" % a.inputs)
    d = pd.concat(rows, ignore_index=True)
    d = d[d["map"] == "striped"].copy()
    d["vendor"] = d["backend"].map(vend)
    locks = int(pd.to_numeric(d["locks"], errors="coerce").max())
    def col_max(name, default=0):
        if name not in d.columns: return default
        v = pd.to_numeric(d[name], errors="coerce").max()
        return int(v) if pd.notna(v) else default
    nodes = col_max("nodes"); segs = col_max("segs"); cont = col_max("contention")
    if nodes <= 0:                       # legacy CSV without node-mode columns
        nodes, segs = locks // 4, 4
    outdir = os.path.join(a.out, "plots"); os.makedirs(outdir, exist_ok=True)

    variants = [v for v in ["leader", "atomic"] if v in d["variant"].unique()]
    vendors = [v for v in ["AMD (MI300A)", "NVIDIA (H100)"] if v in d["vendor"].unique()]
    stats, summary = {}, []
    for v in vendors:
        for var in variants:
            m, sd = trim(d[(d.vendor == v) & (d.variant == var)]["mops_per_s"])
            stats[(v, var)] = (m, sd); summary.append(dict(vendor=v, variant=var, mops=m, std=sd))
    pd.DataFrame(summary).to_csv(os.path.join(outdir, "gnnd_summary.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(variants)); w = 0.38
    for i, v in enumerate(vendors):
        ys = [stats[(v, var)][0] for var in variants]
        es = [stats[(v, var)][1] for var in variants]
        bars = ax.bar(x + (i - 0.5) * w, ys, w, yerr=es, capsize=5, color=VC.get(v), label=v)
        for b, y in zip(bars, ys):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                    f"{y/1000:.0f}k" if y >= 1000 else f"{y:.0f}",
                    ha="center", va="bottom", fontsize=11, fontweight="bold", color=VC.get(v))
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([f"{v}\n(lock)" if v=="leader" else f"{v}\n(lock-free)" for v in variants], fontsize=12)
    ax.set_ylabel("throughput (Mops/s, log)"); ax.set_xlabel("")
    # Equal-leaders config: warps-per-lock (the lock contenders) is held equal across vendors,
    # so report that. (atomic threads/lock then differ by warp width: AMD 2x64, NVIDIA 2x32.)
    contvals = sorted({int(x) for x in pd.to_numeric(d["contention"], errors="coerce").fillna(0) if x > 0})
    sub = f"{nodes/1e6:.0f}M nodes, {locks/1e6:.0f}M locks ({segs}/node)"
    if contvals:
        sub += f", {contvals[0]} warps/lock" if len(contvals) == 1 else f", {contvals[0]}-{contvals[-1]} warps/lock"
    ax.set_title(f"Lock vs lock-free at GNND scale\n{sub}, lightweight increment")
    ax.legend(fontsize=12)
    fig.tight_layout(); path = os.path.join(outdir, "gnnd_lock_vs_atomic.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", path)

    print("\nlock penalty (atomic / leader):")
    for v in vendors:
        if (v,"leader") in stats and (v,"atomic") in stats:
            le, at = stats[(v,"leader")][0], stats[(v,"atomic")][0]
            if le and le > 0: print(f"  {v}: {at/le:.1f}x   (leader {le:.0f}  atomic {at:.0f} Mops/s)")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
