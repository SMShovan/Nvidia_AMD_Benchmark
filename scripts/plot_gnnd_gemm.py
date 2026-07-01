#!/usr/bin/env python3
"""
Plot the GNND batched-GEMM result: regular-core (fp32) vs tensor-core (fp16-in/fp32-out),
AMD vs NVIDIA, as grouped bars of throughput (GFLOP/s) with IQR-trimmed mean and std.

Usage:
  python3 scripts/plot_gnnd_gemm.py results/experiments/raw_csv/gnnd_gemm_*.csv --out results/experiments
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
    ap.add_argument("inputs", nargs="*", default=["results/experiments/raw_csv/gnnd_gemm_*.csv"])
    ap.add_argument("--out", default="results/experiments")
    a = ap.parse_args(argv)
    files = []
    for p in (a.inputs or ["results/experiments/raw_csv/gnnd_gemm_*.csv"]):
        files += glob.glob(p) or ([p] if os.path.isfile(p) else [])
    rows = [pd.read_csv(f) for f in sorted(set(files)) if os.path.isfile(f)]
    if not rows: sys.exit("no gnnd_gemm CSVs found in: %s" % a.inputs)
    d = pd.concat(rows, ignore_index=True)
    d["vendor"] = d["backend"].map(vend)
    pts = int(pd.to_numeric(d["points"], errors="coerce").max())
    dim = int(pd.to_numeric(d["dim"], errors="coerce").max())
    lst = int(pd.to_numeric(d["list"], errors="coerce").max())
    outdir = os.path.join(a.out, "plots"); os.makedirs(outdir, exist_ok=True)

    variants = [v for v in ["regular", "tensor"] if v in d["variant"].unique()]
    vendors  = [v for v in ["AMD (MI300A)", "NVIDIA (H100)"] if v in d["vendor"].unique()]
    stats, summary = {}, []
    for v in vendors:
        for var in variants:
            sub = d[(d.vendor == v) & (d.variant == var)]
            g, gsd = trim(sub["gflops"]); t, tsd = trim(sub["kernel_ms"])
            stats[(v, var)] = (g, gsd, t, tsd)
            summary.append(dict(vendor=v, variant=var, gflops=g, gflops_std=gsd, ms=t, ms_std=tsd))
    pd.DataFrame(summary).to_csv(os.path.join(outdir, "gnnd_gemm_summary.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(variants)); w = 0.38
    for i, v in enumerate(vendors):
        ys = [stats[(v, var)][0] / 1000.0 for var in variants]   # TFLOP/s
        es = [stats[(v, var)][1] / 1000.0 for var in variants]
        bars = ax.bar(x + (i - 0.5) * w, ys, w, yerr=es, capsize=5, color=VC.get(v), label=v)
        for b, y in zip(bars, ys):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(), f"{y:.1f}",
                    ha="center", va="bottom", fontsize=11, fontweight="bold", color=VC.get(v))
    ax.set_xticks(x); ax.set_xticklabels(
        [f"{v}\n({'fp32' if v=='regular' else 'fp16 tensor'})" for v in variants], fontsize=12)
    ax.set_ylabel("throughput (TFLOP/s)"); ax.set_xlabel("")
    ax.set_title(f"GNND batched GEMM: regular vs tensor core\n"
                 f"{pts/1e6:.0f}M points, {lst}x{lst} per point, K={dim}")
    ax.legend(fontsize=12)
    fig.tight_layout(); path = os.path.join(outdir, "gnnd_gemm.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", path)

    print("\ntensor / regular speedup (per vendor):")
    for v in vendors:
        if (v,"regular") in stats and (v,"tensor") in stats:
            rg, tg = stats[(v,"regular")][0], stats[(v,"tensor")][0]
            rt, tt = stats[(v,"regular")][2], stats[(v,"tensor")][2]
            if rg and tg: print(f"  {v}: {tg/rg:.2f}x throughput  (regular {rg/1000:.1f} TF, tensor {tg/1000:.1f} TF;"
                                 f" time {rt:.1f} vs {tt:.1f} ms)")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
