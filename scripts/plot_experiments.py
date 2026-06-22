#!/usr/bin/env python3
"""
Seaborn plots for the H100 (NVIDIA) vs MI300A (AMD) study.

Reads the raw CSVs produced by scripts/run_experiments.sh (which writes M rows per
configuration), takes an interquartile-trimmed statistic per configuration
(keep the 25th-75th percentile, then report the trimmed MEAN as the point and the
trimmed STD as the error bar, so extreme runs do not skew the figure), and writes
one presentation-quality figure per experiment plus processed summary tables.

Run this ON YOUR OWN MACHINE after downloading the results/experiments/ tree:

    pip install --user pandas seaborn matplotlib        # once
    python3 scripts/plot_experiments.py                 # uses results/experiments/
    # or point it at the files / a folder you downloaded:
    python3 scripts/plot_experiments.py ~/Downloads/experiments --out ~/Downloads/plots

Figures (x-axis is matrix size for GEMM, so you can see whether the gap widens):
  exp2_tiled_kernel_time_<dtype>.png   kernel-only time, AMD vs NVIDIA
  exp2_vendor_gap_ratio.png            NVIDIA/AMD kernel-time ratio vs size
  exp3_e2e_overhead.png                end-to-end / kernel (the memcpy story)
  exp3_kernel_vs_e2e.png               kernel vs end-to-end lines, per vendor
  exp4_matrix_kernel_time.png          matrix-core time, MFMA vs WMMA
  exp5_regular_vs_matrix_speedup.png   per-vendor speedup (tiled fp16 / matrix)
  exp6_spinlock_headline.png           naive/leader/atomic at perwarp (bail flagged)
  exp6_spinlock_contention.png         throughput vs #locks
  bonus_gflops.png                     GFLOP/s vs size
Tables: gemm_summary.csv, spin_summary.csv (trimmed mean/std/n per config).
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception as e:
    sys.exit("Missing a plotting library (%s).\nInstall with: pip install --user pandas seaborn matplotlib" % e)

sns.set_theme(style="whitegrid", context="talk")
VENDOR_COLORS = {"NVIDIA (H100)": "#76B900", "AMD (MI300A)": "#E2231A", "CPU": "#888888"}
VENDOR_ORDER  = ["AMD (MI300A)", "NVIDIA (H100)", "CPU"]

# ----------------------------------------------------------------------------- io
def vendor_of(backend):
    b = str(backend).upper()
    return {"CUDA": "NVIDIA (H100)", "HIP": "AMD (MI300A)", "CPU": "CPU"}.get(b, b)

def load(paths):
    gemm, spin = [], []
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += glob.glob(os.path.join(p, "**", "*.csv"), recursive=True)
        else:
            files += glob.glob(p) or [p]
    for q in sorted(set(files)):
        if not os.path.isfile(q):
            continue
        try:
            df = pd.read_csv(q)
        except Exception:
            continue
        cols = set(df.columns)
        if {"variant", "mops_per_s"}.issubset(cols):
            spin.append(df)
        elif {"kernel", "kernel_ms", "M"}.issubset(cols):
            gemm.append(df)
    g = pd.concat(gemm, ignore_index=True) if gemm else pd.DataFrame()
    s = pd.concat(spin, ignore_index=True) if spin else pd.DataFrame()
    for d in (g, s):
        if not d.empty:
            d["vendor"] = d["backend"].map(vendor_of)
    return g, s

# ------------------------------------------------------------------- statistics
def trimmed(series):
    """Trimmed mean and std over the 25th-75th percentile (extremes removed)."""
    s = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if len(s) == 0:
        return np.nan, np.nan, 0
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    t = s[(s >= q1) & (s <= q3)]
    if len(t) == 0:
        t = s
    mean = float(t.mean())
    std = float(t.std(ddof=1)) if len(t) > 1 else 0.0
    return mean, std, int(len(t))

def agg(df, group_cols, value_cols):
    rows = []
    for key, gdf in df.groupby(group_cols):
        rec = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        n = 0
        for v in value_cols:
            if v in gdf:
                m, sd, n = trimmed(gdf[v])
                rec[v], rec[v + "_std"] = m, sd
        rec["n"] = n
        rows.append(rec)
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------- helpers
def _color(label):
    return VENDOR_COLORS.get(label, None)

def lines(ax, data, x, y, yerr, group, logx=True, palette=None):
    keys = sorted(data[group].unique(), key=lambda k: (VENDOR_ORDER.index(k) if k in VENDOR_ORDER else 99, str(k)))
    pal = palette or sns.color_palette("colorblind", n_colors=len(keys))
    for i, key in enumerate(keys):
        g = data[data[group] == key].sort_values(x)
        if g.empty:
            continue
        ax.errorbar(g[x], g[y], yerr=g.get(yerr), marker="o", capsize=3, lw=2.2, ms=7,
                    color=_color(key) or pal[i % len(pal)], label=str(key))
    if logx:
        ax.set_xscale("log", base=2)
    ax.legend(fontsize=11, title=None)

def save(fig, out, name):
    path = os.path.join(out, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", path)

# -------------------------------------------------------------------- GEMM plots
def gemm_plots(g, out):
    if g.empty:
        print("[no GEMM rows]"); return
    sq = g[(g["M"] == g["N"]) & (g["N"] == g["K"])].copy()
    sq["size"] = sq["M"].astype(int)
    a = agg(sq, ["vendor", "kernel", "dtype", "size"], ["kernel_ms", "e2e_ms", "gflops"])
    a.to_csv(os.path.join(out, "gemm_summary.csv"), index=False)
    print("  wrote", os.path.join(out, "gemm_summary.csv"))

    tiled = a[a["kernel"] == "tiled"]
    matrix = a[a["kernel"] == "matrix"]

    # exp2: tiled kernel-only time vs size, one figure per dtype
    for dt in sorted(tiled["dtype"].unique()):
        d = tiled[tiled["dtype"] == dt]
        if d.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 5.5))
        lines(ax, d, "size", "kernel_ms", "kernel_ms_std", "vendor")
        ax.set_xlabel("matrix size  M=N=K"); ax.set_ylabel("kernel time (ms)")
        ax.set_title(f"GEMM tiled kernel time ({dt})  -- lower is better")
        ax.set_xticks(sorted(d["size"].unique())); ax.set_xticklabels(sorted(d["size"].unique()))
        save(fig, out, f"exp2_tiled_kernel_time_{dt}.png")

    # exp2: vendor-gap ratio (NVIDIA / AMD) vs size -- shows if the gap widens
    _vendor_ratio(tiled, "kernel_ms", out, "exp2_vendor_gap_ratio.png",
                  "GEMM tiled: NVIDIA / AMD kernel-time ratio", per="dtype")

    # exp3: end-to-end overhead = e2e / kernel, per vendor (the memcpy story)
    d = tiled.copy()
    d = d[d["kernel_ms"] > 0]
    if not d.empty:
        d["overhead"] = d["e2e_ms"] / d["kernel_ms"]
        for dt in sorted(d["dtype"].unique()):
            dd = d[d["dtype"] == dt]
            fig, ax = plt.subplots(figsize=(8, 5.5))
            for key in sorted(dd["vendor"].unique(), key=lambda k: VENDOR_ORDER.index(k) if k in VENDOR_ORDER else 99):
                gg = dd[dd["vendor"] == key].sort_values("size")
                ax.plot(gg["size"], gg["overhead"], marker="o", lw=2.2, ms=7, color=_color(key), label=str(key))
            ax.axhline(1.0, ls="--", c="gray", lw=1)
            ax.set_xscale("log", base=2); ax.set_xticks(sorted(dd["size"].unique()))
            ax.set_xticklabels(sorted(dd["size"].unique()))
            ax.set_xlabel("matrix size  M=N=K"); ax.set_ylabel("end-to-end / kernel  (1.0 = no transfer cost)")
            ax.set_title(f"GEMM transfer overhead ({dt}): unified memory vs memcpy")
            ax.legend(fontsize=11)
            save(fig, out, f"exp3_e2e_overhead_{dt}.png")

    # exp3: kernel vs end-to-end absolute lines, per vendor (fp32 if present else first)
    dts = sorted(tiled["dtype"].unique())
    if dts:
        dt = "fp32" if "fp32" in dts else dts[0]
        d = tiled[tiled["dtype"] == dt]
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for key in sorted(d["vendor"].unique(), key=lambda k: VENDOR_ORDER.index(k) if k in VENDOR_ORDER else 99):
            gg = d[d["vendor"] == key].sort_values("size")
            c = _color(key)
            ax.errorbar(gg["size"], gg["kernel_ms"], yerr=gg["kernel_ms_std"], marker="o", lw=2.2, color=c, label=f"{key} kernel")
            ax.errorbar(gg["size"], gg["e2e_ms"], yerr=gg["e2e_ms_std"], marker="s", ls="--", lw=2.0, color=c, label=f"{key} end-to-end")
        ax.set_xscale("log", base=2); ax.set_xticks(sorted(d["size"].unique()))
        ax.set_xticklabels(sorted(d["size"].unique()))
        ax.set_xlabel("matrix size  M=N=K"); ax.set_ylabel("time (ms)")
        ax.set_title(f"GEMM kernel vs end-to-end ({dt})")
        ax.legend(fontsize=10)
        save(fig, out, "exp3_kernel_vs_e2e.png")

    # exp4: matrix-core kernel time vs size
    if not matrix.empty:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        lines(ax, matrix, "size", "kernel_ms", "kernel_ms_std", "vendor")
        ax.set_xlabel("matrix size  M=N=K"); ax.set_ylabel("kernel time (ms)")
        ax.set_title("GEMM matrix-core time (WMMA vs MFMA)  -- lower is better")
        ax.set_xticks(sorted(matrix["size"].unique())); ax.set_xticklabels(sorted(matrix["size"].unique()))
        save(fig, out, "exp4_matrix_kernel_time.png")
        _vendor_ratio(matrix, "kernel_ms", out, "exp4_matrix_vendor_gap_ratio.png",
                      "Matrix core: NVIDIA / AMD kernel-time ratio", per=None)

    # exp5: regular-vs-matrix speedup per vendor (tiled fp16 / matrix), vs size
    base = tiled[tiled["dtype"] == "fp16"][["vendor", "size", "kernel_ms"]].rename(columns={"kernel_ms": "tiled_ms"})
    mat = matrix[["vendor", "size", "kernel_ms"]].rename(columns={"kernel_ms": "matrix_ms"})
    sp = base.merge(mat, on=["vendor", "size"], how="inner")
    if not sp.empty:
        sp["speedup"] = sp["tiled_ms"] / sp["matrix_ms"]
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for key in sorted(sp["vendor"].unique(), key=lambda k: VENDOR_ORDER.index(k) if k in VENDOR_ORDER else 99):
            gg = sp[sp["vendor"] == key].sort_values("size")
            ax.plot(gg["size"], gg["speedup"], marker="o", lw=2.4, ms=8, color=_color(key), label=str(key))
        ax.axhline(1.0, ls="--", c="gray", lw=1)
        ax.set_xscale("log", base=2); ax.set_xticks(sorted(sp["size"].unique()))
        ax.set_xticklabels(sorted(sp["size"].unique()))
        ax.set_xlabel("matrix size  M=N=K"); ax.set_ylabel("speedup  (tiled fp16 time / matrix time)")
        ax.set_title("Matrix-core speedup over regular cores (per vendor)")
        ax.legend(fontsize=11)
        save(fig, out, "exp5_regular_vs_matrix_speedup.png")
    else:
        print("  [exp5 skipped: need both tiled fp16 and matrix rows for a vendor]")

    # bonus: GFLOP/s vs size (tiled fp16 + matrix)
    b = a[((a["kernel"] == "tiled") & (a["dtype"] == "fp16")) | (a["kernel"] == "matrix")].copy()
    if not b.empty:
        b["series"] = b["vendor"] + " " + b["kernel"]
        fig, ax = plt.subplots(figsize=(8, 5.5))
        lines(ax, b, "size", "gflops", "gflops_std", "series",
              palette=sns.color_palette("colorblind", n_colors=b["series"].nunique()))
        ax.set_xlabel("matrix size  M=N=K"); ax.set_ylabel("GFLOP/s  -- higher is better")
        ax.set_title("GEMM throughput (fp16)")
        ax.set_xticks(sorted(b["size"].unique())); ax.set_xticklabels(sorted(b["size"].unique()))
        save(fig, out, "bonus_gflops.png")

def _vendor_ratio(a, metric, out, fname, title, per="dtype"):
    """Plot NVIDIA/AMD ratio of `metric` vs size; one line per `per` (or single)."""
    nv = a[a["vendor"] == "NVIDIA (H100)"]; am = a[a["vendor"] == "AMD (MI300A)"]
    if nv.empty or am.empty:
        return
    keycols = ["size"] + ([per] if per else [])
    m = nv[keycols + [metric]].merge(am[keycols + [metric]], on=keycols, suffixes=("_nv", "_am"))
    if m.empty:
        return
    m["ratio"] = m[metric + "_nv"] / m[metric + "_am"]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    groups = sorted(m[per].unique()) if per else [None]
    pal = sns.color_palette("colorblind", n_colors=len(groups))
    for i, gkey in enumerate(groups):
        gg = (m[m[per] == gkey] if per else m).sort_values("size")
        lab = (str(gkey) if per else "ratio")
        ax.plot(gg["size"], gg["ratio"], marker="o", lw=2.4, ms=8, color=pal[i], label=lab)
    ax.axhline(1.0, ls="--", c="gray", lw=1)
    ax.set_xscale("log", base=2); ax.set_xticks(sorted(m["size"].unique()))
    ax.set_xticklabels(sorted(m["size"].unique()))
    ax.set_xlabel("matrix size  M=N=K")
    ax.set_ylabel("NVIDIA / AMD  (>1: AMD faster,  <1: NVIDIA faster)")
    ax.set_title(title)
    ax.legend(fontsize=11)
    save(fig, out, fname)

# ------------------------------------------------------------------- spin plots
def spin_plots(s, out):
    if s.empty:
        print("[no spinlock rows]"); return
    s = s.copy()
    s["locks"] = s["locks"].astype(int)
    a = agg(s, ["vendor", "variant", "map", "locks"], ["mops_per_s", "kernel_ms"])
    # carry a "bailed" flag (any abandoned>0 or correct==no in the group)
    flag = (s.assign(bad=(pd.to_numeric(s["abandoned"], errors="coerce").fillna(0) > 0) |
                         (s.get("correct", "yes").astype(str) == "no"))
              .groupby(["vendor", "variant", "map", "locks"])["bad"].any().reset_index())
    a = a.merge(flag, on=["vendor", "variant", "map", "locks"], how="left")
    a.to_csv(os.path.join(out, "spin_summary.csv"), index=False)
    print("  wrote", os.path.join(out, "spin_summary.csv"))

    bailed = a[a["bad"] == True]
    if not bailed.empty:
        print("  LIVELOCK / INCOMPLETE (the headline):")
        for _, r in bailed.iterrows():
            print(f"    {r['vendor']:14s} {r['variant']:6s}/{r['map']:8s} locks={int(r['locks'])}")

    # exp6 headline: perwarp, bars by variant x vendor, bail annotated
    head = a[a["map"] == "perwarp"]
    if not head.empty:
        order = [v for v in ["naive", "leader", "atomic"] if v in head["variant"].unique()]
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        sns.barplot(data=head, x="variant", y="mops_per_s", hue="vendor", order=order,
                    hue_order=[v for v in VENDOR_ORDER if v in head["vendor"].unique()],
                    palette=VENDOR_COLORS, ax=ax, errorbar=None)
        # annotate bailed bars
        for cont, vend in zip(ax.containers, [v for v in VENDOR_ORDER if v in head["vendor"].unique()]):
            for bar, var in zip(cont, order):
                row = head[(head["vendor"] == vend) & (head["variant"] == var)]
                if not row.empty and bool(row["bad"].iloc[0]):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), "bailed",
                            ha="center", va="bottom", color="crimson", fontsize=10, rotation=0)
        ax.set_xlabel("lock variant (map = perwarp)"); ax.set_ylabel("throughput (Mops/s)")
        ax.set_title("Spinlock headline: same code, one bails on AMD")
        ax.legend(fontsize=11, title=None)
        save(fig, out, "exp6_spinlock_headline.png")

    # exp6 contention: striped, mops vs locks, line per vendor x variant
    cont = a[a["map"] == "striped"].copy()
    if not cont.empty:
        cont["series"] = cont["vendor"] + " " + cont["variant"]
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        keys = sorted(cont["series"].unique())
        pal = sns.color_palette("colorblind", n_colors=len(keys))
        for i, key in enumerate(keys):
            g = cont[cont["series"] == key].sort_values("locks")
            ax.errorbar(g["locks"], g["mops_per_s"], yerr=g["mops_per_s_std"], marker="o",
                        capsize=3, lw=2.2, ms=7, color=pal[i], label=key)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("# locks  (left = maximum contention)"); ax.set_ylabel("throughput (Mops/s)")
        ax.set_title("Spinlock throughput vs contention")
        ax.legend(fontsize=10)
        save(fig, out, "exp6_spinlock_contention.png")

# --------------------------------------------------------------------------- main
def main(argv):
    ap = argparse.ArgumentParser(description="Seaborn plots for the H100 vs MI300A study.")
    ap.add_argument("inputs", nargs="*", default=["results/experiments"],
                    help="CSV files, globs, or a folder (default: results/experiments)")
    ap.add_argument("--out", default=None, help="output folder for figures/tables")
    args = ap.parse_args(argv)
    inputs = args.inputs or ["results/experiments"]
    out = args.out or (inputs[0] if os.path.isdir(inputs[0]) else "results/experiments")
    out = os.path.join(out, "plots")
    os.makedirs(out, exist_ok=True)

    g, s = load(inputs)
    if g.empty and s.empty:
        sys.exit("No CSVs found in: %s" % inputs)
    print("Loaded %d GEMM rows, %d spinlock rows. Writing to %s/" % (len(g), len(s), out))
    if not g.empty:
        print("Vendors (GEMM):", ", ".join(sorted(g["vendor"].unique())))
    gemm_plots(g, out)
    spin_plots(s, out)
    print("Done.")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
