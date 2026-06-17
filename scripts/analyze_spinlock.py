#!/usr/bin/env python3
"""Analyze spinlock-microbenchmark CSVs: summary table + (optional) plots.

Usage:
  python3 scripts/analyze_spinlock.py results/spin_h100.csv results/spin_mi300a.csv
  python3 scripts/analyze_spinlock.py results/spin*.csv --out results

Reads rows with columns:
  backend,device,variant,map,threads,locks,work,critsize,maxspin,
  kernel_ms,mops_per_s,correct,abandoned

Writes results/spinlock_summary.md and (if matplotlib is available) PNGs:
  spin_throughput_vs_contention.png  (B: Mops/s vs #locks)
  spin_headline.png                  (A: variant x backend, perwarp)
  spin_abandoned.png                 (livelock signal: abandoned count)
"""
import csv, glob, os, sys

def vendor_of(backend):
    b = (backend or "").upper()
    if b == "CUDA": return "NVIDIA"
    if b == "HIP":  return "AMD"
    return "CPU"

def load(paths):
    rows = []
    for p in paths:
        for q in (glob.glob(p) or [p]):
            if not os.path.isfile(q): continue
            with open(q) as f:
                for d in csv.DictReader(f):
                    if "variant" not in d or "mops_per_s" not in d:
                        continue  # skip non-spinlock CSVs
                    try:
                        d["threads"]   = int(d["threads"]);  d["locks"] = int(d["locks"])
                        d["work"]      = int(d["work"]);      d["critsize"] = int(d["critsize"])
                        d["kernel_ms"] = float(d["kernel_ms"])
                        d["mops_per_s"]= float(d["mops_per_s"])
                        d["abandoned"] = int(d["abandoned"])
                    except Exception:
                        continue
                    d["vendor"] = vendor_of(d["backend"])
                    rows.append(d)
    return rows

def summary(rows, out):
    rows = sorted(rows, key=lambda r: (r["vendor"], r["variant"], r["map"], r["locks"]))
    lines = ["# Spinlock benchmark summary", "",
             "| vendor | device | variant | map | threads | locks | work | crit | "
             "kernel ms | Mops/s | correct | abandoned |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f'| {r["vendor"]} | {r.get("device","")[:20]} | {r["variant"]} | {r["map"]} '
            f'| {r["threads"]} | {r["locks"]} | {r["work"]} | {r["critsize"]} '
            f'| {r["kernel_ms"]:.3f} | {r["mops_per_s"]:.1f} | {r.get("correct","")} '
            f'| {r["abandoned"]} |')
    md = "\n".join(lines) + "\n"
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "spinlock_summary.md"), "w") as f:
        f.write(md)
    print(md)
    # Call out any livelock/incomplete rows explicitly.
    bad = [r for r in rows if r.get("correct") == "no" or r["abandoned"] > 0]
    if bad:
        print("LIVELOCK / INCOMPLETE rows (the headline result):")
        for r in bad:
            print(f'  {r["vendor"]:6s} {r["variant"]:6s}/{r["map"]:9s} '
                  f'locks={r["locks"]:<8d} abandoned={r["abandoned"]} correct={r.get("correct")}')

def plots(rows, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plots skipped: matplotlib unavailable: {e}]")
        return

    # (B) throughput vs contention: striped map, one line per (vendor, variant).
    sweep = [r for r in rows if r["map"] == "striped"]
    if sweep:
        groups = {}
        for r in sweep:
            groups.setdefault((r["vendor"], r["variant"]), []).append((r["locks"], r["mops_per_s"]))
        for k in groups: groups[k].sort()
        plt.figure(figsize=(7,5))
        for (vend, var), pts in sorted(groups.items()):
            xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
            plt.plot(xs, ys, marker="o", label=f"{vend}/{var}")
        plt.xscale("log", base=2); plt.yscale("log")
        plt.xlabel("# locks  (left = max contention)"); plt.ylabel("Mops/s")
        plt.title("Spinlock throughput vs contention"); plt.legend(); plt.grid(True, which="both", alpha=.3)
        plt.savefig(os.path.join(out, "spin_throughput_vs_contention.png"), dpi=130, bbox_inches="tight")

    # (A) headline: perwarp map, grouped bars (variant) x vendor.
    head = [r for r in rows if r["map"] == "perwarp"]
    if head:
        vendors = sorted({r["vendor"] for r in head})
        variants = ["naive", "leader", "atomic"]
        val = {(r["vendor"], r["variant"]): r for r in head}
        import numpy as np
        x = np.arange(len(variants)); w = 0.8/max(1,len(vendors))
        plt.figure(figsize=(7,5))
        for i, vend in enumerate(vendors):
            ys = [val.get((vend, v), {}).get("mops_per_s", 0.0) for v in variants]
            bars = plt.bar(x + i*w, ys, w, label=vend)
            for v, b in zip(variants, bars):
                r = val.get((vend, v))
                if r and (r.get("correct") == "no" or r["abandoned"] > 0):
                    plt.text(b.get_x()+b.get_width()/2, b.get_height(), "bailed",
                             ha="center", va="bottom", fontsize=8, color="crimson")
        plt.xticks(x + w*(len(vendors)-1)/2, variants)
        plt.ylabel("Mops/s"); plt.title("Headline: lock variant x vendor (map=perwarp)")
        plt.legend(); plt.grid(True, axis="y", alpha=.3)
        plt.savefig(os.path.join(out, "spin_headline.png"), dpi=130, bbox_inches="tight")

    # Abandoned (livelock signal) across all rows that have any.
    ab = [r for r in rows if r["abandoned"] > 0]
    if ab:
        ab.sort(key=lambda r: (r["vendor"], r["variant"], r["map"]))
        labels = [f'{r["vendor"]}\n{r["variant"]}/{r["map"]}' for r in ab]
        plt.figure(figsize=(max(7,len(ab)*1.2),5))
        plt.bar(range(len(ab)), [r["abandoned"] for r in ab], color="crimson")
        plt.xticks(range(len(ab)), labels, fontsize=8)
        plt.ylabel("abandoned critical sections"); plt.title("Livelock signal (bounded-spin bail-outs)")
        plt.grid(True, axis="y", alpha=.3)
        plt.savefig(os.path.join(out, "spin_abandoned.png"), dpi=130, bbox_inches="tight")
    print(f"[wrote plots to {out}/]")

def main(argv):
    out = "results"; paths = []; i = 0
    while i < len(argv):
        if argv[i] == "--out": out = argv[i+1]; i += 2
        else: paths.append(argv[i]); i += 1
    if not paths: paths = ["results/spin*.csv"]
    rows = load(paths)
    if not rows:
        print("no spinlock rows found in:", paths); return 1
    summary(rows, out)
    plots(rows, out)
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
