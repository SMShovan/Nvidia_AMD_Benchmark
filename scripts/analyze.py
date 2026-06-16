#!/usr/bin/env python3
"""Analyze gemm_bench CSVs: summary table + (optional) plots.

Usage:
  python3 scripts/analyze.py results/h100.csv results/mi300a.csv
  python3 scripts/analyze.py results/*.csv --out results

Reads rows with columns:
  backend,device,kernel,dtype,M,N,K,kernel_ms,e2e_ms,gflops,gbps,verified,max_rel

Writes results/summary.md and (if matplotlib is available) PNG plots:
  gflops_vs_size.png, pct_peak_vs_size.png, e2e_overhead_vs_size.png
"""
import csv, glob, os, sys

# ----------------------------------------------------------------------------
# PEAK throughput references (TFLOP/s). EDIT to match YOUR exact parts (SXM vs
# PCIe, MI300A clock) from the vendor white papers. Keyed by (vendor, klass, dtype).
#   klass: "vector" = regular cores (tiled kernel); "matrix" = tensor/matrix cores.
# These are approximate dense figures and only drive the optional %-of-peak column.
PEAKS_TFLOPS = {
    ("NVIDIA", "vector", "fp32"): 66.9,    # H100 SXM FP32 (CUDA cores)
    ("NVIDIA", "vector", "fp16"): 133.8,   # FP16 on CUDA cores (~2x fp32)
    ("NVIDIA", "matrix", "fp16"): 989.4,   # H100 SXM FP16 Tensor Core, fp32 acc, dense
    ("AMD",    "vector", "fp32"): 122.6,   # MI300A FP32
    ("AMD",    "vector", "fp16"): 245.2,   # FP16 on vector ALUs (~2x fp32)
    ("AMD",    "matrix", "fp16"): 980.6,   # MI300A FP16 Matrix Core, fp32 acc, dense
}

def vendor_of(backend, device):
    b = (backend or "").upper()
    if b == "CUDA": return "NVIDIA"
    if b == "HIP":  return "AMD"
    return "CPU"

def klass_of(kernel):
    return "matrix" if kernel == "matrix" else "vector"

def load(paths):
    rows = []
    for p in paths:
        for q in (glob.glob(p) or [p]):
            if not os.path.isfile(q): continue
            with open(q) as f:
                for d in csv.DictReader(f):
                    try:
                        d["M"] = int(d["M"]); d["N"] = int(d["N"]); d["K"] = int(d["K"])
                        d["kernel_ms"] = float(d["kernel_ms"]); d["e2e_ms"] = float(d["e2e_ms"])
                        d["gflops"] = float(d["gflops"])
                    except Exception:
                        continue
                    rows.append(d)
    return rows

def pct_peak(r):
    v = vendor_of(r["backend"], r.get("device"))
    key = (v, klass_of(r["kernel"]), r["dtype"])
    peak = PEAKS_TFLOPS.get(key)
    if not peak or r["gflops"] <= 0: return None
    return 100.0 * (r["gflops"] / 1000.0) / peak

def summary(rows, out):
    rows = [r for r in rows if r["kernel"] in ("tiled", "matrix")]
    rows.sort(key=lambda r: (r["backend"], r["kernel"], r["dtype"], r["M"]))
    lines = ["# GEMM benchmark summary", "",
             "| backend | device | kernel | dtype | M=N=K | kernel ms | GFLOP/s | % peak | e2e/kernel | verified |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        sq = r["M"] if (r["M"] == r["N"] == r["K"]) else f'{r["M"]}x{r["N"]}x{r["K"]}'
        pp = pct_peak(r); pps = f"{pp:.1f}%" if pp is not None else "-"
        ratio = (r["e2e_ms"] / r["kernel_ms"]) if r["kernel_ms"] > 0 else 0.0
        lines.append(f'| {r["backend"]} | {r.get("device","")[:24]} | {r["kernel"]} | {r["dtype"]} '
                     f'| {sq} | {r["kernel_ms"]:.3f} | {r["gflops"]:.1f} | {pps} | {ratio:.2f}x | {r.get("verified","")} |')
    md = "\n".join(lines) + "\n"
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "summary.md"), "w") as f: f.write(md)
    print(md)

def plots(rows, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plots skipped: matplotlib unavailable: {e}]")
        return
    sq = [r for r in rows if r["M"] == r["N"] == r["K"] and r["kernel"] in ("tiled", "matrix")]
    def series(rs, y):
        groups = {}
        for r in rs:
            k = (r["backend"], r["kernel"], r["dtype"])
            groups.setdefault(k, []).append((r["M"], y(r)))
        for k in groups: groups[k].sort()
        return groups
    # 1) GFLOP/s vs size
    g = series(sq, lambda r: r["gflops"])
    plt.figure(figsize=(7,5))
    for k, pts in g.items():
        xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
        plt.plot(xs, ys, marker="o", label=f"{k[0]}/{k[1]}/{k[2]}")
    plt.xlabel("M=N=K"); plt.ylabel("GFLOP/s"); plt.title("GEMM throughput"); plt.legend(); plt.grid(True, alpha=.3)
    plt.savefig(os.path.join(out, "gflops_vs_size.png"), dpi=130, bbox_inches="tight")
    # 2) % peak vs size
    gp = series([r for r in sq if pct_peak(r) is not None], lambda r: pct_peak(r))
    if gp:
        plt.figure(figsize=(7,5))
        for k, pts in gp.items():
            xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
            plt.plot(xs, ys, marker="o", label=f"{k[0]}/{k[1]}/{k[2]}")
        plt.xlabel("M=N=K"); plt.ylabel("% of peak"); plt.title("Efficiency"); plt.legend(); plt.grid(True, alpha=.3)
        plt.savefig(os.path.join(out, "pct_peak_vs_size.png"), dpi=130, bbox_inches="tight")
    # 3) end-to-end overhead (memcpy vs unified memory)
    go = series(sq, lambda r: (r["e2e_ms"]/r["kernel_ms"]) if r["kernel_ms"]>0 else 0.0)
    plt.figure(figsize=(7,5))
    for k, pts in go.items():
        xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
        plt.plot(xs, ys, marker="o", label=f"{k[0]}/{k[1]}/{k[2]}")
    plt.xlabel("M=N=K"); plt.ylabel("end-to-end / kernel time"); plt.title("Transfer overhead (1.0 = none)")
    plt.legend(); plt.grid(True, alpha=.3)
    plt.savefig(os.path.join(out, "e2e_overhead_vs_size.png"), dpi=130, bbox_inches="tight")
    print(f"[wrote plots to {out}/]")

def main(argv):
    out = "results"
    paths = []
    i = 0
    while i < len(argv):
        if argv[i] == "--out": out = argv[i+1]; i += 2
        else: paths.append(argv[i]); i += 1
    if not paths: paths = ["results/*.csv"]
    rows = load(paths)
    if not rows:
        print("no rows found in:", paths); return 1
    summary(rows, out)
    plots(rows, out)
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
