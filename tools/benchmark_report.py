#!/usr/bin/env python3
"""Reproducible benchmark + chart generator for the README.

Generates a 200-pair evaluation set spanning BOTH structures (FinFET, DRAM) x
fixed-10x and variable-magnification (9x-11x) acquisition -- each pair carrying
independent per-capture noise, blur, rotation and a random ground-truth
position -- then runs the localiser over all of it and renders the result
charts embedded in the README.

    python tools/benchmark_report.py --out data_bench

Charts are written to docs/images/{passrate,breakdown,error_cdf,latency}.png
and the aggregate numbers are printed as JSON. Re-running with an existing
--out dir reuses the generated images (only re-runs the localiser + charts).
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from localize import localize  # noqa: E402

# ---- validated categorical palette (dataviz skill reference instance) --------
SURFACE = "#fcfcfb"; INK = "#141413"; INK2 = "#52514e"; GRID = "#e6e5e1"
BLUE = "#2a78d6"; AQUA = "#1baf7a"; AMBER = "#eda100"; GOOD = "#0ca30c"; CRIT = "#d03b3b"

SPLITS = [  # (label, style, mag_jitter, seed)
    ("FinFET",         "finfet", False, 100),
    ("FinFET var-mag", "finfet", True,  200),
    ("DRAM",           "dram",   False, 300),
    ("DRAM var-mag",   "dram",   True,  400),
]
# tolerance bands: (px, label). 10 nm/px in the search frame.
BANDS = [(5, "50 nm\n5 px"), (10, "100 nm\n10 px"), (30, "300 nm\n30 px"),
         (50, "500 nm\n50 px"), (100, "1 µm\n100 px")]


def _style_ax(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, length=0, labelsize=10)
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)


def generate(out, n=50):
    for label, style, mag, seed in SPLITS:
        d = os.path.join(out, style + ("_mag" if mag else ""))
        if os.path.exists(os.path.join(d, "ground_truth.json")):
            continue
        cmd = [sys.executable, "src/dataset_gen.py", "--style", style,
               "--n", str(n), "--out", d, "--seed", str(seed)]
        if mag:
            cmd.append("--mag-jitter")
        print("generating", d, file=sys.stderr)
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def run(out):
    # Preload every pair into memory first so disk I/O is never inside the timed
    # region, and warm the OpenCV/NumPy allocators, then time the localiser itself.
    items = []  # (label, ref, search, gt_x, gt_y)
    for label, style, mag, seed in SPLITS:
        d = os.path.join(out, style + ("_mag" if mag else ""))
        for m in json.load(open(os.path.join(d, "ground_truth.json"))):
            i = m["pair_id"]
            ref = cv2.imread(os.path.join(d, f"pair_{i:03d}_ref.png"), cv2.IMREAD_UNCHANGED)
            se = cv2.imread(os.path.join(d, f"pair_{i:03d}_search.png"), cv2.IMREAD_UNCHANGED)
            items.append((label, ref, se, m["gt_x"], m["gt_y"]))
    for _, ref, se, _, _ in items[:5]:
        localize(ref, se)  # warm up (allocators, code paths)
    rows = []  # (label, err, ms)
    for label, ref, se, gx, gy in items:
        best_ms, x, y = None, None, None
        for _ in range(2):  # min-of-2: reject transient OS/CPU scheduling spikes
            t0 = time.perf_counter()
            x, y, _ = localize(ref, se)
            ms = (time.perf_counter() - t0) * 1000
            best_ms = ms if best_ms is None else min(best_ms, ms)
        err = float(np.hypot(x - gx, y - gy))
        rows.append((label, err, best_ms))
    return rows


def pct(errs, tol):
    return 100.0 * float((np.asarray(errs) <= tol).mean())


def chart_passrate(errs, path):
    fig, ax = plt.subplots(figsize=(7.6, 3.9), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    vals = [pct(errs, t) for t, _ in BANDS]
    xs = np.arange(len(BANDS))
    ax.bar(xs, vals, width=0.62, color=BLUE, zorder=3)
    for x, v in zip(xs, vals):
        ax.text(x, v + 1.5, f"{v:.1f}%", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=INK)
    ax.set_xticks(xs); ax.set_xticklabels([b for _, b in BANDS])
    ax.set_ylim(0, 108); ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.set_major_formatter(PercentFormatter())
    _style_ax(ax)
    ax.set_title(f"Localisation accuracy — {len(errs)} pairs (FinFET + DRAM)",
                 fontsize=12.5, fontweight="bold", color=INK, pad=12, loc="left")
    ax.set_ylabel("pairs within tolerance", color=INK2, fontsize=10)
    ax.set_xlabel("error tolerance  (search-frame pixels / physical distance)", color=INK2, fontsize=10)
    fig.tight_layout(); fig.savefig(path, facecolor=SURFACE); plt.close(fig)


def chart_breakdown(rows, path):
    # grouped bars: x = FinFET/DRAM structure, series = fixed vs var-mag; % within 300 nm
    struct = ["FinFET", "DRAM"]
    fixed = [pct([e for l, e, _ in rows if l == s], 30) for s in struct]
    varm = [pct([e for l, e, _ in rows if l == s + " var-mag"], 30) for s in struct]
    xs = np.arange(len(struct)); w = 0.34
    fig, ax = plt.subplots(figsize=(7.6, 3.9), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    b1 = ax.bar(xs - w/2, fixed, w, color=BLUE, label="fixed 10×", zorder=3)
    b2 = ax.bar(xs + w/2, varm, w, color=AQUA, label="variable 9–11×", zorder=3)
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width()/2, r.get_height() + 1.4,
                    f"{r.get_height():.0f}%", ha="center", va="bottom",
                    fontsize=10.5, fontweight="bold", color=INK)
    ax.set_xticks(xs); ax.set_xticklabels(struct, fontsize=11)
    ax.set_ylim(0, 112); ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.set_major_formatter(PercentFormatter())
    _style_ax(ax)
    ax.set_title("Accuracy by structure and magnification",
                 fontsize=12.5, fontweight="bold", color=INK, pad=12, loc="left")
    ax.set_ylabel("pairs within 300 nm (30 px)", color=INK2, fontsize=10)
    leg = ax.legend(loc="lower center", ncol=2, frameon=False, fontsize=10,
                    bbox_to_anchor=(0.5, -0.30))
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout(); fig.savefig(path, facecolor=SURFACE); plt.close(fig)


def chart_cdf(errs, path):
    e = np.sort(np.clip(np.asarray(errs), 0.02, None))
    cum = 100.0 * np.arange(1, len(e) + 1) / len(e)
    med = float(np.median(errs))
    fig, ax = plt.subplots(figsize=(7.6, 3.9), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.plot(e, cum, color=BLUE, linewidth=2.2, zorder=4)
    ax.fill_between(e, cum, color=BLUE, alpha=0.08, zorder=2)
    ax.axvline(30, color=INK2, linewidth=1, linestyle=(0, (4, 3)), zorder=3)
    ax.text(30, 6, " 300 nm success\n footprint", color=INK2, fontsize=9, va="bottom")
    ax.plot([med], [50], "o", color=AMBER, markersize=8, zorder=5)
    ax.annotate(f"median {med:.2f} px", (med, 50), textcoords="offset points",
                xytext=(10, -4), fontsize=10, fontweight="bold", color=INK)
    ax.set_xscale("log")
    ax.set_xlim(0.05, 1000); ax.set_ylim(0, 103)
    ax.set_xticks([0.1, 1, 10, 100, 1000])
    ax.set_xticklabels(["0.1", "1", "10", "100", "1000"])
    ax.yaxis.set_major_formatter(PercentFormatter())
    _style_ax(ax)
    ax.grid(axis="x", color=GRID, linewidth=1, zorder=0)
    ax.set_title("Error distribution — most revisits are sub-pixel",
                 fontsize=12, fontweight="bold", color=INK, pad=12, loc="left")
    ax.set_xlabel("localisation error  (search-frame pixels, log scale)", color=INK2, fontsize=10)
    ax.set_ylabel("cumulative % of pairs", color=INK2, fontsize=10)
    fig.tight_layout(); fig.savefig(path, facecolor=SURFACE); plt.close(fig)


def chart_latency(times, path):
    t = np.asarray(times)
    fig, ax = plt.subplots(figsize=(7.6, 3.9), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.hist(t, bins=24, color=BLUE, zorder=3)
    ax.axvline(200, color=CRIT, linewidth=1.6, zorder=4)
    ax.text(200, ax.get_ylim()[1]*0.92, " 200 ms budget", color=CRIT,
            fontsize=9.5, fontweight="bold", va="top")
    ax.axvline(t.mean(), color=GOOD, linewidth=1.6, zorder=4)
    ax.text(t.mean(), ax.get_ylim()[1]*0.72, f" mean {t.mean():.0f} ms",
            color=GOOD, fontsize=9.5, fontweight="bold", va="top")
    _style_ax(ax)
    ax.set_title(f"Inference latency — mean {t.mean():.0f} ms, max {t.max():.0f} ms  (CPU, single-threaded)",
                 fontsize=12, fontweight="bold", color=INK, pad=12, loc="left")
    ax.set_xlabel("time per pair  (ms)", color=INK2, fontsize=10)
    ax.set_ylabel("number of pairs", color=INK2, fontsize=10)
    fig.tight_layout(); fig.savefig(path, facecolor=SURFACE); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_bench")
    ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    imgdir = os.path.join(root, "docs", "images")
    os.makedirs(imgdir, exist_ok=True)
    generate(a.out, a.n)
    rows = run(a.out)
    errs = [e for _, e, _ in rows]
    times = [ms for _, _, ms in rows]
    chart_passrate(errs, os.path.join(imgdir, "bench_passrate.png"))
    chart_breakdown(rows, os.path.join(imgdir, "bench_breakdown.png"))
    chart_cdf(errs, os.path.join(imgdir, "bench_error_cdf.png"))
    chart_latency(times, os.path.join(imgdir, "bench_latency.png"))
    summary = {
        "n_pairs": len(errs),
        "median_px": round(float(np.median(errs)), 3),
        "mean_px": round(float(np.mean(errs)), 2),
        "within": {lbl: round(pct(errs, t), 1) for t, lbl in
                   [(5, "50nm"), (10, "100nm"), (30, "300nm"), (50, "500nm"), (100, "1um")]},
        "by_split": {lbl: round(pct([e for l, e, _ in rows if l == lbl], 30), 1)
                     for lbl, *_ in SPLITS},
        "sub_1px_pct": round(100.0 * float((np.asarray(errs) <= 1).mean()), 1),
        "mean_ms": round(float(np.mean(times)), 1),
        "median_ms": round(float(np.median(times)), 1),
        "max_ms": round(float(np.max(times)), 1),
        "pct_under_200ms": round(100.0 * float((np.asarray(times) <= 200).mean()), 1),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
