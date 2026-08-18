#!/usr/bin/env python3
"""Demo / evaluation report — the tables and figures for a walkthrough video.

For every category (structure x noise level) it generates N labelled pairs, runs
the localiser, and prints:

  1. a PER-IMAGE table for one chosen category (image, measured scale, predicted
     x / y, error px, time ms, confidence),
  2. a CATEGORY table (dram/finfet/mixed x low/medium/high): n, <=5px %, <=10px %,
     mean px, median px, mean ms,
  3. an OVERALL summary row (accuracy % and mean latency),

writes everything to <out>/demo_report.csv, and saves reference | search | located
triptych figures to <out>/figures/ for the on-screen "where the reference goes" shots.

    python tools/demo_report.py --out demo_out --n 10

Notes
-----
Columns are this project's own quantities (measured magnification, predicted x/y,
Euclidean error, per-pair time, selection confidence). It does NOT reproduce
another team's custom columns (e.g. "sabase / drbase / cov%"), whose definitions
are not published.
"""
from __future__ import annotations
import argparse, csv, os, sys, time
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from localize import localize                      # noqa: E402
import dataset_gen as G                            # noqa: E402

STYLES = ["dram", "finfet", "mixed"]
LEVELS = ["low", "medium", "high"]


def gen_category(style, level, n, seed):
    """Generate n pairs of one (style, level) in memory. Returns list of dicts."""
    pairs = []
    for i in range(n):
        pair_style = style
        if style == "mixed":
            pair_style = "finfet" if i % 2 == 0 else "dram"
        ref, search, meta = G.generate_pair(i, seed + i, style=pair_style,
                                            noise_level=level, superstructure=True)
        pairs.append(dict(ref=ref, search=search, gt_x=meta.gt_x, gt_y=meta.gt_y,
                          style=pair_style))
    return pairs


def run_category(pairs):
    rows = []
    for k, p in enumerate(pairs):
        t0 = time.perf_counter()
        x, y, info = localize(p["ref"], p["search"])
        ms = (time.perf_counter() - t0) * 1000
        err = float(np.hypot(x - p["gt_x"], y - p["gt_y"]))
        ncc = float(info.chosen.score) if info.chosen is not None else 0.0
        rows.append(dict(image=k, style=p["style"], scale=float(info.magnification),
                         x=float(x), y=float(y), gt_x=p["gt_x"], gt_y=p["gt_y"],
                         error_px=err, time_ms=ms, ncc=ncc))
    return rows


def agg(rows):
    e = np.array([r["error_px"] for r in rows]); t = np.array([r["time_ms"] for r in rows])
    return dict(n=len(rows),
                le5=100.0 * float((e <= 5).mean()), le10=100.0 * float((e <= 10).mean()),
                n_le5=int((e <= 5).sum()), n_le10=int((e <= 10).sum()),
                mean_px=float(e.mean()), median_px=float(np.median(e)),
                p95_px=float(np.percentile(e, 95)), mean_ms=float(t.mean()))


def triptych(ref, search, x, y, gt_x, gt_y, path, title):
    """reference | search | search-with-located-box, side by side, labelled."""
    def to_bgr(img):
        g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    H = 360
    def fit(img):
        b = to_bgr(img); h, w = b.shape[:2]
        return cv2.resize(b, (int(w * H / h), H), interpolation=cv2.INTER_AREA)
    ref_p = fit(ref)
    loc = to_bgr(search); h, w = loc.shape[:2]
    box = int(0.05 * w)
    cv2.rectangle(loc, (int(x - box), int(y - box)), (int(x + box), int(y + box)), (60, 220, 60), 3)
    cv2.drawMarker(loc, (int(x), int(y)), (60, 220, 60), cv2.MARKER_CROSS, 26, 3)
    cv2.drawMarker(loc, (int(gt_x), int(gt_y)), (60, 60, 230), cv2.MARKER_TILTED_CROSS, 20, 2)
    loc = cv2.resize(loc, (int(w * H / h), H), interpolation=cv2.INTER_AREA)
    search_p = fit(search)
    gap = np.full((H, 14, 3), 255, np.uint8)
    strip = np.hstack([ref_p, gap, search_p, gap, loc])
    header = np.full((44, strip.shape[1], 3), 255, np.uint8)
    w0 = ref_p.shape[1]; w1 = search_p.shape[1]; g = gap.shape[1]
    centers = [w0 // 2, w0 + g + w1 // 2, w0 + g + w1 + g + loc.shape[1] // 2]
    for label, cx, scale in [("REFERENCE", centers[0], 0.6), ("SEARCH", centers[1], 0.6),
                             ("LOCATED  green=pred red=truth", centers[2], 0.48)]:
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        x0 = int(np.clip(cx - tw // 2, 2, strip.shape[1] - tw - 2))
        cv2.putText(header, label, (x0, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (20, 20, 20), 1, cv2.LINE_AA)
    out = np.vstack([header, strip])
    cv2.putText(out, title, (6, out.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (140, 60, 20), 1, cv2.LINE_AA)
    cv2.imwrite(path, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo_out")
    ap.add_argument("--n", type=int, default=10, help="pairs per category")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--detail", default="mixed/medium",
                    help="category shown in the per-image table, e.g. finfet/high")
    ap.add_argument("--figures", type=int, default=3, help="triptych figures to save")
    a = ap.parse_args()
    figdir = os.path.join(a.out, "figures"); os.makedirs(figdir, exist_ok=True)

    all_rows, cat_summ = [], []
    detail_rows = None
    dstyle, dlevel = a.detail.split("/")
    seed = a.seed
    for style in STYLES:
        for level in LEVELS:
            pairs = gen_category(style, level, a.n, seed); seed += 1000
            rows = run_category(pairs)
            for r in rows:
                r["category"] = f"{style}/{level}"; all_rows.append(r)
            cat_summ.append((f"{style}/{level}", agg(rows)))
            if style == dstyle and level == dlevel:
                detail_rows = rows
                for k in range(min(a.figures, len(pairs))):
                    triptych(pairs[k]["ref"], pairs[k]["search"], rows[k]["x"], rows[k]["y"],
                             pairs[k]["gt_x"], pairs[k]["gt_y"],
                             os.path.join(figdir, f"{style}_{level}_img{k}.png"),
                             f"{style}/{level}  img{k}   err={rows[k]['error_px']:.2f}px   {rows[k]['time_ms']:.0f}ms")

    # ---------- 1. per-image table ----------
    print(f"\n=== PER-IMAGE RESULTS  ({a.detail}, {len(detail_rows)} images) ===")
    print(f"{'image':>5} {'style':>6} {'scale':>6} {'x':>8} {'y':>8} {'error_px':>9} {'time_ms':>8} {'ncc':>5}")
    for r in detail_rows:
        print(f"{r['image']:>5} {r['style']:>6} {r['scale']:>6.2f} {r['x']:>8.2f} {r['y']:>8.2f} "
              f"{r['error_px']:>9.2f} {r['time_ms']:>8.0f} {r['ncc']:>5.2f}")

    # ---------- 2. category table ----------
    print(f"\n=== CATEGORY SUMMARY  ({a.n} pairs each) ===")
    hdr = (f"{'category':>14} {'<=5px':>7} {'rate':>5} {'<=10px':>7} {'mean_px':>8} "
           f"{'median':>7} {'p95':>7} {'mean_ms':>8}")
    print(hdr)
    for name, s in cat_summ:
        print(f"{name:>14} {s['n_le5']:>3}/{s['n']:<3} {s['le5']:>4.0f}% "
              f"{s['n_le10']:>3}/{s['n']:<3} {s['mean_px']:>8.2f} "
              f"{s['median_px']:>7.2f} {s['p95_px']:>7.2f} {s['mean_ms']:>8.0f}")

    # ---------- 3. overall row ----------
    o = agg(all_rows)
    print("-" * len(hdr))
    print(f"{'ALL pairs':>14} {o['n_le5']:>3}/{o['n']:<3} {o['le5']:>4.0f}% "
          f"{o['n_le10']:>3}/{o['n']:<3} {o['mean_px']:>8.2f} "
          f"{o['median_px']:>7.2f} {o['p95_px']:>7.2f} {o['mean_ms']:>8.0f}")
    print(f"\nACCURACY: {o['le5']:.0f}% within 5 px ({o['le5']:.0f}% within 50 nm) | "
          f"median {o['median_px']:.2f} px | LATENCY: mean {o['mean_ms']:.0f} ms/pair")

    with open(os.path.join(a.out, "demo_report.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["category", "image", "style", "scale", "x", "y",
                                          "gt_x", "gt_y", "error_px", "time_ms", "ncc"])
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"\nrows -> {a.out}/demo_report.csv    figures -> {figdir}/")


if __name__ == "__main__":
    main()
