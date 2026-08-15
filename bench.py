"""
V1 vs V2 benchmark for the Drift-Sense localizer.

Requires `localize_v1.py` (snapshot of the previous version) and the current
`localize.py` (V2) to both expose `localize(ref, search) -> (x, y, info)`.
Runs both over every pair in the dataset and prints a comparison table.

    python bench.py --data data

Pixel-to-physical conversion: the search image is 10 nm/px, so an error of
N px corresponds to 10*N nm. "within 1 um" == error <= 100 px.
"""
from __future__ import annotations
import argparse, json, os, time
import cv2
import numpy as np

import localize as v2
try:
    import localize_v1 as v1
    HAVE_V1 = True
except Exception:
    HAVE_V1 = False

SEARCH_NM_PER_PX = 10.0


def load(data_dir):
    metas = json.load(open(os.path.join(data_dir, "ground_truth.json")))
    out = []
    for m in metas:
        i = m["pair_id"]
        ref = cv2.imread(os.path.join(data_dir, f"pair_{i:03d}_ref.png"), cv2.IMREAD_GRAYSCALE)
        search = cv2.imread(os.path.join(data_dir, f"pair_{i:03d}_search.png"), cv2.IMREAD_GRAYSCALE)
        out.append((ref, search, m))
    return out


def run_module(mod, pairs):
    errs, times = [], []
    for ref, search, m in pairs:
        t = time.perf_counter()
        x, y, _ = mod.localize(ref, search)
        times.append(time.perf_counter() - t)
        errs.append(float(np.hypot(x - m["gt_x"], y - m["gt_y"])))
    return np.array(errs), np.array(times)


def summary(name, errs, times):
    within_um = 100.0  # px, = 1 um at 10 nm/px
    return {
        "name": name,
        "median_px": float(np.median(errs)),
        "mean_px": float(errs.mean()),
        "max_px": float(errs.max()),
        "pct_1px": 100.0 * float((errs <= 1).mean()),
        "pct_10px": 100.0 * float((errs <= 10).mean()),
        "pct_100px": 100.0 * float((errs <= within_um).mean()),
        "pct_1um": 100.0 * float((errs <= within_um).mean()),
        "catastrophic": int((errs > 100).sum()),
        "time_s": float(times.mean()),
    }


def print_table(rows):
    cols = [("name", "algorithm", 26), ("median_px", "median px", 10),
            ("mean_px", "mean px", 9), ("max_px", "max px", 9),
            ("pct_1px", "<1px %", 8), ("pct_10px", "<10px %", 8),
            ("pct_100px", "<100px(1um) %", 14),
            ("catastrophic", ">100px #", 9), ("time_s", "sec/pair", 9)]
    header = "".join(f"{h:>{w}}" if k != "name" else f"{h:<{w}}" for k, h, w in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        line = ""
        for k, _h, w in cols:
            v = r[k]
            if k == "name":
                line += f"{v:<{w}}"
            elif isinstance(v, float):
                line += f"{v:>{w}.1f}"
            else:
                line += f"{v:>{w}}"
        print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    pairs = load(args.data)

    rows = []
    if HAVE_V1:
        e1, t1 = run_module(v1, pairs)
        rows.append(summary("V1 (NCC + center prior)", e1, t1))
    e2, t2 = run_module(v2, pairs)
    rows.append(summary("V2 (generate + verify)", e2, t2))

    print_table(rows)

    if HAVE_V1:
        print("\nPer-pair (px error), sorted by V2-vs-V1 improvement:")
        deltas = sorted(zip([m["pair_id"] for _, _, m in pairs],
                            [m["forced_periodic"] for _, _, m in pairs], e1, e2),
                        key=lambda z: (z[3] - z[2]))
        print(f"{'id':>3} {'periodic':>9} {'V1_err':>8} {'V2_err':>8}")
        for pid, fp, a, b in deltas:
            print(f"{pid:>3} {str(fp):>9} {a:>8.1f} {b:>8.1f}")
