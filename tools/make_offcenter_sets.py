"""Generate off-center test sets by forcing the true site toward the corners.

Uses generate_pair(drift_nm=...) to plant the reference at an explicit offset
from the search-FOV center, so we can stress-test positions OUTSIDE the bounded-
drift envelope (where the center prior no longer helps and the full-image
fallback must carry the localization).

    python make_offcenter_sets.py --mode corner   # near the 4 corners (extreme)
    python make_offcenter_sets.py --mode inner     # mid-way to the corners
"""
from __future__ import annotations
import argparse, json, os
from dataclasses import asdict
import cv2, numpy as np
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from dataset_gen import generate_pair, SEARCH_FOV_NM, REF_FOV_NM

# max drift that still keeps the whole 1 um crop inside the 10 um FOV
MARGIN = REF_FOV_NM / 2 + 10
MAX_DRIFT = SEARCH_FOV_NM / 2 - MARGIN            # ~4490 nm  (-> ~px 51 or 949)
CORNERS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]


def gen_set(out, n, mag_lo, mag_hi, seed0):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(seed0)
    metas = []
    for i in range(n):
        sx, sy = CORNERS[i % 4]
        dx = sx * rng.uniform(mag_lo, mag_hi)
        dy = sy * rng.uniform(mag_lo, mag_hi)
        ref, search, meta = generate_pair(i, seed0 + i, forced_periodic=False,
                                          drift_nm=(dx, dy))
        cv2.imwrite(os.path.join(out, f"pair_{i:03d}_ref.png"), ref)
        cv2.imwrite(os.path.join(out, f"pair_{i:03d}_search.png"), search)
        metas.append(asdict(meta))
    json.dump(metas, open(os.path.join(out, "ground_truth.json"), "w"), indent=2)
    d = [np.hypot(m["gt_x"] - 500, m["gt_y"] - 500) for m in metas]
    print(f"wrote {out}: {n} pairs, distance-from-center {min(d):.0f}..{max(d):.0f} px "
          f"(mean {np.mean(d):.0f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["corner", "inner"], required=True)
    args = ap.parse_args()
    if args.mode == "corner":
        # extreme: as far into the corners as the FOV margin allows (~450 px out)
        gen_set("data_corner", 30, 0.90 * MAX_DRIFT, MAX_DRIFT, seed0=6000)
    else:
        # inner corners: about half-way between center and the corner (~300 px out)
        gen_set("data_inner", 30, 0.40 * MAX_DRIFT, 0.55 * MAX_DRIFT, seed0=6100)
