"""Run the localizer on YOUR OWN reference + search image pair.

    python predict.py --ref myref.png --search mysearch.png
    python predict.py --ref r.png --search s.png --out result.png   # save overlay
    python predict.py --ref r.png --search s.png --gt 512 488        # if you know GT

Prints the predicted center (x, y) in SEARCH-image pixels and, unless --no_overlay,
saves an annotated copy of the search image with a box + crosshair at the match so
you can visually verify (there is no ground truth for a real capture).

Image requirements:
  * Reference  = high-mag (e.g. 100x) close-up of the site.
  * Search     = low-mag (e.g. 10x) wide image containing that site, shrunk by the
                 magnification ratio (default 10x -- pass --ratio if different).
  * Any size / channel count: they are read as grayscale automatically. 1000x1000
    is what the tool was designed for, but other sizes work as long as --ratio is
    the true magnification ratio between the two captures.
"""
from __future__ import annotations
import argparse
import cv2
import numpy as np
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from localize import localize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--search", required=True)
    ap.add_argument("--ratio", type=float, default=10.0, help="magnification ratio (search low-mag : ref high-mag)")
    ap.add_argument("--gt", type=float, nargs=2, metavar=("X", "Y"), help="known ground-truth center, if any")
    ap.add_argument("--out", default="prediction_overlay.png")
    ap.add_argument("--no_overlay", action="store_true")
    args = ap.parse_args()

    ref = cv2.imread(args.ref, cv2.IMREAD_GRAYSCALE)
    search = cv2.imread(args.search, cv2.IMREAD_GRAYSCALE)
    if ref is None:
        raise SystemExit(f"Could not read reference image: {args.ref}")
    if search is None:
        raise SystemExit(f"Could not read search image: {args.search}")
    print(f"reference: {ref.shape[1]}x{ref.shape[0]}   search: {search.shape[1]}x{search.shape[0]}   ratio: {args.ratio}x")

    x, y, info = localize(ref, search, nominal_ratio=args.ratio)

    print(f"\n>>> Predicted center (search px):  x = {x:.2f}   y = {y:.2f}")
    print(f"    match NCC = {info.chosen.score:.3f}   fingerprint = {info.chosen.fingerprint:.3f}   "
          f"candidates = {len(info.candidates)}   time = {info.elapsed_s*1000:.0f} ms")
    if info.full_image_fallback:
        print("    (full-image fallback fired: match may lie outside the assumed drift ROI)")

    if args.gt:
        err = float(np.hypot(x - args.gt[0], y - args.gt[1]))
        nm = err * (10000.0 / search.shape[1])  # 10 um FOV assumption for nm readout
        print(f"    error vs GT = {err:.2f} px  (~{nm:.0f} nm at a 10 um FOV)")

    if not args.no_overlay:
        vis = cv2.cvtColor(search, cv2.COLOR_GRAY2BGR)
        # box sized to the matched template (ref shrunk by the refined scale)
        bw = ref.shape[1] / max(info.chosen.scale, 1e-6)
        bh = ref.shape[0] / max(info.chosen.scale, 1e-6)
        p1 = (int(round(x - bw / 2)), int(round(y - bh / 2)))
        p2 = (int(round(x + bw / 2)), int(round(y + bh / 2)))
        cv2.rectangle(vis, p1, p2, (0, 255, 0), 2)
        cv2.drawMarker(vis, (int(round(x)), int(round(y))), (0, 255, 0),
                       cv2.MARKER_CROSS, 24, 2)
        if args.gt:
            cv2.drawMarker(vis, (int(round(args.gt[0])), int(round(args.gt[1]))),
                           (0, 0, 255), cv2.MARKER_TILTED_CROSS, 24, 2)  # red = GT
        cv2.imwrite(args.out, vis)
        print(f"\n    overlay saved -> {args.out}  (green box/cross = prediction"
              + (", red = ground truth)" if args.gt else ")"))


if __name__ == "__main__":
    main()
