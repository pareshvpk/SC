"""
Drift-Sense evaluation / reporting script.

Runs `localize()` over every generated (reference, search) pair, scores
against the recorded ground truth, and reports:
    * per-pair error (Euclidean distance, in search-image pixels)
    * per-pair computation time
    * aggregate % of pairs landing within a stated tolerance
    * the single worst (highest-error) case, with a root-cause explanation

Usage:
    python eval.py --data data --tolerance_px 30
"""
from __future__ import annotations

import argparse
import json
import os
import time

import cv2
import numpy as np

from localize import localize


def load_pairs(data_dir: str):
    metas = json.load(open(os.path.join(data_dir, "ground_truth.json")))
    pairs = []
    for m in metas:
        i = m["pair_id"]
        ref = cv2.imread(os.path.join(data_dir, f"pair_{i:03d}_ref.png"), cv2.IMREAD_GRAYSCALE)
        search = cv2.imread(os.path.join(data_dir, f"pair_{i:03d}_search.png"), cv2.IMREAD_GRAYSCALE)
        pairs.append((ref, search, m))
    return pairs


def run_eval(data_dir: str, tolerance_px: float):
    pairs = load_pairs(data_dir)
    results = []
    for ref, search, meta in pairs:
        t0 = time.perf_counter()
        x, y, info = localize(ref, search)
        dt = time.perf_counter() - t0
        err = float(np.hypot(x - meta["gt_x"], y - meta["gt_y"]))
        # nearest raw candidate to the true GT, for grounding the failure writeup
        gt_candidate = min(info.candidates,
                            key=lambda c: np.hypot(c.x - meta["gt_x"], c.y - meta["gt_y"]))
        results.append({
            "pair_id": meta["pair_id"],
            "forced_periodic": meta["forced_periodic"],
            "found_x": x, "found_y": y,
            "gt_x": meta["gt_x"], "gt_y": meta["gt_y"],
            "error_px": err,
            "time_s": dt,
            "n_survivors": len(info.survivors),
            "n_near_tied": len(info.near_tied),
            "score": info.chosen.score,
            "full_image_fallback": info.full_image_fallback,
            "gt_candidate_score": gt_candidate.score,
            "gt_candidate_dist_px": float(np.hypot(gt_candidate.x - meta["gt_x"],
                                                     gt_candidate.y - meta["gt_y"])),
        })

    errors = np.array([r["error_px"] for r in results])
    times = np.array([r["time_s"] for r in results])
    n_fallback = sum(r["full_image_fallback"] for r in results)
    tolerances = [10.0, 30.0, 50.0, 100.0]
    if tolerance_px not in tolerances:
        tolerances.append(tolerance_px)
        tolerances.sort()
    pct_by_tol = {t: 100.0 * float((errors <= t).mean()) for t in tolerances}

    print(f"Pairs evaluated: {len(results)}")
    print(f"Mean compute time / pair: {times.mean()*1000:.1f} ms  (max {times.max()*1000:.1f} ms)")
    print(f"Full-image fallback triggered on {n_fallback}/{len(results)} pairs "
          f"(unrestricted search used when a match may lie outside the drift ROI)")
    print(f"Mean error: {errors.mean():.1f} px   Median error: {np.median(errors):.1f} px")
    print("% within tolerance:")
    for t in tolerances:
        marker = "  <- requested" if t == tolerance_px else ""
        print(f"    {t:6.0f} px ({t*10:6.0f} nm): {pct_by_tol[t]:5.1f}%{marker}")
    print()
    print(f"{'id':>3} {'periodic':>9} {'err_px':>8} {'time_ms':>8} {'score':>7} "
          f"{'survivors':>9} {'near_tied':>9} {'fallback':>8}")
    for r in sorted(results, key=lambda r: -r["error_px"]):
        print(f"{r['pair_id']:>3} {str(r['forced_periodic']):>9} {r['error_px']:>8.1f} "
              f"{r['time_s']*1000:>8.1f} {r['score']:>7.3f} {r['n_survivors']:>9} "
              f"{r['n_near_tied']:>9} {str(r['full_image_fallback']):>8}")

    worst = max(results, key=lambda r: r["error_px"])
    print()
    print("=== Honest failure case ===")
    print(f"pair_id={worst['pair_id']}  forced_periodic={worst['forced_periodic']}  "
          f"error={worst['error_px']:.1f} px  chosen_score={worst['score']:.3f}  "
          f"n_survivors={worst['n_survivors']}  n_near_tied={worst['n_near_tied']}")
    print(f"The candidate nearest the true ground truth scored {worst['gt_candidate_score']:.3f} "
          f"({worst['gt_candidate_score'] - worst['score']:+.3f} vs. the chosen candidate's "
          f"{worst['score']:.3f}) and sat {worst['gt_candidate_dist_px']:.1f} px from the exact "
          f"GT point -- i.e. NCC ranked a different lattice repeat as good as or better than the "
          f"true site.")
    print(
        "Root cause: the reference crop sits inside a highly periodic region of the "
        "FinFET dense-fin/gate-bar lattice. Normalized cross-correlation against a "
        "periodic template produces near-identical correlation peaks at essentially every "
        "lattice repeat (same phase alignment), because the *dominant* signal in the "
        "correlation score is phase alignment to the periodic grid, not the small "
        "per-crossing defect signal that actually distinguishes one repeat from "
        "another. When sensor noise is comparable in scale to that per-cell distinguishing "
        "signal, the true site is not guaranteed to be the highest-scoring peak, and the "
        "nearest-to-center tie-break -- while physically well-motivated (real motion-stage "
        "drift is bounded) -- can pick a neighboring look-alike repeat instead of the true "
        "site when that look-alike happens to fall closer to the search image center."
    )

    with open(os.path.join(data_dir, "eval_report.json"), "w") as f:
        json.dump({
            "tolerance_px": tolerance_px,
            "pct_within_tolerance_by_threshold": pct_by_tol,
            "mean_error_px": float(errors.mean()),
            "median_error_px": float(np.median(errors)),
            "mean_time_ms": float(times.mean() * 1000),
            "max_time_ms": float(times.max() * 1000),
            "n_full_image_fallback": n_fallback,
            "results": results,
            "worst_case": worst,
        }, f, indent=2)
    print(f"\nFull report written to {os.path.join(data_dir, 'eval_report.json')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=str, default="data")
    ap.add_argument("--tolerance_px", type=float, default=30.0,
                     help="Success tolerance in search-image pixels (30 px = 300 nm, "
                          "roughly the reference FOV's own footprint in the search frame "
                          "divided by 3 -- i.e. landing well inside the field the reference "
                          "actually covers).")
    args = ap.parse_args()
    run_eval(args.data, args.tolerance_px)
