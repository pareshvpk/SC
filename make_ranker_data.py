"""Build the training/test feature tables for the hybrid ML candidate ranker.

For each generated pair we run the CLASSICAL candidate pipeline (localize with
use_ml=False), take every candidate, compute candidate_features(), and label a
candidate 1 iff it lies within LABEL_PX of the ground-truth site. The ML ranker
(train_ranker.ipynb) then learns to pick the true candidate from these features.

Output: ranker_data.npz with X_train/y_train/g_train and X_test/y_test/g_test
(+ gt tables) -- pairs are split by GROUP so no pair leaks across the split.

    python make_ranker_data.py --n_train 200 --n_test 40
"""
from __future__ import annotations
import argparse
import numpy as np

from dataset_gen import generate_pair
from localize import localize, candidate_features

LABEL_PX = 5.0
NOMINAL_RATIO, ROT_MAX_DEG, MAX_DRIFT_FRAC = 10.0, 4.0, 0.24


def build(seeds, forced_frac, rng):
    Xs, ys, gs, gts = [], [], [], []
    for gi, seed in enumerate(seeds):
        forced = rng.random() < forced_frac
        ref, search, meta = generate_pair(gi, seed, forced_periodic=forced)
        _, _, info = localize(ref, search)  # classical candidate generation
        cands = info.survivors
        if not cands:
            continue
        X = candidate_features(cands, 1000, 1000, NOMINAL_RATIO, ROT_MAX_DEG, MAX_DRIFT_FRAC)
        d = np.array([np.hypot(c.x - meta.gt_x, c.y - meta.gt_y) for c in cands])
        Xs.append(X)
        ys.append((d < LABEL_PX).astype(np.int64))
        gs.append(np.full(len(cands), gi))
        gts.append((meta.gt_x, meta.gt_y, int(forced)))
        if (gi + 1) % 25 == 0:
            print(f"  {gi+1}/{len(seeds)} pairs")
    return (np.vstack(Xs), np.concatenate(ys), np.concatenate(gs), np.array(gts))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=200)
    ap.add_argument("--n_test", type=int, default=40)
    ap.add_argument("--out", default="ranker_data.npz")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    print(f"Building TRAIN ({args.n_train} pairs, ~35% forced-periodic)...")
    Xtr, ytr, gtr, gttr = build(range(1000, 1000 + args.n_train), 0.35, rng)
    print(f"Building TEST ({args.n_test} pairs, ~20% forced-periodic)...")
    Xte, yte, gte, gtte = build(range(3000, 3000 + args.n_test), 0.20, rng)

    np.savez(args.out,
             X_train=Xtr, y_train=ytr, g_train=gtr, gt_train=gttr,
             X_test=Xte, y_test=yte, g_test=gte, gt_test=gtte)
    print(f"\nsaved {args.out}")
    print(f"  train rows {Xtr.shape}, positives {ytr.sum()} ({100*ytr.mean():.1f}%)")
    print(f"  test  rows {Xte.shape}, positives {yte.sum()} ({100*yte.mean():.1f}%)")
