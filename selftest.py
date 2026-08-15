"""One-command sanity test for the Drift-Sense localizer.

Tests on FRESH, UNSEEN pairs (seeds not used for tuning/training) -- this is the
honest test, closest to how the official hidden test set will judge the solution.
Also runs a NOISIER variant, since the official search images are stated to be
noisier than the self-eval set.

    python selftest.py            # default: 20 pairs
    python selftest.py --n 40     # more pairs
    python selftest.py --hybrid   # also test the optional ML ranker

PASS criteria (self-eval, tune-able): median < 5 px AND >= 85% within 1 um.
"""
from __future__ import annotations
import argparse
import numpy as np

from dataset_gen import generate_pair
from localize import localize


def summarize(name, errs, times=None):
    e = np.array(errs)
    line = ("  %-16s median %5.2f px  mean %6.1f px  <1px %3.0f%%  <30px %3.0f%%  "
            "<1um %3.0f%%  >100px %d" % (name, np.median(e), e.mean(),
            100*(e < 1).mean(), 100*(e < 30).mean(), 100*(e < 100).mean(), (e > 100).sum()))
    if times is not None:
        line += "  | %.0f ms/pair" % (1000*np.mean(times))
    print(line)
    return np.median(e), 100*(e < 100).mean()


def add_extra_noise(search_u8, rng, sigma=12.0):
    """Approximate the 'noisier official test set' by adding read noise."""
    noisy = search_u8.astype(np.float32) + rng.normal(0, sigma, search_u8.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=9000, help="unseen seed base")
    ap.add_argument("--hybrid", action="store_true")
    args = ap.parse_args()

    import time
    rng = np.random.default_rng(123)
    e_cls, t_cls, e_ml, e_noisy = [], [], [], []

    print(f"Generating {args.n} fresh unseen pairs (seeds {args.seed0}..{args.seed0+args.n-1})...")
    for i in range(args.n):
        forced = rng.random() < 0.15   # ~15% hard periodic pairs, like the eval mix
        ref, search, m = generate_pair(i, args.seed0 + i, forced_periodic=forced)

        t = time.perf_counter()
        x, y, _ = localize(ref, search)                  # classical (default)
        t_cls.append(time.perf_counter() - t)
        e_cls.append(np.hypot(x - m.gt_x, y - m.gt_y))

        if args.hybrid:
            xm, ym, _ = localize(ref, search, use_ml=True)
            e_ml.append(np.hypot(xm - m.gt_x, ym - m.gt_y))

        noisy = add_extra_noise(search, rng)             # noisier-test-set proxy
        xn, yn, _ = localize(ref, noisy)
        e_noisy.append(np.hypot(xn - m.gt_x, yn - m.gt_y))

    print("\nResults on UNSEEN data:")
    med, within = summarize("classical", e_cls, t_cls)
    if args.hybrid:
        summarize("hybrid (ML)", e_ml)
    summarize("classical+noise", e_noisy)

    ok = (med < 5.0) and (within >= 85.0)
    print("\n%s  (median %.2f px, %.0f%% within 1 um)" %
          ("[PASS]" if ok else "[CHECK - below threshold]", med, within))
    print("Note: 1-2 failures are expected by design (forced-periodic honest-failure cases).")


if __name__ == "__main__":
    main()
