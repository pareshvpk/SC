"""Generate a BLIND positional test set: 30 realistic (superstructure) pairs whose
true site is placed across five radial zones -- center, inner-center, middle, ring,
corner -- WITHOUT recording which zone each pair belongs to. The published ground
truth is only (x, y); the zone labels are kept in a separate hidden file for
analysis, so a localizer (or reviewer) gets no positional hint.

    python make_blind_set.py --out blind_test --style finfet
"""
from __future__ import annotations
import argparse, json, os
import numpy as np, cv2
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from dataset_gen import generate_pair, SEARCH_NM_PER_PX

# radial zone -> (min, max) distance of the true site from frame center, in search px
ZONES = {"center": (0, 80), "inner-center": (80, 160), "middle": (160, 260),
         "ring": (260, 360), "corner": (360, 449)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="blind_test")
    ap.add_argument("--style", choices=["finfet", "dram"], default="finfet")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    zones = (list(ZONES) * ((args.n // len(ZONES)) + 1))[:args.n]
    rng.shuffle(zones)                       # shuffle so pair index never reveals the zone

    public, hidden = [], []
    for i, zone in enumerate(zones):
        r_lo, r_hi = ZONES[zone]
        r_px = rng.uniform(r_lo, r_hi)
        if zone == "corner":                 # bias toward a true diagonal corner
            ang = np.radians(rng.choice([45, 135, 225, 315]) + rng.uniform(-18, 18))
        else:
            ang = rng.uniform(0, 2 * np.pi)
        dx = r_px * SEARCH_NM_PER_PX * np.cos(ang)
        dy = r_px * SEARCH_NM_PER_PX * np.sin(ang)
        ref, search, meta = generate_pair(i, args.seed * 7 + i, style=args.style,
                                          superstructure=True, drift_nm=(float(dx), float(dy)))
        cv2.imwrite(os.path.join(args.out, f"pair_{i:03d}_ref.png"), ref)
        cv2.imwrite(os.path.join(args.out, f"pair_{i:03d}_search.png"), search)
        public.append({"pair_id": i, "gt_x": round(meta.gt_x, 3), "gt_y": round(meta.gt_y, 3)})
        hidden.append({"pair_id": i, "zone": zone, "dist_px": round(float(r_px), 1)})
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{args.n}")

    json.dump(public, open(os.path.join(args.out, "ground_truth.json"), "w"), indent=2)
    json.dump(hidden, open(os.path.join(args.out, "_zones_hidden.json"), "w"), indent=2)
    print(f"wrote {args.n} blind pairs -> {args.out}/ (zones hidden in _zones_hidden.json)")


if __name__ == "__main__":
    main()
