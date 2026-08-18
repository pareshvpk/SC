#!/usr/bin/env python3
"""Drift-Sense smoke test — one command for a judge to verify the pipeline runs.

It (1) prints the environment, (2) generates a small low-noise set via the real
`dataset_gen.py`, (3) localises every pair through the REAL `localize.py` CLI as a
subprocess — exactly the call a grader makes — and (4) asserts pass/fail
thresholds. Exit code 0 = PASS, 1 = FAIL, so it is CI-friendly.

    python tools/smoke_test.py
"""
from __future__ import annotations
import json, os, subprocess, sys, tempfile, time
import numpy as np

N = 12                 # pairs to generate
FLOOR = 8              # require >= this many within TOL_PX
TOL_PX = 10.0
MEDIAN_LIMIT = 5.0     # median error must be under this
TIME_LIMIT = 60.0      # whole test must finish under this (s)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _versions():
    import cv2, numpy
    v = [f"numpy {numpy.__version__}", f"opencv {cv2.__version__}"]
    try:
        import scipy; v.append(f"scipy {scipy.__version__}")
    except Exception:
        pass
    return "  ".join(v)


def main() -> int:
    py = sys.executable
    print("Drift-Sense smoke test")
    print(f"  python {sys.version.split()[0]}  ({py})")
    print(f"  {_versions()}")
    print(f"  checks  {N} pairs (FinFET+DRAM): >= {FLOOR} within {TOL_PX:.0f} px, "
          f"median error < {MEDIAN_LIMIT:.0f} px, total < {TIME_LIMIT:.0f} s")

    t_start = time.perf_counter()
    work = tempfile.mkdtemp(prefix="driftsense_smoke_")

    print("\n[1/2] generating dataset via dataset_gen.py")
    subprocess.run([py, "src/dataset_gen.py", "--style", "both", "--noise-level", "medium",
                    "--n", str(N), "--out", work, "--seed", "1"],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    metas = json.load(open(os.path.join(work, "ground_truth.json")))
    print(f"      {len(metas)} pairs written")

    print("\n[2/2] localizing via the real localize.py CLI (subprocess)")
    print(f"      {'pair':<10} {'predicted':>14} {'truth':>18} {'error px':>10} {'ms':>7}")
    errs = []
    for m in metas:
        i = m["pair_id"]
        ref = os.path.join(work, f"pair_{i:03d}_ref.png")
        se = os.path.join(work, f"pair_{i:03d}_search.png")
        t0 = time.perf_counter()
        out = subprocess.run([py, "src/localize.py", ref, se],
                             capture_output=True, text=True, cwd=ROOT)
        ms = (time.perf_counter() - t0) * 1000
        try:
            x, y = (float(v) for v in out.stdout.strip().split(","))
        except Exception:
            print(f"      pair_{i+1:04d}  BAD OUTPUT: {out.stdout!r} {out.stderr!r}")
            return 1
        err = float(np.hypot(x - m["gt_x"], y - m["gt_y"]))
        errs.append(err)
        flag = "  <-- miss (tail case)" if err > TOL_PX else ""
        print(f"      pair_{i+1:04d}  ({x:6.0f}, {y:6.0f})  ({m['gt_x']:7.1f}, {m['gt_y']:7.1f})"
              f"  {err:9.2f}  {ms:6.0f}{flag}")

    errs = np.array(errs)
    total = time.perf_counter() - t_start
    hits = int((errs <= TOL_PX).sum())
    med = float(np.median(errs))

    print(f"\n  hits          {hits}/{N} within {TOL_PX:.0f} px  (floor {FLOOR})")
    print(f"  median error  {med:.2f} px  (limit {MEDIAN_LIMIT:.0f} px)")
    print(f"  total time    {total:.1f} s  (limit {TIME_LIMIT:.0f} s)")

    ok = hits >= FLOOR and med < MEDIAN_LIMIT and total < TIME_LIMIT
    print(f"\n  {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
