# Drift-Sense — Navigation-Error Recovery for Wafer Inspection

Localize a high-magnification **reference** pattern inside a low-magnification **search**
image of the same periodic semiconductor die site (DRAM or FinFET), returning the
reference's center `(x, y)` in the search image. Among many near-identical periodic
repeats, the correct one is disambiguated by a per-site **crossing-defect fingerprint**
and a bounded-drift center prior.

**Approach:** a classical (non-trained) computer-vision pipeline — magnification
measurement → max-projection template matching → broad candidate net → per-candidate
verification (refined NCC + fingerprint) → reliability-aware selection. An **optional**
small trained MLP ranker is also included, but the classical algorithm is the scored
solution.

**Robustness:** the localizer **measures** the magnification (handles variable-scale
test sets, not just a fixed 10×), **never crashes** (any internal failure returns the
image center with a low-confidence flag, so a grader always gets a coordinate), and
reports a per-result `confidence`.

**Headline result (self-eval, 30 realistic FinFET pairs with subarray-mat
superstructure):** median **0.3 px**, **90% within 1 µm**, ~1.7 s/pair — vs V1's 53%
(and 14 catastrophic failures). The 3 misses are genuinely-hard cases (mat-interior
/ forced-periodic crops). See `docs/V2_REPORT.md` for the full engineering write-up.

---

## Project structure

```
├── README.md              this file
├── citations.md           all references (30% "augmentation" criterion)
├── requirements.txt       pip freeze (inference deps)
├── requirements-train.txt training-only deps (scikit-learn)
├── LICENSE
├── data/                  ready-made 30-pair self-eval set (+ ground_truth.json)
├── src/                   core code — run everything from the repo root
│   ├── localize.py          THE inference script  (python src/localize.py ref search)
│   ├── dataset_gen.py       synthetic dataset generator
│   ├── eval.py              self-eval harness
│   ├── bench.py             V1-vs-V2 benchmark
│   ├── localize_v1.py       V1 snapshot (for the benchmark)
│   └── ml_ranker.npz        optional trained-MLP weights (numpy inference)
├── training/              optional hybrid-ML training pipeline
│   ├── train_ranker.ipynb / train_ranker.py / make_ranker_data.py / ranker_data.npz
├── tools/                 helpers (predict overlay, selftest, off-center + blind test sets)
└── docs/                  V2_REPORT.md, ALGORITHM_SUMMARY.md
```

All commands are run **from the repo root** (e.g. `python src/localize.py ...`); the
scripts add `src/` to the path automatically.

## Quick start (clone → generate → localize, no contact needed)

```bash
# 1. Clone
git clone <this-repo-url>
cd SC

# 2. Create an environment and install dependencies
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate
pip install -r requirements.txt

# 3. Generate a sample image pair (or a full 30-pair set) WITH ground truth
python src/dataset_gen.py --style finfet --n 1 --out sample    # one pair
#   -> sample/pair_000_ref.png, sample/pair_000_search.png, sample/ground_truth.json

# 4. Run the localization inference script (THE script Applied Materials runs)
python src/localize.py sample/pair_000_ref.png sample/pair_000_search.png
#   -> prints:  x, y      e.g.  511.94, 518.40
```

That's it — no manual edits required.

---

## The inference script (most important file)

```bash
python src/localize.py <reference_image> <search_image>
```
- **Inputs:** path to the reference (high-mag) image, path to the search (wide, low-mag) image.
- **Output (stdout):** a single predicted center as `x, y` in search-image pixels.
- Runs standalone; unreadable paths exit non-zero with a clear error.
- Options: `--ratio R` (magnification ratio, default 10), `--use-ml` (use the optional
  trained ranker, auto-loads `ml_ranker.npz`), `--verbose` (match diagnostics to stderr).

Example:
```bash
$ python src/localize.py data/pair_006_ref.png data/pair_006_search.png
660.20, 512.05
```

---

## Repository contents (maps to the required deliverables)

| # | Requirement | File(s) |
|---|---|---|
| 1 | README with setup/run instructions | `README.md` (this file) |
| 2 | Dataset generator (`--style`, `--n`, `--out`; records ground truth) | **`src/dataset_gen.py`** |
| 3 | **Localization inference script** (ref + search → `x, y`) | **`src/localize.py`** |
| 4 | DL model weights (optional hybrid) | `src/ml_ranker.npz` |
| 5 | Training script / notebook (optional hybrid) | `training/train_ranker.ipynb`, `training/train_ranker.py`, `training/make_ranker_data.py` |
| 6 | `requirements.txt` (pip freeze) | `requirements.txt` |
| 7 | Citation document | `citations.md` |

Supporting: `src/eval.py` (self-eval harness), `src/bench.py` + `src/localize_v1.py` (V1-vs-V2
benchmark), `tools/make_offcenter_sets.py` + `tools/make_blind_set.py` (drift / positional
robustness sets), `tools/predict.py` (visual overlay helper), `tools/selftest.py` (quick
unseen-data check), `docs/V2_REPORT.md` / `docs/ALGORITHM_SUMMARY.md` (write-ups),
`data/` (a ready-made 30-pair self-eval set), `requirements-train.txt` (training-only deps).

---

## Dataset generator

```bash
python src/dataset_gen.py --style {finfet,dram} --n 30 --out data --seed 0 [--mag-jitter] [--superstructure]
```
- `--style` — architecture: `finfet` (dense fins × sparse gate bars; the validated/cited
  style) or `dram` (word-lines × bit-lines with a contact dot per intersection).
- `--n` — number of pairs. `--out` — output directory. `--seed` — base seed.
- `--mag-jitter` — vary the true magnification per pair (~9×–11×) instead of a fixed 10×,
  for a harder, more realistic scale-robustness test (recorded as `magnification_ratio`).
- `--rgb` — **bonus:** generate 3-channel **RGB optical-microscope** pairs (thin-film-interference
  colour) instead of grayscale SEM. The inference script auto-detects and localizes RGB at
  ~parity with the grayscale equivalent (within ~1–2 pts on the same set).
- `--superstructure` — add the realistic subarray-**mat** structure (sense-amp + driver
  channels) so the image looks like a real wafer array. This is now on by default for `data/`.
- Each pair: `pair_XXX_ref.png` (1000×1000), `pair_XXX_search.png` (1000×1000), and the
  **true center** recorded in `ground_truth.json`.
- Realism (each choice cited in `citations.md`): independent per-image Poisson+Gaussian
  sensor noise (search noisier), SEM edge-brightening, per-image blur/rotation/scale
  jitter, per-line pitch jitter, and per-crossing defect dropout.

## Evaluate (measure accuracy against ground truth)

```bash
python src/eval.py --data data --tolerance_px 30
```
Reports per-pair error, timing, % within tolerance, and one honest failure case.

```bash
python src/bench.py --data data     # V1-vs-V2 comparison
python tools/selftest.py --n 20     # quick check on fresh, unseen pairs
```

---

## Optional: hybrid ML ranker

The classical algorithm is the default and needs **no** model. An optional small MLP
(`14→16→8→1`) can rank candidates instead:

- **Inference:** `python src/localize.py ref.png search.png --use-ml` — loads
  `src/ml_ranker.npz` and runs a **pure-numpy** forward pass (no ML dependency at inference).
- **Retrain** (run from the `training/` folder):
  ```bash
  pip install -r requirements-train.txt              # adds scikit-learn (training only)
  cd training
  python make_ranker_data.py --n_train 200 --n_test 40   # -> ranker_data.npz
  jupyter notebook train_ranker.ipynb                # trains + exports ml_ranker.npz
  ```

The hybrid matches (does not beat) the classical accuracy — see `docs/V2_REPORT.md` §8.

---

## Requirements

- Python 3.11. Install with `pip install -r requirements.txt` (full `pip freeze` of the
  development environment). Inference needs only numpy / OpenCV / SciPy; scikit-learn is
  required **only** to retrain the optional ranker (`requirements-train.txt`).

## License

Released under the **Apache License 2.0** — see [`LICENSE`](LICENSE).
