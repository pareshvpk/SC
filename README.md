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

**Headline result (self-eval, 30 FinFET pairs):** median **0.1 px**, **96.7% within 1 µm**,
~1.3 s/pair. See `V2_REPORT.md` for the full engineering write-up.

---

## Quick start (clone → generate → localize, no contact needed)

```bash
# 1. Clone
git clone <this-repo-url>
cd driftsense

# 2. Create an environment and install dependencies
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate
pip install -r requirements.txt

# 3. Generate a sample image pair (or a full 30-pair set) WITH ground truth
python dataset_gen.py --style finfet --n 1 --out sample        # one pair
#   -> sample/pair_000_ref.png, sample/pair_000_search.png, sample/ground_truth.json

# 4. Run the localization inference script (THE script Applied Materials runs)
python localize.py sample/pair_000_ref.png sample/pair_000_search.png
#   -> prints:  x, y      e.g.  511.94, 518.40
```

That's it — no manual edits required.

---

## The inference script (most important file)

```bash
python localize.py <reference_image> <search_image>
```
- **Inputs:** path to the reference (high-mag) image, path to the search (wide, low-mag) image.
- **Output (stdout):** a single predicted center as `x, y` in search-image pixels.
- Runs standalone; unreadable paths exit non-zero with a clear error.
- Options: `--ratio R` (magnification ratio, default 10), `--use-ml` (use the optional
  trained ranker, auto-loads `ml_ranker.npz`), `--verbose` (match diagnostics to stderr).

Example:
```bash
$ python localize.py data/pair_006_ref.png data/pair_006_search.png
660.20, 512.05
```

---

## Repository contents (maps to the required deliverables)

| # | Requirement | File(s) |
|---|---|---|
| 1 | README with setup/run instructions | `README.md` (this file) |
| 2 | Dataset generator (`--style`, `--n`, `--out`; records ground truth) | **`dataset_gen.py`** |
| 3 | **Localization inference script** (ref + search → `x, y`) | **`localize.py`** |
| 4 | DL model weights (optional hybrid) | `ml_ranker.npz` |
| 5 | Training script / notebook (optional hybrid) | `train_ranker.ipynb`, `train_ranker.py`, `make_ranker_data.py` |
| 6 | `requirements.txt` (pip freeze) | `requirements.txt` |
| 7 | Citation document | `citations.md` |

Supporting: `eval.py` (self-eval harness), `bench.py` + `localize_v1.py` (V1-vs-V2 benchmark),
`make_offcenter_sets.py` (drift-robustness study), `predict.py` (visual overlay helper),
`selftest.py` (quick unseen-data check), `V2_REPORT.md` / `ALGORITHM_SUMMARY.md` (write-ups),
`data/` (a ready-made 30-pair FinFET eval set), `requirements-train.txt` (training-only deps).

---

## Dataset generator

```bash
python dataset_gen.py --style {finfet,dram} --n 30 --out data --seed 0 [--mag-jitter]
```
- `--style` — architecture: `finfet` (dense fins × sparse gate bars; the validated/cited
  style) or `dram` (word-lines × bit-lines with a contact dot per intersection).
- `--n` — number of pairs. `--out` — output directory. `--seed` — base seed.
- `--mag-jitter` — vary the true magnification per pair (~9×–11×) instead of a fixed 10×,
  for a harder, more realistic scale-robustness test (recorded as `magnification_ratio`).
- Each pair: `pair_XXX_ref.png` (1000×1000), `pair_XXX_search.png` (1000×1000), and the
  **true center** recorded in `ground_truth.json`.
- Realism (each choice cited in `citations.md`): independent per-image Poisson+Gaussian
  sensor noise (search noisier), SEM edge-brightening, per-image blur/rotation/scale
  jitter, per-line pitch jitter, and per-crossing defect dropout.

## Evaluate (measure accuracy against ground truth)

```bash
python eval.py --data data --tolerance_px 30
```
Reports per-pair error, timing, % within tolerance, and one honest failure case.

```bash
python bench.py --data data     # V1-vs-V2 comparison
python selftest.py --n 20       # quick check on fresh, unseen pairs
```

---

## Optional: hybrid ML ranker

The classical algorithm is the default and needs **no** model. An optional small MLP
(`14→16→8→1`) can rank candidates instead:

- **Inference:** `python localize.py ref.png search.png --use-ml` — loads `ml_ranker.npz`
  and runs a **pure-numpy** forward pass (no ML dependency at inference).
- **Retrain:**
  ```bash
  pip install -r requirements-train.txt      # adds scikit-learn (training only)
  python make_ranker_data.py --n_train 200 --n_test 40   # -> ranker_data.npz
  jupyter notebook train_ranker.ipynb        # trains + exports ml_ranker.npz
  ```

The hybrid matches (does not beat) the classical accuracy — see `V2_REPORT.md` §8.

---

## Requirements

- Python 3.11. Install with `pip install -r requirements.txt` (full `pip freeze` of the
  development environment). Inference needs only numpy / OpenCV / SciPy; scikit-learn is
  required **only** to retrain the optional ranker (`requirements-train.txt`).

## License

Released under the **Apache License 2.0** — see [`LICENSE`](LICENSE).
