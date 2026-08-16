# Drift-Sense — Algorithm Summary

> For the full write-up — dataset design, why plain cross-correlation fails, the
> reliability-aware pipeline, held-out validation, parameters, and the honest
> failure case — see **`REPORT.md`**.

## One-paragraph overview

Given a 100× reference close-up and a 10× wide-search image of the same **FinFET**
die site (dense vertical fins crossed by 1–2 horizontal gate bars, with ~40 % of
fin–gate crossings dropped as defects), return the reference's center `(x, y)` in
search-image pixels; among periodic look-alikes, return the match nearest the
search-image center. The localizer does **not** use cross-correlation as both
generator and selector (fatal on a periodic lattice — a wrong repeat out-scores
the true site). It builds a **max-projection NCC response map** (so the true site
appears at its best-transform score), takes a **broad candidate net** of
spatially-distinct peaks, then **verifies each candidate independently** with a
locally-refined NCC and a **fin–gate crossing fingerprint** (samples intensity at
the crossings where the ~40 % dropout lives). Crucially, selection is
**reliability-aware**: it measures whether the fingerprint carries signal
(reference contrast `fp_ref_std` and best match `max_fp`) and switches among three
regimes — a confident fingerprint wins outright, an unreliable one defers to the
NCC + center prior, and genuine ties are broken by center.

## Headline result (30-pair self-eval, realistic FinFET with mat superstructure)

| median px | <1 px | <1 µm | honest failures (>100 px) |
|---|---|---|---|
| **0.3** | **80.0 %** | **90.0 %** | **3 / 30** |

The dataset carries a realistic subarray-**mat superstructure** (dense cell blocks
separated by sense-amp / driver channels — looks like a real wafer array). The 3
remaining failures are all genuinely-hard cases: two `forced_periodic`
honest-failure crops and one crop that landed deep inside a mat interior (locally
periodic, so ambiguous by construction).

## Files to review

- `REPORT.md` — the canonical engineering report (read this).
- `localize.py` — the localizer, API `localize(ref, search) -> (x, y, info)`.
- `eval.py` — self-eval harness (`python eval.py --data data`).
- `dataset_gen.py`, `citations.md` — dataset generator and references.
