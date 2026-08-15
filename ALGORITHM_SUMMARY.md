# Drift-Sense — Algorithm Summary

> For the full write-up — dataset design, V1 diagnosis, the reliability-aware V2
> pipeline, the V1→V2 benchmark, held-out validation, parameters, and the honest
> failure case — see **`V2_REPORT.md`**.

## One-paragraph overview

Given a 100× reference close-up and a 10× wide-search image of the same **FinFET**
die site (dense vertical fins crossed by 1–2 horizontal gate bars, with ~40 % of
fin–gate crossings dropped as defects), return the reference's center `(x, y)` in
search-image pixels; among periodic look-alikes, return the match nearest the
search-image center. V2 stops using cross-correlation as both generator and
selector. It builds a **max-projection NCC response map** (so the true site
appears at its best-transform score), takes a **broad candidate net** of
spatially-distinct peaks, then **verifies each candidate independently** with a
locally-refined NCC and a **fin–gate crossing fingerprint** (samples intensity at
the crossings where the ~40 % dropout lives). Crucially, selection is
**reliability-aware**: it measures whether the fingerprint carries signal
(reference contrast `fp_ref_std` and best match `max_fp`) and switches among three
regimes — a confident fingerprint wins outright, an unreliable one defers to the
NCC + center prior, and genuine ties are broken by center.

## Headline result (30-pair self-eval, deliberately-hard FinFET)

| | median px | <1 px | <1 µm | catastrophic (>100 px) |
|---|---|---|---|---|
| V1 (NCC + center) | 60.6 | 43.3 % | 56.7 % | 13 |
| **V2** | **0.1** | **83.3 %** | **96.7 %** | **1** |

The 1 remaining failure is the designed-hard case: a defect-free periodic region
where the true drifted site sits farther from center than a look-alike repeat and
no dropout signal exists to break the tie. Held-out seed 500 (params frozen):
median 0.1 px, 93.3 % within 1 µm.

## Files to review

- `V2_REPORT.md` — the canonical engineering report (read this).
- `localize.py` — V2 implementation, API `localize(ref, search) -> (x, y, info)`.
- `bench.py` — reproduces the V1-vs-V2 table (`python bench.py`).
- `localize_v1.py` — V1 snapshot used by the benchmark.
- `dataset_gen.py`, `eval.py`, `citations.md` — dataset, self-eval harness, references.
