# Drift-Sense localizer — engineering report (FinFET, V2)

## 0. What this dataset is, and why it is deliberately hard

The pipeline generates a **FinFET-style** layout: dense parallel vertical **fins**
(pitch 24–48 nm) crossed by sparse horizontal **gate bars**, with a per-crossing
**dropout defect** (≈40 % of fin–gate crossings drop their contact feature). A
1 µm reference crop appears, shrunk 10×, somewhere inside the 10 µm wide-search
image, which is highly periodic because the fin field repeats every ~3 px in the
search frame.

The gate pitch is set to **500–900 nm** so a 1 µm crop shows only **one or two
gate bars** (the brief's description). This matters: an earlier setting (150–350 nm
→ 3–7 gate bars per crop) produced a dense 2-D fin×gate grid whose 40 % dropout
pattern was a rich enough fingerprint that even plain NCC localized correctly, so
V1 and V2 tied and the multi-peak disambiguation the task actually targets was
never exercised. With only 1–2 gate bars the crop is dominated by the near-pure
periodic fin field, and disambiguation must come from a *thin, sparse* strip of
crossing-defects plus per-line pitch jitter — genuinely ambiguous, which is the
regime the localizer must survive.

## 1. Diagnosis of the V1 algorithm

V1 used normalized cross-correlation (NCC) for **both** candidate generation and
candidate selection. On a periodic FinFET lattice this is self-defeating:

- At any single fixed `(scale, rotation)` trial, a wrong lattice repeat routinely
  out-scores the true site, because the dominant term in the correlation is phase
  alignment to the periodic grid, not the small per-crossing defect signal that
  actually distinguishes one repeat from another.
- V1 recorded each spatial peak's score from **whichever single trial found it**,
  never at that site's own best transform, deflating the true site.
- V1's NMS suppression radius was ~½ template width, so the true site was
  routinely **merged with and discarded in favour of** a nearby higher-scoring
  wrong repeat.

Net effect on the hard FinFET set: a **bimodal** error distribution — exact, or
locked onto a wrong repeat 100–300 px away. V1 lands **13/30 catastrophic**
(> 100 px). The dominant error is **wrong-repeat selection**, not precision.

Two measurements reframed the problem:

- **Oracle test:** at its own best `(scale, rotation)`, the true site is the
  strongest local match on essentially every pair — the signal is there; V1 just
  never scored each site at its best transform.
- **Fingerprint test:** sampling intensity exactly at the fin–gate crossings
  (where the dropout lives) ranks the true site above the wrong repeat as an
  independent vote free of NCC's periodicity blind spot — **but only where dropout
  signal actually exists** (see §3, the reliability gate).

## 2. V2 strategy — generate broadly, verify, then select by reliability

1. **Max-projection response map** — per pixel, the max NCC over the whole
   `(scale, rotation)` sweep, so the true site appears at its oracle score.
2. **Broad candidate net** — top-N spatially-distinct local maxima with an NMS
   radius **< 1 lattice pitch**, so adjacent repeats are kept as *separate*
   candidates (pure recall; wrong repeats intentionally retained).
3. **Per-candidate verification** at each site's best transform: locally-refined
   NCC (scale/rotation/sub-pixel) + a **fin–gate crossing fingerprint**.
4. **Reliability-aware fusion + selection** (§3) — the core of V2.
5. **Sub-pixel** parabolic fit on the local correlation surface.

## 3. Reliability-aware selection (the part that makes V2 work on hard data)

The fingerprint is decisive when it carries signal and actively misleading when it
does not. V2 measures its reliability per pair and selects in one of three
regimes.

**Two reliability signals:**

- `fp_ref_std` — std of the *reference* template's own intersection-contrast
  vector. Low ⇒ the reference has little/no dropout variation (a near-defect-free
  region), so any fingerprint NCC is noise.
- `max_fp` — the best fingerprint achieved across all candidates. Low ⇒ even where
  the reference has contrast, no candidate actually matched it (noise floor).

These fold into a smooth `fp_gate ∈ [0,1]` that scales the fingerprint's weight in
the fusion `combined = z(NCC) + fp_gate·fp_weight·z(fingerprint)`.

**Three selection regimes:**

- **(a) Decisive fingerprint** — `fp_gate` high AND one in-ROI candidate has a
  confidently high fingerprint (`≥ fp_confident`) that is clearly separated
  (`≥ fp_gap`) from every other spatially-distinct candidate. That candidate *is*
  the site identity: pick it outright. **The center prior must not override a
  confident identity match.** (Fixes the case where the true site has fp = 0.92 but
  a slightly lower NCC, and the center tie-break otherwise discarded it for a wrong
  repeat 40 px nearer center — pair 5: 335 → 0.1 px.)
- **(b) Unreliable fingerprint** — `fp_gate` low (defect-free or noise-floor). The
  fused score is essentially NCC alone, and periodic repeats have near-identical
  NCC, so widen the near-tie to every candidate within `ncc_tie_margin` of the best
  NCC and let the **center prior** arbitrate — the only trustworthy signal for a
  defect-free revisit is that the true drifted site is nearest center. (Fixes
  pair 1: 220 → 18 px; pair 11: 149 → 98 px.)
- **(c) Otherwise** — fused-z near-tie with a center tie-break among genuine ties.

A bounded-drift ROI is searched first (a prior, not a hard bound); an out-of-ROI
candidate must beat the best in-ROI candidate by a decisive margin to compete, and
a comparative full-image fallback covers sites outside the drift envelope.

## 4. Benchmark (30-pair self-eval, `python bench.py --data data`)

| algorithm | median px | mean px | max px | <1px | <10px | <100px (1 µm) | >100px # | sec/pair |
|---|---|---|---|---|---|---|---|---|
| V1 (NCC + center prior) | 60.6 | 90.8 | 299.6 | 43.3% | 46.7% | 56.7% | 13 | 2.3 |
| **V2 (generate + verify + reliability)** | **0.1** | **12.7** | 141.9 | **83.3%** | **83.3%** | **96.7%** | **1** | **1.3** |

V2 rescues 11 pairs from 130–300 px wrong-repeat errors down to sub-pixel, cuts
catastrophic failures **13 → 1**, and is faster than V1 (no full-image response map
on the common path). Runtime is machine-dependent; the V1/V2 ratio is the stable
quantity.

**Held-out generalization (seed 500, params frozen):** median 0.1 px, **93.3 %
within 1 µm**, 2 catastrophic — consistent with the tuning set, confirming the
reliability thresholds are not overfit to seed 0.

## 5. Honest failure case

**Pair 2** (`forced_periodic`, defect-free): 141.9 px. The crop sits deep in a
near-defect-free region, so (i) the fingerprint has no dropout signal to exploit
(`fp_gate` → 0, correctly), and (ii) under the center-prior fallback a look-alike
lattice repeat happens to sit closer to the search-image center than the true
drifted site. Both independent disambiguators are exhausted, so the answer locks
onto the nearer repeat. This is the genuinely-unsolvable case the brief requires:
a highly periodic array region with no site-identifying signal within the assumed
drift envelope. (V1 lands it 53 px off by luck of the same center prior; neither
algorithm can *know* which repeat is real here.)

## 6. Parameters (with defaults)

| parameter | default | note |
|---|---|---|
| `nominal_ratio` | 10.0 | known zoom ratio; keep. |
| `scale_jitter` | (0.94, 1.06) | coarse scale sweep. |
| `rot_max_deg` | 4.0 | coarse rotation half-range. |
| `n_candidates` | 50 | candidate net size (recall). |
| `fp_weight` | 1.8 | base fingerprint weight (before the gate). |
| `fp_gate_lo / hi` | 1.0 / 4.0 | `fp_ref_std` ramp: below `lo` the fingerprint is treated as pure noise. |
| `fp_floor_lo / hi` | 0.30 / 0.45 | `max_fp` ramp: if no candidate matches the fingerprint, down-gate it. |
| `fp_confident` | 0.60 | fingerprint value above which a match may win outright (regime a). |
| `fp_gap` | 0.12 | required separation of the top fingerprint from the next distinct site (regime a). |
| `ncc_tie_margin` | 0.012 | raw-NCC near-tie width used for the center prior when the fingerprint is unreliable (regime b). |
| `tie_z` | 0.7 | fused-z near-tie width AND the decisive out-of-ROI margin. |
| `max_drift_frac` | 0.24 | center-ROI radius; a prior, not a hard bound. |

## 7. V3 suggestions (only if the official set proves harder)

- **Grid-anchored fingerprint:** register the lattice phase globally (Fourier peak
  fit) and match dropout maps in lattice coordinates — removes residual rotation
  sensitivity of the current intersection sampler.
- **Log-polar / Fourier-Mellin** refinement for larger scale/rotation jitter than
  the sweep covers, at constant cost.
- **Learned local descriptor** keyed on the defect pattern — only if the official
  test set proves harder than the fingerprint handles; a 30-pair set does not
  justify DL yet.
- **Confidence output:** expose the `combined`-score gap between the top two
  candidates as a calibrated confidence, so the tool can flag genuinely ambiguous
  revisits (like pair 2) for a re-scan instead of returning a wrong site.

## 8. Optional hybrid ML ranker (`use_ml=True`)

A small **trained neural network** (MLP, 14→16→8→1) can replace the hand-tuned
three-regime selector. It scores each classical candidate's probability of being
the true site from the **same** per-candidate features the reliability logic uses
(`candidate_features` in `localize.py`), so there is no train/serve skew; the
highest-probability candidate within the ROI-eligible pool is selected.

- **Training** (`make_ranker_data.py` + `train_ranker.ipynb`): 200 generated pairs
  → classical candidates → label = within 5 px of GT. scikit-learn `MLPClassifier`,
  L2 + early stopping. Weights exported to `ml_ranker.npz`.
- **Inference** is a **pure-numpy forward pass** in `localize.py` — no
  sklearn/torch at run time. If `ml_ranker.npz` is absent, the classical selector
  is used, so the pipeline always works.
- **Result:** test ROC-AUC **0.977**; it *reproduces* the classical accuracy
  rather than beating it — on the tuning set classical is slightly ahead (97% vs
  93% within 1 µm), on a fresh held-out set the hybrid is marginally ahead (82% vs
  80%, 7 vs 8 catastrophic). Interpretation: the MLP **learned** the reliability
  rules from data, a useful cross-check, but not a headline accuracy win.
- **Positioning:** the **classical selector is the default and the scored
  solution** (interpretable, no model on the critical path). The hybrid ships as
  the trained-model deliverable (enable with `localize(..., use_ml=True)`).
  Training-only dependency: `requirements-train.txt`.

## 9. Off-center robustness (drift-envelope stress test)

The localizer assumes **bounded drift** — the true site sits near the search
center — which is physically correct and is what makes the "nearest-center"
tie-break meaningful. To characterize behavior when that assumption is violated
(the official test set's drift envelope is unknown), the true site was forced to
three distance bands from center (`make_offcenter_sets.py`, 30 pairs each):

| true-site band | dist. from center | **default** (scored) | `always_full_search=True` |
|---|---|---|---|
| Center (realistic) | 0–215 px | **96.7%** within 1 µm | 93.3% |
| Inner-corner | 270–336 px | 0% | 17% |
| Corner (extreme) | 561–641 px | 56.7% | 73.3% |

**Findings (per-candidate diagnosis):**
- An off-center site is recoverable **only when its fingerprint is strong**. The
  fingerprint is position-independent identity, so a confident match wins
  regardless of location; a weak-fingerprint off-center site has **no usable
  signal** (identical repeats + a location prior that now points the wrong way) —
  it is **fundamentally ambiguous**, not a code defect.
- Counterintuitively the inner band (~300 px) fails *worse* than the far corners:
  near the ROI edge the comparative full-image fallback stays silent (the central
  region still matches well), whereas at the extreme corners the central match is
  poor enough to trigger the fallback, which then recovers the site.
- `always_full_search=True` closes part of the gap (corner 57→73 %) by always
  generating a full-image candidate net, but it costs ~3–4 % on the realistic
  center case (a spurious far repeat can occasionally win). **No free lunch.**

**Decision:** default is the center-optimized path (matches the stated bounded-
drift physics and maximizes the scored metric); `always_full_search=True` is an
opt-in knob for when the drift envelope proves wide. Reproduce with
`python make_offcenter_sets.py --mode corner|inner` then `python eval.py`.

## Files

- `train_ranker.ipynb` — training notebook for the optional hybrid MLP ranker.
- `make_ranker_data.py` — builds the ranker's feature/label table.
- `ml_ranker.npz` — exported MLP weights (numpy-only inference).

- `localize.py` — V2 (public API `localize(ref, search) -> (x, y, info)`).
- `localize_v1.py` — V1 snapshot, for benchmarking.
- `bench.py` — V1 vs V2 benchmark table + per-pair deltas.
- `eval.py` — self-eval harness (V2 is drop-in compatible).
- `dataset_gen.py` — FinFET dataset generator.
- `make_offcenter_sets.py` — off-center (inner/corner) drift-robustness test sets.
- `citations.md` — literature backing every noise/blur/structural choice.
- `train_ranker.py` — script mirror of the training notebook.
- `requirements.txt` / `requirements-train.txt` — inference / training dependencies.
