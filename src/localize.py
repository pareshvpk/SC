"""
Drift-Sense localization algorithm.

Public API:

    x, y, info = localize(ref_img, search_img)

Given a high-res reference capture (100x zoom) and a low-res search capture
(10x zoom) of the same physical FinFET die site, return the center (x, y) of the
reference pattern inside the search image, in search-image pixel coordinates.
If several regions match (the layout is highly periodic), return the one
closest to the search-image center.

------------------------------------------------------------------------------
WHY NOT PLAIN CROSS-CORRELATION
------------------------------------------------------------------------------
Using normalized cross-correlation (NCC) both to *generate* and to *select*
candidates is fatal on a periodic FinFET lattice: at any single fixed
(scale, rotation) trial, a wrong lattice repeat routinely out-scores the true
site (measured: true site NCC ~0.35 vs a wrong repeat ~0.86 at a fixed trial).
Recording each spatial peak's score from whichever single trial found it, and
suppressing peaks within ~half a template width, merges the true site with --
and discards it in favour of -- a nearby higher-scoring wrong repeat. That
produces a bimodal error distribution (either exact, or locked onto a repeat
100-250 px away).

Two empirical facts drive the design (both measured on the dataset):

  1. At its OWN best (scale, rotation), the true site is the strongest local
     match 19/19 times on the otherwise-failing pairs. So the signal is there;
     it just has to be scored at each site's best transform.

  2. A fin-gate crossing "fingerprint" -- sampling intensity exactly at the
     grid intersections, where the local crossing-defect variation lives --
     ranks the true site above the wrong repeat 18/19 times, as an
     INDEPENDENT vote that does not share NCC's periodicity blind spot.

------------------------------------------------------------------------------
PIPELINE
------------------------------------------------------------------------------
  reference
     |
     v
  (1) MAX-PROJECTION response map: for every pixel, the max NCC over the whole
      (scale, rotation) sweep. This makes the true site a strong local peak
      (its oracle score), instead of being deflated by one unlucky trial.
     |
     v
  (2) BROAD candidate net: top-N spatially-distinct local maxima of the
      response map (NMS radius < one lattice pitch, so adjacent repeats are
      kept as *separate* candidates rather than merged away). This is pure
      recall -- we deliberately keep wrong repeats too and let verification
      sort them out.
     |
     v
  (3) PER-CANDIDATE VERIFICATION (each candidate scored independently):
        - refined NCC: local (scale, rotation, sub-pixel position) optimisation
          so every candidate is scored at ITS best transform (fixes fact 1);
        - fin-gate crossing fingerprint correlation (fixes fact 2).
     |
     v
  (4) RELIABILITY-AWARE FUSION: combined = z(NCC) + fp_gate*fp_weight*z(fp).
      The fingerprint is decisive where dropout signal exists and actively
      misleading where it does not, so its weight is gated by two per-pair
      reliability signals: fp_ref_std (does the REFERENCE carry dropout
      contrast?) and max_fp (did ANY candidate actually match it?). Both low
      => the fingerprint is noise and is down-weighted toward zero.
     |
     v
  (5) SELECTION (see localize() body for details):
      * LANDMARK override (texture-agnostic): a dedicated single-transform,
        multi-scale global NCC (_global_landmark) measures whether the reference
        contains APERIODIC content (a mat boundary / sense-amp channel / field
        edge). Because it uses ONE fixed template -- not per-candidate refinement,
        which would let every repeat borrow its own best transform and erase the
        margin -- only the true site scores decisively higher when a landmark
        exists. A decisive lead pins the location outright, even far off-center
        (UNIFORM placement), before the center prior can grab a nearer repeat.
        On a purely periodic crop the lead ~= 0, so this stays silent and the
        fingerprint + center prior below resolve the periodic case.
      Otherwise, one of three regimes:
        (a) DECISIVE fingerprint (reliable + a clear, isolated high-fp winner):
            that candidate IS the site identity -- pick it outright; the center
            prior must NOT override a confident identity match.
        (b) UNRELIABLE fingerprint (defect-free / noise-floor): fall back to
            NCC + the center prior over the true periodic repeats (the true
            drifted site is nearest center).
        (c) OTHERWISE: fused-z near-tie broken by center.
     |
     v
  (6) SUB-PIXEL refinement via parabolic fit on the local correlation surface.

Result (30-pair self-eval, deliberately-hard FinFET set, see eval.py):
    median ~0.2 px, ~90% within 1 um, ~0.65 s/pair. The few misses are the
    forced-periodic, defect-free pairs that carry no fingerprint by design and
    whose true site sits farther from center than a look-alike repeat -- the
    intended honest-failure cases. The landmark override additionally recovers
    off-center / cross-generator placements that the center prior alone would miss.

Performance: scale and rotation are per-CAPTURE (global), so the verification loop
    refines every candidate in a NARROW sweep around the one global transform found
    by the probe (fine_neighbors=1: a 3x3 scale/rotation neighbourhood), instead of
    re-searching the whole grid per site. Verification is capped to the strongest
    refine_topk=30 candidates by response score (the max-projection map always
    surfaces the true site in the top ~30; going tighter drops it on the ambiguous
    foreign set). Together with a scipy-free peak finder these hold accuracy exactly
    while cutting runtime ~1.5x vs. an all-candidate 5x5 refine (matchTemplate
    dominates runtime; the per-candidate refine loop was ~50% of it).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np


# ----------------------------------------------------------------------------
# Data structures (field names kept compatible with eval.py)
# ----------------------------------------------------------------------------
@dataclass
class Candidate:
    x: float
    y: float
    score: float          # refined NCC at this candidate (interpretable [-1, 1])
    scale: float          # search/ref pixel ratio at the refined match
    rotation: float       # degrees
    fingerprint: float = 0.0
    fp_ref_std: float = 0.0  # std of the reference template's own intersection-contrast
                             # vector -- a candidate-independent measure of how much
                             # dropout signal the fingerprint can possibly carry here.
    envelope: float = 0.0    # texture-agnostic low-frequency identity match: NCC of the
                             # slowly-varying per-site brightness/shading envelope (lattice
                             # removed). Distinguishes periodic repeats when NCC cannot --
                             # and, unlike the fin-gate fingerprint, does not assume a
                             # particular defect model, so it survives a foreign texture.
    res_score: float = 0.0   # periodic-residual match: NCC of the APERIODIC remainder
                             # (lattice subtracted) of the reference vs this candidate's
                             # search window. Unlike the envelope (low-freq) it keeps the
                             # high-freq aperiodic content -- mat boundaries, defects,
                             # line-edge roughness -- that pins one lattice repeat vs the
                             # next. This is the array-mode/cell-to-cell disambiguator.
    combined: float = 0.0  # fused z-score used for ranking


@dataclass
class LocalizeInfo:
    candidates: list = field(default_factory=list)
    survivors: list = field(default_factory=list)
    near_tied: list = field(default_factory=list)
    chosen: "Candidate | None" = None
    best_score: float = 0.0
    elapsed_s: float = 0.0
    full_image_fallback: bool = False
    # fallback-trigger instrumentation (best single-template NCC of the ROI's
    # own refined probe, evaluated inside vs strictly outside the drift ROI --
    # the two quantities the comparative trigger compares; see localize()).
    roi_probe_best: float = 0.0
    outside_probe_best: float = 0.0
    fallback_reason: str = ""
    # selection confidence: derived from the combined-score margin between the top
    # two candidates (a genuine near-tie -> low confidence). low_confidence is also
    # set True when the whole call fell back to the image center after an internal
    # failure (never-raise guarantee).
    confidence: float = 1.0
    low_confidence: bool = False
    magnification: float = 0.0   # magnification actually used (measured or given)


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Zero-normalized cross-correlation of two equal-shape arrays."""
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / (denom + 1e-8))


def _make_template(ref_f: np.ndarray, scale: float, rot_deg: float) -> np.ndarray:
    """Reference downsampled by `scale` and rotated by `rot_deg`."""
    tw = max(8, int(round(ref_f.shape[1] / scale)))
    th = max(8, int(round(ref_f.shape[0] / scale)))
    t = cv2.resize(ref_f, (tw, th), interpolation=cv2.INTER_AREA)
    if rot_deg != 0.0:
        M = cv2.getRotationMatrix2D((tw / 2, th / 2), rot_deg, 1.0)
        t = cv2.warpAffine(t, M, (tw, th), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
    return t


def _estimate_pitch_px(img: np.ndarray) -> "float | None":
    """Estimate the periodic lattice pitch (pixels) via 1-D autocorrelation of
    the row/column mean profiles. Used to size the candidate-NMS radius and the
    fingerprint sampling grid. Returns None if no clear period is found."""
    f = img - img.mean()
    col = f.mean(axis=0)
    row = f.mean(axis=1)

    def first_period(sig):
        ac = np.correlate(sig, sig, mode="full")[len(sig) - 1:]
        if ac[0] <= 0:
            return None
        ac = ac.copy()
        ac[0] = 0.0
        for i in range(3, len(ac) - 1):
            if ac[i] > ac[i - 1] and ac[i] >= ac[i + 1] and ac[i] > 0.2 * ac.max():
                return i
        return None

    periods = [p for p in (first_period(col), first_period(row)) if p]
    return float(np.mean(periods)) if periods else None


# ---------------------------------------------------------------------------
# Periodic-residual disambiguation ("array-mode" / cell-to-cell comparison).
#
# On a periodic layout every lattice repeat has an essentially identical
# high-frequency pattern, so raw NCC scores them all alike and picks the true
# site only by luck. Subtract the lattice and what remains -- mat boundaries,
# periphery strips, defects, array edges, line-edge roughness -- is exactly the
# aperiodic content that distinguishes ONE repeat from another. This is the
# signal driftsense's envelope (low-freq) and fin-gate fingerprint (a specific
# defect model) do NOT capture; it is what a wafer tool uses in array mode and
# what a strong classical competitor relies on for its periodic accuracy.
#
# The operator residual(I) = I - w*(I<<p + I>>p) per axis is separable, linear
# and shift-invariant, so NCC between two residual-filtered images stays a valid
# match score and the same filter applies to template and search patch alike.
# ---------------------------------------------------------------------------
def _pitch_yx(img: np.ndarray, min_lag: int = 2, max_lag: "int | None" = None
              ) -> "tuple[float, float]":
    """Per-axis lattice pitch (pitch_y, pitch_x) in pixels via 1-D autocorrelation
    of the row/column mean profiles. Either may be 0.0 if undetected."""
    if max_lag is None:
        max_lag = min(img.shape[:2]) // 4

    def fundamental(profile):
        x = profile.astype(np.float64)
        x = x - x.mean()
        if x.std() < 1e-9:
            return 0.0
        n = len(x)
        fsize = 1 << int(np.ceil(np.log2(2 * n)))
        F = np.fft.rfft(x, fsize)
        ac = np.fft.irfft(F * np.conj(F), fsize)[:n]
        ac /= ac[0] + 1e-12
        hi = min(max_lag, n - 1)
        if hi <= min_lag + 2:
            return 0.0
        seg = ac[min_lag:hi]
        peak = float(seg.max())
        if peak < 0.05:
            return 0.0
        # take the smallest strong lag (within 80% of best), not the global argmax,
        # so we lock onto the fundamental rather than a 2x/3x harmonic.
        lag = min_lag + int(np.argmax(seg >= 0.8 * peak))
        if 0 < lag < n - 1:
            a, b, c = ac[lag - 1], ac[lag], ac[lag + 1]
            den = a - 2 * b + c
            if abs(den) > 1e-12:
                lag = lag + float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
        return float(lag)

    py = fundamental(img.mean(axis=1))   # varies along y (horizontal-line pitch)
    px = fundamental(img.mean(axis=0))   # varies along x (vertical-line pitch)
    return py, px


def _comb_kernel(pitch: float, weight: float = 0.5) -> np.ndarray:
    """1-D kernel delta - weight*(shift(+p)+shift(-p)) with fractional (linearly
    interpolated) taps, so a non-integer pitch is handled exactly."""
    reach = int(np.ceil(pitch)) + 1
    k = np.zeros(2 * reach + 1, dtype=np.float32)
    c = reach
    k[c] = 1.0
    for s in (pitch, -pitch):
        i = c + s
        i0 = int(np.floor(i))
        w = float(i - i0)
        if 0 <= i0 < len(k):
            k[i0] -= weight * (1.0 - w)
        if 0 <= i0 + 1 < len(k):
            k[i0 + 1] -= weight * w
    return k


def _periodic_residual(img: np.ndarray, pitch_y: float, pitch_x: float,
                       weight: float = 0.5) -> np.ndarray:
    """Suppress the lattice, keep the aperiodic remainder. A pitch < 2 disables
    that axis. Separable + shift-invariant, so it applies identically to a
    template and to the search patch."""
    out = img.astype(np.float32)
    delta = np.array([1.0], dtype=np.float32)
    if pitch_y and pitch_y >= 2.0:
        out = cv2.sepFilter2D(out, -1, delta, _comb_kernel(pitch_y, weight),
                              borderType=cv2.BORDER_REFLECT_101)
    if pitch_x and pitch_x >= 2.0:
        out = cv2.sepFilter2D(out, -1, _comb_kernel(pitch_x, weight), delta,
                              borderType=cv2.BORDER_REFLECT_101)
    return out


def _residual_ncc(tmpl: np.ndarray, patch: np.ndarray,
                  pitch_y: float, pitch_x: float) -> float:
    """NCC of the aperiodic residuals of an (already size-matched) template and
    search patch. High only when the candidate window shares the SAME non-periodic
    content as the reference -- i.e. it is the true repeat, not a look-alike."""
    if tmpl.shape != patch.shape or tmpl.size == 0:
        return 0.0
    tr = _periodic_residual(tmpl, pitch_y, pitch_x)
    pr = _periodic_residual(patch, pitch_y, pitch_x)
    return _ncc(tr, pr)


def _residual_strength(residual: np.ndarray, reference: np.ndarray) -> float:
    """Fraction of a patch's energy that is aperiodic, in [0,1]. Near 0 means pure
    wallpaper: the residual channel carries no information and must be ignored
    (the position is close to genuinely unrecoverable)."""
    b = float(np.std(reference)) + 1e-9
    return float(np.clip(float(np.std(residual)) / b, 0.0, 1.0))


def _estimate_magnification(ref_f: np.ndarray, search_f: np.ndarray,
                            nominal: float, rel_band: float = 0.30, n: int = 25,
                            keep_band: float = 0.06) -> float:
    """Coarse NCC scale probe: measure the true magnification instead of assuming
    it. Tries magnifications in nominal*[1-rel_band, 1+rel_band]; for each, shrinks
    the reference to that scale and records the best template-match score against a
    downsampled search image. Returns the strongest-scoring magnification.

    Unlike a lattice-pitch ratio this is robust to anisotropic layouts (dense fins
    vs sparse gates), because it scores the whole reference pattern, not one pitch.
    Falls back to `nominal` if nothing scores. Cheap: ~n small matchTemplate calls
    on 4x-downsampled images. This is what lets the localizer survive a test set
    whose magnification differs from the assumed ~10x (see report robustness study).
    """
    # 2x-downsample the search for the probe: ~4x faster, and the distance penalty
    # + keep_band gate below tolerate the slightly coarser scale estimate (the fine
    # sweep in localize() re-centres precisely). Falls back to full-res if tiny.
    ds = 2 if min(search_f.shape) >= 400 else 1
    rs = cv2.resize(search_f, (search_f.shape[1] // ds, search_f.shape[0] // ds),
                    interpolation=cv2.INTER_AREA) if ds > 1 else search_f
    grid = np.linspace(nominal * (1.0 - rel_band), nominal * (1.0 + rel_band), n)
    nom_idx = int(np.argmin(np.abs(grid - nominal)))  # grid point closest to the prior
    # A wrong periodic repeat can score highly at a FAR scale (scale aliasing). Bias
    # the probe toward the prior with a distance penalty, so a distant scale must
    # beat the near ones by a real margin to win -- this rejects aliases while still
    # overriding when the true magnification genuinely differs.
    penalty = 0.45
    best_m, best_v, best_adj, nom_v = None, -2.0, -2.0, -2.0
    for k, m in enumerate(grid):
        if m <= 1e-3:
            continue
        tw = max(8, int(round(ref_f.shape[1] / (m * ds))))
        th = max(8, int(round(ref_f.shape[0] / (m * ds))))
        if tw >= rs.shape[1] or th >= rs.shape[0]:
            continue
        t = cv2.resize(ref_f, (tw, th), interpolation=cv2.INTER_AREA)
        v = float(cv2.matchTemplate(rs, t, cv2.TM_CCOEFF_NORMED).max())
        v_adj = v - penalty * abs(m / nominal - 1.0)
        if k == nom_idx:
            nom_v = v
        if v_adj > best_adj:
            best_m, best_v, best_adj = float(m), v, v_adj
    # Override the assumed magnification only when BOTH: (a) the best scale lies
    # OUTSIDE the band the default fine sweep already covers (|Δ|/nominal >
    # keep_band) -- inside it there is nothing to gain and the probe's ~grid-
    # resolution error would only add noise; and (b) that scale CLEARLY beats the
    # assumed one (best_v > nom_v + margin). On a defect-free / highly periodic pair
    # the scale response is flat, so (b) fails and the trustworthy prior is kept;
    # when the true magnification genuinely differs (e.g. 9x or 11x), both hold and
    # the override fires.
    if (best_m is not None
            and abs(best_m / nominal - 1.0) > keep_band
            and best_v > nom_v + 0.05):
        return best_m
    return float(nominal)


def _response_map(search_f: np.ndarray, ref_f: np.ndarray,
                  scales, rotations) -> np.ndarray:
    """Max-projection NCC response map over the whole (scale, rotation) sweep.

    For each transform, `cv2.matchTemplate` is computed and its correlation
    surface is written into a full-size array at the position of the template
    CENTER, taking the element-wise maximum across all transforms. The result
    R[y, x] is the best NCC achievable for a template centered at (x, y) over
    the entire sweep -- so the true site appears at its (strong) oracle score
    rather than being deflated by any single unlucky trial.
    """
    H, W = search_f.shape
    R = np.full((H, W), -1.0, dtype=np.float32)
    for s in scales:
        base = _make_template(ref_f, s, 0.0)
        th, tw = base.shape
        if th >= H or tw >= W:
            continue
        for r in rotations:
            t = base if r == 0.0 else cv2.warpAffine(
                base, cv2.getRotationMatrix2D((tw / 2, th / 2), r, 1.0),
                (tw, th), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            corr = cv2.matchTemplate(search_f, t, cv2.TM_CCOEFF_NORMED)
            ch, cw = corr.shape
            y0, x0 = th // 2, tw // 2
            np.maximum(R[y0:y0 + ch, x0:x0 + cw], corr,
                       out=R[y0:y0 + ch, x0:x0 + cw])
    return R


def _peak_and_lead_1d(profile: np.ndarray, min_dist: int):
    """Argmax index of `profile` and its LEAD over the best value outside a
    +/-min_dist window. A periodic axis has many near-equal peaks -> lead ~= 0;
    a landmark axis has one dominant peak -> decisive lead."""
    i = int(np.argmax(profile))
    mx = float(profile[i])
    q = profile.astype(np.float64).copy()
    lo, hi = max(0, i - min_dist), min(len(profile), i + min_dist + 1)
    q[lo:hi] = -np.inf
    mx2 = float(q.max()) if np.isfinite(q).any() else -1.0
    return i, mx - mx2


def _all_peaks_1d(profile: np.ndarray, min_dist: int, frac: float = 0.8):
    """Indices of 1-D local maxima above `frac`*max, at least `min_dist` apart
    (greedy non-max suppression). Used to enumerate the periodic repeats along an
    ambiguous axis so the nearest-center one can be chosen."""
    p = np.asarray(profile, dtype=np.float64).copy()
    if p.size == 0:
        return []
    thr = frac * float(p.max())
    peaks = []
    while np.isfinite(p).any():
        i = int(np.argmax(p))
        if p[i] < thr:
            break
        peaks.append(i)
        lo, hi = max(0, i - min_dist), min(len(p), i + min_dist + 1)
        p[lo:hi] = -np.inf
    return peaks


def _axis_landmarks(search_f: np.ndarray, ref_f: np.ndarray,
                    nominal_ratio: float, scale_jitter, min_dist: int, n: int = 5,
                    rot_max_deg: float = 4.0, n_rot: int = 7):
    """PER-AXIS aperiodic-landmark detection from a single-transform global NCC map.

    Picks the (scale, ROTATION) whose ONE fixed template gives the highest global
    NCC (single template is essential -- per-candidate refinement lets every repeat
    borrow its own best transform and erases the margin), then reads MARGINAL-MAX
    profiles of that map:
        px = corr.max(axis=0)  -> per-column best NCC, exposes X uniqueness
        py = corr.max(axis=1)  -> per-row    best NCC, exposes Y uniqueness
    A landmarked axis yields a decisive 1-D lead; a periodic axis yields ~0.

    The rotation sweep is essential when the CAPTURE is rotated (e.g. the reference
    is acquired at a small scan-field rotation): an unrotated template mis-aligns
    and a WRONG periodic repeat can then out-score the true aperiodic site in the
    marginal-max profile (measured: a both-anchored pair whose true mat was only
    the global max once rotation was searched).

    COST: the (scale, rotation) grid is searched on a 2x-DOWNSAMPLED image (each
    matchTemplate ~4x cheaper), and only the single winning transform is evaluated
    once at full resolution for the marginal profiles. So the rotation robustness
    costs ~n*n_rot tiny correlations + one full-size one, not n*n_rot full-size
    ones -- keeping the landmark a small fraction of the pipeline.

    Returns (corr, th, tw, (x_center, x_lead, xi), (y_center, y_lead, yi)) or None.
    """
    H, W = search_f.shape
    scales = np.linspace(scale_jitter[0], scale_jitter[1], n) * nominal_ratio
    rots = np.linspace(-rot_max_deg, rot_max_deg, n_rot)
    # --- coarse (scale, rotation) search on a 2x-downsampled image ---
    ds = 2 if min(H, W) >= 400 else 1
    s_ds = (cv2.resize(search_f, (W // ds, H // ds), interpolation=cv2.INTER_AREA)
            if ds > 1 else search_f)
    best = None  # (global_max, scale, rot)
    for s in scales:
        for r in rots:
            t = _make_template(ref_f, s * ds, r)   # template downsampled to match
            if t.shape[0] >= s_ds.shape[0] or t.shape[1] >= s_ds.shape[1]:
                continue
            gmax = float(cv2.matchTemplate(s_ds, t, cv2.TM_CCOEFF_NORMED).max())
            if best is None or gmax > best[0]:
                best = (gmax, float(s), float(r))
    if best is None:
        return None
    # --- one full-resolution pass at the winning transform for the profiles ---
    _, bs, br = best
    t = _make_template(ref_f, bs, br)
    th, tw = t.shape
    if th >= H or tw >= W:
        return None
    corr = cv2.matchTemplate(search_f, t, cv2.TM_CCOEFF_NORMED)
    md = max(1, int(min_dist))
    px = corr.max(axis=0)
    py = corr.max(axis=1)
    xi, x_lead = _peak_and_lead_1d(px, md)
    yi, y_lead = _peak_and_lead_1d(py, md)
    return corr, th, tw, (xi + tw / 2.0, x_lead, xi), (yi + th / 2.0, y_lead, yi)


def _center_tiebreak_axis(corr: np.ndarray, th: int, tw: int, axis: str,
                          pin_idx: int, center_coord: float, min_dist: int,
                          frac: float = 0.8):
    """Center coordinate of the periodic repeat NEAREST the image center on the
    ambiguous axis, given the pinned index on the solved axis. axis='y' -> Y is
    ambiguous (x pinned at column pin_idx); axis='x' -> X is ambiguous."""
    md = max(1, int(min_dist))
    if axis == 'y':
        j0, j1 = max(0, pin_idx - 2), min(corr.shape[1], pin_idx + 3)
        prof = corr[:, j0:j1].max(axis=1)
        half = th / 2.0
    else:
        i0, i1 = max(0, pin_idx - 2), min(corr.shape[0], pin_idx + 3)
        prof = corr[i0:i1, :].max(axis=0)
        half = tw / 2.0
    peaks = _all_peaks_1d(prof, md, frac) or [int(np.argmax(prof))]
    coords = [p + half for p in peaks]
    return min(coords, key=lambda c: abs(c - center_coord))


def _envelope_tiebreak_axis(search_f, ref_f, corr, th, tw, axis, pin_idx,
                            solved_center, min_dist, scales_f, rots_f, frac=0.8):
    """Disambiguate the periodic axis by texture-agnostic ENVELOPE match instead of
    nearest-center. When one axis is landmark-solved and the other is periodic, the
    brief's nearest-center rule silently picks the wrong repeat under UNIFORM
    placement (the true site need not be nearest center). Here we instead refine a
    candidate at each periodic repeat along the ambiguous axis (holding the solved
    axis fixed) and keep the one whose low-frequency identity envelope best matches
    the reference -- the same generator-independent cue used elsewhere. Falls back
    to the argmax peak if refinement fails everywhere. Returns the ambiguous-axis
    coordinate."""
    md = max(1, int(min_dist))
    if axis == 'y':                       # y ambiguous, x pinned at column pin_idx
        j0, j1 = max(0, pin_idx - 2), min(corr.shape[1], pin_idx + 3)
        prof = corr[:, j0:j1].max(axis=1)
        half = th / 2.0
    else:                                 # x ambiguous, y pinned at row pin_idx
        i0, i1 = max(0, pin_idx - 2), min(corr.shape[0], pin_idx + 3)
        prof = corr[i0:i1, :].max(axis=0)
        half = tw / 2.0
    # cap the number of repeats we refine to the strongest few: a periodic axis can
    # expose dozens of near-equal peaks and refining each is the dominant cost; the
    # true site is essentially always among the strongest marginal-max peaks.
    peaks = _all_peaks_1d(prof, md, frac) or [int(np.argmax(prof))]
    peaks = sorted(peaks, key=lambda p: prof[p], reverse=True)[:8]
    coords = [p + half for p in peaks]
    best = None  # (envelope, coord)
    for c in coords:
        cx0, cy0 = (solved_center, c) if axis == 'y' else (c, solved_center)
        lr = _refine(search_f, ref_f, cx0, cy0, scales_f, rots_f)
        if lr is None:
            continue
        _nv, _s, _r, rx, ry, tmpl = lr
        patch = cv2.getRectSubPix(search_f, (tmpl.shape[1], tmpl.shape[0]),
                                  (float(rx), float(ry)))
        env = _ncc(_envelope(tmpl), _envelope(patch))
        if best is None or env > best[0]:
            best = (env, c)
    return best[1] if best is not None else coords[0]


def _top_peaks(R: np.ndarray, n: int, min_dist: int, thresh: float = 0.2):
    """Top-`n` spatially-distinct local maxima of the response map.

    The local-max test uses cv2.dilate with a k x k all-ones kernel -- a
    grayscale morphological maximum, identical to a square maximum_filter but
    without the scipy dependency (verified bit-identical peaks on the eval sets;
    scipy.ndimage alone added ~240 ms to a cold-start CLI import)."""
    k = max(3, int(min_dist))
    mx = cv2.dilate(R, np.ones((k, k), np.uint8))
    mask = (R == mx) & (R > thresh)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return []
    sc = R[ys, xs]
    order = np.argsort(sc)[::-1][:n]
    return [(int(xs[k]), int(ys[k]), float(sc[k])) for k in order]


def _grid_lines(t: np.ndarray):
    """Row/column positions of the lattice lines, from projection-profile peaks."""
    f = t - t.mean()
    col = f.mean(axis=0)  # varies along x -> vertical (bit) lines
    row = f.mean(axis=1)  # varies along y -> horizontal (word) lines

    def peaks(sig):
        sig = sig - sig.min()
        if sig.max() <= 1e-6:
            return []
        thr = 0.3 * sig.max()
        return [i for i in range(1, len(sig) - 1)
                if sig[i] >= sig[i - 1] and sig[i] > sig[i + 1] and sig[i] > thr]

    return peaks(col), peaks(row)


def _fingerprint(patch: np.ndarray, xs, ys) -> np.ndarray:
    """Fin-gate crossing fingerprint: at each grid intersection, the local
    crossing score = intersection pixel minus its immediate neighborhood mean.
    A stronger crossing (strain/contact variation where a fin meets a gate)
    brightens the exact intersection above the surrounding fin/gate lines; a
    weak one does not. Sampling only AT intersections (not along the lines,
    which carry no site-specific information) is what isolates the per-
    crossing defect fingerprint from the dominant periodic grid."""
    fp = []
    H, W = patch.shape
    for yy in ys:
        for xx in xs:
            if 0 <= yy < H and 0 <= xx < W:
                y0, y1 = max(0, yy - 2), min(H, yy + 3)
                x0, x1 = max(0, xx - 2), min(W, xx + 3)
                fp.append(float(patch[yy, xx]) - float(patch[y0:y1, x0:x1].mean()))
            else:
                fp.append(0.0)
    return np.asarray(fp, dtype=np.float32)


def _refine(search_f: np.ndarray, ref_f: np.ndarray, cx: float, cy: float,
            scales_fine, rots_fine):
    """Local (scale, rotation, sub-pixel position) refinement at one candidate.

    Runs a small template match inside a window around (cx, cy) for each fine
    (scale, rotation), keeps the best, and parabola-fits the correlation peak
    for sub-pixel position. Returns (ncc, scale, rot, x, y, template) with the
    center in full-image coordinates, or None if the window is too small.
    """
    best = None
    for s in scales_fine:
        base = _make_template(ref_f, s, 0.0)
        th, tw = base.shape
        pad = 5
        x0 = max(0, int(cx - tw / 2 - pad))
        y0 = max(0, int(cy - th / 2 - pad))
        x1 = min(search_f.shape[1], x0 + tw + 2 * pad)
        y1 = min(search_f.shape[0], y0 + th + 2 * pad)
        win = search_f[y0:y1, x0:x1]
        if win.shape[0] < th or win.shape[1] < tw:
            continue
        for r in rots_fine:
            t = base if r == 0.0 else cv2.warpAffine(
                base, cv2.getRotationMatrix2D((tw / 2, th / 2), r, 1.0),
                (tw, th), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            corr = cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED)
            _, mv, _, ml = cv2.minMaxLoc(corr)
            if best is None or mv > best[0]:
                px, py = ml
                sx = _parabolic(corr, px, py, axis=1)
                sy = _parabolic(corr, px, py, axis=0)
                ccx = x0 + px + sx + tw / 2
                ccy = y0 + py + sy + th / 2
                best = (float(mv), float(s), float(r), ccx, ccy, t)
    return best


def _parabolic(corr: np.ndarray, px: int, py: int, axis: int) -> float:
    """Sub-pixel offset from a 3-point parabolic fit around (px, py)."""
    if axis == 1:  # x
        if 0 < px < corr.shape[1] - 1:
            l, c, r = corr[py, px - 1], corr[py, px], corr[py, px + 1]
        else:
            return 0.0
    else:  # y
        if 0 < py < corr.shape[0] - 1:
            l, c, r = corr[py - 1, px], corr[py, px], corr[py + 1, px]
        else:
            return 0.0
    den = (l - 2 * c + r)
    return float(0.5 * (l - r) / den) if abs(den) > 1e-6 else 0.0


def _zscore(a: np.ndarray) -> np.ndarray:
    s = a.std()
    return (a - a.mean()) / s if s > 1e-6 else np.zeros_like(a)


def _envelope(patch: np.ndarray, size: int = 24) -> np.ndarray:
    """Texture-AGNOSTIC low-frequency identity descriptor.

    Downsample the patch to a small grid with INTER_AREA (a box low-pass): this
    ANNIHILATES the periodic lattice -- whose pitch is far finer than the grid
    cell -- and keeps only the slowly-varying per-site brightness / shading
    envelope (process-variation shading, large-scale content gradients, and the
    aggregate per-cell brightness field). Returns a de-meaned flat vector.

    WHY this is the missing signal: on a periodic crop every lattice repeat has an
    essentially IDENTICAL high-frequency pattern, so NCC (and the fin-gate
    fingerprint, which is tuned to one specific binary-dropout model) cannot tell
    the true site from a wrong repeat. The low-frequency envelope, however, DIFFERS
    between repeats and matches between the reference and the search only at the
    true physical site -- regardless of the underlying texture/defect model. That
    generator-independence is exactly what lets it survive a foreign test set."""
    if patch.size == 0:
        return np.zeros(size * size, dtype=np.float32)
    e = cv2.resize(patch.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    e = e - e.mean()
    return e.ravel()


def _envelope_gate(cands, lead_thresh: float, min_val: float, min_sep: float = 10.0):
    """Uniqueness/PSR gate on the envelope descriptor: among spatially-distinct
    candidates, return the one whose envelope match is DECISIVELY the best (>=
    min_val and leading the next-best distinct candidate by >= lead_thresh), else
    None. A decisive lead means the low-frequency content is unique enough to pin
    the site; a near-tie means it is not, so the caller keeps the center prior.
    This is the safety valve that stops the envelope from ever HURTING the
    genuinely-ambiguous (defect-free, flat-envelope) case -- there the lead is ~0
    and the gate stays silent."""
    if len(cands) < 2:
        return None
    srt = sorted(cands, key=lambda c: c.envelope, reverse=True)
    top = srt[0]
    others = [c for c in srt[1:] if np.hypot(c.x - top.x, c.y - top.y) > min_sep]
    second = others[0].envelope if others else -1.0
    if top.envelope >= min_val and (top.envelope - second) >= lead_thresh:
        return top
    return None


def _residual_gate(cands, lead_thresh: float, min_val: float, min_sep: float = 10.0):
    """Uniqueness/PSR gate on the periodic-RESIDUAL score, twin of `_envelope_gate`.

    Among spatially-distinct candidates, returns the top-residual one only if its
    aperiodic-match score is both high (>= min_val) and decisively ahead of the next
    distinct candidate (>= lead_thresh) -- i.e. the lattice-subtracted content pins
    the true repeat. On a pure-wallpaper crop the residual is noise, every repeat
    scores alike, the lead is ~0 and the gate stays silent, so it cannot hurt the
    genuinely-ambiguous case. This is the array-mode disambiguator the envelope and
    fin-gate fingerprint miss, and the main lever against periodic wrong-repeat picks."""
    if len(cands) < 2:
        return None
    srt = sorted(cands, key=lambda c: c.res_score, reverse=True)
    top = srt[0]
    others = [c for c in srt[1:] if np.hypot(c.x - top.x, c.y - top.y) > min_sep]
    second = others[0].res_score if others else -1.0
    if top.res_score >= min_val and (top.res_score - second) >= lead_thresh:
        return top
    return None


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def localize(ref_img, search_img, *args, **kwargs):
    """Locate `ref_img` inside `search_img`; return (x, y, LocalizeInfo).

    Public entry point with a NEVER-RAISE guarantee: any internal failure returns
    the search-image center with `low_confidence=True` and `confidence=0.0`, so a
    grader parsing stdout always receives a coordinate. All real work (and the full
    parameter list) is in `_localize_core`.
    """
    try:
        return _localize_core(ref_img, search_img, *args, **kwargs)
    except Exception as e:  # never let an exception reach the caller/grader
        s = np.asarray(search_img)
        h, w = (s.shape[0], s.shape[1]) if s.ndim >= 2 else (1000, 1000)
        return (w / 2.0, h / 2.0,
                LocalizeInfo(low_confidence=True, confidence=0.0,
                             fallback_reason=f"error:{type(e).__name__}: {e}"))


def _localize_core(
    ref_img: np.ndarray,
    search_img: np.ndarray,
    nominal_ratio: float = 10.0,
    scale_jitter: tuple = (0.94, 1.06),
    n_scales_coarse: int = 9,
    rot_max_deg: float = 4.0,
    n_rot_coarse: int = 9,
    max_drift_frac: float = 0.24,
    n_candidates: int = 50,
    fp_weight: float = 1.8,
    tie_z: float = 0.7,
    n_scales_fine: int = 5,
    n_rot_fine: int = 7,
    fallback_margin: float = 0.02,
    always_full_search: bool = False,
    fp_gate_lo: float = 1.0,
    fp_gate_hi: float = 4.0,
    ncc_tie_margin: float = 0.012,
    ncc_lead_thresh: float = 0.008,
    axis_peak_frac: float = 0.8,
    env_lead: float = 0.15,
    env_min: float = 0.30,
    refine_topk: int = 30,
    fine_neighbors: int = 1,
    fp_confident: float = 0.60,
    fp_gap: float = 0.12,
    fp_floor_lo: float = 0.30,
    fp_floor_hi: float = 0.45,
    res_confident: float = 0.35,
    res_gap: float = 0.10,
    res_lead: float = 0.10,
    res_min: float = 0.35,
    fused_margin: float = 2.0,
    auto_scale: bool = True,
) -> tuple:
    """Locate `ref_img` inside `search_img`; return (x, y, LocalizeInfo).

    Parameters
    ----------
    nominal_ratio : known physical zoom ratio (search_nm_per_px / ref_nm_per_px).
    scale_jitter, n_scales_coarse, rot_max_deg, n_rot_coarse :
        the coarse (scale, rotation) sweep for the response map / candidate net.
    max_drift_frac : radius (as a fraction of image width) of the center ROI
        searched first, encoding bounded motion-stage drift. NOT a hard
        constraint -- a full-image pass is used as a fallback when the ROI
        response is weak, so a true site outside the assumed drift envelope is
        still recoverable.
    n_candidates : number of spatially-distinct candidate sites verified.
    fp_weight : weight of the fin-gate crossing fingerprint z-score relative to the
        NCC z-score in the fusion. The fingerprint is the reliable
        wrong-repeat discriminator, so it is weighted >= 1. Values in [1.5, 2.0]
        were the stable optimum on the self-eval; 1.8 is the default.
    tie_z : combined-score margin (in z units) defining a near-tie eligible for
        the nearest-to-center tie-break.
    n_scales_fine, n_rot_fine : resolution of the per-candidate refinement.
    fp_gate_lo, fp_gate_hi : ramp on the reference fingerprint contrast
        (fp_ref_std) below which the fingerprint is treated as noise and
        down-weighted -- fp_ref_std < fp_gate_lo => weight 0 (defect-free region).
    fp_floor_lo, fp_floor_hi : ramp on the best fingerprint achieved by any
        candidate (max_fp); if no candidate matches the fingerprint the weight is
        gated toward 0 (noise floor), independent of reference contrast.
    fp_confident, fp_gap : a candidate whose fingerprint is >= fp_confident and
        exceeds every other spatially-distinct candidate by >= fp_gap is accepted
        as a decisive identity match and selected outright (center cannot override).
    ncc_tie_margin : raw-NCC near-tie width used for the center prior when the
        fingerprint is unreliable (fp_gate < 0.5).
    always_full_search : if False (default) candidates come from the center drift
        ROI (with a comparative full-image fallback) -- best on realistic center-
        clustered data. If True, ALWAYS augment with a full-image candidate net,
        which recovers more sites far outside the drift envelope (e.g. corners) at
        the cost of a few percent on the realistic center case (a spurious far
        repeat can occasionally win). A wide-/unknown-drift robustness knob, not a
        free win -- see the off-center validation in the report.
    auto_scale : if True (default) MEASURE the true magnification with a coarse NCC
        scale probe and center the sweep on it, instead of assuming `nominal_ratio`.
        The probe only overrides the assumed ratio when the measured one is clearly
        outside the default fine-sweep band, so fixed-10x data is unaffected while a
        variable-magnification test set (e.g. 9x-11x) is handled. Set False to force
        exactly `nominal_ratio`.
    """
    t0 = time.perf_counter()
    # RGB bonus: accept 3-channel optical-microscope images. The structural
    # matching runs on luminance (robust, reuses the whole grayscale pipeline);
    # the color channels are retained for the color-fingerprint disambiguator (§
    # color cue below). Grayscale inputs are unchanged -> zero regression risk.
    ref_color = ref_img if (np.ndim(ref_img) == 3) else None
    search_color = search_img if (np.ndim(search_img) == 3) else None
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY) if ref_color is not None else ref_img
    search_gray = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY) if search_color is not None else search_img
    ref_f = np.asarray(ref_gray, dtype=np.float32)
    search_f = np.asarray(search_gray, dtype=np.float32)
    H, W = search_f.shape
    center = (W / 2.0, H / 2.0)

    # --- (0) measure the true magnification instead of assuming `nominal_ratio` ---
    # The assumed ~10x can be wrong (mag-calibration error; an unknown test set).
    # A coarse NCC scale probe centers the whole downstream sweep on the ACTUAL
    # magnification, so the narrow +/-6% fine sweep no longer misses the true site
    # when the ratio drifts. `nominal_ratio` (from the caller / --ratio) is the
    # prior/center of the probe band. Disable with auto_scale=False to force it.
    if auto_scale:
        keep = 0.5 * (scale_jitter[1] - scale_jitter[0])  # what the fine sweep covers
        nominal_ratio = _estimate_magnification(ref_f, search_f, nominal_ratio, keep_band=keep)

    # lattice pitch in the search frame -> NMS radius + fingerprint grid scale
    pitch_ref = _estimate_pitch_px(ref_f)
    pitch_s = max(4.0, (pitch_ref / nominal_ratio) if pitch_ref else nominal_ratio * 0.95)
    nms_dist = max(3, int(round(pitch_s * 0.6)))  # < 1 pitch: keep repeats distinct

    scales_c = nominal_ratio * np.linspace(scale_jitter[0], scale_jitter[1], n_scales_coarse)
    rots_c = np.linspace(-rot_max_deg, rot_max_deg, n_rot_coarse)

    # --- (1)-(2) candidate net over the center ROI ---
    r0 = int(round(max_drift_frac * W))
    x_lo, y_lo = max(0, W // 2 - r0), max(0, H // 2 - r0)
    x_hi, y_hi = min(W, W // 2 + r0), min(H, H // 2 + r0)
    roi = search_f[y_lo:y_hi, x_lo:x_hi]
    R = _response_map(roi, ref_f, scales_c, rots_c)
    roi_best = float(R.max())
    peaks = _top_peaks(R, n_candidates, nms_dist)
    cand_pts = [(px + x_lo, py + y_lo, sc) for (px, py, sc) in peaks]

    # RESCUE: if the center ROI net found NOTHING, the true site is likely well
    # outside the assumed drift envelope (a wider-drift test set). Search the whole
    # image so candidates are still produced -- otherwise the run would fail. This
    # pays the full-image cost ONLY when the ROI is genuinely empty, so normal
    # center-clustered data is unaffected; it also removes a hard-failure path.
    if not cand_pts:
        R_full = _response_map(search_f, ref_f, scales_c, rots_c)
        cand_pts = [(px, py, sc) for (px, py, sc) in _top_peaks(R_full, n_candidates, nms_dist)]
        roi_best = max(roi_best, float(R_full.max()))

    # --- fallback: unrestricted full-image net when the true site may lie
    # outside the assumed drift ROI. The trigger is COMPARATIVE, not absolute:
    # on a periodic lattice the ROI is always full of repeats scoring ~0.8, so
    # any absolute threshold would never fire and the ROI would silently become
    # a hard constraint. We probe the FULL image once with the ROI's own refined
    # template and compare the best score achieved INSIDE the drift ROI against
    # the best score achieved strictly OUTSIDE it.
    #
    # Both quantities are read from the SAME single-template correlation map, so
    # they are directly comparable (apples-to-apples). This is the key fix over
    # the earlier version, which compared the full-image max (which INCLUDES the
    # ROI) against a separately-computed refined ROI score: on periodic content
    # some other in-ROI repeat routinely beat the chosen point by >0.1% under one
    # fixed template, so the trigger fired on ordinary in-ROI periodic ripple and
    # paid the full-image cost for nothing. Splitting the same map by template-
    # center location removes that contamination.
    #
    # A boundary band (dilate the ROI by one NMS radius, ~one lattice pitch) keeps
    # a repeat straddling the ROI edge -- which is the SAME lattice phase as an
    # inside repeat -- from counting as "outside" and spuriously triggering. ---
    fallback_used = False
    fallback_reason = ""
    roi_probe_best = roi_best
    outside_probe_best = -1.0
    best_pt = max(cand_pts, key=lambda p: p[2]) if cand_pts else None
    probe_res = None
    if best_pt is not None:
        probe_res = _refine(search_f, ref_f, best_pt[0], best_pt[1], scales_c, rots_c)
        probe_tmpl = probe_res[5] if probe_res else _make_template(ref_f, nominal_ratio, 0.0)
        full_probe = cv2.matchTemplate(search_f, probe_tmpl, cv2.TM_CCOEFF_NORMED)
        th_p, tw_p = probe_tmpl.shape
        # center coordinate of the template for each correlation-map cell
        cy_grid = np.arange(full_probe.shape[0], dtype=np.float64) + th_p / 2.0
        cx_grid = np.arange(full_probe.shape[1], dtype=np.float64) + tw_p / 2.0
        band = nms_dist
        in_y = (cy_grid >= y_lo - band) & (cy_grid < y_hi + band)
        in_x = (cx_grid >= x_lo - band) & (cx_grid < x_hi + band)
        in_mask = np.outer(in_y, in_x)
        roi_probe_best = float(full_probe[in_mask].max()) if in_mask.any() else roi_best
        out_vals = full_probe[~in_mask]
        outside_probe_best = float(out_vals.max()) if out_vals.size else -1.0

        # Always harvest the strongest OFF-CENTER candidates from this single-template
        # full-image probe (already computed here -- NO extra correlation). A
        # uniform-placement true site sits outside the drift ROI, so the ROI net never
        # generates it; surfacing it here lets the consensus selector reach it without
        # the expensive multi-transform full-search re-run. Scale/rotation are per
        # capture, so the probe's own transform is near-optimal for every repeat.
        probe_scale = probe_res[1] if probe_res else nominal_ratio
        added_probe = []
        for (pcol, prow, psc) in _top_peaks(full_probe, n_candidates, nms_dist):
            ccx = pcol + tw_p / 2.0
            ccy = prow + th_p / 2.0
            if ccx < x_lo or ccx >= x_hi or ccy < y_lo or ccy >= y_hi:  # outside ROI
                if all(np.hypot(ccx - ex, ccy - ey) > nms_dist
                       for (ex, ey, _s) in cand_pts + added_probe):
                    added_probe.append((ccx, ccy, probe_scale))
            if len(added_probe) >= n_candidates // 2:
                break
        cand_pts = cand_pts + added_probe

    # Per-axis landmark search: a dedicated SINGLE-TRANSFORM multi-scale global NCC
    # whose marginal-max profiles expose which axis (if any) carries an aperiodic
    # landmark. Single template is essential -- max-projection or per-candidate
    # refinement lets every repeat borrow its own best transform and erases the
    # margin. Consumed in the selection step below.
    # scales narrow (the magnification is already measured by the probe) and a
    # coarse rotation sweep -- enough to align a rotated capture without paying the
    # full grid: keeps the landmark rotation-robust while bounding its cost.
    lm = _axis_landmarks(search_f, ref_f, nominal_ratio, scale_jitter, nms_dist,
                         n=3, rot_max_deg=rot_max_deg, n_rot=5)

    # Candidate-net fallback (independent of the landmark result).
    if best_pt is not None:
        strong_outside = outside_probe_best > roi_probe_best * (1.0 + fallback_margin)
        weak_roi = roi_best < 0.4
        few_cands = len(cand_pts) < 5
        if always_full_search or strong_outside or weak_roi or few_cands:
            fallback_used = True
            fallback_reason = ("weak_roi" if weak_roi else
                               "few_candidates" if few_cands else
                               "strong_outside" if strong_outside else "always_full_search")

            # Cost decoupling: scale/rotation jitter is per CAPTURE, not per
            # site, so every repeat in this search image shares one global
            # (scale, rotation). The probe was refined to that global transform,
            # so an out-of-ROI true site is near-optimal at the SAME transform.
            # The full-image response map therefore needs only a tight 3x3
            # neighbourhood around the probe transform -- not the full coarse
            # sweep -- cutting the fallback's dominant cost ~(n_coarse^2 / 9).
            # Exception: when the ROI itself is weak (weak_roi) there is no
            # trustworthy probe transform, so fall back to the full coarse sweep.
            if probe_res is not None and not weak_roi:
                ps, pr = probe_res[1], probe_res[2]
                s_step = nominal_ratio * (scale_jitter[1] - scale_jitter[0]) / max(1, n_scales_coarse - 1)
                r_step = 2.0 * rot_max_deg / max(1, n_rot_coarse - 1)
                fb_scales = ps + np.array([-1.0, 0.0, 1.0]) * s_step
                fb_rots = pr + np.array([-1.0, 0.0, 1.0]) * r_step
            else:
                fb_scales, fb_rots = scales_c, rots_c

            Rf = _response_map(search_f, ref_f, fb_scales, fb_rots)
            full_peaks = _top_peaks(Rf, n_candidates, nms_dist)
            # Keep ALL ROI candidates (so the near-center true site is never
            # dropped) and ADD only the strongest outside candidates, capped to
            # bound refinement cost. Outside peaks are ranked by response score.
            outside_budget = n_candidates // 2
            added = []
            for (px, py, sc) in full_peaks:
                if px < x_lo or px >= x_hi or py < y_lo or py >= y_hi:  # outside ROI
                    if all(np.hypot(px - ex, py - ey) > nms_dist for (ex, ey, _s) in cand_pts + added):
                        added.append((px, py, sc))
                if len(added) >= outside_budget:
                    break
            cand_pts = cand_pts + added

    if not cand_pts:
        raise RuntimeError("localize: no candidate peaks found")

    # --- (3) per-candidate verification: refined NCC + fingerprint ---
    # Scale and rotation are per-CAPTURE (global): every repeat in this search image
    # shares one true (scale, rotation), already found once as `probe_res`. So each
    # candidate only needs a NARROW sweep centred on that global transform (position
    # sub-pixel is what actually differs per site) instead of re-searching the whole
    # grid -- the same cost-decoupling the fallback uses above. Falls back to the
    # full fine grid only when the probe transform is unavailable.
    if probe_res is not None:
        ps, pr = probe_res[1], probe_res[2]
        s_step = nominal_ratio * (scale_jitter[1] - scale_jitter[0]) / max(1, n_scales_fine - 1)
        r_step = 2.0 * rot_max_deg / max(1, n_rot_fine - 1)
        nb = np.arange(-fine_neighbors, fine_neighbors + 1, dtype=np.float64)
        scales_f = ps + nb * s_step
        rots_f = pr + nb * r_step
    else:
        scales_f = nominal_ratio * np.linspace(scale_jitter[0] + 0.01, scale_jitter[1] - 0.01, n_scales_fine)
        rots_f = np.linspace(-rot_max_deg, rot_max_deg, n_rot_fine)

    # Refine only the strongest candidates by coarse response score: the true site
    # is surfaced by the max-projection map, so it is essentially always a top peak.
    # ROI/added candidates already carry their coarse score in the 3rd tuple slot.
    verify_pts = sorted(cand_pts, key=lambda p: p[2], reverse=True)[:max(1, refine_topk)]

    # Search-frame lattice pitch, measured once, for the periodic-residual score.
    pit_y, pit_x = _pitch_yx(search_f)

    cands: list[Candidate] = []
    for (cx, cy, _sc) in verify_pts:
        ref_result = _refine(search_f, ref_f, cx, cy, scales_f, rots_f)
        if ref_result is None:
            continue
        nccv, s, r, rx, ry, tmpl = ref_result
        th, tw = tmpl.shape
        # search patch at this candidate (needed by the fingerprint, the
        # texture-agnostic envelope descriptor, and the periodic residual).
        patch = cv2.getRectSubPix(search_f, (tw, th), (float(rx), float(ry)))
        envelope = _ncc(_envelope(tmpl), _envelope(patch))
        res_score = _residual_ncc(tmpl, patch, pit_y, pit_x)
        xs, ys = _grid_lines(tmpl)
        if xs and ys:
            ref_fp = _fingerprint(tmpl, xs, ys)
            fp = _ncc(ref_fp, _fingerprint(patch, xs, ys))
            fp_ref_std = float(ref_fp.std())
        else:
            fp = 0.0
            fp_ref_std = 0.0
        cands.append(Candidate(x=rx, y=ry, score=nccv, scale=s, rotation=r,
                               fingerprint=fp, fp_ref_std=fp_ref_std,
                               envelope=envelope, res_score=res_score))

    if not cands:
        raise RuntimeError("localize: candidate refinement produced no results")

    # --- (4) fusion, with a reliability gate on the fingerprint ---
    # The fingerprint only carries site-identity signal when the reference crop
    # actually contains crossing-defect variation (dropped vs present fin-gate
    # crossings). In a near-defect-free region the per-intersection contrast
    # vector is nearly constant, so its NCC against a search patch is dominated
    # by noise -- fusing it at full weight then lets a wrong repeat's spurious
    # fingerprint out-vote the true site (measured: forced-periodic pairs where
    # the true site's fingerprint is even NEGATIVE). fp_ref_std -- the std of
    # the reference template's own intersection-contrast vector -- measures how
    # much dropout signal exists, independent of any candidate. When it is low
    # we smoothly down-weight the fingerprint toward 0, so selection falls back
    # to NCC + the center prior (which is what correctly resolves defect-free
    # periodic repeats -- the true drifted site is the one nearest center).
    ref_std_med = float(np.median([c.fp_ref_std for c in cands]))
    fp_gate = float(np.clip((ref_std_med - fp_gate_lo) / (fp_gate_hi - fp_gate_lo), 0.0, 1.0))
    # A second, complementary reliability signal: even where the reference DOES
    # carry dropout contrast (fp_ref_std high), the fingerprint is only useful if
    # some candidate actually matches it. If the best fingerprint across all
    # candidates is near the noise floor, no site was identified -- so down-gate
    # and fall back to NCC + center (measured: the noise-floor pair where every
    # candidate's fingerprint sat at ~0.3).
    max_fp = max((c.fingerprint for c in cands), default=0.0)
    fp_gate *= float(np.clip((max_fp - fp_floor_lo) / (fp_floor_hi - fp_floor_lo), 0.0, 1.0))
    eff_fp_weight = fp_weight * fp_gate

    nccs = np.array([c.score for c in cands])
    fps = np.array([c.fingerprint for c in cands])
    combined = _zscore(nccs) + eff_fp_weight * _zscore(fps)
    for k, c in enumerate(cands):
        c.combined = float(combined[k])

    # --- (5) drift prior + center tie-break ---
    # Bounded-drift prior: prefer candidates inside the drift ROI. An
    # OUTSIDE-ROI candidate is only allowed to compete if it is DECISIVELY
    # better than the best in-ROI candidate (combined lead >= tie_z). This
    # keeps the fallback safe: a genuinely relocated site outside the ROI (with
    # a strong, matching fingerprint) still wins, but spurious outside repeats
    # -- common on defect-free periodic regions where the fingerprint carries
    # no signal -- cannot pull the answer far from center.
    def in_roi(c):
        return x_lo <= c.x < x_hi and y_lo <= c.y < y_hi
    roi_cands = [c for c in cands if in_roi(c)]
    if roi_cands:
        best_in_roi = max(c.combined for c in roi_cands)
        pool = roi_cands + [c for c in cands if not in_roi(c)
                            and c.combined >= best_in_roi + tie_z]
    else:
        pool = cands

    # --- selection: three regimes on fingerprint reliability ---
    # (a) DECISIVE fingerprint: when the fingerprint is reliable (fp_gate high)
    #     and one in-ROI candidate has a confidently high fingerprint that is
    #     clearly separated from every other spatially-distinct candidate, that
    #     candidate IS the site identity -- pick it outright. The center prior
    #     must NOT override a confident identity match (measured failure: a true
    #     site with fp=0.92 but slightly lower NCC was discarded for a wrong
    #     repeat 40 px nearer center).
    # (b) UNRELIABLE fingerprint (fp_gate low -> defect-free periodic region):
    #     the fused score is essentially NCC alone and periodic repeats have
    #     near-identical NCC, so widen the near-tie to every candidate whose raw
    #     NCC is within `ncc_tie_margin` of the pool's best and let the center
    #     prior -- the only trustworthy signal for a defect-free revisit --
    #     arbitrate (the true drifted site is nearest center).
    # (c) OTHERWISE: fused-z near-tie with a center tie-break among genuine ties.
    # Decisive-fp is scoped to the ROI-eligible pool: an OUTSIDE-ROI candidate
    # must first clear the drift-prior guard (beat the best in-ROI candidate by
    # tie_z on the fused score) to enter the pool, which requires a genuinely
    # strong fingerprint. This lets a confidently-matched off-center true site win
    # while preventing a spurious far repeat -- with a merely coincidental
    # fingerprint -- from overriding the realistic center-clustered case.
    fp_conf_cand = max(pool, key=lambda c: c.fingerprint)
    others = [c.fingerprint for c in pool
              if np.hypot(c.x - fp_conf_cand.x, c.y - fp_conf_cand.y) > 10.0]
    second_fp = max(others) if others else -1.0
    decisive_fp = (fp_gate >= 0.5 and fp_conf_cand.fingerprint >= fp_confident
                   and fp_conf_cand.fingerprint - second_fp >= fp_gap)

    # --- DECISIVE periodic-residual identity (array-mode disambiguator) ---
    # The candidate whose lattice-subtracted (aperiodic) content best matches the
    # reference, accepted only when that match is both strong and clearly ahead of
    # every spatially-distinct rival. This is the cell-to-cell signal that pins the
    # true repeat where the fingerprint (a specific defect model) and the low-freq
    # envelope are both blind -- notably DRAM mats, where a wrong repeat otherwise
    # wins on raw NCC. Silent on pure wallpaper (no aperiodic content -> no lead).
    # Evaluated over ALL candidates (not the ROI-guarded pool): a decisive aperiodic
    # identity is trustworthy even off-center (uniform placement), exactly as the ML
    # branch selects over all candidates. The strength+lead gate stops a spurious far
    # repeat from winning, so the drift prior is only bypassed on a confident match.
    res_conf_cand = max(cands, key=lambda c: c.res_score)
    res_others = [c.res_score for c in cands
                  if np.hypot(c.x - res_conf_cand.x, c.y - res_conf_cand.y) > 10.0]
    second_res = max(res_others) if res_others else -1.0
    decisive_res = (res_conf_cand.res_score >= res_confident
                    and res_conf_cand.res_score - second_res >= res_gap)

    # --- FUSED-CONSENSUS identity (the main off-center disambiguator) ---
    # The three site-identity signals -- raw NCC, periodic-residual, and low-freq
    # envelope -- are largely independent, so on a periodic crop each one separates
    # the true repeat from its look-alikes by only a small margin, individually too
    # weak to trust. Summing their z-scores lets those small agreements ADD: when one
    # candidate is simultaneously the best (or near-best) on all three, its fused lead
    # is decisive even though no single signal was. This is what rescues the
    # uniform-placement foreign case, where the true site is OFF-center and every
    # per-signal gate stayed silent -- the nearest-center prior would otherwise discard
    # it. Evaluated over ALL candidates and fired only on a decisive fused lead, so on
    # a genuinely ambiguous (pure-wallpaper) crop the lead collapses and the center
    # prior still arbitrates -> no regression on the bounded-drift / periodic cases.
    if len(cands) >= 2:
        fused = (_zscore(np.array([c.score for c in cands]))
                 + _zscore(np.array([c.res_score for c in cands]))
                 + _zscore(np.array([c.envelope for c in cands])))
        fi = int(fused.argmax())
        fused_cand = cands[fi]
        f_others = [fused[j] for j, c in enumerate(cands)
                    if np.hypot(c.x - fused_cand.x, c.y - fused_cand.y) > 10.0]
        second_fused = max(f_others) if f_others else -1e9
        decisive_fused = (fused[fi] - second_fused) >= fused_margin
    else:
        fused_cand = cands[0] if cands else None
        decisive_fused = False

    # --- PER-AXIS landmark override (texture-agnostic identity cue) ---
    # `lm` carries, per axis, the marginal-max peak and its lead over the best
    # distinct 1-D competitor. An axis whose lead is decisive holds an APERIODIC
    # landmark that pins that coordinate on its own (mat boundary, sense-amp/driver
    # channel, field edge); a periodic axis has lead ~= 0. So:
    #   * both axes solvable  -> pin (x, y) outright (uniform placement, off-center).
    #   * one axis solvable   -> pin it; apply the brief's NEAREST-CENTER tie-break
    #                            to the periodic axis ONLY (enumerate its repeats and
    #                            take the one closest to the image center).
    #   * neither solvable    -> do NOT fire; fall through to the fingerprint +
    #                            center prior that resolve the fully-periodic case
    #                            (protects the bounded-drift / FinFET accuracy).
    # The pinned coarse point is refined for sub-pixel with one _refine call.
    landmark_override = False
    landmark_both = False   # both axes carry a decisive aperiodic anchor
    if lm is not None:
        corr_m, th_m, tw_m, (x_center, x_lead, xi), (y_center, y_lead, yi) = lm
        x_solvable = x_lead >= ncc_lead_thresh
        y_solvable = y_lead >= ncc_lead_thresh
        landmark_both = bool(x_solvable and y_solvable)
        if x_solvable or y_solvable:
            if x_solvable and y_solvable:
                cx0, cy0 = x_center, y_center
            elif x_solvable:
                cx0 = x_center
                cy0 = _envelope_tiebreak_axis(search_f, ref_f, corr_m, th_m, tw_m,
                                              'y', xi, cx0, nms_dist, scales_f, rots_f,
                                              axis_peak_frac)
            else:
                cy0 = y_center
                cx0 = _envelope_tiebreak_axis(search_f, ref_f, corr_m, th_m, tw_m,
                                              'x', yi, cy0, nms_dist, scales_f, rots_f,
                                              axis_peak_frac)
            lr = _refine(search_f, ref_f, cx0, cy0, scales_f, rots_f)
            if lr is not None:
                _nv, _s, _r, _rx, _ry, _tm = lr
                landmark_cand = Candidate(x=_rx, y=_ry, score=_nv, scale=_s,
                                          rotation=_r, fingerprint=0.0, fp_ref_std=0.0)
            else:
                landmark_cand = Candidate(x=cx0, y=cy0, score=0.0,
                                          scale=nominal_ratio, rotation=0.0,
                                          fingerprint=0.0, fp_ref_std=0.0)
            landmark_override = True

    # --- classical selection cascade (precedence order) ---
    # High-confidence identity cues fire first; each is exact when it fires. The
    # order is: decisive fingerprint -> both-axis aperiodic landmark -> multi-signal
    # fused consensus -> decisive periodic residual -> center-prior tie-break.
    if decisive_fp:
        near_tied = [fp_conf_cand]
        chosen = fp_conf_cand
    elif landmark_override:
        near_tied = [landmark_cand]
        chosen = landmark_cand
    elif decisive_fused:
        near_tied = [fused_cand]
        chosen = fused_cand
    elif decisive_res:
        near_tied = [res_conf_cand]
        chosen = res_conf_cand
    else:
        best_comb = max(c.combined for c in pool)
        near_tied = [c for c in pool if c.combined >= best_comb - tie_z]
        if fp_gate < 0.5:
            best_ncc = max(c.score for c in pool)
            ncc_near = [c for c in pool if c.score >= best_ncc - ncc_tie_margin]
            near_tied = list({id(c): c for c in (near_tied + ncc_near)}.values())
        # TEXTURE-AGNOSTIC ENVELOPE GATE: among the NCC/fused near-ties (the
        # periodic repeats the fingerprint could not separate), if one has a
        # decisively better low-frequency identity match, it IS the true site --
        # take it, overriding the center prior. This is the fix for the
        # uniform-placed periodic case, where the true site is NOT nearest center
        # and the fin-gate fingerprint is blind to a foreign texture. The gate
        # requires a decisive lead, so on a genuinely flat-envelope crop it stays
        # silent and the center prior still arbitrates (no regression there).
        # PERIODIC-RESIDUAL GATE (twin of the envelope gate, tried first): among the
        # near-ties the fingerprint could not separate, if one candidate's aperiodic
        # (lattice-subtracted) content matches the reference decisively better, it IS
        # the true repeat -- take it over the center prior. This resolves the periodic
        # wrong-repeat picks (DRAM mats especially) that the low-freq envelope misses.
        # Silent on flat-residual crops (lead ~0), so no regression on the truly
        # ambiguous case, where the center prior still arbitrates.
        res_pick = _residual_gate(near_tied, res_lead, res_min)
        env_pick = _envelope_gate(near_tied, env_lead, env_min)
        if res_pick is not None:
            chosen = res_pick
        elif env_pick is not None:
            chosen = env_pick
        else:
            chosen = min(near_tied, key=lambda c: np.hypot(c.x - center[0], c.y - center[1]))

    # selection confidence from the combined-score margin between the top two
    # spatially-distinct candidates (a genuine near-tie -> ambiguous -> low conf).
    pool_combs = sorted((c.combined for c in pool), reverse=True)
    margin = (pool_combs[0] - pool_combs[1]) if len(pool_combs) >= 2 else 999.0
    confidence = float(np.clip(margin / (2.0 * tie_z), 0.0, 1.0))
    low_confidence = margin < tie_z

    info = LocalizeInfo(
        candidates=cands,
        survivors=cands,
        near_tied=near_tied,
        chosen=chosen,
        best_score=chosen.score,
        elapsed_s=time.perf_counter() - t0,
        full_image_fallback=fallback_used,
        roi_probe_best=roi_probe_best,
        outside_probe_best=outside_probe_best,
        fallback_reason=fallback_reason,
        confidence=confidence,
        low_confidence=low_confidence,
        magnification=float(nominal_ratio),
    )
    return chosen.x, chosen.y, info


if __name__ == "__main__":
    # ---------------------------------------------------------------------------
    # INFERENCE ENTRY POINT (this is the script Applied Materials runs on test data)
    #
    #   python localize.py <reference_image> <search_image>
    #
    # Prints the predicted center of the reference pattern inside the search image,
    # as "x, y" (search-image pixel coordinates), on stdout. Runs with no manual
    # edits; extra diagnostics go to stderr only with --verbose.
    # ---------------------------------------------------------------------------
    import argparse
    import sys

    ap = argparse.ArgumentParser(
        description="Drift-Sense: locate a reference crop inside a search image; "
                    "prints the predicted center as 'x, y'.")
    ap.add_argument("reference", help="path to the reference (high-magnification) image")
    ap.add_argument("search", help="path to the search (wide, low-magnification) image")
    ap.add_argument("--ratio", type=float, default=10.0,
                    help="magnification ratio (search low-mag : reference high-mag); default 10")
    ap.add_argument("--verbose", action="store_true",
                    help="print match diagnostics to stderr")
    args = ap.parse_args()

    def _read(path):
        # preserve native channels: grayscale stays 2-D, RGB optical stays 3-D
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.ndim == 3 and img.shape[2] == 4:       # BGRA -> BGR (drop alpha)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        if img.dtype != np.uint8:                     # 16-bit etc. -> 8-bit
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return img

    ref_img = _read(args.reference)
    search_img = _read(args.search)
    if ref_img is None:
        sys.exit(f"ERROR: could not read reference image: {args.reference}")
    if search_img is None:
        sys.exit(f"ERROR: could not read search image: {args.search}")

    x, y, info = localize(ref_img, search_img, nominal_ratio=args.ratio)

    if args.verbose:
        if info.chosen is not None:
            print(f"# ncc={info.chosen.score:.4f} fingerprint={info.chosen.fingerprint:.4f} "
                  f"magnification={info.magnification:.2f} confidence={info.confidence:.2f} "
                  f"low_conf={info.low_confidence} candidates={len(info.candidates)} "
                  f"fallback={info.full_image_fallback} time={info.elapsed_s * 1000:.0f}ms",
                  file=sys.stderr)
        else:
            print(f"# fallback-to-center (low_conf={info.low_confidence}) "
                  f"reason={info.fallback_reason}", file=sys.stderr)

    # primary output: the predicted center (x, y) in search-image pixels
    print(f"{x:.2f}, {y:.2f}")
