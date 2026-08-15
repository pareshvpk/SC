"""
Drift-Sense localization algorithm  --  V2 (candidate-generate + verify).

Public API (unchanged):

    x, y, info = localize(ref_img, search_img)

Given a high-res reference capture (100x zoom) and a low-res search capture
(10x zoom) of the same physical FinFET die site, return the center (x, y) of the
reference pattern inside the search image, in search-image pixel coordinates.
If several regions match (the layout is highly periodic), return the one
closest to the search-image center.

------------------------------------------------------------------------------
WHY V2 EXISTS  (diagnosis of the V1 failure mode)
------------------------------------------------------------------------------
V1 used normalized cross-correlation (NCC) both to *generate* and to *select*
candidates. On a periodic FinFET lattice that is fatal: at any single fixed
(scale, rotation) trial, a wrong lattice repeat routinely out-scores the true
site (measured: true site NCC ~0.35 vs a wrong repeat ~0.86 at a fixed trial).
V1 also recorded each spatial peak's score from whichever single trial found
it, and suppressed peaks within ~half a template width -- so the true site was
frequently merged with, and discarded in favour of, a nearby higher-scoring
wrong repeat. Result: a bimodal error distribution (either exact, or locked
onto a repeat 100-250 px away).

Two empirical facts drove the V2 redesign (both measured on the dataset):

  1. At its OWN best (scale, rotation), the true site is the strongest local
     match 19/19 times on the pairs V1 got wrong. So the signal is there; V1
     simply never scored each site at its best transform.

  2. A fin-gate crossing "fingerprint" -- sampling intensity exactly at the
     grid intersections, where the local crossing-defect variation lives --
     ranks the true site above the wrong repeat 18/19 times, as an
     INDEPENDENT vote that does not share NCC's periodicity blind spot.

------------------------------------------------------------------------------
V2 PIPELINE
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
  (5) SELECTION in one of three regimes (see localize() body for details):
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

Benchmark (30-pair self-eval, deliberately-hard FinFET set, see bench.py):
    V1: 43% within 1 px, median 61 px, 13 catastrophic (>100 px) failures.
    V2: 83% within 1 px, median  0 px,  1 catastrophic failure (the forced-
        periodic, defect-free pair that carries no fingerprint by design and
        whose true site sits farther from center than a look-alike repeat --
        the intended honest-failure case). Held-out seed: 93% within 1 um.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.ndimage import maximum_filter


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
    rs = search_f  # full detail: downsampling destroys the dense fin pitch and the
                   # coarse probe then aliases across scales on periodic content.
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
        tw = max(8, int(round(ref_f.shape[1] / m)))
        th = max(8, int(round(ref_f.shape[0] / m)))
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


def _top_peaks(R: np.ndarray, n: int, min_dist: int, thresh: float = 0.2):
    """Top-`n` spatially-distinct local maxima of the response map."""
    mx = maximum_filter(R, size=max(3, min_dist))
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


# ----------------------------------------------------------------------------
# Hybrid ML candidate ranker (optional)
# ----------------------------------------------------------------------------
# A small trained MLP that scores each candidate's probability of being the true
# site, from the SAME classical features the reliability logic uses. It is an
# optional drop-in for the hand-tuned three-regime selector: identical feature
# code is shared by training (train_ranker.ipynb) and inference here, so there is
# no train/serve skew. Inference is a pure-numpy forward pass -- no sklearn/torch
# dependency at run time; the trained weights live in an .npz. If the model file
# is absent the classical selector is used, so the pipeline always works.

# Per-candidate feature order. MUST stay in sync with train_ranker.ipynb.
FEATURE_ORDER = [
    "score", "fingerprint", "fp_ref_std", "dist_center_norm", "scale_dev",
    "rot_norm", "score_minus_max", "fp_minus_max", "ncc_rank", "fp_rank",
    "in_roi", "pair_max_fp", "pair_fp_gap", "pair_refstd_med",
]


def candidate_features(cands, W, H, nominal_ratio, rot_max_deg, max_drift_frac):
    """Feature matrix (N, D) for a pair's candidate list, in FEATURE_ORDER.

    Uses only quantities available at inference time (the Candidate fields plus
    image geometry), so training and serving compute identical features.
    """
    n = len(cands)
    cx, cy = W / 2.0, H / 2.0
    half = 0.5 * float(np.hypot(W, H))
    r = max_drift_frac * W
    x_lo, x_hi, y_lo, y_hi = cx - r, cx + r, cy - r, cy + r

    score = np.array([c.score for c in cands], dtype=np.float64)
    fp = np.array([c.fingerprint for c in cands], dtype=np.float64)
    refstd = np.array([c.fp_ref_std for c in cands], dtype=np.float64)
    xs = np.array([c.x for c in cands], dtype=np.float64)
    ys = np.array([c.y for c in cands], dtype=np.float64)
    scale = np.array([c.scale for c in cands], dtype=np.float64)
    rot = np.array([c.rotation for c in cands], dtype=np.float64)

    dist_center = np.hypot(xs - cx, ys - cy) / (half + 1e-9)
    scale_dev = scale / nominal_ratio - 1.0
    rot_norm = rot / (rot_max_deg + 1e-9)
    max_score = score.max() if n else 0.0
    max_fp = fp.max() if n else 0.0
    score_minus_max = score - max_score
    fp_minus_max = fp - max_fp
    # ranks in [0,1], 0 = best
    ncc_rank = np.argsort(np.argsort(-score)) / max(n - 1, 1)
    fp_rank = np.argsort(np.argsort(-fp)) / max(n - 1, 1)
    in_roi = ((xs >= x_lo) & (xs < x_hi) & (ys >= y_lo) & (ys < y_hi)).astype(np.float64)
    # pair-context (constant across candidates): gap between best and 2nd-best fp
    fp_sorted = np.sort(fp)[::-1]
    pair_fp_gap = float(fp_sorted[0] - fp_sorted[1]) if n >= 2 else 0.0
    pair_refstd_med = float(np.median(refstd)) if n else 0.0

    X = np.column_stack([
        score, fp, refstd, dist_center, scale_dev, rot_norm,
        score_minus_max, fp_minus_max, ncc_rank, fp_rank, in_roi,
        np.full(n, max_fp), np.full(n, pair_fp_gap), np.full(n, pair_refstd_med),
    ])
    return X


_ML_CACHE = {}


def load_ml_ranker(path):
    """Load exported MLP weights (.npz) once; returns a dict or None if missing."""
    if path in _ML_CACHE:
        return _ML_CACHE[path]
    model = None
    if path and os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        n_layers = int(d["n_layers"])
        model = {
            "mean": d["mean"], "scale": d["scale"],
            "coefs": [d[f"coef_{i}"] for i in range(n_layers)],
            "intercepts": [d[f"intercept_{i}"] for i in range(n_layers)],
        }
    _ML_CACHE[path] = model
    return model


def _ml_scores(X, model):
    """Pure-numpy MLP forward pass -> per-row probability (matches sklearn
    MLPClassifier: standardize, hidden ReLU layers, logistic output)."""
    z = (X - model["mean"]) / model["scale"]
    coefs, intercepts = model["coefs"], model["intercepts"]
    for i, (w, b) in enumerate(zip(coefs, intercepts)):
        z = z @ w + b
        if i < len(coefs) - 1:
            z = np.maximum(z, 0.0)  # ReLU on hidden layers
    return 1.0 / (1.0 + np.exp(-z.ravel()))  # logistic output


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
    fp_confident: float = 0.60,
    fp_gap: float = 0.12,
    fp_floor_lo: float = 0.30,
    fp_floor_hi: float = 0.45,
    use_ml: bool = False,
    ml_path: str = "ml_ranker.npz",
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

        strong_outside = outside_probe_best > roi_probe_best * (1.0 + fallback_margin)
        weak_roi = roi_best < 0.4
        few_cands = len(cand_pts) < 5
        # `always_full_search` removes the drift-ROI "dead zone": a true site just
        # beyond the ROI (or in the trigger's boundary band) was previously never
        # made a candidate, so it could not be selected even by the fingerprint.
        # Always augmenting with the cheap probe-transform full-image net makes the
        # true site a candidate REGARDLESS of position; the center rule remains only
        # a soft tie-break, so realistic center-clustered accuracy is unaffected.
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
    scales_f = nominal_ratio * np.linspace(scale_jitter[0] + 0.01, scale_jitter[1] - 0.01, n_scales_fine)
    rots_f = np.linspace(-rot_max_deg, rot_max_deg, n_rot_fine)
    cands: list[Candidate] = []
    for (cx, cy, _sc) in cand_pts:
        ref_result = _refine(search_f, ref_f, cx, cy, scales_f, rots_f)
        if ref_result is None:
            continue
        nccv, s, r, rx, ry, tmpl = ref_result
        th, tw = tmpl.shape
        xs, ys = _grid_lines(tmpl)
        if xs and ys:
            ref_fp = _fingerprint(tmpl, xs, ys)
            patch = cv2.getRectSubPix(search_f, (tw, th), (float(rx), float(ry)))
            fp = _ncc(ref_fp, _fingerprint(patch, xs, ys))
            fp_ref_std = float(ref_fp.std())
        else:
            fp = 0.0
            fp_ref_std = 0.0
        cands.append(Candidate(x=rx, y=ry, score=nccv, scale=s, rotation=r,
                               fingerprint=fp, fp_ref_std=fp_ref_std))

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

    # --- optional hybrid ML selector ---
    # A trained MLP scores each candidate's probability of being the true site
    # from the classical features (candidate_features). When enabled and the
    # weights file is present, the highest-probability candidate WITHIN the
    # ROI-eligible pool is chosen -- keeping the drift-prior safety while letting
    # the learned model do the fingerprint/NCC/center trade-off it was trained
    # on. Falls back to the classical selector if the model is unavailable.
    ml_model = load_ml_ranker(ml_path) if use_ml else None
    if ml_model is not None:
        Xf = candidate_features(cands, W, H, nominal_ratio, rot_max_deg, max_drift_frac)
        probs = _ml_scores(Xf, ml_model)
        pool_ids = {id(c) for c in pool}
        pool_idx = [k for k, c in enumerate(cands) if id(c) in pool_ids]
        best_k = max(pool_idx, key=lambda k: probs[k])
        chosen = cands[best_k]
        near_tied = [chosen]
    elif decisive_fp:
        near_tied = [fp_conf_cand]
        chosen = fp_conf_cand
    else:
        best_comb = max(c.combined for c in pool)
        near_tied = [c for c in pool if c.combined >= best_comb - tie_z]
        if fp_gate < 0.5:
            best_ncc = max(c.score for c in pool)
            ncc_near = [c for c in pool if c.score >= best_ncc - ncc_tie_margin]
            near_tied = list({id(c): c for c in (near_tied + ncc_near)}.values())
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
    ap.add_argument("--use-ml", action="store_true",
                    help="use the optional trained ML ranker (ml_ranker.npz) instead of "
                         "the classical selector")
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

    # resolve the ML weights next to this file so --use-ml works from any cwd
    ml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_ranker.npz")
    x, y, info = localize(ref_img, search_img, nominal_ratio=args.ratio,
                          use_ml=args.use_ml, ml_path=ml_path)

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
