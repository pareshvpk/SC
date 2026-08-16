"""
Drift-Sense localization algorithm.

Given a high-res reference capture (100x zoom) and a low-res search capture
(10x zoom) of the same physical site, find where the reference pattern sits
inside the search image and return its center (x, y) in the search image's
pixel frame.

Approach: multi-scale / multi-rotation normalized cross-correlation (NCC).
The nominal zoom ratio (10x) is known and used to size the template, but a
small scale/rotation sweep is done around that nominal value because the
dataset generator (and presumably the real tool) applies small per-capture
scale/rotation jitter that a single fixed-ratio template would miss.

The key twist required by the brief: DRAM/FinFET layouts are highly
periodic, so classical NCC produces many near-identical correlation peaks.
Naive argmax picks an arbitrary repeat. Instead we:
    1. Collect ALL local correlation peaks across the scale/rotation sweep.
    2. Non-max-suppress candidates that are closer than ~1 template width
       apart (they're the same physical peak found at slightly different
       scale/rotation trials).
    3. Keep every surviving candidate within `margin` of the best score
       (near-equal peaks -- not exact ties, because noise never gives an
       exact tie).
    4. Among those survivors, return the one closest to the CENTER of the
       search image, not the highest-scoring one.

Usage:
    from localize import localize
    x, y, info = localize(ref_img, search_img)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.ndimage import maximum_filter


@dataclass
class Candidate:
    x: float
    y: float
    score: float
    scale: float
    rotation: float


@dataclass
class LocalizeInfo:
    candidates: list = field(default_factory=list)
    survivors: list = field(default_factory=list)
    near_tied: list = field(default_factory=list)
    chosen: "Candidate | None" = None
    best_score: float = 0.0
    elapsed_s: float = 0.0
    full_image_fallback: bool = False


def _rotate_template(tmpl: np.ndarray, angle_deg: float) -> np.ndarray:
    if angle_deg == 0.0:
        return tmpl
    h, w = tmpl.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(tmpl, M, (w, h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)


def _find_local_peaks(corr: np.ndarray, min_score: float, nbhd: int) -> list:
    """Local maxima of the correlation surface above min_score."""
    local_max = maximum_filter(corr, size=nbhd, mode="nearest")
    mask = (corr == local_max) & (corr >= min_score)
    ys, xs = np.nonzero(mask)
    return list(zip(xs.tolist(), ys.tolist(), corr[ys, xs].tolist()))


def _sweep(search_f: np.ndarray, ref_f: np.ndarray, x_lo: int, y_lo: int,
           scales, rotations, peak_rel_threshold: float):
    """Run the full scale/rotation sweep against `search_f`, returning
    candidates in the *full-image* coordinate frame (offset by x_lo, y_lo)."""
    candidates: list[Candidate] = []
    best_score = 0.0
    for s in scales:
        tw = max(8, int(round(ref_f.shape[1] / s)))
        th = max(8, int(round(ref_f.shape[0] / s)))
        if tw >= search_f.shape[1] or th >= search_f.shape[0]:
            continue
        base_tmpl = cv2.resize(ref_f, (tw, th), interpolation=cv2.INTER_AREA)
        for r in rotations:
            tmpl = _rotate_template(base_tmpl, r)
            corr = cv2.matchTemplate(search_f, tmpl, cv2.TM_CCOEFF_NORMED)
            best_score = max(best_score, float(corr.max()))
            nbhd = max(3, min(tw, th) // 2)
            for (px, py, score) in _find_local_peaks(corr, peak_rel_threshold * best_score, nbhd):
                candidates.append(Candidate(x=px + tw / 2 + x_lo, y=py + th / 2 + y_lo,
                                             score=score, scale=s, rotation=r))
    return candidates, best_score


def _nms_and_select(candidates: list, ref_width: int, tie_margin: float, center: tuple):
    candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
    suppress_radius = float(np.median([ref_width / c.scale for c in candidates])) * 0.5
    survivors: list[Candidate] = []
    for c in candidates:
        if all(np.hypot(c.x - s.x, c.y - s.y) > suppress_radius for s in survivors):
            survivors.append(c)
    global_best = survivors[0].score
    near_tied = [c for c in survivors if c.score >= tie_margin * global_best]
    chosen = min(near_tied, key=lambda c: np.hypot(c.x - center[0], c.y - center[1]))
    return survivors, near_tied, chosen


def localize(
    ref_img: np.ndarray,
    search_img: np.ndarray,
    nominal_ratio: float = 10.0,
    scale_jitter: tuple = (0.95, 1.05),
    n_scales: int = 9,
    rotation_jitter_deg: float = 3.0,
    n_rotations: int = 9,
    peak_rel_threshold: float = 0.5,
    tie_margin: float = 0.95,
    max_drift_frac: float = 0.2,
) -> tuple:
    """Locate `ref_img` inside `search_img`.

    Parameters
    ----------
    nominal_ratio : known physical zoom ratio (search_nm_per_px / ref_nm_per_px).
    scale_jitter : multiplicative sweep range applied to nominal_ratio, to
        absorb small per-capture scale variation.
    rotation_jitter_deg : +/- degrees swept, to absorb small per-capture
        rotation.
    peak_rel_threshold : local maxima below this fraction of the running
        best score are discarded early (keeps candidate lists small).
    tie_margin : after NMS, any candidate scoring >= tie_margin * best_score
        is treated as a "near-equal" peak and is eligible for the
        nearest-to-center tie-break, per the task's required disambiguation
        rule (return the match closest to image center, not the highest
        score).
    max_drift_frac : search is restricted to a region of interest around the
        search image's center, of radius `max_drift_frac * search_width`.
        This encodes the physical prior that motion-stage drift between
        visits is bounded (thermal expansion / vibration / mechanical slack
        are all small relative to the full field of view) -- exactly the
        assumption that makes "closest to center" a meaningful disambiguator
        in the first place. On a highly periodic lattice, an *unrestricted*
        search finds near-identical correlation peaks at essentially every
        lattice repeat across the whole image, which makes nearest-to-center
        degenerate into "nearest lattice node to center" rather than
        "correct site among a handful of true look-alikes". Restricting to
        the plausible drift envelope keeps the tie-break meaningful, and
        mirrors how real tools bound the search using known stage
        repeatability specs.

    Returns
    -------
    (x, y, info) : chosen center in search-image pixel coordinates, plus a
        LocalizeInfo with all candidates / survivors for diagnostics.
    """
    t0 = time.perf_counter()
    ref_f = ref_img.astype(np.float32)
    full_search_f = search_img.astype(np.float32)
    H, W = full_search_f.shape
    center = (W / 2.0, H / 2.0)

    ratios = nominal_ratio * np.linspace(scale_jitter[0], scale_jitter[1], n_scales)
    rotations = np.linspace(-rotation_jitter_deg, rotation_jitter_deg, n_rotations)

    # --- pass 1: fast path, restricted to the plausible drift envelope ---
    roi_r = int(round(max_drift_frac * W))
    cx0, cy0 = W // 2, H // 2
    x_lo, x_hi = max(0, cx0 - roi_r), min(W, cx0 + roi_r)
    y_lo, y_hi = max(0, cy0 - roi_r), min(H, cy0 + roi_r)
    roi_search_f = full_search_f[y_lo:y_hi, x_lo:x_hi]

    candidates, roi_best_score = _sweep(roi_search_f, ref_f, x_lo, y_lo,
                                         ratios, rotations, peak_rel_threshold)
    if not candidates:
        raise RuntimeError("localize: no correlation peaks found")
    fallback_used = False

    # --- pass 2: fall back to an unrestricted full-image search if some
    # region outside the drift envelope scores meaningfully better than
    # anything found inside it. This guards against the drift-envelope
    # assumption being wrong on data we didn't generate (e.g. the official
    # Phase 2 test set) -- the ROI is a speed/ambiguity-reduction prior, not
    # a hard constraint on where the answer is allowed to be. We only pay
    # for a full sweep when the fast path's own best score looks beaten. ---
    best_roi_candidate = max(candidates, key=lambda c: c.score)
    tw_check = max(8, int(round(ref_f.shape[1] / best_roi_candidate.scale)))
    th_check = max(8, int(round(ref_f.shape[0] / best_roi_candidate.scale)))
    check_tmpl = _rotate_template(
        cv2.resize(ref_f, (tw_check, th_check), interpolation=cv2.INTER_AREA),
        best_roi_candidate.rotation,
    )
    full_probe = cv2.matchTemplate(full_search_f, check_tmpl, cv2.TM_CCOEFF_NORMED)
    if float(full_probe.max()) > roi_best_score * (1.0 + 1e-3):
        fallback_used = True
        candidates, _ = _sweep(full_search_f, ref_f, 0, 0, ratios, rotations, peak_rel_threshold)

    survivors, near_tied, chosen = _nms_and_select(candidates, ref_f.shape[1], tie_margin, center)

    info = LocalizeInfo(
        candidates=candidates,
        survivors=survivors,
        near_tied=near_tied,
        chosen=chosen,
        best_score=survivors[0].score,
        elapsed_s=time.perf_counter() - t0,
        full_image_fallback=fallback_used,
    )
    return chosen.x, chosen.y, info


if __name__ == "__main__":
    import argparse
    import cv2 as _cv2

    ap = argparse.ArgumentParser(description="Localize a reference crop inside a search image.")
    ap.add_argument("ref", type=str)
    ap.add_argument("search", type=str)
    args = ap.parse_args()

    ref_img = _cv2.imread(args.ref, _cv2.IMREAD_GRAYSCALE)
    search_img = _cv2.imread(args.search, _cv2.IMREAD_GRAYSCALE)
    x, y, info = localize(ref_img, search_img)
    print(f"match center: ({x:.2f}, {y:.2f})  score={info.chosen.score:.4f}  "
          f"candidates={len(info.candidates)} survivors={len(info.survivors)}  "
          f"time={info.elapsed_s*1000:.1f} ms")
