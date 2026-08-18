"""
Drift-Sense synthetic dataset generator (FinFET-style layout).

Generates paired (reference, search) SEM-style image pairs that mimic the
Navigation-Error Recovery task described in the Applied Materials PS 2 brief:

    * Reference capture : 100x zoom, 1 nm/px,  1 um x 1 um FOV,  1000x1000 px
    * Search capture     : 10x  zoom, 10 nm/px, 10 um x 10 um FOV, 1000x1000 px

The reference pattern is a genuine crop of the same continuous FinFET layout
that the search image is rendered from (dense parallel vertical fins, crossed
by one or two horizontal gate bars per reference-sized crop), so the two are
physically consistent -- the reference really does appear, shrunk 10x, inside
the search image, at the recorded ground-truth location.

Realism knobs (each choice is cited in citations.md):
    * Independent sensor noise per capture (shot + read noise), search
      noisier than reference.
    * SEM-style edge brightening (secondary-electron edge effect).
    * Independent Gaussian blur per capture (search bigger + noisier).
    * Small random rotation/scale jitter applied to the search capture,
      with the ground-truth point transformed through the *same* affine
      matrix so labels stay exact.
    * Deterministic per-line pitch jitter so the lattice is not perfectly
      crystalline (real litho has some pitch-walk), while remaining highly
      periodic -- which is exactly what makes localization hard.
    * Fins are dense and tightly pitched (tens of nm); gate bars are sparse
      and widely pitched (hundreds of nm), so a 1 um reference crop shows
      many fins crossed by only one or two gates -- matching the brief's
      "dense parallel fins x one or two gate bars" description.

Run as a script to generate a full dataset:
    python dataset_gen.py --n 30 --out data --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# Physical / imaging constants (match the problem statement exactly)
# ----------------------------------------------------------------------------
SEARCH_FOV_NM = 10_000.0       # 10 um x 10 um
SEARCH_PX = 1000               # search image is 1000x1000 px -> 10 nm/px
REF_FOV_NM = 1_000.0            # 1 um x 1 um
REF_PX = 1000                  # reference image is 1000x1000 px -> 1 nm/px
SCALE_RATIO = SEARCH_FOV_NM / REF_FOV_NM  # = 10, nominal zoom ratio
SEARCH_NM_PER_PX = SEARCH_FOV_NM / SEARCH_PX   # 10 nm/px
REF_NM_PER_PX = REF_FOV_NM / REF_PX            # 1 nm/px

SUPERSAMPLE = 4  # supersample factor used before downsampling the search image


@dataclass
class LayoutParams:
    """FinFET fin/gate geometry, in nanometres."""
    pitch_x: float          # fin pitch (dense)
    pitch_y: float          # gate pitch (sparse -- 1-2 bars per ref crop)
    line_width: float       # 1-sigma width of a fin's brightness profile
    gate_width: float       # 1-sigma width of a gate bar's brightness profile
    contact_radius: float   # 1-sigma radius of the fin-gate crossing feature
    line_brightness: float
    contact_brightness: float
    background: float
    pitch_jitter_std: float  # per-fin random pitch-walk, nm


def sample_layout_params(rng: np.random.Generator, style: str = "finfet") -> LayoutParams:
    """Sample layout geometry for the requested architecture style.

    style="dram": periodic horizontal word-lines x vertical bit-lines crossing at
    right angles, with a contact/via dot at every intersection -- an (near-)
    isotropic fine grid. Both pitches are tens of nm (DRAM cell scale).

    style="finfet" (default): dense parallel vertical fins (tens of nm) crossed by
    sparse horizontal gate bars (hundreds of nm), contact feature at fin-gate
    crossings. See citations.md. (FinFET is the validated/cited style used for the
    submission; DRAM is provided so the generator supports both, as required.)
    """
    if style == "dram":
        bit_pitch = rng.uniform(40.0, 80.0)    # nm, vertical bit-line pitch
        word_pitch = rng.uniform(40.0, 80.0)   # nm, horizontal word-line pitch
        return LayoutParams(
            pitch_x=bit_pitch,
            pitch_y=word_pitch,
            line_width=bit_pitch * rng.uniform(0.25, 0.40),
            gate_width=word_pitch * rng.uniform(0.25, 0.40),   # word-lines as dense as bit-lines
            contact_radius=min(bit_pitch, word_pitch) * rng.uniform(0.30, 0.45),
            line_brightness=rng.uniform(0.35, 0.55),
            contact_brightness=rng.uniform(0.40, 0.60),
            background=rng.uniform(0.12, 0.20),
            pitch_jitter_std=bit_pitch * rng.uniform(0.01, 0.03),
        )
    # --- FinFET (default) ---
    fin_pitch = rng.uniform(24.0, 48.0)     # nm, advanced-node fin pitch
    gate_pitch = rng.uniform(500.0, 900.0)  # nm, cell-row / gate-bar spacing. Chosen so a
                                             # 1000 nm reference crop shows only ONE OR TWO
                                             # horizontal gate bars (the brief's description),
                                             # leaving the crop dominated by the dense, highly
                                             # periodic fin field -- this is what makes
                                             # localization genuinely ambiguous (see citations.md #1).
    return LayoutParams(
        pitch_x=fin_pitch,
        pitch_y=gate_pitch,
        line_width=fin_pitch * rng.uniform(0.25, 0.40),      # CD/pitch ratio, citations.md #8
        gate_width=gate_pitch * rng.uniform(0.02, 0.035),    # CD/pitch ratio, citations.md #8
        contact_radius=fin_pitch * rng.uniform(0.35, 0.55),  # crossing feature size, citations.md #8
        line_brightness=rng.uniform(0.35, 0.55),
        contact_brightness=rng.uniform(0.35, 0.55),
        background=rng.uniform(0.12, 0.20),
        pitch_jitter_std=fin_pitch * rng.uniform(0.01, 0.03),  # LER / pitch-walk, citations.md #9
    )


def _line_offsets(n_lines: int, jitter_std: float, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, jitter_std, size=n_lines)


def render_layout(x_nm: np.ndarray, y_nm: np.ndarray, params: LayoutParams,
                   offsets_x: np.ndarray, offsets_y: np.ndarray,
                   cell_scale: np.ndarray) -> np.ndarray:
    """Analytic FinFET dense-fin x sparse-gate-bar pattern, vectorised.

    x_nm, y_nm: broadcastable arrays of physical coordinates (nm), evaluated
    on whatever grid (full-res reference crop or supersampled search grid)
    the caller wants. Domain must be >= 0 in both axes (offsets are indexed
    by floor(coord / pitch)).

    cell_scale: per-crossing brightness multiplier, shape
    (len(offsets_y), len(offsets_x)). A real fin-gate crossing is never
    perfectly uniform -- local strain/contact variation at the crossing
    (see citations.md) breaks exact long-range periodicity of the *lattice
    content*, even though the fin/gate grid itself stays highly regular.
    This is what keeps ambiguity local (a handful of genuinely near-identical
    neighbouring sites) instead of making every repeat across the full field
    indistinguishable.
    """
    idx_x = np.clip(np.floor(x_nm / params.pitch_x).astype(np.int64), 0, len(offsets_x) - 1)
    idx_y = np.clip(np.floor(y_nm / params.pitch_y).astype(np.int64), 0, len(offsets_y) - 1)

    center_x = idx_x * params.pitch_x + params.pitch_x / 2 + offsets_x[idx_x]
    center_y = idx_y * params.pitch_y + params.pitch_y / 2 + offsets_y[idx_y]

    dist_x = x_nm - center_x
    dist_y = y_nm - center_y

    vert = np.exp(-0.5 * (dist_x / params.line_width) ** 2)   # fins (vertical, dense)
    horiz = np.exp(-0.5 * (dist_y / params.gate_width) ** 2)  # gate bars (horizontal, sparse)
    lines = np.clip(vert + horiz, 0.0, 1.0) * params.line_brightness

    contact_r2 = dist_x ** 2 + dist_y ** 2
    scale = cell_scale[idx_y, idx_x]
    contact = np.exp(-0.5 * contact_r2 / params.contact_radius ** 2) * params.contact_brightness * scale

    img = params.background + lines + contact
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def sem_edge_brighten(img: np.ndarray, strength: float) -> np.ndarray:
    """Approximate the SEM secondary-electron edge-brightening effect by
    adding scaled gradient magnitude back onto the image. See citations.md."""
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = mag / (mag.max() + 1e-8)
    return np.clip(img + strength * mag, 0.0, 1.0).astype(np.float32)


def add_sensor_noise(img: np.ndarray, rng: np.random.Generator,
                      shot_scale: float, read_std: float) -> np.ndarray:
    """Independent Poisson shot noise + Gaussian read noise per capture.
    Never share an rng draw between reference and search -- each call here
    must be given its own rng / be invoked separately per image."""
    peak = 255.0 * shot_scale
    counts = rng.poisson(np.clip(img, 0, 1) * peak) / peak
    noisy = counts + rng.normal(0.0, read_std, size=img.shape)
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


# Thin-film-interference tint (BGR): shadows read cool/bluish, highlights warm/gold
# -- the characteristic optical-microscope colour cast. Applied as a MULTIPLIER on
# luminance so a grayscale (luminance) conversion recovers the structure almost
# exactly, preserving the fingerprint the localizer relies on. See citations.md.
_OPTICAL_COOL_BGR = np.array([0.95, 0.75, 0.55], dtype=np.float32)   # dark regions
_OPTICAL_WARM_BGR = np.array([0.75, 0.95, 1.00], dtype=np.float32)   # bright regions


def colorize_optical(gray: np.ndarray) -> np.ndarray:
    """Map a grayscale layout (0..1 luminance) to a 3-channel BGR optical-
    microscope image. Colour is a deterministic, luminance-preserving function of
    the structure (a smooth cool->warm thin-film tint), so a matching physical site
    has matching colour in the reference and search captures, and luminance
    conversion recovers the structure. Returns float32 BGR in 0..1."""
    g = np.clip(gray, 0.0, 1.0)[..., None]
    tint = _OPTICAL_COOL_BGR * (1.0 - g) + _OPTICAL_WARM_BGR * g
    return np.clip(g * tint * 1.15, 0.0, 1.0)


def _stripe_positions(fov_nm, gap_lo, gap_hi, rng):
    """Aperiodic (jittered) stripe centres across [0, fov_nm), as absolute nm.
    Irregular spacing makes the stripe *sequence* a unique code, so a crop that
    straddles stripes is individually identifiable -- this is what makes the
    superstructure break the periodic lattice's translation ambiguity."""
    pos, x = [], rng.uniform(gap_lo, gap_hi)
    while x < fov_nm:
        pos.append(x)
        x += rng.uniform(gap_lo, gap_hi)
    return np.asarray(pos, dtype=np.float32)


def _channel_field(coords_nm, positions, width_nm):
    """1-D proximity to the nearest channel centre: ~1 inside a channel, ~0 in the
    mat interior. Evaluated identically for reference and search -> consistent."""
    f = np.zeros_like(coords_nm, dtype=np.float32)
    for p in positions:
        f = np.maximum(f, np.exp(-0.5 * ((coords_nm - p) / width_nm) ** 2))
    return f


def apply_superstructure(img, x_nm, y_nm, sup):
    """Turn a uniform lattice into a real subarray-MAT array: dense cell blocks
    separated by CLEAR sense-amplifier stripes (horizontal) and wordline-driver
    channels (vertical). The lattice is SUPPRESSED inside the channels (so blocks
    are separated by gaps, as on a real DRAM/FinFET array -- CITE citations.md),
    the horizontal channels carry a bright sense-amp band and the vertical ones a
    darker driver band. Irregular channel spacing makes each mat identifiable."""
    hc = _channel_field(y_nm, sup["h_pos"], sup["h_w"])     # rows: sense-amp channels
    vc = _channel_field(x_nm, sup["v_pos"], sup["v_w"])     # cols: driver channels
    chan = np.maximum(hc[:, None], vc[None, :])             # 2-D: 1 in any channel
    out = img * (1.0 - 0.9 * chan)                          # clear the cells in channels
    out = out + sup["h_amp"] * hc[:, None] * (1.0 - vc[None, :])   # bright sense-amp bands
    out = out - sup["v_amp"] * vc[None, :] * (1.0 - 0.5 * hc[:, None])  # dark driver bands
    return np.clip(out + sup["bg"], 0.0, 1.0).astype(np.float32)


@dataclass
class PairMeta:
    pair_id: int
    seed: int
    gt_x: float
    gt_y: float
    crop_x0_nm: float
    crop_y0_nm: float
    rotation_deg: float
    scale: float
    ref_blur_sigma: float
    search_blur_sigma: float
    forced_periodic: bool
    magnification_ratio: float = 10.0   # effective search:reference magnification


def generate_pair(pair_id: int, seed: int, forced_periodic: bool = False,
                  drift_nm: "tuple | None" = None, style: str = "finfet",
                  mag_jitter: bool = False, rgb: bool = False,
                  superstructure: bool = False):
    """Generate one (reference, search, ground_truth) sample.

    forced_periodic: if True, the reference crop is deliberately placed away
    from any layout edge, deep inside the repeating lattice, so the search
    image contains multiple near-identical candidate matches. Used to build
    the mandatory "honest failure case" and to stress-test the nearest-to-
    center tie-break.

    drift_nm: if given, an explicit (drift_x, drift_y) offset from the FOV
    center in nanometres, overriding the random bounded-drift draw. Used to
    construct controlled out-of-ROI test pairs (true site deliberately placed
    beyond the localizer's drift ROI) for the fallback-path regression test.
    """
    rng = np.random.default_rng(seed)
    params = sample_layout_params(rng, style=style)

    n_lines_x = int(SEARCH_FOV_NM / params.pitch_x) + 4
    n_lines_y = int(SEARCH_FOV_NM / params.pitch_y) + 4
    offsets_x = _line_offsets(n_lines_x, params.pitch_jitter_std, rng)
    offsets_y = _line_offsets(n_lines_y, params.pitch_jitter_std, rng)

    # per-intersection contact defect grid (see render_layout docstring).
    # forced_periodic pairs get a near-defect-free patch so the search
    # image genuinely contains several indistinguishable repeats.
    # NB: 0.4 is a deliberate task-driven exaggeration, NOT a real defect
    # rate -- it stands in for the aggregate site-distinguishing content that
    # breaks periodicity; see the honesty note in citations.md #7.
    defect_prob = 0.03 if forced_periodic else 0.4
    is_defect = rng.random((n_lines_y, n_lines_x)) < defect_prob
    cell_scale = np.where(is_defect, 0.0, 1.0).astype(np.float32)

    # aperiodic superstructure (subarray mats separated by sense-amp stripes /
    # driver channels) -- makes the image look like a real array AND breaks the
    # pure-lattice translation ambiguity so localization is well-posed everywhere.
    sup = None
    if superstructure:
        sup = dict(
            h_pos=_stripe_positions(SEARCH_FOV_NM, 1500.0, 2800.0, rng),   # sense-amp rows
            v_pos=_stripe_positions(SEARCH_FOV_NM, 1500.0, 2800.0, rng),   # driver columns
            h_w=float(rng.uniform(180.0, 300.0)), v_w=float(rng.uniform(160.0, 280.0)),
            h_amp=float(rng.uniform(0.35, 0.50)), v_amp=float(rng.uniform(0.20, 0.30)),
            bg=float(rng.uniform(0.10, 0.16)),
        )

    # --- choose reference crop location (in nm, within the search FOV) ---
    # Physically, the search capture is centered on the *previous* site and
    # the reference must be re-localized after small motion-stage drift
    # (thermal expansion, vibration, mechanical slack) -- so the true site
    # sits near the search image's center with a bounded offset, not
    # anywhere in the 10 um FOV. This is also what makes the task's
    # nearest-to-center tie-break a physically sensible disambiguation rule
    # rather than an arbitrary one: among several near-identical periodic
    # matches, the true (drifted) site is expected to be the one closest to
    # center.
    margin = REF_FOV_NM / 2 + 10
    max_drift = SEARCH_FOV_NM * 0.18  # bounded drift radius from FOV center
    if forced_periodic:
        # keep drift small but deep inside the lattice (several pitches from
        # any edge) so many repeats of the pattern surround the true site --
        # used for the mandatory "honest failure case".
        max_drift = SEARCH_FOV_NM * 0.08
    center_fov = SEARCH_FOV_NM / 2
    if drift_nm is not None:
        drift_x, drift_y = float(drift_nm[0]), float(drift_nm[1])
    else:
        drift_x = rng.uniform(-max_drift, max_drift)
        drift_y = rng.uniform(-max_drift, max_drift)
    cx = float(np.clip(center_fov + drift_x, margin, SEARCH_FOV_NM - margin))
    cy = float(np.clip(center_fov + drift_y, margin, SEARCH_FOV_NM - margin))
    x0 = cx - REF_FOV_NM / 2
    y0 = cy - REF_FOV_NM / 2

    # --- render reference at native 1 nm/px, 2x supersampled for AA ---
    ss = 2
    lin = (np.arange(REF_PX * ss) + 0.5) / ss * REF_NM_PER_PX
    ref_x = x0 + lin
    ref_y = y0 + lin
    RX, RY = np.meshgrid(ref_x, ref_y)
    ref_super = render_layout(RX, RY, params, offsets_x, offsets_y, cell_scale)
    if sup is not None:
        ref_super = apply_superstructure(ref_super, ref_x, ref_y, sup)
    ref_base = cv2.resize(ref_super, (REF_PX, REF_PX), interpolation=cv2.INTER_AREA)

    # --- render full search FOV supersampled, then box-downsample ---
    lin_s = (np.arange(SEARCH_PX * SUPERSAMPLE) + 0.5) / SUPERSAMPLE * SEARCH_NM_PER_PX
    SX, SY = np.meshgrid(lin_s, lin_s)
    search_super = render_layout(SX, SY, params, offsets_x, offsets_y, cell_scale)
    if sup is not None:
        search_super = apply_superstructure(search_super, lin_s, lin_s, sup)
    search_base = cv2.resize(search_super, (SEARCH_PX, SEARCH_PX), interpolation=cv2.INTER_AREA)

    gt_x = cx / SEARCH_NM_PER_PX
    gt_y = cy / SEARCH_NM_PER_PX

    # --- independent per-capture blur ---
    ref_blur_sigma = rng.uniform(0.25, 0.45)
    search_blur_sigma = rng.uniform(0.4, 0.7)  # search is noisier/blurrier
    ref_img = cv2.GaussianBlur(ref_base, (0, 0), ref_blur_sigma)
    search_img = cv2.GaussianBlur(search_base, (0, 0), search_blur_sigma)

    # --- SEM edge brightening (independent strength draw per capture, but
    #     kept close between the two so the structural match stays strong --
    #     the two captures are the same physical site, not different ones) ---
    edge_strength = rng.uniform(0.15, 0.22)
    ref_img = sem_edge_brighten(ref_img, edge_strength * rng.uniform(0.9, 1.1))
    search_img = sem_edge_brighten(search_img, edge_strength * rng.uniform(0.9, 1.1))

    # --- random rotation/scale jitter applied to the *search* capture ---
    # propagate the exact same affine transform into the ground truth point.
    # rotation = scan-field rotation/distortion; scale = magnification-
    # calibration uncertainty. See citations.md #5.
    rot_deg = rng.uniform(-2.0, 2.0)
    # scale jitter = magnification-calibration uncertainty. With --mag-jitter the
    # range widens to a REAL magnification variation (effective ratio ~9x-11.1x),
    # to model an unknown/harder test set and to validate the localizer's scale
    # measurement. The effective magnification is SCALE_RATIO / scale.
    scale = rng.uniform(0.90, 1.10) if mag_jitter else rng.uniform(0.97, 1.03)
    center = (SEARCH_PX / 2, SEARCH_PX / 2)
    M = cv2.getRotationMatrix2D(center, rot_deg, scale)
    search_img = cv2.warpAffine(search_img, M, (SEARCH_PX, SEARCH_PX),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    gt_vec = np.array([gt_x, gt_y, 1.0])
    gt_x, gt_y = M @ gt_vec

    # --- independent sensor noise, drawn from independent rng streams,
    #     search strictly noisier than reference ---
    ref_rng = np.random.default_rng(seed * 2 + 1)
    search_rng = np.random.default_rng(seed * 2 + 2)
    if rgb:
        # RGB optical-microscope bonus: colorize (BGR) then add INDEPENDENT noise
        # per channel per capture, so the reference and search are separate colour
        # acquisitions. Colour is a function of luminance -> physically consistent.
        ref_c = colorize_optical(ref_img)
        search_c = colorize_optical(search_img)
        ref_c = np.stack([add_sensor_noise(ref_c[..., k], ref_rng, 6.0, 0.004) for k in range(3)], axis=-1)
        search_c = np.stack([add_sensor_noise(search_c[..., k], search_rng, 3.0, 0.008) for k in range(3)], axis=-1)
        ref_u8 = (ref_c * 255).astype(np.uint8)
        search_u8 = (search_c * 255).astype(np.uint8)
    else:
        ref_img = add_sensor_noise(ref_img, ref_rng, shot_scale=6.0, read_std=0.004)
        search_img = add_sensor_noise(search_img, search_rng, shot_scale=3.0, read_std=0.008)
        ref_u8 = (ref_img * 255).astype(np.uint8)
        search_u8 = (search_img * 255).astype(np.uint8)

    meta = PairMeta(
        pair_id=pair_id, seed=seed, gt_x=float(gt_x), gt_y=float(gt_y),
        crop_x0_nm=float(x0), crop_y0_nm=float(y0),
        rotation_deg=float(rot_deg), scale=float(scale),
        ref_blur_sigma=float(ref_blur_sigma), search_blur_sigma=float(search_blur_sigma),
        forced_periodic=forced_periodic,
        magnification_ratio=float(SCALE_RATIO / scale),
    )
    return ref_u8, search_u8, meta


def generate_dataset(n: int, out_dir: str, seed0: int = 0, n_forced_periodic: "int | None" = None,
                     style: str = "finfet", mag_jitter: bool = False, rgb: bool = False,
                     superstructure: bool = False):
    # Number of deliberately-hard "forced periodic" (near-defect-free) pairs scales
    # with the set size: none for a tiny quick-start set (so the first pair a
    # reviewer runs is a normal, solvable one), up to 3 for a full >=30-pair set
    # (which then contains the mandatory highly-periodic honest-failure cases).
    if n_forced_periodic is None:
        n_forced_periodic = min(3, n // 10)
    os.makedirs(out_dir, exist_ok=True)
    all_meta = []
    for i in range(n):
        forced = i < n_forced_periodic
        # style="both" produces a single mixed set, alternating FinFET/DRAM per pair
        # so one command exercises both architectures.
        pair_style = ("finfet" if i % 2 == 0 else "dram") if style == "both" else style
        # forced-periodic pairs are kept pure-lattice (no superstructure) so the
        # mandatory "genuinely-hard periodic region" honest-failure case survives
        # even when the rest of the set carries the disambiguating mat structure.
        ref_img, search_img, meta = generate_pair(i, seed0 + i, forced_periodic=forced,
                                                  style=pair_style, mag_jitter=mag_jitter, rgb=rgb,
                                                  superstructure=superstructure and not forced)
        cv2.imwrite(os.path.join(out_dir, f"pair_{i:03d}_ref.png"), ref_img)
        cv2.imwrite(os.path.join(out_dir, f"pair_{i:03d}_search.png"), search_img)
        all_meta.append(asdict(meta))
    with open(os.path.join(out_dir, "ground_truth.json"), "w") as f:
        json.dump(all_meta, f, indent=2)
    return all_meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--style", type=str.lower, choices=["finfet", "dram", "both"], default="finfet",
                    help="architecture style to generate: finfet, dram, or both "
                         "(both = one mixed set alternating FinFET/DRAM per pair; "
                         "case-insensitive; default: finfet)")
    ap.add_argument("--n", type=int, default=30, help="number of pairs to generate")
    ap.add_argument("--out", type=str, default="data", help="output directory")
    ap.add_argument("--seed", type=int, default=0, help="base random seed")
    ap.add_argument("--mag-jitter", action="store_true",
                    help="vary the true magnification per pair (~9x-11x) instead of a "
                         "fixed 10x -- a harder, more realistic set for scale-robustness")
    ap.add_argument("--rgb", action="store_true",
                    help="generate 3-channel RGB optical-microscope-style pairs "
                         "(thin-film-interference colour) instead of grayscale SEM (bonus)")
    ap.add_argument("--superstructure", action="store_true",
                    help="add the realistic subarray-MAT superstructure (sense-amp + "
                         "driver channels) -- looks like a real wafer array and makes "
                         "off-mat sites uniquely identifiable")
    args = ap.parse_args()

    t0 = time.time()
    meta = generate_dataset(args.n, args.out, seed0=args.seed, style=args.style,
                            mag_jitter=args.mag_jitter, rgb=args.rgb,
                            superstructure=args.superstructure)
    dt = time.time() - t0
    label = "finfet+dram" if args.style == "both" else args.style
    print(f"Generated {len(meta)} {label} pairs in {dt:.2f}s -> {args.out}/")
