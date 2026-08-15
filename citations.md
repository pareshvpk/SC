# Citations

Public references backing every noise model, blur/rotation/scale choice, and
structural parameter used in `dataset_gen.py`. Only publicly known structural
characteristics are used — no proprietary fab data.

## 1. FinFET fin pitch / gate-bar pitch (`sample_layout_params`, 24-48 nm fin / 500-900 nm gate-row)

Advanced-node FinFET fin pitch is documented in the tens-of-nm range:

- IOPscience, "A breakthrough in contact engineering for sub-3 nm FinFETs:
  overcoming the fin-pitch bottleneck" — discusses fin-pitch scaling limits
  for sub-3 nm FinFETs. https://iopscience.iop.org/article/10.1088/1361-6528/ae2513
- ResearchGate, "Pitch ranges for the Fin, Contact to Poly, and Metal layers
  at 10 nm, 7 nm, and 5 nm" — tabulates public fin-pitch and contacted-poly-
  pitch (CPP) figures for these nodes, both in the tens-of-nm range (CPP
  roughly 30-64 nm, fin pitch roughly 24-34 nm).
  https://www.researchgate.net/figure/Pitch-ranges-for-the-Fin-Contact-to-Poly-and-Metal-layers-at-10-nm-7-nm-and-5-nm_tbl1_338341938
- ASIC North, "FinFET Technology and Layout — Part 1" — general public
  overview of fin/gate layout conventions.
  https://www.asicnorth.com/blog/part-one-finfet-technology-and-layout/

**Note on the gate-bar (`pitch_y`) range**: literal contacted-poly-pitch is
the *same* order of magnitude as fin pitch (tens of nm), which would put
dozens of gate bars in a 1 um reference crop — not the "one or two gate bars"
visual the problem brief describes. The brief's constraint directly sets the
scale: to show **one or two** horizontal gate bars across a 1000 nm crop,
`pitch_y` must be roughly 500-1000 nm, so `pitch_y` models a coarser periodic
feature at the (multi-track / relaxed) standard-cell-row scale rather than a
single contacted-poly line. Public advanced-node cell-height figures are
documented in the low-hundreds-of-nm range, and multi-height cell rows scale
that up by an integer factor into the range used here:

- Angstronomics, "The TRUTH of TSMC 5nm" — reports practical standard-cell
  height figures for advanced nodes.
  https://www.angstronomics.com/p/the-truth-of-tsmc-5nm
- Georgia Tech CAD Lab, "Performance, Power, and Area of Standard Cells in
  Sub-3 nm Node" — reports cell/track-height figures (e.g. 120 nm cell height
  at the 3 nm node) for advanced standard-cell libraries.
  https://gtcad.gatech.edu/www/papers/ted22-a.pdf

This is a **deliberate task-driven simplification**, not a claim that real
gate pitch is hundreds of nm — flagged explicitly here and in the
`sample_layout_params` docstring so it's not mistaken for a literal CPP
citation.

## 2. SEM secondary-electron edge brightening (`sem_edge_brighten`)

Modeled as image intensity + (strength x local gradient magnitude), which
approximates the well-documented "edge effect" in secondary-electron SEM
imaging:

- JEOL Ltd., SEM glossary, "edge effect" — "the tip of a protrusion and the
  edge of a step on a specimen surface become extremely bright" because more
  secondary electrons escape near edges than from flat regions.
  https://www.jeol.com/words/semterms/20121024.012800.php
- ETH Zurich, ScopeM, "SEM – Imaging with Secondary Electrons" — describes
  edge/topographic contrast as the dominant secondary-electron contrast
  mechanism. https://scopem.ethz.ch/education/EM-ken/Methods/SEM/SEM_Imaging_1.html
- "Secondary electron emission in the scanning electron microscope,"
  J. Appl. Phys. 54, R1 (1983) — foundational treatment of secondary-electron
  yield vs. surface topography/tilt, i.e. the physical basis of edge
  brightening. https://pubs.aip.org/aip/jap/article/54/11/R1/13393/Secondary-electron-emission-in-the-scanning
- (recent) "Impact of photoexcitation on secondary electron emission: a Monte
  Carlo study," arXiv:2210.14470 (2022) — up-to-date modeling of secondary-
  electron yield, confirming the topography/angle dependence behind edge
  contrast. https://arxiv.org/abs/2210.14470
- (recent) "Model sensitivity analysis of Monte-Carlo based SEM simulations"
  (2020) — simulates SEM secondary-electron images and explicitly reproduces
  the "edge-blooming" bright-edge regions we approximate.
  https://www.researchgate.net/publication/345072113_Model_sensitivity_analysis_of_Monte-Carlo_based_SEM_simulations

## 3. Independent sensor noise: Poisson shot noise + Gaussian read noise (`add_sensor_noise`)

Modeled as `counts ~ Poisson(image * peak) / peak + Gaussian(0, read_std)`,
directly following the standard Poisson-Gaussian sensor noise model used for
electron/photon-counting imaging systems:

- Available at https://arxiv.org/abs/2210.04866 — PoGaIN: Poisson-Gaussian
  Image Noise Modeling from Paired Samples — states the standard model:
  shot noise as Poisson (from the discrete/particle nature of the signal)
  plus a zero-mean Gaussian read-noise term combining thermal, amplifier, and
  ADC quantization noise.
- Imatest, "Image Sensor Noise – measurement and modeling" — practical
  description of shot noise (Poisson-distributed, signal-dependent) and read
  noise (approximately Gaussian, signal-independent) in imaging sensors.
  https://www.imatest.com/imaging/image-sensor-noise/
- Reason the two captures use independent noise draws and never share an RNG
  stream: the reference and search images are stated to be separate physical
  acquisitions, and shot/read noise in a real detector is independent
  per-exposure by construction (same Poisson-Gaussian references above).

## 4. Gaussian blur as a beam/optical point-spread function proxy (`cv2.GaussianBlur`)

- "Measurement of the Electron Beam Point Spread Function (PSF)
  in a Scanning Electron Microscope (SEM)" — the SEM probe's PSF is
  well-approximated by a Gaussian intensity distribution over a range of
  specimen conditions.
  https://www.researchgate.net/publication/311529053_Measurement_of_the_Electron_Beam_Point_Spread_Function_PSF_in_a_Scanning_Electron_Microscope_SEM
- "The Determination and Application of the Point Spread Function in the
  Scanning Electron Microscope," PubMed 30175706 — beam diameter is defined
  via the full-width-at-fraction-maximum of a Gaussian intensity
  distribution; deconvolution/PSF literature treats the SEM blur kernel as
  Gaussian. https://pubmed.ncbi.nlm.nih.gov/30175706/
- (recent) Hwang, Park, Jung & Ogawa, "Enhanced Scanning Electron Microscopy
  Using Auto-Optimized Image Restoration with Constrained Least Squares Filter,"
  Microscopy and Microanalysis (2023) — current SEM image-restoration work that
  treats the SEM blur as a determinable point-spread function.
  https://www.cambridge.org/core/journals/microscopy-and-microanalysis/article/abs/exploring-the-parameter-space-of-point-spread-function-determination-for-the-scanning-electron-microscopepart-ii-effect-on-image-restoration-quality/04139C05EB390CA6EE5D7B7E516B8B2C
- Search-image blur sigma is drawn larger than the reference's, consistent
  with the wide-search (10x/low-mag) capture integrating a larger effective
  probe/pixel footprint than the high-res reference capture.

## 5. Small random rotation / scale jitter per capture

Modeled as a small affine perturbation (±2 deg rotation, ~3% scale) applied
independently to the search capture. The two components have distinct, documented
physical origins:

**Scale (~3%) — SEM magnification-calibration uncertainty.** SEM magnification is
not exact and depends on operating conditions (magnification setting, accelerating
voltage, working distance), which is precisely why traceable calibration standards
exist; a few-percent scale mismatch between two captures of the same site is a
standard metrology reality:

- NIST, "Design and Development of a Measurement and Control System for Measuring
  SEM Magnification Calibration Samples" (SRM 2090 / RM 8090 magnification
  standard) — establishes that SEM magnification requires traceable calibration
  and depends on operating conditions.
  https://www.nist.gov/publications/design-and-development-measurement-and-control-system-measuring-sem-magnification
- "The Measurement and Uncertainty of a Calibration Standard for the Scanning
  Electron Microscope" (J. Res. NIST, PMC8345241) — quantifies the uncertainty of
  SEM dimensional/magnification calibration.
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8345241/

**Rotation (±2 deg) — scan-field rotation / geometric distortion.** SEM images
carry geometric distortion and scan-to-scan rotational offset between the scan
coordinate system and the specimen, a documented artifact that correction methods
are explicitly built to remove:

- Jin et al., "Correction of image drift and distortion in a scanning electron
  microscopy," Journal of Microscopy 260(3), 2015 — characterizes and corrects SEM
  geometric distortion and drift between captures.
  https://onlinelibrary.wiley.com/doi/abs/10.1111/jmi.12293
- (recent) "Identification and correction of temporal and spatial distortions in
  scanning transmission electron microscopy," Ultramicroscopy (2021) — recent
  treatment of scan-induced spatial distortion between captures.
  https://www.sciencedirect.com/science/article/abs/pii/S0304399121001212
- (recent) "Deep convolutional neural networks to restore single-shot electron
  microscopy images," npj Computational Materials (2023) — modern restoration of
  distortion/drift artifacts in single SEM/STEM captures.
  https://www.nature.com/articles/s41524-023-01188-0
- The bounded stage-drift references in section 6 additionally support small
  scan-to-scan positional offsets of the same physical site.

## 6. Bounded motion-stage drift -> reference site lies near the search image center

`generate_pair` places the true site within a bounded radius of the search
FOV's center (drift_x/drift_y), rather than uniformly across the whole field.
This reflects documented SEM/stage-drift magnitudes being small relative to
a 10 um field of view between successive visits:

- Element Pi, "SEM Drift Causes: Understanding the Factors" — thermal drift
  is the most common cause of SEM image drift; even a 0.1 degC change shifts
  beam/sample position, and mechanical/friction effects in the stage also
  contribute. https://elementpi.com/sem-drift-causes/
- DEMCON, "Thermal drift in precision engineering" — quantifies thermal
  drift in precision stages and its impact on repeatability at the
  nanometre scale. https://hightechsystems.demcon.com/showcases/thermal-drift-in-precision-engineering
- ScienceDirect, "Long term thermal drift study on SPM scanners" — reports
  drift on the order of a few nm/s for conventional mechanical scanning
  stages, i.e. bounded and slow relative to a single revisit, supporting the
  "small bounded offset from the previous site" model used for ground-truth
  placement. https://www.sciencedirect.com/science/article/abs/pii/S095741581100105X
- (recent) "Design, modeling and control of high-bandwidth nano-positioning
  stages for ultra-precise measurement and manufacturing: a survey," Int. J.
  Extreme Manufacturing (2024) — current survey confirming thermal drift as a
  dominant, bounded error source in precision stages.
  https://iopscience.iop.org/article/10.1088/2631-7990/ad6ecc
- (recent) "Vision-based thermal drift monitoring method for machine tools,"
  CIRP Annals (2023) — quantifies stage thermal drift (reduced from 23 to
  8.7 nm/min), i.e. bounded and slow relative to a single revisit.
  https://www.sciencedirect.com/science/article/abs/pii/S0007850623000914

This bounded-drift assumption is also what makes the task's required
disambiguation rule (return the match nearest to the search image center)
physically meaningful rather than arbitrary — see the `max_drift_frac`
docstring in `localize.py`.

## 7. Fin-gate crossing defects break long-range lattice periodicity (`cell_scale` / defect grid)

`generate_pair` randomly suppresses a fraction of fin-gate crossing
"strength" per pair, representing real contact-engineering / strain
non-uniformity at the fin-gate overlap, so that the FinFET lattice is highly
periodic locally (creating the intended matching ambiguity) but not
perfectly periodic globally (real chips are not defect-free lattices):

- IOPscience, "A breakthrough in contact engineering for sub-3 nm FinFETs:
  overcoming the fin-pitch bottleneck" — discusses contact-resistance and
  strain non-uniformity at the fin-gate/source-drain contact interface as a
  known scaling and yield concern.
  https://iopscience.iop.org/article/10.1088/1361-6528/ae2513
- USPTO, "Structure and method for FinFET device with asymmetric contact" —
  documents contact-formation variability at fin-gate crossings as a known
  FinFET process/defect mechanism.
  https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10115825
- USPTO, "FinFET device with contact over dielectric gate" — further public
  documentation of contact-placement/formation variability in FinFET
  structures.
  https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11527651

**Note on the ~40% dropout magnitude (`defect_prob = 0.4`)**: real semiconductor
defect densities are *vanishingly small per feature* — of order 0.1-0.5 defects
per cm^2 at a mature process, i.e. essentially zero per individual crossing:

- AnySilicon Semipedia, "Defect Density (DD)" — industry benchmark defect density
  is < 0.5 def/cm^2. https://anysilicon.com/semipedia/defect-density-dd/
- SIA / ITRS, "Yield Enhancement" chapter — defect-density targets and the
  Poisson/Murphy yield models (Y = e^(-D*A)).
  https://www.semiconductors.org/wp-content/uploads/2018/08/YieldEnhanc2003.pdf
- B. Parhami, "Defects: Physical Imperfections" (UCSB course text) — textbook
  treatment of defect density and defect-tolerance modeling.
  https://web.ece.ucsb.edu/~parhami/docs_folder/f33-book-dep-comp-pt2.pdf

So the ~40% figure is **not** a literal defect rate. It is a deliberate,
task-driven modeling device: the per-crossing "dropout" stands in for the
*aggregate* of all site-distinguishing content variation that breaks perfect
periodicity in a real image (contact-formation variability, strain, fill and CMP
non-uniformity, plus true defects), exaggerated to a density high enough to yield
a usable per-site fingerprint at the 10 nm/px search resolution. It is flagged
here and in the `generate_pair` docstring so it is not mistaken for a claimed
real-world defect density. `forced_periodic` pairs drop this to 3% precisely to
model the near-defect-free regions where this fingerprint is unavailable (the
intended honest-failure case).

## 8. Feature-width / critical-dimension ratios (`line_width`, `gate_width`, `contact_radius`)

The rendered feature widths are set as fractions of their pitch: the fin
brightness profile `line_width ~ 0.25-0.40 x fin_pitch`, the gate-bar profile
`gate_width ~ 0.02-0.035 x gate_pitch`, and the crossing feature
`contact_radius ~ 0.35-0.55 x fin_pitch`. These fractions reflect the documented
fin critical-dimension (CD) to pitch relationship, in which the physical fin CD is
a small fraction (~1/5 to 1/3) of the fin pitch (e.g. ~5-6 nm fin on a ~30 nm fin
pitch at the 7 nm node). The values used here are broadened relative to the bare
physical CD because they parameterize the *rendered SEM brightness profile* (widened
by the section-2 edge-blooming and section-4 PSF blur), not a hard-edged CD:

- WikiChip, "7 nm lithography process" — tabulates public fin pitch (~30 nm), fin
  width (~5-6 nm), and contacted gate pitch (~57 nm) figures.
  https://en.wikichip.org/wiki/7_nm_lithography_process
- L. T. Clark et al., "ASAP7: A 7-nm finFET predictive process design kit,"
  Microelectronics Journal 53 (2016) — a public, fully specified 7 nm FinFET PDK
  giving fin, gate, and metal CD/pitch dimensions.
  https://www.sciencedirect.com/science/article/pii/S002626921630026X
- "Scaling Challenges of FinFET Architecture below 40 nm Contacted Gate Pitch,"
  DRC 2017 — discusses fin-width / gate-length / CPP scaling at advanced nodes.
  https://www.ece.ucdavis.edu/~bbaas/116/docs/paper.FinFET.below.40nm.pitch.2017.06_DRC.conference.pdf
- (recent) "A Review of Reliability in Gate-All-Around Nanosheet Devices,"
  Micromachines (2024) — current-node dimensions succeeding FinFET (nanosheet
  width ~14/30 nm, CPP ~44-48 nm, gate length ~12 nm), confirming the
  fraction-of-pitch CD relationship used here.
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10892190/
- (recent) "Scaling opportunities for Gate-All-Around: A patterning
  perspective," IEDM 2023 — advanced-node fin/sheet patterning pitches and CDs.
  https://mys.mapyourshow.com/mys_shared/iedm23/handouts/2-3_Mon_16241.pdf

## 9. Per-line pitch jitter -> line-edge roughness / pitch walking (`pitch_jitter_std`)

Each fin/gate line is given a small random center offset
(`pitch_jitter_std ~ 0.01-0.03 x fin_pitch`) so the lattice is highly periodic but
not perfectly crystalline. This models two documented, distinct real phenomena:

**Line-edge / line-width roughness (LER/LWR)** — stochastic per-line position and
width variation intrinsic to advanced lithography:

- IBM Research / SPIE Advanced Lithography 2017, "Comprehensive analysis of
  line-edge and line-width roughness for EUV lithography" — LER/LWR across
  lithographic transfer steps.
  https://research.ibm.com/publications/comprehensive-analysis-of-line-edge-and-line-width-roughness-for-euv-lithography
- "Comprehensive Modeling of Line Edge Roughness and Line Width Roughness in
  Semiconductor Nanofabrication," Russian Microelectronics (Springer), 2025 —
  review of LER/LWR characterization and modeling.
  https://link.springer.com/article/10.1134/S1063739725602036

**Pitch walking** — systematic fin-to-fin pitch variation from multi-patterning
(SAQP), i.e. the exact "some lines shifted from the ideal grid" effect modeled:

- K. Bhattacharyya et al., "Measuring self-aligned quadruple patterning pitch
  walking with scatterometry-based metrology utilizing virtual reference,"
  J. Micro/Nanolith. MEMS MOEMS 15(4), 044004 (2016).
  https://www.spiedigitallibrary.org/journals/journal-of-micro-nanopatterning-materials-and-metrology/volume-15/issue-04/044004/Measuring-self-aligned-quadruple-patterning-pitch-walking-with-scatterometry-based/10.1117/1.JMM.15.4.044004.full
- "N7 FinFET Self-Aligned Quadruple Patterning Modeling," SISPAD 2018 — models
  SAQP-induced pitch walk at the 7 nm (24 nm fin-pitch) node.
  http://in4.iue.tuwien.ac.at/pdfs/sispad2018/SISPAD_2018_344-347.pdf
