# Current EM/VDAM development scope

Updated September 28, 2026. Cleanup and scientific qualification remain in
progress. The [task queue](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/pr179_coordination/CURRENT_TASK.md)
contains the active work list; detailed experiment histories live in the private
`ma-gilles/recovar-experiments` repository.

## Subtomogram InitialModel and first-iteration CC (October 1, 2026)

Subtomogram particles (RELION 5 2D stacks) run RELION's VDAM InitialModel
(`relax initial_model --ios optimisation_set.star`, RELION's `--grad --denovo_3dref`) and
`--firstiter_cc` in Refine3D and Class3D. One implementation: the VDAM driver takes a
`TomoDataset` and scores through the subtomogram Refine3D/Class3D pass
(`relax/vdam/tomo_estep.py` over `tomo_half.score_tomo_half`).

- Start-up: RELION counts tilt images against `minimum_nr_particles_sigma2_noise` (10,
  ml_optimiser.cpp:2574, :3058), so the first particle of each optics group gives the noise
  spectrum and the bootstrap; each of its tilt images backprojects at `Aproj R` with its
  dose-damped CTF, and the reference gets no blobs or soft mask (:2707). relax matches RELION's
  `run_it000` (sigma2 noise to 4e-7, class map to 3e-8 relative L2; et09_box64, et15_k2conf_box64 seed 1;
  `em_work/cryoet_vdam_20261001/checks/bootstrap_vs_relion.py`).
- E-step: every tilt image backprojects VDAM's residual into its particle's pseudo-halfset slot
  `class + K * (part_id % 2)`; sigma2_offset divides by 3; auto-sampling keeps at least half the
  offset step (:9832). A tilt chunk whose (slot, row) projections do not fit is projected and scored
  a block of image slots at a time (`resident_tilts.tilt_projection_slot_block`).
- RELION behaviour, reproduced: with one optics group the bootstrap fills class 0 only, and an empty
  class stays an exact zero map (its tau2 is zero), so K>1 VDAM on one optics group is K=1 plus empty
  classes in RELION and relax alike (et15 seed 1: class 2 std 0.0 at iterations 0-100 in both,
  pdf_class at iteration 10 0.655/0.345 relax, 0.643/0.357 RELION). Its tied rotations give single
  particles tens of thousands of significant samples.
- `--firstiter_cc`: the coarse pass adds every tilt image's normalized CC on RELION's square
  current-size crop (DC and x=0 column included), with RELION's 128 atomic additions per image, and
  keeps the first maximum in RELION's orientation order (`tomo_coarse.particle_coarse_cc_winners`);
  pass 2 scores its children (`resident_tilts.tilt_cc_scores`: each image's CC operands translated per
  slot). Class3D's CC iteration is K=1 against class 0 (ml_optimiser.cpp:4389). Iteration 1 of
  et01_base seed 1 against RELION 5.0.1 (mpiscale build, MPI 3x4, H100), `run_it001_data.star`:
  orientations identical for 2000/2000 particles, offsets for 1998/2000
  (`em_work/cryoet_vdam_20261001/cc_it1`). Iteration 1 took 332 s in relax, about 60 min in RELION.
- Speed (open): at et09 iteration 100 (590 particles, HEALPix 3, 180 3D coarse translations) the
  relax iteration is 2.1x RELION's (165 s vs about 78 s), 120 s of it the coarse pass: the per-image
  coarse kernel takes 128 translations per launch, so 180 need two launches that each project all
  36864 orientations. The coarse pass also pulled every particle's tilt-image matrices to the host
  (a sync per particle); they now stay on the device, built once per particle batch (identical
  coarse cut): one late et09 E-step (1000 particles) 137 s -> 107 s, coarse 120 s -> 89 s, et13
  Class3D 25 iterations -4.7%. The coarse kernel `relion_coarse_diff2_projector_f32_kernel` is now
  93% of the E-step's GPU time (85 ms per particle; nsys,
  `em_work/cryoet_vdam_20261001/profile`).
- Subtomogram Refine3D `--firstiter_cc` (et01_base, RELION's default command, RELION 5.0.1
  mpiscale MPI 3x4, H100): masked GT FSC-AUC relax / RELION (two same-seed runs, identical)
  s1 0.99161 / 0.99162, s2 0.99180 / 0.99180, s3 0.99212 / 0.99223; map gate PASS on every seed
  (merged cross-engine 0.99998 at s1-s2, 0.99962 at s3); wall 0.43-0.46x. OPEN: the s3 gap of
  1.1e-4 reproduces in two relax runs (66ff6c8, d6ba262), outside RELION's same-seed range.
- Subtomogram VDAM K=1 (et09_box64, one optics group, stock seeding, s2): masked GT FSC-AUC
  0.98808 inside RELION's same-seed range [0.98795, 0.98816]; wall about 2x RELION before the
  coarse matrices fix.
- With one optics group RELION's subtomogram start-up seeds only class 1 (each group's first
  particle fills the 10-image quota, into class `position % K`), so K>1 VDAM on the etbench
  fixtures keeps classes 2..K empty in both programs (ma-gilles/relax#11). With one optics group
  per tomogram, as RELION 5 imports, every class is seeded; K>1 tomo VDAM is qualified on
  fixtures re-labelled that way (in progress, together with InitialModel on several optics groups).
  Evidence and job IDs: `/scratch/gpfs/CRYOEM/gilleslab/em_work/cryoet_vdam_20261001/HANDOFF.json`.

## PPCA coarse optimizer comparison (September 28, 2026)

The opt-in [PPCA momentum SGD and two-pass coarse route](../math/ppca_momentum_sgd.md)
are on relax main at `2abfe2b2`; VDAM and the existing engine remain defaults.
At noise level 1, two paired 600-update runs used one mean plus two loadings,
20,000 particles, radius 31, HP3, 29 shifts, batch 300 and positivity off.
The metric is per-state rigid+hand-fitted mean FSC over shells 1–15 on the
same fixed 600-particle evaluation subset. GT labels were used only to score
state maps, not during refinement.

| Seed | PPCA VDAM state FSC | PPCA momentum SGD state FSC | Median pose error, VDAM / SGD |
| --- | --- | --- | --- |
| 11 | .181 / .230 / .200 | .282 / .323 / .235 | 130.2° / 125.1° |
| 12 | .202 / .238 / .239 | .300 / .264 / .269 | 134.6° / 128.7° |

SGD has higher final fitted FSC in all six comparisons and avoids VDAM's late
high-band map-power growth. Neither optimizer recovers reliable poses or three
states; latent between-state R² remains at most .106. The archived native K3
VDAM run recovers state 2 in both seeds (FSC .884/.892) but is a different model
and protocol, so it is context rather than a matched optimizer arm. Whole-update
medians were 3.017/3.026 seconds for VDAM and 2.980/2.988 seconds for SGD
(seeds 11/12). Training used 2.074 allocated A100 GPU-hours. Full science
scorecard, registered plots, signed MRCs, exact jobs and reproduction commands
are in `/scratch/gpfs/CRYOEM/gilleslab/em_work/ppca_momentum_gemm_20260928/science_compare/REVIEW.md`.

The integrated source passed local A100 smoke, all 28 medium-tier items and a
source-exact full-grid PPCA gate (24/24 float32 comparison fields, zero pose
flips). Medium job 14643773 passed its tier but its appended gate launcher
failed before execution; gate retry 14654414 passed. The two-pass route saves
residency but was slower than retained scoring at matched B150 (2.89 vs 2.17 s
warm); no dense CUDA/GEMM PPCA adapter was integrated. The next bounded science
check is the curvature-maximum frequency and per-band gradient distribution at
a saved checkpoint before changing the optimizer.

## Coarse K3 SGD comparison (September28,2026)

The opt-in momentum SGD/noise implementation has passed scoped numerical checks,
the medium tier and fresh integration smoke. Matched fixed-HP3, 600-update
noise-1 runs with positivity off recover state 2 in both seeds; states 0/1
remain poorly separated. The completed fixed-settings robustness matrix finds
condition-dependent recovery: preferred orientations improve the candidate's
three-state result, while high noise and nonzero shifts leave both arms with
poor poses. General robustness is not established. See the
[method](../math/momentum_sgd.md), [noise-1 scorecard](../math/momentum_sgd_science_scorecard.md),
[robustness scorecard](../math/momentum_sgd_robustness_scorecard.md),
[source and validation report](../benchmarks/momentum_sgd_coarse_20260928.md),
and [robustness report](../benchmarks/momentum_sgd_robustness_20260928.md).
Native VDAM defaults are unchanged.

## Dense GEMM coarse engine options (September 30)

`coarse_engine=auto|gemm_hybrid|gemm_dense` keeps `auto` as the production default
in standard EM and VDAM, including K>1. `gemm_hybrid` selects GEMM coarse
scoring followed by the existing pruned fine pass. `gemm_dense` is an explicitly
experimental no-pruning route over the full configured pose grid, with exact
per-image normalization; it is not a RELION-parity or performance-qualified
replacement. On the matched 5k-image, box-128, K4 os1 A100 update, warmed
whole-process times were 37.64 s for `auto` and 37.55 s for cached full dense
(job 14694770). Their minimum per-class one-update RELION FSC-AUC was
0.999999993 versus 0.963218, respectively. Neither the measured process tie
nor the divergent result supports using full dense by default. The isolated
kernel speedups are not end-to-end production speedups. Detailed source and
evidence are in the [integration record](dense_gemm_coarse_engine_plan_20260928.md).

## Opt-in dense GEMM K1 experiment (September 28)

The requested full dense grid experiment has exact two-sweep and lagged one-sweep
normalization, image/projection translation expansion, and serial/mixed/full
translation tiles. Exact three-iteration reconstruction closely agrees with the
native dense control; lagged absolute normalization overflows at iteration two.
Production defaults are unchanged. [Math and measured scope](../math/dense_gemm_experiment.md)
and the [detailed plan](dense_gemm_experiment_plan_20260928.md) link the evidence.
The final single raw-batch staging seam passes real GPU operand/upload checks
14632476. Medium14632505 passed all 27 items and 13 FSC comparisons; integrated
smoke14633325 passed all four items and three FSC comparisons. Both enforced
pinned comparisons passed; detailed receipts record the optional fixture skips.
The historical timing control used tomography scoring and a deprecated adjoint
schedule, so its speed ratio does not measure current SPA performance. A corrected
control now uses production-selected GEMM scoring and SPA CUDA row accumulation,
retaining every dense hypothesis. Its matched H100 benchmark 14634396 measured 20.3×/27.9× warmed dense-operator gains
at B100/B512. Full5000-particle, three-iteration run 14634727 measured 271.34 s versus
41.43 s process time, with worst map relative L2 difference 2.75e-5 and half-FSC
difference 1.55e-6. Both warmed profiles show no host transfers or CUDA allocations.
Final post-rebase smoke 14634842 and focused CPU checks passed on main d02b2c46.
These are controlled dense comparisons; historical production-runtime extrapolations
remain withdrawn. The optional experiment does not change production defaults.

## Scope and invariants

Refactor EM/VDAM for readable, succinct code and direct APIs; integrate qualified
peer work; remove demonstrated dead or duplicate code and completed experiments.
CUDA, FFI and their unique numerical coverage are in scope. Avoid generic
executors or mode switches that obscure distinct algorithms.

Validation is EM/VDAM-only. Do not run RECOVAR-wide SPA/ET, downstream, outlier,
indices, stress or heterogeneity suites, and do not regenerate their baselines.
Structural cleanup preserves scientific defaults, casts, reduction order, JIT
boundaries, layouts and buffer lifetime. EM/VDAM APIs and CLIs may change when
maintained callers migrate together; main heterogeneity APIs and serialized
formats remain compatible. Production EM is float32; double precision is a
labeled diagnostic only.

## Feature gaps (moved from the README, 2026-09-29)

- Several optics groups: the default Refine3D (K=1) command runs them, including groups on other
  pixel sizes and boxes, with no settings (2026-09-30). The fresh K=1 pass always uses RELION's
  powerClass spectrum and exact BPref operands; the `RELAX_K1_RELION_POWERCLASS_SPECTRUM_NORM` and
  `RELAX_K1_RELION_EXACT_BPREF_OPERANDS` switches are retired. The `--firstiter_cc` iteration takes
  them too (each shape class's projection matrices, each image's noise row). Qualification on S3b
  (10k particles, 4.25 A/128 px + 5.44 A/112 px), `--no-firstiter_cc` (the reference is on the
  images' greyscale), same H100 node as MPI-scale-patched RELION 3x4 (job 14748349): map gate PASS
  (merged/half1/half2 0.9962/0.9940/0.9945 against two same-command RELION runs), masked GT band
  FSC-AUC 0.9950 vs RELION 0.9946, masked resolution 8.5 A both, wall 400 s vs 784 s (0.51x).
  Evidence: `/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_multioptics_default_20260929/score_7c3ddbe`.
  With `--firstiter_cc` (2026-09-30): group 2's (s = 1.12) Gaussian coarse window (56 px, r_max 25)
  lies between 2 r_max and 2 s r_max from iteration 2, where RELION's coarse kernel projects and
  shifts the rows beyond maxR at `i - window` (see "Multi-optics on another grid" below); the coarse
  GEMM operands now do the same (`helpers.projection.relion_coarse_relabel`, the window's Nyquist
  row included). S3b against two same-command MPI-scale-patched RELION 3x4 runs (they agree to 1.0):
  map gate PASS (merged/half1/half2 0.9989/0.9983/0.9982, job 14796180 at 30cfae8), masked GT band
  FSC-AUC 0.9954 vs RELION 0.9954, masked resolution 8.5 A both; wall 459 s vs 783 s (0.59x) on one
  H100 node (pair 14762768). GEMM scores match the fused coarse kernel to 2e-7 in the band
  (`tests/unit/test_relion_coarse_relabel_gpu.py`). Evidence:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_multioptics_default_20260929/score_30cfae8_fcc`.
  Class3D (K>1) runs several optics groups on one image shape, single-particle (2026-09-30) and
  subtomogram (et16_k2conf_optics2, `/scratch/gpfs/CRYOEM/gilleslab/em_work/cryoet_class3d_20260929`).
  The per-group noise table is never read as per-class (a [G, P] table with G == K was taken as one
  row per class, so every image backprojected with group 1's spectrum). Qualified relative to one
  optics group on the multioptics_k2same_10k128_20260930 fixture (K=2, one reference, --firstiter_cc,
  against MPI-scale-patched RELION 3x4): the CC iteration matches (maps 2.9e-5, one CC pose tie). The
  seed iteration's gap to stock RELION (maps 2.9e-3; 5.6e-3 with one optics group) is stock RELION's
  float32 backprojection: the double-accumulation RELION build differs from stock by the same 5.6e-3,
  and relax matches it at 6.6e-5/1.0e-4 (one optics group, iteration 2). InitialModel (VDAM, single
  particles and subtomograms) takes several optics groups on one image grid (2026-10-02): RELION's
  per-group start-up particles for the noise and the bootstrap (`relion_startup_positions`), each
  image scored with its group's noise row, the noise updated per group, the expected accuracy per
  group. Start-up against stock RELION non-MPI: every group's sigma2_noise to 4e-7, class maps to
  1.5e-8 (et15 with one group per tomogram) and exactly (SPA two groups). End to end, masked GT
  FSC-AUC (181e99d scorer) inside or above stock RELION's same-seed range at all three seeds on
  et09 with one group per tomogram (8 groups, K=1); the SPA two-group K=2 fixture collapses to one
  class in both programs (a plumbing check only). Groups on other grids stay refused (etw's
  multishape K>1 route). Evidence: `/scratch/gpfs/CRYOEM/gilleslab/em_work/cryoet_vdam_20261001`.
  Evidence: `/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_k2optics_20260930`.
- Class3D (K>1) on optics groups of several image shapes (2026-10-01): each half is scored per shape
  class and merged (`optics_shapes.merge_k_class_outputs`), as K=1 is. Qualified on
  multioptics_k2_10k128_20260930 (K=2, 128 px at 4.25 A and 112 px at 5.44 A, --firstiter_cc), one
  seed (1), against stock RELION 5.0.1 on the CPU as the reference, whose classification pass applies
  the scale difference (ma-gilles/relax#12): masked GT FSC-AUC 0.4659 vs 0.4612, class accuracy 0.7134
  vs 0.7105 (two identical CPU runs); relax-vs-RELION per-class cross FSC-AUC 0.970/0.976. RELION's GPU run, which skips the
  scale difference in pass 1, reaches 0.3880/0.6102. VDAM on several shapes is not wired yet; its
  engine-level loop will use `optics_shapes.shape_class_engine_inputs` and `merge_k_class_engine_results`.
  Evidence: `/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_k2optics_20260930/score_twoshape_68d7ea1_out.json`.
- Optics features in Refine3D (K=1): CTF-premultiplied particles
  (`rlnCtfDataAreCtfPremultiplied`), beam tilt and odd Zernike aberrations (image
  demodulation), even Zernike aberrations and anisotropic magnification (`rlnMagMat*`).
  Class3D (K>1) takes all four, including RELION's average-CTF² correction of
  data_vs_prior for premultiplied data (`setAverageCTF2`); magnification is qualified against
  stock RELION on the CPU (below).
  InitialModel refuses all four until it has its own qualification against RELION.
  Qualification, 2026-09-30: 10k/256 fixtures under
  `/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_fixtures/optics_*_k1_10k256_20260929`, one relax run
  (cand_ccb229f, H100) against same-command, same-seed RELION 5.0.1 runs (patched MPI build,
  3 ranks x 4 threads, H100); map gate from `scripts/score_k1_map_gate.py`, masked GT FSC-AUC
  from `scripts/masked_fsc.py` (mask `synth_k1_50k256_snr05_c1`).

  | Feature | Status | relax GT FSC-AUC | RELION runs | Map gate |
  | --- | --- | --- | --- | --- |
  | Premultiplied (job 14744731) | qualified | 0.94516 | 0.94464, 0.94460 | pass |
  | Even Zernike (14744732) | qualified | 0.94908 | 0.94891, 0.94829 | pass |
  | Beam tilt + odd Zernike (14720565, 14735008) | provisional | 0.94858, 0.94891 | 0.95462, 0.94915, 0.94901 | pass |
  | Magnification (14744733) | provisional | 0.94954 | 0.95099, 0.94966, 0.94985 | pass |

  Beam tilt: relax is 1.0e-4 and 4.3e-4 below the three-run RELION band; relax's own spread is
  3.3e-4, and its map agrees with the third RELION run at 0.998 (masked merged FSC band AUC).
  Its phase matches RELION's code in unit tests. Magnification: relax is 1.2e-4 below the
  three-run band, with 0.998 map agreement against the third RELION run. Whether sub-5e-4 gaps
  where relax's map reproduces a RELION run count as ties is with the user.

  Class3D K=2 qualification, 2026-09-30: 10k/256 fixtures
  `/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_fixtures/optics_*_k2_10k256_20260930` (first 10k of
  synth_pdb_k2_50k256_snr05), the synth_pdb_k2 Class3D command at seed 42; one relax run
  (cand_76ae34a, job 14795328) against four same-command RELION 5.0.1 non-MPI runs, two float
  builds (14774524-7) and two double-precision back-projection builds (14795376-7). Masked mean
  GT FSC-AUC (mask `synth_pdb_k2_50k256_snr05_c1`, classes matched to GT by Hungarian
  assignment; scorer em_work/relax_bench_k1plus_20260925/tools/score_class3d.py); scores in
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_bindw_20260930/classes/<feature>/score_76ae34a.json`.

  | Feature | Status | relax | RELION float | RELION double | class accuracy relax / RELION |
  | --- | --- | --- | --- | --- | --- |
  | Premultiplied | qualified | 0.288584 | 0.284788, 0.287513 | 0.287528, 0.288820 | 0.9531 / 0.9526-0.9567 |
  | Beam tilt + odd Zernike | provisional | 0.208885 | 0.208436, 0.208290 | 0.209495, 0.209034 | 0.8439 / 0.8433-0.8448 |
  | Even Zernike | provisional | 0.213436 | 0.213352, 0.213999 | 0.213522, 0.214689 | 0.8494 / 0.8493-0.8536 |
  | Magnification | qualified against RELION CPU (below) | 0.208381 | 0.213430, 0.215135 | 0.214712, 0.214480 | 0.8373 / 0.8471-0.8543 |

  Beam tilt and even Zernike are inside the four-run band and 1.5e-4 and 8.6e-5 below the
  double-accumulation pair.

  Magnification (K>1), qualified against stock RELION CPU (2026-10-02). RELION's GPU
  classification scores pass 1 without the optics group's magnification and scale difference
  (predefined coarse projector plans, relax#12), so the GPU runs above are not the reference;
  relax applies them in every pass, as RELION's CPU path and its GPU auto-refine do. Reference:
  stock RELION 5.0.1 f2c1a38 on the CPU (no `--gpu`, `--j 32`; deterministic, `--j 24` gives the
  same run bit for bit); relax cand_460ff32 (main fd64ead with magnification accepted); jobs
  14835479, 14835480, 14852027 (relax), 14824531, 14835481, 14835482, 14852025, 14852026 (RELION
  CPU); scores in `/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_bindw_20260930/classes/mag/`
  (`score_seeds.json`, `score_seeds2_relax.json`, `score_seeds3.json`). Benchmark multi-seed rule:
  every relax run is inside or above the across-seed RELION band [0.205033, 0.211092], and relax
  is not below its same-seed RELION run at three of five seeds. It is below at s42 and s29; the
  worst seed is s29 at -1.7e-3. The reference is RELION CPU, not RELION GPU, because of relax#12.

  | Seed | relax (masked mean GT FSC-AUC) | RELION CPU | relax vs same-seed RELION |
  | --- | --- | --- | --- |
  | 42 | 0.206533, 0.207609, 0.209553 | 0.211092 (repeated) | below (best run -1.5e-3) |
  | 29 | 0.209362, 0.209036 | 0.211077 | below (-1.7e-3) |
  | 53 | 0.211691, 0.211337 | 0.210008 | above |
  | 61 | 0.210088 | 0.205033 | above |
  | 67 | 0.208088 | 0.205250 | above |

  Iteration-1 poses: relax and RELION CPU agree for 98.6-98.9% of particles at each seed. RELION
  disagrees with itself at the same level on this data: its GPU auto-refine K=1 (magnification
  applied, equal to relax K=1 in 100% of poses) and its CPU K=1 agree for 98.45% (jobs 14812274,
  14851745), against 99.24% on the even Zernike fixture, where RELION CPU is 2.4-3.7e-3 below
  RELION GPU and relax (0.210942 vs 0.213352-0.214689).
- CTF: relax evaluates RELION's CTF rows (`CTF::getFftwImage`, including the per-particle
  `rlnCtfBfactor` and `rlnCtfScalefactor`) in its own host float64 code
  (`relax/relion/relion_ctf.py`), cast to float32 before GPU scoring. Production CTF no longer
  calls RELION code; the RELION binding is only the unit-test oracle
  (`tests/unit/test_relion_ctf_formula.py`).
- Follow-up: the exact-CTF device row cache (`relion_ctf._exact_ctf_device_rows`) still keeps
  float64 rows on the GPU, and the elementwise ops on them run in float64 before the cast to
  float32. They are cheap. Moving them to float32 is a parity-qualified change that has not
  been made yet.
- Refused permanently, with a clear message: `rlnCtfDataAreCtfCorrected` and several different
  `rlnMtfFileName` values.
- Subtomogram particles (RELION 5 2D stacks) run the first-iteration cross-correlation
  (`--firstiter_cc`) in Refine3D and Class3D since 2026-10-01 (see "Subtomogram InitialModel
  and first-iteration CC" below); its end-to-end qualification against RELION's default command
  is running.
- Subtomogram InitialModel (`relax initial_model --ios`) takes several optics groups on one image
  grid (above). RELION's subtomogram VDAM with one optics group leaves classes 2..K empty (below), so
  K>1 tests use one optics group per tomogram, as RELION 5 imports (et15_ogtomo; the recovar tomo
  simulator writes that by default since recovar 24e3fa27b).

## RELION binding removal (started 2026-09-29)

User rule (2026-09-29): production code must not rely on RELION code. The compiled binding
(`relax/relion_bind`, `_relion_bind_core`) stays only as the oracle of unit tests, diagnostics
and parity tools. Each port keeps RELION's exact semantics and gets a CPU test against the
binding (integers exact, floats at a measured relative tolerance). A lint test will fail on any
import of `relion_bind` outside `relax/relion_bind`, `relax/diagnostics`, `relax/reference` and
`tests`; until the last port lands it carries an allowlist that shrinks with each port.

Production call sites at relax main `93bf20e`; the status column records each port. Every port has a CPU test against
the binding in `tests/unit/test_*_vs_relion_bind.py` (and `tests/unit/initial_model/`).

| Module | Function | Binding call | Runs that use it | What it computes | Status |
| --- | --- | --- | --- | --- | --- |
| `symmetry.py` | `_operators_float64` | `get_symmetry_operators` | every run (C1 hash/point-group code; operators for non-C1) | RELION `SymList` ordered L/R operators and point-group code | ported (`symmetry._operators_float64`) |
| `sampling.py` | `_get_relion_grid_metadata` | `get_healpix_directions` (C1), `get_healpix_sampling_metadata` (non-C1) | every run | HEALPix NEST directions (rot, tilt), retained pixel ids, psi grid; asymmetric-unit pruning for non-C1 | ported (`healpix_sampling.py`) |
| `sampling.py` | `get_relion_rotation_grid`, `_get_relion_rotation_grid_eulers_float64` | `get_coarse_orientations` | every run | the coarse (direction, psi) Euler grid in RELION order | ported (`healpix_sampling.py`) |
| `sampling.py` | oversampled-orientation builder | `get_oversampled_orientations_batch` (NumPy fallback for C1; required for non-C1) | every run (fine pass) | `HealpixSampling::getOrientations` children with random perturbation | ported (`healpix_sampling.py`) |
| `sampling.py` | inverse scoring matrices | `euler_angles_to_inverse_matrices` (NumPy fallback exists) | every run | `Euler_angles2matrix` then `Matrix2D::inv` | ported (`healpix_sampling.py`) |
| `sampling.py` | device scoring perturbation | `euler_angles_to_matrix` (NumPy fallback exists) | every run with perturbation | perturbation Euler matrix | ported (`healpix_sampling.py`) |
| `sampling.py` | `_relion_rnd_unif_scaled_first_draw` | `vdam_rnd_unif_range_sequence` (glibc `ctypes` fallback) | every run (perturbation draw) | `init_random_generator(seed)` then `rnd_unif(low, high)` | ported (`helpers/relion_random.py`) |
| `relion/relion_projector_setup.py` | `reference_to_relion_projector_half_maps_and_power` | `compute_fourier_transform_map` | `--projector_setup_backend native` and geometry the JAX path does not take (odd box, padding > 2, interpolator 0) | `Projector::computeFourierTransformMap` | removed: one device setup, other geometry refused; RELION's transform is `diagnostics/native_projector_setup.py` |
| `reconstruction/regularization_relion.py` | tau2 from reference | `compute_fourier_transform_map` | callers that pass no `projector_power_spectrum` | projector power spectrum | ported (device projector power) |
| `relion/relion_ctf.py` | exact CTF | `get_ctf_images_batch` | CTF images | `CTF::getFftwImage` | ported by optics (0cbfb8b) |
| `helpers/expected_accuracy.py` | `estimate_relion_expected_accuracy_from_prepared_inputs` | `vdam_expected_angular_errors` | auto-refine, Class3D, InitialModel, tomo | `MlOptimiser::calculateExpectedAngularErrors` | ported (`helpers/relion_expected_accuracy.py`, CTF from `relion_ctf.relion_ctf_fftw_half`) |
| `helpers/expected_accuracy.py` | `relion_auto_refine_half_orders`, `relion_half1_trial_order`, `relion_class3d_trial_layout` | `auto_refine_randomise_half_orders_mt19937` | auto-refine, Class3D | `std::shuffle` with `mt19937(seed + iter)` | ported (`relion_random.shuffled_orders`) |
| `vdam/subset_schedule.py` | subset shuffle | `vdam_randomise_particles_order` | InitialModel | `std::shuffle` with `mt19937(seed + iter)` | ported (`relion_random.shuffled_orders`) |
| `vdam/iteration_loop.py` | `refresh_tau2_from_projector_power` | `vdam_projector_power_spectrum` | InitialModel | `MlModel::setFourierTransformMaps` tau2 | ported (device projector power) |
| `vdam/bootstrap_iref.py` | `compute_bootstrap_iref_via_cpp` | `vdam_bootstrap_iref` | InitialModel | random-angle bootstrap reconstruction (ml_optimiser.cpp:3127-3205) | ported (`vdam/bootstrap_reconstruction.py`, now `compute_bootstrap_iref` / `postprocess_bootstrap_iref`) |
| `vdam/bootstrap_iref.py` | `postprocess_bootstrap_iref_via_cpp` | `vdam_postprocess_initial_iref` | InitialModel | blobs, low-pass and soft mask with the C `rand()` stream | ported (`vdam/bootstrap_reconstruction.py`, now `compute_bootstrap_iref` / `postprocess_bootstrap_iref`) |
| `vdam/mstep_single_class.py` | `_get_bindings` and the step-by-step path | `vdam_reweight_grad`, `vdam_first_moment`, `vdam_second_moment`, `vdam_apply_momenta`, `vdam_update_ssnr_arrays_from_bpref`, `vdam_reconstruct_grad` | InitialModel (module required by the default JAX transaction; step path only with dumps/replay or `use_native_transaction=False`) | VDAM M-step | done: production runs the transaction; RELION's steps are `diagnostics/vdam_native_mstep.py` |
| `relion/relion_vdam_mstep.py` | `_first_moment_initializes`, `relion_vdam_m_step_host` | `vdam_first_moment_initializes`, `vdam_m_step_transaction` (FFT grid < 16) | InitialModel | serial first-moment sum test; small-grid M-step | ported (serial host sum; grids < 16 refused) |
| `commands/ppca_initial_model.py` | `source_identity` | module file hash | PPCA initial model | provenance hash of the binding | removed (no binding in provenance) |

`commands/build_relion_bind.py` builds the oracle and is a tool, not a production path.

## Architecture and ownership

`relax/` is the implementation root. Standard refinement and VDAM retain
separate controllers and schedules while sharing sampling, scoring, candidate
layouts and accumulation where their semantics match. RELION runtime adapters,
replay tools, diagnostics and independent references have explicit owners. See
the [codebase map](codebase.md) and [implementation guide](em_implementation.md).

The lead integration branch is `codex/integrate-pr180` in
`recovar_structural_cleanup_20260907`. It contains the selected EM peer head
`1923240c0` and VDAM peer head `5a7fc9037`, plus integration fixes and cleanup.
The two peer branches remain independent sources of future candidates; integrate
at deliberate checkpoints without interrupting their work. Continue selective
review of Roey's `dense_em_refactor`, keeping the clearer implementation where
approaches overlap.

Completed experiment material belongs in the private
[recovar-experiments](https://github.com/ma-gilles/recovar-experiments) archive
with source identity and reproduction records. Main retains production code,
maintained workflows, current scorecards and unique numerical tests. The exact
pre-cleanup status and real-data evidence inventories are preserved in
[archive commit 8b62d4e](https://github.com/ma-gilles/recovar-experiments/tree/8b62d4e1389cb7c106de2436ac38df6e0b7ca172/snapshots/em_development_records_20260921).

## One engine: removal TODO

User decision (2026-09-26): relax keeps one pass-2 engine, the device-resident one
(`relax/sparse_pass2/resident_*.py`), as RELION keeps one algorithm. Every other pass-2
engine or route is deprecated: its docstring says so, and a run that routes a pass to it
logs one `DEPRECATED engine` warning per engine, pass kind and reason
(`relax.sparse_pass2.engine_record.warn_deprecated_engine`), next to the per-iteration
`pass2_engine_trajectory` entry. They are removed in the order below once resident covers
what still routes to them. Inventory and line estimates (relax bc6d3e1):
`/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_onengine_20260926/PLAN.md`.

| Deprecated engine or route | What still routes to it on main | Resident work needed |
|---|---|---|
| Generic dense K-class coarse scorer (pass 1: `scoring.significance._compute_k_class_significance_batched`, `_score_block`, `_add_priors`, `_e_step_block_scores_normalized_cc`) | K=1 runs without the fresh BPref order (RELION-seeded or replay starts) and every normalized-CC (`--firstiter_cc`) pass; Class3D and VDAM at every K already score on RELION's exact coarse operands | Exact-operand coarse scoring for those passes, then delete the generic scorer with the temporary `relion_exact_coarse` switch (kspeed; Coarse scorer TODO below) |
| Dense `run_em` (`dense/em_engine.py`, `dense_big_jit.py`, `k_class.run_dense_k_class_em`) and the per-image reference route (`reference/sparse_pass2.py`) | Nothing in production (the CLI always builds scale groups and supplies RELION's projector): oversampling 0 without scale groups, `RELAX_K1_DENSE_PASS2` / `RELAX_K_CLASS_DENSE_PASS2`, VDAM `RELAX_DISABLE_SPARSE_PASS2`, the dense K-class fallbacks, a full-grid C1 pass without supports | Move the joint `--firstiter_cc` coarse probe (pass 1) out of the dense K-class wrapper |

Removal order:

1. Tests first (done 2026-09-26): `tests/helpers/relion_estep_reference.py` restates RELION's
   GPU fine pass in NumPy float64, and `tests/unit/test_resident_relion_reference.py` pins the
   resident drivers to it on the 8x8 fixtures: K=1 at current size 6 and the full box, K=2/3
   Class3D, wide translation grids, VDAM's residual backprojection, zero oversampling, C4 and
   the local fine pass. Sibling-engine agreement cannot validate a convention, so the tests
   that pin resident against compact or exact local are deleted with the engine they compare
   against.
2. Compact, with the K=1 exact-local adaptive route and the dispatch fallback (done
   2026-09-26, about 22k lines of `relax/`): a configuration the resident checks refuse is now
   an error, subset and focused replays take RELION's atomic Wavg arithmetic (user decision),
   the compact-only diagnostic dumps are retired, and pass 2 needs a CUDA GPU (user decision:
   no CPU back end; CPU correctness runs on the NumPy reference). The one memory refusal left is
   a smallest chunk that does not fit the device (the joint chunk planner, 4c3b2af, shrinks
   every larger plan to fit).
3. The full-box final pass on resident local (done, 1178448), then the parent probe.
4. The VDAM exact-local route (done 2026-09-27: `vdam/sparse_pass2_estep.py`, `k_class.run_local_k_class_em`,
   `--pass2_engine local/local_segmented` and the four options only it read are removed; the
   adaptive route is VDAM's only E-step route, and a configuration it refuses is an error).
5. The exact local engine (done 2026-09-30, about 24k lines of `relax/`): the parent probe runs on the
   resident local pass, and `local/local_em_engine.run_local_em_exact`, the `local/` modules only it reached
   and the helpers, diagnostics and tests that served it are removed; `local_layout` and the M-step row
   helpers in `local_backprojection` stay.
6. Dense `run_em` and the per-image reference (about 4k lines).

Tomography (S4) runs only on the resident engine (`compute_tilt_pass2_stats_resident`)
and pins none of these.

Coarse scorer TODO (team-lead, 2026-09-27): one coarse path for every K. Class3D and VDAM at every
K score pass 1 on RELION's exact coarse operands (`relion_exact_coarse`, set by
`relax.refinement.half_scoring` and `relax.vdam.adaptive_estep`), and fresh K=1 Refine3D does too.
K=1 runs without the fresh order (RELION-seeded or replay starts) and every normalized-CC pass
(`--firstiter_cc`) still take the generic dense K-class scorer in
`relax.scoring.significance._compute_k_class_significance_batched` (`_score_block`, `_add_priors`,
`_e_step_block_scores_normalized_cc`). Owner kspeed: move them, then delete the generic scorer and
the temporary `relion_exact_coarse` switch (`relax.classification.k_class`) together.

Projection kernel (2026-09-27, kspeed, from team-lead's TODO): `project_relion_half_capacity`
and the half-storage branch of `relax.helpers.projection._project_relion_projector_texture` take
every slab whose texels fit the staging kernel's int32 indexing (box 800 at padding 2 is 1603 x
1603 x 802), in launches of at most 65535 rotations, so one kernel serves every size. A resident
local half stages one `RelionCapacityHalfTextureF32` for every slab, a plane group at a time
(1 GiB of staging beside the texture); `resident_local_pass2._open_resident_local_projector_texture`
and its persistent texture are gone. The per-call `relion_projector_half_texture_f32` branch now
serves only the geometry the half-storage kernel does not take (odd output sizes, padding other
than 1 or 2). The XLA pool reserve (`relax/helpers/xla_memory_reserve.py`) sizes the texture.

## Current evidence and open gates

The integrated CPU EM/InitialModel scope and focused GPU kernel suites pass. The
VDAM merge guard passes its maintained 12-case panel. The current consolidated
scorecards report K1 31/34, K4 direct 41/60, K4 all-class 9/15 and VDAM 12/12;
these are progress measures, not completion. The three recorded real-data K1
calibration cases remain in the maintained science-equivalence scorecard, while
the 10202 target is still pending.

Class3D seed-iteration gap (resolved 2026-10-01): on a seed iteration with flat posteriors (Pmax 0.004,
about 4500 significant samples per particle) relax's class maps were 0.27% below stock RELION's,
uniformly. The double-accumulation RELION build (double device backprojection) moves RELION by the same
5.6e-3 and relax matches it to 1e-4 at iteration 2: stock RELION's float32 accumulation of many small
weights, not a relax difference. Per-particle significant-sample counts differ for 25% of particles by
float32-level score differences under flat posteriors (the seed iteration scores one class per
particle, through the K=1 RELION coarse weight order). Evidence: `/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_k2optics_20260930`
(diag_oneopt_relion_bpd). Subtomogram Class3D (et16) hard single-particle offset switches remain OPEN.

EMPIAR-10202 (box 800, I1) progress on the resident K=1 engine: iteration 3
(current size 304) spent 4,530 s per half in the pass-2 M-step on 589ce09, which
visited all 55.4M candidate rows (bigbox 14450997, cancelled). With the live-row
M-step of 68a3cf0 the pass-2 chunk loop takes 52.9 s and 49.4 s per half and pass 1
about 128 s per half (bigbox 14456981), against RELION's 485 s for the whole
iteration on 2 H100s. The same runs exposed two box-800 memory faults, both fixed:
the cached-path chunk gather (131072 rows x 36514 pixels = 35.7 GiB, now bounded by
measured free memory) and the low-resolution half join, which moved a physically
large grid's host accumulators back to the device before the 1600^3 inverse FFT.

RELION's projector (`Projector::computeFourierTransformMap`) has one device build
(`relax/relion/relion_projector_setup.py::_build_projector_window`, bigbox 2026-09-27): the padded
transform one axis at a time inside a window, mask and shell power on the device. It replaced the
whole-volume rfftn build, which needed about 100 GB at box 800 and left 10202 on the native host
binding (two single-threaded builds, 117 s per iteration in py-spy 14561585). At 10202 it22 the
logged projector step is 14.1 s, against 61.8 s native (bigbox 14564062, 14561585); padded
128-512 costs 0.85-1.02x the old device build at full radius and
0.40-0.79x at half radius (bench 14564054). The device build is the default; native is the test
reference.

K=1 auto-refine HEALPix cap removed (2026-09-27, team-lead decision): relax capped K=1 sampling
at HEALPix order 7 by default, and at the cap `convergence.update_angular_sampling` never latches
`has_fine_enough_angular_sampling`, so a dataset that needs a finer grid never converges. RELION's
auto-refine has no cap (ml_optimiser.cpp `updateAngularSampling`, 11731-11753: the order goes up
while the old step is at least 75% of `acc_rot`, and the flag latches below it). On EMPIAR-10202
RELION went to order 8 at iteration 21 and 9 at 24 and converged at 27 (14475516), while relax
continuations sat at order 7 and 2.96 A for 23 and 44 iterations (bigbox 14522322, 14514496). The
default is now uncapped; `--max_healpix_order` remains an explicit opt-in cap.

Knife-edge note, pdb K=1 100k/256 (bench 14445854): relax converged at HEALPix 6, RELION refined to 7 at
its iteration 16 and converged at 18. Every latch input matched through iteration 15 (resolution, current
size, stall counters, the 0.46875 deg step) except `acc_rot` at the start of iteration 16: relax
0.6250000000000003, RELION 0.624, around the 0.625 threshold of `old_rottilt_step < 0.75 * acc_rot`
(ml_optimiser.cpp 11746-11748). relax's expected-accuracy binding accumulates the trial errors in RELION's
order and type (double, trial order, one division) and reproduces RELION exactly on RELION's own inputs:
0.6270000000000002 from `run_it014`, 0.6240000000000003 from `run_it015` (bigbox 2026-09-27). The flip is
therefore trajectory, one 0.1 deg step in one of the 100 trial particles from relax's own iteration-15 maps
and poses, not a parity defect.

Run files and `--continue` (2026-09-26): auto-refine and Class3D write RELION's
`run_itNNN_{optimiser,model,data,sampling}.star` and maps every iteration and continue from
them ([algorithm map section 8](../math/relion_refinement_algorithm.md)). Two departures from
relion_refine `--continue` keep a continued run equal to the uninterrupted one: each half keeps
its own noise (RELION's MPI restart broadcasts half 1's, ml_optimiser_mpi.cpp:750-758) and the
particle order stays the run's first mt19937 order (RELION reshuffles with random_seed + iter,
exp_model.cpp:406-446); STAR floats are written at full precision. On K1 5k/128 to convergence
(17 iterations, local search from 8) and K4 5k/128 (12 iterations), runs continued from several
iterations stay inside the same-code repeat band at every later iteration, with identical
schedules and convergence iteration (Slurm 14470000/1, 14470259, 14470264). On K1 C4 (14470328,
14470680, 14475183) continuations from iterations 4, 6, 8, 12, 17 and 20 match to 1e-6; from 5, the
iteration after a HEALPix refinement, every restored input equals the uninterrupted run's except the
references, which differ by 2.6e-7 after the float32 MRC round trip that RELION's `--continue` also
makes, and that flips one half-2 particle's significance support (44071 vs 44072 samples): half maps
differ by 1.9e-4 at iteration 6, the kind of single-particle flip a same-code repeat shows there. This
C4 fixture is chaotic from iteration 7 (one of two repeats converges at 19 instead of 21). VDAM has
no continuation yet.

The supported K4 comparison still has a class below its FSC-AUC gate. Saved
launch-ladder contributors agree in a bounded sample, while GPU accumulators are
not bitwise repeatable even on the control. Neither finding justifies a tolerance
change or a rounding-noise dismissal. A BPref accumulator guard now stops a
known non-finite compact-engine failure at its source; the mechanism remains an
open defect.

Class3D (K>1) starts standalone by default: it reads only relion_refine's inputs
([launch recipe](em_parity_runbook.md#standalone-class3d-launch)). On the K4
50k/256 fixture two standalone runs match the non-MPI RELION reference's final
resolution, ground-truth FSC-AUC and class agreement. OPEN (small): their
per-class FSC-AUC against the RELION reference is 0.0002-0.0007 below the band
of three same-seed RELION repeats, comparable to the 8e-4 difference between the
two relax runs (A100 and H100); the cause is unexplained. Evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_kclassstandalone_20260923/band50k/GATE.json`.

Multi-optics on another grid (2026-09-25): an optics group whose box x pixel size
exceeds the reference's (scale s > 1) gets a Fourier window wider than the model
sphere. RELION's accelerated kernels relabel the image rows beyond
maxR = min(PPref.r_max, imgX - 1): the coarse diff2 kernel wraps them negative
(acc/cuda/cuda_kernels/diff2.cuh:86-90, 163-164), and the fine diff2 and wavg kernels
move them to the pixel (maxR, i) for both the projection and the image shift
(diff2.cuh:268-304, wavg.cuh:81-86). relax reproduces this
([sparse_projection_radius.md](../math/sparse_projection_radius.md)). With it, the
S3b fast case matches RELION to 1e-8. OPEN, a RELION defect not yet reproduced:
for s >= sqrt(2) the moved fine pixel falls inside the sphere and RELION scores a
wrong pixel at a wrong phase. relax refuses such groups (list the widest optics
group first); the reproduction is planned inside the S4.2 scorer changes
(MANAGER_DECISIONS items 11 and 13). Resolved: the M-step of a class at its full box (S3b optics
group 2 from iteration 3) dropped RELION's N/2..N/2+1/2 ring (shells 50-51). RELION backprojects the
rounded support and bounds only the rotated reference radius (BP.cuh:322), so the class window now
backprojects that support and its reference-sphere clip is the exact cut; the multi-optics replay of
iteration 3 then matches RELION's BPref, and the S3b end-to-end run follows RELION's trajectory
through iteration 6. A coarse window strictly between
2 maxR and about 2 s maxR, where RELION's wrapped coarse rows land inside the
sphere: the fused coarse scorer and the coarse GEMM operands reproduce it (the
projection and the image shift at the relabelled row, `relion_coarse_relabel`).
Still OPEN: the local parent pass refuses such a window (the S3b `--firstiter_cc`
run never reaches one); its image-shift relabel belongs in the resident parent probe.

Resolved (2026-09-29, cryo-ET): subtomogram runs write RELION's run files and continue from them (e435bfa4). The half
rows already map the particles onto the tomo particles.star; data.star now carries rlnOriginZAngst for the 3D offsets.
Check 14643890 (S1 seed 20260925): a run continued from iteration 6 ends on the uninterrupted run's map (FSC-AUC 1.0,
same iterations 7-9).

Resolved (2026-09-28, cryo-ET S4.2): relax's ground-truth FSC-AUC on the S1 depth-fix fixture was 0.003-0.004 below RELION's
(0.9346-0.9367 against 0.9381-0.9409). Two bugs, both found by comparing each iteration with RELION's `_model.star`/`_data.star`
and by one-iteration replays from RELION's state: the tilt fold added tilt-image power above the norm cutoff with mass 1 instead
of 1 / n_images (noise beyond the current size 39x RELION's; 3409d7bc), and pass 2 backprojected the tilt images without their
group scale (RELION's ctfs = Fctf * scale, acc_ml_optimiser_impl.h:4370-4404; the it002 replay from RELION's it001 state had
BPref data 4.9% off, 1.5% after the fix, the rest being a few particles' pose flips; 9f8efc89). Gate 14624974: relax GT
0.9354-0.9409 against RELION 0.9381-0.9409; seeds 20260926 and 20260927 reproduce the same-seed RELION maps (relax-vs-RELION
FSC-AUC 1.0000), seed 20260925 leaves RELION's trajectory at iteration 5 (0.9354). Also RELION-matched on the way: no norm
correction for subtomograms (608a997b).

RELION float32 BPref accumulation band (2026-09-25, not reproduced; lead decision). RELION's GPU backprojector adds every
particle of a half into one float32 volume with `atomicAdd` (acc/acc_backprojector.h:41; acc/cuda/cuda_kernels/BP.cuh:157-169)
and reads it back once per iteration (ml_optimiser_mpi.cpp:1719). At iteration 1, when posteriors are broad, the rounding drops
tiny terms. On S3b the DC BPref weight is 0.32 % below `0.999 sum Q^2/sigma2[0]` for RELION and 0.083 % below it for relax,
decaying with radius. RELION's retained mass is 0.999 (14428592), and a float32 sequential-sum simulation reproduces RELION's DC
to 1e-4. By it3 the formula holds to 1e-5. relax loses less because each optics shape class starts from its own zero
accumulator. Single-optics runs are unaffected. This is not an algorithmic choice and depends on accumulation order, so relax
does not reproduce it. It also does not explain the S3b final gap: a free run from RELION's it001 (14431963) ends at GT FSC-AUC
0.8929, where RELION reaches 0.9055 and relax standalone 0.8923. Resolved (2026-09-25): that gap was the exact local
engine at the full box, which scored the whole half grid. RELION's resolution pointers still cut the corners there
(ires < image_current_size / 2 + 1), and for optics group 2 (s = 1.12) the corners project inside the model sphere, so
relax's local scores carried a rotation- and shift-dependent term (RELION ACC dumps at it013: coarse diff2 residual std
10-11, with the same argmin). The local search now scores RELION's window at the box (window_at_box, as the resident
drivers). One-step replays from it012: group-2 Pmax correlation 0.59 -> 0.998 (group 1 unchanged, 0.9999); standalone
S3b GT FSC-AUC 0.9057 (RELION 0.9055, main before the fix 0.8923), relax vs RELION 0.9957 (was 0.9755), 844 s wall
(job 14451343). Evidence: `/scratch/gpfs/CRYOEM/gilleslab/em_work/multioptics_spa_s3_20260924/s3b_it013_dumps_20260925`,
`/scratch/gpfs/CRYOEM/gilleslab/em_work/exactlocal_box_20260925`, `s3b_winbox_{ctrl_01fddc2,cand_9c41737}_20260925`.

VDAM K>1 (2026-09-25): relax main 237e76b normalizes sigma2_noise, pdf_class,
sigma2_offset and ave_Pmax by RELION's retained (significant-pruned) class mass;
it used the full mass before, which moved every K>1 trajectory from iteration 1.
On pdb K=2 5k/128 the class splits now match RELION at seeds 29/41/53. The seed-29
spread is closed as basin sampling (2026-09-25): 8 relax vs 7 RELION runs have equal
means (0.3275 vs 0.3312, Welch p = 0.29) and relax's sd is 2.1x RELION's (0.0081 vs
0.0039, F p = 0.049, Levene p = 0.023). There is no per-step mechanism: repeats of both
engines first differ at iterations 48-61 and diverge at the same rate through iteration
70, and one-step relax replays from RELION's checkpoints at iterations 60-199 match
RELION's next iteration at its 6-decimal STAR rounding (class-2 weight within 4.5e-7,
alternating sign; class assignment 100%). Evidence and tools:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_vdamk2_20260924/HANDOFF.json`.

VDAM defaults (2026-09-25, landed from the vdamspeed stack): RELION's ternary max/min in the JAX
M-step FSC estimate; the exact K=1 pass 2 always packs flat local rows; the float32 JAX M-step
transaction is the only default route (float64 stays a diagnostic); one projector setup, the device
FFT in double narrowed to the complex64 slab RELION's GPU projector holds as a float texture
(`--projector-setup-backend` is gone from InitialModel); the noise carry starts at its block dtype.
PPCA: the VDAM comparison arm of the PPCA pilots now builds its projector in double for float32 runs
too (before, the JAX setup followed the M-step dtype through `_projector_setup_dtype`, removed with
its test); PPCA's own code does not call the VDAM projector. Quality, scored on cand_04e38a9 (this
stack plus bc30d20, the per-class device coarse significance that follows it on main; the same code
before the rebase): pdb K=2 5k/128
population-weighted GT FSC-AUC 0.3204 / 0.3883 / 0.4485 at seeds 29 / 41 / 53, against RELION
0.3229-0.3354 (7 runs) / 0.3797 / 0.4528-0.4575 and the same-job pre-stack control 0.3110 / 0.1360 /
0.4410; that is 0.0025 / 0.0043 below RELION's seed-29 / seed-53 range, inside relax's seed-29 basin
spread (0.3161-0.3381). EMPIAR-10097 0.2248 / 0.3565 (unmasked / masked) inside RELION seed-41 runs
0.2225-0.2254 / 0.3550-0.3588. Evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_vdamspeed_20260924/scoring` (jobs 14427596, 14427597).

VDAM on the resident engine (2026-09-25; K>1 default 2026-09-26, K=1 default 2026-09-26): the adaptive
route runs the InitialModel E-step on auto-refine's adaptive route (`relax/vdam/adaptive_estep.py`) with
the device-resident pass 2. `--pass2_engine auto` (the default) selects it for every K
(`relax.vdam.dense_adapter.vdam_pass2_route`); a configuration the adaptive route
refuses before device work runs exact-local with a logged reason, `adaptive` makes the refusal an
error, and each iteration's `run_itNNN_recovar_meta.json` records `pass2_engine` and `pass2_engines`.
Gate (200 iterations, relax 47103e5, one H100 + 8 CPUs per arm, uncapped; GT FSC-AUC against the
same-seed RELION runs, `/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_kclass_20260925/vdam_gate/GATE_SUMMARY.json`):

| case | RELION band | exact-local | resident | walls RELION / local / resident (s) |
|---|---|---|---|---|
| pdb K2 s29 | 0.20307-0.20755 | 0.20480 | 0.20706 | 883 / 2665 / 1682 |
| pdb K2 s41 | 0.22357-0.22536 | 0.22500 | 0.22650 | 962 / 2537 / 1518 |
| pdb K2 s53 | 0.24476-0.24661 | 0.24450 | 0.24677 | 966 / 2548 / 1519 |
| pdb K4 s29 | 0.07542-0.07554 | 0.07439 | 0.07577 | 773* / 1932 / 1750 |
| pdb K4 s41 | 0.12389-0.12641 | 0.12733 | 0.12141 | 1087* / 2502 / 1933 |
| pdb K4 s53 | 0.07777-0.07800 | 0.07860 | 0.07757 | 794* / 2087 / 1995 |
| noise1 50k K1 s29 | 0.34125-0.34141 | 0.34134 | 0.34134 | 1565 / 2232 / 2382 |

(* bench's RELION jobs 14445855-7 on other nodes.) Resident is 0.60-0.96x the exact-local wall at K>1 and
1.07x at K=1 at 47103e5; K=1 moved to resident on 2026-09-26 (below). K4 needs more RELION seeds before any
quality claim: single seeds scatter about ±2.5e-3 around RELION's two same-seed runs, for both routes.
RELION's three `--grad` E-step differences map onto it: the residual backprojection
(`mstep_subtract_ctf_projection`, now on resident), the pseudo-halfset BPref slots (one resident pass
whose accumulator slot is `class + K * pseudo-halfset`, `docs/development/resident_segments.md`) and the
coarse-only `maximum_significants = 100 K`. K>1 runs through the same pass (2026-09-25; pdb K2 seed 29
iterations 1-12 match the exact-local route to 7.9e-6 in the maps with identical classes and angles,
job 14444404). The exact-local VDAM route was removed on 2026-09-27; resident is VDAM's only route.
Evidence: `/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_vdamres_20260925/HANDOFF.json`.
Quality on noise1 50k/256 seed 29 (200 iterations, job 14422222, scored 14425217): relax resident
GT FSC-AUC 0.34131 unmasked / 0.75598 masked, inside four RELION runs 0.34125-0.34141 / 0.75515-0.75603;
relax-RELION pair FSC-AUC 0.971-0.984 against RELION-RELION 0.977-0.983. Speed gate (manager item 11,
not met): wall 4549 s against RELION 1451 s on the same node, slower than the exact-local default; the
E-step is 90% of it and grows from 9 s to 31 s per iteration. The resident route stays opt-in and the
exact-local route stays until resident is no slower at equal quality on 10097 and noise1 50k.
EMPIAR-10097 seed 41 (200 iterations, job 14425100 at the cache fix cbd7b0b, scored 14434198 against the
RELION auto-refine map in the frozen mask): relax resident FSC-AUC 0.2248 unmasked / 0.3578 masked, at the top
of four seed-41 RELION runs 0.2225-0.2243 / 0.3550-0.3573; relax-RELION pair FSC-AUC 0.701-0.732 against
RELION-RELION 0.688-0.773. Wall 6954 s against RELION 3025 s on the same node (2.3x); the exact-local default
measured 6835 s in a separate job (vdamspeed j14, not a matched pair), so the 10097 speed gate is not shown
met. The run predates the image-capacity change (834b3b3), which removes the per-subset re-trace.
K=1 default flip (2026-09-26): with stable Fourier windows and S2 (relax 5802a5e) resident is faster
than exact-local at K=1: noise1 50k s29 resident 1303 s, 0.92x its same-node RELION (job 14502014), where
exact-local took 1437 s against RELION 1421 s (job 14497728); EMPIAR-10097 s41 on one node 2469 s against
exact-local 4274 s and RELION 2957 s (job 14504146).
10097 quality (scored 14508334, `/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_vdamfast_20260926/e10097_route_1dafaae/scores`):
resident GT FSC-AUC 0.2269 unmasked / 0.3611 masked, exact-local 0.2236 / 0.3552, five seed-41 RELION
runs 0.2225-0.2259 / 0.3550-0.3605. `--pass2_engine auto` now selects resident for every K.

VDAM coarse scorer, native vs GEMM (2026-09-26, relax main bc6d3e1; same node, one H100 + 8 CPUs per
arm, uncapped). The default exact-operand coarse pass is a real-packed float32 GEMM pair
(`relax/scoring/scoring.py::_relion_coarse_gaussian_gemm_scores_jit`); the native arm is RELION's fused
per-pair projector/diff2 kernel (`RECOVAR_COARSE_GAUSSIAN_GEMM_MACRO=0`, and for the K>1 resident route
also `RECOVAR_K1_COARSE_GAUSSIAN_FFI=1 RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF=1
RECOVAR_K1_RELION_EXACT_COARSE_OPERANDS=1 RECOVAR_K1_COARSE_FUSED_PROJECTOR=1`, because K>1 resident
otherwise scores pass 1 with the generic dense GEMM):

| case, route | RELION | native coarse | GEMM coarse (default) |
|---|---|---|---|
| noise1 50k K1 s29, exact-local | 1422 s | 2238 s (1.57x) | 1592 s (1.12x) |
| pdb K2 s29, resident | 993 s | 2181 s (2.20x) | 1249 s (1.26x) |

GEMM quality (relax 12fe042 gate, `/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_vdamfast_20260926/gate_s1_12fe042/scores`):
mean GT FSC-AUC inside or above RELION's same-seed runs for pdb K2 s29/s41/s53 and K4 s29, noise1 masked
0.7554 (exact-local) and 0.7547/0.7555 (resident repeat) against RELION 0.7552-0.7560. Float32 against
binary64 GEMM scores differ by at most 0.02 (about 20 ulp) near the best pose. The late iterations are
already faster than RELION on the GEMM path (noise1 its 151-200: 544 s against about 766 s); the remaining
gap is early and middle iterations, where XLA compilation is 46-70% of main-thread time
(whole-run py-spy, `/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_vdamfast_20260926/e11_latepy`,
`e15_localpy`). Evidence: gates 14474812 and 14475131 in `.../relax_vdamfast_20260926/gate_main_bc6d3e1`.

OPEN (compact, not fixed: the compact engine is to be deleted): without scale-correction groups the
compact sparse pass 2 takes its non-atomic noise arithmetic, 19% apart in `wsum_sigma2_noise` from
RELION's scale-1 Wavg triplet on the algebraic-Wavg test fixture (repro: `_vdam_args(residual=True,
groups=False)` in `tests/unit/test_resident_vdam_estep.py`, compact against resident, job 14421464).
The resident engine runs RELION's triplet at scale 1 (relax 40b14ac).

Class3D local searches, K>1 (2026-09-25): the per-class local route now keeps
RELION's joint per-particle pass-2 support (3601775); it matches the
class-segmented pass to 1e-7. No benchmark reaches this route (Class3D keeps
HEALPix 1-2, below auto_local_healpix_order 4). Later engine task: give the
class-segmented pass padding factor 2 and a per-class scale-correction mask
(RELION's `data_vs_prior_class[iclass] > 3`), route Class3D local K>1 through
it, then delete the per-class K>1 loop (one implementation; K=1 keeps its path).
Update 2026-09-25: Class3D no longer reaches local searches at all. RELION switches
to local searches from the HEALPix order only under auto-refine: the iteration-0
switch is inside `if (do_auto_refine)` (ml_optimiser.cpp:2302-2326) and later
switches come from `updateAngularSampling`, which Class3D never calls
(ml_optimiser.cpp:3550-3552); relax has no `--sigma_ang`. relax switched Class3D
to local searches at HEALPix >= 4 (fixed 7381f84, replays 016e261), and the
Class3D local K>1 route (the K>1 branch of `_run_local_search_iteration` and the
class arms of the local half scorer) is deleted as unreachable. The per-class
and segmented K-class local passes (`run_local_k_class_em`) were removed with
the VDAM exact-local route on 2026-09-27.

Class3D noise statistics, compact K>1 (2026-09-25): OPEN against the compact
K-class engine (`compute_k_class_pass2_stats_sparse_fused`, today's default Class3D
pass 2). It has no RELION direct low-shell Wavg residual, so its
`wsum_sigma2_noise` is not RELION's, and on a small fixture one shell goes
negative. Repro (on the branch below): `tests/unit/test_resident_k_class_pass2.py::_k_class_args(3)`
(the 8x8 resident driver fixture with 3 classes). The class sum of the compact
noise tuples is -21.4 at shell 0, and 3835.6 against the resident 4604.7 at K=2.
Job 14421852 (candidate 91596ff) measured rel L2 0.24 at K=2 and 0.71 at K=3
against the resident K-class pass. The two agree on everything else to the
default band: evidence, per-class winners, joint Pmax, class mass and both BPrefs.
The resident K-class pass computes the noise with RELION's arithmetic
(`resident_pass2.compute_k_class_pass2_stats_resident`). Its noise is pinned by the
duplicated-class test against the K=1 resident pass. The compact K>1 engine is not
fixed; it is deleted with compact.

Class3D on the resident engine (2026-09-25): the K-class pass 2 runs on the resident
engine (the only pass-2 engine since the compact removal of 2026-09-26) and records its engine. Rows carry the class axis
(`docs/development/resident_segments.md`). K4 50k/256 long (15 iterations, candidate e2f5197,
job 14428127 against the RELION repeat of 14427006): GT min FSC-AUC 0.216797 (masked 0.216068),
inside RELION's band [0.216623, 0.216957] (masked [0.215882, 0.216170]); the compact long run
14410744 scores 0.216536, 8.7e-5 below it. K4 100k/256 completion (job 14426696, CPU-matched
pair): GT min FSC-AUC 0.26407 (masked 0.26795), above RELION's [0.26276, 0.26322] (masked
[0.26642, 0.26679]); wall 14620 s against RELION 4666 s.

RELION MPI scale-group pack/unpack defect (RELION 5.0.1 f2c1a38; relax issue #1): the pieced
`MlWsumModel::pack` / `unpack` that `relion_refine_mpi` uses to combine weighted sums in CUDA builds
(`src/ml_model.cpp:2056`, `:2241`) size the scale-group loop with `sigma2_noise.size()`, the number of
optics groups, so only scale group 1's `wsum_signal_product` / `wsum_reference_power` cross ranks. With
`--scale` and several followers per data set, every other group's scale correction comes from one
follower's particles; with one particle per group (the simulated fixtures) half the groups collapse to a
common value. relax implements the non-MPI behaviour. Affected: Class3D (no split halves) run with more
than one follower. Auto-refine with one follower per half-set (the 3-rank arms of every K=1 benchmark row)
skips the within-half combine, and its final join's scale update is not used for any reconstruction, so the
K=1 rows are unaffected (checked on the 10097 row: per-group scales vary normally in both halves). Evidence:
K4 5k/128 GUI Class3D, three seeds: relax equals non-MPI RELION (class agreement 1.0000 at all 25
iterations; GT class accuracy and FSC-AUC identical on seed 29), while MPI RELION splits from it at
iteration 3 (0.979-0.981) and ends in another solution (GT class accuracy 0.992 vs 0.976); two MPI repeats
also split at iteration 3 (0.981). A two-line patch (use `wsum_signal_product.size()`) makes
`relion_refine_mpi` equal to non-MPI RELION on that case (agreement 1.0000 at every iteration, scale
corrections to 4e-5; jobs 14456979, 14457083). Patched build, patch, verification and report draft:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/relion_patched_mpi_scale_20260926/` (run with
`SLURM_MPI_TYPE=pmix_v3`). Class3D benchmark references should be non-MPI RELION or this patched build.

K4 100k/256 class agreement (formerly OPEN; explained by the defect above): relax's class agreement with the
two MPI RELION runs was 0.878-0.894 at iteration 15 while they agree with each other at 0.9335. Non-MPI RELION
with the same command (job 14455196, 3643 s) agrees with those MPI runs at 0.8785-0.8954, the same level, and
with relax at 0.9780; map FSC-AUC relax vs non-MPI RELION 0.9696 against 0.842-0.857 for non-MPI vs MPI RELION.
GT min FSC-AUC: relax 0.26407 (masked 0.26795), non-MPI RELION 0.26182 (0.26522), MPI RELION 0.26276-0.26322
(0.26642-0.26679). Compact scores like resident (long run 14410744). The one-step replays are no longer needed.
Evidence: `/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_kclass_20260925/band_100k/BAND_nompi.json`,
`/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_kclass_20260925/k4_100k_relion_nompi/TRAJ_relax_vs_nompi.json`.

Final all-data maps are now always gridding-corrected, as in RELION; the former
default-off selector made the K1 100k/256 masked GT FSC 0.0008 lower than both
same-command RELION repeats over shells 1-60. Pinned merged-map records were
regenerated post hoc from the saved maps (scorecard v2). OPEN (small): with the
correction, relax's GT FSC-AUC sits just below both same-command RELION repeats
on two synthetic fixtures, by about as much as the repeats differ from each
other. K1 100k/256: masked 3.3e-5 / 5.2e-5 below rep2 / rep1 in the end-to-end
qualification run 14365794 (1.6-3.5e-5 post hoc on job 14320204, merged map and
each unfiltered half; repeats differ by 1.9e-5). K1 50k/256 noise 1 (aligned GT): relax
0.35122 vs RELION 0.35129 and its repeat 0.35127. More RELION repeats are needed
to tell a defect from run-to-run variation.

On the 10k EMPIAR-10097 fixture at 256 px on one H100, the resident K1 path
reduced the cold auto-refine gap from 3.26x to about 1.7-1.8x RELION, with a
further small gain from building the projector on device. This qualifies only
that fixture. Remaining cost is concentrated in new-shape transitions and the
final all-data iteration. The 100k/256 K1 and exactly-K4 quality and
performance gates remain open.

Final all-data pass on resident local (2026-09-27, 1178448..73c2a22). The full-box
pass routes to the resident local driver, whose M-step now walks only rows with
weight and translates unshifted per-image operands inside the translate-sum kernel.
Same-node 10097 10k gate (job 14518565, scored 14518568): final iteration 123.3 s
on the exact local engine, 92.6 s on resident; run walls main c423fb7 603 s,
candidate 595 s, RELION MPI 3x4 592 s; map gate passes against all five
same-command RELION runs (the candidate landed in the RELION basin of run a1,
cross-engine merged 0.9938). The fast parity tier (13/13) and the resident GPU unit
files (78 passed, job 14518566) pass. The NumPy reference test covers the box case
at noise 200; at noise 2 the resident Ft_y relative L2 is 1.18e-5 against the exact
engine's 2.3e-5 (job 14508921), above the test's 1e-5 bound, pending a decision on a
derived float32 bound.

Wall references (2026-09-26, user decision). The speed target is at most 0.5x RELION's cold
wall for one full-size run, as a same-node pair: K=1 on EMPIAR-10097 and the 50k/100k synthetic
fixtures, Class3D on K4 100k. Compile amortized over a run's iterations counts; warm-cache walls
are not quoted. 5k fixtures serve iteration and profiling only, since compile dominates them.
Quote each wall against the fastest RELION configuration that runs, and show both when both run.
Auto-refine (`--split_random_halves`) runs only under MPI (non-MPI exits at start), so K=1
references are `relion_refine_mpi` 3 ranks x 4 threads (K=1 5k/128: 196 and 200 s, jobs 14497840
and 14499533). Class3D runs both; non-MPI `--j 12` is faster (K4 5k/128: 268-271 s against MPI
382-390 s) and is also the quality reference (MPI scale-group defect above). The 268-271 s walls in
`tests/baselines/relion_vs_relax_benchmarks.json` rows `pdb_k4_5k_class3d_25it_*` are that Class3D
non-MPI reference and are labelled correctly; no non-MPI K=1 wall exists. Since ed7d5d7 the
refinement entry point activates the persistent compilation cache, so a cold arm needs its own
empty `JAX_COMPILATION_CACHE_DIR` (or `RECOVAR_DISABLE_JAX_CACHE=1`).

Explained (2026-09-24, test tiers): the medium K1 5k/128 standalone end-to-end run (medium
14375481, relax 319cd10) scored GT FSC-AUC 0.6084 on relax's `final_merged.mrc` against
0.5959-0.5960 for RELION's `run_class001.mrc`. The cause is the map kind. RELION's map is
gridding-corrected, and relax at 319cd10 did not apply a final gridding correction. Applying
RELION's pad-2 sinc^2 correction to relax's map gives 0.5959, inside the band. The unfiltered half
maps agree: average GT FSC-AUC relax 0.59519, RELION 0.59516-0.59519. This is resolved by the
always-on final gridding correction (user decision; relax df88eab retires the option). The
tier gates both the unfiltered half-map average and the merged map, and requires
`final_all_data_grid_correct` to be recorded True in `refinement_results.npz`.

K=1 engine (2026-09-25): the device-resident pass 2, local search and device significance are the
Refine3D default, with the qualified flag set, including the first-iteration CC pass and
`--adaptive_oversampling 0`. A global pass the resident checks refuse is an error (the compact
engine is deleted), and so is a local fine pass they refuse (2026-09-27); only the pass-1 parent
probe still runs on the deprecated exact-local engine, with a logged reason recorded per iteration
in `pass2_engine_trajectory`. The jitted stage glue and the local image-capacity
ladder are set at the K=1 entry points (`apply_k1_refine3d_env_defaults`), since VDAM shares that
code and qualifies its own defaults. Transitional A/B off switches (not permanent variants;
`RELAX_SPARSE_PASS2_RESIDENT` and `RELAX_LOCAL_SEARCH_RESIDENT` are retired and setting either is an
error): `RELAX_EM_PROTOTYPE_SOFT_POSTERIOR_BLOCK_BPREF=0`,
`RELAX_K1_RELION_WAVG_SEQUENTIAL_CUDA=0`, `RELAX_COARSE_PAD_FINAL_IMAGE_BATCH=0`,
`RELAX_EM_JIT_STAGE_GLUE=0`, `RELAX_LOCAL_IMAGE_CAPACITY_LADDER=0`. Device coarse significance has no
switch: every class's coarse support is compacted on the device whenever the ids are collected
(K>1 and VDAM too); score dumps and the compact-hybrid diagnostics keep the host mask
(`relax/sparse_pass2/resident_significance.py`). Flip pairs (relax vs
RELION wall, same node): EMPIAR-10073 1.05x and 10345 1.01x pass the scorecard thresholds; K1
50k/256 runs at 0.65x, with the resident masked GT FSC 9e-5 below the three-run RELION band
(compact inside it). OPEN (small, both engines): on 10345 and K1 50k/256 relax's map agreement with RELION
sits just below RELION's run-to-run band with equal own quality (10345 merged band FSC-AUC
0.982 vs 0.986-0.988, masked 0.9980 vs 0.9985-0.9987; 50k 0.9994 vs 0.9997). Compact shows the
same gap, and RELION native FFT units did not change it. Per-step attribution (measured, K1 50k/256): one-step replays from
the uninterrupted RELION run's states, scoring each half with its own sigma2_noise, deviate from
RELION's next iteration by 4.0e-5 / 9.5e-6 (half 1 / half 2) at it2, 3.2e-5 / 4.7e-6 at it5,
1.5e-4 / 1.8e-4 at it9 and 5.9e-5 / 1.1e-4 at it13 (half-map rel L2 inside the frozen mask; resident
and compact agree to the digits shown except it13, where resident sits closer, mean abs dPmax 3.5e-4 vs 6.6e-4).
RELION's own CPU path, stepped from the same stored states, differs from its GPU path by 3.6e-4 /
3.1e-4 at it2 and 1.4e-4 / 1.1e-4 at it13, while two GPU repeats agree to 3e-7-5e-6. relax's per-step
deviation is therefore at or inside RELION's CPU/GPU arithmetic band (1.0-33x smaller; equal at
it13 half 2). End to end, such
arithmetic-path differences compound into a larger map disagreement than relax's (measured on the K1
5k/128 os0 fixture, auto-refine to convergence plus the final all-data iteration; cross FSC-AUC of the
unfiltered half-map average / merged map against the RELION GPU oracle): RELION's CPU path 0.99772 /
0.99719 (job 14426055, converged at iteration 14 like the GPU runs); relax main b0276ba 0.99990 /
0.99986 (two identical arms 14446428, 1.000000 / 0.999999 to each other); RELION GPU same-command
repeats 0.99999 / 0.99998 and 0.99900 / 0.99814. relax therefore sits inside RELION's GPU repeat band
and about 20x closer to the GPU oracle than RELION's own CPU path. The 50k/256 full-run gap (0.9994 vs
0.9997) has no CPU-path end-to-end run; it is attributed by the same mechanism, not measured at 50k.
Scores: `/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_landing_20260925/score_5k_all.json`.
The earlier attribution (1.3-4.2e-4 per step, "about 2.5x RELION's drift") is superseded: the replay
harness then scored half 2 with half 1's sigma2_noise (the RELION MPI restart broadcast); those
replays (relax a7977c8) read 4.3e-4-7.6e-4 for half 2 at it2/it5, against the per-half-noise values
above (relax 64b08ef, whose half 1 also moved from 6.6e-5 to 4.0e-5 at it2). Replays with per-half noise: jobs 14425333 (relax), 14411352 (RELION CPU),
under `/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_speed_20260923/replay_50k_iters_cns_20260925`
and `replay_50k_iters_20260925/relion_cont`.

Tested, no effect (2026-09-25, not landed): RELION's CUDA path normalizes each real image by
`(XFLOAT)(avg_norm_correction / normcorr)` (`acc/acc_ml_optimiser_impl.h:432`, f2c1a38). relax
recovers that factor as `float32(combined) / float32(group_scale)`. On the K1 50k/256 fixture this
differs by 1-2 ULP (1.2e-7 relative) for 34% of images at it2 and it13. A port of a parallel
session's fix carried the once-rounded host ratio separately. It moved relax itself by a mean
|dPmax| of 2e-5, but not toward RELION. One-step replays from the uninterrupted RELION run's state
(job 14421433, main a1786bf, H100) were compared with RELION's next iteration of that run:

| Replay | Map rel L2, control -> fix | Mean abs dPmax, control -> fix |
| --- | --- | --- |
| it2, resident | 2.49566e-4 -> 2.49566e-4 | 5.975e-4 -> 5.975e-4 |
| it2, compact | 2.49567e-4 -> 2.49566e-4 | 5.975e-4 -> 5.975e-4 |
| it13, resident | 1.25412e-4 -> 1.25457e-4 | 9.834e-4 -> 9.830e-4 |
| it13, compact | 1.34754e-4 -> 1.34758e-4 | 1.1648e-3 -> 1.1648e-3 |

The table's metrics are the half-map rel L2 inside the frozen mask and per-particle Pmax, both
against RELION's next iteration. The no-effect conclusion holds (both arms share the setup), but the
absolute gaps are superseded: these replays scored half 2 with half 1's sigma2_noise (the harness's
former restart-broadcast default), which inflates half 2 against the uninterrupted run; see the K=1
engine paragraph above for the per-half numbers. Observation: the it13 compact
arm's per-particle Pmax was identical between control and fix, so that route may not consume the
carried factor. Evidence: `/scratch/gpfs/CRYOEM/gilleslab/em_work/relax_normfix_20260925/ab50k_it2_it13/`
(`PERSTEP_DEVIATION.json`, `PMAX_AB.json`).

Map sign convention: every map relax writes is in RELION's map convention, so a relax map and
the RELION map of the same run agree voxel for voxel and in sign (`relax/helpers/map_io.py`).
relax holds volumes internally in RECOVAR's frame, the negated transpose of RELION's file array
(`relion_volume_to_recovar`); the sign entered at the map-file boundary, where the refinement,
the parity harness and the per-iteration dumps wrote with RECOVAR's `write_mrc` and the K1
refinement read its reference with `load_mrc`. Maps are now written with `write_map` (labelled
`relax map, RELION sign and axis convention`) and references are read with `load_relion_volume`:
`--init_volume` and `--init_class_volumes` take the same file as relion_refine's `--ref`, and the
data-directory defaults are `reference_init_relion.mrc` and `reference_init_class00K_relion.mrc`.
Read a relax map with `load_relax_map`; a map written before this convention has no label and
holds the negated array, and the scorers read it as such (`legacy_recovar_sign=True`).
Ground-truth `reference_gt*.mrc` files stay in RECOVAR's frame and are read with `load_mrc`.

Follow the unchanged [quantitative gates](../math/em_parity_program.md) and
[validation ladder](em_parity_runbook.md#validation-ladder): matched-state
scores, support, posteriors, poses and accumulators; synthetic then real K1;
then exactly K4 with Hungarian matching and per-class FSC/FSC-AUC. Completion
requires production float32, matched inputs/seeds/maps/masks, matching
convergence/finalization and at least 100,000 particles at 256x256 or larger on
matched GPU classes. Correlation, diagnostic double, partial iterations and
missing measurements do not satisfy those gates. Scientific acceptance precedes
speed qualification.

Use frozen pixi environments, Slurm for integration and long GPU work, and
sealed identified native libraries. Follow the local GPU0 reservation, the
[EM contract](../../relax/AGENTS.md), [benchmark contract](benchmarks.md)
and [agent workflow](agent_workflow.md). Repeat checks only for changed behavior,
failures, unresolved concerns or required qualification.
