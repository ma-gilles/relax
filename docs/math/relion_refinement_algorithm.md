# RELION-style refinement: algorithm and code map

## Start-up noise

Every refinement starts from RELION's start-up noise estimate from the images;
`scripts/run_full_refinement.py` has no other estimator. It computes only the
initial noise from the particles: up to 1000 particles per optics group and
rounded-radius half-spectrum shell power. K1 takes the stable subset-1 then
subset-2 source order of its half sets. Class3D (K>1) splits no halves, so
RELION's `sorted_idx` is the micrograph-sorted input order itself
([`_relion_class3d_initial_noise_layout`](../../scripts/run_full_refinement.py),
`ml_optimiser.cpp:2804-2808`, `exp_model.cpp:900-901`). It does not load oracle
tau2, poses, priors or normalization corrections. On the K4 50k/256 and 5k/128
fixtures the Class3D spectrum agrees with RELION's `run_it000_model.star` to its
six-digit serialization (maximum relative difference 3.4e-7); the former RECOVAR
pipeline estimator, removed from relax refinement on 2026-09-24, differed by up
to 8.4%.

Three debug inputs replace it: a frozen boundary's sealed noise, the
diagnostic `--init_noise_from_npz`, and a `--relion_init_dir` start, which
loads RELION's `run_it000` model noise. A `--perturb_replay_relion_dir` replay
starts from the estimate and injects RELION's model noise from its first loaded
state on. K1 without half sets and multiple optics groups are rejected.

[`_compute_relion_startup_noise`](../../scripts/run_full_refinement.py)
uses the host float64 estimate, scales its native sigma2 by the image side
length to the fourth power, and supplies a float32 pixel-noise array to the
controller (float64 for double-scoring diagnostics). Tests in
[`test_relion_startup_noise.py`](../../tests/unit/test_relion_startup_noise.py)
check ordering, units/dtype and mode exclusions. The standalone K1 start was
qualified end to end on the K1 50k/256 fixture (see the standalone start-up
section below).

## Class3D standalone start-up

A fresh Class3D (K>1) run starts from what `relion_refine` reads, without any
RELION output; this standalone start is the Class3D default (launch recipe and
qualification in the [runbook](../development/em_parity_runbook.md#standalone-class3d-launch)).
The pieces:

- Startup noise: RELION's estimate, always (section above).
- Input origins: the default `--initial-pose-source auto` (or an explicit
  `input-star`) loads only the input `rlnOriginX/YAngst` (or pixel origins;
  absent origins are zero) in the all-data particle order
  ([`_load_input_star_class3d_translations`](../../relax/relion/input_poses.py)).
  RELION rounds and applies them before the image FFT but does not centre the
  first global search on the input angles. The `--relion_init_dir` route
  (`_kclass_firstiter_translation_seed`, from `run_it000_data.star`) is kept
  for debug comparison; on the K4 50k/256 fixture the two are bitwise equal, and
  on EMPIAR-10097 (non-zero origins) the input origins equal `run_it000`'s.
- References and class distribution: `--ref_star` reads relion_refine's `--ref`
  STAR ([`read_relion_reference_star`](../../relax/relion/relion_metadata.py));
  its maps are RELION-frame and load with `load_relion_volume`. A fresh run
  starts from `pdf_class = 1/K` whatever `_rlnClassDistribution` says
  (`MlModel::initialise`, `ml_model.cpp:53`; `initialiseFromImages` reads only
  `_rlnReferenceImage`). A RELION 5.0.1 `--iter 0` run on the K4 5k/128
  fixture with a 0.7/0.1/0.1/0.1 `--ref` STAR wrote 0.25 for every class in
  `run_it000_model.star`.
- Expected-accuracy trials (always on for a fresh Class3D run without half
  sets): RELION shuffles the whole micrograph-sorted vector once with
  `std::shuffle(mt19937(random_seed + iter))`, stable-sorts it by optics group
  (`exp_model.cpp:449-456`) and takes the first 100 entries; each trial's
  `part_id` seeds its draws
  ([`relion_class3d_trial_layout`](../../relax/helpers/expected_accuracy.py)).
  The estimator divides by `sigma2_fudge * sigma2_noise` with RELION's default
  `sigma2_fudge = 1` (`ml_optimiser.cpp:1069, 9291`), never by `tau2_fudge`.
  With RELION's iteration-1 state this reproduces `run_it002`'s per-class
  `rlnAccuracyRotations` to 1e-14 on the K4 5k/128 and 50k/256 fixtures.
- Group scales: a standalone run is single-process, like a non-MPI
  `relion_refine`, so it keeps one group-scale state and needs no dispatch
  schedule (`--relion-scale-followers` resolves to 0 without
  `--relion_init_dir`/`--perturb_replay_relion_dir`). Emulating
  `relion_refine_mpi`'s follower-local scales
  ([`relion_worker_scale`](../../relax/relion/relion_worker_scale.py)) needs a
  captured dispatch schedule and stays a debug replay tool.
- No RELION output: without `--relion_optimiser`, `--relion_init_dir`,
  `--perturb_replay_relion_dir` or `--relion_half_sets`, a Class3D run does not
  pick up a RELION optimiser STAR found next to the data
  ([`_find_relion_optimiser_star`](../../scripts/run_full_refinement.py)). The
  mask, `ini_high` and `max_significants` then come from `--particle_diameter_ang`,
  `--apply-initial-lowpass --init_resolution` and relion_refine's `--maxsig -1`.
- Fixed schedule: Class3D keeps `--healpix_order` and runs every `--iter`
  iteration. relion_refine calls `updateAngularSampling` only under auto-refine
  or auto-sampling and `checkConvergence` only under auto-refine
  (`ml_optimiser.cpp:3308-3313, 3550-3552`), so K>1 skips both
  ([`_relion_auto_refine_transitions`](../../relax/refinement/iteration_loop.py));
  its expected accuracy is written to the history but never gates. Before this,
  a K>1 run could latch fine-enough sampling once the overall accuracy exceeded
  4/3 of the order-1 step, then stop early on convergence and run an all-data
  pass RELION Class3D never runs.

This page describes RECOVAR's current dense-volume refinement implementation,
including its K-class and exact local-search routes. Function names identify
implementation owners; line numbers are deliberately omitted because code moves
as it is cleaned up. The [codebase map](../development/codebase.md) covers the
rest of RECOVAR.

This is an implementation guide, not a claim of complete RELION parity.
[Current EM status](../development/em_status.md) records accepted checks,
failed comparisons and pending qualification. Historical RELION source locations
in code comments refer to the source used for those comparisons.

## 1. Controller and state

[`refine_single_volume`](../../relax/refinement/iteration_loop.py)
accepts two half-set datasets, initial Fourier volumes, noise and signal priors,
and refinement settings. Despite its historical name, it supports `n_classes > 1`.
[`RefinementOptions`](../../relax/refinement/refinement_options.py)
groups the settings; supplied option fields override the corresponding individual
arguments.

The controller, `refine_single_volume`, coordinates:

1. Sampling and reference preparation for the current iteration.
2. Scoring each half through `_score_half_dense` or `_score_half_local`.
3. Reconstruction, low-frequency accumulator joining, and updates to noise,
   normalization, scale and translation-prior statistics.
4. Resolution and hidden-variable tracking for the next sampling decision.
5. An optional final all-data expectation and reconstruction.

The order of individual updates within these stages matters for trajectory
comparisons. Replay/oracle inputs can replace selected state boundaries;
results from those modes must remain distinguishable from autonomous refinement.
Their implementation belongs to
[`relion_replay.py`](../../relax/diagnostics/relion_replay.py).

[`score_outputs.py`](../../relax/dense/score_outputs.py)
defines the controller's scoring payloads:

- `HalfScoreResult` holds one half's accumulators, assignments and statistics.
- `PerHalfOutputs` holds two half-set slots per field. Class axes live inside
  those slots; a class axis is not a half-set axis. The halves may contain
  different numbers of images.

These containers store existing references. They do not copy arrays, normalize
precision or release device storage. The controller owns buffer lifetime.
Accumulator shape and half-spectrum axis metadata must travel with the arrays.

### Cold-start K1 translation prior

The global K1 controller supplies an explicit zero score-prior center when no
previous offsets are available. Zero initial offsets do not imply a flat prior:
the native accelerated `pdf_offset` construction still applies its Gaussian.
For unperturbed translations `t` in pixels, voxel size `a` in Angstrom/pixel and
model offset sigma `s` in Angstrom, the source-equivalent log prior is
`-0.5 * ||t||^2 * a^4 / s^2`, up to a candidate-independent constant. Native
sampling stores Angstrom translations and its accelerated code applies another
pixel-size-squared factor. This documents source parity, not a proposed unit
convention. Perturbation affects scoring translations, not this base-grid prior.

[`make_relion_translation_log_prior`](../../relax/helpers/orientation_priors.py)
implements the formula; its explicit `None` center still requests a flat prior.
The K1 correction is at the regular global controller call site, not a blanket
change to shared helper, local, K-class or VDAM semantics. The controller wiring
is tested by `test_k1_coldstart_supplies_gaussian_translation_prior` in
[`test_refine_relion_mode.py`](../../tests/unit/test_refine_relion_mode.py).
Trajectory/FSC qualification remains separate from the fixed-input regression.

### Standalone K1 start-up state

A fresh K1 start builds RELION's iteration-0 model from `relion_refine`'s own
inputs rather than from a RELION `run_it000` output. This is the default when
the run is given no RELION output (no `--relion_half_sets`, `--relion_init_dir`,
replay directory or frozen boundary):
[`_resolve_standalone_k1_start`](../../scripts/run_full_refinement.py) turns on
`--relion-half-sets-from-input`. The mt19937 order and RELION's start-up noise
are the defaults of every run. A run given RELION output is a debug start and
must supply `--relion_half_sets`; relax has no other K1 half split.

- Norm corrections. `relion_refine` reads each particle's `rlnNormCorrection`
  and uses 1 when the label is absent; `avg_norm_correction` starts at 1. The
  E-step multiplies each image by `avg_norm / normcorr`, so the start-up image
  correction is `1 / normcorr` with unit group scales.
  [`_load_input_star_previous_best_poses`](../../relax/relion/input_poses.py)
  reads the column with the input poses in half-local order and
  [`_initial_corrections_from_norm`](../../relax/relion/input_poses.py) forms the
  corrections; unit norms keep the unit (`None`) representation.
- Signal prior and data-vs-prior. `MlModel::initialiseDataVersusPrior` sets
  `tau2 = tau2_fudge * P(Iref) * N^2 / 2`, with `P` the rounded-radius
  half-spectrum power of the low-passed start-up reference, and
  `data_vs_prior = n_half * tau2 / (sigma2 * 2 i)` (shell `i > 0`), with the
  half set's particle count and the initial noise.
  [`relion_initial_tau2_and_data_vs_prior`](../../relax/vdam/init.py) implements
  one class and is shared with the InitialModel start;
  [`_relion_k1_start_tau2_and_data_vs_prior`](../../scripts/run_full_refinement.py)
  applies it to every fresh K1 start that does not load a noise or tau2 state
  (frozen boundary, `--init_noise_from_npz`, `--relion_init_dir`). The start-up `data_vs_prior` selects the iteration-1
  scale-correction shells (`data_vs_prior > 3`), which matters for starts
  without `--firstiter_cc`; later iterations use the updated spectrum. Like
  those later iterations, the controller uses half 1's spectrum for both halves.
- Offset prior. `--offset_sigma_angstrom` defaults to RELION's `--offset` 10 A.

`test_relion_start_tau2.py` checks the formula and RELION's `run_it000` model
on the 5k and 50k K1 fixtures; `test_run_full_refinement_input_pose_seed.py`
checks the norm column.

Qualification of the standalone default (K1 50k/256, relax fc670f0, Slurm
14339756/14339757 with RELION's seed and 14339758 with a second seed; the code
of the default flip, before the final gridding correction). It passes the
functional rule: same resolution, GT not worse, standalone with its own draws.
All three arms converge in 14 iterations (RELION 14, 14, 15). The unmasked
half-map resolution at 0.143 equals RELION's with the same seed (11.83 A, and
11.57 A for the second seed); the masked postprocess resolution is 10.88 A for
every relax and RELION run. Unmasked GT FSC-AUC is 0.2393/0.2393/0.2411 against
RELION's 0.2384/0.2384/0.2342. Masked GT band FSC-AUC (frozen mask
`noise1_k1_50k256_c1`, shells 1-45) is 0.679021 against RELION's same-seed
0.679357/0.679359; after RELION's griddingCorrect applied post hoc it is
0.679394, so the 3.4e-4 shortfall is the missing final gridding correction
(fixed separately on branch `fix/final-gridding-always-on`). Same-seed map
agreement with RELION is FSC-AUC 0.998 (RELION's repeat 0.99977; a different
seed 0.899). Evidence: `em_work/relax_defaults_20260924/qual50k_accept/`.

## 2. Sampling grids and units

[`sampling.py`](../../relax/sampling.py) owns rotation and translation grids,
Euler conversions, oversampled children and perturbations. For the full C1 grid,
`rotation_grid_n_in_planes` and `rotation_grid_size` give

```text
n_directions = 12 * 4**order
n_psi        = 6 * 2**order
n_rotations  = n_directions * n_psi
angular_step = 360 / (6 * 2**order) degrees
```

| Base order | Directions | Psi steps | Rotations | Angular step |
| --- | ---: | ---: | ---: | ---: |
| 1 | 48 | 12 | 576 | 30° |
| 2 | 192 | 24 | 4,608 | 15° |
| 3 | 768 | 48 | 36,864 | 7.5° |
| 4 | 3,072 | 96 | 294,912 | 3.75° |

`get_relion_rotation_grid` constructs the grid from the RELION binding.
RECOVAR's grid indexing is psi-slow and direction-fast. Preserve index order
when comparing hard assignments; equal sets of rotations are insufficient.
K1/K4 coarse significance forwards the explicit projection precision option
before uploading the supplied projector slab. Production selects complex64;
double scoring or projection diagnostics retain complex128. The original host
setup array is preserved. Direct significance callers that omit the projection
option retain their legacy input precision. A float32 score array alone does
not establish float32 projection arithmetic.

Numbered replay retains the source sampling order of a learned direction prior
in [`apply_iter_replay_overrides`](../../relax/diagnostics/relion_replay.py).
When that order differs from the scoring grid,
[`relion_direction_log_priors_for_half`](../../relax/helpers/orientation_priors.py)
uses a uniform prior, matching RELION's `updateAngularSampling` reset. Remapping
the old distribution and labeling it with the new order would incorrectly
preserve learned directional preferences after a grid change. The file and
explicit-override routes share this rule for K1 and K4.

The source Euler and matrix precision can also matter at score ties.

For oversampling level `s`, `get_oversampled_rotation_grid_from_samples`
generates `4**s` direction children and `2**s` psi children per rotation parent:
`8**s` rotations. `get_oversampled_translation_grid` generates `4**s` children
per 2D translation parent. At `s=1`, one rotation/translation parent therefore
has up to **8 × 4 = 32 pose children**, before support restrictions. Here `s` is
an oversampling level; `K` below is the number of classes.

Translation grids and stored offsets use pixels unless an argument explicitly
names Angstroms. Local prior sigmas are stored in radians; angular sampling
and RELION Euler metadata use degrees. `relion_angular_sampling_deg` includes
oversampling when requested.

`advance_relion_perturbation` and the seed/replay helpers determine the scalar
sampling perturbation. `apply_relion_rotation_perturbation` and
`apply_relion_translation_perturbation` apply it to their respective grids.
Preserve the seed, iteration number, unperturbed grid and perturbation together
when reproducing a scoring boundary.

## 3. Gaussian scoring and posterior normalization

For image `i`, class `c`, rotation `r` and translation `t`, write the predicted
Fourier image as `a = S_t C_i P_r mu_c`. Ignoring constants common to all
hypotheses for that image, the Gaussian score is

```text
residual = sum_k w[k] * |y_i[k] - a[k]|² / sigma²_i[k]
         = data + cross + norm
cross    = -2 * sum_k w[k] * Re(conj(y_i[k]) * a[k]) / sigma²_i[k]
norm     = sum_k w[k] * |a[k]|² / sigma²_i[k]
score    = -0.5 * (cross + norm)
```

The image-only `data` term cancels when normalizing pose probabilities within
a common scoring convention. Evidence comparisons and external class
normalizers must account for the same omitted offset. Image normalization,
group scale, CTF and noise factors are applied by the preprocessing/scoring
path; the formula above is schematic about where those factors are stored.

Let `log_prior` include the applicable class, direction and translation priors.
The full posterior is

```text
log_weight[i,c,r,t] = score[i,c,r,t] + log_prior[i,c,r,t]
log_Z[i]           = logsumexp over all allowed (c,r,t) of log_weight[i,c,r,t]
gamma[i,c,r,t]     = exp(log_weight[i,c,r,t] - log_Z[i])
```

Priors belong inside the normalization. In K-class EM, independently
normalizing each class's pose distribution would discard class probabilities.
Support pruning and first-iteration winner selection are additional policies;
retained M-step mass need not equal the full posterior mass.

The implementation owners are:

| Work | Owner |
| --- | --- |
| Image/CTF/noise preparation and translation phases | [`preprocessing.py`](../../relax/helpers/preprocessing.py), `preprocess_batch` and `preprocess_batch_firstiter_cc` |
| Projection and projection-dependent residual statistics | [`projection.py`](../../relax/helpers/projection.py), `compute_projections_block` and `compute_relion_projector_projections_block` |
| Gaussian and normalized-CC block scores | [`scoring.py`](../../relax/scoring/scoring.py), `_score_rotation_block` and `_e_step_block_scores_windowed` |
| Priors, candidate masks and class/external-normalizer constraints | [`score_constraints.py`](../../relax/scoring/score_constraints.py), `DenseScoreConstraints` |
| Scoring weights for the selected Fourier convention | [`half_spectrum.py`](../../relax/helpers/half_spectrum.py), `make_scoring_half_image_weights` |

The half-image layout has `H * (W//2 + 1)` entries. RELION half-sum scoring and
Hermitian full-image inner-product weights are separate conventions. Gaussian
RELION scoring masks redundant centered `kx=0` rows; normalized-CC callers can
retain them. Changing these weights is a numerical change, not a missing
optimization to enable during cleanup.

RELION evaluates the float32 fine-pass `diff2` on unnormalised FFT
coefficients. RECOVAR's shifted image and projected reference carry an extra
`N²` and its score `corr_img` an extra `N⁻⁴`. The factors cancel over the reals,
and bit for bit when `N²` is a power of two, but not in float32 pixel products
otherwise. The fresh K=1 exact-Gaussian pass therefore scores native-unit
operands (condition `_relion_native_fine_units_enabled`): each complex operand
divided by `N²` in binary64 and rounded once
([`sparse_pass2_scoring.py`](../../relax/sparse_pass2/sparse_pass2_scoring.py),
`_relion_native_fine_units`), and RELION's own `corr_img` before its `N⁻⁴`
conversion, with the zero origin of `Minvsigma2`
(`_relion_native_score_corr_img`). The compact engine divides the translated
image; the device-resident driver divides the unshifted image, which its
kernel translates, and the reference rows of its score projections
([`resident_operands.py`](../../relax/sparse_pass2/resident_operands.py),
`prepare_resident_half_operands`;
[`resident_pass2.py`](../../relax/sparse_pass2/resident_pass2.py),
`_prepare_chunk_reconstruction_operands`). The two engines therefore agree to
rounding, not bit for bit. Reconstruction and noise operands keep RECOVAR units
in both. The exact local engine scores in RECOVAR units. The regressions are in
[`test_relion_native_fine_score_units.py`](../../tests/unit/test_relion_native_fine_score_units.py).

For bounded normalized-CC rescoring, the stored projector radius and the
current image radius are distinct. The native rescorer in
[`relion_scoring.cuh`](../../relax/cuda/relion_scoring.cuh),
`launch_relion_coarse_normalized_cc_native_texture_pairs_f32`, preserves the
model-sized texture but limits rotated frequency support to
`min(projector_max_r, current_size // 2)`. This follows RELION 5.0.1's
`AccProjectorKernel::makeKernel`; using the model radius alone admits extra
frequencies when first-iteration scoring uses a smaller window. The exact
pixel-count regression is
[`test_native_cc_rescore_limits_support_to_current_image`](../../tests/unit/test_normalized_cc_replay.py).

## 4. Dense, adaptive and local execution

There are two different uses of “two pass.” Keep them separate when profiling
or comparing intermediate results.

**Blockwise normalization within one grid.**
[`em_engine.run_em`](../../relax/dense/em_engine.py) processes
image batches and rotation blocks. Its first sweep collects normalization and
best-pose statistics; its second sweep recomputes scores for accumulation.
`_update_logsumexp` and `_merge_block_logsumexp` in `helpers/scoring.py` combine
block normalizers. An external `normalization_log_evidence` can normalize this
class against a joint class/pose distribution. `score_only` skips accumulation;
other options can skip negligible second-sweep blocks or use fused execution.
The complete image × rotation × translation score tensor is not required.

`run_em` returns `DenseEMResult`, defined in
[`helpers/types.py`](../../relax/helpers/types.py).
Read `mean`, `hard_assignments`, `Ft_y` and `Ft_ctf` by name. Optional `stats`,
`noise_stats` and `profile` fields are `None` when disabled; the corresponding
flags still control the same computations. The result container stores existing
array references. Callers no longer decode a different tuple layout for each
flag combination.

**Adaptive coarse-to-fine search.**
[`k_class.py`](../../relax/classification/k_class.py) owns
`run_dense_k_class_em` and `run_dense_k_class_em_adaptive`.
[`significance.py`](../../relax/scoring/significance.py)
computes joint coarse class/pose evidence and significant support, including
K=1 routed through the class-aware implementation.
[`oversampling.py`](../../relax/helpers/oversampling.py)
owns cumulative-mass selection and coarse/fine mappings. Significance selects
rotation/translation pairs; it is not simply an independent probability cutoff
on every orientation.

Fine execution can use dense or sparse routes.
[`sparse_pass2_bucketed.py`](../../relax/sparse_pass2/sparse_pass2_bucketed.py)
owns bucketed and compact-pair scoring, posterior reconstruction policies and
accumulation, including `compute_k_class_pass2_stats_sparse_fused`.
The support representation, execution buckets and float32 posterior policy
are part of the comparison contract. Preserving only final MAP assignments
does not establish equivalent soft M-step contributions.

**Exact local search.**
[`local_search_iteration.py`](../../relax/refinement/local_search_iteration.py)
constructs per-image neighborhoods, applies the batch budget and dispatches
`local_em_engine.run_local_em_exact` or `k_class.run_local_k_class_em`.
[`local_layout.py`](../../relax/local/local_layout.py)
builds the per-image hypothesis layout. This route does not use the retired
sort-and-split union helper formerly described on this page.

`sampling.get_local_rotation_grid_fast` implements C1 factored direction/psi
priors. Viewing directions use the third **row** of the RELION rotation matrix.
Direction and psi cutoffs use their respective sigmas; psi width must not widen
the direction cone. Neighborhoods follow previous best poses, while M-step
weights inside the neighborhood remain soft unless a winner-selection policy
is active. This limits exploration of separated modes; it does not prove that
a particle can never leave its initial neighborhood over later iterations.

**Fourier windows and performance.**
[`fourier_window.py`](../../relax/helpers/fourier_window.py)
defines `FourierWindowSpec` and the score/projection window mappings.
`current_size` is an image diameter in pixels. Window shape, pixel order and
redundant-axis treatment depend on the scoring route. Smaller windows reduce
operand sizes, but neither the layout size nor a GEMM formulation establishes
a fixed speedup. Use paired measurements under the
[benchmark contract](../development/benchmarks.md).

## 5. Accumulation, reconstruction and parameter updates

Conceptually, each class accumulates a weighted-image numerator and a
CTF/noise precision denominator:

```text
Ft_y[c]   = sum_(i,r,t) gamma[i,c,r,t] * P_r* (conj(S_t C_i) y_i / sigma²_i)
Ft_ctf[c] = sum_(i,r,t) gamma[i,c,r,t] * P_r* (|C_i|² / sigma²_i)
mu_c      ≈ Ft_y[c] / (Ft_ctf[c] + prior_precision[c])
```

These equations omit layout, interpolation, normalization and padding details.
`P_r*` inserts a 2D slice into the 3D accumulator. Dense accumulation belongs to
`em_engine._dense_mstep_block` and the scoring/adjoint helpers; local and sparse
routes have their own implementations. `Ft_y` is complex. `Ft_ctf` represents
real weights, although some return layouts store it in a complex array.

[`half_volume_mstep.py`](../../relax/helpers/half_volume_mstep.py)
owns packed-half conventions, the Hermitian `x=0` plane and conversions to
public layouts. Do not assume all accumulators have the full native volume
shape: padding and current-size backprojector grids change their dimensions.

[`mean_helpers.py`](../../relax/refinement/mean_helpers.py) owns
`compute_unregularized_halfmaps_and_align_signs` and
`_reconstruct_and_postprocess_means`.
For K-class refinement, regularized and diagnostic unregularized maps retain
the sign determined by the image/CTF convention. They are not negated to match
the previous reference: a weak class can have unreliable overlap. Both half
slots share the reconstructed class stack. K1 retains its existing sign
continuity check.
The EM reconstruction wrapper explicitly selects FFT computation precision from
its numerator accumulator: complex64 for the float32 path, complex128 for a
double diagnostic. `post_process_from_filter_v2` accepts `fft_compute_dtype`
at both transform boundaries; deliberate higher-precision Wiener denominator
and gridding calculations retain their existing arithmetic. The shared helper's
unspecified option preserves legacy behavior for non-EM callers. Returned dtype
alone is not evidence of transform precision; regression checks inspect both
FFT operations as well as analytic DC normalization in full and packed layouts.
[`noise_updates.py`](../../relax/refinement/noise_updates.py) owns
`update_posterior_noise_variance`, `update_c1_sigma_offset_from_posterior`
and the half-set noise helpers.
These updates consume posterior-weighted residual and moment statistics as
well as accumulators. The input noise representation can be a per-pixel array
or separate half-set inputs; radial statistics and group corrections have
explicit conversion/update paths.

[`relion_normalization.py`](../../relax/relion/relion_normalization.py)
owns `update_relion_norm_scale_corrections` and its result type. It computes
per-image normalization and per-group scales from M-step statistics, using
retained posterior mass for the average normalization. The controller installs
the returned corrections; follower-specific installation remains in
`relion_worker_scale.py`. Host arithmetic stays float64 and returned arrays
use the caller's selected dtype, float32 by default.

[`regularization.py`](https://github.com/ma-gilles/recovar/blob/a6e6b64dd864aefa78b6953ffe2185ffb3be0578/recovar/reconstruction/regularization.py) owns FSC,
tau2 and data/prior helpers. `compute_data_vs_prior` uses shell-average weight
**multiplied by** tau2, tau2 fudge and the padding-volume correction; it is not
`Ft_ctf / tau2`. The controller's K1 scheduling path also uses
`_k1_data_vs_prior_for_scheduling`; the generic weight-based helper is not an
exhaustive description of its resolution policy.
[`relion_reconstruct`](https://github.com/ma-gilles/recovar/blob/a6e6b64dd864aefa78b6953ffe2185ffb3be0578/recovar/reconstruction/relion_functions.py) applies
the actual regularized reconstruction and postprocessing conventions.

During numbered split-half iterations, `join_halves_at_low_resolution` averages
**accumulators**, then writes the result into both halves inside the join sphere.
It does not average already reconstructed maps. The effective joining
resolution is the larger Angstrom value of the configured threshold (40 Å by
default) and the available current resolution. Thus “low resolution” means
low Fourier frequency, not spatial wavelengths smaller than 40 Å. At iteration 1
the current resolution is the `--ini_high` shell, which RELION sets before the
first iteration (`initialize_resolution_from_ini_high`); at the GUI default of
60 Å it, not the 40 Å threshold, bounds the first join.

## 6. Sampling transitions and convergence

[`convergence.py`](../../relax/helpers/convergence.py)
owns `RefinementState`, `update_refinement_state`, `update_angular_sampling`,
`refine_angular_sampling` and `check_convergence`.

Convergence requires the latched `has_fine_enough_angular_sampling` flag,
sufficient resolution stall and sufficiently stable hidden-variable changes.
When per-particle change tracking has not been populated, the implementation
uses its assignment-counter fallback. Reaching `max_healpix_order` prevents
further grid growth; it does **not** establish convergence.

At the sampling transition, the old effective angular step is compared with
75% of the measured angular accuracy. The controller latches the fine-enough
flag at that boundary. Recomputing it from a newly refined grid would move the
convergence decision. `refine_angular_sampling` increases the order, updates
translation range/step, resets the change counters and activates local search
at `auto_local_healpix_order`. Its local sigma is
`2 * radians(new_angular_step / 2**adaptive_oversampling)`.

[`expected_accuracy.py`](../../relax/helpers/expected_accuracy.py)
owns the RELION-style accuracy trial calculation. The approximate posterior
helper `calculate_expected_angular_errors` is a different route. Likewise,
`convergence._relion_optimizer_average_pmax` uses the split-half optimizer's
mass normalization, rather than just averaging all recorded Pmax values.

## 7. Final output and validation boundaries

When `_should_run_final_all_data_iteration` allows it, the controller performs a
final expectation at full image size and reconstructs from the combined
half-set accumulators. This is distinct from low-frequency joining during
numbered iterations. `skip_final_iteration`, convergence state and the explicit
final-iteration policies affect whether it runs.

The return dictionary includes `mean`, per-half/class products, assignments,
`fsc`, `convergence_state` and trajectories. `final_all_data_ran` identifies the
final route. `fsc` is the last numbered-iteration FSC; `final_all_data_fsc` is a
separate field when final all-data processing runs. Consumers should inspect
the actual returned fields instead of assuming one fixed four-item tuple.

`convergence_state.current_resolution` follows RELION's
`updateCurrentResolution`: the last shell before data_vs_prior drops below 1,
computed by `relion_current_resolution_shell` in
[`resolution.py`](../../relax/helpers/resolution.py). In numbered split-half
iterations the K=1 curve is the half-map SSNR `fsc / (1 - fsc)`, so the
crossing is at FSC 0.5. RELION also runs the update after the final all-data
iteration, where `updateSSNRarrays` converts the FSC to whole-data form
`sqrt(2 fsc / (1 + fsc))`, so the crossing moves to FSC 1/7 (0.143). The
controller applies the same final update. Nothing schedules on the value after
the final iteration, so this update only affects what is reported.

First-iteration normalized-CC scoring and winner-take-all reconstruction are
implemented policies. They are not an unimplemented parity gap. Similarly,
no universal gridding-equivalence or speedup claim follows from this map.
Use [EM status](../development/em_status.md) for evidence tied to exact source,
inputs and hardware. The recorded fixed-input backprojection repeatability
finding is one reason to retain intermediate posterior and accumulator checks
alongside final-map, accuracy, wall-time and memory comparisons.

Hierarchical candidate propagation and multiple local-search centers remain
future engine design questions. They would change search support, state and
memory requirements and need their own scientific validation after this cleanup.

## 8. Per-iteration run files and `--continue`

RELION writes its state after every numbered iteration and restarts from it with
`--continue <run_itNNN_optimiser.star>` (`MlOptimiser::write`/`read`,
ml_optimiser.cpp:1359-1557 and 1089-1357). relax does the same.
[`iteration_snapshot.py`](../../relax/refinement/iteration_snapshot.py) defines the state
one numbered iteration hands to the next (`IterationSnapshot`): the half or class
references, tau2, data_vs_prior, the FSC and the growth FSC, per-half and per-optics-group
noise, per-half sigma offsets, class weights, direction priors, current size, `incr_size`
and `has_high_fsc_at_limit`, the sampling perturbation, every `RefinementState` scalar,
and per particle the pose, offset, norm and scale corrections, group and class.
[`run_files.py`](../../relax/refinement/run_files.py) writes it as RELION's
`run_itNNN_optimiser.star`, `run_itNNN_half{1,2}_model.star` (Class3D: `run_itNNN_model.star`),
`run_itNNN_data.star`, `run_itNNN_sampling.star`, the reference maps and, for auto-refine,
the unregularized `run_itNNN_half{1,2}_class001_unfil.mrc`, and reads them back.
`run_full_refinement.py --write-iteration-every N` (default 1) sets the frequency and
`--keep-iterations N` (default 0, keep all, as RELION) keeps only the newest N iterations'
files. A background thread writes each iteration's files while the next one runs.

`--continue` restores that state before the loop (`options.checkpoint.resume`), and the
loop's first iteration then takes the same branches as iteration N+1 of the uninterrupted
run (`has_previous_iteration` in `iteration_loop.py`): the current-size growth reads the
restored FSC, current size and data_vs_prior, sampling is updated at the expectation
boundary and convergence is checked at the top. A continued run reproduces the
uninterrupted run within same-code repeat noise, with three deliberate differences from
RELION's restart:

* Each half keeps its own noise spectrum. RELION's MPI restart broadcasts rank 1's
  `sigma2_noise` to every rank (ml_optimiser_mpi.cpp:750-758), so half 2 restarts on
  half 1's noise; an uninterrupted RELION run keeps both (relax issue #7).
* The particle order stays the one the run started with. RELION randomises the order once
  per process with `random_seed + iter` (exp_model.cpp:406-446), so its restart processes
  the particles in a new order and draws new expected-accuracy trial particles.
* STAR floats are written at full precision, and the optimiser STAR carries the scalars
  RELION's files lack (the latched fine-enough flag, the growth FSC and the rest of
  `RefinementState`) in extra `data_relax_state`/`data_relax_shells` blocks, which
  `MlOptimiser::read` does not parse. The references pass through float32 real-space
  MRCs, as RELION's do.

`--max_iter` stays the last numbered iteration of the whole run, like RELION's `--iter`.
Multi-shape particle STARs and VDAM (InitialModel) have no run files yet.

## BPref translation arithmetic

The float32 BPref translation computes the imaginary component as
`fma(sine, real, round(cosine * imag))`, matching captured RELION output bits.
[`relion_translate_bpref_f32_kernel`](https://github.com/ma-gilles/recovar/blob/a6e6b64dd864aefa78b6953ffe2185ffb3be0578/recovar/cuda/cuda_backproject.cu)
and the fused [`translate_rotate_bpref_f32`](../../relax/cuda/relion_translate_sum.cuh)
use that same explicit operand order; weighted CTF multiplication follows the
complex rotation. The positive-Nyquist coordinate mapping is unchanged.
[`test_relion_translate_bpref_f32_matches_native_captured_bits`](../../tests/unit/test_cuda_relion_translation.py)
is the independent captured-bit guard; the fused translation-sum tests compare
against the standalone primitive. These checks do not establish full trajectory
or performance equivalence.
