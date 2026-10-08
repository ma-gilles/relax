# Codebase map for contributors

Numbered convergence policy lives in [`refinement/convergence.py`](../../relax/refinement/convergence.py):
physical-unit conversion, accuracy admission and optimiser replay precedence.
The same owner selects native versus explicit angular sampling at the start of
expectation, after accuracy and before the numbered grid is constructed.
The mathematical scheduler/metrics remain in `helpers/convergence.py`; the
controller installs state and checks preceding-iteration convergence at its
existing top boundary. See the [complete operation](final_local_sampling_patch_review.md#iteration-convergence-policy).

Start at the entry point for the workflow you are changing, then follow the
state and array layouts into its kernels. RECOVAR has several EM workflows;
sharing a numerical primitive does not make their controllers interchangeable.
The [development contract](../../AGENTS.md) defines change scope and validation.

## Workflow entry points

`refinement.iteration_planning.initialize_refinement_state` owns startup state
construction and replay/FSC/low-pass/frozen/continuation precedence. It returns
the existing convergence state directly. The numerical controller retains phase
timing, grid construction, subsequent model/particle installation and later
scientific state updates.

`relion.input_particle_table` owns physical particle identity, paired-half and
Class3D row preparation. `ParticleLayout` pairs selected input rows with their
expected-accuracy source/local frame throughout command consumers. Source/mode
selection, diagnostic admission and important state writes remain in the
command; this run-constant identity result is separate from mutable half poses.

`refinement.particle_loading` owns command particle-format admission, reading
setup and mask configuration. `LoadedParticles` returns the existing dataset with
its format/grid/preprocessing metadata. RECOVAR remains the image reader and
Fourier preprocessor. Optimiser-source discovery belongs to
`refinement.command_options`; the controller retains half-set construction,
startup-noise source selection, explicit state updates and invocation timing.

Refinement flag definitions, job defaults and supported-mode validation belong to
[`command_options.py`](../../relax/refinement/command_options.py). The execution
controller retains cache/import setup, seed assignment and validation timing.

| Workflow | Entry point | Main implementation |
| --- | --- | --- |
| Covariance pipeline | [`standard_recovar_pipeline`](https://github.com/ma-gilles/recovar/blob/dev/recovar/commands/pipeline.py), the `recovar pipeline` command | [`principal_components`](https://github.com/ma-gilles/recovar/blob/dev/recovar/heterogeneity/principal_components.py), covariance estimation, then embedding |
| Pipeline PPCA | The same pipeline with `--use-ppca`; `_run_ppca_refinement` selects the PPCA path | [`recovar.ppca.ppca.EM`](https://github.com/ma-gilles/recovar/blob/dev/recovar/ppca/ppca.py), using the supplied dataset poses |
| RELION-style K1/K-class refinement | `relax refine` / `relax class3d`: [`refinement/full_refinement.py`](../../relax/refinement/full_refinement.py) resolves inputs and options | [`iteration_loop.refine_single_volume`](../../relax/refinement/iteration_loop.py); despite the name, this controller also handles K-class refinement |
| Pose-marginal PPCA refinement | [`refinement_loop`](../../relax/ppca_refinement/refinement_loop.py) exposes dense and local refinement loops | [`dense_dataset`](../../relax/ppca_refinement/dense_dataset.py), [`local_dataset`](../../relax/ppca_refinement/local_dataset.py), and their fused kernels |
| InitialModel/VDAM | [`vdam.iteration_loop.run_vdam_iterations`](../../relax/vdam/iteration_loop.py) | Initial-model schedules, subset selection, state and reconstruction |
| Earlier and independent EM references | Test oracles in `tests/oracles/`, imported by tests as `oracles.<module>`; no relax command uses them | [`states`](../../tests/oracles/states.py), [`iterations`](../../tests/oracles/iterations.py), the E-step/M-step and heterogeneity modules, and the independent [normalized-CC](../../tests/oracles/normalized_cc_replay.py) and [Gaussian-reduction](../../tests/oracles/gaussian_reduction_replay.py) replays |

Pipeline PPCA and pose-marginal PPCA have different entry points and state
contracts. Choose the implementation reached by the actual command. The
pipeline PPCA path currently rejects tilt-series input. Consult the
[PPCA refinement guide](../../relax/ppca_refinement/AGENTS.md) when working
on pose refinement, and the [paper-data runbook](della.md) for pinned inputs.

Unperturbed exhaustive rotation matrices, working Euler rows, HEALPix order and
point group are produced together as `sampling.RotationGrid` in
[`sampling.py`](../../relax/sampling.py). Initial planning and numbered replacement
retain that owner; final sampling consumes it. The command and numbered controllers
remain unfinished; the [current review](final_local_sampling_patch_review.md#rotation-grid-ownership-and-remaining-controller-work)
shows the actual producer, updates and final consumer.

Numbered normalization preparation, numerical estimation and reporting live in
[`relion_normalization.py`](../../relax/relion/relion_normalization.py). The controller
selects policy and visibly installs corrections; strict follower installation is
owned by [`relion_worker_scale.py`](../../relax/relion/relion_worker_scale.py).

## EM package layout

`relax/` is the common implementation package. Standard refinement and
VDAM have separate controllers and schedules, and share numerical owners where
their semantics already match:

| Directory | Responsibility |
| --- | --- |
| `refinement/` | Standard EM iteration, convergence/finalization, options and map updates |
| `vdam/` | InitialModel driver, subset schedule, learning rates and VDAM state transitions |
| `classification/` | K-class routing, inputs and joint result assembly |
| `dense/`, `local/` | Dense and exact-local execution |
| `scoring/`, `sparse_pass2/` | Coarse scores/support and sparse second-pass execution |
| `helpers/` | Shared array layouts, operators, batching, precision and statistics |
| `relion/` | Runtime RELION metadata, normalization, CTF and native adapters |
| `cuda/` | The EM CUDA library `librelax_cuda.so`: translation unit `relax_kernels.cu`, its headers and Makefile, and the Python side (`kernels.py`: FFI wrappers, target tables and the `NativeLibrary` loader; `noise_residual.py`) |
| `reconstruction/` | EM-only RELION variants split from `recovar.reconstruction` (`regularization_relion`, `relion_functions_relion`, `noise_relion`) |
| `diagnostics/` | Optional capture writers, replay and intervention tools |
| `reference/` | Independent earlier EM/covariance formulations and deterministic numerical replays |
| `ppca_refinement/` | Pose-marginal PPCA workflow and its K-class bridge |

The [dense reference formulation](../math/em_dense_reference.md) documents the
independent algorithmic core under `reference/`.

RELION diagnostic checkpoint restoration lives in [`relion/vdam_checkpoint.py`](../../relax/relion/vdam_checkpoint.py), separate from the VDAM execution driver. Native moment/reference and BPref overrides, including post-M-step reference-map replay, live in [`diagnostics/vdam_mstep_replay.py`](../../relax/diagnostics/vdam_mstep_replay.py); [`vdam/mstep_single_class.py`](../../relax/vdam/mstep_single_class.py) retains the reconstruction transaction and its numerical boundary calls.

InitialModel STAR import/export is owned by
[`relion/initial_model_io.py`](../../relax/relion/initial_model_io.py).
[`vdam/output.py`](../../relax/vdam/output.py) handles artifact paths,
startup metadata and write cadence; it delegates STAR serialization to that
adapter. Optics and particle records live in
[`vdam/state.py`](../../relax/vdam/state.py), so sampling does not import
serialization. Column lookup is shared in `data_io.starfile.star_column`, while
strict list-style checkpoint scalars live in `relion.relion_metadata`.
The refinement CLI also reads RELION initialization metadata there; captured
iteration overrides and final-reference readers belong to `diagnostics.relion_replay`.
`helpers.iteration_history` owns class/pose/history artifact formatting and
restart readers; `diagnostics.parity_dump` owns timing artifact readers and
summaries. The CLI retains argument declarations, input/option precedence,
refinement invocation, destination selection and final output writes.

Particle bootstrap is owned by [`vdam/bootstrap_iref.py`](../../relax/vdam/bootstrap_iref.py): it loads the bootstrap images and constructs the initial reference/state. [`vdam/init.py`](../../relax/vdam/init.py) contains the state-only initialization formulas, while [`relion/initial_noise.py`](../../relax/relion/initial_noise.py) owns the image iterator, initial noise estimate, single-optics noise input and MPI process-start half-set noise policy. The driver coordinates these stages; sampling geometry stays in [`vdam/native_sampling.py`](../../relax/vdam/native_sampling.py).

VDAM coarse-call naming and result diagnostic packaging live with the shared [`coarse_gaussian_diagnostics.py`](../../relax/diagnostics/coarse_gaussian_diagnostics.py) and [`coarse_score_diagnostics.py`](../../relax/diagnostics/coarse_score_diagnostics.py) owners. The sparse E-step invokes them but does not implement report bookkeeping. These extracted helpers remain counted in the VDAM size budget.

There is no second EM stack for VDAM. Its adapters supply the existing shared
kernels with VDAM-specific inputs. Scheduling and state transitions remain with
their workflow. The retired `dense_single_volume/` and `initial_model/` source
directories are gone. Import current owners directly: EM/VDAM is work in
progress, with no backward compatibility requirement for its Python APIs, CLIs
or old Python object names. Main heterogeneity pipeline compatibility remains
required. Historical logger names remain stable.

## Shared data and numerical boundaries

| Boundary | Owner | Contract to inspect |
| --- | --- | --- |
| Particle loading and batch identity | [`CryoEMDataset`](https://github.com/ma-gilles/recovar/blob/dev/recovar/data_io/cryoem_dataset.py), image loaders and half-set utilities | Original image/particle IDs, subset-local positions, half-set membership, image backend and CTF metadata |
| Forward-model configuration and state | [`core.configs`](https://github.com/ma-gilles/recovar/blob/dev/recovar/core/configs.py) | `ForwardModelConfig` static fields versus dynamic `ModelState` arrays; changing a static value may change JIT specialization |
| Fourier transforms and volume I/O | [`fourier_transform_utils`](https://github.com/ma-gilles/recovar/blob/dev/recovar/core/fourier_transform_utils.py), [`utils.helpers`](https://github.com/ma-gilles/recovar/blob/dev/recovar/utils/helpers.py) | Centered Fourier conventions, flattened arrays, full versus half spectrum, and the RELION axis/sign conversion; relax map files are in RELION's convention ([`helpers/map_io.py`](../../relax/helpers/map_io.py): `write_map`, `load_relax_map`; references through `load_relion_volume`) |
| Mean, noise and regularization | [`homogeneous`](https://github.com/ma-gilles/recovar/blob/dev/recovar/reconstruction/homogeneous.py), [`noise`](https://github.com/ma-gilles/recovar/blob/dev/recovar/reconstruction/noise.py), [`regularization`](https://github.com/ma-gilles/recovar/blob/dev/recovar/reconstruction/regularization.py); EM-only RELION variants in [`em/reconstruction`](../../relax/reconstruction/regularization_relion.py) | Half-set ownership, shell support, normalization, prior construction and reconstruction units |
| Saved results | [`output`](https://github.com/ma-gilles/recovar/blob/dev/recovar/output/output.py), [`ResultPaths`](https://github.com/ma-gilles/recovar/blob/dev/recovar/output/output_paths.py) | Serialized field names, shapes, original IDs, and downstream `PipelineOutput` consumers |
| CUDA and RELION references | [`cuda_backproject`](https://github.com/ma-gilles/recovar/blob/dev/recovar/cuda_backproject.py) (pipeline library `libcuda_backproject.so` and its loader), [`cuda_build`](https://github.com/ma-gilles/recovar/blob/dev/recovar/cuda_build.py) (`NativeLibrary`, public CUDA headers), [`em/cuda/kernels`](../../relax/cuda/kernels.py) (EM library), [`relion_bind`](../../relax/relion_bind/__init__.py) | Loaded binary identity, device placement, native layouts and independent reference behavior |

The [source conventions](https://github.com/ma-gilles/recovar/blob/dev/recovar/CLAUDE.md) give the exact FFT and
RELION-frame rules. Follow those helpers when loading volumes for a comparison;
raw MRC arrays and uncentered FFT calls are not interchangeable with them.

## Dense and local EM ownership

Particle pose snapshots, engine interpretation and half-ordered convergence inputs
belong to `prepare_particle_pose_update` and `prepare_pose_comparison` in
[`half_inputs.py`](../../relax/refinement/half_inputs.py). Their computed results
keep previous/current frames together. The controller installs state and records
history/particle files between those two operations, preserving their timing.

Post-reconstruction observed/scheduling resolution shells belong to
[`estimate_k1_iteration_resolution` and `estimate_class_iteration_resolution`](../../relax/helpers/resolution.py);
the controller passes each its mode's curve. Their result
is consumed by diagnostics, history, next-iteration planning and convergence while
the controller retains the timing of state writes and native convergence checks.

Use the [EM implementation reference](em_implementation.md#dense-and-local-em-ownership)
for detailed module contracts. Start with the boundary being changed:

| Boundary | Main owner |
| --- | --- |
| Command oracle admission and resolved controls | [`command_options.py`](../../relax/refinement/command_options.py); portable dispatch manifest/particle/file admission and CLI/optimiser cap, CTF and initial-filter precedence. `full_refinement.py` retains source choice, strict replay decision, follower preparation and visible option installation. |
| Sealed restart CLI admission and runtime adaptation | [`diagnostics/frozen_boundary_cli.py`](../../relax/diagnostics/frozen_boundary_cli.py); invocation/source binding, effective-config checks, projector-only replay slots, capture attachment with source identity and float32 scoring-noise expansion. Schema/file loaders keep their existing owners. Command retains modes, state installation and provenance publication. |
| Prejoin capture and finite auditing | [`diagnostics/reconstruction.py`](../../relax/diagnostics/reconstruction.py); target selection and native-half audit before cross-half mixing, plus pre/postjoin BPref serialization. Numerical join and state writes remain in the controller. |
| Iteration scheduling and state mutation | [`iteration_loop.py`](../../relax/refinement/iteration_loop.py) |
| Refinement scoring projector preparation | [`projector_preparation.py`](../../relax/refinement/projector_preparation.py); accuracy-produced `ProjectorReuse` binds reference identity and image/window support; `prepare_scoring_projector` owns reuse, transform, disk cache and dumps. Controller retains captured admission, half order and release/install boundaries. |
| Checkpoint capture and saved iteration schema | [`refinement/iteration_snapshot.py`](../../relax/refinement/iteration_snapshot.py); header replacement preserves retained-map lifetime, then complete array/schema capture returns `IterationSnapshot`. The controller owns scheduling/growth; `run_files.py` owns disk formats. |
| Iteration-zero model-file replay | [`diagnostics/initial_model_replay.py`](../../relax/diagnostics/initial_model_replay.py); ordered file/table identity, NPZ/live/STAR noise precedence, MPI broadcast, prior expansion and optimiser controls. Command retains admission and explicit installation/reporting. Shared formulas remain in RECOVAR and `relion/initial_noise.py`. |
| Global E/M execution | [`k_class.py`](../../relax/classification/k_class.py) `run_dense_k_class_em_adaptive`: coarse pass 1 in [`significance.py`](../../relax/scoring/significance.py), fine pass 2 and M-step sums in [`resident_pass2.py`](../../relax/sparse_pass2/resident_pass2.py) |
| Local search orchestration and kernels | [`local_search_iteration.py`](../../relax/refinement/local_search_iteration.py), [`resident_local_pass2.py`](../../relax/sparse_pass2/resident_local_pass2.py) |
| Class routing and joint result assembly | [`k_class.py`](../../relax/classification/k_class.py), [`k_class_results.py`](../../relax/classification/k_class_results.py) |
| Replay selection and final all-data admission/execution | [`relion_replay.py`](../../relax/diagnostics/relion_replay.py), [`finalization.py`](../../relax/refinement/finalization.py) |
| Final native/replay sampling and prepared grids | [`final_sampling.py`](../../relax/refinement/final_sampling.py) |
| Numbered prior/reconstruction and first-CC reporting | [`mean_helpers.py`](../../relax/refinement/mean_helpers.py): ordered Class3D replay/CTF/diagnostic aggregation, K1 split-half estimation, the K1 half-map and Class3D class-map regularized solve/capture/filter/flatten operations and reporting tapers. Controller installs maps/tau2 and publishes class scheduling/history before detail taper. Final retains distinct policies over shared primitives. |
| Final reconstruction and half-map/class priors | [`final_reconstruction.py`](../../relax/refinement/final_reconstruction.py) |
| Shared reference startup formulas | [`reference_initialization.py`](../../relax/relion/reference_initialization.py), used by refinement and VDAM |
| Refinement mask widths and padding defaults | [`geometry.py`](../../relax/relion/geometry.py); VDAM retains its own padding and mask settings |
| Resident candidate tables and chunk layout | [`sparse_pass2/resident_candidates.py`](../../relax/sparse_pass2/resident_candidates.py); shared host cell offsets for global, local parent and local fine scoring. Drivers retain placement, posterior kernels and accumulation. |
| Sparse engine dispatch and independent reference | [`sparse_pass2/dispatch.py`](../../relax/sparse_pass2/dispatch.py), [`tests/helpers/relion_estep_reference.py`](../../tests/helpers/relion_estep_reference.py); grid and support arithmetic stay in `helpers/oversampling.py` |
| Standard half-set and first-iteration adapters | [`refinement/half_scoring.py`](../../relax/refinement/half_scoring.py), [`refinement/firstiter_cc.py`](../../relax/refinement/firstiter_cc.py) |
| Particle pose interpretation and persistent half state | [`refinement/half_inputs.py`](../../relax/refinement/half_inputs.py); resolves explicit or grid poses into computed matrices/Euler rows and relative/absolute pixel shifts. The controller installs state and records history. |
| Strict RELION follower topology preparation | [`relion/relion_worker_scale.py`](../../relax/relion/relion_worker_scale.py); validates MPI admission, replay and numbered particle ownership before execution |
| Particle-table source and group identity | [`relion/input_particle_table.py`](../../relax/relion/input_particle_table.py); admits half-set STAR schema/optics geometry and produces authoritative group source plus matching physical/optics axes in half image order |
| Input geometry and replay units | `ImageGeometry` in [`helpers/resolution.py`](../../relax/helpers/resolution.py); fixed shape and validated physical pixel size. [`diagnostics/relion_replay.py`](../../relax/diagnostics/relion_replay.py) consumes geometry directly. Controller retains the borrowed source pixel scalar where host promotion requires its original type; changing windows/model support have separate owners. |
| Refinement memory planning | [`helpers/batch_planning.py`](../../relax/helpers/batch_planning.py) and [`refinement/expectation_batches.py`](../../relax/refinement/expectation_batches.py); live dense/local/compact budgets and half adaptation. The disconnected future whole-local descriptor/fingerprint family is retired; supported batch decisions are unchanged. |
| Numbered half expectation preparation, execution and recording | [`refinement/expectation.py`](../../relax/refinement/expectation.py); `NumberedExpectation` and `prepare_numbered_expectation` bind the shared canonical grid, dense/local support and diagnostic policy. The same owner prepares per-half priors/batches/optics, dispatches, and records profiles/captures and ordered counts. Controller retains modes, tomography adaptation, publication, offloading, serial/overlap choice and release. |
| Final SPA half prior/optics preparation | `prepare_final_half` in the same expectation owner; local/dense scoring and manifest export consume its prepared operands |
| Half image preprocessing | `configure_half_image_preprocessing` in [`refinement/particle_loading.py`](../../relax/refinement/particle_loading.py); backend registration/selection, source-faithful admission and mask units; paired row/noise/accuracy preparation owns the selected CTF copy and releases unused source tables. Controller retains source/mode admission, dataset subsetting and setup order/timing; image datasets and backends stay rooted by the half input owners. |
| Refinement startup noise | [`refinement/startup_noise.py`](../../relax/refinement/startup_noise.py); ordered source rows, SPA/multi-shape/tomography image adaptation, live host sigma2 and scoring expansion. Command retains source selection; host FFT/mask formulas remain in `relion/initial_noise.py`. |
| Numbered image-size and startup grid planning | [`refinement/iteration_planning.py`](../../relax/refinement/iteration_planning.py); initial precedence, K1 raw/corrected resolution versus growth signals, Class3D prior curves, computed results. Controller retains mode selection, writes, oracle and replay/angular order. |
| Numbered perturbation and scoring windows | [`refinement/iteration_planning.py`](../../relax/refinement/iteration_planning.py); sealed/STAR/native perturbation precedence, independent model/particle cutoffs and adaptive pass-1 sizing from the incoming order. Grid execution remains in `sampling.py` and the controller. |
| Completed-iteration correction reporting and parity capture | `NormScaleCorrectionReport` in [`relion/relion_normalization.py`](../../relax/relion/relion_normalization.py), consumed by checkpoints and `dump_numbered_iteration` in [`diagnostics/iteration.py`](../../relax/diagnostics/iteration.py). Controller retains correction installation, capture selection and checkpoint order. |
| Final refinement result files | [`refinement/result_files.py`](../../relax/refinement/result_files.py); reused result schemas, final diagnostic formatting, array layouts, NPZ compression, profiles and final maps. Controllers retain model selection and publication order. |
| Coarse/sparse scoring | [`scoring/significance.py`](../../relax/scoring/significance.py), [`sparse_pass2/resident_pass2.py`](../../relax/sparse_pass2/resident_pass2.py) |

Coarse window metadata is published by `scoring/coarse_publication.py`.
Pass 1 (the coarse pass) scores, adds the priors and reduces an image batch in one program,
`scoring.significance._coarse_pass1_blocks` (see [EM status](em_status.md), "Pass 1 as one program per
image batch") on RELION's exact coarse operands only: pass 1 needs a CUDA GPU and RELION's CUDA image
preprocessing, and the generic dense coarse scorer was removed on 2026-10-02 (its arithmetic is the test
oracle `tests/helpers/generic_coarse_reference.py`). The coarse GEMM scorer is its only Gaussian scorer; the fused, native-texture and
rectangular coarse scorers, the certified K=1 GEMM hybrid (`coarse_gemm_hybrid`, `coarse_partition`,
`coarse_device_*`), the fused pass-1 block and the paired and streaming GEMM score captures
(`coarse_gemm_streaming`) were removed on 2026-10-02, and their switches are listed as retired in
`relax/renamed_environment.json`. The `RELAX_SIGNIFICANCE_DUMP_*` target scores come from the pass-1
program, and a dump records RELION's exact coarse operands (`coarse_gaussian_*`); the dumps are written
by `diagnostics/coarse_gaussian_diagnostics.py`. The generic-operand dump fields (`shifted_data`,
`ctf2_data`, `window_indices`, `half_weights`), the `RELAX_SIGNIFICANCE_DUMP_PROJECTION_ROTATIONS`
projection extras and the scripts that read them (`scripts/analyze_em_k1_live_reference_counterfactual.py`,
`analyze_em_k1_fine_ppref_source_boundary.py`, `analyze_em_k1_fine_top_pair_operands.py` and their unit
tests) were removed on 2026-10-02; recover them from git history (before the removal commit) if needed.
Live BPref execution modes are selected by `sparse_pass2/sparse_pass2_policy.py`;
capture scopes and shadow comparisons stay with the diagnostic owners.

Import execution APIs directly from their defining modules; helpers must not
initialize controllers or scoring engines. During structural cleanup preserve
casts, reduction/JIT order, array lifetime, scientific defaults and saved formats.
Canonical source Euler angles and host pixel geometry remain metadata; derive
computation arrays from them. Required validation comes from the scoped guides,
not from the size of this overview. Current evidence belongs in [EM status](em_status.md).

Numbered K1 current-FSC and independent half-weight tau2 estimation live in
`mean_helpers.estimate_split_half_prior` with its `SplitHalfPrior` result. The
controller retains joining, K1/Class3D policy, reference replacement and later
CC taper/host parking. Raw FSC and corrected growth FSC remain distinct.
The same `ReconstructionSettings` is constructed once for numbered/final prior
estimation, reconstruction, masks/initial filtering and Class3D captures. It owns
their invariant regularization and mask settings; current windows, accumulator
layout and phase decisions remain explicit inputs.

## Diagnostics and reusable evidence

Class3D M-step capture is owned by
[`diagnostics/reconstruction.py`](../../relax/diagnostics/reconstruction.py):
`write_class_mstep` computes captured floor-shell weight summaries and writes the
historical NPZ schema from the prior result and reconstruction settings. The
per-class gate and numerical class loop remain in the refinement controller.

Sealed diagnostic CLI admission lives in
[`frozen_boundary_cli.py`](../../relax/diagnostics/frozen_boundary_cli.py): flags,
source/arm checks, manifest binding and effective-runtime adaptation. Bundle
schemas and array validation remain in
[`frozen_boundary.py`](../../relax/diagnostics/frozen_boundary.py). The refinement
command owns when admission and the resolved-runtime check run.

The optional iteration, reconstruction, pass-2 operand and normalization capture
writers live in [`relax/diagnostics`](../../relax/diagnostics/__init__.py).
Production engines call them at the existing capture boundaries. The package
initializer imports nothing; the individual writers still use shared numerical
utilities and the BPref capture context. This is an ownership boundary, not a
claim that all diagnostics have already been removed from normal import paths.
Local operand comparison and parity-worktree provenance also live there; they
are EM/RELION diagnostics rather than general RECOVAR utilities.

[`relax/relion_bind`](../../relax/relion_bind/__init__.py) currently mixes
native runtime dependencies with independent validation interfaces. RELION-style
EM uses its sampling, particle ordering, CTF and reconstruction routines; those
are not removable merely because the package also supports parity tests.
Oracle-only bindings, replay tools and historical experiment scripts need a
consumer/reproduction audit before relocation or deletion. Keep the independent
references under tests separate from the production functions they validate.

See [diagnostic owners](em_implementation.md#diagnostics-and-reusable-evidence)
for score, posterior, BPref, noise, normalization and output capture boundaries.
Use existing provenance/test wrappers and immutable evidence roots. Historical
runs qualify only their recorded source and inputs; missing cells are not passes.

### Replay state diagnostics

See [replay/state ownership](em_implementation.md#replay-state-diagnostics)
for frozen snapshots, intervention ordering, sampling, convergence, resolution
and projector preparation. Read these contracts before changing those boundaries.

## Ground-truth reporting

See [GT reporting owners](em_implementation.md#ground-truth-reporting) and
[the reporting guide](gt_reporting.md). Rigid fit-once/apply-many reporting is
opt-in; it does not change E/M execution or scientific acceptance gates.

The completion reporter and final-BPref replay share NumPy FSC calculations in
[`scripts/fsc_metrics.py`](../../scripts/fsc_metrics.py). This module remains
independent of production scoring and does not select a JAX backend. Include it
in source manifests when freezing or copying either reporter; the reporter file
alone no longer contains the full metric implementation.

Shared RELION projector construction lives in
[`relion_projector_setup.py`](../../relax/relion/relion_projector_setup.py):
`reference_to_relion_projector_half_maps_and_power` selects native/JAX setup and
performs the established frame and dtype conversion; the maps-only wrapper
releases the unused power spectrum. EM projector caching and VDAM both use this
owner directly. VDAM's `dense_adapter` retains state-specific preparation and
accumulator conversion, so EM no longer imports the VDAM execution adapter to
construct projectors. The same shared owner normalizes local projector slab
shapes without changing dtype. Shared host/device x=0 Hermitian enforcement
lives in [`helpers/half_volume_mstep.py`](../../relax/helpers/half_volume_mstep.py);
resident capacity ladders use [`helpers/env_flags.py`](../../relax/helpers/env_flags.py).

## VDAM code budgets

The original `fbdf23f9` InitialModel snapshot contains 5,014 Python lines.
At `afa3d6d46`, the same accounting scope contains 8,626, including code moved
into shared owners. The user approved replacing the inherited 6,100-line cap
with audited responsibility budgets on September 13, 2026. The table below
reflects the September 28 audit, including the intervening adaptive E-step and
multi-optics owners and the new opt-in SGD optimizer. Numerical tolerances,
baselines and scientific gates are unchanged.

| Responsibility | Audited lines | Budget | Retained scope |
| --- | ---: | ---: | --- |
| Controller and schedules | 1,781 | 1,797 | Driver, iteration/subset schedules, options and launcher defaults |
| Initialization | 500 | 500 | Bootstrap, initial state and shared initial-reference filter |
| Sampling and layout | 940 | 950 | Native sampling updates, canonical pose metadata and frame conversions |
| E-step | 1,880 | 1,936 | Adaptive E-step, configuration, statistics, support, projector setup and probability updates |
| SGD optimizer and noise | 309 | 309 | Scalar-curvature momentum update and discounted masked-noise estimator |
| Reconstruction and state | 777 | 790 | Single-class M-step transaction, precision checks, state and class dispatch |
| Input/output | 1,329 | 1,330 | STAR metadata, startup artifacts, RELION checkpoint import and initial noise |
| Diagnostics | 1,092 | 1,240 | GT registration, native moment/reference replay and coarse report bookkeeping |
| **Total** | **8,608** | **8,852** | **244 lines of total headroom** |

Subtomogram InitialModel (October 1, 2026) raises four budgets by exactly the lines it adds, with no
headroom: controller to 1,896 (+99: `--ios` input, the `TomoDataset` branch, the E-step closure split),
initialization to 1,071 (+126: tilt-image start-up and seeding, per-image `Aproj R` in the bootstrap),
sampling to 974 (+24: tilt-image expected accuracy, the subtomogram offset-step rule), E-step to 2,070
(+134: `tomo_estep.py`) and input/output to 1,348 (+18: subtomogram particle state and 3D offsets);
`tests/unit/initial_model/test_refactor_invariants.py` holds the ceilings.

Several optics groups on one image grid (October 1, 2026) raises five budgets by the lines added, with
no headroom: controller to 1,910 (+14: the E-step's optics-group ids, the momentum-SGD refusal),
initialization to 1,087 (+16: per-group start-up particles, per-image bootstrap optics and class
positions), sampling to 1,003 (+29: the expected accuracy per group, recombined), E-step to 2,121
(+47: per-group noise rows and update) and input/output to 1,381 (+31: per-particle optics, one noise
table per group, `relion_startup_positions`).

Optics features in InitialModel (October 2, 2026) raise three budgets by the lines added: controller
to 1,932 (+22: the optics gate, the premultiplied flag, the subset's average CTF^2), initialization to
1,133 (+46: the bootstrap CTF with each optics group's even Zernike gamma) and sampling to 1,009 (+6:
the trials' optics-table CTFs in the expected accuracy).

RELION's per-particle `rlnLogLikeliContribution` in InitialModel's data.star (October 3, 2026), which
VDAM had copied from its input STAR, raises controller to 1,942 (+10: dLL from the E-step's log
evidence), E-step to 2,148 (+27: the evidence through the E-step meta and
`estep_meta_updates.relion_log_likelihood_contributions`) and input/output to 1,382 (+1: the writer).

Optics groups on several image shapes in InitialModel (October 5, 2026; RELION S3b), measured against
main 7d7f3c92: controller 1,952 -> 1,977 lines (+25; budget 1,958 -> 1,977: one dataset per image shape,
the refusal of features not wired for several shapes), initialization 1,141 -> 1,185 (+44; budget 1,185:
start-up images masked at their own pixel size and resized and windowed to the model grid), reconstruction
and state 549 -> 552 (+3; budget 552: per-particle pixel size and box), input/output 1,379 -> 1,385 (+6;
budget 1,385), E-step 1,884 -> 1,923 and sampling 991 -> 1,008 (within their budgets), and a new
`shape_class_estep` budget of 152 for
[`vdam/shape_class_estep.py`](../../relax/vdam/shape_class_estep.py), the per-shape-class orchestration
of the adaptive E-step, so the E-step budget does not grow.

The September 28 architectural review charges the new functionality explicitly.
Relative to `e2401c4c`, the controller adds 24 lines in the driver for SGD noise
initialization and wiring, 89 in the loop for optimizer selection, fixed
bandwidth/subsets and prior/occupancy reporting, and 29 in typed options.
The controller allowance therefore grows by exactly 142 lines, with no added
headroom at the pre-removal review. Uniform joint class/direction priors add 36
lines to probability updates; their E-step allowance grows by 36 over the
upstream 1,900-line ceiling. The upstream exact-local VDAM removal integrates
`adaptive_estep.py` into that responsibility and removes `sparse_pass2_estep.py`;
its merged E-step count is 1,880 lines. All 309 lines of the new
`sgd_initial_model` package have a separate owner and allowance, and the
inventory guard checks that package dynamically. Other responsibility limits
are unchanged. These are reviewed allowances for
the [new optimizer and shared controls](../math/momentum_sgd.md), not changes
to accuracy thresholds or exclusions of extracted code from accounting.

Noise failure reports and optional noise-boundary captures now live in
[`diagnostics/vdam_noise.py`](../../relax/diagnostics/vdam_noise.py);
`vdam/estep_meta_updates.py` owns the numerical update. This move transfers
110 budget lines from E-step to diagnostics without increasing the 8,850 total.
Solvent masking now lives with reconstruction in `vdam/m_step.py`; state precision
preparation lives beside its dtype definitions in `vdam/mstep_single_class.py`.
Their move transfers 90 budget lines from controllers to reconstruction/state,
again preserving the 8,850 total.
VDAM translation and class-orientation prior construction now lives beside the
sampling state/plan in `vdam/native_sampling.py`; this transfers 100 budget lines
from controller to sampling. The earlier `deeb4b5ed` noise-adapter extraction
added 55 shared lines to the counted input/output owner, reaching 1,269; 20 more
lines of controller headroom now cover that responsibility (1,270 allowance).
The combined allowance remains 8,850.
Projector refresh/consume lifecycle now lives beside its builders in
`vdam/dense_adapter.py`; its unchanged stale-state checks and single-use handoff
transfer 40 budget lines from controllers to E-step, preserving the total.
Image-mask setup and normalized-spectrum conversion also live in the E-step
adapter; this transfers another 35 budget lines from controllers to E-step.
These historical transfers predate the current table. Package A counts both
serialization modules in full, the shared column/scalar definitions with their
spacing allowance, and `NativeOpticsState` once in reconstruction/state. Its
net accounting change is +9 lines (module/import overhead); no ceiling increased.

Before its removal, the largest exact-local routine grew from 228 to 733 lines:
sparse pass-2 orchestration had accumulated compact/local, zero-oversampling,
exact-operand and execution-policy cases. The current InitialModel route uses
the shared adaptive E-step.
Other identifiable additions include the 430-line native checkpoint adapter,
96-line continuation subset-order replay, 409-line rigid-registration owner,
and native M-step/reference replay diagnostics. These have distinct scientific
or diagnostic consumers; their size alone does not justify deletion. The audit
also identifies controller, particle-input and reconstruction growth for further
simplification. The budgets are limits, not a declaration that all code is necessary.

The [budget guard](../../tests/unit/initial_model/test_refactor_invariants.py)
assigns every VDAM Python module to exactly one responsibility, requires all
listed files to exist, and counts shared extractions with the previous spacing,
import and alias allowances. A move must migrate its accounting; a new module
must receive an explicit owner. Each responsibility must fit independently, so
spare diagnostic budget cannot conceal growth in the E-step. The budgets are
review signals with slack (owner ruling, 2026-10-05), as refinement's ceilings:
a total above its budget by at most `max(1, ceil(budget * 5 / 100))` lines warns,
and only a total beyond that fails. Review justified
new functionality before revising any budget. Preserve separate numerical paths
when merging them would complicate control flow or change arithmetic.

The source inventory and growth audit (artifact lost 2026-09-26, see em_work/LOSS_AUDIT_20260926.md)
records the original comparison at `4f83abed4` (8,633 lines). The subsequent
prior cleanup removed 11 lines and the startup metadata boundary added four;
the table above accounts for both. Historical file/name counts distinguish
relocation from new names but are not a semantic proof of dead-code completeness.


Startup sampling policy is produced by
`relax/refinement/command_options.py:resolve_initial_sampling` and reused as
`InitialSampling`. Archive metadata conversion belongs to
`relax/refinement/result_files.py:build_archive_metadata`; the command owns
publication order. The current review shows both producers and all calling facts.

Startup pose source selection, loading and norm-correction preparation belong to
`relax/relion/input_poses.py:prepare_initial_poses`. Its `InitialPoses` result
retains the source-specific seed payload; `PoseProvenance` is reused by profile
and archive reporting. The command retains frozen-correction selection and
visible replay options.
