# EM implementation reference

Start with the [workflow map](codebase.md) to select the correct controller.
Read the relevant owner below for the current task; this is implementation
reference, not a requirement to load every module description on each resume.
The [EM contract](../../relax/AGENTS.md) remains authoritative for scientific
rules and the [current status](em_status.md) identifies reviewed source/evidence.

## Dense and local EM ownership

Pose-stack preparation for convergence belongs to
[`helpers.convergence.concatenate_pose_stacks_or_none`](../../relax/refinement/refinement_state.py).
The iteration controller supplies precision and logging context and retains the
four current/previous rotation/translation call sites. Empty half-sets, missing
poses, malformed-shape warnings and concatenation ownership are preserved.
The completed state also owns `fraction_changed`; the controller reuses it for
history instead of repeating the assignment reduction after replay overrides.
The assignment metric takes only the two index stacks and translation count;
it compares decoded rotation indices, with no angular-distance threshold.
`convergence.concatenate_assignments` and `concatenate_assignments_or_none` join
both half-sets' int32 assignment indices for that comparison; the strict form
is used for the current iteration and the tolerant form for a previous
iteration that may not have recorded assignments. The optimizer Pmax
normalization mass comes from `convergence._relion_pmax_normalization_mass_per_half`:
Class3D uses each half's retained M-step posterior mass, K=1 the half's noise
`sumw`, and `_relion_optimizer_average_pmax` divides half 1's Pmax sum by it.

Direction-prior construction, learning and scoring belong to
[`orientation_priors`](../../relax/sampling/orientation_priors.py).
`DirectionPrior` pairs one half-model's probabilities with their HEALPix order:
a vector for K=1, one row per class for Class3D, fixed where the prior is produced.
`learn_k1_direction_priors` and `learn_class_direction_priors` return updates.
K=1 collapses each half's rotation posterior at the order used for scoring and
skips a half whose prior cannot form a RELION log prior, with the warning routed
through the controller logger; K-class combines both halves' per-class posteriors
on the exhaustive grid for global scoring only and stores an independent copy per
half. The controller decides when both posteriors are present, derives the K=1
order from its sampling state and supplies the grid sizes, so the controller's
sampling policy stays the only source of grid geometry. `RefinementHistory`
owns the float64 snapshot copies of the rotation posteriors and of the learned
priors (class 0 per half for K-class); the controller passes the prior owners.
The controller applies learned payloads explicitly. Tests that stub the collapse
step patch `orientation_priors`.

Snapshot initialization of those priors belongs to
`orientation_priors.initial_direction_priors_from_snapshot`: a RELION restart
carries the previous iteration's `pdf_direction` per half (per class for K-class),
normalized to the scoring dtype with the HEALPix order inferred from its length.
Scoring with those priors follows RELION through one owner,
[`orientation_priors.relion_direction_log_priors_for_half`](../../relax/sampling/orientation_priors.py),
which both the regular iterations and the final all-data pass call per half.
RELION (`ml_optimiser.cpp`) multiplies orientation weights by the class's
`pdf_direction` value at the sampled direction only in `NOPRIOR` mode, so local searches
get no direction prior; `initialisePdfDirection` resets every class to an even
distribution on a sampling change, so a prior at another HEALPix order is not
used; RELION keeps one prior per class, and a K-class run reads only its
per-class priors (the one-reference start copies class 0 to every class in the
controller; no Class3D path fills the K=1 `shared` field, so the former
shared-to-every-class fallback is removed); and each half scores with its own model, including the
joined final iteration. Sealed captured sampling expands the prior onto the
captured direction rows the scorer uses (`_sealed_direction_log_prior`, now
also owned here); otherwise the canonical sample ordering is used. The controller
supplies the scoring order, the sealed state only when the scored grid is the
sealed grid, and its logger. Adopting one rule made K-class priors follow sealed
rows in both passes. [`test_direction_log_prior_owner.py`](../../tests/unit/test_direction_log_prior_owner.py)
pins each rule.
`orientation_priors.relion_local_search_sigmas` owns the local-search
orientational prior widths (configured widths kept, psi falling back to rot,
twice the oversampled angular step when unset; `updateAngularSampling`) for
both passes ([`test_local_search_sigma_owner.py`](../../tests/unit/test_local_search_sigma_owner.py)).

The [refinement controller](../../relax/refinement/iteration_loop.py)
owns iteration history, half-set dispatch, sampling updates, convergence and
finalization scheduling/state mutation. The existing [sampling module](../../relax/sampling/__init__.py)
owns pure grid construction. Its helpers
`_relion_mstep_source_eulers` and `_perturbed_trial_grid` hold the two sampling rules
both passes share: the exact M-step rotations are seeded from the sealed grid's own
angles or RELION's canonical grid at the perturbation order (the scoring grid's
angles when the row counts differ), and one RELION `SamplingPerturbation` rotates
the trial orientations, rebuilds the M-step rotations and shifts the translation
grid. [`iteration_planning.build_initial_coarse_grids`](../../relax/refinement/iteration_planning.py)
materializes the first exhaustive grid from a caller translation table or the
RELION translation grid; `relax/parity/relion_replay_source.build_sealed_initial_coarse_grids` handles sealed
captures. [`prepare_final_sampling`](../../relax/refinement/trial_grids.py)
resolves final native/replay settings and returns ready-to-score grids together
with their metadata. It owns the final NumPy/JAX conversion boundary and preserves
rounding before perturbation. [`test_initial_coarse_grid_owner.py`](../../tests/unit/test_initial_coarse_grid_owner.py)
and [`test_final_sampling.py`](../../tests/unit/test_final_sampling.py) cover these
boundaries. `expected_accuracy.Half1AccuracyInputs` bundles the run-constant inputs of RELION's
expected-accuracy estimation and owns the `estimate` method used by both passes.
The same module owns `_expected_accuracy_class_ids`, which supplies half-1 class labels ([`test_expected_accuracy_inputs_owner.py`](../../tests/unit/test_expected_accuracy_inputs_owner.py)).
`sampling._advance_relion_perturbation` owns the update beside its seeded and generator primitives, and advances RELION's SamplingPerturbation to an iteration
(seeded `random_seed + iteration`, or the run's generator) for both passes
([`test_perturbation_advance_owner.py`](../../tests/unit/test_perturbation_advance_owner.py)).
The controller transports coarse scoring geometry separately from the effective
rotation grid used by both dense half-scoring routes
([`test_zero_coarse_geometry.py`](../../tests/unit/test_zero_coarse_geometry.py)). `_exact_local_fine_grid` materializes RELION's fine local-search grid
once with its perturbation and exact M-step rotations, and
`_local_search_mstep_rotations` reuses or rebuilds the M-step matrices of a
scoring grid; the final pass sizes its parent pass with
`resolution.relion_local_pass1_current_size` only under adaptive oversampling
([`test_local_fine_grid_owner.py`](../../tests/unit/test_local_fine_grid_owner.py)).
Grid tests substitute primitives at their sampling owner; sealed-state replay remains
at the refinement boundary. The exact local-search stage is implemented in
[`local_half`](../../relax/local_search/half.py).
That module builds local pose neighborhoods, asks
[`batch_planning`](../../relax/runtime/batch_planning.py) for
batch sizes, calls the selected kernel and returns `LocalSearchResult` (named accumulators,
pose fields and statistics) or, for a Class3D fine pass, `LocalClassSearchResult` (the
engine's class-segmented output).
The controller reads those fields directly.
The resident pass-2 operands in
[`sparse_pass2_scoring`](../../relax/fine_pass/scoring.py)
reproduce RELION's CUDA `powerClass` through one operand owner:
`_relion_powerclass_packed_image` (RELION's unshifted `Faux` layout and
amplitude convention), `_relion_powerclass_operands` (CUDA shell map and pixel
validity) and `_relion_powerclass_native_spectrum_highres` (native atomics),
shared by the JAX reproductions and the native wrappers
([`test_powerclass_operand_owner.py`](../../tests/unit/test_powerclass_operand_owner.py)).
`_relion_powerclass_noise_terms` selects the `highres_Xi2` and high-shell norm
terms the resident scoring and operand stages need. `_sparse_pass2_window_setup` builds the
forward-model configuration, score and reconstruction windows, RELION x-half reconstruction
indices and the windowed-prepare decision of a resident pass.
In [`k_class`](../../relax/classification/k_class.py),
`_override_class_assignments_with_coarse_winner` applies RELION's coarse-grid
binarization to a pass-2 result (winning class, that class's fine pose, decoded
best-pose details) for both adaptive pass-2 paths
([`test_kclass_results_owner.py`](../../tests/unit/test_kclass_results_owner.py)).

Replay and finalization have separate selection and mutation boundaries:

| Responsibility | Owner | Inputs and preserved behavior |
| --- | --- | --- |
| Final-pass admission | [`finalization.py`](../../relax/refinement/finalization.py) | Admission receives convergence/cap state and the controller logger and does not mutate state. The same owner executes final accuracy/sampling, expectation and reconstruction, with explicit final state/history writes. |
| Replay numbering and cutoff | [`relion_replay.py`](../../relax/parity/relion_replay.py) | `_numbered_relion_iteration` maps restart-local indices; `_native_sampling_boundary_for_iteration` checks cutoff and sealed state. The controller retains scheduling. |
| Numbered optimiser accuracy override | `relion_replay.read_optimiser_accuracy_replay` | Selects this iteration's numbered optimiser STAR when replay is active and unsealed; finite RELION rotation/translation accuracies replace the reported and convergence accuracies. Read or parse failures warn and keep values assigned before the failure. Returns `OptimiserAccuracyReplay`; the controller passes its metadata to `apply_optimiser_convergence_replay` after the state update. |
| Class3D captured tau2 selection | `relion_replay._class_tau2_replay` | Selects same-iteration captured spectra when the diagnostic is enabled, preserves fallback logging and validates captured shapes even when disabled. Returns spectra, enable flag and source label; M-step arithmetic stays in refinement. |
| Final override selection | `relion_replay._select_final_replay_override` | Receives the requested index, explicit override, recorded history and its already-computed presence flag. Returns an index and the original override object; no copying or state updates. |
| Final reference substitution | `relion_replay._prepare_final_replay_references` | Validates source iteration, K1 restriction, two-map count and shapes in order; casts each map to its half's dtype. With no maps, returns the original reference list. |
| Applying selected state | [`iteration_loop.py`](../../relax/refinement/iteration_loop.py) | Retains sigma, pose, corrections, noise and direction-prior updates in their original order, including casts and half-set handling. |

Read `_should_run_final_all_data_iteration` in decision order: forced-cap mode
rejects the extra pass first; otherwise convergence admits it. Without
convergence, the optional after-cap diagnostic can admit K1 only when the cap
has been reached. K-class still requires convergence. Every final map is
gridding-corrected, as in RELION.

Replay admission is also explicit. An initial-only override does not activate
numbered final replay. An explicit final diagnostic override wins, even an
empty dictionary; otherwise automatic replay requires numbered overrides or
its force flag and must not be disabled. Selection clamps to the final stored
slot when needed. A `None` slot in a nonempty history raises an error; an absent
history logs that no override exists. The selected dictionary retains identity.
These final-selection diagnostics use the controller logger passed by the caller.

The replay owner also validates physical BPref ordering for fresh/imported/sealed
boundaries, selects final sampling STARs and checks required files. Numerical
grid construction after selection remains in the controller. Diagnostic map
loading includes half/class and shared-class fallback files, with unchanged
casts and Fourier/frame conversion; these map-loading messages use the replay
module's logger. Review [replay-state tests](../../tests/unit/test_relion_replay_state.py)
and [controller tests](../../tests/unit/test_refine_relion_mode.py) for selection
identity, missing-slot errors, cutoff behavior and cold-start finalization.

[`numbered_reconstruction`](../../relax/refinement/numbered_reconstruction.py) owns two
M-step boundaries that the regular iterations and the final all-data pass used
to repeat inline. `join_half_accumulators_at_low_resolution` applies RELION's
`--low_resol_join_halves` to the K=1 half accumulators before the Wiener solve;
the join radius is capped by the last recorded shell resolution, a non-positive
recorded shell leaves it uncapped, and without history a finite state resolution
is used (a fresh run's `--ini_high` shell, seeded by `initialize_resolution_from_ini_high`). `_class_tau2_from_iref_power_spectrum`, `_class_tau2_update_details`
and `_stack_class_tau2_update_details` produce the Class3D per-class tau2
volume, the RELION- and RECOVAR-frame shells, data-vs-prior and the stacked host
detail record with `fsc_shells` left `None`. The controller keeps the enabling
conditions, the replay `class_tau2` branch, the round/floor weight statistics
and the dump calls, and passes the regular or final accumulators, current size
and layout explicitly. Both owners reach `regularization` through the module
attribute, so monkeypatched controller tests keep working. The final-iteration
cases in [`test_refine_relion_mode.py`](../../tests/unit/test_refine_relion_mode.py)
exercise the controller boundary, while
[`test_class_tau2_lowres_join_owner.py`](../../tests/unit/test_class_tau2_lowres_join_owner.py)
pins the owner arguments, dtypes and record layout.

The local kernel returns `LocalEMResult` from
[`helpers.types`](../../relax/types.py):
`Ft_y`, `Ft_ctf`, `hard_assignments`, `stats`, optional best-pose fields,
`noise_stats`, `profile` and `significant_counts`. All sixteen return-flag
combinations have the same field layout; disabled fields are `None`. The
local-search wrapper and K-class orchestration read these fields directly;
the positional packer and both decoders are removed. The result stores array
references without copying or synchronizing them.

Requesting reconstruction probabilities or sample IDs enables the engine's
profile that carries those captures. The wrapper still exposes a profile only
when requested, and copies its dictionary before adding wrapper timings.
Significant counts retain their own field even when that internal profile is
hidden. `tests/unit/test_local_search_result_contract.py` covers this routing,
plus K2/exact-K4 pose, noise and class-summary settings. Returned arrays retain
their layouts, dtypes and identities; saved refinement field names are unchanged.

Per-half dispatch belongs to
[`half_scoring`](../../relax/refinement/dense_half.py).
Its dense and local adapters prepare engine arguments, retain adaptive/first-CC
routing, and write class/pose fields into the caller-owned `PerHalfOutputs`.
Every global K=1 and K-class scoring call runs the adaptive/sparse engine at the requested
oversampling order, including 0, where its single coarse pass on the current grid is RELION's
single pass (RELION's `storeWeightedSums` accumulates the group-scale `XA`/`AA` sums and the
norm-correction residuals in every pass); positive oversampling keeps the two-pass adaptive
expectation (pass 1 at the current size when no reduced coarse size exists). The direct dense
engine (`run_em`) that served runs without scale groups at oversampling 0 was removed on
2026-10-03
([`test_dense_scale_group_routing.py`](../../tests/unit/test_dense_scale_group_routing.py)).
[`prepare_adaptive_pass2_grids`](../../relax/sampling/oversampling.py) materializes the perturbed coarse grid, the
oversampled children with parent maps, the fine M-step rotations and the coarse
translation phase source for both routes
([`test_adaptive_pass2_grids_owner.py`](../../tests/unit/test_adaptive_pass2_grids_owner.py)).
Both routes pass the same keywords to `run_dense_k_class_em_adaptive` (noise accumulation,
RELION's adaptive fraction, the fine M-step rotations pruned only for sparse pass 2), each at
its own call; the K=1 call adds
significance skipping, the diagnostic float64 pass 2 and the host-double coarse
translation phases, and the K-class call plans its own batches. Pass 2 always runs on the
device-resident sparse engine (the dense pass 2 and its `RELAX_K1_DENSE_PASS2` /
`RELAX_K_CLASS_DENSE_PASS2` switches were removed on 2026-10-03), and `_collapse_fine_pose_assignments_to_coarse` collapses fine pose
assignments onto the coarse grid when a fine pass ran
([`test_adaptive_engine_call_owner.py`](../../tests/unit/test_adaptive_engine_call_owner.py)).
[`heterogeneity._fixed_rotation_covariance_images`](../../tests/oracles/heterogeneity.py) accumulates the
fixed-rotation covariance-column update in image space (right-hand side and normal
operator per rotation) for both the Equinox and the classic accumulator, which only
convert to half images and back-project. In [`vdam.estep_common`](../../relax/vdam/estep_common.py),
`_centered_bpref_sources` validates and centers the data/weight cubes once for the
dense and the RELION-x-half BPref converters, and `_bpref_slab_outputs` applies RELION's
double-precision cast and denormal-weight clamp
([`test_covariance_rhs_and_bpref_source_owner.py`](../../tests/unit/test_covariance_rhs_and_bpref_source_owner.py)).
[`state_swap_runtime._apply_state_swap_probe`](../../relax/parity/state_swap_runtime.py)
returns a `_StateSwapValues` named tuple (current size, maps, tau2, noise, poses, sigma offset and
direction priors), which `RelionReplaySource.swapped_state` hands to the controller as a `ScoringState`
read by field; the unchanged value is built once from the
inputs and returned by both early exits
([`test_state_swap_values_owner.py`](../../tests/unit/test_state_swap_values_owner.py)).
[`k_class._PerClassSubsetResults`](../../relax/classification/k_class.py) collects the
per-class outputs of the dense and sparse firstiter-CC global-winner subset passes in class
order: a class without images gets zero accumulators, `-inf` best scores and zero posteriors,
and a scored class has its subset accumulators, statistics, noise and best poses expanded to
the full image axis. Each route states whether it hosts the appended accumulators
([`test_kclass_results_owner.py`](../../tests/unit/test_kclass_results_owner.py)).
[`scoring._e_step_block_score_components`](../../relax/scoring/coarse_kernels.py)
computes the two HIGHEST-precision GEMMs every dense scorer is built from (the cross term
`-2 Re(conj(shifted) . proj_weighted)` and the model energy `ctf2_over_nv . proj_abs2`);
the residual, windowed, normalized-CC and coarse Gaussian scorers only combine them
([`test_score_components_owner.py`](../../tests/unit/test_score_components_owner.py)).
[`projection._relion_projector_fftw_block`](../../relax/projection/projection.py)
projects one rotation block through RELION's Projector onto the clamped `2 r_max` (or
requested) square with the scorer rotations transposed at the handoff; the centered-row
projector reorders its rows and the indexed projector gathers its pixels from that block
([`test_projector_fftw_block_owner.py`](../../tests/unit/test_projector_fftw_block_owner.py)).
[`types.SparsePass2Output`](../../relax/types.py) carries sparse
pass-2 accumulators, poses and optional diagnostics in named fields. The resident driver
and the per-image reference path construct it directly; K-class callers no longer
need a positional decoder or flags to locate fields. Unrequested diagnostics are
`None`. The resident driver's independent check is the NumPy RELION E-step reference
([`test_resident_relion_reference.py`](../../tests/unit/test_resident_relion_reference.py)).
In [`significance`](../../relax/scoring/significance.py),
`_coarse_gaussian_ffi_default` applies the fresh-InitialModel coarse Gaussian FFI
default only when the supplied RELION projector operands exist; a dense pass
without a projector keeps the JAX coarse path and an explicit environment request
still fails closed.
`preprocessing.uses_relion_cuda_image_preprocessing` (with `relion_preprocess_backend`,
which follows subset parents) is the one detection of RELION's CUDA image path; the
local engine, the InitialModel adapter and the controller's early fresh-K=1 check use
it ([`test_relion_cuda_preprocess_owner.py`](../../tests/unit/test_relion_cuda_preprocess_owner.py)).
The controller calls its two BPref-scoped entry points and retains iteration
scheduling, state transitions, reconstruction and device-buffer lifetime.
[`scoring_policy`](../../relax/refinement/scoring_policy.py)
owns shared padding/window constants, the import-time static kwargs object,
and the existing call-time environment selectors. Their override precedence,
invalid-value handling and float32 defaults are preserved. The controller and
scorers share the same kwargs object; policy imports do not load engine modules.
These owners have no imports back into the controller. Log messages are unchanged,
with namespaces following the owner of each moved function.

Local searches (the fine pass and RELION's pass-1 parent probe) run on the device-resident
local pass, [`resident_local_pass2`](../../relax/local_search/resident_pass2.py).
[`k_class`](../../relax/classification/k_class.py) supplies adaptive
and local K-class orchestration.
[`k_class_inputs`](../../relax/classification/k_class_inputs.py) owns
class-axis validation, shared/per-class array selection and local prior layouts.
It imports no execution engines; engine-specific keyword filtering stays in `k_class`.
[`k_class_results`](../../relax/classification/k_class_results.py) owns
the shared result type, joint result assembly and host/device publication.
Accumulator offloading and scheduling stay in the orchestrator.
Class evidence and posterior mass
must be handled at the K-class level, not inferred from independently normalized
single-class probabilities.

Scale-group ID validation and full-axis sizing have one host owner,
[`helpers/scale_groups.py`](../../relax/relion/scale_groups.py).
Local EM, both sparse scorers and the K-class subset router use it. Explicit
counts retain groups absent from a class subset; missing IDs disable engine
scale-statistics allocation, while routing still retains an explicit count.
Empty ID arrays retain the existing one-group convention. Engine callers check
the flattened image axis; the router has no image-count constraint. The helper
preserves existing casts and errors and imports independently of execution.

Local projector slab normalization has one owner,
[`relion_projector_setup.prepare_local_projector_slab`](../../relax/relion/projector_setup.py).
Bucket projection, packed-noise projection and the main BigJIT path accept the
same three-dimensional slab or singleton class axis. The helper preserves JAX
dtype conversion and path-specific errors. Radius requirements, pixel selection,
interpolation, masking and projection execution stay with the callers.

The exact coarse Gaussian path in
[`helpers/significance.py`](../../relax/scoring/significance.py)
passes its existing host pixel indices to the shared source-precision CTF owner
[`helpers/relion_ctf.py`](../../relax/relion/ctf.py).
Coarse, local and sparse scoring call that owner directly; its parsed STAR
tables and its one float64 CTF program remain independent of the execution engines.
It evaluates the requested rows at the requested pixels on the device at every
call and caches none; index order and duplicates are preserved. Omitting pixel
indices evaluates the full grid. Only scoring operands are compacted: full-image
powerClass inputs, source precision, scale correction and padding semantics
remain intact. The [current evidence](em_status.md) separates operand equivalence
and the allocation microbenchmark from pending full-runtime/trajectory checks.

The production fine-grid significance mask is lazy: `_ClassFineGridSignificanceMask`
and `_PerClassFineGridSignificanceMask` generate only the requested image/rotation
block. The materialized NumPy comparison lives in
[`tests/helpers/fine_grid_significance_reference.py`](../../tests/helpers/fine_grid_significance_reference.py).
It has no production callers and retains a separate mask-building algorithm
for checking lazy blocks and explicit/complement coarse support.

[`score_outputs`](../../relax/refinement/score_outputs.py) owns
the scoring containers and class/coarse-grid result adapters. It also owns
optional half-accumulator combination, shape/axis resolution and profile-row
recording. The controller retains scheduling and device-buffer offloading.
`HalfScoreResult` carries one halfset's common scoring output.
`PerHalfOutputs` owns separate two-slot lists for a scoring phase; slot 0/1
always selects the halfset, including for class-related fields. Image arrays
retain each halfset's local order and size. The K-class adapters populate class
assignments, posterior summaries and per-class noise statistics separately from
`update_from`. That method preserves existing optional pose fields when the
new result omits them, while always replacing accumulator-layout metadata.

Local-search dependencies are imported from their owners. Tests that replace a
kernel for a local dispatch check patch its binding in `local_search_iteration`.
`half_scoring` has its own active `build_local_hypothesis_layout` binding for
adaptive parent-layout construction. Tests of whole-controller dispatch patch
engine bindings in `half_scoring`; tests of local chunk execution patch
`local_search_iteration`. A shared sizing function mocked across both the
controller and scorer must be patched at both consumers. Patch the call site
exercised by the test; do not add reverse imports to preserve an old monkeypatch
location.

[`diagnostics.iteration`](../../relax/diagnostics/iteration.py) owns the
half-selection policy for terminating significance/noise captures and numbered
BPref device captures, together with iteration dump writers. The controller
calls those selectors at the same dispatch boundaries and retains diagnostic
completion/stop control. Selector errors and log messages are unchanged; log records identify the
module that owns each operation. Capture state and counters belong to the
separate diagnostic owner below.

[`diagnostics.reconstruction`](../../relax/diagnostics/reconstruction.py)
serializes pre-mask maps, K-class current-size/M-step and tau2-update NPZ
captures. Refinement callers retain the environment gates and call
boundaries; writers preserve historical fields, casts and optional entries.

[`helpers.coarse_score_diagnostics`](../../relax/diagnostics/coarse_score_diagnostics.py)
owns host NumPy summaries of direct/GEMM score deltas, ULPs, winner margins,
support changes, repeated runs and scale panels, plus the qualification decision.
It also validates selector audits, hashes exact support and attaches diagnostic
profiles to results. Scoring, K-class/InitialModel controllers and reporting
scripts import these helpers directly; saved-audit validation does not require
loading `significance`. `significance` calls this owner directly. JAX scoring/M-step kernels remain in
`helpers.scoring`; diagnostic imports do not initialize those execution modules.

The compact engine's K1/K-class score dumps and normalization-residual captures
(`diagnostics.pass2`, `diagnostics.norm_scale`) were deleted with it (2026-09-26); the
resident drivers refuse the dump environment variables. The diagnostics package itself
imports no engines or capture modules; import specific writers directly.

K1 and fused K-class preprocessing captures share
`bpref_diagnostics.build_bpref_preprocess_capture`. Callers retain the raw
preprocessing tuple to preserve operand lifetime and invoke the builder at the
original capture gate; schema, defaults, masks and dtype casts stay unchanged.

[`helpers.bpref_diagnostics`](../../relax/diagnostics/bpref_diagnostics.py)
owns the numbered-half capture context, contribution and membership counters,
membership selectors/rotation-mass writers, device-panel state, capture validation
and artifact writers shared by sparse and exact-local EM. Fused K-class capture-row
materialization lives here beside the signature/shadow consumer; compact-pair
expansion, selected-row order and reconstruction-posterior fields are preserved. The controller and replay scripts set and clear that context through this
owner. Sparse scoring retains candidate planning, numerical kernels and live
accumulation; it asks the diagnostic owner for scoped capture decisions. The
diagnostic module has no direct import of sparse scoring or the iteration
controller. The package initializer exposes options and sampling/statistics
helpers; import K-class execution from `k_class.py` and result assembly/types
from `k_class_results.py`. The historical result type alias in `k_class` preserves
pickle compatibility; importing the new result owner does not load engines.
Standalone helper imports do not load dense/local engines or sparse scoring.
Tests replace capture functions and state at this owner, including optional
native signature panels. Dump schemas, precision, counter order and error
behavior remain unchanged. The boolean parser is shared through
`helpers.env_flags.parse_env_flag`; file identities use `utils.file_hash`.

Strict local capacity/packing selectors call `helpers.env_flags.parse_env_binary_flag`
directly. It accepts only `0` and `1` after stripping whitespace, defaults to
false when unset, and rejects blank or textual boolean values. Its behavior
differs from the permissive diagnostic parser above; do not interchange them
during structural cleanup. The engine retains the order of reads and mode checks.

Unused constant copies in `iteration_loop` have also been retired. Batch and
raw-image-cache limits, first-iteration reconstruction caps, dense K-class
hypothesis budgets and adaptive pass plans belong to `batch_planning`.
`firstiter_cc` constructs the first-iteration coarse/fine grids; the fine-grid
precomputation limit belongs to `local_search_iteration`. Their values and
environment overrides are unchanged.

The maintained dense PPCA runner uses `ppca_refinement.refinement_loop` for its
halfset resolution gate. The dense-to-local workflow calls the single-iteration
dense and exact-local owners directly, so it does not carry a second schedule
controller. Helper-only callers import sign alignment and combined noise
statistics from `mean_helpers`, rotation metadata from `relion_metadata`, and
replay iteration mapping from `relion_replay`.

[`relion_normalization`](../../relax/relion/normalization.py)
owns per-image norm and per-group scale formulas and their result type. It
depends on NumPy/JAX, not the controller, mean reconstruction or follower
dispatch. The controller retains state installation and temporary lifetimes;
`relion_worker_scale` handles follower-specific corrections. The seven formula
tests live in `tests/unit/test_relion_normalization.py` and import this owner
directly. The old `mean_helpers` normalization exports are removed.

Dense and local scoring share `orientation_priors.relion_translation_prior_center`;
the duplicate `relion_local_translation_prior_center` entry point has been removed.
`orientation_priors.relion_half_translation_prior_inputs` builds one half-set's
score and sigma-offset prior centers, the zero-centered cold-start engine
center and the prior translation grid for both the regular iterations and the
final all-data pass; the controller keeps the search base, the per-half sigma
offset and the dense-only score log-prior call. The local adapter receives
its own center array, and the engine center aliases the sigma center.
Both use `(prior - rounded_old_offset) / pixel_size`, as before. The separate
`relion_sigma_offset_prior_center` serves sufficient statistics and keeps its
pixel-space formula without that division.

The former runtime `compute_e_step_weights` API had only test consumers.
Its materialized dense posterior implementation is preserved in
[`tests/helpers/dense_posterior_reference.py`](../../tests/helpers/dense_posterior_reference.py).
The adaptive-oversampling tests still compare its complete posterior with active
significance paths. This reference keeps separate orchestration but shares
production preprocessing/scoring kernels; it does not independently validate
those kernels. Production significance belongs to `helpers/significance.py`.

Pure convergence-policy cases live in `tests/unit/test_convergence.py`; collecting
this module does not import the iteration controller. Full iteration smoke tests
remain in `test_refine_relion_mode.py`.

The first-iteration winner-take-all dispatcher lives with its grid builder in
`dense_half.py`. It calls the batch planner and K-class engine directly for
both K=1 and K-class scoring. The controller supplies its logger and chooses
whether the batch clamp also updates the caller’s argument dictionary.

Precision selectors belong to `helpers/dtype_policy.py`. The controller passes
its existing static argument mapping; diagnostic iteration selection still
reads the environment at call time. Moving the selectors does not evaluate a
second set of import-time defaults or change any selected dtype.

Angular-grid order policies belong to `helpers/convergence.py`: exhaustive-grid
capping, final parent/fine orders, perturbation order and direction-prior order.
The controller supplies the active state and captured final-sampling metadata;
the helpers preserve their distinct order choices.

For an extraction, identify the actual boundary first: array layout, casts,
reduction order, JIT scope, device placement, buffer ownership and returned
statistics. Preserve those contracts during structural cleanup. The
[EM development guide](../../relax/AGENTS.md) and
[mathematical algorithm map](../math/relion_refinement_algorithm.md) describe the
validation ladder and scientific state transitions.

## Diagnostics and reusable evidence

Diagnostic scripts compare specific captures, layouts and policies. Similar
names or similar-looking reductions are insufficient evidence of duplication.
Keep an independent numerical reference separate from the implementation it tests.

[`tests/helpers/sparse_pass2_test_support.py`](../../tests/helpers/sparse_pass2_test_support.py)
retains six former runtime helpers used only by tests: cached/packed scoring
variants, a lane-tree wrapper, pair normalization and materialized noise-row
gathers. These share production primitives and are comparison utilities, not
independent numerical oracles. The tests keep their separate NumPy references;
production imports no test helpers.

Common transport and command mechanics have narrow owners:

- [`file_hash.sha256_file`](https://github.com/ma-gilles/recovar/blob/dev/recovar/utils/file_hash.py) hashes files in
  8 MiB blocks for RECOVAR-dependent diagnostics. A hash alone does not make a
  mutable file immutable or validate a manifest.
- [`json_utils.to_jsonable`](https://github.com/ma-gilles/recovar/blob/dev/recovar/utils/json_utils.py) converts NumPy
  values, paths and nested containers. Finite-value and report acceptance rules
  remain the caller's responsibility.
- [`scripts.file_hash.sha256_file`](../../scripts/file_hash.py) supplies the same
  8 MiB hashing contract to standalone diagnostics without importing RECOVAR or
  JAX. Keep the runtime helper in the installed package and this script helper
  usable through both direct entry points and package imports.
- [`scorecard_cli`](../../scripts/scorecard_cli.py) supplies print/write/check
  handling for compatible historical scorecards. Each renderer retains its own
  fixed case inventory, validator and Markdown format. These CLIs remain usable
  without importing the scientific environment.

Use `python -m scripts.<name>` from the checkout for diagnostics that import
other script modules. Some older direct-file entry points still fail their
imports; the current review records that debt rather than treating failed help
commands as successful checks.

The [benchmark contract](benchmarks.md) defines source, fixture, library,
quality and performance evidence. [Current EM status](em_status.md) separates
the selected source from historical results and records open qualification gaps.
Historical scorecards describe their pinned runs; they do not qualify a new
checkout merely because the same report can still be rendered.

### Replay state diagnostics

`helpers/state_swap_probe.py` owns the supported component variants and CLI
validation. It can enumerate variants without importing the refinement
controller. `helpers/state_swap_runtime.py` owns snapshot copying, map-amplitude
scaling and restoration of the selected components. The controller still owns
when the snapshot is taken and applies it after the RELION replay override.

Snapshots preserve the existing ownership contract: array inputs are copied,
while `state_fields` is a shallow copy of `state.__dict__`. Changing that
ownership, the ordered return tuple or the restoration sequence requires its
own behavior review. The in-memory scoring-state inventory and overwrite guard are owned by
`parity/frozen_boundary.py`, alongside the sealed-boundary loader.
The controller takes and checks those snapshots at the existing boundaries.

Captured sampling grids belong to `parity/relion_replay.py`, which
also applies replay state overrides. Its helpers construct Euler/translation
grids, canonical coarse rotation IDs and direction log priors directly from
sealed sampling metadata. They preserve the recorded direction/psi order and
convert translations from Angstroms to pixels using the supplied voxel size.
The controller selects when to use these grids.

Refinement now receives one `RefinementOptions` container. Its groups own
scheduling, adaptive search, parity behavior, local search, class setup, replay,
diagnostics and batching. `refinement_options.with_validated_sampling_schedule`
owns explicit current-size/HEALPix schedule admission and shallow option copies;
the entry point invokes it before starting the loop. Option construction does
not trigger these checks. `helpers/iteration_history.py` owns the per-iteration
history lists and their established result-dictionary keys. Its noise/tau2 recorder
also owns host diagnostic formatting: it prepares all float64 shell fields
before appending, retains aliases to existing float64 shell arrays, and stacks
the halves into a fresh array. `noise_updates._noise_radial_history` constructs
the radial history used by initialization and replay; pixel-noise normalization
and estimation remain with their callers. The controller
still chooses when each snapshot is recorded.

Precision is explicit at extracted boundaries: replay grids, resolution
curves and scoring-output adapters receive the caller's dtype. These helpers
do not import the controller to discover runtime settings. PR180's numerical
changes and their qualification state are tracked on the
[EM status page](em_status.md).

`helpers/convergence.py` owns angular-refinement state transitions, including
validation and application of explicit HEALPix schedules used for oracle runs.
The controller selects the iteration's requested order; the convergence helper
advances through the existing angular and translation updates without coarsening
an active state.
It also owns the approximate-accuracy convergence gate and its environment
overrides. The controller supplies its logger so malformed-override warnings
keep their existing routing.

`helpers/resolution.py` owns current-size growth inputs and first-iteration
resolution rules: the inclusive FSC/data-vs-prior boundary, raw versus corrected
K1 scheduling, the initial high-resolution cutoff, and the tau2 reporting taper.
It also owns expectation-boundary coarse sizing: replay selects the incoming
HEALPix order, and local pass 1 sizes its Fourier window from that order while
child expansion retains the updated order. Callers import these policies directly.
Initial FSC/low-pass resolution seeding also lives here, with explicit FSC dtype
and the shared ini_high shell calculation. The controller retains replay/FSC/
low-pass precedence; seeding preserves input copies and state-assignment order.
The controller retains their timing within the refinement loop. The pure
scheduling cases live in `tests/unit/test_resolution_scheduling.py`; the
reconstruction/taper ordering check remains with the controller tests.

`refinement/projector_preparation.py` prepares RELION reference slabs
for the controller's scoring calls. It owns native reference conversion, cache
keys and files and optional dumps. (Replaying a captured RELION `Projector::data`
was retired; its code is at git tag `retired/captured-projector-20261006`.)

Captured sampling-state tests live in `tests/unit/test_relion_replay_state.py`.
They exercise the replay owner directly, including immutable copied arrays and
suppression of external metadata reads. End-to-end controller behavior remains
in `test_refine_relion_mode.py`.

Import execution entry points explicitly from their owners:

```python
from relax.refinement.iteration_loop import refine_single_volume
from relax.classification.k_class_results import KClassEMResult
from relax.classification.k_class import run_dense_k_class_em_adaptive
```

The package initializer does not re-export these names. The CPU fast guard
checks that importing replay, normalization, projector, result and diagnostic
helpers leaves the controller, K-class orchestration, dense/local engines and
sparse scoring unloaded. Existing callers already import from these owners;
the definitions and their serialized module identities are unchanged.

## Ground-truth reporting

[`diagnostics/gt_registration.py`](../../relax/diagnostics/gt_registration.py)
owns the optional CPU rigid fitter and immutable fit-once transform. The existing
[`gt_metrics.py`](../../relax/diagnostics/gt_metrics.py) keeps its legacy
rotation-only alignment API and result type. The reporting CLI
[`evaluate_ab_initio_gt.py`](../../scripts/evaluate_ab_initio_gt.py) opts into the
new fitter or applies a saved transform without fitting. [The reporting guide](gt_reporting.md)
explains geometry, common-frame comparisons and limitations. These are diagnostic
reporting tools; E/M execution, precision and quality gates are independent.

Noise initialization, half-set aggregation and posterior updates live in
[`refinement/noise_updates.py`](../../relax/refinement/noise_updates.py).
The refinement controller and replay diagnostics import that owner directly;
volume reconstruction remains in `refinement/numbered_reconstruction.py`.
