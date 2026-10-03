# Final-search preparation: first review proposal

Geometry design is superseded by the
[grid ownership proposal](grid_ownership_proposal.md). The branch and consumer
inventory below remains evidence; the proposed image-sizing container is not
approved for implementation.

Start with the [worked before/after example](final_search_review_example.md)
for a complete small calculation, its input construction, and its consumer.
This document describes the broader local/global interface proposal.

Proposed design, September 30, 2026; awaiting user review. Source inspected at
`f17505a220769e488f4cf3eceb33ef8a7831ea27` on `main`, with user comments
preserved. No production edits, imports of numerical modules, numerical tests,
or GPU jobs were performed for this proposal. Apply the
[agreed principles](refactor_principles.md).

The first extraction is the section in
[iteration_loop.py](../../relax/refinement/iteration_loop.py) beginning at
`final_use_local` (line 5296) and ending before projector construction (line
5413). It can use explicit inputs and distinct local/global results. The local
result can contain the existing `LocalSamplingSpec` directly. Global preparation
computes only adaptive image sizes; construction of `DenseSamplingSpec` remains
at its existing scoring call site.

## Calling code to review first

This is proposed Python, with the signatures and fields specified below.
Existing logging statements retain their text and position within each branch;
`log_final_local_search` and `log_final_global_search` below denote the visible
logging blocks, not proposed forwarding functions.

```python
final_use_local = bool(
    (not k_class_enabled)
    and state.do_local_search
    and all(e is not None for e in relion_half_inputs.previous_best_rotation_eulers)
    and all(t is not None for t in relion_half_inputs.previous_best_translations)
)
# Keep this calculation before the branch, including its global-mode behavior.
final_sigma_rot, final_sigma_psi = relion_local_search_sigmas(
    state.sigma_rot,
    state.sigma_psi,
    use_local=final_use_local,
    healpix_order=state.healpix_order,
    adaptive_oversampling=state.adaptive_oversampling,
)
final_image_sizing = FinalSearchImageSizing(
    original_size=grid_size,
    current_size=final_current_size,
    pixel_size_angstrom=cryo.voxel_size if cryo.voxel_size > 0 else 1.0,
    particle_diameter_angstrom=particle_diameter_ang,
)
if final_use_local:
    final_search = prepare_final_local_search(
        TrialOrientations(
            rotations=final_effective_rotations,
            eulers_deg=final_effective_rotation_eulers,
            mstep_rotations=final_effective_mstep_rotations,
            healpix_order=final_current_healpix_order,
            symmetry=symmetry,
        ),
        settings=FinalLocalSearchSettings(
            state_healpix_order=int(state.healpix_order),
            adaptive_oversampling=int(state.adaptive_oversampling),
            replay_parent_order=(
                int(final_perturbation_healpix_order)
                if final_sampling_star is not None else None
            ),
            perturbation=(
                final_random_perturbation if final_perturbation_applied else None
            ),
            precision_iteration=final_sampling_relion_iteration,
        ),
        image_sizing=final_image_sizing,
        sigma_rot=final_sigma_rot,
        sigma_psi=final_sigma_psi,
        current_translations=final_current_translations,
        base_translations=final_base_translations,
        disc_type=options.disc_type,
    )
    # Existing local-search logger.info block, using final_search fields.
else:
    # Existing dense-global logger.info block stays BEFORE sizing arithmetic.
    final_search = prepare_final_global_search(
        healpix_order=final_current_healpix_order,
        adaptive_oversampling=int(state.adaptive_oversampling),
        k_class_enabled=k_class_enabled,
        image_sizing=final_image_sizing,
    )
    # Existing adaptive K-class logger.info block, only when sizes are present.

# Projector construction, precision resolution and half loop stay here.
```

Use `final_search.sampling` directly as the `sampling=` argument of
`_score_half_local_in_bpref_scope`; it is built once and shared read-only by the
two half calls. Use `final_search.pass1_step_deg` in `_optics_group_kwargs`.
For tomography, select fields explicitly:

```python
local_sampling = final_search.sampling  # Inside the existing local/tomo branch.
sampling = local_tomo_sampling(
    fine_order=int(local_sampling.local_search_order),
    oversampling_order=int(state.adaptive_oversampling),
    translation_range_px=final_translation_range,
    translation_step_px=final_translation_step,
    voxel_size=cryo.voxel_size,
    random_perturbation=local_sampling.local_search_random_perturbation,
    pass1_size=local_sampling.local_pass1_current_size,
    current_size=final_current_size,
)
```

The existing tomography rejection when `final_use_local` is false stays intact.
In global scoring, keep the existing `DenseSamplingSpec(...)` construction;
replace only `DenseVariantPolicy.firstiter_coarse_current_size` and
`firstiter_fine_current_size` with `final_search.pass1_current_size` and
`pass2_current_size`. The global direction-prior call currently reads the
misnamed `final_local_parent_oversampling_order`: replace that read with
`int(state.adaptive_oversampling)`, without changing its value. Local precision
and adaptive-support consumers read
`final_search.sampling.local_parent_oversampling_order`.

## Actual proposed types and signatures

Place these small inputs/results and the two computational functions in the
existing [half_scoring.py](../../relax/refinement/half_scoring.py), alongside
`LocalSamplingSpec`. This is the current owner of the consuming sampling
interfaces and already imports sampling functions and the local-iteration owner.
Add direct imports from `sampling`, `helpers.convergence`, `helpers.resolution`,
`helpers.dtype_policy`, and the existing local-grid owner for the extracted work;
none imports `half_scoring` back. It avoids a new module and moving types across
the engine import graph.
The helper performs grid preparation and sizing; it does not forward scoring.
No changes to regular iteration preparation are proposed in this first example.

```python
@dataclass(frozen=True, kw_only=True)
class TrialOrientations:
    rotations: object                 # (R, 3, 3), existing NumPy/JAX operand
    eulers_deg: object                # (R, 3), existing canonical/source rows
    mstep_rotations: object | None    # (R, 3, 3), absent on unperturbed path
    healpix_order: int                # Declared exhaustive trial-grid order
    symmetry: str                    # Existing point group, unchanged

@dataclass(frozen=True, kw_only=True)
class FinalSearchImageSizing:
    original_size: int                # Model grid width, pixels
    current_size: int                 # Final scoring Fourier window, pixels
    pixel_size_angstrom: float        # Existing nonpositive-voxel fallback applied
    particle_diameter_angstrom: float | None

@dataclass(frozen=True, kw_only=True)
class FinalLocalSearchSettings:
    state_healpix_order: int
    adaptive_oversampling: int
    replay_parent_order: int | None   # None means use the optimiser state order
    perturbation: float | None        # None means no perturbation was applied
    precision_iteration: int

@dataclass(frozen=True, kw_only=True)
class LocalFinalSearch:
    sampling: LocalSamplingSpec
    parent_order: int                 # Logging metadata; never inferred from R
    pass1_step_deg: float | None      # None preserves ordinary optics sizing

@dataclass(frozen=True, kw_only=True)
class GlobalFinalSearch:
    pass1_current_size: int | None
    pass2_current_size: int | None
    # Both None for ordinary global scoring; both sizes for adaptive K-class.
    # Construct only through prepare_final_global_search in this first scope.


def prepare_final_local_search(
    trial: TrialOrientations,
    *,
    settings: FinalLocalSearchSettings,
    image_sizing: FinalSearchImageSizing,
    sigma_rot: float,
    sigma_psi: float,
    current_translations: object,
    base_translations: object,
    disc_type: str,
) -> LocalFinalSearch: ...


def prepare_final_global_search(
    *,
    healpix_order: int,
    adaptive_oversampling: int,
    k_class_enabled: bool,
    image_sizing: FinalSearchImageSizing,
) -> GlobalFinalSearch: ...
```

`LocalSamplingSpec` is reused without an extra grid conversion layer. Populate
its unchanged fields as follows: the computed grid/order/sizing fields from the
branch table below; `sigma_rot` and `sigma_psi` from the explicit arguments;
`current_translations`, `base_translations` and `disc_type` from their arguments;
`cs_for_engine=image_sizing.current_size`; `symmetry=trial.symmetry`;
`model_current_size_for_engine=None`, as today. Its existing annotations for
`local_search_rotations` and `local_search_angular_sampling_deg` should describe
the existing `None` meanings (`object | None` and `float | None`). No new
materialization policy, copies, casts or serialization is implied by these types.

The new `TrialOrientations` is justified because existing
`sampling._PerturbedTrialGrid` is a *producer result for an applied perturbation*:
it has mandatory M-step matrices and translations but lacks order/symmetry.
This section also handles unperturbed trials with missing M-step matrices.
`InitialCoarseGrids` similarly lacks M-step matrices and symmetry and describes
startup. Do not relabel either as a generic final-search input.
`LocalSearchGridSpec` describes per-image neighborhoods with prior poses and
translation priors, unavailable until the half loop. Its iteration result type
contains accumulators and statistics, so neither fits preparation.
`DenseSamplingSpec` requires mutable `state`, reconstruction/engine choices,
and translations; passing it to a function that needs only angular order and
image geometry would hide unused dependencies.

## Branch mapping and algorithm inside the helpers

Local preparation first calls `_final_local_sampling_orders` with the three
order settings, and compares the existing trial row count with
`rotation_grid_size(fine_order, symmetry=...)`. The count is an existing reuse
check, not a substitute for declared order metadata. Keep the current
conditional omission of the symmetry keyword for `C1` where owners/test doubles
currently depend on it.

| Condition | Grid result and work | Pass-one result |
| --- | --- | --- |
| Trial row count matches fine grid | Reuse `trial.rotations`; call `_local_search_mstep_rotations(trial.mstep_rotations, trial.eulers_deg, fine_order, ...)` | Full current size; step `None` |
| Count differs, oversampling is zero, precompute limit permits grid | Compute fine angular step; resolve fine-grid precision at this point; call `_exact_local_fine_grid` with the existing dtype selection and `settings.perturbation`; discard its returned Euler rows exactly as today | Full current size; step `None` |
| Count differs, oversampling is zero, limit forbids grid | Rotations/M-step matrices `None`; angular step present; perturbation `0.0` if absent, otherwise existing `float(...)` conversion | Full current size; step `None`; never parent sizing |
| Count differs, oversampling is positive | Defer grid regardless of precompute limit; rotations/M-step matrices `None`; fine angular step and deferred perturbation supplied | Compute `healpix_angular_step(parent_order)` and `relion_local_pass1_current_size` with existing geometry |

Default local deferred perturbation remains `0.0` and fine angular step remains
`None` on the reused-grid branch. On the eager branch perturbation goes into the
materialized arrays, while the downstream deferred-perturbation scalar remains
`0.0`. Preserve that distinction, including applied-zero versus absent
perturbation. `None` rotations request per-image generation by local layout;
materializing them would change both work and memory use.

Global preparation returns `(None, None)` except when K-class is enabled and
adaptive oversampling is positive. Only then call `compute_coarse_image_size`
with `healpix_angular_step(healpix_order)` and the existing geometry, followed by
`clamp_relion_coarse_image_size`; return that pass-one size and the full current
pass-two size. K1 global with positive oversampling still returns `(None, None)`.

## Numerical and resource invariants

Preserve the selected replay order separately from the state order: final STAR
metadata can advance the local parent order, but the angular-sigma calculation
still uses `state.healpix_order` today. Replay file selection, expected-accuracy
updates, random advancement and `_perturbed_trial_grid` remain before this
section. Projector construction and the later per-pass precision resolution
remain after it. Settings resolution must retain existing environment precedence
and branch timing; this proposal does not introduce unconditional precision reads.

Keep each existing NumPy/JAX operation, dtype, cast, conditional grid allocation,
source Euler operand and M-step helper call. Do not reconstruct Euler rows from
matrices. Construct containers by referencing arrays only; they perform no
validation that imports numerical modules, array copies or host/device transfers.
There is no state mutation in preparation and no `result.apply(...)` method.
The new local sampling object replaces the two short-lived spec constructions;
it holds only arrays already retained by this section, never scoring outputs or
projectors. Full finalization and its accumulator release order are outside scope.

One existing defect needs separate handling before claiming eager-branch
coverage: [iteration_loop.py](../../relax/refinement/iteration_loop.py), line
5341, calls `_local_search_precision_flags(..., pass_index=2)` without its
required keyword-only `static_em_kwargs` argument
([dtype_policy.py](../../relax/helpers/dtype_policy.py), line 231). Other final
precision calls supply `_DENSE_EM_STATIC_KWARGS`. Source inspection predicts a
`TypeError` on the eager mismatch branch; it was not executed here. A pure
extraction must preserve the existing failure, or the primary must qualify a
separate repair first. Silently adding the argument and calling the batch a
behavior-preserving refactor would hide a correctness change.

## Affected checks for the implementation

The following are the actual maintained tests/callers to inventory and migrate.
No full-suite run is proposed for this design document.

| Existing inventory | Relevance |
| --- | --- |
| [test_final_local_pass1_size_guard.py](../../tests/unit/test_final_local_pass1_size_guard.py) | AST guard for non-expanded deferred pass-one size; move its target to the preparation owner and retain the semantic guard |
| [test_local_fine_grid_owner.py](../../tests/unit/test_local_fine_grid_owner.py) | Numerical grid/M-step owner tests plus AST call counts/symmetry propagation; count regular and final owners together after extraction |
| [test_refine_relion_mode.py](../../tests/unit/test_refine_relion_mode.py), lines 567–591 and 8742–9630 | Final parent-order fallback; half maps, regularization, priors/noise; final local branch; exactly K4 and sparse adaptive K-class routing |
| Same file, lines 12749–13527 | Deferred parent-expanded grid, perturbation, local translation-prior modes and oversampling-zero support; helper monkeypatch ownership changes where used |
| [test_relion_replay_state.py](../../tests/unit/test_relion_replay_state.py), lines 323–360; [test_final_boundary_factorial.py](../../tests/unit/test_final_boundary_factorial.py) | Final sampling source selection and replay group/settings boundaries |
| [test_em_symmetry.py](../../tests/unit/test_em_symmetry.py), lines 790–795; [test_optics_shapes.py](../../tests/unit/test_optics_shapes.py) | Symmetry-aware precompute threshold and local parent-window/optics sizing |
| [test_resident_tilts.py](../../tests/unit/test_resident_tilts.py), line 259 | Tomography adaptation with absent parent-pass sizing |
| [test_relion_worker_scale.py](../../tests/unit/test_relion_worker_scale.py), line 962 | Source-order guard locating final dispatch before local selection; migrate if branch spelling changes |
| [tests/helpers/refinement_specs.py](../../tests/helpers/refinement_specs.py) and [test_firstiter_cc_batch_budget.py](../../tests/unit/test_firstiter_cc_batch_budget.py) | Existing `LocalSamplingSpec` construction; annotation-only changes should preserve it |

Add a focused behavioral preparation check for the four local branch rows and
ordinary global/adaptive K4 sizing. Verify array identity on reuse, selected
order and symmetry, deferred `None`, call suppression under parent expansion,
pass sizes, and preserved perturbation placement; compare floating operands
with existing approved bands. A targeted eager-branch failure/repair check is
needed for the defect above. Do not retain source-count assertions that require
these operations to live inside `refine_single_volume`.

Implementation then requires the applicable engine smoke/EM checks, using a
frozen control and both K1 and exactly K4 coverage. No tolerance change, baseline
regeneration, NumPy-to-JAX migration or performance claim is authorized by this
proposal. The current checkout lacks `.pixi/envs/default/bin/python`; the primary
must establish the frozen environment before numerical validation.

The remaining user-review decision is whether this concrete calling flow and
reuse of `LocalSamplingSpec` is readable enough to implement as the first
example. The eager-branch defect requires a separate integration decision by
the primary. Full finalization remains a later proposal after this example is
reviewed and validated.
