"""Prepare numbered and final half expectations without publishing model updates.

See docs/math/relion_refinement_algorithm.md, section 3.
"""

import logging
import os
from dataclasses import dataclass, field, replace
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from relax.dense.score_outputs import (
    HalfScoreResult,
    PerHalfOutputs,
    _maybe_host_offload_half0_local_accumulators,
    _record_score_profile,
    empty_half_result,
)
from relax.dense.scoring_policy import local_precision
from relax.diagnostics import bpref_diagnostics
from relax.diagnostics import parity_dump as _parity_dump
from relax.helpers.dtype_policy import _diagnostic_float64_pass2_matches
from relax.helpers.host_memory import return_freed_heap
from relax.helpers.orientation_priors import (
    DirectionPrior,
    HalfDirectionLogPriors,
    HalfTranslationPriorInputs,
    make_relion_translation_log_prior,
    relion_direction_log_priors_for_half,
    relion_half_translation_prior_inputs,
    relion_translation_search_base,
)
from relax.helpers.resolution import ImageGeometry
from relax.refinement import half_inputs, iteration_planning, local_sampling, optics_shapes
from relax.refinement.expectation_batches import BatchPlanner, prepare_half_batches
from relax.refinement.final_sampling import FinalSampling
from relax.refinement.half_scoring import (
    DenseBatchPolicy,
    DenseExecutionPolicy,
    DensePriorSpec,
    DenseSamplingSpec,
    DenseVariantPolicy,
    HalfScoringData,
    LocalBatchPolicy,
    LocalDiagnosticPolicy,
    LocalExecutionPolicy,
    LocalPriorSpec,
    _score_half_dense_in_bpref_scope,
    _score_half_local_in_bpref_scope,
)
from relax.refinement.optics_shapes import OpticsSpec
from relax.refinement.ports import DenseHalfScored, ExpectationProbe
from relax.refinement.refinement_options import RefinementOptions
from relax.refinement.tomo_half import TomoSampling, numbered_iteration_tomo_sampling
from relax.refinement.tomo_half import score_tomo_half_in_loop as _score_tomo_half_in_loop
from relax.sampling import TrialGrid, rotation_grid_size

if TYPE_CHECKING:
    from relax.refinement.setup_checks import RunContext

logger = logging.getLogger("relax.refinement.iteration_loop")


@dataclass
class SignificanceStatistics:
    """Half counts and aggregates in recording order for one expectation.

    See ``docs/math/relion_refinement_algorithm.md#iteration-convergence-policy``.
    """

    per_half: list = field(default_factory=lambda: [None, None], init=False)
    recorded: np.ndarray | None = field(default=None, init=False)
    convergence: np.ndarray | None = field(default=None, init=False)
    _recorded_parts: list = field(default_factory=list, init=False, repr=False)
    _convergence_parts: list = field(default_factory=list, init=False, repr=False)

    def record(self, half_index: int, counts, *, for_convergence: bool) -> None:
        if counts is not None:
            counts = np.asarray(counts, dtype=np.int32)
            self.per_half[half_index] = counts
            self._recorded_parts.append(counts)
            if for_convergence:
                self._convergence_parts.append(counts)

    def combine(self) -> None:
        """Combine after both halves and preprocessing checks have completed."""
        if self._convergence_parts:
            self.convergence = np.concatenate(self._convergence_parts, axis=0)
        if self._recorded_parts:
            self.recorded = np.concatenate(self._recorded_parts, axis=0)


def record_numbered_half(
    score_result: HalfScoreResult,
    particle_half: half_inputs.HalfSet,
    per_half: PerHalfOutputs,
    significance: SignificanceStatistics,
    *,
    profile_history: list,
    iteration: int,
    image_window_size: int | None,
    healpix_order: int,
    k_class_enabled: bool,
) -> None:
    """Record an already-published half's profile, support counts and captures.

    Empty halves keep the existing zero capture without a profile or device
    panel flush. Scoring and accumulator offloading remain with the caller.
    """
    k = particle_half.index
    if particle_half.dataset.n_units == 0:
        _parity_dump.collect_e_step(
            half=k,
            em_stats=score_result.em_stats,
            hard_assignment=score_result.ha,
            coarse_hard_assignment=per_half.coarse_ha[k],
            noise_stats=score_result.noise_stats,
            Ft_y=score_result.Ft_y,
            Ft_ctf=score_result.Ft_ctf,
            pose_rotation_eulers=per_half.pose_rotation_eulers[k],
            best_pose_rotation_eulers=per_half.best_pose_rotation_eulers[k],
            best_pose_translations=per_half.best_pose_translations[k],
            translation_search_base=per_half.translation_search_bases[k],
            original_image_indices=np.zeros(0, dtype=np.int64),
        )
        return

    _record_score_profile(
        profile_history,
        score_result,
        phase="iteration",
        iteration=iteration,
        relion_iteration=iteration + 1,
        half_index=k,
        current_size=image_window_size,
        healpix_order=healpix_order,
        k_class_enabled=k_class_enabled,
    )
    significance.record(k, score_result.significant_counts, for_convergence=not k_class_enabled)
    bpref_diagnostics.flush_selected_bpref_device_panel(iteration_index=iteration, half_index=k)

    # The parity dump records each image's row in the input stack; a dataset without an index layout has none.
    index_layout = getattr(particle_half.dataset, "_index_layout", None)
    original_image_indices = (
        np.asarray(
            index_layout.original_image_indices_for_local(np.arange(particle_half.dataset.n_images, dtype=np.int32)),
            dtype=np.int64,
        )
        if index_layout is not None and _parity_dump.is_active()
        else None
    )
    _parity_dump.collect_e_step(
        half=k,
        em_stats=score_result.em_stats,
        hard_assignment=score_result.ha,
        coarse_hard_assignment=per_half.coarse_ha[k],
        noise_stats=per_half.noise_stats[k],
        Ft_y=score_result.Ft_y,
        Ft_ctf=score_result.Ft_ctf,
        pose_rotation_eulers=per_half.pose_rotation_eulers[k],
        best_pose_rotation_eulers=per_half.best_pose_rotation_eulers[k],
        best_pose_translations=per_half.best_pose_translations[k],
        translation_search_base=per_half.translation_search_bases[k],
        original_image_indices=original_image_indices,
    )


_BPREF_DUMP_ENV_VARS = (
    "RELAX_BPREF_MEMBERSHIP_DUMP_DIR",
    "RELAX_BPREF_CONTRIBUTION_DUMP_DIR",
    "RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR",
)


@dataclass(frozen=True)
class NumberedHalfInputs:
    """What scoring one half of a numbered iteration reads that differs between the halves.

    ``data`` holds the half's particles (centred for this iteration's local search), its reference, tau2,
    noise, projector and scale groups. The controller builds one per half before the halves are scored; every
    array is one the controller holds for the whole expectation, so the record extends no lifetime.
    """

    data: HalfScoringData
    direction_priors: HalfDirectionLogPriors
    sigma_offset_angstrom: object


@dataclass(frozen=True)
class NumberedHalfRecording:
    """Where a numbered iteration's scored halves go: the iteration's half slots and significance counts, the
    run's profile history, and the labels its profile rows carry. One per iteration; ``per_half``,
    ``significance`` and ``profile_history`` are written in place."""

    per_half: PerHalfOutputs
    significance: SignificanceStatistics
    profile_history: list
    iteration: int
    image_window_size: int | None
    healpix_order: int
    k_class_enabled: bool


def finish_numbered_half(
    half: NumberedHalfInputs,
    score_result: HalfScoreResult,
    recording: NumberedHalfRecording,
    *,
    use_local: bool,
    dtype,
) -> None:
    """Publish one scored half into ``recording.per_half`` and record it, right after the half is scored.

    Writes half ``k``'s slots of ``per_half`` (``k = half.data.particles.index``), then its profile and
    significance counts (``record_numbered_half``). Before that, half 0's local-search M-step accumulators move to
    host (``_maybe_host_offload_half0_local_accumulators``, which edits ``score_result`` in place), so that when
    the halves run one after another they are off the device while half 1 scores (code rule 3).
    """
    particle_half = half.data.particles
    k = particle_half.index
    recording.per_half.translation_search_bases[k] = score_result.translation_search_base
    recording.per_half.pose_rotations[k] = score_result.pose_rotations
    recording.per_half.pose_rotation_eulers[k] = score_result.pose_rotation_eulers
    recording.per_half.coarse_ha[k] = score_result.coarse_ha
    if particle_half.dataset.n_units != 0:
        score_result = _maybe_host_offload_half0_local_accumulators(
            half_index=k,
            use_local=use_local,
            score_result=score_result,
            log=logger,
        )
    recording.per_half.update_from(k, score_result, dtype=dtype)
    record_numbered_half(
        score_result,
        particle_half,
        recording.per_half,
        recording.significance,
        profile_history=recording.profile_history,
        iteration=recording.iteration,
        image_window_size=recording.image_window_size,
        healpix_order=recording.healpix_order,
        k_class_enabled=recording.k_class_enabled,
    )


def _score_and_finish_half(score_half, finish_half, half_inputs, k) -> None:
    """Score half ``k`` and finish it before anything else runs on that half's thread, then return the heap its
    passes freed (:func:`~relax.helpers.host_memory.return_freed_heap`)."""
    finish_half(half_inputs[k], score_half(half_inputs[k]))
    return_freed_heap(f"half {k + 1}'s E-step")


def _half_overlap_active(requested: bool, *, diagnostic_half_indices, log) -> bool:
    """Decide whether the two halves' E-steps may run concurrently.

    Refuses rather than degrades. Each guard is a place where running the two
    halves at once would change what is observed, not merely when:

    * a subset of halves is being scored, so there is nothing to overlap;
    * a BPref dump is armed. The dump context is process-global and is set per
      half at the top of the half's work, so two halves in flight would write
      each other's context. The dump is a diagnostic, so the overlap yields.
    """

    if not requested:
        return False
    if tuple(diagnostic_half_indices) != (0, 1):
        log.info("Half overlap off: scoring halves %s, not both", tuple(diagnostic_half_indices))
        return False
    armed = [name for name in _BPREF_DUMP_ENV_VARS if os.environ.get(name, "").strip()]
    if armed:
        log.info("Half overlap off: a BPref dump is armed (%s)", ", ".join(armed))
        return False
    log.info("Half overlap ON: the two halves' E-steps run in one thread each")
    return True


def _run_halves_overlapped(run_half, diagnostic_half_indices) -> None:
    """Run each half's E-step in its own thread and re-raise in half order.

    Kernels still serialise on JAX's single compute stream, so the device order
    within a half is unchanged and the two halves' kernels cannot interleave
    mid-kernel. What overlaps is host work: one half's dispatch and operand
    preparation proceed while the other's kernels run.

    The halves write disjoint state, each indexed by its own half, so no
    accumulator is shared. Exceptions are collected and re-raised in half order
    so a failure reports the same way it would have when the halves ran one
    after another.
    """

    import threading

    from relax.sparse_pass2.sparse_pass2_budget import set_concurrent_device_shares

    errors: dict[int, BaseException] = {}

    def _target(half_index):
        try:
            run_half(half_index)
        except BaseException as exc:  # re-raised below, in half order
            errors[half_index] = exc

    # A Python-level name only; do not name the OS thread through ctypes (a truncated pthread handle segfaults).
    threads = [
        threading.Thread(target=_target, args=(int(k),), name=f"em-half-{int(k)}")
        for k in diagnostic_half_indices
    ]
    # Every pass-2 cache budget is a fraction of the device, written for one
    # worker at a time. Both halves sizing against the whole device is what
    # made the order-3 overlap fail with RESOURCE_EXHAUSTED while building the
    # second half's projection cache, so each is told it owns its share.
    previous_shares = set_concurrent_device_shares(len(threads))
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        set_concurrent_device_shares(previous_shares)
    for k in diagnostic_half_indices:
        if int(k) in errors:
            raise errors[int(k)]


def run_numbered_halves(
    score_half,
    finish_half,
    half_inputs,
    diagnostic_half_indices,
    significance: SignificanceStatistics,
    *,
    overlap_halves: bool,
    log: logging.Logger,
) -> None:
    """Score the halves of a numbered iteration and close its expectation.

    For each half ``k`` of ``diagnostic_half_indices``: ``score_half(half_inputs[k])`` returns its score result
    and ``finish_half(half_inputs[k], result)`` publishes it at once, before the next half is scored (so half 0's
    finished accumulators can leave the device first). The halves run one after another, or one thread each
    (score then finish) when ``overlap_halves`` is requested and allowed. A run that scored a subset of the
    halves for a diagnostic stops here. Then the deferred preprocess checks are drained (dropped when the halves
    raise, :func:`relax.cuda.kernels.expectation_relion_preprocess_checks`) and the halves' significant-sample
    counts combined; the caller marks the end of the expectation for its observer.
    """
    from relax.cuda.kernels import expectation_relion_preprocess_checks

    _overlap_active = _half_overlap_active(
        overlap_halves,
        diagnostic_half_indices=diagnostic_half_indices,
        log=log,
    )
    run_half = partial(_score_and_finish_half, score_half, finish_half, half_inputs)
    # The deferred preprocess checks are read when the halves are done, and dropped if they raise.
    with expectation_relion_preprocess_checks():
        if _overlap_active:
            _run_halves_overlapped(run_half, diagnostic_half_indices)
        else:
            for k in diagnostic_half_indices:
                run_half(k)
        if diagnostic_half_indices != (0, 1):
            raise RuntimeError(
                "targeted half-only significance diagnostic returned without writing its "
                "complete target set; refusing to continue with one half missing"
            )
    significance.combine()


@dataclass(frozen=True, kw_only=True)
class NumberedExpectationResult:
    """A numbered expectation's published per-half outputs and significance statistics."""

    per_half: PerHalfOutputs
    significance: SignificanceStatistics


def _numbered_dense_variant(first_iteration, k_class, *, use_adaptive: bool, coarse_cs, fine_window_size):
    """A numbered iteration's dense route: its first-iteration mode, the adaptive sizes and the class options.

    ``coarse_cs`` is the adaptive pass-1 size, None off the adaptive route; ``fine_window_size`` is the pass-2
    scoring window (the E-step size).
    """

    return DenseVariantPolicy(
        score_mode=first_iteration.score_mode,
        winner_take_all=first_iteration.winner_take_all,
        k_class_enabled=int(k_class.n_classes) > 1,
        firstiter_cc=first_iteration.relion_firstiter_cc,
        coarse_window_size=coarse_cs,
        fine_window_size=fine_window_size if use_adaptive else None,
        skip_align=bool(k_class.skip_align),
    )


def run_numbered_expectation(
    ctx: "RunContext",
    carry: iteration_planning.IterationCarry,
    plan: local_sampling.NumberedSamplingPlan,
    this_iteration: iteration_planning.NumberedIteration,
    options: RefinementOptions,
    *,
    half_inputs: list,
    diagnostic_half_indices,
    history,
) -> NumberedExpectationResult:
    """Score both halves of a numbered iteration (step S3): the dense route's variant, the expectation phase
    shared by the halves, the subtomogram sampling, then each half scored and published (half 0's
    accumulators leave the device before half 1 is scored; ``options.execution.overlap_halves`` may run the two
    on two threads).

    ``half_inputs`` are the halves' scoring inputs (``numbered_half_inputs``); ``diagnostic_half_indices`` the
    halves to score (both, except for a targeted significance diagnostic, which stops the run). Reads from
    ``carry``: ``state``, ``coarse_grids``, ``random_perturbation`` and the class log priors; from ``ctx``: the
    run's scoring settings and its expectation probe (the observer's per-half hooks). Appends the local and
    global profile rows to ``history``'s profile lists as the scorers produce them.
    """
    use_local = carry.state.do_local_search
    variant = _numbered_dense_variant(
        this_iteration.first_iteration, options.k_class, use_adaptive=plan.use_adaptive, coarse_cs=plan.coarse_cs,
        fine_window_size=plan.sampling_plan.windows.score_window_size,
    )
    phase = prepare_numbered_expectation(
        plan.trial_grid, plan.sampling_plan.windows, local_sampling=plan.sampling_plan.local, variant=variant,
        use_adaptive=plan.use_adaptive, base_translations=carry.coarse_grids.base_translations,
        current_healpix_order=carry.coarse_grids.rotation_grid.healpix_order,
        oversampling_order=carry.state.adaptive_oversampling, translation_step=carry.state.translation_step,
        random_perturbation=carry.random_perturbation, adaptive_pass1=plan.adaptive_pass1,
        coarse_rotation_ids=plan.scoring_rotation_ids, coarse_angular_step_deg=plan.coarse_angular_step_deg,
        options=options, iteration=this_iteration.iteration,
        numbered_relion_iteration=this_iteration.numbered_relion_iteration,
        collect_local_search_profile=ctx.collect_local_search_profile,
        local_profile_history=history.local_profile_history, probe=ctx.expectation_probe,
    )
    tomo_sampling = (
        numbered_iteration_tomo_sampling(
            carry.state, ctx.image_geometry, local_sampling=plan.sampling_plan.local,
            grid_healpix_order=carry.coarse_grids.rotation_grid.healpix_order,
            random_perturbation=carry.random_perturbation,
            coarse_size=plan.sampling_plan.local.coarse_image_window_size if use_local else plan.coarse_cs,
            fine_size=plan.sampling_plan.windows.score_window_size,
        )
        if ctx.tomo_halves
        else None
    )
    significance = SignificanceStatistics()
    # per_half.coarse_ha holds the coarse-grid assignments (trial_grid.rotations indices on every route).
    per_half = PerHalfOutputs()
    run_numbered_halves(
        partial(
            score_numbered_half, phase=phase, tomo_sampling=tomo_sampling,
            class_log_priors=carry.class_mixture.log_priors, batch_planner=ctx.batch_planner,
            image_geometry=ctx.image_geometry, padded_volume_shape=ctx.padded_volume_shape,
            multi_shape_halves=ctx.multi_shape_halves, options=options,
            replay_prior_translations=plan.replay_prior_translations,
            initial_class_assignments=this_iteration.seeding.seed_classes,
            single_class_iteration=this_iteration.seeding.single_class_iteration, scoring_dtype=ctx.scoring_dtype,
            relion_translation_angle_scale=ctx.relion_translation_angle_scale,
            relion_projection_scale=ctx.relion_projection_scale, iteration=this_iteration.iteration,
            numbered_relion_iteration=this_iteration.numbered_relion_iteration, probe=ctx.expectation_probe,
        ),
        partial(
            finish_numbered_half,
            recording=NumberedHalfRecording(
                per_half, significance, profile_history=history.global_profile_history,
                iteration=this_iteration.iteration, image_window_size=plan.sampling_plan.windows.image_window_size,
                healpix_order=carry.coarse_grids.rotation_grid.healpix_order, k_class_enabled=ctx.k_class_enabled,
            ),
            use_local=use_local, dtype=ctx.scoring_dtype,
        ),
        half_inputs, diagnostic_half_indices, significance,
        overlap_halves=options.execution.overlap_halves, log=logger,
    )
    return NumberedExpectationResult(per_half=per_half, significance=significance)


def numbered_half_inputs(
    ctx: "RunContext",
    carry: iteration_planning.IterationCarry,
    plan: local_sampling.NumberedSamplingPlan,
    options: RefinementOptions,
    *,
    halves,
    reference_model,
    projectors,
    follower_setup,
) -> list[NumberedHalfInputs]:
    """Each half's scoring inputs of one numbered iteration: its particles (centred for a local search), its
    reference, tau2, noise and projector, its scale groups and the iteration-start scale-gate curve, its
    direction log priors and translation prior width.

    Reads from ``carry``: ``state``, ``noise_model``, ``sigma_offset`` and
    ``previous_data_vs_prior_for_scheduling`` (still the iteration-start curve: RELION's scale XA/AA shell
    gate reads it, the scheduling curve changes later); from ``ctx``: ``tomo_halves``; from
    ``follower_setup``: the halves' scale-statistics groups; from ``options``: ``replay.init_angle_priors``.
    """
    return [
        NumberedHalfInputs(
            data=HalfScoringData(
                particles=local_sampling.local_search_centre_half(
                    halves[k], (options.replay.init_angle_priors or (None, None))[k], carry.state
                ),
                reference=reference_model.maps[k],
                mean_variance=reference_model.half_tau2(k),
                noise_variance=carry.noise_model.variance_per_half[k],
                noise_radial=(
                    carry.noise_model.radial_per_half[k]
                    if not ctx.tomo_halves and halves[k].dataset.n_units
                    else None
                ),
                projector=projectors[k],
                scale_group_ids=follower_setup.scale_stats_group_ids_per_half[k],
                scale_group_count=follower_setup.scale_stats_group_count_per_half[k],
                scale_correction_data_vs_prior=carry.previous_data_vs_prior_for_scheduling,
            ),
            direction_priors=plan.direction_log_priors[k],
            sigma_offset_angstrom=carry.sigma_offset.for_half(k),
        )
        for k in (0, 1)
    ]


@dataclass(frozen=True)
class PreparedFinalHalf:
    """Priors and optics resolved on one final expectation's scoring frame."""

    translations: HalfTranslationPriorInputs
    translation_log_prior: object
    directions: HalfDirectionLogPriors
    optics: OpticsSpec


def prepare_final_half(
    half: half_inputs.HalfSet,
    sampling: FinalSampling,
    *,
    image_geometry: ImageGeometry,
    sigma_offset_angstrom,
    noise_radial,
    direction_prior: DirectionPrior,
    n_classes: int,
    use_local: bool,
    coarse_angular_step_deg,
    particle_diameter_angstrom,
    sealed_sampling_state,
    symmetry: str,
    dtype,
    projection_scale: float,
) -> PreparedFinalHalf:
    """Resolve SPA priors and optics in the final grid's image-pixel frame.

    See docs/math/relion_refinement_algorithm.md, section 7.
    """
    translations = relion_half_translation_prior_inputs(
        half.translations,
        voxel_size=image_geometry.pixel_size_angstrom,
        base_translations=sampling.base_translations,
        current_translations=sampling.grid.translations,
        dtype=dtype,
    )
    translation_log_prior = make_relion_translation_log_prior(
        translations.prior_translations,
        image_geometry.pixel_size_angstrom,
        sigma_offset_angstrom,
        translations.prior_center,
        offset_range_pixels=None,
        dtype=dtype,
    )
    directions = relion_direction_log_priors_for_half(
        use_local=use_local,
        scoring_healpix_order=None if use_local else sampling.settings.grid_order,
        n_classes=n_classes,
        prior=direction_prior,
        sealed_sampling_state=sealed_sampling_state,
        dtype=dtype,
        log=logger,
        half_index=half.index,
        symmetry=symmetry,
    )
    optics = optics_shapes.prepare_optics(
        half.dataset,
        noise_radial=noise_radial,
        coarse_step_deg=coarse_angular_step_deg,
        particle_diameter_ang=particle_diameter_angstrom,
        previous_translations=half.translations,
        sigma_offset_angstrom=sigma_offset_angstrom,
        base_translations=sampling.base_translations,
        current_translations=sampling.grid.translations,
        with_log_prior=not use_local,
        zero_cold_center=False,
        dtype=dtype,
        single_shape_projection_scale=projection_scale,
    )
    return PreparedFinalHalf(translations, translation_log_prior, directions, optics)


@dataclass(frozen=True, eq=False, kw_only=True)
class NumberedExpectation:
    """Trial pose grid and resolved sampling/policies shared by both numbered halves."""

    grid: TrialGrid
    sampling: local_sampling.LocalSampling | DenseSamplingSpec
    variant: DenseVariantPolicy
    use_adaptive: bool
    local_diagnostics: LocalDiagnosticPolicy | None


def prepare_numbered_expectation(
    grid: TrialGrid,
    windows: iteration_planning.ExpectationWindows,
    *,
    local_sampling: local_sampling.LocalSampling | None,
    variant: DenseVariantPolicy,
    use_adaptive: bool,
    base_translations,
    current_healpix_order: int,
    oversampling_order: int,
    translation_step: float,
    random_perturbation: float,
    adaptive_pass1,
    coarse_rotation_ids,
    coarse_angular_step_deg,
    options: RefinementOptions,
    iteration: int,
    numbered_relion_iteration: int,
    collect_local_search_profile: bool,
    local_profile_history: list,
    probe: ExpectationProbe,
) -> NumberedExpectation:
    """Bind the numbered grid to dense/local support and local diagnostic policy.

    Dense scoring may use device coarse rotations while poses retain the supplied
    canonical trial rows.
    See ``docs/math/relion_refinement_algorithm.md#numbered-expectation-preparation``.
    """
    if local_sampling is not None:
        numbered_sampling = local_sampling
        numbered_local_diagnostics = LocalDiagnosticPolicy(
            iteration=iteration,
            debug_iteration=numbered_relion_iteration,
            # The half's activation is known only when it is scored (score_numbered_half sets it).
            bpref_device_signature_active=False,
            probe=probe,
            collect_local_search_profile=collect_local_search_profile,
            local_profile_history=local_profile_history,
        )
    else:
        adaptive_pass1_rotations, adaptive_pass1_source = (None, None) if adaptive_pass1 is None else adaptive_pass1
        effective_from_pass1 = use_adaptive and adaptive_pass1_rotations is not None
        numbered_sampling = DenseSamplingSpec(
            effective_rotations=adaptive_pass1_rotations if effective_from_pass1 else grid.rotations,
            effective_device_source=adaptive_pass1_source if effective_from_pass1 else None,
            current_translations=grid.translations,
            base_translations=base_translations,
            current_healpix_order=current_healpix_order,
            oversampling_order=oversampling_order,
            translation_step=translation_step,
            coarse_engine=options.adaptive.coarse_engine,
            random_perturbation=random_perturbation,
            cs_for_engine=windows.score_window_size,
            model_current_size_for_engine=windows.engine_model_window_size,
            wsum_current_size_for_engine=windows.wsum_size_for_engine,
            coarse_rotation_ids=coarse_rotation_ids,
            coarse_scoring_rotations=adaptive_pass1_rotations if int(oversampling_order) == 0 else None,
            coarse_scoring_device_source=adaptive_pass1_source if int(oversampling_order) == 0 else None,
            coarse_angular_step_deg=coarse_angular_step_deg,
            symmetry=options.symmetry.point_group,
        )
        numbered_local_diagnostics = None
    return NumberedExpectation(
        grid=grid,
        sampling=numbered_sampling,
        variant=variant,
        use_adaptive=use_adaptive,
        local_diagnostics=numbered_local_diagnostics,
    )


def empty_half_rotation_count(sampling, grid_rotation_count: int, *, use_local: bool, symmetry: str) -> int:
    """The rotation-sum length an empty half reports, the grid its occupied twin sums over.

    A local pass bins its rotation posterior on the parent (pass-1) grid, which is the
    direction-prior grid of a local search (``direction_prior_healpix_order_for_scoring``);
    a global pass on its trial grid.
    """
    if use_local:
        return int(rotation_grid_size(sampling.search.parent_order, symmetry=symmetry))
    return int(grid_rotation_count)


def score_numbered_half(
    half: NumberedHalfInputs,
    phase: NumberedExpectation,
    *,
    tomo_sampling: TomoSampling | None,
    class_log_priors,
    batch_planner: BatchPlanner,
    image_geometry: ImageGeometry,
    padded_volume_shape,
    multi_shape_halves: bool,
    options: RefinementOptions,
    replay_prior_translations,
    initial_class_assignments,
    single_class_iteration: bool,
    scoring_dtype,
    relion_translation_angle_scale: float,
    relion_projection_scale: float,
    iteration: int,
    numbered_relion_iteration: int,
    probe: ExpectationProbe,
) -> HalfScoreResult:
    """Build half-specific priors/batches and accumulate an empty, SPA or tomography half.

    Canonical pose grids and the applied translation base accompany the result. RELION's source-faithful
    powerClass normalisation (and with it the exact BPref operands) is on wherever RELION's particle order is
    preserved (``options.parity.preserve_bpref_particle_order``).
    Publication, accumulator offloading and post-score capture are ``finish_numbered_half``'s.
    """
    sampling = phase.sampling
    particle_half = half.data.particles
    k = particle_half.index
    use_local = isinstance(sampling, local_sampling.LocalSampling)
    tomo_halves = tomo_sampling is not None
    k_class_enabled = phase.variant.k_class_enabled
    image_window_size = sampling.image_window_size if use_local else sampling.cs_for_engine
    model_support_size = sampling.model_support_size if use_local else sampling.model_current_size_for_engine
    symmetry = sampling.search.symmetry if use_local else sampling.symmetry
    coarse_size_step_deg = sampling.coarse_angular_step_deg
    particle_diameter_ang = options.schedule.particle_diameter_ang
    source_faithful_spectrum_norm = options.parity.preserve_bpref_particle_order
    bpref_diagnostics.set_bpref_contribution_dump_context(
        iteration=iteration + 1,
        half=particle_half.index + 1,
    )
    bpref_device_signature_active = (
        options.execution.bpref_device_signature_target == (iteration + 1, particle_half.index + 1)
    )
    logger.info(
        "BPREF_DEVICE_SIGNATURE_ACTIVATION iteration=%d half=%d "
        "final_all_data=false active=%s",
        iteration + 1,
        particle_half.index + 1,
        str(bpref_device_signature_active).lower(),
    )
    previous_translations_k = particle_half.translations
    translation_search_base = relion_translation_search_base(
        previous_translations_k, dtype=scoring_dtype
    )
    half_batching = prepare_half_batches(
        particle_half.dataset,
        half.data.projector,
        planner=batch_planner,
        rotations=phase.grid.rotations if phase.use_adaptive or k_class_enabled else None,
        translations=phase.grid.translations,
        cs_for_engine=image_window_size,
        coarse_cs=phase.variant.coarse_window_size if phase.use_adaptive else None,
        model_current_size_for_engine=model_support_size,
        use_adaptive=phase.use_adaptive,
        use_local=use_local,
        relion_firstiter_cc_this_iter=phase.variant.firstiter_cc,
        firstiter_cc_tree_rescore_max_margin=options.parity.firstiter_cc_tree_rescore_max_margin,
        firstiter_winner_take_all_this_iter=phase.variant.winner_take_all,
        source_faithful_spectrum_norm=source_faithful_spectrum_norm,
        preserve_bpref_particle_order=options.parity.preserve_bpref_particle_order,
        bpref_device_signature_active=bpref_device_signature_active,
        use_relion_x_half_mstep=options.variants.k1_relion_x_half_mstep,
        multi_shape_halves=multi_shape_halves,
        coarse_sizing=(coarse_size_step_deg, particle_diameter_ang) if phase.use_adaptive and multi_shape_halves else None,
    )
    # RELION translation priors: relion_half_translation_prior_inputs
    # documents the pdf_offset / wsum_sigma2_offset centers, the
    # cold-start engine center and the prior-grid selection.
    if tomo_halves:
        # A subtomogram half builds its 3D offset priors per particle (score_tomo_half).
        translation_prior_inputs = None
        trans_prior_center = local_trans_prior_center = trans_prior_center_for_engine = None
    else:
        translation_prior_inputs = relion_half_translation_prior_inputs(
            previous_translations_k,
            voxel_size=image_geometry.pixel_size_angstrom,
            base_translations=sampling.base_translations,
            current_translations=phase.grid.translations,
            dtype=scoring_dtype,
        )
        trans_prior_center = translation_prior_inputs.prior_center
        local_trans_prior_center = translation_prior_inputs.local_prior_center
        trans_prior_center_for_engine = translation_prior_inputs.engine_prior_center
    translation_log_prior = None
    if not use_local and not tomo_halves:
        if not k_class_enabled and trans_prior_center is None:
            # A fresh K1 half has implicit zero offsets, not a flat
            # pdf_offset. Native ACC applies the Gaussian even at
            # iteration 1; see relion_refinement_algorithm.md.
            trans_prior_center = np.zeros(2, dtype=scoring_dtype)
        translation_log_prior = make_relion_translation_log_prior(
            translation_prior_inputs.prior_translations,
            image_geometry.pixel_size_angstrom,
            half.sigma_offset_angstrom,
            trans_prior_center,
            offset_range_pixels=None,
            dtype=scoring_dtype,
        )
    if particle_half.dataset.n_units == 0:
        logger.info("Skipping E-step/M-step accumulation for empty half-%d dataset", particle_half.index + 1)
        n_shells = int(image_geometry.image_shape[0] // 2 + 1)
        n_rot_for_stats = empty_half_rotation_count(
            sampling, phase.grid.rotations.shape[0], use_local=use_local, symmetry=symmetry
        )
        empty_k1_x_half_mstep = (
            (not k_class_enabled)
            and (use_local or phase.use_adaptive)
            and options.variants.k1_relion_x_half_mstep
        )
        empty_result = empty_half_result(
            volume_shape=particle_half.dataset.volume_shape if empty_k1_x_half_mstep else None,
            padded_volume_shape=padded_volume_shape,
            n_classes=batch_planner.n_classes,
            n_shells=n_shells,
            n_rotations=n_rot_for_stats,
            translation_dimension=3 if tomo_halves else phase.grid.translations.shape[1],
            image_window_size=image_window_size,
            model_support_size=model_support_size,
            use_x_half_mstep=empty_k1_x_half_mstep,
        )
        empty_result.coarse_ha = empty_result.ha
        empty_result.translation_search_base = translation_search_base
        return empty_result
    # The half's units' classes in RELION's seed iteration, by particle row (all class 0 in the CC
    # iteration before it).
    seed_classes_k = (
        np.asarray(initial_class_assignments)[
            particle_half.dataset._index_layout.original_image_indices_for_local(
                np.arange(particle_half.dataset.n_units)
            )
        ]
        if initial_class_assignments is not None
        else np.zeros(particle_half.dataset.n_units, dtype=np.int64)
        if single_class_iteration
        else None
    )
    if tomo_halves:
        score_result = _score_tomo_half_in_loop(
            half.data,
            use_adaptive=phase.use_adaptive,
            sampling=tomo_sampling,
            local_search=(
                dict(
                    previous_eulers_deg=particle_half.rotation_eulers,
                    sigma_rot=sampling.search.sigma_rot,
                    sigma_psi=sampling.search.sigma_psi,
                )
                if use_local
                else None
            ),
            rotation_log_prior=half.direction_priors.rotation_log_prior,
            sigma_offset_angst=half.sigma_offset_angstrom,
            max_significants=options.adaptive.max_significants,
            reconstruction_current_size=model_support_size,
            symmetry=symmetry,
            class_log_priors=class_log_priors if k_class_enabled else None,
            class_rotation_log_prior=half.direction_priors.class_rotation_log_prior,
            unit_seed_classes=seed_classes_k,
            normalized_cc=phase.variant.score_mode == "normalized_cc",
        )
    elif use_local:
        local_optics = optics_shapes.prepare_optics(
            particle_half.dataset,
            noise_radial=half.data.noise_radial,
            coarse_step_deg=coarse_size_step_deg,
            particle_diameter_ang=particle_diameter_ang,
            previous_translations=previous_translations_k,
            sigma_offset_angstrom=half.sigma_offset_angstrom,
            base_translations=sampling.base_translations,
            current_translations=phase.grid.translations,
            with_log_prior=False,
            zero_cold_center=True,
            dtype=scoring_dtype,
            single_shape_projection_scale=relion_projection_scale,
        )
        local_result = _score_half_local_in_bpref_scope(
            half=replace(half.data, mean_variance=None, image_seed_classes=seed_classes_k),
            sampling=sampling,
            priors=LocalPriorSpec(
                trans_prior_center=local_trans_prior_center,
                trans_prior_center_for_engine=trans_prior_center_for_engine,
                current_sigma_offset_angstrom=half.sigma_offset_angstrom,
                translation_search_base=translation_search_base,
                local_search_translation_prior_mode=(options.local_search.local_search_translation_prior_mode),
                replay_prior_translations=replay_prior_translations,
            ),
            batching=LocalBatchPolicy(
                max_significants=options.adaptive.max_significants,
            ),
            execution=LocalExecutionPolicy(
                disc_type="linear_interp",
                disable_adjoint_y=options.debug.disable_adjoint_y,
                disable_adjoint_ctf=options.debug.disable_adjoint_ctf,
                source_faithful_spectrum_norm=source_faithful_spectrum_norm,
                relion_translation_angle_scale=(relion_translation_angle_scale),
                nyquist_column_counting=options.consistency.nyquist_column_counting,
                relion_x_half_mstep=options.variants.k1_relion_x_half_mstep,
                adaptive_pass2=options.variants.local_adaptive_pass2.at(sampling.search.oversampling_order),
                precision=options.precision,
                fine_precision=local_precision(options.precision, numbered_relion_iteration, pass_index=2),
                score_only=options.local_search.stop_after_local_search_score_only,
                # A search local from iteration 1 (--sigma_ang) scores RELION's --firstiter_cc
                # iteration with the normalized CC, as the subtomogram local path above does.
                firstiter_cc=phase.variant.score_mode == "normalized_cc",
            ),
            diagnostics=replace(
                phase.local_diagnostics, bpref_device_signature_active=bpref_device_signature_active,
            ),
            optics=local_optics,
        )
        if local_result.coarse_ha is None:
            local_result.coarse_ha = local_result.ha
        score_result = local_result

    else:
        # Shared dense half-scoring operands; the adaptive branch adds its
        # pass-1 grid and batch/size overrides.
        dense_optics = optics_shapes.prepare_optics(
            particle_half.dataset,
            noise_radial=half.data.noise_radial,
            coarse_step_deg=coarse_size_step_deg,
            particle_diameter_ang=particle_diameter_ang,
            previous_translations=previous_translations_k,
            sigma_offset_angstrom=half.sigma_offset_angstrom,
            base_translations=sampling.base_translations,
            current_translations=phase.grid.translations,
            with_log_prior=True,
            zero_cold_center=not k_class_enabled,
            dtype=scoring_dtype,
            single_shape_projection_scale=relion_projection_scale,
        )
        dense_half = replace(half.data, image_seed_classes=seed_classes_k)
        dense_sampling = sampling
        dense_priors = DensePriorSpec(
            rotation_log_prior_k=half.direction_priors.rotation_log_prior,
            class_rotation_log_prior_k=half.direction_priors.class_rotation_log_prior,
            translation_log_prior=translation_log_prior,
            translation_search_base=translation_search_base,
            trans_prior_center_for_engine=trans_prior_center_for_engine,
            class_log_priors=class_log_priors,
        )
        dense_batching = DenseBatchPolicy(
            image_batch_size=options.execution.image_batch_size,
            max_significants=options.adaptive.max_significants,
            safe_batch_sizes=half_batching.safe_batch_sizes,
            significance_safe_batch_sizes=half_batching.significance_safe_batch_sizes,
            class_batch_overrides=half_batching.class_overrides,
            k_class_image_batch_size_override=(half_batching.fine_image_batch_size if phase.use_adaptive else None),
            k_class_rotation_block_size_override=(half_batching.fine_rotation_block_size if phase.use_adaptive else None),
            significance_image_batch_size_override=(half_batching.coarse_image_batch_size if phase.use_adaptive else None),
            significance_rotation_block_size_override=(half_batching.coarse_rotation_block_size if phase.use_adaptive else None),
        )
        dense_variant = phase.variant
        dense_execution = DenseExecutionPolicy(
            disc_type="linear_interp",
            disable_adjoint_y=options.debug.disable_adjoint_y,
            disable_adjoint_ctf=options.debug.disable_adjoint_ctf,
            return_best_pose_details=True,
            bpref_device_signature_active=bpref_device_signature_active,
            debug_iteration=numbered_relion_iteration,
            diagnostic_float64_pass2=_diagnostic_float64_pass2_matches(
                numbered_relion_iteration
            ),
            preserve_bpref_particle_order=options.parity.preserve_bpref_particle_order,
            source_faithful_spectrum_norm=source_faithful_spectrum_norm,
            relion_translation_angle_scale=relion_translation_angle_scale,
            firstiter_cc_tree_rescore_max_margin=options.parity.firstiter_cc_tree_rescore_max_margin,
            firstiter_cc_support=options.consistency.firstiter_cc_support,
            nyquist_column_counting=options.consistency.nyquist_column_counting,
            relion_x_half_mstep=options.variants.relion_x_half_mstep(k_class=k_class_enabled),
            precision=options.precision,
        )
        dense_result = _score_half_dense_in_bpref_scope(
            dense_half,
            dense_sampling,
            dense_priors,
            dense_batching,
            dense_variant,
            dense_execution,
            dense_optics,
        )
        if not phase.use_adaptive or dense_result.pose_rotations is None:
            if dense_result.pose_rotations is None:
                dense_result.pose_rotations = phase.grid.rotations
            if dense_result.pose_rotation_eulers is None:
                dense_result.pose_rotation_eulers = phase.grid.rotation_eulers
        if dense_result.coarse_ha is None:
            dense_result.coarse_ha = dense_result.ha
        score_result = dense_result

        if not phase.use_adaptive:
            probe.dense_half_scored(DenseHalfScored(
                iteration, k, grid=phase.grid, sampling=dense_sampling, direction_priors=half.direction_priors,
                translation_log_prior=translation_log_prior, particles=particle_half,
                translation_search_base=translation_search_base, previous_translations=previous_translations_k,
                half=half.data, image_window_size=image_window_size, perturb_factor=options.parity.perturb_factor,
                result=dense_result,
            ))

    score_result.translation_search_base = translation_search_base
    return score_result
