"""Prepare numbered and final half expectations without publishing model updates.

See docs/math/relion_refinement_algorithm.md, section 3.
"""

import logging
import os
from dataclasses import dataclass, field, replace

import numpy as np

from relax.dense import scoring_policy
from relax.dense.score_outputs import HalfScoreResult, PerHalfOutputs, _record_score_profile, empty_half_result
from relax.dense.scoring_policy import (
    _k1_relion_x_half_mstep_enabled,
    _local_adaptive_pass2_denominator_support_mode,
    _local_adaptive_pass2_full_parent_enabled,
    _local_adaptive_pass2_rotation_only_enabled,
)
from relax.diagnostics import bpref_diagnostics
from relax.diagnostics import parity_dump as _parity_dump
from relax.diagnostics.iteration import _bpref_device_signature_active_for_numbered_half, _replay_manifest_array
from relax.helpers.dtype_policy import _diagnostic_float64_pass2_matches
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
from relax.refinement import optics_shapes
from relax.refinement.expectation_batches import BatchPlanner, prepare_half_batches
from relax.refinement.final_sampling import FinalSampling
from relax.refinement.half_inputs import HalfSet
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
from relax.refinement.iteration_planning import ExpectationWindows
from relax.refinement.local_sampling import LocalSampling
from relax.refinement.optics_shapes import OpticsSpec
from relax.refinement.refinement_options import RefinementOptions
from relax.refinement.tomo_half import TomoSampling
from relax.refinement.tomo_half import score_tomo_half_in_loop as _score_tomo_half_in_loop
from relax.relion.geometry import PROJECTION_PADDING_FACTOR, RECONSTRUCTION_PADDING_FACTOR
from relax.sampling import TrialGrid, rotation_grid_size

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
    particle_half: HalfSet,
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

    try:
        original_image_indices = np.asarray(
            particle_half.dataset._index_layout.original_image_indices_for_local(
                np.arange(particle_half.dataset.n_images, dtype=np.int32)
            ),
            dtype=np.int64,
        )
    except Exception:
        original_image_indices = None
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

    # The name is a process-level label; CPython does not push it to the OS, so
    # a profile shows these threads unnamed. Naming them through libc was tried
    # and removed: ctypes defaults a return type to int, which truncates a
    # 64-bit pthread handle, and the truncated handle segfaults. A profiling
    # convenience is not worth a crash in the driver. Profiles identify the two
    # half threads by dispatch volume instead, which is unambiguous: in the
    # order-1 trace they issued 514,849 and 496,720 CUDA calls against 7,500
    # for the next busiest thread.
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
    run_half,
    diagnostic_half_indices,
    significance: SignificanceStatistics,
    *,
    overlap_halves: bool,
    iteration: int,
    log: logging.Logger,
) -> None:
    """Score the halves of a numbered iteration and close its expectation.

    ``run_half(k)`` scores half ``k`` and records its outputs; the halves run one after another, or one
    thread each when ``overlap_halves`` is requested and allowed. A run that scored a subset of the
    halves for a diagnostic stops here. Then the deferred preprocess checks are drained (dropped when
    the halves raise, :func:`relax.cuda.kernels.expectation_relion_preprocess_checks`) and the halves'
    significant-sample counts combined.
    """
    from relax.cuda.kernels import expectation_relion_preprocess_checks

    _overlap_active = _half_overlap_active(
        overlap_halves,
        diagnostic_half_indices=diagnostic_half_indices,
        log=log,
    )
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

        # E-step + per-half M-step accumulators are now both populated.
        _parity_dump.mark_stage(iteration, "e_step")
    significance.combine()


@dataclass(frozen=True)
class PreparedFinalHalf:
    """Priors and optics resolved on one final expectation's scoring frame."""

    translations: HalfTranslationPriorInputs
    translation_log_prior: object
    directions: HalfDirectionLogPriors
    optics: OpticsSpec


def prepare_final_half(
    half: HalfSet,
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
    )
    return PreparedFinalHalf(translations, translation_log_prior, directions, optics)


@dataclass(frozen=True, eq=False, kw_only=True)
class NumberedExpectation:
    """Trial pose grid and resolved sampling/policies shared by both numbered halves."""

    grid: TrialGrid
    sampling: LocalSampling | DenseSamplingSpec
    variant: DenseVariantPolicy
    use_adaptive: bool
    local_diagnostics: LocalDiagnosticPolicy | None


def prepare_numbered_expectation(
    grid: TrialGrid,
    windows: ExpectationWindows,
    *,
    local_sampling: LocalSampling | None,
    variant: DenseVariantPolicy,
    use_adaptive: bool,
    base_translations,
    current_healpix_order: int,
    oversampling_order: int,
    translation_step: float,
    random_perturbation: float,
    adaptive_pass1_rotations,
    coarse_rotation_ids,
    coarse_angular_step_deg,
    options: RefinementOptions,
    iteration: int,
    numbered_relion_iteration: int,
    collect_local_search_profile: bool,
    local_profile_history: list,
) -> NumberedExpectation:
    """Bind the numbered grid to dense/local support and local diagnostic policy.

    Dense scoring may use device coarse rotations while poses retain the supplied
    canonical trial rows.
    See ``docs/math/relion_refinement_algorithm.md#numbered-expectation-preparation``.
    """
    if local_sampling is not None:
        if local_sampling.search.oversampling_order > 0:
            local_adaptive_full_parent = _local_adaptive_pass2_full_parent_enabled()
            local_adaptive_rotation_only = _local_adaptive_pass2_rotation_only_enabled()
            local_adaptive_denominator_mode = (
                _local_adaptive_pass2_denominator_support_mode()
            )
        else:
            local_adaptive_full_parent = False
            local_adaptive_rotation_only = False
            local_adaptive_denominator_mode = None
        numbered_sampling = local_sampling
        numbered_local_diagnostics = LocalDiagnosticPolicy(
            iteration=iteration,
            debug_iteration=numbered_relion_iteration,
            save_intermediates_dir=options.debug.save_intermediates_dir,
            collect_local_search_profile=collect_local_search_profile,
            diagnostic_score_only=bool(options.debug.stop_after_local_search_score_only),
            local_profile_history=local_profile_history,
            adaptive_pass2_full_parent=local_adaptive_full_parent,
            adaptive_pass2_rotation_only=local_adaptive_rotation_only,
            adaptive_pass2_denominator_mode=local_adaptive_denominator_mode,
        )
    else:
        numbered_sampling = DenseSamplingSpec(
            effective_rotations=(
                adaptive_pass1_rotations
                if use_adaptive and adaptive_pass1_rotations is not None
                else grid.rotations
            ),
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
    direction-prior grid of a local search (``_direction_prior_healpix_order_for_scoring``);
    a global pass on its trial grid.
    """
    if use_local:
        return int(rotation_grid_size(sampling.search.parent_order, symmetry=symmetry))
    return int(grid_rotation_count)


def score_numbered_half(
    half: HalfScoringData,
    phase: NumberedExpectation,
    *,
    tomo_sampling: TomoSampling | None,
    direction_priors: HalfDirectionLogPriors,
    class_log_priors,
    sigma_offset_angstrom,
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
    iteration: int,
    numbered_relion_iteration: int,
) -> HalfScoreResult:
    """Build half-specific priors/batches and accumulate an empty, SPA or tomography half.

    Canonical pose grids and the applied translation base accompany the result. RELION's source-faithful
    powerClass normalisation (and with it the exact BPref operands) is on wherever RELION's particle order is
    preserved (``options.parity.preserve_bpref_particle_order``).
    The controller owns publication, accumulator offloading and post-score capture.
    """
    sampling = phase.sampling
    particle_half = half.particles
    k = particle_half.index
    use_local = isinstance(sampling, LocalSampling)
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
        _bpref_device_signature_active_for_numbered_half(
            iteration=iteration + 1,
            half=particle_half.index + 1,
        )
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
        half.projector,
        planner=batch_planner,
        rotations=phase.grid.rotations if phase.use_adaptive or k_class_enabled else None,
        translations=phase.grid.translations,
        cs_for_engine=image_window_size,
        coarse_cs=phase.variant.firstiter_coarse_current_size if phase.use_adaptive else None,
        model_current_size_for_engine=model_support_size,
        use_adaptive=phase.use_adaptive,
        use_local=use_local,
        relion_firstiter_cc_this_iter=phase.variant.relion_firstiter_cc_this_iter,
        firstiter_cc_tree_rescore_max_margin=options.parity.firstiter_cc_tree_rescore_max_margin,
        firstiter_winner_take_all_this_iter=phase.variant.firstiter_winner_take_all_this_iter,
        source_faithful_spectrum_norm=source_faithful_spectrum_norm,
        preserve_bpref_particle_order=options.parity.preserve_bpref_particle_order,
        image_fourier_backend=options.parity.image_fourier_backend,
        bpref_device_signature_active=bpref_device_signature_active,
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
            sigma_offset_angstrom,
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
            and _k1_relion_x_half_mstep_enabled()
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
            half,
            use_local=use_local,
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
            rotation_log_prior=direction_priors.rotation_log_prior,
            sigma_offset_angst=sigma_offset_angstrom,
            max_significants=options.adaptive.max_significants,
            reconstruction_current_size=model_support_size,
            symmetry=symmetry,
            class_log_priors=class_log_priors if k_class_enabled else None,
            class_rotation_log_prior=direction_priors.class_rotation_log_prior,
            unit_seed_classes=seed_classes_k,
            normalized_cc=phase.variant.firstiter_score_mode_this_iter == "normalized_cc",
        )
    elif use_local:
        local_optics = optics_shapes.prepare_optics(
            particle_half.dataset,
            noise_radial=half.noise_radial,
            coarse_step_deg=coarse_size_step_deg,
            particle_diameter_ang=particle_diameter_ang,
            previous_translations=previous_translations_k,
            sigma_offset_angstrom=sigma_offset_angstrom,
            base_translations=sampling.base_translations,
            current_translations=phase.grid.translations,
            with_log_prior=False,
            zero_cold_center=True,
            dtype=scoring_dtype,
        )
        local_result = _score_half_local_in_bpref_scope(
            half=replace(half, mean_variance=None, image_seed_classes=seed_classes_k),
            sampling=sampling,
            priors=LocalPriorSpec(
                trans_prior_center=local_trans_prior_center,
                trans_prior_center_for_engine=trans_prior_center_for_engine,
                current_sigma_offset_angstrom=sigma_offset_angstrom,
                translation_search_base=translation_search_base,
                local_search_translation_prior_mode=(options.local_search.local_search_translation_prior_mode),
                replay_prior_translations=replay_prior_translations,
            ),
            batching=LocalBatchPolicy(
                max_significants=options.adaptive.max_significants,
                safe_batch_sizes=half_batching.safe_batch_sizes,
            ),
            execution=LocalExecutionPolicy(
                disc_type=options.disc_type,
                disable_adjoint_y=options.debug.disable_adjoint_y,
                disable_adjoint_ctf=options.debug.disable_adjoint_ctf,
                source_faithful_spectrum_norm=source_faithful_spectrum_norm,
                relion_translation_angle_scale=(relion_translation_angle_scale),
                nyquist_column_counting=options.consistency.nyquist_column_counting,
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
            noise_radial=half.noise_radial,
            coarse_step_deg=coarse_size_step_deg,
            particle_diameter_ang=particle_diameter_ang,
            previous_translations=previous_translations_k,
            sigma_offset_angstrom=sigma_offset_angstrom,
            base_translations=sampling.base_translations,
            current_translations=phase.grid.translations,
            with_log_prior=True,
            zero_cold_center=not k_class_enabled,
            dtype=scoring_dtype,
        )
        dense_half = replace(half, image_seed_classes=seed_classes_k)
        dense_sampling = sampling
        dense_priors = DensePriorSpec(
            rotation_log_prior_k=direction_priors.rotation_log_prior,
            class_rotation_log_prior_k=direction_priors.class_rotation_log_prior,
            translation_log_prior=translation_log_prior,
            translation_search_base=translation_search_base,
            trans_prior_center_for_engine=trans_prior_center_for_engine,
            class_log_priors=class_log_priors,
        )
        dense_batching = DenseBatchPolicy(
            image_batch_size=options.batching.image_batch_size,
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
            disc_type=options.disc_type,
            disable_adjoint_y=options.debug.disable_adjoint_y,
            disable_adjoint_ctf=options.debug.disable_adjoint_ctf,
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

        # --- Manifest dump for deterministic replay (Phase 0.1) ---
        if not phase.use_adaptive and options.debug.save_intermediates_dir is not None:
            _manifest_path = os.path.join(
                options.debug.save_intermediates_dir,
                f"manifest_iter{iteration}_half{k}.npz",
            )
            _manifest = {
                "effective_rotations": np.asarray(phase.grid.rotations),
                "coarse_scoring_rotations": _replay_manifest_array(
                    dense_sampling.coarse_scoring_rotations,
                ),
                "current_translations": np.asarray(phase.grid.translations),
                "rotation_log_prior": _replay_manifest_array(direction_priors.rotation_log_prior, dtype=np.float64),
                "translation_log_prior": _replay_manifest_array(translation_log_prior, dtype=np.float64),
                "image_corrections": _replay_manifest_array(
                    particle_half.image_corrections, dtype=np.float64,
                ),
                "scale_corrections": _replay_manifest_array(
                    particle_half.scale_corrections, dtype=np.float64,
                ),
                "image_pre_shifts": _replay_manifest_array(translation_search_base, dtype=np.float32),
                "absolute_previous_translations": _replay_manifest_array(
                    previous_translations_k, dtype=np.float32,
                ),
                "mean_vol_ft": np.asarray(half.reference),
                "mean_variance": np.asarray(half.mean_variance),
                "noise_variance": np.asarray(half.noise_variance),
                "current_size": np.int32(image_window_size) if image_window_size is not None else np.int32(-1),
                "half_spectrum_scoring": np.bool_(True),
                "use_float64_scoring": np.bool_(scoring_policy.DENSE_PRECISION.use_float64_scoring),
                "projection_padding_factor": np.int32(PROJECTION_PADDING_FACTOR),
                "reconstruction_padding_factor": np.int32(RECONSTRUCTION_PADDING_FACTOR),
                "score_with_masked_images": np.bool_(True),
                "perturbation_instance": np.float64(sampling.random_perturbation),
                "perturbation_factor": np.float64(options.parity.perturb_factor),
                "iteration": np.int32(iteration),
                "half_index": np.int32(k),
                "ave_Pmax": np.float64(float(np.mean(dense_result.em_stats.max_posterior_per_image))),
            }
            np.savez(_manifest_path, **_manifest)
            logger.info("Manifest dumped: %s", _manifest_path)

    score_result.translation_search_base = translation_search_base
    return score_result
