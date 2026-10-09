"""Orchestrate dense single-volume and K-class EM refinement.

``refine_single_volume`` validates options and manages refinement state and dispatch.
``half_scoring`` owns the per-half dense/local engine calls; ``scoring_policy``
owns their shared execution defaults and diagnostic selectors. Local chunks
are implemented in ``local_search_iteration``; state-swap diagnostics belong
to ``parity.state_swap_runtime``. Pure trial-grid construction belongs to
``sampling``.
See ``docs/math/relion_refinement_algorithm.md`` for the algorithm map.
"""


import logging
from dataclasses import replace
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from recovar.data_io import cryoem_dataset

import relax.parity.relion_replay as replay_policy
from relax.dense.score_outputs import (
    PerHalfOutputs,
    _resolve_mstep_accumulator_shape,
    _resolve_mstep_full_half_axis,
)
from relax.diagnostics import bpref_diagnostics
from relax.diagnostics.iteration import (
    _significance_dump_half_indices,
)
from relax.diagnostics.reconstruction import check_half_accumulators_before_join
from relax.helpers.convergence import (
    _direction_prior_healpix_order_for_scoring,
    _exhaustive_grid_order_for_state,
    _relion_pmax_normalization_mass_per_half,
    check_convergence,
    concatenate_assignments,
    concatenate_assignments_or_none,
    expectation_statistics,
    hard_class_change_fraction,
)
from relax.helpers.expected_accuracy import estimate_iteration_accuracy
from relax.helpers.iteration_history import RefinementHistory
from relax.helpers.orientation_priors import (
    initial_direction_priors_from_snapshot,
    learn_class_direction_priors,
    learn_k1_direction_priors,
    relion_direction_log_priors,
)
from relax.helpers.resolution import (
    ImageGeometry,
    _zero_shells_past_current_size,
    estimate_class_iteration_resolution,
    estimate_k1_iteration_resolution,
    relion_expectation_coarse_size_order,
    shell_index_to_resolution_angstrom,
)
from relax.helpers.timing import Stopwatch
from relax.parity.relion_replay import _validate_bpref_particle_order_scope
from relax.reconstruction.regularization_relion import (
    update_relion_growth_state_from_fsc,
)
from relax.refinement import finalization
from relax.refinement.convergence import (
    advance_expectation_sampling,
    reset_follower_counter_once,
    update_iteration_convergence,
    uses_native_auto_refine,
)
from relax.refinement.expectation import (
    NumberedHalfInputs,
    NumberedHalfRecording,
    SignificanceStatistics,
    finish_numbered_half,
    prepare_numbered_expectation,
    run_numbered_halves,
    score_numbered_half,
)
from relax.refinement.expectation_batches import BatchPlanner
from relax.refinement.half_inputs import (
    SigmaOffset,
    _sigma_offset_for_half,
    as_sigma_offset_half_pair,
    best_rotation_matrices,
    configure_half_image_preprocessing,
    copy_optional_float_pair,
    initialize_halfsets,
    normalize_sigma_offset_per_half,
    prepare_particle_pose_update,
    prepare_pose_comparison,
    sigma_offset_from_halves,
)
from relax.refinement.half_scoring import (
    DenseVariantPolicy,
    HalfScoringData,
)
from relax.refinement.iteration_planning import (
    adaptive_pass1_size,
    build_initial_coarse_grids,
    builds_coarse_pass1_rotations,
    class_seeding,
    coarse_pass1_rotations,
    first_iteration_policy,
    initial_random_perturbation,
    initialize_refinement_state,
    iteration_trial_grid,
    plan_adaptive_image_size,
    plan_class_image_size,
    plan_halfmap_image_size,
    plan_initial_image_size,
    refresh_coarse_grids,
    resolve_current_size,
    resolve_numbered_perturbation,
    strict_e_step_size,
)
from relax.refinement.iteration_snapshot import SnapshotCapture, direction_priors_from_snapshot
from relax.refinement.local_sampling import local_search_centre_half, plan_expectation_sampling
from relax.refinement.maximization import (
    class_maximization,
    copy_first_class_to_every_class,
    k1_maximization,
)
from relax.refinement.mean_helpers import (
    _class_weights_from_posterior,
    _initialize_class_log_priors,
    _normalize_initial_means,
    align_k1_volume_signs,
    class_mixture_from_weights,
    initialize_reference_model,
    merged_half_map,
    reconstruct_unregularized_class_means,
    reconstruct_unregularized_k1_halfmaps,
    reference_model_from_snapshot,
    shared_tau2_per_half,
)
from relax.refinement.noise_updates import (
    _mean_noise_variance,
    _normalize_noise_variance_per_half,
    datasets_store_premultiplied_ctf,
    initialize_noise_model,
    noise_model_from_shells,
    update_c1_sigma_offset_from_posterior,
    update_posterior_noise_variance,
)
from relax.refinement.optics_shapes import MultiShapeHalf
from relax.refinement.ports import (
    FinalState,
    FinishedIteration,
    InputSource,
    NumberedState,
    ReconstructedIteration,
    RunObserver,
    ScoringArrays,
    ScoringState,
)
from relax.refinement.projector_preparation import (
    build_numbered_projectors,
    prepare_initial_real_references,
)
from relax.refinement.refinement_options import (
    FINAL_ALL_DATA_USE_MERGED_REFERENCE_ENV,
    RefinementOptions,
    require_consistency_route,
    require_process_precision,
    with_validated_sampling_schedule,
)
from relax.refinement.refinement_result import (
    NumberedMetadata,
    ProfileStop,
    RefinementResult,
    ReplayTelemetry,
)
from relax.refinement.setup_checks import (
    checked_optics_group_ids,
    checked_run_optics,
    expected_accuracy_inputs_for_run,
    reconstruction_settings_for_run,
    translation_angle_scale_for_run,
)
from relax.refinement.tomo_half import TomoHalf, numbered_iteration_tomo_sampling
from relax.relion.geometry import (
    RECONSTRUCTION_PADDING_FACTOR,
)
from relax.relion.relion_normalization import (
    NormScaleCorrectionReport,
    log_norm_scale_update,
    norm_scale_report,
    numbered_norm_scale_update,
)
from relax.relion.relion_worker_scale import (
    _dispatch_relion_follower_scale_for_final_all_data,
    _dispatch_relion_follower_scale_for_numbered_iteration,
    _finalize_relion_follower_scale_replay_telemetry,
    _update_relion_follower_corrections,
    setup_relion_follower_scale_state,
)
from relax.sampling import (
    rotation_grid_size,
)
from relax.sparse_pass2.engine_record import take_coarse_engine_calls, take_pass_engines

logger = logging.getLogger(__name__)




def _should_use_adaptive_search(state, options: RefinementOptions, *, use_local: bool, n_rotations: int) -> bool:
    """Keep non-C1 refinement on its supported sparse/x-half route.

    Reads ``state.adaptive_oversampling`` and ``options.symmetry.point_group``.

    Small C1 grids may use the direct dense path. Point-group symmetry cannot:
    symmetry reduction itself can make a valid grid smaller than that cutoff
    (O has 12 coarse rotations at HEALPix order 1, I1 fewer), while its
    scoring and reconstruction still require the adaptive RELION x-half path.
    Ported from final Q 22efd8065.
    """

    if int(state.adaptive_oversampling) <= 0 or bool(use_local):
        return False
    return int(n_rotations) > 16 or str(options.symmetry.point_group).upper() != "C1"


class PublishedAccuracy(NamedTuple):
    """The latest completed expected-accuracy estimate, as the run reports it.

    Its trials (half-1 local rows and particle ids; None before the first estimate) and RELION's per-class
    ``MlModel::acc_rot``/``acc_trans`` for model.star (zero until the first estimate, ml_model.cpp:68).
    """

    trial_local_indices: np.ndarray | None
    trial_particle_ids: np.ndarray | None
    acc_rot_per_class: np.ndarray
    acc_trans_per_class_angstrom: np.ndarray

    @classmethod
    def before_first_estimate(cls, n_classes, acc_rot_per_class=None, acc_trans_per_class_angstrom=None):
        """No estimate yet in this run: no trials; the continued run's per-class values, else zeros."""
        if acc_rot_per_class is None:
            return cls(None, None, np.zeros(n_classes, dtype=np.float64), np.zeros(n_classes, dtype=np.float64))
        return cls(
            None,
            None,
            np.array(acc_rot_per_class, dtype=np.float64),
            np.array(acc_trans_per_class_angstrom, dtype=np.float64),
        )


def _follower_replay_telemetry(source, history) -> ReplayTelemetry:
    """The follower-scale replay's requested and applied iterations (both None without a replay), validated
    against the iterations the run applied; raises if the replay was not applied as requested."""

    requested, applied = _finalize_relion_follower_scale_replay_telemetry(
        source, applied_iterations=history.relion_follower_scale_replay_applied_iterations, logger=logger,
    )
    return ReplayTelemetry(requested_iterations=requested, applied_iterations=applied)


def _numbered_dense_variant(first_iteration, k_class, *, use_adaptive: bool, coarse_cs, fine_window_size):
    """A numbered iteration's dense route: its first-iteration mode, the adaptive sizes and the class options.

    ``coarse_cs`` is the adaptive pass-1 size, None off the adaptive route; ``fine_window_size`` is the pass-2
    scoring window (the E-step size).
    """

    return DenseVariantPolicy(
        firstiter_score_mode_this_iter=first_iteration.score_mode,
        firstiter_winner_take_all_this_iter=first_iteration.winner_take_all,
        k_class_enabled=int(k_class.n_classes) > 1,
        relion_firstiter_cc_this_iter=first_iteration.relion_firstiter_cc,
        firstiter_coarse_current_size=coarse_cs,
        firstiter_fine_current_size=fine_window_size if use_adaptive else None,
        firstiter_log_label="" if use_adaptive else "(non-adaptive site) ",
        skip_align=bool(k_class.skip_align),
    )


def refine_single_volume(
    experiment_datasets: list[cryoem_dataset.CryoEMDataset],
    init_volume: list[jnp.ndarray] | jnp.ndarray,
    init_noise_variance: jnp.ndarray,
    init_mean_variance: jnp.ndarray,
    translations: jnp.ndarray | None,
    options: RefinementOptions,
    observer: RunObserver | None = None,
    source: InputSource | None = None,
) -> RefinementResult:
    """Multi-iteration RELION-parity EM refinement.

    Run it inside ``relax.sparse_pass2.resident_pass2.stable_window_class_history()``, as the commands do, so a
    refinement's resident passes reuse the window classes they already ran (about 5 s of recompiles saved per
    current-size crossing on the 5k K=1 run); outside one, every pass picks its class afresh.

    Implements the loop described in ``docs/math/relion_refinement_algorithm.md``.

    Parameters
    ----------
    experiment_datasets : list of 2 dataset objects
        Half-set datasets (same format as run_halfset_em_iteration expects).
    init_volume : list of 2 jnp.ndarray, shape (volume_size,) or jnp.ndarray, shape (volume_size,)
        Initial volume in Fourier space for each half-set.
    init_noise_variance : jnp.ndarray, shape (2,image_size)
        Initial per-pixel noise variance for each half-set.
    init_mean_variance : jnp.ndarray, shape (volume_size,)
        Initial signal prior (tau^2).
    translations : jnp.ndarray, shape (n_trans, 2)
        Translation grid.
    options : `RefinementOptions` struct that bundles the schedule / adaptive / parity
        / local-search / K-class / replay / debug / batching kwarg groups, built by the command.
    observer : the run's ``RunObserver`` (``relax.refinement.ports``): dumps and captures that watch the run
        and never change it. Defaults to one that does nothing.
    source : the run's ``InputSource`` (``relax.refinement.ports``): what a comparison run takes from
        elsewhere (a RELION run) instead of computing it. Defaults to the native source.

    Returns
    -------
    RefinementResult (``relax.refinement.refinement_result``): the maps (``maps.mean``, ``maps.means``),
    the last ``convergence_state``, the run's ``history`` (``current_sizes``, ``fsc_history``,
    ``pixel_resolutions``, ``wall_times``, ``significant_counts``, the trajectories), the set-up and
    numbered-iteration metadata (``numbered``), the final pass's outputs (``final_pass``, None when it did
    not run) and ``profile_stop`` for a local-search diagnostic stop. ``archive_fields()`` is the flat
    mapping the archive and reports read.
    """
    if observer is None:
        observer = RunObserver()
    if source is None:
        source = InputSource()

    options = with_validated_sampling_schedule(options)
    require_process_precision(options)
    # The dtype of the controller's float64-sensitive host operands (rotation grids, priors), from the precision.
    scoring_dtype = options.precision.rotation_real_dtype

    # Each set-up phase's cumulative seconds since the set-up started (a result: the archive and the ledger).
    setup_clock = Stopwatch()
    setup_phase_seconds = {}

    volume_shape = experiment_datasets[0].volume_shape
    # Keep the input scalar type for host arithmetic; geometry validates its value.
    source_pixel_size_angstrom = experiment_datasets[0].voxel_size
    image_geometry = ImageGeometry(
        image_shape=experiment_datasets[0].image_shape, pixel_size_angstrom=source_pixel_size_angstrom,
    )
    k_class_enabled = options.k_class.n_classes > 1
    multi_shape_halves = isinstance(experiment_datasets[0], MultiShapeHalf)
    optics = checked_run_optics(options.parity, image_geometry, multi_shape_halves=multi_shape_halves)
    # Subtomogram particles (S4.2): units are particles over their tilt images, offsets are 3D.
    tomo_halves = isinstance(experiment_datasets[0], TomoHalf)
    # Opt-in corrections of RELION's inconsistencies, refused on the routes that keep RELION's rules.
    consistency = require_consistency_route(
        options, subtomograms=tomo_halves, several_image_shapes=multi_shape_halves,
        replays_relion_state=source.replays_relion_state(),
    )
    relion_translation_angle_scale = translation_angle_scale_for_run(
        optics, n_classes=options.k_class.n_classes, subtomograms=tomo_halves,
    )
    _validate_bpref_particle_order_scope(
        preserve_bpref_particle_order=options.parity.preserve_bpref_particle_order,
        n_classes=options.k_class.n_classes,
        init_relion_iteration=options.schedule.init_relion_iteration,
        perturb_replay_relion_dir=None if source.relion_replay is None else source.relion_replay.perturb_replay_relion_dir,
        replay_iteration_overrides=(
            None if source.relion_replay is None else source.relion_replay.replay_iteration_overrides
        ),
        sealed_sampling_state=source.sealed_sampling_state,
        sealed_scoring_context=source.sealed_scoring_context,
        allow_replayed_bpref_particle_order=options.parity.allow_replayed_bpref_particle_order,
        allow_state_swap_fresh_bpref_particle_order=source.swaps_state,
        continues_own_run=options.checkpoint.resume is not None,
    )
    class_mixture = _initialize_class_log_priors(options.k_class.n_classes, options.replay.init_direction_prior)

    reconstruction_settings = reconstruction_settings_for_run(options, image_geometry, volume_shape, consistency)
    snapshot_capture = SnapshotCapture(
        n_classes=options.k_class.n_classes, box_size=image_geometry.box_size,
        voxel_size=image_geometry.pixel_size_angstrom, tau2_fudge=options.parity.tau2_fudge,
        consistency=consistency.non_default(),
    )

    configure_half_image_preprocessing(
        experiment_datasets,
        pixel_size_angstrom=source_pixel_size_angstrom,
        particle_diameter_angstrom=options.schedule.particle_diameter_ang,
        width_mask_edge_px=options.schedule.width_mask_edge_px,
        fourier_backend=options.parity.image_fourier_backend,
        # RELION's source-faithful powerClass normalisation applies wherever its particle order is preserved.
        source_faithful_spectrum_norm=options.parity.preserve_bpref_particle_order,
        log=logger,
    )

    setup_phase_seconds["mask_and_image_cache"] = setup_clock.seconds

    state = initialize_refinement_state(
        options, image_geometry, subtomogram=tomo_halves, dtype=scoring_dtype, source=source,
    )
    resume = options.checkpoint.resume
    setup_phase_seconds["state_init"] = setup_clock.seconds

    # The refinement schedule owns the initial coarse HEALPix grid; a continuation
    # rebuilds the grid of its restored sampling state.
    initial_grid_order = (
        int(options.schedule.init_healpix_order) if resume is None else _exhaustive_grid_order_for_state(state)
    )
    coarse_grids = source.initial_coarse_grids(
        initialized_healpix_order=options.schedule.init_healpix_order if resume is None else state.healpix_order,
        voxel_size=source_pixel_size_angstrom, symmetry=options.symmetry.point_group,
        native=partial(
            build_initial_coarse_grids, initial_grid_order, translations if resume is None else None,
            translation_range=options.schedule.init_translation_range if resume is None else state.translation_range,
            translation_step=options.schedule.init_translation_step if resume is None else state.translation_step,
            n_classes=options.k_class.n_classes, voxel_size=source_pixel_size_angstrom,
            symmetry=options.symmetry.point_group, dtype=scoring_dtype,
        ),
    )
    # coarse_grids keeps the unperturbed host-RFLOAT base translations: each iteration perturbs a fresh copy.
    collect_local_search_profile = options.local_search.collects_profile(observer.collects_local_search_profiles)
    setup_phase_seconds["sampling_grid"] = setup_clock.seconds

    padded_volume_shape = tuple(d * RECONSTRUCTION_PADDING_FACTOR for d in volume_shape)

    batch_planner = BatchPlanner(
        requested=options.execution, image_shape=image_geometry.image_shape, volume_shape=volume_shape,
        n_classes=options.k_class.n_classes, precision=options.precision, log=logger,
    )

    initial_real_references_by_half = prepare_initial_real_references(
        options.replay.init_reference_real, volume_shape=volume_shape, n_classes=options.k_class.n_classes,
        init_relion_iteration=options.schedule.init_relion_iteration, log=logger,
    )
    initial_noise_variance_per_half = _normalize_noise_variance_per_half(init_noise_variance)
    optics_group_ids_per_half = checked_optics_group_ids(
        options.parity.optics_group_ids_per_half, initial_noise_variance_per_half, experiment_datasets
    )
    setup_phase_seconds["initial_arrays"] = setup_clock.seconds

    history = RefinementHistory(keep_rotation_posteriors=observer.keeps_rotation_posteriors)
    take_pass_engines()  # entries from before this run's first iteration belong to no iteration
    take_coarse_engine_calls()
    previous_assignments = [None, None]
    halves = initialize_halfsets(
        experiment_datasets, optics_group_ids=optics_group_ids_per_half,
        previous_best_translations=options.replay.init_previous_best_translations,
        previous_best_rotation_eulers=options.replay.init_previous_best_rotation_eulers,
        image_corrections=options.replay.init_image_corrections,
        scale_corrections=options.replay.init_scale_corrections, group_ids=options.replay.init_group_ids,
        group_count=options.replay.init_group_count,
    )
    if int(options.schedule.init_relion_incr_size) <= 0:
        raise ValueError("init_relion_incr_size must be positive")

    expected_accuracy_inputs = expected_accuracy_inputs_for_run(
        options, experiment_datasets[0], volume_shape,
        optics_group_ids=optics_group_ids_per_half[0], gridding_kernel=consistency.gridding_kernel,
    )

    follower_setup = setup_relion_follower_scale_state(
        options,
        topology=source.follower_topology,
        relion_half_inputs=halves,
        experiment_datasets=experiment_datasets,
        k_class_enabled=k_class_enabled,
        # The STAR replay's restart iterations (a strict Class3D replay restarts its follower scales with them).
        restart_state_iterations=(
            () if source.relion_replay is None else source.relion_replay.perturb_replay_restart_state_iterations
        ),
    )
    # The replayed run's follower-scale replay (None without one).
    follower_scale_replay = None if source.follower_topology is None else source.follower_topology.replay

    if resume is None:
        # --- A fresh run starts from the caller's references, noise and replayed particle state ---
        # Each half stores its references in the loop's layout: an explicit leading
        # class axis for K classes, one flat reference for K=1.
        reference_model = initialize_reference_model(
            _normalize_initial_means(init_volume, options.k_class.n_classes), jnp.asarray(init_mean_variance),
            use_per_half_mean_variance=options.parity.use_per_half_mean_variance, k_class_enabled=k_class_enabled,
            dtype=scoring_dtype, log=logger,
        )
        # Per-shell radial profiles of the input pixel-array noise variances, for the
        # diagnostic log ("noise update per shell: old=... new=...").
        noise_model = initialize_noise_model(
            initial_noise_variance_per_half, average_variance=_mean_noise_variance(initial_noise_variance_per_half),
            image_shape=image_geometry.image_shape, dtype=scoring_dtype,
        )
        # Drop the start-up arrays: where the caller passed temporaries (the parity script), the models
        # are then their only holders and the first updates free them; relax refine keeps its own copies.
        del init_volume, init_mean_variance, initial_noise_variance_per_half
        class_assignments = [None, None]
        previous_class_assignments = [None, None]
        previous_data_vs_prior_for_scheduling = (
            None
            if options.schedule.init_data_vs_prior is None
            else np.asarray(options.schedule.init_data_vs_prior, dtype=scoring_dtype)
        )
        # RELION's sigma2_offset in Angstrom^2 (min_sigma2_offset=2 A^2), updated each iteration from the data.
        sigma_offset = sigma_offset_from_halves(
            as_sigma_offset_half_pair(options.schedule.init_translation_sigma_angstrom)
        )
        relion_incr_size = int(options.schedule.init_relion_incr_size)
        relion_has_high_fsc_at_limit = (
            bool(options.schedule.init_has_high_fsc_at_limit)
            if options.schedule.init_has_high_fsc_at_limit is not None
            else False
        )
        direction_priors = initial_direction_priors_from_snapshot(
            options.replay.init_direction_prior, n_classes=options.k_class.n_classes, dtype=scoring_dtype, log=logger,
            symmetry=options.symmetry.point_group, expected_order=coarse_grids.rotation_grid.healpix_order,
        )
        random_perturbation = initial_random_perturbation(options, log=logger)
        published_accuracy = PublishedAccuracy.before_first_estimate(options.k_class.n_classes)
    else:
        # --- A continued run starts from the run files of an earlier run (RELION --continue) ---
        # The snapshot replaces every value the next numbered iteration reads, so the
        # first loop iteration runs as iteration init_relion_iteration + 1 of the
        # uninterrupted run (see relax/refinement/iteration_snapshot.py). It reads none of
        # the start-up arrays: they are released before the snapshot's model is built.
        del init_volume, init_mean_variance, initial_noise_variance_per_half
        reference_model = reference_model_from_snapshot(
            resume, volume_shape, k_class_enabled=k_class_enabled, dtype=scoring_dtype,
        )
        noise_model = noise_model_from_shells(resume.noise_shells, image_geometry.image_shape)
        for half, eulers, translations, images, scales in zip(
            halves, resume.rotation_eulers, resume.translations,
            resume.image_corrections, resume.scale_corrections, strict=True,
        ):
            half.rotation_eulers = eulers
            half.translations = translations
            half.image_corrections = images
            half.scale_corrections = scales
        class_assignments = [None, None]
        previous_class_assignments = [None, None]
        if k_class_enabled:
            class_assignments = [None if c is None else np.asarray(c) for c in resume.class_assignments]
            previous_class_assignments = [None if c is None else c.copy() for c in class_assignments]
            class_mixture = class_mixture_from_weights(np.asarray(resume.class_weights, dtype=np.float64))
        previous_data_vs_prior_for_scheduling = np.asarray(resume.data_vs_prior, dtype=scoring_dtype)
        sigma_offset = sigma_offset_from_halves(as_sigma_offset_half_pair(resume.sigma_offset_angstrom))
        relion_incr_size = int(resume.incr_size)
        relion_has_high_fsc_at_limit = bool(resume.has_high_fsc_at_limit)
        direction_priors = direction_priors_from_snapshot(
            resume, options.replay.init_direction_prior, n_classes=options.k_class.n_classes,
            grid_healpix_order=coarse_grids.rotation_grid.healpix_order, symmetry=options.symmetry.point_group,
            dtype=scoring_dtype, log=logger,
        )
        random_perturbation = resume.random_perturbation
        published_accuracy = PublishedAccuracy.before_first_estimate(
            options.k_class.n_classes, resume.acc_rot_per_class, resume.acc_trans_per_class_angstrom
        )
        logger.info(
            "Continuing after numbered iteration %d: current_size=%d healpix_order=%d "
            "local_search=%s resolution=%.3f A",
            int(resume.relion_iteration), int(resume.current_size), int(state.healpix_order),
            bool(state.do_local_search), float(state.current_resolution),
        )
    # Both start-up states end here; the archive keeps the two phase names it has always had.
    setup_phase_seconds["direction_prior"] = setup_phase_seconds["noise_radial_init"] = setup_clock.seconds
    # RELION measures the first iteration's orientation changes from the input angles, as its offset
    # changes from the input offsets (updateOverallChangesInHiddenVariables); they seed the smallest-change
    # trackers of the hidden-variable stall counter. An empty half keeps an empty stack, as the loop does.
    previous_best_rotations = best_rotation_matrices(halves, dtype=scoring_dtype)
    perturb_rng = None if options.parity.perturb_seed is not None else np.random.default_rng()
    iteration = 0
    setup_phase_seconds["before_iterations"] = setup_clock.seconds
    logger.info(
        "RELION mode setup timing before iteration loop: %s",
        ", ".join(f"{key}={value:.1f}s" for key, value in setup_phase_seconds.items()),
    )
    # The RELION run whose numbered STAR files supply the sampling (an input source's; None natively).
    star_directory = source.relion_run_directory(iteration - 1)
    native_sampling_boundary = source.relion_run_directory(iteration) is None and source.sealed_sampling_state is None
    # A numbered RELION sampling STAR is the state *after* the expectation
    # transition that produced it.  The next expectation computes
    # image_coarse_size from that saved state before updateAngularSampling.
    # Keep this replay boundary separate from RECOVAR's end-of-iteration
    # RefinementState, which may already have advanced one order.
    replay_saved_healpix_order = (
        None if native_sampling_boundary else int(state.healpix_order)
    )
    frozen_initial_scoring_state_sha256 = None
    source.scoring_state_bound(
        ScoringArrays(reference_model, noise_model, sigma_offset, halves, direction_priors, experiment_datasets)
    )
    # Per-half numbered-iteration assignments; a final-only replay
    # (--max_iter 0 --force-final-after-zero-iterations) runs no numbered
    # iteration and reports none.
    hard_assignments = [None, None]
    # Set when a local-search diagnostic stops the run after its first local search.
    profile_stop = None
    while (
        options.schedule.force_max_iter_after_convergence or not state.has_converged
    ) and iteration < options.schedule.max_iter:
        # A continued run's first iteration follows the snapshot's iteration.
        has_previous_iteration = iteration > 0 or resume is not None
        seeding = class_seeding(options, continued=resume is not None, iteration=iteration)
        star_directory = source.relion_run_directory(iteration)
        native_sampling_boundary = star_directory is None and source.sealed_sampling_state is None
        if native_sampling_boundary:
            replay_saved_healpix_order = None
        # RELION checks convergence at the top of iteration n from the
        # completed n-1 statistics and the fine-enough decision latched during
        # expectation n-1.  If true, iteration n is the unnumbered joined
        # all-data pass rather than another numbered half-set iteration.
        if (
            uses_native_auto_refine(
                native_sampling_boundary=native_sampling_boundary,
                n_classes=options.k_class.n_classes,
            )
            and not options.schedule.force_max_iter_after_convergence
            and has_previous_iteration
            and check_convergence(state)
        ):
            state.has_converged = True
            logger.info(
                "Convergence reached after numbered iteration %d. "
                "Entering RELION final all-data iteration.",
                iteration,
            )
            break
        iteration_clock = Stopwatch()
        observer.iteration_started(iteration)
        first_iteration = first_iteration_policy(options, iteration=iteration)
        numbered_relion_iteration = replay_policy._numbered_relion_iteration(
            options.schedule.init_relion_iteration, iteration
        )

        if follower_setup.follower_scale_state is not None:
            _dispatch_relion_follower_scale_for_numbered_iteration(
                follower_setup, history, iteration=iteration, numbered_relion_iteration=numbered_relion_iteration,
                relion_half_inputs=halves, relion_follower_scale_replay_source=follower_scale_replay,
                dtype=scoring_dtype, logger=logger,
            )

        # Image support uses the preceding iteration's spectra, before replay
        # and angular sampling select this expectation's grid.
        if not has_previous_iteration:
            image_size_plan = plan_initial_image_size(
                options, box_size=image_geometry.box_size, pixel_size_angstrom=source_pixel_size_angstrom,
                incr_size=relion_incr_size, has_high_fsc_at_limit=relion_has_high_fsc_at_limit,
                dtype=scoring_dtype, log=logger,
            )
            if image_size_plan.data_vs_prior is not None:
                previous_data_vs_prior_for_scheduling = image_size_plan.data_vs_prior
            relion_incr_size = image_size_plan.incr_size
            relion_has_high_fsc_at_limit = image_size_plan.has_high_fsc_at_limit
        else:
            prev_cs = history.current_sizes[-1] if history.current_sizes else int(resume.current_size)
            if k_class_enabled:
                image_size_plan = plan_class_image_size(
                    previous_data_vs_prior_for_scheduling, previous_size=prev_cs, box_size=image_geometry.box_size,
                    pixel_size_angstrom=source_pixel_size_angstrom, incr_size=relion_incr_size, ave_pmax=state.ave_Pmax,
                    completed_relion_iteration=int(options.schedule.init_relion_iteration) + int(iteration),
                    parity=options.parity, dtype=scoring_dtype, log=logger,
                )
                observer.class_image_size_planned(
                    iteration, image_size_plan, previous_size=prev_cs, box_size=image_geometry.box_size,
                    has_high_fsc_at_limit=relion_has_high_fsc_at_limit, incr_size=relion_incr_size, state=state,
                )
            else:
                image_size_plan = plan_halfmap_image_size(
                    history.fsc_history, growth_fsc_history=history.fsc_for_growth_history, restart=resume,
                    data_vs_prior=previous_data_vs_prior_for_scheduling, previous_size=prev_cs,
                    box_size=image_geometry.box_size, pixel_size_angstrom=source_pixel_size_angstrom,
                    incr_size=relion_incr_size, has_high_fsc_at_limit=relion_has_high_fsc_at_limit,
                    ave_pmax=state.ave_Pmax,
                    completed_relion_iteration=int(options.schedule.init_relion_iteration) + int(iteration),
                    parity=options.parity, dtype=scoring_dtype, log=logger,
                )
                previous_data_vs_prior_for_scheduling = image_size_plan.data_vs_prior
                relion_incr_size = image_size_plan.incr_size
                relion_has_high_fsc_at_limit = image_size_plan.has_high_fsc_at_limit

        current_size = resolve_current_size(
            image_size_plan, options, previous_size=prev_cs if has_previous_iteration else None,
            incr_size=relion_incr_size, has_high_fsc_at_limit=relion_has_high_fsc_at_limit,
            ave_pmax=state.ave_Pmax, iteration=iteration, box_size=image_geometry.box_size, log=logger,
        )

        # RELION updates image_coarse_size before updateAngularSampling at the
        # start of expectation(). Preserve that incoming sampling order even
        # when replay/native scheduling advances state.healpix_order below.
        coarse_size_healpix_order = relion_expectation_coarse_size_order(
            state_healpix_order=state.healpix_order, replay_saved_healpix_order=replay_saved_healpix_order,
        )

        source.state_swap_snapshot(
            iteration,
            ScoringState(
                state, current_size, reference_model, noise_model, halves, previous_best_rotations, sigma_offset,
                direction_priors,
            ),
        )
        # The input source supplies the state this iteration scores with (the native source: the run's own).
        numbered = source.numbered_state(
            iteration,
            NumberedState(
                current_size=current_size, noise_model=noise_model, sigma_offset=sigma_offset,
                previous_best_rotations=previous_best_rotations, mean_variance=reference_model.tau2,
                class_mixture=class_mixture, prior_translations=None, sampling_healpix_order=None,
            ),
            state=state, halves=halves, direction_priors=direction_priors, image_geometry=image_geometry,
        )
        current_size = numbered.current_size
        _replay_prior_translations = numbered.prior_translations
        previous_best_rotations = numbered.previous_best_rotations
        noise_model = numbered.noise_model
        reference_model.tau2 = numbered.mean_variance
        class_mixture = numbered.class_mixture
        # Both halves' translation prior widths, a scalar width repeated.
        sigma_offset = SigmaOffset(
            numbered.sigma_offset.shared_angstrom,
            as_sigma_offset_half_pair(
                numbered.sigma_offset.shared_angstrom
                if numbered.sigma_offset.per_half_angstrom is None
                else numbered.sigma_offset.per_half_angstrom
            ),
        )
        if replay_saved_healpix_order is not None:
            replay_saved_healpix_order = int(state.healpix_order)

        reference_model.maps = source.scoring_references(iteration, reference_model, volume_shape=volume_shape)
        swapped = source.swapped_state(
            iteration,
            ScoringState(
                state, current_size, reference_model, noise_model, halves, previous_best_rotations, sigma_offset,
                direction_priors,
            ),
            volume_shape=volume_shape,
        )
        if swapped is not None:
            (
                current_size,
                reference_model,
                noise_model,
                previous_best_rotations,
                swapped_sigma_offset_angstrom,
                swapped_sigma_offset_angstrom_per_half,
                direction_priors,
            ) = swapped
            sigma_offset = SigmaOffset(swapped_sigma_offset_angstrom, swapped_sigma_offset_angstrom_per_half)
            history.state_swap_probe_applied_relion_iterations.append(
                int(options.schedule.init_relion_iteration) + int(iteration) + 1
            )
        if not options.parity.use_per_half_mean_variance:
            # State-swap diagnostics historically replace the one shared tau2.
            # Do not leave the scorer pointing at pre-swap aliases.
            reference_model.tau2_per_half = shared_tau2_per_half(reference_model.tau2)
        checked = source.scoring_state_checked(
            iteration,
            ScoringArrays(reference_model, noise_model, sigma_offset, halves, direction_priors, experiment_datasets),
        )
        if checked is not None:
            frozen_initial_scoring_state_sha256 = checked

        # Half 1's projector of this iteration's references, built for the
        # expected-accuracy estimate and reused by the scoring projector setup
        # below: RELION computes each class's projector once per iteration.
        # Release the previous iteration's projector before the estimate builds this one.
        shared_projector_half1 = None
        iteration_accuracy, shared_projector_half1 = estimate_iteration_accuracy(
            expected_accuracy_inputs, reference_model.maps[0], best_eulers_deg=halves[0].rotation_eulers,
            class_assignments=class_assignments[0], class_weights=class_mixture.weights,
            sigma2_noise_native=noise_model.radial_per_half[0], current_size=current_size,
            accuracy_image_size=strict_e_step_size(current_size, optics, options),
            image_box_size=image_geometry.box_size, n_classes=options.k_class.n_classes, iteration=iteration,
            native_sampling_boundary=native_sampling_boundary,
            relion_firstiter_cc_this_iter=first_iteration.relion_firstiter_cc,
            build_shared_projector=has_previous_iteration, log=logger,
        )
        if iteration_accuracy.sampling_accuracy is not None:
            # The estimate, or infinity when it was due and could not be made (convergence stays fail-closed).
            state.acc_rot, state.acc_trans = iteration_accuracy.sampling_accuracy
        if iteration_accuracy.published:
            published_accuracy = PublishedAccuracy(
                iteration_accuracy.trial_local_indices, iteration_accuracy.trial_particle_ids,
                iteration_accuracy.acc_rot_per_class.copy(), iteration_accuracy.acc_trans_per_class_angstrom.copy(),
            )

        # Accuracy and the preceding iteration's stall counters select this
        # expectation's grid; completed-iteration updates remain after M-step.
        state = advance_expectation_sampling(
            state, options.adaptive, iteration=iteration,
            may_advance_natively=has_previous_iteration and uses_native_auto_refine(
                native_sampling_boundary=native_sampling_boundary, n_classes=options.k_class.n_classes,
            ),
            log=logger,
        )

        history.record_scheduling(
            current_size, state.healpix_order, float(sigma_offset.shared_angstrom),
            copy_optional_float_pair(sigma_offset.per_half_angstrom),
        )
        scoring_current_size = int(current_size)

        logger.info(
            "=== RELION Iteration %d/%d: current_size=%d, healpix_order=%d, local_search=%s ===", iteration + 1,
            options.schedule.max_iter, scoring_current_size, state.healpix_order, state.do_local_search,
        )

        # --- Rotation grid at the state's order: from order 5 the base grid stays at order 4 (the full order-5
        # grid has 2.4M+ rotations) and local search plus oversampling give the finer steps. ---
        coarse_grids = source.coarse_grids(
            iteration, coarse_grids, state, voxel_size=source_pixel_size_angstrom, dtype=scoring_dtype,
        )
        coarse_grids = refresh_coarse_grids(
            coarse_grids, state, options, voxel_size=source_pixel_size_angstrom, dtype=scoring_dtype, log=logger,
        )

        # --- Local angular search: each image searches around its previous exact rotation at the current order ---
        adaptive_pass1 = None
        if state.do_local_search:
            # A half without poses is centred at Euler angles (0, 0, 0) with zero offsets.
            for half in halves:
                half.centre_absent_poses(offset_dims=3 if tomo_halves else 2)
        use_local = state.do_local_search
        if use_local and k_class_enabled and options.local_search.sigma_ang_deg is None:
            # Class3D searches locally only with --sigma_ang: RELION switches from the HEALPix
            # order only under auto-refine (ml_optimiser.cpp:2541-2565, 3936-3938).
            raise RuntimeError("K>1 (Class3D) reached local angular searches without --sigma_ang")
        # --- RELION's SamplingPerturbation of the trial grid (healpix_sampling.cpp:1810-1820, 1909-1934): a rigid
        # rotation applied after oversampling; at OS0 the coarse grid is the trial grid. ---
        random_perturbation = source.random_perturbation(
            iteration,
            native=partial(
                resolve_numbered_perturbation, random_perturbation, options, iteration=iteration, rng=perturb_rng,
                log=logger,
            ),
        )
        # The HEALPix order whose angular step scales the perturbation: a replayed sampling's (RELION's grid
        # order; the run's may be capped at the exhaustive-grid order), else the grid's; None: none applies.
        perturbation_order = (
            numbered.sampling_healpix_order
            if numbered.sampling_healpix_order is not None
            else coarse_grids.rotation_grid.healpix_order if options.parity.perturb_factor > 0 else None
        )
        trial_grid = iteration_trial_grid(
            coarse_grids, state, options, random_perturbation, perturbation_order=perturbation_order,
            sealed_grid=source.sealed_sampling_state is not None, dtype=scoring_dtype,
        )
        coarse_grids = replace(coarse_grids, translations=trial_grid.translations)
        if builds_coarse_pass1_rotations(state, options, first_iteration, use_local=use_local):
            adaptive_pass1 = coarse_pass1_rotations(
                coarse_grids.rotation_grid, random_perturbation, options,
                perturbation_order=perturbation_order, dtype=scoring_dtype, log=logger,
            )
        # First-iteration CC scores the full translation grid before choosing
        # its single winning pose (ml_optimiser.cpp:9181-9207).
        sampling_plan = plan_expectation_sampling(
            trial_grid, coarse_grids, state, options, optics, current_size=scoring_current_size,
            use_local=use_local, perturbation=random_perturbation, coarse_size_healpix_order=coarse_size_healpix_order,
        )
        direction_prior_healpix_order = _direction_prior_healpix_order_for_scoring(
            state, use_local=use_local, grid_healpix_order=coarse_grids.rotation_grid.healpix_order,
            local_search_order=sampling_plan.local.search.healpix_order if use_local else None,
        )
        coarse_rotation_ids_for_scoring = source.scoring_rotation_ids(trial_grid, use_local=use_local)

        direction_log_priors = relion_direction_log_priors(
            direction_priors, options, use_local=use_local, scoring_healpix_order=direction_prior_healpix_order,
            sealed_sampling_state=source.sealed_sampling_state, dtype=scoring_dtype, log=logger,
        )

        # --- E-step on each half: one pass (adaptive_oversampling=0) or coarse then fine (>=1) ---
        significance = SignificanceStatistics()
        use_adaptive = _should_use_adaptive_search(
            state, options, use_local=use_local, n_rotations=trial_grid.rotations.shape[0],
        )
        # per_half.coarse_ha holds the coarse-grid assignments (trial_grid.rotations indices on every route).
        per_half = PerHalfOutputs()
        # The assignments outlive the iteration: the result, the next accuracy estimate and the final pass read them.
        hard_assignments = per_half.hard_assignments
        class_assignments = per_half.class_assignments
        # Two-pass adaptive oversampling: pass 1 at a reduced size finds the significant orientations, pass 2
        # scores them oversampled at current_size. Off the adaptive route coarse_cs is None (pass 1, where a
        # scorer has one, takes the full window).
        coarse_image_plan = (
            source.adaptive_coarse_size(
                plan_adaptive_image_size(coarse_size_healpix_order, sampling_plan.windows, optics, options, log=logger),
                model_size=sampling_plan.windows.model_size,
            )
            if use_adaptive
            else None
        )
        coarse_cs = adaptive_pass1_size(
            coarse_image_plan, sampling_plan.windows, state, options, box_size=image_geometry.box_size, log=logger,
        )

        # The previous iteration's slabs are released before this iteration's are built.
        projectors = [None, None]
        # Every dense, local and tomo scorer reads this projector: pass 1 scores RELION's exact
        # coarse operands on every route, as RELION builds Projector::data every iteration.
        projectors = build_numbered_projectors(
            halves, reference_model.maps, reconstruction_settings, current_size=sampling_plan.windows.model_window_size,
            n_classes=options.k_class.n_classes, reusable_half1=shared_projector_half1,
            real_references_by_half=initial_real_references_by_half if iteration == 0 else None, iteration=iteration,
            log=logger,
        )

        # The iteration-start curve RELION's scale XA/AA shell gate reads (the scheduling curve changes later).
        scale_correction_data_vs_prior_this_iter = previous_data_vs_prior_for_scheduling

        diagnostic_half_indices = _significance_dump_half_indices(
            numbered_iteration=numbered_relion_iteration, n_classes=options.k_class.n_classes,
            experiment_datasets=experiment_datasets,
        )
        numbered_variant = _numbered_dense_variant(
            first_iteration, options.k_class, use_adaptive=use_adaptive, coarse_cs=coarse_cs,
            fine_window_size=sampling_plan.windows.score_window_size,
        )
        numbered_expectation = prepare_numbered_expectation(
            trial_grid, sampling_plan.windows, local_sampling=sampling_plan.local, variant=numbered_variant,
            use_adaptive=use_adaptive, base_translations=coarse_grids.base_translations,
            current_healpix_order=coarse_grids.rotation_grid.healpix_order,
            oversampling_order=state.adaptive_oversampling, translation_step=state.translation_step,
            random_perturbation=random_perturbation, adaptive_pass1=adaptive_pass1,
            coarse_rotation_ids=coarse_rotation_ids_for_scoring,
            coarse_angular_step_deg=None if coarse_image_plan is None else coarse_image_plan.angular_step_deg,
            options=options, iteration=iteration, numbered_relion_iteration=numbered_relion_iteration,
            collect_local_search_profile=collect_local_search_profile,
            local_profile_history=history.local_profile_history, observer=observer,
        )
        if tomo_halves:
            numbered_tomo_sampling = numbered_iteration_tomo_sampling(
                state, image_geometry, local_sampling=sampling_plan.local,
                grid_healpix_order=coarse_grids.rotation_grid.healpix_order, random_perturbation=random_perturbation,
                coarse_size=sampling_plan.local.coarse_image_window_size if use_local else coarse_cs,
                fine_size=sampling_plan.windows.score_window_size,
            )
        else:
            numbered_tomo_sampling = None

        half_inputs = [
            NumberedHalfInputs(
                data=HalfScoringData(
                    particles=local_search_centre_half(
                        halves[k], (options.replay.init_angle_priors or (None, None))[k], state
                    ),
                    reference=reference_model.maps[k],
                    mean_variance=reference_model.tau2_per_half[k],
                    noise_variance=noise_model.variance_per_half[k],
                    noise_radial=(
                        noise_model.radial_per_half[k] if not tomo_halves and halves[k].dataset.n_units else None
                    ),
                    projector=projectors[k],
                    scale_group_ids=follower_setup.scale_stats_group_ids_per_half[k],
                    scale_group_count=follower_setup.scale_stats_group_count_per_half[k],
                    scale_correction_data_vs_prior=scale_correction_data_vs_prior_this_iter,
                ),
                direction_priors=direction_log_priors[k],
                sigma_offset_angstrom=_sigma_offset_for_half(
                    sigma_offset.shared_angstrom, sigma_offset.per_half_angstrom, k,
                ),
            )
            for k in (0, 1)
        ]
        # Each half is scored, then at once published into per_half and recorded (half 0's local accumulators
        # leave the device before half 1 is scored); overlap_halves may run the two on two threads.
        run_numbered_halves(
            partial(
                score_numbered_half, phase=numbered_expectation, tomo_sampling=numbered_tomo_sampling,
                class_log_priors=class_mixture.log_priors, batch_planner=batch_planner, image_geometry=image_geometry,
                padded_volume_shape=padded_volume_shape, multi_shape_halves=multi_shape_halves, options=options,
                replay_prior_translations=_replay_prior_translations, initial_class_assignments=seeding.seed_classes,
                single_class_iteration=seeding.single_class_iteration, scoring_dtype=scoring_dtype,
                relion_translation_angle_scale=relion_translation_angle_scale, iteration=iteration,
                numbered_relion_iteration=numbered_relion_iteration, observer=observer,
            ),
            partial(
                finish_numbered_half,
                recording=NumberedHalfRecording(
                    per_half, significance, profile_history=history.global_profile_history, iteration=iteration,
                    image_window_size=sampling_plan.windows.image_window_size,
                    healpix_order=coarse_grids.rotation_grid.healpix_order, k_class_enabled=k_class_enabled,
                ),
                use_local=use_local, dtype=scoring_dtype,
            ),
            half_inputs, diagnostic_half_indices, significance, overlap_halves=options.execution.overlap_halves,
            iteration=iteration, observer=observer, log=logger,
        )
        # Drop the inputs' references to this iteration's projectors and maps, which the M-step and the next
        # iteration release (code rule 3).
        half_inputs = None

        Ft_y_0, Ft_y_1 = per_half.Ft_y
        Ft_ctf_0, Ft_ctf_1 = per_half.Ft_ctf

        if options.local_search.stops_after_local_search and use_local:
            elapsed = iteration_clock.seconds
            logger.info(
                "Stopping after local-search diagnostic at iteration %d: profiles=%d score_only=%s wall=%.1fs",
                iteration + 1, len(history.local_profile_history),
                bool(options.local_search.stop_after_local_search_score_only), elapsed,
            )
            profile_stop = ProfileStop(
                score_only=bool(options.local_search.stop_after_local_search_score_only), wall_seconds=elapsed,
                significant_count=significance.recorded,
            )
            break
        if k_class_enabled:
            class_mixture = class_mixture_from_weights(
                _class_weights_from_posterior(
                    per_half.class_posterior, options.k_class.n_classes, class_mixture.weights,
                ),
            )
            history.record_class_weights(
                class_mixture.weights,
                _class_weights_from_posterior(
                    per_half.class_full_posterior, options.k_class.n_classes, class_mixture.weights,
                ),
            )
            logger.info(
                "K-class occupancies: %s",
                ", ".join(f"class {idx + 1}={weight:.4f}" for idx, weight in enumerate(class_mixture.weights)),
            )
        mstep_accumulator_shape = _resolve_mstep_accumulator_shape(
            per_half.mstep_accumulator_shape, padded_volume_shape,
        )
        mstep_full_half_axis = _resolve_mstep_full_half_axis(per_half.mstep_full_half_axis, default_axis=-1)

        # The raw half accumulators, before any join: the observer sees them, then the finite guard checks them
        # (a report names the half that is actually damaged, not the one a join copied it into).
        observer.half_accumulators_ready(
            iteration, numerators=(Ft_y_0, Ft_y_1), denominators=(Ft_ctf_0, Ft_ctf_1),
            settings=reconstruction_settings, current_size=current_size, accumulator_shape=mstep_accumulator_shape,
            k_class_enabled=k_class_enabled, pixel_size_angstrom=source_pixel_size_angstrom,
        )
        check_half_accumulators_before_join(
            (Ft_y_0, Ft_y_1), (Ft_ctf_0, Ft_ctf_1), iteration=iteration,
            init_relion_iteration=options.schedule.init_relion_iteration, log=logger,
        )

        # --- RELION-exact M-step: K=1 on the split-half auto-refine path (compareTwoHalves -> updateSSNRarrays
        # -> reconstruct); Class3D joins the halves per class, carries the previous Iref power spectrum forward
        # as tau2 and solves once per class. mstep is the mode's record (ClassMaximization or K1Maximization). ---
        if k_class_enabled:
            mstep = class_maximization(
                reference_model, (Ft_y_0, Ft_y_1), (Ft_ctf_0, Ft_ctf_1), reconstruction_settings, options,
                halves=halves, iteration=iteration, current_size=current_size,
                image_current_size=sampling_plan.windows.image_current_size,
                mstep_accumulator_shape=mstep_accumulator_shape, mstep_full_half_axis=mstep_full_half_axis,
                projector_power_spectrum=(
                    None
                    if not has_previous_iteration or projectors[0] is None
                    else projectors[0].power_spectrum
                ),
                class_tau2=source.class_tau2(iteration, options.k_class.n_classes), scoring_dtype=scoring_dtype,
                relion_firstiter_cc_this_iter=first_iteration.relion_firstiter_cc,
                source_pixel_size_angstrom=source_pixel_size_angstrom, observer=observer,
            )
            history.data_vs_prior_trajectory.append(mstep.data_vs_prior)
            previous_data_vs_prior_for_scheduling = mstep.data_vs_prior
        else:
            mstep = k1_maximization(
                reference_model, (Ft_y_0, Ft_y_1), (Ft_ctf_0, Ft_ctf_1), reconstruction_settings, parity=options.parity,
                pixel_resolutions=history.pixel_resolutions, current_resolution=state.current_resolution,
                iteration=iteration, current_size=current_size, mstep_accumulator_shape=mstep_accumulator_shape,
                mstep_full_half_axes=per_half.mstep_full_half_axis, scoring_dtype=scoring_dtype,
                relion_firstiter_cc_this_iter=first_iteration.relion_firstiter_cc,
                source_pixel_size_angstrom=source_pixel_size_angstrom, observer=observer,
            )
            # The accumulators the solve used (joined at low resolution when that is on) replace the scored ones.
            Ft_y_0, Ft_y_1 = mstep.Ft_y_per_half
            Ft_ctf_0, Ft_ctf_1 = mstep.Ft_ctf_per_half
        observer.stage_finished(iteration, "recon")

        history.significant_counts.append(significance.recorded)

        history.record_rotation_posterior(per_half.rotation_posterior)
        if all(rot_sum is not None for rot_sum in per_half.rotation_posterior):
            if not k_class_enabled:
                learned_priors = learn_k1_direction_priors(
                    per_half.rotation_posterior, direction_prior_order=direction_prior_healpix_order,
                    expected_rotation_count=rotation_grid_size(
                        direction_prior_healpix_order, symmetry=options.symmetry.point_group,
                    ),
                    dtype=scoring_dtype, log=logger, symmetry=options.symmetry.point_group,
                )
                for half_index, learned in enumerate(learned_priors):
                    if learned is not None:
                        direction_priors[half_index] = learned
            else:
                exhaustive_grid_size = rotation_grid_size(
                    direction_prior_healpix_order,  # a local search's posterior grid (pdf_direction still accumulates)
                    symmetry=options.symmetry.point_group,
                )
                if (
                    (use_local or trial_grid.rotations.shape[0] == exhaustive_grid_size)
                    and all(
                        rot_sum is not None
                        for rot_sum in per_half.class_rotation_posterior
                    )
                ):
                    learned_priors = learn_class_direction_priors(
                        per_half.class_rotation_posterior, n_classes=options.k_class.n_classes,
                        healpix_order=direction_prior_healpix_order, dtype=scoring_dtype,
                        symmetry=options.symmetry.point_group,
                    )
                    for half_index, learned in enumerate(learned_priors):
                        direction_priors[half_index] = learned
        if seeding.single_class_iteration:
            mstep, class_mixture = copy_first_class_to_every_class(
                reference_model, direction_priors, mstep, class_mixture, n_classes=options.k_class.n_classes,
            )
            history.data_vs_prior_trajectory[-1] = mstep.data_vs_prior
            previous_data_vs_prior_for_scheduling = mstep.data_vs_prior
        history.record_direction_prior(direction_priors, k_class_enabled=k_class_enabled)

        # --- Unregularized half-maps, reconstructed only for the run files and the observer ---
        need_unreg_means = (
            observer.wants_unfiltered_maps(numbered_relion_iteration)
            or (
                options.checkpoint.writer is not None
                and options.checkpoint.writer.wants_unfiltered_maps(
                    numbered_relion_iteration, n_classes=options.k_class.n_classes
                )
            )
        )
        unreg_clock = Stopwatch()
        if k_class_enabled:
            unreg_means = (
                reconstruct_unregularized_class_means(
                    mstep.Ft_y_combined,
                    mstep.Ft_ctf_combined,
                    reconstruction_settings,
                    options.k_class.n_classes,
                    accumulator_volume_shape=mstep_accumulator_shape,
                )
                if need_unreg_means
                else [None, None]
            )
        else:
            unreg_means = (
                reconstruct_unregularized_k1_halfmaps(
                    (Ft_y_0, Ft_y_1),
                    (Ft_ctf_0, Ft_ctf_1),
                    reconstruction_settings,
                    accumulator_volume_shape=mstep_accumulator_shape,
                )
                if need_unreg_means
                else [None, None]
            )
            align_k1_volume_signs(reference_model.maps, mstep.previous_means, unreg_means, volume_shape)
        logger.info(
            "Unregularized reconstruction (2 halves): %.1fs%s", unreg_clock.seconds,
            "" if need_unreg_means else " (skipped; diagnostics disabled)",
        )

        # K=1's FSC (the M-step's, which set tau2 before the Wiener solve) also drives size growth; Class3D has
        # none (its shared per-class curve drives growth).
        fsc = None if k_class_enabled else mstep.fsc
        history.record_fsc(fsc, fsc)
        observer.stage_finished(iteration, "fsc")

        observer.maps_reconstructed(ReconstructedIteration(
            iteration, numerators=(Ft_y_0, Ft_y_1), denominators=(Ft_ctf_0, Ft_ctf_1), reference_model=reference_model,
            noise_model=noise_model, per_half=per_half, trial_grid=trial_grid, sampling_plan=sampling_plan,
            options=options, unfiltered_maps=unreg_means, fsc=fsc, current_size=current_size, state=state,
            volume_shape=volume_shape, voxel_size=source_pixel_size_angstrom,
        ))

        # --- This expectation's particle statistics: joined assignments, posterior maxima, optimizer Pmax ---
        statistics = expectation_statistics(
            per_half, previous_assignments,
            _relion_pmax_normalization_mass_per_half(per_half, k_class_enabled=k_class_enabled),
        )
        if k_class_enabled:
            logger.info(
                "Class3D optimizer Pmax: value=%.9f numerator=%.9f "
                "half1_mstep_posterior_mass=%.9f half1_particle_count=%d",
                statistics.ave_pmax, float(np.sum(np.asarray(per_half.max_posterior[0]), dtype=np.float64)),
                statistics.ave_pmax_mass, int(np.asarray(per_half.max_posterior[0]).size),
            )
        history.record_pmax(statistics.ave_pmax, statistics.ave_pmax_mass, statistics.max_posterior.copy())
        history.record_pass2_engines(take_pass_engines())
        history.record_coarse_engines(take_coarse_engine_calls())

        # --- Resolution from updated FSC-derived SSNR (RELION auto-refine) ---
        if k_class_enabled:
            current_combined_classes = concatenate_assignments(class_assignments)
            history.class_assignment_history.append(current_combined_classes.copy())
            previous_combined_classes = concatenate_assignments_or_none(previous_class_assignments)
            # K>1: from the shared per-class prior and the combined class accumulators.
            resolution_estimate = estimate_class_iteration_resolution(
                history.data_vs_prior_trajectory[-1], current_size=current_size, box_size=image_geometry.box_size,
                voxel_size=source_pixel_size_angstrom,
                emulate_relion_firstiter_cc=options.parity.emulate_relion_firstiter_cc,
                ini_high_angstrom=options.parity.relion_firstiter_ini_high_angstrom,
                relion_iteration=int(options.schedule.init_relion_iteration) + int(iteration) + 1, dtype=scoring_dtype,
            )
        else:
            current_combined_classes = None
            previous_combined_classes = None
            # K=1: data_vs_prior comes from the half-map FSC.
            resolution_estimate = estimate_k1_iteration_resolution(
                mstep.tau2_update_details["ssnr_shells"], current_size=current_size, box_size=image_geometry.box_size,
                voxel_size=source_pixel_size_angstrom,
                emulate_relion_firstiter_cc=options.parity.emulate_relion_firstiter_cc,
                ini_high_angstrom=options.parity.relion_firstiter_ini_high_angstrom,
                relion_iteration=int(options.schedule.init_relion_iteration) + int(iteration) + 1, dtype=scoring_dtype,
            )
        if int(resolution_estimate.scheduling_shell) != int(resolution_estimate.observed_shell):
            logger.info(
                "RELION firstiter_cc resolution state: using ini_high=%.2f A shell %d "
                "instead of live data-vs-prior shell %d",
                float(options.parity.relion_firstiter_ini_high_angstrom), int(resolution_estimate.scheduling_shell),
                int(resolution_estimate.observed_shell),
            )
        history.pixel_resolutions.append(resolution_estimate.scheduling_shell)

        # --- Update poses and noise (the previous best poses stay for RELION's change metrics, B3) ---
        pose_update = prepare_particle_pose_update(
            per_half, halves, coarse_grids.translations, previous_rotations=previous_best_rotations,
            local_sampling=sampling_plan.local if use_local else None, dtype=scoring_dtype,
        )
        previous_best_rotations = [poses.rotations for poses in pose_update.current]
        for half, poses in zip(halves, pose_update.current, strict=True):
            half.rotation_eulers = poses.eulers_deg
            half.translations = poses.translations_pixels
        history.record_pose_history(
            [np.asarray(poses.eulers_deg).copy() for poses in pose_update.current],
            [np.asarray(poses.translations_pixels).copy() for poses in pose_update.current],
        )
        observer.poses_updated(
            iteration, poses=pose_update.current, per_half=per_half, significance=significance,
            datasets=experiment_datasets,
        )

        pose_comparison = prepare_pose_comparison(
            pose_update, translation_dimension=3 if tomo_halves else 2, dtype=scoring_dtype, log=logger,
        )

        if not k_class_enabled:
            previous_data_vs_prior_for_scheduling = np.asarray(resolution_estimate.data_vs_prior, dtype=scoring_dtype)
            history.data_vs_prior_trajectory.append(previous_data_vs_prior_for_scheduling)

        # RELION's posterior-weighted noise update: the radial sigma2_noise and the engine's pixel rows.
        noise_update = update_posterior_noise_variance(
            per_half.noise_stats,
            noise_model,
            image_geometry.image_shape,
            k_class_enabled=k_class_enabled,
            firstiter_cc=first_iteration.relion_firstiter_cc,
            ctf_premultiplied=datasets_store_premultiplied_ctf(experiment_datasets),
            summed_current_size=sampling_plan.windows.image_window_size if consistency.noise_shell_count == "summed" else None,
            nyquist_column_counting=consistency.nyquist_column_counting,
        )
        if not first_iteration.relion_firstiter_cc:
            observer.noise_updated(
                iteration, current_size=current_size, image_shape=image_geometry.image_shape,
                noise_stats_per_half=per_half.noise_stats, previous_noise_radial_per_half=noise_model.radial_per_half,
                noise_from_res_per_half=noise_update.noise_from_res_per_half, noise_from_res=noise_update.noise_from_res,
            )
            observer.stage_finished(iteration, "noise_update")
        noise_from_res = noise_update.noise_from_res
        noise_from_res_per_half = noise_update.noise_from_res_per_half
        noise_model = noise_update.model

        correction_report = NormScaleCorrectionReport()
        norm_scale_update = numbered_norm_scale_update(
            per_half, halves, firstiter_cc=first_iteration.relion_firstiter_cc, do_norm_correction=not tomo_halves,
            do_scale_correction=follower_setup.follower_scale_state is None, dtype=scoring_dtype,
            iteration=iteration, current_size=current_size,
        )
        if follower_setup.follower_scale_state is not None and norm_scale_update is None:
            raise RuntimeError(
                "Strict RELION follower-scale topology requires per-half norm/scale "
                "statistics at every numbered M-step"
            )
        if norm_scale_update is not None:
            if follower_setup.follower_scale_state is None:
                for half, images, scales in zip(
                    halves,
                    norm_scale_update.image_corrections_per_half,
                    norm_scale_update.scale_corrections_per_half,
                    strict=True,
                ):
                    half.image_corrections = images
                    half.scale_corrections = scales
                group_scale_corrections = norm_scale_update.group_scale_corrections_per_half
            else:
                # Class3D follower-scale emulation: the follower state owns the scales and installs them.
                group_scale_corrections = _update_relion_follower_corrections(
                    follower_setup, noise_stats_per_half=per_half.noise_stats, norm_scale_update=norm_scale_update,
                    relion_half_inputs=halves, relion_firstiter_cc_this_iter=first_iteration.relion_firstiter_cc,
                    dtype=scoring_dtype, logger=logger,
                )
            correction_report = norm_scale_report(norm_scale_update, group_scale_corrections)
            log_norm_scale_update(norm_scale_update, log=logger)
        if follower_setup.follower_scale_state is not None:
            history.relion_scale_follower_scales_numbered_post_mstep_trajectory.append(
                np.asarray(follower_setup.follower_scale_state.scales, dtype=np.float64).copy()
            )

        # The iteration's sigma2 shells (after the noise update) and the tau2 ingredients of its Wiener update.
        history.record_noise_and_tau2(noise_from_res, noise_from_res_per_half, mstep.tau2_update_details)

        # --- Convergence state: assignment changes, resolution stalls, angular-step refinement ---
        state, accuracy_replay = update_iteration_convergence(
            state, pose_comparison, options, image_geometry=image_geometry, iteration=iteration,
            sampling_decision_now=not k_class_enabled and not native_sampling_boundary,
            class_change_fraction=(
                hard_class_change_fraction(current_combined_classes, previous_combined_classes)
                if k_class_enabled
                else 0.0
            ),
            scheduling_resolution_shell=resolution_estimate.scheduling_shell, source=source,
            translations=coarse_grids.translations, statistics=statistics, significant_counts=significance.convergence,
            exact_acc_rot=iteration_accuracy.acc_rot, exact_acc_trans=iteration_accuracy.acc_trans_angstrom, log=logger,
        )
        if not k_class_enabled:
            state = reset_follower_counter_once(state, options, iteration=iteration)

        # Reuse the assignment statistic computed by update_refinement_state.
        # Sampling transitions and optimiser replay preserve this field.
        history.frac_changed_trajectory.append(float(state.fraction_changed))

        # --- sigma2_offset from the posterior-weighted offsets (RELION parity, C1) ---
        sigma_offset_result = update_c1_sigma_offset_from_posterior(
            per_half, sigma_offset, n_classes=options.k_class.n_classes,
            state_fallback_offsets_angstrom=state.current_changes_optimal_offsets_angstrom,
            offset_dims=3 if tomo_halves else 2,
        )
        sigma_offset = SigmaOffset(
            sigma_offset_result.current_sigma_offset_angstrom,
            normalize_sigma_offset_per_half(sigma_offset_result.current_sigma_offset_angstrom_per_half),
        )
        per_class_sigma_offset = sigma_offset_result.per_class_sigma_offset_angstrom
        history.record_sigma_offset_update(
            float(sigma_offset.shared_angstrom), copy_optional_float_pair(sigma_offset.per_half_angstrom),
            None if per_class_sigma_offset is None else per_class_sigma_offset.tolist(),
        )
        history.record_pose_accuracy_diagnostics(
            accuracy_replay, iteration_accuracy, state, n_classes=options.k_class.n_classes
        )

        # The next iteration's change tracking reads the coarse-grid assignments (trial-grid indices on every
        # route, adaptive or not).
        previous_assignments = [ha.copy() if ha is not None else None for ha in per_half.coarse_ha]
        previous_class_assignments = [cls.copy() if cls is not None else None for cls in class_assignments]
        observer.stage_finished(iteration, "convergence")

        # --- RELION's run_itNNN files (ml_optimiser.cpp:3489) ---
        checkpoint_writer = options.checkpoint.writer
        if checkpoint_writer is not None and checkpoint_writer.due(numbered_relion_iteration):
            # The files hold incr_size/has_high_fsc_at_limit after this iteration's FSC
            # update, which the loop applies (idempotently) at the top of the next one.
            incr_size_after, high_fsc_after = relion_incr_size, relion_has_high_fsc_at_limit
            if not k_class_enabled:
                incr_size_after, high_fsc_after = update_relion_growth_state_from_fsc(
                    _zero_shells_past_current_size(
                        fsc, current_size=current_size, box_size=image_geometry.box_size, dtype=scoring_dtype,
                    ),
                    current_size, incr_size=relion_incr_size, has_high_fsc_at_limit=relion_has_high_fsc_at_limit,
                )
            snapshot = snapshot_capture.begin(
                numbered_relion_iteration, state, sigma_offset_angstrom_per_half=sigma_offset.per_half_angstrom,
                current_size=current_size, incr_size=incr_size_after, has_high_fsc_at_limit=high_fsc_after,
                random_perturbation=random_perturbation, acc_rot_per_class=published_accuracy.acc_rot_per_class,
                acc_trans_per_class_angstrom=published_accuracy.acc_trans_per_class_angstrom,
            )
            snapshot = snapshot_capture.finish(
                snapshot,
                reference_model.maps,
                unreg_means,
                (
                    mstep.tau2_shells
                    if k_class_enabled
                    else [details["prior_shells"] for details in mstep.tau2_update_details_per_half]
                ),
                previous_data_vs_prior_for_scheduling,
                noise_model.radial_per_half,
                fsc=fsc,
                fsc_for_growth=None if k_class_enabled else fsc,
                class_weights=class_mixture.weights if k_class_enabled else None,
                direction_priors=direction_priors,
                half_inputs=halves,
                class_assignments=class_assignments if k_class_enabled else None,
                max_posterior=per_half.max_posterior,
                significant_counts=significance.per_half,
                avg_norm_correction=correction_report.avg_norm_correction_per_half,
            )
            checkpoint_writer(snapshot)

        observer.iteration_finished(FinishedIteration(
            iteration, init_relion_iteration=options.schedule.init_relion_iteration, state=state,
            current_size=current_size, sigma_offset_angstrom=sigma_offset.shared_angstrom,
            random_perturbation=random_perturbation, settings=reconstruction_settings,
            pixel_size_angstrom=source_pixel_size_angstrom, ave_pmax=statistics.ave_pmax, fsc=fsc,
            noise_variance=noise_model.average_variance, means=reference_model.maps, unfiltered_means=unreg_means,
            poses=pose_update.current, half_inputs=halves, corrections=correction_report,
            scale_correction_data_vs_prior=scale_correction_data_vs_prior_this_iter,
        ))

        elapsed = iteration_clock.seconds
        history.wall_times.append(elapsed)

        res_angstrom = shell_index_to_resolution_angstrom(
            resolution_estimate.scheduling_shell, image_geometry.image_shape[0], source_pixel_size_angstrom,
        )
        logger.info(
            "RELION Iteration %d: current_size=%d, pixel_res=%.1f, "
            "res=%.2f A, ave_Pmax=%.4f, healpix_order=%d, "
            "converged=%s, time=%.1fs",
            iteration + 1, current_size, resolution_estimate.scheduling_shell, res_angstrom, statistics.ave_pmax,
            state.healpix_order, state.has_converged, elapsed,
        )

        # End-of-iteration memory boundary. The next iteration pads each half-map to the projection
        # grid; keeping the previous accumulators (also in the pass outputs), unregularized maps or the
        # run-files snapshot live can make high-resolution runs OOM before the batch-size estimator acts
        # (41 GB of host carried into the final pass at box 800).
        jax.block_until_ready(reference_model.maps)
        Ft_y_0 = Ft_y_1 = None
        Ft_ctf_0 = Ft_ctf_1 = None
        unreg_means = mstep = per_half = snapshot = None
        # Pass containers must not retain the previous grids while the next projector is built.
        numbered_expectation = numbered_tomo_sampling = numbered_variant = None
        if options.execution.clear_jax_caches_between_iterations:
            jax.clear_caches()

        if state.has_converged and not options.schedule.force_max_iter_after_convergence:
            logger.info(
                "Convergence reached at iteration %d. Final resolution: %.2f A (pixel_res=%.1f)", iteration + 1,
                res_angstrom, resolution_estimate.scheduling_shell,
            )
            break
        if state.has_converged:
            logger.info(
                "Convergence reached at iteration %d, continuing because force_max_iter_after_convergence=True",
                iteration + 1,
            )

        iteration += 1

    # Numbered contribution/native dump identities must never leak into the
    # return path or RELION's unnumbered final all-data pass.
    bpref_diagnostics.clear_bpref_contribution_dump_context()

    # RELION enters the final pass only when checkConvergence() ran at the top of a permitted iteration; a run
    # made convergence-ready by its last numbered iteration ends at the cap (``iter <= nr_iter``). Set-up and
    # numbered-iteration facts; the final pass leaves them as they are.
    numbered = NumberedMetadata(
        hard_assignments=hard_assignments, frozen_initial_scoring_state_sha256=frozen_initial_scoring_state_sha256,
        expected_accuracy_trial_local_indices=published_accuracy.trial_local_indices,
        expected_accuracy_trial_particle_ids=published_accuracy.trial_particle_ids,
        setup_phase_seconds=setup_phase_seconds,
    )
    if profile_stop is not None:
        # Local search is K=1 (Class3D was rejected above), so there are no class products.
        return RefinementResult(
            maps=finalization.numbered_k1_maps(reference_model.maps),
            replay=_follower_replay_telemetry(follower_scale_replay, history), follower_scale=None,
            convergence_state=state, numbered=numbered, history=history, profile_stop=profile_stop,
        )
    if not finalization.final_pass_due(state, options, iteration=iteration, k_class_enabled=k_class_enabled):
        return RefinementResult(
            maps=(
                finalization.numbered_class_maps(reference_model.maps, class_mixture.weights, class_assignments)
                if k_class_enabled
                else finalization.numbered_k1_maps(reference_model.maps)
            ),
            replay=_follower_replay_telemetry(follower_scale_replay, history),
            follower_scale=follower_setup.result_outputs(history), convergence_state=state, numbered=numbered,
            history=history,
        )
    # --- RELION's final iteration (do_join_random_halves + do_use_all_data, ml_optimiser.cpp:10157-10160 and
    # 5707-5708): one more E+M at full Nyquist, each half against its own map, the weighted sums joined. ---
    final_join_means = [reference_model.maps[0], reference_model.maps[1]]
    if not k_class_enabled and options.final_pass.merged_reference:
        final_merged_reference = merged_half_map(reference_model.maps)
        final_join_means = [final_merged_reference, final_merged_reference]
        logger.info(
            "Diagnostic %s=1: final all-data K=1 E-step uses merged reference for both halves",
            FINAL_ALL_DATA_USE_MERGED_REFERENCE_ENV,
        )
    final_join_means, sigma_offset, noise_model = source.final_state(
        FinalState(final_join_means, sigma_offset, noise_model),
        means=reference_model.maps, numbered_iteration_count=len(history.current_sizes), halves=halves,
        direction_priors=direction_priors, healpix_order=state.healpix_order, image_geometry=image_geometry,
    )
    if follower_setup.follower_scale_state is not None:
        _dispatch_relion_follower_scale_for_final_all_data(
            follower_setup, init_relion_iteration=options.schedule.init_relion_iteration,
            numbered_iteration_count=len(history.current_sizes), relion_half_inputs=halves, dtype=scoring_dtype,
            logger=logger,
        )
    final_use_local = not k_class_enabled and state.do_local_search
    if final_use_local:
        for half in halves:
            half.require_local_search_poses()
    # Each half is still scored with its own numbered-iteration noise (one MPI follower per half); K=1 joins
    # the half sums after this E-step.
    final_result = finalization.run_final_all_data(
        halves, reference_model=reference_model, noise_model=noise_model, direction_priors=direction_priors,
        state=state, history=history, options=options, optics=optics, reconstruction_settings=reconstruction_settings,
        batch_planner=batch_planner, follower_setup=follower_setup, expected_accuracy_inputs=expected_accuracy_inputs,
        final_join_means=final_join_means, final_use_local=final_use_local, tomo_halves=tomo_halves,
        native_sampling_boundary=native_sampling_boundary, rotation_grid=coarse_grids.rotation_grid,
        random_perturbation=random_perturbation, perturb_rng=perturb_rng, source=source, sigma_offset=sigma_offset,
        class_mixture=class_mixture, class_assignments=class_assignments,
        previous_data_vs_prior_for_scheduling=previous_data_vs_prior_for_scheduling, iteration=iteration,
        collect_local_search_profile=collect_local_search_profile,
        relion_translation_angle_scale=relion_translation_angle_scale, observer=observer,
    )
    return replace(final_result, numbered=numbered, replay=_follower_replay_telemetry(follower_scale_replay, history))

### THIS FILE SHOULD BE MUCH SHORTER - REMOVE UN-NECESSARY IF STATEMENTS/RELION THINGS THAT DONT MATTER/ WE DONT USE
## MOVE THINGS AWAY TO DIFFERNET FILE E.G. I/O PERHAPS. IT SHOULD BE EASY TO UDNERSTAND WHERE THE MAIN ENGINE OF ITER IS GOING, WHAT ARE THE MAIN STEPS (E-M ACCUMULATION) POSTPROCESSING, NOISE UPDATING, ETC
