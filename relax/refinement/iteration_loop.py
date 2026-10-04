"""Orchestrate dense single-volume and K-class EM refinement.

``refine_single_volume`` validates options and manages refinement state and dispatch.
``half_scoring`` owns the per-half dense/local engine calls; ``scoring_policy``
owns their shared execution defaults and diagnostic selectors. Local chunks
are implemented in ``local_search_iteration``; state-swap diagnostics belong
to ``diagnostics.state_swap_runtime``. Pure trial-grid construction belongs to
``sampling``.
See ``docs/math/relion_refinement_algorithm.md`` for the algorithm map.
"""


import logging
import os
import time
from functools import partial, wraps

import jax
import jax.numpy as jnp
import numpy as np
from recovar import utils
from recovar.data_io import cryoem_dataset

from relax import sampling
from relax.dense import scoring_policy
from relax.dense.score_outputs import (
    PerHalfOutputs,
    _combine_optional_half_accumulators,
    _maybe_host_offload_half0_local_accumulators,
    _resolve_mstep_accumulator_shape,
    _resolve_mstep_full_half_axis,
)
from relax.dense.scoring_policy import (
    _dense_global_scoring_dtype,
)
from relax.diagnostics import bpref_diagnostics
from relax.diagnostics import parity_dump as _parity_dump
from relax.diagnostics import reconstruction as reconstruction_diagnostics
from relax.diagnostics import relion_replay as replay_policy
from relax.diagnostics.frozen_boundary import (
    _assert_frozen_scoring_state_unchanged,
    _frozen_scoring_state_arrays,
)
from relax.diagnostics.iteration import (
    _maybe_dump_noise_update_debug,
    _save_iteration_intermediates,
    _save_iteration_particle_states,
    _significance_dump_half_indices,
    _source_image_indices,
    dump_numbered_iteration,
)
from relax.diagnostics.relion_replay import (
    _apply_replay_correction_overrides,
    _has_numbered_replay_iteration_overrides,
    _maybe_debug_replay_relion_references,
    _sealed_sampling_rotation_ids,
    _validate_bpref_particle_order_scope,
    apply_iter_replay_overrides,
)
from relax.diagnostics.state_swap_runtime import (
    _apply_state_swap_probe,
    _copy_half_pair,
    _copy_optional_float_pair,
    _snapshot_state_swap_inputs,
)
from relax.helpers.convergence import (
    _direction_prior_healpix_order_for_scoring,
    _exhaustive_grid_order_for_state,
    _relion_optimizer_average_pmax,
    _relion_pmax_normalization_mass_per_half,
    check_convergence,
    concatenate_assignments,
    concatenate_assignments_or_none,
)
from relax.helpers.env_flags import parse_env_true_flag
from relax.helpers.expected_accuracy import (
    RELION_DEFAULT_SIGMA2_FUDGE,
    Half1AccuracyInputs,
    _expected_accuracy_class_ids,
    prepare_relion_half1_trial_order,
)
from relax.helpers.fourier_window import quantize_current_size
from relax.helpers.iteration_history import RefinementHistory
from relax.helpers.orientation_priors import (
    DirectionPrior,
    infer_direction_prior_healpix_order,
    initial_direction_priors_from_snapshot,
    learn_class_direction_priors,
    learn_k1_direction_priors,
    normalize_class_direction_prior,
    normalize_class_direction_prior_per_half,
    normalize_direction_prior_per_half,
    relion_direction_log_priors_for_half,
    relion_local_search_sigmas,
    remap_half_direction_prior_to_healpix_order,
)
from relax.helpers.resolution import (
    ImageGeometry,
    _firstiter_cc_ini_high_tapered,
    _zero_shells_past_current_size,
    estimate_class_iteration_resolution,
    estimate_k1_iteration_resolution,
    relion_expectation_coarse_size_order,
    shell_index_to_resolution_angstrom,
)
from relax.reconstruction.regularization_relion import (
    RELION_MINRES_MAP,
    update_relion_growth_state_from_fsc,
)
from relax.refinement import finalization
from relax.refinement.convergence import (
    advance_expectation_sampling,
    update_iteration_convergence,
    uses_native_auto_refine,
)
from relax.refinement.expectation import (
    SignificanceStatistics,
    _half_overlap_active,
    _run_halves_overlapped,
    prepare_numbered_expectation,
    record_numbered_half,
    score_numbered_half,
)
from relax.refinement.expectation_batches import BatchPlanner
from relax.refinement.half_inputs import (
    _as_sigma_offset_half_pair,
    _mean_sigma_offset_per_half,
    _normalize_sigma_offset_per_half,
    _sigma_offset_for_half,
    initialize_halfsets,
    prepare_particle_pose_update,
    prepare_pose_comparison,
)
from relax.refinement.half_scoring import (
    DenseVariantPolicy,
    HalfScoringData,
)
from relax.refinement.iteration_planning import (
    build_initial_coarse_grids,
    build_sealed_initial_coarse_grids,
    initialize_refinement_state,
    plan_adaptive_image_size,
    plan_class_image_size,
    plan_expectation_windows,
    plan_halfmap_image_size,
    plan_initial_image_size,
    resolve_numbered_perturbation,
)
from relax.refinement.iteration_snapshot import (
    SnapshotCapture,
)
from relax.refinement.local_sampling import (
    LocalSearchSettings,
    prepare_numbered_local_sampling,
)
from relax.refinement.mean_helpers import (
    ReconstructionSettings,
    _class_weights_from_posterior,
    _host_tau2_volumes,
    _initialize_class_log_priors,
    _merged_mean_from_halves,
    _normalize_initial_means,
    _snapshot_and_release_previous_k1_means,
    _stack_class_tau2_update_details,
    _updated_mean_variance_per_half,
    align_k1_volume_signs,
    estimate_class_priors,
    estimate_split_half_prior,
    initialize_reference_model,
    join_half_accumulators_at_low_resolution,
    reconstruct_numbered_class_maps,
    reconstruct_numbered_k1_halfmaps,
    reconstruct_unregularized_class_means,
    reconstruct_unregularized_k1_halfmaps,
    reference_model_from_snapshot,
    share_kclass_volume_signs,
    shared_tau2_per_half,
    taper_first_cc_class_prior,
    taper_first_cc_k1_prior,
)
from relax.refinement.noise_updates import (
    _mean_noise_variance,
    _normalize_noise_variance_per_half,
    initialize_noise_model,
    noise_model_from_pixels,
    noise_model_from_shells,
    update_c1_sigma_offset_from_posterior,
    update_posterior_noise_variance,
)
from relax.refinement.optics_shapes import MultiShapeHalf
from relax.refinement.particle_loading import configure_half_image_preprocessing
from relax.refinement.projector_preparation import (
    ProjectorReuse,
    _validate_captured_relion_projector_for_iteration,
    prepare_initial_real_references,
    prepare_scoring_projector,
)
from relax.refinement.refinement_options import RefinementOptions, with_validated_sampling_schedule
from relax.refinement.result_files import (
    _model_result_fields,
    _numbered_result_metadata,
)
from relax.refinement.tomo_half import TomoHalf, TomoSampling
from relax.relion.geometry import (
    IMAGE_MASK_EDGE_PIXELS,
    PROJECTION_PADDING_FACTOR,
    RECONSTRUCTION_PADDING_FACTOR,
    REFERENCE_FILTER_EDGE_SHELLS,
)
from relax.relion.relion_normalization import (
    NormScaleCorrectionReport,
    log_norm_scale_update,
    prepare_norm_scale_update,
)
from relax.relion.relion_worker_scale import (
    _dispatch_relion_follower_scale_for_final_all_data,
    _dispatch_relion_follower_scale_for_numbered_iteration,
    _finalize_relion_follower_scale_replay_telemetry,
    _update_relion_follower_corrections,
    setup_relion_follower_scale_state,
)
from relax.sampling import (
    _relion_adaptive_pass1_rotations,
    relion_angular_sampling_deg,
    relion_sampling_perturbation_for_iteration,
    rotation_grid_size,
)
from relax.sparse_pass2.engine_record import take_coarse_engine_calls, take_pass_engines
from relax.sparse_pass2.resident_pass2 import stable_window_class_history

logger = logging.getLogger(__name__)


_FINAL_ALL_DATA_USE_MERGED_REFERENCE_ENV = "RELAX_FINAL_ALL_DATA_USE_MERGED_REFERENCE"
_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE_ENV = "RELAX_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE"
_FINAL_ALL_DATA_DISABLE_REPLAY_LAST_NUMBERED_STATE_ENV = "RELAX_FINAL_ALL_DATA_DISABLE_REPLAY_LAST_NUMBERED_STATE"


def _fresh_k1_spectrum_norm_default(
    *,
    preserve_bpref_particle_order: bool,
) -> bool:
    """Enable the production K=1 arithmetic wherever RELION's order is preserved.

    This selects source-faithful powerClass normalization and, through it, the
    exact BPref operands and the atomic Wavg triplet. Imported-boundary
    replays preserve RELION's native order too, so they run the same
    arithmetic as a fresh run rather than a second, replay-only variant.
    """

    return bool(preserve_bpref_particle_order)


def _relion_k1_translation_angle_scale(
    *,
    n_classes: int,
    model_pixel_size: float,
    optics_pixel_sizes,
) -> float:
    """Convert K=1 model-pixel translations to the shared optics pixel size.

    RELION keeps sampling translations in Angstrom and converts them with the
    particle's optics pixel size (HealpixSampling::getTranslationsInPixel),
    while RECOVAR's translation grid is in model pixels. The scale
    model_pixel_size / optics_pixel_size is exactly 1.0 when the serialized
    sizes agree; it multiplies only the RELION translation-phase operand.
    """

    if int(n_classes) != 1 or optics_pixel_sizes is None:
        return 1.0
    model_pixel_size = float(model_pixel_size)
    optics = np.asarray(optics_pixel_sizes, dtype=np.float64).reshape(-1)
    if optics.size == 0 or not np.all(np.isfinite(optics)) or np.any(optics <= 0.0):
        raise ValueError("RELION optics pixel sizes must be non-empty, positive, and finite")
    unique_optics = np.unique(optics)
    if unique_optics.size != 1:
        raise NotImplementedError(
            "K=1 exact RELION translation phases currently require one shared optics pixel size; "
            "per-particle optics scaling is not yet implemented"
        )
    return model_pixel_size / float(unique_optics[0])




def _should_use_adaptive_search(
    *,
    adaptive_oversampling: int,
    use_local: bool,
    n_rotations: int,
    symmetry: str,
) -> bool:
    """Keep non-C1 refinement on its supported sparse/x-half route.

    Small C1 grids may use the direct dense path. Point-group symmetry cannot:
    symmetry reduction itself can make a valid grid smaller than that cutoff
    (O has 12 coarse rotations at HEALPix order 1, I1 fewer), while its
    scoring and reconstruction still require the adaptive RELION x-half path.
    Ported from final Q 22efd8065.
    """

    if int(adaptive_oversampling) <= 0 or bool(use_local):
        return False
    return int(n_rotations) > 16 or str(symmetry).upper() != "C1"


def _optics_group_ids_per_half(optics_group_ids_per_half, noise_variance_per_half, experiment_datasets):
    """Each half's per-image optics-group rows, or ``[None, None]`` for one group.

    A half's noise is a flat vector (one optics group) or ``[G, P]`` rows
    (:mod:`relax.helpers.optics_noise`); with rows every image needs its group.
    """

    n_groups = 1 if noise_variance_per_half[0].ndim == 1 else int(noise_variance_per_half[0].shape[0])
    if n_groups == 1:
        return [None, None]
    if optics_group_ids_per_half is None or len(optics_group_ids_per_half) != 2:
        raise ValueError(
            f"a {n_groups}-optics-group noise table needs parity.optics_group_ids_per_half for both halves"
        )
    ids = []
    for half, (values, dataset) in enumerate(zip(optics_group_ids_per_half, experiment_datasets)):
        values = np.asarray(values, dtype=np.int32).reshape(-1)
        if values.shape != (int(dataset.n_units),) or np.any(values < 0) or np.any(values >= n_groups):
            raise ValueError(
                f"half {half + 1} optics-group ids must give each of {int(dataset.n_units)} images "
                f"a row 0..{n_groups - 1}"
            )
        ids.append(values)
    return ids


def _copy_first_class(stacked):
    """``stacked`` (a leading class axis) with every class set to class 0, as the input's array type."""

    if isinstance(stacked, np.ndarray):
        return np.broadcast_to(stacked[:1], stacked.shape).copy()
    return jnp.broadcast_to(stacked[:1], stacked.shape)


def _with_stable_window_class_history(refine):
    """Run a refinement inside one stable-window class history (resident_pass2)."""

    @wraps(refine)
    def run(*args, **kwargs):
        with stable_window_class_history():
            return refine(*args, **kwargs)

    return run


@_with_stable_window_class_history
def refine_single_volume(
    experiment_datasets: list[cryoem_dataset.CryoEMDataset],
    init_volume: list[jnp.ndarray] | jnp.ndarray,
    init_noise_variance: jnp.ndarray,
    init_mean_variance: jnp.ndarray,
    translations: jnp.ndarray | None,
    options: RefinementOptions | None = None,
) -> dict:
    """Multi-iteration RELION-parity EM refinement.

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
        / local-search / K-class / replay / debug / batching kwarg groups.
        Defaults to ``RefinementOptions()`` when omitted.

    Returns
    -------
    dict with keys:
        mean : jnp.ndarray -- final merged mean volume
        means : list of 2 jnp.ndarray -- per-half-set means
        fsc : jnp.ndarray -- final FSC curve
        hard_assignments : list of 2 np.ndarray -- per-half-set assignments
        current_sizes : list of int -- current_size at each iteration
        fsc_history : list of jnp.ndarray -- FSC curve at each iteration
        pixel_resolutions : list of float -- pixel resolution at each iter
        wall_times : list of float -- wall time per iteration
        significant_counts : list of (jnp.ndarray or None) -- per-image
            significant sample counts at each iteration (None when
            adaptive_oversampling=0).

    RELION-specific keys:
        convergence_state : RefinementState -- final convergence state
        data_vs_prior_trajectory : list of jnp.ndarray -- per-iteration
            data_vs_prior curves
        healpix_order_trajectory : list of int -- HEALPix order per iter
        ave_Pmax_trajectory : list of float -- average Pmax per iter
    """
    if options is None:
        options = RefinementOptions()

    options = with_validated_sampling_schedule(options)

    scoring_dtype = _dense_global_scoring_dtype()
    symmetry = options.symmetry.point_group
    schedule = options.schedule
    adaptive = options.adaptive
    parity = options.parity
    local_search = options.local_search
    k_class = options.k_class
    replay = options.replay
    debug = options.debug
    batching = options.batching
    expected_accuracy = debug.expected_accuracy

    particle_diameter_ang = schedule.particle_diameter_ang
    tau2_fudge = parity.tau2_fudge
    perturb_replay_relion_dir = parity.perturb_replay_relion_dir
    perturb_replay_relion_prefix = parity.perturb_replay_relion_prefix
    perturb_replay_max_iter = parity.perturb_replay_max_iter
    init_relion_iteration = schedule.init_relion_iteration
    final_replay_override = replay.final_replay_override
    n_classes = k_class.n_classes
    stop_after_local_search = debug.stop_after_local_search
    sealed_sampling_state = debug.sealed_sampling_state

    if options.parity.perturb_replay_restart_state_iterations:
        logger.info(
            "Perturbation replay restart provenance: saved-state iterations=%s",
            list(options.parity.perturb_replay_restart_state_iterations),
        )

    setup_t0 = time.time()
    setup_phase_seconds = {}

    def _mark_setup_phase(name: str) -> None:
        setup_phase_seconds[name] = time.time() - setup_t0

    volume_shape = experiment_datasets[0].volume_shape
    # Keep the input scalar type for host arithmetic; geometry validates its value.
    source_pixel_size_angstrom = experiment_datasets[0].voxel_size
    image_geometry = ImageGeometry(
        image_shape=experiment_datasets[0].image_shape,
        pixel_size_angstrom=source_pixel_size_angstrom,
    )
    grid_size = image_geometry.box_size
    n_classes = int(n_classes)
    k_class_enabled = n_classes > 1
    if (parity.relion_optics_image_sizes is None) != (parity.relion_optics_pixel_sizes is None):
        raise ValueError(
            "relion_optics_image_sizes and relion_optics_pixel_sizes must be supplied together",
        )
    optics_image_sizes = None
    optics_pixel_sizes = None
    if parity.relion_optics_image_sizes is not None:
        optics_image_sizes = np.asarray(parity.relion_optics_image_sizes, dtype=np.int64).reshape(-1)
        optics_pixel_sizes = np.asarray(parity.relion_optics_pixel_sizes, dtype=np.float64).reshape(-1)
        if optics_image_sizes.shape != optics_pixel_sizes.shape or optics_image_sizes.size == 0:
            raise ValueError("RELION optics image geometry arrays must be non-empty and aligned")
    model_pixel_size = (
        image_geometry.pixel_size_angstrom
        if parity.relion_model_pixel_size is None
        else float(parity.relion_model_pixel_size)
    )
    if not np.isfinite(model_pixel_size) or model_pixel_size <= 0.0:
        raise ValueError(f"RELION model pixel size must be positive, got {model_pixel_size}")
    multi_shape_halves = isinstance(experiment_datasets[0], MultiShapeHalf)
    # Subtomogram particles (S4.2): units are particles over their tilt images, offsets are 3D.
    tomo_halves = isinstance(experiment_datasets[0], TomoHalf)
    relion_translation_angle_scale = (
        # Shape classes carry their translations in class pixels already; tilt images have their own phases.
        1.0
        if multi_shape_halves or tomo_halves
        else _relion_k1_translation_angle_scale(
            n_classes=n_classes,
            model_pixel_size=model_pixel_size,
            optics_pixel_sizes=optics_pixel_sizes,
        )
    )
    if relion_translation_angle_scale != 1.0:
        logger.info(
            "RELION K=1 translation phases: model_pixel_size=%.12g "
            "optics_pixel_size=%.12g angle_scale=%.17g",
            model_pixel_size,
            float(optics_pixel_sizes[0]),
            relion_translation_angle_scale,
        )
    _validate_bpref_particle_order_scope(
        preserve_bpref_particle_order=parity.preserve_bpref_particle_order,
        n_classes=n_classes,
        init_relion_iteration=init_relion_iteration,
        perturb_replay_relion_dir=perturb_replay_relion_dir,
        replay_iteration_overrides=replay.replay_iteration_overrides,
        sealed_sampling_state=sealed_sampling_state,
        sealed_scoring_context=debug.sealed_scoring_context,
        allow_replayed_bpref_particle_order=parity.allow_replayed_bpref_particle_order,
        allow_state_swap_fresh_bpref_particle_order=debug.state_swap_probe is not None,
        continues_own_run=options.checkpoint.resume is not None,
    )
    source_faithful_spectrum_norm = _fresh_k1_spectrum_norm_default(
        preserve_bpref_particle_order=parity.preserve_bpref_particle_order,
    )
    class_log_priors, class_weights = _initialize_class_log_priors(
        n_classes,
        k_class.init_class_log_priors,
        replay.init_direction_prior,
    )

    reconstruction_settings = ReconstructionSettings(
        grid_size=grid_size,
        voxel_size=image_geometry.pixel_size_angstrom,
        volume_shape=volume_shape,
        padding_factor=RECONSTRUCTION_PADDING_FACTOR,
        projection_padding_factor=PROJECTION_PADDING_FACTOR,
        minres_map=RELION_MINRES_MAP,
        width_mask_edge=IMAGE_MASK_EDGE_PIXELS,
        fmask_edge=REFERENCE_FILTER_EDGE_SHELLS,
        tau2_fudge=parity.tau2_fudge,
        particle_diameter_angstrom=schedule.particle_diameter_ang,
        first_iteration_lowpass_angstrom=parity.relion_firstiter_ini_high_angstrom,
    )
    snapshot_capture = SnapshotCapture(
        n_classes=n_classes,
        grid_size=grid_size,
        voxel_size=image_geometry.pixel_size_angstrom,
        tau2_fudge=tau2_fudge,
    )

    configure_half_image_preprocessing(
        experiment_datasets,
        pixel_size_angstrom=source_pixel_size_angstrom,
        particle_diameter_angstrom=particle_diameter_ang,
        fourier_backend=parity.image_fourier_backend,
        source_faithful_spectrum_norm=source_faithful_spectrum_norm,
        log=logger,
    )

    _mark_setup_phase("mask_and_image_cache")

    state = initialize_refinement_state(
        options,
        image_geometry,
        subtomogram=tomo_halves,
        dtype=scoring_dtype,
    )
    resume = options.checkpoint.resume
    _mark_setup_phase("state_init")

    # The refinement schedule owns the initial coarse HEALPix grid; a continuation
    # rebuilds the grid of its restored sampling state.
    initial_grid_order = (
        int(schedule.init_healpix_order) if resume is None else _exhaustive_grid_order_for_state(state)
    )
    if sealed_sampling_state is not None:
        initial_grids = build_sealed_initial_coarse_grids(
            sealed_sampling_state,
            initialized_healpix_order=(
                schedule.init_healpix_order if resume is None else state.healpix_order
            ),
            voxel_size=source_pixel_size_angstrom,
            symmetry=symmetry,
            log=logger,
        )
    else:
        initial_grids = build_initial_coarse_grids(
            initial_grid_order,
            translations if resume is None else None,
            translation_range=(
                schedule.init_translation_range if resume is None else state.translation_range
            ),
            translation_step=(
                schedule.init_translation_step if resume is None else state.translation_step
            ),
            n_classes=n_classes,
            voxel_size=source_pixel_size_angstrom,
            symmetry=symmetry,
        )
    current_rotation_grid = initial_grids.rotation_grid
    base_translations = initial_grids.base_translations
    current_translations = initial_grids.translations
    # Unperturbed base grid — `current_translations` may be replaced per-iter by
    # a perturbed copy (SamplingPerturbation). Keep the base so each iter
    # perturbs a fresh copy rather than compounding prior perturbations.
    # Keep RELION's host-RFLOAT base grid separate so each perturbation starts
    # from the unrounded coordinates.  In double mode the score/pose grid is
    # also RFLOAT; explicit CUDA-f32 helpers cast only at their ABI boundary.
    if debug.save_intermediates_dir is not None:
        os.makedirs(debug.save_intermediates_dir, exist_ok=True)

    collect_local_search_profile = (
        debug.save_intermediates_dir is not None if local_search.local_search_profile_mode == "auto" else local_search.local_search_profile_mode == "on"
    )
    if debug.stop_after_local_search_profile:
        collect_local_search_profile = True
    if debug.stop_after_local_search_score_only:
        stop_after_local_search = True
    _mark_setup_phase("sampling_grid")

    padded_volume_shape = tuple(d * RECONSTRUCTION_PADDING_FACTOR for d in volume_shape)

    batch_planner = BatchPlanner(
        requested=batching,
        image_shape=image_geometry.image_shape,
        volume_shape=volume_shape,
        n_classes=n_classes,
        log=logger,
    )

    # State: two half-set references.  For K-class refinement each half stores
    # an explicit leading class axis; single-class callers keep the historical
    # flat per-half reference layout.
    initial_maps = _normalize_initial_means(init_volume, n_classes)
    del init_volume
    initial_real_references_by_half = prepare_initial_real_references(
        replay.init_reference_real,
        volume_shape=volume_shape,
        n_classes=n_classes,
        init_relion_iteration=init_relion_iteration,
        log=logger,
    )
    initial_noise_variance_per_half = _normalize_noise_variance_per_half(
        init_noise_variance,
        n_halves=2,
    )
    initial_noise_variance = _mean_noise_variance(initial_noise_variance_per_half)
    optics_group_ids_per_half = _optics_group_ids_per_half(
        parity.optics_group_ids_per_half, initial_noise_variance_per_half, experiment_datasets
    )
    # A continued run takes its start-up model state from the snapshot, below.
    if resume is None:
        reference_model = initialize_reference_model(
            initial_maps,
            jnp.asarray(init_mean_variance),
            use_per_half_mean_variance=parity.use_per_half_mean_variance,
            k_class_enabled=k_class_enabled,
            log=logger,
        )
    # The reference owner alone retains the start-up tau2, so the first M-step
    # replacement releases it from the device.
    del initial_maps, init_mean_variance
    _mark_setup_phase("initial_arrays")

    # History tracking: one RefinementHistory instance accumulates every
    # per-iteration trajectory (see helpers/iteration_history.py).
    history = RefinementHistory()
    take_pass_engines()  # entries from before this run's first iteration belong to no iteration
    take_coarse_engine_calls()
    previous_assignments = [None, None]
    class_assignments = [None, None]
    previous_class_assignments = [None, None]
    previous_best_rotations = [None, None]
    halves = initialize_halfsets(
        experiment_datasets,
        optics_group_ids=optics_group_ids_per_half,
        previous_best_translations=replay.init_previous_best_translations,
        previous_best_rotation_eulers=replay.init_previous_best_rotation_eulers,
        image_corrections=replay.init_image_corrections,
        scale_corrections=replay.init_scale_corrections,
        group_ids=replay.init_group_ids,
        group_count=replay.init_group_count,
    )
    # RELION measures the first iteration's orientation changes from the input angles, as its offset
    # changes from the input offsets (updateOverallChangesInHiddenVariables); they seed the smallest-change
    # trackers of the hidden-variable stall counter.
    if resume is None:
        previous_best_rotations = [
            None
            if half.rotation_eulers is None
            else np.zeros((0, 3, 3), dtype=scoring_dtype)
            if len(half.rotation_eulers) == 0
            else np.asarray(utils.R_from_relion(np.asarray(half.rotation_eulers), degrees=True), dtype=scoring_dtype)
            for half in halves
        ]
    previous_data_vs_prior_for_scheduling = (
        None
        if schedule.init_data_vs_prior is None
        else np.asarray(schedule.init_data_vs_prior, dtype=scoring_dtype)
    )
    tau2_update_details = None
    tau2_update_details_per_half = None

    # C1 (RELION-parity): per-iter sigma2_offset update from data. Initialized
    # from `init_translation_sigma_angstrom`; updated from RELION's
    # posterior-weighted offset moment when the E-step path propagates it.
    # RELION stores and updates this quantity in Angstrom², and its default
    # lower bound is min_sigma2_offset=2 Å² (ml_optimiser.cpp).
    current_sigma_offset_angstrom_per_half = _as_sigma_offset_half_pair(schedule.init_translation_sigma_angstrom)
    current_sigma_offset_angstrom = _mean_sigma_offset_per_half(current_sigma_offset_angstrom_per_half)
    expected_accuracy_trial_local_indices = None
    expected_accuracy_trial_particle_ids = None
    relion_incr_size = int(schedule.init_relion_incr_size)
    if relion_incr_size <= 0:
        raise ValueError("init_relion_incr_size must be positive")
    relion_has_high_fsc_at_limit = bool(schedule.init_has_high_fsc_at_limit) if schedule.init_has_high_fsc_at_limit is not None else False

    # --- Direction prior from snapshot ---
    if resume is None or resume.direction_prior is None:
        direction_priors = initial_direction_priors_from_snapshot(
            replay.init_direction_prior,
            n_classes=n_classes,
            dtype=scoring_dtype,
            log=logger,
            symmetry=symmetry, expected_order=current_rotation_grid.healpix_order,
        )
    _mark_setup_phase("direction_prior")

    # Extract per-shell radial profiles from the input pixel-array noise
    # variances for diagnostic logging ("noise update per shell: old=... new=...").
    if resume is None:
        noise_model = initialize_noise_model(
            initial_noise_variance_per_half,
            average_variance=initial_noise_variance,
            image_shape=image_geometry.image_shape,
            dtype=scoring_dtype,
        )
    del initial_noise_variance_per_half, initial_noise_variance
    _mark_setup_phase("noise_radial_init")

    # RELION randomises each half once at the first iteration and then uses
    # the first 100 half-1 particles for calculateExpectedAngularErrors.
    # Build that immutable local order once.  A missing/rebuilt-without-this-
    # helper binding is handled fail-closed below: acc_rot stays infinite and
    # cannot trigger convergence.
    effective_optimizer_random_seed = (
        parity.perturb_seed if parity.optimizer_random_seed is None else parity.optimizer_random_seed
    )
    expected_accuracy_trial_order = prepare_relion_half1_trial_order(
        expected_accuracy=expected_accuracy,
        half1_dataset=experiment_datasets[0],
        optimizer_random_seed=effective_optimizer_random_seed,
        init_relion_iteration=init_relion_iteration,
        log=logger,
    )
    expected_accuracy_inputs = Half1AccuracyInputs(
        trial_order_local=expected_accuracy_trial_order,
        dataset=experiment_datasets[0],
        volume_shape=volume_shape,
        padding_factor=PROJECTION_PADDING_FACTOR,
        sigma2_fudge=RELION_DEFAULT_SIGMA2_FUDGE,
        optimizer_random_seed=effective_optimizer_random_seed,
        expected_accuracy=expected_accuracy,
        optics_group_ids=optics_group_ids_per_half[0],
    )

    follower_setup = setup_relion_follower_scale_state(
        options,
        relion_half_inputs=halves,
        experiment_datasets=experiment_datasets,
        k_class_enabled=k_class_enabled,
    )

    # --- Continue from the run files of an earlier run (RELION --continue) ---
    # The snapshot replaces every value the next numbered iteration reads, so the
    # first loop iteration runs as iteration init_relion_iteration + 1 of the
    # uninterrupted run (see relax/refinement/iteration_snapshot.py).
    if resume is not None:
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
        # An empty half (Class3D's second accumulator) keeps an empty stack, as the loop does.
        previous_best_rotations = [
            None
            if eulers is None
            else np.zeros((0, 3, 3), dtype=scoring_dtype)
            if len(eulers) == 0
            else np.asarray(utils.R_from_relion(np.asarray(eulers), degrees=True), dtype=scoring_dtype)
            for eulers in resume.rotation_eulers
        ]
        if k_class_enabled:
            class_assignments = [None if c is None else np.asarray(c) for c in resume.class_assignments]
            previous_class_assignments = [None if c is None else c.copy() for c in class_assignments]
            class_weights = np.asarray(resume.class_weights, dtype=np.float64)
            class_log_priors = np.log(class_weights)
        previous_data_vs_prior_for_scheduling = np.asarray(resume.data_vs_prior, dtype=scoring_dtype)
        current_sigma_offset_angstrom_per_half = _as_sigma_offset_half_pair(resume.sigma_offset_angstrom)
        current_sigma_offset_angstrom = _mean_sigma_offset_per_half(current_sigma_offset_angstrom_per_half)
        relion_incr_size = int(resume.incr_size)
        relion_has_high_fsc_at_limit = bool(resume.has_high_fsc_at_limit)
        random_perturbation = float(resume.random_perturbation)
        if resume.direction_prior is not None:
            # The saved order resolves a prior length that is ambiguous under symmetry.
            saved_orders = [int(resume.extra.get(f"direction_prior_order_half{h + 1}", -1)) for h in range(2)]
            saved_orders = [None if order < 0 else order for order in saved_orders]
            direction_priors = initial_direction_priors_from_snapshot(
                resume.direction_prior,
                n_classes=n_classes,
                dtype=scoring_dtype,
                log=logger,
                symmetry=symmetry, expected_order=saved_orders[0],
            )
            direction_priors = [
                DirectionPrior(prior.values, order)
                for prior, order in zip(direction_priors, saved_orders, strict=True)
            ]
        logger.info(
            "Continuing after numbered iteration %d: current_size=%d healpix_order=%d "
            "local_search=%s resolution=%.3f A",
            int(resume.relion_iteration),
            int(resume.current_size),
            int(state.healpix_order),
            bool(state.do_local_search),
            float(state.current_resolution),
        )
    else:
        # --- RELION SamplingPerturbation state (healpix_sampling.cpp:167-174) ---
        # RELION applies a random rigid rotation of the entire SO(3) trial grid at
        # each iteration: A -> A @ R_perturb with R_perturb = R_from_relion([m,m,m])
        # and m = random_perturbation * angular_sampling. The random_perturbation
        # is advanced per iter via realWRAP(prev + rnd_unif(0.5*pf, pf), -pf, +pf).
        # For exact parity replay, read _rlnSamplingPerturbInstance from RELION's
        # per-iter sampling.star.
        if parity.perturb_factor > 0 and parity.perturb_seed is not None:
            random_perturbation = relion_sampling_perturbation_for_iteration(
                parity.perturb_factor,
                parity.perturb_seed,
                init_relion_iteration,
            )
            logger.info(
                "Perturbation init: relion_iter=%d random_seed=%d rp=%+.5f",
                int(init_relion_iteration),
                int(parity.perturb_seed),
                random_perturbation,
            )
        else:
            random_perturbation = 0.0
    perturb_rng = None if parity.perturb_seed is not None else np.random.default_rng()
    # RELION's per-class MlModel::acc_rot/acc_trans for model.star: zero until the
    # first expected-accuracy estimate (ml_model.cpp:68), then the latest estimate.
    if resume is not None and resume.acc_rot_per_class is not None:
        model_acc_rot_per_class = np.array(resume.acc_rot_per_class, dtype=np.float64)
        model_acc_trans_per_class = np.array(resume.acc_trans_per_class_angstrom, dtype=np.float64)
    else:
        model_acc_rot_per_class = np.zeros(n_classes, dtype=np.float64)
        model_acc_trans_per_class = np.zeros(n_classes, dtype=np.float64)
    iteration = 0
    _mark_setup_phase("before_iterations")
    logger.info(
        "RELION mode setup timing before iteration loop: %s",
        ", ".join(f"{key}={value:.1f}s" for key, value in setup_phase_seconds.items()),
    )
    native_sampling_boundary = replay_policy._native_sampling_boundary_for_iteration(
        iteration=iteration,
        perturb_replay_relion_dir=perturb_replay_relion_dir,
        perturb_replay_max_iter=perturb_replay_max_iter,
        sealed_sampling_state=sealed_sampling_state,
    )
    # A numbered RELION sampling STAR is the state *after* the expectation
    # transition that produced it.  The next expectation computes
    # image_coarse_size from that saved state before updateAngularSampling.
    # Keep this replay boundary separate from RECOVAR's end-of-iteration
    # RefinementState, which may already have advanced one order.
    replay_saved_healpix_order = (
        None if native_sampling_boundary else int(state.healpix_order)
    )
    frozen_initial_scoring_state = None
    frozen_initial_scoring_state_sha256 = None
    def _state_swap_inputs():
        """The scoring-state values a state-swap probe snapshots and later swaps, as bound right now."""

        return dict(
            state=state,
            cs=current_size,
            reference_model=reference_model,
            noise_model=noise_model,
            relion_half_inputs=halves,
            previous_best_rotations=previous_best_rotations,
            current_sigma_offset_angstrom=current_sigma_offset_angstrom,
            current_sigma_offset_angstrom_per_half=current_sigma_offset_angstrom_per_half,
            direction_priors=direction_priors,
        )

    def _frozen_scoring_state_now():
        """The scoring-state arrays as bound right now; the loop re-binds several of them per iteration."""

        return _frozen_scoring_state_arrays(
            means=reference_model.maps,
            mean_variance=reference_model.tau2,
            mean_variance_per_half=(
                reference_model.tau2_per_half if parity.use_per_half_mean_variance else None
            ),
            relion_half_inputs=halves,
            noise_variance_per_half=noise_model.variance_per_half,
            current_sigma_offset_angstrom_per_half=current_sigma_offset_angstrom_per_half,
            direction_priors=direction_priors,
            experiment_datasets=experiment_datasets,
            sealed_sampling_state=sealed_sampling_state,
            sealed_scoring_context=debug.sealed_scoring_context,
        )

    if debug.assert_initial_scoring_state_immutable:
        if k_class_enabled:
            raise RuntimeError(
                "Frozen scoring-state immutability assertion currently supports K=1 only"
            )
        frozen_initial_scoring_state = _frozen_scoring_state_now()
    # Per-half numbered-iteration assignments; a final-only replay
    # (--max_iter 0 --force-final-after-zero-iterations) runs no numbered
    # iteration and reports none.
    hard_assignments = [None, None]
    while (schedule.force_max_iter_after_convergence or not state.has_converged) and iteration < schedule.max_iter:
        # A continued run's first iteration follows the snapshot's iteration.
        has_previous_iteration = iteration > 0 or resume is not None
        # RELION's Class3D from one reference scores each particle against one random class in its first
        # iteration (do_generate_seeds, ml_optimiser.cpp:4626-4633, :4880-4898). With --firstiter_cc the
        # first iteration scores class 0 alone (CC), its model is then copied to every class, and the random
        # classes are seeded in the second iteration.
        seeded_start = (
            k_class.first_iteration_seed_classes is not None and resume is None and int(init_relion_iteration) == 0
        )
        seed_after_cc = seeded_start and bool(parity.emulate_relion_firstiter_cc)
        single_class_iteration = seed_after_cc and iteration == 0
        seed_iteration = seeded_start and iteration == (1 if seed_after_cc else 0)
        if perturb_replay_relion_dir is not None and replay_policy._past_perturb_replay_max_iter(
            iteration, perturb_replay_max_iter
        ):
            logger.info(
                "Replay override: disabling RELION per-iteration STAR replay from "
                "iteration %d onward (--replay-override-max-iter %d)",
                iteration + 1,
                perturb_replay_max_iter,
            )
            perturb_replay_relion_dir = None
            replay_saved_healpix_order = None
        native_sampling_boundary = replay_policy._native_sampling_boundary_for_iteration(
            iteration=iteration,
            perturb_replay_relion_dir=perturb_replay_relion_dir,
            perturb_replay_max_iter=perturb_replay_max_iter,
            sealed_sampling_state=sealed_sampling_state,
        )
        # RELION checks convergence at the top of iteration n from the
        # completed n-1 statistics and the fine-enough decision latched during
        # expectation n-1.  If true, iteration n is the unnumbered joined
        # all-data pass rather than another numbered half-set iteration.
        if (
            uses_native_auto_refine(
                native_sampling_boundary=native_sampling_boundary,
                n_classes=n_classes,
            )
            and not schedule.force_max_iter_after_convergence
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
        t0 = time.time()
        _parity_dump.start_iteration(iteration)
        iter_replay_override = None
        if replay.replay_iteration_overrides is not None and iteration < len(replay.replay_iteration_overrides):
            iter_replay_override = replay.replay_iteration_overrides[iteration]
        relion_firstiter_cc_this_iter = bool(
            parity.emulate_relion_firstiter_cc and init_relion_iteration == 0 and iteration == 0
        )
        first_iter_normalized_cc_this_iter = bool(
            parity.first_iteration_score_mode == "normalized_cc" and init_relion_iteration == 0 and iteration == 0
        )
        first_iter_hard_reconstruction_this_iter = bool(
            parity.first_iteration_reconstruction_mode == "hard" and init_relion_iteration == 0 and iteration == 0
        )
        firstiter_score_mode_this_iter = (
            "normalized_cc" if (relion_firstiter_cc_this_iter or first_iter_normalized_cc_this_iter) else "gaussian"
        )
        firstiter_winner_take_all_this_iter = bool(
            relion_firstiter_cc_this_iter or first_iter_hard_reconstruction_this_iter
        )
        numbered_relion_iteration = replay_policy._numbered_relion_iteration(init_relion_iteration, iteration)

        if follower_setup.follower_scale_state is not None:
            _dispatch_relion_follower_scale_for_numbered_iteration(
                follower_setup,
                history,
                iteration=iteration,
                numbered_relion_iteration=numbered_relion_iteration,
                relion_half_inputs=halves,
                relion_follower_scale_replay_source=replay.relion_follower_scale_replay,
                dtype=scoring_dtype,
                logger=logger,
            )

        # Image support uses the preceding iteration's spectra, before replay
        # and angular sampling select this expectation's grid.
        if not has_previous_iteration:
            image_size_plan = plan_initial_image_size(
                schedule,
                parity=parity,
                grid_size=grid_size,
                pixel_size_angstrom=source_pixel_size_angstrom,
                incr_size=relion_incr_size,
                has_high_fsc_at_limit=relion_has_high_fsc_at_limit,
                dtype=scoring_dtype,
                log=logger,
            )
            current_size = image_size_plan.size
            data_vs_prior_iter = image_size_plan.data_vs_prior
            if data_vs_prior_iter is not None:
                previous_data_vs_prior_for_scheduling = data_vs_prior_iter
            relion_incr_size = image_size_plan.incr_size
            relion_has_high_fsc_at_limit = image_size_plan.has_high_fsc_at_limit
        else:
            prev_cs = history.current_sizes[-1] if history.current_sizes else int(resume.current_size)
            if k_class_enabled:
                image_size_plan = plan_class_image_size(
                    previous_data_vs_prior_for_scheduling,
                    previous_size=prev_cs,
                    grid_size=grid_size,
                    pixel_size_angstrom=source_pixel_size_angstrom,
                    incr_size=relion_incr_size,
                    ave_pmax=state.ave_Pmax,
                    completed_relion_iteration=int(init_relion_iteration) + int(iteration),
                    parity=parity,
                    dtype=scoring_dtype,
                    log=logger,
                )
                _kclass_dump_dir = os.environ.get("RELAX_KCLASS_DUMP_DIR")
                if _kclass_dump_dir:
                    reconstruction_diagnostics.write_class_image_size(
                        image_size_plan,
                        output_dir=_kclass_dump_dir,
                        previous_size=prev_cs,
                        grid_size=grid_size,
                        iteration=iteration,
                        has_high_fsc_at_limit=relion_has_high_fsc_at_limit,
                        incr_size=relion_incr_size,
                        state=state,
                    )
                current_size = image_size_plan.size
            else:
                image_size_plan = plan_halfmap_image_size(
                    history.fsc_history,
                    growth_fsc_history=history.fsc_for_growth_history,
                    restart=resume,
                    data_vs_prior=previous_data_vs_prior_for_scheduling,
                    previous_size=prev_cs,
                    grid_size=grid_size,
                    pixel_size_angstrom=source_pixel_size_angstrom,
                    incr_size=relion_incr_size,
                    has_high_fsc_at_limit=relion_has_high_fsc_at_limit,
                    ave_pmax=state.ave_Pmax,
                    completed_relion_iteration=int(init_relion_iteration) + int(iteration),
                    parity=parity,
                    dtype=scoring_dtype,
                    log=logger,
                )
                current_size = image_size_plan.size
                data_vs_prior_iter = image_size_plan.data_vs_prior
                previous_data_vs_prior_for_scheduling = data_vs_prior_iter
                relion_incr_size = image_size_plan.incr_size
                relion_has_high_fsc_at_limit = image_size_plan.has_high_fsc_at_limit

        current_size = quantize_current_size(current_size, ori_size=grid_size)
        if has_previous_iteration:
            logger.info(
                "RELION current-size decision: iter=%d prev=%d res_shell=%d "
                "incr_size=%d high_fsc_at_limit=%s ave_Pmax=%.6f raw=%d quantized=%d",
                iteration + 1,
                int(prev_cs),
                int(image_size_plan.resolution_shell),
                int(relion_incr_size),
                bool(relion_has_high_fsc_at_limit),
                float(state.ave_Pmax),
                int(image_size_plan.raw_size),
                int(current_size),
            )
        del image_size_plan
        if adaptive.relion_current_sizes is not None:
            if iteration < len(adaptive.relion_current_sizes):
                oracle_cs = int(adaptive.relion_current_sizes[iteration])
            else:
                oracle_cs = int(adaptive.relion_current_sizes[-1])
            if oracle_cs <= 0:
                oracle_cs = int(schedule.init_current_size)
            current_size = quantize_current_size(oracle_cs, ori_size=grid_size)
            logger.info(
                "Current-size oracle: iteration %d using current_size=%d",
                iteration + 1,
                current_size,
            )

        # RELION updates image_coarse_size before updateAngularSampling at the
        # start of expectation(). Preserve that incoming sampling order even
        # when replay/native scheduling advances state.healpix_order below.
        coarse_size_healpix_order = relion_expectation_coarse_size_order(
            state_healpix_order=state.healpix_order,
            replay_saved_healpix_order=replay_saved_healpix_order,
        )

        # --- Replay override: force recovar's sampling state to mirror RELION ---
        # When replaying, RELION's per-iter sampling.star / model.star /
        # iter_replay_override dict dictate the actual hp_order, sigma priors,
        # translation grid, current_size, direction priors, noise, etc. used
        # at this iteration. Helper mutates state + relion_half_inputs +
        # direction-prior lists in place; returns explicit new values for
        # everything else.
        recovar_state_swap_snapshot = None
        state_swap_target_this_iteration = (
            debug.state_swap_probe is not None
            and int(debug.state_swap_probe.get("iteration", -1)) == int(iteration)
        )
        if state_swap_target_this_iteration:
            recovar_state_swap_snapshot = _snapshot_state_swap_inputs(**_state_swap_inputs())
        replay_result = apply_iter_replay_overrides(
            iter_replay_override=iter_replay_override,
            perturb_replay_relion_dir=perturb_replay_relion_dir,
            perturb_replay_relion_prefix=perturb_replay_relion_prefix,
            init_relion_iteration=init_relion_iteration,
            iteration=iteration,
            state=state,
            cs=current_size,
            image_geometry=image_geometry,
            n_classes=n_classes,
            relion_half_inputs=halves,
            previous_best_rotations=previous_best_rotations,
            noise_model=noise_model,
            current_sigma_offset_angstrom=current_sigma_offset_angstrom,
            current_sigma_offset_angstrom_per_half=current_sigma_offset_angstrom_per_half,
            direction_priors=direction_priors,
            preserve_existing_direction_prior=replay.preserve_initial_direction_prior,
            sealed_sampling_state=sealed_sampling_state,
            dtype=scoring_dtype,
            symmetry=symmetry,
        )
        current_size = replay_result.cs
        _replay_prior_translations = replay_result.prior_translations
        _replay_meta = replay_result.replay_meta
        previous_best_rotations = replay_result.previous_best_rotations
        noise_model = replay_result.noise_model
        replay_mean_variance = (
            None
            if iter_replay_override is None
            else iter_replay_override.get("mean_variance")
        )
        if replay_mean_variance is not None:
            replay_mean_variance = np.asarray(replay_mean_variance, dtype=np.float64).reshape(-1)
            expected_mean_variance_shape = tuple(reference_model.tau2.shape)
            if replay_mean_variance.shape != expected_mean_variance_shape:
                raise ValueError(
                    "K=1 replay mean_variance shape mismatch: "
                    f"expected {expected_mean_variance_shape}, "
                    f"got {replay_mean_variance.shape}"
                )
            reference_model.tau2 = jnp.asarray(replay_mean_variance)
            logger.info("Replay override: K=1 tau2/mean_variance <- model.star")
        current_sigma_offset_angstrom = replay_result.current_sigma_offset_angstrom
        current_sigma_offset_angstrom_per_half = _as_sigma_offset_half_pair(
            replay_result.current_sigma_offset_angstrom_per_half
        )
        if replay_saved_healpix_order is not None:
            replay_saved_healpix_order = int(state.healpix_order)
        if k_class_enabled and replay_result.class_weights is not None:
            class_weights = np.asarray(replay_result.class_weights, dtype=np.float64)
            class_log_priors = np.log(class_weights)
            logger.info(
                "Replay override: class priors <- direction-prior row sums (%s)",
                ", ".join(f"class {idx + 1}={weight:.4f}" for idx, weight in enumerate(class_weights)),
            )

        reference_model.maps = _maybe_debug_replay_relion_references(
            means=reference_model.maps,
            perturb_replay_relion_dir=(
                None
                if sealed_sampling_state is not None and not state_swap_target_this_iteration
                else perturb_replay_relion_dir
            ),
            perturb_replay_relion_prefix=perturb_replay_relion_prefix,
            init_relion_iteration=init_relion_iteration,
            iteration=iteration,
            volume_shape=volume_shape,
            n_classes=n_classes,
            force=(
                state_swap_target_this_iteration
                and bool(debug.state_swap_probe.get("replay_relion_references", False))
            ),
        )

        (
            current_size,
            reference_model,
            noise_model,
            previous_best_rotations,
            current_sigma_offset_angstrom,
            current_sigma_offset_angstrom_per_half,
            direction_priors,
        ) = _apply_state_swap_probe(
            probe=debug.state_swap_probe,
            iteration=iteration,
            recovar_snapshot=recovar_state_swap_snapshot,
            volume_shape=volume_shape,
            **_state_swap_inputs(),
        )
        if not parity.use_per_half_mean_variance:
            # State-swap diagnostics historically replace the one shared tau2.
            # Do not leave the scorer pointing at pre-swap aliases.
            reference_model.tau2_per_half = shared_tau2_per_half(reference_model.tau2)
        if state_swap_target_this_iteration:
            history.state_swap_probe_applied_relion_iterations.append(int(init_relion_iteration) + int(iteration) + 1)
        if frozen_initial_scoring_state is not None and iteration == 0:
            frozen_initial_scoring_state_sha256 = _assert_frozen_scoring_state_unchanged(
                frozen_initial_scoring_state,
                _frozen_scoring_state_now(),
            )
            logger.info(
                "Frozen scoring-state ownership verified immediately before physical iteration %d scoring",
                int(init_relion_iteration) + iteration + 1,
            )

        # Half 1's projector of this iteration's references, built for the
        # expected-accuracy estimate and reused by the scoring projector setup
        # below: RELION computes each class's projector once per iteration.
        shared_projector_half1 = None
        exact_acc_rot_this_iter = None
        exact_acc_trans_this_iter = None
        exact_acc_rot_per_class_this_iter = None
        exact_acc_trans_per_class_this_iter = None
        exact_accuracy_class_counts_this_iter = None
        exact_accuracy_status_this_iter = "skipped_firstiter_cc"
        should_estimate_exact_accuracy = not relion_firstiter_cc_this_iter
        if native_sampling_boundary and should_estimate_exact_accuracy:
            previous_eulers_half1 = halves[0].rotation_eulers
            if expected_accuracy_trial_order is None or previous_eulers_half1 is None:
                exact_accuracy_status_this_iter = "unavailable_inputs"
                state.acc_rot = float("inf")
                state.acc_trans = float("inf")
                logger.warning(
                    "RELION exact expected accuracy unavailable at iteration %d; "
                    "convergence remains fail-closed",
                    iteration + 1,
                )
            else:
                accuracy_class_ids = _expected_accuracy_class_ids(
                    class_assignments[0],
                    k_class_enabled=k_class_enabled,
                    n_units=experiment_datasets[0].n_units,
                )
                if has_previous_iteration and replay_result.relion_projector_state is None:
                    # Iteration 1 may project the initial real references and a
                    # replay may supply a captured projector; both keep their own.
                    shared_projector_size = min(int(current_size), int(grid_size))
                    shared_projector_half1 = ProjectorReuse(
                        references=reference_model.maps[0],
                        current_size=shared_projector_size,
                        image_box_size=grid_size,
                        projector=prepare_scoring_projector(
                            reference_model.maps[0],
                            volume_shape=volume_shape,
                            current_size=shared_projector_size,
                            padding_factor=PROJECTION_PADDING_FACTOR,
                            n_classes=n_classes,
                            dump_label=f"iter{iteration:03d}_half0",
                        ),
                    )
                try:
                    accuracy = expected_accuracy_inputs.estimate(
                        projector_data=None if shared_projector_half1 is None else shared_projector_half1.projector.data,
                        reference_fourier=reference_model.maps[0],
                        best_eulers_deg=previous_eulers_half1,
                        class_ids=accuracy_class_ids,
                        class_weights=class_weights,
                        sigma2_noise_native=noise_model.radial_per_half[0],
                        current_image_size=current_size,
                    )
                    exact_acc_rot_this_iter = float(accuracy.acc_rot)
                    exact_acc_trans_this_iter = float(accuracy.acc_trans_angstrom)
                    exact_acc_rot_per_class_this_iter = np.asarray(
                        accuracy.acc_rot_per_class,
                        dtype=np.float64,
                    ).copy()
                    exact_acc_trans_per_class_this_iter = np.asarray(
                        accuracy.acc_trans_per_class_angstrom,
                        dtype=np.float64,
                    ).copy()
                    exact_accuracy_class_counts_this_iter = np.asarray(
                        accuracy.class_counts,
                        dtype=np.int64,
                    ).copy()
                    expected_accuracy_trial_local_indices = np.asarray(
                        accuracy.trial_local_indices,
                        dtype=np.int64,
                    ).copy()
                    expected_accuracy_trial_particle_ids = np.asarray(
                        accuracy.trial_particle_ids,
                        dtype=np.int64,
                    ).copy()
                    exact_accuracy_status_this_iter = "ok"
                    model_acc_rot_per_class = exact_acc_rot_per_class_this_iter.copy()
                    model_acc_trans_per_class = exact_acc_trans_per_class_this_iter.copy()
                    state.acc_rot = exact_acc_rot_this_iter
                    state.acc_trans = exact_acc_trans_this_iter
                    logger.info(
                        "RELION exact expected accuracy: acc_rot=%.3f deg, acc_trans=%.4f A "
                        "(trials=%d, first_particle_ids=%s)",
                        exact_acc_rot_this_iter,
                        exact_acc_trans_this_iter,
                        int(accuracy.trial_local_indices.size),
                        accuracy.trial_particle_ids[:5].tolist(),
                    )
                except Exception as exc:
                    exact_accuracy_status_this_iter = f"error:{type(exc).__name__}:{exc}"
                    state.acc_rot = float("inf")
                    state.acc_trans = float("inf")
                    logger.warning(
                        "RELION exact expected-accuracy estimation failed at iteration %d; "
                        "convergence remains fail-closed: %s",
                        iteration + 1,
                        exc,
                    )

        # Accuracy and the preceding iteration's stall counters select this
        # expectation's grid; completed-iteration updates remain after M-step.
        state = advance_expectation_sampling(
            state,
            adaptive,
            iteration=iteration,
            has_previous_iteration=has_previous_iteration,
            native_sampling_boundary=native_sampling_boundary,
            n_classes=n_classes,
            log=logger,
        )

        history.record_scheduling(
            current_size,
            state.healpix_order,
            float(current_sigma_offset_angstrom),
            _copy_optional_float_pair(current_sigma_offset_angstrom_per_half),
        )
        scoring_current_size = int(current_size)

        logger.info(
            "=== RELION Iteration %d/%d: current_size=%d, healpix_order=%d, local_search=%s ===",
            iteration + 1,
            schedule.max_iter,
            scoring_current_size,
            state.healpix_order,
            state.do_local_search,
        )

        # --- Angular step refinement: regenerate rotation grid if needed ---
        # When update_refinement_state incremented healpix_order, we need
        # a new rotation grid at the finer level.
        # IMPORTANT: At order >= 5, the full grid has 2.4M+ rotations which
        # OOMs the GPU.  Instead, keep the order-4 grid as the "base" and
        # rely on local search + oversampling to achieve finer angular steps.
        # The order is still tracked for sigma calculation.
        if state.healpix_order != current_rotation_grid.healpix_order:
            new_order = _exhaustive_grid_order_for_state(state)
            if new_order != current_rotation_grid.healpix_order:
                logger.info(
                    "Regenerating rotation grid: order %d -> %d",
                    current_rotation_grid.healpix_order,
                    new_order,
                )
                current_rotation_grid = sampling.relion_scoring_rotation_grid(
                    new_order, dtype=scoring_dtype,
                    symmetry=symmetry,
                )
            else:
                logger.info(
                    "Angular step refined to order %d (exhaustive grid stays at order %d — local search handles finer sampling)",
                    state.healpix_order,
                    current_rotation_grid.healpix_order,
                )

            # Regenerate translation grid based on updated parameters
            base_translations = sampling._relion_base_translation_grid(
                state.translation_range,
                state.translation_step,
                n_classes=n_classes,
                voxel_size=source_pixel_size_angstrom,
            )
            current_translations = jnp.asarray(base_translations, dtype=scoring_dtype)
            logger.info(
                "New grid: %d rotations, %d translations (range=%.1f, step=%.1f)",
                current_rotation_grid.rotations.shape[0],
                current_translations.shape[0],
                state.translation_range,
                state.translation_step,
            )
        elif perturb_replay_relion_dir is not None and sealed_sampling_state is None:
            # Translation params may have changed under replay without an
            # hp_order bump. Regenerate the translation grid to match RELION.
            _new_t_source = sampling._relion_base_translation_grid(
                state.translation_range,
                state.translation_step,
                n_classes=n_classes,
                voxel_size=source_pixel_size_angstrom,
            )
            _new_t = jnp.asarray(_new_t_source, dtype=scoring_dtype)
            if _new_t.shape != base_translations.shape or not jnp.allclose(
                _new_t,
                np.asarray(base_translations, dtype=scoring_dtype),
            ):
                current_translations = _new_t
                base_translations = _new_t_source
                logger.info(
                    "Replay: regenerated translation grid: %d translations (range=%.2f px, step=%.2f px)",
                    current_translations.shape[0],
                    state.translation_range,
                    state.translation_step,
                )

        # --- Local angular search bookkeeping ---
        # Once RELION enters local search, each image should search around its
        # own previous orientation on the true current HEALPix order. Use the
        # exact rotations selected in the previous iteration, not the nearest
        # snapped grid indices.
        effective_rotations = current_rotation_grid.rotations
        effective_rotation_eulers = np.asarray(
            current_rotation_grid.rotation_eulers,
            dtype=scoring_dtype,
        )
        effective_mstep_rotations = None
        adaptive_pass1_rotations = None
        direction_log_priors = [None, None]
        use_local = state.do_local_search and all(half.rotation_eulers is not None for half in halves)
        if use_local and k_class_enabled:
            # Class3D keeps global searches: RELION switches to local searches from the
            # HEALPix order only under auto-refine (ml_optimiser.cpp:2541-2565, 3936-3938).
            raise RuntimeError("K>1 (Class3D) reached local angular searches; RELION never does")
        adaptive_pass1_source_eulers = np.asarray(effective_rotation_eulers, dtype=np.float64)
        # --- Apply RELION SamplingPerturbation to the trial grid for this iter ---
        # healpix_sampling.cpp:1909-1934 (rotations) + 1810-1820 (translations)
        # Perturbation is a rigid rotation of SO(3): A := A @ R_perturb applied
        # AFTER oversampling. At adaptive_oversampling=0 (os0 RELION runs),
        # the coarse grid IS the trial grid so we apply directly here.
        random_perturbation = resolve_numbered_perturbation(
            random_perturbation,
            parity,
            iteration=iteration,
            init_relion_iteration=init_relion_iteration,
            replay_metadata=_replay_meta,
            replay_dir=perturb_replay_relion_dir,
            rng=perturb_rng,
            log=logger,
        )
        if _replay_meta is not None or parity.perturb_factor > 0:
            # Use RELION's actual hp_order when replaying (recovar's current
            # grid order may be capped at MAX_FULL_GRID_ORDER=4 for memory).
            _angsamp_order = int(_replay_meta["healpix_order"]) if _replay_meta is not None else current_rotation_grid.healpix_order
            angsamp_deg = relion_angular_sampling_deg(_angsamp_order, adaptive_oversampling=0)
            trial_grid = sampling._perturbed_trial_grid(
                rotation_eulers=effective_rotation_eulers,
                mstep_source_eulers=sampling._relion_mstep_source_eulers(
                    effective_rotation_eulers,
                    _angsamp_order,
                    use_grid_eulers=sealed_sampling_state is not None,
                    symmetry=symmetry,
                ),
                base_translations=base_translations,
                translation_step=float(state.translation_step),
                random_perturbation=random_perturbation,
                angular_sampling_deg=angsamp_deg,
                dtype=scoring_dtype,
            )
            effective_rotations = trial_grid.rotations
            effective_rotation_eulers = trial_grid.rotation_eulers
            effective_mstep_rotations = trial_grid.mstep_rotations
            current_translations = trial_grid.translations
        # RELION's coarse device geometry also applies at OS0. Keep this
        # separate from host fine/M-step geometry; see docs/math/zero_coarse_geometry.md.
        if not use_local and (
            int(state.adaptive_oversampling) > 0
            or (
                int(state.adaptive_oversampling) == 0
                and n_classes == 1
                and firstiter_score_mode_this_iter == "gaussian"
                and not firstiter_winner_take_all_this_iter
                and not scoring_policy.DENSE_PRECISION.use_float64_scoring
            )
        ):
            adaptive_pass1_order = (
                int(_replay_meta["healpix_order"])
                if _replay_meta is not None
                else int(current_rotation_grid.healpix_order)
            )
            adaptive_pass1_use_float64 = bool(scoring_policy.DENSE_PRECISION.use_float64_scoring)
            adaptive_pass1_rotations = _relion_adaptive_pass1_rotations(
                adaptive_pass1_source_eulers,
                random_perturbation if (_replay_meta is not None or parity.perturb_factor > 0) else 0.0,
                relion_angular_sampling_deg(adaptive_pass1_order, adaptive_oversampling=0),
                use_float64=adaptive_pass1_use_float64,
            )
            if adaptive_pass1_rotations is not None:
                logger.info(
                    "RELION adaptive pass 1: using %s-built coarse scorer rotations; "
                    "fine/M-step rotations remain host-generated",
                    "double-precision CUDA" if adaptive_pass1_use_float64 else "CUDA",
                )
        # First-iteration CC scores the full translation grid before choosing
        # its single winning pose (ml_optimiser.cpp:9181-9207).
        expectation_windows = plan_expectation_windows(
            scoring_current_size,
            image_geometry,
            model_pixel_size=model_pixel_size,
            optics_image_sizes=None if multi_shape_halves else optics_image_sizes,
            optics_pixel_sizes=optics_pixel_sizes,
            log=logger,
        )
        model_current_size_for_engine = expectation_windows.model_window_size
        image_current_size = expectation_windows.image_size
        cs_for_engine = expectation_windows.image_window_size
        sigma_rot, sigma_psi = relion_local_search_sigmas(
            state.sigma_rot,
            state.sigma_psi,
            use_local=use_local,
            healpix_order=state.healpix_order,
            adaptive_oversampling=state.adaptive_oversampling,
        )

        # Angular step behind this iteration's pass-1 coarse size, when RELION's
        # adaptive formula sets it (shape classes recompute their own from it).
        if use_local:
            local_sampling = prepare_numbered_local_sampling(
                LocalSearchSettings(
                    healpix_order=state.healpix_order + state.adaptive_oversampling,
                    oversampling_order=int(state.adaptive_oversampling) if state.adaptive_oversampling > 0 else 0,
                    sigma_rot=sigma_rot,
                    sigma_psi=sigma_psi,
                    symmetry=symmetry,
                ),
                sampling.TrialGrid(
                    rotations=effective_rotations,
                    rotation_eulers=effective_rotation_eulers,
                    mstep_rotations=effective_mstep_rotations,
                    translations=current_translations,
                ),
                base_translations=base_translations,
                image_window_size=cs_for_engine,
                model_support_size=model_current_size_for_engine,
                base_healpix_order=current_rotation_grid.healpix_order,
                coarse_size_healpix_order=coarse_size_healpix_order,
                perturbation=random_perturbation,
                model_pixel_size=model_pixel_size,
                original_model_size=grid_size,
                optics_image_sizes=optics_image_sizes,
                optics_pixel_sizes=optics_pixel_sizes,
                particle_diameter_angstrom=particle_diameter_ang,
                log=logger,
            )
        else:
            local_sampling = None
        direction_prior_healpix_order = _direction_prior_healpix_order_for_scoring(
            use_local=use_local,
            current_healpix_order=current_rotation_grid.healpix_order,
            state_healpix_order=state.healpix_order,
            adaptive_oversampling=state.adaptive_oversampling,
            local_search_order=local_sampling.search.healpix_order if use_local else None,
        )
        coarse_rotation_ids_for_scoring = (
            _sealed_sampling_rotation_ids(sealed_sampling_state)
            if sealed_sampling_state is not None and not use_local
            else None
        )
        if (
            coarse_rotation_ids_for_scoring is not None
            and coarse_rotation_ids_for_scoring.shape != (int(effective_rotations.shape[0]),)
        ):
            raise RuntimeError(
                "sealed captured rotation IDs do not match the directly materialized scorer grid"
            )

        for _half_idx in range(2):
            half_direction_priors = relion_direction_log_priors_for_half(
                use_local=use_local,
                scoring_healpix_order=direction_prior_healpix_order,
                n_classes=n_classes,
                prior=direction_priors[_half_idx],
                sealed_sampling_state=sealed_sampling_state,
                dtype=scoring_dtype,
                log=logger,
                half_index=_half_idx,
                symmetry=symmetry,
            )
            direction_log_priors[_half_idx] = half_direction_priors

        # --- Run E+M on each half-set ---
        # Two modes: single-pass (adaptive_oversampling=0) or two-pass
        # coarse/fine (adaptive_oversampling>=1).
        significance = SignificanceStatistics()
        use_adaptive = _should_use_adaptive_search(
            adaptive_oversampling=state.adaptive_oversampling,
            use_local=use_local,
            n_rotations=effective_rotations.shape[0],
            symmetry=symmetry,
        )
        # Track the rotation grids used for pose extraction.
        # When adaptive oversampling is active, ha_k indices refer to the
        # oversampled grid (from pass 2), not effective_rotations.
        per_half = PerHalfOutputs()
        hard_assignments = per_half.hard_assignments
        class_assignments = per_half.class_assignments
        class_posterior_per_half = per_half.class_posterior
        class_full_posterior_per_half = per_half.class_full_posterior
        max_posterior_per_half = per_half.max_posterior
        rotation_posterior_per_half = per_half.rotation_posterior
        class_rotation_posterior_per_half = per_half.class_rotation_posterior
        pose_rotations = per_half.pose_rotations  # rotations to use with ha for poses
        pose_rotation_eulers = per_half.pose_rotation_eulers
        best_pose_rotations = per_half.best_pose_rotations
        best_pose_rotation_eulers = per_half.best_pose_rotation_eulers
        best_pose_translations = per_half.best_pose_translations
        translation_search_bases = per_half.translation_search_bases
        # Coarse-grid assignments for local search tracking (always indexed
        # into effective_rotations, even when adaptive oversampling is used).
        coarse_ha = per_half.coarse_ha
        if use_adaptive:
            # --- TWO-PASS ADAPTIVE OVERSAMPLING (RELION parity) ---
            # Pass 1: coarse E-step at reduced resolution to find
            #         significant orientations.
            # Pass 2: oversampled E+M at full current_size for significant
            #         orientations only.

            coarse_image_plan = plan_adaptive_image_size(
                coarse_size_healpix_order,
                expectation_windows,
                image_geometry,
                particle_diameter_angstrom=particle_diameter_ang,
                optics_image_sizes=optics_image_sizes,
                optics_pixel_sizes=optics_pixel_sizes,
                sealed_sampling_state=sealed_sampling_state,
                log=logger,
            )
            coarse_size = coarse_image_plan.size
            coarse_cs = coarse_size if coarse_size < grid_size else None

            logger.info(
                "Adaptive oversampling: pass 1 at coarse_size=%s, "
                "pass 2 at current_size=%s (oversampling=%d, particle_diameter=%s)",
                coarse_cs,
                cs_for_engine,
                state.adaptive_oversampling,
                (f"{float(particle_diameter_ang):.1f} A" if particle_diameter_ang is not None else "box_size"),
            )

        # D.2: per-class noise stats (K-tuple of NoiseStats per half) for the
        # per-class sigma_offset C1 update at end-of-iter. K=1 paths leave
        # this None; K-class paths populate from k_class_result.noise_stats.
        noise_stats_per_half = per_half.noise_stats
        noise_stats_per_half_per_class = per_half.noise_stats_per_class

        projectors = [None, None]
        captured_projector_state = replay_result.relion_projector_state
        # Every dense, local and tomo scorer reads this projector: pass 1 scores RELION's exact
        # coarse operands on every route, as RELION builds Projector::data every iteration.
        projector_t0 = time.time()
        if captured_projector_state is not None:
            projectors = _validate_captured_relion_projector_for_iteration(
                captured_projector_state,
                current_size=model_current_size_for_engine,
                volume_shape=volume_shape,
                padding_factor=PROJECTION_PADDING_FACTOR,
                n_classes=n_classes,
            )
            logger.info(
                "RELION mode: using captured exact Projector::data at current_size=%s "
                "r_max=%s manifest=%s",
                model_current_size_for_engine,
                None if projectors[0] is None else projectors[0].r_max,
                captured_projector_state.source_manifest_sha256,
            )
        else:
            for half in halves:
                if half.dataset.n_units == 0:
                    logger.info(
                        "RELION mode: skipping Projector::data build for empty half-%d dataset",
                        half.index + 1,
                    )
                    continue
                projector = prepare_scoring_projector(
                    reference_model.maps[half.index],
                    volume_shape=volume_shape,
                    current_size=model_current_size_for_engine,
                    padding_factor=PROJECTION_PADDING_FACTOR,
                    n_classes=n_classes,
                    reusable=shared_projector_half1 if half.index == 0 else None,
                    real_references=(
                        initial_real_references_by_half[half.index]
                        if iteration == 0
                        else None
                    ),
                    dump_label=f"iter{iteration:03d}_half{half.index}",
                )
                projectors[half.index] = projector
            logger.info(
                # The slab dtype decides whether pass-2 projection runs on
                # the native texture projector or the vmapped JAX fallback
                # (_relion_projector_texture_enabled requires complex64),
                # so record it rather than leaving the path implicit.
                "RELION mode: built exact Projector::data for scoring at current_size=%s r_max=%s "
                "dtype=%s in %.2fs",
                model_current_size_for_engine,
                None if projectors[0] is None else projectors[0].r_max,
                None if projectors[0] is None else projectors[0].data.dtype,
                time.time() - projector_t0,
            )

        # Freeze the exact iteration-start curve used by RELION's scale XA/AA
        # shell gate.  The scheduling variable is updated again after the
        # reconstruction, before parity diagnostics are written.
        scale_correction_data_vs_prior_this_iter = previous_data_vs_prior_for_scheduling

        diagnostic_half_indices = _significance_dump_half_indices(
            numbered_iteration=numbered_relion_iteration,
            n_classes=n_classes,
            experiment_datasets=experiment_datasets,
        )
        # The two halves are independent inside the E-step. Extracting one
        # half's work into a function changes neither what runs nor its
        # order; it makes the two callable independently, which is what the
        # overlap option uses. Serial dispatch stays the default.
        numbered_variant = DenseVariantPolicy(
            firstiter_score_mode_this_iter=firstiter_score_mode_this_iter,
            firstiter_winner_take_all_this_iter=firstiter_winner_take_all_this_iter,
            k_class_enabled=k_class_enabled,
            relion_firstiter_cc_this_iter=relion_firstiter_cc_this_iter,
            firstiter_coarse_current_size=coarse_cs if use_adaptive else None,
            firstiter_fine_current_size=cs_for_engine if use_adaptive else None,
            firstiter_log_label="" if use_adaptive else "(non-adaptive site) ",
            firstiter_updates_em_kwargs_ibs=bool(use_adaptive),
        )
        numbered_expectation = prepare_numbered_expectation(
            sampling.TrialGrid(
                rotations=effective_rotations,
                rotation_eulers=effective_rotation_eulers,
                mstep_rotations=effective_mstep_rotations,
                translations=current_translations,
            ),
            expectation_windows,
            local_sampling=local_sampling,
            variant=numbered_variant,
            use_adaptive=use_adaptive,
            base_translations=base_translations,
            current_healpix_order=current_rotation_grid.healpix_order,
            oversampling_order=state.adaptive_oversampling,
            translation_step=state.translation_step,
            random_perturbation=random_perturbation,
            adaptive_pass1_rotations=adaptive_pass1_rotations,
            coarse_rotation_ids=coarse_rotation_ids_for_scoring,
            coarse_angular_step_deg=coarse_image_plan.angular_step_deg if use_adaptive else None,
            options=options,
            iteration=iteration,
            numbered_relion_iteration=numbered_relion_iteration,
            collect_local_search_profile=collect_local_search_profile,
            local_profile_history=history.local_profile_history,
        )
        if tomo_halves:
            tomo_oversampling = int(state.adaptive_oversampling)
            tomo_coarse_size = local_sampling.coarse_image_window_size if use_local else coarse_cs
            numbered_tomo_sampling = TomoSampling(
                healpix_order=int(local_sampling.search.healpix_order) - tomo_oversampling if use_local else int(current_rotation_grid.healpix_order),
                oversampling_order=tomo_oversampling,
                offset_range_angst=float(state.translation_range) * image_geometry.pixel_size_angstrom,
                offset_step_angst=float(state.translation_step) * image_geometry.pixel_size_angstrom,
                random_perturbation=float(local_sampling.perturbation if use_local else random_perturbation),
                coarse_size=int(image_geometry.image_shape[0] if tomo_coarse_size is None else tomo_coarse_size),
                fine_size=int(image_geometry.image_shape[0] if cs_for_engine is None else cs_for_engine),
            )
        else:
            numbered_tomo_sampling = None

        def _run_half_estep(k):
            particle_half = halves[k]
            score_result = score_numbered_half(
                HalfScoringData(
                    particles=particle_half,
                    reference=reference_model.maps[k],
                    mean_variance=reference_model.tau2_per_half[k],
                    noise_variance=noise_model.variance_per_half[k],
                    noise_radial=noise_model.radial_per_half[k] if not tomo_halves and particle_half.dataset.n_units else None,
                    projector=projectors[k],
                    scale_group_ids=follower_setup.scale_stats_group_ids_per_half[k],
                    scale_group_count=follower_setup.scale_stats_group_count_per_half[k],
                    scale_correction_data_vs_prior=scale_correction_data_vs_prior_this_iter,
                ),
                numbered_expectation,
                tomo_sampling=numbered_tomo_sampling,
                direction_priors=direction_log_priors[k],
                class_log_priors=class_log_priors,
                sigma_offset_angstrom=_sigma_offset_for_half(
                    current_sigma_offset_angstrom, current_sigma_offset_angstrom_per_half, k,
                ),
                batch_planner=batch_planner,
                image_geometry=image_geometry,
                padded_volume_shape=padded_volume_shape,
                multi_shape_halves=multi_shape_halves,
                options=options,
                replay_prior_translations=_replay_prior_translations,
                initial_class_assignments=k_class.first_iteration_seed_classes if seed_iteration else None,
                single_class_iteration=single_class_iteration,
                scoring_dtype=scoring_dtype,
                source_faithful_spectrum_norm=source_faithful_spectrum_norm,
                relion_translation_angle_scale=relion_translation_angle_scale,
                iteration=iteration,
                numbered_relion_iteration=numbered_relion_iteration,
            )
            per_half.translation_search_bases[k] = score_result.translation_search_base
            per_half.pose_rotations[k] = score_result.pose_rotations
            per_half.pose_rotation_eulers[k] = score_result.pose_rotation_eulers
            per_half.coarse_ha[k] = score_result.coarse_ha
            if particle_half.dataset.n_units != 0:
                score_result = _maybe_host_offload_half0_local_accumulators(
                    half_index=k,
                    use_local=use_local,
                    score_result=score_result,
                    log=logger,
                )
            per_half.update_from(k, score_result, dtype=scoring_dtype)
            record_numbered_half(
                score_result,
                particle_half,
                per_half,
                significance,
                profile_history=history.global_profile_history,
                iteration=iteration,
                image_window_size=cs_for_engine,
                healpix_order=current_rotation_grid.healpix_order,
                k_class_enabled=k_class_enabled,
            )

        _overlap_active = _half_overlap_active(
            options.overlap.overlap_halves,
            diagnostic_half_indices=diagnostic_half_indices,
            log=logger,
        )
        if _overlap_active:
            _run_halves_overlapped(_run_half_estep, diagnostic_half_indices)
            k = diagnostic_half_indices[-1]
        else:
            for k in diagnostic_half_indices:
                _run_half_estep(k)
        if diagnostic_half_indices != (0, 1):
            raise RuntimeError(
                "targeted half-only significance diagnostic returned without writing its "
                "complete target set; refusing to continue with one half missing"
            )

        Ft_y_0, Ft_y_1 = per_half.Ft_y
        Ft_ctf_0, Ft_ctf_1 = per_half.Ft_ctf

        # E-step + per-half M-step accumulators are now both populated.
        _parity_dump.mark_stage(iteration, "e_step")
        from relax.cuda.kernels import drain_relion_preprocess_checks
        drain_relion_preprocess_checks()
        significance.combine()
        if (debug.stop_after_local_search_profile or stop_after_local_search) and use_local:
            elapsed = time.time() - t0
            logger.info(
                "Stopping after local-search diagnostic at iteration %d: profiles=%d score_only=%s wall=%.1fs",
                iteration + 1,
                len(history.local_profile_history),
                bool(debug.stop_after_local_search_score_only),
                elapsed,
            )
            # Local search is K=1 (Class3D was rejected above), so there are no class products.
            merged_mean, merged_class_means = _merged_mean_from_halves(reference_model.maps, None)
            (
                replay_requested_iterations,
                replay_applied_iterations,
            ) = _finalize_relion_follower_scale_replay_telemetry(
                replay.relion_follower_scale_replay,
                applied_iterations=history.relion_follower_scale_replay_applied_iterations,
                logger=logger,
            )
            return {
                "profile_only": True,
                **_model_result_fields(merged_mean, reference_model.maps, merged_class_means, None, None),
                "relion_follower_scale_replay_requested_iterations": replay_requested_iterations,
                "relion_follower_scale_replay_applied_iterations": replay_applied_iterations,
                "convergence_state": state,
                **_numbered_result_metadata(
                    hard_assignments, frozen_initial_scoring_state_sha256,
                    expected_accuracy_trial_local_indices, expected_accuracy_trial_particle_ids,
                    setup_phase_seconds,
                ),
                "final_all_data_ran": False,
                "stop_after_local_search_score_only": bool(debug.stop_after_local_search_score_only),
                **history.to_dict(),
                "wall_times": [elapsed],
                "significant_counts": [significance.recorded],
            }
        if k_class_enabled:
            class_weights = _class_weights_from_posterior(
                class_posterior_per_half,
                n_classes,
                class_weights,
            )
            class_log_priors = np.log(class_weights)
            history.record_class_weights(
                class_weights,
                _class_weights_from_posterior(
                    class_full_posterior_per_half,
                    n_classes,
                    class_weights,
                ),
            )
            logger.info(
                "K-class occupancies: %s",
                ", ".join(f"class {idx + 1}={weight:.4f}" for idx, weight in enumerate(class_weights)),
            )
        mstep_accumulator_shape = _resolve_mstep_accumulator_shape(
            per_half.mstep_accumulator_shape,
            padded_volume_shape,
        )
        mstep_full_half_axis = _resolve_mstep_full_half_axis(
            per_half.mstep_full_half_axis,
            default_axis=-1,
        )

        _bpref_boundary_iteration_matches = reconstruction_diagnostics.audit_prejoin_accumulators(
            (Ft_y_0, Ft_y_1),
            (Ft_ctf_0, Ft_ctf_1),
            reconstruction_settings,
            iteration=iteration,
            current_size=current_size,
            accumulator_shape=mstep_accumulator_shape,
            k_class_enabled=k_class_enabled,
            init_relion_iteration=init_relion_iteration,
            pixel_size_angstrom=source_pixel_size_angstrom,
            log=logger,
        )

        # --- RELION-exact M-step ordering ---
        # K=1 stays on RELION's split-half auto-refine path
        # (compareTwoHalves -> updateSSNRarrays -> reconstruct).
        # K>1 switches to RELION Class3D semantics:
        #   1. combine the two half accumulators per class
        #   2. carry the previous Iref power spectrum forward as tau2
        #   3. run one Wiener solve per class
        #
        # Snapshot the previous-iter means BEFORE the reconstruction so sign
        # alignment has a reference at iter 1.
        if k_class_enabled:
            Ft_y_combined = _combine_optional_half_accumulators(Ft_y_0, Ft_y_1, label="Ft_y")
            Ft_ctf_combined = _combine_optional_half_accumulators(Ft_ctf_0, Ft_ctf_1, label="Ft_ctf")
            # K-class 256px maps are large enough that materializing both
            # previous class stacks on the host immediately after pass 2 can
            # SIGBUS under Slurm/tmp quota pressure.  JAX arrays are immutable;
            # keep device references here and let the later per-class tau2/sign
            # code transfer only the slices it actually needs.
            previous_means = [jnp.asarray(mean) if mean is not None else None for mean in reference_model.maps]
            _t_unreg_first = time.time()
            class_priors = estimate_class_priors(
                previous_means,
                Ft_y_combined,
                Ft_ctf_combined,
                reconstruction_settings,
                half_denominators=(Ft_ctf_0, Ft_ctf_1),
                prior_tau2=reference_model.tau2,
                halves=halves,
                n_classes=n_classes,
                iteration=iteration,
                current_size=current_size,
                image_current_size=image_current_size,
                accumulator_shape=mstep_accumulator_shape,
                full_half_axis=mstep_full_half_axis,
                projector_power_spectrum=(
                    None
                    if not has_previous_iteration or projectors[0] is None
                    else projectors[0].power_spectrum
                ),
                iter_replay_override=iter_replay_override,
                replay=replay,
                scoring_dtype=scoring_dtype,
                started_at=_t_unreg_first,
                log=logger,
            )
            mean_signal_variance = class_priors.variance
            mean_signal_variance_shells = class_priors.shells
            data_vs_prior_iter = class_priors.data_vs_prior
            tau2_update_details_per_class = class_priors.details_per_class
            kclass_tau2_source = class_priors.source
            del class_priors
            history.data_vs_prior_trajectory.append(data_vs_prior_iter)
            previous_data_vs_prior_for_scheduling = data_vs_prior_iter
            tau2_update_details = _stack_class_tau2_update_details(tau2_update_details_per_class)
            del tau2_update_details_per_class
            logger.info(
                "Computed iter-%d Class3D tau2 from %s: %.1fs",
                iteration + 1,
                kclass_tau2_source,
                time.time() - _t_unreg_first,
            )
            reference_model.tau2 = mean_signal_variance
            reference_model.tau2_per_half = shared_tau2_per_half(reference_model.tau2)

            # --- Free previous-iteration means to reclaim GPU memory ---
            # (previous_means already snapshotted earlier for FSC sign alignment)
            for k in range(2):
                reference_model.maps[k] = None

            # --- Now reconstruct the regularized means ---
            _t_recon = time.time()
            reference_model.maps[:] = reconstruct_numbered_class_maps(
                Ft_y_combined,
                Ft_ctf_combined,
                mean_signal_variance_shells,
                reconstruction_settings,
                n_classes=n_classes,
                iteration=iteration,
                current_size=current_size,
                accumulator_volume_shape=mstep_accumulator_shape,
                relion_firstiter_cc_this_iter=relion_firstiter_cc_this_iter,
            )
            logger.info(
                "Regularized reconstruction (2 halves + flatten): %.1fs",
                time.time() - _t_recon,
            )
            if relion_firstiter_cc_this_iter and parity.relion_firstiter_ini_high_angstrom is not None:
                # Class3D tapers each class's tau2_class and data_vs_prior_class the
                # same way (ml_optimiser.cpp:6389-6420). RELION's comment calls this
                # output only, but the next E-step gates each class's scale sums on
                # data_vs_prior_class > 3 (:10473), so the untapered curve let
                # shells past ini_high into iteration 2's scale correction. The
                # class tau2 volumes are recomputed from the Iref power next
                # iteration, so only the shell curves carry the taper.
                data_vs_prior_iter = _firstiter_cc_ini_high_tapered(
                    data_vs_prior_iter,
                    grid_size,
                    source_pixel_size_angstrom,
                    parity.relion_firstiter_ini_high_angstrom,
                    filter_edgewidth=REFERENCE_FILTER_EDGE_SHELLS,
                )
                history.data_vs_prior_trajectory[-1] = data_vs_prior_iter
                previous_data_vs_prior_for_scheduling = data_vs_prior_iter
                tapered_prior = taper_first_cc_class_prior(
                    mean_signal_variance_shells,
                    tau2_update_details,
                    reconstruction_settings,
                    pixel_size_angstrom=source_pixel_size_angstrom,
                )
                mean_signal_variance_shells = tapered_prior.shells
                tau2_update_details = tapered_prior.details
                del tapered_prior
                logger.info(
                    "RELION iter-1 CC emulation: tapered Class3D tau2/data-vs-prior with ini_high=%.2f A",
                    float(parity.relion_firstiter_ini_high_angstrom),
                )
        else:
            retained_Ft_y_0_device = None
            # RELION's --low_resol_join_halves averages the low-resolution shells of
            # the K=1 half accumulators before the Wiener solve; see
            # join_half_accumulators_at_low_resolution for the rationale and cap.
            if parity.low_resol_join_halves_angstrom is not None and parity.low_resol_join_halves_angstrom > 0:
                Ft_y_0, Ft_y_1, Ft_ctf_0, Ft_ctf_1, retained_Ft_y_0_device = join_half_accumulators_at_low_resolution(
                    (Ft_y_0, Ft_y_1),
                    (Ft_ctf_0, Ft_ctf_1),
                    accumulator_volume_shape=mstep_accumulator_shape,
                    grid_size=grid_size,
                    voxel_size=source_pixel_size_angstrom,
                    padding_factor=RECONSTRUCTION_PADDING_FACTOR,
                    low_resolution_angstrom=parity.low_resol_join_halves_angstrom,
                    pixel_resolutions=history.pixel_resolutions,
                    current_resolution=getattr(state, "current_resolution", float("inf")),
                    preserve_inputs=False,
                    return_retained_first_numerator=True,
                )
            previous_means = _snapshot_and_release_previous_k1_means(reference_model.maps)
            _t_unreg_first = time.time()
            # Optional dump of post-join Ft_y, Ft_ctf for shell-by-shell parity
            # comparison against RELION's RELAX_MSTEP_DUMP_DIR. Activated by
            # RELAX_BPREF_ACCUM_DUMP_DIR. One npz per iteration.
            _bpref_accum_dir = os.environ.get("RELAX_BPREF_ACCUM_DUMP_DIR")
            if _bpref_accum_dir and _bpref_boundary_iteration_matches:
                reconstruction_diagnostics.write_bpref_accumulators(
                    _bpref_accum_dir,
                    stage="accum",
                    iteration=iteration,
                    current_size=current_size,
                    padding_factor=RECONSTRUCTION_PADDING_FACTOR,
                    grid_size=grid_size,
                    voxel_size=source_pixel_size_angstrom,
                    volume_shape=volume_shape,
                    accumulator_shape=mstep_accumulator_shape,
                    Ft_y_0=Ft_y_0,
                    Ft_y_1=Ft_y_1,
                    Ft_ctf_0=Ft_ctf_0,
                    Ft_ctf_1=Ft_ctf_1,
                )
            split_prior = estimate_split_half_prior(
                (Ft_y_0, Ft_y_1),
                (Ft_ctf_0, Ft_ctf_1),
                reconstruction_settings,
                current_size=current_size,
                accumulator_shape=mstep_accumulator_shape,
                full_half_axes=per_half.mstep_full_half_axis,
                iteration=iteration,
                scoring_dtype=scoring_dtype,
                started_at=_t_unreg_first,
                log=logger,
            )
            current_iter_fsc = split_prior.fsc
            tau2_fsc_for_update = split_prior.fsc_for_update
            mean_signal_variance = split_prior.variance
            mean_signal_variance_per_half = split_prior.variance_per_half
            mean_signal_variance_shells_per_half = split_prior.shells_per_half
            tau2_update_details_per_half = split_prior.details_per_half
            # Diagnostics follow the half-1 model.star, matching the parity report.
            tau2_update_details = tau2_update_details_per_half[0]
            del split_prior
            logger.info(
                "tau2 update from THIS-iter FSC: old_max=%.4e new_max=%.4e half_max=(%.4e, %.4e)",
                float(jnp.max(jnp.abs(reference_model.tau2))),
                float(jnp.max(jnp.abs(mean_signal_variance))),
                float(jnp.max(jnp.abs(mean_signal_variance_per_half[0]))),
                float(jnp.max(jnp.abs(mean_signal_variance_per_half[1]))),
            )
            reference_model.tau2 = mean_signal_variance
            reference_model.tau2_per_half = _updated_mean_variance_per_half(
                reference_model.tau2,
                mean_signal_variance_per_half,
                use_per_half_mean_variance=parity.use_per_half_mean_variance,
            )

            # --- Now reconstruct the regularized means ---
            # (the previous K=1 references were released by the snapshot above)
            _t_recon = time.time()
            reference_model.maps[:] = reconstruct_numbered_k1_halfmaps(
                (Ft_y_0, Ft_y_1),
                (Ft_ctf_0, Ft_ctf_1),
                mean_signal_variance_shells_per_half,
                reconstruction_settings,
                iteration=iteration,
                current_size=current_size,
                accumulator_volume_shape=mstep_accumulator_shape,
                relion_firstiter_cc_this_iter=relion_firstiter_cc_this_iter,
                retained_first_numerator=retained_Ft_y_0_device,
            )
            logger.info(
                "Regularized reconstruction (2 halves + flatten): %.1fs",
                time.time() - _t_recon,
            )
            retained_Ft_y_0_device = None

            # RELION reconstructs the first-iteration CC maps with the untapered
            # updateSSNRarrays tau2.  Only afterwards does
            # initialLowPassFilterReferences taper tau2/data_vs_prior for the
            # model state and reporting; that tapered spectrum is explicitly not
            # used in the reconstruction calculation (ml_optimiser.cpp:5296-5328).
            if relion_firstiter_cc_this_iter and parity.relion_firstiter_ini_high_angstrom is not None:
                tapered_prior = taper_first_cc_k1_prior(
                    mean_signal_variance_per_half,
                    tau2_update_details_per_half,
                    reconstruction_settings,
                    pixel_size_angstrom=source_pixel_size_angstrom,
                    scoring_dtype=scoring_dtype,
                )
                mean_signal_variance = tapered_prior.variance
                mean_signal_variance_per_half = tapered_prior.variance_per_half
                tau2_update_details_per_half = tapered_prior.details_per_half
                del tapered_prior
                reference_model.tau2 = mean_signal_variance
                reference_model.tau2_per_half = _updated_mean_variance_per_half(
                    reference_model.tau2,
                    mean_signal_variance_per_half,
                    use_per_half_mean_variance=parity.use_per_half_mean_variance,
                )
                tau2_update_details = tau2_update_details_per_half[0]
                logger.info(
                    "RELION iter-1 CC emulation: tapered post-reconstruction tau2/data-vs-prior "
                    "with ini_high=%.2f A",
                    float(parity.relion_firstiter_ini_high_angstrom),
                )
            # The K=1 tau2 volumes are read again only by the next M-step (the
            # resident E-step does not use them). Keep them on the host between
            # uses, as RELION keeps tau2 as a host spectrum: at box 800 the four
            # float32 volumes are 8 GB of the device floor (GPU census, bigbox
            # 14480607). The per-half reconstruction volumes are not read again:
            # drop them here instead of copying them.
            (
                reference_model.tau2,
                reference_model.tau2_per_half,
                mean_signal_variance,
            ) = _host_tau2_volumes(
                reference_model.tau2,
                reference_model.tau2_per_half,
                mean_signal_variance,
            )
            mean_signal_variance_per_half = None
        _parity_dump.mark_stage(iteration, "recon")

        history.significant_counts.append(significance.recorded)

        history.record_rotation_posterior(rotation_posterior_per_half)
        if all(rot_sum is not None for rot_sum in rotation_posterior_per_half):
            if not k_class_enabled:
                learned_priors = learn_k1_direction_priors(
                    rotation_posterior_per_half,
                    direction_prior_order=direction_prior_healpix_order,
                    expected_rotation_count=rotation_grid_size(
                        direction_prior_healpix_order,
                        symmetry=symmetry,
                    ),
                    dtype=scoring_dtype,
                    log=logger,
                    symmetry=symmetry,
                )
                for half_index, learned in enumerate(learned_priors):
                    if learned is not None:
                        direction_priors[half_index] = learned
            else:
                exhaustive_grid_size = rotation_grid_size(
                    current_rotation_grid.healpix_order,
                    symmetry=symmetry,
                )
                if (
                    effective_rotations.shape[0] == exhaustive_grid_size
                    and all(
                        rot_sum is not None
                        for rot_sum in class_rotation_posterior_per_half
                    )
                ):
                    learned_priors = learn_class_direction_priors(
                        class_rotation_posterior_per_half,
                        n_classes=n_classes,
                        healpix_order=current_rotation_grid.healpix_order,
                        dtype=scoring_dtype,
                        symmetry=symmetry,
                    )
                    for half_index, learned in enumerate(learned_priors):
                        direction_priors[half_index] = learned
        if single_class_iteration:
            # After the CC iteration RELION copies class 0's model to every class for the seed iteration:
            # Iref, tau2_class, data_vs_prior_class and pdf_direction, each class taking pdf_class[0] / K
            # (maximizationOtherParameters, ml_optimiser.cpp:6423-6437).
            reference_model.maps = [None if mean is None else _copy_first_class(mean) for mean in reference_model.maps]
            mean_signal_variance = _copy_first_class(mean_signal_variance)
            reference_model.tau2 = mean_signal_variance
            reference_model.tau2_per_half = shared_tau2_per_half(reference_model.tau2)
            mean_signal_variance_shells = _copy_first_class(mean_signal_variance_shells)
            data_vs_prior_iter = _copy_first_class(data_vs_prior_iter)
            history.data_vs_prior_trajectory[-1] = data_vs_prior_iter
            previous_data_vs_prior_for_scheduling = data_vs_prior_iter
            tau2_update_details = {
                key: None if value is None else _copy_first_class(value) for key, value in tau2_update_details.items()
            }
            for half_index, prior in enumerate(direction_priors):
                if prior.values is not None:
                    direction_priors[half_index] = DirectionPrior(
                        _copy_first_class(prior.values), prior.healpix_order,
                    )
            class_weights = np.full(n_classes, float(class_weights[0]) / n_classes, dtype=np.float64)
            class_log_priors = np.log(class_weights)
            logger.info("Class3D one-reference start: copied class 1 to every class after the CC iteration")
        history.record_direction_prior(
            direction_priors,
            k_class_enabled=k_class_enabled,
        )

        # --- Compute unregularized half-maps only when diagnostics need them ---
        # K=1 FSC was already computed above directly from the BackProjector
        # accumulators (current_iter_fsc), matching RELION ordering. For K>1
        # the shared class3D prior is from the previous Iref power spectrum.
        # Reconstructing unreg here is only needed for saved intermediates /
        # parity dumps.
        need_unreg_means = (
            (debug.save_intermediates_dir is not None and not debug.save_intermediates_skip_unregularized)
            or _parity_dump.is_active()
            or (
                options.checkpoint.writer is not None
                and options.checkpoint.writer.wants_unfiltered_maps(numbered_relion_iteration, n_classes=n_classes)
            )
        )
        _t_unreg = time.time()
        if k_class_enabled:
            unreg_means = (
                reconstruct_unregularized_class_means(
                    Ft_y_combined,
                    Ft_ctf_combined,
                    reconstruction_settings,
                    n_classes,
                    accumulator_volume_shape=mstep_accumulator_shape,
                )
                if need_unreg_means
                else [None, None]
            )
            share_kclass_volume_signs(reference_model.maps, unreg_means)
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
            align_k1_volume_signs(reference_model.maps, previous_means, unreg_means, volume_shape)
        logger.info(
            "Unregularized reconstruction (2 halves): %.1fs%s",
            time.time() - _t_unreg,
            "" if need_unreg_means else " (skipped; diagnostics disabled)",
        )

        # K>1 uses the shared per-class data_vs_prior curve to drive growth;
        # K=1 keeps the split-half FSC history.
        if k_class_enabled:
            fsc = None
            history.record_fsc(fsc, None)
            _parity_dump.mark_stage(iteration, "fsc")
        else:
            # FSC was already computed above in the RELION-exact ordering block
            # (current_iter_fsc) and used to derive tau2 BEFORE the Wiener solve.
            # Reuse it here — recomputing would give the same value (same
            # underlying unreg accumulators).
            fsc = current_iter_fsc
            history.record_fsc(fsc, tau2_fsc_for_update)
            _parity_dump.mark_stage(iteration, "fsc")

        # --- Save intermediate volumes if requested ---
        if debug.save_intermediates_dir is not None:
            _save_iteration_intermediates(
                debug.save_intermediates_dir,
                iteration=iteration,
                Ft_y_0=Ft_y_0,
                Ft_y_1=Ft_y_1,
                Ft_ctf_0=Ft_ctf_0,
                Ft_ctf_1=Ft_ctf_1,
                means=reference_model.maps,
                unreg_means=unreg_means,
                fsc=fsc,
                noise_variance=noise_model.average_variance,
                noise_variance_per_half=noise_model.variance_per_half,
                mean_variance=reference_model.tau2,
                hard_assignments=hard_assignments,
                coarse_ha=coarse_ha,
                effective_rotations=effective_rotations,
                current_translations=current_translations,
                use_local=use_local,
                local_search_order=local_sampling.search.healpix_order if use_local else None,
                cs=current_size,
                state=state,
                n_classes=n_classes,
                volume_shape=volume_shape,
                voxel_size=source_pixel_size_angstrom,
                symmetry=symmetry,
            )

        # --- Compute ave_Pmax from the actual E-step maxima ---
        if any(pmax is None for pmax in max_posterior_per_half):
            raise RuntimeError(
                "RELION mode expected per-image posterior maxima from the EM engine",
            )
        combined_max_posterior, ave_pmax, ave_pmax_denominator = _relion_optimizer_average_pmax(
            max_posterior_per_half,
            _relion_pmax_normalization_mass_per_half(
                k_class_enabled=k_class_enabled,
                class_posterior_per_half=class_posterior_per_half,
                noise_stats_per_half=noise_stats_per_half,
            ),
        )
        if k_class_enabled:
            logger.info(
                "Class3D optimizer Pmax: value=%.9f numerator=%.9f "
                "half1_mstep_posterior_mass=%.9f half1_particle_count=%d",
                ave_pmax,
                float(np.sum(np.asarray(max_posterior_per_half[0]), dtype=np.float64)),
                ave_pmax_denominator,
                int(np.asarray(max_posterior_per_half[0]).size),
            )
        history.record_pmax(ave_pmax, ave_pmax_denominator, combined_max_posterior.copy())
        history.record_pass2_engines(take_pass_engines())
        history.record_coarse_engines(take_coarse_engine_calls())

        # --- Track per-image best assignments for convergence detection ---
        # Combine both half-sets' assignments into a single array for
        # update_refinement_state.  Use coarse_ha (indexed into
        # effective_rotations) for consistent convergence tracking.
        current_combined_ha = concatenate_assignments(coarse_ha)
        previous_combined_ha = concatenate_assignments_or_none(previous_assignments)

        # tau2 was already updated BEFORE the Wiener solve (matching RELION's
        # reconstruct() which calls updateSSNRarrays before the filter).

        # --- Resolution from updated FSC-derived SSNR (RELION auto-refine) ---
        if k_class_enabled:
            current_combined_classes = concatenate_assignments(class_assignments)
            history.class_assignment_history.append(current_combined_classes.copy())
            previous_combined_classes = concatenate_assignments_or_none(previous_class_assignments)
            # K>1: data_vs_prior comes from the shared per-class prior and the
            # combined class accumulators.
            resolution_estimate = estimate_class_iteration_resolution(
                history.data_vs_prior_trajectory[-1],
                current_size=current_size,
                grid_size=grid_size,
                voxel_size=source_pixel_size_angstrom,
                emulate_relion_firstiter_cc=parity.emulate_relion_firstiter_cc,
                ini_high_angstrom=parity.relion_firstiter_ini_high_angstrom,
                relion_iteration=int(init_relion_iteration) + int(iteration) + 1,
                dtype=scoring_dtype,
            )
        else:
            current_combined_classes = None
            previous_combined_classes = None
            # K=1: data_vs_prior comes from the half-map FSC.
            resolution_estimate = estimate_k1_iteration_resolution(
                tau2_update_details["ssnr_shells"],
                current_size=current_size,
                grid_size=grid_size,
                voxel_size=source_pixel_size_angstrom,
                emulate_relion_firstiter_cc=parity.emulate_relion_firstiter_cc,
                ini_high_angstrom=parity.relion_firstiter_ini_high_angstrom,
                relion_iteration=int(init_relion_iteration) + int(iteration) + 1,
                dtype=scoring_dtype,
            )
        if int(resolution_estimate.scheduling_shell) != int(resolution_estimate.observed_shell):
            logger.info(
                "RELION firstiter_cc resolution state: using ini_high=%.2f A shell %d "
                "instead of live data-vs-prior shell %d",
                float(parity.relion_firstiter_ini_high_angstrom),
                int(resolution_estimate.scheduling_shell),
                int(resolution_estimate.observed_shell),
            )
        _tau2_debug_dump_dir = os.environ.get("RELAX_RELION_TAU2_DEBUG_DUMP_DIR")
        if _tau2_debug_dump_dir:
            reconstruction_diagnostics.write_tau2_update(
                _replay_meta=_replay_meta,
                output_dir=_tau2_debug_dump_dir,
                voxel_size=source_pixel_size_angstrom,
                current_size=current_size,
                dvp_iter=resolution_estimate.data_vs_prior,
                fsc=fsc,
                grid_size=grid_size,
                iteration=iteration,
                mstep_accumulator_shape=mstep_accumulator_shape,
                perturb_replay_relion_dir=perturb_replay_relion_dir,
                perturb_replay_relion_prefix=perturb_replay_relion_prefix,
                pixel_res=resolution_estimate.scheduling_shell,
                sealed_sampling_state=sealed_sampling_state,
                tau2_update_details=tau2_update_details,
                tau2_update_details_per_half=tau2_update_details_per_half,
                logger=logger,
            )
        history.pixel_resolutions.append(resolution_estimate.scheduling_shell)

        # --- Update poses and noise ---
        # Snapshot the iter K-1 best rotations / translations BEFORE the
        # loop overwrites them, so update_refinement_state below can compute
        # the RELION-exact change metrics (B3) between iter K-1 and iter K.
        pose_update = prepare_particle_pose_update(
            per_half,
            halves,
            current_translations,
            previous_rotations=previous_best_rotations,
            local_sampling=local_sampling if use_local else None,
            dtype=scoring_dtype,
        )
        previous_best_rotations = [poses.rotations for poses in pose_update.current]
        for half, poses in zip(halves, pose_update.current, strict=True):
            half.rotation_eulers = poses.eulers_deg
            half.translations = poses.translations_pixels
        history.record_pose_history(
            [np.asarray(poses.eulers_deg).copy() for poses in pose_update.current],
            [np.asarray(poses.translations_pixels).copy() for poses in pose_update.current],
        )
        if debug.save_intermediates_dir is not None:
            _save_iteration_particle_states(
                debug.save_intermediates_dir,
                iteration=iteration,
                rotation_matrices_per_half=[poses.rotations for poses in pose_update.current],
                rotation_eulers_deg_per_half=[poses.eulers_deg for poses in pose_update.current],
                relative_translations_pixels_per_half=[poses.relative_translations_pixels for poses in pose_update.current],
                absolute_translations_pixels_per_half=[poses.translations_pixels for poses in pose_update.current],
                max_posterior_per_half=max_posterior_per_half,
                significant_counts_per_half=significance.per_half,
                hard_assignments_per_half=hard_assignments,
                coarse_hard_assignments_per_half=coarse_ha,
                original_image_indices_per_half=[_source_image_indices(ds) for ds in experiment_datasets],
            )

        pose_comparison = prepare_pose_comparison(
            pose_update,
            translation_dimension=3 if tomo_halves else 2,
            dtype=scoring_dtype,
            log=logger,
        )

        if not k_class_enabled:
            history.data_vs_prior_trajectory.append(np.asarray(resolution_estimate.data_vs_prior, dtype=scoring_dtype))
            previous_data_vs_prior_for_scheduling = np.asarray(
                resolution_estimate.data_vs_prior,
                dtype=scoring_dtype,
            )

        # RELION-style posterior-weighted noise update. Helper folds the
        # K-class (shared) / K=1 (per-half) / firstiter_cc-skip variants;
        # returns updated radial sigma2_noise + the unrolled
        # ``noise_variance`` representation consumed by the engine.
        noise_debug_dump = partial(
            _maybe_dump_noise_update_debug,
            iteration=iteration,
            current_size=current_size,
            image_shape=image_geometry.image_shape,
        )
        noise_update = update_posterior_noise_variance(
            noise_stats_per_half,
            noise_model,
            image_geometry.image_shape,
            k_class_enabled=k_class_enabled,
            firstiter_cc=relion_firstiter_cc_this_iter,
            dump_debug=noise_debug_dump,
        )
        noise_from_res = noise_update.noise_from_res
        noise_from_res_per_half = noise_update.noise_from_res_per_half
        noise_model = noise_update.model
        if not relion_firstiter_cc_this_iter:
            _parity_dump.mark_stage(iteration, "noise_update")

        correction_report = NormScaleCorrectionReport()
        can_update_norm_scale = (
            noise_stats_per_half is not None
            and all(
                stats_k is not None
                and (
                    getattr(stats_k, "wsum_norm_correction", None) is not None
                    or int(experiment_datasets[_half_idx].n_units) == 0
                )
                for _half_idx, stats_k in enumerate(noise_stats_per_half)
            )
        )
        if follower_setup.follower_scale_state is not None and not can_update_norm_scale:
            raise RuntimeError(
                "Strict RELION follower-scale topology requires per-half norm/scale "
                "statistics at every numbered M-step"
            )
        if can_update_norm_scale:
            group_ids_per_half = [
                np.zeros(int(experiment_datasets[_half_idx].n_units), dtype=np.int64)
                if group_ids_k is None
                else group_ids_k
                for _half_idx, group_ids_k in enumerate([particle_half.group_ids for particle_half in halves])
            ]
            norm_scale_update = prepare_norm_scale_update(
                noise_stats_per_half,
                halves,
                group_ids_per_half=group_ids_per_half,
                firstiter_cc=relion_firstiter_cc_this_iter,
                do_norm_correction=not tomo_halves,
                do_scale_correction=follower_setup.follower_scale_state is None,
                dtype=scoring_dtype,
                iteration=iteration,
                current_size=current_size,
            )
            if follower_setup.follower_scale_state is None:
                for half, images, scales in zip(
                    halves,
                    norm_scale_update.image_corrections_per_half,
                    norm_scale_update.scale_corrections_per_half,
                    strict=True,
                ):
                    half.image_corrections = images
                    half.scale_corrections = scales
                correction_report.group_scale_corrections_per_half = norm_scale_update.group_scale_corrections_per_half
            else:
                correction_report.group_scale_corrections_per_half = _update_relion_follower_corrections(
                    follower_setup,
                    noise_stats_per_half=noise_stats_per_half,
                    norm_scale_update=norm_scale_update,
                    relion_half_inputs=halves,
                    relion_firstiter_cc_this_iter=relion_firstiter_cc_this_iter,
                    dtype=scoring_dtype,
                    logger=logger,
                )
            correction_report.norm_corrections_per_half = norm_scale_update.norm_corrections_per_half
            correction_report.avg_norm_correction_per_half = norm_scale_update.avg_norm_correction_per_half
            correction_report.zero_norm_residual_counts = norm_scale_update.zero_norm_residual_counts
            log_norm_scale_update(norm_scale_update, log=logger)
        if follower_setup.follower_scale_state is not None:
            history.relion_scale_follower_scales_numbered_post_mstep_trajectory.append(
                np.asarray(follower_setup.follower_scale_state.scales, dtype=np.float64).copy()
            )

        # Save per-iter per-shell sigma2 (after this iter's noise update) and
        # the exact shell-wise tau2 ingredients used in the Wiener update.
        history.record_noise_and_tau2(noise_from_res, noise_from_res_per_half, tau2_update_details)

        # --- Update convergence state ---
        # This checks assignment changes, resolution stalls, and may trigger
        # angular step refinement or convergence.
        state, accuracy_replay = update_iteration_convergence(
            state,
            pose_comparison,
            options,
            image_geometry=image_geometry,
            iteration=iteration,
            native_sampling_boundary=native_sampling_boundary,
            scheduling_resolution_shell=resolution_estimate.scheduling_shell,
            replay_dir=perturb_replay_relion_dir,
            translations=current_translations,
            current_assignments=current_combined_ha,
            previous_assignments=previous_combined_ha,
            current_classes=current_combined_classes,
            previous_classes=previous_combined_classes,
            max_posterior=combined_max_posterior,
            ave_pmax=ave_pmax,
            significant_counts=significance.convergence,
            exact_acc_rot=exact_acc_rot_this_iter,
            exact_acc_trans=exact_acc_trans_this_iter,
            log=logger,
        )
        iter_acc_rot = accuracy_replay.acc_rot
        iter_acc_trans = accuracy_replay.acc_trans

        # Reuse the assignment statistic computed by update_refinement_state.
        # Sampling transitions and optimiser replay preserve this field.
        history.frac_changed_trajectory.append(float(state.fraction_changed))

        # --- C1 (RELION-parity): update sigma2_offset from data ---
        # Posterior-weighted RELION update with fallback to hard-assignment
        # proxy; see ``update_c1_sigma_offset_from_posterior`` for details.
        sigma_offset_result = update_c1_sigma_offset_from_posterior(
            noise_stats_per_half=noise_stats_per_half,
            noise_stats_per_half_per_class=noise_stats_per_half_per_class,
            current_sigma_offset_angstrom_per_half=current_sigma_offset_angstrom_per_half,
            n_classes=n_classes,
            state_fallback_offsets_angstrom=state.current_changes_optimal_offsets_angstrom,
            offset_dims=3 if tomo_halves else 2,
        )
        current_sigma_offset_angstrom = sigma_offset_result.current_sigma_offset_angstrom
        current_sigma_offset_angstrom_per_half = _normalize_sigma_offset_per_half(
            sigma_offset_result.current_sigma_offset_angstrom_per_half
        )
        per_class_sigma_offset = sigma_offset_result.per_class_sigma_offset_angstrom
        history.record_sigma_offset_update(
            float(current_sigma_offset_angstrom),
            _copy_optional_float_pair(current_sigma_offset_angstrom_per_half),
            None if per_class_sigma_offset is None else per_class_sigma_offset.tolist(),
        )
        history.record_pose_accuracy_diagnostics(
            float(iter_acc_rot) if iter_acc_rot is not None else np.nan,
            float(iter_acc_trans) if iter_acc_trans is not None else np.nan,
            np.full(n_classes, np.nan, dtype=np.float64)
            if exact_acc_rot_per_class_this_iter is None
            else exact_acc_rot_per_class_this_iter,
            np.full(n_classes, np.nan, dtype=np.float64)
            if exact_acc_trans_per_class_this_iter is None
            else exact_acc_trans_per_class_this_iter,
            np.full(n_classes, -1, dtype=np.int64)
            if exact_accuracy_class_counts_this_iter is None
            else exact_accuracy_class_counts_this_iter,
            exact_accuracy_status_this_iter,
            float(state.current_changes_optimal_orientations),
            float(state.current_changes_optimal_offsets_angstrom),
        )

        # Save assignments for next iteration's change tracking.
        # Use coarse_ha (indexed into effective_rotations/base rotation grid)
        # so that local search and convergence detection work correctly
        # regardless of whether adaptive oversampling was used.
        previous_assignments = [ha.copy() if ha is not None else None for ha in coarse_ha]
        previous_class_assignments = [cls.copy() if cls is not None else None for cls in class_assignments]
        _parity_dump.mark_stage(iteration, "convergence")

        # --- RELION's run_itNNN files (ml_optimiser.cpp:3489) ---
        checkpoint_writer = options.checkpoint.writer
        if checkpoint_writer is not None and checkpoint_writer.due(numbered_relion_iteration):
            # The files hold incr_size/has_high_fsc_at_limit after this iteration's FSC
            # update, which the loop applies (idempotently) at the top of the next one.
            incr_size_after, high_fsc_after = relion_incr_size, relion_has_high_fsc_at_limit
            if not k_class_enabled:
                incr_size_after, high_fsc_after = update_relion_growth_state_from_fsc(
                    _zero_shells_past_current_size(
                        tau2_fsc_for_update,
                        current_size=current_size,
                        grid_size=grid_size,
                        dtype=scoring_dtype,
                    ),
                    current_size,
                    incr_size=relion_incr_size,
                    has_high_fsc_at_limit=relion_has_high_fsc_at_limit,
                )
            snapshot = snapshot_capture.begin(
                numbered_relion_iteration,
                state,
                sigma_offset_angstrom_per_half=current_sigma_offset_angstrom_per_half,
                current_size=current_size,
                incr_size=incr_size_after,
                has_high_fsc_at_limit=high_fsc_after,
                random_perturbation=random_perturbation,
                acc_rot_per_class=model_acc_rot_per_class,
                acc_trans_per_class_angstrom=model_acc_trans_per_class,
            )
            snapshot = snapshot_capture.finish(
                snapshot,
                reference_model.maps,
                unreg_means,
                (
                    mean_signal_variance_shells
                    if k_class_enabled
                    else [details["prior_shells"] for details in tau2_update_details_per_half]
                ),
                previous_data_vs_prior_for_scheduling,
                noise_model.radial_per_half,
                fsc=fsc,
                fsc_for_growth=None if k_class_enabled else tau2_fsc_for_update,
                class_weights=class_weights if k_class_enabled else None,
                direction_priors=direction_priors,
                half_inputs=halves,
                class_assignments=class_assignments if k_class_enabled else None,
                max_posterior=max_posterior_per_half,
                significant_counts=significance.per_half,
                avg_norm_correction=correction_report.avg_norm_correction_per_half,
            )
            checkpoint_writer(snapshot)

        if _parity_dump.is_active():
            dump_numbered_iteration(
                iteration,
                init_relion_iteration=init_relion_iteration,
                state=state,
                current_size=current_size,
                sigma_offset_angstrom=current_sigma_offset_angstrom,
                random_perturbation=random_perturbation,
                settings=reconstruction_settings,
                pixel_size_angstrom=source_pixel_size_angstrom,
                ave_pmax=ave_pmax,
                fsc=fsc,
                noise_variance=noise_model.average_variance,
                means=reference_model.maps,
                unfiltered_means=unreg_means,
                poses=pose_update.current,
                half_inputs=halves,
                corrections=correction_report,
                scale_correction_data_vs_prior=scale_correction_data_vs_prior_this_iter,
                log=logger,
            )
        elif _parity_dump.timing_is_active():
            try:
                _parity_dump.dump_timing_iteration(
                    iteration=iteration,
                    init_relion_iteration=int(init_relion_iteration),
                    iteration_start=t0,
                )
            except Exception as exc:
                logger.warning("parity_dump.dump_timing_iteration failed at iter %d: %s", iteration, exc)

        # --- Timing ---
        elapsed = time.time() - t0
        history.wall_times.append(elapsed)

        res_angstrom = shell_index_to_resolution_angstrom(
            resolution_estimate.scheduling_shell,
            image_geometry.image_shape[0],
            source_pixel_size_angstrom,
        )
        logger.info(
            "RELION Iteration %d: current_size=%d, pixel_res=%.1f, "
            "res=%.2f A, ave_Pmax=%.4f, healpix_order=%d, "
            "converged=%s, time=%.1fs",
            iteration + 1,
            current_size,
            resolution_estimate.scheduling_shell,
            res_angstrom,
            ave_pmax,
            state.healpix_order,
            state.has_converged,
            elapsed,
        )

        # End-of-iteration memory boundary.  The next iteration immediately
        # pads each half-map to the projection grid; keeping previous
        # backprojector accumulators or unregularized diagnostic maps live can
        # make high-resolution runs OOM before the batch-size estimator can act.
        try:
            jax.block_until_ready(reference_model.maps)
        except Exception:
            pass
        Ft_y_0 = Ft_y_1 = None
        Ft_ctf_0 = Ft_ctf_1 = None
        Ft_y_combined = Ft_ctf_combined = None
        unreg_means = previous_means = None
        mean_signal_variance_per_half = mean_signal_variance_shells_per_half = tau2_update_details_per_half = None
        noise_stats_per_half = noise_stats_per_half_per_class = None
        # Pass containers must not retain the previous grids while the next projector is built.
        numbered_expectation = numbered_tomo_sampling = numbered_variant = None
        if parse_env_true_flag("RELAX_RELION_CLEAR_JAX_CACHES_BETWEEN_ITERS"):
            jax.clear_caches()

        if state.has_converged and not schedule.force_max_iter_after_convergence:
            logger.info(
                "Convergence reached at iteration %d. Final resolution: %.2f A (pixel_res=%.1f)",
                iteration + 1,
                res_angstrom,
                resolution_estimate.scheduling_shell,
            )
            break
        if state.has_converged and schedule.force_max_iter_after_convergence:
            logger.info(
                "Convergence reached at iteration %d, continuing because force_max_iter_after_convergence=True",
                iteration + 1,
            )

        iteration += 1

    # Numbered contribution/native dump identities must never leak into the
    # return path or RELION's unnumbered final all-data pass.
    bpref_diagnostics.clear_bpref_contribution_dump_context()

    # RELION can enter final all-data only when checkConvergence() ran at the
    # top of a permitted loop iteration.  If the last numbered iteration
    # merely makes the state convergence-ready, ``iter <= nr_iter`` ends and
    # RELION does not synthesize another boundary after the cap.
    should_run_final_iteration = finalization._should_run_final_all_data_iteration(
        logger=logger,
        has_converged=state.has_converged,
        iteration=iteration,
        max_iter=schedule.max_iter,
        force_max_iter_after_convergence=schedule.force_max_iter_after_convergence,
        k_class_enabled=k_class_enabled,
    )
    if schedule.skip_final_iteration or not should_run_final_iteration:
        if not schedule.skip_final_iteration and not should_run_final_iteration:
            logger.info(
                "Skipping RELION final all-data iteration: has_converged=%s, "
                "iteration=%d, max_iter=%d, force_max_iter_after_convergence=%s",
                state.has_converged,
                iteration,
                schedule.max_iter,
                schedule.force_max_iter_after_convergence,
            )
        merged_mean, merged_class_means = _merged_mean_from_halves(
            reference_model.maps,
            class_weights if k_class_enabled else None,
        )
        (
            replay_requested_iterations,
            replay_applied_iterations,
        ) = _finalize_relion_follower_scale_replay_telemetry(
            replay.relion_follower_scale_replay,
            applied_iterations=history.relion_follower_scale_replay_applied_iterations,
            logger=logger,
        )
        return {
            **_model_result_fields(
                merged_mean, reference_model.maps, merged_class_means,
                class_weights if k_class_enabled else None,
                class_assignments if k_class_enabled else None,
            ),
            "relion_follower_scale_replay_requested_iterations": replay_requested_iterations,
            "relion_follower_scale_replay_applied_iterations": replay_applied_iterations,
            **follower_setup.to_result_dict(history),
            "convergence_state": state,
            **_numbered_result_metadata(
                hard_assignments, frozen_initial_scoring_state_sha256,
                expected_accuracy_trial_local_indices, expected_accuracy_trial_particle_ids,
                setup_phase_seconds,
            ),
            "final_all_data_ran": False,
            **history.to_dict(),
        }
    if not state.has_converged:
        logger.info(
            "Diagnostic %s=1: running RELION final all-data iteration after max_iter exhaustion "
            "(iteration=%d, max_iter=%d)",
            finalization._FINAL_ALL_DATA_AFTER_MAX_ITER_ENV,
            iteration,
            schedule.max_iter,
        )
    # --- RELION's final iteration: do_join_random_halves + do_use_all_data ---
    # After convergence, RELION runs ONE more iter with:
    #   - current_size = ori_size (Nyquist, all shells)
    #   - joined weighted sums for reconstruction
    #   - each half still scored against its own half-map
    # See ml_optimiser.cpp:10157-10160 (sets do_join_random_halves and
    # do_use_all_data) and ml_optimiser.cpp:5707-5708 (forces current_size to
    # ori_size when do_use_all_data is true).
    #
    # Implementation: run one more E+M at full Nyquist for each half, using
    # that half's own reference map, then join the weighted sums into one final
    # reconstruction.
    final_join_means = [reference_model.maps[0], reference_model.maps[1]]
    if not k_class_enabled and parse_env_true_flag(_FINAL_ALL_DATA_USE_MERGED_REFERENCE_ENV):
        final_merged_reference, _ = _merged_mean_from_halves(reference_model.maps)
        final_join_means = [final_merged_reference, final_merged_reference]
        logger.info(
            "Diagnostic %s=1: final all-data K=1 E-step uses merged reference for both halves",
            _FINAL_ALL_DATA_USE_MERGED_REFERENCE_ENV,
        )
    final_replay_forced = parse_env_true_flag(_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE_ENV)
    final_replay_disabled = parse_env_true_flag(_FINAL_ALL_DATA_DISABLE_REPLAY_LAST_NUMBERED_STATE_ENV)
    final_replay_has_overrides = replay.replay_iteration_overrides is not None and len(replay.replay_iteration_overrides) > 0
    final_replay_has_numbered_overrides = _has_numbered_replay_iteration_overrides(
        replay.replay_iteration_overrides
    )
    diagnostic_final_replay_override = final_replay_override
    final_join_means = replay_policy._prepare_final_replay_references(
        replay=replay,
        diagnostic_override=diagnostic_final_replay_override,
        numbered_iteration_count=len(history.current_sizes),
        means=reference_model.maps,
        final_join_means=final_join_means,
        k_class_enabled=k_class_enabled,
        logger=logger,
    )
    final_replay_last_numbered_state = (
        diagnostic_final_replay_override is not None
        or (
            not final_replay_disabled
            and (final_replay_forced or final_replay_has_numbered_overrides)
        )
    )
    if final_replay_last_numbered_state:
        final_replay_override_index, final_replay_override = replay_policy._select_final_replay_override(
            requested_index=len(history.current_sizes),
            diagnostic_override=diagnostic_final_replay_override,
            replay_overrides=replay.replay_iteration_overrides,
            has_overrides=final_replay_has_overrides,
            logger=logger,
        )
        if final_replay_override is not None:
            _final_replay_fields = []
            _final_replay_sigma_per_half = final_replay_override.get("translation_sigma_angstrom_per_half")
            if _final_replay_sigma_per_half is not None:
                current_sigma_offset_angstrom_per_half = _normalize_sigma_offset_per_half(
                    _final_replay_sigma_per_half
                )
                current_sigma_offset_angstrom = _mean_sigma_offset_per_half(
                    current_sigma_offset_angstrom_per_half
                )
                _final_replay_fields.append("translation_sigma_angstrom_per_half")
            _final_replay_sigma = final_replay_override.get("translation_sigma_angstrom")
            if _final_replay_sigma is not None and _final_replay_sigma_per_half is None:
                current_sigma_offset_angstrom = float(_final_replay_sigma)
                current_sigma_offset_angstrom_per_half = _as_sigma_offset_half_pair(
                    current_sigma_offset_angstrom
                )
                _final_replay_fields.append("translation_sigma_angstrom")
            _final_replay_prev_trans = final_replay_override.get("previous_best_translations")
            if _final_replay_prev_trans is not None:
                for particle_half, value_for_half in zip(halves, _copy_half_pair(_final_replay_prev_trans), strict=True):
                    particle_half.translations = value_for_half
                _final_replay_fields.append("previous_best_translations")
            _final_replay_prev_eulers = final_replay_override.get("previous_best_rotation_eulers")
            if _final_replay_prev_eulers is not None:
                for particle_half, value_for_half in zip(halves, _copy_half_pair(_final_replay_prev_eulers), strict=True):
                    particle_half.rotation_eulers = value_for_half
                _final_replay_fields.append("previous_best_rotation_eulers")
            _final_replay_fields.extend(
                _apply_replay_correction_overrides(
                    relion_half_inputs=halves,
                    replay_override=final_replay_override,
                )
            )
            _final_replay_noise = final_replay_override.get("noise_variance")
            if _final_replay_noise is not None:
                noise_model = noise_model_from_pixels(
                    _final_replay_noise, image_geometry.image_shape, dtype=scoring_dtype,
                )
                _final_replay_fields.append("noise_variance")
            _final_replay_dir_prior = final_replay_override.get("direction_prior")
            if _final_replay_dir_prior is not None:
                if k_class_enabled:
                    _final_replay_priors = normalize_class_direction_prior_per_half(
                        _final_replay_dir_prior,
                        n_classes,
                        dtype=scoring_dtype,
                    )
                else:
                    _final_replay_priors = normalize_direction_prior_per_half(
                        _final_replay_dir_prior, dtype=scoring_dtype
                    )
                for _half_idx in range(2):
                    if _final_replay_priors[_half_idx] is None:
                        continue
                    _prior_k = np.asarray(_final_replay_priors[_half_idx], dtype=scoring_dtype)
                    _prior_order_k = infer_direction_prior_healpix_order(
                        _prior_k[0] if k_class_enabled else _prior_k,
                        symmetry=symmetry, expected_order=state.healpix_order,
                    )
                    if _prior_order_k != state.healpix_order:
                        _prior_k = remap_half_direction_prior_to_healpix_order(
                            _prior_k,
                            _prior_order_k,
                            state.healpix_order,
                            n_classes=n_classes if k_class_enabled else None,
                            dtype=scoring_dtype,
                            symmetry=symmetry,
                        )
                        _prior_order_k = state.healpix_order
                    if k_class_enabled:
                        _prior_k = normalize_class_direction_prior(_prior_k, n_classes, dtype=scoring_dtype)
                    direction_priors[_half_idx] = DirectionPrior(_prior_k, _prior_order_k)
                _final_replay_fields.append("direction_prior")
            logger.info(
                "RELION replay: final all-data replays last numbered RELION state "
                "(previous_state_index=%d, fields=%s)",
                final_replay_override_index,
                ",".join(_final_replay_fields) if _final_replay_fields else "<none>",
            )
    elif not k_class_enabled and final_replay_disabled and final_replay_has_overrides:
        logger.info(
            "Diagnostic %s=1: final all-data skips automatic last-numbered RELION state replay",
            _FINAL_ALL_DATA_DISABLE_REPLAY_LAST_NUMBERED_STATE_ENV,
        )
    if follower_setup.follower_scale_state is not None:
        _dispatch_relion_follower_scale_for_final_all_data(
            follower_setup,
            init_relion_iteration=init_relion_iteration,
            numbered_iteration_count=len(history.current_sizes),
            relion_half_inputs=halves,
            dtype=scoring_dtype,
            logger=logger,
        )
    final_use_local = not k_class_enabled and state.do_local_search
    if final_use_local:
        for half in halves:
            half.require_local_search_poses()
    if not k_class_enabled:
        # RELION joins the half-set weighted sums after the post-convergence
        # expectation step.  During that E-step each MPI follower still owns
        # its numbered-iteration half model, so particles from random subset
        # 1 and 2 are scored with sigma2_noise from half 1 and 2 respectively.
        logger.info(
            "RELION final all-data: scoring each particle half with its own sigma2_noise",
        )
    final_result = finalization.run_final_all_data(
        halves,
        reference_model=reference_model,
        noise_model=noise_model,
        direction_priors=direction_priors,
        state=state,
        history=history,
        options=options,
        image_geometry=image_geometry,
        reconstruction_settings=reconstruction_settings,
        batch_planner=batch_planner,
        follower_setup=follower_setup,
        expected_accuracy_inputs=expected_accuracy_inputs,
        final_join_means=final_join_means,
        final_use_local=final_use_local,
        tomo_halves=tomo_halves,
        native_sampling_boundary=native_sampling_boundary,
        n_classes=n_classes,
        rotation_grid=current_rotation_grid,
        random_perturbation=random_perturbation,
        perturb_rng=perturb_rng,
        perturb_replay_relion_dir=perturb_replay_relion_dir,
        current_sigma_offset_angstrom=current_sigma_offset_angstrom,
        current_sigma_offset_angstrom_per_half=current_sigma_offset_angstrom_per_half,
        class_weights=class_weights,
        class_assignments=class_assignments,
        class_log_priors=class_log_priors,
        previous_data_vs_prior_for_scheduling=previous_data_vs_prior_for_scheduling,
        iteration=iteration,
        collect_local_search_profile=collect_local_search_profile,
        source_faithful_spectrum_norm=source_faithful_spectrum_norm,
        relion_translation_angle_scale=relion_translation_angle_scale,
        scoring_dtype=scoring_dtype,
        logger=logger,
    )
    # Setup and numbered-iteration metadata retain their existing caller ownership.
    final_result.update(_numbered_result_metadata(
        hard_assignments, frozen_initial_scoring_state_sha256,
        expected_accuracy_trial_local_indices, expected_accuracy_trial_particle_ids,
        setup_phase_seconds,
    ))
    return final_result

### THIS FILE SHOULD BE MUCH SHORTER - REMOVE UN-NECESSARY IF STATEMENTS/RELION THINGS THAT DONT MATTER/ WE DONT USE
## MOVE THINGS AWAY TO DIFFERNET FILE E.G. I/O PERHAPS. IT SHOULD BE EASY TO UDNERSTAND WHERE THE MAIN ENGINE OF ITER IS GOING, WHAT ARE THE MAIN STEPS (E-M ACCUMULATION) POSTPROCESSING, NOISE UPDATING, ETC
