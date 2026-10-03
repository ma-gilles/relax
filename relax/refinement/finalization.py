"""Final all-data admission, expectation and reconstruction.

The numbered controller owns convergence and replay. This module owns the
post-convergence sampling/scoring/reconstruction sequence and its result.
"""


import os
import time

import jax.numpy as jnp
import numpy as np

from relax import sampling
from relax.dense.score_outputs import (
    PerHalfOutputs,
    _record_score_profile,
    _resolve_mstep_accumulator_shape,
    _resolve_mstep_full_half_axis,
)
from relax.dense.scoring_policy import (
    _local_adaptive_pass2_denominator_support_mode,
    _local_adaptive_pass2_full_parent_enabled,
    _local_adaptive_pass2_rotation_only_enabled,
    local_precision,
)
from relax.diagnostics import bpref_diagnostics
from relax.diagnostics.iteration import _replay_manifest_array
from relax.helpers.convergence import healpix_angular_step, update_angular_sampling
from relax.helpers.dtype_policy import _diagnostic_float64_pass2_matches
from relax.helpers.env_flags import parse_env_flag_or_false
from relax.helpers.expected_accuracy import _expected_accuracy_class_ids
from relax.helpers.orientation_priors import relion_local_search_sigmas, relion_translation_search_base
from relax.helpers.resolution import (
    clamp_relion_coarse_image_size,
    compute_coarse_image_size,
    relion_current_resolution_shell,
    shell_index_to_resolution_angstrom,
)
from relax.refinement import final_reconstruction
from relax.refinement.expectation import prepare_final_half
from relax.refinement.final_sampling import prepare_final_sampling
from relax.refinement.half_inputs import _sigma_offset_for_half
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
from relax.refinement.local_sampling import LocalSearchSettings, prepare_final_local_sampling
from relax.refinement.mean_helpers import _class_weights_from_posterior, join_half_accumulators_at_low_resolution
from relax.refinement.projector_preparation import prepare_scoring_projector
from relax.refinement.result_files import (
    _model_result_fields,
    final_pass_result_fields,
)
from relax.refinement.tomo_half import local_tomo_sampling
from relax.refinement.tomo_half import score_tomo_half_in_loop as _score_tomo_half_in_loop
from relax.relion.geometry import PROJECTION_PADDING_FACTOR, RECONSTRUCTION_PADDING_FACTOR
from relax.relion.relion_metadata import _relion_metadata_translations
from relax.relion.relion_worker_scale import _finalize_relion_follower_scale_replay_telemetry

_FINAL_ALL_DATA_AFTER_MAX_ITER_ENV = "RELAX_FINAL_ALL_DATA_AFTER_MAX_ITER"


def _should_run_final_all_data_iteration(
    *,
    logger,
    has_converged: bool,
    iteration: int,
    max_iter: int,
    force_max_iter_after_convergence: bool,
    k_class_enabled: bool = False,
) -> bool:
    """Return whether to run RELION's final all-data reconstruction pass."""

    if force_max_iter_after_convergence:
        return False
    if bool(has_converged):
        return True
    if not (
        parse_env_flag_or_false(_FINAL_ALL_DATA_AFTER_MAX_ITER_ENV, logger=logger)
        and int(iteration) >= int(max_iter)
    ):
        return False
    if bool(k_class_enabled):
        logger.warning(
            "Ignoring %s=1 for K-class after max_iter exhaustion; final all-data "
            "is only valid for K-class after convergence",
            _FINAL_ALL_DATA_AFTER_MAX_ITER_ENV,
        )
        return False
    return True


def run_final_all_data(
    halves,
    *,
    reference_model,
    noise_model,
    direction_priors,
    state,
    history,
    options,
    image_geometry,
    reconstruction_settings,
    batch_planner,
    follower_setup,
    expected_accuracy_inputs,
    final_join_means,
    final_use_local,
    tomo_halves,
    native_sampling_boundary,
    n_classes,
    rotation_grid: sampling.RotationGrid,
    random_perturbation,
    perturb_rng,
    perturb_replay_relion_dir,
    current_sigma_offset_angstrom,
    current_sigma_offset_angstrom_per_half,
    class_weights,
    class_assignments,
    class_log_priors,
    previous_data_vs_prior_for_scheduling,
    iteration,
    collect_local_search_profile,
    source_faithful_spectrum_norm,
    relion_translation_angle_scale,
    scoring_dtype,
    logger,
) -> dict:
    """Score the converged halves at full size and reconstruct the final maps.

    Replay and final-pass admission are resolved by the numbered controller.
    The returned mapping contains this phase's results; the caller publishes
    setup and numbered-iteration metadata after execution.
    See ``docs/math/relion_refinement_algorithm.md``, section 7.
    """
    parity = options.parity
    adaptive = options.adaptive
    batching = options.batching
    debug = options.debug
    replay = options.replay
    local_search = options.local_search
    symmetry = options.symmetry.point_group
    init_relion_iteration = options.schedule.init_relion_iteration
    particle_diameter_ang = options.schedule.particle_diameter_ang
    sealed_sampling_state = debug.sealed_sampling_state
    k_class_enabled = n_classes > 1
    grid_size = image_geometry.box_size
    volume_shape = reconstruction_settings.volume_shape
    padded_volume_shape = tuple(d * RECONSTRUCTION_PADDING_FACTOR for d in volume_shape)
    final_expected_accuracy = None
    final_expected_accuracy_status = "not_run"
    final_iter_t0 = time.time()
    final_current_size = image_geometry.box_size
    if native_sampling_boundary:
        final_eulers_half1 = halves[0].rotation_eulers
        if expected_accuracy_inputs.trial_order_local is None or final_eulers_half1 is None:
            final_expected_accuracy_status = "unavailable_inputs"
            state.acc_rot = float("inf")
            state.acc_trans = float("inf")
            logger.warning(
                "RELION final all-data expected accuracy unavailable; "
                "final expectation remains fail-closed",
            )
        else:
            final_accuracy_class_ids = _expected_accuracy_class_ids(
                class_assignments[0],
                k_class_enabled=k_class_enabled,
                n_units=expected_accuracy_inputs.dataset.n_units,
            )
            try:
                final_expected_accuracy = expected_accuracy_inputs.estimate(
                    reference_fourier=final_join_means[0],
                    best_eulers_deg=final_eulers_half1,
                    class_ids=final_accuracy_class_ids,
                    class_weights=class_weights,
                    sigma2_noise_native=noise_model.radial_per_half[0],
                    current_image_size=final_current_size,
                )
                state.acc_rot = final_expected_accuracy.acc_rot
                state.acc_trans = final_expected_accuracy.acc_trans_angstrom
                state = update_angular_sampling(state)
                final_expected_accuracy_status = "ok"
                logger.info(
                    "RELION final all-data expected accuracy: acc_rot=%.3f deg, "
                    "acc_trans=%.4f A",
                    state.acc_rot,
                    state.acc_trans,
                )
            except Exception as exc:
                final_expected_accuracy_status = f"error:{type(exc).__name__}:{exc}"
                state.acc_rot = float("inf")
                state.acc_trans = float("inf")
                logger.warning(
                    "RELION final all-data expected-accuracy estimation failed: %s",
                    exc,
                )
    final_sampling = prepare_final_sampling(
        state,
        image_geometry,
        previous_rotation_grid=rotation_grid,
        last_numbered_iteration=init_relion_iteration + len(history.current_sizes),
        parity=parity,
        active_replay_dir=perturb_replay_relion_dir,
        require_final_state=replay.replay_iteration_overrides is not None,
        previous_perturbation=random_perturbation,
        rng=perturb_rng,
        n_classes=n_classes,
        dtype=scoring_dtype,
    )
    final_precision = local_precision(final_sampling.settings.relion_iteration, pass_index=2)
    ## THIS ALL SHOULD BE AN OBJECT/DICT/ENUM OR SOMETHING PROBABLY WITH THESE DECISIONS OR SOMETHING? 
    ## SHOULD BE WANTING TO DEFINE 100 THINGS LIKE THIS
    if final_use_local:
        final_sigma_rot, final_sigma_psi = relion_local_search_sigmas(
            state.sigma_rot,
            state.sigma_psi,
            use_local=True,
            healpix_order=state.healpix_order,
            adaptive_oversampling=state.adaptive_oversampling,
        )
        final_search = LocalSearchSettings(
            healpix_order=final_sampling.settings.perturbation_order + state.adaptive_oversampling,
            oversampling_order=state.adaptive_oversampling,
            sigma_rot=final_sigma_rot,
            sigma_psi=final_sigma_psi,
            symmetry=symmetry,
        )
        final_local_sampling = prepare_final_local_sampling(
            final_search,
            image_geometry,
            translations=final_sampling.grid.translations,
            base_translations=final_sampling.base_translations,
            image_window_size=final_current_size,
            particle_diameter_angstrom=particle_diameter_ang,
            perturbation=final_sampling.settings.perturbation,
            rotation_dtype=final_precision.rotation_real_dtype,
        )
        logger.info(
            "RELION final all-data iteration using local search: parent_order=%d fine_order=%d, "
            "sigma_rot=%.4f rad (%.2f deg), sigma_psi=%.4f rad, perturbation=%+.5f",
            final_sampling.settings.perturbation_order,
            final_local_sampling.search.healpix_order,
            final_sigma_rot,
            np.rad2deg(final_sigma_rot),
            final_sigma_psi,
            final_sampling.settings.random_perturbation,
        )
    else:
        logger.info(
            "RELION final all-data iteration using dense global scoring: healpix_order=%d, perturbation=%+.5f",
            final_sampling.settings.grid_order,
            final_sampling.settings.random_perturbation,
        )
        if k_class_enabled and state.adaptive_oversampling > 0:
            final_coarse_size = compute_coarse_image_size(
                healpix_angular_step(final_sampling.settings.grid_order),
                image_geometry.pixel_size_angstrom,
                image_geometry.box_size,
                particle_diameter=particle_diameter_ang,
            )
            final_adaptive_pass1_current_size = clamp_relion_coarse_image_size(
                final_coarse_size,
                final_current_size,
                image_geometry.box_size,
            )
            final_adaptive_pass2_current_size = final_current_size
            logger.info(
                "RELION final all-data adaptive K-class pass-1: coarse_current_size=%d "
                "fine_current_size=%d oversampling=%d",
                final_adaptive_pass1_current_size,
                final_adaptive_pass2_current_size,
                state.adaptive_oversampling,
            )
        else:
            final_adaptive_pass1_current_size = None
            final_adaptive_pass2_current_size = None
    # Pass 1 scores RELION's exact coarse operands on every route, so the final pass builds the projector too.
    projector_t0 = time.time()
    final_projectors = [
        prepare_scoring_projector(
            reference,
            volume_shape=volume_shape,
            current_size=final_current_size,
            padding_factor=PROJECTION_PADDING_FACTOR,
            n_classes=n_classes,
            dump_label=f"final_half{half.index}",
        )
        for half, reference in zip(halves, final_join_means, strict=True)
    ]
    logger.info(
        "RELION final all-data: built exact Projector::data for scoring at current_size=%d r_max=%s in %.2fs",
        final_current_size,
        final_projectors[0].r_max,
        time.time() - projector_t0,
    )
    logger.info("=== RELION final all-data Nyquist iteration ===")
    if final_use_local:
        if state.adaptive_oversampling > 0:
            final_local_adaptive_full_parent = _local_adaptive_pass2_full_parent_enabled()
            final_local_adaptive_rotation_only = _local_adaptive_pass2_rotation_only_enabled()
            final_local_adaptive_denominator_mode = (
                _local_adaptive_pass2_denominator_support_mode()
            )
        else:
            final_local_adaptive_full_parent = False
            final_local_adaptive_rotation_only = False
            final_local_adaptive_denominator_mode = None
        final_local_batching = LocalBatchPolicy(
            max_significants=adaptive.max_significants,
            safe_batch_sizes=batch_planner,
        )
        final_local_execution = LocalExecutionPolicy(
            disc_type=options.disc_type,
            disable_adjoint_y=debug.disable_adjoint_y,
            disable_adjoint_ctf=debug.disable_adjoint_ctf,
            relion_translation_angle_scale=relion_translation_angle_scale,
        )
        final_local_diagnostics = LocalDiagnosticPolicy(
            iteration=iteration + 1,
            debug_iteration=final_sampling.settings.relion_iteration,
            save_intermediates_dir=debug.save_intermediates_dir,
            collect_local_search_profile=collect_local_search_profile,
            diagnostic_score_only=False,
            local_profile_history=history.local_profile_history,
            adaptive_pass2_full_parent=final_local_adaptive_full_parent,
            adaptive_pass2_rotation_only=final_local_adaptive_rotation_only,
            adaptive_pass2_denominator_mode=final_local_adaptive_denominator_mode,
        )
    else:
        final_dense_sampling = DenseSamplingSpec(
            effective_rotations=final_sampling.grid.rotations,
            current_translations=final_sampling.grid.translations,
            base_translations=final_sampling.base_translations,
            current_healpix_order=final_sampling.settings.grid_order,
            oversampling_order=state.adaptive_oversampling,
            translation_step=state.translation_step,
            coarse_engine=adaptive.coarse_engine,
            random_perturbation=final_sampling.settings.random_perturbation,
            cs_for_engine=final_current_size,
            symmetry=symmetry,
        )
        final_dense_batching = DenseBatchPolicy(
            image_batch_size=batching.image_batch_size,
            safe_batch_sizes=batch_planner,
            max_significants=adaptive.max_significants,
        )
        final_dense_variant = DenseVariantPolicy(
            firstiter_score_mode_this_iter="gaussian",
            firstiter_winner_take_all_this_iter=False,
            k_class_enabled=k_class_enabled,
            relion_firstiter_cc_this_iter=False,
            firstiter_coarse_current_size=final_adaptive_pass1_current_size,
            firstiter_fine_current_size=final_adaptive_pass2_current_size,
            firstiter_log_label="final all-data ",
            firstiter_updates_em_kwargs_ibs=True,
        )
        final_dense_execution = DenseExecutionPolicy(
            disc_type=options.disc_type,
            disable_adjoint_y=debug.disable_adjoint_y,
            disable_adjoint_ctf=debug.disable_adjoint_ctf,
            return_best_pose_details=not k_class_enabled,
            debug_iteration=final_sampling.settings.relion_iteration,
            diagnostic_float64_pass2=_diagnostic_float64_pass2_matches(
                final_sampling.settings.relion_iteration
            ),
            preserve_bpref_particle_order=parity.preserve_bpref_particle_order,
            source_faithful_spectrum_norm=source_faithful_spectrum_norm,
            relion_translation_angle_scale=relion_translation_angle_scale,
            firstiter_cc_tree_rescore_max_margin=parity.firstiter_cc_tree_rescore_max_margin,
        )
    final_outs = PerHalfOutputs()
    for half, projector in zip(halves, final_projectors, strict=True):
        bpref_diagnostics.clear_bpref_contribution_dump_context()
        final_half_t0 = time.time()
        logger.info(
            "BPREF_DEVICE_SIGNATURE_ACTIVATION iteration=%d half=%d "
            "final_all_data=true active=false",
            iteration + 1,
            half.index + 1,
        )
        logger.info(
            "RELION final all-data half-%d start: images=%d current_size=%d "
            "healpix_order=%d n_rot=%d n_trans=%d local_search=%s",
            half.index + 1,
            half.dataset.n_units,
            final_current_size,
            final_sampling.settings.grid_order,
            final_sampling.base_rotations.shape[0],
            final_sampling.grid.translations.shape[0],
            final_use_local,
        )
        translation_search_base = relion_translation_search_base(
            half.translations, dtype=scoring_dtype
        )
        final_outs.translation_search_bases[half.index] = translation_search_base
        final_sigma_offset_k = _sigma_offset_for_half(
            current_sigma_offset_angstrom,
            current_sigma_offset_angstrom_per_half,
            half.index,
        )
        if tomo_halves:
            # Subtomograms: the tomo half pass at the final sampling (RELION's local search on the
            # previous poses), with the merged reference for both halves.
            if not final_use_local:
                raise NotImplementedError(
                    "a subtomogram final all-data pass without local search needs the global direction priors"
                )
            final_result = _score_tomo_half_in_loop(
                half.dataset,
                use_local=True,
                use_adaptive=final_local_sampling.search.oversampling_order > 0,
                volume=final_join_means[half.index],
                noise_variance=noise_model.variance_per_half[half.index],
                relion_projector_half=projector.data,
                relion_projector_r_max=projector.r_max,
                sampling=local_tomo_sampling(
                    fine_order=final_local_sampling.search.healpix_order,
                    oversampling_order=final_local_sampling.search.oversampling_order,
                    translation_range_px=final_sampling.settings.translation_range,
                    translation_step_px=final_sampling.settings.translation_step,
                    voxel_size=image_geometry.pixel_size_angstrom,
                    random_perturbation=final_local_sampling.perturbation,
                    pass1_size=final_local_sampling.coarse_image_window_size,
                    current_size=final_current_size,
                ),
                local_search=dict(
                    previous_eulers_deg=half.rotation_eulers,
                    sigma_rot=final_local_sampling.search.sigma_rot,
                    sigma_psi=final_local_sampling.search.sigma_psi,
                ),
                rotation_log_prior=None,
                previous_translations=half.translations,
                sigma_offset_angst=final_sigma_offset_k,
                max_significants=adaptive.max_significants,
                unit_groups=half.optics_group_ids,
                scale_corrections=half.scale_corrections,
                group_ids=follower_setup.scale_stats_group_ids_per_half[half.index],
                scale_correction_group_count=follower_setup.scale_stats_group_count_per_half[half.index],
                scale_correction_data_vs_prior=previous_data_vs_prior_for_scheduling,
                reconstruction_current_size=final_current_size,
                symmetry=symmetry,
            )
        else:
            # The final pass scores each half with its own priors on the grid rows it
            # actually uses; sealed rows apply only when the sealed grid is reused.
            final_inputs = prepare_final_half(
                half,
                final_sampling,
                image_geometry=image_geometry,
                sigma_offset_angstrom=final_sigma_offset_k,
                noise_radial=noise_model.radial_per_half[half.index],
                direction_priors=direction_priors[half.index],
                n_classes=n_classes,
                use_local=final_use_local,
                coarse_angular_step_deg=final_local_sampling.coarse_angular_step_deg if final_use_local else None,
                particle_diameter_angstrom=particle_diameter_ang,
                sealed_sampling_state=(
                    sealed_sampling_state if final_sampling.base_rotations is rotation_grid.rotations else None
                ),
                symmetry=symmetry,
                dtype=scoring_dtype,
            )
            scoring_half = HalfScoringData(
                particles=half,
                projector=projector,
                reference=final_join_means[half.index],
                mean_variance=reference_model.tau2,
                noise_variance=noise_model.variance_per_half[half.index],
                scale_group_ids=follower_setup.scale_stats_group_ids_per_half[half.index],
                scale_group_count=follower_setup.scale_stats_group_count_per_half[half.index],
                scale_correction_data_vs_prior=previous_data_vs_prior_for_scheduling,
            )
            if final_use_local:
                final_result = _score_half_local_in_bpref_scope(
                    half=scoring_half,
                    sampling=final_local_sampling,
                    priors=LocalPriorSpec(
                        trans_prior_center=final_inputs.translations.local_prior_center,
                        trans_prior_center_for_engine=final_inputs.translations.engine_prior_center,
                        current_sigma_offset_angstrom=final_sigma_offset_k,
                        translation_search_base=translation_search_base,
                        local_search_translation_prior_mode=local_search.local_search_translation_prior_mode,
                    ),
                    batching=final_local_batching,
                    execution=final_local_execution,
                    diagnostics=final_local_diagnostics,
                    optics=final_inputs.optics,
                )
            else:
                final_result = _score_half_dense_in_bpref_scope(
                    half=scoring_half,
                    sampling=final_dense_sampling,
                    priors=DensePriorSpec(
                        rotation_log_prior_k=final_inputs.directions.rotation_log_prior,
                        class_rotation_log_prior_k=final_inputs.directions.class_rotation_log_prior,
                        translation_log_prior=final_inputs.translation_log_prior,
                        translation_search_base=translation_search_base,
                        trans_prior_center_for_engine=final_inputs.translations.engine_prior_center,
                        class_log_priors=class_log_priors,
                    ),
                    batching=final_dense_batching,
                    variant=final_dense_variant,
                    execution=final_dense_execution,
                    optics=final_inputs.optics,
                )
        if final_result.best_pose_translations is not None:
            final_result.best_pose_translations = _relion_metadata_translations(
                half.translations,
                final_result.best_pose_translations,
                dtype=scoring_dtype,
            )
        final_outs.update_from(half.index, final_result, dtype=scoring_dtype)
        _record_score_profile(
            history.global_profile_history,
            final_result,
            phase="final_all_data",
            iteration=iteration + 1,
            relion_iteration=final_sampling.settings.relion_iteration,
            half_index=half.index,
            current_size=final_current_size,
            healpix_order=final_sampling.settings.grid_order,
            k_class_enabled=k_class_enabled,
        )
        logger.info(
            "RELION final all-data half-%d done: wall=%.1fs",
            half.index + 1,
            time.time() - final_half_t0,
        )
        # --- Manifest dump for final all-data iteration (Phase 0.1) ---
        if debug.save_intermediates_dir is not None:
            _manifest_path = os.path.join(
                debug.save_intermediates_dir,
                f"manifest_final_half{half.index}.npz",
            )
            _manifest = {
                "effective_rotations": np.asarray(final_sampling.grid.rotations, dtype=np.float32),
                "current_translations": np.asarray(final_sampling.grid.translations, dtype=np.float32),
                "rotation_log_prior": _replay_manifest_array(final_inputs.directions.rotation_log_prior, dtype=np.float64),
                "translation_log_prior": np.asarray(final_inputs.translation_log_prior, dtype=np.float64),
                "translation_prior_centers": np.asarray(final_inputs.translations.engine_prior_center, dtype=np.float64),
                "image_corrections": _replay_manifest_array(half.image_corrections, dtype=np.float64),
                "scale_corrections": _replay_manifest_array(half.scale_corrections, dtype=np.float64),
                "image_pre_shifts": _replay_manifest_array(translation_search_base, dtype=np.float32),
                "absolute_previous_translations": _replay_manifest_array(
                    half.translations, dtype=np.float32,
                ),
                "mean_vol_ft": np.asarray(final_join_means[half.index]),
                "mean_variance": np.asarray(reference_model.tau2),
                "noise_variance": np.asarray(noise_model.variance_per_half[half.index]),
                "current_size": np.int32(final_current_size),
                "half_spectrum_scoring": np.bool_(True),
                "use_float64_scoring": np.bool_(final_precision.use_float64_scoring),
                "use_float64_projections": np.bool_(final_precision.use_float64_projections),
                "projection_padding_factor": np.int32(PROJECTION_PADDING_FACTOR),
                "reconstruction_padding_factor": np.int32(RECONSTRUCTION_PADDING_FACTOR),
                "score_with_masked_images": np.bool_(True),
                "perturbation_instance": np.float64(final_sampling.settings.random_perturbation),
                "perturbation_factor": np.float64(final_sampling.settings.perturbation_factor),
                "perturbation_applied": np.bool_(final_sampling.settings.perturbation is not None),
                "perturbation_relion_iteration": np.int32(final_sampling.settings.relion_iteration),
                "local_search": np.bool_(final_use_local),
                "iteration": np.int32(-1),
                "half_index": np.int32(half.index),
            }
            np.savez(_manifest_path, **_manifest)
            logger.info("Final manifest dumped: %s", _manifest_path)

    final_Ft_y_0 = final_outs.Ft_y[0]
    final_Ft_y_1 = final_outs.Ft_y[1]
    final_Ft_ctf_0 = final_outs.Ft_ctf[0]
    final_Ft_ctf_1 = final_outs.Ft_ctf[1]
    final_mstep_accumulator_shape = _resolve_mstep_accumulator_shape(
        final_outs.mstep_accumulator_shape,
        padded_volume_shape,
    )
    final_reconstruct_t0 = time.time()
    final_unfiltered_means_for_output = None
    if not k_class_enabled:
        # K1 pre-join sequence, in this order: unfiltered half maps, optional
        # low-resolution join, release of the pass outputs' references. Class3D
        # has no half maps to keep unjoined and sums its two partitions below.
        #
        # RELION writes run_half{1,2}_class001_unfil.mrc from the converged half
        # BackProjectors saved before joinTwoHalvesAtLowResolution mutates their
        # low-frequency voxels, with do_map=false.  Keep this separate from the
        # Wiener-regularized half maps below so parity audits compare like
        # products.  Default RELION refinement sets BackProjector's
        # skip_gridding=true, so reconstruct uses the radial denominator floor
        # and direct division path before its final real-space gridding
        # correction and always applies softMaskOutsideMap inside
        # windowToOridimRealSpace; do_map=false only omits the tau2 prior.
        # They are reconstructed first, from the pre-join arrays, so the join
        # below can update those arrays in place instead of copying them: at
        # EMPIAR-10202's full box a copy of both halves' accumulators is 49 GB
        # of host memory (bigbox 14592943, 14594535). Each map is kept on the
        # host so the device holds only the reconstruction in progress.
        final_unfiltered_means_for_output = final_reconstruction.reconstruct_unfiltered_halfmaps(
            (final_Ft_y_0, final_Ft_y_1),
            (final_Ft_ctf_0, final_Ft_ctf_1),
            settings=reconstruction_settings,
            current_size=final_current_size,
            accumulator_shape=final_mstep_accumulator_shape,
        )
        if parity.low_resol_join_halves_angstrom is not None and parity.low_resol_join_halves_angstrom > 0:
            final_Ft_y_0, final_Ft_y_1, final_Ft_ctf_0, final_Ft_ctf_1 = join_half_accumulators_at_low_resolution(
                (final_Ft_y_0, final_Ft_y_1),
                (final_Ft_ctf_0, final_Ft_ctf_1),
                accumulator_volume_shape=final_mstep_accumulator_shape,
                grid_size=grid_size,
                voxel_size=image_geometry.pixel_size_angstrom,
                padding_factor=RECONSTRUCTION_PADDING_FACTOR,
                low_resolution_angstrom=parity.low_resol_join_halves_angstrom,
                pixel_resolutions=history.pixel_resolutions,
                current_resolution=state.current_resolution,
                preserve_inputs=False,
            )
        # The unfiltered maps are made; drop the pass outputs' references so the
        # pre-join accumulators live only as long as the joined ones do.
        final_outs.Ft_y[0] = final_outs.Ft_y[1] = None
        final_outs.Ft_ctf[0] = final_outs.Ft_ctf[1] = None

    final_ft_y = final_Ft_y_0 + final_Ft_y_1
    final_ft_ctf = final_Ft_ctf_0 + final_Ft_ctf_1
    final_iter_fsc = None
    final_mstep_full_half_axis = _resolve_mstep_full_half_axis(
        final_outs.mstep_full_half_axis,
        default_axis=-1,
    )
    if k_class_enabled:
        class_weights = _class_weights_from_posterior(
            final_outs.class_posterior,
            n_classes,
            class_weights,
        )
        history.record_class_weights(
            class_weights,
            _class_weights_from_posterior(
                final_outs.class_full_posterior,
                n_classes,
                class_weights,
            ),
        )
        _t_final_tau2 = time.time()
        final_class_priors = final_reconstruction.compute_final_class_priors(
            final_ft_ctf,
            final_join_means[0],
            projector=final_projectors[0],
            n_classes=n_classes,
            settings=reconstruction_settings,
            current_size=final_current_size,
            accumulator_shape=final_mstep_accumulator_shape,
            full_half_axis=final_mstep_full_half_axis,
        )
        final_mean_variance = final_class_priors.variance
        final_mean_variance_shells = final_class_priors.shells
        final_data_vs_prior = final_class_priors.data_vs_prior
        final_tau2_update_details = final_class_priors.details
        logger.info(
            "RELION final all-data Class3D tau2 from Iref power spectra: old_max=%.4e new_max=%.4e "
            "dvp_shell_1=%.4f wall=%.1fs",
            float(jnp.max(jnp.abs(reference_model.tau2))),
            float(jnp.max(jnp.abs(final_mean_variance))),
            float(np.asarray(final_data_vs_prior)[0, 1]) if np.asarray(final_data_vs_prior).shape[-1] > 1 else float("nan"),
            time.time() - _t_final_tau2,
        )
    else:
        _t_final_tau2 = time.time()
        final_halfmap_prior = final_reconstruction.compute_final_halfmap_prior(
            (final_Ft_y_0, final_Ft_y_1),
            (final_Ft_ctf_0, final_Ft_ctf_1),
            settings=reconstruction_settings,
            current_size=final_current_size,
            accumulator_shape=final_mstep_accumulator_shape,
            full_half_axis=final_mstep_full_half_axis,
            scoring_dtype=scoring_dtype,
        )
        final_iter_fsc = final_halfmap_prior.fsc
        final_mean_variance = final_halfmap_prior.variance
        final_tau2_update_details = final_halfmap_prior.details
        logger.info(
            "RELION final all-data tau2 from joined FSC: old_max=%.4e new_max=%.4e "
            "fsc_shell_1=%.4f wall=%.1fs",
            float(jnp.max(jnp.abs(reference_model.tau2))),
            float(jnp.max(jnp.abs(final_mean_variance))),
            float(np.asarray(final_iter_fsc)[1]) if np.asarray(final_iter_fsc).size > 1 else float("nan"),
            time.time() - _t_final_tau2,
        )

    # RELION calls updateCurrentResolution after the final all-data
    # iteration too (ml_optimiser_mpi.cpp:4329), from that iteration's
    # whole-data DVP, so rlnCurrentResolution reports the final half-map FSC
    # at 0.143 rather than the last split-half iteration's 0.5 crossing.
    # Nothing after this point schedules on the resolution.
    final_dvp = (
        final_data_vs_prior
        if k_class_enabled
        else np.asarray(final_tau2_update_details["ssnr_shells"], dtype=scoring_dtype)
    )
    final_res_shell = relion_current_resolution_shell(
        final_dvp,
        k_class_enabled=k_class_enabled,
        current_size=final_current_size,
        grid_size=grid_size,
        dtype=scoring_dtype,
    )
    state.previous_resolution = state.current_resolution
    state.current_resolution = shell_index_to_resolution_angstrom(
        final_res_shell, image_geometry.box_size, image_geometry.pixel_size_angstrom
    )
    logger.info(
        "RELION final all-data current resolution: shell=%d res=%.2f A (last split-half iteration %.2f A)",
        int(final_res_shell),
        state.current_resolution,
        state.previous_resolution,
    )

    # Reconstruct the final volume from the COMBINED Ft_y/Ft_ctf accumulators
    # at the full Nyquist resolution. Skip the join_halves step (we're already
    # combining the two halves into one dataset for this final iter).
    logger.info(
        "RELION final all-data reconstruction start: current_size=%d n_classes=%d",
        final_current_size,
        n_classes,
    )
    if k_class_enabled:
        final_maps = final_reconstruction.reconstruct_final_class_maps(
            final_ft_y,
            final_ft_ctf,
            final_mean_variance_shells,
            class_weights=class_weights,
            n_classes=n_classes,
            settings=reconstruction_settings,
            current_size=final_current_size,
            accumulator_shape=final_mstep_accumulator_shape,
        )
        final_class_means = final_maps.halves[0]
        class_assignments = final_outs.class_assignments
    else:
        final_class_means = None
        final_backprojections = [
            (final_ft_ctf, final_ft_y),
            (final_Ft_ctf_0, final_Ft_y_0),
            (final_Ft_ctf_1, final_Ft_y_1),
        ]
        del final_ft_ctf, final_ft_y
        del final_Ft_ctf_0, final_Ft_y_0, final_Ft_ctf_1, final_Ft_y_1
        final_maps = final_reconstruction.reconstruct_final_halfmaps(
            final_backprojections,
            final_mean_variance,
            settings=reconstruction_settings,
            current_size=final_current_size,
            accumulator_shape=final_mstep_accumulator_shape,
        )
    merged_mean = final_maps.merged
    final_means_for_output = final_maps.halves
    logger.info(
        "RELION final all-data reconstruction done: wall=%.1fs",
        time.time() - final_reconstruct_t0,
    )
    final_iter_elapsed = time.time() - final_iter_t0
    logger.info(
        "Final iter complete: current_size=%d (Nyquist), wall=%.1fs",
        final_current_size,
        final_iter_elapsed,
    )
    history.wall_times.append(final_iter_elapsed)

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
            merged_mean, final_means_for_output, final_class_means,
            class_weights if k_class_enabled else None,
            class_assignments if k_class_enabled else None,
        ),
        "unfiltered_means": final_unfiltered_means_for_output,
        "relion_follower_scale_replay_requested_iterations": replay_requested_iterations,
        "relion_follower_scale_replay_applied_iterations": replay_applied_iterations,
        **follower_setup.to_result_dict(history),
        # RELION-mode specific outputs
        "convergence_state": state,
        **history.to_dict(),
        **final_pass_result_fields(
            final_outs, final_sampling.settings,
            accuracy=final_expected_accuracy,
            accuracy_status=final_expected_accuracy_status,
            prior_details=final_tau2_update_details,
            fsc=final_iter_fsc,
            k_class_enabled=k_class_enabled,
        ),
    }
