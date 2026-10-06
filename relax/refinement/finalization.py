"""Final all-data admission, expectation and reconstruction.

The numbered controller owns convergence and replay. This module owns the
post-convergence sampling/scoring/reconstruction sequence and its result.
"""


import logging
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
    _dense_global_scoring_dtype,
    _local_adaptive_pass2_denominator_support_mode,
    _local_adaptive_pass2_full_parent_enabled,
    _local_adaptive_pass2_rotation_only_enabled,
    local_precision,
)
from relax.diagnostics import bpref_diagnostics
from relax.diagnostics.iteration import write_final_half_manifest
from relax.helpers.convergence import healpix_angular_step, update_angular_sampling
from relax.helpers.dtype_policy import _diagnostic_float64_pass2_matches
from relax.helpers.expected_accuracy import _expected_accuracy_class_ids
from relax.helpers.orientation_priors import relion_local_search_sigmas, relion_translation_search_base
from relax.helpers.resolution import (
    clamp_relion_coarse_image_size,
    class_current_resolution_shell,
    compute_coarse_image_size,
    k1_current_resolution_shell,
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
from relax.refinement.local_sampling import LocalSearchSettings, local_search_centre_half, prepare_final_local_sampling
from relax.refinement.mean_helpers import _class_weights_from_posterior, join_half_accumulators_at_low_resolution
from relax.refinement.projector_preparation import prepare_scoring_projector
from relax.refinement.refinement_options import FINAL_ALL_DATA_AFTER_MAX_ITER_ENV
from relax.refinement.refinement_result import ModelMaps, RefinementResult, ReplayTelemetry
from relax.refinement.result_files import final_pass_result
from relax.refinement.tomo_half import local_tomo_sampling
from relax.refinement.tomo_half import score_tomo_half_in_loop as _score_tomo_half_in_loop
from relax.relion.geometry import PROJECTION_PADDING_FACTOR, RECONSTRUCTION_PADDING_FACTOR
from relax.relion.relion_metadata import _relion_metadata_translations
from relax.relion.relion_worker_scale import _finalize_relion_follower_scale_replay_telemetry

# The numbered controller's log: the final pass logs under its name.
logger = logging.getLogger("relax.refinement.iteration_loop")



def _should_run_final_all_data_iteration(
    *,
    logger,
    has_converged: bool,
    iteration: int,
    max_iter: int,
    force_max_iter_after_convergence: bool,
    after_max_iter: bool,
    k_class_enabled: bool = False,
) -> bool:
    """Return whether to run RELION's final all-data reconstruction pass.

    ``after_max_iter`` (``RELAX_FINAL_ALL_DATA_AFTER_MAX_ITER``): K=1 runs it after the last numbered
    iteration without convergence.
    """

    if force_max_iter_after_convergence:
        return False
    if bool(has_converged):
        return True
    if not (
        after_max_iter
        and int(iteration) >= int(max_iter)
    ):
        return False
    if bool(k_class_enabled):
        logger.warning(
            "Ignoring %s=1 for K-class after max_iter exhaustion; final all-data "
            "is only valid for K-class after convergence",
            FINAL_ALL_DATA_AFTER_MAX_ITER_ENV,
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
    rotation_grid: sampling.RotationGrid,
    random_perturbation,
    perturb_rng,
    perturb_replay_relion_dir,
    sigma_offset,
    class_mixture,
    class_assignments,
    previous_data_vs_prior_for_scheduling,
    iteration,
    collect_local_search_profile,
    relion_translation_angle_scale,
) -> RefinementResult:
    """Score the converged halves at full size and reconstruct the final maps.

    Replay and final-pass admission are resolved by the numbered controller.
    The returned result holds this phase's results with ``numbered`` None; the caller adds
    the set-up and numbered-iteration metadata after execution.
    ``sigma_offset`` is the run's ``SigmaOffset`` (shared and per-half translation prior widths) and
    ``class_mixture`` its ``ClassMixture`` (class weights and their log priors; one class for K=1). The
    class count is ``options.k_class.n_classes``; arrays are in the global scoring dtype.
    See ``docs/math/relion_refinement_algorithm.md``, section 7.
    """
    scoring_dtype = _dense_global_scoring_dtype()
    n_classes = int(options.k_class.n_classes)
    class_weights = class_mixture.weights
    class_log_priors = class_mixture.log_priors
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
            # A failure raises: the slab is sized to the device by its budget
            # (accuracy_slab_resident_bytes), and an infinite accuracy would hide it.
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
    final_sampling = prepare_final_sampling(
        state,
        image_geometry,
        options,
        previous_rotation_grid=rotation_grid,
        numbered_iteration_count=len(history.current_sizes),
        active_replay_dir=perturb_replay_relion_dir,
        previous_perturbation=random_perturbation,
        rng=perturb_rng,
        dtype=scoring_dtype,
    )
    final_precision = local_precision(final_sampling.settings.relion_iteration, pass_index=2)
    ## THIS ALL SHOULD BE AN OBJECT/DICT/ENUM OR SOMETHING PROBABLY WITH THESE DECISIONS OR SOMETHING?
    ## SHOULD BE WANTING TO DEFINE 100 THINGS LIKE THIS
    if final_use_local:
        final_sigma_rot, final_sigma_psi = relion_local_search_sigmas(state, use_local=True)
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
            gridding_kernel=reconstruction_settings.gridding_kernel,
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
            nyquist_column_counting=options.consistency.nyquist_column_counting,
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
            skip_align=bool(options.k_class.skip_align),
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
            # RELION's source-faithful powerClass normalisation applies wherever its particle order is preserved.
            source_faithful_spectrum_norm=parity.preserve_bpref_particle_order,
            relion_translation_angle_scale=relion_translation_angle_scale,
            firstiter_cc_tree_rescore_max_margin=parity.firstiter_cc_tree_rescore_max_margin,
            nyquist_column_counting=options.consistency.nyquist_column_counting,
        )
    final_outs = PerHalfOutputs()
    for half, projector in zip(halves, final_projectors, strict=True):
        half = local_search_centre_half(half, (options.replay.init_angle_priors or (None, None))[half.index], state)
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
            sigma_offset.shared_angstrom,
            sigma_offset.per_half_angstrom,
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
                HalfScoringData(
                    particles=half,
                    reference=final_join_means[half.index],
                    noise_variance=noise_model.variance_per_half[half.index],
                    projector=projector,
                    scale_group_ids=follower_setup.scale_stats_group_ids_per_half[half.index],
                    scale_group_count=follower_setup.scale_stats_group_count_per_half[half.index],
                    scale_correction_data_vs_prior=previous_data_vs_prior_for_scheduling,
                ),
                use_local=True,
                use_adaptive=final_local_sampling.search.oversampling_order > 0,
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
                sigma_offset_angst=final_sigma_offset_k,
                max_significants=adaptive.max_significants,
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
                direction_prior=direction_priors[half.index],
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
            write_final_half_manifest(
                debug.save_intermediates_dir, half, final_sampling, final_inputs,
                translation_search_base=translation_search_base, reference=final_join_means[half.index],
                reference_model=reference_model, noise_variance=noise_model.variance_per_half[half.index],
                current_size=final_current_size, precision=final_precision, use_local=final_use_local, log=logger,
            )

    final_mstep_accumulator_shape = _resolve_mstep_accumulator_shape(
        final_outs.mstep_accumulator_shape,
        padded_volume_shape,
    )
    final_reconstruct_t0 = time.time()
    # The one mode decision of the final reconstruction. Each branch is its mode's whole sequence over the
    # steps of final_reconstruction.py, in order: accumulators, prior, resolution, maps. Both bind final_maps,
    # final_tau2_update_details, final_iter_fsc, final_model_maps and
    # the two mode operands of final_pass_result.
    #
    # RELION calls updateCurrentResolution after the final all-data
    # iteration too (ml_optimiser_mpi.cpp:4329), from that iteration's
    # whole-data DVP, so rlnCurrentResolution reports the final half-map FSC
    # at 0.143 rather than the last split-half iteration's 0.5 crossing.
    # Nothing after that point schedules on the resolution.
    if k_class_enabled:
        # Class3D: the two partitions sum to one data set. Class weights from the posterior, one prior per
        # class from its reference's power spectrum, the resolution, the class maps.
        final_ft_y = final_outs.Ft_y[0] + final_outs.Ft_y[1]
        final_ft_ctf = final_outs.Ft_ctf[0] + final_outs.Ft_ctf[1]
        final_mstep_full_half_axis = _resolve_mstep_full_half_axis(final_outs.mstep_full_half_axis, default_axis=-1)
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
        final_data_vs_prior = final_class_priors.data_vs_prior
        logger.info(
            "RELION final all-data Class3D tau2 from Iref power spectra: old_max=%.4e new_max=%.4e "
            "dvp_shell_1=%.4f wall=%.1fs",
            float(jnp.max(jnp.abs(reference_model.tau2))),
            float(jnp.max(jnp.abs(final_class_priors.variance))),
            float(np.asarray(final_data_vs_prior)[0, 1]) if np.asarray(final_data_vs_prior).shape[-1] > 1 else float("nan"),
            time.time() - _t_final_tau2,
        )
        final_res_shell = class_current_resolution_shell(
            final_data_vs_prior, current_size=final_current_size, grid_size=grid_size, dtype=scoring_dtype,
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
        logger.info(
            "RELION final all-data reconstruction start: current_size=%d n_classes=%d", final_current_size, n_classes,
        )
        final_maps = final_reconstruction.reconstruct_final_class_maps(
            final_ft_y,
            final_ft_ctf,
            final_class_priors.shells,
            class_weights=class_weights,
            n_classes=n_classes,
            settings=reconstruction_settings,
            current_size=final_current_size,
            accumulator_shape=final_mstep_accumulator_shape,
        )
        final_tau2_update_details = final_class_priors.details
        final_iter_fsc = None
        final_prior_weight_combination = "class_iref"
        final_class_assignments = final_outs.class_assignments
        final_model_maps = ModelMaps(
            mean=final_maps.merged,
            means=final_maps.halves,
            class_means=final_maps.halves[0],
            class_weights=class_weights,
            class_assignments=final_class_assignments,
        )
    else:
        # K=1: unfiltered half maps, optional low-resolution join, release of the pass outputs' references;
        # then the prior from the joined halves' FSC, the resolution, the merged map and the two half maps.
        final_Ft_y_0 = final_outs.Ft_y[0]
        final_Ft_y_1 = final_outs.Ft_y[1]
        final_Ft_ctf_0 = final_outs.Ft_ctf[0]
        final_Ft_ctf_1 = final_outs.Ft_ctf[1]
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
        final_mstep_full_half_axis = _resolve_mstep_full_half_axis(final_outs.mstep_full_half_axis, default_axis=-1)
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
        final_tau2_update_details = final_halfmap_prior.details
        logger.info(
            "RELION final all-data tau2 from joined FSC: old_max=%.4e new_max=%.4e "
            "fsc_shell_1=%.4f wall=%.1fs",
            float(jnp.max(jnp.abs(reference_model.tau2))),
            float(jnp.max(jnp.abs(final_halfmap_prior.variance))),
            float(np.asarray(final_iter_fsc)[1]) if np.asarray(final_iter_fsc).size > 1 else float("nan"),
            time.time() - _t_final_tau2,
        )
        final_res_shell = k1_current_resolution_shell(
            np.asarray(final_tau2_update_details["ssnr_shells"], dtype=scoring_dtype),
            current_size=final_current_size, grid_size=grid_size, dtype=scoring_dtype,
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
        logger.info(
            "RELION final all-data reconstruction start: current_size=%d n_classes=%d", final_current_size, n_classes,
        )
        # The merged map from the COMBINED accumulators (summed only now: 24.7 GB of host at box 800), each half
        # map from its own, at full Nyquist. The list is the only owner; each slot is freed after its solve.
        final_ft_y = final_Ft_y_0 + final_Ft_y_1
        final_ft_ctf = final_Ft_ctf_0 + final_Ft_ctf_1
        final_backprojections = [
            (final_ft_ctf, final_ft_y),
            (final_Ft_ctf_0, final_Ft_y_0),
            (final_Ft_ctf_1, final_Ft_y_1),
        ]
        del final_ft_ctf, final_ft_y
        del final_Ft_ctf_0, final_Ft_y_0, final_Ft_ctf_1, final_Ft_y_1
        final_maps = final_reconstruction.reconstruct_final_halfmaps(
            final_backprojections,
            final_halfmap_prior.variance,
            settings=reconstruction_settings,
            current_size=final_current_size,
            accumulator_shape=final_mstep_accumulator_shape,
        )
        final_prior_weight_combination = "sum"
        final_class_assignments = None
        final_model_maps = ModelMaps(
            mean=final_maps.merged, means=final_maps.halves, unfiltered_means=final_unfiltered_means_for_output,
        )
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

    return RefinementResult(
        maps=final_model_maps,
        replay=ReplayTelemetry(
            requested_iterations=replay_requested_iterations, applied_iterations=replay_applied_iterations,
        ),
        follower_scale=follower_setup.result_outputs(history),
        # RELION-mode specific outputs
        convergence_state=state,
        history=history,
        numbered=None,
        final_pass=final_pass_result(
            final_outs, final_sampling.settings,
            accuracy=final_expected_accuracy,
            accuracy_status=final_expected_accuracy_status,
            prior_details=final_tau2_update_details,
            fsc=final_iter_fsc,
            prior_weight_combination=final_prior_weight_combination,
            class_assignments=final_class_assignments,
            gridding_kernel=reconstruction_settings.gridding_kernel,
        ),
    )
