"""Final all-data admission, expectation and reconstruction.

The numbered controller owns convergence and replay. This module owns the
post-convergence sampling/scoring/reconstruction sequence and its result.
"""


import logging
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from relax.numerics.precision import diagnostic_float64_pass2_matches
from relax.reconstruction.resolution import (
    ImageGeometry,
    class_current_resolution_shell,
    k1_current_resolution_shell,
    relion_coarse_image_size,
    shell_index_to_resolution_angstrom,
)
from relax.refinement import final_reconstruction
from relax.refinement.dense_half import score_half_dense_in_bpref_scope
from relax.refinement.dense_specs import (
    DenseBatchPolicy,
    DenseExecutionPolicy,
    DensePriorSpec,
    DenseSamplingSpec,
    DenseVariantPolicy,
)
from relax.refinement.iteration_planning import IterationCarry
from relax.refinement.local_half import (
    LocalBatchPolicy,
    LocalDiagnosticPolicy,
    LocalExecutionPolicy,
    LocalPriorSpec,
    score_half_local_in_bpref_scope,
)
from relax.refinement.local_sampling import LocalSampling, LocalSearchSettings, prepare_final_local_sampling
from relax.refinement.numbered_reconstruction import merged_half_map, weighted_class_merge
from relax.refinement.ports import FinalHalfScored, InputSource
from relax.refinement.priors import join_half_accumulators_at_low_resolution
from relax.refinement.reference_state import class_weights_from_posterior
from relax.refinement.refinement_options import FINAL_ALL_DATA_AFTER_MAX_ITER_ENV, RefinementOptions
from relax.refinement.refinement_result import ModelMaps, RefinementResult
from relax.refinement.refinement_state import update_angular_sampling
from relax.refinement.result_files import final_pass_result
from relax.refinement.trial_grids import FinalSampling, prepare_final_sampling
from relax.relion.geometry import PROJECTION_PADDING_FACTOR, RECONSTRUCTION_PADDING_FACTOR
from relax.runtime.timing import Stopwatch
from relax.sampling.healpix import angular_sampling_deg
from relax.sampling.orientation_priors import (
    DirectionPrior,
    HalfDirectionLogPriors,
    HalfTranslationPriorInputs,
    make_relion_translation_log_prior,
    relion_direction_log_priors_for_half,
    relion_half_translation_prior_inputs,
    relion_local_search_sigmas,
    relion_translation_search_base,
)
from relax.star.metadata import relion_metadata_translations
from relax.steps import half_inputs, shape_class_scoring
from relax.steps.expected_accuracy import expected_accuracy_class_ids
from relax.steps.half_inputs import HalfScoringData, local_search_centre_half
from relax.steps.optics_shapes import image_translation_factors
from relax.steps.projector_preparation import prepare_scoring_projector
from relax.steps.score_outputs import (
    PerHalfOutputs,
    record_score_profile,
    resolve_mstep_accumulator_shape,
    resolve_mstep_full_half_axis,
)
from relax.steps.scoring_policy import (
    local_precision,
)
from relax.steps.shape_class_scoring import OpticsSpec
from relax.steps.tomo_half import local_tomo_sampling
from relax.steps.tomo_scoring import score_tomo_half_in_loop

if TYPE_CHECKING:
    from relax.refinement.setup_checks import RunContext

# The numbered controller's log: the final pass logs under its name.
logger = logging.getLogger("relax.refinement.iteration_loop")


def _should_run_final_all_data_iteration(
    *,
    has_converged: bool,
    iteration: int,
    max_iter: int,
    force_max_iter_after_convergence: bool,
    after_max_iter: bool,
) -> bool:
    """Return whether to run RELION's final all-data reconstruction pass.

    ``after_max_iter``: run it after the last numbered iteration without convergence (K=1's
    ``RELAX_FINAL_ALL_DATA_AFTER_MAX_ITER``; Class3D passes False, and refuses the option at set-up).
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
    return True


def final_pass_due(state, options: RefinementOptions, *, iteration: int, after_max_iter: bool) -> bool:
    """Whether the run ends with RELION's final all-data pass, after ``iteration`` numbered iterations.

    ``_should_run_final_all_data_iteration`` decides (``after_max_iter``: the mode's admission of a pass
    without convergence); ``schedule.skip_final_iteration`` skips the pass
    silently, any other skip is logged, and a pass run without convergence (``final_pass.after_max_iter``) is
    logged as the diagnostic it is.
    """
    should_run_final_iteration = _should_run_final_all_data_iteration(
        has_converged=state.has_converged,
        iteration=iteration,
        max_iter=options.schedule.max_iter,
        force_max_iter_after_convergence=options.schedule.force_max_iter_after_convergence,
        after_max_iter=after_max_iter,
    )
    if options.schedule.skip_final_iteration or not should_run_final_iteration:
        if not options.schedule.skip_final_iteration:
            logger.info(
                "Skipping RELION final all-data iteration: has_converged=%s, "
                "iteration=%d, max_iter=%d, force_max_iter_after_convergence=%s",
                state.has_converged,
                iteration,
                options.schedule.max_iter,
                options.schedule.force_max_iter_after_convergence,
            )
        return False
    if not state.has_converged:
        logger.info(
            "Diagnostic %s=1: running RELION final all-data iteration after max_iter exhaustion "
            "(iteration=%d, max_iter=%d)",
            FINAL_ALL_DATA_AFTER_MAX_ITER_ENV,
            iteration,
            options.schedule.max_iter,
        )
    return True


def numbered_k1_maps(means) -> ModelMaps:
    """The published maps of a K=1 run that ends after its numbered iterations: the half maps and their merge."""
    return ModelMaps(mean=merged_half_map(means), means=means, class_means=None)


def numbered_class_maps(means, class_weights, class_assignments) -> ModelMaps:
    """The published maps of a Class3D run that ends after its numbered iterations: the half pair of class
    stacks, their merged class maps, the class-weighted merge of those, and the class weights and assignments."""
    merged_class_means = merged_half_map(means)
    return ModelMaps(
        mean=weighted_class_merge(merged_class_means, class_weights),
        means=means,
        class_means=merged_class_means,
        class_weights=class_weights,
        class_assignments=class_assignments,
    )


def run_final_all_data(
    halves,
    *,
    ctx: "RunContext",
    carry: IterationCarry,
    reference_model,
    history,
    options,
    follower_setup,
    expected_accuracy_inputs,
    final_join_means,
    final_use_local,
    source: InputSource,
    iteration,
    remote_halves=None,
) -> RefinementResult:
    """Score the converged halves at full size and reconstruct the final maps.

    ``remote_halves`` (a multi-GPU run's ``RemoteHalves``; None on one GPU) scores the halves another GPU owns.

    Replay and final-pass admission are resolved by the numbered controller.
    The returned result holds this phase's results with ``numbered`` None; the caller adds the set-up and
    numbered-iteration metadata after execution.
    Reads the run's context (``ctx``: optics, reconstruction settings, batch planner, perturbation RNG,
    profile switch, expectation probe) and the last numbered iteration's ``carry`` (state, noise model and
    sigma offset as the input source's final state left them, direction priors, grids, perturbation, class
    mixture and assignments, scheduling curve); the final pass updates its own copy of the carry's state
    (the RefinementState object itself is updated in place, as before) and returns it as the convergence
    state. The class count is ``options.k_class.n_classes``; arrays are in ``options.precision``'s dtype.
    See ``docs/math/relion_refinement_algorithm.md``, section 7.
    """
    image_geometry = ctx.optics.image_geometry
    scoring_dtype = options.precision.rotation_real_dtype
    class_weights = carry.class_mixture.weights
    k_class_enabled = options.k_class.n_classes > 1
    volume_shape = ctx.reconstruction_settings.volume_shape
    padded_volume_shape = tuple(d * RECONSTRUCTION_PADDING_FACTOR for d in volume_shape)
    final_expected_accuracy = None
    final_expected_accuracy_status = "not_run"
    final_iter_clock = Stopwatch()
    final_current_size = image_geometry.box_size
    if carry.native_sampling_boundary:
        final_eulers_half1 = halves[0].rotation_eulers
        if expected_accuracy_inputs.trial_order_local is None or final_eulers_half1 is None:
            final_expected_accuracy_status = "unavailable_inputs"
            carry.state.acc_rot = float("inf")
            carry.state.acc_trans = float("inf")
            logger.warning(
                "RELION final all-data expected accuracy unavailable; "
                "final expectation remains fail-closed",
            )
        else:
            final_accuracy_class_ids = expected_accuracy_class_ids(
                carry.class_assignments[0],
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
                sigma2_noise_native=carry.noise_model.radial_per_half[0],
                current_image_size=final_current_size,
            )
            carry.state.acc_rot = final_expected_accuracy.acc_rot
            carry.state.acc_trans = final_expected_accuracy.acc_trans_angstrom
            carry = replace(carry, state=update_angular_sampling(
                carry.state, low_pmax_guard=options.environment.low_pmax_refine_guard,
            ))
            final_expected_accuracy_status = "ok"
            logger.info(
                "RELION final all-data expected accuracy: acc_rot=%.3f deg, "
                "acc_trans=%.4f A",
                carry.state.acc_rot,
                carry.state.acc_trans,
            )
    final_sampling = prepare_final_sampling(
        carry.state,
        image_geometry,
        options,
        previous_rotation_grid=carry.coarse_grids.rotation_grid,
        numbered_iteration_count=len(history.current_sizes),
        source=source,
        previous_perturbation=carry.random_perturbation,
        rng=ctx.perturb_rng,
        dtype=scoring_dtype,
    )
    final_precision = local_precision(
        options.precision, final_sampling.settings.relion_iteration, pass_index=2,
        float64_pass2_iterations=options.environment.diagnostic_float64_pass2_iterations,
    )
    ## THIS ALL SHOULD BE AN OBJECT/DICT/ENUM OR SOMETHING PROBABLY WITH THESE DECISIONS OR SOMETHING?
    ## SHOULD BE WANTING TO DEFINE 100 THINGS LIKE THIS
    if final_use_local:
        final_sigma_rot, final_sigma_psi = relion_local_search_sigmas(carry.state, use_local=True)
        final_search = LocalSearchSettings(
            healpix_order=final_sampling.settings.perturbation_order + carry.state.adaptive_oversampling,
            oversampling_order=carry.state.adaptive_oversampling,
            sigma_rot=final_sigma_rot,
            sigma_psi=final_sigma_psi,
            symmetry=options.symmetry.point_group,
        )
        final_local_sampling = prepare_final_local_sampling(
            final_search,
            ctx.optics,
            translations=final_sampling.grid.translations,
            base_translations=final_sampling.base_translations,
            image_window_size=final_current_size,
            particle_diameter_angstrom=options.schedule.particle_diameter_ang,
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
        if k_class_enabled and carry.state.adaptive_oversampling > 0:
            final_adaptive_pass1_current_size = relion_coarse_image_size(
                angular_sampling_deg(final_sampling.settings.grid_order),
                image_geometry.pixel_size_angstrom,
                image_geometry.box_size,
                particle_diameter=options.schedule.particle_diameter_ang,
                current_size=final_current_size,
            )
            final_adaptive_pass2_current_size = final_current_size
            logger.info(
                "RELION final all-data adaptive K-class pass-1: coarse_current_size=%d "
                "fine_current_size=%d oversampling=%d",
                final_adaptive_pass1_current_size,
                final_adaptive_pass2_current_size,
                carry.state.adaptive_oversampling,
            )
        else:
            final_adaptive_pass1_current_size = None
            final_adaptive_pass2_current_size = None
    logger.info("=== RELION final all-data Nyquist iteration ===")
    if final_use_local:
        final_local_batching = LocalBatchPolicy(
            max_significants=options.adaptive.max_significants,
        )
        final_local_execution = LocalExecutionPolicy(
            disc_type="linear_interp",
            # As the numbered local pass and the final dense pass: RELION's high-shell norm and sigma2 terms are
            # the same powerClass sums in the final iteration (ml_optimiser.cpp:10848-10856).
            source_faithful_spectrum_norm=options.parity.preserve_bpref_particle_order,
            relion_translation_angle_scale=ctx.relion_translation_angle_scale,
            nyquist_column_counting=options.consistency.nyquist_column_counting,
            relion_x_half_mstep=options.variants.k1_relion_x_half_mstep,
            adaptive_pass2=options.variants.local_adaptive_pass2.at(carry.state.adaptive_oversampling),
            precision=options.precision,
            fine_precision=final_precision,
            score_only=False,
            firstiter_cc=False,
            engine_tuning=options.environment.engine,
            scoring_overrides=options.environment.scoring,
        )
        final_local_diagnostics = LocalDiagnosticPolicy(
            iteration=iteration + 1,
            debug_iteration=final_sampling.settings.relion_iteration,
            bpref_device_signature_active=False,
            probe=ctx.expectation_probe,
            collect_local_search_profile=ctx.collect_local_search_profile,
            local_profile_history=history.local_profile_history,
        )
    else:
        # Resolved once: the batch plan and the execution policy read the same answer.
        diagnostic_float64_pass2 = diagnostic_float64_pass2_matches(
            final_sampling.settings.relion_iteration, options.environment.diagnostic_float64_pass2_iterations,
        )
        final_dense_sampling = DenseSamplingSpec(
            effective_rotations=final_sampling.grid.rotations,
            current_translations=final_sampling.grid.translations,
            base_translations=final_sampling.base_translations,
            current_healpix_order=final_sampling.settings.grid_order,
            oversampling_order=carry.state.adaptive_oversampling,
            translation_step=carry.state.translation_step,
            coarse_engine=options.adaptive.coarse_engine,
            random_perturbation=final_sampling.settings.random_perturbation,
            image_window_size=final_current_size,
            symmetry=options.symmetry.point_group,
        )
        final_dense_batching = DenseBatchPolicy(
            safe_batch_sizes=ctx.batch_planner.for_expectation(diagnostic_float64_pass2=diagnostic_float64_pass2),
            max_significants=options.adaptive.max_significants,
            significance_safe_batch_sizes=None,
            k_class_image_batch_size_override=None,
            k_class_rotation_block_size_override=None,
            significance_image_batch_size_override=None,
            significance_rotation_block_size_override=None,
            class_batch_overrides=None,
        )
        final_dense_variant = DenseVariantPolicy(
            score_mode="gaussian",
            winner_take_all=False,
            k_class_enabled=k_class_enabled,
            firstiter_cc=False,
            coarse_window_size=final_adaptive_pass1_current_size,
            fine_window_size=final_adaptive_pass2_current_size,
            skip_align=bool(options.k_class.skip_align),
        )
        final_dense_execution = DenseExecutionPolicy(
            disc_type="linear_interp",
            return_best_pose_details=not k_class_enabled,
            # Unlike the numbered pass, the final pass keeps the BPREF signature off and RELION's CC support:
            # the two pass builders differ here, and no parity run has checked which one RELION follows.
            bpref_device_signature_active=False,
            debug_iteration=final_sampling.settings.relion_iteration,
            diagnostic_float64_pass2=diagnostic_float64_pass2,
            preserve_bpref_particle_order=options.parity.preserve_bpref_particle_order,
            # RELION's source-faithful powerClass normalisation applies wherever its particle order is preserved.
            source_faithful_spectrum_norm=options.parity.preserve_bpref_particle_order,
            relion_translation_angle_scale=ctx.relion_translation_angle_scale,
            firstiter_cc_tree_rescore_max_margin=options.parity.firstiter_cc_tree_rescore_max_margin,
            firstiter_cc_support="relion",
            nyquist_column_counting=options.consistency.nyquist_column_counting,
            relion_x_half_mstep=options.variants.relion_x_half_mstep(k_class=k_class_enabled),
            precision=options.precision,
            k_class_probe=ctx.k_class_probe,
            keep_pose_rotation_eulers=ctx.expectation_probe.keeps_pose_rotation_eulers,
            engine_tuning=options.environment.engine,
            scoring_overrides=options.environment.scoring,
        )
    final_pass = FinalPass(
        current_size=final_current_size,
        volume_shape=volume_shape,
        use_local=final_use_local,
        sampling=final_sampling,
        route=(
            FinalLocalRoute(
                sampling=final_local_sampling,
                batching=final_local_batching,
                execution=final_local_execution,
                diagnostics=final_local_diagnostics,
            )
            if final_use_local
            else FinalDenseRoute(
                sampling=final_dense_sampling,
                batching=final_dense_batching,
                variant=final_dense_variant,
                execution=final_dense_execution,
            )
        ),
        references=final_join_means,
        tau2=reference_model.tau2,
        state=carry.state,
        noise_model=carry.noise_model,
        sigma_offset=carry.sigma_offset,
        direction_priors=carry.direction_priors,
        class_log_priors=carry.class_mixture.log_priors,
        scale_gate=carry.previous_data_vs_prior_for_scheduling,
        sealed_sampling_state=(
            source.sealed_sampling_state if final_sampling.base_rotations is carry.coarse_grids.rotation_grid.rotations else None
        ),
    )
    final_outs = PerHalfOutputs()
    first_projector_power_spectrum = None
    score_half = partial(score_final_half, ctx=ctx, options=options, follower_setup=follower_setup)
    # A multi-GPU run sends the halves another GPU owns first and publishes each in its turn.
    remote = () if remote_halves is None else tuple(half.index for half in halves if remote_halves.remote(half.index))
    for half in halves:
        if half.index in remote:
            remote_halves.submit(half.index, score_half, half, final_pass)
    for half in halves:
        scored = remote_halves.result(half.index) if half.index in remote else score_half(half, final_pass)
        if first_projector_power_spectrum is None:
            first_projector_power_spectrum = scored.projector_power_spectrum
        final_outs.translation_search_bases[half.index] = scored.translation_search_base
        final_outs.update_from(half.index, scored.result, dtype=scoring_dtype)
        record_score_profile(
            history.global_profile_history,
            scored.result,
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
            scored.clock.seconds,
        )
        ctx.expectation_probe.final_half_scored(FinalHalfScored(
            scored.half, final_sampling, scored.prepared,
            translation_search_base=scored.translation_search_base, reference=final_join_means[half.index],
            reference_model=reference_model, noise_variance=carry.noise_model.variance_per_half[half.index],
            current_size=final_current_size, precision=final_precision, use_local=final_use_local,
        ))
        # The half is published: drop this loop's references to its result before the next half is scored.
        del scored

    final_mstep_accumulator_shape = resolve_mstep_accumulator_shape(
        final_outs.mstep_accumulator_shape,
        padded_volume_shape,
    )
    final_reconstruct_clock = Stopwatch()
    # The one mode decision of the final reconstruction. Each branch is its mode's whole sequence over the
    # steps of final_reconstruction.py, in order: accumulators, prior, resolution, maps.
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
        # As for K=1, drop the pass outputs' references to the partition accumulators: the summed ones are the
        # only ones read from here on.
        final_outs.Ft_y[0] = final_outs.Ft_y[1] = None
        final_outs.Ft_ctf[0] = final_outs.Ft_ctf[1] = None
        final_mstep_full_half_axis = resolve_mstep_full_half_axis(final_outs.mstep_full_half_axis, default_axis=-1)
        class_weights = class_weights_from_posterior(
            final_outs.class_posterior,
            options.k_class.n_classes,
            class_weights,
        )
        history.record_class_weights(
            class_weights,
            class_weights_from_posterior(
                final_outs.class_full_posterior,
                options.k_class.n_classes,
                class_weights,
            ),
        )
        final_tau2_clock = Stopwatch()
        final_class_priors = final_reconstruction.compute_final_class_priors(
            final_ft_ctf,
            final_join_means[0],
            projector_power_spectrum=first_projector_power_spectrum,
            n_classes=options.k_class.n_classes,
            settings=ctx.reconstruction_settings,
            current_size=final_current_size,
            accumulator_shape=final_mstep_accumulator_shape,
            full_half_axis=final_mstep_full_half_axis,
            stable_window_quantum=options.environment.engine.stable_window_quantum,
        )
        final_data_vs_prior = final_class_priors.data_vs_prior
        logger.info(
            "RELION final all-data Class3D tau2 from Iref power spectra: dvp_shell_1=%.4f wall=%.1fs",
            float(np.asarray(final_data_vs_prior)[0, 1]) if np.asarray(final_data_vs_prior).shape[-1] > 1 else float("nan"),
            final_tau2_clock.seconds,
        )
        final_res_shell = class_current_resolution_shell(
            final_data_vs_prior, current_size=final_current_size, box_size=image_geometry.box_size,
            dtype=scoring_dtype,
        )
        carry.state.previous_resolution = carry.state.current_resolution
        carry.state.current_resolution = shell_index_to_resolution_angstrom(
            final_res_shell, image_geometry.box_size, image_geometry.pixel_size_angstrom
        )
        logger.info(
            "RELION final all-data current resolution: shell=%d res=%.2f A (last split-half iteration %.2f A)",
            int(final_res_shell),
            carry.state.current_resolution,
            carry.state.previous_resolution,
        )
        logger.info(
            "RELION final all-data reconstruction start: current_size=%d n_classes=%d",
            final_current_size, options.k_class.n_classes,
        )
        final_maps = final_reconstruction.reconstruct_final_class_maps(
            final_ft_y,
            final_ft_ctf,
            final_class_priors.shells,
            class_weights=class_weights,
            n_classes=options.k_class.n_classes,
            settings=ctx.reconstruction_settings,
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
            settings=ctx.reconstruction_settings,
            current_size=final_current_size,
            accumulator_shape=final_mstep_accumulator_shape,
        )
        if (
            options.parity.low_resol_join_halves_angstrom is not None
            and options.parity.low_resol_join_halves_angstrom > 0
        ):
            final_Ft_y_0, final_Ft_y_1, final_Ft_ctf_0, final_Ft_ctf_1 = join_half_accumulators_at_low_resolution(
                (final_Ft_y_0, final_Ft_y_1),
                (final_Ft_ctf_0, final_Ft_ctf_1),
                accumulator_volume_shape=final_mstep_accumulator_shape,
                box_size=image_geometry.box_size,
                voxel_size=image_geometry.pixel_size_angstrom,
                padding_factor=RECONSTRUCTION_PADDING_FACTOR,
                low_resolution_angstrom=options.parity.low_resol_join_halves_angstrom,
                pixel_resolutions=history.pixel_resolutions,
                current_resolution=carry.state.current_resolution,
                preserve_inputs=False,
                route=options.variants.reconstruction.lowres_join,
            )
        # The unfiltered maps are made; drop the pass outputs' references so the
        # pre-join accumulators live only as long as the joined ones do.
        final_outs.Ft_y[0] = final_outs.Ft_y[1] = None
        final_outs.Ft_ctf[0] = final_outs.Ft_ctf[1] = None
        final_mstep_full_half_axis = resolve_mstep_full_half_axis(final_outs.mstep_full_half_axis, default_axis=-1)
        final_tau2_clock = Stopwatch()
        final_halfmap_prior = final_reconstruction.compute_final_halfmap_prior(
            (final_Ft_y_0, final_Ft_y_1),
            (final_Ft_ctf_0, final_Ft_ctf_1),
            settings=ctx.reconstruction_settings,
            current_size=final_current_size,
            accumulator_shape=final_mstep_accumulator_shape,
            full_half_axis=final_mstep_full_half_axis,
            scoring_dtype=scoring_dtype,
        )
        final_iter_fsc = final_halfmap_prior.fsc
        final_tau2_update_details = final_halfmap_prior.details
        logger.info(
            "RELION final all-data tau2 from joined FSC: fsc_shell_1=%.4f wall=%.1fs",
            float(np.asarray(final_iter_fsc)[1]) if np.asarray(final_iter_fsc).size > 1 else float("nan"),
            final_tau2_clock.seconds,
        )
        final_res_shell = k1_current_resolution_shell(
            np.asarray(final_tau2_update_details["ssnr_shells"], dtype=scoring_dtype),
            current_size=final_current_size, box_size=image_geometry.box_size, dtype=scoring_dtype,
        )
        carry.state.previous_resolution = carry.state.current_resolution
        carry.state.current_resolution = shell_index_to_resolution_angstrom(
            final_res_shell, image_geometry.box_size, image_geometry.pixel_size_angstrom
        )
        logger.info(
            "RELION final all-data current resolution: shell=%d res=%.2f A (last split-half iteration %.2f A)",
            int(final_res_shell),
            carry.state.current_resolution,
            carry.state.previous_resolution,
        )
        logger.info(
            "RELION final all-data reconstruction start: current_size=%d n_classes=%d",
            final_current_size, options.k_class.n_classes,
        )
        # Each half map from its own accumulators, then the merged map from their sum, at full Nyquist. The list
        # is the only owner: the sum is formed in the first half's arrays after both half maps are solved.
        final_backprojections = [
            (final_Ft_ctf_0, final_Ft_y_0),
            (final_Ft_ctf_1, final_Ft_y_1),
        ]
        del final_Ft_ctf_0, final_Ft_y_0, final_Ft_ctf_1, final_Ft_y_1
        final_maps = final_reconstruction.reconstruct_final_halfmaps(
            final_backprojections,
            final_halfmap_prior.variance,
            settings=ctx.reconstruction_settings,
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
        final_reconstruct_clock.seconds,
    )
    final_iter_elapsed = final_iter_clock.seconds
    logger.info(
        "Final iter complete: current_size=%d (Nyquist), wall=%.1fs",
        final_current_size,
        final_iter_elapsed,
    )
    history.wall_times.append(final_iter_elapsed)

    return RefinementResult(
        maps=final_model_maps,
        follower_scale=follower_setup.result_outputs(history),
        convergence_state=carry.state,
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
            gridding_kernel=ctx.reconstruction_settings.gridding_kernel,
        ),
    )


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
    optics = shape_class_scoring.prepare_optics(
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


@dataclass(frozen=True, kw_only=True)
class FinalLocalRoute:
    """The exact-local route of the final all-data pass: its sampling and the local engine's policies."""

    sampling: LocalSampling
    batching: LocalBatchPolicy
    execution: LocalExecutionPolicy
    diagnostics: LocalDiagnosticPolicy


@dataclass(frozen=True, kw_only=True)
class FinalDenseRoute:
    """The dense route of the final all-data pass: its sampling and the dense engine's policies."""

    sampling: DenseSamplingSpec
    batching: DenseBatchPolicy
    variant: DenseVariantPolicy
    execution: DenseExecutionPolicy


@dataclass(frozen=True, kw_only=True)
class FinalPass:
    """What each half of the final all-data pass is scored with, resolved once before the halves.

    ``route`` is the subtomogram pass's local sampling too (a ``FinalLocalRoute`` when ``use_local``); a
    subtomogram global pass reads ``sampling`` and ``state`` alone. ``references`` are the halves' scoring maps
    (the joined means), ``tau2`` the reference model's shared tau2, ``scale_gate`` the iteration-start
    scale-correction curve (``previous_data_vs_prior_for_scheduling``) and ``sealed_sampling_state`` the input
    source's sealed rows when the final grid is the sealed one, else None. ``state`` and the carry's records
    are the run's own objects: scoring reads them and changes none.
    """

    current_size: int
    volume_shape: tuple
    use_local: bool
    sampling: FinalSampling
    route: FinalLocalRoute | FinalDenseRoute
    references: list
    tau2: object
    state: object
    noise_model: object
    sigma_offset: object
    direction_priors: list
    class_log_priors: object
    scale_gate: object
    sealed_sampling_state: object | None


@dataclass(frozen=True, kw_only=True)
class FinalHalfScore:
    """One half of the final all-data pass scored by :func:`score_final_half`.

    ``half`` is the half as scored (centred for this pass's local search), ``prepared`` its single-particle
    priors and optics (None for subtomograms), ``projector_power_spectrum`` its scoring projector's power
    spectrum (Class3D's final tau2 reads half 1's) and ``clock`` started when the half's scoring started, after
    its projector was built.
    """

    half: half_inputs.HalfSet
    result: object
    translation_search_base: object
    prepared: PreparedFinalHalf | None
    projector_power_spectrum: object
    clock: Stopwatch


def score_final_half(
    half: half_inputs.HalfSet, final_pass: FinalPass, *, ctx: "RunContext", options: RefinementOptions, follower_setup
) -> FinalHalfScore:
    """Build one half's scoring projector and score the half on the final pass's route.

    Each half's Projector::data lives only inside this call. The caller publishes the result.
    """
    image_geometry = ctx.optics.image_geometry
    scoring_dtype = options.precision.rotation_real_dtype
    # Pass 1 scores RELION's exact coarse operands on every route, so the final pass builds the projector too.
    # Each half's Projector::data is built when its pass starts and released when it ends: complex128, 33 GiB
    # per half at EMPIAR-10202's box, and both were held for the whole final pass (relax#39).
    projector_clock = Stopwatch()
    projector = prepare_scoring_projector(
        final_pass.references[half.index],
        volume_shape=final_pass.volume_shape,
        current_size=final_pass.current_size,
        padding_factor=PROJECTION_PADDING_FACTOR,
        n_classes=options.k_class.n_classes,
        dump_label=f"final_half{half.index}",
        gridding_kernel=ctx.reconstruction_settings.gridding_kernel,
        files=options.environment.projector_files,
        stable_window_quantum=options.environment.engine.stable_window_quantum,
    )
    logger.info(
        "RELION final all-data half-%d: built exact Projector::data for scoring at current_size=%d r_max=%s "
        "in %.2fs",
        half.index + 1,
        final_pass.current_size,
        projector.r_max,
        projector_clock.seconds,
    )
    half = local_search_centre_half(half, (options.start.init_angle_priors or (None, None))[half.index], final_pass.state)
    final_half_clock = Stopwatch()
    logger.info(
        "RELION final all-data half-%d start: images=%d current_size=%d "
        "healpix_order=%d n_rot=%d n_trans=%d local_search=%s",
        half.index + 1,
        half.dataset.n_units,
        final_pass.current_size,
        final_pass.sampling.settings.grid_order,
        final_pass.sampling.base_rotations.shape[0],
        final_pass.sampling.grid.translations.shape[0],
        final_pass.use_local,
    )
    translation_search_base = relion_translation_search_base(
        half.translations, dtype=scoring_dtype
    )
    final_sigma_offset_k = final_pass.sigma_offset.for_half(half.index)
    if ctx.tomo_halves:
        final_inputs = None  # the half preparation below is single-particle only
        # Subtomograms: the tomo half pass at the final sampling (RELION's local search on the
        # previous poses), with the merged reference for both halves.
        # RELION's final iteration keeps the orientational prior the run converged with: a local search,
        # or below --auto_local_healpix_order the --sigma_tilt prior or none, on the global sampling
        # (the convergence branch sets has_converged, do_join_random_halves and do_use_all_data and
        # nothing of the prior or the sampling, ml_optimiser.cpp:10152-10160).
        tilt_prior = half_inputs.tilt_prior_for_half(
            half,
            (options.start.init_angle_priors or (None, None))[half.index],
            options.local_search.tilt_prior_deg(final_pass.state.do_local_search),
        )
        if final_pass.use_local:
            tomo_final_sampling = local_tomo_sampling(
                fine_order=final_pass.route.sampling.search.healpix_order,
                oversampling_order=final_pass.route.sampling.search.oversampling_order,
                translation_range_px=final_pass.sampling.settings.translation_range,
                translation_step_px=final_pass.sampling.settings.translation_step,
                model_pixel_size=ctx.optics.model_pixel_size,
                random_perturbation=final_pass.route.sampling.perturbation,
                pass1_size=final_pass.route.sampling.coarse_image_window_size,
                current_size=final_pass.current_size,
            )
            tomo_final_search = dict(
                previous_eulers_deg=half.rotation_eulers,
                sigma_rot=final_pass.route.sampling.search.sigma_rot,
                sigma_psi=final_pass.route.sampling.search.sigma_psi,
                sigma_tilt_deg=0.0,
            )
        else:
            tomo_final_sampling = local_tomo_sampling(
                fine_order=final_pass.sampling.settings.grid_order + final_pass.state.adaptive_oversampling,
                oversampling_order=final_pass.state.adaptive_oversampling,
                translation_range_px=final_pass.sampling.settings.translation_range,
                translation_step_px=final_pass.sampling.settings.translation_step,
                model_pixel_size=ctx.optics.model_pixel_size,
                random_perturbation=final_pass.sampling.settings.random_perturbation,
                pass1_size=relion_coarse_image_size(
                    angular_sampling_deg(final_pass.sampling.settings.grid_order),
                    image_geometry.pixel_size_angstrom,
                    image_geometry.box_size,
                    particle_diameter=options.schedule.particle_diameter_ang,
                    current_size=final_pass.current_size,
                ),
                current_size=final_pass.current_size,
            )
            if tilt_prior is not None:
                logger.info(
                    "RELION final all-data iteration under the --sigma_tilt prior: healpix_order=%d, "
                    "sigma_tilt=%g deg",
                    final_pass.sampling.settings.grid_order, tilt_prior.sigma_deg,
                )
                tomo_final_search = dict(
                    previous_eulers_deg=(
                        np.zeros((half.dataset.n_units, 3), dtype=np.float64)
                        if half.rotation_eulers is None
                        else half.rotation_eulers
                    ),
                    sigma_rot=0.0,
                    sigma_psi=0.0,
                    sigma_tilt_deg=tilt_prior.sigma_deg,
                    prior_tilt_deg=tilt_prior.centre_deg,
                )
            else:
                # No orientational prior (RELION's NOPRIOR): the global search of the numbered iterations,
                # each orientation weighted by this half's direction distribution
                # (acc_ml_optimiser_impl.h:1968-1969), which is even when the sampling changed since it
                # was learned.
                logger.info(
                    "RELION final all-data iteration as a global subtomogram search: healpix_order=%d",
                    final_pass.sampling.settings.grid_order,
                )
                tomo_final_search = None
                tomo_final_direction_log_prior = relion_direction_log_priors_for_half(
                    use_local=False,
                    scoring_healpix_order=final_pass.sampling.settings.grid_order,
                    n_classes=options.k_class.n_classes,
                    prior=final_pass.direction_priors[half.index],
                    sealed_sampling_state=None,
                    dtype=scoring_dtype,
                    log=logger,
                    half_index=half.index,
                    symmetry=options.symmetry.point_group,
                ).rotation_log_prior
        final_result = score_tomo_half_in_loop(
            HalfScoringData(
                particles=half,
                reference=final_pass.references[half.index],
                noise_variance=final_pass.noise_model.variance_per_half[half.index],
                projector=projector,
                scale_group_ids=follower_setup.scale_stats_group_ids_per_half[half.index],
                scale_group_count=follower_setup.scale_stats_group_count_per_half[half.index],
                scale_correction_data_vs_prior=final_pass.scale_gate,
            ),
            use_adaptive=tomo_final_sampling.oversampling_order > 0,
            sampling=tomo_final_sampling,
            local_search=tomo_final_search,
            rotation_log_prior=None if tomo_final_search is not None else tomo_final_direction_log_prior,
            sigma_offset_angst=final_sigma_offset_k,
            max_significants=options.adaptive.max_significants,
            reconstruction_current_size=final_pass.current_size,
            symmetry=options.symmetry.point_group,
            engine_tuning=options.environment.engine,
            scoring_overrides=options.environment.scoring,
        )
    else:
        # The final pass scores each half with its own priors on the grid rows it
        # actually uses; sealed rows apply only when the sealed grid is reused.
        final_inputs = prepare_final_half(
            half,
            final_pass.sampling,
            image_geometry=image_geometry,
            sigma_offset_angstrom=final_sigma_offset_k,
            noise_radial=final_pass.noise_model.radial_per_half[half.index],
            direction_prior=final_pass.direction_priors[half.index],
            n_classes=options.k_class.n_classes,
            use_local=final_pass.use_local,
            coarse_angular_step_deg=final_pass.route.sampling.coarse_angular_step_deg if final_pass.use_local else None,
            particle_diameter_angstrom=options.schedule.particle_diameter_ang,
            sealed_sampling_state=final_pass.sealed_sampling_state,
            symmetry=options.symmetry.point_group,
            dtype=scoring_dtype,
            projection_scale=ctx.relion_projection_scale,
        )
        scoring_half = HalfScoringData(
            particles=half,
            projector=projector,
            reference=final_pass.references[half.index],
            mean_variance=final_pass.tau2,
            noise_variance=final_pass.noise_model.variance_per_half[half.index],
            scale_group_ids=follower_setup.scale_stats_group_ids_per_half[half.index],
            scale_group_count=follower_setup.scale_stats_group_count_per_half[half.index],
            scale_correction_data_vs_prior=final_pass.scale_gate,
        )
        if final_pass.use_local:
            final_result = score_half_local_in_bpref_scope(
                half=scoring_half,
                sampling=final_pass.route.sampling,
                priors=LocalPriorSpec(
                    trans_prior_center=final_inputs.translations.local_prior_center,
                    trans_prior_center_for_engine=final_inputs.translations.engine_prior_center,
                    current_sigma_offset_angstrom=final_sigma_offset_k,
                    translation_search_base=translation_search_base,
                    class_log_priors=final_pass.class_log_priors,
                ),
                batching=final_pass.route.batching,
                execution=final_pass.route.execution,
                diagnostics=final_pass.route.diagnostics,
                optics=final_inputs.optics,
            )
        else:
            final_result = score_half_dense_in_bpref_scope(
                half=scoring_half,
                sampling=final_pass.route.sampling,
                priors=DensePriorSpec(
                    rotation_log_prior_k=final_inputs.directions.rotation_log_prior,
                    class_rotation_log_prior_k=final_inputs.directions.class_rotation_log_prior,
                    translation_log_prior=final_inputs.translation_log_prior,
                    translation_search_base=translation_search_base,
                    trans_prior_center_for_engine=final_inputs.translations.engine_prior_center,
                    class_log_priors=final_pass.class_log_priors,
                ),
                batching=final_pass.route.batching,
                variant=final_pass.route.variant,
                execution=final_pass.route.execution,
                optics=final_inputs.optics,
            )
    if final_result.best_pose_translations is not None:
        final_result.best_pose_translations = relion_metadata_translations(
            half.translations,
            final_result.best_pose_translations,
            own_pixel_factors=image_translation_factors(half.dataset),
            dtype=scoring_dtype,
        )

    return FinalHalfScore(
        half=half,
        result=final_result,
        translation_search_base=translation_search_base,
        prepared=final_inputs,
        projector_power_spectrum=projector.power_spectrum,
        clock=final_half_clock,
    )
