"""Orchestrate dense single-volume and K-class EM refinement.

``refine_single_volume`` validates options and manages refinement state and dispatch.
``dense_half`` and ``local_half`` own the per-half dense and local engine calls; ``scoring_policy``
owns their shared execution defaults and diagnostic selectors. Local chunks
are implemented in ``local_search_iteration``; state-swap diagnostics belong
to ``parity.state_swap_runtime``. Pure trial-grid construction belongs to
``sampling``.
See ``docs/math/relion_refinement_algorithm.md`` for the algorithm map.
"""


import logging
from dataclasses import replace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from recovar.data_io import cryoem_dataset

import relax.diagnostics.iteration as iteration_diagnostics
import relax.diagnostics.reconstruction as reconstruction_diagnostics
import relax.helpers.convergence as convergence_helpers
import relax.parity.relion_replay as replay_policy
from relax.diagnostics import bpref_diagnostics
from relax.helpers import expected_accuracy, iteration_history, orientation_priors, resolution, timing
from relax.refinement import (
    expectation,
    finalization,
    half_inputs,
    iteration_planning,
    iteration_snapshot,
    local_sampling,
    map_postprocess,
    maximization,
    noise_updates,
    numbered_reconstruction,
    numbered_transitions,
    particle_poses,
    ports,
    projector_preparation,
    reference_state,
    refinement_options,
    refinement_result,
    setup_checks,
    sigma_offset,
    startup_references,
    trial_grids,
)
from relax.relion import relion_normalization, relion_worker_scale
from relax.sparse_pass2 import engine_record

logger = logging.getLogger(__name__)


# An iteration whose significant samples grow by more than this factor is reported: pass 2's cost follows them.
SIGNIFICANT_SAMPLE_JUMP_FACTOR = 20.0


def _warn_if_significant_samples_jumped(significant_counts, iteration: int) -> None:
    """One warning when this iteration's significant samples exceed the previous iteration's by a large factor."""
    if len(significant_counts) < 2 or significant_counts[-1] is None or significant_counts[-2] is None:
        return
    previous, current = (float(np.sum(counts, dtype=np.float64)) for counts in significant_counts[-2:])
    if previous > 0.0 and current > SIGNIFICANT_SAMPLE_JUMP_FACTOR * previous:
        logger.warning(
            "Iteration %d kept %.0f significant samples, %.0f times the previous iteration's %.0f: its second pass "
            "and the next iterations will be that much slower", iteration + 1, current, current / previous, previous,
        )




class _ClassStart(NamedTuple):
    """A continued run's class state from its snapshot: each half's assignments, the previous ones the first
    change statistic reads, and the class mixture."""

    class_assignments: list
    previous_class_assignments: list
    class_mixture: object


class _K1Iteration:
    """The K=1 auto-refine iteration's own controller statements. ``_ClassIteration`` has the same methods; the
    loop chooses one of the two once per run and calls them at the same points (code rule 6)."""

    run_files_snapshot = staticmethod(iteration_snapshot.k1_run_files_snapshot)
    # K=1 keeps each half's own sigma2_noise.
    update_noise_variance = staticmethod(noise_updates.update_k1_posterior_noise_variance)

    def follower_scale_setup(self, options, source, halves, experiment_datasets):
        """K=1 runs no follower-scale emulation (a follower topology is refused)."""
        return relion_worker_scale.k1_follower_scale_state(
            options, topology=source.follower_topology, relion_half_inputs=halves,
            experiment_datasets=experiment_datasets,
        )

    def initial_reference_model(self, ctx, options, init_volume, init_mean_variance):
        """Each half's flat start-up reference with the shared or per-half tau2."""
        return reference_state.initialize_reference_model(
            reference_state.initial_half_references(init_volume, options.k_class.n_classes), jnp.asarray(init_mean_variance),
            use_per_half_mean_variance=options.parity.use_per_half_mean_variance, dtype=ctx.scoring_dtype,
            log=logger,
        )

    def snapshot_reference_model(self, ctx, resume):
        return reference_state.reference_model_from_snapshot(resume, ctx.volume_shape, dtype=ctx.scoring_dtype)

    def class_start_from_snapshot(self, resume, class_mixture):
        """One class: no assignments, the start-up class mixture."""
        return _ClassStart([None, None], [None, None], class_mixture)

    def record_direction_prior(self, history, direction_priors):
        """K=1 records each half's global prior."""
        history.record_k1_direction_prior(direction_priors)

    def pmax_normalization_mass(self, per_half):
        """K=1's optimizer Pmax normalises by the noise particle mass."""
        return convergence_helpers.relion_k1_pmax_normalization_mass_per_half(per_half.noise_stats)

    def final_pass_due(self, state, options, iteration):
        """K=1 may run the final pass after max_iter exhaustion (``final_pass.after_max_iter``)."""
        return finalization.final_pass_due(
            state, options, iteration=iteration, after_max_iter=options.final_pass.after_max_iter,
        )

    def plan_image_size(self, ctx, carry, options, history, observer, *, previous_size, resume, iteration):
        """The image size from the half-map FSC history; installs the growth state and the scheduling curve."""
        image_size_plan = iteration_planning.plan_halfmap_image_size(
            history.fsc_history, growth_fsc_history=history.fsc_for_growth_history, restart=resume,
            data_vs_prior=carry.previous_data_vs_prior_for_scheduling, previous_size=previous_size,
            box_size=ctx.image_geometry.box_size, pixel_size_angstrom=ctx.source_pixel_size_angstrom,
            incr_size=carry.relion_incr_size, has_high_fsc_at_limit=carry.relion_has_high_fsc_at_limit,
            ave_pmax=carry.state.ave_Pmax,
            completed_relion_iteration=options.schedule.numbered_relion_iteration(iteration) - 1,
            parity=options.parity, dtype=ctx.scoring_dtype, log=logger,
        )
        carry = replace(
            carry, previous_data_vs_prior_for_scheduling=image_size_plan.data_vs_prior,
            relion_incr_size=image_size_plan.incr_size,
            relion_has_high_fsc_at_limit=image_size_plan.has_high_fsc_at_limit,
        )
        return carry, image_size_plan

    def refuse_local_search(self, options):
        """Auto-refine switches to local searches from the HEALPix order: nothing to refuse."""

    def record_class_weights(self, carry, options, history, per_half):
        """One class: no class weights to update."""
        return carry

    def maximize(
        self, ctx, carry, options, history, this_iteration, operands, reference_model, source, scored_class_weights
    ):
        """RELION's split-half M-step (compareTwoHalves -> updateSSNRarrays -> reconstruct)."""
        mstep = maximization.k1_maximization(
            reference_model, operands, ctx, this_iteration, parity=options.parity,
            pixel_resolutions=history.pixel_resolutions, current_resolution=carry.state.current_resolution,
        )
        return carry, mstep

    def solved_accumulators(self, mstep, operands):
        """The accumulators the solve used (joined at low resolution when that is on) replace the scored ones."""
        return mstep.Ft_y_per_half, mstep.Ft_ctf_per_half

    def learned_direction_priors(self, ctx, options, per_half, plan, use_local):
        return maximization.k1_learned_direction_priors(
            per_half.rotation_posterior, direction_prior_order=plan.direction_prior_healpix_order,
            symmetry=options.symmetry.point_group, dtype=ctx.scoring_dtype, log=logger,
        )

    def unfiltered_maps(self, ctx, options, reference_model, mstep, accumulator_shape, needed):
        """The unregularized half-maps (when ``needed``), then each new map's sign aligned to the previous."""
        unreg_means = (
            numbered_reconstruction.reconstruct_unregularized_k1_halfmaps(
                mstep.Ft_y_per_half,
                mstep.Ft_ctf_per_half,
                ctx.reconstruction_settings,
                accumulator_volume_shape=accumulator_shape,
            )
            if needed
            else [None, None]
        )
        map_postprocess.align_k1_volume_signs(reference_model.maps, mstep.previous_means, unreg_means, ctx.volume_shape)
        return unreg_means

    def fsc(self, mstep):
        """K=1's FSC (the M-step's, which set tau2 before the Wiener solve) also drives size growth."""
        return mstep.fsc

    def log_statistics(self, statistics, per_half):
        """Nothing beyond the history's Pmax row."""

    def resolution(self, ctx, carry, options, history, this_iteration, mstep):
        """The resolution from the half-map FSC's SSNR, and no class change (one class)."""
        resolution_estimate = resolution.estimate_k1_iteration_resolution(
            mstep.tau2_update_details["ssnr_shells"], current_size=this_iteration.current_size,
            box_size=ctx.image_geometry.box_size,
            voxel_size=ctx.source_pixel_size_angstrom,
            emulate_relion_firstiter_cc=options.parity.emulate_relion_firstiter_cc,
            ini_high_angstrom=options.parity.relion_firstiter_ini_high_angstrom,
            relion_iteration=int(options.schedule.init_relion_iteration) + int(this_iteration.iteration) + 1,
            dtype=ctx.scoring_dtype,
        )
        return resolution_estimate, 0.0

    def scheduling_curve_after_resolution(self, ctx, carry, history, resolution_estimate):
        """K=1's next scheduling curve is the resolution estimate's data-vs-prior: installed and recorded."""
        carry = replace(carry, previous_data_vs_prior_for_scheduling=np.asarray(
            resolution_estimate.data_vs_prior, dtype=ctx.scoring_dtype,
        ))
        history.data_vs_prior_trajectory.append(carry.previous_data_vs_prior_for_scheduling)
        return carry

    def sampling_decision_now(self, carry):
        """Auto-refine decides the sampling at the end of a natively sampled iteration."""
        return not carry.native_sampling_boundary

    def after_convergence(self, carry, options, iteration):
        """RELION's one-time reset of the follower's counter after the convergence update."""
        return replace(carry, state=numbered_transitions.reset_follower_counter_once(carry.state, options, iteration=iteration))

    def numbered_maps(self, reference_model, carry):
        return finalization.numbered_k1_maps(reference_model.maps)

    def final_join_means(self, reference_model, options):
        """Each half's own map, or (diagnostic) the merged map for both halves."""
        if not options.final_pass.merged_reference:
            return [reference_model.maps[0], reference_model.maps[1]]
        final_merged_reference = numbered_reconstruction.merged_half_map(reference_model.maps)
        logger.info(
            "Diagnostic %s=1: final all-data K=1 E-step uses merged reference for both halves",
            refinement_options.FINAL_ALL_DATA_USE_MERGED_REFERENCE_ENV,
        )
        return [final_merged_reference, final_merged_reference]

    def final_use_local(self, carry):
        """The final pass searches locally where the numbered iterations did."""
        return carry.state.do_local_search


class _ClassIteration:
    """The Class3D iteration's own controller statements; the methods of ``_K1Iteration``."""

    run_files_snapshot = staticmethod(iteration_snapshot.class_run_files_snapshot)
    # Class3D shares one sigma2_noise across classes.
    update_noise_variance = staticmethod(noise_updates.update_class_posterior_noise_variance)

    def follower_scale_setup(self, options, source, halves, experiment_datasets):
        """RELION's per-follower group-scale emulation where the input source has a follower topology."""
        return relion_worker_scale.setup_relion_follower_scale_state(
            options,
            topology=source.follower_topology,
            relion_half_inputs=halves,
            experiment_datasets=experiment_datasets,
            # The STAR replay's restart iterations (a strict Class3D replay restarts its follower scales with them).
            restart_state_iterations=(
                () if source.relion_replay is None else source.relion_replay.perturb_replay_restart_state_iterations
            ),
        )

    def initial_reference_model(self, ctx, options, init_volume, init_mean_variance):
        """Each half's start-up class stack with the shared tau2."""
        return reference_state.initialize_class_reference_model(
            reference_state.initial_half_references(init_volume, options.k_class.n_classes), jnp.asarray(init_mean_variance),
            use_per_half_mean_variance=options.parity.use_per_half_mean_variance,
        )

    def snapshot_reference_model(self, ctx, resume):
        return reference_state.class_reference_model_from_snapshot(resume, ctx.volume_shape, dtype=ctx.scoring_dtype)

    def class_start_from_snapshot(self, resume, class_mixture):
        """The snapshot's class assignments (also the previous ones) and class weights."""
        class_assignments = [None if c is None else np.asarray(c) for c in resume.class_assignments]
        return _ClassStart(
            class_assignments, [None if c is None else c.copy() for c in class_assignments],
            reference_state.class_mixture_from_weights(np.asarray(resume.class_weights, dtype=np.float64)),
        )

    def record_direction_prior(self, history, direction_priors):
        """Class3D records class 0 of each half's prior."""
        history.record_first_class_direction_prior(direction_priors)

    def pmax_normalization_mass(self, per_half):
        """Class3D's optimizer Pmax normalises by the class posterior mass."""
        return convergence_helpers.relion_class_pmax_normalization_mass_per_half(per_half.class_posterior)

    def final_pass_due(self, state, options, iteration):
        """Class3D runs the final pass only after convergence."""
        return finalization.final_pass_due(
            state, options, iteration=iteration,
            after_max_iter=finalization.class_after_max_iter(state, options, iteration=iteration),
        )

    def plan_image_size(self, ctx, carry, options, history, observer, *, previous_size, resume, iteration):
        """The image size from the shared per-class curve; the observer sees the plan (the carry is unchanged)."""
        image_size_plan = iteration_planning.plan_class_image_size(
            carry.previous_data_vs_prior_for_scheduling, previous_size=previous_size,
            box_size=ctx.image_geometry.box_size,
            pixel_size_angstrom=ctx.source_pixel_size_angstrom, incr_size=carry.relion_incr_size,
            ave_pmax=carry.state.ave_Pmax,
            completed_relion_iteration=options.schedule.numbered_relion_iteration(iteration) - 1,
            parity=options.parity, dtype=ctx.scoring_dtype, log=logger,
        )
        observer.class_image_size_planned(
            iteration, image_size_plan, previous_size=previous_size, box_size=ctx.image_geometry.box_size,
            has_high_fsc_at_limit=carry.relion_has_high_fsc_at_limit, incr_size=carry.relion_incr_size,
            state=carry.state,
        )
        return carry, image_size_plan

    def refuse_local_search(self, options):
        """Class3D searches locally only with --sigma_ang: RELION switches from the HEALPix order only under
        auto-refine (ml_optimiser.cpp:2541-2565, 3936-3938)."""
        if options.local_search.sigma_ang_deg is None:
            raise RuntimeError("K>1 (Class3D) reached local angular searches without --sigma_ang")

    def record_class_weights(self, carry, options, history, per_half):
        """The class weights from this expectation's posterior: installed, recorded with the full-posterior
        weights, and logged."""
        scored_weights = carry.class_mixture.weights
        carry = replace(carry, class_mixture=reference_state.class_mixture_from_weights(
            reference_state._class_weights_from_posterior(
                per_half.class_posterior, options.k_class.n_classes, carry.class_mixture.weights,
            ),
        ))
        for class_idx in reference_state.emptied_classes(scored_weights, carry.class_mixture.weights):
            logger.warning(
                "Class %d received no particle weight: it is empty and stays out of every later expectation "
                "(RELION's pdf_class == 0)", class_idx + 1,
            )
        history.record_class_weights(
            carry.class_mixture.weights,
            reference_state._class_weights_from_posterior(
                per_half.class_full_posterior, options.k_class.n_classes, carry.class_mixture.weights,
            ),
        )
        logger.info(
            "K-class occupancies: %s",
            ", ".join(f"class {idx + 1}={weight:.4f}" for idx, weight in enumerate(carry.class_mixture.weights)),
        )
        return carry

    def maximize(
        self, ctx, carry, options, history, this_iteration, operands, reference_model, source, scored_class_weights
    ):
        """Class3D joins the halves per class, carries the previous Iref power spectrum forward as tau2 and
        solves once per class; its data-vs-prior curve is recorded and becomes the scheduling curve."""
        mstep = maximization.class_maximization(
            reference_model, operands, ctx, options, this_iteration,
            class_tau2=source.class_tau2(this_iteration.iteration, options.k_class.n_classes),
            scored_class_weights=scored_class_weights,
            previous_data_vs_prior=carry.previous_data_vs_prior_for_scheduling,
        )
        history.data_vs_prior_trajectory.append(mstep.data_vs_prior)
        carry = replace(carry, previous_data_vs_prior_for_scheduling=mstep.data_vs_prior)
        return carry, mstep

    def solved_accumulators(self, mstep, operands):
        """The scored accumulators (the joined class sums stay in the M-step's record)."""
        return operands.numerators, operands.denominators

    def learned_direction_priors(self, ctx, options, per_half, plan, use_local):
        return maximization.class_learned_direction_priors(
            per_half.rotation_posterior, per_half.class_rotation_posterior, n_classes=options.k_class.n_classes,
            direction_prior_order=plan.direction_prior_healpix_order, symmetry=options.symmetry.point_group,
            use_local=use_local, n_trial_rotations=plan.trial_grid.rotations.shape[0], dtype=ctx.scoring_dtype,
        )

    def unfiltered_maps(self, ctx, options, reference_model, mstep, accumulator_shape, needed):
        """The unregularized class means from the joined class sums (when ``needed``)."""
        return (
            numbered_reconstruction.reconstruct_unregularized_class_means(
                mstep.Ft_y_combined,
                mstep.Ft_ctf_combined,
                ctx.reconstruction_settings,
                options.k_class.n_classes,
                accumulator_volume_shape=accumulator_shape,
            )
            if needed
            else [None, None]
        )

    def fsc(self, mstep):
        """Class3D has no half-map FSC (its shared per-class curve drives growth)."""
        return None

    def log_statistics(self, statistics, per_half):
        logger.info(
            "Class3D optimizer Pmax: value=%.9f numerator=%.9f "
            "half1_mstep_posterior_mass=%.9f half1_particle_count=%d",
            statistics.ave_pmax, float(np.sum(np.asarray(per_half.max_posterior[0]), dtype=np.float64)),
            statistics.ave_pmax_mass, int(np.asarray(per_half.max_posterior[0]).size),
        )

    def resolution(self, ctx, carry, options, history, this_iteration, mstep):
        """The joined class assignments (recorded) and their change since the previous iteration; the
        resolution from the shared per-class prior and the combined class accumulators."""
        current_combined_classes = convergence_helpers.concatenate_assignments(carry.class_assignments)
        history.class_assignment_history.append(current_combined_classes.copy())
        previous_combined_classes = convergence_helpers.concatenate_assignments_or_none(carry.previous_class_assignments)
        resolution_estimate = resolution.estimate_class_iteration_resolution(
            history.data_vs_prior_trajectory[-1], current_size=this_iteration.current_size,
            box_size=ctx.image_geometry.box_size,
            voxel_size=ctx.source_pixel_size_angstrom,
            emulate_relion_firstiter_cc=options.parity.emulate_relion_firstiter_cc,
            ini_high_angstrom=options.parity.relion_firstiter_ini_high_angstrom,
            relion_iteration=int(options.schedule.init_relion_iteration) + int(this_iteration.iteration) + 1,
            dtype=ctx.scoring_dtype,
        )
        return resolution_estimate, convergence_helpers.hard_class_change_fraction(current_combined_classes, previous_combined_classes)

    def scheduling_curve_after_resolution(self, ctx, carry, history, resolution_estimate):
        """Class3D's scheduling curve is the M-step's (already installed)."""
        return carry

    def sampling_decision_now(self, carry):
        """Class3D takes no auto-refine sampling decision."""
        return False

    def after_convergence(self, carry, options, iteration):
        return carry

    def numbered_maps(self, reference_model, carry):
        return finalization.numbered_class_maps(
            reference_model.maps, carry.class_mixture.weights, carry.class_assignments,
        )

    def final_join_means(self, reference_model, options):
        """Each half's own class stack."""
        return [reference_model.maps[0], reference_model.maps[1]]

    def final_use_local(self, carry):
        """The Class3D final pass searches globally."""
        return False


def _follower_replay_telemetry(source, history) -> refinement_result.ReplayTelemetry:
    """The follower-scale replay's requested and applied iterations (both None without a replay), validated
    against the iterations the run applied; raises if the replay was not applied as requested."""

    requested, applied = relion_worker_scale._finalize_relion_follower_scale_replay_telemetry(
        source, applied_iterations=history.relion_follower_scale_replay_applied_iterations, logger=logger,
    )
    return refinement_result.ReplayTelemetry(requested_iterations=requested, applied_iterations=applied)


def refine_single_volume(
    experiment_datasets: list[cryoem_dataset.CryoEMDataset],
    startup: startup_references.StartupHandoff,
    init_noise_variance: half_inputs.HalfPair,
    translations: jnp.ndarray | None,
    options: refinement_options.RefinementOptions,
    observer: ports.RunObserver,
    source: ports.InputSource,
) -> refinement_result.RefinementResult:
    """Multi-iteration RELION-parity EM refinement.

    Run it inside ``relax.sparse_pass2.resident_pass2.stable_window_class_history()``, as the commands do, so a
    refinement's resident passes reuse the window classes they already ran (about 5 s of recompiles saved per
    current-size crossing on the 5k K=1 run); outside one, every pass picks its class afresh.

    Implements the loop described in ``docs/math/relion_refinement_algorithm.md``.

    Parameters
    ----------
    experiment_datasets : list of 2 dataset objects
        Half-set datasets (same format as run_halfset_em_iteration expects).
    startup : ``StartupHandoff`` (``relax.refinement.startup_references``) holding the initial Fourier volume
        (a ``HalfPair``), the initial signal prior (tau^2, shape (volume_size,)) and the first projector's real
        maps (a ``HalfPair`` or None). The loop takes them once, so they are freed after the start-up (relax#26).
    init_noise_variance : ``HalfPair`` (``relax.refinement.half_inputs``)
        Each half-set's initial per-pixel noise variance: a flat image vector, or ``[G, P]`` rows for G > 1
        optics groups.
    translations : jnp.ndarray, shape (n_trans, 2)
        Translation grid.
    options : `RefinementOptions` struct that bundles the schedule / adaptive / parity
        / local-search / K-class / replay / debug / batching kwarg groups, built by the command.
    observer : the run's ``RunObserver`` (``relax.refinement.ports``): dumps and captures that watch the run
        and never change it (``RunObserver()``: one that does nothing).
    source : the run's ``InputSource`` (``relax.refinement.ports``): what a comparison run takes from
        elsewhere (a RELION run) instead of computing it (``InputSource()``: the native computation).

    Returns
    -------
    RefinementResult (``relax.refinement.refinement_result``): the maps (``maps.mean``, ``maps.means``),
    the last ``convergence_state``, the run's ``history`` (``current_sizes``, ``fsc_history``,
    ``pixel_resolutions``, ``wall_times``, ``significant_counts``, the trajectories), the set-up and
    numbered-iteration metadata (``numbered``), the final pass's outputs (``final_pass``, None when it did
    not run) and ``profile_stop`` for a local-search diagnostic stop. ``archive_fields()`` is the flat
    mapping the archive and reports read.
    """
    options = refinement_options.with_validated_sampling_schedule(options)
    refinement_options.require_process_precision(options)

    # Each set-up phase's cumulative seconds since the set-up started (a result: the archive and the ledger).
    setup_clock = timing.Stopwatch()
    setup_phase_seconds = {}

    ctx = setup_checks.build_run_context(
        experiment_datasets, options, replays_relion_state=source.replays_relion_state(),
        observer=observer,
    )
    observer.run_started(ctx)
    # The run's one mode decision: the K=1 or the Class3D run's own set-up and iteration statements.
    mode = _ClassIteration() if ctx.k_class_enabled else _K1Iteration()
    replay_policy._validate_bpref_particle_order_scope(
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
    class_mixture = reference_state._initialize_class_log_priors(options.k_class.n_classes, options.start.init_direction_prior)

    setup_phase_seconds["mask_and_image_cache"] = setup_clock.seconds

    state = iteration_planning.initialize_refinement_state(
        options, ctx.image_geometry, subtomogram=ctx.tomo_halves, dtype=ctx.scoring_dtype, source=source,
        hands_reference_real=startup.hands_reference_real,
    )
    resume = options.checkpoint.resume
    setup_phase_seconds["state_init"] = setup_clock.seconds

    # The refinement schedule owns the initial coarse HEALPix grid; a continuation
    # rebuilds the grid of its restored sampling state.
    initial_grid_order = (
        int(options.schedule.init_healpix_order) if resume is None else convergence_helpers._exhaustive_grid_order_for_state(state)
    )
    coarse_grids = source.initial_coarse_grids(
        initialized_healpix_order=options.schedule.init_healpix_order if resume is None else state.healpix_order,
        voxel_size=ctx.source_pixel_size_angstrom, symmetry=options.symmetry.point_group,
    )
    if coarse_grids is None:
        coarse_grids = trial_grids.build_initial_coarse_grids(
            initial_grid_order, translations if resume is None else None,
            translation_range=options.schedule.init_translation_range if resume is None else state.translation_range,
            translation_step=options.schedule.init_translation_step if resume is None else state.translation_step,
            n_classes=options.k_class.n_classes, voxel_size=ctx.source_pixel_size_angstrom,
            symmetry=options.symmetry.point_group, dtype=ctx.scoring_dtype,
        )
    # coarse_grids keeps the unperturbed host-RFLOAT base translations: each iteration perturbs a fresh copy.
    setup_phase_seconds["sampling_grid"] = setup_clock.seconds

    init_volume, init_mean_variance, init_reference_real = startup.take()
    initial_real_references_by_half = projector_preparation.prepare_initial_real_references(
        init_reference_real, volume_shape=ctx.volume_shape, n_classes=options.k_class.n_classes,
        init_relion_iteration=options.schedule.init_relion_iteration, log=logger,
    )
    del init_reference_real
    initial_noise_variance_per_half = noise_updates.noise_rows_per_half(init_noise_variance)
    optics_group_ids_per_half = setup_checks.checked_optics_group_ids(
        options.optics_geometry.optics_group_ids_per_half, initial_noise_variance_per_half, experiment_datasets
    )
    setup_phase_seconds["initial_arrays"] = setup_clock.seconds

    history = iteration_history.RefinementHistory(keep_rotation_posteriors=observer.keeps_rotation_posteriors)
    engine_record.take_pass_engines()  # entries from before this run's first iteration belong to no iteration
    engine_record.take_coarse_engine_calls()
    previous_assignments = [None, None]
    halves = half_inputs.initialize_halfsets(
        experiment_datasets, optics_group_ids=optics_group_ids_per_half,
        previous_best_translations=options.start.init_previous_best_translations,
        previous_best_rotation_eulers=options.start.init_previous_best_rotation_eulers,
        image_corrections=options.start.init_image_corrections,
        scale_corrections=options.start.init_scale_corrections, group_ids=options.start.init_group_ids,
        group_count=options.start.init_group_count,
    )
    if int(options.start.init_relion_incr_size) <= 0:
        raise ValueError("init_relion_incr_size must be positive")

    expected_accuracy_inputs = setup_checks.expected_accuracy_inputs_for_run(
        options, experiment_datasets[0], ctx.volume_shape,
        optics_group_ids=optics_group_ids_per_half[0], gridding_kernel=ctx.consistency.gridding_kernel,
    )

    follower_setup = mode.follower_scale_setup(options, source, halves, experiment_datasets)
    # The replayed run's follower-scale replay (None without one).
    follower_scale_replay = None if source.follower_topology is None else source.follower_topology.replay

    if resume is None:
        # --- A fresh run starts from the caller's references, noise and replayed particle state ---
        # Each half stores its references in the loop's layout: an explicit leading
        # class axis for K classes, one flat reference for K=1.
        reference_model = mode.initial_reference_model(ctx, options, init_volume, init_mean_variance)
        # Per-shell radial profiles of the input pixel-array noise variances, for the
        # diagnostic log ("noise update per shell: old=... new=...").
        noise_model = noise_updates.initialize_noise_model(
            initial_noise_variance_per_half, average_variance=noise_updates._mean_noise_variance(initial_noise_variance_per_half),
            image_shape=ctx.image_geometry.image_shape, dtype=ctx.scoring_dtype,
        )
        # Drop the start-up arrays: where the caller passed temporaries (the parity script), the models
        # are then their only holders and the first updates free them; relax refine keeps its own copies.
        del init_volume, init_mean_variance, initial_noise_variance_per_half
        class_assignments = [None, None]
        previous_class_assignments = [None, None]
        previous_data_vs_prior_for_scheduling = (
            None
            if options.start.init_data_vs_prior is None
            else np.asarray(options.start.init_data_vs_prior, dtype=ctx.scoring_dtype)
        )
        # RELION's sigma2_offset in Angstrom^2 (min_sigma2_offset=2 A^2), updated each iteration from the data.
        start_sigma_offset = sigma_offset.SigmaOffset.from_halves(options.start.init_translation_sigma_angstrom)
        relion_incr_size = int(options.start.init_relion_incr_size)
        relion_has_high_fsc_at_limit = (
            bool(options.start.init_has_high_fsc_at_limit)
            if options.start.init_has_high_fsc_at_limit is not None
            else False
        )
        direction_priors = orientation_priors.initial_direction_priors_from_snapshot(
            options.start.init_direction_prior, n_classes=options.k_class.n_classes, dtype=ctx.scoring_dtype,
            log=logger,
            symmetry=options.symmetry.point_group, expected_order=coarse_grids.rotation_grid.healpix_order,
        )
        random_perturbation = iteration_planning.initial_random_perturbation(options, log=logger)
        published_accuracy = expected_accuracy.PublishedAccuracy.before_first_estimate(options.k_class.n_classes)
    else:
        # --- A continued run starts from the run files of an earlier run (RELION --continue) ---
        # The snapshot replaces every value the next numbered iteration reads, so the
        # first loop iteration runs as iteration init_relion_iteration + 1 of the
        # uninterrupted run (see relax/refinement/iteration_snapshot.py). It reads none of
        # the start-up arrays: they are released before the snapshot's model is built.
        del init_volume, init_mean_variance, initial_noise_variance_per_half
        reference_model = mode.snapshot_reference_model(ctx, resume)
        noise_model = noise_updates.noise_model_from_shells(resume.noise_shells, ctx.image_geometry.image_shape)
        for half, eulers, translations, images, scales in zip(
            halves, resume.rotation_eulers, resume.translations,
            resume.image_corrections, resume.scale_corrections, strict=True,
        ):
            half.rotation_eulers = eulers
            half.translations = translations
            half.image_corrections = images
            half.scale_corrections = scales
        class_start = mode.class_start_from_snapshot(resume, class_mixture)
        class_assignments = class_start.class_assignments
        previous_class_assignments = class_start.previous_class_assignments
        class_mixture = class_start.class_mixture
        previous_data_vs_prior_for_scheduling = np.asarray(resume.data_vs_prior, dtype=ctx.scoring_dtype)
        start_sigma_offset = sigma_offset.SigmaOffset.from_halves(resume.sigma_offset_angstrom)
        relion_incr_size = int(resume.incr_size)
        relion_has_high_fsc_at_limit = bool(resume.has_high_fsc_at_limit)
        direction_priors = iteration_snapshot.direction_priors_from_snapshot(
            resume, options.start.init_direction_prior, n_classes=options.k_class.n_classes,
            grid_healpix_order=coarse_grids.rotation_grid.healpix_order, symmetry=options.symmetry.point_group,
            dtype=ctx.scoring_dtype, log=logger,
        )
        random_perturbation = resume.random_perturbation
        published_accuracy = expected_accuracy.PublishedAccuracy.before_first_estimate(
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
    previous_best_rotations = particle_poses.best_rotation_matrices(halves, dtype=ctx.scoring_dtype)
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
        ports.ScoringArrays(reference_model, noise_model, start_sigma_offset, halves, direction_priors, experiment_datasets)
    )
    carry = iteration_planning.IterationCarry(
        state=state, class_mixture=class_mixture, direction_priors=direction_priors, noise_model=noise_model,
        sigma_offset=start_sigma_offset, coarse_grids=coarse_grids, random_perturbation=random_perturbation,
        replay_saved_healpix_order=replay_saved_healpix_order, native_sampling_boundary=native_sampling_boundary,
        relion_incr_size=relion_incr_size, relion_has_high_fsc_at_limit=relion_has_high_fsc_at_limit,
        previous_data_vs_prior_for_scheduling=previous_data_vs_prior_for_scheduling,
        previous_assignments=previous_assignments, previous_class_assignments=previous_class_assignments,
        previous_best_rotations=previous_best_rotations, class_assignments=class_assignments,
        # Per-half numbered-iteration assignments; a final-only replay
        # (--max_iter 0 --force-final-after-zero-iterations) runs no numbered
        # iteration and reports none.
        hard_assignments=[None, None],
        published_accuracy=published_accuracy,
    )
    # The carry is the only holder of the start-up values from here, so the loop's replacements release them
    # as the loop's reassignments did (code rule 3).
    del (
        state, class_mixture, direction_priors, noise_model, start_sigma_offset, coarse_grids, random_perturbation,
        replay_saved_healpix_order, native_sampling_boundary, relion_incr_size, relion_has_high_fsc_at_limit,
        previous_data_vs_prior_for_scheduling, previous_assignments, previous_class_assignments,
        previous_best_rotations, class_assignments, published_accuracy,
    )
    # Set when a local-search diagnostic stops the run after its first local search.
    profile_stop = None
    while (
        options.schedule.force_max_iter_after_convergence or not carry.state.has_converged
    ) and iteration < options.schedule.max_iter:
        # A continued run's first iteration follows the snapshot's iteration.
        has_previous_iteration = iteration > 0 or resume is not None
        seeding = iteration_planning.class_seeding(options, continued=resume is not None, iteration=iteration)
        star_directory = source.relion_run_directory(iteration)
        carry = replace(carry, native_sampling_boundary=star_directory is None and source.sealed_sampling_state is None)
        if carry.native_sampling_boundary:
            carry = replace(carry, replay_saved_healpix_order=None)
        # RELION checks convergence at the top of iteration n from the
        # completed n-1 statistics and the fine-enough decision latched during
        # expectation n-1.  If true, iteration n is the unnumbered joined
        # all-data pass rather than another numbered half-set iteration.
        if (
            numbered_transitions.uses_native_auto_refine(
                native_sampling_boundary=carry.native_sampling_boundary,
                n_classes=options.k_class.n_classes,
            )
            and not options.schedule.force_max_iter_after_convergence
            and has_previous_iteration
            and convergence_helpers.check_convergence(carry.state)
        ):
            carry.state.has_converged = True
            logger.info(
                "Convergence reached after numbered iteration %d. "
                "Entering RELION final all-data iteration.",
                iteration,
            )
            break
        iteration_clock = timing.Stopwatch()
        observer.iteration_started(iteration)
        numbered_relion_iteration = options.schedule.numbered_relion_iteration(iteration)

        if follower_setup.follower_scale_state is not None:
            relion_worker_scale._dispatch_relion_follower_scale_for_numbered_iteration(
                follower_setup, history, iteration=iteration, numbered_relion_iteration=numbered_relion_iteration,
                relion_half_inputs=halves, relion_follower_scale_replay_source=follower_scale_replay,
                dtype=ctx.scoring_dtype, logger=logger,
            )

        # Image support uses the preceding iteration's spectra, before replay
        # and angular sampling select this expectation's grid.
        if not has_previous_iteration:
            image_size_plan = iteration_planning.plan_initial_image_size(
                options, box_size=ctx.image_geometry.box_size, pixel_size_angstrom=ctx.source_pixel_size_angstrom,
                incr_size=carry.relion_incr_size, has_high_fsc_at_limit=carry.relion_has_high_fsc_at_limit,
                dtype=ctx.scoring_dtype, log=logger,
            )
            if image_size_plan.data_vs_prior is not None:
                carry = replace(carry, previous_data_vs_prior_for_scheduling=image_size_plan.data_vs_prior)
            carry = replace(
                carry, relion_incr_size=image_size_plan.incr_size,
                relion_has_high_fsc_at_limit=image_size_plan.has_high_fsc_at_limit,
            )
        else:
            prev_cs = history.current_sizes[-1] if history.current_sizes else int(resume.current_size)
            carry, image_size_plan = mode.plan_image_size(
                ctx, carry, options, history, observer, previous_size=prev_cs, resume=resume, iteration=iteration,
            )

        current_size = iteration_planning.resolve_current_size(
            image_size_plan, options, previous_size=prev_cs if has_previous_iteration else None,
            incr_size=carry.relion_incr_size, has_high_fsc_at_limit=carry.relion_has_high_fsc_at_limit,
            ave_pmax=carry.state.ave_Pmax, iteration=iteration, box_size=ctx.image_geometry.box_size, log=logger,
        )

        # RELION updates image_coarse_size before updateAngularSampling at the
        # start of expectation(). Preserve that incoming sampling order even
        # when replay/native scheduling advances state.healpix_order below.
        coarse_size_healpix_order = resolution.relion_expectation_coarse_size_order(
            state_healpix_order=carry.state.healpix_order, replay_saved_healpix_order=carry.replay_saved_healpix_order,
        )

        source.state_swap_snapshot(
            iteration,
            ports.ScoringState(
                carry.state, current_size, reference_model, carry.noise_model, halves, carry.previous_best_rotations,
                carry.sigma_offset,
                carry.direction_priors,
            ),
        )
        # The input source supplies the state this iteration scores with (the native source: the run's own).
        numbered = source.numbered_state(
            iteration,
            ports.NumberedState(
                current_size=current_size, noise_model=carry.noise_model, sigma_offset=carry.sigma_offset,
                previous_best_rotations=carry.previous_best_rotations, mean_variance=reference_model.tau2,
                class_mixture=carry.class_mixture, prior_translations=None, sampling_healpix_order=None,
            ),
            state=carry.state, halves=halves, direction_priors=carry.direction_priors,
            image_geometry=ctx.image_geometry,
        )
        current_size = numbered.current_size
        reference_model.tau2 = numbered.mean_variance
        carry = replace(
            carry, previous_best_rotations=numbered.previous_best_rotations, noise_model=numbered.noise_model,
            class_mixture=numbered.class_mixture, sigma_offset=numbered.sigma_offset,
        )
        if carry.replay_saved_healpix_order is not None:
            carry = replace(carry, replay_saved_healpix_order=int(carry.state.healpix_order))

        reference_model.maps = source.scoring_references(iteration, reference_model, volume_shape=ctx.volume_shape)
        swapped = source.swapped_state(
            iteration,
            ports.ScoringState(
                carry.state, current_size, reference_model, carry.noise_model, halves, carry.previous_best_rotations,
                carry.sigma_offset,
                carry.direction_priors,
            ),
            volume_shape=ctx.volume_shape,
        )
        if swapped is not None:
            current_size = swapped.current_size
            reference_model = swapped.reference_model
            carry = replace(
                carry, noise_model=swapped.noise_model, previous_best_rotations=swapped.previous_best_rotations,
                sigma_offset=swapped.sigma_offset, direction_priors=swapped.direction_priors,
            )
            history.state_swap_probe_applied_relion_iterations.append(numbered_relion_iteration)
        checked = source.scoring_state_checked(
            iteration,
            ports.ScoringArrays(
                reference_model, carry.noise_model, carry.sigma_offset, halves, carry.direction_priors,
                experiment_datasets,
            ),
        )
        if checked is not None:
            frozen_initial_scoring_state_sha256 = checked
        # The iteration's identity, its decisions taken at the top and its current size (final after the ports).
        this_iteration = iteration_planning.NumberedIteration(
            iteration=iteration, numbered_relion_iteration=numbered_relion_iteration,
            has_previous_iteration=has_previous_iteration,
            first_iteration=iteration_planning.first_iteration_policy(options, iteration=iteration), seeding=seeding,
            current_size=current_size,
        )

        # Half 1's projector of this iteration's references, built for the
        # expected-accuracy estimate and reused by the scoring projector setup
        # below: RELION computes each class's projector once per iteration.
        # Release the previous iteration's projector before the estimate builds this one.
        shared_projector_half1 = None
        iteration_accuracy, shared_projector_half1 = expected_accuracy.estimate_iteration_accuracy(
            expected_accuracy_inputs, reference_model.maps[0], best_eulers_deg=halves[0].rotation_eulers,
            class_assignments=carry.class_assignments[0], class_weights=carry.class_mixture.weights,
            sigma2_noise_native=carry.noise_model.radial_per_half[0], current_size=this_iteration.current_size,
            accuracy_image_size=iteration_planning.strict_e_step_size(this_iteration.current_size, ctx.optics, options),
            image_box_size=ctx.image_geometry.box_size, n_classes=options.k_class.n_classes, iteration=iteration,
            native_sampling_boundary=carry.native_sampling_boundary,
            relion_firstiter_cc_this_iter=this_iteration.first_iteration.relion_firstiter_cc,
            build_shared_projector=this_iteration.has_previous_iteration, log=logger,
        )
        if iteration_accuracy.sampling_accuracy is not None:
            # The estimate, or infinity when it was due and could not be made (convergence stays fail-closed).
            carry.state.acc_rot, carry.state.acc_trans = iteration_accuracy.sampling_accuracy
        if iteration_accuracy.published:
            carry = replace(carry, published_accuracy=expected_accuracy.PublishedAccuracy(
                iteration_accuracy.trial_local_indices, iteration_accuracy.trial_particle_ids,
                iteration_accuracy.acc_rot_per_class.copy(), iteration_accuracy.acc_trans_per_class_angstrom.copy(),
            ))

        # Accuracy and the preceding iteration's stall counters select this
        # expectation's grid; completed-iteration updates remain after M-step.
        carry = replace(carry, state=numbered_transitions.advance_expectation_sampling(
            carry.state, options.adaptive, iteration=iteration,
            may_advance_natively=this_iteration.has_previous_iteration and numbered_transitions.uses_native_auto_refine(
                native_sampling_boundary=carry.native_sampling_boundary, n_classes=options.k_class.n_classes,
            ),
            log=logger,
        ))

        history.record_scheduling(
            this_iteration.current_size, carry.state.healpix_order, carry.sigma_offset.shared_angstrom,
            list(carry.sigma_offset.per_half_angstrom),
        )
        scoring_current_size = int(this_iteration.current_size)

        logger.info(
            "=== RELION Iteration %d/%d: current_size=%d, healpix_order=%d, local_search=%s ===",
            this_iteration.numbered_relion_iteration, int(options.schedule.init_relion_iteration) + int(options.schedule.max_iter),
            scoring_current_size, carry.state.healpix_order, carry.state.do_local_search,
        )

        # --- Rotation grid at the state's order: from order 5 the base grid stays at order 4 (the full order-5
        # grid has 2.4M+ rotations) and local search plus oversampling give the finer steps. ---
        carry = replace(carry, coarse_grids=source.coarse_grids(
            iteration, carry.coarse_grids, carry.state, voxel_size=ctx.source_pixel_size_angstrom,
            dtype=ctx.scoring_dtype,
        ))
        carry = replace(carry, coarse_grids=trial_grids.refresh_coarse_grids(
            carry.coarse_grids, carry.state, options, voxel_size=ctx.source_pixel_size_angstrom,
            dtype=ctx.scoring_dtype, log=logger,
        ))

        # --- Local angular search: each image searches around its previous exact rotation at the current order ---
        if carry.state.do_local_search:
            # A half without poses is centred at Euler angles (0, 0, 0) with zero offsets.
            for half in halves:
                half.centre_absent_poses(offset_dims=ctx.offset_dims)
        use_local = carry.state.do_local_search
        if use_local:
            mode.refuse_local_search(options)
        # --- RELION's SamplingPerturbation of the trial grid (healpix_sampling.cpp:1810-1820, 1909-1934): a rigid
        # rotation applied after oversampling; at OS0 the coarse grid is the trial grid. ---
        perturbation = source.random_perturbation(iteration)
        if perturbation is None:
            perturbation = iteration_planning.resolve_numbered_perturbation(
                carry.random_perturbation, options, iteration=iteration, rng=ctx.perturb_rng, log=logger,
            )
        carry = replace(carry, random_perturbation=perturbation)
        plan = local_sampling.plan_numbered_sampling(
            ctx, carry, options, first_iteration=this_iteration.first_iteration,
            replayed_sampling_healpix_order=numbered.sampling_healpix_order,
            coarse_size_healpix_order=coarse_size_healpix_order, current_size=scoring_current_size,
            sealed_sampling_state=source.sealed_sampling_state, log=logger,
        )
        carry = replace(carry, coarse_grids=plan.coarse_grids)
        # The input source's scoring rotation ids (None: the grid's own) and the adaptive pass-1 width.
        # Two-pass adaptive oversampling: pass 1 at a reduced size finds the significant orientations, pass 2
        # scores them oversampled at current_size. Off the adaptive route coarse_cs is None (pass 1, where a
        # scorer has one, takes the full window).
        coarse_image_plan = (
            source.adaptive_coarse_size(
                iteration_planning.plan_adaptive_image_size(
                    coarse_size_healpix_order, plan.sampling_plan.windows, ctx.optics, options, log=logger,
                ),
                model_size=plan.sampling_plan.windows.model_size,
            )
            if plan.use_adaptive
            else None
        )
        plan = replace(
            plan, scoring_rotation_ids=source.scoring_rotation_ids(plan.trial_grid, use_local=use_local),
            coarse_angular_step_deg=None if coarse_image_plan is None else coarse_image_plan.angular_step_deg,
            coarse_cs=iteration_planning.adaptive_pass1_size(
                coarse_image_plan, plan.sampling_plan.windows, carry.state, options,
                box_size=ctx.image_geometry.box_size, log=logger,
            ),
            replay_prior_translations=numbered.prior_translations,
        )

        # The iteration-start curve RELION's scale XA/AA shell gate reads (the scheduling curve changes later).
        scale_correction_data_vs_prior_this_iter = carry.previous_data_vs_prior_for_scheduling
        # Every dense, local and tomo scorer reads this projector: pass 1 scores RELION's exact
        # coarse operands on every route, as RELION builds Projector::data every iteration.
        projectors = projector_preparation.build_numbered_projectors(
            halves, reference_model.maps, ctx.reconstruction_settings,
            current_size=plan.sampling_plan.windows.model_window_size,
            n_classes=options.k_class.n_classes, reusable_half1=shared_projector_half1,
            real_references_by_half=initial_real_references_by_half if iteration == 0 else None, iteration=iteration,
            log=logger,
        )
        # The start-up real maps serve iteration 0's projector only (relax#26).
        initial_real_references_by_half = [None, None]
        scoring_inputs = expectation.numbered_half_inputs(
            ctx, carry, plan, options, halves=halves, reference_model=reference_model, projectors=projectors,
            follower_setup=follower_setup,
        )
        # --- E-step on each half: one pass (adaptive_oversampling=0) or coarse then fine (>=1) ---
        expected = expectation.run_numbered_expectation(
            ctx, carry, plan, this_iteration, options, half_inputs=scoring_inputs, history=history,
            diagnostic_half_indices=iteration_diagnostics._significance_dump_half_indices(
                numbered_iteration=this_iteration.numbered_relion_iteration, n_classes=options.k_class.n_classes,
                experiment_datasets=experiment_datasets,
            ),
        )
        # The assignments outlive the iteration: the result, the next accuracy estimate and the final pass read them.
        carry = replace(
            carry, hard_assignments=expected.per_half.hard_assignments, class_assignments=expected.per_half.class_assignments,
        )
        # E-step + per-half M-step accumulators are now both populated.
        observer.stage_finished(iteration, "e_step")
        # The Class3D prior reads half 1's projected reference power (none before a previous iteration's).
        projector_power_spectrum = (
            None if not this_iteration.has_previous_iteration or projectors[0] is None else projectors[0].power_spectrum
        )
        # Release this iteration's projectors now, not when the next iteration builds its own (code rule 3: an
        # earlier release), and the inputs' references to its maps, which the M-step releases.
        scoring_inputs = projectors = shared_projector_half1 = None


        if options.local_search.stops_after_local_search and use_local:
            elapsed = iteration_clock.seconds
            logger.info(
                "Stopping after local-search diagnostic at iteration %d: profiles=%d score_only=%s wall=%.1fs",
                iteration + 1, len(history.local_profile_history),
                bool(options.local_search.stop_after_local_search_score_only), elapsed,
            )
            profile_stop = refinement_result.ProfileStop(
                score_only=bool(options.local_search.stop_after_local_search_score_only), wall_seconds=elapsed,
                significant_count=expected.significance.recorded,
            )
            break
        scored_class_weights = carry.class_mixture.weights  # the weights this expectation scored with
        carry = mode.record_class_weights(carry, options, history, expected.per_half)
        operands = maximization.mstep_operands(
            expected.per_half, padded_volume_shape=ctx.padded_volume_shape, halves=halves,
            image_current_size=plan.sampling_plan.windows.image_current_size,
            projector_power_spectrum=projector_power_spectrum,
        )

        # The raw half accumulators, before any join: the observer sees them, then the finite guard checks them
        # (a report names the half that is actually damaged, not the one a join copied it into).
        observer.half_accumulators_ready(
            iteration, numerators=operands.numerators, denominators=operands.denominators,
            settings=ctx.reconstruction_settings, current_size=this_iteration.current_size,
            accumulator_shape=operands.accumulator_shape,
            pixel_size_angstrom=ctx.source_pixel_size_angstrom,
        )
        reconstruction_diagnostics.check_half_accumulators_before_join(
            operands.numerators, operands.denominators, iteration=iteration,
            init_relion_iteration=options.schedule.init_relion_iteration, log=logger,
        )

        # --- RELION-exact M-step: K=1 on the split-half auto-refine path (compareTwoHalves -> updateSSNRarrays
        # -> reconstruct); Class3D joins the halves per class, carries the previous Iref power spectrum forward
        # as tau2 and solves once per class. mstep is the mode's record (ClassMaximization or K1Maximization). ---
        carry, mstep = mode.maximize(
            ctx, carry, options, history, this_iteration, operands, reference_model, source, scored_class_weights
        )
        numerators, denominators = mode.solved_accumulators(mstep, operands)
        observer.stage_finished(iteration, "recon")

        history.significant_counts.append(expected.significance.recorded)
        _warn_if_significant_samples_jumped(history.significant_counts, iteration)

        history.record_rotation_posterior(expected.per_half.rotation_posterior)
        # pdf_direction: each half's next direction prior from this iteration's posteriors (None: kept).
        learned_priors = mode.learned_direction_priors(ctx, options, expected.per_half, plan, use_local)
        for half_index, learned in enumerate(learned_priors):
            if learned is not None:
                carry.direction_priors[half_index] = learned
        if this_iteration.seeding.single_class_iteration:
            mstep, copied_mixture = maximization.copy_first_class_to_every_class(
                reference_model, carry.direction_priors, mstep, carry.class_mixture,
                n_classes=options.k_class.n_classes,
            )
            history.data_vs_prior_trajectory[-1] = mstep.data_vs_prior
            carry = replace(
                carry, class_mixture=copied_mixture, previous_data_vs_prior_for_scheduling=mstep.data_vs_prior,
            )
        mode.record_direction_prior(history, carry.direction_priors)

        # --- Unregularized half-maps, reconstructed only for the run files and the observer ---
        need_unreg_means = (
            observer.wants_unfiltered_maps(this_iteration.numbered_relion_iteration)
            or (
                options.checkpoint.writer is not None
                and options.checkpoint.writer.wants_unfiltered_maps(
                    this_iteration.numbered_relion_iteration, n_classes=options.k_class.n_classes
                )
            )
        )
        unreg_clock = timing.Stopwatch()
        unreg_means = mode.unfiltered_maps(
            ctx, options, reference_model, mstep, operands.accumulator_shape, need_unreg_means,
        )
        logger.info(
            "Unregularized reconstruction (2 halves): %.1fs%s", unreg_clock.seconds,
            "" if need_unreg_means else " (skipped; diagnostics disabled)",
        )

        # K=1's FSC also drives size growth; Class3D has none (its shared per-class curve drives growth).
        fsc = mode.fsc(mstep)
        history.record_fsc(fsc, fsc)
        observer.stage_finished(iteration, "fsc")

        observer.maps_reconstructed(ports.ReconstructedIteration(
            iteration, numerators=numerators, denominators=denominators, reference_model=reference_model,
            noise_model=carry.noise_model, per_half=expected.per_half, trial_grid=plan.trial_grid, sampling_plan=plan.sampling_plan,
            options=options, unfiltered_maps=unreg_means, fsc=fsc, current_size=this_iteration.current_size, state=carry.state,
            volume_shape=ctx.volume_shape, voxel_size=ctx.source_pixel_size_angstrom,
        ))

        # --- This expectation's particle statistics: joined assignments, posterior maxima, optimizer Pmax ---
        statistics = convergence_helpers.expectation_statistics(
            expected.per_half, carry.previous_assignments,
            mode.pmax_normalization_mass(expected.per_half),
        )
        mode.log_statistics(statistics, expected.per_half)
        history.record_pmax(statistics.ave_pmax, statistics.ave_pmax_mass, statistics.max_posterior.copy())
        history.record_pass2_engines(engine_record.take_pass_engines())
        history.record_coarse_engines(engine_record.take_coarse_engine_calls())

        # --- Resolution from updated FSC-derived SSNR (RELION auto-refine) ---
        resolution_estimate, class_change_fraction = mode.resolution(
            ctx, carry, options, history, this_iteration, mstep,
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
        pose_update = particle_poses.prepare_particle_pose_update(
            expected.per_half, halves, carry.coarse_grids.translations, previous_rotations=carry.previous_best_rotations,
            local_sampling=plan.sampling_plan.local if use_local else None, dtype=ctx.scoring_dtype,
        )
        carry = replace(carry, previous_best_rotations=[poses.rotations for poses in pose_update.current])
        for half, poses in zip(halves, pose_update.current, strict=True):
            half.rotation_eulers = poses.eulers_deg
            half.translations = poses.translations_pixels
        history.record_pose_history(
            [np.asarray(poses.eulers_deg).copy() for poses in pose_update.current],
            [np.asarray(poses.translations_pixels).copy() for poses in pose_update.current],
        )
        observer.poses_updated(
            iteration, poses=pose_update.current, per_half=expected.per_half, significance=expected.significance,
            datasets=experiment_datasets,
        )

        pose_comparison = particle_poses.prepare_pose_comparison(
            pose_update, translation_dimension=ctx.offset_dims, dtype=ctx.scoring_dtype, log=logger,
        )

        carry = mode.scheduling_curve_after_resolution(ctx, carry, history, resolution_estimate)

        # RELION's posterior-weighted noise update: the radial sigma2_noise and the engine's pixel rows.
        noise_update = mode.update_noise_variance(
            expected.per_half.noise_stats,
            carry.noise_model,
            ctx.image_geometry.image_shape,
            firstiter_cc=this_iteration.first_iteration.relion_firstiter_cc,
            ctf_premultiplied=noise_updates.datasets_store_premultiplied_ctf(experiment_datasets),
            summed_current_size=(
                plan.sampling_plan.windows.image_window_size if ctx.consistency.noise_shell_count == "summed" else None
            ),
            nyquist_column_counting=ctx.consistency.nyquist_column_counting,
        )
        if not this_iteration.first_iteration.relion_firstiter_cc:
            observer.noise_updated(
                iteration, current_size=this_iteration.current_size, image_shape=ctx.image_geometry.image_shape,
                noise_stats_per_half=expected.per_half.noise_stats,
                previous_noise_radial_per_half=carry.noise_model.radial_per_half,
                noise_from_res_per_half=noise_update.noise_from_res_per_half,
                noise_from_res=noise_update.noise_from_res,
            )
            observer.stage_finished(iteration, "noise_update")
        noise_from_res = noise_update.noise_from_res
        noise_from_res_per_half = noise_update.noise_from_res_per_half
        carry = replace(carry, noise_model=noise_update.model)

        correction_report = relion_normalization.NormScaleCorrectionReport()
        norm_scale_update = relion_normalization.numbered_norm_scale_update(
            expected.per_half, halves, firstiter_cc=this_iteration.first_iteration.relion_firstiter_cc, do_norm_correction=not ctx.tomo_halves,
            do_scale_correction=follower_setup.follower_scale_state is None, dtype=ctx.scoring_dtype,
            iteration=iteration, current_size=this_iteration.current_size,
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
                group_scale_corrections = relion_worker_scale._update_relion_follower_corrections(
                    follower_setup, noise_stats_per_half=expected.per_half.noise_stats, norm_scale_update=norm_scale_update,
                    relion_half_inputs=halves, relion_firstiter_cc_this_iter=this_iteration.first_iteration.relion_firstiter_cc,
                    dtype=ctx.scoring_dtype, logger=logger,
                )
            correction_report = relion_normalization.norm_scale_report(norm_scale_update, group_scale_corrections)
            relion_normalization.log_norm_scale_update(norm_scale_update, log=logger)
        if follower_setup.follower_scale_state is not None:
            history.relion_scale_follower_scales_numbered_post_mstep_trajectory.append(
                np.asarray(follower_setup.follower_scale_state.scales, dtype=np.float64).copy()
            )

        # The iteration's sigma2 shells (after the noise update) and the tau2 ingredients of its Wiener update.
        history.record_noise_and_tau2(noise_from_res, noise_from_res_per_half, mstep.tau2_update_details)

        # --- Convergence state: assignment changes, resolution stalls, angular-step refinement ---
        updated_state, accuracy_replay = numbered_transitions.update_iteration_convergence(
            carry.state, pose_comparison, options, image_geometry=ctx.image_geometry, iteration=iteration,
            sampling_decision_now=mode.sampling_decision_now(carry), class_change_fraction=class_change_fraction,
            scheduling_resolution_shell=resolution_estimate.scheduling_shell, source=source,
            translations=carry.coarse_grids.translations, statistics=statistics,
            significant_counts=expected.significance.convergence,
            exact_acc_rot=iteration_accuracy.acc_rot, exact_acc_trans=iteration_accuracy.acc_trans_angstrom, log=logger,
        )
        carry = mode.after_convergence(replace(carry, state=updated_state), options, iteration)

        # Reuse the assignment statistic computed by update_refinement_state.
        # Sampling transitions and optimiser replay preserve this field.
        history.frac_changed_trajectory.append(float(carry.state.fraction_changed))

        # --- sigma2_offset from the posterior-weighted offsets (RELION parity, C1) ---
        sigma_offset_result = sigma_offset.update_c1_sigma_offset_from_posterior(
            expected.per_half, carry.sigma_offset, n_classes=options.k_class.n_classes,
            state_fallback_offsets_angstrom=carry.state.current_changes_optimal_offsets_angstrom,
            offset_dims=ctx.offset_dims,
        )
        carry = replace(carry, sigma_offset=sigma_offset.SigmaOffset(
            sigma_offset_result.current_sigma_offset_angstrom, sigma_offset_result.current_sigma_offset_angstrom_per_half,
        ))
        per_class_sigma_offset = sigma_offset_result.per_class_sigma_offset_angstrom
        history.record_sigma_offset_update(
            carry.sigma_offset.shared_angstrom, list(carry.sigma_offset.per_half_angstrom),
            None if per_class_sigma_offset is None else per_class_sigma_offset.tolist(),
        )
        history.record_pose_accuracy_diagnostics(
            accuracy_replay, iteration_accuracy, carry.state, n_classes=options.k_class.n_classes
        )

        # The next iteration's change tracking reads the coarse-grid assignments (trial-grid indices on every
        # route, adaptive or not).
        carry = replace(
            carry, previous_assignments=[ha.copy() if ha is not None else None for ha in expected.per_half.coarse_ha],
            previous_class_assignments=[cls.copy() if cls is not None else None for cls in carry.class_assignments],
        )
        observer.stage_finished(iteration, "convergence")

        # --- RELION's run_itNNN files (ml_optimiser.cpp:3489) ---
        checkpoint_writer = options.checkpoint.writer
        if checkpoint_writer is not None and checkpoint_writer.due(this_iteration.numbered_relion_iteration):
            snapshot = mode.run_files_snapshot(
                ctx, carry, this_iteration, reference_model=reference_model, halves=halves, mstep=mstep,
                unfiltered_maps=unreg_means, expected=expected, corrections=correction_report,
            )
            checkpoint_writer(snapshot)

        observer.iteration_finished(ports.FinishedIteration(
            iteration, init_relion_iteration=options.schedule.init_relion_iteration, state=carry.state,
            current_size=this_iteration.current_size, sigma_offset_angstrom=carry.sigma_offset.shared_angstrom,
            random_perturbation=carry.random_perturbation, settings=ctx.reconstruction_settings,
            pixel_size_angstrom=ctx.source_pixel_size_angstrom, ave_pmax=statistics.ave_pmax, fsc=fsc,
            noise_variance=carry.noise_model.average_variance, means=reference_model.maps, unfiltered_means=unreg_means,
            poses=pose_update.current, half_inputs=halves, corrections=correction_report,
            scale_correction_data_vs_prior=scale_correction_data_vs_prior_this_iter,
        ))

        elapsed = iteration_clock.seconds
        history.wall_times.append(elapsed)

        res_angstrom = resolution.shell_index_to_resolution_angstrom(
            resolution_estimate.scheduling_shell, ctx.image_geometry.image_shape[0], ctx.source_pixel_size_angstrom,
        )
        logger.info(
            "RELION Iteration %d: current_size=%d, pixel_res=%.1f, "
            "res=%.2f A, ave_Pmax=%.4f, healpix_order=%d, "
            "converged=%s, time=%.1fs",
            this_iteration.numbered_relion_iteration, this_iteration.current_size, resolution_estimate.scheduling_shell, res_angstrom,
            statistics.ave_pmax,
            carry.state.healpix_order, carry.state.has_converged, elapsed,
        )

        # End-of-iteration memory boundary. The next iteration pads each half-map to the projection
        # grid; keeping the previous accumulators (also in the pass outputs), unregularized maps or the
        # run-files snapshot live can make high-resolution runs OOM before the batch-size estimator acts
        # (41 GB of host carried into the final pass at box 800).
        jax.block_until_ready(reference_model.maps)
        numerators = denominators = None
        unreg_means = mstep = snapshot = None
        # Pass containers must not retain the previous grids while the next projector is built.
        expected = projector_power_spectrum = operands = None
        if options.execution.clear_jax_caches_between_iterations:
            jax.clear_caches()

        if carry.state.has_converged and not options.schedule.force_max_iter_after_convergence:
            logger.info(
                "Convergence reached at iteration %d. Final resolution: %.2f A (pixel_res=%.1f)", iteration + 1,
                res_angstrom, resolution_estimate.scheduling_shell,
            )
            break
        if carry.state.has_converged:
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
    numbered = refinement_result.NumberedMetadata(
        hard_assignments=carry.hard_assignments,
        frozen_initial_scoring_state_sha256=frozen_initial_scoring_state_sha256,
        expected_accuracy_trial_local_indices=carry.published_accuracy.trial_local_indices,
        expected_accuracy_trial_particle_ids=carry.published_accuracy.trial_particle_ids,
        setup_phase_seconds=setup_phase_seconds,
    )
    if profile_stop is not None:
        # Local search is K=1 (Class3D was rejected above), so there are no class products.
        return refinement_result.RefinementResult(
            maps=finalization.numbered_k1_maps(reference_model.maps),
            replay=_follower_replay_telemetry(follower_scale_replay, history), follower_scale=None,
            convergence_state=carry.state, numbered=numbered, history=history, profile_stop=profile_stop,
        )
    if not mode.final_pass_due(carry.state, options, iteration):
        return refinement_result.RefinementResult(
            maps=mode.numbered_maps(reference_model, carry),
            replay=_follower_replay_telemetry(follower_scale_replay, history),
            follower_scale=follower_setup.result_outputs(history), convergence_state=carry.state, numbered=numbered,
            history=history,
        )
    # --- RELION's final iteration (do_join_random_halves + do_use_all_data, ml_optimiser.cpp:10157-10160 and
    # 5707-5708): one more E+M at full Nyquist, each half against its own map, the weighted sums joined. ---
    final_join_means = mode.final_join_means(reference_model, options)
    final_state = source.final_state(
        ports.FinalState(final_join_means, carry.sigma_offset, carry.noise_model),
        means=reference_model.maps, numbered_iteration_count=len(history.current_sizes), halves=halves,
        direction_priors=carry.direction_priors, healpix_order=carry.state.healpix_order,
        image_geometry=ctx.image_geometry,
    )
    # The final pass scores with the source's state (its own, natively); the replaced numbered values are released.
    final_join_means = final_state.join_means
    carry = replace(carry, sigma_offset=final_state.sigma_offset, noise_model=final_state.noise_model)
    if follower_setup.follower_scale_state is not None:
        relion_worker_scale._dispatch_relion_follower_scale_for_final_all_data(
            follower_setup, init_relion_iteration=options.schedule.init_relion_iteration,
            numbered_iteration_count=len(history.current_sizes), relion_half_inputs=halves, dtype=ctx.scoring_dtype,
            logger=logger,
        )
    final_use_local = mode.final_use_local(carry)
    if final_use_local:
        for half in halves:
            half.require_local_search_poses()
    # Each half is still scored with its own numbered-iteration noise (one MPI follower per half); K=1 joins
    # the half sums after this E-step.
    final_result = finalization.run_final_all_data(
        halves, ctx=ctx, carry=carry, reference_model=reference_model, history=history, options=options,
        follower_setup=follower_setup, expected_accuracy_inputs=expected_accuracy_inputs,
        final_join_means=final_join_means, final_use_local=final_use_local, source=source, iteration=iteration,
    )
    return replace(final_result, numbered=numbered, replay=_follower_replay_telemetry(follower_scale_replay, history))

### THIS FILE SHOULD BE MUCH SHORTER - REMOVE UN-NECESSARY IF STATEMENTS/RELION THINGS THAT DONT MATTER/ WE DONT USE
## MOVE THINGS AWAY TO DIFFERNET FILE E.G. I/O PERHAPS. IT SHOULD BE EASY TO UDNERSTAND WHERE THE MAIN ENGINE OF ITER IS GOING, WHAT ARE THE MAIN STEPS (E-M ACCUMULATION) POSTPROCESSING, NOISE UPDATING, ETC
