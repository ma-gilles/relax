"""Convergence and sampling transitions at their numbered-iteration boundaries."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from relax.diagnostics.relion_replay import (
    OptimiserAccuracyReplay,
    apply_optimiser_convergence_replay,
    read_optimiser_accuracy_replay,
)
from relax.helpers.convergence import (
    RefinementState,
    _apply_relion_healpix_order_oracle,
    _approx_acc_rot_policy_for_convergence,
    calculate_expected_angular_errors,
    update_angular_sampling,
    update_refinement_state,
)
from relax.helpers.resolution import ImageGeometry, shell_index_to_resolution_angstrom

if TYPE_CHECKING:
    from relax.refinement.half_inputs import PoseComparison
    from relax.refinement.refinement_options import AdaptiveOptions, RefinementOptions


def uses_native_auto_refine(*, native_sampling_boundary: bool, n_classes: int) -> bool:
    """Class3D reports accuracy but never advances auto-refine or converges."""
    return native_sampling_boundary and n_classes == 1


def _scheduled_healpix_order_state(
    state: RefinementState, adaptive: AdaptiveOptions, *, iteration: int, log: logging.Logger,
) -> RefinementState:
    """Install this iteration's order of an explicit HEALPix schedule."""
    target_order = int(adaptive.relion_healpix_orders[iteration])
    state = _apply_relion_healpix_order_oracle(
        state, target_order, iteration_number=iteration + 1,
    )
    log.info(
        "HEALPix-order oracle: iteration %d using healpix_order=%d",
        iteration + 1, target_order,
    )
    return state


def advance_k1_expectation_sampling(
    state: RefinementState,
    adaptive: AdaptiveOptions,
    *,
    iteration: int,
    has_previous_iteration: bool,
    native_sampling_boundary: bool,
    log: logging.Logger,
) -> RefinementState:
    """Choose the next K=1 angular grid after accuracy, before expectation.

    Native auto-refine uses the preceding iteration's stall counters. An
    explicit HEALPix schedule takes precedence, including on the first iteration.
    See ``docs/math/relion_refinement_algorithm.md#iteration-convergence-policy``.
    """
    if adaptive.relion_healpix_orders is not None:
        state = _scheduled_healpix_order_state(state, adaptive, iteration=iteration, log=log)
    elif has_previous_iteration and native_sampling_boundary:
        state = update_angular_sampling(state)
    return state


def advance_class_expectation_sampling(
    state: RefinementState,
    adaptive: AdaptiveOptions,
    *,
    iteration: int,
    log: logging.Logger,
) -> RefinementState:
    """Class3D keeps its angular grid unless an explicit HEALPix schedule sets this iteration's order."""
    if adaptive.relion_healpix_orders is not None:
        state = _scheduled_healpix_order_state(state, adaptive, iteration=iteration, log=log)
    return state


def advance_expectation_sampling(
    state: RefinementState,
    adaptive: AdaptiveOptions,
    *,
    iteration: int,
    has_previous_iteration: bool,
    native_sampling_boundary: bool,
    n_classes: int,
    log: logging.Logger,
) -> RefinementState:
    """The one remaining mode decision of the expectation sampling transition.

    K=1 may advance natively (``advance_k1_expectation_sampling``); Class3D
    never does (``advance_class_expectation_sampling``). Remove this dispatch
    when the K1 and Class3D trajectories call those directly.
    """
    if n_classes == 1:
        return advance_k1_expectation_sampling(
            state,
            adaptive,
            iteration=iteration,
            has_previous_iteration=has_previous_iteration,
            native_sampling_boundary=native_sampling_boundary,
            log=log,
        )
    return advance_class_expectation_sampling(state, adaptive, iteration=iteration, log=log)


class ConvergenceUpdate(NamedTuple):
    """Updated counters/sampling state and the accuracy used for reporting."""

    state: RefinementState
    accuracy: OptimiserAccuracyReplay


def update_iteration_convergence(
    state: RefinementState,
    pose_comparison: PoseComparison,
    options: RefinementOptions,
    *,
    image_geometry: ImageGeometry,
    iteration: int,
    native_sampling_boundary: bool,
    replay_dir: str | None,
    scheduling_resolution_shell: float,
    translations,
    current_assignments,
    previous_assignments,
    current_classes,
    previous_classes,
    max_posterior,
    ave_pmax: float,
    significant_counts,
    exact_acc_rot: float | None,
    exact_acc_trans: float | None,
    log: logging.Logger,
) -> ConvergenceUpdate:
    """Update completed-iteration statistics, then apply optimiser controls.

    Native auto-refine tests convergence at the next permitted loop boundary;
    this operation preserves that timing. Approximate support accuracy only
    drives convergence when explicitly enabled and exact accuracy is absent.
    See ``docs/math/relion_refinement_algorithm.md#iteration-convergence-policy``.
    """
    parity = options.parity
    init_relion_iteration = options.schedule.init_relion_iteration
    k_class_enabled = int(options.k_class.n_classes) > 1
    sealed_sampling_state = options.debug.sealed_sampling_state
    n_trans_current = translations.shape[0]
    new_res_angstrom = shell_index_to_resolution_angstrom(
        scheduling_resolution_shell,
        image_geometry.box_size,
        image_geometry.pixel_size_angstrom,
    )

    iter_acc_rot = exact_acc_rot
    iter_acc_trans = exact_acc_trans
    convergence_acc_rot = None
    convergence_acc_trans = None
    if significant_counts is not None and len(significant_counts) > 0:
        approx_acc_rot, _ = calculate_expected_angular_errors(
            state.healpix_order,
            significant_counts,
            n_translations=n_trans_current,
        )
        approx_for_convergence, approx_convergence_reason = _approx_acc_rot_policy_for_convergence()
        if approx_for_convergence and exact_acc_rot is None:
            convergence_acc_rot = approx_acc_rot
        log.info(
            "approx_acc_rot=%.3f deg (from %d images, mean n_sig=%.1f, convergence=%s)",
            approx_acc_rot,
            len(significant_counts),
            float(np.mean(significant_counts)),
            approx_convergence_reason,
        )

    accuracy_replay = read_optimiser_accuracy_replay(
        replay_dir=replay_dir,
        replay_prefix=parity.perturb_replay_relion_prefix,
        init_relion_iteration=init_relion_iteration,
        iteration=iteration,
        sealed_sampling_state=sealed_sampling_state,
        acc_rot=iter_acc_rot,
        acc_trans=iter_acc_trans,
        convergence_acc_rot=convergence_acc_rot,
        convergence_acc_trans=convergence_acc_trans,
        logger=log,
    )
    state = update_refinement_state(
        state,
        current_assignments=current_assignments,
        previous_assignments=previous_assignments,
        n_translations=n_trans_current,
        translations=translations,
        new_resolution=new_res_angstrom,
        max_posterior_per_image=max_posterior,
        acc_rot=accuracy_replay.convergence_acc_rot,
        acc_trans=accuracy_replay.convergence_acc_trans,
        current_rotation_matrices=pose_comparison.current_rotations,
        previous_rotation_matrices=pose_comparison.previous_rotations,
        current_translations_pixel=pose_comparison.current_translations_pixels,
        previous_translations_pixel=pose_comparison.previous_translations_pixels,
        current_classes=current_classes,
        previous_classes=previous_classes,
        ave_pmax_override=ave_pmax,
        voxel_size_angstrom=image_geometry.pixel_size_angstrom,
        update_sampling=not native_sampling_boundary and not k_class_enabled,
        check_convergence_now=not native_sampling_boundary and not k_class_enabled,
        symmetry_label=options.symmetry.point_group,
    )
    if (
        iteration == 0
        and options.checkpoint.resume is None
        and int(init_relion_iteration) == 0
        and not k_class_enabled
    ):
        # Fresh auto-refine followers reset the hidden-variable counter before
        # iteration 2 (ml_optimiser_mpi.cpp:1233-1234); Class3D does not.
        state = replace(state, suppress_hidden_variable_increment_once=True)
    if accuracy_replay.metadata is not None:
        apply_optimiser_convergence_replay(
            state,
            metadata=accuracy_replay.metadata,
            optimiser_star=accuracy_replay.optimiser_star,
            optimiser_iteration=accuracy_replay.optimiser_iteration,
            replay_dir=replay_dir,
            replay_prefix=parity.perturb_replay_relion_prefix,
            logger=log,
        )
    return ConvergenceUpdate(state, accuracy_replay)
