"""Convergence and sampling transitions at their numbered-iteration boundaries."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from relax.helpers.convergence import (
    ExpectationStatistics,
    RefinementState,
    apply_relion_healpix_order_oracle,
    approx_acc_rot_policy_for_convergence,
    calculate_expected_angular_errors,
    update_angular_sampling,
    update_refinement_state,
)
from relax.helpers.resolution import ImageGeometry, shell_index_to_resolution_angstrom
from relax.refinement.ports import InputSource, OptimiserAccuracyReplay

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
    state = apply_relion_healpix_order_oracle(
        state, target_order, iteration_number=iteration + 1,
    )
    log.info(
        "HEALPix-order oracle: iteration %d using healpix_order=%d",
        iteration + 1, target_order,
    )
    return state


def advance_expectation_sampling(
    state: RefinementState,
    adaptive: AdaptiveOptions,
    *,
    iteration: int,
    may_advance_natively: bool,
    log: logging.Logger,
) -> RefinementState:
    """Choose this expectation's angular grid after accuracy, before expectation.

    An explicit HEALPix schedule takes precedence, including on the first iteration. Otherwise the
    grid advances only where ``may_advance_natively``: native K=1 auto-refine after a completed
    iteration, from its stall counters (the controller decides; Class3D keeps its grid).
    See ``docs/math/relion_refinement_algorithm.md#iteration-convergence-policy``.
    """
    if adaptive.relion_healpix_orders is not None:
        return _scheduled_healpix_order_state(state, adaptive, iteration=iteration, log=log)
    return update_angular_sampling(state) if may_advance_natively else state


class ConvergenceUpdate(NamedTuple):
    """Updated counters/sampling state and the accuracy used for reporting."""

    state: RefinementState
    accuracy: OptimiserAccuracyReplay


def _iteration_accuracy_for_convergence(
    healpix_order: int,
    *,
    iteration: int,
    source: InputSource,
    n_translations: int,
    significant_counts,
    exact_acc_rot: float | None,
    exact_acc_trans: float | None,
    approx_acc_rot_for_convergence: bool,
    log: logging.Logger,
) -> OptimiserAccuracyReplay:
    """Admit this iteration's accuracy: the exact estimate, the support width when opted in, then the input
    source's (a replayed optimiser's); natively nothing admits a translation accuracy for convergence."""
    convergence_acc_rot = None
    if significant_counts is not None and len(significant_counts) > 0:
        approx_acc_rot, _ = calculate_expected_angular_errors(
            healpix_order,
            significant_counts,
            n_translations=n_translations,
        )
        approx_for_convergence, approx_convergence_reason = approx_acc_rot_policy_for_convergence(
            approx_acc_rot_for_convergence
        )
        if approx_for_convergence and exact_acc_rot is None:
            convergence_acc_rot = approx_acc_rot
        log.info(
            "approx_acc_rot=%.3f deg (from %d images, mean n_sig=%.1f, convergence=%s)",
            approx_acc_rot,
            len(significant_counts),
            float(np.mean(significant_counts)),
            approx_convergence_reason,
        )

    return source.convergence_accuracy(
        iteration,
        OptimiserAccuracyReplay(
            metadata=None,
            optimiser_star=None,
            optimiser_iteration=None,
            acc_rot=exact_acc_rot,
            acc_trans=exact_acc_trans,
            convergence_acc_rot=convergence_acc_rot,
            convergence_acc_trans=None,
        ),
    )


def update_iteration_convergence(
    state: RefinementState,
    pose_comparison: PoseComparison,
    options: RefinementOptions,
    *,
    image_geometry: ImageGeometry,
    iteration: int,
    sampling_decision_now: bool,
    class_change_fraction: float,
    source: InputSource,
    scheduling_resolution_shell: float,
    translations,
    statistics: ExpectationStatistics,
    significant_counts,
    exact_acc_rot: float | None,
    exact_acc_trans: float | None,
    log: logging.Logger,
) -> ConvergenceUpdate:
    """Update completed-iteration statistics, then apply optimiser controls.

    Reads from ``statistics`` (the expectation's ``ExpectationStatistics``): ``assignments``,
    ``previous_assignments``, ``max_posterior`` and ``ave_pmax``.

    ``sampling_decision_now``: advance the sampling and test convergence in this update. Native K=1
    auto-refine does both at the next permitted loop boundary instead (this preserves that timing), and
    Class3D never does; the controller passes True only for K=1 at a replayed boundary.
    ``class_change_fraction``: Class3D's fraction of hard class changes (0.0 for K=1). Approximate support
    accuracy only drives convergence when explicitly enabled and exact accuracy is absent.
    See ``docs/math/relion_refinement_algorithm.md#iteration-convergence-policy``.
    """
    n_trans_current = translations.shape[0]
    new_res_angstrom = shell_index_to_resolution_angstrom(
        scheduling_resolution_shell,
        image_geometry.box_size,
        image_geometry.pixel_size_angstrom,
    )
    accuracy_replay = _iteration_accuracy_for_convergence(
        state.healpix_order,
        iteration=iteration,
        source=source,
        n_translations=n_trans_current,
        significant_counts=significant_counts,
        exact_acc_rot=exact_acc_rot,
        exact_acc_trans=exact_acc_trans,
        approx_acc_rot_for_convergence=options.variants.approx_acc_rot_for_convergence,
        log=log,
    )
    state = update_refinement_state(
        state,
        current_assignments=statistics.assignments,
        previous_assignments=statistics.previous_assignments,
        n_translations=n_trans_current,
        translations=translations,
        new_resolution=new_res_angstrom,
        max_posterior_per_image=statistics.max_posterior,
        acc_rot=accuracy_replay.convergence_acc_rot,
        acc_trans=accuracy_replay.convergence_acc_trans,
        current_rotation_matrices=pose_comparison.current_rotations,
        previous_rotation_matrices=pose_comparison.previous_rotations,
        current_translations_pixel=pose_comparison.current_translations_pixels,
        previous_translations_pixel=pose_comparison.previous_translations_pixels,
        current_changes_classes=class_change_fraction,
        ave_pmax_override=statistics.ave_pmax,
        voxel_size_angstrom=image_geometry.pixel_size_angstrom,
        update_sampling=sampling_decision_now,
        check_convergence_now=sampling_decision_now,
        symmetry_label=options.symmetry.point_group,
    )
    source.apply_optimiser_controls(iteration, state, accuracy_replay)
    return ConvergenceUpdate(state, accuracy_replay)


def reset_follower_counter_once(state: RefinementState, options: RefinementOptions, *, iteration: int) -> RefinementState:
    """Fresh auto-refine followers reset the hidden-variable counter before iteration 2
    (ml_optimiser_mpi.cpp:1233-1234); the controller calls this after a K=1 update (Class3D does not reset)."""
    if (
        iteration == 0
        and options.checkpoint.resume is None
        and int(options.schedule.init_relion_iteration) == 0
    ):
        return replace(state, suppress_hidden_variable_increment_once=True)
    return state
