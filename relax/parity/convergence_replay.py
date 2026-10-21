"""RELION replay of the optimiser's convergence state: counters at a restart and accuracies per iteration.

``restore_convergence_state_from_replay_restart`` restores the convergence counters from RELION's optimiser
and model stars at a perturbation-replay restart; ``read_optimiser_accuracy_replay`` reads this iteration's
rotation and translation accuracies, and ``apply_optimiser_convergence_replay`` installs RELION's convergence
fields after the state update.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from relax.refinement.ports import OptimiserAccuracyReplay
from relax.star.metadata import read_relion_model_metadata, read_relion_optimiser_metadata

logger = logging.getLogger("relax.diagnostics.relion_replay")  # the name before the move to relax.parity: log rows keep it


def restore_convergence_state_from_replay_restart(state, replay, init_relion_iteration) -> None:
    """Restore convergence counters from a RELION optimiser/model STAR at a
    perturbation-replay restart iteration.

    RELION's convergence counters are not initialized against an infinite
    previous resolution -- a replay restart resumes them from the previous
    optimiser/model STAR instead of the fresh-run FSC/ini_high state. This is
    the loop-boundary counterpart to ``relion_replay.apply_iter_replay_overrides``,
    which handles the same ``perturb_replay_relion_dir`` source per mid-loop
    iteration.
    """
    init_opt_star = os.path.join(
        replay.perturb_replay_relion_dir,
        f"{replay.perturb_replay_relion_prefix}_it{init_relion_iteration:03d}_optimiser.star",
    )
    init_model_star = os.path.join(
        replay.perturb_replay_relion_dir,
        f"{replay.perturb_replay_relion_prefix}_it{init_relion_iteration:03d}_half1_model.star",
    )
    if os.path.exists(init_model_star):
        model_meta = read_relion_model_metadata(init_model_star)
        resolution_angstrom = float(model_meta["current_resolution"])
        if np.isfinite(resolution_angstrom) and resolution_angstrom > 0.0:
            state.current_resolution = resolution_angstrom
            state.previous_resolution = resolution_angstrom
    if os.path.exists(init_opt_star):
        opt_meta = read_relion_optimiser_metadata(init_opt_star)
        state.nr_iter_wo_resol_gain = int(opt_meta.get("number_iter_without_resolution_gain") or 0)
        hidden_variable_changes = int(opt_meta.get("number_iter_without_changing_assignments") or 0)
        state.nr_iter_wo_large_hidden_variable_changes = hidden_variable_changes
        state.nr_iter_wo_assignment_changes = hidden_variable_changes
        if opt_meta.get("overall_accuracy_rotations") is not None:
            state.acc_rot = float(opt_meta["overall_accuracy_rotations"])
        if opt_meta.get("overall_accuracy_translations_angst") is not None:
            state.acc_trans = float(opt_meta["overall_accuracy_translations_angst"])
        if opt_meta.get("smallest_changes_orientations") is not None:
            state.smallest_changes_optimal_orientations = float(opt_meta["smallest_changes_orientations"])
        if opt_meta.get("smallest_changes_offsets") is not None:
            state.smallest_changes_optimal_offsets_angstrom = float(opt_meta["smallest_changes_offsets"])
        if opt_meta.get("smallest_changes_classes") is not None:
            state.smallest_changes_optimal_classes = float(opt_meta["smallest_changes_classes"])
        if opt_meta.get("has_converged") is not None:
            state.has_converged = bool(int(opt_meta["has_converged"]))
    logger.info(
        "Replay convergence init from RELION iter %03d: res=%.2f A, "
        "stalls=(res=%d,hvc=%d), smallest=(rot=%.3f deg, trans=%.3f A, class=%.3f)",
        init_relion_iteration,
        state.current_resolution,
        state.nr_iter_wo_resol_gain,
        state.nr_iter_wo_large_hidden_variable_changes,
        state.smallest_changes_optimal_orientations,
        state.smallest_changes_optimal_offsets_angstrom,
        state.smallest_changes_optimal_classes,
    )


def read_optimiser_accuracy_replay(
    *,
    replay_dir,
    replay_prefix,
    init_relion_iteration,
    iteration: int,
    sealed_sampling_state,
    acc_rot,
    acc_trans,
    convergence_acc_rot,
    convergence_acc_trans,
    logger,
) -> OptimiserAccuracyReplay:
    """Read RELION's numbered optimiser accuracies for the convergence update.

    With a replay directory and no sealed sampling state, the numbered
    optimiser STAR for this iteration is selected. When it exists, its finite
    ``overall_accuracy_rotations`` / ``overall_accuracy_translations_angst``
    replace both the reported and the convergence accuracies. Read or parse
    failures are logged as warnings and keep whatever was assigned before the
    failure, as the controller did inline. Sealed replay leaves every input
    unchanged.
    """

    metadata = None
    optimiser_star = None
    optimiser_iteration = None
    if replay_dir is not None and sealed_sampling_state is None:
        optimiser_iteration = int(init_relion_iteration) + iteration + 1
        optimiser_star = os.path.join(
            replay_dir,
            f"{replay_prefix}_it{optimiser_iteration:03d}_optimiser.star",
        )
        if os.path.exists(optimiser_star):
            try:
                metadata = read_relion_optimiser_metadata(optimiser_star)
                relion_acc_rot = metadata.get("overall_accuracy_rotations")
                relion_acc_trans_angst = metadata.get("overall_accuracy_translations_angst")
                if relion_acc_rot is not None and np.isfinite(float(relion_acc_rot)):
                    acc_rot = float(relion_acc_rot)
                    convergence_acc_rot = acc_rot
                if relion_acc_trans_angst is not None and np.isfinite(float(relion_acc_trans_angst)):
                    acc_trans = float(relion_acc_trans_angst)
                    convergence_acc_trans = acc_trans
                logger.info(
                    "Replay override: optimiser accuracy <- %s (acc_rot=%.3f deg, acc_trans=%s Å)",
                    optimiser_star,
                    float(acc_rot) if acc_rot is not None else float("nan"),
                    f"{acc_trans:.3f}" if acc_trans is not None else "unset",
                )
            except Exception as exc:
                logger.warning(
                    "Replay override: failed to read optimiser metadata from %s: %s", optimiser_star, exc
                )
    return OptimiserAccuracyReplay(
        metadata=metadata,
        optimiser_star=optimiser_star,
        optimiser_iteration=optimiser_iteration,
        acc_rot=acc_rot,
        acc_trans=acc_trans,
        convergence_acc_rot=convergence_acc_rot,
        convergence_acc_trans=convergence_acc_trans,
    )


def apply_optimiser_convergence_replay(
    state,
    *,
    metadata,
    optimiser_star,
    optimiser_iteration,
    replay_dir,
    replay_prefix,
    logger,
):
    """Apply numbered optimiser controls after the native state update.

    ``metadata`` exists only for a replay directory without sealed sampling
    (``read_optimiser_accuracy_replay``), so ``replay_dir`` is a path here.
    Mutate the existing state in metadata order. Missing fields retain their
    computed values. An unnumbered final optimiser may close the numbered
    replay only when the next sampling STAR is absent.
    ``read_optimiser_accuracy_replay`` supplies the accuracy overrides before
    the controller's state update; the controller calls this afterward.

    Unlike restart restoration, missing numbered counters retain computed
    values rather than defaulting to zero. Keep those contracts separate.
    """
    _relion_res_stalls = metadata.get("number_iter_without_resolution_gain")
    _relion_hvc_stalls = metadata.get("number_iter_without_changing_assignments")
    if _relion_res_stalls is not None:
        state.nr_iter_wo_resol_gain = int(_relion_res_stalls)
    if _relion_hvc_stalls is not None:
        _hvc = int(_relion_hvc_stalls)
        state.nr_iter_wo_large_hidden_variable_changes = _hvc
        state.nr_iter_wo_assignment_changes = _hvc
    _relion_changes = (
        ("changes_optimal_orientations", "current_changes_optimal_orientations", float),
        ("changes_optimal_offsets", "current_changes_optimal_offsets_angstrom", float),
        ("changes_optimal_classes", "current_changes_optimal_classes", float),
        ("smallest_changes_orientations", "smallest_changes_optimal_orientations", float),
        ("smallest_changes_offsets", "smallest_changes_optimal_offsets_angstrom", float),
        ("smallest_changes_classes", "smallest_changes_optimal_classes", float),
    )
    for _meta_key, _state_attr, _cast in _relion_changes:
        _value = metadata.get(_meta_key)
        if _value is not None:
            setattr(state, _state_attr, _cast(_value))
    _relion_has_converged = metadata.get("has_converged")
    if _relion_has_converged is not None:
        state.has_converged = bool(int(_relion_has_converged))

    # RELION's final all-data pass is stored as unnumbered
    # run_sampling.star/run_optimiser.star.  Numbered strict-replay
    # streams therefore end one iteration before the final pass; do not
    # request run_it{N+1}_sampling.star when RELION already recorded the
    # final convergence state in run_optimiser.star.
    if not state.has_converged:
        _next_sampling_star = os.path.join(
            replay_dir,
            f"{replay_prefix}_it{optimiser_iteration + 1:03d}_sampling.star",
        )
        _final_sampling_star = os.path.join(
            replay_dir,
            f"{replay_prefix}_sampling.star",
        )
        _final_optimiser_star = os.path.join(
            replay_dir,
            f"{replay_prefix}_optimiser.star",
        )
        if (
            not os.path.exists(_next_sampling_star)
            and os.path.exists(_final_sampling_star)
            and os.path.exists(_final_optimiser_star)
        ):
            try:
                _final_optimiser_meta = read_relion_optimiser_metadata(_final_optimiser_star)
                _final_has_converged = _final_optimiser_meta.get("has_converged")
                if _final_has_converged is not None and bool(int(_final_has_converged)):
                    state.has_converged = True
                    logger.info(
                        "Replay override: RELION final optimiser convergence <- %s "
                        "(numbered replay ended after %s)",
                        _final_optimiser_star,
                        optimiser_star,
                    )
            except Exception as exc:
                logger.warning(
                    "Replay override: failed to read final optimiser metadata from %s: %s",
                    _final_optimiser_star,
                    exc,
                )
    logger.info(
        "Replay override: optimiser control <- %s "
        "(res_stalls=%d, hvc_stalls=%d, changes=(rot=%.3f deg, trans=%.3f A, class=%.0f), "
        "smallest=(rot=%.3f deg, trans=%.3f A, class=%.0f), converged=%s)",
        optimiser_star,
        state.nr_iter_wo_resol_gain,
        state.nr_iter_wo_large_hidden_variable_changes,
        state.current_changes_optimal_orientations,
        state.current_changes_optimal_offsets_angstrom,
        state.current_changes_optimal_classes,
        state.smallest_changes_optimal_orientations,
        state.smallest_changes_optimal_offsets_angstrom,
        state.smallest_changes_optimal_classes,
        state.has_converged,
    )
