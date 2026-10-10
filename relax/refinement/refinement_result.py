"""What ``refine_single_volume`` and ``run_final_all_data`` return: named records, one per concept.

A run's result is a :class:`RefinementResult`. Its records hold the objects of the archive mapping and the
``RefinementHistory`` its trajectory entries are read from (rule 3: the history's other lists are host curves
and owner lists of the same run, kilobytes). Every record is
frozen; that freezes the bindings only: no consumer may mutate the arrays, lists or the convergence state a
record refers to.

:meth:`RefinementResult.archive_fields` is the saved-format and report mapping (code rule 2): it returns
the flat dict of saved keys of the return site that built the result, in the saved order, with the keys a
return path does not produce absent. The archive and report writers read it at their
boundary; other consumers read the records' attributes.

Lifetime: one run. The result is built at a return site of the controller and lives as long as its caller
keeps it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from relax.refinement.iteration_history import RefinementHistory
    from relax.refinement.refinement_state import RefinementState
    from relax.relion.worker_scale import FollowerScaleOutputs


@dataclass(frozen=True, kw_only=True)
class ModelMaps:
    """The run's published maps, Fourier-space, in the layout the controller produced them.

    ``mean``: the merged map (K=1) or the merged class maps (Class3D). ``means``: the per-half maps (a pair
    for K=1; for Class3D each half is ``(K, V)``). ``class_means``: the class maps, None for K=1.
    ``class_weights`` and ``class_assignments``: Class3D only, None otherwise. ``unfiltered_means``: the
    K=1 final pass's unfiltered half maps (RELION's ``_unfil`` maps); None otherwise.
    """

    mean: Any
    means: Any
    class_means: Any = None
    class_weights: Any = None
    class_assignments: Any = None
    unfiltered_means: Any = None

    def archive_fields(self) -> dict:
        """The five model keys every return path publishes (``unfiltered_means`` is placed by the caller)."""
        return {
            "mean": self.mean,
            "means": self.means,
            "class_means": self.class_means,
            "class_weights": self.class_weights,
            "class_assignments": self.class_assignments,
        }


@dataclass(frozen=True, kw_only=True)
class NumberedMetadata:
    """Set-up and numbered-iteration facts, in their existing precision and frames.

    Missing metadata stays None. The trial ids keep their half-local and particle-id frames; the archive
    converts them when it writes.
    """

    hard_assignments: Any
    frozen_initial_scoring_state_sha256: Any
    expected_accuracy_trial_local_indices: Any
    expected_accuracy_trial_particle_ids: Any
    setup_phase_seconds: Any

    def archive_fields(self) -> dict:
        return {
            "hard_assignments": self.hard_assignments,
            "frozen_initial_scoring_state_sha256": self.frozen_initial_scoring_state_sha256,
            "expected_accuracy_trial_local_indices": self.expected_accuracy_trial_local_indices,
            "expected_accuracy_trial_particle_ids": self.expected_accuracy_trial_particle_ids,
            "setup_phase_seconds": self.setup_phase_seconds,
        }


@dataclass(frozen=True, kw_only=True)
class ReplayTelemetry:
    """The RELION follower-scale replay's validated iterations; both None when the replay is off."""

    requested_iterations: Any
    applied_iterations: Any

    def archive_fields(self) -> dict:
        return {
            "relion_follower_scale_replay_requested_iterations": self.requested_iterations,
            "relion_follower_scale_replay_applied_iterations": self.applied_iterations,
        }


@dataclass(frozen=True, kw_only=True)
class FinalPassResult:
    """The final all-data pass's outputs and diagnostics (``final_all_data_*`` in the archive).

    Host casts and None sentinels are applied when the record is built (``result_files.final_pass_result``).
    ``grid_correct`` is the archive's legacy constant (scripts/masked_fsc.py and run_multi_iter_parity.py
    read it).
    """

    grid_correct: ClassVar[bool] = True

    pass2_engines: Any
    coarse_engines: Any
    expected_accuracy_status: Any
    acc_rot: Any
    acc_trans: Any
    acc_rot_per_class: Any
    acc_trans_per_class: Any
    expected_accuracy_class_counts: Any
    fsc: Any
    tau2_radial: Any
    tau2_fsc_used: Any
    tau2_ssnr: Any
    tau2_weight_combination: str
    best_rotation_eulers: Any
    best_translations: Any
    max_posterior: Any
    class_assignments: Any
    sampling_perturbation: Any
    sampling_perturbation_applied: bool
    sampling_relion_iteration: Any
    sampling_star: Any
    sampling_star_source: Any
    sampling_offset_range: Any
    sampling_offset_step: Any
    gridding_correct: Any

    def archive_fields(self) -> dict:
        return {
            "final_all_data_ran": True,
            "final_all_data_pass2_engines": self.pass2_engines,
            "final_all_data_coarse_engines": self.coarse_engines,
            "final_all_data_expected_accuracy_status": self.expected_accuracy_status,
            "final_all_data_acc_rot": self.acc_rot,
            "final_all_data_acc_trans": self.acc_trans,
            "final_all_data_acc_rot_per_class": self.acc_rot_per_class,
            "final_all_data_acc_trans_per_class": self.acc_trans_per_class,
            "final_all_data_expected_accuracy_class_counts": self.expected_accuracy_class_counts,
            "final_all_data_fsc": self.fsc,
            "tau2_radial_final_all_data": self.tau2_radial,
            "tau2_fsc_used_final_all_data": self.tau2_fsc_used,
            "tau2_ssnr_final_all_data": self.tau2_ssnr,
            "tau2_weight_combination_final_all_data": self.tau2_weight_combination,
            "final_all_data_best_rotation_eulers": self.best_rotation_eulers,
            "final_all_data_best_translations": self.best_translations,
            "final_all_data_max_posterior": self.max_posterior,
            "final_all_data_class_assignments": self.class_assignments,
            "final_all_data_sampling_perturbation": self.sampling_perturbation,
            "final_all_data_sampling_perturbation_applied": self.sampling_perturbation_applied,
            "final_all_data_sampling_relion_iteration": self.sampling_relion_iteration,
            "final_all_data_sampling_star": self.sampling_star,
            "final_all_data_sampling_star_source": self.sampling_star_source,
            "final_all_data_sampling_offset_range": self.sampling_offset_range,
            "final_all_data_sampling_offset_step": self.sampling_offset_step,
            "final_all_data_grid_correct": self.grid_correct,
            "final_all_data_gridding_correct": self.gridding_correct,
        }


@dataclass(frozen=True, kw_only=True)
class ProfileStop:
    """A run stopped after a local-search diagnostic iteration (``stop_after_local_search*``).

    ``wall_seconds`` and ``significant_count`` are that iteration's wall time and significant-sample counts;
    the archive reports them, alone, as ``wall_times`` and ``significant_counts``.
    """

    score_only: bool
    wall_seconds: float
    significant_count: Any


@dataclass(frozen=True, kw_only=True)
class RefinementResult:
    """One run of the numbered-iteration controller, by the path it returned on.

    - completed with the final all-data pass: ``final_pass`` set, ``profile_stop`` None;
    - completed without it (not converged, or skipped): ``final_pass`` and ``profile_stop`` None;
    - stopped after a local-search diagnostic: ``profile_stop`` set, ``follower_scale`` None.

    ``numbered`` is None only for the result ``run_final_all_data`` hands the controller, which adds it.
    ``history`` is the run's ``RefinementHistory``; the convergence state is the controller's last.
    """

    maps: ModelMaps
    convergence_state: RefinementState
    history: RefinementHistory
    replay: ReplayTelemetry
    follower_scale: FollowerScaleOutputs | None
    numbered: NumberedMetadata | None
    final_pass: FinalPassResult | None = None
    profile_stop: ProfileStop | None = None

    @property
    def profile_only(self) -> bool:
        return self.profile_stop is not None

    @property
    def final_all_data_ran(self) -> bool:
        return self.final_pass is not None

    def archive_fields(self) -> dict:
        """The flat result mapping the archive and report writers read (a saved-format contract, rule 2).

        The keys, their order and values are those each return site of ``refine_single_volume`` returned
        as a dict; a key a path did not return is absent, a key returned as None is present with None.
        """
        model = self.maps.archive_fields()
        replay = self.replay.archive_fields()
        history = self.history.to_dict()
        numbered = {} if self.numbered is None else self.numbered.archive_fields()
        if self.profile_stop is not None:
            # The stopped iteration's own wall time and counts replace the history's, in their place.
            history["wall_times"] = [self.profile_stop.wall_seconds]
            history["significant_counts"] = [self.profile_stop.significant_count]
            return {
                "profile_only": True,
                **model,
                **replay,
                "convergence_state": self.convergence_state,
                **numbered,
                "final_all_data_ran": False,
                "stop_after_local_search_score_only": self.profile_stop.score_only,
                **history,
            }
        follower_scale = self.follower_scale.archive_fields()
        if self.final_pass is None:
            return {
                **model,
                **replay,
                **follower_scale,
                "convergence_state": self.convergence_state,
                **numbered,
                "final_all_data_ran": False,
                **history,
            }
        return {
            **model,
            "unfiltered_means": self.maps.unfiltered_means,
            **replay,
            **follower_scale,
            "convergence_state": self.convergence_state,
            **history,
            **self.final_pass.archive_fields(),
            **numbered,
        }
