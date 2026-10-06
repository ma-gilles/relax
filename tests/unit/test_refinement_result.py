"""RefinementResult.archive_fields(): the saved flat mapping of each return path of the controller.

The expected mappings are written out as the return sites of ``refine_single_volume`` and
``run_final_all_data`` composed them before they returned records (a1f78ce): the same keys, in the same
order, holding the same objects. The local-search diagnostic stop has no CPU run, so this is its only check.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers.refinement_results import numbered_metadata

from relax.helpers.iteration_history import RefinementHistory
from relax.refinement.refinement_result import (
    FinalPassResult,
    ModelMaps,
    ProfileStop,
    RefinementResult,
    ReplayTelemetry,
)
from relax.relion.relion_worker_scale import FollowerScaleOutputs

pytestmark = pytest.mark.unit


def _history():
    history = RefinementHistory()
    history.current_sizes.extend([8, 12])
    history.fsc_history.extend([np.ones(5), np.full(5, 0.5)])
    history.wall_times.extend([1.0, 2.0])
    history.significant_counts.extend([None, np.arange(3)])
    history.state_swap_probe_applied_relion_iterations.append(4)
    return history


def _records():
    maps = ModelMaps(
        mean=np.zeros(4), means=[np.ones(4), np.ones(4)], class_means=np.zeros((2, 4)),
        class_weights=np.array([0.5, 0.5]), class_assignments=[np.zeros(2), np.ones(2)],
        unfiltered_means=[np.full(4, 2.0), np.full(4, 3.0)],
    )
    replay = ReplayTelemetry(requested_iterations=np.array([2]), applied_iterations=np.array([2]))
    follower = FollowerScaleOutputs(
        follower_scales=np.ones(3), rank1_serialized=np.ones(2), follower_owners_half1=np.zeros(2, np.int64),
        follower_owners_half1_trajectory=np.zeros((1, 2), np.int64),
        follower_scales_numbered_pre_score_trajectory=np.ones((1, 3)),
        follower_scales_numbered_post_mstep_trajectory=np.ones((1, 3)),
    )
    numbered = numbered_metadata(
        hard_assignments=[np.zeros(2), np.zeros(2)], setup_phase_seconds={"state_init": 0.5},
    )
    return maps, replay, follower, numbered


def _model(maps):
    return {
        "mean": maps.mean, "means": maps.means, "class_means": maps.class_means,
        "class_weights": maps.class_weights, "class_assignments": maps.class_assignments,
    }


def _replay(replay):
    return {
        "relion_follower_scale_replay_requested_iterations": replay.requested_iterations,
        "relion_follower_scale_replay_applied_iterations": replay.applied_iterations,
    }


def _follower(follower):
    return {
        "relion_scale_follower_scales": follower.follower_scales,
        "relion_scale_rank1_serialized": follower.rank1_serialized,
        "relion_scale_follower_owners_half1": follower.follower_owners_half1,
        "relion_scale_follower_owners_half1_trajectory": follower.follower_owners_half1_trajectory,
        "relion_scale_follower_scales_numbered_pre_score_trajectory":
            follower.follower_scales_numbered_pre_score_trajectory,
        "relion_scale_follower_scales_numbered_post_mstep_trajectory":
            follower.follower_scales_numbered_post_mstep_trajectory,
    }


def _numbered(numbered):
    return {
        "hard_assignments": numbered.hard_assignments,
        "frozen_initial_scoring_state_sha256": numbered.frozen_initial_scoring_state_sha256,
        "expected_accuracy_trial_local_indices": numbered.expected_accuracy_trial_local_indices,
        "expected_accuracy_trial_particle_ids": numbered.expected_accuracy_trial_particle_ids,
        "setup_phase_seconds": numbered.setup_phase_seconds,
    }


def _assert_same_mapping(got, expected):
    assert list(got) == list(expected)
    for key, value in expected.items():
        if isinstance(value, list) and key not in ("means", "class_assignments", "hard_assignments"):
            # A history list (or, for the profile stop and the state-swap list, a fresh list): equal items.
            assert len(got[key]) == len(value), key
            for got_item, item in zip(got[key], value):
                assert got_item is item, key
        else:
            assert got[key] is value, key


def test_numbered_return_without_final_pass():
    maps, replay, follower, numbered = _records()
    history, state = _history(), object()
    result = RefinementResult(
        maps=maps, convergence_state=state, history=history, replay=replay, follower_scale=follower,
        numbered=numbered,
    )
    expected = {
        **_model(maps), **_replay(replay), **_follower(follower), "convergence_state": state,
        **_numbered(numbered), "final_all_data_ran": False, **history.to_dict(),
    }
    fields = result.archive_fields()
    _assert_same_mapping(fields, expected)
    assert "unfiltered_means" not in fields and "profile_only" not in fields
    assert fields["fsc"] is history.fsc_history[-1]
    assert fields["wall_times"] is history.wall_times
    assert result.final_all_data_ran is False and result.profile_only is False


def test_profile_stop_reports_the_stopped_iteration_in_place():
    maps, replay, _, numbered = _records()
    history, state = _history(), object()
    counts = np.arange(6)
    result = RefinementResult(
        maps=maps, convergence_state=state, history=history, replay=replay, follower_scale=None,
        numbered=numbered, profile_stop=ProfileStop(score_only=True, wall_seconds=7.5, significant_count=counts),
    )
    expected = {
        "profile_only": True, **_model(maps), **_replay(replay), "convergence_state": state,
        **_numbered(numbered), "final_all_data_ran": False, "stop_after_local_search_score_only": True,
        **history.to_dict(), "wall_times": [7.5], "significant_counts": [counts],
    }
    fields = result.archive_fields()
    _assert_same_mapping(fields, expected)
    assert fields["wall_times"] == [7.5] and fields["significant_counts"][0] is counts
    assert history.wall_times == [1.0, 2.0]  # the history is not changed
    assert not any(key.startswith("relion_scale_") for key in fields) and "unfiltered_means" not in fields
    assert result.profile_only is True and result.final_all_data_ran is False


def _final_pass():
    return FinalPassResult(
        pass2_engines=["resident"], coarse_engines=["dense"], expected_accuracy_status="ok", acc_rot=1.5,
        acc_trans=0.5, acc_rot_per_class=None, acc_trans_per_class=None, expected_accuracy_class_counts=None,
        fsc=np.ones(5), tau2_radial=np.ones(5), tau2_fsc_used=None, tau2_ssnr=np.ones(5),
        tau2_weight_combination="sum", best_rotation_eulers=[np.zeros((2, 3))],
        best_translations=[np.zeros((2, 2))], max_posterior=[np.ones(2)], class_assignments=None,
        sampling_perturbation=0.25, sampling_perturbation_applied=True, sampling_relion_iteration=3,
        sampling_star=None, sampling_star_source=None, sampling_offset_range=4.0, sampling_offset_step=1.0,
        gridding_correct="separable",
    )


@pytest.mark.parametrize("with_numbered", [False, True])
def test_final_pass_return(with_numbered):
    maps, replay, follower, numbered = _records()
    numbered = numbered if with_numbered else None
    history, state, final_pass = _history(), object(), _final_pass()
    result = RefinementResult(
        maps=maps, convergence_state=state, history=history, replay=replay, follower_scale=follower,
        numbered=numbered, final_pass=final_pass,
    )
    expected = {
        **_model(maps), "unfiltered_means": maps.unfiltered_means, **_replay(replay), **_follower(follower),
        "convergence_state": state, **history.to_dict(),
        "final_all_data_ran": True,
        "final_all_data_pass2_engines": final_pass.pass2_engines,
        "final_all_data_coarse_engines": final_pass.coarse_engines,
        "final_all_data_expected_accuracy_status": final_pass.expected_accuracy_status,
        "final_all_data_acc_rot": final_pass.acc_rot,
        "final_all_data_acc_trans": final_pass.acc_trans,
        "final_all_data_acc_rot_per_class": None,
        "final_all_data_acc_trans_per_class": None,
        "final_all_data_expected_accuracy_class_counts": None,
        "final_all_data_noise_source_half": -1,
        "final_all_data_noise_source_halves": (0, 1),
        "final_all_data_fsc": final_pass.fsc,
        "tau2_radial_final_all_data": final_pass.tau2_radial,
        "tau2_fsc_used_final_all_data": None,
        "tau2_ssnr_final_all_data": final_pass.tau2_ssnr,
        "tau2_weight_combination_final_all_data": final_pass.tau2_weight_combination,
        "final_all_data_best_rotation_eulers": final_pass.best_rotation_eulers,
        "final_all_data_best_translations": final_pass.best_translations,
        "final_all_data_max_posterior": final_pass.max_posterior,
        "final_all_data_class_assignments": None,
        "final_all_data_sampling_perturbation": final_pass.sampling_perturbation,
        "final_all_data_sampling_perturbation_applied": True,
        "final_all_data_sampling_relion_iteration": final_pass.sampling_relion_iteration,
        "final_all_data_sampling_star": None,
        "final_all_data_sampling_star_source": None,
        "final_all_data_sampling_offset_range": final_pass.sampling_offset_range,
        "final_all_data_sampling_offset_step": final_pass.sampling_offset_step,
        "final_all_data_grid_correct": True,
        "final_all_data_gridding_correct": final_pass.gridding_correct,
        # The controller adds the set-up and numbered metadata after the final pass returns.
        **({} if numbered is None else _numbered(numbered)),
    }
    fields = result.archive_fields()
    for key in ("final_all_data_noise_source_half", "final_all_data_noise_source_halves",
                "final_all_data_grid_correct", "final_all_data_ran", "final_all_data_sampling_perturbation_applied"):
        assert fields[key] == expected[key]
        expected[key] = fields[key]
    _assert_same_mapping(fields, expected)
    assert result.final_all_data_ran is True
