from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from scripts.analyze_vdam_bpref_accumulator_boundary import (
    _geometry,
    _inline_projector_replays,
    _production_names,
    _rank_particle_sources,
    _to_relion_bpref_frame,
)
from scripts.lib.bpref_contribution_replay import BPrefAccumulatorReplay

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("half", "iteration", "expected_native"),
    [
        (1, 1, ("pipe_it1_c0_bp_data_pre_reweight.bin", "pipe_it1_c0_bp_weight.bin")),
        (1, 58, ("pipe_it58_c0_bp_data_pre_reweight.bin", "pipe_it58_c0_bp_weight.bin")),
        (2, 58, ("pipe_it58_c0_bp_data_h_pre_reweight.bin", "pipe_it58_c0_bp_weight_h.bin")),
    ],
)
def test_production_names_select_requested_iteration(half, iteration, expected_native):
    names = _production_names(half, iteration=iteration)

    assert names[:2] == expected_native


@pytest.mark.unit
def test_production_names_reject_nonpositive_iteration():
    with pytest.raises(ValueError, match="iteration must be positive"):
        _production_names(1, iteration=0)


@pytest.mark.unit
def test_rank_particle_sources_aggregates_rows_and_orders_by_data_norm():
    rows = {
        "active_original_indices": np.asarray([9, 2, 9], dtype=np.int64),
        "active_summed": np.asarray([[3 + 4j, 0], [2 + 0j, 0], [0, 12 + 0j]]),
        "active_ctf_probs": np.asarray([[1.0, 2.0], [4.0, 0.0], [3.0, 0.0]]),
    }

    ranking = _rank_particle_sources(rows)

    assert [row["original_index"] for row in ranking] == [9, 2]
    assert ranking[0]["row_count"] == 2
    assert ranking[0]["data_l2"] == 13.0
    assert ranking[0]["weight_l1"] == 6.0


@pytest.mark.unit
def test_geometry_separates_parallel_and_orthogonal_candidate_error():
    control = np.zeros(2, dtype=np.float64)
    native = np.asarray([2.0, 0.0])
    candidate = np.asarray([1.0, 3.0])

    metric = _geometry(candidate, native, control)

    assert metric == {
        "candidate_projection_on_native_delta": 0.5,
        "candidate_orthogonal_over_native_delta": 1.5,
    }


@pytest.mark.unit
def test_relion_bpref_frame_conversion_applies_fft_sign_and_scales():
    replay = BPrefAccumulatorReplay(
        data=np.asarray([1 + 2j], dtype=np.complex128),
        weight=np.asarray([3.0], dtype=np.float64),
        backend="raw",
        order="execution",
        precision="complex128/float64",
        launch_topology="fixture",
    )

    converted = _to_relion_bpref_frame(replay, ori_size=4)

    assert_matches(converted.data, np.asarray([-16 - 32j]))
    assert_matches(converted.weight, np.asarray([768.0]))
    assert converted.backend == "raw_relion_bpref_frame"
    assert converted.order == replay.order
    assert converted.precision == replay.precision
    assert converted.launch_topology == replay.launch_topology


@pytest.mark.unit
def test_inline_projector_replay_selects_joint_reconstruction_group():
    half_size = 3 * 3 * 2
    shard = SimpleNamespace(
        values={
            "inline_projector_data_volumes": np.asarray(
                [np.ones(half_size), np.full(half_size, 2.0)],
                dtype=np.complex64,
            ),
            "inline_projector_weight_volumes": np.asarray(
                [np.ones(half_size), np.full(half_size, 3.0)],
                dtype=np.float32,
            ),
            "inline_projector_original_indices": np.asarray([10, 20], dtype=np.int64),
            "original_indices": np.asarray([10, 20], dtype=np.int64),
            "reconstruction_group_ids": np.asarray([0, 1], dtype=np.int32),
        }
    )
    bundle = SimpleNamespace(
        shards=(shard,),
        boundary_values={"volume_shape": np.asarray([3, 3, 3], dtype=np.int32)},
    )

    replays, summary = _inline_projector_replays(
        bundle,
        reconstruction_group=1,
        ori_size=4,
    )
    group_zero, _ = _inline_projector_replays(
        bundle,
        reconstruction_group=0,
        ori_size=4,
    )

    assert_matches(
        replays["sequential_float32"].data,
        2.0 * group_zero["sequential_float32"].data,
    )
    assert_matches(
        replays["sequential_float32"].weight,
        3.0 * group_zero["sequential_float32"].weight,
    )
    assert summary["particle_count"] == 1
    assert summary["first_particle_original_index"] == 20


