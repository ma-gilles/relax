from __future__ import annotations

import numpy as np
import pytest

from relax.classification.k_class import _apply_bpref_particle_order_policy
from relax.diagnostics.relion_replay import _validate_bpref_particle_order_scope
from relax.helpers.batch_planning import _plan_consecutive_padded_batches
from relax.helpers.env_flags import parse_env_flag
from relax.scoring.sparse_bucket_arrays import _bucket_pass2_inputs
from relax.sparse_pass2.sparse_pass2_policy import _BPREF_EXECUTION_GROUP_BY_BUCKET_SIZE_ENV


def test_sparse_pass2_execution_order_override_is_exact_and_single_particle():
    counts = [16, 64, 32, 16]
    per_image = {
        "oversampled_rots": [
            np.zeros((count, 3, 3), dtype=np.float32) for count in counts
        ],
    }
    order = np.asarray([2, 0, 3, 1], dtype=np.int64)

    buckets = _bucket_pass2_inputs(
        per_image,
        n_fine_trans=4,
        processing_order_override=order,
    )

    assert [bucket["image_indices"].tolist() for bucket in buckets] == [
        [2],
        [0],
        [3],
        [1],
    ]
    assert [int(bucket["bucket_size"]) for bucket in buckets] == [32, 16, 16, 64]

    with pytest.raises(ValueError, match="must be a permutation"):
        _bucket_pass2_inputs(
            per_image,
            n_fine_trans=4,
            processing_order_override=np.asarray([0, 0, 2, 3]),
        )


def test_sparse_pass2_execution_order_override_chunks_only_adjacent_particles():
    counts = [16, 64, 32, 16, 48]
    per_image = {
        "oversampled_rots": [
            np.zeros((count, 3, 3), dtype=np.float32) for count in counts
        ],
    }
    order = np.asarray([2, 0, 3, 1, 4], dtype=np.int64)

    buckets = _bucket_pass2_inputs(
        per_image,
        n_fine_trans=4,
        processing_order_override=order,
        processing_order_chunk_size=2,
    )

    assert [bucket["image_indices"].tolist() for bucket in buckets] == [
        [2, 0],
        [3, 1],
        [4],
    ]
    assert [int(bucket["bucket_size"]) for bucket in buckets] == [32, 64, 64]
    with pytest.raises(ValueError, match="must be positive"):
        _bucket_pass2_inputs(
            per_image,
            n_fine_trans=4,
            processing_order_override=order,
            processing_order_chunk_size=0,
        )


def test_shared_consecutive_padded_planner_preserves_order_and_pool_alignment():
    padded_sizes = np.asarray([32, 64, 16, 128, 32, 16, 256], dtype=np.int64)
    order = np.asarray([2, 0, 1, 4, 3, 5, 6], dtype=np.int64)

    plans = _plan_consecutive_padded_batches(
        padded_sizes,
        processing_order=order,
        target_items_per_batch=4,
        max_items_per_batch=20,
        max_padded_values_per_batch=10_000,
        item_alignment=3,
    )

    assert [plan.item_indices.tolist() for plan in plans] == [
        [2, 0, 1],
        [4, 3, 5, 6],
    ]
    assert [plan.padded_size for plan in plans] == [64, 256]
    assert [plan.padded_item_capacity for plan in plans] == [3, 4]
    assert np.cumsum([0] + [plan.item_indices.size for plan in plans[:-1]]).tolist() == [0, 3]
    assert np.concatenate([plan.item_indices for plan in plans]).tolist() == order.tolist()


def test_shared_consecutive_padded_planner_validates_processing_order():
    with pytest.raises(ValueError, match="must be a permutation"):
        _plan_consecutive_padded_batches(
            [16, 32, 64],
            processing_order=[0, 0, 2],
            target_items_per_batch=2,
            max_items_per_batch=2,
            max_padded_values_per_batch=1024,
        )


@pytest.mark.parametrize(
    ("target_items", "max_items"),
    [(2, 8), (8, 2)],
)
def test_shared_consecutive_padded_planner_rejects_limits_that_split_a_pool(
    target_items,
    max_items,
):
    with pytest.raises(ValueError, match="too small to preserve"):
        _plan_consecutive_padded_batches(
            [32, 32, 32],
            target_items_per_batch=target_items,
            max_items_per_batch=max_items,
            max_padded_values_per_batch=512,
            item_alignment=3,
        )


def test_shared_consecutive_padded_planner_rejects_an_oversize_aligned_pool():
    with pytest.raises(ValueError, match="too small for one aligned item group"):
        _plan_consecutive_padded_batches(
            [32, 32, 32],
            target_items_per_batch=3,
            max_items_per_batch=3,
            max_padded_values_per_batch=95,
            item_alignment=3,
        )


@pytest.mark.parametrize("padded_sizes", [[32], [32, 32], [32, 32, 32, 32], [32, 32, 32, 32, 32]])
def test_shared_consecutive_padded_planner_allows_a_fitting_final_incomplete_pool(padded_sizes):
    plans = _plan_consecutive_padded_batches(
        padded_sizes,
        target_items_per_batch=3,
        max_items_per_batch=3,
        max_padded_values_per_batch=96,
        item_alignment=3,
    )

    assert np.concatenate([plan.item_indices for plan in plans]).tolist() == list(
        range(len(padded_sizes))
    )
    assert all(plan.item_indices.size <= 3 for plan in plans)


def test_shared_consecutive_padded_planner_keeps_alignment_one_oversize_progress():
    plans = _plan_consecutive_padded_batches(
        [128],
        target_items_per_batch=1,
        max_items_per_batch=1,
        max_padded_values_per_batch=64,
        item_alignment=1,
    )

    assert len(plans) == 1
    assert plans[0].item_indices.tolist() == [0]
    assert plans[0].padded_size == 128


def test_sparse_pass2_ordered_chunks_respect_hypothesis_and_image_caps():
    counts = [16, 16, 256, 16, 16, 16]
    per_image = {
        "oversampled_rots": [
            np.zeros((count, 3, 3), dtype=np.float32) for count in counts
        ],
    }

    buckets = _bucket_pass2_inputs(
        per_image,
        n_fine_trans=4,
        processing_order_override=np.arange(len(counts), dtype=np.int64),
        processing_order_chunk_size=4,
        max_hypotheses_per_microbatch=2048,
        max_images_per_microbatch=3,
    )

    assert [bucket["image_indices"].tolist() for bucket in buckets] == [
        [0, 1],
        [2, 3],
        [4, 5],
    ]
    assert [int(bucket["bucket_size"]) for bucket in buckets] == [16, 256, 16]
    assert np.concatenate([bucket["image_indices"] for bucket in buckets]).tolist() == list(
        range(len(counts))
    )


def test_sparse_pass2_ordered_chunks_do_not_scale_with_support_runs():
    counts = ([16, 32] * 500) + [256] + ([16, 32] * 100)
    per_image = {
        "oversampled_rots": [
            np.zeros((count, 3, 3), dtype=np.float32) for count in counts
        ],
    }
    max_hypotheses = 814_509
    n_fine_trans = 116

    buckets = _bucket_pass2_inputs(
        per_image,
        n_fine_trans=n_fine_trans,
        processing_order_override=np.arange(len(counts), dtype=np.int64),
        processing_order_chunk_size=220,
        max_hypotheses_per_microbatch=max_hypotheses,
        max_images_per_microbatch=156,
    )

    assert np.concatenate([bucket["image_indices"] for bucket in buckets]).tolist() == list(
        range(len(counts))
    )
    assert len(buckets) < 20
    for bucket in buckets:
        image_count = len(bucket["image_indices"])
        assert image_count <= 156
        assert image_count * int(bucket["bucket_size"]) * n_fine_trans <= max_hypotheses


def test_sparse_pass2_execution_order_batches_only_consecutive_equal_sizes():
    counts = [16, 16, 64, 64, 16, 32, 32, 32]
    per_image = {
        "oversampled_rots": [
            np.zeros((count, 3, 3), dtype=np.float32) for count in counts
        ],
    }
    order = np.asarray([4, 0, 1, 2, 3, 7, 5, 6], dtype=np.int64)

    buckets = _bucket_pass2_inputs(
        per_image,
        n_fine_trans=4,
        processing_order_override=order,
        processing_order_batch_consecutive_bucket_sizes=True,
        max_hypotheses_per_microbatch=256,
        max_images_per_microbatch=8,
    )

    assert [bucket["image_indices"].tolist() for bucket in buckets] == [
        [4, 0, 1],
        [2],
        [3],
        [7, 5],
        [6],
    ]
    assert [int(bucket["bucket_size"]) for bucket in buckets] == [16, 64, 64, 32, 32]
    assert np.concatenate([bucket["image_indices"] for bucket in buckets]).tolist() == order.tolist()


def test_sparse_pass2_execution_order_can_stay_stable_within_size_buckets():
    counts = [16, 64, 32, 16, 48]
    per_image = {
        "oversampled_rots": [
            np.zeros((count, 3, 3), dtype=np.float32) for count in counts
        ],
    }
    order = np.asarray([2, 0, 3, 1, 4], dtype=np.int64)

    buckets = _bucket_pass2_inputs(
        per_image,
        n_fine_trans=4,
        processing_order_override=order,
        processing_order_group_by_bucket_size=True,
        max_images_per_microbatch=8,
    )

    assert [bucket["image_indices"].tolist() for bucket in buckets] == [
        [0, 3],
        [2],
        [1, 4],
    ]
    assert [int(bucket["bucket_size"]) for bucket in buckets] == [16, 32, 64]


def test_grouped_execution_order_environment_flag_uses_module_parser(monkeypatch):
    monkeypatch.setenv(_BPREF_EXECUTION_GROUP_BY_BUCKET_SIZE_ENV, "1")
    assert parse_env_flag(_BPREF_EXECUTION_GROUP_BY_BUCKET_SIZE_ENV)


def test_sparse_pass_order_policy_is_k1_only_and_dormant_by_default():
    common = {"sentinel": object()}
    assert not _apply_bpref_particle_order_policy(
        common,
        {},
        n_classes=4,
    )
    assert "preserve_bpref_particle_order" not in common

    assert _apply_bpref_particle_order_policy(
        common,
        {"preserve_bpref_particle_order": True},
        n_classes=1,
    )
    assert common["preserve_bpref_particle_order"] is True

    with pytest.raises(ValueError, match="K=1-only"):
        _apply_bpref_particle_order_policy(
            {},
            {"preserve_bpref_particle_order": True},
            n_classes=4,
        )


def test_fresh_k1_bpref_order_scope_accepts_perturbation_replay_from_iteration_zero():
    """A fresh run that replays RELION's perturbations keeps RELION's order."""

    _validate_bpref_particle_order_scope(
        preserve_bpref_particle_order=True,
        n_classes=1,
        init_relion_iteration=0,
        perturb_replay_relion_dir="relion",
        replay_iteration_overrides=[None, {"state": 1}],
        sealed_sampling_state=None,
        sealed_scoring_context=None,
    )


def test_fresh_k1_bpref_order_scope_accepts_only_unsealed_iteration_zero():
    kwargs = {
        "preserve_bpref_particle_order": True,
        "n_classes": 1,
        "init_relion_iteration": 0,
        "perturb_replay_relion_dir": None,
        "replay_iteration_overrides": [None],
        "sealed_sampling_state": None,
        "sealed_scoring_context": None,
    }
    _validate_bpref_particle_order_scope(**kwargs)

    for override, match in (
        ({"n_classes": 4}, "K=1-only"),
        ({"init_relion_iteration": 1}, "requires the native RELION order"),
        ({"replay_iteration_overrides": [None, {"state": 1}]}, "without perturbation replay"),
        ({"sealed_sampling_state": object()}, "sealed boundary"),
        ({"sealed_scoring_context": object()}, "sealed boundary"),
    ):
        invalid = dict(kwargs)
        invalid.update(override)
        with pytest.raises(ValueError, match=match):
            _validate_bpref_particle_order_scope(**invalid)


def test_bpref_order_scope_is_dormant_when_preservation_is_disabled():
    _validate_bpref_particle_order_scope(
        preserve_bpref_particle_order=False,
        n_classes=4,
        init_relion_iteration=9,
        perturb_replay_relion_dir="relion",
        replay_iteration_overrides=[None, {"state": 1}],
        sealed_sampling_state=object(),
        sealed_scoring_context=object(),
    )


def test_state_swap_fresh_k1_bpref_order_scope_requires_complete_unsealed_replay():
    kwargs = {
        "preserve_bpref_particle_order": True,
        "n_classes": 1,
        "init_relion_iteration": 0,
        "perturb_replay_relion_dir": "relion",
        "replay_iteration_overrides": [None, {"state": 1}],
        "sealed_sampling_state": None,
        "sealed_scoring_context": None,
        "allow_state_swap_fresh_bpref_particle_order": True,
    }
    _validate_bpref_particle_order_scope(**kwargs)

    for override, match in (
        ({"init_relion_iteration": 1}, "fresh iteration-0"),
        ({"perturb_replay_relion_dir": None}, "requires perturbation replay"),
        ({"replay_iteration_overrides": [None]}, "requires numbered replay state"),
        ({"sealed_sampling_state": object()}, "cannot alter a sealed boundary"),
    ):
        invalid = dict(kwargs)
        invalid.update(override)
        with pytest.raises(ValueError, match=match):
            _validate_bpref_particle_order_scope(**invalid)
