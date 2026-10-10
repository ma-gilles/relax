"""Image-axis capacity ladder for the exact local engine's bucket programs.

The leading axis of every per-bucket local program is the PADDED images-per-
bucket capacity ``min(image_batch_size, max_hypotheses_per_microbatch //
bucket_rotations)``.  ``image_batch_size`` is a memory estimate that moves by a
few images between iterations, so that axis is unstable and each new value
recompiles the whole bucket program set.  The ladder snaps the capacity to a
fixed set of rungs, always downward so the planner's own memory bound still
holds.

These tests pin the resolver, the snapping rule, the planner's use of it, the
two layouts it must not disturb (RELION's InitialModel particle pools and the
consecutive mixed-bucket plan), and the contract that makes the ladder safe:
padding a bucket's image axis does not change any engine output.
"""

from __future__ import annotations

import numpy as np
import pytest

from relax.local_search.layout import (
    DEFAULT_LOCAL_IMAGE_CAPACITY_LADDER,
    LOCAL_IMAGE_CAPACITY_LADDER_ENV,
    LocalHypothesisLayout,
    _ladder_image_capacity,
    plan_local_hypothesis_buckets,
    resolve_local_image_capacity_ladder,
)

pytestmark = pytest.mark.unit


def _layout(rotation_counts, n_trans=1):
    rotation_counts = np.asarray(rotation_counts, dtype=np.int32)
    offsets = np.concatenate(([0], np.cumsum(rotation_counts))).astype(np.int64)
    total = int(offsets[-1])
    return LocalHypothesisLayout(
        n_global_rotations=total,
        n_pixels=1,
        n_psi=1,
        rotation_offsets=offsets,
        rotation_ids_flat=np.arange(total, dtype=np.int32),
        rotations_flat=np.broadcast_to(
            np.eye(3, dtype=np.float32), (total, 3, 3)
        ).copy(),
        rotation_log_priors_flat=np.zeros(total, dtype=np.float32),
        rotation_counts=rotation_counts,
        translation_grid=np.zeros((n_trans, 2), dtype=np.float32),
        translation_log_priors=np.zeros((rotation_counts.size, n_trans), dtype=np.float32),
    )


def _plan(ladder, *, counts, image_batch_size, microbatch=65536, **kwargs):
    return plan_local_hypothesis_buckets(
        _layout(counts),
        image_batch_size=image_batch_size,
        rotation_block_size=8192,
        max_hypotheses_per_microbatch=microbatch,
        image_capacity_ladder=ladder,
        **kwargs,
    )


def _capacities(plans):
    return sorted({int(plan.bucket_image_count) for plan in plans})


def _padding_ratio(plans):
    physical = sum(int(plan.image_indices.shape[0]) for plan in plans)
    padded = sum(
        max(int(plan.image_indices.shape[0]), int(plan.bucket_image_count))
        for plan in plans
    )
    return padded / max(1, physical)


# ----------------------------------------------------------------- resolver ---


def test_ladder_is_off_by_default_and_reads_the_environment(monkeypatch):
    monkeypatch.delenv(LOCAL_IMAGE_CAPACITY_LADDER_ENV, raising=False)
    assert resolve_local_image_capacity_ladder() == ()

    monkeypatch.setenv(LOCAL_IMAGE_CAPACITY_LADDER_ENV, "1")
    assert resolve_local_image_capacity_ladder() == DEFAULT_LOCAL_IMAGE_CAPACITY_LADDER

    monkeypatch.setenv(LOCAL_IMAGE_CAPACITY_LADDER_ENV, "8, 64,16")
    assert resolve_local_image_capacity_ladder() == (8, 16, 64)

    monkeypatch.setenv(LOCAL_IMAGE_CAPACITY_LADDER_ENV, "0")
    assert resolve_local_image_capacity_ladder() == ()


def test_explicit_argument_overrides_the_environment(monkeypatch):
    monkeypatch.setenv(LOCAL_IMAGE_CAPACITY_LADDER_ENV, "1")
    assert resolve_local_image_capacity_ladder(False) == ()
    assert resolve_local_image_capacity_ladder([32, 16]) == (16, 32)
    assert resolve_local_image_capacity_ladder("off") == ()
    assert resolve_local_image_capacity_ladder(True) == DEFAULT_LOCAL_IMAGE_CAPACITY_LADDER


@pytest.mark.parametrize("value", ["16,not-an-integer", "16,0", [-4]])
def test_invalid_ladders_are_rejected(value):
    with pytest.raises(ValueError):
        resolve_local_image_capacity_ladder(value)


# ------------------------------------------------------------------ snapping ---


@pytest.mark.parametrize(
    ("max_images", "expected"),
    [
        (14, 14),  # below the ladder minimum: left alone, never raised
        (16, 16),
        (17, 16),
        (23, 16),
        (46, 32),
        (64, 64),
        (92, 64),
        (129, 128),
        (135, 128),
        (136, 128),
        (343, 256),
    ],
)
def test_capacity_snaps_down_to_a_rung(max_images, expected):
    assert (
        _ladder_image_capacity(max_images, DEFAULT_LOCAL_IMAGE_CAPACITY_LADDER)
        == expected
    )


def test_capacity_is_unchanged_when_the_ladder_is_off():
    assert _ladder_image_capacity(135, ()) == 135


@pytest.mark.parametrize("max_images", [1, 3, 15, 17, 63, 135, 136, 255, 1000])
def test_snapping_never_raises_the_planner_memory_bound(max_images):
    assert (
        _ladder_image_capacity(max_images, DEFAULT_LOCAL_IMAGE_CAPACITY_LADDER)
        <= max_images
    )


# ------------------------------------------------------------------- planner ---


def test_planners_ignore_the_environment_unless_the_caller_opts_in(monkeypatch):
    """`scripts/lib/local_dataset.py` also plans local buckets.

    It passes no ladder, so an EM-scoped environment variable must not
    re-bucket a pipeline with its own validation. Only `run_local_em_exact`
    reads the environment.
    """

    monkeypatch.setenv(LOCAL_IMAGE_CAPACITY_LADDER_ENV, "1")
    counts = np.full(500, 180, dtype=np.int32)
    assert _capacities(_plan(None, counts=counts, image_batch_size=135)) == [135]
    assert _capacities(_plan(True, counts=counts, image_batch_size=135)) == [128]


def test_planner_stabilizes_the_image_axis_across_batch_size_estimates():
    """135 and 136 images per bucket are two programs; both snap to one rung."""

    counts = np.full(500, 180, dtype=np.int32)
    off = [_plan(None, counts=counts, image_batch_size=n) for n in (135, 136)]
    on = [_plan(True, counts=counts, image_batch_size=n) for n in (135, 136)]

    assert [_capacities(p) for p in off] == [[135], [136]]
    assert [_capacities(p) for p in on] == [[128], [128]]


def test_planner_capacities_are_rungs():
    counts = np.repeat([40, 100, 300], 400).astype(np.int32)
    off = _plan(None, counts=counts, image_batch_size=135)
    on = _plan(True, counts=counts, image_batch_size=135)

    assert set(_capacities(on)) <= set(DEFAULT_LOCAL_IMAGE_CAPACITY_LADDER)
    assert max(_capacities(on)) <= max(_capacities(off))


def test_one_capacity_per_rotation_class_including_the_remainder_group():
    """A second capacity would be a second compiled program per class.

    Giving the remainder its own smaller rung pads less but costs an extra
    ``run_local_bucket_big_jit`` compile per rotation class; that variant
    measured 7.96 s -> 16.26 s of local-engine compile at the 10k/256 order-4
    state, so the remainder keeps the class capacity.
    """

    counts = np.full(400, 180, dtype=np.int32)
    plans = _plan(True, counts=counts, image_batch_size=135)

    assert [int(plan.image_indices.shape[0]) for plan in plans] == [128, 128, 128, 16]
    assert _capacities(plans) == [128]


def test_padding_stays_small_when_a_class_is_much_larger_than_its_rung():
    """The production case: one rung per class over thousands of images."""

    counts = np.full(4966, 180, dtype=np.int32)
    on = _plan(True, counts=counts, image_batch_size=135)
    assert _capacities(on) == [128]
    assert _padding_ratio(on) < 1.05


def test_planner_preserves_the_image_set_and_its_order():
    counts = np.repeat([40, 100, 300], 137).astype(np.int32)
    for ladder in (None, True):
        plans = _plan(ladder, counts=counts, image_batch_size=135)
        seen = np.concatenate([np.asarray(p.image_indices) for p in plans])
        assert sorted(seen.tolist()) == list(range(counts.size))


def test_initial_model_particle_pools_win_over_the_ladder():
    """``preserve_image_order`` keeps RELION's pool-of-three FFI boundary."""

    counts = np.full(90, 180, dtype=np.int32)
    plans = _plan(
        True, counts=counts, image_batch_size=136, preserve_image_order=True
    )
    capacities = _capacities(plans)
    assert capacities == [135]  # max(3, (136 // 3) * 3), not a ladder rung
    assert all(capacity % 3 == 0 for capacity in capacities)


def test_consecutive_mixed_buckets_are_not_snapped():
    counts = np.full(60, 180, dtype=np.int32)
    plans = _plan(
        True,
        counts=counts,
        image_batch_size=136,
        preserve_image_order=True,
        consecutive_mixed_bucket_size=9,
    )
    assert _capacities(plans) not in ([128], [64], [32])
