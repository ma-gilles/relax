"""A half scored as shards (disjoint particle subsets) joins into the result of the whole half.

``merge_half_shards`` (relax.steps.score_outputs) puts per-particle rows back at their particles and adds the
additive sums in float64 in shard order. Every field of the result records has a rule, so a field added later
without one fails here instead of being dropped by a multi-GPU run.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from relax.steps.score_outputs import (
    HALF_RESULT_MERGE_RULES,
    NOISE_STATS_MERGE_RULES,
    RELION_STATS_MERGE_RULES,
    HalfScoreResult,
    merge_half_shards,
)
from relax.types import NoiseStats, RelionStats, make_noise_stats, make_relion_stats

pytestmark = pytest.mark.unit


def test_every_result_field_has_a_merge_rule():
    assert {f.name for f in dataclasses.fields(HalfScoreResult)} == set(HALF_RESULT_MERGE_RULES)
    assert set(RelionStats._fields) == set(RELION_STATS_MERGE_RULES)
    assert set(NoiseStats._fields) == set(NOISE_STATS_MERGE_RULES)


def _result(rng, units, *, n_rot=5, n_shells=4, volume=12):
    n = units.size
    return HalfScoreResult(
        ha=rng.integers(0, 100, n).astype(np.int32),
        Ft_y=(rng.standard_normal(volume) + 1j * rng.standard_normal(volume)).astype(np.complex64),
        Ft_ctf=rng.random(volume).astype(np.float32),
        em_stats=make_relion_stats(
            log_evidence_per_image=rng.standard_normal(n),
            best_log_score_per_image=rng.standard_normal(n),
            max_posterior_per_image=rng.random(n),
            rotation_posterior_sums=rng.random(n_rot),
            host_arrays=True,
        ),
        noise_stats=make_noise_stats(
            wsum_sigma2_noise=rng.random(n_shells),
            wsum_img_power=rng.random(n_shells),
            wsum_sigma2_offset=float(rng.random()),
            sumw=float(n),
            wsum_norm_correction=rng.random(n),
            wsum_scale_correction_xa=rng.random(2),
            wsum_scale_correction_aa=rng.random(2),
            host_arrays=True,
        ),
        best_pose_rotations=rng.standard_normal((n, 3, 3)).astype(np.float32),
        best_pose_rotation_eulers=rng.standard_normal((n, 3)),
        best_pose_translations=rng.standard_normal((n, 3)).astype(np.float32),
        coarse_ha=rng.integers(0, 50, n).astype(np.int32),
        significant_counts=rng.integers(1, 9, n).astype(np.int32),
        mstep_full_half_axis=0,
        mstep_accumulator_shape=(5, 5, 3),
    )


def test_shards_join_rows_by_particle_and_sums_in_float64():
    rng = np.random.default_rng(3)
    shards = [np.array([4, 0, 6]), np.array([1, 2]), np.array([5, 3])]  # a non-contiguous partition of 7
    parts = [_result(rng, units) for units in shards]
    merged = merge_half_shards(parts, shards, 7)

    for name in ("ha", "coarse_ha", "significant_counts", "best_pose_rotations", "best_pose_translations"):
        expected = np.empty((7,) + np.asarray(getattr(parts[0], name)).shape[1:], getattr(parts[0], name).dtype)
        for part, units in zip(parts, shards):
            expected[units] = getattr(part, name)
        np.testing.assert_array_equal(getattr(merged, name), expected)
        assert getattr(merged, name).dtype == getattr(parts[0], name).dtype
    for units, part in zip(shards, parts):
        np.testing.assert_array_equal(
            merged.em_stats.max_posterior_per_image[units], part.em_stats.max_posterior_per_image
        )
        np.testing.assert_array_equal(
            merged.noise_stats.wsum_norm_correction[units], part.noise_stats.wsum_norm_correction
        )

    total = np.zeros(12, np.complex128)
    for part in parts:
        total += part.Ft_y
    assert merged.Ft_y.dtype == np.complex64
    np.testing.assert_array_equal(merged.Ft_y, total.astype(np.complex64))
    np.testing.assert_allclose(
        merged.em_stats.rotation_posterior_sums, sum(p.em_stats.rotation_posterior_sums for p in parts)
    )
    assert merged.noise_stats.sumw == 7.0 and isinstance(merged.noise_stats.wsum_sigma2_offset, float)
    assert merged.mstep_accumulator_shape == (5, 5, 3)


def test_one_shard_is_the_result_itself():
    rng = np.random.default_rng(0)
    units = np.arange(4)
    part = _result(rng, units)
    assert merge_half_shards([part], [units], 4) is part


def test_shards_that_miss_a_particle_or_disagree_are_refused():
    rng = np.random.default_rng(1)
    shards = [np.array([0, 1]), np.array([3])]
    with pytest.raises(ValueError, match="do not cover"):
        merge_half_shards([_result(rng, u) for u in shards], shards, 4)
    parts = [_result(rng, u) for u in (np.array([0]), np.array([1]))]
    parts[1].mstep_accumulator_shape = (6, 6, 4)
    with pytest.raises(ValueError, match="differs between shards"):
        merge_half_shards(parts, [np.array([0]), np.array([1])], 2)


def test_a_field_of_an_unsharded_route_is_refused():
    rng = np.random.default_rng(2)
    shards = [np.array([0]), np.array([1])]
    parts = [_result(rng, u) for u in shards]
    parts[0].profile_summary = {"seconds": 1.0}
    with pytest.raises(NotImplementedError, match="profile_summary"):
        merge_half_shards(parts, shards, 2)
