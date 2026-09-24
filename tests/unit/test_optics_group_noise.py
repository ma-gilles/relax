"""Per-optics-group sigma2_noise: state layout and the M-step update.

RELION keeps one noise spectrum per optics group and normalises each group's sums
with that group's own weight (``maximizationOtherParameters``, ml_optimiser.cpp
5246-5285). A run with G groups must update group g exactly as a one-group run on
group g's sums would, and a one-group run must keep the flat layout.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from relax.helpers.types import make_noise_stats
from relax.refinement import noise_updates

SHAPE = (16, 16)
N_SHELLS = SHAPE[0] // 2 + 1


def _stats(rng, n_groups=None, sumw=None):
    shape = (N_SHELLS,) if n_groups is None else (n_groups, N_SHELLS)
    return make_noise_stats(
        wsum_sigma2_noise=rng.uniform(-1.0, 0.0, shape),
        wsum_img_power=rng.uniform(2.0, 3.0, shape),
        wsum_sigma2_offset=1.0,
        sumw=rng.uniform(5.0, 9.0) if sumw is None else sumw,
    )


def _update(stats_per_half, noise_per_half, radial_per_half):
    return noise_updates.update_posterior_noise_variance(
        noise_stats_per_half=stats_per_half,
        noise_variance_per_half=list(noise_per_half),
        previous_noise_radial_per_half=radial_per_half,
        previous_noise_radial=np.mean(np.stack(radial_per_half), axis=0),
        cryo=SimpleNamespace(image_shape=SHAPE),
        k_class_enabled=False,
        relion_firstiter_cc_this_iter=False,
        iteration=3,
        cs=SHAPE[0],
    )


@pytest.mark.unit
def test_per_group_update_equals_one_group_updates():
    rng = np.random.default_rng(0)
    n_groups = 3
    group_stats = [[_stats(rng, sumw=float(g + 4)) for g in range(n_groups)] for _ in range(2)]
    stacked = [
        make_noise_stats(
            wsum_sigma2_noise=np.stack([s.wsum_sigma2_noise for s in half]),
            wsum_img_power=np.stack([s.wsum_img_power for s in half]),
            wsum_sigma2_offset=2.0,
            sumw=np.array([float(s.sumw) for s in half]),
        )
        for half in group_stats
    ]
    previous_rows = [np.ones((n_groups, SHAPE[0] * SHAPE[1])) for _ in range(2)]
    previous_radial = [np.ones((n_groups, N_SHELLS)) for _ in range(2)]
    result = _update(stacked, previous_rows, previous_radial)

    for k in range(2):
        assert result.noise_from_res_per_half[k].shape == (n_groups, N_SHELLS)
        assert result.noise_variance_per_half[k].shape == (n_groups, SHAPE[0] * SHAPE[1])
        for g in range(n_groups):
            single = _update(
                [group_stats[0][g], group_stats[1][g]],
                [np.ones(SHAPE[0] * SHAPE[1])] * 2,
                [np.ones(N_SHELLS)] * 2,
            )
            np.testing.assert_array_equal(result.noise_from_res_per_half[k][g], single.noise_from_res_per_half[k])
            np.testing.assert_array_equal(
                np.asarray(result.noise_variance_per_half[k][g]), np.asarray(single.noise_variance_per_half[k])
            )


@pytest.mark.unit
def test_group_without_noise_sums_keeps_its_spectrum():
    rng = np.random.default_rng(1)
    stats = _stats(rng, n_groups=2, sumw=np.array([6.0, 0.0]))
    stats = stats._replace(
        wsum_sigma2_noise=np.stack([stats.wsum_sigma2_noise[0], np.zeros(N_SHELLS)]),
        wsum_img_power=np.stack([stats.wsum_img_power[0], np.zeros(N_SHELLS)]),
    )
    previous_radial = np.stack([np.ones(N_SHELLS), np.full(N_SHELLS, 7.0)])
    previous_rows = np.stack([np.ones(SHAPE[0] * SHAPE[1]), np.full(SHAPE[0] * SHAPE[1], 7.0)])
    result = _update([stats, stats], [previous_rows, previous_rows], [previous_radial, previous_radial])
    np.testing.assert_array_equal(result.noise_from_res_per_half[0][1], np.full(N_SHELLS, 7.0))
    np.testing.assert_array_equal(np.asarray(result.noise_variance_per_half[0][1]), previous_rows[1])
    assert not np.allclose(result.noise_from_res_per_half[0][0], 1.0)


@pytest.mark.unit
def test_noise_state_layout_per_group():
    one = noise_updates._normalize_noise_variance_per_half([np.ones((1, 256)), np.ones(256)])
    assert [a.shape for a in one] == [(256,), (256,)]
    rows = noise_updates._normalize_noise_variance_per_half([np.ones((2, 256)), 2 * np.ones((2, 256))])
    assert [a.shape for a in rows] == [(2, 256), (2, 256)]
    np.testing.assert_array_equal(np.asarray(noise_updates._mean_noise_variance(rows)), 1.5 * np.ones((2, 256)))
    per_half, mean = noise_updates._noise_radial_history(rows, SHAPE, dtype=np.float64)
    assert per_half[0].shape == (2, N_SHELLS) and np.asarray(mean).shape == (2, N_SHELLS)
