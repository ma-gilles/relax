"""Per-optics-group sigma2_noise: state layout and the M-step update.

RELION keeps one noise spectrum per optics group and normalises each group's sums
with that group's own weight (``maximizationOtherParameters``, ml_optimiser.cpp
5246-5285). A run with G groups must update group g exactly as a one-group run on
group g's sums would, and a one-group run must keep the flat layout.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from helpers.float_compare import assert_matches

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
            assert_matches(result.noise_from_res_per_half[k][g], single.noise_from_res_per_half[k])
            assert_matches(
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
    assert_matches(result.noise_from_res_per_half[0][1], np.full(N_SHELLS, 7.0))
    assert_matches(np.asarray(result.noise_variance_per_half[0][1]), previous_rows[1])
    assert not np.allclose(result.noise_from_res_per_half[0][0], 1.0)


@pytest.mark.unit
def test_noise_state_layout_per_group():
    one = noise_updates._normalize_noise_variance_per_half([np.ones((1, 256)), np.ones(256)])
    assert [a.shape for a in one] == [(256,), (256,)]
    rows = noise_updates._normalize_noise_variance_per_half([np.ones((2, 256)), 2 * np.ones((2, 256))])
    assert [a.shape for a in rows] == [(2, 256), (2, 256)]
    assert_matches(np.asarray(noise_updates._mean_noise_variance(rows)), 1.5 * np.ones((2, 256)))
    per_half, mean = noise_updates._noise_radial_history(rows, SHAPE, dtype=np.float64)
    assert per_half[0].shape == (2, N_SHELLS) and np.asarray(mean).shape == (2, N_SHELLS)


@pytest.mark.unit
def test_k1_class_mass_and_sums_take_per_group_weight_sums():
    # A K=1 result with one weight sum per optics group: the class M-step mass is their total,
    # and summing noise statistics keeps the per-group layout (one scalar stays one float).
    from relax.classification import k_class_results as results

    groups = make_noise_stats(
        wsum_sigma2_noise=np.ones((2, 5)), wsum_img_power=np.zeros((2, 5)),
        wsum_sigma2_offset=1.0, sumw=np.array([3.0, 4.5]),
    )
    single = make_noise_stats(
        wsum_sigma2_noise=np.ones(5), wsum_img_power=np.zeros(5), wsum_sigma2_offset=1.0, sumw=7.5,
    )
    for stats in (groups, single):
        mass = results._resolve_class_mstep_posterior_sums(
            noise_stats=(stats,), class_posterior_sums_full=np.array([9.0]), class_posterior_sums_override=None,
        )
        np.testing.assert_array_equal(mass, [7.5])
    np.testing.assert_array_equal(results._summed_sumw((groups, groups)), [6.0, 9.0])
    assert results._summed_sumw((single, single)) == 15.0 and type(results._summed_sumw((single,))) is float
    np.testing.assert_array_equal(results._sum_noise_stats((groups,), host_arrays=True).sumw, [3.0, 4.5])


@pytest.mark.unit
def test_k1_aggregate_noise_keeps_per_group_weight_sums():
    from relax.classification import k_class_results as results

    groups = make_noise_stats(
        wsum_sigma2_noise=np.ones((2, 5)), wsum_img_power=np.ones((2, 5)),
        wsum_sigma2_offset=1.0, sumw=np.array([3.0, 4.5]),
    )
    for host in (True, False):
        aggregate = results._sum_k_class_noise_stats((groups,), np.array([7.5]), host_arrays=host)
        np.testing.assert_array_equal(np.asarray(aggregate.sumw), [3.0, 4.5])
        np.testing.assert_array_equal(np.asarray(aggregate.wsum_img_power), np.ones((2, 5)))


@pytest.mark.unit
def test_sigma_offset_update_uses_the_total_weight_over_optics_groups():
    groups = make_noise_stats(
        wsum_sigma2_noise=np.ones((2, 5)), wsum_img_power=np.zeros((2, 5)),
        wsum_sigma2_offset=60.0, sumw=np.array([3.0, 7.0]),
    )
    single = make_noise_stats(wsum_sigma2_noise=np.ones(5), wsum_img_power=np.zeros(5), wsum_sigma2_offset=60.0, sumw=10.0)
    kwargs = dict(noise_stats_per_half_per_class=None, current_sigma_offset_angstrom_per_half=[5.0, 5.0],
                  n_classes=1, state_fallback_offsets_angstrom=np.nan)
    a = noise_updates.update_c1_sigma_offset_from_posterior(noise_stats_per_half=[groups, groups], **kwargs)
    b = noise_updates.update_c1_sigma_offset_from_posterior(noise_stats_per_half=[single, single], **kwargs)
    assert a.current_sigma_offset_angstrom_per_half == b.current_sigma_offset_angstrom_per_half


@pytest.mark.unit
def test_empty_high_shells_take_the_previous_shell():
    # A group on a coarser grid leaves the outer reference shells empty; RELION fills an
    # empty shell from the previous one (ml_optimiser.cpp:5281-5284).
    from relax.reconstruction import noise_relion

    shape = (8, 8)
    wsum = np.array([0.0, 4.0, 3.0, 2.0, 0.0])
    sigma2 = np.asarray(noise_relion.normalize_wsum_to_sigma2_noise(wsum, np.zeros(5), 1.0, shape))
    assert sigma2[4] == sigma2[3] > 1e-14
