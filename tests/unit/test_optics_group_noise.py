"""Per-optics-group sigma2_noise: state layout and the M-step update.

RELION keeps one noise spectrum per optics group and normalises each group's sums
with that group's own weight (``maximizationOtherParameters``, ml_optimiser.cpp
5246-5285). A run with G groups must update group g exactly as a one-group run on
group g's sums would, and a one-group run must keep the flat layout.
"""

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
        stats_per_half,
        noise_updates.NoiseModel(
            variance_per_half=list(noise_per_half),
            radial_per_half=radial_per_half,
            average_variance=noise_updates._mean_noise_variance(noise_per_half),
            average_radial=np.mean(np.stack(radial_per_half), axis=0),
        ),
        SHAPE,
        k_class_enabled=False,
        firstiter_cc=False,
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
        assert result.model.variance_per_half[k].shape == (n_groups, SHAPE[0] * SHAPE[1])
        for g in range(n_groups):
            single = _update(
                [group_stats[0][g], group_stats[1][g]],
                [np.ones(SHAPE[0] * SHAPE[1])] * 2,
                [np.ones(N_SHELLS)] * 2,
            )
            assert_matches(result.noise_from_res_per_half[k][g], single.noise_from_res_per_half[k])
            assert_matches(
                np.asarray(result.model.variance_per_half[k][g]), np.asarray(single.model.variance_per_half[k])
            )


@pytest.mark.unit
def test_class3d_update_keeps_one_spectrum_per_group_and_ignores_the_empty_accumulator():
    """Class3D (subtomograms, several groups): the all-data half's per-group update; the empty second
    accumulator (one flat placeholder, no mass) adds nothing."""

    rng = np.random.default_rng(2)
    stats = _stats(rng, n_groups=2, sumw=np.array([6.0, 4.0]))
    empty = make_noise_stats(
        wsum_sigma2_noise=np.zeros(N_SHELLS), wsum_img_power=np.zeros(N_SHELLS), wsum_sigma2_offset=0.0, sumw=0.0
    )
    previous_radial = np.ones((2, N_SHELLS))
    previous_rows = np.ones((2, SHAPE[0] * SHAPE[1]))
    result = noise_updates.update_posterior_noise_variance(
        [stats, empty],
        noise_updates.NoiseModel(
            variance_per_half=[previous_rows, previous_rows],
            radial_per_half=[previous_radial, previous_radial],
            average_variance=previous_rows,
            average_radial=previous_radial,
        ),
        SHAPE,
        k_class_enabled=True,
        firstiter_cc=False,
    )
    k1 = _update([stats, stats], [previous_rows, previous_rows], [previous_radial, previous_radial])
    for k in range(2):
        assert_matches(result.noise_from_res_per_half[k], k1.noise_from_res_per_half[0])
        assert_matches(np.asarray(result.model.variance_per_half[k]), np.asarray(k1.model.variance_per_half[0]))
    assert result.model.variance_per_half[0] is result.model.variance_per_half[1]
    assert result.model.radial_per_half[0] is not result.model.radial_per_half[1]


@pytest.mark.unit
def test_firstiter_cc_retains_noise_buffers_and_history_precision():
    rows = [np.ones(256, dtype=np.float32), np.full(256, 2.0, dtype=np.float32)]
    model = noise_updates.noise_model_from_pixels(rows, SHAPE, dtype=np.float32)
    original_pixels = list(model.variance_per_half)
    result = noise_updates.update_posterior_noise_variance(
        [_stats(np.random.default_rng(3)), _stats(np.random.default_rng(4))],
        model, SHAPE, k_class_enabled=False, firstiter_cc=True,
    )
    assert result.model.variance_per_half is model.variance_per_half
    assert all(a is b for a, b in zip(result.model.variance_per_half, original_pixels, strict=True))
    assert result.model.radial_per_half is model.radial_per_half
    assert result.model.average_radial is model.average_radial
    assert result.model.average_radial.dtype == np.float32
    assert result.noise_from_res.dtype == np.float64
    assert_matches(result.noise_from_res, np.mean(np.stack(model.radial_per_half), axis=0))


@pytest.mark.unit
def test_k1_update_replaces_pixels_in_the_owned_list():
    model = noise_updates.noise_model_from_pixels(
        [np.ones(256, dtype=np.float32), np.ones(256, dtype=np.float32)], SHAPE, dtype=np.float32,
    )
    original_pixels = list(model.variance_per_half)
    result = noise_updates.update_posterior_noise_variance(
        [_stats(np.random.default_rng(5)), _stats(np.random.default_rng(6))],
        model, SHAPE, k_class_enabled=False, firstiter_cc=False,
    )
    assert result.model.variance_per_half is model.variance_per_half
    assert all(a is not b for a, b in zip(result.model.variance_per_half, original_pixels, strict=True))
    assert result.model.radial_per_half is result.noise_from_res_per_half


@pytest.mark.unit
@pytest.mark.parametrize("n_groups", [1, 3])
def test_checkpoint_noise_restores_shell_precision_and_group_layout(n_groups):
    shell_shape = (N_SHELLS,) if n_groups == 1 else (n_groups, N_SHELLS)
    shells = [np.ones(shell_shape, dtype=np.float32), np.full(shell_shape, 3.0, dtype=np.float32)]
    restored = noise_updates.noise_model_from_shells(shells, SHAPE)
    pixel_shape = (256,) if n_groups == 1 else (n_groups, 256)
    assert [row.shape for row in restored.variance_per_half] == [pixel_shape, pixel_shape]
    assert restored.average_radial.dtype == np.float64
    assert all(row.dtype == np.float64 for row in restored.radial_per_half)
    assert_matches(restored.average_radial, np.full(shell_shape, 2.0))


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
    assert_matches(np.asarray(result.model.variance_per_half[0][1]), previous_rows[1])
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
        assert_matches(mass, [7.5])
    assert_matches(results._summed_sumw((groups, groups)), [6.0, 9.0])
    assert results._summed_sumw((single, single)) == 15.0 and type(results._summed_sumw((single,))) is float
    assert_matches(results._sum_noise_stats((groups,), host_arrays=True).sumw, [3.0, 4.5])


@pytest.mark.unit
def test_k1_aggregate_noise_keeps_per_group_weight_sums():
    from relax.classification import k_class_results as results

    groups = make_noise_stats(
        wsum_sigma2_noise=np.ones((2, 5)), wsum_img_power=np.ones((2, 5)),
        wsum_sigma2_offset=1.0, sumw=np.array([3.0, 4.5]),
    )
    for host in (True, False):
        aggregate = results._sum_k_class_noise_stats((groups,), np.array([7.5]), host_arrays=host)
        assert_matches(np.asarray(aggregate.sumw), [3.0, 4.5])
        assert_matches(np.asarray(aggregate.wsum_img_power), np.ones((2, 5)))


@pytest.mark.unit
def test_k_class_aggregate_noise_sums_per_group_weight_sums_over_classes():
    """Class3D with several optics groups: each class carries ``[G]`` weight sums; the aggregate
    is their per-group total, rescaled with the image power when a class's mass is not its
    responsibility."""
    from relax.classification import k_class_results as results

    a = make_noise_stats(
        wsum_sigma2_noise=np.ones((2, 5)), wsum_img_power=np.ones((2, 5)),
        wsum_sigma2_offset=1.0, sumw=np.array([3.0, 1.0]),
    )
    b = make_noise_stats(
        wsum_sigma2_noise=np.ones((2, 5)), wsum_img_power=2.0 * np.ones((2, 5)),
        wsum_sigma2_offset=1.0, sumw=np.array([2.0, 4.0]),
    )
    aggregate = results._sum_k_class_noise_stats((a, b), np.array([4.0, 6.0]))
    assert_matches(np.asarray(aggregate.sumw), [5.0, 5.0])
    assert_matches(np.asarray(aggregate.wsum_img_power), 3.0 * np.ones((2, 5)))
    # Responsibilities 2 and 3 against class masses 4 and 6: every class's sums scale by 1/2.
    rescaled = results._sum_k_class_noise_stats((a, b), np.array([2.0, 3.0]))
    assert_matches(np.asarray(rescaled.sumw), [2.5, 2.5])
    assert_matches(np.asarray(rescaled.wsum_img_power), 1.5 * np.ones((2, 5)))


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
def test_class3d_sigma_offset_per_class_diagnostic_takes_per_group_weight_sums():
    """Class3D with several optics groups: each class's noise statistics carry ``[G]`` weight sums."""
    groups = [
        make_noise_stats(
            wsum_sigma2_noise=np.ones((2, 5)), wsum_img_power=np.zeros((2, 5)),
            wsum_sigma2_offset=w, sumw=np.array(n),
        )
        for w, n in ((30.0, [1.0, 2.0]), (30.0, [3.0, 4.0]))
    ]
    single = [
        make_noise_stats(wsum_sigma2_noise=np.ones(5), wsum_img_power=np.zeros(5), wsum_sigma2_offset=30.0, sumw=n)
        for n in (3.0, 7.0)
    ]
    kwargs = dict(current_sigma_offset_angstrom_per_half=[5.0, 5.0], n_classes=2, state_fallback_offsets_angstrom=np.nan)
    a = noise_updates.update_c1_sigma_offset_from_posterior(
        noise_stats_per_half=[groups[0], groups[1]], noise_stats_per_half_per_class=[groups, None], **kwargs
    )
    b = noise_updates.update_c1_sigma_offset_from_posterior(
        noise_stats_per_half=[single[0], single[1]], noise_stats_per_half_per_class=[single, None], **kwargs
    )
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
