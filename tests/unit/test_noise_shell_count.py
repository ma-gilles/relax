"""The ``noise_shell_count`` option: RELION's full-image pixel count or the pixels actually summed.

RELION normalises the noise sums of an expectation by ``Npix_per_shell``, counted on the full
image (``ml_optimiser.cpp:5717-5730``). Below the box, the shells up to ``current_size / 2`` are
summed on the cropped image (``windowFourierTransform``, rows ``-(cs/2 - 1) .. +cs/2``), which has no
row ``-cs/2``; the pixels of that row whose shell is ``cs/2`` are in the count and in no sum, so
``sigma2_noise[cs/2]`` comes out low. ``"summed"`` divides every shell by the pixels it summed.
The oracles enumerate RELION's FFTW labels in plain numpy; RELION's image and current sizes are even.
"""

import numpy as np
import pytest
from helpers.float_compare import assert_matches
from helpers.tiny_refinement import record_calls, run_tiny_refinement

from relax.helpers.types import make_noise_stats
from relax.reconstruction import noise_relion
from relax.refinement import noise_updates
from relax.refinement.refinement_options import RelionConsistencyOptions

pytestmark = pytest.mark.unit


def _fftw_half_labels(size):
    """RELION's (ip, jp) of a ``size x (size / 2 + 1)`` FFTW half image (fftw.h:99-109)."""
    half_width = size // 2 + 1
    rows = np.arange(size)
    ip = np.where(rows < half_width, rows, rows - size)
    return ip[:, None], np.arange(half_width)[None, :]


def _relion_round(values):
    return np.floor(values + 0.5).astype(np.int64)


def _counted(size, max_shell):
    """Per-shell count of the stored pixels with ires <= max_shell, without the redundant jp = 0, ip < 0."""
    ip, jp = _fftw_half_labels(size)
    shell = _relion_round(np.sqrt((ip * ip + jp * jp).astype(np.float64)))
    keep = (shell <= max_shell) & ~((jp == 0) & (ip < 0))
    return np.bincount(shell[keep], minlength=max_shell + 1)


def _npix_per_shell(box):
    """RELION's Npix_per_shell: every shell counted on the full image."""
    return _counted(box, box // 2)


def _summed_per_shell(box, current_size):
    """Pixels each shell sums: the cropped image up to shell cs / 2, the full image (power_img) above."""
    count = _npix_per_shell(box).copy()
    if current_size is not None and current_size < box:
        count[: current_size // 2 + 1] = _counted(current_size, current_size // 2)
    return count


@pytest.mark.parametrize("box,current_size", [(16, 8), (16, 12), (32, 16), (32, 30), (64, 32), (64, 48), (128, 64)])
def test_summed_count_is_the_cropped_image_up_to_the_current_shell(box, current_size):
    summed = noise_relion.summed_noise_pixels_per_shell((box, box), current_size)
    expected = _summed_per_shell(box, current_size)
    assert_matches(np.asarray(summed, dtype=np.int64), expected)
    full = _npix_per_shell(box)
    # Only shell cs / 2 differs, and it is short of RELION's count.
    differing = np.flatnonzero(expected != full)
    assert differing.tolist() == [current_size // 2]
    assert expected[current_size // 2] < full[current_size // 2]


@pytest.mark.parametrize("box,current_size,summed,counted", [(64, 32, 52, 56), (128, 64, 89, 94), (256, 128, 212, 220)])
def test_the_deficit_of_the_current_shell(box, current_size, summed, counted):
    """The audit's numbers: 7.1%, 5.3% and 3.6% of shell cs / 2 are counted but never summed."""
    shell = current_size // 2
    assert _summed_per_shell(box, current_size)[shell] == summed and _npix_per_shell(box)[shell] == counted
    assert int(noise_relion.summed_noise_pixels_per_shell((box, box), current_size)[shell]) == summed


@pytest.mark.parametrize("box,current_size", [(16, 16), (32, None), (32, 32)])
def test_at_the_box_the_two_counts_agree(box, current_size):
    summed = noise_relion.summed_noise_pixels_per_shell((box, box), current_size)
    assert_matches(np.asarray(summed, dtype=np.int64), _npix_per_shell(box))


@pytest.mark.parametrize("box,current_size", [(16, 8), (32, 16), (32, 22), (32, 32), (64, 32)])
def test_sigma2_default_divides_by_relions_count_and_summed_by_the_summed_pixels(box, current_size):
    """Every summed pixel carries power 2 sigma2: the summed count recovers sigma2 in every shell."""
    true_sigma2 = 3.0
    sumw = 7.0
    sums = 2.0 * sumw * true_sigma2 * _summed_per_shell(box, current_size).astype(np.float64)
    residual, image_power = 0.25 * sums, 0.75 * sums

    default = np.asarray(noise_relion.normalize_wsum_to_sigma2_noise(residual, image_power, sumw, (box, box)))
    summed = np.asarray(
        noise_relion.normalize_wsum_to_sigma2_noise(
            residual, image_power, sumw, (box, box), summed_current_size=current_size
        )
    )

    relion = sums / (2.0 * sumw * _npix_per_shell(box))
    assert_matches(default, relion, rtol=1e-6)  # Npix_per_shell is float32 in production
    assert_matches(summed, np.full(box // 2 + 1, true_sigma2), rtol=1e-6)
    if current_size < box:
        shell = current_size // 2
        low = _summed_per_shell(box, current_size)[shell] / _npix_per_shell(box)[shell]
        assert 0.8 < low < 0.97  # 12.5% at box 16, 7.1% at box 64
        assert_matches(default[shell], true_sigma2 * low, rtol=1e-6)
        others = np.arange(box // 2 + 1) != shell
        assert_matches(default[others], summed[others], rtol=1e-6)


SHAPE = (16, 16)
N_SHELLS = SHAPE[0] // 2 + 1


def _stats(rng, n_groups=None):
    shape = (N_SHELLS,) if n_groups is None else (n_groups, N_SHELLS)
    return make_noise_stats(
        wsum_sigma2_noise=rng.uniform(-1.0, 0.0, shape),
        wsum_img_power=rng.uniform(2.0, 3.0, shape),
        wsum_sigma2_offset=1.0,
        sumw=5.0 if n_groups is None else np.full(n_groups, 5.0),
    )


def _model(n_groups=None):
    rows = np.ones(SHAPE[0] * SHAPE[1]) if n_groups is None else np.ones((n_groups, SHAPE[0] * SHAPE[1]))
    radial = np.ones(N_SHELLS) if n_groups is None else np.ones((n_groups, N_SHELLS))
    return noise_updates.NoiseModel(
        variance_per_half=[rows, rows.copy()],
        radial_per_half=[radial, radial.copy()],
        average_variance=rows,
        average_radial=radial,
    )


@pytest.mark.parametrize("k_class,n_groups", [(False, None), (True, None), (False, 2), (True, 2)])
@pytest.mark.parametrize("summed_current_size", [None, 8])
def test_every_noise_update_branch_hands_the_normalisation_the_size(monkeypatch, k_class, n_groups, summed_current_size):
    """K=1 per half, Class3D shared, and both with one spectrum per optics group."""
    calls = record_calls(monkeypatch, noise_relion, "normalize_wsum_to_sigma2_noise")
    rng = np.random.default_rng(0)
    result = noise_updates.update_posterior_noise_variance(
        [_stats(rng, n_groups), _stats(rng, n_groups)],
        _model(n_groups),
        SHAPE,
        k_class_enabled=k_class,
        firstiter_cc=False,
        summed_current_size=summed_current_size,
    )
    assert calls and {kwargs.get("summed_current_size") for _, kwargs in calls} == {summed_current_size}
    assert np.all(np.isfinite(np.asarray(result.noise_from_res)))


def test_noise_update_changes_only_the_current_shell():
    rng_a, rng_b = np.random.default_rng(3), np.random.default_rng(3)
    relion = noise_updates.update_posterior_noise_variance(
        [_stats(rng_a), _stats(rng_a)], _model(), SHAPE, k_class_enabled=False, firstiter_cc=False
    )
    summed = noise_updates.update_posterior_noise_variance(
        [_stats(rng_b), _stats(rng_b)], _model(), SHAPE, k_class_enabled=False, firstiter_cc=False,
        summed_current_size=8,
    )
    ratio = np.asarray(summed.noise_from_res) / np.asarray(relion.noise_from_res)
    expected = np.ones(N_SHELLS)
    expected[4] = _npix_per_shell(16)[4] / _summed_per_shell(16, 8)[4]
    assert expected[4] > 1.05
    assert_matches(ratio, expected, rtol=1e-6)


@pytest.mark.parametrize("count", ["relion", "summed"])
def test_refinement_hands_the_noise_update_the_expectations_image_size(monkeypatch, count):
    """The image window of each numbered expectation reaches the K=1 noise update (None at the box)."""
    import dataclasses

    from relax.refinement import local_sampling

    # plan_expectation_sampling (local_sampling.py) plans the windows the loop then reads.
    plan = local_sampling.plan_expectation_windows
    image_sizes = iter([6, 8])  # the 8-pixel mock box itself never plans a window below the box

    def plan_below_the_box(*args, **kwargs):
        return dataclasses.replace(plan(*args, **kwargs), image_current_size=next(image_sizes))

    monkeypatch.setattr(local_sampling, "plan_expectation_windows", plan_below_the_box)
    calls = record_calls(monkeypatch, noise_relion, "normalize_wsum_to_sigma2_noise")

    run_tiny_refinement(monkeypatch, consistency=RelionConsistencyOptions(noise_shell_count=count))

    assert len(calls) == 4  # one update per half and iteration
    received = [kwargs["summed_current_size"] for _, kwargs in calls]
    assert received == ([None] * 4 if count == "relion" else [6, 6, None, None])
