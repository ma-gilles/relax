"""The shell numbers scripts/plot_ppca_mean_vs_truth.py reports for a run's mean against a truth mean."""

import numpy as np
import pytest

from scripts import plot_ppca_mean_vs_truth as mean_vs_truth

pytestmark = pytest.mark.unit

SHELLS = np.repeat([0, 1, 2, 3], 3)


def _volume(seed: int) -> np.ndarray:
    """Twelve complex coefficients, three per shell 0..3."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=12) + 1j * rng.normal(size=12)


def test_the_shell_correlation_is_one_for_a_copy_minus_one_for_its_negative_and_nan_for_an_empty_shell():
    """Each shell is its own correlation; a shell with no coefficients (or a zero mean there) is nan."""
    truth = _volume(1)
    assert np.allclose(mean_vs_truth.shell_correlation(truth, truth, SHELLS, 4)[:4], 1.0)
    assert np.allclose(mean_vs_truth.shell_correlation(-truth, truth, SHELLS, 4)[:4], -1.0)
    assert np.isnan(mean_vs_truth.shell_correlation(truth, truth, SHELLS, 5)[4])
    zero_outside = truth * (SHELLS < 2)
    out = mean_vs_truth.shell_correlation(zero_outside, truth, SHELLS, 4)
    assert np.allclose(out[:2], 1.0) and np.isnan(out[2]) and np.isnan(out[3])


def test_the_active_radius_is_the_outermost_shell_with_a_coefficient():
    """The model is band-limited, so its radius is where its coefficients stop."""
    mean = _volume(2) * (SHELLS <= 2)
    assert mean_vs_truth.active_radius(mean, SHELLS) == 2


def test_the_in_band_numbers_of_a_scaled_copy():
    """A half-size copy has correlation 1, relative error 1/2 and power ratio 1/4 over the band it occupies."""
    truth = _volume(3)
    numbers = mean_vs_truth.in_band_numbers(0.5 * truth, truth, SHELLS, 3)
    assert numbers == pytest.approx({"fsc": 1.0, "relative_error": 0.5, "power_ratio": 0.25})
    inside = mean_vs_truth.in_band_numbers(truth * (SHELLS <= 1), truth, SHELLS, 1)
    assert inside == pytest.approx({"fsc": 1.0, "relative_error": 0.0, "power_ratio": 1.0})


@pytest.mark.parametrize(
    ("fsc", "shell"), [([1, 0.9, 0.8, 0.4, 0.2], 3), ([1, 0.4, 0.9], 1), ([1, 0.9, 0.8], 2), ([1, 0.9, np.nan, 0.8], 2)]
)
def test_the_half_fsc_shell_is_the_first_shell_below_one_half(fsc, shell):
    """The first shell from 1 whose correlation is below 0.5 or missing; the last shell when none is."""
    assert mean_vs_truth.half_fsc_shell(fsc) == shell


def test_the_band_numbers_of_a_half_size_mean_and_two_half_size_loadings():
    """Per band: power ratio 1/4 and correlation 1 for a half-size mean, loading power 2 x 1/4 of the truth's."""
    truth = _volume(4)
    loadings = 0.5 * np.stack([truth, truth], axis=1)
    out = mean_vs_truth.band_numbers(0.5 * truth, loadings, truth, SHELLS, 3, bands=((1, 1), (2, 3)))
    assert out["band_power_ratio"] == pytest.approx([0.25, 0.25])
    assert out["band_fsc"] == pytest.approx([1.0, 1.0])
    assert out["band_loading_ratio"] == pytest.approx([0.5, 0.5])


def test_a_band_above_the_radius_or_without_truth_power_is_nan_and_an_empty_mean_has_no_correlation():
    """Missing measurements stay nan; a mean that is zero in a band has power ratio 0 and no correlation."""
    truth, loadings = _volume(5), np.zeros((12, 1), complex)
    above = mean_vs_truth.band_numbers(truth, loadings, truth, SHELLS, 1, bands=((1, 1), (2, 3)))
    assert above["band_power_ratio"][0] == pytest.approx(1.0) and np.isnan(above["band_power_ratio"][1])
    assert np.isnan(above["band_fsc"][1]) and np.isnan(above["band_loading_ratio"][1])
    empty = mean_vs_truth.band_numbers(0 * truth, loadings, truth, SHELLS, 3, bands=((1, 1),))
    assert empty["band_power_ratio"] == [0.0] and np.isnan(empty["band_fsc"][0])
    no_truth = mean_vs_truth.band_numbers(truth, loadings, 0 * truth, SHELLS, 3, bands=((1, 1),))
    assert np.isnan(no_truth["band_power_ratio"][0])
