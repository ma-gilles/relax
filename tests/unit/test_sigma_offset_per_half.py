import numpy as np
import pytest

from relax.helpers.types import make_noise_stats
from relax.refinement.half_inputs import SigmaOffset
from relax.refinement.noise_updates import (
    update_class_sigma_offset_from_posterior,
    update_k1_sigma_offset_from_posterior,
)

pytestmark = pytest.mark.unit


def _noise_stats(wsum_sigma2_offset, sumw, n_shells=4):
    return make_noise_stats(
        wsum_sigma2_noise=np.ones(n_shells),
        wsum_img_power=np.ones(n_shells),
        wsum_sigma2_offset=wsum_sigma2_offset,
        sumw=sumw,
    )


def test_non_kclass_sigma_offset_is_independent_per_half():
    result = update_k1_sigma_offset_from_posterior(
        noise_stats_per_half=[_noise_stats(12.0, 2.0), _noise_stats(40.0, 2.0)],
        current_sigma_offset_angstrom_per_half=[10.0, 10.0],
        state_fallback_offsets_angstrom=float("nan"),
    )
    assert result.current_sigma_offset_angstrom_per_half == pytest.approx([np.sqrt(3.0), np.sqrt(10.0)])
    assert result.current_sigma_offset_angstrom == pytest.approx((np.sqrt(3.0) + np.sqrt(10.0)) / 2.0)


def test_kclass_sigma_offset_is_shared_across_halves():
    result = update_class_sigma_offset_from_posterior(
        noise_stats_per_half=[_noise_stats(12.0, 2.0), _noise_stats(40.0, 2.0)],
        noise_stats_per_half_per_class=[None, None],
        current_sigma_offset_angstrom_per_half=[10.0, 10.0],
        n_classes=2,
        state_fallback_offsets_angstrom=float("nan"),
    )
    expected = np.sqrt(52.0 / 8.0)
    assert result.current_sigma_offset_angstrom_per_half == pytest.approx([expected, expected])


def test_subtomogram_kclass_sigma_offset_divides_by_three_dimensions():
    """Class3D of subtomograms: one shared sigma2_offset = wsum / (3 sum_weight) (ml_optimiser.cpp:5222-5227)."""

    result = update_class_sigma_offset_from_posterior(
        noise_stats_per_half=[_noise_stats(60.0, 2.0), None],
        noise_stats_per_half_per_class=[None, None],
        current_sigma_offset_angstrom_per_half=[10.0, 10.0],
        n_classes=2,
        state_fallback_offsets_angstrom=float("nan"),
        offset_dims=3,
    )
    assert result.current_sigma_offset_angstrom_per_half == pytest.approx([np.sqrt(10.0), np.sqrt(10.0)])


@pytest.mark.parametrize("missing_stats", [None, _noise_stats(0.0, 0.0)])
def test_missing_half_uses_hard_assignment_fallback_independently(missing_stats):
    result = update_k1_sigma_offset_from_posterior(
        noise_stats_per_half=[missing_stats, _noise_stats(40.0, 2.0)],
        current_sigma_offset_angstrom_per_half=[10.0, 10.0],
        state_fallback_offsets_angstrom=3.0,
    )
    assert result.current_sigma_offset_angstrom_per_half == pytest.approx([3.0, np.sqrt(10.0)])


def test_sigma_offset_half_pair_normalizes_scalar_and_pair():
    assert SigmaOffset.from_halves(1.5) == SigmaOffset(1.5, (1.5, 1.5))
    assert SigmaOffset.from_halves([1.5, 2.5]) == SigmaOffset(2.0, (1.5, 2.5))
    assert SigmaOffset(3.0).for_half(1) == 3.0
    with pytest.raises(ValueError, match="exactly two"):
        SigmaOffset.from_halves([1.0, 2.0, 3.0])


def test_a_zero_offset_moment_takes_relions_lower_bound():
    # Every particle on its prior centre (offsets all zero): RELION's sigma2_offset is 0, raised to
    # min_sigma2_offset = 2 A^2, not kept at its previous value.
    result = update_k1_sigma_offset_from_posterior(
        noise_stats_per_half=[_noise_stats(0.0, 2.0), _noise_stats(0.0, 3.0)],
        current_sigma_offset_angstrom_per_half=[10.0, 10.0],
        state_fallback_offsets_angstrom=3.0,
    )
    assert result.current_sigma_offset_angstrom_per_half == pytest.approx([np.sqrt(2.0), np.sqrt(2.0)])
