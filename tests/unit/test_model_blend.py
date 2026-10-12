"""RELION's blend of the model's other parameters (relax.steps.model_blend).

Expectations are written out from RELION's statements (``x *= mu; x += (1. - mu) * wsum / weight``,
maximizationOtherParameters) with values whose results are exact in binary floating point, plus the grouping
check: ``((1 - mu) * wsum) / weight``, which differs from ``(1 - mu) * (wsum / weight)`` in the last bit.
"""

import numpy as np
import pytest

from relax.steps.model_blend import MIN_SIGMA2_OFFSET_ANGSTROM2, blend_offset_variance, relion_blend

pytestmark = pytest.mark.unit


def test_relion_blend_moves_previous_towards_the_measured_value():
    # mu 0.5: half of 4, plus half of 12 / 2.
    assert relion_blend(4.0, 12.0, 2.0, 0.5) == 5.0
    # mu 0: the measured value alone; mu 1: the previous value alone.
    np.testing.assert_array_equal(relion_blend(np.asarray([1.0, 2.0]), np.asarray([6.0, 8.0]), 2.0, 0.0), [3.0, 4.0])
    np.testing.assert_array_equal(relion_blend(np.asarray([1.0, 2.0]), np.asarray([6.0, 8.0]), 2.0, 1.0), [1.0, 2.0])


def test_relion_blend_groups_as_relion():
    previous, wsum, weight, mu = 0.3, 0.3, 7.0, 0.9
    relion = previous * mu
    relion += (1.0 - mu) * wsum / weight
    assert relion_blend(previous, wsum, weight, mu) == relion
    # Values where the other grouping rounds differently, so the test sees the grouping.
    assert (1.0 - mu) * (wsum / weight) != (1.0 - mu) * wsum / weight


def test_relion_blend_keeps_float32_operands_in_float32():
    out = relion_blend(np.ones(3, np.float32), np.full(3, 2.0, np.float32), np.float32(1.0), np.float32(0.9))
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, np.float32(0.9) + (np.float32(1) - np.float32(0.9)) * np.float32(2.0))


def test_offset_variance_is_the_per_dimension_mean_square_blended_and_floored():
    # 2D: 40 / (2 * 4) = 5, blended halfway with 3.
    assert blend_offset_variance(3.0, 40.0, 4.0, 2, 0.5, MIN_SIGMA2_OFFSET_ANGSTROM2) == 4.0
    # 3D subtomogram offsets divide by 3.
    assert blend_offset_variance(3.0, 60.0, 4.0, 3, 0.0, MIN_SIGMA2_OFFSET_ANGSTROM2) == 5.0
    # Below RELION's floor of 2 Angstrom squared.
    assert blend_offset_variance(0.5, 0.0, 4.0, 2, 0.5, MIN_SIGMA2_OFFSET_ANGSTROM2) == 2.0
    # In pixels squared the caller converts the floor: 2 A^2 at 2 A per pixel is 0.5 px^2.
    assert blend_offset_variance(0.25, 0.0, 1.0, 2, 0.0, MIN_SIGMA2_OFFSET_ANGSTROM2 / 2.0**2) == 0.5
    assert isinstance(blend_offset_variance(3.0, np.float64(40.0), 4.0, 2, 0.5, 2.0), float)
