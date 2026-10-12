"""RELION's soft mask weights (relax.relion.geometry.soft_mask_weights) against a transcription of mask.cpp.

The reference walks the grid as ``softMaskOutsideMap`` does (Xmipp origin, ``r = sqrt(k*k + i*i + j*j)``, the three
branches) one voxel at a time with ``math.cos``.
"""

import itertools
import math

import numpy as np
import pytest

from relax.relion.geometry import soft_mask_weights

pytestmark = pytest.mark.unit


def _relion_weights(shape, radius, width):
    radius_p = radius + width
    weights = np.empty(shape)
    for index in itertools.product(*(range(n) for n in shape)):
        logical = [i - int(n / 2.0) for i, n in zip(index, shape)]
        r = math.sqrt(sum(x * x for x in logical))
        if r < radius:
            weights[index] = 0.0
        elif r > radius_p:
            weights[index] = 1.0
        else:
            weights[index] = 0.5 + 0.5 * math.cos(math.pi * (radius_p - r) / width)
    return weights


@pytest.mark.parametrize("shape", [(9, 9), (10, 10), (7, 7, 7), (8, 8, 8)])
@pytest.mark.parametrize("radius, width", [(2.0, 1.0), (2.5, 1.7), (3.0, 3.0), (0.4, 2.0)])
def test_weights_are_relions_three_branches(shape, radius, width):
    np.testing.assert_allclose(
        soft_mask_weights(shape, radius, width), _relion_weights(shape, radius, width), rtol=0, atol=1e-15
    )


def test_inside_edge_and_outside_values():
    w = soft_mask_weights((11, 11), 2.0, 2.0)
    centre = 5
    assert w[centre, centre] == 0.0  # r = 0
    assert w[centre, centre + 2] == 0.0  # r = radius: the taper starts at 0
    assert w[centre, centre + 3] == pytest.approx(0.5)  # half way through the edge
    assert w[centre, centre + 4] == 1.0  # r = radius + width
    assert w[0, 0] == 1.0


def test_zero_width_is_nan_on_the_radius_as_in_relion():
    w = soft_mask_weights((9, 9), 3.0, 0.0)
    assert np.isnan(w[4, 7]) and np.isnan(w[4, 1])  # r == 3
    finite = ~np.isnan(w)
    np.testing.assert_array_equal(np.unique(w[finite]), [0.0, 1.0])
