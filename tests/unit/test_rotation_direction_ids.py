"""Each rotation's HEALPix direction, in both rotation index orders of RELION's grid.

The expectation is read off the grid's own Euler angles: two rotations share a direction exactly when they share
(rot, tilt), and every direction holds one rotation per in-plane angle.
"""

import numpy as np
import pytest

from relax.sampling.grids import (
    get_relion_rotation_grid_eulers_float64,
    rotation_direction_ids,
    rotation_grid_n_in_planes,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("rotation_index_order", ["relion", "recovar"])
@pytest.mark.parametrize("symmetry", ["C1", "D2"])
def test_rotations_share_a_direction_exactly_when_they_share_rot_and_tilt(order, rotation_index_order, symmetry):
    eulers = get_relion_rotation_grid_eulers_float64(order, rotation_index_order=rotation_index_order, symmetry=symmetry)
    ids = rotation_direction_ids(order, rotation_index_order=rotation_index_order, symmetry=symmetry)
    assert ids.shape == (eulers.shape[0],) and ids.dtype == np.int64
    _, by_angles = np.unique(eulers[:, :2], axis=0, return_inverse=True)
    pairs = np.unique(np.stack([ids, by_angles.reshape(-1)]), axis=1)
    # A bijection between the ids and the distinct (rot, tilt) pairs.
    assert pairs.shape[1] == len(np.unique(ids)) == len(np.unique(by_angles))
    np.testing.assert_array_equal(np.bincount(ids), rotation_grid_n_in_planes(order))


def test_unknown_rotation_index_order_refuses():
    with pytest.raises(ValueError, match="rotation_index_order"):
        rotation_direction_ids(1, rotation_index_order="healpix")
