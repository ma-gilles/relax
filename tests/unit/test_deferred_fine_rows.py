"""A deferred adaptive fine grid, filled per coarse parent, holds the whole grid's rows."""

import numpy as np
import pytest

from relax.helpers.oversampling import DeferredFineRows
from relax.refinement.half_scoring import _adaptive_pass2_grids
from relax.sampling import get_oversampled_rotation_grid_from_samples, get_relion_rotation_grid

pytestmark = pytest.mark.unit


def _grids(defer):
    coarse = get_relion_rotation_grid(1).astype(np.float32)
    base = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    return _adaptive_pass2_grids(
        coarse, base.astype(np.float32), base, healpix_order=1, adaptive_oversampling=1, translation_step=2.0,
        random_perturbation=0.37, coarse_rotation_ids=None, defer_fine_rotations=defer,
    )


def test_filled_rows_equal_the_whole_grid_rows():
    full, deferred = _grids(False), _grids(True)
    eulers = get_oversampled_rotation_grid_from_samples(
        np.arange(full.coarse_rotations.shape[0]), 1, oversampling_order=1, random_perturbation=0.37,
        return_source_eulers=True,
    )[-1]
    np.testing.assert_array_equal(deferred.rotation_parent_map, full.rotation_parent_map)
    np.testing.assert_array_equal(deferred.fine_translations, full.fine_translations)
    fill = DeferredFineRows(deferred, 1, 1, 0.37)
    parents = np.array([3, 3, 100, 575, 0])
    fill(parents)
    rows = np.flatnonzero(np.isin(full.rotation_parent_map, parents))
    assert rows.size == 4 * 8
    np.testing.assert_array_equal(deferred.fine_rotations[rows], full.fine_rotations[rows])
    np.testing.assert_array_equal(deferred.fine_mstep_rotations[rows], full.fine_mstep_rotations[rows])
    np.testing.assert_array_equal(fill.source_eulers[rows], eulers[rows])
    unfilled = np.setdiff1d(np.arange(full.rotation_parent_map.size), rows)
    assert np.all(np.isnan(fill.source_eulers[unfilled]))
    np.testing.assert_array_equal(deferred.fine_rotations[unfilled], np.broadcast_to(np.eye(3), (unfilled.size, 3, 3)))
    fill(None)  # every parent
    np.testing.assert_array_equal(deferred.fine_rotations, full.fine_rotations)
    np.testing.assert_array_equal(deferred.fine_mstep_rotations, full.fine_mstep_rotations)
    np.testing.assert_array_equal(fill.source_eulers, eulers)
