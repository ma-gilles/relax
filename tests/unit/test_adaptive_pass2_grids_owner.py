"""The K=1 and K-class dense routes materialize RELION's two-pass trial grids through one owner."""

from __future__ import annotations

import numpy as np
import pytest
from helpers.float_compare import matches

from relax.refinement import dense_specs
from relax.sampling.grids import rotation_grid_size
from relax.sampling.oversampling import build_adaptive_pass2_grids
from relax.sampling.perturbation import apply_relion_translation_perturbation

pytestmark = pytest.mark.unit

ORDER = 1
N_ROT = rotation_grid_size(ORDER)
BASE = np.asarray([[-1.0, 0.0], [0.0, 0.0], [1.0, 1.0]], dtype=np.float64)


def _same(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    return x.dtype == y.dtype and x.shape == y.shape and matches(x, y)


@pytest.mark.parametrize("oversampling", [0, 1])
def test_owner_returns_the_builder_grids_and_perturbed_phase_source(oversampling):
    rot = np.repeat(np.eye(3, dtype=np.float32)[None], N_ROT, axis=0)
    grids = dense_specs.prepare_adaptive_pass2_grids(
        rot, BASE.astype(np.float32), BASE, healpix_order=ORDER, adaptive_oversampling=oversampling,
        translation_step=1.0, random_perturbation=0.25, coarse_rotation_ids=None,
    )
    expected = build_adaptive_pass2_grids(rot, BASE.astype(np.float32), BASE, ORDER, oversampling, 1.0, 0.25, return_mstep_rotations=True)
    for got, exp in zip(grids[:7], expected):
        assert _same(got, exp)
    assert _same(grids.coarse_translation_phase_source, apply_relion_translation_perturbation(BASE, 0.25, 1.0))
    assert grids.n_fine_translations == int(expected[3].shape[0]) and isinstance(grids.n_fine_translations, int)


def test_zero_oversampling_keeps_the_coarse_rotation_grid():
    rot = np.repeat(np.eye(3, dtype=np.float32)[None], N_ROT, axis=0)
    grids = dense_specs.prepare_adaptive_pass2_grids(
        rot, BASE.astype(np.float32), BASE, healpix_order=ORDER, adaptive_oversampling=0,
        translation_step=1.0, random_perturbation=0.0, coarse_rotation_ids=None,
    )
    assert _same(grids.fine_rotations, rot) and matches(grids.rotation_parent_map, np.arange(N_ROT))


def test_zero_oversampling_backprojects_with_relion_host_inverse_rows():
    """relax#81: without oversampling RELION scores with its device-built rows but backprojects with
    generateEulerMatrices(inverse=true) of the same perturbed orientations (the cofactor inverse, cast to float32).
    The two differ by a float32 unit in part of the entries, which moves the outer-shell radius test of the M-step,
    so the M-step rows are the host-inverse builder's, whatever rows pass 1 and pass 2 score."""
    from relax.sampling.oversampled_grids import get_oversampled_rotation_grid_from_samples

    order, rp = 2, -0.11470896005630493
    exact = get_oversampled_rotation_grid_from_samples(
        np.arange(rotation_grid_size(order)), parent_nside_level=order, oversampling_order=0, random_perturbation=rp,
        return_mstep_rotations=True, dtype=np.float32,
    )[2]
    # Scored rows one float32 unit off in every other entry, as device-built rows can be.
    scoring = exact.copy()
    scoring.reshape(-1)[::2] = np.nextafter(scoring.reshape(-1)[::2], np.float32(2.0))
    grids = build_adaptive_pass2_grids(
        scoring, BASE.astype(np.float32), BASE, order, 0, 1.0, rp, return_mstep_rotations=True,
    )
    assert np.array_equal(grids[0], scoring) and np.array_equal(grids[2], scoring)  # both passes score them
    assert grids[6].dtype == scoring.dtype and np.array_equal(grids[6], exact)
