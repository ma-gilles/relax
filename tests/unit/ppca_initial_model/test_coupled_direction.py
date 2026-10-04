"""The coupled VDAM-PPCA direction is solved and checked on the update's support only (algorithm section 17)."""

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches
from recovar.ppca.triangular import pack_upper_tri

from relax.ppca_initial_model.update import coupled_direction, metric_floor

pytestmark = pytest.mark.unit

FLOOR = metric_floor(128)
# Row 555307 (radius 33.30, stage radius 32) of the EMPIAR-10499 pilot's failure_metric_0165_half0.npz
# (em_work/relax_ppca_cryoet_20261002/empiar10499/pilot_14944779/run, relax e6a0521): interpolation spill, near rank
# one at the floor 1.86e-9, smallest eigenvalue -1.6e-14 against the check's bound 7.9e-15.
SPILL_TRI = np.asarray(
    [
        3.16836249e-12, -3.22785771e-13, 6.81305637e-11, 8.80160788e-12, -3.89515330e-11,
        4.35932350e-13, -7.26576872e-12, -9.27959315e-13, 4.24822287e-12, 1.54018309e-09,
        1.95660502e-10, -8.76591022e-10, 2.52495854e-11, -1.11552906e-10, 4.99105590e-10,
    ],
    np.float32,
)  # fmt: skip
SPILL_GRADIENT = np.asarray(
    [
        1.2578175e-09 - 7.8880440e-09j, -1.3341075e-10 + 8.5338375e-10j, 2.7881761e-08 - 1.7873415e-07j,
        3.5654599e-09 - 2.2685384e-08j, -1.5893388e-08 + 1.0169820e-07j,
    ],
    np.complex64,
)  # fmt: skip


def _definite_row(rng):
    factor = rng.standard_normal((5, 5))
    matrix = (factor @ factor.T + np.eye(5)).astype(np.float32) * np.float32(1e-3)
    gradient = (rng.standard_normal(5) + 1j * rng.standard_normal(5)).astype(np.complex64)
    return matrix, gradient


def _rows(rng):
    matrix, gradient = _definite_row(rng)
    tri = np.stack([np.asarray(pack_upper_tri(jnp.asarray(matrix))), SPILL_TRI])
    return matrix, gradient, tri, np.stack([gradient, SPILL_GRADIENT])


def test_spill_row_outside_the_support_does_not_stop_the_update():
    matrix, gradient, tri, gradients = _rows(np.random.default_rng(0))
    direction, info = coupled_direction(tri, gradients, np.asarray([True, False]), floor=FLOOR)
    direction = np.asarray(direction)
    # float32 eigh solve of a matrix with condition number below 1e3 against the float64 solve.
    expected = np.linalg.solve(matrix.astype(np.float64), gradient.astype(np.complex128))
    assert_matches(direction[0], expected, rtol=1e-3)
    assert np.count_nonzero(direction[1]) == 0
    assert info["minimum_eigenvalue"] > 0 and info["floored_eigenvalues"] == 0


def test_indefinite_row_inside_the_support_stops_the_update():
    _, _, tri, gradients = _rows(np.random.default_rng(0))
    with pytest.raises(ValueError, match="materially indefinite PPCA metric"):
        coupled_direction(tri, gradients, np.asarray([True, True]), floor=FLOOR)


def test_nonfinite_statistics_outside_the_support_are_ignored_and_inside_stop_the_update():
    _, _, tri, gradients = _rows(np.random.default_rng(0))
    tri[1, 0] = np.nan
    coupled_direction(tri, gradients, np.asarray([True, False]), floor=FLOOR)
    with pytest.raises(ValueError, match="Nonfinite or materially indefinite PPCA metric"):
        coupled_direction(tri, gradients, np.asarray([True, True]), floor=FLOOR)
