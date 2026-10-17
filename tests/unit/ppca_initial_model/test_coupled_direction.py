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


def _definite_blocks(count, rng):
    """``count`` packed 3x3 (q = 2) definite blocks and their complex gradients."""
    factor = rng.standard_normal((count, 3, 3)).astype(np.float32)
    matrix = factor @ factor.swapaxes(-1, -2) + np.eye(3, dtype=np.float32)
    tri = np.asarray(pack_upper_tri(jnp.asarray(matrix)))
    gradient = (rng.standard_normal((count, 3)) + 1j * rng.standard_normal((count, 3))).astype(np.complex64)
    return tri, gradient


def test_chunked_eigh_solves_each_block_as_one_batch_does(monkeypatch):
    """Chunking the batch and solving only the support rows change no value of the direction."""
    from relax.ppca_initial_model import update

    tri, gradient = _definite_blocks(50, np.random.default_rng(5))
    support = np.random.default_rng(6).random(50) < 0.7
    whole, whole_info = coupled_direction(tri, gradient, support, floor=FLOOR)
    monkeypatch.setattr(update, "EIGH_BATCH_BLOCKS", 7)
    chunked, chunked_info = coupled_direction(tri, gradient, support, floor=FLOOR)
    np.testing.assert_array_equal(np.asarray(chunked), np.asarray(whole))
    assert chunked_info == whole_info
    assert not np.any(np.asarray(whole)[~support])


@pytest.mark.gpu
def test_a_box_192_half_volume_of_blocks_is_solved_on_the_gpu():
    """192 x 192 x 97 = 3,575,808 blocks: one batched cuSolver syevj call rejects it (jaxlib 0.9.0.1,
    'gpusolverDnSsyevjBatched_bufferSize ... invalid value' above about 2M blocks); the chunked solve gives each
    block the value a small batch gives it."""
    count = 192 * 192 * 97
    tri, gradient = _definite_blocks(count, np.random.default_rng(7))
    direction, _ = coupled_direction(tri, gradient, np.ones(count, bool), floor=FLOOR)
    direction = np.asarray(direction)
    for start in (0, 2_000_000, count - 100):
        part, _ = coupled_direction(
            tri[start : start + 100], gradient[start : start + 100], np.ones(100, bool), floor=FLOOR
        )
        np.testing.assert_array_equal(direction[start : start + 100], np.asarray(part))
