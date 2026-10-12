"""relax's HEALPix sampling port against RELION's ``HealpixSampling`` (the binding is the oracle).

Pixel ids, counts and ordering must agree exactly; angles agree to libm rounding
(measured differences are zero on della; the tolerance allows one ulp of a
different libm at the angle scale).
"""

from __future__ import annotations

import numpy as np
import pytest

from relax.sampling import healpix as hs

bind = pytest.importorskip("relax.relion_bind._relion_bind_core")

ANGLE_ATOL = 1e-11
SYMMETRIES = ["C1", "C2", "C3", "C4", "C7", "D1", "D2", "D3", "D7", "T", "O", "I1", "I2", "I3", "I4"]


def _angles_close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=ANGLE_ATOL)


@pytest.mark.parametrize("order", [0, 1, 2, 3, 4])
def test_c1_directions_match(order):
    expected = np.asarray(bind.get_healpix_directions(order), dtype=np.float64)
    sampling = hs.healpix_sampling(order, "C1")
    np.testing.assert_array_equal(sampling["directions_ipix"], np.arange(expected.shape[0]))
    _angles_close(np.stack([sampling["rot"], sampling["tilt"]], axis=-1), expected)


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("symmetry", SYMMETRIES)
def test_symmetric_sampling_metadata_match(order, symmetry):
    expected = bind.get_healpix_sampling_metadata(order, -1.0, symmetry)
    sampling = hs.healpix_sampling(order, symmetry)
    np.testing.assert_array_equal(sampling["directions_ipix"], np.asarray(expected["directions_ipix"]))
    _angles_close(sampling["rot"], np.asarray(expected["rot"]))
    _angles_close(sampling["tilt"], np.asarray(expected["tilt"]))
    _angles_close(sampling["psi"], np.asarray(expected["psi"]))


@pytest.mark.parametrize("order", [0, 2, 3])
@pytest.mark.parametrize("symmetry", ["C1", "C4", "D2", "O", "I2"])
def test_coarse_orientations_match(order, symmetry):
    expected = np.asarray(bind.get_coarse_orientations(order, -1.0, symmetry), dtype=np.float64)
    actual = hs.coarse_orientations(order, symmetry)
    assert actual.shape == expected.shape
    _angles_close(actual, expected)


@pytest.mark.parametrize("symmetry", ["C1", "C3", "D2", "T", "I4"])
@pytest.mark.parametrize("oversampling_order", [0, 1, 2])
@pytest.mark.parametrize("perturbation", [0.0, 0.37, -0.21])
def test_oversampled_orientations_match(symmetry, oversampling_order, perturbation):
    order = 2
    sampling = hs.healpix_sampling(order, symmetry)
    rng = np.random.default_rng(7)
    n = 40
    idirs = rng.integers(0, sampling["rot"].size, size=n)
    ipsis = rng.integers(0, sampling["psi"].size, size=n)
    expected = np.asarray(
        bind.get_oversampled_orientations_batch(order, oversampling_order, idirs, ipsis, perturbation, symmetry),
        dtype=np.float64,
    )
    actual = hs.oversampled_orientations(order, oversampling_order, idirs, ipsis, perturbation, symmetry)
    assert actual.shape == expected.shape
    _angles_close(actual, expected)


def test_euler_round_trip_matches_relion():
    rng = np.random.default_rng(3)
    eulers = np.column_stack([rng.uniform(-180, 180, 500), rng.uniform(0, 180, 500), rng.uniform(-180, 180, 500)])
    eulers[:5, 1] = [0.0, 180.0, 1e-9, 90.0, 179.9999999]
    matrices = hs.euler_angles_to_matrix(eulers)
    for row, matrix in zip(eulers, matrices, strict=True):
        np.testing.assert_allclose(matrix, np.asarray(bind.euler_angles_to_matrix(*row)), rtol=0.0, atol=1e-15)
    angles = hs.euler_matrix_to_angles(matrices)
    for matrix, row in zip(matrices, angles, strict=True):
        _angles_close(row, np.asarray(bind.matrix_to_euler_angles(matrix), dtype=np.float64))
