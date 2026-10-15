"""librelax_host_math.so (relax/host_math) against Python's math module, which calls the same C library functions."""

import math

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.sampling import libm

pytestmark = pytest.mark.unit

_ATAN2 = np.frompyfunc(math.atan2, 2, 1)
_EDGES = np.array(
    [0.0, -0.0, 1.0, -1.0, np.inf, -np.inf, np.nan, 5e-324, -5e-324, 2.2250738585072014e-308, 1e-300, 1e300, 1.7976931348623157e308]
)


def _reference_atan2(y, x):
    return np.asarray(_ATAN2(y, x), dtype=np.float64)


def _reference_acos(values):
    return np.array([math.acos(v) for v in values])


def test_atan2_matches_math_on_random_and_edge_pairs():
    rng = np.random.default_rng(11)
    y = np.concatenate([rng.standard_normal(200_000) * 10.0 ** rng.integers(-300, 300, 200_000), np.repeat(_EDGES, _EDGES.size)])
    x = np.concatenate([rng.standard_normal(200_000) * 10.0 ** rng.integers(-300, 300, 200_000), np.tile(_EDGES, _EDGES.size)])
    got, expected = libm.atan2(y, x), _reference_atan2(y, x)
    assert got.shape == y.shape and got.dtype == np.float64
    assert_matches(got, expected)
    # Signed zeros and NaNs are discrete outcomes: they must agree exactly.
    assert np.array_equal(np.signbit(got), np.signbit(expected)) and np.array_equal(np.isnan(got), np.isnan(expected))


def test_acos_matches_math_and_keeps_its_domain_error():
    rng = np.random.default_rng(12)
    values = np.concatenate([rng.uniform(-1.0, 1.0, 200_000), [1.0, -1.0, 0.0, -0.0, np.nan, np.nextafter(1.0, 0.0)]])
    got, expected = libm.acos(values), _reference_acos(values)
    assert_matches(got, expected)
    assert np.array_equal(np.isnan(got), np.isnan(expected))
    for outside in (np.nextafter(1.0, 2.0), np.nextafter(-1.0, -2.0), np.inf):
        with pytest.raises(ValueError):
            math.acos(outside)
        with pytest.raises(ValueError):
            libm.acos(np.array([0.5, outside]))


def test_shapes_broadcast_and_scalars_are_kept():
    assert libm.atan2(np.float64(1.0), -1.0).shape == ()
    assert libm.acos(0.5).shape == ()
    grid = libm.atan2(np.ones((3, 1)), np.arange(4.0)[None, :])
    assert grid.shape == (3, 4)
    assert_matches(grid, _reference_atan2(np.ones((3, 4)), np.broadcast_to(np.arange(4.0), (3, 4))))
