"""Iteration-boundary checks for the fixed real-fixture trajectory adapter."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from scripts.benchmark_dense_gemm_trajectory import _checked_particle_rows, _failure_diagnostics


def _result(*, masses=(0.73, 1.0), numerator=(1 + 2j, 3 - 1j)):
    pair = np.full((5, 2), -np.inf, dtype=np.float32)
    pair[3] = (-1280.0, np.log(4.0))
    pair[1] = (-1260.0, np.log(3.0))
    mass = np.zeros(5, dtype=np.float32)
    mass[[3, 1]] = masses
    return SimpleNamespace(
        next_pair_table=jnp.asarray(pair),
        mass_table=jnp.asarray(mass),
        invalid_normalizer_table=jnp.zeros(5, dtype=jnp.bool_),
        invalid_weight_table=jnp.zeros(5, dtype=jnp.bool_),
        numerator=jnp.asarray(numerator, dtype=jnp.complex64),
        denominator=jnp.asarray((2.0, 1.0), dtype=jnp.float32),
    )


def test_iteration_boundary_preserves_lagged_mass_by_stable_id():
    report = _checked_particle_rows(_result(), (3, 1))
    assert report["particles"] == 2
    np.testing.assert_allclose(report["mass_min_max"], (0.73, 1.0), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(report["mass_mean"], 0.865, rtol=1e-6, atol=1e-6)
    assert report["logz_min_max"][0] < report["logz_min_max"][1]


@pytest.mark.parametrize("masses", [(0.0, 1.0), (np.nan, 1.0), (np.inf, 1.0)])
def test_iteration_boundary_rejects_invalid_weight_mass(masses):
    with pytest.raises(RuntimeError, match="invalid normalizer or posterior mass"):
        _checked_particle_rows(_result(masses=masses), (3, 1))


def test_iteration_boundary_rejects_nonfinite_volume():
    with pytest.raises(RuntimeError, match="nonfinite BPref accumulators"):
        _checked_particle_rows(_result(numerator=(complex(np.nan, 0), 1 + 0j)), (3, 1))


def test_failure_census_keeps_stable_ids_and_finite_ranges():
    result = _result(masses=(np.inf, 0.0), numerator=(complex(np.nan, 0), 1 + 0j))
    report = _failure_diagnostics(result, (3, 1))
    assert report["nonfinite_mass_ids"] == [3]
    assert report["nonpositive_mass_ids"] == [1]
    assert report["finite_mass_min_max"] == [0.0, 0.0]
    assert report["nonfinite_numerator_voxels"] == 1
