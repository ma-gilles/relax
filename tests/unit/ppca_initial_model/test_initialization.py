"""Independent empirical covariance of the three selected random seeds."""

import jax.numpy as jnp
import numpy as np
import pytest

from relax.ppca_initial_model.initialization import seed_maps_to_model

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype,tol", [(np.float32, 2e-6), (np.float64, 2e-12)])
def test_seed_covariance(dtype, tol):
    volumes = np.random.default_rng(318).normal(size=(3, 8)).astype(dtype)
    model = np.asarray(seed_maps_to_model(volumes, compute_dtype=dtype))
    centered = volumes.astype(np.float64) - np.mean(volumes.astype(np.float64), axis=0)
    reference = centered.T @ centered / 3
    np.testing.assert_allclose(model[1:].T @ model[1:], reference, atol=tol, rtol=tol)
    np.testing.assert_allclose(model[0], volumes.astype(np.float64).mean(axis=0), atol=tol, rtol=tol)
    assert model.dtype == dtype


def test_initialization_rejects_wrong_seed_count():
    with pytest.raises(ValueError, match="exactly three"):
        seed_maps_to_model(jnp.zeros((2, 8)))
