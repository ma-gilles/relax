"""Orthogonal equivariance and independent scalar update checks."""

import jax.numpy as jnp
import numpy as np
import pytest
from recovar.ppca.triangular import pack_upper_tri

from relax.ppca_initial_model.update import coupled_direction, empty_moments, metric_floor, stochastic_update

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype,tol", [(np.float32, 4e-5), (np.float64, 4e-11)])
def test_latent_rotation_equivariance(dtype, tol):
    rng = np.random.default_rng(81)
    complex_dtype = np.complex64 if dtype == np.float32 else np.complex128
    theta = (rng.normal(size=(9, 3)) + 1j * rng.normal(size=(9, 3))).astype(complex_dtype)
    matrices = rng.normal(size=(2, 9, 3, 3)).astype(dtype)
    matrices = matrices @ matrices.swapaxes(-1, -2) + np.eye(3, dtype=dtype) * 0.2
    residual = (rng.normal(size=(2, 9, 3)) + 1j * rng.normal(size=(2, 9, 3))).astype(complex_dtype)
    Q = np.eye(3, dtype=dtype)
    Q[1:, 1:] = np.array([[0.6, -0.8], [0.8, 0.6]], dtype=dtype)
    dirs = jnp.stack([coupled_direction(pack_upper_tri(a), r, floor=0.4)[0] for a, r in zip(matrices, residual)])
    transformed = Q.T @ matrices @ Q
    dirs_q = jnp.stack(
        [coupled_direction(pack_upper_tri(a), r, floor=0.4)[0] for a, r in zip(transformed, residual @ Q)]
    )
    np.testing.assert_allclose(dirs_q, dirs @ Q, atol=tol, rtol=tol)
    shells = np.arange(9) // 3
    args = dict(coverage=np.ones((2, 9), bool), shells=shells, step=0.8, fudge=2, image_size=8)
    out, moments, info = stochastic_update(jnp.asarray(theta), empty_moments(jnp.asarray(theta)), dirs, **args)
    out_q, moments_q, info_q = stochastic_update(
        jnp.asarray(theta @ Q), empty_moments(jnp.asarray(theta @ Q)), dirs_q, **args
    )
    np.testing.assert_allclose(out_q, out @ Q, atol=tol, rtol=tol)
    np.testing.assert_allclose(moments_q.second, moments.second, atol=tol, rtol=tol)
    np.testing.assert_allclose(info_q["gates"], info["gates"], atol=tol, rtol=tol)


def test_zero_disagreement_and_new_support():
    theta = jnp.ones((2, 3), jnp.complex64)
    moments = empty_moments(theta)
    out, next_moments, info = stochastic_update(
        theta,
        moments,
        jnp.ones((2, 2, 3), jnp.complex64),
        jnp.array([[True, False], [True, False]]),
        np.array([0, 1]),
        step=0.5,
        fudge=1,
        image_size=8,
    )
    np.testing.assert_array_equal(out, [[0.5] * 3, [1] * 3])
    np.testing.assert_array_equal(next_moments.initialized, [[True, False], [True, False]])
    assert info["zero_disagreement_shell_channels"] == 2


def test_indefinite_metric_rejected():
    with pytest.raises(ValueError, match="indefinite"):
        coupled_direction(jnp.array([[-1.0, 0.0, 1.0]], jnp.float32), jnp.ones((1, 2), jnp.complex64), floor=1.0)


def test_metric_units():
    # RELION= N^4 * engine; coefficient noise is 2*component noise.
    assert metric_floor(8) * 2 * 8**4 == 1


def test_scalar_vdam_reference_in_normalized_units():
    # Diagnostic Python complex128 transcription of reconstructGrad's scalar
    # recurrence, before spatial postprocessing. Production stays complex64.
    n = 8
    scale = n**2
    value = 0.7 + 0.3j
    first = [0j, 0j]
    second = 1.0
    theta = jnp.asarray([[scale * value]], jnp.complex64)
    moments = empty_moments(theta)
    for iteration, directions in enumerate([(0.3 + 0.2j, -0.1 + 0.4j), (0.1 - 0.2j, 0.5 + 0.1j)]):
        for half in range(2):
            first[half] = directions[half] if iteration == 0 else 0.9 * first[half] + 0.1 * directions[half]
        ratio = abs(directions[1] - directions[0]) ** 2 / (abs(sum(directions) / 2) ** 2 + 1e-12)
        second = 0.999 * second + 0.001 * ratio
        gradient = sum(first) / 2 / (np.sqrt(second) + 1e-12)
        rho = 2 * 1.7 * abs(value) / abs(first[0] - first[1])
        gate = rho / (1 + rho)
        value += 0.8 * (gate * gradient - (1 - gate) * value)
        theta, moments, _ = stochastic_update(
            theta,
            moments,
            jnp.asarray(directions, jnp.complex64).reshape(2, 1, 1) * scale,
            np.ones((2, 1), bool),
            np.zeros(1, np.int32),
            step=0.8,
            fudge=1.7,
            image_size=n,
        )
        np.testing.assert_allclose(theta[0, 0] / scale, value, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(moments.second[0, 0], second, rtol=2e-6, atol=2e-6)
