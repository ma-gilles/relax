"""Independent real Gaussian/FFT references; float64 cases are diagnostic."""

import jax.numpy as jnp
import numpy as np
import pytest
from recovar.core import fourier_transform_utils as ftu
from recovar.ppca.triangular import unpack_tri_to_full

from relax.helpers.half_spectrum import make_half_image_weights
from relax.ppca_initial_model.noise import expected_residual_power, relion_to_coefficient_variance
from relax.ppca_refinement.engine import dense_pose_ppca_score_with_moments_blocked
from relax.ppca_refinement.residual_statistics import full_float32

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype,tol", [(np.float32, 2e-5), (np.float64, 2e-11)])
@pytest.mark.parametrize("q", [0, 2])
@pytest.mark.parametrize("zero_loading", [False, True])
def test_real_covariance_reference(dtype, tol, q, zero_loading):
    rng = np.random.default_rng(1729)
    n = 4
    y = rng.normal(size=(n, n)).astype(dtype)
    mu = rng.normal(size=(n, n)).astype(dtype)
    W = (0.2 * rng.normal(size=(q, n, n))).astype(dtype)
    if zero_loading:
        W *= 0
    variance = dtype(0.7)
    yft = ftu.get_dft2_real(y).reshape(-1)
    projections = ftu.get_dft2_real(np.concatenate([mu[None], W])).reshape(1, q + 1, -1)
    weights = make_half_image_weights((n, n)).astype(dtype)
    nv = n * n * variance
    # The production E-step runs this engine under full float32 (no TF32 on GPU).
    result = full_float32(dense_pose_ppca_score_with_moments_blocked)(
        (yft * weights / nv)[None, None],
        projections,
        (weights / nv)[None],
        jnp.array([jnp.sum(weights * jnp.abs(yft) ** 2) / nv]),
    )
    w = W.reshape(q, n * n).T.astype(np.float64)
    r = (y - mu).ravel().astype(np.float64)
    covariance = variance * np.eye(n * n) + w @ w.T
    reference = -0.5 * (
        r @ np.linalg.solve(covariance, r) + np.linalg.slogdet(covariance)[1] - n * n * np.log(variance)
    )
    latent_cov = np.linalg.inv(np.eye(q) + w.T @ w / variance)
    latent_mean = latent_cov @ w.T @ r / variance
    # Scores exclude the pose-invariant image constant; the absolute score adds it back.
    np.testing.assert_allclose(result.score[0, 0, 0] + result.score_offset[0], reference, atol=tol, rtol=tol)
    np.testing.assert_allclose(result.alpha[0, 0, 0, 1:], latent_mean, atol=tol, rtol=tol)
    g = unpack_tri_to_full(result.G_tri, q + 1)
    np.testing.assert_allclose(g[0, 0, 0, 1:, 1:], latent_cov + np.outer(latent_mean, latent_mean), atol=tol, rtol=tol)
    np.testing.assert_allclose(
        jnp.sum(weights * jnp.abs(yft) ** 2) / nv, np.sum(y.astype(np.float64) ** 2) / variance, atol=tol, rtol=tol
    )


@pytest.mark.parametrize("dtype,tol", [(np.float32, 2e-5), (np.float64, 2e-11)])
def test_residual_gaussian_moment_reference(dtype, tol):
    rng = np.random.default_rng(31)
    y, mu = rng.normal(size=(2, 7)).astype(dtype)
    w = rng.normal(size=(7, 2)).astype(dtype)
    mean = np.array([0.2, -0.3], dtype=dtype)
    cov = np.array([[0.8, 0.2], [0.2, 0.5]], dtype=dtype)
    # Symmetric sigma points integrate any quadratic exactly.
    root = np.linalg.cholesky(cov.astype(np.float64)) * np.sqrt(2)
    points = np.concatenate([mean[:, None] + root, mean[:, None] - root], axis=1)
    reference = np.mean((y[:, None] - mu[:, None] - w @ points) ** 2, axis=1)
    np.testing.assert_allclose(expected_residual_power(y, mu, w, mean, cov), reference, atol=tol, rtol=tol)


def test_relion_normalization_conversion():
    # Initial-noise owner uses half the power of FFT/N², including DC.
    variance = np.float32(0.125)
    n = 8
    np.testing.assert_array_equal(relion_to_coefficient_variance(variance / (2 * n * n), (n, n)), variance * n * n)


@pytest.mark.parametrize("dtype,tol", [(np.float32, 3e-5), (np.float64, 3e-11)])
def test_direct_residual_gradient_fisher_identity(dtype, tol):
    from relax.ppca_refinement.residual_statistics import residual_image_statistics

    rng = np.random.default_rng(91)
    y = rng.normal(size=7).astype(dtype)
    theta = rng.normal(size=(3, 7)).astype(dtype) * dtype(0.2)
    variance = dtype(0.8)
    result = dense_pose_ppca_score_with_moments_blocked(
        jnp.asarray(y[None, None] / variance),
        jnp.asarray(theta[None]),
        jnp.full((1, 7), 1 / variance, dtype=dtype),
        jnp.array([np.sum(y * y) / variance]),
    )
    gradient, correction, embedding = residual_image_statistics(
        result.score,
        result.alpha,
        result.G_tri,
        result.logZ,
        jnp.asarray(y[None, None] / variance),
        jnp.full((1, 7), 1 / variance, dtype=dtype),
        jnp.asarray(theta[None]),
    )

    def objective(value):
        mu, w = value[0], value[1:].T
        cov = variance * np.eye(7) + w @ w.T
        residual = y - mu
        return -0.5 * (residual @ np.linalg.solve(cov, residual) + np.linalg.slogdet(cov)[1])

    reference = np.empty(theta.shape)
    step = 1e-3
    for index in np.ndindex(theta.shape):
        plus, minus = theta.astype(np.float64).copy(), theta.astype(np.float64).copy()
        plus[index] += step
        minus[index] -= step
        far_plus, far_minus = plus.copy(), minus.copy()
        far_plus[index] += step
        far_minus[index] -= step
        reference[index] = (
            -objective(far_plus) + 8 * objective(plus) - 8 * objective(minus) + objective(far_minus)
        ) / (12 * step)
    np.testing.assert_allclose(gradient[:, 0], reference, atol=tol, rtol=tol)
    w = theta[1:].T.astype(np.float64)
    cov = np.linalg.inv(np.eye(2) + w.T @ w / variance)
    m = cov @ w.T @ (y - theta[0]) / variance
    expected = (y - theta[0] - w @ m) ** 2 + np.diag(w @ cov @ w.T)
    np.testing.assert_allclose(y * y + variance * correction, expected, atol=tol, rtol=tol)
    np.testing.assert_allclose(embedding[0], m, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype,tol", [(np.float32, 2e-5), (np.float64, 2e-8)])
def test_off_grid_adjoint_reference(dtype, tol):
    """Double companion for the dataset directional derivative: same packed adjoint."""
    from recovar import core
    from scipy.spatial.transform import Rotation

    from relax.ppca_refinement.residual_statistics import residual_image_statistics

    rng = np.random.default_rng(112)
    n = 6
    shape = (n, n, n)
    theta = ftu.get_dft3_real(rng.normal(size=(3, n, n, n)).astype(dtype) * dtype(0.03)).reshape(3, -1)
    perturb = ftu.get_dft3_real(rng.normal(size=(3, n, n, n)).astype(dtype) * dtype(0.03)).reshape(3, -1)
    rotations = Rotation.from_euler("xyz", [[14, 23, 7], [48, -19, 31]], degrees=True).as_matrix().astype(np.float32)

    def project(value):
        return core.batch_slice_volume(
            value,
            rotations,
            (n, n),
            shape,
            "linear_interp",
            half_volume=True,
            half_image=True,
            relion_texture_interp=False,
            max_r=2,
        ).swapaxes(0, 1)

    projections = project(theta)
    perturbations = project(perturb)
    y = ftu.get_dft2_real(rng.normal(size=(1, n, n)).astype(dtype)).reshape(1, 1, -1)
    weights = make_half_image_weights((n, n)).astype(dtype)
    nv = dtype(n * n)
    result = dense_pose_ppca_score_with_moments_blocked(
        y * weights / nv, projections, (weights / nv)[None], jnp.sum(jnp.abs(y) ** 2 * weights / nv, axis=(1, 2))
    )
    residual, _, _ = residual_image_statistics(
        result.score, result.alpha, result.G_tri, result.logZ, y * weights / nv, (weights / nv)[None], projections
    )
    gradient = core.batch_adjoint_slice_volume(
        residual / weights[None, None],
        rotations,
        (n, n),
        shape,
        "linear_interp",
        half_image=True,
        half_volume=True,
        max_r=2,
    )
    actual = jnp.vdot(gradient, perturb).real
    reference = jnp.vdot(residual, perturbations.swapaxes(0, 1)).real
    np.testing.assert_allclose(actual, reference, atol=tol, rtol=tol)
