"""The contrast latent (model A): the tied column, the chain-rule fold, the contrast estimate and the sd update.

The references are independent: the marginal likelihood of ``y = (1 + delta) mu + W z + noise`` written as one Gaussian
with an explicit covariance, finite differences of it, and explicit ``T^T L T``.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from relax.ppca_refinement import contrast
from relax.ppca_refinement.engine import dense_pose_ppca_score_with_moments_blocked
from relax.ppca_refinement.residual_statistics import full_float32, residual_image_statistics

pytestmark = pytest.mark.unit


def _engine(y, theta, variance):
    """The per-pose engine on a real toy problem: ``y`` ``(B, n)``, ``theta`` ``(P, n)`` columns, one pose."""
    count, n = y.shape
    scaled = jnp.asarray(y[:, None] / variance)
    weights = jnp.full((count, n), 1 / variance, dtype=y.dtype)
    norms = jnp.asarray(np.sum(y * y, axis=1) / variance)
    return full_float32(dense_pose_ppca_score_with_moments_blocked)(scaled, jnp.asarray(theta[None]), weights, norms)


def _model_a_log_likelihood(y, mu, w, sd, variance):
    """log N(y; mu, sd^2 mu mu^T + W W^T + variance I) up to the engine's pose-invariant constant, which it
    reports as ``score_offset``: the engine's absolute score is this, less ``-n log(variance) / 2`` of pure noise."""
    covariance = variance * np.eye(len(y)) + sd**2 * np.outer(mu, mu) + w @ w.T
    residual = y - mu
    return -0.5 * (
        residual @ np.linalg.solve(covariance, residual) + np.linalg.slogdet(covariance)[1] - len(y) * np.log(variance)
    )


def test_tied_column_is_the_mean_scaled():
    mu = jnp.arange(5.0) + 1j * jnp.arange(5.0)
    loadings = jnp.ones((5, 2), jnp.complex64)
    volumes = contrast.with_contrast_column(mu, loadings, 0.3)
    assert volumes.shape == (5, 4)
    np.testing.assert_allclose(volumes[:, 0], mu)
    np.testing.assert_allclose(volumes[:, 1:3], loadings)
    np.testing.assert_allclose(volumes[:, 3], 0.3 * mu)
    assert contrast.with_contrast_column(mu, jnp.zeros((5, 0), jnp.complex64), 0.3).shape == (5, 2)


@pytest.mark.parametrize("q", [0, 1, 3])
def test_engine_on_the_tied_column_is_model_a(q):
    rng = np.random.default_rng(7 + q)
    n, sd, variance = 9, 0.35, 0.8
    mu, w = rng.normal(size=n), 0.4 * rng.normal(size=(n, q))
    y = rng.normal(size=(1, n))
    theta = np.concatenate([mu[None], w.T, sd * mu[None]])
    result = _engine(y, theta, variance)
    reference = _model_a_log_likelihood(y[0], mu, w, sd, variance)
    np.testing.assert_allclose(result.score[0, 0, 0] + result.score_offset[0], reference, rtol=1e-4, atol=1e-4)
    loadings = np.concatenate([w, sd * mu[:, None]], axis=1)
    cov = np.linalg.inv(np.eye(q + 1) + loadings.T @ loadings / variance)
    mean = cov @ loadings.T @ (y[0] - mu) / variance
    np.testing.assert_allclose(result.alpha[0, 0, 0, 1:], mean, rtol=1e-4, atol=1e-5)
    embeddings, estimate = contrast.split_embeddings(np.asarray(result.alpha[:, 0, 0, 1:]), sd)
    np.testing.assert_allclose(embeddings[0], mean[:-1], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(estimate[0], 1 + sd * mean[-1], rtol=1e-4, atol=1e-5)


def test_folded_gradient_is_the_gradient_of_the_model_a_likelihood():
    rng = np.random.default_rng(91)
    n, sd, variance, q = 7, 0.4, 0.8, 2
    y = rng.normal(size=n)
    theta = 0.3 * rng.normal(size=(q + 1, n))
    tied = np.concatenate([theta, sd * theta[:1]])
    weights = jnp.full((1, n), 1 / variance)
    result = _engine(y[None], tied, variance)
    gradient, _, _ = residual_image_statistics(
        result.score, result.alpha, result.G_tri, result.logZ, jnp.asarray(y[None, None] / variance), weights,
        jnp.asarray(tied[None]),
    )
    folded = np.asarray(contrast.fold_gradient(jnp.asarray(gradient[:, 0]).T, sd)).T  # (q + 1, n)

    def objective(value):
        return _model_a_log_likelihood(y, value[0], value[1:].T, sd, variance)

    reference = np.empty(theta.shape)
    step = 1e-3
    for index in np.ndindex(theta.shape):
        plus, minus = theta.copy(), theta.copy()
        far_plus, far_minus = theta.copy(), theta.copy()
        plus[index] += step
        minus[index] -= step
        far_plus[index] += 2 * step
        far_minus[index] -= 2 * step
        reference[index] = (-objective(far_plus) + 8 * objective(plus) - 8 * objective(minus) + objective(far_minus)) / (
            12 * step
        )
    np.testing.assert_allclose(folded, reference, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("q", [0, 2, 5])
def test_fold_metric_is_t_transpose_l_t(q):
    rng = np.random.default_rng(3 + q)
    size, sd = q + 2, 0.45
    raw = rng.normal(size=(6, size, size))
    full = raw @ raw.transpose(0, 2, 1)  # symmetric positive semidefinite per frequency
    packed = full[:, np.triu_indices(size)[0], np.triu_indices(size)[1]]
    transform = np.zeros((size, size - 1))
    transform[: size - 1] = np.eye(size - 1)
    transform[size - 1, 0] = sd
    expected = transform.T @ full @ transform
    folded = np.asarray(contrast.fold_metric(jnp.asarray(packed), sd))
    rows, cols = np.triu_indices(size - 1)
    np.testing.assert_allclose(folded, expected[:, rows, cols], rtol=1e-5, atol=1e-5)


def test_fold_statistics_keeps_embeddings_and_rejects_trace_only():
    from recovar.ppca.pose_accumulators import AugmentedPPCAStats

    rng = np.random.default_rng(1)
    stats = AugmentedPPCAStats(
        rhs=None, lhs_tri=jnp.asarray(rng.normal(size=(4, 10)).astype(np.float32)),
        residual_gradient=jnp.asarray(rng.normal(size=(4, 4)).astype(np.float32)),
        embeddings=jnp.zeros((3, 3)),
    )
    folded = contrast.fold_statistics(stats, 0.2)
    assert folded.lhs_tri.shape == (4, 6) and folded.residual_gradient.shape == (4, 3) and folded.rhs is None
    assert folded.embeddings.shape == (3, 3)
    with pytest.raises(ValueError, match="packed metric"):
        contrast.fold_statistics(AugmentedPPCAStats(rhs=None, lhs_tri=None), 0.2)


def test_updated_sd_is_em_clipped_and_blended():
    kwargs = dict(sd_min=0.01, sd_max=0.5)
    assert contrast.updated_sd(0.2, 1.0, previous_weight=0.0, **kwargs) == pytest.approx(0.2)
    assert contrast.updated_sd(0.2, 4.0, previous_weight=0.0, **kwargs) == pytest.approx(0.4)
    assert contrast.updated_sd(0.2, 4.0, previous_weight=0.9, **kwargs) == pytest.approx(np.sqrt(0.9 * 0.04 + 0.1 * 0.16))
    assert contrast.updated_sd(0.2, 100.0, previous_weight=0.0, **kwargs) == 0.5
    assert contrast.updated_sd(0.2, 0.0, previous_weight=0.0, **kwargs) == 0.01


def test_em_recovers_the_contrast_sd_from_single_pose_images():
    """With the model known and one pose, iterating the EM update of sd from the engine's E[e^2] reaches the true sd."""
    rng = np.random.default_rng(5)
    n, variance, true_sd, count, q = 60, 1.0, 0.25, 3000, 2
    mu, w = 1.5 * rng.normal(size=n), 0.3 * rng.normal(size=(n, q))
    delta = true_sd * rng.normal(size=count)
    y = (1 + delta)[:, None] * mu + rng.normal(size=(count, q)) @ w.T + np.sqrt(variance) * rng.normal(size=(count, n))
    sd = 0.05
    for _ in range(12):
        theta = np.concatenate([mu[None], w.T, sd * mu[None]])
        result = _engine(y, theta, variance)
        from recovar.ppca.triangular import unpack_tri_to_full

        second = np.asarray(unpack_tri_to_full(result.G_tri, q + 2))[:, 0, 0, -1, -1]
        sd = contrast.updated_sd(sd, float(np.mean(second)), previous_weight=0.0, sd_min=1e-3, sd_max=1.0)
    assert sd == pytest.approx(true_sd, rel=0.1)
    theta = np.concatenate([mu[None], w.T, sd * mu[None]])
    _, estimate = contrast.split_embeddings(np.asarray(_engine(y, theta, variance).alpha[:, 0, 0, 1:]), sd)
    assert np.corrcoef(estimate, 1 + delta)[0, 1] > 0.85


def test_latents_over_contrast_divides_and_floors_and_counts():
    scaled = np.array([[2.0, 4.0], [1.0, 1.0], [3.0, 6.0]])
    estimate = np.array([2.0, 0.05, -0.4])
    latents, floored = contrast.latents_over_contrast(scaled, estimate, floor=0.1)
    np.testing.assert_allclose(latents, [[1.0, 2.0], [10.0, 10.0], [30.0, 60.0]])
    assert floored == 2
    _, none = contrast.latents_over_contrast(scaled, np.array([1.0, 1.1, 0.9]))
    assert none == 0
