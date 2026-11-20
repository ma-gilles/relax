"""Orthogonal equivariance and independent scalar update checks."""

import jax.numpy as jnp
import numpy as np
import pytest
from recovar.ppca.triangular import pack_upper_tri

from helpers.float_compare import assert_matches

from relax.ppca_initial_model.update import (
    apply_px_em,
    coupled_direction,
    empty_moments,
    metric_floor,
    px_em_reduction,
    stochastic_update,
)

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
    dirs = jnp.stack(
        [
            coupled_direction(pack_upper_tri(a), r, np.ones(len(r), bool), floor=0.4)[0]
            for a, r in zip(matrices, residual)
        ]
    )
    transformed = Q.T @ matrices @ Q
    dirs_q = jnp.stack(
        [
            coupled_direction(pack_upper_tri(a), r, np.ones(len(r), bool), floor=0.4)[0]
            for a, r in zip(transformed, residual @ Q)
        ]
    )
    # Rotate the references on the host: a default-precision GPU matmul would be TF32.
    np.testing.assert_allclose(dirs_q, np.asarray(dirs) @ Q, atol=tol, rtol=tol)
    shells = np.arange(9) // 3
    args = dict(coverage=np.ones((2, 9), bool), shells=shells, step=0.8, fudge=2, image_size=8)
    out, moments, info = stochastic_update(jnp.asarray(theta), empty_moments(jnp.asarray(theta)), dirs, **args)
    out_q, moments_q, info_q = stochastic_update(
        jnp.asarray(theta @ Q), empty_moments(jnp.asarray(theta @ Q)), dirs_q, **args
    )
    np.testing.assert_allclose(out_q, np.asarray(out) @ Q, atol=tol, rtol=tol)
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
        coupled_direction(
            jnp.array([[-1.0, 0.0, 1.0]], jnp.float32), jnp.ones((1, 2), jnp.complex64), np.ones(1, bool), floor=1.0
        )


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


def test_px_em_reduction_maps_the_expanded_prior_onto_the_model():
    """theta T is (mu + W a, W chol(S)) and the half-set first moments move the same way."""
    rng = np.random.default_rng(4)
    z = rng.normal(size=(50, 2)) * [3.0, 0.5] + [2.0, -1.0]
    T, record = px_em_reduction(z, 50 * 2 * 0.1)
    second = np.cov(z.T, bias=True) + 0.1 * np.eye(2)
    chol = np.linalg.cholesky(second)
    assert_matches(np.asarray(record["px_em_zbar"]), z.mean(0), rtol=1e-12)
    assert_matches(T[1:, 1:] @ T[1:, 1:].T, second, rtol=1e-12)
    theta = (rng.normal(size=(7, 3)) + 1j * rng.normal(size=(7, 3))).astype(np.complex64)
    moments = empty_moments(jnp.asarray(theta))
    moments.first = jnp.asarray((rng.normal(size=(2, 7, 3)) + 1j * rng.normal(size=(2, 7, 3))).astype(np.complex64))
    reduced, moved = apply_px_em(jnp.asarray(theta), moments, T)
    expected = np.concatenate([theta[:, :1] + theta[:, 1:] @ z.mean(0)[:, None], theta[:, 1:] @ chol], axis=1)
    assert_matches(np.asarray(reduced), expected, rtol=2e-6)
    assert_matches(np.asarray(moved.first[1]), np.asarray(moments.first[1]) @ T, rtol=2e-6)
    assert moved.second is moments.second and moved.initialized is moments.initialized


def _ppca_em_update(theta, x, s2, px_em):
    """One exact EM update of plain PPCA (no projection) through RELAX's M-step solve; optionally PX-EM after."""
    q = theta.shape[1] - 1
    W = theta[:, 1:]
    M = W.T @ W + s2 * np.eye(q)
    m = (x - theta[:, 0]) @ W @ np.linalg.inv(M)
    S = s2 * np.linalg.inv(M)
    a = np.concatenate([np.ones((len(x), 1)), m], axis=1)
    second = a.T @ a
    second[1:, 1:] += len(x) * S
    lhs = np.broadcast_to(pack_upper_tri(jnp.asarray(second / s2))[None], (theta.shape[0], (q + 1) * (q + 2) // 2))
    gradient = ((x.T @ a - theta @ second) / s2).astype(np.complex64)
    direction, _ = coupled_direction(jnp.asarray(lhs, jnp.float32), jnp.asarray(gradient), np.ones(len(theta), bool), floor=1e-12)
    new = theta + np.real(np.asarray(direction, np.complex128))
    if px_em:
        T, _ = px_em_reduction(m, len(x) * S)
        new = new @ T
    return new, m.mean(axis=0), np.cov(m.T, bias=True) + S


def test_px_em_removes_the_slow_em_mode_of_the_latent_mean():
    """Exact EM moves zbar by (I - s2 M^-1) per update (0.97-0.99 here: the mean-loading direction is EM's slow mode);
    with the PX-EM reduction zbar and the latent second moment reach 0 and the identity within a few updates."""
    rng = np.random.default_rng(0)
    F, q, n, s2 = 60, 2, 200, 1.0
    mu = rng.normal(size=F) * 3
    W = np.linalg.qr(rng.normal(size=(F, q)))[0] * np.sqrt([100.0, 30.0])
    x = mu + rng.normal(size=(n, q)) @ W.T + rng.normal(size=(n, F)) * np.sqrt(s2)
    start = np.concatenate([(mu - W @ [2.0, -1.5])[:, None], W / 4], axis=1)
    theta = start.copy()
    for _ in range(3):
        theta, zbar, _ = _ppca_em_update(theta, x, s2, px_em=False)
    nxt, zbar, _ = _ppca_em_update(theta, x, s2, px_em=False)
    Wn = nxt[:, 1:]
    predicted = (np.eye(q) - s2 * np.linalg.inv(Wn.T @ Wn + s2 * np.eye(q))) @ zbar
    _, measured, _ = _ppca_em_update(nxt, x, s2, px_em=False)
    # float32 metric solve against float64: measured 1e-6 relative.
    assert_matches(measured, predicted, rtol=1e-4)
    assert np.linalg.norm(measured) > 0.9 * np.linalg.norm(zbar)
    theta = start.copy()
    history = []
    for _ in range(6):
        theta, zbar, second = _ppca_em_update(theta, x, s2, px_em=True)
        history.append((np.linalg.norm(zbar), np.max(np.abs(second - np.eye(q)))))
    assert history[0][0] > 5 and history[2][0] < 1e-3 and history[2][1] < 0.02
    assert_matches(second, np.eye(q), rtol=1e-3)
    # The maximum-likelihood point (fixed noise) is a fixed point of the PX-EM update: there T = I.
    centred = x - x.mean(axis=0)
    values, vectors = np.linalg.eigh(centred.T @ centred / n)
    ml = np.concatenate([x.mean(axis=0)[:, None], vectors[:, ::-1][:, :q] * np.sqrt(values[::-1][:q] - s2)], axis=1)
    moved, zbar, second = _ppca_em_update(ml, x, s2, px_em=True)
    assert np.linalg.norm(zbar) < 1e-8
    assert_matches(second, np.eye(q), rtol=1e-6)
    # float32 metric solve: measured 2e-7 of the largest loading entry.
    assert_matches(moved, ml, rtol=1e-5)
