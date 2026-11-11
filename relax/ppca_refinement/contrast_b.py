"""Per-particle contrast on the whole signal (``docs/math/ppca_contrast.md``, model B), integrated by adaptive quadrature.

The physical model is ``y = c A (mu + W z) + noise`` with ``c ~ N(kappa, sigma^2)`` and ``z ~ N(0, I)``. Given the pose, ``z``
integrates in closed form for every ``c``; the one-dimensional integral over ``c`` is done per pose by Gauss-Hermite
quadrature centred on model A's Gaussian posterior of the contrast (a closed form, exact to about three digits against the
physical model), so a few nodes follow the narrow, pose-dependent likelihood that a grid on the prior would miss.

Everything is computed from the engine's two GEMM outputs for the columns ``[mu, W]``: ``inner`` ``(P, R, B, T)`` (the
pose-dependent ``t = <a, y>`` and ``g = B^* D y``) and the packed Gram ``(tri(P), R, B)`` (``nu``, ``h``, ``H``, which depend only
on the rotation and the image). In the eigenbasis of ``H`` each node costs ``O(q)`` per pose; the eigendecomposition is per
(rotation, image). :func:`latent_scores_b` is pass 1 (the pose evidence), :func:`posterior_moments_b` pass 2 (the M-step
moments ``E[c alpha]`` and ``E[c^2 alpha alpha^T]`` with ``alpha = (1, z)``, in the layout the streamed engine's M-step reads).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

HIGHEST = jax.lax.Precision.HIGHEST


class ContrastPrior(NamedTuple):
    """The contrast model of one expectation: ``nodes`` Gauss-Hermite nodes per pose and the prior ``c ~ N(mean, sd^2)``.

    With ``normalise`` the M-step's moments are those of ``c / mean`` (the population mean of the contrast is taken as the
    gain between the mean volume and the images, which the volumes carry; docs/math/ppca_contrast.md).
    """

    nodes: int
    mean: float
    sd: float
    normalise: bool


class GramFactors(NamedTuple):
    """Per (rotation, image) factors of the Gram of ``[mu, W]``: ``nu = <a, a>``, the eigendecomposition ``H = U diag(lam) U^T``
    and ``h`` rotated into that basis."""

    nu: jax.Array  # (R, B)
    lam: jax.Array  # (q, R, B) eigenvalues of H, at least 0
    rot: jax.Array  # (R, B, q, q) eigenvectors U (columns)
    h_rot: jax.Array  # (q, R, B) U^T h


class PoseMoments(NamedTuple):
    """Pass-2 outputs of one rotation block, in the layout of the engine's M-step."""

    weights: jax.Array  # (P, R, B, T) gamma E[c alpha]
    sums: jax.Array  # (tri(P), R, B) sum_t gamma E[c^2 alpha alpha^T], packed upper
    embedding: jax.Array  # (P, R, B) sum_t gamma [E z, E c]
    latent_trace: jax.Array  # (R, B) sum_t gamma tr Cov[z | y, pose]


def hermite_rule(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes ``x`` and weights ``w`` (summing to 1) of the ``n_nodes``-point Gauss-Hermite rule for the standard normal."""
    x, w = np.polynomial.hermite_e.hermegauss(n_nodes)
    return x, w / np.sqrt(2.0 * np.pi)


def packed_index(size: int) -> dict[tuple[int, int], int]:
    """Position of entry ``(i, j)``, ``i <= j``, in the packed upper triangle of a ``size x size`` matrix."""
    return {(int(i), int(j)): k for k, (i, j) in enumerate(zip(*np.triu_indices(size)))}


def gram_factors(gram: jax.Array, q: int) -> GramFactors:
    """Factors of the packed Gram ``(tri(q + 1), R, B)`` of ``[mu, W_1 .. W_q]``."""
    index = packed_index(q + 1)
    nu = gram[index[(0, 0)]]
    if q == 0:
        return GramFactors(nu, jnp.zeros((0,) + nu.shape, nu.dtype), jnp.zeros(nu.shape + (0, 0), nu.dtype),
                           jnp.zeros((0,) + nu.shape, nu.dtype))
    rows = [jnp.stack([gram[index[(min(i, j), max(i, j))]] for j in range(1, q + 1)], axis=-1) for i in range(1, q + 1)]
    loadings_gram = jnp.stack(rows, axis=-2)  # (R, B, q, q)
    h = jnp.stack([gram[index[(0, j)]] for j in range(1, q + 1)])  # (q, R, B)
    lam, rot = jnp.linalg.eigh(loadings_gram)
    lam = jnp.maximum(jnp.moveaxis(lam, -1, 0), 0.0)
    return GramFactors(nu, lam, rot, jnp.einsum("rbij,irb->jrb", rot, h, precision=HIGHEST))


def rotate_in(factors: GramFactors, vectors: jax.Array) -> jax.Array:
    """``U^T v`` for vectors ``(q, R, B, T)`` (the loading components of ``g``)."""
    return jnp.einsum("rbij,irbt->jrbt", factors.rot, vectors, precision=HIGHEST)


def rotate_out(factors: GramFactors, vectors: jax.Array) -> jax.Array:
    """``U v`` for eigenbasis vectors ``(q, R, B, T)``."""
    return jnp.einsum("rbij,jrbt->irbt", factors.rot, vectors, precision=HIGHEST)


def contrast_proposal(t, g_rot, factors: GramFactors, kappa, sigma):
    """Model A's Gaussian posterior of ``c`` per pose: ``(mean, sd)`` ``(R, B, T)``.

    Contrast on the mean only, ``c = kappa + sigma e``: eliminating ``z`` leaves the precision
    ``1 + sigma^2 (nu - sum_j h_j^2 / (1 + lam_j))`` for ``e`` and the mean follows from the same eigenbasis sums.
    """
    nu = factors.nu[..., None]
    if g_rot.shape[0]:
        h = factors.h_rot[..., None]
        shifted = 1.0 + factors.lam[..., None]
        coupling = jnp.sum(h * h / shifted, axis=0)
        linear = jnp.sum(h * (g_rot - kappa * h) / shifted, axis=0)
    else:
        coupling, linear = 0.0, 0.0
    precision = 1.0 + sigma**2 * (nu - coupling)
    return kappa + sigma**2 * ((t - kappa * nu) - linear) / precision, sigma / jnp.sqrt(precision)


def node_log_integrand(c, t, g_rot, factors: GramFactors, kappa, sigma):
    """``log p(y | c, pose) + log p(c)`` at contrasts ``c`` ``(K, R, B, T)``, without the pose-independent ``-<y, y> / 2``.

    ``z`` is integrated in closed form: ``-1/2 [ r(c) - sum_j c^2 (g_j - c h_j)^2 / (1 + c^2 lam_j) + sum_j log(1 + c^2 lam_j) ]``
    with ``r(c) = c^2 nu - 2 c t``, in the eigenbasis of ``H``; ``p(c)`` is the normal density ``N(kappa, sigma^2)``.
    """
    c2 = c * c
    nu = factors.nu[None, ..., None]
    loglik = -0.5 * (c2 * nu - 2.0 * c * t[None])
    if g_rot.shape[0]:
        lam = factors.lam[None, ..., None]
        denominator = 1.0 + c2[:, None] * lam
        residual = g_rot[None] - c[:, None] * factors.h_rot[None, ..., None]
        loglik = loglik + 0.5 * jnp.sum(c2[:, None] * residual**2 / denominator, axis=1)
        loglik = loglik - 0.5 * jnp.sum(jnp.log(denominator), axis=1)
    return loglik - 0.5 * ((c - kappa) / sigma) ** 2 - jnp.log(sigma * jnp.sqrt(2.0 * jnp.pi))


def adaptive_nodes(inner, factors: GramFactors, kappa, sigma, rule):
    """``(c, log_weight, g_rot)``: Gauss-Hermite nodes ``(K, R, B, T)`` on model A's posterior of the contrast, their log
    importance weights and the pose's inner products with the loadings in the eigenbasis ``(q, R, B, T)``.

    ``sum_k exp(log_weight_k)`` is the pose's evidence ``int p(y | c, pose) p(c) dc`` (up to ``-<y, y> / 2``) and the
    responsibilities are the softmax of ``log_weight``: the integrand over its Gaussian proposal, ``f(c) / N(c; m, s^2)``, is
    smooth, so a few nodes integrate it.
    """
    x, w = (jnp.asarray(a, inner.dtype) for a in rule)
    t, g_rot = inner[0], rotate_in(factors, inner[1:])
    mean, sd = contrast_proposal(t, g_rot, factors, kappa, sigma)
    c = mean[None] + sd[None] * x[:, None, None, None]
    log_measure = (jnp.log(w) + 0.5 * x**2)[:, None, None, None] + jnp.log(sd * jnp.sqrt(2.0 * jnp.pi))[None]
    return c, log_measure + node_log_integrand(c, t, g_rot, factors, kappa, sigma), g_rot


def latent_scores_b(inner, gram, pose_log_prior, kappa, sigma, rule, q: int):
    """Pass 1: the pose log-scores ``(R, B, T)`` ``log p(y | pose) + log pi(pose)`` with the contrast integrated.

    ``inner`` ``(q + 1, R, B, T)``, ``gram`` ``(tri(q + 1), R, B)``, ``pose_log_prior`` ``(B, R, T)``; ``rule`` the Gauss-Hermite
    ``(nodes, weights)``. At ``sigma -> 0`` this is the unit-contrast score of :func:`relax.ppca_refinement.full_row_kernels.latent_scores`.
    """
    _, log_weight, _ = adaptive_nodes(inner, gram_factors(gram, q), kappa, sigma, rule)
    return jax.scipy.special.logsumexp(log_weight, axis=0) + jnp.transpose(pose_log_prior, (1, 0, 2))


def posterior_moments_b(inner, gram, gamma, kappa, sigma, rule, q: int) -> PoseMoments:
    """Pass 2 with the adaptive nodes: :func:`moments_from_nodes` of one rotation block given the pose posterior ``gamma``."""
    factors = gram_factors(gram, q)
    c, log_weight, g_rot = adaptive_nodes(inner, factors, kappa, sigma, rule)
    return moments_from_nodes(c, log_weight, g_rot, factors, gamma, q)


def moments_from_nodes(c, log_weight, g_rot, factors: GramFactors, gamma, q: int) -> PoseMoments:
    """The M-step moments of one rotation block from contrast nodes ``c`` ``(K, R, B, T)`` and their log weights.

    With node responsibilities ``rho_k`` and, per node, the latent posterior ``N(m_k, M_k^-1)`` of ``z`` given ``c_k`` (``M_k = I +
    c_k^2 H``): ``E[c alpha] = sum_k rho_k c_k (1, m_k)`` per pose and ``E[c^2 alpha alpha^T] = sum_k rho_k c_k^2 [[1, m_k^T],
    [m_k, M_k^-1 + m_k m_k^T]]``, summed over translations with ``gamma``; the sums are formed in the eigenbasis and rotated once
    per (rotation, image).
    """
    rho = jax.nn.softmax(log_weight, axis=0)  # (K, R, B, T)
    c2 = c * c
    shifted = 1.0 + c2[:, None] * factors.lam[None, ..., None]  # (K, q, R, B, T)
    mean = c[:, None] * (g_rot[None] - c[:, None] * factors.h_rot[None, ..., None]) / shifted  # E[z_rot | c_k]
    first = jnp.sum(rho * c, axis=0)  # E[c]
    second = jnp.sum(rho * c2, axis=0)  # E[c^2]
    mean_c = jnp.sum(rho[:, None] * c[:, None] * mean, axis=0)  # E[c z_rot]
    index = packed_index(q + 1)
    P = q + 1
    weights = jnp.concatenate([(gamma * first)[None], gamma[None] * rotate_out(factors, mean_c)], axis=0)
    weight_k = gamma[None] * rho * c2  # (K, R, B, T): gamma rho_k c_k^2
    sums = [None] * len(index)
    sums[index[(0, 0)]] = jnp.sum(gamma * second, axis=-1)
    embedding_z = jnp.sum(gamma[None] * jnp.sum(rho[:, None] * mean, axis=0), axis=-1)  # (q, R, B) rotated
    trace = jnp.zeros(gamma.shape[:2], gamma.dtype)
    if q:
        linear = jnp.sum(weight_k[:, None] * mean, axis=(0, -1))  # (q, R, B) sum gamma rho c^2 m
        outer = jnp.einsum("krbt,kjrbt,klrbt->jlrb", weight_k, mean, mean, precision=HIGHEST)
        outer = outer + jnp.einsum("jrb,jl->jlrb", jnp.sum(weight_k[:, None] / shifted, axis=(0, -1)), jnp.eye(q, dtype=gamma.dtype))
        linear_out = jnp.einsum("rbij,jrb->irb", factors.rot, linear, precision=HIGHEST)
        outer_out = jnp.einsum("rbij,jlrb,rbkl->ikrb", factors.rot, outer, factors.rot, precision=HIGHEST)
        for i in range(1, P):
            sums[index[(0, i)]] = linear_out[i - 1]
            for j in range(i, P):
                sums[index[(i, j)]] = outer_out[i - 1, j - 1]
        embedding_z = jnp.einsum("rbij,jrb->irb", factors.rot, embedding_z, precision=HIGHEST)
        node_trace = jnp.sum(rho * jnp.sum(1.0 / shifted + mean * mean, axis=1), axis=0)  # sum_k rho_k (tr M_k^-1 + |m_k|^2)
        centre = jnp.sum(rho[:, None] * mean, axis=0)
        trace = jnp.sum(gamma * (node_trace - jnp.sum(centre * centre, axis=0)), axis=-1)
    embedding = jnp.concatenate([embedding_z, jnp.sum(gamma * first, axis=-1)[None]], axis=0)
    return PoseMoments(weights=weights, sums=jnp.stack(sums, axis=0), embedding=embedding, latent_trace=trace)
