"""Momentum step for the augmented PPCA Fourier model: one curvature scalar for the mean, one for the loadings.

See docs/math/ppca_momentum_sgd.md. The direct residual gradient
comes from the existing pose-marginal statistics path; its gridded block is
only a curvature scale, not a replacement for that gradient.
"""

import jax.numpy as jnp
import numpy as np


def metric_curvature(lhs_tri, p):
    """Per-frequency ``(n_frequency, 2)`` curvature of packed upper ``(n_frequency, tri(p))`` metrics: the mean's
    entry ``LHS_00`` and the trace ``sum_{k>=1} LHS_kk`` of the loading block (zero when ``p == 1``)."""
    tri_i, tri_j = np.triu_indices(p)
    diagonal = jnp.asarray(lhs_tri)[:, np.flatnonzero(tri_i == tri_j)]
    return jnp.stack([diagonal[:, 0], jnp.sum(diagonal[:, 1:], axis=-1)], axis=-1)


def momentum_step(theta, previous, gradient, curvature, active, *, learning_rate, floor, loading_scale=1.0):
    """Return an unprojected step and momentum in raw full-real FFT units.

    ``curvature`` is the per-frequency ``(mean, loading trace)`` of the augmented LHS metric
    (:func:`metric_curvature`). The mean column is divided by the largest mean entry over the active frequencies
    and every loading column by the largest loading trace, so the mean's step does not depend on how many
    loadings the model has. The trace of the loading block bounds that positive-semidefinite block and is
    invariant under rotations of the loading coordinates; a diagonal maximum or separate scales per loading would
    break that symmetry. The loading columns' new-gradient term is multiplied by ``loading_scale`` (their learning
    rate relative to the mean's), the same for every loading column. The caller applies its common Fourier/solvent
    operation to both outputs.
    """
    theta = jnp.asarray(theta)
    previous = jnp.asarray(previous)
    gradient = jnp.asarray(gradient)
    curvature = jnp.asarray(curvature)
    active = jnp.asarray(active, bool)
    if theta.ndim != 2 or gradient.shape != theta.shape or previous.shape != theta.shape:
        raise ValueError("PPCA model, momentum and gradient must have the same [frequency, channel] shape")
    if curvature.shape != (theta.shape[0], 2) or active.shape != (theta.shape[0],):
        raise ValueError("PPCA curvature or active Fourier support has the wrong shape")
    if theta.dtype != jnp.complex64 or previous.dtype != theta.dtype or gradient.dtype != theta.dtype:
        raise TypeError("Production PPCA model, momentum and gradient must be complex64")
    if curvature.dtype != jnp.float32:
        raise TypeError("Production PPCA curvature must be float32")
    if not np.isfinite(learning_rate) or learning_rate <= 0 or not np.isfinite(floor) or floor <= 0:
        raise ValueError("Learning rate and curvature floor must be finite and positive")
    if not np.isfinite(loading_scale) or loading_scale <= 0:
        raise ValueError("Loading scale must be finite and positive")

    in_band = jnp.where(active[:, None], curvature, 0.0)
    curvature_max = jnp.max(in_band, axis=0)
    scale = jnp.maximum(curvature_max, jnp.asarray(floor, jnp.float32))
    # A group with zero curvature cannot provide an update, including stale velocity.
    has_data = curvature_max > 0
    if not bool(jnp.all(jnp.isfinite(gradient))) or not bool(jnp.all(jnp.isfinite(curvature))):
        raise ValueError("Nonfinite PPCA gradient or curvature")
    if bool(jnp.any(jnp.min(in_band, axis=0) < -32 * jnp.finfo(jnp.float32).eps * scale)):
        raise ValueError("Materially negative PPCA curvature")
    group = (jnp.arange(theta.shape[1]) > 0).astype(jnp.int32)  # 0: the mean column, 1: the loading columns
    increment = jnp.where(active[:, None], gradient, 0) / scale[group]
    rate = jnp.asarray([0.1 * learning_rate, 0.1 * learning_rate * loading_scale], jnp.float32)
    velocity = jnp.where(
        has_data[group],
        jnp.asarray(0.9, jnp.float32) * jnp.where(active[:, None], previous, 0) + rate[group] * increment,
        jnp.zeros_like(previous),
    )
    diagnostics = {
        "curvature_mean": float(scale[0]),
        "curvature_loading": float(scale[1]) if theta.shape[1] > 1 else None,
        "gradient_l2": float(jnp.linalg.norm(gradient)),
        "velocity_l2_before_projection": float(jnp.linalg.norm(velocity)),
        "active_frequency_count": int(jnp.sum(active)),
    }
    return theta + velocity, velocity, diagnostics
