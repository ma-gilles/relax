"""Scalar-curvature momentum step for the augmented PPCA Fourier model.

See docs/math/ppca_momentum_sgd.md. The direct residual gradient
comes from the existing pose-marginal statistics path; its gridded block is
only a curvature scale, not a replacement for that gradient.
"""

import jax.numpy as jnp
import numpy as np


def metric_trace(lhs_tri, p):
    """Per-frequency trace ``sum_k LHS_kk`` of packed upper ``(n_frequency, tri(p))`` metrics."""
    tri_i, tri_j = np.triu_indices(p)
    return jnp.sum(jnp.asarray(lhs_tri)[:, np.flatnonzero(tri_i == tri_j)], axis=-1)


def momentum_step(theta, previous, gradient, trace, active, *, learning_rate, floor):
    """Return an unprojected step and momentum in raw full-real FFT units.

    ``trace`` is the per-frequency trace of the augmented LHS metric
    (:func:`metric_trace`, or a trace-only stream's ``metric_trace``).
    One scalar bounds every positive-semidefinite augmented block by its trace.
    Trace is invariant under rotations of the two loading coordinates; a
    diagonal maximum or separate loading scales would break that symmetry.
    The caller applies its common Fourier/solvent operation to both outputs.
    """
    theta = jnp.asarray(theta)
    previous = jnp.asarray(previous)
    gradient = jnp.asarray(gradient)
    trace = jnp.asarray(trace)
    active = jnp.asarray(active, bool)
    if theta.ndim != 2 or gradient.shape != theta.shape or previous.shape != theta.shape:
        raise ValueError("PPCA model, momentum and gradient must have the same [frequency, channel] shape")
    if trace.shape != (theta.shape[0],) or active.shape != (theta.shape[0],):
        raise ValueError("PPCA curvature or active Fourier support has the wrong shape")
    if theta.dtype != jnp.complex64 or previous.dtype != theta.dtype or gradient.dtype != theta.dtype:
        raise TypeError("Production PPCA model, momentum and gradient must be complex64")
    if trace.dtype != jnp.float32:
        raise TypeError("Production PPCA curvature must be float32")
    if not np.isfinite(learning_rate) or learning_rate <= 0 or not np.isfinite(floor) or floor <= 0:
        raise ValueError("Learning rate and curvature floor must be finite and positive")

    trace_max = jnp.max(jnp.where(active, trace, 0.0))
    max_curvature = jnp.maximum(trace_max, jnp.asarray(floor, jnp.float32))
    # A zero-curvature batch cannot provide an update, including stale velocity.
    has_data = trace_max > 0
    if not bool(jnp.all(jnp.isfinite(gradient))) or not bool(jnp.all(jnp.isfinite(trace))):
        raise ValueError("Nonfinite PPCA gradient or curvature")
    if bool(jnp.min(jnp.where(active, trace, 0.0)) < -32 * jnp.finfo(jnp.float32).eps * max_curvature):
        raise ValueError("Materially negative PPCA curvature trace")
    increment = jnp.where(active[:, None], gradient, 0) / max_curvature
    velocity = jnp.where(
        has_data,
        jnp.asarray(0.9, jnp.float32) * jnp.where(active[:, None], previous, 0)
        + jnp.asarray(0.1 * learning_rate, jnp.float32) * increment,
        jnp.zeros_like(previous),
    )
    diagnostics = {
        "curvature_trace_max": float(trace_max),
        "curvature_scalar": float(max_curvature),
        "gradient_l2": float(jnp.linalg.norm(gradient)),
        "velocity_l2_before_projection": float(jnp.linalg.norm(velocity)),
        "active_frequency_count": int(jnp.sum(active)),
    }
    return theta + velocity, velocity, diagnostics
