"""Scalar-curvature momentum step for the augmented PPCA Fourier model.

See docs/math/ppca_momentum_sgd.md. The direct residual gradient
comes from the existing pose-marginal statistics path; its gridded block is
only a curvature scale, not a replacement for that gradient.
"""

import jax.numpy as jnp
import numpy as np


def metric_trace(lhs_tri, p):
    """Per-frequency trace ``sum_k LHS_kk`` of packed upper ``(n_frequency, tri(p))`` metrics."""
    return jnp.sum(metric_diagonal(lhs_tri, p), axis=-1)


def metric_diagonal(lhs_tri, p):
    """Per-frequency diagonal ``LHS_kk``, ``(n_frequency, p)``, of packed upper ``(n_frequency, tri(p))`` metrics."""
    tri_i, tri_j = np.triu_indices(p)
    return jnp.asarray(lhs_tri)[:, np.flatnonzero(tri_i == tri_j)]


def gated_gradient(theta, gradient, diagonal, gate):
    """The ascent direction ``gate G - (1 - gate) diag(H) theta`` of the gated momentum step.

    ``gradient`` ``G`` is the direct residual gradient and ``diagonal`` the metric diagonal
    ``(n_frequency, 1+q)`` (:func:`metric_diagonal`) at ``theta``; ``gate`` ``(n_frequency, 1+q)`` in
    [0, 1] is the half-set shell gate (:meth:`relax.ppca_initial_model.update.ShellGates.per_frequency`).
    For a diagonal metric the direction vanishes at ``gate`` times the unregularised solution: a
    ridge penalty proportional to the data precision, whose strength the half-sets' disagreement sets.
    Gate 1 is the plain gradient.
    """
    theta = jnp.asarray(theta)
    gradient = jnp.asarray(gradient)
    diagonal = jnp.asarray(diagonal)
    gate = jnp.asarray(gate)
    if gradient.shape != theta.shape or diagonal.shape != theta.shape or gate.shape != theta.shape:
        raise ValueError("PPCA model, gradient, metric diagonal and gate must have the same [frequency, channel] shape")
    if diagonal.dtype != jnp.float32 or gate.dtype != jnp.float32:
        raise TypeError("Production PPCA metric diagonal and gate must be float32")
    return gate * gradient - (1 - gate) * diagonal * theta


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
