"""Coupled VDAM-PPCA direction and latent-basis-invariant shrinkage.

See docs/math/vdam_ppca_algorithm.md sections 5, 6.2 and 8.1. This is a
stochastic heuristic, not a claim of monotone marginal likelihood ascent.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from recovar.ppca.triangular import unpack_tri_to_full

from relax.ppca_refinement.residual_statistics import full_float32


@dataclass
class Moments:
    first: object  # [2, frequency, 1+q], complex64
    second: object  # [frequency, 2], float32: mean and shared loading scalar
    initialized: object  # [2, frequency], bool, independent of latent basis


def empty_moments(theta):
    return Moments(
        jnp.zeros((2,) + theta.shape, theta.dtype),
        jnp.ones((theta.shape[0], 2), theta.real.dtype),
        jnp.zeros((2, theta.shape[0]), bool),
    )


# The first moments: first = FIRST_MOMENT_DECAY * first + FIRST_MOMENT_WEIGHT * direction; the first iteration sets first = direction.
FIRST_MOMENT_DECAY = 0.9
FIRST_MOMENT_WEIGHT = 0.1


def advance_noise_weight(weight: float | None, count: int) -> float:
    """The noise weight of the first moments after one more iteration whose batch holds ``count`` particles.

    A direction's noise power is inversely proportional to the particles of its half (count / 2 each), so the two halves'
    first moments differ by noise of power s^2 * weight, with s^2 the noise power of one particle and ``weight`` the
    sum, over the iterations averaged into the moments, of (their coefficient)^2 * 4 / count. ``weight`` is None
    before the first iteration, whose direction is the moment itself.
    """
    inverse = 4.0 / count
    if weight is None:
        return inverse
    return FIRST_MOMENT_DECAY**2 * weight + FIRST_MOMENT_WEIGHT**2 * inverse


def full_data_disagreement_scale(weight: float, n_images: int) -> float:
    """The factor on the halves' disagreement power that gives the noise power of two full-data half maps.

    The full-data half map of n_images / 2 particles has noise power 2 s^2 / n_images, and the disagreement of two such
    maps is twice that; the measured disagreement is s^2 * ``weight``. For a constant batch of b particles ``weight``
    tends to 4 e / b with e = (1 - decay) / (1 + decay) the running average's noise reduction, so the factor is
    (b / n_images) / e.
    """
    return 4.0 / (n_images * weight)


def metric_floor(image_size):
    """VDAM BPref floor=1 in full-real-image PPCA statistic units.

    estep_common.relion_bpref_frame_scales gives weight_relion=N^4*weight_engine
    for per-component noise. Total-coefficient noise is twice that variance,
    so the PPCA weight/floor is half the engine value. No N/b scaling.
    """
    return np.float32(0.5 / int(image_size) ** 4)


# Blocks per batched eigh call: cuSolver's syevjBatched rejects batches between 2.0M and 3.6M 3x3 blocks
# (jaxlib 0.9.0.1, "gpusolverDnSsyevjBatched_bufferSize ... invalid value"); each block is solved on its own,
# so the chunking does not change a value. Remove it once the pinned jaxlib's syevjBatched takes a 3.6M-block batch.
EIGH_BATCH_BLOCKS = 1 << 20


def _batched_eigh(matrix):
    """``jnp.linalg.eigh`` of ``[F, n, n]`` in batches of at most :data:`EIGH_BATCH_BLOCKS` blocks."""
    if matrix.shape[0] <= EIGH_BATCH_BLOCKS:
        return jnp.linalg.eigh(matrix)
    parts = [
        jnp.linalg.eigh(matrix[start : start + EIGH_BATCH_BLOCKS])
        for start in range(0, matrix.shape[0], EIGH_BATCH_BLOCKS)
    ]
    return jnp.concatenate([p[0] for p in parts]), jnp.concatenate([p[1] for p in parts])


@full_float32
def coupled_direction(lhs_tri, residual_gradient, support, *, floor):
    """Solve the symmetric block with an orthogonally invariant spectral floor on the update's support.

    ``support`` marks the rows the update keeps (radius within the stage radius, as
    ``initialization.bandlimit_and_mask``). Rows outside hold only interpolation spill at the float floor;
    they are neither solved nor checked, and their direction is zero (algorithm section 17).
    """
    support = np.asarray(support, bool)
    rows = jnp.asarray(np.flatnonzero(support))
    matrix = unpack_tri_to_full(jnp.asarray(lhs_tri)[rows], residual_gradient.shape[-1])
    matrix = (matrix + matrix.swapaxes(-1, -2)) * 0.5
    kept_gradient = jnp.asarray(residual_gradient)[rows]
    values, vectors = _batched_eigh(matrix)
    values_host = np.asarray(values)
    bound = 32 * np.finfo(values_host.dtype).eps * np.maximum(np.max(np.abs(values_host), axis=-1), floor)
    if not np.all(np.isfinite(values_host)) or np.any(values_host[:, 0] < -bound):
        raise ValueError("Nonfinite or materially indefinite PPCA metric")
    if not np.all(np.isfinite(np.asarray(kept_gradient))):
        raise ValueError("Nonfinite direct PPCA residual gradient")
    projected = jnp.einsum("fji,fj->fi", vectors, kept_gradient)
    kept_direction = jnp.einsum("fij,fj->fi", vectors, projected / jnp.maximum(values, floor))
    direction = jnp.zeros(residual_gradient.shape, kept_direction.dtype).at[rows].set(kept_direction)
    return direction, {
        "minimum_eigenvalue": float(values_host.min()),
        "floored_eigenvalues": int(np.sum(values_host < floor)),
        "floor": float(floor),
    }


@full_float32
def stochastic_update(
    theta: jnp.ndarray,
    moments: Moments,
    directions: jnp.ndarray,
    coverage: jnp.ndarray,
    shells: np.ndarray,
    *,
    step: float,
    fudge: float,
    image_size: int,
    disagreement_scale: float,
    gate_override: np.ndarray | None = None,
    log_moments: bool = False,
) -> tuple[jnp.ndarray, Moments, dict]:
    """Simultaneous old-model update; one scalar adaptation/gate for all PCs.

    New coverage initializes each half's first moment directly. Frequencies
    outside coverage retain their moments. Caller handles solvent/Hermitian
    projection separately. Exactly zero disagreement gives rho=0 by contract.

    The shell gate is rho / (1 + rho) with rho = 2 * fudge * signal / disagreement, where signal and
    disagreement are shell means of the power of theta (|x|^2, ||W||^2 / q for the loadings) and of the half first
    moments' difference, so that at fudge 1 the gate is the half-map FSC under independent noise (RELION takes shell
    means of the amplitudes instead; docs/math/vdam_ppca_algorithm.md section 3.3). ``disagreement_scale`` multiplies the
    disagreement: full_data_disagreement_scale turns the noise of the applied direction into that of full-data half maps.
    The returned "signal" and "disagreement" are the shell means of the power, the disagreement after the scale.

    Gate study (docs/math/vdam_gate_study.md; both off by default). ``gate_override``, a ``[shells, 2]`` table (mean,
    shared loadings), replaces the estimated gate (the oracle arm); the estimate is still returned, as
    "gates_estimated". ``log_moments`` adds per-shell summaries of the second moment u and of the effective gate
    phi / (phi + (1 - phi) sqrt(u)), the weight on the average direction at the update's fixed point.
    """
    theta, directions = jnp.asarray(theta), jnp.asarray(directions)
    coverage = jnp.asarray(coverage, bool)
    first = jnp.where(
        moments.initialized[..., None],
        FIRST_MOMENT_DECAY * moments.first + FIRST_MOMENT_WEIGHT * directions,
        directions,
    )
    first = jnp.where(coverage[..., None], first, moments.first)
    initialized = moments.initialized | coverage
    active = jnp.any(coverage, axis=0)
    difference = directions[1] - directions[0]
    average = (directions[0] + directions[1]) * 0.5

    def channel_power(value):
        return jnp.stack([jnp.abs(value[:, 0]) ** 2, jnp.sum(jnp.abs(value[:, 1:]) ** 2, axis=-1)], axis=-1)

    # VDAM directions live in FFT/N² units; here they live in raw FFT units.
    eps_power = np.float32(1e-12 * int(image_size) ** 4)
    ratio = channel_power(difference) / (channel_power(average) + eps_power)
    second = jnp.where(active[:, None], 0.999 * moments.second + 0.001 * ratio, moments.second)
    mean_first = (first[0] + first[1]) * 0.5
    divisor = jnp.sqrt(second) + 1e-12
    adapted = jnp.concatenate([mean_first[:, :1] / divisor[:, :1], mean_first[:, 1:] / divisor[:, 1:]], axis=-1)
    q = theta.shape[-1] - 1

    def powers(value):
        return channel_power(value) / jnp.asarray([1, max(q, 1)], theta.real.dtype)

    shells = jnp.asarray(shells, jnp.int32)
    nshells = int(np.max(np.asarray(shells))) + 1
    counts = jnp.zeros(nshells, theta.real.dtype).at[shells].add(active.astype(theta.real.dtype))

    def shell_average(value):
        sums = jnp.zeros((nshells, 2), theta.real.dtype).at[shells].add(value * active[:, None])
        return sums / jnp.maximum(counts[:, None], 1)

    signal = shell_average(powers(theta))
    disagreement = shell_average(powers(first[0] - first[1])) * disagreement_scale
    safe_disagreement = jnp.where(disagreement > 0, disagreement, 1)
    rho = jnp.where(disagreement > 0, 2 * fudge * signal / safe_disagreement, 0)
    estimated = rho / (1 + rho)
    gates = estimated if gate_override is None else _override_table(gate_override, nshells, theta.real.dtype)
    gate_channels = gates[shells]  # [frequency, 2]
    gate = jnp.concatenate(
        [gate_channels[:, :1], jnp.broadcast_to(gate_channels[:, 1:], (theta.shape[0], q))], axis=-1
    )
    updated = jnp.where(active[:, None], theta + step * (gate * adapted - (1 - gate) * theta), theta)
    info = {
        "gates": np.asarray(gates),
        "signal": np.asarray(signal),
        "disagreement": np.asarray(disagreement),
        "zero_disagreement_shell_channels": int(
            np.sum((np.asarray(disagreement) == 0) & (np.asarray(counts)[:, None] > 0))
        ),
    }
    if gate_override is not None:
        info["gates_estimated"] = np.asarray(estimated)
    if log_moments:
        info.update(_moment_summaries(second, gate_channels, active, shells, nshells))
    return updated, Moments(first, second, initialized), info


def _override_table(table, nshells, dtype):
    """A ``[shells, 2]`` gate table cut or zero-padded to ``nshells`` shells."""
    table = np.asarray(table, np.float64)
    if table.ndim != 2 or table.shape[1] != 2 or not np.all(np.isfinite(table)) or np.any(table < 0) or np.any(table > 1):
        raise ValueError("A gate table is [shells, 2] (mean, loadings) with values in [0, 1]")
    out = np.zeros((nshells, 2))
    out[: min(nshells, len(table))] = table[:nshells]
    return jnp.asarray(out, dtype)


def _moment_summaries(second, gate_channels, active, shells, nshells):
    """Per shell and channel over the updated voxels: mean and median of sqrt(u), mean of 1/sqrt(u), and the mean
    effective gate phi / (phi + (1 - phi) sqrt(u)) (the fixed point of v + eta (phi m / sqrt(u) - (1 - phi) v) is
    that weight times the average direction's target)."""
    root = np.sqrt(np.asarray(second, np.float64))
    phi = np.asarray(gate_channels, np.float64)
    effective = phi / np.maximum(phi + (1 - phi) * root, 1e-30)
    keep = np.asarray(active, bool)
    shells = np.asarray(shells)[keep]
    counts = np.bincount(shells, minlength=nshells).astype(np.float64)

    def mean(value):
        sums = np.stack([np.bincount(shells, value[keep, c], nshells) for c in range(2)], axis=-1)
        return np.where(counts[:, None] > 0, sums / np.maximum(counts[:, None], 1), np.nan)

    median = np.full((nshells, 2), np.nan)
    order = np.argsort(shells, kind="stable")
    sorted_shells = shells[order]
    bounds = np.searchsorted(sorted_shells, np.arange(nshells + 1))
    kept_root = root[keep][order]
    for s in range(nshells):
        if bounds[s + 1] > bounds[s]:
            median[s] = np.median(kept_root[bounds[s] : bounds[s + 1]], axis=0)
    return {
        "sqrt_u_mean": mean(root),
        "sqrt_u_median": median,
        "inv_sqrt_u_mean": mean(1 / np.maximum(root, 1e-30)),
        "effective_gate_mean": mean(effective),
    }
