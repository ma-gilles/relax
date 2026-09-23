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


def metric_floor(image_size):
    """VDAM BPref floor=1 in full-real-image PPCA statistic units.

    layout.relion_bpref_frame_scales gives weight_relion=N^4*weight_engine
    for per-component noise. Total-coefficient noise is twice that variance,
    so the PPCA weight/floor is half the engine value. No N/b scaling.
    """
    return np.float32(0.5 / int(image_size) ** 4)


@full_float32
def coupled_direction(lhs_tri, residual_gradient, *, floor):
    """Solve the symmetric block with an orthogonally invariant spectral floor."""
    matrix = unpack_tri_to_full(jnp.asarray(lhs_tri), residual_gradient.shape[-1])
    matrix = (matrix + matrix.swapaxes(-1, -2)) * 0.5
    values, vectors = jnp.linalg.eigh(matrix)
    values_host = np.asarray(values)
    bound = 32 * np.finfo(values_host.dtype).eps * np.maximum(np.max(np.abs(values_host), axis=-1), floor)
    if not np.all(np.isfinite(values_host)) or np.any(values_host[:, 0] < -bound):
        raise ValueError("Nonfinite or materially indefinite PPCA metric")
    if not np.all(np.isfinite(np.asarray(residual_gradient))):
        raise ValueError("Nonfinite direct PPCA residual gradient")
    projected = jnp.einsum("fji,fj->fi", vectors, residual_gradient)
    direction = jnp.einsum("fij,fj->fi", vectors, projected / jnp.maximum(values, floor))
    return direction, {
        "minimum_eigenvalue": float(values_host.min()),
        "floored_eigenvalues": int(np.sum(values_host < floor)),
        "floor": float(floor),
    }


@full_float32
def stochastic_update(theta, moments, directions, coverage, shells, *, step, fudge, image_size):
    """Simultaneous old-model update; one scalar adaptation/gate for all PCs.

    New coverage initializes each half's first moment directly. Frequencies
    outside coverage retain their moments. Caller handles solvent/Hermitian
    projection separately. Exactly zero disagreement gives rho=0 by contract.
    """
    theta, directions = jnp.asarray(theta), jnp.asarray(directions)
    coverage = jnp.asarray(coverage, bool)
    first = jnp.where(moments.initialized[..., None], 0.9 * moments.first + 0.1 * directions, directions)
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

    def amplitudes(value):
        return jnp.stack(
            [
                jnp.abs(value[:, 0]),
                jnp.linalg.norm(value[:, 1:], axis=-1) / jnp.sqrt(jnp.asarray(max(q, 1), theta.real.dtype)),
            ],
            axis=-1,
        )

    shells = jnp.asarray(shells, jnp.int32)
    nshells = int(np.max(np.asarray(shells))) + 1
    counts = jnp.zeros(nshells, theta.real.dtype).at[shells].add(active.astype(theta.real.dtype))

    def shell_average(value):
        sums = jnp.zeros((nshells, 2), theta.real.dtype).at[shells].add(value * active[:, None])
        return sums / jnp.maximum(counts[:, None], 1)

    signal = shell_average(amplitudes(theta))
    disagreement = shell_average(amplitudes(first[0] - first[1]))
    safe_disagreement = jnp.where(disagreement > 0, disagreement, 1)
    rho = jnp.where(disagreement > 0, 2 * fudge * signal / safe_disagreement, 0)
    gates = rho / (1 + rho)
    gate = gates[shells]
    gate = jnp.concatenate([gate[:, :1], jnp.broadcast_to(gate[:, 1:], (theta.shape[0], q))], axis=-1)
    updated = jnp.where(active[:, None], theta + step * (gate * adapted - (1 - gate) * theta), theta)
    return (
        updated,
        Moments(first, second, initialized),
        {
            "gates": np.asarray(gates),
            "signal": np.asarray(signal),
            "disagreement": np.asarray(disagreement),
            "zero_disagreement_shell_channels": int(
                np.sum((np.asarray(disagreement) == 0) & (np.asarray(counts)[:, None] > 0))
            ),
        },
    )
