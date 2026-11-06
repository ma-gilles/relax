"""The data_vs_prior Fourier radius of the PPCA InitialModel: RELION's current-resolution rule for the mean.

RELION's VDAM sets the next iteration's image size from the last shell where the mean's data_vs_prior stays at or
above 1, plus ``--incr_size`` shells (ml_optimiser.cpp ``updateCurrentResolution`` and
``updateImageSizeAndResolutionPointers``). Here the data is the mean-mean entry of the PPCA metric and the prior the
mean's own power. The ``data_vs_prior_pcs`` rule also asks the loadings' total variance to stand above its own
estimation noise. docs/math/vdam_ppca_algorithm.md sections 18.6 and 18.14.
"""

import jax.numpy as jnp
import numpy as np
from recovar.core import fourier_transform_utils as ftu
from recovar.ppca.triangular import unpack_tri_to_full

from relax.ppca_initial_model.update import metric_floor
from relax.reconstruction.regularization import compute_data_vs_prior, compute_relion_weight_shell_stats
from relax.reconstruction.resolution_criteria import resolution_from_data_vs_prior


def mean_data_vs_prior(
    mean_weight, mean, volume_shape: tuple[int, int, int], *, fudge: float, radius: int
) -> np.ndarray:
    """RELION's data_vs_prior per shell for the mean, from one pseudo-halfset's statistics.

    ``mean_weight``: the halfset's mean-mean entry of the PPCA metric (``lhs_tri[:, 0]``) and ``mean`` the reference
    its E-step used, both on the packed half volume. RELION's BPref weight is the metric over ``metric_floor``
    (the floor is RELION's weight of 1) and its reference power the raw-FFT power over N^4
    (``estep_common.relion_bpref_frame_scales``); the shells beyond ``radius`` are zero, as in RELION. Returns
    ``(N // 2 + 1,)`` float64 values.
    """
    n = int(volume_shape[0])
    weight = jnp.real(mean_weight) / metric_floor(n)
    power = jnp.abs(mean) ** 2 / n**4
    tau2 = compute_relion_weight_shell_stats(power, volume_shape, r_max=radius, shell_rounding="round")[
        "avg_weight_shells"
    ]
    return np.asarray(
        compute_data_vs_prior(weight, tau2, volume_shape, tau2_fudge=fudge, current_size=2 * radius), np.float64
    )


def loading_data_vs_prior(
    metric_tri, theta, volume_shape: tuple[int, int, int], *, fudge: float, radius: int
) -> np.ndarray:
    """The data_vs_prior per shell of the loadings' total variance, from one pseudo-halfset's statistics.

    ``metric_tri``: the halfset's packed ``(1 + q) x (1 + q)`` PPCA metric per half-volume row and ``theta`` the
    reference its E-step used (mean column first, then the ``q >= 1`` active loadings). The mean's weight is the
    back-projected CTF^2 over noise, ``metric[0, 0]``; the loadings' is not: their block carries the latents'
    second moments and the mean is solved with them. The precision of the loadings alone is the Schur complement
    ``S = L_WW - L_W0 L_00^-1 L_0W`` of the metric, and ``tr(S^-1)`` the total variance of their estimation noise
    (eigenvalues floored at ``metric_floor``). A row's weight is ``q / tr(S^-1)`` over the floor, the prior is the
    shell's mean power per loading, so the ratio is the total loading power over the total noise, in RELION's
    ``data_vs_prior`` form and invariant to a rotation of the latent basis. Shells beyond ``radius`` are zero.
    Returns ``(N // 2 + 1,)`` float64 values.
    """
    n = int(volume_shape[0])
    q = int(theta.shape[-1]) - 1
    radii = np.asarray(ftu.get_grid_of_radial_distances_real(volume_shape, rounded=False)).reshape(-1)
    rows = np.flatnonzero(radii <= radius)
    block = np.asarray(unpack_tri_to_full(jnp.asarray(metric_tri)[rows], q + 1)).real.astype(np.float64)
    block = (block + block.swapaxes(-1, -2)) / 2
    floor = float(metric_floor(n))
    cross = block[:, 1:, 0]
    schur = block[:, 1:, 1:] - cross[:, :, None] * cross[:, None, :] / np.maximum(block[:, 0, 0], floor)[:, None, None]
    noise = np.sum(1 / np.maximum(np.linalg.eigvalsh(schur), floor), axis=-1)
    weight = np.zeros(radii.shape, np.float64)
    weight[rows] = q / noise / floor
    power = np.sum(np.abs(np.asarray(theta)[:, 1:]) ** 2, axis=-1) / (q * n**4)
    tau2 = compute_relion_weight_shell_stats(power, volume_shape, r_max=radius, shell_rounding="round")[
        "avg_weight_shells"
    ]
    return np.asarray(
        compute_data_vs_prior(weight, tau2, volume_shape, tau2_fudge=fudge, current_size=2 * radius), np.float64
    )


def next_radius(data_vs_prior: np.ndarray, *, radius: int, grid_size: int, incr_size: int) -> int:
    """The next iteration's Fourier radius: RELION's resolution shell plus ``incr_size`` shells.

    The radius never decreases (the update zeroes the model outside it) and stays one shell below Nyquist, the
    PPCA engine's own limit. Under ``data_vs_prior_pcs`` ``data_vs_prior`` is the shell-wise minimum of the mean's
    and the loadings' curves, so a shell counts only when both stand at or above 1.
    """
    shell = resolution_from_data_vs_prior(data_vs_prior, box_size=grid_size)
    return int(min(max(shell + incr_size, radius), grid_size // 2 - 1))


def radius_for(config, state, iteration: int, grid_size: int) -> int:
    """The Fourier radius of ``iteration``'s E-step and update.

    The stages table gives it under the ``stages`` rule. Under the ``data_vs_prior`` rules the table's first radius starts
    the run, the state carries the radius planned after its last update, and the iteration of the state itself (the
    final embeddings) uses the radius that update used.
    """
    cap = grid_size // 2 - 1
    if config.radius_rule == "stages":
        return min(config.stage(iteration)[0], cap)
    if iteration == state.iteration:
        return state.radius
    return state.next_radius if state.next_radius is not None else min(config.stage(1)[0], cap)
