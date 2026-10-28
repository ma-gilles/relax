"""The data_vs_prior Fourier radius of the PPCA InitialModel: RELION's current-resolution rule for the mean.

RELION's VDAM sets the next iteration's image size from the last shell where the mean's data_vs_prior stays at or
above 1, plus ``--incr_size`` shells (ml_optimiser.cpp ``updateCurrentResolution`` and
``updateImageSizeAndResolutionPointers``). Here the data is the mean-mean entry of the PPCA metric and the prior the
mean's own power. docs/math/vdam_ppca_algorithm.md section 18.6.
"""

import jax.numpy as jnp
import numpy as np

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


def next_radius(data_vs_prior: np.ndarray, *, radius: int, grid_size: int, incr_size: int) -> int:
    """The next iteration's Fourier radius: RELION's resolution shell plus ``incr_size`` shells.

    The radius never decreases (the update zeroes the model outside it) and stays one shell below Nyquist, the
    PPCA engine's own limit.
    """
    shell = resolution_from_data_vs_prior(data_vs_prior, box_size=grid_size)
    return int(min(max(shell + incr_size, radius), grid_size // 2 - 1))


def radius_for(config, state, iteration: int, grid_size: int) -> int:
    """The Fourier radius of ``iteration``'s E-step and update.

    The stages table gives it under the ``stages`` rule. Under ``data_vs_prior`` the table's first radius starts
    the run, the state carries the radius planned after its last update, and the iteration of the state itself (the
    final embeddings) uses the radius that update used.
    """
    cap = grid_size // 2 - 1
    if config.radius_rule == "stages":
        return min(config.stage(iteration)[0], cap)
    if iteration == state.iteration:
        return state.radius
    return state.next_radius if state.next_radius is not None else min(config.stage(1)[0], cap)
