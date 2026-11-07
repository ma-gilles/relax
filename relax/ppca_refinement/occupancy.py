"""An explicit empty component for the streamed PPCA posterior.

An observation is either empty (nothing beyond the fixed part already removed from it) or occupied by one placed
copy of the model, with prior probability ``occupancy`` (eta). The stream's pose scores exclude the pose-invariant
image energy ``-|y|^2 / 2`` (``pose_invariant_score_offset``), so in score units the empty hypothesis scores exactly
0: its log-likelihood is that same offset. With pose log-priors that sum to one over the grid, the evidences are

    occupied:  log eta + log sum_g exp(score_g) = log eta + center + centered_logZ
    empty:     log(1 - eta) + 0

Pass 2 weights every pose by ``exp(score - center - centered_logZ)``. Replacing ``centered_logZ`` by
``log Z - log eta - center``, with ``Z`` the two-component normalizer, multiplies each occupied pose's weight by
eta / Z: the joint posterior of (occupied, pose). Every pass-2 statistic (moments, embeddings, rotation masses,
offsets) is then weighted by P(occupied) with no kernel change. The residual power needs nothing: it starts at the
observation power ``sum |y|^2`` of every image with weight 1, which is the empty hypothesis's residual, and pass 2
adds only the weighted occupied corrections.
"""

import jax.numpy as jnp
import numpy as np

from relax.ppca_refinement.full_row_kernels import Posterior


def check_occupancy(occupancy) -> float | None:
    """``None`` (no empty component) or a prior in (0, 1]."""
    if occupancy is None:
        return None
    occupancy = float(occupancy)
    if not 0.0 < occupancy <= 1.0:
        raise ValueError(f"Occupancy prior must be in (0, 1], got {occupancy}")
    return occupancy


def with_empty_component(posterior: Posterior, occupancy: float) -> tuple[Posterior, jnp.ndarray]:
    """``(posterior, p_occupied)``: the pass-1 normalization of the two-component model, and P(occupied) per image.

    Images with an infinite ``center`` (tile padding, :func:`drop_padding`) keep it and get ``p_occupied`` 0.
    ``occupancy`` 1 returns the posterior unchanged and ``p_occupied`` 1 for every real image.
    """
    real = jnp.isfinite(posterior.center)
    if occupancy == 1.0:
        return posterior, real.astype(jnp.float32)
    log_eta = jnp.float32(np.log(occupancy))
    log_empty = jnp.float32(np.log1p(-occupancy))
    center = jnp.where(real, posterior.center, 0.0)
    log_occupied = log_eta + center + posterior.centered_logZ
    log_z = jnp.logaddexp(log_occupied, log_empty)
    centered_logZ = jnp.where(real, log_z - log_eta - center, posterior.centered_logZ)
    p_occupied = jnp.where(real, jnp.exp(log_occupied - log_z), 0.0)
    return posterior._replace(centered_logZ=centered_logZ), p_occupied


def log_partition_correction(occupancy: float | None) -> float:
    """What to add to ``center + centered_logZ`` after :func:`with_empty_component` for the absolute log-partition
    of the two-component model (``log eta``; 0 without an empty component)."""
    return 0.0 if occupancy is None else float(np.log(occupancy))
