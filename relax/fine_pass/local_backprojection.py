"""Weighted-sum helpers of the M-step rows (the resident pass-2 driver and its layout tests)."""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp

_RELION_X_HALF_SEQUENTIAL_TRANSLATION_REDUCTION_ENV = (
    "RELAX_RELION_X_HALF_SEQUENTIAL_TRANSLATION_REDUCTION"
)


def relion_x_half_sequential_translation_reduction_enabled() -> bool:
    """Return whether the diagnostic RELION-order translation reduction is enabled."""

    raw = os.environ.get(_RELION_X_HALF_SEQUENTIAL_TRANSLATION_REDUCTION_ENV)
    return raw is not None and raw.strip().lower() not in {"", "0", "false", "no", "off"}


@jax.jit
def compute_local_weighted_sums(probs, shifted):
    """Compute weighted image sums for one exact local bucket.

    probs: (B, R, T)
    shifted: (B, T, N)
    returns: (B, R, N)

    Use full float32 products for this EM-local contraction.  Accelerator
    defaults may otherwise select reduced-precision products, which produces a
    systematic BPref numerator mismatch against RELION even for one-hot
    translation posteriors.
    """

    return jnp.matmul(probs, shifted, precision=jax.lax.Precision.HIGHEST)


@jax.jit
def compute_local_ctf_sums_from_probs_sum_t(probs_sum_t, ctf2_over_nv):
    """Compute weighted CTF^2/noise sums from precomputed rotation posterior sums."""

    return jnp.where(
        probs_sum_t[..., None] != 0.0,
        probs_sum_t[..., None] * ctf2_over_nv[:, None, :],
        0.0,
    )


