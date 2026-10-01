"""Posterior helpers of the sparse pass 2.

RELION's ``convertAllSquaredDifferencesToWeights`` counterpart: the float32 fine posterior of
the RELION x-half reconstruction path and the helpers the resident drivers and the exact
local engine share. The resident drivers run the segmented CUDA posterior.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from relax.helpers.env_flags import parse_env_flag
from relax.helpers.oversampling import _relion_cuda_f32_tail_target

_RELION_FINE_ROTATION_EXECUTION_ORDER_ENV = (
    "RELAX_RELION_FINE_ROTATION_EXECUTION_ORDER"
)


_RELION_X_HALF_F32_FINE_POSTERIOR_ENV = "RELAX_RELION_X_HALF_F32_FINE_POSTERIOR"


@jax.jit
def _normalize_pass2_bucket_with_log_z(scores, log_z):
    """Normalize sparse candidate scores with a precomputed full-grid log-Z."""
    scores = jnp.where(jnp.isfinite(scores), scores, -jnp.inf)
    flat = scores.reshape(scores.shape[0], -1)
    best_log_score = jnp.max(flat, axis=1)
    has_finite_score = jnp.isfinite(best_log_score) & jnp.isfinite(log_z)
    safe_log_z = jnp.where(has_finite_score, log_z, 0.0)
    probs = jnp.exp(scores - safe_log_z[:, None, None])
    probs = jnp.where(has_finite_score[:, None, None] & jnp.isfinite(probs), probs, 0.0)
    best_argmax = jnp.where(has_finite_score, jnp.argmax(flat, axis=1), 0)
    max_posterior = jnp.exp(best_log_score - safe_log_z)
    max_posterior = jnp.where(has_finite_score & jnp.isfinite(max_posterior), max_posterior, 0.0)
    best_log_score = jnp.where(has_finite_score, best_log_score, -jnp.inf)
    return safe_log_z, probs, best_log_score, best_argmax, max_posterior


@partial(jax.jit, static_argnames=("adaptive_fraction", "keep_all"))
def _relion_f32_fine_posterior(
    scores,
    *,
    adaptive_fraction: float,
    normalization_sum_weight=None,
    keep_all: bool = False,
):
    """Build full and pruned fine probabilities with RELION GPU arithmetic.

    The reference GPU path shifts its float32 log weights so the maximum is
    50, applies ``expf``, sorts the raw weights in ascending order, and obtains
    both ``sum_weight`` and the lower-tail significance cutoff from a float32
    cumulative scan; the cut here has that definition with float64 cumulative
    sums (:func:`relax.cuda.kernels.relion_coarse_cut_f32`).  Surviving weights
    are divided by the full pre-pruning ``sum_weight``; they are intentionally
    not renormalized afterward.
    """

    scores_f32 = jnp.asarray(scores, dtype=jnp.float32)
    flat_scores = scores_f32.reshape(scores_f32.shape[0], -1)
    finite = jnp.isfinite(flat_scores)
    best = jnp.max(jnp.where(finite, flat_scores, -jnp.inf), axis=1)
    has_finite = jnp.isfinite(best)
    safe_best = jnp.where(has_finite, best, jnp.float32(0.0))
    exponent_add = jnp.float32(50.0) - safe_best
    use_native_cuda = False
    if jax.default_backend() == "gpu":
        from recovar import cuda_backproject

        from relax.cuda import kernels as em_cuda_kernels

        use_native_cuda = cuda_backproject.custom_cuda_requested()
    if use_native_cuda:
        # RELION computes ``50 - weights_max`` once in XFLOAT, then its CUDA
        # kernel evaluates ``expf(score + add)``.  Reassociating this as
        # ``exp(score - best + 50)`` changes hundreds of thousands of raw
        # weights at the iteration-2 case-22 boundary.
        finite_scores = jnp.where(finite, flat_scores, -jnp.inf)
        raw_weights = em_cuda_kernels.relion_exponentiate_batched_f32(
            finite_scores,
            exponent_add,
        )
        fine_sum_weight, cut_weight, _ = em_cuda_kernels.relion_coarse_cut_f32(
            raw_weights, adaptive_fraction=float(adaptive_fraction), max_significants=0
        )
    else:
        shifted = jnp.where(
            finite,
            flat_scores + exponent_add[:, None],
            -jnp.inf,
        )
        raw_weights = jnp.where(
            shifted < jnp.float32(-88.0),
            jnp.float32(0.0),
            jnp.exp(shifted),
        )
        raw_weights = jnp.where(
            finite & jnp.isfinite(raw_weights),
            raw_weights,
            jnp.float32(0.0),
        )
        # The CUDA cut's definition (float64 cumulative sums), by sort, for the CPU unit tests.
        sorted_weights = jnp.sort(raw_weights, axis=1)
        cumulative = jnp.cumsum(sorted_weights.astype(jnp.float64), axis=1)
        fine_sum_weight = cumulative[:, -1].astype(jnp.float32)
        cut_index = jax.vmap(lambda row, target: jnp.searchsorted(row, target, side="right"))(
            cumulative, _relion_cuda_f32_tail_target(fine_sum_weight, adaptive_fraction).astype(jnp.float64)
        )
        cut_weight = sorted_weights[jnp.arange(flat_scores.shape[0]), jnp.minimum(cut_index, cumulative.shape[1] - 1)]
    if normalization_sum_weight is None:
        sum_weight = fine_sum_weight
    else:
        # Gradient InitialModel with adaptive_oversampling==0 does not update
        # op.sum_weight in the symbolic fine pass.  The CUDA fine weights are
        # shifted by their own maximum, then divided directly by the numeric
        # float32 denominator retained from the coarse pass.
        sum_weight = jnp.asarray(normalization_sum_weight, dtype=jnp.float32)
    has_mass = has_finite & jnp.isfinite(sum_weight) & (sum_weight > jnp.float32(0.0))
    if keep_all:
        threshold = jnp.zeros_like(sum_weight)
        mask_flat = has_mass[:, None] & finite & (raw_weights > jnp.float32(0.0))
    else:
        threshold = cut_weight
        mask_flat = has_mass[:, None] & finite & (raw_weights >= threshold[:, None])
    safe_sum_weight = jnp.where(has_mass, sum_weight, jnp.float32(1.0))
    if use_native_cuda:
        normalized_weights = em_cuda_kernels.relion_divide_batched_f32(
            raw_weights,
            safe_sum_weight,
        )
    else:
        normalized_weights = raw_weights / safe_sum_weight[:, None]
    reconstruction_probs_flat = jnp.where(
        mask_flat,
        normalized_weights,
        jnp.float32(0.0),
    )
    n_significant = jnp.sum(mask_flat, axis=1).astype(jnp.int32)
    output_shape = scores_f32.shape
    return (
        normalized_weights.reshape(output_shape),
        reconstruction_probs_flat.reshape(output_shape),
        mask_flat.reshape(output_shape),
        n_significant,
        sum_weight,
        threshold,
    )



def _relion_f32_fine_reconstruction_probs(
    scores,
    *,
    adaptive_fraction: float,
    normalization_sum_weight=None,
    keep_all: bool = False,
):
    """Return the legacy pruned view of :func:`_relion_f32_fine_posterior`."""

    full = _relion_f32_fine_posterior(
        scores,
        adaptive_fraction=adaptive_fraction,
        normalization_sum_weight=normalization_sum_weight,
        keep_all=keep_all,
    )
    return full[1:]


def relion_x_half_f32_fine_posterior_enabled(*, default: bool = False) -> bool:
    """Return whether the RELION float32 fine posterior is enabled.

    Preserve the PR179 production default; enable this separately qualified
    arithmetic path explicitly for boundary experiments.
    """

    return parse_env_flag(_RELION_X_HALF_F32_FINE_POSTERIOR_ENV, default=default)


def _relion_fine_parent_execution_order_enabled(
    *,
    use_relion_f32_fine_posterior: bool,
) -> bool:
    """Keep exact fine-posterior diagnostics in RELION's candidate order."""

    return bool(use_relion_f32_fine_posterior) or parse_env_flag(
        _RELION_FINE_ROTATION_EXECUTION_ORDER_ENV,
        default=False,
    )


