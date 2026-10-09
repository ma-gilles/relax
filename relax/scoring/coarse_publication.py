"""Shared posterior and source-order publication for grouped K=1 coarse work."""

from functools import partial

import jax
import jax.numpy as jnp

from relax.helpers.oversampling import relion_cuda_f32_coarse_log_weights, relion_cuda_f32_coarse_posterior


@partial(jax.jit, static_argnames=("adaptive_fraction", "max_significants", "tie_score_ulps"))
def posterior_statistics(values, raw_max, source_blocks, *, adaptive_fraction, max_significants, tie_score_ulps):
    probabilities, mask, count, cutoff, total, threshold = relion_cuda_f32_coarse_posterior(
        values,
        adaptive_fraction=adaptive_fraction,
        max_significants=max_significants,
        tie_score_ulps=tie_score_ulps,
        min_diff2_offsets=-raw_max,
    )
    winner = jnp.argmax(probabilities, axis=1)
    best = jnp.argmax(values, axis=1)
    if source_blocks is not None:
        per_slot = values.shape[1] // source_blocks.shape[1]

        def global_pose(local):
            block = jnp.take_along_axis(source_blocks, (local // per_slot)[:, None], axis=1)[:, 0]
            return block * per_slot + local % per_slot

        winner = global_pose(winner)
        best = global_pose(best)
    return dict(
        mask=mask,
        winner=winner,
        best_pose=best,
        best_score=jnp.max(values, axis=1),
        pmax=jnp.max(probabilities, axis=1),
        n_significant=count,
        cutoff_count=cutoff,
        sum_weight=total,
        threshold=threshold,
    )


@partial(
    jax.jit,
    static_argnames=("exact_weight_order", "n_trans", "adaptive_fraction", "max_significants", "tie_score_ulps"),
)
def coarse_support_posterior(
    values,
    raw_max,
    rotation_log_prior,
    translation_log_prior,
    *,
    exact_weight_order,
    n_trans,
    adaptive_fraction,
    max_significants,
    tie_score_ulps,
):
    """One image batch's RELION float32 coarse posterior and what pass 1 publishes from it, as one program.

    ``values`` is ``[B, K * R * T]``, class-major. With ``exact_weight_order`` (K=1) it holds the
    pre-prior scores and the program forms RELION's log weights ``pdf_orientation + pdf_offset +
    min_diff2 - diff2`` from ``rotation_log_prior`` (``[R]`` or ``[B, R]``) and ``translation_log_prior``
    (``[T]`` or ``[B, T]``) first (:func:`relax.helpers.oversampling.relion_cuda_f32_coarse_log_weights`);
    otherwise it holds the with-prior scores and ``raw_max`` (``[B]``) restores RELION's ``min_diff2``
    frame. Returns the posterior's ``weights``, ``mask``, ``n_significant``, ``cutoff_count``,
    ``sum_weight`` and ``significant_weight``
    (:func:`relax.helpers.oversampling.relion_cuda_f32_coarse_posterior`), each row's ``winner``
    (argmax of the weights, int32) and ``pmax``, and ``rotation_mask`` ``[B, K * R]``, the rotations
    with a significant translation.
    """

    batch = values.shape[0]
    if exact_weight_order:
        values = relion_cuda_f32_coarse_log_weights(
            values.reshape(batch, -1, n_trans), rotation_log_prior, translation_log_prior
        ).reshape(batch, -1)
        # The RELION-order log weights already carry min_diff2.
        offsets = None
    else:
        offsets = -raw_max
    weights, mask, count, cutoff, total, significant = relion_cuda_f32_coarse_posterior(
        values,
        adaptive_fraction=adaptive_fraction,
        max_significants=max_significants,
        tie_score_ulps=tie_score_ulps,
        min_diff2_offsets=offsets,
    )
    return dict(
        weights=weights,
        mask=mask,
        n_significant=count,
        cutoff_count=cutoff,
        sum_weight=total,
        significant_weight=significant,
        winner=jnp.argmax(weights, axis=1).astype(jnp.int32),
        pmax=jnp.max(weights, axis=1),
        rotation_mask=jnp.any(mask.reshape(batch, -1, n_trans), axis=2),
    )
