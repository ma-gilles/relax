"""Shared posterior and source-order publication for grouped K=1 coarse work."""

from functools import partial

import jax
import jax.numpy as jnp

from relax.helpers.oversampling import relion_cuda_f32_coarse_posterior


@partial(jax.jit, static_argnames=("adaptive_fraction", "max_significants", "tie_score_ulps"))
def _posterior_statistics(values, raw_max, source_blocks, *, adaptive_fraction, max_significants, tie_score_ulps):
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


def coarse_square_layout_metadata(coarse_gaussian_square_layout, *, stable_fourier_window_shapes):
    """Describe logical and physical score windows without touching device arrays."""
    return {
        "stable_fourier_window_shapes_requested": bool(
            stable_fourier_window_shapes
        ),
        "stable_fourier_window_shapes_effective": bool(
            stable_fourier_window_shapes
            and coarse_gaussian_square_layout.physical_current_size
            != coarse_gaussian_square_layout.logical_current_size
        ),
        "logical_current_size": int(
            coarse_gaussian_square_layout.logical_current_size
        ),
        "physical_current_size": int(
            coarse_gaussian_square_layout.physical_current_size
        ),
        "logical_square_pixels": int(
            coarse_gaussian_square_layout.logical_square_count
        ),
        "physical_square_pixels": int(
            coarse_gaussian_square_layout.physical_square_count
        ),
        "executed_square_pixels": int(
            coarse_gaussian_square_layout.logical_square_count
            if stable_fourier_window_shapes
            else coarse_gaussian_square_layout.physical_square_count
        ),
        "logical_issue_stream_is_prefix": True,
        "physical_tail_zero_weighted": True,
        "physical_tail_skipped_by_runtime_count": bool(
            stable_fourier_window_shapes
            and coarse_gaussian_square_layout.physical_square_count
            != coarse_gaussian_square_layout.logical_square_count
        ),
    }
