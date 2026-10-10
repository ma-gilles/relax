"""Dense half-spectrum block scorers kept for test comparisons.

The dense EM engine that ran these per rotation block was removed on 2026-10-03.
Production keeps their shared GEMM owner
(:func:`relax.scoring.coarse_kernels._e_step_block_score_components`) and the windowed
Gaussian scorer the coarse GEMM path reuses
(:func:`relax.scoring.coarse_kernels._e_step_block_scores_windowed`); the residual and
normalized-CC wrappers and the per-block dispatcher below are test-only.
"""

from functools import partial

import jax
import jax.numpy as jnp

from relax.refinement.precision import DensePrecisionPolicy
from relax.scoring.coarse_kernels import _e_step_block_score_components, _e_step_block_scores_windowed


def _score_rotation_block(
    window_spec,
    *,
    shifted_score,
    batch_norm,
    score_weight,
    proj_half,
    proj_abs2_half,
    half_weights,
    n_images,
    n_trans,
    image_shape,
    volume_shape,
    score_mode: str,
    precision_policy: DensePrecisionPolicy,
):
    """Score one rotation block against the active Fourier-window spec."""

    proj_score = window_spec.score_values(proj_half)
    proj_abs2_score = window_spec.score_values(proj_abs2_half)
    proj_score, proj_abs2_score = precision_policy.cast_projection_scores(
        proj_score,
        proj_abs2_score,
    )
    weights = window_spec.score_values(half_weights)
    proj_weighted = proj_score * weights
    proj_abs2_weighted = proj_abs2_score * weights
    n_score = window_spec.n_score
    if score_mode == "normalized_cc":
        return _e_step_block_scores_windowed_normalized_cc(
            shifted_score,
            batch_norm,
            score_weight,
            proj_weighted,
            proj_abs2_weighted,
            n_images,
            n_trans,
            n_score,
            image_shape,
            volume_shape,
        )
    return _e_step_block_scores_windowed(
        shifted_score,
        batch_norm,
        score_weight,
        proj_weighted,
        proj_abs2_weighted,
        weights,
        n_images,
        n_trans,
        n_score,
        image_shape,
        volume_shape,
    )


@partial(jax.jit, static_argnums=(4, 5))
def _e_step_block_scores(
    shifted_half,
    ctf2_over_nv_half,
    proj_half_weighted,
    proj_abs2_half,
    n_images,
    n_trans,
):
    """E-step for one rotation block using half-spectrum GEMMs.

    The cross-term GEMM uses weighted projections (half_weights absorbed into
    projections, precomputed once per rotation block) to recover the full inner
    product from half-spectrum data:

        cross[i,r] = -2 Re(conj(shifted_half) @ proj_half_weighted.T)

    The norm-term similarly uses half-weighted |proj|^2.
    """
    cross, norms = _e_step_block_score_components(
        shifted_half,
        ctf2_over_nv_half,
        proj_half_weighted,
        proj_abs2_half,
        n_images,
        n_trans,
    )
    residuals = cross + norms[..., None]
    return -0.5 * residuals


@partial(jax.jit, static_argnums=(5, 6, 7, 8))
def _e_step_block_scores_normalized_cc(
    shifted_half,
    batch_norm,
    ctf2_over_nv_half,
    proj_half_weighted,
    proj_abs2_half,
    n_images,
    n_trans,
    image_shape,
    volume_shape,
):
    """RELION iter-1 normalized cross-correlation score."""
    del batch_norm, image_shape, volume_shape
    cross, norms = _e_step_block_score_components(
        shifted_half,
        ctf2_over_nv_half,
        proj_half_weighted,
        proj_abs2_half,
        n_images,
        n_trans,
    )
    denom = jnp.sqrt(jnp.maximum(norms, jnp.asarray(1e-30, dtype=norms.dtype)))
    return (-0.5 * cross) / denom[..., None]


@partial(jax.jit, static_argnums=(5, 6, 7, 8, 9))
def _e_step_block_scores_windowed_normalized_cc(
    shifted_windowed,
    batch_norm,
    ctf2_over_nv_windowed,
    proj_windowed_weighted,
    proj_abs2_windowed,
    n_images,
    n_trans,
    n_windowed,
    image_shape,
    volume_shape,
):
    """Windowed RELION iter-1 normalized cross-correlation score."""
    del batch_norm, n_windowed, image_shape, volume_shape
    cross, norms = _e_step_block_score_components(
        shifted_windowed,
        ctf2_over_nv_windowed,
        proj_windowed_weighted,
        proj_abs2_windowed,
        n_images,
        n_trans,
    )
    denom = jnp.sqrt(jnp.maximum(norms, jnp.asarray(1e-30, dtype=norms.dtype)))
    return (-0.5 * cross) / denom[..., None]
