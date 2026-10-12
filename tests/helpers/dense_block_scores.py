"""Dense half-spectrum block scorers kept for test comparisons.

The dense EM engine that ran these per rotation block was removed on 2026-10-03, and the coarse pass scores with
``relax.scoring.coarse_kernels.relion_coarse_gaussian_gemm_scores_jit``; the shared GEMM owner
(``_e_step_block_score_components``), the windowed Gaussian scorer, the residual and normalized-CC wrappers and the
per-block dispatcher below are test-only.
"""

from functools import partial

import jax
import jax.numpy as jnp

from relax.numerics.precision import DensePrecisionPolicy


def _e_step_block_score_components(
    shifted,
    ctf2_over_nv,
    proj_weighted,
    proj_abs2,
    n_images,
    n_trans,
):
    """Return the cross and model-energy GEMM components of one rotation block.

    ``cross[i, r, t] = -2 Re(conj(shifted[i, t]) . proj_weighted[r])`` recovers
    the full inner product from half-spectrum pixels (the half weights are
    absorbed into the projections once per block), and
    ``norms[i, r] = ctf2_over_nv[i] . proj_abs2[r]`` is the model energy. Every
    dense scorer (residual, windowed and normalized-CC) builds its score from
    these two HIGHEST-precision GEMMs. The cross term is one real GEMM over
    ``[Re, Im]``-packed operands, ``Re(conj(s) . p) = s_re . p_re + s_im . p_im``,
    so the unused imaginary half of a complex product is never formed.
    """

    rot_block_size = proj_weighted.shape[0]
    shifted_packed = jnp.concatenate([shifted.real, shifted.imag], axis=-1)
    proj_packed = jnp.concatenate([proj_weighted.real, proj_weighted.imag], axis=-1)
    cross = -2.0 * jnp.matmul(
        shifted_packed,
        proj_packed.T,
        precision=jax.lax.Precision.HIGHEST,
    )
    cross = cross.reshape(n_images, n_trans, rot_block_size)
    cross = cross.swapaxes(1, 2)
    norms = jnp.matmul(
        ctf2_over_nv,
        proj_abs2.T,
        precision=jax.lax.Precision.HIGHEST,
    )
    return cross, norms


@partial(jax.jit, static_argnums=(6, 7, 8, 9, 10))
def _e_step_block_scores_windowed(
    shifted_windowed,
    batch_norm,
    ctf2_over_nv_windowed,
    proj_windowed_weighted,
    proj_abs2_windowed,
    half_weights_windowed,
    n_images,
    n_trans,
    n_windowed,
    image_shape,
    volume_shape,
):
    """E-step for one rotation block using windowed half-spectrum GEMMs."""

    del batch_norm, half_weights_windowed, n_windowed, image_shape, volume_shape
    cross, norms = _e_step_block_score_components(
        shifted_windowed,
        ctf2_over_nv_windowed,
        proj_windowed_weighted,
        proj_abs2_windowed,
        n_images,
        n_trans,
    )
    residuals = cross + norms[..., None]
    return -0.5 * residuals


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
    if not precision_policy.use_float64_scoring:
        proj_score = proj_score.astype(jnp.complex64)
        proj_abs2_score = proj_abs2_score.astype(jnp.float32)
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
