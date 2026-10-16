"""Dataset-facing dense PPCA refinement iteration.

This module is the bridge between the current K-class dense EM infrastructure
and PPCA-specific score/moment algebra. It deliberately reuses the dense
single-volume preprocessing, Fourier-window, CTF/noise, translation, and
projection helpers rather than recreating the stale PPCA branch engine layout.
The dataset blocks it iterates (preprocessed images, augmented projections, pose
priors) are built by ``dense_blocks``.
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from recovar.ppca.augmented_mstep import augmented_ppca_mstep_objective, solve_augmented_ppca_mstep
from recovar.ppca.pose_accumulators import AugmentedPPCAStats
from recovar.ppca.triangular import tri_size as _tri_size

from relax.ppca_refinement.config import (
    GeometryConfig,
    PoseSelectionConfig,
    ScheduleConfig,
    ScoringConfig,
    SparsePass2Config,
)
from relax.ppca_refinement.dense_blocks import coerce_augmented_half_volumes, iter_dense_ppca_dataset_block_groups
from relax.ppca_refinement.diagnostics import build_iteration_diagnostics, resolve_image_scale_range
from relax.ppca_refinement.engine import (
    DensePPCAFusedEMResult,
    DenseScoreStats,
    accumulate_pose_ppca_block_cached,
    add_compensated_blocks,
    compensated_add,
    dense_pose_ppca_score_stats_blocked,
    dense_pose_ppca_score_tensor_stats_blocked,
    dense_pose_ppca_score_with_moments_blocked,
    dense_pose_ppca_score_with_moments_factor_once,
    enforce_augmented_x0,
    fused_dense_pose_ppca_block,
    pose_invariant_score_offset,
)
from relax.ppca_refinement.mean_regularization import (
    MeanRegularizationConfig,
    resolve_mean_precision,
)
from relax.ppca_refinement.pose_selection import (
    merge_top_p_pose_scores,
    top_p_from_score_block,
    top_pose_candidate_count,
)
from relax.ppca_refinement.postprocess import PostprocessConfig, postprocess_ppca_half_volumes
from relax.ppca_refinement.residual_statistics import full_float32, residual_statistics_precision
from relax.ppca_refinement.state import PoseMarginalPPCAEMState


def _latent_covariance_trace_from_packed_moments(G_tri, alpha, basis_size: int):
    """Read only latent diagonal entries of the packed second moment."""
    diagonal = np.asarray(
        [i * basis_size - i * (i - 1) // 2 for i in range(1, basis_size)], dtype=np.int32
    )
    return jnp.sum(jnp.take(G_tri, diagonal, axis=-1), axis=-1) - jnp.sum(alpha[..., 1:] ** 2, axis=-1)


def _top_pose_from_score_stats(score_stats: DenseScoreStats, *, rotation_offset: int):
    return (
        jnp.asarray(score_stats.best_log_score_per_image, dtype=jnp.float32)[:, None],
        (
            jnp.asarray(score_stats.best_rotation_idx, dtype=jnp.int32)
            + jnp.asarray(int(rotation_offset), dtype=jnp.int32)
        )[:, None],
        jnp.asarray(score_stats.best_translation_idx, dtype=jnp.int32)[:, None],
    )


def _centered_score_partition(block_scores: list[jax.Array]) -> tuple[jax.Array, jax.Array]:
    """Normalize pose scores in float32 without rounding an absolute logZ.

    Each score block is already retained by the cached-moments path. Reduce
    within those blocks, keeping memory bounded by the computational schedule.
    See the pose-marginal normalization in ``docs/math/vdam_ppca_algorithm.md``.
    """
    center = jnp.max(jnp.stack([
        jnp.max(score, axis=(1, 2)) for score in block_scores
    ]), axis=0)
    partition = jnp.zeros_like(center, dtype=jnp.float32)
    for score in block_scores:
        partition = partition + jnp.sum(
            jnp.exp(score - center[:, None, None]), axis=(1, 2)
        )
    return center, jnp.log(partition)


@residual_statistics_precision
def accumulate_dense_ppca_statistics(
    experiment_dataset,
    mu,
    W=None,
    *,
    noise_variance,
    rotations,
    translations,
    geometry: GeometryConfig | None = None,
    schedule: ScheduleConfig | None = None,
    scoring: ScoringConfig | None = None,
    sparse_pass2: SparsePass2Config | None = None,
    mean_reg: MeanRegularizationConfig | None = None,
    disc_type: str = "linear_interp",
    image_indices: np.ndarray | None = None,
    rotation_log_prior: np.ndarray | None = None,
    translation_log_prior: np.ndarray | None = None,
    rotation_translation_mask: np.ndarray | None = None,
    enforce_x0: bool = True,
    collect_residuals: bool = False,
    factor_once_score: bool = False,
    skip_empty_pose_blocks: bool = False,
    top_pose_count: int = 1,
    pose_selection: PoseSelectionConfig | None = None,
) -> AugmentedPPCAStats:
    """Accumulate unregularized PPCA statistics without solving or postprocessing.

    Model geometry determines sizes; no artificial prior arrays are needed.
    """
    geometry = geometry if geometry is not None else GeometryConfig()
    schedule = schedule if schedule is not None else ScheduleConfig()
    scoring = scoring if scoring is not None else ScoringConfig()
    sparse_pass2_cfg = sparse_pass2 if sparse_pass2 is not None else SparsePass2Config()
    mean_reg = mean_reg if mean_reg is not None else MeanRegularizationConfig()
    # Local aliases keep the rest of the body readable; configs are the public API.
    image_scale_corrections = scoring.image_scale_corrections
    sparse_pass2_enabled = sparse_pass2_cfg.enabled
    sparse_pass2_log_threshold = sparse_pass2_cfg.log_threshold
    pose_selection = (
        pose_selection
        if pose_selection is not None
        else PoseSelectionConfig(top_p_poses=max(1, int(top_pose_count)))
    )
    top_pose_count = int(pose_selection.top_p_poses)

    if collect_residuals and sparse_pass2_cfg.enabled:
        raise ValueError("Residual statistics require all declared posterior support (disable sparse culling)")
    block_groups = iter_dense_ppca_dataset_block_groups(
        experiment_dataset,
        mu,
        W,
        noise_variance,
        rotations,
        translations,
        disc_type=disc_type,
        image_batch_size=schedule.image_batch_size,
        rotation_block_size=schedule.rotation_block_size,
        current_size=geometry.current_size,
        q=geometry.q,
        volume_domain=geometry.volume_domain,
        image_indices=image_indices,
        rotation_log_prior=rotation_log_prior,
        translation_log_prior=translation_log_prior,
        rotation_translation_mask=rotation_translation_mask,
        image_scale_corrections=scoring.image_scale_corrections,
        class_log_prior=scoring.class_log_prior,
        score_with_masked_images=scoring.score_with_masked_images,
        relion_unit_half_weights=scoring.relion_unit_half_weights,
        square_window=scoring.square_window,
        full_real_observation=scoring.full_real_observation,
        relion_texture_interp=scoring.relion_texture_interp,
        skip_empty_pose_blocks=skip_empty_pose_blocks,
        collect_observation=collect_residuals,
    )
    augmented, q_resolved = coerce_augmented_half_volumes(
        mu,
        W,
        volume_shape=experiment_dataset.volume_shape,
        q=geometry.q,
        volume_domain=geometry.volume_domain,
    )
    half_size = augmented.shape[1]
    P = q_resolved + 1
    tri = _tri_size(P)
    image_shape = tuple(int(x) for x in experiment_dataset.image_shape)
    volume_shape = tuple(int(x) for x in experiment_dataset.volume_shape)
    image_scale_min, image_scale_max = resolve_image_scale_range(image_scale_corrections, image_indices)
    rhs_volume = jnp.zeros((P, half_size), dtype=jnp.complex64)
    lhs_tri_volume = jnp.zeros((tri, half_size), dtype=jnp.float32)
    residual_volume = jnp.zeros_like(rhs_volume) if collect_residuals else None
    # Blocks are backprojected into zero volumes and Kahan-added (compensated_add).
    volume_compensation = (jnp.zeros_like(rhs_volume), jnp.zeros_like(lhs_tri_volume))
    residual_compensation = jnp.zeros_like(rhs_volume) if collect_residuals else None
    residual_power = jnp.zeros(int(np.prod(image_shape[:-1])) * (image_shape[-1] // 2 + 1), jnp.float32)
    embedding_batches = []
    offset_second_sum = jnp.float32(0)
    rotation_mass = jnp.zeros(len(rotations), jnp.float32)
    latent_covariance_trace_sum = jnp.float32(0)
    pose_entropy_sum = jnp.float32(0)
    log_likelihood = 0.0
    n_images = 0
    pmax_values = []
    nsig_values = []
    best_rotations = []
    best_translations = []
    top_rotation_values = []
    top_translation_values = []
    top_log_score_values = []
    top_posterior_values = []
    sparse_pass2_total_blocks = 0
    sparse_pass2_skipped_blocks = 0
    sparse_pass2_omitted_mass_upper_sum = 0.0
    sparse_pass2_omitted_mass_upper_max = 0.0
    sparse_pass2_omitted_mass_upper_image_count = 0
    score_fourier_size = None
    recon_fourier_size = None
    postprocess_bandlimit_max_r = None

    for group in block_groups:
        if not group:
            continue
        batch_size = int(group[0].Y1.shape[0])
        if collect_residuals:
            residual_power = residual_power + group[0].observation_power
            batch_embedding = jnp.zeros((batch_size, q_resolved), jnp.float32)
        if postprocess_bandlimit_max_r is None and bool(group[0].use_recon_window):
            postprocess_bandlimit_max_r = group[0].backprojection_max_r
        if score_fourier_size is None:
            score_fourier_size = int(group[0].Y1.shape[-1])
            recon_fourier_size = (
                int(group[0].Y1_recon.shape[-1]) if group[0].Y1_recon is not None else score_fourier_size
            )
        block_logZ = []
        block_score_stats = []
        block_pose_counts = []
        block_top_scores = []
        block_top_rotations = []
        block_top_translations = []
        # Cached-moments fast path: when sparse-pass2 cannot cull any blocks
        # (disabled, or only one block in the group), compute moments (α, G_tri)
        # in pass 1 so pass 2 can skip its duplicate `_per_pose_stats_block`
        # einsum — the dominant cost in the early-iter regime where every
        # block must be backprojected anyway.
        cache_moments = (not sparse_pass2_enabled) or len(group) == 1
        block_alphas: list = []
        block_G_tris: list = []
        block_scores: list = []
        for block in group:
            if cache_moments:
                score_function = (
                    dense_pose_ppca_score_with_moments_factor_once
                    if factor_once_score else dense_pose_ppca_score_with_moments_blocked
                )
                full = score_function(
                    block.Y1,
                    block.proj_aug,
                    block.ctf2_over_noise,
                    block.y_norm,
                    block.pose_log_prior,
                )
                score_stats = DenseScoreStats(
                    logZ=full.logZ,
                    best_log_score_per_image=full.best_log_score_per_image,
                    best_rotation_idx=full.best_rotation_idx,
                    best_translation_idx=full.best_translation_idx,
                )
                block_alphas.append(full.alpha)
                block_G_tris.append(full.G_tri)
                block_scores.append(full.score)
                top_scores, top_rot, top_trans = top_p_from_score_block(
                    full.score,
                    rotation_offset=int(block.rotation_start),
                    candidate_count=top_pose_candidate_count(
                        pose_selection,
                        int(full.score.shape[1]) * int(full.score.shape[2]),
                    ),
                )
            else:
                if top_pose_count > 1:
                    score_tensor_stats = dense_pose_ppca_score_tensor_stats_blocked(
                        block.Y1,
                        block.proj_aug,
                        block.ctf2_over_noise,
                        block.y_norm,
                        block.pose_log_prior,
                    )
                    score_stats = DenseScoreStats(
                        logZ=score_tensor_stats.logZ,
                        best_log_score_per_image=score_tensor_stats.best_log_score_per_image,
                        best_rotation_idx=score_tensor_stats.best_rotation_idx,
                        best_translation_idx=score_tensor_stats.best_translation_idx,
                    )
                    top_scores, top_rot, top_trans = top_p_from_score_block(
                        score_tensor_stats.score,
                        rotation_offset=int(block.rotation_start),
                        candidate_count=top_pose_candidate_count(
                            pose_selection,
                            int(score_tensor_stats.score.shape[1]) * int(score_tensor_stats.score.shape[2]),
                        ),
                    )
                else:
                    score_stats = dense_pose_ppca_score_stats_blocked(
                        block.Y1,
                        block.proj_aug,
                        block.ctf2_over_noise,
                        block.y_norm,
                        block.pose_log_prior,
                    )
                    top_scores, top_rot, top_trans = _top_pose_from_score_stats(
                        score_stats,
                        rotation_offset=int(block.rotation_start),
                    )
                block_alphas.append(None)
                block_G_tris.append(None)
                block_scores.append(None)
            block_score_stats.append(score_stats)
            block_logZ.append(score_stats.logZ)
            block_pose_counts.append(int(block.Y1.shape[1]) * int(block.rotations.shape[0]))
            block_top_scores.append(top_scores)
            block_top_rotations.append(top_rot)
            block_top_translations.append(top_trans)
        if cache_moments:
            # Keep posterior normalization near zero. Adding absolute block
            # log-partitions at scores around -1700 loses an entire float32
            # ULP when the computational rotation block width changes.
            # Scores are already retained for the cached-moments pass, and
            # each reduction stays within one block's bounded score tensor.
            score_center, centered_logZ = _centered_score_partition(block_scores)
            # Cached scores exclude the pose-invariant image constant; the
            # absolute log-partition and top scores add it back.
            score_offset = pose_invariant_score_offset(group[0].y_norm)
            logZ = score_center + centered_logZ + score_offset
            top_scores_for_posterior = [
                scores - score_center[:, None] for scores in block_top_scores
            ]
        else:
            logZ = block_logZ[0]
            for next_logZ in block_logZ[1:]:
                logZ = jnp.logaddexp(logZ, next_logZ)
            centered_logZ = logZ
            top_scores_for_posterior = block_top_scores

        log_likelihood += float(jnp.sum(logZ))
        n_images += batch_size
        batch_nsig = jnp.zeros((batch_size,), dtype=jnp.int32)
        top_selection = merge_top_p_pose_scores(
            top_scores_for_posterior,
            block_top_rotations,
            block_top_translations,
            centered_logZ,
            rotations=np.asarray(rotations, dtype=np.float32),
            translations=np.asarray(translations, dtype=np.float32),
            config=pose_selection,
        )
        top_scores = jnp.asarray(top_selection.log_score, dtype=jnp.float32)
        if cache_moments:
            top_scores = top_scores + score_center[:, None] + score_offset[:, None]
        top_rotations = jnp.asarray(top_selection.rotation_idx, dtype=jnp.int32)
        top_translations = jnp.asarray(top_selection.translation_idx, dtype=jnp.int32)
        top_posteriors = jnp.asarray(top_selection.posterior, dtype=jnp.float32)
        batch_pmax = top_posteriors[:, 0].astype(jnp.float32)
        batch_best_rotation = top_rotations[:, 0].astype(jnp.int32)
        batch_best_translation = top_translations[:, 0].astype(jnp.int32)
        retained_group = tuple(group)
        if sparse_pass2_enabled and len(group) > 1:
            block_best_matrix = jnp.stack(
                [stats.best_log_score_per_image for stats in block_score_stats],
                axis=0,
            )
            block_log_pose_counts = jnp.log(jnp.asarray(block_pose_counts, dtype=logZ.dtype))[:, None]
            finite_log_z = jnp.isfinite(logZ)
            log_omitted_mass_upper = jnp.where(
                finite_log_z[None, :],
                block_log_pose_counts + block_best_matrix.astype(logZ.dtype) - logZ[None, :].astype(logZ.dtype),
                jnp.inf,
            )
            skip_candidate = log_omitted_mass_upper < float(sparse_pass2_log_threshold)
            skip_pass2_block = np.asarray(jnp.all(skip_candidate, axis=1), dtype=bool)
            if np.all(skip_pass2_block):
                best_block_idx = int(np.argmax(np.asarray(jnp.max(block_best_matrix, axis=1))))
                skip_pass2_block[best_block_idx] = False
            sparse_pass2_total_blocks += len(group)
            sparse_pass2_skipped_blocks += int(skip_pass2_block.sum())
            if np.any(skip_pass2_block):
                skipped_mass_upper = jnp.sum(
                    jnp.where(
                        jnp.asarray(skip_pass2_block)[:, None],
                        jnp.exp(jnp.minimum(log_omitted_mass_upper, 50.0)),
                        0.0,
                    ),
                    axis=0,
                )
                skipped_mass_upper_np = np.asarray(skipped_mass_upper, dtype=np.float64)
                sparse_pass2_omitted_mass_upper_sum += float(np.sum(skipped_mass_upper_np))
                sparse_pass2_omitted_mass_upper_max = max(
                    sparse_pass2_omitted_mass_upper_max,
                    float(np.max(skipped_mass_upper_np)),
                )
                sparse_pass2_omitted_mass_upper_image_count += batch_size
                retained_group = tuple(block for block, skip in zip(group, skip_pass2_block, strict=True) if not skip)

        retained_block_ids = {id(b) for b in retained_group}
        for block_idx, block in enumerate(group):
            if id(block) not in retained_block_ids:
                continue
            if collect_residuals:
                from relax.fourier.half_spectrum import make_half_image_weights
                from relax.ppca_refinement.residual_statistics import residual_image_statistics
                from relax.projection.adjoint import batch_adjoint_slice_volume_maybe_windowed

                residual_images, correction, embedding = residual_image_statistics(
                    block_scores[block_idx] - score_center[:, None, None],
                    block_alphas[block_idx],
                    block_G_tris[block_idx],
                    centered_logZ,
                    block.Y1,
                    block.ctf2_over_noise,
                    block.proj_aug,
                )
                indices = block.score_window_indices
                weights = make_half_image_weights(image_shape)
                weights = weights if indices is None else weights[indices]
                # The half-image adjoint supplies conjugate scatters itself.
                residual_block = batch_adjoint_slice_volume_maybe_windowed(
                    residual_images / weights[None, None, :],
                    indices,
                    block.rotations,
                    jnp.zeros_like(residual_volume),
                    image_shape,
                    volume_shape,
                    disc_type,
                    True,
                    True,
                    use_window=indices is not None,
                    max_r=block.backprojection_max_r,
                )
                residual_volume, residual_compensation = compensated_add(
                    residual_volume, residual_compensation, residual_block
                )
                nv = jnp.broadcast_to(block.coefficient_noise, (residual_power.size,))
                if indices is None:
                    residual_power = residual_power + correction * nv / weights
                else:
                    residual_power = residual_power.at[indices].add(correction * nv[indices] / weights)
                batch_embedding = batch_embedding + embedding
                gamma = jnp.exp(block_scores[block_idx] - score_center[:, None, None] - centered_logZ[:, None, None])
                latent_covariance_trace = _latent_covariance_trace_from_packed_moments(
                    block_G_tris[block_idx], block_alphas[block_idx], P
                )
                latent_covariance_trace_sum += jnp.sum(gamma * latent_covariance_trace)
                centered_score = block_scores[block_idx] - score_center[:, None, None] - centered_logZ[:, None, None]
                pose_entropy_sum -= jnp.sum(jnp.where(gamma > 0, gamma * centered_score, 0))
                start = block.rotation_start
                rotation_mass = rotation_mass.at[start : start + gamma.shape[2]].add(jnp.sum(gamma, axis=(0, 1)))
                shift_squared = jnp.sum(jnp.asarray(translations, jnp.float32) ** 2, axis=-1)
                offset_second_sum = offset_second_sum + jnp.sum(gamma * shift_squared[None, :, None])
            if cache_moments and block_scores[block_idx] is not None:
                # Fast path: pass-1 already computed score + (α, G_tri). Skip
                # the duplicate `_per_pose_stats_block` einsum that
                # `fused_dense_pose_ppca_block` would otherwise redo.
                rhs_block, lhs_block, n_sig_block, _pmax_block = accumulate_pose_ppca_block_cached(
                    block_scores[block_idx] - score_center[:, None, None],
                    block_alphas[block_idx],
                    block_G_tris[block_idx],
                    centered_logZ,
                    block.Y1_recon if block.Y1_recon is not None else block.Y1,
                    block.ctf2_over_noise_recon if block.ctf2_over_noise_recon is not None else block.ctf2_over_noise,
                    block.rotations,
                    image_shape,
                    volume_shape,
                    jnp.zeros_like(rhs_volume),
                    jnp.zeros_like(lhs_tri_volume),
                    disc_type_backproject=disc_type,
                    recon_window_indices=block.recon_window_indices,
                    use_recon_window=block.use_recon_window,
                    backprojection_max_r=block.backprojection_max_r,
                )
                batch_nsig = batch_nsig + n_sig_block
            else:
                rhs_block, lhs_block, posterior = fused_dense_pose_ppca_block(
                    block.Y1,
                    block.proj_aug,
                    block.ctf2_over_noise,
                    block.y_norm,
                    block.rotations,
                    image_shape,
                    volume_shape,
                    jnp.zeros_like(rhs_volume),
                    jnp.zeros_like(lhs_tri_volume),
                    block.pose_log_prior,
                    Y1_recon=block.Y1_recon,
                    ctf2_over_noise_recon=block.ctf2_over_noise_recon,
                    normalization_logZ=logZ,
                    disc_type_backproject=disc_type,
                    recon_window_indices=block.recon_window_indices,
                    use_recon_window=block.use_recon_window,
                    backprojection_max_r=block.backprojection_max_r,
                )
                batch_nsig = batch_nsig + posterior.n_significant_per_image
            (rhs_volume, lhs_tri_volume), volume_compensation = add_compensated_blocks(
                (rhs_volume, lhs_tri_volume), volume_compensation, (rhs_block, lhs_block)
            )
        if collect_residuals:
            embedding_batches.append(batch_embedding)
        pmax_values.append(batch_pmax)
        nsig_values.append(batch_nsig)
        best_rotations.append(batch_best_rotation)
        best_translations.append(batch_best_translation)
        top_rotation_values.append(top_rotations)
        top_translation_values.append(top_translations)
        top_log_score_values.append(top_scores)
        top_posterior_values.append(top_posteriors)

    if enforce_x0:
        rhs_volume = enforce_augmented_x0(rhs_volume, volume_shape)
        lhs_tri_volume = enforce_augmented_x0(lhs_tri_volume.astype(jnp.complex64), volume_shape).real.astype(
            jnp.float32
        )

    full_half_fourier_size = int(image_shape[0] * (image_shape[1] // 2 + 1))
    diagnostics = build_iteration_diagnostics(
        pmax_values=pmax_values,
        nsig_values=nsig_values,
        best_rotations=best_rotations,
        best_translations=best_translations,
        top_rotations=top_rotation_values,
        top_rotation_ids=top_rotation_values,
        top_translations=top_translation_values,
        top_log_scores=top_log_score_values,
        top_posteriors=top_posterior_values,
        log_likelihood=log_likelihood,
        n_images=n_images,
        mean_reg=mean_reg,
        image_scale_min=image_scale_min,
        image_scale_max=image_scale_max,
        image_scale_corrections=image_scale_corrections,
        extras={
            "sparse_pass2_enabled": bool(sparse_pass2_enabled),
            "sparse_pass2_log_threshold": float(sparse_pass2_log_threshold),
            "sparse_pass2_total_blocks": int(sparse_pass2_total_blocks),
            "sparse_pass2_skipped_blocks": int(sparse_pass2_skipped_blocks),
            "sparse_pass2_skipped_fraction": (
                float(sparse_pass2_skipped_blocks / sparse_pass2_total_blocks) if sparse_pass2_total_blocks else 0.0
            ),
            "sparse_pass2_omitted_mass_upper_sum": float(sparse_pass2_omitted_mass_upper_sum),
            "sparse_pass2_omitted_mass_upper_max": float(sparse_pass2_omitted_mass_upper_max),
            "sparse_pass2_omitted_mass_upper_mean": (
                float(sparse_pass2_omitted_mass_upper_sum / sparse_pass2_omitted_mass_upper_image_count)
                if sparse_pass2_omitted_mass_upper_image_count
                else 0.0
            ),
            "score_fourier_size": int(score_fourier_size) if score_fourier_size is not None else 0,
            "recon_fourier_size": int(recon_fourier_size) if recon_fourier_size is not None else 0,
            "full_half_fourier_size": full_half_fourier_size,
            "uses_fourier_window": bool(
                score_fourier_size is not None and int(score_fourier_size) < full_half_fourier_size
            ),
            "top_p_poses": int(pose_selection.top_p_poses),
            "top_pose_max_log_score_gap": float(pose_selection.top_pose_max_log_score_gap),
            "top_pose_min_angle_deg": float(pose_selection.top_pose_min_angle_deg),
            "top_pose_min_translation_px": float(pose_selection.top_pose_min_translation_px),
        },
    )
    residual_num = residual_den = original_ids = None
    if collect_residuals:
        from relax.fourier.half_spectrum import make_half_image_weights, make_shell_indices_half

        weights = make_half_image_weights(image_shape)
        shells = make_shell_indices_half(image_shape)
        n_shells = int(np.max(np.asarray(shells))) + 1
        residual_num = jnp.zeros(n_shells, jnp.float32).at[shells].add(weights * residual_power)
        residual_den = jnp.zeros(n_shells, jnp.float32).at[shells].add(weights * n_images)
        local_ids = np.arange(experiment_dataset.n_images) if image_indices is None else np.asarray(image_indices)
        original_ids = experiment_dataset.original_image_indices_from_local(local_ids)
    stats = AugmentedPPCAStats(
        rhs=jnp.swapaxes(rhs_volume, 0, 1),
        lhs_tri=jnp.swapaxes(lhs_tri_volume, 0, 1),
        log_likelihood=log_likelihood,
        n_images=n_images,
        residual_gradient=None if residual_volume is None else residual_volume.T,
        residual_num=residual_num,
        residual_den=residual_den,
        embeddings=jnp.concatenate(embedding_batches) if collect_residuals else None,
        original_image_ids=original_ids,
        diagnostics=diagnostics,
    )
    diagnostics["postprocess_bandlimit_max_r"] = postprocess_bandlimit_max_r
    if collect_residuals:
        diagnostics["offset_second_sum_px2"] = float(offset_second_sum)
        diagnostics["rotation_mass"] = np.asarray(rotation_mass)
        diagnostics["latent_covariance_trace_mean"] = float(latent_covariance_trace_sum / n_images)
        diagnostics["pose_entropy_mean"] = float(pose_entropy_sum / n_images)
    return stats


class DensePPCAEmbeddings(NamedTuple):
    """Final pose-marginal coordinates and stable original particle IDs."""

    embeddings: jax.Array
    original_image_ids: np.ndarray
    n_images: int


@full_float32
def compute_dense_ppca_embeddings(
    experiment_dataset,
    mu,
    W=None,
    *,
    noise_variance,
    rotations,
    translations,
    geometry: GeometryConfig,
    schedule: ScheduleConfig,
    scoring: ScoringConfig,
    disc_type: str = "linear_interp",
    image_indices: np.ndarray | None = None,
    rotation_log_prior: np.ndarray | None = None,
    translation_log_prior: np.ndarray | None = None,
    rotation_translation_mask: np.ndarray | None = None,
) -> DensePPCAEmbeddings:
    """Compute final embeddings on the same complete pose support as accumulation.

    This repeats the dense pass-1 score/moment blocks and centered posterior
    normalization. It omits only the M-step adjoints and expected-noise terms, which are
    unused after the final full-data update. See algorithm section 14.
    """
    blocks = iter_dense_ppca_dataset_block_groups(
        experiment_dataset,
        mu,
        W,
        noise_variance,
        rotations,
        translations,
        disc_type=disc_type,
        image_batch_size=schedule.image_batch_size,
        rotation_block_size=schedule.rotation_block_size,
        current_size=geometry.current_size,
        q=geometry.q,
        volume_domain=geometry.volume_domain,
        image_indices=image_indices,
        rotation_log_prior=rotation_log_prior,
        translation_log_prior=translation_log_prior,
        rotation_translation_mask=rotation_translation_mask,
        image_scale_corrections=scoring.image_scale_corrections,
        class_log_prior=scoring.class_log_prior,
        score_with_masked_images=scoring.score_with_masked_images,
        relion_unit_half_weights=scoring.relion_unit_half_weights,
        square_window=scoring.square_window,
        full_real_observation=scoring.full_real_observation,
        relion_texture_interp=scoring.relion_texture_interp,
    )
    embedding_batches = []
    n_images = 0
    for group in blocks:
        scored = [
            dense_pose_ppca_score_with_moments_blocked(
                block.Y1,
                block.proj_aug,
                block.ctf2_over_noise,
                block.y_norm,
                block.pose_log_prior,
            )
            for block in group
        ]
        center, centered_logZ = _centered_score_partition([block.score for block in scored])
        batch_embedding = jnp.zeros((int(group[0].Y1.shape[0]), geometry.q), jnp.float32)
        for block in scored:
            gamma = jnp.exp(block.score - center[:, None, None] - centered_logZ[:, None, None])
            batch_embedding = batch_embedding + jnp.einsum("btr,btrq->bq", gamma, block.alpha[..., 1:])
        embedding_batches.append(batch_embedding)
        n_images += int(group[0].Y1.shape[0])
    selected = (
        np.arange(experiment_dataset.n_images, dtype=np.int64)
        if image_indices is None else np.asarray(image_indices, dtype=np.int64)
    )
    original_ids = experiment_dataset.original_image_indices_from_local(selected)
    if n_images != len(original_ids):
        raise ValueError("Embedding pass did not cover each selected image")
    return DensePPCAEmbeddings(jnp.concatenate(embedding_batches), original_ids, n_images)


def run_dense_ppca_fused_em_iteration(
    experiment_dataset,
    mu,
    W=None,
    *,
    mean_prior,
    W_prior,
    noise_variance,
    rotations,
    translations,
    geometry: GeometryConfig | None = None,
    schedule: ScheduleConfig | None = None,
    scoring: ScoringConfig | None = None,
    sparse_pass2: SparsePass2Config | None = None,
    mean_reg: MeanRegularizationConfig | None = None,
    postprocess: PostprocessConfig | None = None,
    disc_type: str = "linear_interp",
    image_indices: np.ndarray | None = None,
    rotation_log_prior: np.ndarray | None = None,
    translation_log_prior: np.ndarray | None = None,
    rotation_translation_mask: np.ndarray | None = None,
    enforce_x0: bool = True,
    freeze_mean: bool = False,
    skip_empty_pose_blocks: bool = False,
    top_pose_count: int = 1,
    pose_selection: PoseSelectionConfig | None = None,
) -> DensePPCAFusedEMResult:
    """Run the established regularized refinement M-step after shared accumulation."""
    geometry = geometry if geometry is not None else GeometryConfig()
    schedule = schedule if schedule is not None else ScheduleConfig()
    mean_reg = mean_reg if mean_reg is not None else MeanRegularizationConfig()
    postprocess = postprocess if postprocess is not None else PostprocessConfig()
    stats = accumulate_dense_ppca_statistics(
        experiment_dataset,
        mu,
        W,
        noise_variance=noise_variance,
        rotations=rotations,
        translations=translations,
        geometry=geometry,
        schedule=schedule,
        scoring=scoring,
        sparse_pass2=sparse_pass2,
        mean_reg=mean_reg,
        disc_type=disc_type,
        image_indices=image_indices,
        rotation_log_prior=rotation_log_prior,
        translation_log_prior=translation_log_prior,
        rotation_translation_mask=rotation_translation_mask,
        enforce_x0=enforce_x0,
        skip_empty_pose_blocks=skip_empty_pose_blocks,
        top_pose_count=top_pose_count,
        pose_selection=pose_selection,
    )
    diagnostics = stats.diagnostics
    n_images = stats.n_images
    volume_shape = tuple(int(x) for x in experiment_dataset.volume_shape)
    q_resolved = stats.rhs.shape[1] - 1
    mean_prior, W_prior = jnp.asarray(mean_prior), jnp.asarray(W_prior)
    if W_prior.shape != (mean_prior.shape[0], q_resolved):
        raise ValueError("Prior shape does not match the model")
    mstep_chunk_size = schedule.mstep_chunk_size
    postprocess_bandlimit_max_r = diagnostics.pop("postprocess_bandlimit_max_r")
    mean_precision = resolve_mean_precision(stats, mean_prior, volume_shape, mean_reg)
    input_augmented_half, _input_q = coerce_augmented_half_volumes(
        mu,
        W,
        volume_shape=volume_shape,
        q=q_resolved,
        volume_domain=geometry.volume_domain,
    )
    input_mu_half = input_augmented_half[0]
    input_W_half = (
        jnp.swapaxes(input_augmented_half[1:], 0, 1)
        if q_resolved
        else jnp.zeros((mean_prior.shape[0], 0), dtype=input_mu_half.dtype)
    )
    input_objective = augmented_ppca_mstep_objective(
        stats,
        input_mu_half,
        input_W_half,
        mean_prior=mean_prior,
        W_prior=W_prior,
        mean_precision=mean_precision,
        chunk_size=mstep_chunk_size,
    )
    fixed_mean_half = None
    if freeze_mean:
        fixed_mean_half = input_mu_half
    mu_half, W_half = solve_augmented_ppca_mstep(
        stats,
        mean_prior=mean_prior,
        W_prior=W_prior,
        mean_precision=mean_precision,
        fixed_mean=fixed_mean_half,
        chunk_size=mstep_chunk_size,
    )
    solved_objective = augmented_ppca_mstep_objective(
        stats,
        mu_half,
        W_half,
        mean_prior=mean_prior,
        W_prior=W_prior,
        mean_precision=mean_precision,
        chunk_size=mstep_chunk_size,
    )
    postprocessed = postprocess_ppca_half_volumes(
        mu_half,
        W_half,
        volume_shape,
        config=dataclasses.replace(postprocess, bandlimit_max_r=postprocess_bandlimit_max_r),
    )
    diagnostics.update(postprocessed.diagnostics)
    mu_half, W_half = postprocessed.mu_half, postprocessed.W_half
    diagnostics["mean_frozen"] = bool(freeze_mean)
    diagnostics["mstep_mode"] = "fixed_mean_conditional_W" if freeze_mean else "joint_mu_W"
    if freeze_mean:
        mu_half = fixed_mean_half
    output_objective = augmented_ppca_mstep_objective(
        stats,
        mu_half,
        W_half,
        mean_prior=mean_prior,
        W_prior=W_prior,
        mean_precision=mean_precision,
        chunk_size=mstep_chunk_size,
    )
    diagnostics.update(input_objective.diagnostics("mstep_objective_input", n_images=n_images))
    diagnostics.update(solved_objective.diagnostics("mstep_objective_solved", n_images=n_images))
    diagnostics.update(output_objective.diagnostics("mstep_objective_output", n_images=n_images))
    diagnostics["mstep_objective_solved_delta"] = float(solved_objective.total - input_objective.total)
    diagnostics["mstep_objective_solved_delta_per_image"] = (
        float((solved_objective.total - input_objective.total) / n_images) if n_images else float("nan")
    )
    diagnostics["mstep_objective_output_delta"] = float(output_objective.total - input_objective.total)
    diagnostics["mstep_objective_output_delta_per_image"] = (
        float((output_objective.total - input_objective.total) / n_images) if n_images else float("nan")
    )
    diagnostics["mstep_objective_postprocess_delta"] = float(output_objective.total - solved_objective.total)
    diagnostics["mstep_objective_postprocess_delta_per_image"] = (
        float((output_objective.total - solved_objective.total) / n_images) if n_images else float("nan")
    )
    diagnostics["mstep_objective_scope"] = "fixed_e_step_augmented_quadratic_without_constants"
    diagnostics["mstep_objective_postprocess_in_objective"] = False
    return DensePPCAFusedEMResult(mu_half=mu_half, W_half=W_half, stats=stats, diagnostics=diagnostics)


def combine_halfset_scoring_model(mu_half, W_half):
    """Combine halfset PPCA estimates for scoring with per-column sign alignment."""

    mu0, mu1 = (jnp.asarray(mu_half[0]), jnp.asarray(mu_half[1]))
    W0, W1 = (jnp.asarray(W_half[0]), jnp.asarray(W_half[1]))
    mu_score = 0.5 * (mu0 + mu1)
    if W0.shape != W1.shape:
        raise ValueError(f"halfset W shapes differ: {W0.shape} vs {W1.shape}")
    if W0.shape[-1] == 0:
        return mu_score, W0
    dots = jnp.sum(jnp.conj(W0) * W1, axis=0).real
    signs = jnp.where(dots < 0.0, -1.0, 1.0).astype(W1.real.dtype)
    W_score = 0.5 * (W0 + W1 * signs[None, :])
    return mu_score, W_score


def run_dense_ppca_halfset_fused_em_iteration(
    state: PoseMarginalPPCAEMState,
    experiment_dataset,
    *,
    rotations,
    translations,
    geometry: GeometryConfig | None = None,
    schedule: ScheduleConfig | None = None,
    scoring: ScoringConfig | None = None,
    mean_reg: MeanRegularizationConfig | None = None,
    postprocess: PostprocessConfig | None = None,
    pose_selection: PoseSelectionConfig | None = None,
    top_pose_count: int = 1,
    disc_type: str = "linear_interp",
) -> PoseMarginalPPCAEMState:
    """Run one gold-standard halfset dense PPCA iteration and update state."""
    geometry = geometry if geometry is not None else GeometryConfig(volume_domain="fourier_half")
    schedule = schedule if schedule is not None else ScheduleConfig()
    scoring = scoring if scoring is not None else ScoringConfig()
    mean_reg = mean_reg if mean_reg is not None else MeanRegularizationConfig()
    postprocess = postprocess if postprocess is not None else PostprocessConfig()

    half_datasets = (experiment_dataset.get_halfset(0), experiment_dataset.get_halfset(1))
    results = []
    for half_dataset in half_datasets:
        results.append(
            run_dense_ppca_fused_em_iteration(
                half_dataset,
                state.mu_score,
                state.W_score,
                mean_prior=state.mean_prior,
                W_prior=state.W_prior,
                noise_variance=state.noise_variance,
                rotations=rotations,
                translations=translations,
                geometry=geometry,
                schedule=schedule,
                scoring=scoring,
                mean_reg=mean_reg,
                postprocess=postprocess,
                pose_selection=pose_selection,
                top_pose_count=top_pose_count,
                disc_type=disc_type,
            )
        )

    mu_half = (results[0].mu_half, results[1].mu_half)
    W_half = (results[0].W_half, results[1].W_half)
    mu_score, W_score = combine_halfset_scoring_model(mu_half, W_half)
    old_mu = jnp.asarray(state.mu_score)
    old_W = jnp.asarray(state.W_score)
    pose_diagnostics = {
        "halfset0": results[0].diagnostics,
        "halfset1": results[1].diagnostics,
        "delta_rms_mu": float(jnp.sqrt(jnp.mean(jnp.abs(mu_score - old_mu) ** 2))),
        "delta_rms_W": float(jnp.sqrt(jnp.mean(jnp.abs(W_score - old_W) ** 2))) if old_W.size else 0.0,
    }
    return state.replace(
        mu_half=mu_half,
        W_half=W_half,
        mu_score=mu_score,
        W_score=W_score,
        pose_diagnostics=pose_diagnostics,
    )
