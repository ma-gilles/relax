"""Adaptive coarse significance of dense PPCA, for the oversampled local refinement script.

``compute_dense_ppca_adaptive_significance`` scores the coarse grid over the dense dataset blocks
(``relax.ppca_refinement.dense_blocks``) and keeps each image's significant rotations and translations
(``DensePPCASignificanceResult``). Only ``scripts/run_ppca_dense_os_local_from_init_npz.py`` uses it, so it lives
with the scripts (relax rule 37).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from relax.ppca_refinement.config import GeometryConfig, PoseSelectionConfig, ScheduleConfig, ScoringConfig
from relax.ppca_refinement.dense_blocks import iter_dense_ppca_dataset_blocks
from relax.ppca_refinement.engine import dense_pose_ppca_score_tensor_stats_blocked
from relax.ppca_refinement.pose_selection import select_distinct_top_poses, top_pose_candidate_count
from relax.sampling.oversampling import find_significant_mask, top_k_rows


class DensePPCASignificanceResult(NamedTuple):
    """Coarse PPCA pass-1 significance needed for adaptive pass 2."""

    significant_sample_indices: list[np.ndarray | None]
    n_significant_per_image: np.ndarray
    hard_assignment: np.ndarray
    best_rotation_idx: np.ndarray
    best_translation_idx: np.ndarray
    logZ: np.ndarray
    max_posterior_per_image: np.ndarray
    diagnostics: dict


def _dense_top_rotation_matrices(rotations: np.ndarray, top_rotation_idx: np.ndarray) -> np.ndarray:
    """Return padded top-p rotation matrices for dense-grid diagnostics."""

    top_rotation_idx = np.asarray(top_rotation_idx, dtype=np.int64)
    out = np.zeros(top_rotation_idx.shape + (3, 3), dtype=np.float32)
    valid = top_rotation_idx >= 0
    if np.any(valid):
        out[valid] = np.asarray(rotations, dtype=np.float32)[top_rotation_idx[valid]]
    return out


def _score_block_to_rotation_major_flat(score) -> jax.Array:
    """Convert ``(B, T, R)`` PPCA scores to ``(B, R*T)`` flat pose order."""

    return jnp.swapaxes(jnp.asarray(score), 1, 2).reshape(jnp.asarray(score).shape[0], -1)


def compute_dense_ppca_adaptive_significance(
    experiment_dataset,
    mu,
    W=None,
    *,
    noise_variance,
    rotations,
    translations,
    adaptive_fraction: float = 0.999,
    max_significants: int = -1,
    geometry: GeometryConfig | None = None,
    schedule: ScheduleConfig | None = None,
    scoring: ScoringConfig | None = None,
    disc_type: str = "linear_interp",
    image_indices: np.ndarray | None = None,
    rotation_log_prior: np.ndarray | None = None,
    translation_log_prior: np.ndarray | None = None,
    rotation_translation_mask: np.ndarray | None = None,
    skip_empty_pose_blocks: bool = False,
    pose_selection: PoseSelectionConfig | None = None,
) -> DensePPCASignificanceResult:
    """Run PPCA pass 1 and return RELION-style significant coarse pose samples.

    The returned ``significant_sample_indices`` use rotation-major packed pose
    IDs (``rotation_idx * n_translations + translation_idx``), matching
    :func:`relax.local_search.layout.build_pass2_hypothesis_layout`.
    """

    geometry = geometry if geometry is not None else GeometryConfig()
    schedule = schedule if schedule is not None else ScheduleConfig()
    scoring = scoring if scoring is not None else ScoringConfig()
    pose_selection = pose_selection if pose_selection is not None else PoseSelectionConfig()
    rotations = np.asarray(rotations, dtype=np.float32)
    translations = np.asarray(translations, dtype=np.float32)
    n_trans = int(translations.shape[0])
    top_pose_count = int(pose_selection.top_p_poses)
    selected_indices = (
        np.arange(getattr(experiment_dataset, "n_units", experiment_dataset.n_images), dtype=np.int64)
        if image_indices is None
        else np.asarray(image_indices, dtype=np.int64).reshape(-1)
    )
    n_selected = int(selected_indices.shape[0])
    significant_sample_indices: list[np.ndarray | None] = [None] * n_selected
    n_significant = np.zeros(n_selected, dtype=np.int32)
    hard_assignment = np.zeros(n_selected, dtype=np.int32)
    best_rotation = np.zeros(n_selected, dtype=np.int32)
    best_translation = np.zeros(n_selected, dtype=np.int32)
    logZ_all = np.full(n_selected, -np.inf, dtype=np.float64)
    pmax_all = np.zeros(n_selected, dtype=np.float32)
    top_rotation_all = np.full((n_selected, top_pose_count), -1, dtype=np.int32)
    top_translation_all = np.full((n_selected, top_pose_count), -1, dtype=np.int32)
    top_log_score_all = np.full((n_selected, top_pose_count), -np.inf, dtype=np.float32)
    top_posterior_all = np.zeros((n_selected, top_pose_count), dtype=np.float32)
    blocks = iter_dense_ppca_dataset_blocks(
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
        image_indices=selected_indices,
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
    )

    def _finish_batch(batch_start: int, batch_size: int, score_parts: list[np.ndarray]) -> None:
        if not score_parts:
            return
        score_flat = jnp.asarray(np.concatenate(score_parts, axis=1), dtype=jnp.float32)
        logZ = jax.scipy.special.logsumexp(score_flat, axis=-1)
        weights = jnp.exp(score_flat - logZ[:, None])
        sig_mask, n_sig = find_significant_mask(
            weights,
            adaptive_fraction=float(adaptive_fraction),
            max_significants=int(max_significants),
        )
        best_flat = jnp.argmax(score_flat, axis=-1).astype(jnp.int32)
        pmax = jnp.max(weights, axis=-1).astype(jnp.float32)
        raw_top_count = top_pose_candidate_count(pose_selection, int(score_flat.shape[1]))
        top_scores_raw, top_flat_raw = top_k_rows(score_flat, raw_top_count)
        top_rotation_raw = (top_flat_raw // int(n_trans)).astype(jnp.int32)
        top_translation_raw = (top_flat_raw % int(n_trans)).astype(jnp.int32)
        top_selection = select_distinct_top_poses(
            np.asarray(jax.block_until_ready(top_scores_raw), dtype=np.float32),
            np.asarray(jax.block_until_ready(top_rotation_raw), dtype=np.int32),
            np.asarray(jax.block_until_ready(top_translation_raw), dtype=np.int32),
            logZ=np.asarray(jax.block_until_ready(logZ), dtype=np.float64),
            rotations=rotations,
            translations=translations,
            config=pose_selection,
        )

        row_slice = slice(batch_start, batch_start + batch_size)
        sig_mask_np = np.asarray(jax.block_until_ready(sig_mask), dtype=bool)
        n_significant[row_slice] = np.asarray(jax.block_until_ready(n_sig), dtype=np.int32)
        hard_assignment[row_slice] = np.asarray(jax.block_until_ready(best_flat), dtype=np.int32)
        best_rotation[row_slice] = hard_assignment[row_slice] // n_trans
        best_translation[row_slice] = hard_assignment[row_slice] % n_trans
        logZ_all[row_slice] = np.asarray(jax.block_until_ready(logZ), dtype=np.float64)
        pmax_all[row_slice] = np.asarray(jax.block_until_ready(pmax), dtype=np.float32)
        top_rotation_all[row_slice] = top_selection.rotation_idx
        top_translation_all[row_slice] = top_selection.translation_idx
        top_log_score_all[row_slice] = top_selection.log_score
        top_posterior_all[row_slice] = top_selection.posterior
        for local_row in range(batch_size):
            if bool(np.all(sig_mask_np[local_row])):
                significant_sample_indices[batch_start + local_row] = None
            else:
                significant_sample_indices[batch_start + local_row] = np.flatnonzero(
                    sig_mask_np[local_row],
                ).astype(np.int32)

    current_batch_start = None
    current_batch_size = 0
    score_parts: list[np.ndarray] = []
    for block in blocks:
        block_batch_start = int(block.batch_start)
        if current_batch_start is None:
            current_batch_start = block_batch_start
            current_batch_size = int(block.Y1.shape[0])
        elif block_batch_start != current_batch_start:
            _finish_batch(current_batch_start, current_batch_size, score_parts)
            score_parts = []
            current_batch_start = block_batch_start
            current_batch_size = int(block.Y1.shape[0])
        score_stats = dense_pose_ppca_score_tensor_stats_blocked(
            block.Y1,
            block.proj_aug,
            block.ctf2_over_noise,
            block.y_norm,
            block.pose_log_prior,
        )
        score_part = _score_block_to_rotation_major_flat(score_stats.score)
        score_parts.append(np.asarray(jax.block_until_ready(score_part), dtype=np.float32))
        del score_part, score_stats, block
    if current_batch_start is not None:
        _finish_batch(current_batch_start, current_batch_size, score_parts)

    return DensePPCASignificanceResult(
        significant_sample_indices=significant_sample_indices,
        n_significant_per_image=n_significant,
        hard_assignment=hard_assignment,
        best_rotation_idx=best_rotation,
        best_translation_idx=best_translation,
        logZ=logZ_all,
        max_posterior_per_image=pmax_all,
        diagnostics={
            "adaptive_fraction": float(adaptive_fraction),
            "max_significants": int(max_significants),
            "n_images": int(n_selected),
            "n_rotations": int(rotations.shape[0]),
            "n_translations": int(n_trans),
            "nsig_mean": float(np.mean(n_significant)) if n_selected else float("nan"),
            "pmax_mean": float(np.mean(pmax_all)) if n_selected else float("nan"),
            "log_likelihood": float(np.sum(logZ_all)) if n_selected else 0.0,
            "logZ_mean": float(np.mean(logZ_all)) if n_selected else float("nan"),
            "max_posterior_per_image": pmax_all,
            "best_rotation_idx": best_rotation,
            "best_rotation_id": best_rotation,
            "best_rotation_matrix": rotations[best_rotation] if n_selected else np.zeros((0, 3, 3), dtype=np.float32),
            "best_translation_idx": best_translation,
            "top_rotation_idx": top_rotation_all,
            "top_rotation_id": top_rotation_all,
            "top_rotation_matrix": _dense_top_rotation_matrices(rotations, top_rotation_all),
            "top_translation_idx": top_translation_all,
            "top_log_score": top_log_score_all,
            "top_log_score_per_image": top_log_score_all,
            "top_posterior": top_posterior_all,
            "top_posterior_per_image": top_posterior_all,
            "top_p_poses": int(pose_selection.top_p_poses),
            "top_pose_count": int(pose_selection.top_p_poses),
            "top_pose_max_log_score_gap": float(pose_selection.top_pose_max_log_score_gap),
            "top_pose_min_angle_deg": float(pose_selection.top_pose_min_angle_deg),
            "top_pose_min_translation_px": float(pose_selection.top_pose_min_translation_px),
        },
    )
