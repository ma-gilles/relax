"""Bounded resident statistics for the exact full-grid dense GEMM E/M pass.

The GEMM engine owns posterior normalization and collapsed BPref slices.  This
module feeds its same posterior tiles through the resident engine's native
translate/Wavg/noise primitives, then hands their partials to the resident
image-level reducer.  No RELION statistic formula is reimplemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from relax.cuda import kernels as cuda_backproject
from relax.sparse_pass2.resident_pass2 import (
    _accumulate_chunk_image_terms,
    _add_wavg_rectangle_image_power,
    _ChunkImageOperands,
    _ChunkImageTables,
    _fold_class_scale_sums,
    _resident_block_noise_and_norm,
    _resident_block_wavg_algebraic_terms,
    _resident_block_weighted_sums_kernel,
)
from relax.sparse_pass2.resident_statistics import ResidentStatisticsConfig
from relax.sparse_pass2.sparse_pass2_wavg import relion_cuda_translate_wavg_norm_window


class DenseGemmStatisticsOperands(NamedTuple):
    """One capacity batch, with RELION score/reconstruction windows kept distinct."""

    recon_image: jax.Array  # raw BPref input, or already weighted when nonexact [B, Pr]
    recon_weight: jax.Array | None  # weighted CTF after translation in exact mode [B, Pr]
    noise_image: jax.Array  # score-convention unshifted image [B, Pr]
    ctf2_over_nv_recon: jax.Array  # [B, Pr]
    direct_ctf_rfloat_recon: jax.Array | None  # [B, Pr]
    wavg_image_rect: jax.Array  # [B, P_rect]
    scale: jax.Array  # [B]
    optics_groups: jax.Array | None  # [B]
    image_noise_scale: jax.Array | None  # [B]
    noise_variance: jax.Array  # [Pr] or [G, Pr]
    shell_indices_noise: jax.Array  # [Pr]
    translation_angles: jax.Array  # [Tpad, 2]
    recon_pixel_indices: jax.Array  # [Pr]
    rect_indices: jax.Array  # [P_rect]
    exact_positions: jax.Array  # [Pr]
    logical_recon_pixels: jax.Array  # scalar
    logical_rect_pixels: jax.Array  # scalar
    wavg_scale_pixel_mask: jax.Array  # [P_rect] or [K, P_rect]
    image_power_shells: jax.Array  # [B, n_shells]
    relion_norm_high_shell: jax.Array  # [B]
    image_ids: jax.Array  # global ids [B], padding -1
    group_ids: jax.Array  # scale-group ids [B], padding -1
    translation_sqdist_ang: jax.Array | None  # [B,T] or [T]
    fine_to_coarse_parent: jax.Array  # [Rpad], padded id -> n_coarse_rot
    fine_rotation_ids: jax.Array  # [Rpad] canonical global fine IDs
    fine_local_slots: jax.Array  # [Rpad] inverse of the resident class table's full-grid row order
    cc_half_batch_norm: jax.Array  # [B], zero on Gaussian route
    wavg_shell_indices: jax.Array  # [P_rect]
    norm_shell_cutoff: jax.Array | None  # scalar or None


class DenseGemmStatisticsCarry(NamedTuple):
    """Only row-pixel partials; image-level RELION algebra runs once per batch."""

    wavg_triplet_pixels: jax.Array  # [B,P_rect,3]
    noise_shells: jax.Array  # float64 [n_shells] or [G,n_shells]
    a2_per_image: jax.Array  # [B]
    xa_per_image: jax.Array  # [B]
    scale_xa_per_image: jax.Array | None  # [B] for K>1 class masks
    scale_aa_per_image: jax.Array | None  # [B] for K>1 class masks


@dataclass(frozen=True)
class DenseGemmStatisticsPlan:
    image_shape: tuple[int, int]
    image_capacity: int
    rotation_tile: int
    translation_tile: int
    rows_per_statistics_block: int
    n_recon_pixels: int
    n_rect_pixels: int
    n_shells: int
    n_classes: int
    n_coarse_rot: int
    n_fine_rotations: int
    n_optics_groups: int
    use_rfloat_ctf_wavg: bool
    score_mode: str
    stats_config: ResidentStatisticsConfig

    def __post_init__(self):
        if self.rotation_tile % self.rows_per_statistics_block:
            raise ValueError("rotation tile must divide into bounded statistics row blocks")
        if self.score_mode not in {"gaussian", "normalized_cc"}:
            raise ValueError(f"unsupported score mode {self.score_mode!r}")


def initial_statistics_carry(plan: DenseGemmStatisticsPlan, *, noise_real_dtype=jnp.float32) -> DenseGemmStatisticsCarry:
    b = plan.image_capacity
    groups = (plan.n_optics_groups,) if plan.n_optics_groups > 1 else ()
    class_scale = plan.n_classes > 1
    return DenseGemmStatisticsCarry(
        wavg_triplet_pixels=jnp.zeros((b, plan.n_rect_pixels, 3), jnp.float32),
        noise_shells=jnp.zeros(groups + (plan.n_shells,), jnp.float64),
        a2_per_image=jnp.zeros((b,), noise_real_dtype),
        xa_per_image=jnp.zeros((b,), noise_real_dtype),
        scale_xa_per_image=jnp.zeros((b,), jnp.float64) if class_scale else None,
        scale_aa_per_image=jnp.zeros((b,), jnp.float64) if class_scale else None,
    )


def make_statistics_callbacks(plan: DenseGemmStatisticsPlan):
    """Return callbacks traced into the dense program's second sweep."""

    b, q, u = plan.image_capacity, plan.rotation_tile, plan.translation_tile
    qs = plan.rows_per_statistics_block

    def step(carry, ops, posterior, rec_projection, _class_id, rot_start, trans_start, batch, grid):
        del grid
        angles = jax.lax.dynamic_slice_in_dim(ops.translation_angles, trans_start, u, axis=0)
        raw_rect = relion_cuda_translate_wavg_norm_window(
            ops.wavg_image_rect, angles, ops.rect_indices, plan.image_shape
        )
        raw_exact = raw_rect[:, :, ops.exact_positions]

        def one_block(block_index, partial):
            start = block_index * qs
            qrows = jax.lax.dynamic_slice_in_dim(posterior, start, qs, axis=1)
            row_probs = qrows.reshape(b * qs, u)
            row_proj = jax.lax.dynamic_slice_in_dim(rec_projection, start, qs, axis=0)
            row_proj = jnp.broadcast_to(row_proj[None, :, :], (b, qs, plan.n_recon_pixels)).reshape(
                b * qs, plan.n_recon_pixels
            )
            image_slot = jnp.repeat(jnp.arange(b, dtype=jnp.int32), qs)
            # Invalid rotation and image rows have zero posterior. The native
            # translate kernel still receives -1 for padded image slots.
            kernel_ids = jnp.where(batch.valid_images[image_slot], image_slot, -1)
            summed, summed_masked, ctf_probs, _ = _resident_block_weighted_sums_kernel(
                row_probs, kernel_ids, image_slot,
                ops.recon_image, ops.recon_weight, ops.noise_image,
                ops.ctf2_over_nv_recon, ops.recon_pixel_indices, angles,
                image_shape=plan.image_shape, n_recon_pixels=plan.n_recon_pixels,
                kernel_ctf_probs=False, cuda_backproject=cuda_backproject,
            )
            live = jnp.arange(plan.n_recon_pixels, dtype=jnp.int32) < ops.logical_recon_pixels
            summed_masked = jnp.where(live, summed_masked, 0)
            ctf_probs = jnp.where(live, ctf_probs, 0)
            proj_abs2 = (jnp.abs(row_proj) ** 2).astype(jnp.float32)
            row_optics = None if ops.optics_groups is None else ops.optics_groups[image_slot]
            if plan.use_rfloat_ctf_wavg:
                exact = cuda_backproject.relion_wavg_sequential_runtime_flat_rows_triplet_f32(
                    jnp.asarray(row_proj, jnp.complex64), kernel_ids,
                    jnp.asarray(ops.direct_ctf_rfloat_recon, jnp.float32),
                    jnp.asarray(ops.scale, jnp.float32),
                    jnp.asarray(raw_exact, jnp.complex64), row_probs,
                    ops.logical_recon_pixels,
                )
            else:
                exact = _resident_block_wavg_algebraic_terms(
                    row_proj, proj_abs2, summed_masked, ctf_probs,
                    ops.noise_variance, ops.scale, image_slot, row_optics,
                )
            wavg = cuda_backproject.relion_wavg_exact_atomic_flat_rows_triplet_add_f32(
                jnp.asarray(exact, jnp.float32), kernel_ids, ops.exact_positions,
                partial.wavg_triplet_pixels, ops.logical_rect_pixels,
            )
            noise_masked, noise_ctf = summed_masked, ctf_probs
            if ops.image_noise_scale is not None:
                scale = ops.image_noise_scale[image_slot, None].astype(jnp.float32)
                noise_masked = noise_masked * scale
                noise_ctf = noise_ctf * scale
            shells, a2, xa = _resident_block_noise_and_norm(
                row_proj, proj_abs2, noise_masked, noise_ctf,
                ops.noise_variance, ops.shell_indices_noise, image_slot, row_optics,
                n_shells=plan.n_shells, image_capacity=b,
            )
            return partial._replace(
                wavg_triplet_pixels=wavg,
                noise_shells=partial.noise_shells + shells,
                a2_per_image=partial.a2_per_image + a2,
                xa_per_image=partial.xa_per_image + xa,
            )

        carry = jax.lax.fori_loop(0, q // qs, one_block, carry)
        # Add the image-power part of the rectangle from this bounded U tile.
        # A full [B,T,P_rect] expansion defeats the serial-translation plan.
        row_image = jnp.repeat(jnp.arange(b, dtype=jnp.int32), q)
        wavg = _add_wavg_rectangle_image_power(
            carry.wavg_triplet_pixels, raw_rect, posterior.reshape(b * q, u),
            row_image, ops.exact_positions, ops.logical_rect_pixels,
            power_at_exact_positions=not plan.use_rfloat_ctf_wavg,
        )
        return carry._replace(wavg_triplet_pixels=wavg)

    def finish_class(carry, ops, class_id):
        if plan.n_classes == 1:
            return carry
        return _fold_class_scale_sums(carry, ops.wavg_scale_pixel_mask[class_id])

    return step, finish_class


def split_normalizer_metadata(pair, best_score):
    """Preserve float32 posterior precision and float64 evidence metadata."""

    log_z = pair[..., 0].astype(jnp.float64) + pair[..., 1].astype(jnp.float64)
    pmax = jnp.where(
        jnp.isfinite(best_score),
        jnp.exp((best_score - pair[..., 0]) - pair[..., 1]),
        0,
    ).astype(jnp.float32)
    return log_z, pmax


@partial(jax.jit, static_argnames=("plan",))
def accumulate_batch_statistics(stats, result, carry, ops, batch, *, plan: DenseGemmStatisticsPlan):
    """Fold one full-grid image batch into existing device stats, with no host pull."""

    b, k, t = plan.image_capacity, plan.n_classes, plan.stats_config.n_fine_trans
    rows = jnp.transpose(result.class_translation_marginals[:, :, :t], (1, 0, 2)).reshape(b * k, t)
    row_image = jnp.repeat(jnp.arange(b, dtype=jnp.int32), k)
    row_class = jnp.tile(jnp.arange(k, dtype=jnp.int32), b)
    row_coarse = jnp.full((b * k,), k * plan.n_coarse_rot, jnp.int32)
    # Metadata is deliberately float64 in the resident reducer.  Convert the
    # split pair before adding: a large score gauge can erase the small logsum
    # if the pair is first collapsed in float32.
    best_class = jnp.argmax(result.class_best_scores, axis=0)
    best_score = jnp.max(result.class_best_scores, axis=0)
    class_z, _ = split_normalizer_metadata(result.class_normalizers, result.class_best_scores)
    joint_z, gaussian_pmax = split_normalizer_metadata(result.joint_normalizer, best_score)
    best_pose = jnp.take_along_axis(result.class_best_pose_ids, best_class[None, :], axis=0)[0]
    safe_pose = jnp.maximum(best_pose, 0)
    best_rotation = safe_pose // t
    best_fine_rot = ops.fine_rotation_ids[best_rotation]
    best_local_rot = ops.fine_local_slots[best_rotation]
    if plan.score_mode == "normalized_cc":
        pmax = jnp.isfinite(best_score).astype(jnp.float32)
        min_diff2 = ops.cc_half_batch_norm
    else:
        pmax = gaussian_pmax
        min_diff2 = jnp.zeros((b,), jnp.float32)
    class_pose = result.class_best_pose_ids
    safe_class_pose = jnp.maximum(class_pose, 0)
    class_rot = safe_class_pose // t
    class_fine_rot = ops.fine_rotation_ids[class_rot]
    class_best_cell = jnp.where(
        class_pose >= 0, class_fine_rot * t + safe_class_pose % t, -1
    ).T.reshape(-1)
    image_operands = _ChunkImageOperands(
        row_posterior=rows,
        row_image_local=row_image,
        row_coarse_rot=row_coarse,
        image_ids=ops.image_ids,
        group_ids=ops.group_ids,
        image_power_shells=ops.image_power_shells,
        relion_norm_high_shell=ops.relion_norm_high_shell,
        wavg_triplet_pixels=carry.wavg_triplet_pixels,
        block_noise_shells=carry.noise_shells,
        a2_per_image=carry.a2_per_image,
        xa_per_image=carry.xa_per_image,
        class_log_z=joint_z,
        min_diff2=min_diff2,
        best_log_score=best_score,
        max_posterior=pmax,
        best_cell_index=(best_class * plan.n_fine_rotations
                         + best_local_rot) * t + safe_pose % t,
        best_fine_rot=best_fine_rot,
        optics_groups=ops.optics_groups,
        row_class=row_class if k > 1 else None,
        per_class_log_z=class_z.T.reshape(-1) if k > 1 else None,
        per_class_best_log_score=result.class_best_scores.T.reshape(-1) if k > 1 else None,
        per_class_best_cell=class_best_cell if k > 1 else None,
        scale_xa_per_image=carry.scale_xa_per_image,
        scale_aa_per_image=carry.scale_aa_per_image,
    )
    tables = _ChunkImageTables(
        shell_indices_half=ops.shell_indices_noise,
        wavg_shell_indices=ops.wavg_shell_indices,
        wavg_scale_pixel_mask=ops.wavg_scale_pixel_mask,
        translation_sqdist_ang=ops.translation_sqdist_ang,
        norm_shell_cutoff=ops.norm_shell_cutoff,
    )
    stats = _accumulate_chunk_image_terms(stats, image_operands, tables, config=plan.stats_config)
    # The compressed rows deliberately used a drop sentinel above. Replace
    # their empty rotation contribution with the exact fine→coarse/class sum.
    r = result.rotation_posterior_sums.shape[1]
    fine = jnp.broadcast_to(ops.fine_to_coarse_parent[None, :], (k, r))
    class_ids = jnp.arange(k, dtype=jnp.int32)[:, None]
    indices = jnp.where(fine < plan.n_coarse_rot, class_ids * plan.n_coarse_rot + fine, k * plan.n_coarse_rot)
    rotation_mass = jax.ops.segment_sum(
        result.rotation_posterior_sums.reshape(-1), indices.reshape(-1),
        num_segments=k * plan.n_coarse_rot + 1,
    )[:-1]
    return stats._replace(rotation_posterior_sums=stats.rotation_posterior_sums + rotation_mass)
