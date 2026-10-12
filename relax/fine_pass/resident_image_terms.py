"""Per-chunk image-level statistics of the resident pass 2 (every term without a row-pixel axis).

:func:`accumulate_chunk_image_terms` folds a chunk's posterior and the row-pixel partials of its M-step
blocks into the per-image statistics (T9a's arithmetic, once per chunk), and :func:`fold_class_axis` sums
a class axis out of them.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from relax.fine_pass.resident_statistics import ResidentStatistics, _drop_index, segment_sum_by_image
from relax.fine_pass.wavg import (
    _replace_low_shell_noise_with_relion_wavg_direct_residual_jnp,
    weighted_image_power_from_shells,
)


class ChunkImageOperands(NamedTuple):
    """One chunk's image-level statistics operands, all already on the device."""

    row_posterior: jax.Array  # float32 [C_R, T]
    row_image_local: jax.Array  # int32 [C_R]
    row_coarse_rot: jax.Array  # int32 [C_R], padded -> >= n_coarse_rot
    image_ids: jax.Array  # int32 [C_B], padded -> -1
    group_ids: jax.Array  # int32 [C_B], padded -> -1
    image_power_shells: jax.Array  # float64 [C_B, n_shells], each image's power per noise shell
    relion_norm_high_shell: jax.Array  # real [C_B]
    wavg_triplet_pixels: jax.Array  # float32 [C_B, P_rect, 3]
    block_noise_shells: jax.Array  # float64 [n_shells]
    a2_per_image: jax.Array  # real [C_B]
    xa_per_image: jax.Array  # real [C_B]
    class_log_z: jax.Array  # float64 [C_B]
    min_diff2: jax.Array  # real [C_B]
    best_log_score: jax.Array  # float32 [C_B]
    max_posterior: jax.Array  # float32 [C_B]
    best_cell_index: jax.Array  # int64 [C_B], segment-relative (r_local * T + t)
    best_fine_rot: jax.Array  # int64 [C_B], global fine rotation id of the winner
    optics_groups: jax.Array | None = None  # int32 [C_B] with G > 1 optics groups
    # K>1 only: each row's class and the (slot, class) reductions, flat slot * K + class.
    row_class: jax.Array | None = None  # int32 [C_R]
    per_class_log_z: jax.Array | None = None  # float64 [C_B * K]
    per_class_best_log_score: jax.Array | None = None  # float32 [C_B * K]
    per_class_best_cell: jax.Array | None = None  # int64 [C_B * K], fine rotation * T + t, -1 if none
    # K>1: each image's scale sums under its classes' own masks (_fold_class_scale_sums).
    scale_xa_per_image: jax.Array | None = None  # float64 [C_B]
    scale_aa_per_image: jax.Array | None = None  # float64 [C_B]


class ChunkImageTables(NamedTuple):
    """Iteration-global tables the image-level program reads every chunk."""

    shell_indices_half: jax.Array  # int32 [P_half]
    wavg_shell_indices: jax.Array  # int32 [P_rect]
    wavg_scale_pixel_mask: jax.Array  # bool [P_rect]; [K, P_rect] with K>1 classes
    translation_sqdist_ang: jax.Array | None  # real [T] or [C_B, T] or None
    # RELION's powerClass cutoff (current_size // 2) as a device scalar; None
    # takes the config's. Its direct-noise stop is the next shell.
    norm_shell_cutoff: jax.Array | None = None  # int32 []


@partial(jax.jit, static_argnames=("config",))
def accumulate_chunk_image_terms(
    stats: ResidentStatistics,
    operands: ChunkImageOperands,
    tables: ChunkImageTables,
    *,
    config,
) -> ResidentStatistics:
    """Fold one chunk's image-level terms into the device accumulators.

    Statement for statement this is the host bucket tail in its production
    configuration; the only change is that the row-pixel reductions arrive as
    already-summed chunk partials, because the pixel axis is walked in blocks.
    """

    image_capacity = int(operands.image_ids.shape[0])
    norm_shell_cutoff = config.norm_unweighted_shell_cutoff
    direct_noise_shell_stop = config.direct_noise_exclusive_shell_stop
    if norm_shell_cutoff is not None and tables.norm_shell_cutoff is not None:
        norm_shell_cutoff = tables.norm_shell_cutoff
        direct_noise_shell_stop = tables.norm_shell_cutoff + 1

    probs = operands.row_posterior
    row_image = jnp.asarray(operands.row_image_local, dtype=jnp.int32)
    image_ids = jnp.asarray(operands.image_ids, dtype=jnp.int32)
    valid_image = image_ids >= 0
    image_slot = _drop_index(image_ids, int(config.image_capacity))

    # --- 1/2. sigma2 offset and support mass -------------------------------
    translation_posterior = segment_sum_by_image(
        probs, row_image, image_capacity,
        float32_posterior_bucket_size=(
            128 if config.float32_bucketed_image_sums and probs.dtype == jnp.float32 else None
        ),
    )
    sigma2_offset = stats.sigma2_offset
    if tables.translation_sqdist_ang is not None:
        sqdist = jnp.asarray(tables.translation_sqdist_ang, dtype=jnp.float64)
        sigma2_offset = sigma2_offset + jnp.sum(
            translation_posterior.astype(jnp.float64) * sqdist
        )
    support_mass = jnp.sum(translation_posterior, axis=1)
    support_mass = jnp.where(valid_image, support_mass, jnp.zeros((), support_mass.dtype))
    if config.n_optics_groups == 1:
        sumw = stats.sumw + jnp.sum(support_mass.astype(jnp.float64))
    else:
        # RELION's sumw_group[optics_group]: the support mass of each group's images.
        image_optics = jnp.asarray(operands.optics_groups, dtype=jnp.int32)
        sumw = stats.sumw + jax.ops.segment_sum(
            support_mass.astype(jnp.float64), image_optics, num_segments=config.n_optics_groups
        )

    # --- 3. weighted image power shells and per-image norm power -----------
    weighted_img_shells, weighted_img_per_image = weighted_image_power_from_shells(
        operands.image_power_shells,
        support_mass,
        operands.relion_norm_high_shell,
        valid_image,
        norm_unweighted_shell_cutoff=norm_shell_cutoff,
        include_unweighted_high_shell=config.include_unweighted_high_shell,
        deterministic_norm_reduction=config.deterministic_norm_reduction,
    )

    # --- 4. per-particle norm correction (production algebraic mode) -------
    # The host adds the image-power term and the A2-2XA residual in two
    # separate ``+=`` statements; keep both scatters separate.
    norm_correction = stats.norm_correction.at[image_slot].add(
        jnp.where(valid_image, weighted_img_per_image, 0.0).astype(jnp.float64),
        mode="drop",
    )

    # --- 5/6. noise shells with RELION's direct low-shell replacement ------
    if config.n_optics_groups == 1:
        residual_shells, img_power_shells = (
            _replace_low_shell_noise_with_relion_wavg_direct_residual_jnp(
                jnp.asarray(operands.block_noise_shells, dtype=jnp.float64),
                weighted_img_shells.astype(jnp.float64),
                operands.wavg_triplet_pixels[:, :, 2],
                tables.wavg_shell_indices,
                exclusive_shell_stop=direct_noise_shell_stop,
                shell_count=config.n_shells,
            )
        )
    else:
        # The same two steps once per optics group, over that group's images only.
        residual_per_group, power_per_group = [], []
        for group in range(config.n_optics_groups):
            in_group = valid_image & (image_optics == group)
            group_img_shells, _ = weighted_image_power_from_shells(
                operands.image_power_shells,
                jnp.where(in_group, support_mass, jnp.zeros((), support_mass.dtype)),
                operands.relion_norm_high_shell,
                in_group,
                norm_unweighted_shell_cutoff=norm_shell_cutoff,
                include_unweighted_high_shell=config.include_unweighted_high_shell,
                deterministic_norm_reduction=config.deterministic_norm_reduction,
            )
            group_residual, group_power = _replace_low_shell_noise_with_relion_wavg_direct_residual_jnp(
                jnp.asarray(operands.block_noise_shells[group], dtype=jnp.float64),
                group_img_shells.astype(jnp.float64),
                jnp.where(in_group[:, None], operands.wavg_triplet_pixels[:, :, 2], jnp.float32(0.0)),
                tables.wavg_shell_indices,
                exclusive_shell_stop=direct_noise_shell_stop,
                shell_count=config.n_shells,
            )
            residual_per_group.append(group_residual)
            power_per_group.append(group_power)
        residual_shells = jnp.stack(residual_per_group)
        img_power_shells = jnp.stack(power_per_group)
    wsum_sigma2_noise = stats.wsum_sigma2_noise + residual_shells
    wsum_img_power = stats.wsum_img_power + img_power_shells

    # --- 7. per-image norm residual ---------------------------------------
    block_norm_residual = operands.a2_per_image - 2.0 * operands.xa_per_image
    norm_correction = norm_correction.at[image_slot].add(
        jnp.where(valid_image, block_norm_residual, 0.0).astype(jnp.float64),
        mode="drop",
    )

    # --- 8. group scale sufficient statistics ------------------------------
    scale_xa = stats.scale_xa
    scale_aa = stats.scale_aa
    if config.accumulate_scale:
        if operands.scale_xa_per_image is not None:
            # K>1: already summed class by class under each class's own mask.
            scale_xa_per_image = operands.scale_xa_per_image
            scale_aa_per_image = operands.scale_aa_per_image
        else:
            mask_rect = jnp.asarray(tables.wavg_scale_pixel_mask, dtype=bool).reshape(1, -1)
            zero_f32 = jnp.float32(0.0)
            scale_xa_per_image = jnp.sum(
                jnp.where(mask_rect, operands.wavg_triplet_pixels[:, :, 0], zero_f32).astype(
                    jnp.float64
                ),
                axis=1,
            )
            scale_aa_per_image = jnp.sum(
                jnp.where(mask_rect, operands.wavg_triplet_pixels[:, :, 1], zero_f32).astype(
                    jnp.float64
                ),
                axis=1,
            )
        group_slot = _drop_index(operands.group_ids, int(config.n_scale_groups))
        keep_group = valid_image & (jnp.asarray(operands.group_ids, dtype=jnp.int32) >= 0)
        scale_xa = scale_xa.at[group_slot].add(
            jnp.where(keep_group, scale_xa_per_image, 0.0).astype(jnp.float64), mode="drop"
        )
        scale_aa = scale_aa.at[group_slot].add(
            jnp.where(keep_group, scale_aa_per_image, 0.0).astype(jnp.float64), mode="drop"
        )

    # --- 9. rotation posterior sums ----------------------------------------
    probs_sum_t = jnp.sum(probs, axis=-1)
    coarse_slot = _drop_index(operands.row_coarse_rot, int(config.n_coarse_rot))
    rotation_posterior_sums = stats.rotation_posterior_sums.at[coarse_slot].add(
        probs_sum_t.astype(jnp.float64), mode="drop"
    )

    # --- 10. per-image score and pose fields -------------------------------
    # ``_relion_cuda_fine_log_evidence_offset`` is exactly ``-min_diff2``.
    log_score_offset = (-jnp.asarray(operands.min_diff2)).astype(jnp.float64)
    class_log_z = jnp.asarray(operands.class_log_z, dtype=jnp.float64)
    best_log_score_chunk = jnp.asarray(operands.best_log_score, dtype=jnp.float64)
    finite = jnp.isfinite(best_log_score_chunk)
    neg_inf = jnp.asarray(-jnp.inf, dtype=jnp.float64)
    absolute = class_log_z + log_score_offset

    log_evidence = stats.log_evidence.at[image_slot].set(
        jnp.where(finite, absolute, neg_inf), mode="drop"
    )
    score_log_z = stats.score_log_z.at[image_slot].set(
        jnp.where(finite, absolute, neg_inf), mode="drop"
    )
    best_log_score = stats.best_log_score.at[image_slot].set(
        best_log_score_chunk + log_score_offset, mode="drop"
    )
    max_posterior = stats.max_posterior.at[image_slot].set(
        jnp.asarray(operands.max_posterior, dtype=stats.max_posterior.dtype), mode="drop"
    )

    best_cell_index = jnp.asarray(operands.best_cell_index, dtype=jnp.int64)
    best_local_rot = (best_cell_index // jnp.int64(config.n_fine_trans)).astype(jnp.int32)
    best_translation = best_cell_index % jnp.int64(config.n_fine_trans)
    best_cell_values = (
        jnp.asarray(operands.best_fine_rot, dtype=jnp.int64) * jnp.int64(config.n_fine_trans)
        + best_translation
    )
    best_cell = stats.best_cell.at[image_slot].set(best_cell_values, mode="drop")
    best_local_rot_out = stats.best_local_rot.at[image_slot].set(best_local_rot, mode="drop")

    # --- 11. the class axis (K>1) ------------------------------------------
    # Each class's evidence and winner in the image's absolute coordinates, and
    # its pruned M-step mass (thr_wsum_pdf_class, ml_optimiser.cpp:10497).
    classes = stats.classes
    if classes is not None:
        classes = fold_class_axis(
            classes,
            image_slot,
            log_score_offset,
            per_class_log_z=operands.per_class_log_z,
            per_class_best_log_score=operands.per_class_best_log_score,
            per_class_best_cell=operands.per_class_best_cell,
            row_mass=probs_sum_t,
            row_class=operands.row_class,
            n_classes=int(config.n_classes),
        )

    return ResidentStatistics(
        wsum_sigma2_noise=wsum_sigma2_noise,
        wsum_img_power=wsum_img_power,
        sigma2_offset=sigma2_offset,
        sumw=sumw,
        norm_correction=norm_correction,
        scale_xa=scale_xa,
        scale_aa=scale_aa,
        rotation_posterior_sums=rotation_posterior_sums,
        log_evidence=log_evidence,
        best_log_score=best_log_score,
        max_posterior=max_posterior,
        best_cell=best_cell,
        score_log_z=score_log_z,
        best_local_rot=best_local_rot_out,
        invalid_best_rows=stats.invalid_best_rows,
        classes=classes,
    )


def fold_class_axis(
    classes,
    unit_slot,
    log_score_offset,
    *,
    per_class_log_z,
    per_class_best_log_score,
    per_class_best_cell,
    row_mass,
    row_class,
    n_classes: int,
):
    """Fold a chunk's class axis (K>1) into the per-class statistics.

    Each class's evidence and winner in the unit's absolute coordinates (``log_score_offset`` is the
    unit's ``-min_diff2``), and its pruned M-step mass (thr_wsum_pdf_class, ml_optimiser.cpp:10497).
    ``per_class_*`` are flat ``unit * K + class``; ``row_mass`` is each row's posterior summed over
    translations. The unit is an image (SPA) or a subtomogram particle (resident_tilts).
    """

    n_units = int(log_score_offset.shape[0])
    neg_inf = jnp.asarray(-jnp.inf, dtype=jnp.float64)
    class_best = jnp.asarray(per_class_best_log_score, dtype=jnp.float64).reshape(n_units, n_classes)
    class_absolute = (
        jnp.asarray(per_class_log_z, dtype=jnp.float64).reshape(n_units, n_classes) + log_score_offset[:, None]
    )
    class_mass = jax.ops.segment_sum(
        jnp.asarray(row_mass).astype(jnp.float64),
        jnp.asarray(row_class, dtype=jnp.int32),
        num_segments=n_classes,
    )
    return classes._replace(
        log_evidence=classes.log_evidence.at[unit_slot].set(
            jnp.where(jnp.isfinite(class_best), class_absolute, neg_inf), mode="drop"
        ),
        best_log_score=classes.best_log_score.at[unit_slot].set(class_best + log_score_offset[:, None], mode="drop"),
        best_cell=classes.best_cell.at[unit_slot].set(
            jnp.asarray(per_class_best_cell, dtype=jnp.int64).reshape(n_units, n_classes),
            mode="drop",
        ),
        posterior_sums=classes.posterior_sums + class_mass,
    )
