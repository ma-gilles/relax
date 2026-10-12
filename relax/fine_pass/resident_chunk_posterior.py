"""The posterior stage of a resident chunk.

From the chunk's scores to its float32 fine posterior over the CSR segments: the segmented RELION posterior
(per image, or per (image, class) sub-segment for Class3D), the first-iteration cross-correlation
winner-take-all, and the stage's jitted program.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from relax.fine_pass import resident_chunk_records
from relax.fine_pass.resident_scoring import score_resident_chunk


def logical_current_size(tables: resident_chunk_records.ChunkStageTables, spec: resident_chunk_records.ChunkProgramSpec):
    if tables.window_logical is None:
        return jnp.asarray(spec.current_size, dtype=jnp.int32)
    return tables.window_logical.current_size


def resident_chunk_posterior(
    rows: resident_chunk_records.ChunkRowArrays,
    operands: resident_chunk_records.ChunkStageOperands,
    tables: resident_chunk_records.ChunkStageTables,
    *,
    spec: resident_chunk_records.ChunkProgramSpec,
) -> resident_chunk_records.ChunkPosterior:
    """Gather, cached projection, score and the segmented RELION posterior.

    Identical calls to the per-stage path; factored out so the jitted program
    and its oracle cannot drift apart.
    """

    row_index = jnp.arange(spec.row_capacity, dtype=jnp.int32)
    row_is_valid = row_index < rows.n_valid_rows
    kernel_row_image_ids = jnp.where(row_is_valid, rows.row_image_local, jnp.int32(-1))
    image_index = jnp.arange(spec.image_capacity, dtype=jnp.int32)
    chunk_image_ids = jnp.where(image_index < rows.n_valid_images, image_index, jnp.int32(-1))

    if spec.firstiter_cc:
        return _resident_chunk_posterior_firstiter_cc(
            rows,
            operands,
            tables,
            spec=spec,
            row_is_valid=row_is_valid,
            kernel_row_image_ids=kernel_row_image_ids,
        )
    scored = score_resident_chunk(
        rows.row_image_local,
        rows.row_fine_rot,
        rows.row_log_prior,
        rows.row_mask_bits,
        rows.row_mask_mode,
        rows.n_valid_rows,
        chunk_image_ids,
        tables.projection_score_cache,
        operands.score_input,
        operands.corr_img_score,
        operands.highres_xi2_half,
        operands.translation_prior,
        half_weights=tables.half_weights,
        translation_angles=tables.translation_angles,
        full_to_compact=tables.full_to_compact,
        fine_translation_parent=tables.fine_translation_parent,
        logical_current_size=logical_current_size(tables, spec),
        row_capacity=spec.row_capacity,
        image_capacity=spec.image_capacity,
        n_fine_trans=spec.n_fine_trans,
        n_score_pixels=int(spec.n_score_pixels),
        score_take=tables.union_score_take,
        native_fft_size=int(spec.union_native_fft_size),
    )
    return chunk_posterior_from_scores(
        scored,
        rows,
        tables,
        spec=spec,
        row_is_valid=row_is_valid,
        kernel_row_image_ids=kernel_row_image_ids,
    )


def chunk_posterior_from_scores(
    scored,
    rows: resident_chunk_records.ChunkRowArrays,
    tables: resident_chunk_records.ChunkStageTables,
    *,
    spec: resident_chunk_records.ChunkProgramSpec,
    row_is_valid,
    kernel_row_image_ids,
) -> resident_chunk_records.ChunkPosterior:
    """The segmented RELION posterior of a scored chunk (:func:`resident_chunk_posterior`'s tail)."""

    from relax.cuda import posterior as cuda_posterior

    scores_flat = jnp.asarray(scored.scores, dtype=jnp.float32).reshape(-1)

    reuse = bool(spec.reuse_coarse_normalization)
    if reuse:
        # rows.image_ids are the chunk's global image ids, -1 in padded slots,
        # whose values the segmented kernels never read.
        image_slot = jnp.maximum(rows.image_ids, jnp.int32(0))
        coarse_sum_weight = tables.coarse_reuse.sum_weight[image_slot]
        external_sum_weight = coarse_sum_weight.astype(jnp.float32)
    else:
        external_sum_weight = jnp.ones((spec.image_capacity,), dtype=jnp.float32)
    log_z = cuda_posterior.sparse_pass2_segmented_log_z_f64(
        scores_flat, rows.segment_offsets, rows.n_valid_images
    )
    posterior = cuda_posterior.sparse_pass2_segmented_posterior_f32(
        scores_flat,
        rows.segment_offsets,
        rows.n_valid_images,
        log_z,
        external_sum_weight,
        adaptive_fraction=float(spec.adaptive_fraction),
        keep_all=reuse,
        use_external_sum_weight=reuse,
    )
    (
        log_z_out,
        best_log_score,
        best_cell_index,
        max_posterior,
        _probs,
        _normalized_weights,
        reconstruction_probs,
        _mask,
        _n_significant,
        _sum_weight,
        _threshold,
    ) = posterior
    if reuse:
        # The compact engine's zero-oversampling arithmetic
        # (the deleted sparse_pass2_bucketed.py, reuse_coarse_normalization): RELION keeps the
        # coarse numeric sum but the fine pass's own exponent shift,
        # dLL = log(sum_weight) - (50 - best), and the coarse winner and Pmax.
        exponent_add = jnp.float32(50.0) - jnp.asarray(best_log_score, dtype=jnp.float32)
        log_z_out = jnp.log(coarse_sum_weight.astype(jnp.float64)) - exponent_add.astype(jnp.float64)
        best_cell_index = tables.coarse_reuse.winner_cell[image_slot]
        max_posterior = tables.coarse_reuse.max_posterior[image_slot].astype(
            jnp.asarray(max_posterior).dtype
        )
    classes = None
    if rows.classes is not None:
        classes = class_sub_segment_posterior(
            scores_flat.reshape(spec.row_capacity, spec.n_fine_trans),
            rows,
            row_is_valid,
            n_segments=spec.image_capacity * int(spec.n_classes),
            n_classes=int(spec.n_classes),
        )
    return resident_chunk_records.ChunkPosterior(
        row_posterior=jnp.asarray(reconstruction_probs, dtype=jnp.float32).reshape(
            spec.row_capacity, spec.n_fine_trans
        ),
        min_diff2=scored.min_diff2,
        class_log_z=jnp.asarray(log_z_out, dtype=jnp.float64),
        best_log_score=best_log_score,
        best_cell_index=jnp.asarray(best_cell_index, dtype=jnp.int64),
        max_posterior=max_posterior,
        kernel_row_image_ids=kernel_row_image_ids,
        row_is_valid=row_is_valid,
        classes=classes,
    )


def class_sub_segment_posterior(scores, rows, row_is_valid, *, n_segments: int, n_classes: int):
    """Each (slot, class) sub-segment's log-Z and first maximum, for the per-class statistics.

    The scores are the joint-min-centred ones the image's posterior normalizes,
    so a class's log-Z plus the image's offset is its absolute evidence and
    ``best - logZ_image`` its share of Pmax. The log-Z is the segmented kernel
    the image posterior uses, run on the sub-segments; the winner is the first
    maximum in row order, as for the image (:func:`winner_take_all_cells`).
    """

    from relax.cuda import posterior as cuda_posterior

    layout = rows.classes
    log_z = cuda_posterior.sparse_pass2_segmented_log_z_f64(
        scores.reshape(-1),
        layout.segment_offsets,
        rows.n_valid_images * jnp.int32(n_classes),
    )
    best_log_score, best_cell_index, _winner = winner_take_all_cells(
        scores,
        layout.row_segment,
        row_is_valid,
        layout.segment_offsets,
        image_capacity=n_segments,
    )
    return resident_chunk_records.ChunkClassPosterior(
        log_z=jnp.asarray(log_z, dtype=jnp.float64),
        best_log_score=best_log_score,
        best_cell_index=best_cell_index,
    )


def winner_take_all_cells(scores, row_image_local, row_is_valid, segment_offsets, *, image_capacity: int):
    """Each image's first maximum over its segment, and the one-hot posterior on it.

    ``scores`` is ``[C_R, T]`` with ``-inf`` on invalid cells; rows are
    image-major, so a flat cell's segment-relative index is its offset from the
    image's first cell. Ties go to the smallest flat cell, as ``jnp.argmax``
    does over the compact engine's bucket rows in the same order. Returns
    ``(best_log_score [C_B], best_cell_index int64 [C_B], posterior [C_R, T])``;
    an image without a finite score has ``best_log_score = -inf``, cell 0 and
    no posterior mass.
    """

    row_capacity, n_fine_trans = scores.shape
    row_image = jnp.where(row_is_valid, row_image_local, jnp.int32(image_capacity))
    best_log_score = jax.ops.segment_max(
        jnp.max(scores, axis=1), row_image, num_segments=image_capacity + 1, indices_are_sorted=True
    )[:image_capacity]
    cell = jnp.arange(row_capacity * n_fine_trans, dtype=jnp.int64).reshape(row_capacity, n_fine_trans)
    row_best = best_log_score[jnp.minimum(row_image, image_capacity - 1)]
    is_best = row_is_valid[:, None] & jnp.isfinite(scores) & (scores == row_best[:, None])
    first_cell = jnp.min(jnp.where(is_best, cell, jnp.iinfo(jnp.int64).max), axis=1)
    best_flat = jax.ops.segment_min(
        first_cell, row_image, num_segments=image_capacity + 1, indices_are_sorted=True
    )[:image_capacity]
    has_winner = jnp.isfinite(best_log_score)
    segment_start = jnp.asarray(segment_offsets[:image_capacity], dtype=jnp.int64)
    best_cell_index = jnp.where(has_winner, best_flat - segment_start, jnp.int64(0))
    posterior = jnp.zeros((row_capacity * n_fine_trans,), dtype=jnp.float32).at[
        jnp.where(has_winner, best_flat, row_capacity * n_fine_trans)
    ].set(1.0, mode="drop")
    return best_log_score, best_cell_index, posterior.reshape(row_capacity, n_fine_trans)


def _resident_chunk_posterior_firstiter_cc(
    rows: resident_chunk_records.ChunkRowArrays,
    operands: resident_chunk_records.ChunkStageOperands,
    tables: resident_chunk_records.ChunkStageTables,
    *,
    spec: resident_chunk_records.ChunkProgramSpec,
    row_is_valid,
    kernel_row_image_ids,
) -> resident_chunk_records.ChunkPosterior:
    """RELION's ``--firstiter_cc`` iteration: normalized-CC scores, winner takes all.

    RELION zeroes every weight but the best one (ml_optimiser.cpp:9266-9293;
    acc_ml_optimiser_impl.h:2868-2960), so the posterior is one-hot at each
    image's highest CC and Pmax is 1. The winner is the first maximum in the
    image's segment order, which is the compact engine's ``jnp.argmax`` over its
    bucket rows in the same order (``_winner_take_all_bucket_probs``). log-Z is
    the log-sum-exp of the CC scores, the compact engine's reported evidence.
    """

    from relax.fine_pass.resident_scoring import score_resident_chunk_normalized_cc

    scored = score_resident_chunk_normalized_cc(
        rows.row_image_local,
        rows.row_fine_rot,
        rows.row_mask_bits,
        rows.row_mask_mode,
        rows.n_valid_rows,
        tables.projection_score_cache,
        operands.score_shifted_cc,
        operands.corr_img_score,
        operands.cc_half_batch_norm,
        half_weights=tables.half_weights,
        full_to_compact=tables.full_to_compact,
        fine_translation_parent=tables.fine_translation_parent,
        row_capacity=spec.row_capacity,
        n_fine_trans=spec.n_fine_trans,
        block_rows=int(spec.mstep_block_rows),
    )
    return winner_take_all_posterior(
        scored,
        rows,
        image_capacity=spec.image_capacity,
        row_is_valid=row_is_valid,
        kernel_row_image_ids=kernel_row_image_ids,
    )


def winner_take_all_posterior(
    scored, rows, *, image_capacity: int, row_is_valid, kernel_row_image_ids
) -> resident_chunk_records.ChunkPosterior:
    """The ``--firstiter_cc`` posterior of a chunk's normalized-CC scores (one segment per image or particle)."""

    from relax.cuda import posterior as cuda_posterior

    scores = jnp.asarray(scored.scores, dtype=jnp.float32)
    scores_flat = scores.reshape(-1)
    log_z = cuda_posterior.sparse_pass2_segmented_log_z_f64(
        scores_flat, rows.segment_offsets, rows.n_valid_images
    )

    best_log_score, best_cell_index, winner = winner_take_all_cells(
        scores,
        rows.row_image_local,
        row_is_valid,
        rows.segment_offsets,
        image_capacity=image_capacity,
    )
    has_winner = jnp.isfinite(best_log_score)
    return resident_chunk_records.ChunkPosterior(
        row_posterior=winner,
        min_diff2=scored.min_diff2,
        class_log_z=jnp.asarray(log_z, dtype=jnp.float64),
        best_log_score=best_log_score,
        best_cell_index=best_cell_index,
        max_posterior=has_winner.astype(jnp.float32),
        kernel_row_image_ids=kernel_row_image_ids,
        row_is_valid=row_is_valid,
    )


@partial(jax.jit, static_argnames=("spec",))
def resident_chunk_posterior_program(
    rows: resident_chunk_records.ChunkRowArrays,
    operands: resident_chunk_records.ChunkStageOperands,
    tables: resident_chunk_records.ChunkStageTables,
    *,
    spec: resident_chunk_records.ChunkProgramSpec,
) -> resident_chunk_records.ChunkPosterior:
    """:func:`resident_chunk_posterior` as one program per capacity class.

    The stage's own arithmetic is unchanged; what leaves the chunk loop is the
    dozen eager operations around it -- the two ``arange`` row/image masks, the
    logical current size, the score reshape and the posterior's unit external
    weight -- each of which was a dispatch and a single-primitive program.
    """

    return resident_chunk_posterior(rows, operands, tables, spec=spec)
