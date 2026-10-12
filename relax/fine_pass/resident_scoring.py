"""Device-resident scoring stage of the K=1 sparse pass-2 prototype.

Stages 1-3 of ``em_device_resident_pass2_design_20260918.md``: gather,
"project" (a gather from the per-iteration projection cache, not a projector
call) and score. The candidate rows and their capacity chunks come from
:mod:`recovar.em.sparse_pass2.resident_candidates` (T5); the per-image image /
CTF / noise operands come from
:func:`recovar.em.sparse_pass2.sparse_pass2_bucket_io._prepare_bucket_io`, run
once per half instead of once per bucket and kept on the device.

What this module changes relative to the compact engine, and what it does not
-----------------------------------------------------------------------------
Unchanged: the candidate set and its order, the per-cell CUDA diff2 arithmetic
(the same fused translate-then-score kernel body the compact engine's pairs
route calls), the RELION common-minimum score conversion including its
optimization barriers, and the float32 scoring precision.

Changed: the *layout* the kernel is called on. The compact engine dispatches
one rectangular ``(B, R, T)`` program per bucket shape; this module dispatches
one flat ``(C_R, T)`` program per capacity class, with ``row_image_ids``
carrying the row-to-image addressing. ``n_valid_rows`` / ``n_valid_images`` are
runtime int32 operands, and the RELION current size is the runtime scalar of
:func:`relax.cuda.fine.relion_fine_diff2_fused_translate_runtime_flat_rows_f32`,
so a chunk's program is keyed only on
``(row_capacity, image_capacity, n_fine_trans, n_score_pixels)``.

Padding contract
----------------
Rows at or past ``n_valid_rows`` are handed to the kernel with image id ``-1``:
the kernel writes ``+inf`` for the whole row and skips its pixel traversal, so
padding costs no arithmetic. Their candidate mask is all-false as well (T5 pads
``row_mask_mode`` with the empty mode), so the mask alone would already
invalidate them; the ``-1`` is the performance half of the same contract.
Padded image slots carry ``image_ids == -1`` and are never addressed by a valid
row.

Indexing
--------
"image" is the local position of an image inside the half's ``per_image_inputs``
list, exactly as in :mod:`resident_candidates`. Resident operand row ``i`` is
that image. Chunk-local image ids (``row_image_local``) are offsets from
``chunk.image_start``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from relax.fine_pass.projection_blocks import (
    _compute_sparse_pass2_projections_block,
    _compute_sparse_pass2_windowed_projections_block,
    _place_score_window_block,
    _place_windowed_projection_block,
    window_union_applies,
)
from relax.fine_pass.resident_candidates import (
    expand_chunk_mask_jnp,
    expand_local_chunk_mask_jnp,
)
from relax.fine_pass.scoring import (
    _relion_cuda_fine_normalized_cc_score,
    _relion_cuda_fine_pixel_weights,
)

__all__ = [
    "ResidentChunkScores",
    "live_projection_block_rows",
    "project_resident_live_rows",
    "project_resident_rows",
    "resident_row_projection_bytes",
    "score_resident_chunk",
    "score_resident_projected_chunk",
]


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["raw_diff2", "scores", "min_diff2"],
    meta_fields=[],
)
@dataclass(frozen=True)
class ResidentChunkScores:
    """Scoring outputs of one capacity chunk, all on the device.

    ``raw_diff2`` and ``scores`` are ``[row_capacity, n_fine_trans]``;
    ``min_diff2`` is ``[image_capacity]``. Invalid cells, and every cell of a
    padded row, are ``+inf`` in ``raw_diff2`` and ``-inf`` in ``scores``, the
    same invalidation the rectangular consumers already apply.

    Registered as a JAX pytree so the scoring program can return it directly
    from ``jax.jit`` instead of an unnamed tuple.
    """

    raw_diff2: jax.Array
    scores: jax.Array
    min_diff2: jax.Array


@partial(
    jax.jit,
    static_argnames=("row_capacity", "image_capacity", "n_fine_trans", "n_score_pixels", "native_fft_size"),
)
def score_resident_chunk(
    row_image_local,  # int32 [C_R] chunk-local image id of each row
    row_fine_rot,  # int32 [C_R] index into the iteration's fine rotation grid
    row_log_prior,  # float32 [C_R] rotation log prior of each row (nats)
    row_mask_bits,  # uint32 [C_R, n_mask_words] per-row coarse translation bitset
    row_mask_mode,  # int8 [C_R] 0 full, 1 bitset, 2 empty
    n_valid_rows,  # int32 scalar, runtime
    image_ids,  # int32 [C_B] resident image row of each chunk image slot, -1 when padded
    projection_score_cache,  # complex64 [n_fine_rot, N] per-iteration cache
    score_input,  # complex64 [n_images, N]
    corr_img_score,  # real [n_images, N]
    highres_xi2_half,  # float32 [n_images] or None
    translation_prior,  # real [n_images, T]
    *,
    half_weights,  # real [N]
    translation_angles,  # float32 [T, 2]
    full_to_compact,  # int32 [P]
    fine_translation_parent,  # int32 [T]
    logical_current_size,  # int32 scalar, runtime
    row_capacity: int,
    image_capacity: int,
    n_fine_trans: int,
    n_score_pixels: int,
    score_take=None,  # int32 [N] score window positions in a union-cache row, or None
    native_fft_size: int = 0,
):
    """Score one capacity chunk in a single device program.

    Gathers the chunk's images and projection rows, forms the RELION pixel
    weights, calls the flat-row fused translate-and-score CUDA kernel, expands
    the candidate mask from its bitsets, reduces the per-image common minimum
    over the chunk's rows, and applies RELION's diff2-to-log-weight conversion.

    There is no host round trip and no Python loop over rows or images inside.
    ``n_valid_rows`` and ``logical_current_size`` are traced scalars, so two
    chunks of the same capacity class with different occupancy or a different
    RELION current size reuse one compiled program.

    The per-image minimum computed here is the image's *global* minimum,
    because a capacity chunk holds a contiguous image range and an image's rows
    never straddle two chunks.

    The chunk's ``n_valid_images`` is not a separate operand: padded image
    slots are exactly the slots with ``image_ids == -1``, and no valid row ever
    addresses one.

    ``score_take`` reads a union projection cache (score and reconstruction
    windows in one row): the gathered rows keep their score window, and a
    non-zero ``native_fft_size`` then applies RELION's native-unit division
    that the three-cache build applied to its score cache, with the same
    elementwise statements.
    """

    row_image_local = jnp.asarray(row_image_local, dtype=jnp.int32)
    row_fine_rot = jnp.asarray(row_fine_rot, dtype=jnp.int32)
    image_ids = jnp.asarray(image_ids, dtype=jnp.int32)
    if row_image_local.shape != (row_capacity,):
        raise ValueError(f"row_image_local must have shape ({row_capacity},)")
    if row_fine_rot.shape != (row_capacity,):
        raise ValueError(f"row_fine_rot must have shape ({row_capacity},)")
    if image_ids.shape != (image_capacity,):
        raise ValueError(f"image_ids must have shape ({image_capacity},)")
    if int(score_input.shape[1]) != n_score_pixels:
        raise ValueError(f"score_input must have {n_score_pixels} pixels")
    if int(translation_angles.shape[0]) != n_fine_trans:
        raise ValueError(f"translation_angles must have {n_fine_trans} rows")

    # --- stage 1: gather ---------------------------------------------------
    chunk_image, chunk_corr, chunk_translation_prior, chunk_initial_diff2 = _gather_chunk_image_operands(
        image_ids, score_input, corr_img_score, highres_xi2_half, translation_prior, image_capacity=image_capacity
    )

    # --- stage 2: "project" = gather the cached fine-rotation projections ---
    reference = cached_score_reference(
        projection_score_cache, row_fine_rot, score_take=score_take, native_fft_size=native_fft_size
    )

    # --- stage 3: score ----------------------------------------------------
    row_is_valid = jnp.arange(row_capacity, dtype=jnp.int32) < jnp.asarray(
        n_valid_rows, dtype=jnp.int32
    )
    candidate_mask = expand_chunk_mask_jnp(
        row_mask_bits, row_mask_mode, fine_translation_parent
    )
    return _score_flat_rows(
        reference,
        row_image_local,
        row_log_prior,
        chunk_image,
        chunk_corr,
        chunk_translation_prior,
        chunk_initial_diff2,
        candidate_mask,
        row_is_valid,
        half_weights=half_weights,
        translation_angles=translation_angles,
        full_to_compact=full_to_compact,
        logical_current_size=logical_current_size,
        image_capacity=image_capacity,
    )


def _gather_chunk_image_operands(
    image_ids, score_input, corr_img_score, highres_xi2_half, translation_prior, *, image_capacity: int
):
    """A chunk's image slots gathered out of the half's score operands (padded slots read image 0)."""

    safe_image_ids = jnp.where(image_ids >= 0, image_ids, jnp.int32(0))
    chunk_image = jnp.asarray(score_input, dtype=jnp.complex64)[safe_image_ids]
    chunk_corr = corr_img_score[safe_image_ids]
    chunk_translation_prior = translation_prior[safe_image_ids]
    chunk_initial_diff2 = (
        jnp.zeros((image_capacity,), dtype=jnp.float32)
        if highres_xi2_half is None
        else jnp.asarray(highres_xi2_half, dtype=jnp.float32)[safe_image_ids]
    )
    return chunk_image, chunk_corr, chunk_translation_prior, chunk_initial_diff2


def cache_rows(cache, row_ids):
    """Rows ``row_ids`` of a projection cache: one array, or a tuple of equal-height row blocks.

    A whole-grid cache the allocator cannot hand out as one block is held as
    row blocks (:func:`relax.fine_pass.resident_pass2._allocate_projection_cache_blocks`);
    each block is gathered at the row's in-block position and the row's own
    block is selected, so the rows are the one-array gather's.
    """

    row_ids = jnp.asarray(row_ids, dtype=jnp.int32)
    if not isinstance(cache, tuple):
        return jnp.asarray(cache)[row_ids]
    block_height = int(cache[0].shape[0])
    block = row_ids // jnp.int32(block_height)
    local = row_ids - block * jnp.int32(block_height)
    rows = cache[0][jnp.clip(local, 0, block_height - 1)]
    for index, part in enumerate(cache[1:], start=1):
        taken = part[jnp.clip(local, 0, int(part.shape[0]) - 1)]
        rows = jnp.where((block == index)[:, None], taken, rows)
    return rows


def cache_dtype(cache):
    """The element dtype of a projection cache (:func:`cache_rows`)."""

    return (cache[0] if isinstance(cache, tuple) else cache).dtype


def cached_score_reference(projection_score_cache, row_fine_rot, *, score_take=None, native_fft_size: int = 0):
    """Rows' score projections gathered out of a projection cache, in the score window's units.

    ``score_take`` reads a union cache (score and reconstruction windows in one
    row) and a non-zero ``native_fft_size`` then applies RELION's native-unit
    division that the three-cache build applied to its score cache.
    """

    reference = cache_rows(projection_score_cache, row_fine_rot).astype(jnp.complex64)
    if score_take is not None:
        reference = reference[:, jnp.asarray(score_take, dtype=jnp.int32)]
        if native_fft_size:
            from relax.fine_pass.scoring import _relion_native_fine_units

            reference = _relion_native_fine_units(reference, int(native_fft_size))
    return reference


def score_resident_chunk_in_row_blocks(
    block_reference,  # callable: row start -> complex64 [block_rows, N] the block's score projections
    row_image_local,  # int32 [C_R]
    row_log_prior,  # float32 [C_R]
    candidate_mask,  # bool [C_R, T], or None for "every cell is a candidate"
    n_valid_rows: int,  # host int: the rows past it are padding
    image_ids,  # int32 [C_B]
    score_input,
    corr_img_score,
    highres_xi2_half,
    translation_prior,
    *,
    block_rows: int,
    half_weights,
    translation_angles,
    full_to_compact,
    logical_current_size,
    row_capacity: int,
    image_capacity: int,
    n_fine_trans: int,
):
    """:func:`score_resident_chunk` for a chunk whose rows' projections do not fit at once.

    A single image past the largest row class (EMPIAR-10202-like flat posteriors:
    every one of 294,912 fine rotations a candidate, bench 14641043) would hold
    every row's projection. Here each block of ``block_rows`` rows is projected
    (``block_reference``), scored by the same fused kernel and dropped; only the
    kernel's ``[C_R, T]`` diff2 is kept, and the per-image common minimum and the
    log-weight conversion then run once over all rows, exactly as for a chunk
    scored in one call. The kernel's per-row output does not depend on the other
    rows of its call, so the scores are those of the one-call chunk.

    The global pass (cached projections, parent-translation masks) and local
    search (projected rows, per-row translation masks) share it: each caller
    expands its own ``candidate_mask`` and supplies its own ``block_reference``.
    """

    row_image_local = jnp.asarray(row_image_local, dtype=jnp.int32)
    image_ids = jnp.asarray(image_ids, dtype=jnp.int32)
    chunk_image, chunk_corr, chunk_translation_prior, chunk_initial_diff2 = _gather_chunk_image_operands(
        image_ids, score_input, corr_img_score, highres_xi2_half, translation_prior, image_capacity=image_capacity
    )
    row_is_valid = jnp.arange(row_capacity, dtype=jnp.int32) < jnp.int32(int(n_valid_rows))
    kernel_row_image_ids = jnp.where(row_is_valid, row_image_local, jnp.int32(-1))
    raw_blocks = []
    for start in range(0, int(row_capacity), int(block_rows)):
        stop = min(start + int(block_rows), int(row_capacity))
        if start >= int(n_valid_rows):
            # Padded rows only: the kernel writes +inf for them without reading a projection.
            raw_blocks.append(jnp.full((stop - start, int(n_fine_trans)), jnp.inf, dtype=jnp.float32))
            continue
        block_mask = None if candidate_mask is None else candidate_mask[start:stop]
        raw_blocks.append(
            _flat_rows_kernel_diff2(
                block_reference(start)[: stop - start],
                kernel_row_image_ids[start:stop],
                chunk_image,
                chunk_corr,
                chunk_initial_diff2,
                half_weights=half_weights,
                translation_angles=translation_angles,
                full_to_compact=full_to_compact,
                logical_current_size=logical_current_size,
                translation_chunk_live=_translation_chunk_live(block_mask, row_is_valid[start:stop]),
            )
        )
    return _flat_rows_scores(
        jnp.concatenate(raw_blocks, axis=0),
        row_image_local,
        jnp.asarray(row_log_prior),
        chunk_translation_prior,
        candidate_mask,
        row_is_valid,
        segment_capacity=image_capacity,
    )


def _score_flat_rows(
    reference,  # complex64 [C_R, N] the rows' projections, cached or projected
    row_image_local,  # int32 [C_R]
    row_log_prior,  # real [C_R]
    chunk_image,  # complex64 [C_B, N]
    chunk_corr,  # real [C_B, N]
    chunk_translation_prior,  # real [C_B, T]
    chunk_initial_diff2,  # float32 [C_B]
    candidate_mask,  # bool [C_R, T] or None for "every cell is a candidate"
    row_is_valid,  # bool [C_R]
    *,
    half_weights,
    translation_angles,
    full_to_compact,
    logical_current_size,
    image_capacity: int,
):
    """RELION's flat-row diff2, common minimum and log-weight conversion.

    Shared by the cached-projection route (the global pass 2, which gathers a
    per-iteration fine-rotation cache) and the projected-row route (local
    search, whose order-5 fine grid cannot be cached). Only the source of
    ``reference`` and the shape of the candidate mask differ between them;
    every statement below is common, so the two routes cannot drift apart.
    """

    # Padded rows are handed to the kernel as image -1: it writes +inf for the
    # whole row and skips the pixel traversal. With a candidate mask, chunks of
    # translations without a candidate are skipped the same way: RELION scores
    # only the significant (rotation, translation) pairs, and the cells skipped
    # here are masked to +inf by _flat_rows_scores whatever their value.
    kernel_row_image_ids = jnp.where(row_is_valid, row_image_local, jnp.int32(-1))
    raw_from_kernel = _flat_rows_kernel_diff2(
        reference,
        kernel_row_image_ids,
        chunk_image,
        chunk_corr,
        chunk_initial_diff2,
        half_weights=half_weights,
        translation_angles=translation_angles,
        full_to_compact=full_to_compact,
        logical_current_size=logical_current_size,
        translation_chunk_live=_translation_chunk_live(candidate_mask, row_is_valid),
    )
    return _flat_rows_scores(
        raw_from_kernel,
        row_image_local,
        row_log_prior,
        chunk_translation_prior,
        candidate_mask,
        row_is_valid,
        segment_capacity=image_capacity,
    )


@jax.jit
def tilt_image_rows_raw_diff2(
    slot_projections,  # complex64 [S_b * C_R, N] the visited slots' projections, slot-major (slot local to slot_image_ids)
    slot_image_ids,  # int32 [S_b, C_R] the visited slots' chunk-local images of each row
    chunk_image,
    chunk_corr,
    chunk_initial_diff2,
    candidate_mask,
    row_is_valid,
    *,
    half_weights,
    image_translation_angles,
    full_to_compact,
    logical_current_size,
    running=None,  # float32 [C_R, T] the earlier slots' sum, or None (zeros)
):
    """The rows' float32 diff2 summed over the visited slots, in slot order, added to ``running``.

    One program per shape: every chunk and slot block of a pass reuses it (an eager scan over a
    per-call closure compiled again on every call).
    """

    slot_image_ids = jnp.asarray(slot_image_ids, dtype=jnp.int32)
    row_capacity = int(slot_image_ids.shape[1])
    chunk_live = _translation_chunk_live(candidate_mask, row_is_valid)

    def add_slot(running, slot_and_images):
        slot, image_ids = slot_and_images
        kernel_ids = jnp.where(row_is_valid & (image_ids >= 0), image_ids, jnp.int32(-1))
        raw = _flat_rows_kernel_diff2(
            jax.lax.dynamic_slice_in_dim(slot_projections, slot * row_capacity, row_capacity, axis=0),
            kernel_ids,
            chunk_image,
            chunk_corr,
            chunk_initial_diff2,
            half_weights=half_weights,
            translation_angles=image_translation_angles,
            full_to_compact=full_to_compact,
            logical_current_size=logical_current_size,
            translation_chunk_live=chunk_live,
        )
        return jnp.where((kernel_ids >= 0)[:, None], running + raw, running), None

    if running is None:
        running = jnp.zeros((row_capacity, int(image_translation_angles.shape[1])), dtype=jnp.float32)
    slots = jnp.arange(slot_image_ids.shape[0], dtype=jnp.int32)
    raw_sum, _ = jax.lax.scan(add_slot, running, (slots, slot_image_ids))
    return raw_sum


def tilt_rows_scores_from_raw(
    raw_sum, row_unit_local, row_log_prior, unit_translation_prior, candidate_mask, row_is_valid, *, unit_capacity: int
):
    """The particles' flat-row scores from their rows' diff2 summed over every tilt image."""

    return _flat_rows_scores(
        raw_sum,
        jnp.asarray(row_unit_local, dtype=jnp.int32),
        row_log_prior,
        unit_translation_prior,
        candidate_mask,
        row_is_valid,
        segment_capacity=unit_capacity,
    )


def _translation_chunk_live(candidate_mask, row_is_valid):
    """The kernel's live translation chunks of a candidate mask, or None to score every chunk."""

    if candidate_mask is None:
        return None
    from relax.cuda import fine as cuda_fine

    return cuda_fine.relion_fine_diff2_translation_chunk_live(candidate_mask & row_is_valid[:, None])


def _flat_rows_kernel_diff2(
    reference,  # complex64 [C_R, N]
    kernel_row_image_ids,  # int32 [C_R], -1 for a row the kernel skips (+inf)
    chunk_image,  # complex64 [C_B, N]
    chunk_corr,  # real [C_B, N]
    chunk_initial_diff2,  # float32 [C_B]
    *,
    half_weights,
    translation_angles,  # float32 [T, 2], or [C_B, T, 2] per image (tilt images)
    full_to_compact,
    logical_current_size,
    translation_chunk_live=None,  # uint8 [C_R, ceil(T / 4)] or None: chunks the kernel scores
):
    """RELION's fine diff2 of every flat row against its image: the flat-row CUDA kernel."""

    from relax.cuda import fine as cuda_fine

    weights = _relion_cuda_fine_pixel_weights(
        chunk_corr, jnp.asarray(half_weights)[None, :]
    ).astype(jnp.float32)
    return cuda_fine.relion_fine_diff2_fused_translate_runtime_flat_rows_f32(
        reference,
        kernel_row_image_ids,
        chunk_image,
        jnp.asarray(translation_angles, dtype=jnp.float32),
        weights,
        jnp.asarray(full_to_compact, dtype=jnp.int32),
        jnp.asarray(logical_current_size, dtype=jnp.int32),
        chunk_initial_diff2,
        translation_chunk_live=translation_chunk_live,
    )


def _flat_rows_scores(
    raw_from_kernel,  # float32 [C_R, T]
    row_image_local,  # int32 [C_R] the row's posterior segment (image; particle for tilt images)
    row_log_prior,  # real [C_R]
    chunk_translation_prior,  # real [C_S, T] per segment
    candidate_mask,  # bool [C_R, T] or None
    row_is_valid,  # bool [C_R]
    *,
    segment_capacity: int,
):
    """RELION's per-segment common minimum and diff2-to-log-weight conversion of flat rows."""

    if candidate_mask is None:
        candidate_mask = jnp.broadcast_to(row_is_valid[:, None], raw_from_kernel.shape)
    else:
        candidate_mask = candidate_mask & row_is_valid[:, None]
    raw_diff2 = jnp.where(
        candidate_mask,
        raw_from_kernel,
        jnp.asarray(jnp.inf, dtype=raw_from_kernel.dtype),
    )

    # --- per-segment common minimum over the chunk's rows ------------------
    valid = candidate_mask & jnp.isfinite(raw_diff2)
    per_row_min = jnp.min(
        jnp.where(valid, raw_diff2, jnp.asarray(jnp.inf, dtype=raw_diff2.dtype)), axis=1
    )
    segment_min = jax.ops.segment_min(
        per_row_min,
        row_image_local,
        num_segments=segment_capacity,
        indices_are_sorted=True,
    )
    min_diff2 = jnp.where(
        jnp.isfinite(segment_min), segment_min, jnp.asarray(0.0, dtype=segment_min.dtype)
    )

    # --- RELION diff2 -> log weight, in the flat-row layout ----------------
    # Same operation order as ``scripts/lib/bucketed_pass2_scoring._relion_cuda_fine_diff2_to_scores``: the two
    # optimization barriers keep ``(prior + min) - diff2`` from being
    # re-associated, which changes float32 tie-breaking at diff2 ~ 1e3.
    real_dtype = raw_diff2.dtype
    row_min = min_diff2[row_image_local].astype(real_dtype)
    valid = valid & (raw_diff2 >= row_min[:, None])
    scores = jnp.asarray(row_log_prior, dtype=real_dtype)[:, None] + jnp.asarray(
        chunk_translation_prior, dtype=real_dtype
    )[row_image_local]
    scores = jax.lax.optimization_barrier(scores)
    scores = scores + row_min[:, None]
    scores = jax.lax.optimization_barrier(scores)
    scores = scores - raw_diff2
    scores = jnp.where(valid & jnp.isfinite(scores), scores, -jnp.inf)

    return ResidentChunkScores(raw_diff2=raw_diff2, scores=scores, min_diff2=min_diff2)


def _normalized_cc_row_scores(
    projection_rows,
    row_image_local,
    row_projection_index,
    candidate_mask,
    score_shifted_cc,
    cc_score_weight,
    cc_half_batch_norm,
    *,
    half_weights,
    full_to_compact,
    row_capacity: int,
    n_fine_trans: int,
    block_rows: int,
):
    """RELION's fine normalized-CC scores of a chunk's rows, in row blocks.

    RELION's 256-lane ``cuda_kernel_diff2_CC_fine`` numerator and
    reference-norm reduction (ml_optimiser.cpp:8844-8858), shared through
    :func:`_relion_cuda_fine_normalized_cc_score`, evaluated in row blocks so
    the ``[rows, T, N]`` product never exists for a whole chunk. Row ``r`` reads
    its projection at ``projection_rows[row_projection_index[r]]``. Like the
    compact engine it adds no rotation or translation prior.

    ``min_diff2`` carries ``0.5 * |image|^2``, so the evidence offset the
    statistics stage applies (``-min_diff2``) is the compact engine's
    normalized-CC offset ``-0.5 * batch_norm``.
    """

    if row_capacity % block_rows:
        raise ValueError(f"block_rows {block_rows} must divide the row capacity {row_capacity}")
    row_image_local = jnp.asarray(row_image_local, dtype=jnp.int32)
    row_projection_index = jnp.asarray(row_projection_index, dtype=jnp.int32)

    def block_scores(start):
        images = jax.lax.dynamic_slice_in_dim(row_image_local, start, block_rows)
        rows = jax.lax.dynamic_slice_in_dim(row_projection_index, start, block_rows)
        return _normalized_cc_block_scores(
            projection_rows[rows], images, score_shifted_cc, cc_score_weight,
            half_weights=half_weights, full_to_compact=full_to_compact,
        )

    starts = jnp.arange(0, row_capacity, block_rows, dtype=jnp.int32)
    scores = jax.lax.map(block_scores, starts).reshape(row_capacity, n_fine_trans)
    return _normalized_cc_chunk_scores(scores, candidate_mask, cc_half_batch_norm)


@jax.jit
def _normalized_cc_block_scores(
    block_projections,  # complex [B, N] the block rows' score projections
    block_images,  # int32 [B] chunk-local image id of each block row
    score_shifted_cc,
    cc_score_weight,
    *,
    half_weights,
    full_to_compact,
):
    """One row block's ``[B, T]`` normalized-CC scores: the block core every CC chunk scorer shares."""

    return _relion_cuda_fine_normalized_cc_score(
        block_projections[:, None, :],
        score_shifted_cc[block_images],
        cc_score_weight[block_images][:, None, :],
        half_weights,
        full_to_compact,
    )


@jax.jit
def _normalized_cc_chunk_scores(scores, candidate_mask, cc_half_batch_norm):
    """A chunk's ``[C_R, T]`` normalized-CC block scores masked to its candidates, as the chunk's scores."""

    scores = jnp.where(candidate_mask & jnp.isfinite(scores), scores, -jnp.inf)
    return ResidentChunkScores(
        raw_diff2=jnp.where(jnp.isfinite(scores), -scores, jnp.inf),
        scores=scores,
        min_diff2=jnp.asarray(cc_half_batch_norm),
    )


def _local_cc_candidate_mask(row_mask_bits, n_valid_rows, *, row_capacity: int, n_fine_trans: int):
    """A local chunk's ``[C_R, T]`` CC candidates: the layout's per-row packing, valid rows only."""

    row_is_valid = jnp.arange(row_capacity, dtype=jnp.int32) < jnp.asarray(n_valid_rows, dtype=jnp.int32)
    candidate_mask = expand_local_chunk_mask_jnp(row_mask_bits, n_trans=n_fine_trans)
    valid = row_is_valid[:, None] if candidate_mask is None else candidate_mask & row_is_valid[:, None]
    return jnp.broadcast_to(valid, (row_capacity, n_fine_trans))


@partial(
    jax.jit,
    static_argnames=("row_capacity", "n_fine_trans", "block_rows"),
)
def score_resident_chunk_normalized_cc(
    row_image_local,  # int32 [C_R] chunk-local image id of each row
    row_fine_rot,  # int32 [C_R] index into the iteration's fine rotation grid
    row_mask_bits,  # uint32 [C_R, n_mask_words] per-row coarse translation bitset
    row_mask_mode,  # int8 [C_R] 0 full, 1 bitset, 2 empty
    n_valid_rows,  # int32 scalar, runtime
    projection_score_cache,  # complex [n_fine_rot, N] per-iteration cache
    score_shifted_cc,  # complex [C_B, T, N] translated corrected score tile
    cc_score_weight,  # real [C_B, N] the CC pixel weight (the compact engine's ctf2_over_nv_score)
    cc_half_batch_norm,  # real [C_B] 0.5 * |image|^2
    *,
    half_weights,  # real [N]
    full_to_compact,  # int32 [P]
    fine_translation_parent,  # int32 [T]
    row_capacity: int,
    n_fine_trans: int,
    block_rows: int,
):
    """RELION's ``--firstiter_cc`` fine normalized-CC scores of one global chunk.

    The compact engine's ``_score_pass2_bucket_relion_gpu_normalized_cc`` on
    flat rows (:func:`_normalized_cc_row_scores`), each row's projection
    gathered from the per-iteration cache by its fine rotation.
    """

    row_is_valid = jnp.arange(row_capacity, dtype=jnp.int32) < jnp.asarray(n_valid_rows, dtype=jnp.int32)
    candidate_mask = expand_chunk_mask_jnp(row_mask_bits, row_mask_mode, fine_translation_parent)
    return _normalized_cc_row_scores(
        projection_score_cache,
        row_image_local,
        row_fine_rot,
        candidate_mask & row_is_valid[:, None],
        score_shifted_cc,
        cc_score_weight,
        cc_half_batch_norm,
        half_weights=half_weights,
        full_to_compact=full_to_compact,
        row_capacity=row_capacity,
        n_fine_trans=n_fine_trans,
        block_rows=block_rows,
    )


@partial(
    jax.jit,
    static_argnames=("row_capacity", "n_fine_trans", "block_rows"),
)
def score_resident_projected_chunk_normalized_cc(
    reference,  # complex64 [C_R, N] projections of this chunk's own rows
    row_image_local,  # int32 [C_R] chunk-local image id of each row
    row_mask_bits,  # uint8 [C_R, ceil(T/8)] little-endian, or None for full support
    n_valid_rows,  # int32 scalar, runtime
    score_shifted_cc,  # complex [C_B, T, N] translated corrected score tile
    cc_score_weight,  # real [C_B, N] the CC pixel weight
    cc_half_batch_norm,  # real [C_B] 0.5 * |image|^2
    *,
    half_weights,  # real [N]
    full_to_compact,  # int32 [P]
    row_capacity: int,
    n_fine_trans: int,
    block_rows: int,
):
    """RELION's ``--firstiter_cc`` normalized-CC scores of a local-search chunk.

    The local twin of :func:`score_resident_chunk_normalized_cc`, as
    :func:`score_resident_projected_chunk` is of :func:`score_resident_chunk`:
    the rows arrive projected (row ``r`` is ``reference[r]``) and the candidate
    mask is the local layout's per-row packing over the fine translations.
    RELION scores a local search's first iteration with the same CC and keeps
    only the best hidden variable (ml_optimiser.cpp:9266-9292).
    """

    return _normalized_cc_row_scores(
        reference,
        row_image_local,
        jnp.arange(row_capacity, dtype=jnp.int32),
        _local_cc_candidate_mask(row_mask_bits, n_valid_rows, row_capacity=row_capacity, n_fine_trans=n_fine_trans),
        score_shifted_cc,
        cc_score_weight,
        cc_half_batch_norm,
        half_weights=half_weights,
        full_to_compact=full_to_compact,
        row_capacity=row_capacity,
        n_fine_trans=n_fine_trans,
        block_rows=block_rows,
    )


def score_resident_chunk_normalized_cc_in_row_blocks(
    block_reference,  # callable: row start -> complex [<= block_rows, N] the block's score projections
    row_image_local,  # int32 [C_R] chunk-local image id of each row
    row_mask_bits,  # uint8 [C_R, ceil(T/8)] little-endian, or None for full support
    n_valid_rows: int,  # host int: the rows past it are padding
    score_shifted_cc,  # complex [C_B, T, N] translated corrected score tile
    cc_score_weight,  # real [C_B, N] the CC pixel weight
    cc_half_batch_norm,  # real [C_B] 0.5 * |image|^2
    *,
    block_rows: int,
    half_weights,  # real [N]
    full_to_compact,  # int32 [P]
    row_capacity: int,
    n_fine_trans: int,
):
    """:func:`score_resident_projected_chunk_normalized_cc` for a chunk whose rows' projections do not fit at once.

    The normalized-CC twin of :func:`score_resident_chunk_in_row_blocks`: each block of ``block_rows`` rows is
    projected (``block_reference``), scored by the same block core (:func:`_normalized_cc_block_scores`) and
    dropped; the ``[C_R, T]`` scores are then masked once, as for a chunk scored in one call. A row's score does
    not depend on the other rows of its block, so the scores are those of the one-call chunk.
    """

    row_image_local = jnp.asarray(row_image_local, dtype=jnp.int32)
    blocks = []
    for start in range(0, int(row_capacity), int(block_rows)):
        stop = min(start + int(block_rows), int(row_capacity))
        if start >= int(n_valid_rows):
            # Padded rows only: the candidate mask drops them.
            blocks.append(jnp.full((stop - start, int(n_fine_trans)), -jnp.inf, dtype=jnp.float32))
            continue
        blocks.append(
            _normalized_cc_block_scores(
                block_reference(start)[: stop - start],
                row_image_local[start:stop],
                score_shifted_cc,
                cc_score_weight,
                half_weights=half_weights,
                full_to_compact=full_to_compact,
            )
        )
    scores = jnp.concatenate(blocks, axis=0)
    candidate_mask = _local_cc_candidate_mask(
        row_mask_bits, int(n_valid_rows), row_capacity=int(row_capacity), n_fine_trans=int(n_fine_trans)
    )
    return _normalized_cc_chunk_scores(scores, candidate_mask, cc_half_batch_norm)


@partial(
    jax.jit,
    static_argnames=("row_capacity", "image_capacity", "n_fine_trans", "n_score_pixels"),
)
def score_resident_projected_chunk(
    reference,  # complex64 [C_R, N] projections of this chunk's own rows
    row_image_local,  # int32 [C_R] chunk-local image id of each row
    row_log_prior,  # real [C_R] rotation log prior of each row (nats)
    row_mask_bits,  # uint8 [C_R, ceil(T/8)] little-endian, or None for full support
    n_valid_rows,  # int32 scalar, runtime
    image_ids,  # int32 [C_B] resident image row of each slot, -1 when padded
    score_input,  # complex64 [n_images, N]
    corr_img_score,  # real [n_images, N]
    highres_xi2_half,  # float32 [n_images] or None
    translation_prior,  # real [n_images, T]
    *,
    half_weights,  # real [N]
    translation_angles,  # float32 [T, 2]
    full_to_compact,  # int32 [P]
    logical_current_size,  # int32 scalar, runtime
    row_capacity: int,
    image_capacity: int,
    n_fine_trans: int,
    n_score_pixels: int,
):
    """Score one capacity chunk whose rows were projected, not gathered.

    The local-search twin of :func:`score_resident_chunk`: identical arithmetic
    (both call :func:`_score_flat_rows`), with two inputs replaced.
    ``reference`` arrives already projected for this chunk's own rows, because
    the local fine grid at order 5 has 2.4M rotations and is never materialized;
    and the candidate mask is the layout's per-row little-endian packing over
    the fine translation axis rather than a per-parent coarse-translation
    bitset, because local candidates are per row and per fine translation.

    ``row_mask_bits=None`` is the layout's compact spelling of full support and
    is not expanded; padded rows are excluded by ``n_valid_rows`` alone.
    """

    row_image_local = jnp.asarray(row_image_local, dtype=jnp.int32)
    image_ids = jnp.asarray(image_ids, dtype=jnp.int32)
    if row_image_local.shape != (row_capacity,):
        raise ValueError(f"row_image_local must have shape ({row_capacity},)")
    if reference.shape != (row_capacity, n_score_pixels):
        raise ValueError(
            f"reference must have shape ({row_capacity}, {n_score_pixels}), got {reference.shape}"
        )
    if image_ids.shape != (image_capacity,):
        raise ValueError(f"image_ids must have shape ({image_capacity},)")
    if int(translation_angles.shape[0]) != n_fine_trans:
        raise ValueError(f"translation_angles must have {n_fine_trans} rows")

    safe_image_ids = jnp.where(image_ids >= 0, image_ids, jnp.int32(0))
    chunk_image = jnp.asarray(score_input, dtype=jnp.complex64)[safe_image_ids]
    chunk_corr = corr_img_score[safe_image_ids]
    chunk_translation_prior = translation_prior[safe_image_ids]
    chunk_initial_diff2 = (
        jnp.zeros((image_capacity,), dtype=jnp.float32)
        if highres_xi2_half is None
        else jnp.asarray(highres_xi2_half, dtype=jnp.float32)[safe_image_ids]
    )

    row_is_valid = jnp.arange(row_capacity, dtype=jnp.int32) < jnp.asarray(
        n_valid_rows, dtype=jnp.int32
    )
    candidate_mask = expand_local_chunk_mask_jnp(row_mask_bits, n_trans=n_fine_trans)
    return _score_flat_rows(
        jnp.asarray(reference, dtype=jnp.complex64),
        row_image_local,
        row_log_prior,
        chunk_image,
        chunk_corr,
        chunk_translation_prior,
        chunk_initial_diff2,
        candidate_mask,
        row_is_valid,
        half_weights=half_weights,
        translation_angles=translation_angles,
        full_to_compact=full_to_compact,
        logical_current_size=logical_current_size,
        image_capacity=image_capacity,
    )


# ---------------------------------------------------------------------------
# Projection by rows (local search: no per-iteration fine-rotation cache)
# ---------------------------------------------------------------------------


def resident_row_projection_bytes(
    *,
    n_score_pixels: int,
    n_recon_pixels: int,
    complex_bytes: int = 8,
    real_bytes: int = 4,
) -> int:
    """Device bytes one projected row holds while its chunk is being scored.

    A row carries its score-window projection (complex), its reconstruction
    window projection (complex) and that window's ``|proj|^2`` (real). This is
    the term that decides a local chunk's row capacity, because the local route
    cannot amortize projections across chunks the way the global route's
    per-iteration cache does.
    """

    return max(int(n_score_pixels), 0) * int(complex_bytes) + max(
        int(n_recon_pixels), 0
    ) * (int(complex_bytes) + int(real_bytes))


def project_resident_rows(
    mean_for_proj,
    rotations,
    image_shape,
    proj_volume_shape,
    disc_type,
    *,
    score_indices,
    recon_indices,
    max_projected_rotations: int,
    output_complex_dtype=None,
    output_abs2_dtype=None,
    relion_projector_half=None,
    relion_projector_r_max: int | None = None,
    projection_padding_factor: int = 1,
    window_union=None,
    **projection_kwargs,
):
    """Project a chunk's own rows into the score and reconstruction windows.

    Stage 2 of the design for local search. The compact/global route gathers
    ``projection_cache["score"][row_fine_rot]``; the local fine grid has up to
    2.4M rotations at order 5 and is never cached, so a chunk's rows are
    projected here instead, in blocks of ``max_projected_rotations`` rows so the
    projector's transient stays inside its byte budget. The blocking is the
    compact engine's own
    :func:`_compute_sparse_pass2_windowed_projections_block`, so the projector
    call, its window gathers and the ``|proj|^2`` formation are the same code
    the cache build uses, only driven by row rotations instead of grid rotations.

    Returns ``(score_proj [Q, N_score], recon_proj [Q, N_recon],
    recon_abs2 [Q, N_recon])``.
    """

    return _compute_sparse_pass2_windowed_projections_block(
        mean_for_proj,
        rotations,
        image_shape,
        proj_volume_shape,
        disc_type,
        score_indices=score_indices,
        recon_indices=recon_indices,
        max_projected_rotations=int(max_projected_rotations),
        output_complex_dtype=output_complex_dtype,
        output_abs2_dtype=output_abs2_dtype,
        relion_projector_half=relion_projector_half,
        relion_projector_r_max=relion_projector_r_max,
        projection_padding_factor=projection_padding_factor,
        window_union=window_union,
        **projection_kwargs,
    )


# The live-row quantum is a sixteenth of the row capacity, no smaller than this
# many rows: a chunk projects at most capacity / 16 rows past its valid ones.
_MIN_LIVE_PROJECTION_ROWS = 256


def live_projection_block_rows(row_capacity: int, max_projected_rotations: int) -> int:
    """Rows per projector call when only a chunk's valid rows are projected.

    A power of two dividing ``row_capacity`` (the local capacity ladder is
    powers of two), at most ``max_projected_rotations`` and ``row_capacity /
    16`` but at least :data:`_MIN_LIVE_PROJECTION_ROWS`, so every call of a
    capacity class has one shape and the unused tail is at most one block.
    Any other capacity projects in one block if that is within
    ``max_projected_rotations``, and otherwise in blocks of the largest power
    of two dividing it: an overflow chunk's capacity is a multiple of the
    largest row class, and projected in one call a 1536-row EMPIAR-10202
    final-pass chunk held a 2.88 GiB crop past the plan's 609-row projector
    call (bigbox 14684083).
    """

    max_rows = max(int(max_projected_rotations), 1)
    if row_capacity <= 0:
        return 1
    if row_capacity & (row_capacity - 1):
        if row_capacity <= max_rows:
            return row_capacity
        block = row_capacity & -row_capacity
    else:
        block = max(row_capacity // 16, _MIN_LIVE_PROJECTION_ROWS)
    while block > max_rows:
        block //= 2
    return max(min(block, row_capacity), 1)


def project_resident_live_rows(
    mean_for_proj,
    rotations,
    image_shape,
    proj_volume_shape,
    disc_type,
    *,
    n_valid_rows: int,
    score_indices,
    recon_indices,
    max_projected_rotations: int,
    output_complex_dtype,
    output_abs2_dtype,
    relion_projector_half=None,
    relion_projector_r_max: int | None = None,
    projection_padding_factor: int = 1,
    relion_projector_capacity_texture=None,
    window_union=None,
    place_rows=None,
    outputs=None,
    **projection_kwargs,
):
    """:func:`project_resident_rows` for a padded chunk: only the valid rows are projected.

    A local chunk's rows past ``n_valid_rows`` are padding (identity rotations,
    image id ``-1``, zero posterior): the scoring kernel skips them and the
    M-step walks live rows only, so their projections are never read. The rows
    ``[0, n_valid_rows)`` are projected in blocks of
    :func:`live_projection_block_rows` and written into zero-filled
    ``[row_capacity, ...]`` outputs; each projected row is the projector call
    :func:`project_resident_rows` makes for it. At the 10097 local iterations a
    chunk's valid rows are about 30% of its capacity.
    ``relion_projector_capacity_texture`` (a
    :class:`~relax.cuda.projector.RelionCapacityHalfTextureF32` of
    ``relion_projector_half``) projects without restaging the texture per call.
    ``window_union`` (:func:`projection_window_union` of the two windows) has
    the RELION projector compute only the windows' pixels, as
    :func:`project_resident_rows` does with it. ``recon_indices=None`` projects
    the score window alone (the pass-1 parent probe, which has no M-step);
    ``recon_proj`` and ``recon_abs2`` are then ``None``.

    Class3D local chunks project each class's rows with that class's reference:
    ``rotations`` are then one class's rows gathered to the front, ``place_rows``
    (int32, ascending, ``rotations``' length) the chunk rows they belong to, with
    entries past the chunk's capacity for padding (dropped), and ``outputs`` the
    ``(score_proj, recon_proj, recon_abs2)`` the earlier classes filled (donated).

    Returns ``(score_proj, recon_proj, recon_abs2, n_projected_rows)``.
    """

    row_capacity = int(rotations.shape[0])
    block = live_projection_block_rows(row_capacity, max_projected_rotations)
    n_projected = min(-(-max(int(n_valid_rows), 1) // block) * block, row_capacity)
    score_indices = jnp.asarray(score_indices, dtype=jnp.int32)
    recon_indices = None if recon_indices is None else jnp.asarray(recon_indices, dtype=jnp.int32)
    projection_kwargs = dict(projection_kwargs)
    projection_kwargs["return_abs2"] = False

    if outputs is not None:
        score_proj, recon_proj, recon_abs2 = outputs
    else:
        score_proj = jnp.zeros((row_capacity, int(score_indices.shape[0])), dtype=output_complex_dtype)
        recon_proj = recon_abs2 = None
        if recon_indices is not None:
            recon_proj = jnp.zeros((row_capacity, int(recon_indices.shape[0])), dtype=output_complex_dtype)
            recon_abs2 = jnp.zeros((row_capacity, int(recon_indices.shape[0])), dtype=output_abs2_dtype)
    pixel_indices = None
    if window_union_applies(
        window_union, relion_projector=relion_projector_half is not None, projection_kwargs=projection_kwargs
    ):
        if (window_union.recon_take is None) != (recon_indices is None):
            raise ValueError("the window union does not describe these windows")
        pixel_indices = window_union.indices
        score_indices, recon_indices = window_union.score_take, window_union.recon_take
    for start in range(0, n_projected, block):
        proj_block, _ = _compute_sparse_pass2_projections_block(
            mean_for_proj,
            rotations[start : start + block],
            image_shape,
            proj_volume_shape,
            disc_type,
            max_projected_rotations=None,
            relion_projector_half=relion_projector_half,
            relion_projector_r_max=relion_projector_r_max,
            projection_padding_factor=projection_padding_factor,
            relion_projector_capacity_texture=relion_projector_capacity_texture,
            pixel_indices=pixel_indices,
            **projection_kwargs,
        )
        rows = (
            jnp.arange(start, start + int(proj_block.shape[0]), dtype=jnp.int32)
            if place_rows is None
            else jnp.asarray(place_rows[start : start + int(proj_block.shape[0])], dtype=jnp.int32)
        )
        if recon_indices is None:
            score_proj = _place_score_window_block(
                score_proj,
                proj_block,
                score_indices,
                rows,
                output_complex_dtype=output_complex_dtype,
                zero_column=pixel_indices is not None,
            )
        else:
            score_proj, recon_proj, recon_abs2 = _place_windowed_projection_block(
                score_proj,
                recon_proj,
                recon_abs2,
                proj_block,
                score_indices,
                recon_indices,
                rows,
                output_complex_dtype=output_complex_dtype,
                output_abs2_dtype=output_abs2_dtype,
                zero_column=pixel_indices is not None,
            )
        del proj_block
    return score_proj, recon_proj, recon_abs2, n_projected


