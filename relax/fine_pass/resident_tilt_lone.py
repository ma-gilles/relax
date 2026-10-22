"""A subtomogram particle run alone in row blocks whose full posterior arrays do not fit the device (relax#118).

At the first order-3 iteration of a real tilt-series case one particle can keep every fine rotation (294,912
rows x 1,176 translations, 3.5e8 cells). Run alone in row blocks (:func:`~relax.fine_pass.resident_tilts.run_tilt_chunk`
with ``lone_block_rows``) it still holds about 33 bytes per cell (its diff2, scores and the full segmented
posterior), 11 GiB, which a 16 GB card refuses. The lean form of that chunk holds 13: its scores, built in place
a block of rows at a time (:func:`lean_lone_scores`), and the reconstruction-only posterior handler's three
arrays (``resident_chunk_posterior.chunk_posterior_from_scores(reconstruction_only=True)``). Its scores and
posterior are the full form's: every cell is computed by the same operations, and the particle's minimum,
normalization and significance cut are still taken over all of its cells at once.

resident_pass2 plans it only when the full form does not fit (``resident_tilts.tilt_lone_row_bytes`` with
``lean``), so a run that fits keeps the full form.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp


@functools.partial(jax.jit, donate_argnums=0)
def _place_rows(full, block, row_start):
    # The block's rows written into ``full`` in place (the buffer is donated).
    return jax.lax.dynamic_update_slice(full, block, (row_start, jnp.int32(0)))


def lean_lone_scores(
    raw_diff2_of_block,
    *,
    row_capacity: int,
    block_rows: int,
    n_valid_rows: int,
    n_fine_trans: int,
    row_unit_local,
    row_log_prior,
    unit_translation_prior,
    candidate_mask,
    row_is_valid,
    unit_capacity: int,
):
    """A lone particle's scores, its rows scored ``block_rows`` at a time into one ``[row_capacity, T]`` buffer.

    ``raw_diff2_of_block(row_start)`` is the block's diff2 summed over every tilt image (``run_tilt_chunk``'s
    ``raw_diff2_of_rows``). Each block's diff2 is written into the buffer and gives its rows' minima; with the
    particle's minimum known (:func:`~relax.fine_pass.resident_scoring.flat_rows_segment_min_diff2`, exact in any
    grouping), each block's scores replace its diff2 in place. The scores are
    :func:`~relax.fine_pass.resident_scoring.tilt_rows_scores_from_raw`'s of the concatenated diff2, holding one
    float32 per cell instead of three. Padding rows past ``n_valid_rows`` are never scored, as in the full form.
    Returns a ``ResidentChunkScores`` without ``raw_diff2``.
    """

    from relax.fine_pass.resident_scoring import (
        ResidentChunkScores,
        flat_rows_segment_min_diff2,
        tilt_row_block_min_diff2,
        tilt_row_block_scores,
    )

    if int(row_capacity) % block_rows:
        raise ValueError(f"lone row blocks of {block_rows} do not divide the particle's {row_capacity} rows")
    starts = range(0, int(row_capacity), block_rows)
    row_unit_local = jnp.asarray(row_unit_local, dtype=jnp.int32)
    full = jnp.zeros((int(row_capacity), int(n_fine_trans)), dtype=jnp.float32)
    row_minima = []
    for row_start in starts:
        rows_of = slice(row_start, row_start + block_rows)
        if row_start >= int(n_valid_rows):
            # Padding rows: their running diff2 is the zeros it starts from, masked to +inf by row_is_valid.
            raw = jnp.zeros((block_rows, int(n_fine_trans)), dtype=jnp.float32)
        else:
            raw = raw_diff2_of_block(row_start)
            full = _place_rows(full, raw, jnp.int32(row_start))
        row_minima.append(tilt_row_block_min_diff2(raw, candidate_mask[rows_of], row_is_valid[rows_of]))
        del raw
    min_diff2 = flat_rows_segment_min_diff2(
        jnp.concatenate(row_minima), row_unit_local, segment_capacity=int(unit_capacity)
    )
    del row_minima
    for row_start in starts:
        rows_of = slice(row_start, row_start + block_rows)
        block_scores = tilt_row_block_scores(
            full[rows_of],
            min_diff2,
            row_unit_local[rows_of],
            jnp.asarray(row_log_prior)[rows_of],
            unit_translation_prior,
            candidate_mask[rows_of],
            row_is_valid[rows_of],
        )
        full = _place_rows(full, block_scores, jnp.int32(row_start))
        del block_scores
    return ResidentChunkScores(raw_diff2=None, scores=full, min_diff2=min_diff2)

