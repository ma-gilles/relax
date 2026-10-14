"""A resident chunk's stages in order: posterior, M-step blocks, then the chunk's statistics.

The front (posterior and the M-step row order) is enqueued without a host read, so the global loop can
overlap it with the previous chunk's M-step; the finish walks the live blocks of each accumulator slot, folds
each class's Wavg scale sums, and accumulates the chunk's image-level statistics.
"""

from __future__ import annotations

from dataclasses import replace as dataclass_replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from relax.fine_pass import (
    resident_block_programs,
    resident_chunk_posterior,
    resident_chunk_records,
    resident_image_terms,
    resident_mstep_blocks,
)
from relax.reconstruction.half_volume_mstep import RowPartial


@jax.jit
def _fold_class_scale_sums_program(triplet, scale_xa_per_image, scale_aa_per_image, class_masks_rect, class_index):
    """The fold's arithmetic on the three arrays it changes: ``(triplet, scale_xa, scale_aa)``."""

    mask = jnp.asarray(class_masks_rect[class_index], dtype=bool).reshape(1, -1)
    zero = jnp.float32(0.0)
    xa = jnp.sum(jnp.where(mask, triplet[:, :, 0], zero).astype(jnp.float64), axis=1)
    aa = jnp.sum(jnp.where(mask, triplet[:, :, 1], zero).astype(jnp.float64), axis=1)
    return triplet.at[:, :, :2].set(zero), scale_xa_per_image + xa, scale_aa_per_image + aa


def fold_class_scale_sums(mstep: "resident_chunk_records.ChunkMstepCarry", class_masks_rect, class_index) -> "resident_chunk_records.ChunkMstepCarry":
    """Move one class's Wavg XA/AA into the per-image scale sums under its own mask.

    ``class_masks_rect`` is every class's mask and ``class_index`` a device (or
    traced) index: selecting the row inside the program keeps the per-slot
    selection out of eager dispatch, one program for every class.

    RELION keeps XA and AA per class and adds them to the particle's scale sums
    only where that class's ``data_vs_prior_class > 3``
    (acc_ml_optimiser_impl.h:4893-4912); the diff2 channel is summed over
    classes. The class's blocks have just accumulated its XA/AA pixels, so they
    are masked and summed here and the two channels cleared for the next class.

    Only the triplet and the two scale vectors pass through the program; the rest of the carry is
    rebuilt around them. A program that took and returned the whole carry copied the slot's
    BPref accumulators (Ft_y and Ft_ctf) at every call: 2.37 GiB at a multi-optics K=2 Class3D
    slot of current size 114, which ran a 16 GB card out of memory (relax#36).
    """

    triplet, scale_xa, scale_aa = _fold_class_scale_sums_program(
        mstep.wavg_triplet_pixels, mstep.scale_xa_per_image, mstep.scale_aa_per_image, class_masks_rect, class_index
    )
    return mstep._replace(wavg_triplet_pixels=triplet, scale_xa_per_image=scale_xa, scale_aa_per_image=scale_aa)


@partial(jax.jit, static_argnames=("spec",), donate_argnums=(0,))
def resident_chunk_statistics_program(
    stats,
    rows: resident_chunk_records.ChunkRowArrays,
    operands: resident_chunk_records.ChunkStageOperands,
    tables: resident_chunk_records.ChunkStageTables,
    posterior: resident_chunk_records.ChunkPosterior,
    mstep: resident_chunk_records.ChunkMstepCarry,
    *,
    spec: resident_chunk_records.ChunkProgramSpec,
):
    """The chunk's Wavg image power, then :func:`_resident_chunk_statistics`, as one program per capacity class.

    The image power (:func:`add_chunk_wavg_image_power`) is read only by the
    statistics, so it runs inside this program rather than as one of its own.
    The statistics accumulator is donated: it is a running total the driver
    rebinds every chunk, so updating it in place is what the loose dispatch
    already did through ``accumulate_chunk_image_terms``.
    """

    mstep = resident_block_programs.add_chunk_wavg_image_power(
        mstep, operands, tables, posterior.row_posterior, posterior.kernel_row_image_ids, spec=spec
    )
    return _resident_chunk_statistics(
        stats, rows, operands, tables, posterior, mstep, spec=spec
    )


def _resident_chunk_statistics(
    stats,
    rows: resident_chunk_records.ChunkRowArrays,
    operands: resident_chunk_records.ChunkStageOperands,
    tables: resident_chunk_records.ChunkStageTables,
    posterior: resident_chunk_records.ChunkPosterior,
    mstep: resident_chunk_records.ChunkMstepCarry,
    *,
    spec: resident_chunk_records.ChunkProgramSpec,
):
    """Fold one chunk's image-level terms and its padding sanity counter."""

    image_tables = resident_image_terms.ChunkImageTables(
        shell_indices_half=tables.shell_indices_half,
        wavg_shell_indices=tables.wavg_shell_indices,
        wavg_scale_pixel_mask=tables.wavg_scale_pixel_mask,
        translation_sqdist_ang=operands.translation_sqdist_ang,
        norm_shell_cutoff=(
            None
            if tables.window_logical is None
            else (
                tables.window_logical.current_size
                if tables.window_logical.wsum_current_size is None
                else tables.window_logical.wsum_current_size
            )
            // 2
        ),
    )
    best_row_local = posterior.best_cell_index // jnp.int64(int(spec.n_fine_trans))
    slot_is_valid = jnp.arange(int(spec.image_capacity), dtype=jnp.int32) < rows.n_valid_images
    invalid_best = slot_is_valid & (
        (best_row_local < 0) | (best_row_local >= rows.image_row_count)
    )
    best_chunk_row = jnp.clip(
        rows.image_row_start + best_row_local,
        0,
        jnp.int64(max(int(spec.row_capacity) - 1, 0)),
    ).astype(jnp.int32)
    best_fine_rot = jnp.asarray(rows.row_fine_rot, dtype=jnp.int64)[best_chunk_row]
    if tables.cache_slot_fine_rot is not None:
        best_fine_rot = jnp.asarray(tables.cache_slot_fine_rot, dtype=jnp.int64)[best_fine_rot]

    class_fields = {}
    if posterior.classes is not None:
        # Each class's own winner: its sub-segment row, then that row's fine
        # rotation through the class-stacked (or streamed) slot table.
        n_fine_trans = jnp.int64(int(spec.n_fine_trans))
        class_best_row = jnp.clip(
            rows.classes.segment_row_start + posterior.classes.best_cell_index // n_fine_trans,
            0,
            jnp.int64(max(int(spec.row_capacity) - 1, 0)),
        ).astype(jnp.int32)
        class_fine_rot = jnp.asarray(tables.cache_slot_fine_rot, dtype=jnp.int64)[
            jnp.asarray(rows.row_fine_rot, dtype=jnp.int64)[class_best_row]
        ]
        class_fields = dict(
            row_class=rows.classes.row_class,
            per_class_log_z=posterior.classes.log_z,
            per_class_best_log_score=posterior.classes.best_log_score,
            per_class_best_cell=jnp.where(
                jnp.isfinite(posterior.classes.best_log_score),
                class_fine_rot * n_fine_trans + posterior.classes.best_cell_index % n_fine_trans,
                jnp.int64(-1),
            ),
        )

    chunk_operands = resident_image_terms.ChunkImageOperands(
        row_posterior=posterior.row_posterior,
        row_image_local=rows.row_image_local,
        row_coarse_rot=jnp.where(
            posterior.row_is_valid,
            tables.coarse_parent_grid[rows.row_fine_rot],
            jnp.int32(int(spec.stats_config.n_coarse_rot)),
        ),
        image_ids=rows.image_ids,
        group_ids=operands.group_ids,
        image_power_shells=operands.image_power_shells,
        relion_norm_high_shell=operands.relion_norm_high_shell,
        wavg_triplet_pixels=mstep.wavg_triplet_pixels,
        block_noise_shells=mstep.noise_shells,
        a2_per_image=mstep.a2_per_image,
        xa_per_image=mstep.xa_per_image,
        class_log_z=posterior.class_log_z,
        min_diff2=posterior.min_diff2,
        best_log_score=posterior.best_log_score,
        max_posterior=posterior.max_posterior,
        best_cell_index=posterior.best_cell_index,
        best_fine_rot=best_fine_rot,
        optics_groups=operands.optics_groups,
        scale_xa_per_image=mstep.scale_xa_per_image,
        scale_aa_per_image=mstep.scale_aa_per_image,
        **class_fields,
    )
    stats = resident_image_terms.accumulate_chunk_image_terms(
        stats, chunk_operands, image_tables, config=spec.stats_config
    )
    return stats._replace(
        invalid_best_rows=stats.invalid_best_rows + jnp.sum(invalid_best.astype(jnp.int64))
    )


# Divisors of the M-step block for classes with few live rows. The pixel-axis
# block costs in proportion to its rows whether or not they carry weight, and
# at 50k/256 one class of a chunk has about 40 live rows against a 2048-row
# block. Three sizes bound the programs a capacity class compiles.
_LIVE_BLOCK_DIVISORS = (64, 8, 1)


_MIN_LIVE_BLOCK_ROWS = 16


def _live_block_spec(spec: "resident_chunk_records.ChunkProgramSpec", n_live_rows: int) -> "resident_chunk_records.ChunkProgramSpec":
    """The chunk spec whose M-step block is the smallest ladder size holding ``n_live_rows``.

    The ladder is the configured block divided by 64, by 8 and by 1 (no smaller
    than 16 rows); a slot with more live rows than the configured block walks
    it in configured blocks as before. Every ladder size is a power of two that
    divides the row capacity, so a block never runs past the capacity.
    """

    full = int(spec.mstep_block_rows)
    for divisor in _LIVE_BLOCK_DIVISORS:
        rows = max(full // divisor, min(_MIN_LIVE_BLOCK_ROWS, full))
        if int(n_live_rows) <= rows:
            break
    return spec if rows == full else dataclass_replace(spec, mstep_block_rows=rows)


def run_resident_chunk_stages(
    rows: resident_chunk_records.ChunkRowArrays,
    operands: resident_chunk_records.ChunkStageOperands,
    tables: resident_chunk_records.ChunkStageTables,
    carry: tuple,
    *,
    spec: resident_chunk_records.ChunkProgramSpec,
    timing_hook=None,
    row_partial: RowPartial | None = None,
):
    """The chunk's stages, dispatched one at a time.

    The host block loop reads each accumulator slot's live row range back from
    the device once per chunk (:func:`make_mstep_block_inputs`), so it
    launches only the blocks that carry weight.

    ``timing_hook(name)`` is called after each stage when the chunk
    timing diagnostic is on; it synchronizes, so an arm that passes it is a
    diagnostic arm.
    """

    front = resident_chunk_stages_front(rows, operands, tables, spec=spec, timing_hook=timing_hook)
    return resident_chunk_stages_finish(
        front, rows, operands, tables, carry, spec=spec, timing_hook=timing_hook, row_partial=row_partial
    )


def resident_chunk_stages_front(rows, operands, tables, *, spec, timing_hook=None):
    """The per-stage path's posterior and M-step row order, enqueued; no host read."""

    if spec.glue_jit:
        posterior = resident_chunk_posterior.resident_chunk_posterior_program(rows, operands, tables, spec=spec)
    else:
        posterior = resident_chunk_posterior.resident_chunk_posterior(rows, operands, tables, spec=spec)
    if timing_hook is not None:
        timing_hook("posterior", posterior.row_posterior)
    return posterior, resident_mstep_blocks.make_mstep_block_inputs(rows, posterior, n_slots=int(spec.n_slots))


def resident_chunk_stages_finish(
    front,
    rows,
    operands,
    tables,
    carry,
    *,
    spec,
    timing_hook=None,
    block_projections=None,
    projection_dtypes=None,
    row_partial: RowPartial | None = None,
):
    """The per-stage path after its front: the live row ranges, the M-step blocks, the statistics.

    ``block_projections(start, block_rows)``, when given, returns the M-step
    block's ``(proj, |proj|^2, M-step rotations)`` for the block at ``start`` of
    the M-step order, instead of the block gathering them out of the tables'
    caches: a lone overflow chunk of a streamed pass projects each live block's
    rows (:func:`_run_lone_resident_chunk`); ``projection_dtypes`` are then
    the ``(proj, |proj|^2)`` dtypes it returns, which the tables' caches
    (absent there) would otherwise give.

    With ``row_partial`` the blocks scatter into the pass's partial pair, which is added into the slot's
    totals every ``blocks_per_flush`` blocks and when the slot's blocks end (relax#65).
    """

    posterior, blocks = front
    Ft_y_total, Ft_ctf_total, stats = carry
    mstep = resident_mstep_blocks.initial_mstep_carry(
        Ft_y_total[0], Ft_ctf_total[0], operands, tables, spec=spec, projection_dtypes=projection_dtypes
    )
    slot_offsets = np.asarray(jax.device_get(blocks.slot_offsets), dtype=np.int64)
    Ft_y_out, Ft_ctf_out = [], []
    for slot_index in range(int(spec.n_slots)):
        row_lo, row_hi = int(slot_offsets[slot_index]), int(slot_offsets[slot_index + 1])
        slot_y, slot_ctf = Ft_y_total[slot_index], Ft_ctf_total[slot_index]
        mstep = mstep._replace(Ft_y=slot_y, Ft_ctf=slot_ctf) if row_partial is None else row_partial.bind(mstep)
        class_blocks = blocks._replace(class_row_range=blocks.slot_offsets[slot_index : slot_index + 2])
        block_spec = _live_block_spec(spec, row_hi - row_lo)
        block_rows = int(block_spec.mstep_block_rows)
        # Blocks past the slot's live rows hold only rows without weight, of
        # this slot or another: the weighted sums, the Wavg terms, the noise
        # partials and both adjoint scatters would add exact zeros.
        for start in range((row_lo // block_rows) * block_rows, row_hi, block_rows):
            start_blocks = (
                class_blocks
                if block_projections is None
                else class_blocks._replace(row_fine_rot=None, projections=block_projections(start, block_rows))
            )
            if spec.glue_jit:
                mstep = resident_mstep_blocks.resident_mstep_block_program(
                    resident_chunk_records.device_int32(start), start_blocks, operands, tables, mstep, spec=block_spec
                )
            else:
                mstep = resident_mstep_blocks.resident_mstep_block_at(
                    resident_chunk_records.device_int32(start),
                    start_blocks,
                    operands,
                    tables,
                    mstep,
                    spec=block_spec,
                )
            if row_partial is not None:
                mstep, slot_y, slot_ctf = row_partial.block_done(mstep, slot_y, slot_ctf, block_rows=block_rows)
        if row_partial is not None:
            mstep, slot_y, slot_ctf = row_partial.flush(mstep, slot_y, slot_ctf)
        else:
            slot_y, slot_ctf = mstep.Ft_y, mstep.Ft_ctf
        if mstep.scale_xa_per_image is not None:
            # Slot ``class + K * group``: the scale sums are masked by the slot's class.
            mstep = fold_class_scale_sums(
                mstep, tables.wavg_scale_pixel_mask, resident_chunk_records.device_int32(slot_index % int(spec.n_classes))
            )
        Ft_y_out.append(slot_y)
        Ft_ctf_out.append(slot_ctf)
    if timing_hook is not None:
        timing_hook("mstep", (Ft_y_out, Ft_ctf_out))

    if spec.glue_jit:
        # The program adds the chunk's Wavg image power first, as below.
        stats = resident_chunk_statistics_program(
            stats, rows, operands, tables, posterior, mstep, spec=spec
        )
    else:
        mstep = resident_block_programs.add_chunk_wavg_image_power(
            mstep, operands, tables, posterior.row_posterior, posterior.kernel_row_image_ids, spec=spec
        )
        stats = _resident_chunk_statistics(
            stats, rows, operands, tables, posterior, mstep, spec=spec
        )
    return tuple(Ft_y_out), tuple(Ft_ctf_out), stats
