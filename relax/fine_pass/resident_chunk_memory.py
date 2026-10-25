"""Chunk planning of the resident pass 2: capacity ladders, the M-step block and the chunk memory plan.

A pass walks its candidate rows in chunks padded to a row and image capacity class, so every chunk program
is keyed on a static shape. This module owns the default ladders, the M-step block size, the device bytes
one chunk of a class holds (:func:`resident_chunk_bytes`), the plan that keeps the classes that fit the
measured budget (:func:`plan_resident_chunk_memory`), and the lone overflow chunk that runs by itself
(:func:`lone_chunk_row_bytes`, :func:`chunk_runs_alone`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from relax.fine_pass.policy import ResidentConfigurationUnsupported
from relax.fine_pass.resident_statistics import posterior_translation_bucket_scratch_bytes
from relax.runtime.memory_budget import current_device_available_bytes
from relax.runtime.shape_buckets import pow2_floor


@dataclass(frozen=True)
class ResidentPass2Plan:
    """Chunk plan plus the pixel-axis block size the M-step stages run at."""

    chunks: tuple
    row_capacity_ladder: tuple
    image_capacity_ladder: tuple
    mstep_block_rows: int


def _largest_block_dividing(block: int, capacity: int) -> int:
    """``block`` capped by ``capacity`` and halved until it divides it (at least 1)."""

    block = min(block, capacity)
    while block > 1 and capacity % block:
        block //= 2
    return max(block, 1)


def resolve_mstep_block_rows(
    *,
    n_recon_pixels: int,
    max_block_bytes: int,
    row_capacity_ladder: tuple,
    override: int | None,
    per_chunk: bool = False,
) -> int:
    """Rows per pixel-axis block: a power of two dividing every row capacity (``override``:
    ``EngineTuning.mstep_block_rows``, already a positive power of two).

    A block holds, per row and reconstruction pixel, one complex64 projection,
    one float32 ``|proj|^2``, two complex64 weighted sums, one float32 CTF sum
    and the float32 Wavg triplet: 44 bytes. Sizing from the same budget the
    compact engine uses for its adjoint blocks keeps the transient comparable
    to the bucket this replaces; quantizing to a power of two keeps the traced
    pixel-axis program count at one per pixel count.

    ``per_chunk`` (the tilt pass): the block is capped by the largest row capacity and by
    ``half_volume_mstep.ROW_PARTIAL_FLUSH_ROWS`` (the partial pair is added into the volumes every that many rows,
    blocks_per_flush, so a larger block would hold more rows per partial than relax#65 measured) instead of the
    smallest capacity, and each chunk takes :func:`chunk_mstep_block_rows`, which divides its own capacity. Capped
    by the smallest capacity, a tilt pass ran 256-row blocks in 8192-row chunks: launches of half a wave, ~1.16M
    M-step steps per half (EMPIAR-10491 bin 2, iteration 1).
    """

    if override is not None:
        block = override
    else:
        bytes_per_row = max(n_recon_pixels, 1) * 44
        block = pow2_floor(max(max_block_bytes // bytes_per_row, 1))
    if per_chunk:
        from relax.reconstruction.half_volume_mstep import ROW_PARTIAL_FLUSH_ROWS

        return max(min(block, pow2_floor(max(row_capacity_ladder)), ROW_PARTIAL_FLUSH_ROWS), 1)
    return _largest_block_dividing(block, row_capacity_ladder[0])


def chunk_mstep_block_rows(mstep_block_rows: int, row_capacity: int) -> int:
    """A chunk's M-step block: the pass's (:func:`resolve_mstep_block_rows` with ``per_chunk``), a power of two at
    most the chunk's row capacity, halved until it divides it (a power-of-two class, or a multiple of the largest
    class past the classes)."""

    return _largest_block_dividing(min(mstep_block_rows, pow2_floor(max(row_capacity, 1))), row_capacity)


def _cap_image_capacity_ladder(
    ladder: tuple,
    *,
    n_fine_trans: int,
    n_recon_pixels: int,
    max_tile_bytes: int,
) -> tuple:
    """Drop image classes whose translation tiles exceed the tile budget.

    A chunk materializes three ``(images, T, P)`` complex64 tiles: the
    reconstruction operand, the noise operand and RELION's Wavg rectangle.
    When no class fits, the chunk holds the largest power of two of images
    that does (at least one), as RELION translates one particle's images at a
    time; keeping the smallest class instead ran EMPIAR-10202 iteration 13
    (148 translations, 132095 pixels, 469 MiB per image against a 1.5 GiB
    budget) out of memory with 32-image tiles.
    """

    per_image = max(int(n_fine_trans), 1) * max(int(n_recon_pixels), 1) * 8 * 3
    cap = max(int(max_tile_bytes) // max(per_image, 1), 1)
    kept = tuple(value for value in ladder if int(value) <= cap)
    return kept if kept else (pow2_floor(cap),)


# Share of the measured headroom (after the half's reserved operands) that one
# chunk's row block, translation tiles and M-step block may take together; the
# rest covers XLA temporaries of the scoring and M-step programs.
_RESIDENT_CHUNK_FREE_MEMORY_FRACTION = 0.6


@dataclass(frozen=True)
class ResidentChunkMemoryPlan:
    """One chunk's capacity classes, sized together against one memory budget."""

    row_capacity_ladder: tuple
    image_capacity_ladder: tuple
    mstep_block_rows: int
    peak_bytes: int
    budget_bytes: int | None
    # Device bytes the pass allocates once besides its chunks (accumulators).
    fixed_bytes: int = 0
    # ((row capacity, image capacity), peak bytes) of every pair of the ladders' classes, and of the one-image
    # overflow chunks' row capacities (multiples of the largest row class) at the smallest image class.
    class_peak_bytes: tuple = ()

    @property
    def pass_bytes(self) -> int:
        """The pass's total device need: its fixed buffers plus its largest chunk."""

        return int(self.fixed_bytes) + int(self.peak_bytes)

    def chunk_peak_bytes(self, row_capacity: int, image_capacity: int) -> int:
        """The peak of a chunk of these capacity classes (:func:`resident_chunk_bytes`), or of an overflow chunk
        (a particle past the row classes, run alone); ``peak_bytes`` for any other."""

        return dict(self.class_peak_bytes).get((int(row_capacity), int(image_capacity)), int(self.peak_bytes))


def _mstep_block_row_bytes(n_fine_trans: int, n_recon_pixels: int, mstep_tile_pixels: int | None = None) -> int:
    """Device bytes of one M-step block row (:func:`resident_chunk_bytes`)."""

    t, p = max(int(n_fine_trans), 1), max(int(n_recon_pixels), 1)
    gathered = 2 * p if mstep_tile_pixels is None else int(mstep_tile_pixels)
    return t * gathered * 8 + 44 * p


def resident_chunk_bytes(
    *, row_capacity: int, image_capacity: int, mstep_block_rows: int, row_bytes: int, n_fine_trans: int,
    n_recon_pixels: int, held_tile_pixels: int | None = None, prepare_tile_pixels: int = 0,
    rows_live_during_prepare: bool = False,
    mstep_tile_pixels: int | None = None,
    pipelined: bool = False,
    projection_transient_bytes: int = 0,
    float32_posterior_buckets: bool = False,
) -> int:
    """Device bytes one chunk holds at its peak, the larger of its two stages.

    The row stage holds the row block, the translated image tiles the M-step
    reads and the M-step block. ``row_bytes`` is what one row keeps live (the
    cached gather's score row, a streamed or local chunk's own projections).
    ``held_tile_pixels`` is the per-image, per-translation complex64 pixel count
    of the held tiles; the default, three ``P``-pixel tiles, is the
    reconstruction and noise operands and RELION's Wavg rectangle. An M-step
    block row holds its 44-byte-per-pixel sums and ``mstep_tile_pixels``
    complex64 pixels per translation of gathered tiles; the default, ``2 P``, is
    the row's recon and noise tiles. Unshifted operands (T16) gather none: the
    translate-and-sum kernel translates the per-image operands itself.

    The preparation stage holds ``prepare_tile_pixels`` per image and
    translation (:func:`chunk_translated_tile_pixels`); zero means the caller
    builds no translated tiles. The local pass prepares a chunk's operands
    before it projects the chunk's rows; the global pass projects (or gathers)
    them first, so ``rows_live_during_prepare`` adds the row block to that
    stage.

    A ``pipelined`` loop enqueues chunk k+1 up to its posterior before chunk
    k's M-step, so chunk k's rows and held tiles stay live through the next
    chunk's preparation, projection, scoring and its own M-step: both stages
    add one more row block and held tiles. Unmodelled, that let EMPIAR-10202
    iteration 22 plan 4096-row chunks (11.78 GiB of row projections, a 19.03
    GiB modelled peak of a 22.12 GiB budget) and run out of memory.

    ``projection_transient_bytes`` is what one projector call holds while it
    projects a block of the chunk's rows (:func:`projection_call_row_bytes`);
    it is live in the row stage, before scoring, never together with the
    M-step block. A constant reserve of one full-half-spectrum call missed the
    texture crop held beside it: the EMPIAR-10202 final pass (current size 800)
    ran out of memory allocating the second 4 GiB array (bigbox case).
    """

    t, p = max(int(n_fine_trans), 1), max(int(n_recon_pixels), 1)
    held = 3 * p if held_tile_pixels is None else int(held_tile_pixels)
    rows = int(row_capacity) * int(row_bytes)
    tiles = int(image_capacity) * t * held * 8
    mstep = int(mstep_block_rows) * _mstep_block_row_bytes(n_fine_trans, n_recon_pixels, mstep_tile_pixels)
    posterior_scratch = (
        posterior_translation_bucket_scratch_bytes(row_capacity, image_capacity, n_fine_trans)
        if float32_posterior_buckets else 0
    )
    # A2 and XA each bucket one M-step block of scalar F32 rows. Keep room
    # for both while their producer arrays are live; this is much smaller than
    # the translation-posterior scratch but overlaps the M-step stage.
    norm_scratch = (
        2 * posterior_translation_bucket_scratch_bytes(mstep_block_rows, image_capacity, 1)
        if float32_posterior_buckets else 0
    )
    prepare = int(image_capacity) * t * int(prepare_tile_pixels) * 8
    if rows_live_during_prepare:
        prepare += rows
    previous_chunk = rows + tiles if pipelined else 0
    return max(rows + tiles + max(mstep + norm_scratch, int(projection_transient_bytes), posterior_scratch), prepare) + previous_chunk


def projection_call_row_bytes(
    *,
    crop_pixels: int,
    output_pixels: int,
    n_score_pixels: int,
    n_recon_pixels: int,
    complex_bytes: int = 8,
    real_bytes: int = 4,
) -> int:
    """Device bytes per projected row at the peak of one RELION projector call.

    The texture kernel's ``[rows, crop_pixels]`` output is live while the call
    gathers its ``[rows, output_pixels]`` output (the pass's window union, or
    the full half spectrum without one); the zero-row and dense-scale steps each
    make a new output array beside the previous one, and a zero column is
    appended; the placement builds the score and reconstruction windows and
    ``|recon|^2`` from the widened output (kspeed, 50c2542). The call's peak is
    the largest of those three stages.
    """

    c, u = int(crop_pixels), int(output_pixels)
    s, r = int(n_score_pixels), int(n_recon_pixels)
    return max(
        complex_bytes * (c + u),
        complex_bytes * (2 * u + 1),
        complex_bytes * (u + 1 + s + r) + real_bytes * r,
    )


def resident_accumulator_bytes(volume_size: int, y_dtype, ctf_dtype, *, n_slots: int = 1) -> int:
    """Device bytes of a pass's BPref accumulators (``Ft_y`` and ``Ft_ctf`` per slot)."""

    per_slot = np.dtype(y_dtype).itemsize + np.dtype(ctf_dtype).itemsize
    return int(n_slots) * int(volume_size) * per_slot


def chunk_translated_tile_pixels(
    *,
    unshifted_operands: bool,
    n_score_pixels: int,
    n_recon_pixels: int,
    n_rect_pixels: int,
    n_exact_rect_pixels: int,
    normalized_cc: bool = False,
    masked_scoring: bool = True,
) -> dict:
    """Per-image, per-translation complex64 pixels of a chunk's translated arrays, by stage.

    The ``held_tile_pixels`` / ``prepare_tile_pixels`` arguments of
    :func:`plan_resident_chunk_memory` for the global and the local resident
    pass. Counted from the live device arrays of one chunk of the bigbox case;
    ``n_score_pixels`` / ``n_recon_pixels`` are the windows the preparation's
    tiles carry (the half spectrum when the preparation is not
    windowed).

    With the translated tiles (:func:`prepare_chunk_reconstruction_operands`),
    the preparation peak is inside :func:`_chunk_operand_rows`, where
    ``prepare_bucket_io``'s two score-window and two recon-window tiles, the
    Wavg rectangle and the row-order copies (recon, noise, rectangle and its
    exact positions) are live together; the chunk then holds the three
    row-order recon-window tiles and the rectangle through projection, scoring
    and the M-step. ``normalized_cc`` (the ``--firstiter_cc`` iteration) adds
    the row-order corrected score tile to both; without ``masked_scoring`` the
    preparation's recon and noise tiles are one array
    (``bucket_io``), one recon tile fewer. Planning with three recon tiles
    instead let EMPIAR-10202 iteration 22 take 32 images per chunk, a ~30 GiB
    preparation against a 22 GiB budget, and the global pass of
    VDAM K=1 ribosembly 100k run out of memory allocating the rectangle
    (9.93 GiB, bench refresh 2026-09-27).

    With the unshifted operands (T16), gathered per chunk from the half's
    resident arrays or prepared for the chunk's images, the only translated
    arrays are the Wavg rectangle and its exact positions, from preparation to
    the M-step, and its M-step blocks gather no translated tile
    (``mstep_tile_pixels``). The chunk-local preparation's translation-free buffers are
    padded to :func:`resident_image_capacity` rows (256 at least), about 1.7 GB
    at EMPIAR-10202 current size 626, which the budget's free-memory margin
    covers.
    """

    s, p, r, e = int(n_score_pixels), int(n_recon_pixels), int(n_rect_pixels), int(n_exact_rect_pixels)
    if unshifted_operands:
        # The translate-and-sum kernel translates the per-image operands, so an
        # M-step block gathers no translated tile.
        return {"held_tile_pixels": r + e, "prepare_tile_pixels": r + e, "mstep_tile_pixels": 0}
    cc = s if normalized_cc else 0
    shared = 0 if masked_scoring else p
    return {
        "held_tile_pixels": 2 * p + r + e + cc,
        "prepare_tile_pixels": 2 * s + 4 * p + 2 * r + e + cc - shared,
        "mstep_tile_pixels": 2 * p,
    }


# The smallest row class the memory plan halves a single class down to.
MIN_PLANNED_ROW_CAPACITY = 64


def plan_resident_chunk_memory(
    *,
    row_capacity_ladder: tuple,
    image_capacity_ladder: tuple,
    mstep_block_rows: int,
    row_bytes: int,
    n_fine_trans: int,
    n_recon_pixels: int,
    budget_bytes: int | None,
    held_tile_pixels: int | None = None,
    prepare_tile_pixels: int = 0,
    rows_live_during_prepare: bool = False,
    mstep_tile_pixels: int | None = None,
    pipelined: bool = False,
    projection_transient_bytes: int = 0,
    fixed_bytes: int = 0,
    max_image_rows: int = 0,
    lone_row_bytes: int | None = None,
    lone_block_rows: int | None = None,
    overflow_row_bytes: int | None = None,
    float32_posterior_buckets: bool = False,
    per_chunk: bool = False,
) -> ResidentChunkMemoryPlan:
    """Shrink the three per-chunk classes until their sum fits one budget.

    Each class starts from its own device-fraction rule, which alone let the
    row gather, the translation tiles and the M-step block overcommit the device
    together (EMPIAR-10202 iteration 14: a 9.7 GiB M-step block sized without
    its translation axis, bigbox case). While the largest chunk exceeds the
    budget, the largest of the three terms shrinks: the M-step block halves, the
    image ladder drops its largest class and then halves a single class, the row
    ladder does the same down to :data:`MIN_PLANNED_ROW_CAPACITY` rows. When the
    preparation stage is the chunk's peak (:func:`resident_chunk_bytes`), only
    its own terms shrink: the image ladder, and the row ladder where the rows
    are live during it. Plans that already fit are unchanged; an unknown budget
    does not cap. A plan that cannot shrink further is a
    :class:`ResidentConfigurationUnsupported`: there is no other pass-2 engine.

    ``fixed_bytes`` (the pass's accumulators, allocated before the budget is
    read) is recorded in the plan, whose ``pass_bytes`` is the pass's total
    device need.

    ``max_image_rows`` is the largest single image's row count. An image past
    the largest row class is a one-image chunk of the smallest image class
    (:func:`overflow_row_capacity`), which the chunk loops run alone
    (:func:`chunk_runs_alone`), so it is counted without a pipelined neighbour.
    Uncounted, the EMPIAR-10202 final pass planned 1024-row chunks at 16.05 of a
    16.16 GiB budget and ran out of memory in a 2048-row one (bigbox case).
    ``lone_row_bytes`` is set where that chunk runs in blocks of the largest row
    class (the global pass, :func:`~relax.fine_pass.resident_chunk_run.run_lone_resident_chunk`): it is then a
    largest-class chunk of the smallest image class plus ``lone_row_bytes`` for
    each of its rows (the posterior cells every row keeps); ``lone_block_rows`` is that block when it is smaller
    than the largest row class (a lean lone tilt particle, :mod:`~relax.fine_pass.resident_tilt_lone`).
    ``overflow_row_bytes`` (default ``row_bytes``) is
    that chunk's ``row_bytes`` when it differs from the row classes': a subtomogram particle past them projects
    its rows a smaller block of image slots at a time (resident_pass2, ``tilt_overflow_slot_block``).
    ``per_chunk`` (a tilt pass, :func:`resolve_mstep_block_rows`): the planned block is each chunk's largest,
    counted at the largest chunk, and divides the largest row class instead of every class.
    """

    rows = tuple(int(v) for v in row_capacity_ladder)
    images = tuple(int(v) for v in image_capacity_ladder)
    block = int(mstep_block_rows)
    kwargs = dict(
        row_bytes=row_bytes,
        n_fine_trans=n_fine_trans,
        n_recon_pixels=n_recon_pixels,
        held_tile_pixels=held_tile_pixels,
        prepare_tile_pixels=prepare_tile_pixels,
        rows_live_during_prepare=rows_live_during_prepare,
        mstep_tile_pixels=mstep_tile_pixels,
        pipelined=pipelined,
        projection_transient_bytes=projection_transient_bytes,
        float32_posterior_buckets=float32_posterior_buckets,
    )
    held_pixels = 3 * max(int(n_recon_pixels), 1) if held_tile_pixels is None else int(held_tile_pixels)
    projection = int(projection_transient_bytes)

    def regular_peak():
        return resident_chunk_bytes(
            row_capacity=max(rows), image_capacity=max(images), mstep_block_rows=block, **kwargs
        )

    overflow_kwargs = {**kwargs, "pipelined": False}
    if overflow_row_bytes is not None:
        overflow_kwargs["row_bytes"] = int(overflow_row_bytes)

    def overflow_chunk_peak(overflow_rows):
        # A one-image chunk of ``overflow_rows`` rows past the row classes, run alone.
        if lone_row_bytes is not None:
            return resident_chunk_bytes(
                row_capacity=max(rows) if lone_block_rows is None else int(lone_block_rows),
                image_capacity=min(images),
                mstep_block_rows=block,
                **overflow_kwargs,
            ) + overflow_rows * int(lone_row_bytes)
        return resident_chunk_bytes(
            row_capacity=overflow_rows,
            image_capacity=min(images),
            mstep_block_rows=block,
            **overflow_kwargs,
        )

    def overflow_peak():
        overflow_rows = overflow_row_capacity(max_image_rows, rows)
        return overflow_chunk_peak(overflow_rows) if overflow_rows else 0

    def peak():
        return max(regular_peak(), overflow_peak())

    if budget_bytes is not None:
        while peak() > int(budget_bytes):
            t, p = max(int(n_fine_trans), 1), max(int(n_recon_pixels), 1)
            images_can_shrink = len(images) > 1 or max(images) > 1
            mstep_bytes = block * _mstep_block_row_bytes(t, p, mstep_tile_pixels)
            posterior_scratch = (
                posterior_translation_bucket_scratch_bytes(max(rows), max(images), t)
                if float32_posterior_buckets else 0
            )
            if overflow_peak() > regular_peak():
                # A lone overflow chunk is the peak: its rows are the image's own,
                # so only its M-step block and the rounding to the largest row
                # class can shrink.
                terms = {
                    "mstep": mstep_bytes if block > 1 and mstep_bytes > max(projection, posterior_scratch) else -1,
                    "rows": 1 if len(rows) > 1 else -1,
                }
                largest = max(terms, key=terms.get)
                if terms[largest] < 0:
                    raise ResidentConfigurationUnsupported(
                        "The device-resident pass 2 cannot fit this configuration: its largest image "
                        f"({int(max_image_rows)} rows, run alone) needs {peak() / float(1024 ** 3):.2f} GiB "
                        f"against a {int(budget_bytes) / float(1024 ** 3):.2f} GiB budget. There is no other "
                        "pass-2 engine; run on a GPU with more memory."
                    )
                if largest == "mstep":
                    block //= 2
                else:
                    rows = rows[:-1]
                continue
            rows_bytes = max(rows) * int(row_bytes)
            held_bytes = max(images) * t * held_pixels * 8
            prepare_bytes = max(images) * t * int(prepare_tile_pixels) * 8
            prepare_stage = prepare_bytes + (rows_bytes if rows_live_during_prepare else 0)
            # A pipelined loop holds the previous chunk's rows and held tiles
            # next to either stage (resident_chunk_bytes).
            copies = 2 if pipelined else 1
            # The row ladder drops its largest class, then halves a single class
            # down to MIN_PLANNED_ROW_CAPACITY: the EMPIAR-10202 final pass of a
            # cold run had a 13.81 GiB budget for a 13.90 GiB 1024-row chunk
            # (bigbox case), which 512 rows fit.
            rows_can_shrink = len(rows) > 1 or max(rows) > MIN_PLANNED_ROW_CAPACITY
            rows_in_prepare = rows_bytes * (int(rows_live_during_prepare) + int(pipelined))
            if prepare_stage > rows_bytes + held_bytes + max(mstep_bytes, projection, posterior_scratch):
                # The preparation stage is the peak; only its own terms shrink it.
                terms = {
                    "images": prepare_bytes + (held_bytes if pipelined else 0) if images_can_shrink else -1,
                    "rows": rows_in_prepare if rows_in_prepare and rows_can_shrink else -1,
                }
            else:
                terms = {
                    # A smaller M-step block helps only while it outweighs the projector call.
                    "mstep": mstep_bytes if block > 1 and mstep_bytes > max(projection, posterior_scratch) else -1,
                    "images": held_bytes * copies + posterior_scratch if images_can_shrink else -1,
                    "rows": rows_bytes * copies + posterior_scratch if rows_can_shrink else -1,
                }
            largest = max(terms, key=terms.get)
            if terms[largest] < 0:
                raise ResidentConfigurationUnsupported(
                    "The device-resident pass 2 cannot fit this configuration: its smallest chunk "
                    f"(rows {max(rows)}, images {max(images)}, M-step block {block}) needs "
                    f"{peak() / float(1024 ** 3):.2f} GiB "
                    f"against a {int(budget_bytes) / float(1024 ** 3):.2f} GiB budget. There is no other "
                    "pass-2 engine; run on a GPU with more memory."
                )
            if largest == "mstep":
                block //= 2
            elif largest == "images":
                images = images[:-1] if len(images) > 1 else (max(images) // 2,)
            else:
                rows = rows[:-1] if len(rows) > 1 else (max(rows) // 2,)
    # One block for every chunk divides the smallest row class; a per-chunk block (a tilt pass) only the largest's,
    # and each chunk takes chunk_mstep_block_rows of it.
    block = _largest_block_dividing(block, max(rows) if per_chunk else min(rows))
    # The overflow chunks' row capacities: multiples of the largest class (overflow_row_capacity).
    overflow_capacities = range(2 * max(rows), overflow_row_capacity(max_image_rows, rows) + 1, max(rows))
    return ResidentChunkMemoryPlan(
        row_capacity_ladder=rows,
        image_capacity_ladder=images,
        mstep_block_rows=block,
        peak_bytes=peak(),
        budget_bytes=None if budget_bytes is None else int(budget_bytes),
        fixed_bytes=int(fixed_bytes),
        class_peak_bytes=tuple(
            ((r, i), resident_chunk_bytes(row_capacity=r, image_capacity=i, mstep_block_rows=block, **kwargs))
            for r in rows
            for i in images
        )
        + tuple(((r, min(images)), overflow_chunk_peak(r)) for r in overflow_capacities),
    )


def overflow_row_capacity(max_image_rows: int, row_capacity_ladder) -> int:
    """Row capacity of the largest one-image overflow chunk; 0 when every image fits a class.

    :func:`relax.fine_pass.resident_candidates.plan_capacity_chunks` rounds
    an image past the largest row class up to a multiple of that class.
    """

    largest = max(int(v) for v in row_capacity_ladder)
    n_rows = int(max_image_rows)
    return 0 if n_rows <= largest else -(-n_rows // largest) * largest


# The most a chunk's leftover gives its translated tiles (relax#120). Leftovers of 19-30 GiB on an 80 GB card put
# whole chunks' translations in one M-step block, and the local iteration 18 of the tilt-series run took 728 s
# against 714-715 s at the card's 1.43 GiB device share (two same-node pairs); 1.5 GiB ran the final iteration
# in 1409 s on a 16 GB card, 1304 s on the 80 GB card. The uncapped leftover (``cap_bytes=None``) only
# widens blocks the cap cuts small (resident_tilt_mstep.TILT_MSTEP_BLOCK_TRANSLATIONS).
MAX_LEFTOVER_TILE_BUDGET_BYTES = 3 << 29


def leftover_tile_budget_bytes(
    plan: ResidentChunkMemoryPlan,
    *,
    row_capacity: int,
    image_capacity: int,
    reserved_bytes: int,
    floor_bytes: int,
    cap_bytes: int | None = MAX_LEFTOVER_TILE_BUDGET_BYTES,
) -> int:
    """The translated-tile budget of a chunk whose M-step tiles the chunk plan does not count (the tilt M-step).

    The chunk budget is one chunk's share of the device (:data:`_RESIDENT_CHUNK_FREE_MEMORY_FRACTION`: row block,
    translation tiles and M-step block together); a chunk of ``row_capacity`` rows and ``image_capacity`` images
    fills it up to its own peak (:meth:`ResidentChunkMemoryPlan.chunk_peak_bytes`). The tiles take what is left
    after ``reserved_bytes`` (what else lives beside the chunk's M-step that the plan does not count), at most
    ``cap_bytes`` (:data:`MAX_LEFTOVER_TILE_BUDGET_BYTES`; None for the whole leftover) and never less than
    ``floor_bytes`` (the fixed device share they had before). Without a budget the floor stands. A fixed 2% of
    the device gave a 16 GB card's final tilt iteration 8 translations per M-step block at current size 256: one
    half's pass 2 took 9341 s, where the 80 GB card ran the whole iteration in 1316 s (relax#120).
    """

    if plan.budget_bytes is None:
        return int(floor_bytes)
    peak = plan.chunk_peak_bytes(row_capacity, image_capacity)
    leftover = int(plan.budget_bytes) - int(peak) - int(reserved_bytes)
    if cap_bytes is not None:
        leftover = min(leftover, int(cap_bytes))
    return max(int(floor_bytes), leftover)


def max_image_rows(row_offsets) -> int:
    """The largest single image's row count in a CSR ``row_offsets`` table."""

    row_offsets = np.asarray(row_offsets)
    return int(np.max(np.diff(row_offsets))) if row_offsets.size > 1 else 0


# Per posterior cell of a lone overflow chunk: the kernel diff2 and the scores,
# the segmented posterior's probabilities, weights, reconstruction
# probabilities and mask, and the M-step order's copy of the posterior.
_LONE_CHUNK_CELL_BYTES = 4 * 7 + 1


def lone_chunk_row_bytes(n_fine_trans: int, *, class_rows: bool = False) -> int:
    """Device bytes each row of a lone overflow chunk keeps beyond its block (:func:`~relax.fine_pass.resident_chunk_run.run_lone_resident_chunk`).

    A row's projections live only while its block is scored or reconstructed;
    its ``T`` posterior cells and its row tables live for the whole chunk.
    ``class_rows`` is the Class3D local pass's: one class's masked posterior
    (``T`` float32 cells) while that class's M-step order is formed, and the
    row's class and sub-segment ids.
    """

    if class_rows:
        return int(n_fine_trans) * (_LONE_CHUNK_CELL_BYTES + 4) + 64 + 8
    return int(n_fine_trans) * _LONE_CHUNK_CELL_BYTES + 64


def chunk_runs_alone(chunk, row_capacity_ladder) -> bool:
    """Whether ``chunk`` is a one-image overflow chunk, which a pipelined loop runs alone.

    Its rows (the image's own, past the largest row class) are not overlapped
    with a neighbouring chunk's: the loop finishes the previous chunk before
    starting it and finishes it before starting the next, as
    :func:`plan_resident_chunk_memory` counts it.
    """

    return int(chunk.row_capacity) > max(int(v) for v in row_capacity_ladder)


def lone_block_rows(row_capacity: int, row_capacity_ladder) -> int:
    """Row block of a one-image overflow chunk: the largest row class that divides its rows.

    :func:`~relax.fine_pass.resident_candidates.plan_pass_chunks`' one-row-class re-plan rounds an overflow
    image up to a multiple of the class it chose, which need not be the largest (6144 rows under a 1024/4096
    ladder, EMPIAR-10073 Class3D local at --sigma_ang 10). Every block then has one shape and no padding rows.
    """

    dividing = [int(v) for v in row_capacity_ladder if int(row_capacity) % int(v) == 0]
    if not dividing:
        raise ValueError(
            f"no row class of {tuple(row_capacity_ladder)} divides the lone chunk's {row_capacity} rows"
        )
    return max(dividing)


def resident_image_capacity_start(
    ladder: tuple,
    *,
    n_fine_trans: int,
    n_recon_pixels: int,
    max_tile_bytes: int,
    chunk_budget_bytes: int | None,
) -> tuple:
    """Image classes a chunk starts from before the joint memory plan.

    The fixed tile budget decides as before wherever it fits a class. When it
    fits none (EMPIAR-10202 from iteration 4: 2-8 images against 32, which
    doubled the chunk loops, bigbox case) and the device headroom is known,
    the plan starts from the whole ladder and :func:`plan_resident_chunk_memory`
    sizes the tiles from the measured budget together with the rows and the
    M-step block.
    """

    fixed = _cap_image_capacity_ladder(
        ladder, n_fine_trans=n_fine_trans, n_recon_pixels=n_recon_pixels, max_tile_bytes=max_tile_bytes
    )
    if chunk_budget_bytes is None or min(fixed) >= min(int(v) for v in ladder):
        return fixed
    return tuple(int(v) for v in ladder)


def format_budget_gib(budget_bytes: int | None) -> str:
    """A chunk budget for a log line; ``unknown`` when the device reported nothing."""

    return "unknown" if budget_bytes is None else f"{int(budget_bytes) / float(1024**3):.2f} GiB"


def resident_chunk_budget_bytes(*, reserved_bytes: int = 0) -> int | None:
    """The joint chunk budget from the device readings now; ``None`` when nothing is known."""

    available = current_device_available_bytes()
    if available is None:
        return None
    return int(max(0.0, float(available) - float(reserved_bytes)) * _RESIDENT_CHUNK_FREE_MEMORY_FRACTION)
