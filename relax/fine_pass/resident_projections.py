"""Where the resident pass's projections live, and their device budgets.

A pass projects its fine rotations once into a whole-grid cache (one array, or 2 or 4 row blocks when the
allocator refuses one), into a significant-parent cache of slots, or, when neither fits, per chunk
(:func:`stream_chunk_projections`). This module owns those caches, their budgets and capacity ladders, the
admission of the half's resident operands next to them, and the per-projection M-step sums that replace the
per-slot BPref volumes while they fit (:func:`backproject_projection_sums`).
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from relax.fine_pass import policy, resident_chunk_memory, resident_stable_windows
from relax.fine_pass.adjoint import _accumulate_adjoint_block_chunked
from relax.fine_pass.policy import ResidentConfigurationUnsupported
from relax.fine_pass.resident_operands import (
    resident_half_operand_bytes,
    resident_operands_max_bytes,
    resident_prepare_reserved_bytes,
)
from relax.fine_pass.resident_significance import significant_coarse_parents
from relax.fine_pass.resident_statistics import resident_image_capacity
from relax.fine_pass.scoring import relion_powerclass_noise_dtypes
from relax.runtime.memory_budget import (
    _device_free_memory_bytes,
    _jax_allocator_free_memory_bytes,
    _jax_allocator_pool_free_bytes,
    device_available_bytes,
)
from relax.runtime.shape_buckets import pow2_ceil


# Per-projection M-step row sums replace the per-slot BPref volumes of a pass
# while they take at most this share of the free device memory.
def allocator_available_bytes() -> float | None:
    """Bytes the JAX allocator can still hand out now (:func:`device_available_bytes` of the three readings)."""

    return device_available_bytes(
        _device_free_memory_bytes(), _jax_allocator_free_memory_bytes(), _jax_allocator_pool_free_bytes()
    )


PRESUM_ADJOINT_FREE_FRACTION = 0.2


def projection_sums_bytes(*, n_slots, n_fine_rot, n_recon_pixels, y_dtype, ctf_dtype) -> int:
    """Bytes of one pass's per-projection sums: a ``[n_fine_rot, pixels]`` pair per slot."""

    per_row = int(n_recon_pixels) * (np.dtype(y_dtype).itemsize + np.dtype(ctf_dtype).itemsize)
    return int(n_slots) * int(n_fine_rot) * per_row


def projection_sums_displace_resident_operands(sums_bytes, **operand_size_kwargs) -> bool:
    """Whether the sums would push one half's resident operands out of their budget.

    The resident operands are worth more than the sums: at Class3D K4 100k/256
    current size 90, 5.9 GiB of sums next to 21.2 GiB of operands sent the pass
    to the per-chunk operand preparation, and its pass 2 went from 57 to 89 s
    (job 14589109). The same predicate as the operand admission
    (resident_operands_fit), read before either is allocated.
    """

    if not policy.resident_operands_requested():
        return False
    _, operand_peak_bytes = resident_half_operand_sizes(**operand_size_kwargs)
    available = device_available_bytes(
        _device_free_memory_bytes(), _jax_allocator_free_memory_bytes(), _jax_allocator_pool_free_bytes()
    )
    if available is None:
        return False
    return resident_operands_fit(operand_peak_bytes, available) and not resident_operands_fit(
        operand_peak_bytes, float(available) - float(sums_bytes)
    )


def carries_projection_sums(Ft_y_total) -> bool:
    """Whether the chunk carry holds per-projection sums (2-D) rather than BPref volumes (1-D)."""

    return int(jnp.ndim(Ft_y_total[0])) == 2


def backproject_projection_sums(sums_y, sums_ctf, *, mstep_grid, n_classes, n_fine_rot, volume_size, adjoint_kwargs):
    """Each slot's BPref pair from its per-projection row sums, one backprojection per projection.

    The M-step adjoint is linear in its rows, so backprojecting a fine
    rotation's summed rows once equals backprojecting each row (RELION's
    per-image BPref.set2DFourierTransform) up to float32 reassociation: the
    rows of one projection are added before the trilinear scatter instead of
    after it. Slot ``class + K * group`` takes class ``slot % K``'s M-step
    rotations, ``mstep_grid[class * n_fine_rot : (class + 1) * n_fine_rot]``.
    """

    Ft_y_out, Ft_ctf_out = [], []
    for slot, (slot_y, slot_ctf) in enumerate(zip(sums_y, sums_ctf)):
        start = (slot % int(n_classes)) * int(n_fine_rot)
        rotations = mstep_grid[start : start + int(n_fine_rot)]
        Ft_y_out.append(
            _accumulate_adjoint_block_chunked(
                slot_y,
                rotations,
                jnp.zeros(int(volume_size), dtype=slot_y.dtype),
                log_label="resident-y-sums",
                **adjoint_kwargs,
            )
        )
        Ft_ctf_out.append(
            _accumulate_adjoint_block_chunked(
                slot_ctf,
                rotations,
                jnp.zeros(int(volume_size), dtype=slot_ctf.dtype),
                log_label="resident-ctf-sums",
                **adjoint_kwargs,
            )
        )
    return tuple(Ft_y_out), tuple(Ft_ctf_out)


_STREAM_SLOT_QUANTUM = 8192


# Share of the measured free device memory a chunk-local projection cache may
# take; the rest stays for the chunk's scoring and M-step working set, which
# have their own device-fraction budgets.
_STREAM_FREE_MEMORY_FRACTION = 0.5


# While a chunk's projections are padded to the row capacity, the projected
# block and its padded copy are both live (flipqual 10097 it13: a 12.16 GiB pad
# at row capacity 131072 ran the device out of memory, 14397073).
STREAM_PEAK_COPIES = 2


def resident_half_operand_sizes(
    *,
    n_images,
    n_windowed,
    n_recon_windowed,
    n_rect,
    n_shells,
    n_fine_trans,
    precision_policy,
    resolved_spectrum_norm,
    premultiplied_ctf=False,
    image_shape,
):
    """``(bytes, peak bytes)`` of one half's resident operands.

    The preparation holds the operands plus one reordered copy of its largest
    array (resident_operands.stack) and the working set of its smallest image
    batch (``image_shape``; :func:`resident_prepare_batch_size` sizes the batch
    to what is left), hence the peak.
    """

    _, norm_high_shell_dtype = relion_powerclass_noise_dtypes(
        real_dtype=precision_policy.score_real_dtype,
        source_faithful_spectrum_norm=resolved_spectrum_norm,
    )
    operand_bytes = resident_half_operand_bytes(
        n_images=int(n_images),
        n_score_pixels=int(n_windowed),
        n_recon_pixels=int(n_recon_windowed),
        n_rect_pixels=int(n_rect),
        n_noise_shells=int(n_shells),
        n_fine_trans=int(n_fine_trans),
        score_complex_bytes=np.dtype(precision_policy.score_complex_dtype).itemsize,
        real_bytes=np.dtype(precision_policy.score_real_dtype).itemsize,
        norm_high_shell_bytes=np.dtype(norm_high_shell_dtype).itemsize,
        premultiplied_ctf=bool(premultiplied_ctf),
    )
    peak_bytes = operand_bytes + resident_image_capacity(int(n_images)) * max(
        int(n_windowed), int(n_recon_windowed), int(n_rect)
    ) * np.dtype(precision_policy.score_complex_dtype).itemsize
    peak_bytes += resident_prepare_reserved_bytes(image_shape)
    return operand_bytes, peak_bytes


def resident_operands_fit(operand_peak_bytes, available_bytes) -> bool:
    """Whether one half's resident operands are admitted at this reading.

    The one predicate for both the reservation before the projection cache
    decision and the admission after it.
    """

    return int(operand_peak_bytes) <= resident_operands_max_bytes(available_bytes)


def stream_projection_budget_bytes(
    max_projection_cache_bytes,
    *,
    physical_free_bytes,
    allocator_free_bytes,
    pool_free_bytes=None,
    reserved_bytes=0,
):
    """Chunk-local projection budget: the cache share, capped by measured free memory.

    The readings are taken when the pass plans its chunks, so they see whatever
    earlier passes and iterations left resident; the per-iteration cache share
    alone would ignore that. What the allocator can still hand out is
    :func:`~relax.runtime.memory_budget.device_available_bytes`; an
    unknown reading does not cap. ``reserved_bytes`` (the half's resident
    operands, allocated after the reading) comes off first; half of the rest
    stays for the half's accumulators and the chunk working set, which have
    their own budgets.
    """

    available = device_available_bytes(physical_free_bytes, allocator_free_bytes, pool_free_bytes)
    budget = int(max_projection_cache_bytes)
    if available is not None:
        usable = max(0.0, available - float(reserved_bytes))
        budget = min(budget, int(usable * _STREAM_FREE_MEMORY_FRACTION))
    return max(0, budget)


def _fitting_row_capacities(row_ladder, row_bytes: float, budget_bytes: float, what: str):
    """The ladder's row capacities whose ``capacity * row_bytes`` fits ``budget_bytes``.

    When no class fits, the smallest is halved down to :data:`_MIN_PLANNED_ROW_CAPACITY`, as the chunk
    memory plan does, and the largest that fits is the one class (a 16 GB V100's box-192 subtomogram final
    pass needed 4.43 GiB at 8192 rows against 3.78 GiB, relax#32). Below that floor the pass is refused,
    before any device work, with what to change. Ladders with a fitting class are unchanged.
    """

    kept = tuple(int(c) for c in row_ladder if float(c) * float(row_bytes) <= float(budget_bytes))
    if kept:
        return kept
    capacity = min(int(c) for c in row_ladder)
    while capacity > resident_chunk_memory.MIN_PLANNED_ROW_CAPACITY and float(capacity) * float(row_bytes) > float(budget_bytes):
        capacity //= 2
    if float(capacity) * float(row_bytes) <= float(budget_bytes):
        return (capacity,)
    # A configuration refusal, raised before any device work: there is no other pass-2 engine.
    raise ResidentConfigurationUnsupported(
        f"the device-resident sparse pass 2 does not fit this pass: even {capacity} rows need "
        f"{capacity * row_bytes / float(1024 ** 3):.2f} GiB of {what} against a "
        f"{budget_bytes / float(1024 ** 3):.2f} GiB budget. Run on a GPU with more memory, or on particles "
        "extracted at a smaller box."
    )


def stream_row_capacity_ladder(row_ladder, *, bytes_per_rotation, max_projection_bytes):
    """Row capacities whose chunk-local projection cache fits the cache budget.

    A streamed chunk projects at most one rotation per row, so capacity times
    the per-rotation bytes bounds its cache; the budget is the one the
    per-iteration cache failed, which keeps the pass's peak where it would
    have been had that cache fitted. See :func:`_fitting_row_capacities`.
    """

    return _fitting_row_capacities(
        row_ladder, STREAM_PEAK_COPIES * float(bytes_per_rotation), max_projection_bytes, "streamed projections"
    )


def cached_row_capacity_ladder(row_ladder, *, bytes_per_row, max_gather_bytes):
    """Row capacities whose gathered chunk of cached score projections fits the budget.

    With the whole fine grid cached, a chunk still gathers one score projection
    per row (``score_resident_chunk``: ``projection_score_cache[row_fine_rot]``),
    so capacity times the score-row bytes is one live block. RELION projects each
    orientation inside its diff2 kernel and never holds such a block; bounding the
    chunk by measured free memory keeps the gather within the device. The budget
    is :func:`stream_projection_budget_bytes` with the device as its cap, read
    after the cache is built. See :func:`_fitting_row_capacities`.
    """

    return _fitting_row_capacities(row_ladder, bytes_per_row, max_gather_bytes, "cached projections")


class _ProjectionSlots(NamedTuple):
    """A whole-pass cache that holds only the significant coarse parents' fine rotations.

    ``slot_projection`` [n_classes * capacity] is each cache row's projection id
    (``class * n_fine_rot + rotation``), class-major with ``capacity`` rows per
    class (a class's spare rows repeat its first id and are never read).
    ``slot_of_projection`` [n_classes * n_fine_rot] is each projection id's row,
    -1 where it is not cached.
    """

    capacity: int
    slot_projection: np.ndarray
    slot_of_projection: np.ndarray


_MIN_PROJECTION_SLOTS = 4096


def projection_cache_rows(n_classes: int, n_fine_rot: int, slots: "_ProjectionSlots | None") -> int:
    """Rows of a whole-pass projection cache: each class's slot capacity, or each class's whole fine grid."""

    return int(n_classes) * (int(n_fine_rot) if slots is None else int(slots.capacity))


def _projection_slot_capacity(n_needed: int, n_fine_rot: int) -> int | None:
    """Rows per class for ``n_needed`` cached rotations, or None when it would exceed half the grid.

    A power of two, so the chunk programs, which take the cache as an operand,
    see few cache shapes; inside one refinement a capacity it already used up
    to twice as large is reused rather than compiling a smaller one.
    """

    size = max(_MIN_PROJECTION_SLOTS, pow2_ceil(n_needed))
    history = resident_stable_windows.STABLE_WINDOW_CLASSES_RUN.get()
    if history is not None:
        used = history.setdefault(("projection_slots", int(n_fine_rot)), set())
        reusable = [capacity for capacity in used if size <= capacity <= 2 * size]
        size = min(reusable) if reusable else size
    if 2 * size > int(n_fine_rot):
        return None
    if history is not None:
        used.add(size)
    return size


def significant_projection_slots(
    class_supports, *, n_images: int, n_coarse_rot: int, n_coarse_trans: int, n_fine_rot: int, children
) -> _ProjectionSlots | None:
    """The cache rows a pass reads: the fine children of every class's significant coarse parents.

    RELION projects only each particle's significant orientations; the whole
    fine grid is 294912 rotations at HEALPix order 2 with oversampling, and a
    late VDAM iteration's 1000 images keep a few thousand coarse parents. The
    candidate rows of a class are exactly the children of its images' significant
    coarse rotations (:func:`csr_candidate_rows_per_image`: an empty support takes
    parent 0), so caching those children serves every row with the same
    projection (:func:`~relax.fine_pass.resident_significance.significant_coarse_parents`).
    None (cache the whole grid) when a class has no device CSR or an image whose
    support takes every parent, or when the children's capacity exceeds half the grid.
    """

    child_offsets, child_ids = (np.asarray(value, dtype=np.int64) for value in children)
    class_rotations = []
    for support in class_supports:
        parents = significant_coarse_parents(
            support, n_images=n_images, n_coarse_rot=n_coarse_rot, n_coarse_trans=n_coarse_trans
        )
        if parents is None:
            return None
        counts = child_offsets[parents + 1] - child_offsets[parents]
        starts = np.repeat(child_offsets[parents] - np.cumsum(counts) + counts, counts)
        class_rotations.append(np.unique(child_ids[starts + np.arange(int(counts.sum()))]))
    capacity = _projection_slot_capacity(max(rotations.size for rotations in class_rotations), n_fine_rot)
    if capacity is None:
        return None
    n_classes = len(class_rotations)
    slot_projection = np.empty(n_classes * capacity, dtype=np.int64)
    slot_of_projection = np.full(n_classes * int(n_fine_rot), -1, dtype=np.int32)
    for class_index, rotations in enumerate(class_rotations):
        rows = slot_projection[class_index * capacity : (class_index + 1) * capacity]
        rows[:] = rotations[0] if rotations.size else 0
        rows[: rotations.size] = rotations
        rows += class_index * int(n_fine_rot)
        slot_of_projection[rows[: rotations.size]] = class_index * capacity + np.arange(rotations.size, dtype=np.int32)
    return _ProjectionSlots(capacity, slot_projection, slot_of_projection)


def _stream_slot_count(n_unique: int, row_capacity: int) -> int:
    """Projection-call length for ``n_unique`` rotations: a quantum multiple, capped."""

    quantum = _STREAM_SLOT_QUANTUM
    return int(min(int(row_capacity), max(quantum, -(-int(n_unique) // quantum) * quantum)))


def stream_chunk_projections(
    rows,
    host_row_fine_rot,
    *,
    n_valid_rows,
    row_capacity,
    project,
    n_fine_rot,
    mstep_grid,
    coarse_parent_grid,
):
    """Project one chunk's distinct fine rotations and re-index its rows to them.

    ``host_row_fine_rot`` holds the rows' projection ids (the fine rotation, or
    ``class * n_fine_rot + rotation`` with K>1 classes) and ``project(ids, n_rows)``
    maps projection ids to the three projection arrays, ``n_rows`` long and zero
    past the ids.

    Returns ``(rows, slot_fine_rot, caches, mstep_grid, coarse_parent_grid)``
    where ``rows.row_fine_rot`` now holds each row's chunk-local cache slot,
    ``slot_fine_rot`` [row_capacity] maps a slot back to its global fine
    rotation id, and the caches and the two grids are indexed by slot. Every
    consumer gathers by slot exactly as it gathered by id from the
    per-iteration caches, so the gathered projections are the same arrays; the
    winning rotation is mapped back through ``slot_fine_rot``.

    The cache arrays are always ``row_capacity`` long, so the chunk programs
    stay keyed on the capacity class alone; only the projection call is
    quantised to the chunk's distinct-rotation count. Valid rows read only the
    first ``n_unique`` slots; padded rows read slot 0, a real rotation, and
    carry a zero posterior. Slots past the projection call are zero and unread.
    """

    valid = np.asarray(host_row_fine_rot[: int(n_valid_rows)], dtype=np.int64)
    unique, inverse = np.unique(valid, return_inverse=True)
    if unique.size == 0:
        unique = np.zeros(1, dtype=np.int64)
    n_project = _stream_slot_count(unique.size, row_capacity)
    slot_fine_rot = np.full(int(row_capacity), unique[0], dtype=np.int64)
    slot_fine_rot[: unique.size] = unique
    row_slot = np.zeros(int(row_capacity), dtype=np.int32)
    row_slot[: valid.size] = inverse.astype(np.int32, copy=False)

    slot_ids_device = jnp.asarray(slot_fine_rot, dtype=jnp.int32)
    # Projected straight into row_capacity-long arrays: padding the projection
    # call's arrays afterwards held both copies, 9.73 GiB past the plan at K=1
    # 50k/256 (bench 14576794).
    caches = tuple(project(slot_fine_rot[:n_project], int(row_capacity)))
    return (
        rows._replace(row_fine_rot=jnp.asarray(row_slot, dtype=jnp.int32)),
        jnp.asarray(slot_fine_rot % int(n_fine_rot), dtype=jnp.int32),
        caches,
        mstep_grid[slot_ids_device],
        coarse_parent_grid[slot_ids_device],
    )


# The most row blocks a whole-grid cache is split into before the pass streams
# its projections instead; each block adds one gather to every cached read.
MAX_PROJECTION_CACHE_BLOCKS = 4


def allocate_projection_cache_blocks(n_rows: int, n_pixels: int, dtype):
    """The zero whole-grid cache: one array, else a tuple of 2 or 4 equal row blocks, else ``None``.

    A fragmented pool can hold far more free memory than any one block
    (Class3D K4 100k from iteration 19: a 6.32 GiB cache against a 5.25 GiB
    largest block with 56 GiB free, bench 14719385), so a cache the allocator
    refuses whole is tried in row blocks (:func:`~relax.fine_pass.resident_scoring.cache_rows`)
    before the pass streams; ``None`` means even the blocks were refused
    (EMPIAR-10345 it13 half 2, bench 14643272).
    """

    n_blocks = 1
    while n_blocks <= MAX_PROJECTION_CACHE_BLOCKS:
        height = -(-int(n_rows) // n_blocks)
        blocks = []
        for _ in range(n_blocks):
            block = _allocate_projection_cache(height, n_pixels, dtype)
            if block is None:
                break
            blocks.append(block)
        if len(blocks) == n_blocks:
            return blocks[0] if n_blocks == 1 else tuple(blocks)
        del blocks
        n_blocks *= 2
    return None


def _allocate_projection_cache(n_rows: int, n_pixels: int, dtype):
    """A zero ``[n_rows, n_pixels]`` cache, or ``None`` when the allocator cannot hand it out.

    Only an out-of-memory refusal returns ``None``; any other error is raised.
    """

    try:
        cache = jnp.zeros((int(n_rows), int(n_pixels)), dtype=dtype)
        cache.block_until_ready()
    except Exception as exc:  # noqa: BLE001 - JAX raises its runtime error for an allocation refusal
        if "RESOURCE_EXHAUSTED" not in str(exc):
            raise
        return None
    return cache


def build_projection_cache_in_place(
    project_rows, *, n_classes, n_rows_per_class, n_pixels, rows_per_call, dtype, cache=None
):
    """A ``[n_classes * n_rows_per_class, n_pixels]`` cache, class-major, filled one projector call at a time.

    ``project_rows(class_index, start, stop)`` returns that class's rows
    ``[start, stop)``. Each call's rows are written into the one cache in place,
    so the cache is never held twice: concatenating the per-class caches (and
    the per-call chunks inside each) held it twice, 19 GiB past the plan at
    VDAM pdb K=2 (main 46b4f64, iteration 96). ``cache``, when given, is that
    array already allocated (:func:`_allocate_projection_cache`).
    """

    if cache is None:
        cache = jnp.zeros((int(n_classes) * int(n_rows_per_class), int(n_pixels)), dtype=dtype)
    blocks = list(cache) if isinstance(cache, tuple) else [cache]
    height = int(blocks[0].shape[0])
    for class_index in range(int(n_classes)):
        for start in range(0, int(n_rows_per_class), int(rows_per_call)):
            stop = min(start + int(rows_per_call), int(n_rows_per_class))
            rows = project_rows(class_index, start, stop)
            row = class_index * int(n_rows_per_class) + start
            # A call's rows go to the row blocks they fall in (one block when the cache is one array).
            offset = 0
            while offset < int(rows.shape[0]):
                index, local = divmod(row + offset, height)
                count = min(int(rows.shape[0]) - offset, height - local)
                part = rows if count == int(rows.shape[0]) else rows[offset : offset + count]
                blocks[index] = write_projection_cache_rows(blocks[index], part, np.int32(local))
                del part
                offset += count
            del rows
    return blocks[0] if len(blocks) == 1 else tuple(blocks)


@partial(jax.jit, donate_argnums=0)
def write_projection_cache_rows(cache, rows, start):
    """``cache`` with ``rows`` written at row ``start``, in place (the cache is donated)."""

    return jax.lax.dynamic_update_slice(cache, rows.astype(cache.dtype), (start, jnp.int32(0)))
