"""The tile memory planner of the streamed PPCA engine.

``plan_tile_images`` picks the most images per tile whose host reader, stream buffers and block programs fit the
device (``tile_bytes``, ``stream_tile_bytes``, ``tile_program_bytes``, the last traced from the kernels of
``full_row_kernels``); plans are cached per stream shape (``_PLANS``, keyed by ``plan_shape_key``, one entry per
shape for the process) and ``reserve_plan_region`` holds the planned region on the device. Imports nothing from
the driver (``full_row_stream``).
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import numpy as np
from recovar.ppca.triangular import tri_size

from relax.ppca_refinement.full_row_kernels import (
    SHIFT_ALIGN,
    FullRowStream,
    Posterior,
    TileArrays,
    empty_carry,
    empty_kept,
    moment_block,
    moment_groups,
    score_block,
)

logger = logging.getLogger("relax.ppca_refinement.full_row_stream")  # the stream's logger before the split: log rows keep it


# Device memory a tile plan leaves unplanned for allocator fragmentation and XLA program
# temporaries the plan does not count, as a share of the device's memory.
TILE_FRAGMENTATION_HEADROOM = 0.10


def _window_pixels(stream: FullRowStream) -> int:
    """Pixels per frame of the tile operands (the padded GEMM window on GPU streams)."""
    arrays = stream.arrays
    if arrays.gemm_window is not None:
        return int(arrays.gemm_window.shape[0])
    if arrays.score_indices is not None:
        return int(np.asarray(arrays.score_indices).size)
    return int(arrays.coefficient_noise.size)


# What a tile reader other than the single-particle one must report for the tile planner.
READER_PLAN_ATTRIBUTES = ("operand_bytes", "max_frames")


def _plan_reader(stream: FullRowStream):
    """The stream's custom tile reader, after checking it reports what the planner counts; None for
    the single-particle reader. A reader without them would be planned as single particles, which
    can run a card out of memory, so that is an error."""
    reader = stream.tile_loader
    if reader is None:
        return None
    missing = [name for name in READER_PLAN_ATTRIBUTES if not callable(getattr(reader, name, None))]
    if missing:
        raise ValueError(
            f"Tile reader {getattr(reader, '__qualname__', reader)!r} has no {', '.join(missing)}: the tile "
            "planner cannot count its memory (see READER_PLAN_ATTRIBUTES)"
        )
    return reader


def tile_frames(stream: FullRowStream) -> int:
    """Projection frames per tile row: a custom reader's ``max_frames`` (a subtomogram reader's tilts), else 1."""
    reader = _plan_reader(stream)
    return 1 if reader is None else int(reader.max_frames(stream))


def tile_spec(stream: FullRowStream, n_images: int) -> TileArrays:
    """Shapes and dtypes of a tile of ``n_images`` as the stream's readers build it (:class:`TileArrays`)."""
    B, T = int(n_images), int(stream.translations.shape[0])
    K, F = tile_frames(stream), _window_pixels(stream)
    BT = B * T + (-(B * T) % SHIFT_ALIGN if stream.static.cuda_kernels else 0)
    f32 = jnp.float32
    spec = jax.ShapeDtypeStruct
    return TileArrays(
        coarse_mask=spec((B, stream.n_coarse_rotations + 1, stream.n_coarse_translations), jnp.bool_),
        rows=spec((len(stream.block_starts) * stream.rotation_block_size,), jnp.int32),
        Y1=spec((K * 2 * F, BT), f32),
        ctf2=spec((K * F, B), f32),
        Y1_recon=spec((BT, K * 2 * F), f32),
        ctf2_recon=spec((B, K * F), f32),
        y_norm=spec((B,), f32),
        frames=None if _plan_reader(stream) is None else spec((K, 3, 3), f32),
    )


_PROGRAM_BYTES: dict = {}


def tile_program_bytes(stream: FullRowStream, n_images: int) -> int:
    """Device bytes the pass-1 and pass-2 block programs add for a tile of ``n_images``, from XLA.

    The larger of :func:`score_block` and :func:`moment_block` (moments) at the tile's shapes:
    each program's temporaries plus the outputs that do not alias its donated input (the kept
    results, the moment carry), from ``lower(...).compile().memory_analysis()``. Cached per stage
    shape and size. These are separate compiles from the tiles' own (their operands' device
    placement differs, so the cache keys do): about 0.3 s per stage on A100 at EMPIAR-10076's shapes.
    """
    key = (plan_shape_key(stream), int(n_images), getattr(stream.device, "id", None))
    if key not in _PROGRAM_BYTES:
        static, block = stream.static, stream.rotation_block_size
        capacity = len(stream.block_starts) * block
        T, q = int(stream.translations.shape[0]), static.basis_size - 1
        tile = tile_spec(stream, n_images)
        kept = jax.eval_shape(lambda: empty_kept(capacity, int(n_images), T, q, jnp.float32))
        carry = jax.eval_shape(lambda: empty_carry(stream, int(n_images), jnp.float32(0)))
        vector = jax.ShapeDtypeStruct((int(n_images),), jnp.float32)
        index = jax.ShapeDtypeStruct((int(n_images),), jnp.int32)
        posterior = Posterior(vector, vector, vector, index, index)
        start = jax.ShapeDtypeStruct((), jnp.int32)
        with jax.default_device(stream.device):
            programs = (
                score_block.lower(stream.arrays, tile, kept, start, static=static, block_size=block),
                moment_block.lower(
                    carry, stream.arrays, tile, kept, posterior, start, static=static, block_size=block, moments=True
                ),
            )
            analyses = [program.compile().memory_analysis() for program in programs]
        if any(analysis is None for analysis in analyses):
            raise RuntimeError("XLA reported no memory analysis for the stream's block programs")
        _PROGRAM_BYTES[key] = max(
            int(a.temp_size_in_bytes) + int(a.output_size_in_bytes) - int(a.alias_size_in_bytes) for a in analyses
        )
    return _PROGRAM_BYTES[key]


def stream_tile_bytes(stream: FullRowStream, n_images: int) -> int:
    """Device bytes of the stream's own buffers for a tile of ``n_images``.

    The kept pass-1 results over the full row table (:func:`empty_kept`, from their shapes) with,
    under a pass-2 mass floor, its compacted copy of at most half the rows; the moment accumulators
    and their compensation (:func:`empty_carry`); and the block programs' own memory
    (:func:`tile_program_bytes`: projections, GEMM outputs, weights and M-step images).
    """
    static = stream.static
    P = static.basis_size
    T = int(stream.translations.shape[0])
    capacity = len(stream.block_starts) * stream.rotation_block_size
    kept = capacity * (T * P + tri_size(P - 1) + 3)
    if stream.pass2_mass_floor > 0:
        kept = kept + kept // 2  # the compacted pass-2 copy of at most half the rows (_compact_capacity)
    accumulators = 2 * moment_groups(static) * int(stream.arrays.augmented.shape[1]) * 32
    return 4 * (int(n_images) * kept + accumulators) + tile_program_bytes(stream, n_images)


def single_particle_reader_bytes(stream: FullRowStream, n_images: int) -> tuple[int, int]:
    """``(peak, resident)`` device bytes of :func:`_load_tile` for ``n_images``, from its arrays.

    Peak counts every array the reader creates as live at once: the shifted half images
    ``(B T, n_half)`` complex from ``preprocess_batch``, their scaled and masked score and
    reconstruction copies, the windowed copies, the GEMM-window padding, the planar ``[Re | Im]``
    stacks and their padded transposes, and the CTF rows. Resident is the tile's two planar operands
    and CTF rows, live until the tile finishes.
    """
    T = int(stream.translations.shape[0])
    n_half = int(stream.arrays.coefficient_noise.size)
    F = _window_pixels(stream)
    complex_full = 2 * T * n_half  # one (T, n_half) complex64 image stack, in float32 values
    complex_window = 2 * T * F
    planar = 2 * T * F
    peak = 3 * complex_full + 4 * complex_window + 4 * planar + 6 * n_half
    resident = 2 * planar + 2 * F
    return 4 * int(n_images) * peak, 4 * int(n_images) * resident


def _reader_bytes(stream: FullRowStream):
    """The tile reader's ``(peak, resident)`` byte count: a custom reader's ``operand_bytes`` (the
    subtomogram reader's, from XLA's compiled memory analysis of its operand program), else
    :func:`single_particle_reader_bytes`."""
    reader = _plan_reader(stream)
    return single_particle_reader_bytes if reader is None else reader.operand_bytes


def tile_bytes(stream: FullRowStream, n_images: int, reader=None, *, pipelined: bool = True) -> int:
    """Device bytes a tile of ``n_images`` needs at its peak.

    Tiles of one :func:`accumulate_full_row_tiles` call are pipelined: the next tile's reader runs
    while the current tile's operands and the shared kept buffer are live, so the count is the
    stream's own buffers (:func:`stream_tile_bytes`), one tile's resident operands and one reader
    peak. A call of one tile (``pipelined=False``) reads it before its passes run: the larger of the
    reader peak and the stream's buffers with the tile's operands.
    """
    peak, resident = (reader or _reader_bytes(stream))(stream, n_images)
    own = stream_tile_bytes(stream, n_images) + resident
    return own + peak if pipelined else max(own, peak)


def plan_tile_images(
    stream: FullRowStream, requested: int, *, memory_bytes=None, device_bytes=None, tiles_per_call=None
) -> int:
    """Tile images (particles) per tile: ``requested``, or the most that fit the device (:func:`tile_bytes`).

    The budget is ``memory_bytes`` minus :data:`TILE_FRAGMENTATION_HEADROOM` of ``device_bytes``. On a
    GPU stream they default to what the device can still hand out after the stream's upload
    (:func:`relax.runtime.memory_budget.device_available_bytes`) and the device's memory;
    on a CPU stream to the host memory left to this process and the host memory it may use
    (:mod:`relax.runtime.host_memory`). Every stage (radius, window, pose grid) is planned with its
    own shapes, once: with the probed defaults the plan is kept per stage shape, measured at the
    stage's first update. A stage whose one-image tile does not fit is refused: its rotation block
    is too large for the device.

    ``tiles_per_call(size)`` gives the most tiles one :func:`accumulate_full_row_tiles` call holds at
    a tile size (never fewer for smaller tiles). Tiles are read ahead only within a call, so when
    every call holds one tile at the plan without read-ahead, that plan is used; otherwise (or when
    it is None) the next tile's reader is counted.

    The tile size sets the order of the update's sums (each tile's statistics are added in tile order), so a
    plan that shrinks tiles makes the numbers follow the device's memory. At the default 150 images per tile it
    does not on our A100s: every stage of the box-128 and box-192 schedules up to HEALPix 4, radius 37, keeps
    150 images on an 80 GB card and on one limited to a 40 GB card's allocator (largest tile 26.9 of 31 GiB;
    2026-10-18). Larger boxes, finer grids or smaller cards can shrink them.
    """
    if tiles_per_call is not None and tiles_per_call(requested) <= 1:
        single = _planned_tile_images(stream, requested, memory_bytes, device_bytes, pipelined=False)
        if tiles_per_call(single) <= 1:
            return single
    return _planned_tile_images(stream, requested, memory_bytes, device_bytes, pipelined=True)


def _planned_tile_images(stream, requested: int, memory_bytes, device_bytes, *, pipelined):
    if memory_bytes is None and device_bytes is None:
        # The device probes run nvidia-smi (tens of ms); a stage's shapes are planned once.
        key = (plan_shape_key(stream), requested, stream.device.id, pipelined)
        if key not in _PLANS:
            _PLANS[key] = _planned_tile_images(stream, requested, *available_bytes(stream), pipelined=pipelined)
            if stream.static.cuda_kernels:
                reserve_plan_region(stream.device, tile_bytes(stream, _PLANS[key], pipelined=pipelined))
        return _PLANS[key]
    if not memory_bytes:
        return requested
    budget_bytes, device_bytes = fragmentation_budget(stream, memory_bytes, device_bytes)
    reader = _reader_bytes(stream)
    planned = _plan_tile_images(stream, requested, reader, budget_bytes, pipelined)
    peak, resident = reader(stream, planned)
    gib = 2**30
    counted = tile_bytes(stream, planned, reader, pipelined=pipelined)
    if counted > budget_bytes:
        raise ValueError(
            f"A tile of one image needs {counted / gib:.2f} GiB on the {stream.device.platform} "
            f"(block programs {tile_program_bytes(stream, planned) / gib:.2f} GiB at rotation block "
            f"{stream.rotation_block_size}), more than its {budget_bytes / gib:.2f} GiB budget: "
            "use a smaller rotation block"
        )
    logger.info(
        "PPCA tile plan: %d of %d images per tile (%d frames each); counted %.2f GiB of a %.2f GiB budget "
        "(%.2f GiB available, %.2f GiB device, headroom %.0f%%): stream %.2f GiB of which block programs "
        "%.2f GiB, tile operands %.2f GiB, reader %.2f GiB (%s)",
        planned,
        requested,
        tile_frames(stream),
        counted / gib,
        budget_bytes / gib,
        memory_bytes / gib,
        (device_bytes or memory_bytes) / gib,
        100 * TILE_FRAGMENTATION_HEADROOM,
        stream_tile_bytes(stream, planned) / gib,
        tile_program_bytes(stream, planned) / gib,
        resident / gib,
        peak / gib,
        "next tile read ahead" if pipelined else "one tile per call",
    )
    return planned


# Plans by stage shapes (:func:`plan_shape_key`) for the device probes' defaults.
_PLANS: dict = {}


def plan_shape_key(stream: FullRowStream) -> tuple:
    """Everything :func:`tile_bytes` reads from a stream: a stage's pose grid, window, rank and frames."""
    static = stream.static
    return (
        len(stream.block_starts),
        stream.rotation_block_size,
        int(stream.translations.shape[0]),
        _window_pixels(stream),
        int(stream.arrays.coefficient_noise.size),
        int(stream.arrays.augmented.shape[1]),
        static.basis_size,
        static.metric_trace_only,
        tile_frames(stream),
        stream.tile_loader,
    )


def fragmentation_budget(stream, memory_bytes, device_bytes):
    """``(budget, device_bytes)``: ``memory_bytes`` less :data:`TILE_FRAGMENTATION_HEADROOM` of the device's memory.

    ``device_bytes`` None on a GPU stream (what :func:`available_bytes` returns there) is the device's memory
    limit; on a CPU stream, or when no total is known, the headroom is taken of ``memory_bytes``.
    """
    if device_bytes is None and stream.static.cuda_kernels:
        from relax.runtime import memory_budget as budget

        device_bytes = budget.device_memory_limit_bytes(None)  # PPCA takes no EngineTuning: the device's own memory
    return memory_bytes - TILE_FRAGMENTATION_HEADROOM * (device_bytes or memory_bytes), device_bytes


def available_bytes(stream):
    """``(available, total)`` memory for the stream's tiles: the GPU's after the stream's upload and
    its memory, or the host memory left to this process and the host memory it may use."""
    if not stream.static.cuda_kernels:
        from relax.runtime.host_memory import available_memory_bytes, resident_bytes

        host = available_memory_bytes()
        return host - resident_bytes(), host
    from relax.runtime import memory_budget as budget

    available = budget.current_device_available_bytes()
    return available, None


# Memory a plan's region leaves to the device (allocator rounding and the device's other users).
PLAN_REGION_MARGIN_BYTES = 256 * 2**20


def plan_region_bytes(counted_bytes, limit_bytes, pool_bytes, physical_free_bytes) -> int:
    """Size of the one block that gives a plan's counted bytes contiguous memory in the allocator's pool.

    The counted bytes, or, when the allocator may no longer take that much from the device, all it
    still may: its limit minus its pool, bounded by the device's free memory, less
    :data:`PLAN_REGION_MARGIN_BYTES`. Zero when a reading is missing or nothing is left to take (a
    preallocated pool, or a pool earlier plans grew to its limit).
    """
    if limit_bytes is None or pool_bytes is None or physical_free_bytes is None:
        return 0
    unreserved = min(int(limit_bytes) - int(pool_bytes), int(physical_free_bytes)) - PLAN_REGION_MARGIN_BYTES
    return max(0, min(int(counted_bytes), unreserved))


def reserve_plan_region(device, counted_bytes) -> int:
    """Give a plan's counted bytes one contiguous region of the GPU allocator's pool; returns the block's bytes.

    A plan counts bytes, and the allocator hands out a block only from one contiguous region.
    Without preallocation (``XLA_PYTHON_CLIENT_PREALLOCATE=false``) the pool grows by regions sized
    by the requests so far, so the small early stages leave a pool of small regions in which a later
    stage's large block finds no room while its bytes are free: an 8.48 GiB tilt-reader block failed
    with 9.2 GiB in use of a 36.4 GiB pool on a 40 GB A100 (on Polar; reproduced on an H100
    limited to that pool, where the same run with a preallocated pool passes).
    Allocating :func:`plan_region_bytes` as one block and releasing it makes the allocator grow the
    pool by a region of at least that size, unless the pool already has such a block free. The plan
    and the results do not change: the allocator can hand out the same bytes before and after, and
    the pool only takes earlier what the planned tiles would take.
    """
    from relax.runtime import memory_budget as budget

    stats = device.memory_stats() or {}
    size = plan_region_bytes(
        counted_bytes, stats.get("bytes_limit"), stats.get("pool_bytes"), budget.device_free_memory_bytes()
    )
    megabytes = size // 2**20
    if megabytes == 0:
        return 0
    with jax.default_device(device):
        jnp.zeros((megabytes, 2**20), jnp.uint8).block_until_ready()
    grown = int(device.memory_stats()["pool_bytes"]) - int(stats["pool_bytes"])
    if grown:
        logger.info(
            "PPCA tile plan: the allocator pool grew by %.2f GiB for one block of %.2f GiB",
            grown / 2**30,
            megabytes / 2**10,
        )
    return megabytes * 2**20


def _plan_tile_images(stream, requested, reader, budget_bytes, pipelined):
    exact = {requested: tile_bytes(stream, requested, reader, pipelined=pipelined)}
    if exact[requested] <= budget_bytes:
        return requested
    # Tile bytes are affine in the tile size (every counted buffer and program grows linearly in the
    # images): two exact counts (1 and ``requested``) place every size between them, so the search
    # compiles no other program. The chosen size is then counted exactly.
    exact[1] = tile_bytes(stream, 1, reader, pipelined=pipelined)
    slope = (exact[requested] - exact[1]) / max(requested - 1, 1)
    low = max(1, min(requested, 1 + int((budget_bytes - exact[1]) // slope) if slope > 0 else requested))
    while low > 1:
        exact.setdefault(low, tile_bytes(stream, low, reader, pipelined=pipelined))
        if exact[low] <= budget_bytes:
            break
        low = max(1, min(low - 1, int(low * budget_bytes / exact[low])))
    return low
