"""Shared source-precision RELION CTF operands for coarse, local and sparse EM.

The source STAR and native CTF cache retain their original per-process lifetime.
Host callers can gather planned pixels before stacking; the device accessor
places the full operand once. Native RFLOAT values, centered half-spectrum
coordinates, signs and caller-selected cast boundaries are preserved.
"""

from __future__ import annotations

import collections
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from recovar.data_io.starfile import star_column

from relax.helpers.batch_fetch import original_image_indices
from relax.relion.tomo_input import fftw_half_freq_sq, relion_tomo_damping

_RELION_EXACT_CTF_SOURCE_CACHE: dict[tuple[str, tuple[int, int]], dict] = {}

# Assembled-operand memo. The per-image rows below are already cached, but the
# stack that turns them into one operand is rebuilt on every call: pass 2 asks
# for a whole half (4966 rows of 33024 float64 = 1.31 GB of host memcpy, 250 ms
# per half per iteration), and the coarse pass asks for one image batch of
# planned columns. Both requests repeat with identical inputs every iteration,
# so the assembled array is a pure function of the key and the memo returns the
# very same object: bit-for-bit by construction, not by re-derivation.
_EXACT_CTF_RESULT_CACHE: "collections.OrderedDict[tuple, tuple]" = collections.OrderedDict()
_EXACT_CTF_RESULT_BYTES = 0
_EXACT_CTF_CACHE_GB_ENV = "RELAX_RELION_EXACT_CTF_CACHE_GB"


def _exact_ctf_result_cache_budget_bytes() -> int:
    """Host bytes the assembled-operand memo may hold; 0 disables it.

    The default is 4 GB since the P4-I merge, which holds the two 1.31 GB
    whole-half operands of the 256x256 fixture and costs about 2.5 GB of host
    memory for 2.0 s of a steady hp3 iteration. Set the variable to 0 to
    restore the unmemoized assembly, which stays the oracle.

    The memo trades host memory for the repeated stack: a whole-half operand is
    1.31 GB at 256x256, so a budget below that disables the pass-2 entry while
    still holding the much smaller per-batch coarse entries.
    """

    token = os.environ.get(_EXACT_CTF_CACHE_GB_ENV, "4").strip()
    try:
        budget = float(token)
    except ValueError as exc:
        raise ValueError(f"{_EXACT_CTF_CACHE_GB_ENV} must be a number of gigabytes, got {token!r}") from exc
    if budget < 0:
        raise ValueError(f"{_EXACT_CTF_CACHE_GB_ENV} must not be negative, got {token!r}")
    return int(budget * (1024 ** 3))


def _exact_ctf_result_key(source_path, image_shape, original_indices, pixel_indices):
    """A content key for the assembled operand, with the arrays kept for verification."""

    import hashlib

    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.ascontiguousarray(original_indices).view(np.uint8))
    if pixel_indices is None:
        pixel_tag = b"none"
    else:
        pixel_tag = np.ascontiguousarray(pixel_indices).view(np.uint8).tobytes()
        digest.update(b"|pixels|")
        digest.update(pixel_tag)
    return (
        str(source_path),
        tuple(int(size) for size in image_shape),
        int(original_indices.size),
        None if pixel_indices is None else (str(pixel_indices.dtype), int(pixel_indices.size)),
        digest.digest(),
    )


def _exact_ctf_result_lookup(key, original_indices, pixel_indices):
    """Return the memoized operand for an exactly equal request, else None."""

    entry = _EXACT_CTF_RESULT_CACHE.get(key)
    if entry is None:
        return None
    cached_indices, cached_pixels, result = entry
    # A digest match is not trusted on its own: compare the actual selectors, so
    # a collision can never serve one image set's CTFs for another's.
    if not np.array_equal(cached_indices, original_indices):
        return None
    if (cached_pixels is None) != (pixel_indices is None):
        return None
    if cached_pixels is not None and not np.array_equal(cached_pixels, pixel_indices):
        return None
    _EXACT_CTF_RESULT_CACHE.move_to_end(key)
    return result


def _exact_ctf_result_store(key, original_indices, pixel_indices, result):
    """Memoize one assembled operand, evicting least-recently-used entries."""

    global _EXACT_CTF_RESULT_BYTES

    budget = _exact_ctf_result_cache_budget_bytes()
    if budget <= 0 or result.nbytes > budget:
        return result
    # The memo hands the same object to every caller, so make it read-only: no
    # caller mutates it today (each wraps it in np.asarray and uploads), and a
    # future one must not silently corrupt a shared operand.
    result.setflags(write=False)
    _EXACT_CTF_RESULT_CACHE[key] = (
        np.array(original_indices, copy=True),
        None if pixel_indices is None else np.array(pixel_indices, copy=True),
        result,
    )
    _EXACT_CTF_RESULT_BYTES += result.nbytes
    while _EXACT_CTF_RESULT_BYTES > budget and len(_EXACT_CTF_RESULT_CACHE) > 1:
        _, evicted = _EXACT_CTF_RESULT_CACHE.popitem(last=False)
        _EXACT_CTF_RESULT_BYTES -= evicted[2].nbytes
    return result


def clear_exact_ctf_result_cache() -> None:
    """Drop the assembled-operand memo (tests and long-lived processes)."""

    global _EXACT_CTF_RESULT_BYTES

    _EXACT_CTF_RESULT_CACHE.clear()
    _EXACT_CTF_RESULT_BYTES = 0


def _relion_exact_ctf_source_star(experiment_dataset) -> Path:
    """Resolve the immutable source STAR for exact RELION CTF evaluation."""

    source_star = os.environ.get("RELAX_K1_RELION_EXACT_CTF_STAR", "").strip()
    if not source_star:
        dataset_source = getattr(experiment_dataset, "particles_file", None)
        if dataset_source and Path(dataset_source).suffix.lower() == ".star":
            source_star = str(dataset_source)
    if not source_star:
        raise ValueError(
            "exact RELION operands require a STAR-backed dataset or "
            "RELAX_K1_RELION_EXACT_CTF_STAR"
        )
    return Path(source_star).expanduser().resolve()


def _relion_ctf_threads() -> int:
    """Worker threads for the batched RELION CTF binding: the CPUs this process may use."""

    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # pragma: no cover - non-Linux
        return max(1, os.cpu_count() or 1)


def _relion_ctf_column(cache, name: str, default: float | None, *, optics_only: bool = False) -> np.ndarray:
    """One CTF parameter for every particle, resolved as ``CTF::readValue`` resolves it."""

    columns = cache.setdefault("columns", {})
    if name in columns:
        return columns[name]
    particles = cache["particles"]
    groups = np.asarray(star_column(particles, "rlnOpticsGroup", required=True), dtype=np.int64)
    values = None if optics_only else star_column(particles, name)
    if values is not None:
        column = np.asarray(values, dtype=np.float64)
    else:
        by_group = {}
        for group, optics in cache["optics"].items():
            found = [float(optics[key]) for key in (name, f"_{name}") if key in optics]
            if found:
                by_group[group] = found[0]
            elif default is None:
                raise KeyError(f"RELION optics group {group} has no {name}")
            else:
                by_group[group] = float(default)
        column = np.asarray([by_group[int(group)] for group in groups], dtype=np.float64)
    columns[name] = column
    return column


def _relion_ctf_batch_params(cache, original_indices: np.ndarray) -> np.ndarray:
    """``get_ctf_images_batch`` rows for these particles, in the get_ctf_image argument order."""

    idx = np.asarray(original_indices, dtype=np.int64)
    names = (
        ("rlnDefocusU", None, False),
        ("rlnDefocusV", None, False),
        ("rlnDefocusAngle", None, False),
        ("rlnVoltage", None, True),
        ("rlnSphericalAberration", None, True),
        ("rlnAmplitudeContrast", None, True),
        ("rlnCtfBfactor", 0.0, False),
        ("rlnImagePixelSize", None, True),
        ("rlnPhaseShift", 0.0, False),
        ("rlnCtfScalefactor", 1.0, False),
    )
    return np.ascontiguousarray(
        np.stack([_relion_ctf_column(cache, n, d, optics_only=o)[idx] for n, d, o in names], axis=1)
    )


def _exact_ctf_source_cache(experiment_dataset, image_shape):
    """The source STAR, its parsed tables and the per-particle CTF row block."""

    source_path = _relion_exact_ctf_source_star(experiment_dataset)
    cache_key = (str(source_path), tuple(int(size) for size in image_shape))
    cache = _RELION_EXACT_CTF_SOURCE_CACHE.get(cache_key)
    if cache is None:
        from recovar.data_io.starfile import read_star

        from relax.relion_bind import _relion_bind_core as relion_bind

        particles, optics = read_star(str(source_path))
        if optics is None:
            raise ValueError(f"RELION source STAR has no optics table: {source_path}")
        optics_ids = np.asarray(star_column(optics, "rlnOpticsGroup", required=True), dtype=np.int64)
        if np.unique(optics_ids).size != optics_ids.size:
            raise ValueError(f"RELION source STAR has duplicate optics groups: {source_path}")
        # Cached CTF rows live in one 2-D block rather than a dict of rows, so a
        # batch is gathered with two vectorized indexing operations instead of a
        # Python loop per particle. `slots` maps a particle's original index to its
        # row in that block, with -1 meaning "not evaluated yet"; the block has a
        # row for every particle and is allocated once.
        cache = {
            "particles": particles,
            "optics": {
                int(group): optics.iloc[row]
                for row, group in enumerate(optics_ids)
            },
            "relion_bind": relion_bind,
            # Per-tilt rows of RELION tomo particles (tomo_input.flatten_relion5_tomo).
            "tomo": star_column(particles, "rlnMicrographPreExposure") is not None,
            "slots": np.full(len(particles), -1, dtype=np.int64),
            "rows": None,
            "n_cached": 0,
        }
        _RELION_EXACT_CTF_SOURCE_CACHE[cache_key] = cache

    return source_path, cache


def _evaluate_exact_ctf_rows(cache, original_indices, image_h: int, image_w: int) -> np.ndarray:
    """Evaluate the particles' missing CTF rows into the block; return their row slots."""

    slots = cache["slots"]
    missing = np.unique(original_indices[slots[original_indices] < 0])
    if missing.size:
        # One threaded RELION call evaluates every missing particle
        # (get_ctf_images_batch: get_ctf_image's arithmetic per row).
        n_new = int(missing.size)
        params = _relion_ctf_batch_params(cache, missing)
        native = np.asarray(
            cache["relion_bind"].get_ctf_images_batch(
                params, image_w, image_h, False, False, False, _relion_ctf_threads()
            ),
            dtype=np.float64,
        )
        if cache.get("tomo", False):
            # A RELION tomo image (one row per particle-tilt): relion_refine damps its
            # CTF by the tilt's cumulative dose (tomo_input.relion_tomo_damping). RELION
            # multiplies before the scale factor and before the |CTF| >= 1e-8 floor
            # (src/ctf.h:219-253); applying it here reorders one product and moves the
            # floor, both below 1e-8 absolute.
            dose = _relion_ctf_column(cache, "rlnMicrographPreExposure", None)[missing]
            dose_bfactor = _relion_ctf_column(cache, "rlnCtfBfactorPerElectronDose", 0.0)[missing]
            for row in range(n_new):
                native[row] = native[row] * relion_tomo_damping(
                    fftw_half_freq_sq(image_h, image_w, params[row, 7]), dose[row], dose_bfactor[row]
                )
        # RELION/FFTW stores y in standard order and uses the opposite CTF
        # sign from RECOVAR's forward-model convention: one roll and negation
        # of the row axis, written straight into the cache block.
        rows = cache["rows"]
        used = cache["n_cached"]
        width = image_h * (image_w // 2 + 1)
        if rows is None:
            # One block with a row for every particle of the source STAR; pages are
            # touched only as rows are evaluated. Growing by doubling recopied the
            # whole cache (17.5 s of a noise1 50k VDAM run, job 14523070).
            rows = np.empty((slots.size, width), dtype=np.float64)
        shift = image_h // 2            # np.fft.fftshift is np.roll(x, image_h // 2)
        split = image_h - shift
        block = rows[used : used + n_new].reshape(n_new, image_h, image_w // 2 + 1)
        np.negative(native[:, split:], out=block[:, :shift])
        np.negative(native[:, :split], out=block[:, shift:])
        cache["rows"] = rows
        cache["n_cached"] = used + n_new
        slots[missing] = np.arange(used, used + n_new, dtype=np.int64)

    batch_slots = slots[original_indices]
    if np.any(batch_slots < 0):
        raise RuntimeError("a requested RELION CTF row was not evaluated")
    return batch_slots


def _relion_exact_ctf_half_from_source_star_host(
    experiment_dataset,
    image_indices,
    image_shape,
    *,
    pixel_indices=None,
):
    """Evaluate source-precision SPA CTFs into one host-native operand.

    The result uses RECOVAR's centered-y half-spectrum coordinates and sign.
    The source STAR is mandatory because the ordinary dataset metadata has
    already been rounded to float32 before pass 2.  RELION's binding and the
    source cache are host-native; callers that must pad on the image axis use
    this helper so they place the final operand exactly once. Optional host
    pixel indices gather the requested columns before stacking full CTF rows;
    their order and duplicates are preserved without changing source precision.
    """

    source_path, cache = _exact_ctf_source_cache(experiment_dataset, image_shape)
    original_indices = original_image_indices(
        experiment_dataset,
        np.asarray(image_indices, dtype=np.int64),
    )
    image_h, image_w = (int(size) for size in image_shape)
    if image_h != image_w:
        raise ValueError("exact RELION CTF replay currently requires square images")
    if pixel_indices is not None:
        if not isinstance(pixel_indices, np.ndarray):
            raise TypeError("CTF pixel indices must already be a host NumPy array")
        if pixel_indices.ndim != 1 or pixel_indices.dtype.kind not in "iu":
            raise ValueError("CTF pixel indices must be a one-dimensional integer array")
        if np.any(pixel_indices < 0) or np.any(pixel_indices >= image_h * (image_w // 2 + 1)):
            raise ValueError("CTF pixel indices are outside the full half-spectrum")
    original_indices = np.asarray(original_indices, dtype=np.int64)
    cache_key = _exact_ctf_result_key(source_path, image_shape, original_indices, pixel_indices)
    memoized = _exact_ctf_result_lookup(cache_key, original_indices, pixel_indices)
    if memoized is not None:
        return memoized

    batch_slots = _evaluate_exact_ctf_rows(cache, original_indices, image_h, image_w)
    rows = cache["rows"]
    # Gather rows and the planned columns in one step, so the intermediate is the
    # size of the result rather than of the full half-spectrum.
    gathered = (
        rows[batch_slots]
        if pixel_indices is None
        else rows[np.ix_(batch_slots, np.asarray(pixel_indices, dtype=np.int64))]
    )
    assembled = np.asarray(gathered, dtype=np.float64)
    return _exact_ctf_result_store(cache_key, original_indices, pixel_indices, assembled)


def _relion_exact_ctf_half_from_source_star(
    experiment_dataset,
    image_indices,
    image_shape,
    *,
    pixel_indices=None,
):
    """Return the source-precision CTF rows of these images on the JAX device.

    The binary64 rows are evaluated once per particle on the host
    (:func:`_evaluate_exact_ctf_rows`) and served from a bounded device row
    cache (:func:`_exact_ctf_device_rows`): a request uploads only the rows the
    cache does not hold and is one device gather, optionally of planned pixel
    columns, whose order and duplicates are preserved. The host path copied and
    uploaded every requested row on each call, 72.6 s of main-thread gather in a
    noise1 50k VDAM run (py-spy, job 14523070).
    """

    image_h, image_w = (int(size) for size in image_shape)
    if image_h != image_w:
        raise ValueError("exact RELION CTF replay currently requires square images")
    width = image_h * (image_w // 2 + 1)
    if pixel_indices is not None:
        pixel_indices = np.asarray(pixel_indices)
        if pixel_indices.ndim != 1 or pixel_indices.dtype.kind not in "iu":
            raise ValueError("CTF pixel indices must be a one-dimensional integer array")
        if np.any(pixel_indices < 0) or np.any(pixel_indices >= width):
            raise ValueError("CTF pixel indices are outside the full half-spectrum")
    _, cache = _exact_ctf_source_cache(experiment_dataset, image_shape)
    original_indices = np.asarray(
        original_image_indices(experiment_dataset, np.asarray(image_indices, dtype=np.int64)), dtype=np.int64
    )
    batch_slots = _evaluate_exact_ctf_rows(cache, original_indices, image_h, image_w)
    block, device_rows = _exact_ctf_device_rows(cache, batch_slots, width)
    rows = jnp.asarray(device_rows)
    if pixel_indices is None:
        return jnp.take(block, rows, axis=0)
    return block[rows[:, None], jnp.asarray(pixel_indices, dtype=jnp.int32)[None, :]]


_EXACT_CTF_DEVICE_GB_ENV = "RELAX_RELION_EXACT_CTF_DEVICE_GB"
# The cache takes this share of what the allocator can still hand out (plus what
# it already holds), so the passes that plan from free memory keep the rest.
_EXACT_CTF_DEVICE_FREE_FRACTION = 0.25
# Shrink only below this share of the current capacity, so small swings in free
# memory do not rebuild the cache.
_EXACT_CTF_DEVICE_SHRINK_BELOW = 0.75
# Uploads go up in power-of-two row chunks of at most this many rows: few scatter
# shapes compile and no padded rows travel (a box-800 row is 2.57 MB).
_EXACT_CTF_DEVICE_UPLOAD_ROWS = 256


def _exact_ctf_device_budget_bytes(held_bytes: int) -> int:
    """Device bytes the CTF row cache may hold: ``RELAX_RELION_EXACT_CTF_DEVICE_GB``, else
    ``_EXACT_CTF_DEVICE_FREE_FRACTION`` of the allocator's available bytes now plus the
    ``held_bytes`` the cache already occupies (``device_available_bytes``); the held
    bytes alone when the device reports nothing."""

    token = os.environ.get(_EXACT_CTF_DEVICE_GB_ENV, "").strip()
    if token:
        try:
            budget = float(token)
        except ValueError as exc:
            raise ValueError(f"{_EXACT_CTF_DEVICE_GB_ENV} must be a number of gigabytes, got {token!r}") from exc
        if budget <= 0:
            raise ValueError(f"{_EXACT_CTF_DEVICE_GB_ENV} must be positive, got {token!r}")
        return int(budget * (1024**3))
    from relax.sparse_pass2.sparse_pass2_budget import (
        _device_free_memory_bytes,
        _jax_allocator_free_memory_bytes,
        _jax_allocator_pool_free_bytes,
        device_available_bytes,
    )

    available = device_available_bytes(
        _device_free_memory_bytes(), _jax_allocator_free_memory_bytes(), _jax_allocator_pool_free_bytes()
    )
    if available is None:
        return int(held_bytes)
    return int((float(available) + held_bytes) * _EXACT_CTF_DEVICE_FREE_FRACTION)


def release_exact_ctf_device_cache() -> None:
    """Free every device CTF row cache; the next request rebuilds one at the budget then.

    A pass whose fixed allocations need the memory the cache holds (for example a
    box-800 reconstruction accumulator) calls this before allocating them.
    """

    for cache in _RELION_EXACT_CTF_SOURCE_CACHE.values():
        cache.pop("device_cache", None)


def ensure_device_headroom(n_bytes: int) -> bool:
    """Free the device CTF row caches only if the device cannot hand out ``n_bytes`` now.

    For a pass about to allocate large fixed buffers (a box-800 reconstruction
    accumulator): at 256 px it never fires, so the cache keeps its rows across
    passes. Returns whether it released anything. Call it once before the
    allocation, not per chunk.
    """

    from relax.sparse_pass2.sparse_pass2_budget import (
        _device_free_memory_bytes,
        _jax_allocator_free_memory_bytes,
        _jax_allocator_pool_free_bytes,
        device_available_bytes,
    )

    available = device_available_bytes(
        _device_free_memory_bytes(), _jax_allocator_free_memory_bytes(), _jax_allocator_pool_free_bytes()
    )
    held = any("device_cache" in cache for cache in _RELION_EXACT_CTF_SOURCE_CACHE.values())
    if not held or available is None or available >= n_bytes:
        return False
    release_exact_ctf_device_cache()
    return True


def _exact_ctf_device_capacity(n_particles: int, request_rows: int, row_bytes: int, held_rows: int) -> int:
    """Rows the cache should hold: the budget's rows, at least the request, at most one per particle."""

    budget_rows = _exact_ctf_device_budget_bytes(held_rows * row_bytes) // row_bytes
    return int(min(n_particles, max(request_rows, budget_rows)))


def _exact_ctf_device_rows(cache, batch_slots: np.ndarray, width: int):
    """The device row cache and the cache rows of ``batch_slots``.

    The cache holds as many rows as :func:`_exact_ctf_device_budget_bytes` allows,
    re-read at every request, and never fewer than the request needs nor more than
    one per particle. It is rebuilt empty at the budget when a request does not fit
    or the budget falls below ``_EXACT_CTF_DEVICE_SHRINK_BELOW`` of its capacity.
    Rows a request needs and the cache lacks replace rows in clock order, skipping
    rows the same request reads, and go up in place. It grows only when a request
    does not fit (the first request of a pass, as batch sizes are fixed), never
    because memory freed up, so it does not race a chunk loop's next allocation.
    """

    import jax

    state = cache.get("device_cache")
    n_particles = cache["slots"].size
    row_bytes = width * 8
    requested = np.unique(batch_slots)
    capacity = 0 if state is None else state["block"].shape[0]
    target = _exact_ctf_device_capacity(n_particles, requested.size, row_bytes, capacity)
    if capacity < requested.size or target < _EXACT_CTF_DEVICE_SHRINK_BELOW * capacity:
        cache.pop("device_cache", None)
        state = None  # drop the old block before allocating the new one
        state = {
            "block": jnp.zeros((target, width), dtype=jnp.float64),
            "row_of_slot": np.full(n_particles, -1, dtype=np.int64),
            "slot_of_row": np.full(target, -1, dtype=np.int64),
            "clock": 0,
        }
        cache["device_cache"] = state
    capacity = state["block"].shape[0]
    missing = requested[state["row_of_slot"][requested] < 0]
    if missing.size:
        keep = np.zeros(capacity, dtype=bool)
        keep[state["row_of_slot"][requested[state["row_of_slot"][requested] >= 0]]] = True
        order = (np.arange(capacity) + state["clock"]) % capacity
        free = order[~keep[order]][: missing.size]
        evicted = state["slot_of_row"][free]
        state["row_of_slot"][evicted[evicted >= 0]] = -1
        state["row_of_slot"][missing] = free
        state["slot_of_row"][free] = missing
        state["clock"] = int((free[-1] + 1) % capacity)
        start = 0
        while start < missing.size:
            size = min(_EXACT_CTF_DEVICE_UPLOAD_ROWS, 1 << int(np.log2(missing.size - start)))
            state["block"] = _scatter_exact_ctf_rows(
                state["block"],
                jax.device_put(free[start : start + size].astype(np.int32)),
                jax.device_put(cache["rows"][missing[start : start + size]]),
            )
            start += size
    return state["block"], state["row_of_slot"][batch_slots]


def _scatter_exact_ctf_rows(block, positions, values):
    import jax

    global _scatter_exact_ctf_rows_program
    if _scatter_exact_ctf_rows_program is None:
        _scatter_exact_ctf_rows_program = jax.jit(
            lambda block, positions, values: block.at[positions].set(values), donate_argnums=0
        )
    return _scatter_exact_ctf_rows_program(block, positions, values)


_scatter_exact_ctf_rows_program = None


