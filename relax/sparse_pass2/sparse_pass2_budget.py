"""Device-memory budgets of the sparse bucketed pass 2.

Device memory limits and free-memory probes (JAX allocator, nvidia-smi) and
the per-pass byte budgets derived from them: hypotheses per microbatch,
translation tiles, projection gathers and caches, noise and adjoint blocks.
The positive-int/float environment parsers live here because the budgets
and the policy switches both read them.
"""

from __future__ import annotations

import ctypes
import logging
import os
import subprocess

import jax
import numpy as np

from relax.helpers.env_flags import parse_env_nonnegative_int
from relax.helpers.xla_memory_reserve import _nvidia_smi_visible_device_memory_bytes

logger = logging.getLogger(__name__)


_DEFAULT_MAX_TRANSLATION_TILE_BYTES = 384 * 1024**2


_FUSED_KCLASS_SCORE_GATHER_FRACTION_ENV = "RELAX_SPARSE_PASS2_FUSED_KCLASS_SCORE_GATHER_FRACTION"


_AUTO_TRANSLATION_TILE_DEVICE_FRACTION = 0.020


_AUTO_EXTERNAL_NORMALIZATION_TRANSLATION_TILE_DEVICE_FRACTION = 0.014


_AUTO_FUSED_KCLASS_TRANSLATION_TILE_DEVICE_FRACTION = 0.007


# Fine-projection cache cap as a fraction of device memory.  The K=1 sparse
# pass-2 recomputes every image chunk's fine projections when the cache is
# skipped; at HEALPix order 3 (294912 fine rotations, current_size 92, 256^2)
# the score+recon+abs2 cache estimate is 18.4 GiB, which the former 10% cap
# (7.96 GiB on an 80 GB device) rejected in every hp3 iteration of the 10k
# EMPIAR-10097 convergence run while per-chunk recompute cost 1200-3900 s per
# iteration.  25% admitted that cache on 80 GB devices and still rejected it on
# 40 GB devices.  Measured evidence: docs handoff em_soft_posterior_block_bpref
# prototype 2026-09-17, jobs 14045912 / 14046044.  The stable Fourier windows
# pad the window to a physical class, so the same cache now takes 20.00 GiB
# (71.1 KiB per rotation) against the 17.81 GiB 25% share, and the 10097 10k
# hp3 iterations streamed their projections again: 21.0 s and 11.7 s of
# projection per half (job 14514823).  30% admits it on 80 GB devices; the
# chunk plan is read from free memory after the cache exists.
_AUTO_PROJECTION_CACHE_DEVICE_FRACTION = 0.300


_AUTO_PROJECTED_ROTATIONS_DEVICE_FRACTION = 0.040


_AUTO_ADJOINT_BLOCK_DEVICE_FRACTION = 0.006


# The expected-accuracy projector slab (float64 real and imaginary planes) stays on
# the device whole up to this fraction of device memory; above it, each projection
# streams it in z-plane chunks of at most ACCURACY_SLAB_STREAM_CHUNK_BYTES. At
# EMPIAR-10202's full box the slab is 30.7 GiB, which the pool of a 26-iteration run
# could not place (gpuport 15003773, final all-data pass), and chunks of a tenth of
# the device, uploaded and freed at every projection, fragmented the pool until a
# 7.65 GiB reconstruction buffer failed at iteration 23 (gpuport 15073680; the same
# run without streaming passed, 15085606/15085607). Small chunks leave no large hole.
# A full-size box-256 slab (1.1 GB at padding 2) stays whole on a 16 GB device.
_AUTO_ACCURACY_SLAB_DEVICE_FRACTION = 0.100
ACCURACY_SLAB_STREAM_CHUNK_BYTES = 512 * 1024**2


_DEFAULT_ADJOINT_BLOCK_MAX_BYTES = 512 * 1024**2


_MAX_TRANSLATION_TILE_BYTES_ENV = "RELAX_SPARSE_PASS2_MAX_TRANSLATION_TILE_BYTES"


_MAX_ADJOINT_BLOCK_BYTES_ENV = "RELAX_SPARSE_PASS2_MAX_ADJOINT_BLOCK_BYTES"


_MAX_PROJECTED_ROTATIONS_ENV = "RELAX_SPARSE_PASS2_MAX_PROJECTED_ROTATIONS"


_PROJECTION_CACHE_MAX_BYTES_ENV = "RELAX_SPARSE_PASS2_PROJECTION_CACHE_MAX_BYTES"


_DEFAULT_PROJECTION_CACHE_MAX_BYTES = 3 * 1024**3


def _optional_positive_int_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}")
    return value


_CONCURRENT_DEVICE_SHARES = 1


def _share(total_bytes: int | None) -> int | None:
    """This worker's slice of a device total, given the declared share count."""

    if total_bytes is None:
        return None
    shares = _CONCURRENT_DEVICE_SHARES
    if shares <= 1:
        return int(total_bytes)
    return max(1, int(total_bytes) // shares)


def set_concurrent_device_shares(shares: int) -> int:
    """Declare how many workers are sharing this device, and return the previous count.

    Every budget in this module is a fraction of the device's memory and is
    written for one worker at a time. When two half-sets run concurrently they
    size their caches against the same device, so each must be told it owns
    only its share; otherwise both admit a plan that fits alone and neither
    fits together. That is not hypothetical: overlapping the two halves at
    HEALPix order 3 failed with RESOURCE_EXHAUSTED building the second half's
    projection cache, because each half had budgeted the whole 80 GiB device.

    **This declaration covers device memory only.** Host-side caches are not
    fractions of a device and do not pass through this module, so they neither
    shrink nor are checked when the share count rises, while a second
    concurrent worker doubles them just the same. Two at the time of writing:
    the exact-CTF operand memo in ``recovar/em/relion/relion_ctf.py``, about
    2.4-2.6 GB of host RAM for two half operands when enabled, and the image
    loader's prefetch slots at roughly 65 MB each. Budget those on the host
    side; nothing here will.
    """

    global _CONCURRENT_DEVICE_SHARES
    if shares < 1:
        raise ValueError(f"concurrent device shares must be at least 1, got {shares}")
    previous, _CONCURRENT_DEVICE_SHARES = _CONCURRENT_DEVICE_SHARES, shares
    return previous


def concurrent_device_shares() -> int:
    """How many workers are currently declared to share this device."""

    return _CONCURRENT_DEVICE_SHARES


def _jax_allocator_limit_bytes() -> int | None:
    """The JAX GPU allocator's limit (``bytes_limit``), if reported.

    ``XLA_PYTHON_CLIENT_MEM_FRACTION`` sets it; it is the most this process can
    allocate, whatever the device's total.
    """

    try:
        devices = [device for device in jax.devices() if getattr(device, "platform", "") in {"gpu", "cuda"}]
        if not devices:
            return None
        stats = devices[0].memory_stats()
    except Exception:
        return None
    if not stats:
        return None
    for key in ("bytes_limit", "bytesLimit", "memory_limit"):
        value = stats.get(key)
        if value is not None and int(value) > 0:
            return int(value)
    return None


def _device_memory_limit_bytes() -> int | None:
    """Return this worker's share of the memory this process can allocate on its accelerator.

    The device's total from nvidia-smi, capped by the JAX allocator's limit:
    under ``XLA_PYTHON_CLIENT_MEM_FRACTION=.50`` an 80 GB H100 gives the process
    about 40 GB, and every budget taken as a fraction of the device total then
    overshot (K=1 10097 10k, bench 14445196, ran out of memory in the whole-grid
    projection cache at 42 GB). The share is the whole of that unless several
    workers have been declared through :func:`set_concurrent_device_shares`, in
    which case every fraction computed downstream is a fraction of one worker's
    share.
    """

    # ``RELAX_SPARSE_PASS2_DEVICE_MEMORY_GB`` overrides the nvidia-smi probe.
    # Keep this as a manual escape hatch for reserving headroom on shared GPUs
    # or working around inaccurate allocator/device probes.
    _override = os.environ.get("RELAX_SPARSE_PASS2_DEVICE_MEMORY_GB")
    if _override is not None:
        try:
            override_gb = float(_override.strip())
            if override_gb > 0:
                return _share(int(override_gb * (1024 ** 3)))
        except ValueError:
            pass

    # NVML answers in microseconds; the nvidia-smi subprocess (the fallback below) took about 40 ms
    # per pass, 1.25 s of a 60 s window of late 10k VDAM iterations (py-spy). nvidia-smi prints the
    # total in whole MiB, and the budgets were sized from that figure: keep it.
    mebibyte = 1024 * 1024
    nvml_total = _nvml_memory_bytes(os.environ.get("CUDA_VISIBLE_DEVICES"), "total")
    if nvml_total is not None:
        memory_bytes = (nvml_total // mebibyte) * mebibyte
        allocator_limit = _jax_allocator_limit_bytes()
        if allocator_limit is not None:
            memory_bytes = min(int(memory_bytes), int(allocator_limit))
        return _share(memory_bytes)
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if query.returncode == 0:
            memory_bytes = _nvidia_smi_visible_device_memory_bytes(
                query.stdout,
                os.environ.get("CUDA_VISIBLE_DEVICES"),
            )
            if memory_bytes is not None:
                allocator_limit = _jax_allocator_limit_bytes()
                if allocator_limit is not None:
                    memory_bytes = min(int(memory_bytes), int(allocator_limit))
                return _share(memory_bytes)
    except Exception:
        pass
    try:
        devices = [device for device in jax.devices() if getattr(device, "platform", "") in {"gpu", "cuda"}]
        if not devices:
            return None
        stats = devices[0].memory_stats()
    except Exception:
        return None
    if not stats:
        return None
    for key in ("bytes_limit", "bytesLimit", "memory_limit", "total_memory"):
        value = stats.get(key)
        if value is not None and int(value) > 0:
            return _share(int(value))
    return None


class _NvmlMemory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


# (NVML library, {nvidia-smi index or uuid: device handle}) once loaded, False when
# NVML is unavailable here.
_NVML_DEVICES = None


def _nvml_devices():
    """NVML's handle for every GPU, keyed as nvidia-smi's rows are (index and uuid)."""

    global _NVML_DEVICES
    if _NVML_DEVICES is None:
        try:
            lib = ctypes.CDLL("libnvidia-ml.so.1")
            if lib.nvmlInit_v2() != 0:
                raise OSError("nvmlInit_v2 failed")
            count = ctypes.c_uint()
            if lib.nvmlDeviceGetCount_v2(ctypes.byref(count)) != 0:
                raise OSError("nvmlDeviceGetCount_v2 failed")
            handles = {}
            for index in range(count.value):
                handle = ctypes.c_void_p()
                if lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle)) != 0:
                    raise OSError("nvmlDeviceGetHandleByIndex_v2 failed")
                uuid = ctypes.create_string_buffer(96)
                if lib.nvmlDeviceGetUUID(handle, uuid, 96) != 0:
                    raise OSError("nvmlDeviceGetUUID failed")
                handles[str(index)] = handle
                handles[uuid.value.decode()] = handle
            _NVML_DEVICES = (lib, handles)
        except (OSError, AttributeError):
            _NVML_DEVICES = False
    return _NVML_DEVICES


def _nvml_memory_bytes(visible_devices: str | None, field: str) -> int | None:
    """``field`` (``"free"`` or ``"total"``) of the first visible GPU's memory from NVML, as nvidia-smi reads it.

    Selects the device as :func:`_nvidia_smi_visible_device_memory_bytes` does;
    ``None`` when NVML is unavailable or the device is not found.
    """

    devices = _nvml_devices()
    if not devices:
        return None
    lib, handles = devices
    if visible_devices is None:
        handle = handles.get("0")
    else:
        tokens = [
            part.strip()
            for part in visible_devices.split(",")
            if part.strip() and part.strip() not in {"-1", "none", "NoDevFiles"}
        ]
        handle = next((handles[token] for token in tokens if token in handles), None)
    if handle is None:
        return None
    memory = _NvmlMemory()
    if lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)) != 0:
        return None
    return int(getattr(memory, field))


def _nvml_free_memory_bytes(visible_devices: str | None) -> int | None:
    """Free memory of the first visible GPU from NVML, the reading nvidia-smi prints."""

    return _nvml_memory_bytes(visible_devices, "free")


def _device_free_memory_bytes() -> int | None:
    """Return current free memory for the selected physical GPU, if known.

    NVML answers in microseconds; the ``nvidia-smi`` subprocess it replaces took
    15-20 ms and was read per chunk by the exact-CTF device cache's budget, 33 s
    of two late Class3D K4 100k iterations (py-spy, job 14594324). The subprocess
    stays the fallback where NVML cannot be loaded.
    """

    free = _nvml_free_memory_bytes(os.environ.get("CUDA_VISIBLE_DEVICES"))
    if free is not None:
        return free
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if query.returncode == 0:
            return _nvidia_smi_visible_device_memory_bytes(
                query.stdout,
                os.environ.get("CUDA_VISIBLE_DEVICES"),
            )
    except Exception:
        pass
    return None


def _jax_allocator_free_memory_bytes() -> int | None:
    """Return unused bytes in the active JAX GPU allocator, if reported."""

    try:
        devices = [device for device in jax.devices() if getattr(device, "platform", "") in {"gpu", "cuda"}]
        if not devices:
            return None
        stats = devices[0].memory_stats()
    except Exception:
        return None
    if not stats:
        return None

    limit = next(
        (
            int(stats[key])
            for key in ("bytes_limit", "bytesLimit", "memory_limit", "total_memory")
            if stats.get(key) is not None and int(stats[key]) > 0
        ),
        None,
    )
    bytes_in_use = next(
        (
            int(stats[key])
            for key in ("bytes_in_use", "bytesInUse", "memory_in_use")
            if stats.get(key) is not None and int(stats[key]) >= 0
        ),
        None,
    )
    if limit is None or bytes_in_use is None:
        return None
    return max(0, limit - bytes_in_use)


def _jax_allocator_pool_free_bytes() -> int | None:
    """Return bytes the JAX GPU allocator's pool holds but does not use, if reported.

    ``nvidia-smi`` counts that memory as used, so a free-memory reading taken
    after the pool has grown understates what the allocator can still hand out
    by exactly this amount.
    """

    try:
        devices = [device for device in jax.devices() if getattr(device, "platform", "") in {"gpu", "cuda"}]
        if not devices:
            return None
        stats = devices[0].memory_stats()
    except Exception:
        return None
    if not stats:
        return None
    pool = stats.get("pool_bytes")
    bytes_in_use = stats.get("bytes_in_use")
    if pool is None or bytes_in_use is None:
        return None
    return max(0, int(pool) - int(bytes_in_use))


def _dtype_itemsize(dtype) -> int:
    return int(np.dtype(dtype).itemsize)


def _complex_counterpart_real_dtype(complex_dtype):
    complex_dtype = np.dtype(complex_dtype)
    if complex_dtype.itemsize <= np.dtype(np.complex64).itemsize:
        return np.float32
    return np.float64


def _max_translation_tile_bytes_for_pass(
    device_memory_bytes: int | None = None,
    *,
    has_external_normalization: bool = False,
    fused_k_class: bool = False,
) -> int:
    override = _optional_positive_int_env(_MAX_TRANSLATION_TILE_BYTES_ENV)
    if override is not None:
        return override
    if device_memory_bytes is None:
        return _DEFAULT_MAX_TRANSLATION_TILE_BYTES
    if fused_k_class:
        fraction = _AUTO_FUSED_KCLASS_TRANSLATION_TILE_DEVICE_FRACTION
    elif has_external_normalization:
        fraction = _AUTO_EXTERNAL_NORMALIZATION_TRANSLATION_TILE_DEVICE_FRACTION
    else:
        fraction = _AUTO_TRANSLATION_TILE_DEVICE_FRACTION
    return max(1, int(float(device_memory_bytes) * fraction))


def _max_adjoint_block_bytes_for_pass(device_memory_bytes: int | None = None) -> int:
    override = _optional_positive_int_env(_MAX_ADJOINT_BLOCK_BYTES_ENV)
    if override is not None:
        return int(override)
    if device_memory_bytes is None:
        return _DEFAULT_ADJOINT_BLOCK_MAX_BYTES
    return max(1, int(float(device_memory_bytes) * _AUTO_ADJOINT_BLOCK_DEVICE_FRACTION))


def accuracy_slab_resident_bytes(device_memory_bytes: int | None = None) -> int | None:
    """The largest expected-accuracy slab kept whole on the device, or None for no bound (no accelerator)."""

    if device_memory_bytes is None:
        device_memory_bytes = _device_memory_limit_bytes()
    if device_memory_bytes is None:
        return None
    return max(1, int(float(device_memory_bytes) * _AUTO_ACCURACY_SLAB_DEVICE_FRACTION))


def _projection_cache_max_bytes_for_pass(device_memory_bytes: int | None = None) -> int:
    override = parse_env_nonnegative_int(_PROJECTION_CACHE_MAX_BYTES_ENV)
    if override is not None:
        return override
    if device_memory_bytes is None:
        return _DEFAULT_PROJECTION_CACHE_MAX_BYTES
    return max(1, int(float(device_memory_bytes) * _AUTO_PROJECTION_CACHE_DEVICE_FRACTION))


def _projection_call_max_bytes_for_pass(device_memory_bytes: int | None = None) -> int:
    override = parse_env_nonnegative_int(_PROJECTION_CACHE_MAX_BYTES_ENV)
    if override is not None:
        return override
    if device_memory_bytes is None:
        return _DEFAULT_PROJECTION_CACHE_MAX_BYTES
    return max(1, int(float(device_memory_bytes) * _AUTO_PROJECTED_ROTATIONS_DEVICE_FRACTION))


def _projection_cache_transient_bytes(
    n_rotations: int,
    n_half_pixels: int,
    *,
    projection_complex_dtype=np.complex64,
    include_abs2: bool,
) -> int:
    complex_bytes = _dtype_itemsize(projection_complex_dtype)
    total = int(n_rotations) * int(n_half_pixels) * complex_bytes
    if include_abs2:
        real_dtype = _complex_counterpart_real_dtype(projection_complex_dtype)
        total += int(n_rotations) * int(n_half_pixels) * _dtype_itemsize(real_dtype)
    return int(total)


def _projection_cache_budget_complex_dtype(
    projection_source_dtype,
    score_complex_dtype,
    *,
    use_relion_projector: bool = False,
):
    dtype = np.promote_types(np.dtype(projection_source_dtype), np.dtype(score_complex_dtype))
    if use_relion_projector:
        # RELION Projector parity uses float64 interpolation weights, which
        # promotes complex64 projector data to complex128 before the caller's
        # output cast. Budget the transient allocation, not the retained cache.
        dtype = np.promote_types(dtype, np.dtype(np.complex128))
    return dtype


def _projection_cache_fits_budget(transient_bytes: int, max_bytes: int, *, n_classes: int = 1) -> bool:
    return int(transient_bytes) * max(1, int(n_classes)) <= int(max_bytes)


def _max_projected_rotations_per_call_for_pass(
    *,
    device_memory_bytes: int | None,
    n_projection_pixels: int,
    projection_complex_dtype,
    include_abs2: bool,
) -> int | None:
    override = _optional_positive_int_env(_MAX_PROJECTED_ROTATIONS_ENV)
    if override is not None:
        return int(override)
    if device_memory_bytes is None or int(n_projection_pixels) <= 0:
        return None
    max_bytes = _projection_call_max_bytes_for_pass(device_memory_bytes)
    bytes_per_rotation = _projection_cache_transient_bytes(
        1,
        int(n_projection_pixels),
        projection_complex_dtype=projection_complex_dtype,
        include_abs2=bool(include_abs2),
    )
    if max_bytes <= 0 or bytes_per_rotation <= 0:
        return None
    return max(1, int(max_bytes) // int(bytes_per_rotation))


_PROJECTION_CACHE_BUILD_ROTATION_MULTIPLIER = 4


def _projection_cache_build_max_rotations_per_call(
    per_call_max_rotations: int | None,
    n_fine_rotations: int,
) -> int | None:
    """Rotations per projection call while building the fine-projection cache.

    The per-chunk budget ``per_call_max_rotations`` is sized for the scoring
    loop, where per-image translation tiles, score blocks and Wavg operands are
    live next to the projection intermediates.  The cache build runs before
    any of those exist, so it may project several times more rotations per
    call: at HEALPix order 3 (294912 fine rotations, current_size 92) the
    build took 19.9 s per half at 809 rotations per call and 3.6 s at 4096
    (jobs 14051847 / 14052595), while the scoring-loop budget kept the peak
    memory profile unchanged.  An explicit
    ``RELAX_SPARSE_PASS2_MAX_PROJECTED_ROTATIONS`` still applies verbatim.
    """

    if per_call_max_rotations is None:
        return None
    override = _optional_positive_int_env(_MAX_PROJECTED_ROTATIONS_ENV)
    if override is not None:
        return int(override)
    scaled = int(per_call_max_rotations) * _PROJECTION_CACHE_BUILD_ROTATION_MULTIPLIER
    return max(1, min(int(n_fine_rotations), scaled))


def _projection_budget_pixels_for_pass(
    n_half_pixels: int,
    *,
    use_window: bool,
    use_relion_projector: bool,
) -> int:
    """Effective projection pixels for the sparse pass-2 projection cap.

    Windowed sparse pass-2 only keeps score/reconstruction rows after
    projection, but RELION's centered Projector handoff currently materializes
    full-half intermediates before gathering the requested windows. Budget that
    path with extra headroom for the centered-row scatter, dense scaling, and
    other live pass-2 buffers so huge one-image compact-pair buckets still split
    before the projection helper allocates.
    """

    pixels = int(n_half_pixels)
    if bool(use_window) and bool(use_relion_projector):
        return max(1, 8 * pixels)
    return max(1, pixels)


def device_available_bytes(physical_free_bytes, allocator_free_bytes, pool_free_bytes=None) -> float | None:
    """Bytes the JAX allocator can still hand out on this device, from the three readings.

    The allocator's own headroom (limit minus in use), bounded by the device:
    the physically free memory plus what the allocator's pool already holds
    unused, which ``nvidia-smi`` counts as used. The physical reading alone is
    not a bound once the pool has grown (10097 it13 in 14400302: 13.37 GiB
    physically free against 70.04 GiB of headroom). Without a pool reading the
    physical reading bounds only when the allocator reports nothing; ``None``
    when nothing is known.
    """

    available = None if allocator_free_bytes is None else float(allocator_free_bytes)
    if physical_free_bytes is not None:
        if pool_free_bytes is not None:
            device_bound = float(physical_free_bytes) + float(pool_free_bytes)
            available = device_bound if available is None else min(available, device_bound)
        elif available is None:
            available = float(physical_free_bytes)
    return available
