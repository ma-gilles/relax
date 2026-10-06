"""Size the XLA memory pool so RELION's projector texture fits beside it.

recovar sets ``XLA_PYTHON_CLIENT_MEM_FRACTION`` to 0.90, and the BFC pool grows to
that limit during a refinement. The RELION projector texture is a pair of float32
CUDA arrays allocated outside the pool (``relion_scoring.cuh``,
``RelionPersistentHalfTextureF32``; the resident local pass's
``RelionCapacityHalfTextureF32``, ``relion_capacity_texture.cuh``), so it has to fit in the rest of the card
next to the CUDA context and the CUB scratch. At EMPIAR-10202 current size 626
that is 7.9 GB of a ~9 GB remainder, and the allocation failed with CUDA out of
memory (bigbox 14480549, 14507538). A fixed fraction only moves that cliff to
another box, so the reserve is sized from the run's own model box before the
JAX backend initialises: the texture of the largest local-search slab
(``r_max = box // 2``) plus a fixed allowance for the context and scratch.

``import recovar`` initialises the backend (``recovar.jax_config`` queries the
devices), and ``import relax`` imports recovar, so the reserve has to run before
either. ``relax/__init__.py`` (``_reserve_refinement_projector_memory``) calls
:func:`reserve_for_refinement` for ``relax refine`` and ``relax class3d``
before it imports recovar; any other entry point that runs a refinement in its
own process (a benchmark runner, a notebook) calls :func:`reserve_for_refinement`
with the refinement's arguments before its first ``jax``, ``recovar`` or ``relax``
import. This module therefore imports only the standard library (and ``mrcfile``
or ``jax`` inside a function), so it can run while the package is still
initialising, or be loaded by file path. Called after the backend has started,
the reserve does nothing and says so in its record.

The pool cannot be shrunk once the backend runs, so the refinement driver calls
:func:`require_projector_texture_reserve` at start-up: an entry point that
skipped the reserve is refused in the first seconds instead of failing when the
final pass opens its full-box texture (EMPIAR-10202, box 800, after 6 h 17 min
in job 14963923, relax#17).
"""

from __future__ import annotations

import argparse
import functools
import os
import re
import subprocess

MEM_FRACTION_ENV = "XLA_PYTHON_CLIENT_MEM_FRACTION"
# CUDA context, cuFFT/CUB workspaces and the coarse-kernel texture cache.
_CONTEXT_AND_SCRATCH_BYTES = 3 * 1024**3
# Never hand XLA less than this share of the card.
_MIN_FRACTION = 0.5

def relion_projector_texture_bytes(model_box: int, padding_factor: int) -> int:
    """Bytes of the float32 texture pair of RELION's largest projector slab for ``model_box``.

    The slab is ``(2 r pf + 3) x (2 r pf + 3) x (r pf + 2)`` complex values with
    ``r = model_box // 2`` (``Projector::data``, the half-storage layout the
    texture projector uploads), stored as two float32 arrays: 8 bytes a value.
    """

    r = int(model_box) // 2
    pf = int(padding_factor)
    n = 2 * r * pf + 3
    return n * n * (r * pf + 2) * 8


def xla_memory_fraction(
    model_box: int,
    padding_factor: int,
    device_total_bytes: int,
    *,
    current_fraction: float = 0.90,
    context_bytes: int = _CONTEXT_AND_SCRATCH_BYTES,
) -> float:
    """The XLA pool fraction that leaves room for the projector texture, never above ``current_fraction``."""

    reserve = relion_projector_texture_bytes(model_box, padding_factor) + int(context_bytes)
    fitted = (float(device_total_bytes) - float(reserve)) / float(device_total_bytes)
    return float(min(float(current_fraction), max(_MIN_FRACTION, fitted)))


def model_box_from_map_headers(paths) -> int | None:
    """The box of the first existing reference map in ``paths``, read from its MRC header only."""

    import mrcfile

    for path in paths:
        if path is None or not os.path.isfile(path):
            continue
        with mrcfile.open(str(path), header_only=True, permissive=True) as handle:
            return int(handle.header.nx)
    return None


def _parse_nvidia_smi_memory_rows(output: str) -> dict[str, int]:
    rows: dict[str, int] = {}
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        index, uuid, memory_mib = parts[:3]
        try:
            memory_bytes = int(memory_mib.split()[0]) * 1024**2
        except (ValueError, IndexError):
            continue
        if memory_bytes <= 0:
            continue
        rows[index] = memory_bytes
        rows[uuid] = memory_bytes
        if uuid.startswith("GPU-"):
            rows[uuid[4:]] = memory_bytes
    return rows


def _nvidia_smi_visible_device_memory_bytes(output: str, visible_devices: str | None) -> int | None:
    rows = _parse_nvidia_smi_memory_rows(output)
    if not rows:
        return None
    # An unset variable exposes every device; a set but empty one exposes none.
    if visible_devices is not None:
        tokens = [
            part.strip()
            for part in visible_devices.split(",")
            if part.strip() and part.strip() not in {"-1", "none", "NoDevFiles"}
        ]
        if not tokens:
            return None
        for token in tokens:
            if token in rows:
                return rows[token]
        return None
    return next(iter(rows.values()))


def _visible_device_total_bytes() -> int | None:
    """Total memory of the first visible GPU from nvidia-smi, without initialising JAX."""

    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.total", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except Exception:
        return None
    if query.returncode != 0:
        return None
    return _nvidia_smi_visible_device_memory_bytes(query.stdout, os.environ.get("CUDA_VISIBLE_DEVICES"))


def _jax_backend_initialized() -> bool:
    try:
        from jax._src import xla_bridge
    except Exception:
        return False
    return bool(getattr(xla_bridge, "_backends", None))


def reserve_projector_texture_memory(model_box: int | None, padding_factor: int) -> dict | None:
    """Lower ``XLA_PYTHON_CLIENT_MEM_FRACTION`` for this run's projector texture; return what was done.

    Must run before the JAX backend starts; afterwards the pool limit is fixed and
    this does nothing. ``None`` when the box or the device total is unknown.
    """

    if model_box is None:
        return None
    if _jax_backend_initialized():
        return {
            "model_box": int(model_box),
            "skipped": "the JAX backend was already initialised with "
            f"{MEM_FRACTION_ENV}={os.environ.get(MEM_FRACTION_ENV, 'unset')}",
        }
    total = _visible_device_total_bytes()
    if total is None:
        return None
    current = float(os.environ.get(MEM_FRACTION_ENV, "0.90"))
    fraction = xla_memory_fraction(model_box, padding_factor, total, current_fraction=current)
    if fraction < current:
        os.environ[MEM_FRACTION_ENV] = f"{fraction:.4f}"
    return {
        "model_box": int(model_box),
        "padding_factor": int(padding_factor),
        "device_total_bytes": int(total),
        "texture_bytes": relion_projector_texture_bytes(model_box, padding_factor),
        "context_bytes": _CONTEXT_AND_SCRATCH_BYTES,
        "previous_fraction": current,
        "fraction": fraction,
    }


def reference_maps_from_argv(argv) -> list[str | None]:
    """The reference maps a refinement command line names, in the order the driver reads them."""

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--init_volume", default=None)
    parser.add_argument("--init_class_volumes", default=None)
    known, _ = parser.parse_known_args(list(argv))
    if known.data_dir is None and known.init_volume is None and known.init_class_volumes is None:
        return []
    return [
        known.init_volume,
        known.init_class_volumes.split(",")[0].strip() if known.init_class_volumes else None,
        os.path.join(known.data_dir, "reference_init_relion.mrc") if known.data_dir else None,
        os.path.join(known.data_dir, "reference_init_class001_relion.mrc") if known.data_dir else None,
    ]


def reserve_for_reference_maps(paths, padding_factor: int) -> dict | None:
    """:func:`reserve_projector_texture_memory` for the box of the first existing map in ``paths``."""

    return reserve_projector_texture_memory(model_box_from_map_headers(paths), padding_factor)


def reserve_for_refinement(argv, padding_factor: int) -> dict | None:
    """The reserve for a refinement's command-line arguments (without the program name).

    The one call an entry point makes before the JAX backend starts; see the module docstring.
    """

    return reserve_for_reference_maps(reference_maps_from_argv(argv), padding_factor)


def _backend_pool_limit_bytes() -> int | None:
    """The running GPU backend's XLA pool limit; ``None`` on CPU or when the backend does not report one."""

    import jax

    if jax.default_backend() != "gpu":
        return None
    limit = (jax.devices()[0].memory_stats() or {}).get("bytes_limit")
    return None if not limit else int(limit)


def require_projector_texture_reserve(
    model_box: int | None,
    padding_factor: int,
    *,
    pool_limit_bytes: int | None = None,
    device_total_bytes: int | None = None,
) -> None:
    """Refuse a refinement whose running XLA pool leaves no room for its projector texture.

    The pool limit must be no larger than the one :func:`xla_memory_fraction` gives for ``model_box``, which
    is what :func:`reserve_for_refinement` sets. A run that reserved, or whose texture fits beside
    recovar's default pool, passes unchanged. ``pool_limit_bytes`` and ``device_total_bytes`` default to
    the running backend's and the visible device's.
    """

    if model_box is None:
        return
    pool_limit_bytes = _backend_pool_limit_bytes() if pool_limit_bytes is None else int(pool_limit_bytes)
    device_total_bytes = _visible_device_total_bytes() if device_total_bytes is None else int(device_total_bytes)
    if pool_limit_bytes is None or device_total_bytes is None:
        return
    fraction = xla_memory_fraction(model_box, padding_factor, device_total_bytes)
    # The reserve writes the fraction with four decimals, and XLA rounds the limit to its own page size.
    if pool_limit_bytes <= (fraction + 1e-3) * device_total_bytes:
        return
    texture = relion_projector_texture_bytes(model_box, padding_factor)
    raise RuntimeError(
        f"the XLA memory pool may grow to {pool_limit_bytes / 1024**3:.2f} GiB of this "
        f"{device_total_bytes / 1024**3:.2f} GiB device, which leaves no room for the "
        f"{texture / 1024**3:.2f} GiB RELION projector texture of a {int(model_box)}-pixel model at padding "
        f"{int(padding_factor)} plus {_CONTEXT_AND_SCRATCH_BYTES / 1024**3:.1f} GiB context and scratch: the texture "
        "is allocated outside the pool, and the final pass would fail after the pool has grown. The pool limit is "
        "fixed when the JAX backend starts. Start the run with `relax refine` or `relax class3d`, or call "
        "relax.helpers.xla_memory_reserve.reserve_for_refinement(arguments, padding_factor) before the first "
        f"import of jax, recovar or relax (it sets {MEM_FRACTION_ENV}={fraction:.4f} for this run)."
    )


PREALLOCATE_ENV = "XLA_PYTHON_CLIENT_PREALLOCATE"


_GIB = float(1 << 30)
_UNIT_BYTES = {"B": 1, "KiB": 1 << 10, "MiB": 1 << 20, "GiB": 1 << 30, "TiB": 1 << 40}


def _requested_bytes(message: str) -> int | None:
    """The size XLA's out-of-memory message says it tried to allocate (``... allocate 7.65GiB``)."""

    match = re.search(r"allocate ([0-9.]+)\s*(B|KiB|MiB|GiB|TiB)", message)
    return None if match is None else int(float(match.group(1)) * _UNIT_BYTES[match.group(2)])


def _pool_statistics() -> dict | None:
    """The GPU pool's ``bytes_limit`` and ``peak_bytes_in_use``; ``None`` on CPU or when not reported."""

    try:
        import jax

        if jax.default_backend() != "gpu":
            return None
        stats = jax.devices()[0].memory_stats() or {}
    except Exception:
        return None
    if not stats.get("bytes_limit"):
        return None
    return {"limit": int(stats["bytes_limit"]), "peak": int(stats.get("peak_bytes_in_use", 0))}


def _pool_failure_note(message: str, preallocate_off: bool, stats: dict | None) -> str:
    """Why an allocation failed: the pool's limit, its high-water mark and the request, then the likely cause."""

    request = _requested_bytes(message)
    if stats is None or request is None:
        state = "the pool's state was not reported"
        fragmented = False
    else:
        state = (
            f"request {request / _GIB:.2f} GiB, pool limit {stats['limit'] / _GIB:.2f} GiB, "
            f"run peak in use {stats['peak'] / _GIB:.2f} GiB"
        )
        # In use at the failure is at most the run's peak: below the limit by the request, the bytes were free.
        fragmented = stats["peak"] + request <= stats["limit"]
    if preallocate_off:
        cause = (
            f"{PREALLOCATE_ENV}=false is set: XLA's memory pool grew in separate regions and an array "
            "must fit inside one, so this allocation can fail while the pool has the bytes free. "
            f"Unset {PREALLOCATE_ENV} (JAX then preallocates the pool as one region) and rerun; "
            "turn preallocation off only to share a GPU, and not for long large-box refinements."
        )
    elif fragmented:
        cause = (
            "The pool had the bytes free but no contiguous block that large: it is fragmented (relax#20). "
            "Report the run; a smaller GPU memory plan does not help."
        )
    else:
        cause = "The pool may be full: the request does not fit beside the run's peak in use."
    return f"GPU pool at the failure: {state}. {cause}"


def explains_pool_region_failure(run):
    """Decorate a refinement entry: an allocator failure says what the pool held and the likely cause.

    Without preallocation XLA's pool grows in separate regions (1, 2, 4, 8, 16 GiB ...) and an array
    must fit inside one of them, so after many iterations of small requests a large buffer can fail
    while its bytes are free (EMPIAR-10202, box 800: 18.54 GiB at iteration 13 in a 60.9 GiB pool whose
    largest region was 16 GiB, relax#20). With JAX's default, one preallocated region, a long run can
    still fragment that region (EMPIAR-10202, 7.65 GiB at iteration 23, gpuport 15073680). The note
    gives the pool's limit, its peak in use and the request; the error itself is unchanged.
    """

    @functools.wraps(run)
    def explained(*args, **kwargs):
        try:
            return run(*args, **kwargs)
        except Exception as error:
            if "RESOURCE_EXHAUSTED" in str(error):
                preallocate_off = os.environ.get(PREALLOCATE_ENV, "").lower() in ("false", "0")
                error.add_note(_pool_failure_note(str(error), preallocate_off, _pool_statistics()))
            raise

    return explained


def format_reserve_record(record: dict) -> str:
    """One log line for a :func:`reserve_projector_texture_memory` record."""

    if "skipped" in record:
        return f"XLA pool reserve for the RELION projector texture skipped: {record['skipped']}"
    return (
        f"XLA pool fraction {record['fraction']:.4f} (configured {record['previous_fraction']:.4f}): "
        f"{record['texture_bytes'] / 1024**3:.2f} GiB RELION projector texture of a {record['model_box']}-pixel "
        f"model at padding {record['padding_factor']} plus {record['context_bytes'] / 1024**3:.1f} GiB context "
        f"and scratch stay outside the pool, on a {record['device_total_bytes'] / 1024**3:.2f} GiB device"
    )
