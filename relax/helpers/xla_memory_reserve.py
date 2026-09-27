"""Size the XLA memory pool so RELION's projector texture fits beside it.

recovar sets ``XLA_PYTHON_CLIENT_MEM_FRACTION`` to 0.90, and the BFC pool grows to
that limit during a refinement. The RELION projector texture is a pair of float32
CUDA arrays allocated outside the pool (``relion_scoring.cuh``,
``RelionPersistentHalfTextureF32``), so it has to fit in the rest of the card
next to the CUDA context and the CUB scratch. At EMPIAR-10202 current size 626
that is 7.9 GB of a ~9 GB remainder, and the allocation failed with CUDA out of
memory (bigbox 14480549, 14507538). A fixed fraction only moves that cliff to
another box, so the reserve is sized from the run's own model box before the
JAX backend initialises: the texture of the largest local-search slab
(``r_max = box // 2``) plus a fixed allowance for the context and scratch.

``import recovar`` initialises the backend (``recovar.jax_config`` queries the
devices), so the reserve runs as a launch hook of the ``relax`` package, before
it imports recovar, from the reference-map flags on the command line. This
module therefore imports nothing from relax or recovar.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

MEM_FRACTION_ENV = "XLA_PYTHON_CLIENT_MEM_FRACTION"
# CUDA context, cuFFT/CUB workspaces and the coarse-kernel texture cache.
_CONTEXT_AND_SCRATCH_BYTES = 3 * 1024**3
# Never hand XLA less than this share of the card.
_MIN_FRACTION = 0.5
# Padding of relax's projectors (run_full_refinement's projection_padding_factor, RELION's --pad 2).
PROJECTION_PADDING_FACTOR = 2
# What the launch hook did, for the driver to log once logging is configured.
LAUNCH_RESERVE_RECORD: dict | None = None


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
        return {"model_box": int(model_box), "skipped": "the JAX backend was already initialised"}
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


def reserve_for_command_line(argv=None) -> dict | None:
    """The ``relax`` launch hook: reserve the texture of the command line's reference box."""

    global LAUNCH_RESERVE_RECORD
    paths = reference_maps_from_argv(argv if argv is not None else sys.argv[1:])
    if not paths:
        return None
    LAUNCH_RESERVE_RECORD = reserve_projector_texture_memory(
        model_box_from_map_headers(paths), PROJECTION_PADDING_FACTOR
    )
    return LAUNCH_RESERVE_RECORD


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
