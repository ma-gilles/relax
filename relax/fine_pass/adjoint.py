"""Adjoint (BPref) accumulation of the sparse pass 2.

The chunked adjoint-block accumulation over hypothesis rows that the resident M-step uses.
"""

from __future__ import annotations

import logging
from functools import partial

import jax
import jax.numpy as jnp

from relax.projection.adjoint import ReferenceSphereClip
from relax.projection.adjoint import adjoint_slice_volume_half as _adjoint_slice_volume_half
from relax.projection.adjoint import adjoint_slice_volume_windowed as _adjoint_slice_volume_windowed
from relax.runtime.memory_budget import (
    _dtype_itemsize,
)

logger = logging.getLogger(__name__)


_adjoint_block_chunk_log_keys: set[tuple[str, int, int, int, int]] = set()


def _flat_block_row_bytes(flat_block) -> int:
    if flat_block is None or len(flat_block.shape) == 0:
        return 1
    n_pixels = int(flat_block.shape[1]) if len(flat_block.shape) > 1 else 1
    return max(1, n_pixels * _dtype_itemsize(flat_block.dtype))


def _adjoint_block_chunk_rows(flat_block, *, max_block_bytes: int) -> int:
    if flat_block is None:
        return 1
    row_bytes = _flat_block_row_bytes(flat_block)
    return max(1, int(max_block_bytes) // row_bytes)


def _accumulate_adjoint_block_chunked(
    flat_block,
    flat_rotations,
    volume,
    *,
    window_indices=None,
    use_windowed_adjoint: bool,
    image_shape,
    volume_shape,
    disc_type,
    half_image: bool,
    half_volume: bool,
    max_r,
    relion_x_half: bool,
    max_block_bytes: int,
    log_label: str,
    runtime_max_r=None,
):
    """Accumulate adjoint-slice rows in capped chunks for pathological tails.

    ``runtime_max_r`` (windowed adjoint only) clips at a traced radius up to ``max_r``.
    """

    if runtime_max_r is not None and not use_windowed_adjoint:
        raise NotImplementedError("a runtime adjoint radius needs the windowed (indexed) adjoint")
    if isinstance(max_r, ReferenceSphereClip) and max_r.reference_radius is not None and not use_windowed_adjoint:
        raise NotImplementedError("the rotated-radius clip of anisotropic magnification needs the windowed adjoint")
    if flat_block is None:
        return volume
    n_rows = int(flat_block.shape[0])
    if n_rows == 0:
        return volume
    max_rows = _adjoint_block_chunk_rows(flat_block, max_block_bytes=max_block_bytes)
    if n_rows <= max_rows:
        if use_windowed_adjoint:
            return _adjoint_slice_volume_windowed(
                flat_block,
                window_indices,
                flat_rotations,
                volume,
                image_shape,
                volume_shape,
                disc_type,
                half_image,
                half_volume,
                max_r,
                relion_x_half,
                runtime_max_r,
            )
        return _adjoint_slice_volume_half(
            flat_block,
            flat_rotations,
            volume,
            image_shape,
            volume_shape,
            disc_type,
            half_image,
            half_volume,
        )

    n_chunks = (n_rows + max_rows - 1) // max_rows
    n_pixels = int(flat_block.shape[1]) if len(flat_block.shape) > 1 else 1
    log_key = (str(log_label), n_rows, n_pixels, max_rows, int(max_block_bytes))
    if log_key not in _adjoint_block_chunk_log_keys:
        _adjoint_block_chunk_log_keys.add(log_key)
        logger.info(
            "Sparse pass-2 adjoint block chunking: %s rows=%d pixels=%d max_rows=%d chunks=%d max_block_bytes=%.2f GiB",
            log_label,
            n_rows,
            n_pixels,
            max_rows,
            n_chunks,
            float(max_block_bytes) / float(1024**3),
        )

    for start in range(0, n_rows, max_rows):
        volume = _adjoint_rows(
            flat_block,
            flat_rotations,
            jnp.int32(start),
            volume,
            window_indices,
            runtime_max_r,
            rows=min(start + max_rows, n_rows) - start,
            use_windowed_adjoint=use_windowed_adjoint,
            image_shape=image_shape,
            volume_shape=volume_shape,
            disc_type=disc_type,
            half_image=half_image,
            half_volume=half_volume,
            max_r=max_r,
            relion_x_half=relion_x_half,
        )
    return volume


@partial(
    jax.jit,
    static_argnames=(
        "rows",
        "use_windowed_adjoint",
        "image_shape",
        "volume_shape",
        "disc_type",
        "half_image",
        "half_volume",
        "max_r",
        "relion_x_half",
    ),
)
def _adjoint_rows(
    flat_block,
    flat_rotations,
    start,
    volume,
    window_indices,
    runtime_max_r,
    *,
    rows,
    use_windowed_adjoint,
    image_shape,
    volume_shape,
    disc_type,
    half_image,
    half_volume,
    max_r,
    relion_x_half,
):
    """One chunk of rows ``start : start + rows`` adjoint-sliced into ``volume``.

    The row slices are inside the program: taken eagerly, each was a program per
    chunk shape, compiled again at every new current size. ``start + rows`` never
    passes the end, so the slice is never clamped.
    """

    block = jax.lax.dynamic_slice_in_dim(flat_block, start, rows, axis=0)
    rotations = jax.lax.dynamic_slice_in_dim(flat_rotations, start, rows, axis=0)
    if use_windowed_adjoint:
        return _adjoint_slice_volume_windowed(
            block,
            window_indices,
            rotations,
            volume,
            image_shape,
            volume_shape,
            disc_type,
            half_image,
            half_volume,
            max_r,
            relion_x_half,
            runtime_max_r,
        )
    return _adjoint_slice_volume_half(
        block,
        rotations,
        volume,
        image_shape,
        volume_shape,
        disc_type,
        half_image,
        half_volume,
    )
