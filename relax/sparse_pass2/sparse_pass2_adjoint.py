"""Adjoint (BPref) accumulation of the sparse pass 2.

The chunked adjoint-block accumulation over hypothesis rows that the resident M-step uses.
"""

from __future__ import annotations

import logging

from relax.helpers.adjoint import adjoint_slice_volume_half as _adjoint_slice_volume_half
from relax.helpers.adjoint import adjoint_slice_volume_windowed as _adjoint_slice_volume_windowed
from relax.sparse_pass2.sparse_pass2_budget import (
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
        stop = min(start + max_rows, n_rows)
        if use_windowed_adjoint:
            volume = _adjoint_slice_volume_windowed(
                flat_block[start:stop],
                window_indices,
                flat_rotations[start:stop],
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
        else:
            volume = _adjoint_slice_volume_half(
                flat_block[start:stop],
                flat_rotations[start:stop],
                volume,
                image_shape,
                volume_shape,
                disc_type,
                half_image,
                half_volume,
            )
    return volume


