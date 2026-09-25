"""Tilt images of the resident pass 2's units (cryo-ET, S4.2).

The posterior unit of the resident pass 2 is a particle; for subtomograms it owns several
tilt images (docs/development/resident_segments.md, "The scoring side"). A capacity chunk
holds a contiguous range of units, so it also holds the contiguous range of their images.
:func:`chunk_tilt_layout` enumerates, for each image slot ``s``, which chunk image scores
each hypothesis row: the ``s``-th visible tilt of the row's unit, in RELION's ``img_id``
order (frame order, ``tomo_input.relion_image_geometry``). The scorer and the M-step loop
over ``s`` and add each image's contribution in that order, as RELION's ``img_id`` loops
do (acc_ml_optimiser_impl.h:2282, :4268).

Single-particle data are the one-image case: ``unit_image_offsets = arange(n + 1)``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from relax.refinement import tomo_particles


class ChunkTiltLayout(NamedTuple):
    """One chunk's tilt images and each row's image per image slot (host NumPy)."""

    image_ids: np.ndarray  # int64 [C_I], global image of each chunk image, -1 when padded
    image_unit_local: np.ndarray  # int64 [C_I], chunk-local unit of each image, C_U when padded
    slot_image_ids: np.ndarray  # int32 [S, C_R], chunk-local image of a row's unit's s-th image, -1 past its images
    n_valid_images: int


def validate_unit_image_offsets(unit_image_offsets, n_units: int) -> np.ndarray:
    """The CSR of images over units: ``[n_units + 1]``, starting at 0, every unit owning an image."""

    offsets = np.asarray(unit_image_offsets, dtype=np.int64).reshape(-1)
    if offsets.shape != (int(n_units) + 1,) or offsets[0] != 0:
        raise ValueError(f"unit_image_offsets must have shape ({int(n_units) + 1},) and start at 0")
    if np.any(np.diff(offsets) < 1):
        raise ValueError("every unit must own at least one image")
    return offsets


def max_images_per_unit(unit_image_offsets) -> int:
    return int(np.max(np.diff(np.asarray(unit_image_offsets, dtype=np.int64))))


def chunk_tilt_layout(
    unit_image_offsets,
    *,
    unit_start: int,
    n_valid_units: int,
    row_unit_local,
    n_valid_rows: int,
    image_capacity: int,
    slot_capacity: int,
) -> ChunkTiltLayout:
    """The images of units ``unit_start : unit_start + n_valid_units`` and every row's image per slot.

    ``row_unit_local`` is the chunk-local unit of each of the chunk's ``C_R`` rows (rows past
    ``n_valid_rows`` are padding). ``slot_capacity`` is the chunk program's image-slot count
    ``S``, at least the most images any unit of the chunk owns.
    """

    offsets = np.asarray(unit_image_offsets, dtype=np.int64)
    unit_start, n_valid_units, n_valid_rows = int(unit_start), int(n_valid_units), int(n_valid_rows)
    first = int(offsets[unit_start])
    unit_offsets = offsets[unit_start : unit_start + n_valid_units + 1] - first
    counts = np.diff(unit_offsets)
    n_valid_images = int(unit_offsets[-1])
    if n_valid_images > int(image_capacity):
        raise ValueError(f"chunk has {n_valid_images} images, more than its image capacity {image_capacity}")
    if counts.size and int(counts.max()) > int(slot_capacity):
        raise ValueError(f"a unit owns {int(counts.max())} images, more than the {slot_capacity} image slots")

    image_ids = np.full(int(image_capacity), -1, dtype=np.int64)
    image_ids[:n_valid_images] = np.arange(first, first + n_valid_images)
    image_unit_local = np.full(int(image_capacity), n_valid_units, dtype=np.int64)
    image_unit_local[:n_valid_images] = np.repeat(np.arange(n_valid_units), counts)

    row_unit = np.asarray(row_unit_local, dtype=np.int64)[:n_valid_rows]
    if row_unit.size and (row_unit.min() < 0 or row_unit.max() >= n_valid_units):
        raise ValueError("a valid row addresses a unit outside the chunk")
    slot_image_ids = np.full((int(slot_capacity), np.asarray(row_unit_local).shape[0]), -1, dtype=np.int32)
    for slot in range(int(slot_capacity)):
        slot_image_ids[slot, :n_valid_rows] = tomo_particles.image_slot_ids(row_unit, unit_offsets, slot)
    return ChunkTiltLayout(
        image_ids=image_ids,
        image_unit_local=image_unit_local,
        slot_image_ids=slot_image_ids,
        n_valid_images=n_valid_images,
    )


def tilt_slot_rotations(layout: ChunkTiltLayout, row_source_eulers, image_left, *, dtype=np.float32) -> np.ndarray:
    """``[S * C_R, 3, 3]`` fine and M-step matrices of every (image slot, row): ``inv(L_i A_r)``.

    RELION builds a tilt image's fine and backprojection matrices on the host with its left matrix
    ``L_i`` (the image's ``Aproj`` times the optics scale; generateEulerMatrices,
    acc_helper_functions_impl.h:248-255; acc_ml_optimiser_impl.h:2522-2547, 4521-4543), so every
    (image, rotation) pair has its own matrix. ``row_source_eulers`` ``[C_R, 3]`` are the rows' RELION
    Euler angles and ``image_left`` ``[n_images, 3, 3]`` the images' ``L``; entries of rows without an
    ``s``-th image are the identity and are never read. Slot ``s`` of row ``r`` is entry ``s * C_R + r``.
    """

    from relax.sampling import _relion_mstep_rotations_from_eulers

    n_slots, n_rows = layout.slot_image_ids.shape
    out = np.tile(np.eye(3, dtype=dtype), (n_slots * n_rows, 1, 1))
    chunk_image = layout.slot_image_ids.reshape(-1).astype(np.int64)
    valid = chunk_image >= 0
    if np.any(valid):
        rows = np.tile(np.arange(n_rows), n_slots)[valid]
        images = layout.image_ids[chunk_image[valid]]
        out[valid] = _relion_mstep_rotations_from_eulers(
            np.asarray(row_source_eulers, dtype=np.float64)[rows],
            dtype=dtype,
            left_matrices=np.asarray(image_left, dtype=np.float64)[images],
        )
    return out
