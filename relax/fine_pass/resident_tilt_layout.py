"""The tilt-image layout of one resident chunk (cryo-ET, S4.2).

``chunk_tilt_layout`` lays a chunk's subtomogram particles out as image slots (``ChunkTiltLayout``: each slot's
unit, tilt image and validity), and ``tilt_slot_rotations`` gives every (row, slot) pair its tilt image's
rotation. ``resident_tilts.run_tilt_chunk`` calls them; this module imports nothing from it.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from relax.fine_pass.resident_candidates import map_over_classes
from relax.sampling import tomo_particles


class ChunkTiltLayout(NamedTuple):
    """One chunk's tilt images and each row's image per image slot (host NumPy)."""

    image_ids: np.ndarray  # int64 [C_I], global image of each chunk image, -1 when padded
    image_unit_local: np.ndarray  # int64 [C_I], chunk-local unit of each image, n_valid_units when padded
    slot_image_ids: np.ndarray  # int32 [S, C_R], chunk-local image of a row's unit's s-th image, -1 past its images
    n_valid_images: int


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


# Entries of (image slot, row) matrices one host thread builds before another joins it.
_SLOT_ROTATION_THREAD_ENTRIES = 32768


def tilt_slot_rotations(
    layout: ChunkTiltLayout, row_source_eulers, image_left, *, row_spa_matrices=None, dtype=np.float32
) -> np.ndarray:
    """``[S * C_R, 3, 3]`` fine and M-step matrices of every (image slot, row): ``inv(L_i A_r)``.

    RELION builds a tilt image's fine and backprojection matrices on the host with its left matrix
    ``L_i`` (the image's ``Aproj`` times the optics scale; generateEulerMatrices,
    acc_helper_functions_impl.h:248-255; acc_ml_optimiser_impl.h:1709-1734, 3212-3234), so every
    (image, rotation) pair has its own matrix. It passes ``L`` only when it is not the identity
    (``isIdentity``, :1100-1104; ``tomo_particles.relion_left_matrices``); an identity image keeps the
    SPA matrices, ``row_spa_matrices`` ``[C_R, 3, 3]`` when given. ``row_source_eulers`` ``[C_R, 3]``
    are the rows' RELION Euler angles and ``image_left`` ``[n_images, 3, 3]`` the images' ``L``; entries
    of rows without an ``s``-th image are the identity and are never read. Slot ``s`` of row ``r`` is
    entry ``s * C_R + r``.
    """

    from relax.sampling.grids import relion_mstep_rotations_from_euler_matrices, relion_mstep_rotations_from_eulers
    from relax.sampling.healpix import euler_angles_to_matrix
    from relax.sampling.tomo_particles import relion_left_matrices

    n_slots, n_rows = layout.slot_image_ids.shape
    out = np.empty((n_slots * n_rows, 3, 3), dtype=dtype)
    out[:] = np.eye(3, dtype=dtype)
    chunk_image = layout.slot_image_ids.reshape(-1)
    # Only the entries with an image are built: a particle has fewer images than the pass's slots.
    entries = np.flatnonzero(chunk_image >= 0)
    if entries.size == 0:
        return out
    rows = entries % n_rows
    entry_image = chunk_image[entries].astype(np.int64)
    # One L per chunk image and one Euler matrix per row, gathered per entry (the trigonometry of a row
    # serves all of its images).
    left, applies = relion_left_matrices(
        np.asarray(image_left, dtype=np.float64)[layout.image_ids[: int(layout.n_valid_images)]]
    )
    with_left = applies[entry_image]
    if np.any(with_left):
        row_matrices = euler_angles_to_matrix(np.asarray(row_source_eulers, dtype=np.float64))
        left_entries, left_rows, left_images = entries[with_left], rows[with_left], entry_image[with_left]

        def inverses(part):
            return relion_mstep_rotations_from_euler_matrices(
                row_matrices[left_rows[part]], dtype=dtype, left_matrices=left[left_images[part]]
            )

        # Each entry's inverse is its own NumPy arithmetic, so a few host threads build them side by side
        # (a soft global iteration has about 300,000 entries in each of thousands of chunks, relax#80).
        n_parts = min(8, left_entries.size // _SLOT_ROTATION_THREAD_ENTRIES) + 1
        bounds = np.linspace(0, left_entries.size, n_parts + 1, dtype=np.int64)
        parts = [slice(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])]
        for part, block in zip(parts, map_over_classes(inverses, parts)):
            out[left_entries[part]] = block
    if not np.all(with_left):
        spa_rows = rows[~with_left]
        out[entries[~with_left]] = (
            np.asarray(row_spa_matrices, dtype=dtype)[spa_rows]
            if row_spa_matrices is not None
            else relion_mstep_rotations_from_eulers(
                np.asarray(row_source_eulers, dtype=np.float64)[spa_rows], dtype=dtype
            )
        )
    return out
