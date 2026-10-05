"""Tilt images of the resident pass 2's units (cryo-ET, S4.2).

The posterior unit of the resident pass 2 is a particle; for subtomograms it owns several
tilt images (docs/development/resident_segments.md, "The scoring side"). A capacity chunk
holds a contiguous range of units, so it also holds the contiguous range of their images.
:func:`chunk_tilt_layout` enumerates, for each image slot ``s``, which chunk image scores
each hypothesis row: the ``s``-th visible tilt of the row's unit, in RELION's ``img_id``
order (frame order, ``tomo_input.relion_image_geometry``). The scorer and the M-step loop
over ``s`` and add each image's contribution in that order, as RELION's ``img_id`` loops
do (acc_ml_optimiser_impl.h:1490, :2959).

Single-particle data are the one-image case: ``unit_image_offsets = arange(n + 1)``.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from relax.refinement import tomo_particles
from relax.sparse_pass2.sparse_pass2_projection_blocks import project_rows_by_class
from relax.sparse_pass2.sparse_pass2_wavg import weighted_image_power_from_shells


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

    from relax.refinement.tomo_particles import relion_left_matrices
    from relax.sampling import _relion_mstep_rotations_from_eulers

    n_slots, n_rows = layout.slot_image_ids.shape
    out = np.tile(np.eye(3, dtype=dtype), (n_slots * n_rows, 1, 1))
    chunk_image = layout.slot_image_ids.reshape(-1).astype(np.int64)
    valid = chunk_image >= 0
    if not np.any(valid):
        return out
    rows = np.tile(np.arange(n_rows), n_slots)
    images = np.zeros_like(chunk_image)
    images[valid] = layout.image_ids[chunk_image[valid]]
    left, applies = relion_left_matrices(np.asarray(image_left, dtype=np.float64)[images])
    with_left = valid & applies
    if np.any(with_left):
        out[with_left] = _relion_mstep_rotations_from_eulers(
            np.asarray(row_source_eulers, dtype=np.float64)[rows[with_left]],
            dtype=dtype,
            left_matrices=left[with_left],
        )
    spa = valid & ~applies
    if np.any(spa):
        out[spa] = (
            np.asarray(row_spa_matrices, dtype=dtype)[rows[spa]]
            if row_spa_matrices is not None
            else _relion_mstep_rotations_from_eulers(
                np.asarray(row_source_eulers, dtype=np.float64)[rows[spa]], dtype=dtype
            )
        )
    return out


# ---------------------------------------------------------------------------
# One chunk of subtomogram particles on the resident engine
# ---------------------------------------------------------------------------


class TiltPassInputs(NamedTuple):
    """What a resident pass needs beyond the SPA operands when its units are subtomogram particles.

    Units are the half's particles (the rows of the candidate tables), images are the flat per-tilt
    rows of the half's dataset in RELION's ``img_id`` order (tomo_input.relion_image_geometry), and
    ``unit_image_offsets`` is the CSR between them.
    """

    unit_image_offsets: np.ndarray  # int64 [U + 1]
    image_left: np.ndarray  # float64 [I, 3, 3], Aproj_i times the optics scale (generateEulerMatrices' L)
    image_angles: np.ndarray  # float32 [I, T, 2], each image's fine phases (tomo_particles.tilt_translation_angles)
    image_noise_scale: np.ndarray  # float32 [I], 1 / n_images of the image's particle
    unit_translation_prior: np.ndarray  # float32 [U, T], the particle's fine 3D offset log prior
    unit_translation_sqdist_ang: np.ndarray | None  # [U, T] squared offset distance for sigma2_offset, or None
    unit_optics_groups: np.ndarray | None  # int32 [U] dense optics group of each particle, or None
    fine_source_eulers: np.ndarray | None  # float64 [n_fine_rot, 3], the fine grid's RELION Euler angles
    fine_rotations: np.ndarray  # [n_fine_rot, 3, 3], the SPA fine matrices (an identity L keeps them)
    slot_capacity: int  # the most images any particle of the half owns


def tilt_chunk_image_capacity(unit_capacity: int, slot_capacity: int) -> int:
    """A chunk's image capacity: its unit capacity times the most images per particle."""

    return int(unit_capacity) * int(slot_capacity)


def tilt_projection_slot_block(max_rows: int, *, slot_capacity: int, slot_row_bytes: int, budget_bytes) -> int:
    """Image slots whose projections of the largest particle's rows fit half the chunk budget at once.

    ``slot_row_bytes`` is one row's projection bytes for one slot. Every slot (one block, the chunk's
    projections made once) when they fit or the budget is unknown; else the largest count that fits,
    at least one.
    """

    slot_capacity = int(slot_capacity)
    need = int(max_rows) * int(slot_row_bytes)
    if budget_bytes is None or need <= 0 or need * slot_capacity <= int(budget_bytes) // 2:
        return slot_capacity
    return int(min(slot_capacity, max(1, (int(budget_bytes) // 2) // need)))


def _slot_tables_block(slots: "SlotMstepTables", start: int, stop: int) -> "SlotMstepTables":
    """Slots ``start:stop`` of :func:`_slot_mstep_tables`, renumbered from 0 (their block's projections)."""

    block = slots._replace(**{name: getattr(slots, name)[start:stop] for name in slots._fields})
    return block._replace(slot=jnp.arange(stop - start, dtype=jnp.int32))


def run_tilt_chunk(
    chunk,
    *,
    tables,
    tilt: TiltPassInputs,
    resident_operands,
    project_rotations,
    base_tables,
    n_fine_trans,
    spec_kwargs,
    stats,
    Ft_y_total,
    Ft_ctf_total,
    n_fine_rot: int,
    image_shape,
    rect_indices_device,
    exact_positions_device,
    tile_budget_bytes: int,
    operand_image_start: int = 0,
    slot_block: int | None = None,
    score_pixel_indices=None,
):
    """Score, weight and backproject one chunk of particles; return ``(Ft_y_total, Ft_ctf_total, stats)``.

    ``Ft_y_total``/``Ft_ctf_total`` are the pass's BPref pairs, one per accumulator slot (class +
    K * group, resident_pass2); ``resident_operands`` hold the half's images from
    ``operand_image_start`` on: the whole half (0), or only this chunk's images when the half's
    operands do not fit (resident_pass2).

    - Rows are the chunk's particle hypotheses (unit-major, as the candidate tables hold them).
    - Each row is scored by every image of its particle, image slot by image slot, with that image's
      matrix ``inv(L_i A_r)`` (a chunk-local projection cache laid out ``slot * C_R + row``), phases,
      CTF and noise, and the diff2 summed per row (``score_tilt_image_rows``). The minimum, the weights
      and the significance are per particle (RELION's img_id loop, acc_ml_optimiser_impl.h:2282).
    - The M-step visits the slots again: each (image, row) backprojects with the particle's posterior,
      the image's matrix and phases; the noise and norm sums carry 1 / n_images, the scale sums and
      the backprojection do not (docs in PLAN.md, "S4.2 statistics semantics").
    - The M-step visits each particle's own translations with posterior mass (:func:`unit_mstep_translations`):
      a row's posterior and its image's phases are taken at its particle's ``k`` translations, so the
      translated tiles cover only those (the dropped terms carry exactly zero posterior).
    - Memory: the translated Wavg tiles of one slot are ``[C_U, T_b, P_rect]`` (the slot view) and
      ``[block rows, T_b, P_rect]`` (the per-row gather), for blocks of ``T_b`` of the particles'
      ``k`` translations, sized so both fit ``tile_budget_bytes`` (:func:`mstep_translation_blocks`).
    - K>1 classes (RELION subtomogram Class3D): a particle's rows are class-major (the candidate
      tables' class layout), each (slot, row) projects from its class's reference, the posterior
      segment is the particle's over every class, and each accumulator slot's rows backproject into
      its own BPref pair, their scale sums under the class's mask (``_fold_class_scale_sums``).
    - ``slot_block`` (:func:`tilt_projection_slot_block`): when the (slot, row) projections of every slot
      do not fit, they are made a block of slots at a time, for the scores (the running diff2 carried
      across blocks, so its slot order is kept) and again for the M-step.
    The per-stage (non-jitted) path.
    """

    from relax.cuda import kernels as em_cuda_kernels
    from relax.sparse_pass2 import resident_pass2 as rp
    from relax.sparse_pass2.resident_candidates import expand_chunk_mask_jnp
    from relax.sparse_pass2.resident_scoring import tilt_image_rows_raw_diff2, tilt_rows_scores_from_raw

    row_capacity = int(chunk.row_capacity)
    unit_capacity = int(chunk.image_capacity)
    n_valid_rows = int(chunk.n_valid_rows)
    n_slots = int(tilt.slot_capacity)
    image_capacity = tilt_chunk_image_capacity(unit_capacity, n_slots)
    n_fine_trans = int(n_fine_trans)

    n_classes = int(tables.n_classes)
    n_accumulators = len(Ft_y_total)
    if n_accumulators != int(tables.n_slots):
        raise ValueError(f"{n_accumulators} BPref pairs for the tables' {int(tables.n_slots)} accumulator slots")
    rows = rp._make_chunk_row_arrays(tables, chunk, n_fine_trans, place=rp._PLACE_ON_DEVICE, n_fine_rot=n_fine_rot)
    _block_tables, host, _local_chunk = rp._chunk_host_rows(tables, chunk)
    layout = chunk_tilt_layout(
        tilt.unit_image_offsets,
        unit_start=int(chunk.image_start),
        n_valid_units=int(chunk.n_valid_images),
        row_unit_local=np.asarray(host["row_image_local"]),
        n_valid_rows=n_valid_rows,
        image_capacity=image_capacity,
        slot_capacity=n_slots,
    )
    valid_images = layout.image_ids >= 0
    image_slots = np.where(valid_images, layout.image_ids, -1).astype(np.int32)
    operand_slots = np.where(valid_images, layout.image_ids - int(operand_image_start), -1).astype(np.int32)
    safe_images = np.where(valid_images, layout.image_ids, 0)

    # --- projections of every (slot, row): RELION's host inv(L_i A_r) ------
    row_fine_rot = np.asarray(host["row_fine_rot"], dtype=np.int64)
    row_eulers = (
        np.zeros((row_fine_rot.size, 3))
        if tilt.fine_source_eulers is None
        else np.asarray(tilt.fine_source_eulers, dtype=np.float64)[row_fine_rot]
    )
    slot_matrices = tilt_slot_rotations(
        layout, row_eulers, tilt.image_left, row_spa_matrices=np.asarray(tilt.fine_rotations)[row_fine_rot]
    )
    row_class = np.zeros(row_capacity, dtype=np.int64)
    if n_classes > 1:
        row_class[:n_valid_rows] = np.asarray(host["row_class"], dtype=np.int64)[:n_valid_rows]
    entry_class = np.tile(row_class, n_slots)
    slot_block = n_slots if slot_block is None else max(1, min(int(slot_block), n_slots))
    slot_blocks = [(start, min(start + slot_block, n_slots)) for start in range(0, n_slots, slot_block)]

    def project_block(start, stop):
        entries = slice(start * row_capacity, stop * row_capacity)
        return project_slot_rows(project_rotations, slot_matrices[entries], entry_class[entries], n_classes=n_classes)

    # One block: the chunk's projections are made once, for the scores and the M-step.
    whole = project_block(0, n_slots) if len(slot_blocks) == 1 else None
    score_cache, recon_cache, recon_abs2_cache = (None, None, None) if whole is None else whole

    # --- per-image operands: the SPA gather, with each image's own phases ---
    image_angles = jnp.asarray(np.asarray(tilt.image_angles, dtype=np.float32)[safe_images])
    # The translated Wavg tile is [images, T, P_rect]; with a subtomogram's 3D grid (thousands of
    # translations) it is built per image slot in the M-step below, for C_U images at a time, never
    # for the chunk's C_U * S images. The chunk-wide gather takes one zero translation.
    recon = rp.gather_resident_chunk_operands(
        resident_operands,
        operand_slots,
        translation_angles=np.zeros((1, 2), dtype=np.float32),
        rect_indices=rect_indices_device,
        exact_positions=exact_positions_device,
        image_shape=image_shape,
    )
    # Each chunk image's unshifted Wavg window (the resident operands keep only that window of the
    # image); the slot views translate it with each image's own phases.
    from relax.sparse_pass2.resident_operands import _gather_rows

    chunk_wavg_window = _gather_rows(
        resident_operands.wavg_image_rect,
        jnp.asarray(np.maximum(operand_slots, 0), dtype=jnp.int32),
        jnp.asarray(operand_slots >= 0),
    )
    noise_scale = np.where(valid_images, np.asarray(tilt.image_noise_scale, dtype=np.float32)[safe_images], 0.0)
    operands = rp._make_chunk_stage_operands(recon, None)._replace(image_noise_scale=jnp.asarray(noise_scale))
    if resident_operands.cc_half_batch_norm is not None:
        # The CC iteration's per-image evidence offset (the chunk gather carries the Gaussian fields).
        operands = operands._replace(
            cc_half_batch_norm=jnp.where(
                jnp.asarray(operand_slots >= 0),
                jnp.asarray(resident_operands.cc_half_batch_norm)[jnp.asarray(np.maximum(operand_slots, 0))],
                jnp.float32(0.0),
            )
        )

    def block_tables(start, stop, caches):
        return base_tables._replace(
            projection_score_cache=caches[0],
            projection_recon_cache=caches[1],
            projection_recon_abs2_cache=caches[2],
            mstep_grid=jnp.asarray(
                slot_matrices[start * row_capacity : stop * row_capacity], dtype=base_tables.mstep_grid.dtype
            ),
            translation_angles=image_angles,
            cache_slot_fine_rot=None,
        )

    stage_tables = block_tables(0, n_slots, (score_cache, recon_cache, recon_abs2_cache))
    spec = rp._make_chunk_program_spec(
        row_capacity=row_capacity, image_capacity=image_capacity, n_fine_trans=n_fine_trans, **spec_kwargs
    )

    # --- scoring and the per-particle posterior -----------------------------
    row_is_valid = jnp.arange(row_capacity, dtype=jnp.int32) < rows.n_valid_rows
    slot_image_ids = jnp.asarray(layout.slot_image_ids)
    unit_ids = np.asarray(host["image_ids"], dtype=np.int64)
    unit_prior = np.zeros((unit_capacity, n_fine_trans), dtype=np.float32)
    valid_units = unit_ids >= 0
    unit_prior[valid_units] = np.asarray(tilt.unit_translation_prior, dtype=np.float32)[unit_ids[valid_units]]
    candidate_mask = expand_chunk_mask_jnp(rows.row_mask_bits, rows.row_mask_mode, base_tables.fine_translation_parent)
    kernel_row_image_ids = jnp.where(row_is_valid, rows.row_image_local, jnp.int32(-1))
    if spec.firstiter_cc:
        # RELION's --firstiter_cc iteration: each image's normalized CC added over the particle's images,
        # the winner taking all (acc_ml_optimiser_impl.h:1290, diff2.cuh cuda_kernel_diff2_CC_fine).
        scored = tilt_cc_scores(
            lambda start, stop: score_cache if whole is not None else project_block(start, stop)[0],
            slot_blocks,
            np.asarray(layout.slot_image_ids),
            _unit_slot_images(layout, unit_capacity=unit_capacity, slot_capacity=n_slots),
            operands,
            image_angles,
            candidate_mask,
            row_is_valid,
            rows.row_image_local,
            np.where(valid_images, np.asarray(layout.image_unit_local), unit_capacity),
            half_weights=base_tables.half_weights,
            full_to_compact=base_tables.full_to_compact,
            score_pixel_indices=score_pixel_indices,
            image_shape=image_shape,
            row_capacity=row_capacity,
            unit_capacity=unit_capacity,
            block_rows=int(spec.mstep_block_rows),
            tile_budget_bytes=int(tile_budget_bytes),
        )
        posterior = rp._winner_take_all_posterior(
            scored,
            rows,
            image_capacity=unit_capacity,
            cuda_backproject=em_cuda_kernels,
            row_is_valid=row_is_valid,
            kernel_row_image_ids=kernel_row_image_ids,
        )
    else:
        raw_sum = None
        for start, stop in slot_blocks:
            block_score_cache = score_cache if whole is not None else project_block(start, stop)[0]
            raw_sum = tilt_image_rows_raw_diff2(
                block_score_cache,
                slot_image_ids[start:stop],
                operands.score_input,
                operands.corr_img_score,
                (
                    jnp.zeros((image_capacity,), dtype=jnp.float32)
                    if operands.highres_xi2_half is None
                    else jnp.asarray(operands.highres_xi2_half, dtype=jnp.float32)
                ),
                candidate_mask,
                row_is_valid,
                half_weights=base_tables.half_weights,
                image_translation_angles=image_angles,
                full_to_compact=base_tables.full_to_compact,
                logical_current_size=jnp.asarray(spec.current_size, dtype=jnp.int32),
                running=raw_sum,
            )
            del block_score_cache
        scored = tilt_rows_scores_from_raw(
            raw_sum,
            rows.row_image_local,
            rows.row_log_prior,
            jnp.asarray(unit_prior),
            candidate_mask,
            row_is_valid,
            unit_capacity=unit_capacity,
        )
        # The particle's segment spans every class: RELION's joint minimum, normalization and
        # significance (ml_optimiser.cpp:8411, :9225); with K>1 also each (particle, class) sub-segment.
        posterior = rp._chunk_posterior_from_scores(
            scored,
            rows,
            stage_tables,
            spec=rp._make_chunk_program_spec(
                row_capacity=row_capacity, image_capacity=unit_capacity, n_fine_trans=n_fine_trans, **spec_kwargs
            ),
            cuda_backproject=em_cuda_kernels,
            row_is_valid=row_is_valid,
            kernel_row_image_ids=kernel_row_image_ids,
        )
    row_posterior = posterior.row_posterior

    # --- M-step, one pass per image slot ------------------------------------
    # Each slot backprojects the C_U images that are its units' s-th images (a slot view of the chunk's
    # operands, indexed by unit), and its per-image partials land on those images' chunk rows.
    # The carry's dtypes follow the projections' (one slot's when they are made per block).
    mstep = rp._initial_mstep_carry(
        Ft_y_total[0],
        Ft_ctf_total[0],
        operands,
        stage_tables if whole is not None else block_tables(0, 1, project_block(0, 1)),
        spec=spec,
    )
    # Only each particle's translations with posterior mass enter the M-step: a subtomogram's 3D grid
    # has thousands, a particle a handful of significant samples, and the Wavg rectangle is
    # [images, T, P_rect] (gathered per row, [rows, T, P_rect]). Dropped translations carry exactly
    # zero posterior.
    block_rows = int(spec.mstep_block_rows)
    host_posterior = np.asarray(row_posterior)[:n_valid_rows]  # sliced on the host: no program per row count
    # Rows without posterior mass (RELION's non-significant samples) add exact zeros; the M-step
    # blocks visit only the rows with mass (row_has_mass), gathered to the front of each slot's order.
    row_has_mass = np.any(host_posterior > 0.0, axis=1)
    host_row_unit = np.asarray(host["row_image_local"], dtype=np.int64)
    unit_translations = unit_mstep_translations(
        host_posterior, host_row_unit[:n_valid_rows], unit_capacity=unit_capacity, n_fine_trans=n_fine_trans
    )
    n_unit_trans = int(unit_translations.index.shape[1])
    # The rows' posterior at their particle's translations, [C_R, k]; padded entries are zero.
    row_unit_device = np.zeros(row_capacity, dtype=np.int64)
    row_unit_device[:n_valid_rows] = host_row_unit[:n_valid_rows]
    unit_posterior = jnp.where(
        jnp.asarray(unit_translations.valid[row_unit_device]),
        jnp.take_along_axis(row_posterior, jnp.asarray(unit_translations.index[row_unit_device], dtype=jnp.int32), axis=1),
        jnp.zeros((), row_posterior.dtype),
    )
    translation_blocks = mstep_translation_blocks(
        MstepTranslations(index=np.arange(n_unit_trans, dtype=np.int64), valid=np.ones(n_unit_trans, dtype=bool)),
        bytes_per_translation=(unit_capacity + block_rows)
        * (int(rect_indices_device.shape[0]) + int(exact_positions_device.shape[0]))
        * 8,
        tile_budget_bytes=int(tile_budget_bytes),
    )
    slot_spec = rp._make_chunk_program_spec(
        row_capacity=row_capacity,
        image_capacity=unit_capacity,
        n_fine_trans=int(translation_blocks[0].index.size),
        **spec_kwargs,
    )
    unit_slot_images = _unit_slot_images(layout, unit_capacity=unit_capacity, slot_capacity=n_slots)
    kept_index = jnp.asarray(np.stack([kept.index for kept in translation_blocks]), dtype=jnp.int32)
    kept_valid = jnp.asarray(np.stack([kept.valid for kept in translation_blocks]))
    # Each accumulator slot (class + K * group) backprojects its own rows into its BPref pair; the
    # per-image partials add over the slots (_place_slot_partials), the scale sums under each class's
    # mask (RELION keeps XA/AA per class, acc_ml_optimiser_impl.h:4893-4912).
    row_accumulator = np.zeros(row_capacity, dtype=np.int64)
    if n_accumulators > 1:
        row_accumulator[:n_valid_rows] = np.asarray(host["row_slot"], dtype=np.int64)[:n_valid_rows]
    accumulator_slots, accumulator_posteriors = [], []
    for accumulator in range(n_accumulators):
        in_accumulator = row_accumulator == accumulator
        accumulator_slots.append(
            _slot_mstep_tables(
                unit_slot_images,
                row_unit=host_row_unit,
                row_has_mass=row_has_mass & in_accumulator[:n_valid_rows],
                n_valid_rows=n_valid_rows,
                row_capacity=row_capacity,
                block_rows=block_rows,
                image_angles=np.asarray(tilt.image_angles, dtype=np.float32),
                layout_image_ids=np.asarray(layout.image_ids),
                unit_translations=unit_translations.index,
                translation_blocks=translation_blocks,
                image_capacity=image_capacity,
            )
        )
        accumulator_posteriors.append(
            unit_posterior
            if n_accumulators == 1
            else jnp.where(jnp.asarray(in_accumulator)[:, None], unit_posterior, jnp.zeros((), unit_posterior.dtype))
        )
    Ft_y_out, Ft_ctf_out = list(Ft_y_total), list(Ft_ctf_total)
    # Each accumulator's slots are visited in slot order (block by block); an image is one slot's, so its
    # per-image partials add over the accumulators in the same order with or without blocks.
    for start, stop in slot_blocks:
        tables_b = stage_tables if whole is not None else block_tables(start, stop, project_block(start, stop))
        for accumulator in range(n_accumulators):
            slots = accumulator_slots[accumulator]
            mstep = _tilt_mstep_program(
                mstep._replace(Ft_y=Ft_y_out[accumulator], Ft_ctf=Ft_ctf_out[accumulator]),
                slots if whole is not None else _slot_tables_block(slots, start, stop),
                operands,
                tables_b,
                chunk_wavg_window,
                accumulator_posteriors[accumulator],
                kept_index,
                kept_valid,
                rect_indices_device,
                jnp.asarray(exact_positions_device, dtype=jnp.int32),
                slot_spec=slot_spec,
                image_shape=tuple(int(n) for n in image_shape),
            )
            if mstep.scale_xa_per_image is not None:
                mstep = rp._fold_class_scale_sums(mstep, stage_tables.wavg_scale_pixel_mask[accumulator % n_classes])
            Ft_y_out[accumulator], Ft_ctf_out[accumulator] = mstep.Ft_y, mstep.Ft_ctf
        del tables_b

    stats = accumulate_tilt_chunk_terms(
        stats,
        row_posterior=row_posterior,
        row_unit_local=rows.row_image_local,
        # The coarse parent grid is indexed by projection id (class * n_fine_rot + rotation).
        row_coarse_rot=jnp.where(
            row_is_valid,
            base_tables.coarse_parent_grid[rows.row_fine_rot],
            jnp.int32(int(spec.stats_config.n_coarse_rot)),
        ),
        row_fine_rot=jnp.asarray(row_fine_rot_device(host, row_capacity)),
        unit_ids=jnp.asarray(unit_ids, dtype=jnp.int32),
        unit_optics_groups=None
        if tilt.unit_optics_groups is None
        else jnp.asarray(
            np.where(valid_units, np.asarray(tilt.unit_optics_groups)[np.where(valid_units, unit_ids, 0)], 0),
            dtype=jnp.int32,
        ),
        unit_translation_sqdist_ang=None
        if tilt.unit_translation_sqdist_ang is None
        else jnp.asarray(np.asarray(tilt.unit_translation_sqdist_ang)[np.where(valid_units, unit_ids, 0)]),
        image_unit_local=jnp.asarray(layout.image_unit_local, dtype=jnp.int32),
        image_ids=jnp.asarray(image_slots),
        operands=operands,
        tables=stage_tables,
        mstep=mstep,
        posterior=posterior,
        rows=rows,
        spec=spec,
    )
    return tuple(Ft_y_out), tuple(Ft_ctf_out), stats


def tilt_cc_scores(
    block_score_cache,  # callable (start, stop) -> complex [(stop - start) * C_R, N], slots start:stop's projections
    slot_blocks,
    slot_image_ids,  # int [S, C_R] chunk image of each row per slot, -1 past its particle's images
    unit_slot_images,  # int [S, C_U] chunk image of each unit's s-th image, -1 past its images
    operands,  # the chunk's normalized-CC operands: score_input, corr_img_score (CC weight), cc_half_batch_norm
    image_angles,  # float32 [C_B, T, 2] each image's phases
    candidate_mask,  # bool [C_R, T]
    row_is_valid,
    row_unit_local,  # int32 [C_R]
    image_unit_local,  # int [C_B] each image's unit, C_U on padded images
    *,
    half_weights,
    full_to_compact,
    score_pixel_indices,
    image_shape,
    row_capacity: int,
    unit_capacity: int,
    block_rows: int,
    tile_budget_bytes: int,
):
    """The chunk's ``--firstiter_cc`` scores: each row's normalized CC summed over its particle's images.

    RELION's fine CC kernel adds ``-s / sqrt(s_cc)`` of every tilt image into the particle's diff2 in
    ``img_id`` order (diff2.cuh cuda_kernel_diff2_CC_fine; no Xi2 offset, :1288). Per image slot, the
    slot's images are translated with their own phases (RELION's score translation,
    ``relion_translate_score_f32``) over the translations some row may take, in blocks that fit
    ``tile_budget_bytes``, and each row is scored by the SPA reduction
    (``_relion_cuda_fine_normalized_cc_score``); the float32 running sum keeps the slot order.
    ``min_diff2`` is the particle's ``0.5 |X_i|^2`` summed over its images (the SPA evidence offset).
    """

    from relax.cuda import kernels as em_cuda_kernels
    from relax.sparse_pass2.resident_scoring import ResidentChunkScores
    from relax.sparse_pass2.sparse_pass2_scoring import _relion_cuda_fine_normalized_cc_score

    slot_image_ids = np.asarray(slot_image_ids, dtype=np.int64)
    unit_slot_images = np.asarray(unit_slot_images, dtype=np.int64)
    n_trans = int(candidate_mask.shape[1])
    n_pixels = int(operands.score_input.shape[1])
    kept = np.flatnonzero(np.any(np.asarray(candidate_mask & row_is_valid[:, None]), axis=0))
    if kept.size == 0:
        kept = np.zeros(1, dtype=np.int64)
    per_translation = (int(unit_capacity) + int(block_rows)) * n_pixels * 8
    t_block = int(min(kept.size, max(1, int(tile_budget_bytes) // max(per_translation, 1))))
    pixels = jnp.asarray(score_pixel_indices, dtype=jnp.int32)
    score_input = jnp.asarray(operands.score_input, dtype=jnp.complex64)
    weight = jnp.asarray(operands.corr_img_score, dtype=jnp.float32)
    row_unit = jnp.asarray(row_unit_local, dtype=jnp.int32)
    running = jnp.zeros((int(row_capacity), n_trans), dtype=jnp.float32)
    shape = tuple(int(n) for n in image_shape)
    for t_start in range(0, kept.size, t_block):
        index = kept[t_start : t_start + t_block]
        index_device = jnp.asarray(index, dtype=jnp.int32)
        block_angles = jnp.asarray(image_angles)[:, index_device]  # [C_B, T_b, 2]
        for start, stop in slot_blocks:
            cache = block_score_cache(start, stop)
            for slot in range(start, stop):
                units_images = unit_slot_images[slot]
                valid_units = units_images >= 0
                safe = jnp.asarray(np.where(valid_units, units_images, 0), dtype=jnp.int32)
                # [C_U, T_b, N]: the slot's images with their own phases, one launch.
                tiles = em_cuda_kernels.relion_translate_score_f32(
                    score_input[safe], block_angles[safe], pixels, shape
                ).reshape(safe.shape[0], index.size, -1)
                images = jnp.asarray(np.where(slot_image_ids[slot] >= 0, slot_image_ids[slot], 0), dtype=jnp.int32)
                has_image = jnp.asarray(slot_image_ids[slot] >= 0) & row_is_valid
                projections = jax.lax.dynamic_slice_in_dim(cache, (slot - start) * row_capacity, row_capacity, axis=0)

                def block_cc(first, projections=projections, tiles=tiles, images=images):
                    rows = jax.lax.dynamic_slice_in_dim(row_unit, first, block_rows)
                    row_images = jax.lax.dynamic_slice_in_dim(images, first, block_rows)
                    return _relion_cuda_fine_normalized_cc_score(
                        jax.lax.dynamic_slice_in_dim(projections, first, block_rows)[:, None, :],
                        tiles[rows],
                        weight[row_images][:, None, :],
                        half_weights,
                        full_to_compact,
                    )

                starts = jnp.arange(0, int(row_capacity), int(block_rows), dtype=jnp.int32)
                cc = jax.lax.map(block_cc, starts).reshape(int(row_capacity), index.size)
                running = running.at[:, index_device].add(
                    jnp.where(has_image[:, None], -cc.astype(jnp.float32), jnp.float32(0.0))
                )
            del cache
    valid_cells = candidate_mask & row_is_valid[:, None]
    scores = jnp.where(valid_cells & jnp.isfinite(running), -running, -jnp.inf)
    image_units = jnp.asarray(image_unit_local, dtype=jnp.int32)
    half_norm = jax.ops.segment_sum(
        jnp.asarray(operands.cc_half_batch_norm, dtype=jnp.float32), image_units, num_segments=int(unit_capacity) + 1
    )[: int(unit_capacity)]
    return ResidentChunkScores(
        raw_diff2=jnp.where(jnp.isfinite(scores), running, jnp.inf), scores=scores, min_diff2=half_norm
    )


def project_slot_rows(project_rotations, slot_matrices, entry_class, *, n_classes: int):
    """The ``(score, recon, |recon|^2)`` projections of every (slot, row) entry, each from its row's class.

    ``project_rotations(matrices, class_index=k)`` projects from class ``k``'s reference. With K>1 each
    class projects only its own entries (``entry_class``), in one call padded to
    :func:`class_call_length`, written in place at the
    entries' ``slot * C_R + row`` positions (:func:`project_rows_by_class`).
    """

    if int(n_classes) == 1:
        return project_rotations(slot_matrices)
    entry_class = np.asarray(entry_class, dtype=np.int64)
    return project_rows_by_class(
        project_rotations, np.asarray(slot_matrices), entry_class, n_rows=entry_class.size
    )


class SlotMstepTables(NamedTuple):
    """Every image slot's M-step inputs of one tilt chunk, stacked on a leading slot axis ``S``."""

    slot: jax.Array  # int32 [S]
    safe_images: jax.Array  # int32 [S, C_U] chunk image of each unit's s-th image (a valid image on padding)
    valid_images: jax.Array  # bool [S, C_U]
    targets: jax.Array  # int32 [S, C_U] chunk image the slot's partials land on, image_capacity (dropped) on padding
    order: jax.Array  # int32 [S, C_R] the slot's rows: those with mass and an image first, then the rest
    active: jax.Array  # bool [S, C_R] in visiting order
    units: jax.Array  # int32 [S, C_R] in visiting order, 0 where the row has no image in the slot
    n_blocks: jax.Array  # int32 [S] M-step blocks covering the active rows
    angles: jax.Array  # float32 [S, K, C_U, T_b, 2] each unit's image phases at its translations, per block


def _slot_mstep_tables(
    unit_slot_images,
    *,
    row_unit,
    row_has_mass,
    n_valid_rows: int,
    row_capacity: int,
    block_rows: int,
    image_angles,
    layout_image_ids,
    unit_translations,
    translation_blocks,
    image_capacity: int,
) -> SlotMstepTables:
    """Host tables for :func:`_tilt_mstep_program`: one transfer per chunk instead of several per slot.

    Rows without posterior mass (RELION's non-significant samples) add exact zeros, so each slot
    visits only its rows with mass and an image in the slot, gathered to the front (ascending).
    """

    unit_slot_images = np.asarray(unit_slot_images, dtype=np.int64)
    n_slots, _ = unit_slot_images.shape
    row_unit_padded = np.zeros(row_capacity, dtype=np.int64)
    row_unit_padded[:n_valid_rows] = row_unit[:n_valid_rows]
    order = np.zeros((n_slots, row_capacity), dtype=np.int32)
    active = np.zeros((n_slots, row_capacity), dtype=bool)
    units = np.zeros((n_slots, row_capacity), dtype=np.int32)
    n_blocks = np.zeros(n_slots, dtype=np.int32)
    for slot in range(n_slots):
        slot_images = unit_slot_images[slot]
        has_image = np.zeros(row_capacity, dtype=bool)
        has_image[:n_valid_rows] = slot_images[row_unit[:n_valid_rows]] >= 0
        row_active = has_image.copy()
        row_active[:n_valid_rows] &= row_has_mass
        visit = np.concatenate([np.flatnonzero(row_active), np.flatnonzero(~row_active)])
        order[slot] = visit
        active[slot] = row_active[visit]
        units[slot] = np.where(has_image, row_unit_padded, 0)[visit]
        n_blocks[slot] = -(-int(np.count_nonzero(row_active)) // int(block_rows))
    valid = unit_slot_images >= 0
    safe = np.where(valid, unit_slot_images, np.max(unit_slot_images, axis=1, keepdims=True)).clip(min=0)
    image_of_unit = np.where(valid, np.asarray(layout_image_ids)[np.maximum(unit_slot_images, 0)], 0)
    # Gathered once, at each unit's own translations: taken inside the comprehension, every translation
    # block repeated the whole [S, C_U, T, 2] gather (5 s of each late et09 subtomogram VDAM iteration,
    # py-spy 14993731).
    unit_translations = np.asarray(unit_translations, dtype=np.int64)
    unit_angles = np.asarray(image_angles)[image_of_unit[:, :, None], unit_translations[None, :, :]]
    angles = np.stack([unit_angles[:, :, kept.index] for kept in translation_blocks], axis=1)
    return SlotMstepTables(
        slot=jnp.arange(n_slots, dtype=jnp.int32),
        safe_images=jnp.asarray(safe, dtype=jnp.int32),
        valid_images=jnp.asarray(valid),
        targets=jnp.asarray(np.where(valid, unit_slot_images, int(image_capacity)), dtype=jnp.int32),
        order=jnp.asarray(order),
        active=jnp.asarray(active),
        units=jnp.asarray(units),
        n_blocks=jnp.asarray(n_blocks),
        angles=jnp.asarray(angles, dtype=jnp.float32),
    )


@partial(jax.jit, static_argnames=("slot_spec", "image_shape"), donate_argnums=(0,))
def _tilt_mstep_program(
    mstep,
    slots: SlotMstepTables,
    operands,
    stage_tables,
    wavg_window,
    row_posterior,
    kept_index,
    kept_valid,
    rect_indices,
    exact_positions,
    *,
    slot_spec,
    image_shape,
):
    """A tilt chunk's M-step as one program: a scan over its image slots.

    Each slot backprojects the C_U images that are its units' s-th images (a slot view of the chunk's
    operands, indexed by unit), block by block over its active rows, then adds the image power of each
    translation block once (as the SPA chunk does, ``_add_chunk_wavg_image_power``) and puts its
    per-image partials on those images' chunk rows. ``row_posterior`` ``[C_R, k]`` is each row's posterior at
    its unit's ``k`` translations (:func:`unit_mstep_translations`), and ``slots.angles`` the units' phases
    there; ``kept_index``/``kept_valid`` are the translation blocks' ``[K, T_b]`` positions in those ``k``
    and validity: positions outside a block (and padding) carry zero posterior there.
    """

    from relax.cuda import kernels as em_cuda_kernels
    from relax.sparse_pass2 import resident_pass2 as rp

    row_capacity = int(row_posterior.shape[0])
    block_rows = int(slot_spec.mstep_block_rows)
    fields = {name: getattr(operands, name) for name in _SLOT_VIEW_FIELDS}

    def one_slot(full, xs):
        slot, safe, valid, target, order, active, units, n_blocks, angles = xs
        ordered_posterior = row_posterior[order]
        gathered, window = _slot_view_gather(fields, wavg_window, safe, valid)
        kernel_row_image_ids = jnp.where(active, units, jnp.int32(-1))
        # The block stages' dtypes follow from the gathered operands alone (rp._mstep_block_operand_dtypes).
        slot_carry = rp._initial_mstep_carry(
            full.Ft_y,
            full.Ft_ctf,
            operands._replace(**gathered),
            stage_tables._replace(translation_angles=angles[0]),
            spec=slot_spec,
        )

        # One translation block at a time: a loop iteration's translated tiles are dead before the next
        # block translates its own, so the program holds one block's tiles, as mstep_translation_blocks
        # sizes them. Unrolled blocks let XLA schedule every block's translation first (all tiles live).
        def translation_block(k, slot_carry):
            translated, translated_atomic = _slot_view_translate(
                window, valid, angles[k], rect_indices, exact_positions, image_shape=image_shape
            )
            slot_operands = operands._replace(
                **gathered, raw_translated_wavg_rectangle=translated, raw_translated_wavg_for_atomic=translated_atomic
            )
            # The M-step kernels read each row's image phases by its slot-local image (the unit).
            slot_tables = stage_tables._replace(translation_angles=angles[k])
            block_posterior = jnp.where(
                kept_valid[k][None, :], ordered_posterior[:, kept_index[k]], jnp.zeros((), row_posterior.dtype)
            )
            blocks = rp._MstepBlockInputs(
                row_image_local=units,
                kernel_row_image_ids=kernel_row_image_ids,
                row_posterior=jnp.where(active[:, None], block_posterior, jnp.zeros((), block_posterior.dtype)),
                row_fine_rot=slot * jnp.int32(row_capacity) + order,
                projections=None,
            )

            def block(i, carry):
                return rp._resident_mstep_block_at(
                    i * jnp.int32(block_rows),
                    blocks,
                    slot_operands,
                    slot_tables,
                    carry,
                    spec=slot_spec,
                    cuda_backproject=em_cuda_kernels,
                )

            slot_carry = jax.lax.fori_loop(0, n_blocks, block, slot_carry)
            return rp._add_chunk_wavg_image_power(
                slot_carry,
                slot_operands,
                slot_tables,
                blocks.row_posterior,
                blocks.kernel_row_image_ids,
                spec=slot_spec,
            )

        slot_carry = jax.lax.fori_loop(0, int(kept_index.shape[0]), translation_block, slot_carry)
        placed = _place_slot_partials(
            {name: getattr(full, name) for name in _SLOT_PARTIAL_FIELDS},
            {name: getattr(slot_carry, name) for name in _SLOT_PARTIAL_FIELDS},
            full.noise_shells,
            slot_carry.noise_shells,
            target,
        )
        return full._replace(Ft_y=slot_carry.Ft_y, Ft_ctf=slot_carry.Ft_ctf, **placed), None

    xs = (
        slots.slot,
        slots.safe_images,
        slots.valid_images,
        slots.targets,
        slots.order,
        slots.active,
        slots.units,
        slots.n_blocks,
        slots.angles,
    )
    mstep, _ = jax.lax.scan(one_slot, mstep, xs)
    return mstep


class MstepTranslations(NamedTuple):
    """The translations one tilt chunk's M-step visits: ``index`` into the fine grid, ``valid`` false on padding."""

    index: np.ndarray  # int64 [T_cap]
    valid: np.ndarray  # bool [T_cap]


class UnitMstepTranslations(NamedTuple):
    """Each unit's (particle's) translations the M-step visits: ``index`` into the fine grid, ``valid`` false on padding."""

    index: np.ndarray  # int64 [C_U, k] ascending, padded with the unit's last
    valid: np.ndarray  # bool [C_U, k]


def unit_mstep_translations(
    row_posterior, row_unit, *, unit_capacity: int, n_fine_trans: int, minimum: int = 4
) -> UnitMstepTranslations:
    """Each unit's fine translations with posterior mass in any of its rows, ascending, padded to one power of two.

    ``k`` is the next power of two of the largest unit's count (at least ``minimum``, at most the grid),
    so chunks share programs; a unit's padding repeats its last translation with ``valid`` false (a unit
    without rows takes translation 0). Visiting a row's translations in ascending order without the
    others drops only terms whose posterior is exactly zero.
    """

    n_fine_trans = int(n_fine_trans)
    row_unit = np.asarray(row_unit, dtype=np.int64)
    mass = np.asarray(row_posterior) > 0.0
    unit_mass = np.zeros((int(unit_capacity), n_fine_trans), dtype=bool)
    if row_unit.size:
        order = np.argsort(row_unit, kind="stable")
        sorted_units = row_unit[order]
        starts = np.flatnonzero(np.r_[True, sorted_units[1:] != sorted_units[:-1]])
        unit_mass[sorted_units[starts]] = np.logical_or.reduceat(mass[order], starts, axis=0)
    counts = unit_mass.sum(axis=1)
    largest = int(counts.max(initial=0))
    k = max(int(minimum), 1 << max(largest - 1, 0).bit_length())
    if k >= n_fine_trans:
        k = n_fine_trans
    # Translations with mass first, each set ascending (a stable sort of "no mass").
    index = np.argsort(~unit_mass, axis=1, kind="stable")[:, :k].astype(np.int64)
    valid = np.arange(k)[None, :] < counts[:, None]
    last = np.take_along_axis(index, np.maximum(counts - 1, 0)[:, None], axis=1)
    index = np.where(valid, index, np.where(counts[:, None] > 0, last, 0))
    return UnitMstepTranslations(index=index, valid=valid)


# The flat-row translate-and-sum keeps a block's angle, posterior and active-translation tables in
# 32 KB of shared memory: 16 T bytes at one row per block, the tilt launch (relion_translate_sum.cuh).
TILT_MSTEP_MAX_TRANSLATIONS = 2048


def mstep_translation_blocks(
    kept: MstepTranslations, *, bytes_per_translation: int, tile_budget_bytes: int
) -> tuple[MstepTranslations, ...]:
    """``kept`` in blocks of equal power-of-two size whose translated tiles fit ``tile_budget_bytes``.

    ``bytes_per_translation`` is what one translation adds to a slot's tiles (the slot view's units and
    the per-row gather's block rows, times the Wavg rectangle and exact pixels, complex64). Blocks also
    stay within the translate-and-sum kernel's table limit (``TILT_MSTEP_MAX_TRANSLATIONS``). One block
    when ``kept`` fits; otherwise every block has the largest power of two that fits (at least one),
    the last padded with ``valid`` false.
    """

    size = int(kept.index.size)
    fit = min(max(int(tile_budget_bytes) // max(int(bytes_per_translation), 1), 1), TILT_MSTEP_MAX_TRANSLATIONS)
    if size <= fit:
        return (kept,)
    block = 1 << (int(fit).bit_length() - 1)
    out = []
    for start in range(0, size, block):
        index = kept.index[start : start + block]
        valid = kept.valid[start : start + block]
        pad = block - index.size
        out.append(
            MstepTranslations(
                index=np.concatenate([index, np.full(pad, index[-1])]).astype(np.int64),
                valid=np.concatenate([valid, np.zeros(pad, dtype=bool)]),
            )
        )
    return tuple(out)


def _unit_slot_images(layout: ChunkTiltLayout, *, unit_capacity: int, slot_capacity: int) -> np.ndarray:
    """``[S, C_U]`` chunk image of each unit's s-th image, -1 past its images or on a padded unit."""

    n_images = int(layout.n_valid_images)
    unit_of_image = np.asarray(layout.image_unit_local[:n_images], dtype=np.int64)
    counts = np.bincount(unit_of_image, minlength=int(unit_capacity))[: int(unit_capacity)]
    first = np.concatenate([[0], np.cumsum(counts)[:-1]])
    slots = np.arange(int(slot_capacity))[:, None]
    return np.where(slots < counts[None, :], first[None, :] + slots, -1).astype(np.int64)


# The per-image fields a slot view gathers, and their padding value when not zero.
_SLOT_VIEW_FIELDS = (
    "score_input",
    "corr_img_score",
    "highres_xi2_half",
    "translation_prior",
    "recon_image",
    "recon_weight",
    "noise_image",
    "ctf2_over_nv_recon",
    "direct_ctf_rfloat_recon",
    "image_power_shells",
    "relion_norm_high_shell",
    "scale",
    "group_ids",
    "optics_groups",
    "image_noise_scale",
    "bpref_ctf2_over_nv_recon",
)
_SLOT_VIEW_FILL = {"scale": 1.0, "group_ids": -1}
# The chunk operands a slot view does not take per image: the translated Wavg tiles it rebuilds with
# each image's own phases, and the per-chunk tiles and CC operands the resident M-step never reads.
# Every other per-image operand must be in _SLOT_VIEW_FIELDS, or the slot's units read the chunk's
# first images' values (a premultiplied CTF weight did, 2026-10-03).
_SLOT_CHUNK_ONLY_FIELDS = (
    "shifted_recon",
    "shifted_noise",
    "raw_translated_wavg_rectangle",
    "raw_translated_wavg_for_atomic",
    "translation_sqdist_ang",
    "score_shifted_cc",
    "cc_half_batch_norm",
)


def _slot_view_gather(fields, wavg_window, safe, valid):
    """One image slot's chunk operands, one image per unit (``safe``, ``valid`` false on padding), and its Wavg window.

    Padded units read a valid image with every array zeroed, except ``scale`` (1) and ``group_ids``
    (-1), the chunk gather's padding values (resident_operands._gather_chunk_arrays).
    """

    from relax.sparse_pass2.resident_operands import _gather_rows

    gathered = {
        name: _gather_rows(values, safe, valid, fill=_SLOT_VIEW_FILL.get(name, 0)) for name, values in fields.items()
    }
    return gathered, _gather_rows(wavg_window, safe, valid)


def _slot_view_translate(window, valid, angles, rect_indices, exact_positions, *, image_shape):
    """The slot's translated Wavg rectangle ``[C, T, W]``, each image with its own phases (one launch)."""

    from relax.sparse_pass2.sparse_pass2_wavg import relion_cuda_translate_wavg_norm_window

    translated = relion_cuda_translate_wavg_norm_window(window, angles, rect_indices, image_shape)
    translated = jnp.where(valid[:, None, None], translated, jnp.zeros((), translated.dtype))
    return translated, translated[:, :, exact_positions]


@partial(jax.jit, static_argnames=("image_shape",))
def _slot_view_arrays(fields, wavg_window, safe, valid, angles, rect_indices, exact_positions, *, image_shape):
    """One image slot's chunk operands and its translated Wavg rectangle (:func:`_slot_view_gather`, :func:`_slot_view_translate`)."""

    gathered, window = _slot_view_gather(fields, wavg_window, safe, valid)
    translated, translated_atomic = _slot_view_translate(
        window, valid, angles, rect_indices, exact_positions, image_shape=image_shape
    )
    return gathered, translated, translated_atomic


_SLOT_PARTIAL_FIELDS = ("wavg_triplet_pixels", "a2_per_image", "xa_per_image")


@jax.jit
def _place_slot_partials(full_fields, slot_fields, full_noise, slot_noise, target):
    """Add one slot's per-image M-step partials to their chunk images (``target``; out of range drops) and its noise sums.

    An image is its particle's s-th image in one slot only; it takes a partial from every accumulator
    slot (class) of its particle's rows.
    """

    placed = {
        name: None if full_fields[name] is None else full_fields[name].at[target].add(slot_fields[name], mode="drop")
        for name in _SLOT_PARTIAL_FIELDS
    }
    placed["noise_shells"] = full_noise + slot_noise
    return placed


def row_fine_rot_device(host, row_capacity: int) -> np.ndarray:
    """The chunk rows' global fine rotations, padded rows at 0."""

    out = np.zeros(int(row_capacity), dtype=np.int32)
    values = np.asarray(host["row_fine_rot"], dtype=np.int32)
    out[: values.size] = values[: int(row_capacity)]
    return out


def accumulate_tilt_chunk_terms(
    stats,
    *,
    row_posterior,  # float32 [C_R, T], the particle posterior of each hypothesis row
    row_unit_local,  # int32 [C_R]
    row_coarse_rot,  # int32 [C_R], padded -> >= n_coarse_rot
    row_fine_rot,  # int32 [C_R], the row's fine rotation (without the class offset)
    unit_ids,  # int32 [C_U], -1 padded
    unit_optics_groups,  # int32 [C_U] or None
    unit_translation_sqdist_ang,  # [C_U, T] or None
    image_unit_local,  # int32 [C_I], C_U on padded images
    image_ids,  # int32 [C_I], -1 padded
    operands,  # _ChunkStageOperands over the chunk's images (image_noise_scale set)
    tables,  # _ChunkStageTables
    mstep,  # _ChunkMstepCarry after every slot's blocks (noise and norm partials already scaled)
    posterior,  # resident_pass2._ChunkPosterior over the chunk's particles
    rows,
    spec,
):
    """Fold one tilt chunk into the resident statistics, particle terms once and image terms per image.

    The SPA fold (resident_pass2._accumulate_chunk_image_terms) with RELION's subtomogram rules
    (acc_ml_optimiser_impl.h storeWeightedSums): the offset, sumw, rotation and score/pose terms are the
    particle's; the image power, the noise shells and the norm residual are summed over its images with
    1 / n_images (:3490-3491, :3512-3516); the scale sums are summed over its images as they
    are (:3474-3479). The noise and norm block partials arrive already scaled (image_noise_scale).
    With K>1 the class terms are the particle's (``resident_pass2._fold_class_axis``) and the scale
    sums arrive folded class by class (``mstep.scale_xa_per_image``).
    """

    from relax.sparse_pass2 import resident_pass2 as rp
    from relax.sparse_pass2.resident_statistics import ResidentStatistics, _drop_index, segment_sum_by_image

    config = spec.stats_config
    n_shells = int(config.n_shells)
    n_fine_trans = int(config.n_fine_trans)
    unit_capacity = int(unit_ids.shape[0])
    valid_unit = unit_ids >= 0
    unit_slot = _drop_index(unit_ids, int(config.image_capacity))
    valid_image = image_ids >= 0
    scale = jnp.where(valid_image, jnp.asarray(operands.image_noise_scale, dtype=jnp.float32), jnp.float32(0.0))

    # --- 1/2. offset and support mass: once per particle --------------------
    translation_posterior = segment_sum_by_image(row_posterior, row_unit_local, unit_capacity)
    sigma2_offset = stats.sigma2_offset
    if unit_translation_sqdist_ang is not None:
        sigma2_offset = sigma2_offset + jnp.sum(
            jnp.where(valid_unit[:, None], translation_posterior.astype(jnp.float64), 0.0)
            * jnp.asarray(unit_translation_sqdist_ang, dtype=jnp.float64)
        )
    unit_mass = jnp.where(
        valid_unit, jnp.sum(translation_posterior, axis=1), jnp.zeros((), translation_posterior.dtype)
    )
    n_optics_groups = int(config.n_optics_groups)
    if n_optics_groups == 1:
        sumw = stats.sumw + jnp.sum(unit_mass.astype(jnp.float64))
    else:
        sumw = stats.sumw + jax.ops.segment_sum(
            unit_mass.astype(jnp.float64), unit_optics_groups, num_segments=n_optics_groups
        )

    # --- 3. image power: each image with its particle's mass / n_images (1 / n_images above the cutoff)
    safe_unit = jnp.minimum(image_unit_local, jnp.int32(unit_capacity - 1))
    image_mass = jnp.where(valid_image, unit_mass[safe_unit] * scale, jnp.zeros((), unit_mass.dtype))
    weighted_img_shells, weighted_img_per_image = weighted_image_power_from_shells(
        operands.image_power_shells,
        image_mass,
        jnp.where(valid_image, operands.relion_norm_high_shell, 0.0),
        valid_image,
        norm_unweighted_shell_cutoff=config.norm_unweighted_shell_cutoff,
        include_unweighted_high_shell=config.include_unweighted_high_shell,
        deterministic_norm_reduction=config.deterministic_norm_reduction,
        high_shell_mass=scale,
    )

    # --- 4/7. norm correction: the particle's images, / n_images -----------
    image_norm = jnp.where(valid_image, weighted_img_per_image, 0.0).astype(jnp.float64) + jnp.where(
        valid_image, mstep.a2_per_image - 2.0 * mstep.xa_per_image, 0.0
    ).astype(jnp.float64)
    unit_norm = jax.ops.segment_sum(image_norm, image_unit_local, num_segments=unit_capacity + 1)[:unit_capacity]
    norm_correction = stats.norm_correction.at[unit_slot].add(jnp.where(valid_unit, unit_norm, 0.0), mode="drop")

    # --- 5/6. noise shells with the direct low-shell residual --------------
    direct_residual = operands.image_noise_scale[:, None] * mstep.wavg_triplet_pixels[:, :, 2]
    direct_residual = jnp.where(valid_image[:, None], direct_residual, jnp.float32(0.0))
    if n_optics_groups == 1:
        residual_shells, image_power_shells = rp._replace_low_shell_noise_with_relion_wavg_direct_residual_jnp(
            jnp.asarray(mstep.noise_shells, dtype=jnp.float64),
            weighted_img_shells.astype(jnp.float64),
            direct_residual,
            tables.wavg_shell_indices,
            exclusive_shell_stop=int(config.direct_noise_exclusive_shell_stop),
            shell_count=n_shells,
        )
    else:
        image_optics = jnp.asarray(operands.optics_groups, dtype=jnp.int32)
        residual_per_group, power_per_group = [], []
        for group in range(n_optics_groups):
            in_group = valid_image & (image_optics == group)
            group_img_shells, _ = weighted_image_power_from_shells(
                operands.image_power_shells,
                jnp.where(in_group, image_mass, jnp.zeros((), image_mass.dtype)),
                jnp.where(in_group, operands.relion_norm_high_shell, 0.0),
                in_group,
                norm_unweighted_shell_cutoff=config.norm_unweighted_shell_cutoff,
                include_unweighted_high_shell=config.include_unweighted_high_shell,
                deterministic_norm_reduction=config.deterministic_norm_reduction,
                high_shell_mass=scale,
            )
            group_residual, group_power = rp._replace_low_shell_noise_with_relion_wavg_direct_residual_jnp(
                jnp.asarray(mstep.noise_shells[group], dtype=jnp.float64),
                group_img_shells.astype(jnp.float64),
                jnp.where(in_group[:, None], direct_residual, jnp.float32(0.0)),
                tables.wavg_shell_indices,
                exclusive_shell_stop=int(config.direct_noise_exclusive_shell_stop),
                shell_count=n_shells,
            )
            residual_per_group.append(group_residual)
            power_per_group.append(group_power)
        residual_shells = jnp.stack(residual_per_group)
        image_power_shells = jnp.stack(power_per_group)

    # --- 8. scale sums: every image as it is ------------------------------
    scale_xa, scale_aa = stats.scale_xa, stats.scale_aa
    if config.accumulate_scale:
        if mstep.scale_xa_per_image is not None:
            xa, aa = mstep.scale_xa_per_image, mstep.scale_aa_per_image
        else:
            mask_rect = jnp.asarray(tables.wavg_scale_pixel_mask, dtype=bool).reshape(1, -1)
            zero = jnp.float32(0.0)
            xa = jnp.sum(jnp.where(mask_rect, mstep.wavg_triplet_pixels[:, :, 0], zero).astype(jnp.float64), axis=1)
            aa = jnp.sum(jnp.where(mask_rect, mstep.wavg_triplet_pixels[:, :, 1], zero).astype(jnp.float64), axis=1)
        group_slot = _drop_index(operands.group_ids, int(config.n_scale_groups))
        keep = valid_image & (jnp.asarray(operands.group_ids, dtype=jnp.int32) >= 0)
        scale_xa = scale_xa.at[group_slot].add(jnp.where(keep, xa, 0.0), mode="drop")
        scale_aa = scale_aa.at[group_slot].add(jnp.where(keep, aa, 0.0), mode="drop")

    # --- 9. rotation posterior sums: the particle's -------------------------
    probs_sum_t = jnp.sum(row_posterior, axis=-1)
    coarse_slot = _drop_index(row_coarse_rot, int(config.n_coarse_rot))
    rotation_posterior_sums = stats.rotation_posterior_sums.at[coarse_slot].add(
        probs_sum_t.astype(jnp.float64), mode="drop"
    )

    # --- 10. score and pose fields: the particle's --------------------------
    log_score_offset = (-jnp.asarray(posterior.min_diff2)).astype(jnp.float64)
    best = jnp.asarray(posterior.best_log_score, dtype=jnp.float64)
    finite = jnp.isfinite(best)
    absolute = jnp.asarray(posterior.class_log_z, dtype=jnp.float64) + log_score_offset
    neg_inf = jnp.asarray(-jnp.inf, dtype=jnp.float64)
    best_cell_index = jnp.asarray(posterior.best_cell_index, dtype=jnp.int64)
    best_local_rot = (best_cell_index // jnp.int64(n_fine_trans)).astype(jnp.int32)
    last_row = jnp.int64(int(spec.row_capacity) - 1)
    best_chunk_row = jnp.clip(rows.image_row_start + best_local_rot.astype(jnp.int64), 0, last_row).astype(jnp.int32)
    row_fine_rot = jnp.asarray(row_fine_rot, dtype=jnp.int64)
    n_t = jnp.int64(n_fine_trans)
    best_cell_values = row_fine_rot[best_chunk_row] * n_t + best_cell_index % n_t

    # --- 11. the class axis (K>1): the particle's --------------------------
    classes = stats.classes
    if classes is not None:
        class_cell = jnp.asarray(posterior.classes.best_cell_index, dtype=jnp.int64)
        class_best_row = jnp.clip(rows.classes.segment_row_start + class_cell // n_t, 0, last_row).astype(jnp.int32)
        classes = rp._fold_class_axis(
            classes,
            unit_slot,
            log_score_offset,
            per_class_log_z=posterior.classes.log_z,
            per_class_best_log_score=posterior.classes.best_log_score,
            per_class_best_cell=jnp.where(
                jnp.isfinite(posterior.classes.best_log_score),
                row_fine_rot[class_best_row] * n_t + class_cell % n_t,
                jnp.int64(-1),
            ),
            row_mass=probs_sum_t,
            row_class=rows.classes.row_class,
            n_classes=int(config.n_classes),
        )
    return ResidentStatistics(
        wsum_sigma2_noise=stats.wsum_sigma2_noise + residual_shells,
        wsum_img_power=stats.wsum_img_power + image_power_shells,
        sigma2_offset=sigma2_offset,
        sumw=sumw,
        norm_correction=norm_correction,
        scale_xa=scale_xa,
        scale_aa=scale_aa,
        rotation_posterior_sums=rotation_posterior_sums,
        log_evidence=stats.log_evidence.at[unit_slot].set(jnp.where(finite, absolute, neg_inf), mode="drop"),
        best_log_score=stats.best_log_score.at[unit_slot].set(best + log_score_offset, mode="drop"),
        max_posterior=stats.max_posterior.at[unit_slot].set(
            jnp.asarray(posterior.max_posterior, dtype=stats.max_posterior.dtype), mode="drop"
        ),
        best_cell=stats.best_cell.at[unit_slot].set(best_cell_values, mode="drop"),
        score_log_z=stats.score_log_z.at[unit_slot].set(jnp.where(finite, absolute, neg_inf), mode="drop"),
        best_local_rot=stats.best_local_rot.at[unit_slot].set(best_local_rot, mode="drop"),
        invalid_best_rows=stats.invalid_best_rows,
        classes=classes,
    )


def tilt_capacity_ladders(row_ladder, image_ladder, *, slot_capacity: int, min_rows: int = 256):
    """The row and unit capacity ladders of a tilt pass.

    A tilt chunk projects every row once per image slot (``S * C_R`` projections), so the SPA row ladder
    is divided by the slot count ``S``; rows are floored to a power of two (the M-step blocks must divide
    every row capacity) and to at least ``min_rows``. The SPA image ladder is capped by the translated
    Wavg tile per image; a tilt chunk builds that tile one slot at a time, for its ``C_U`` units'
    images (run_tilt_chunk), so the image ladder is the unit ladder. Every ladder keeps at least one entry.
    """

    slots = max(int(slot_capacity), 1)
    rows = sorted({max(int(min_rows), 1 << max(int(r) // slots, 1).bit_length() - 1) for r in row_ladder})
    units = sorted({max(1, int(b)) for b in image_ladder})
    return tuple(rows), tuple(units)
