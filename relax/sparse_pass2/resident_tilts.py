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

import jax
import jax.numpy as jnp
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


def tilt_slot_rotations(
    layout: ChunkTiltLayout, row_source_eulers, image_left, *, row_spa_matrices=None, dtype=np.float32
) -> np.ndarray:
    """``[S * C_R, 3, 3]`` fine and M-step matrices of every (image slot, row): ``inv(L_i A_r)``.

    RELION builds a tilt image's fine and backprojection matrices on the host with its left matrix
    ``L_i`` (the image's ``Aproj`` times the optics scale; generateEulerMatrices,
    acc_helper_functions_impl.h:248-255; acc_ml_optimiser_impl.h:2522-2547, 4521-4543), so every
    (image, rotation) pair has its own matrix. It passes ``L`` only when it is not the identity
    (``isIdentity``, :1614-1618; ``tomo_particles.relion_left_matrices``); an identity image keeps the
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
    image_shape,
    rect_indices_device,
    exact_positions_device,
):
    """Score, weight and backproject one chunk of particles; return ``(Ft_y_total, Ft_ctf_total, stats)``.

    - Rows are the chunk's particle hypotheses (unit-major, as the candidate tables hold them).
    - Each row is scored by every image of its particle, image slot by image slot, with that image's
      matrix ``inv(L_i A_r)`` (a chunk-local projection cache laid out ``slot * C_R + row``), phases,
      CTF and noise, and the diff2 summed per row (``score_tilt_image_rows``). The minimum, the weights
      and the significance are per particle (RELION's img_id loop, acc_ml_optimiser_impl.h:2282).
    - The M-step visits the slots again: each (image, row) backprojects with the particle's posterior,
      the image's matrix and phases; the noise and norm sums carry 1 / n_images, the scale sums and
      the backprojection do not (docs in PLAN.md, "S4.2 statistics semantics").
    K=1, the global fine pass, the per-stage (non-jitted) path.
    """

    from relax.cuda import kernels as em_cuda_kernels
    from relax.sparse_pass2 import resident_pass2 as rp
    from relax.sparse_pass2.resident_candidates import expand_chunk_mask_jnp, materialize_chunk
    from relax.sparse_pass2.resident_scoring import score_tilt_image_rows

    row_capacity = int(chunk.row_capacity)
    unit_capacity = int(chunk.image_capacity)
    n_valid_rows = int(chunk.n_valid_rows)
    n_slots = int(tilt.slot_capacity)
    image_capacity = tilt_chunk_image_capacity(unit_capacity, n_slots)
    n_fine_trans = int(n_fine_trans)

    rows = rp._make_chunk_row_arrays(tables, chunk, n_fine_trans, place=rp._PLACE_ON_DEVICE)
    host = materialize_chunk(tables, chunk)
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
    score_cache, recon_cache, recon_abs2_cache = project_rotations(slot_matrices)

    # --- per-image operands: the SPA gather, with each image's own phases ---
    image_angles = jnp.asarray(np.asarray(tilt.image_angles, dtype=np.float32)[safe_images])
    # The translated Wavg tile is [images, T, P_rect]; with a subtomogram's 3D grid (thousands of
    # translations) it is built per image slot in the M-step below, for C_U images at a time, never
    # for the chunk's C_U * S images. The chunk-wide gather takes one zero translation.
    recon = rp.gather_resident_chunk_operands(
        resident_operands,
        image_slots,
        translation_angles=np.zeros((1, 2), dtype=np.float32),
        rect_indices=rect_indices_device,
        exact_positions=exact_positions_device,
        image_shape=image_shape,
    )
    noise_scale = np.where(valid_images, np.asarray(tilt.image_noise_scale, dtype=np.float32)[safe_images], 0.0)
    operands = rp._make_chunk_stage_operands(recon, None)._replace(image_noise_scale=jnp.asarray(noise_scale))

    stage_tables = base_tables._replace(
        projection_score_cache=score_cache,
        projection_recon_cache=recon_cache,
        projection_recon_abs2_cache=recon_abs2_cache,
        mstep_grid=jnp.asarray(slot_matrices, dtype=base_tables.mstep_grid.dtype),
        translation_angles=image_angles,
        cache_slot_fine_rot=None,
    )
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
    scored = score_tilt_image_rows(
        lambda slot, _ids: jax.lax.dynamic_slice_in_dim(score_cache, slot * row_capacity, row_capacity, axis=0),
        slot_image_ids,
        rows.row_image_local,
        rows.row_log_prior,
        operands.score_input,
        operands.corr_img_score,
        (
            jnp.zeros((image_capacity,), dtype=jnp.float32)
            if operands.highres_xi2_half is None
            else jnp.asarray(operands.highres_xi2_half, dtype=jnp.float32)
        ),
        jnp.asarray(unit_prior),
        expand_chunk_mask_jnp(rows.row_mask_bits, rows.row_mask_mode, base_tables.fine_translation_parent),
        row_is_valid,
        half_weights=base_tables.half_weights,
        image_translation_angles=image_angles,
        full_to_compact=base_tables.full_to_compact,
        logical_current_size=jnp.asarray(spec.current_size, dtype=jnp.int32),
        unit_capacity=unit_capacity,
    )
    scores_flat = jnp.asarray(scored.scores, dtype=jnp.float32).reshape(-1)
    log_z = em_cuda_kernels.sparse_pass2_segmented_log_z_f64(scores_flat, rows.segment_offsets, rows.n_valid_images)
    (
        log_z_out,
        best_log_score,
        best_cell_index,
        max_posterior,
        _probs,
        _normalized,
        reconstruction_probs,
        _mask,
        _n_significant,
        _sum_weight,
        _threshold,
    ) = em_cuda_kernels.sparse_pass2_segmented_posterior_f32(
        scores_flat,
        rows.segment_offsets,
        rows.n_valid_images,
        log_z,
        jnp.ones((unit_capacity,), dtype=jnp.float32),
        adaptive_fraction=float(spec.adaptive_fraction),
        keep_all=False,
        use_external_sum_weight=False,
    )
    row_posterior = jnp.asarray(reconstruction_probs, dtype=jnp.float32).reshape(row_capacity, n_fine_trans)

    # --- M-step, one pass per image slot ------------------------------------
    # Each slot backprojects the C_U images that are its units' s-th images (a slot view of the chunk's
    # operands, indexed by unit), and its per-image partials land on those images' chunk rows.
    mstep = rp._initial_mstep_carry(Ft_y_total, Ft_ctf_total, operands, stage_tables, spec=spec)
    slot_spec = rp._make_chunk_program_spec(
        row_capacity=row_capacity, image_capacity=unit_capacity, n_fine_trans=n_fine_trans, **spec_kwargs
    )
    block_rows = int(spec.mstep_block_rows)
    row_index = jnp.arange(row_capacity, dtype=jnp.int32)
    unit_slot_images = _unit_slot_images(layout, unit_capacity=unit_capacity, slot_capacity=n_slots)
    row_unit = np.asarray(host["row_image_local"], dtype=np.int64)
    for slot in range(n_slots):
        if not np.any(unit_slot_images[slot] >= 0):
            continue
        slot_images = unit_slot_images[slot]
        slot_angles = jnp.asarray(
            np.asarray(tilt.image_angles, dtype=np.float32)[
                np.where(slot_images >= 0, layout.image_ids[np.maximum(slot_images, 0)], 0)
            ]
        )
        slot_operands = _slot_view(
            operands,
            slot_images,
            angles=slot_angles,
            rect_indices=rect_indices_device,
            exact_positions=exact_positions_device,
            image_shape=image_shape,
        )
        # The M-step kernels read each row's image phases by its slot-local image (the unit).
        slot_tables = stage_tables._replace(translation_angles=slot_angles)
        slot_carry = rp._initial_mstep_carry(mstep.Ft_y, mstep.Ft_ctf, slot_operands, slot_tables, spec=slot_spec)
        row_has_image = np.zeros(row_capacity, dtype=bool)
        row_has_image[:n_valid_rows] = slot_images[row_unit[:n_valid_rows]] >= 0
        has_image = jnp.asarray(row_has_image)
        units = jnp.asarray(
            np.where(row_has_image, np.pad(row_unit, (0, row_capacity - row_unit.size)), 0), dtype=jnp.int32
        )
        blocks = rp._MstepBlockInputs(
            row_image_local=units,
            kernel_row_image_ids=jnp.where(has_image, units, jnp.int32(-1)),
            row_posterior=jnp.where(has_image[:, None], row_posterior, jnp.zeros((), row_posterior.dtype)),
            row_fine_rot=jnp.int32(slot * row_capacity) + row_index,
            projections=None,
        )
        for start in range(0, n_valid_rows, block_rows):
            slot_carry = rp._resident_mstep_block_at(
                rp._device_int32(start),
                blocks,
                slot_operands,
                slot_tables,
                slot_carry,
                spec=slot_spec,
                cuda_backproject=em_cuda_kernels,
            )
        mstep = _fold_slot_carry(mstep, slot_carry, slot_images, image_capacity=image_capacity)

    stats = accumulate_tilt_chunk_terms(
        stats,
        row_posterior=row_posterior,
        row_unit_local=rows.row_image_local,
        row_coarse_rot=jnp.where(
            row_is_valid,
            base_tables.coarse_parent_grid[jnp.asarray(row_fine_rot_device(host, row_capacity))],
            jnp.int32(int(spec.stats_config.n_coarse_rot)),
        ),
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
        min_diff2=scored.min_diff2,
        class_log_z=jnp.asarray(log_z_out, dtype=jnp.float64),
        best_log_score=best_log_score,
        max_posterior=max_posterior,
        best_cell_index=jnp.asarray(best_cell_index, dtype=jnp.int64),
        rows=rows,
        spec=spec,
    )
    return mstep.Ft_y, mstep.Ft_ctf, stats


def _unit_slot_images(layout: ChunkTiltLayout, *, unit_capacity: int, slot_capacity: int) -> np.ndarray:
    """``[S, C_U]`` chunk image of each unit's s-th image, -1 past its images or on a padded unit."""

    n_images = int(layout.n_valid_images)
    unit_of_image = np.asarray(layout.image_unit_local[:n_images], dtype=np.int64)
    counts = np.bincount(unit_of_image, minlength=int(unit_capacity))[: int(unit_capacity)]
    first = np.concatenate([[0], np.cumsum(counts)[:-1]])
    slots = np.arange(int(slot_capacity))[:, None]
    return np.where(slots < counts[None, :], first[None, :] + slots, -1).astype(np.int64)


def _slot_view(operands, slot_images, *, angles, rect_indices, exact_positions, image_shape):
    """The chunk operands of one image slot, one image per unit (``slot_images``, -1 padded).

    Padded units read the first image with every array zeroed, except ``scale`` (1) and ``group_ids``
    (-1), the chunk gather's padding values (resident_operands._gather_chunk_arrays). The slot's
    translated Wavg rectangle is built here with each image's own phases.
    """

    from relax.sparse_pass2.resident_operands import _gather_rows
    from relax.sparse_pass2.sparse_pass2_wavg import _relion_cuda_translate_wavg_norm_images

    slot_images = np.asarray(slot_images, dtype=np.int64)
    valid = jnp.asarray(slot_images >= 0)
    safe = jnp.asarray(np.where(slot_images >= 0, slot_images, int(np.max(slot_images))), dtype=jnp.int32)

    def take(values, fill=0):
        return _gather_rows(values, safe, valid, fill=fill)

    processed = take(operands.processed_image_half)
    translated = jax.lax.map(
        lambda pair: _relion_cuda_translate_wavg_norm_images(pair[0][None], pair[1], rect_indices, image_shape)[0],
        (processed, jnp.asarray(angles, dtype=jnp.float32)),
    )
    translated = jnp.where(valid[:, None, None], translated, jnp.zeros((), translated.dtype))
    return operands._replace(
        score_input=take(operands.score_input),
        corr_img_score=take(operands.corr_img_score),
        highres_xi2_half=take(operands.highres_xi2_half),
        translation_prior=take(operands.translation_prior),
        recon_image=take(operands.recon_image),
        recon_weight=take(operands.recon_weight),
        noise_image=take(operands.noise_image),
        ctf2_over_nv_recon=take(operands.ctf2_over_nv_recon),
        direct_ctf_rfloat_recon=take(operands.direct_ctf_rfloat_recon),
        processed_image_half=processed,
        relion_norm_high_shell=take(operands.relion_norm_high_shell),
        raw_translated_wavg_rectangle=translated,
        raw_translated_wavg_for_atomic=translated[:, :, jnp.asarray(exact_positions, dtype=jnp.int32)],
        scale=take(operands.scale, fill=1.0),
        group_ids=take(operands.group_ids, fill=-1),
        optics_groups=take(operands.optics_groups),
        image_noise_scale=take(operands.image_noise_scale),
    )


def _fold_slot_carry(full, slot, slot_images, *, image_capacity: int):
    """Put one slot's per-image M-step partials on their chunk images; carry its volume and noise sums."""

    slot_images = np.asarray(slot_images, dtype=np.int64)
    # Padded units write past the end and are dropped (an out-of-range positive index, never -1).
    target = jnp.asarray(np.where(slot_images >= 0, slot_images, int(image_capacity)), dtype=jnp.int32)

    def place(full_values, slot_values):
        if full_values is None:
            return None
        return full_values.at[target].set(slot_values, mode="drop")

    return full._replace(
        Ft_y=slot.Ft_y,
        Ft_ctf=slot.Ft_ctf,
        wavg_triplet_pixels=place(full.wavg_triplet_pixels, slot.wavg_triplet_pixels),
        noise_shells=full.noise_shells + slot.noise_shells,
        a2_per_image=place(full.a2_per_image, slot.a2_per_image),
        xa_per_image=place(full.xa_per_image, slot.xa_per_image),
    )


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
    unit_ids,  # int32 [C_U], -1 padded
    unit_optics_groups,  # int32 [C_U] or None
    unit_translation_sqdist_ang,  # [C_U, T] or None
    image_unit_local,  # int32 [C_I], C_U on padded images
    image_ids,  # int32 [C_I], -1 padded
    operands,  # _ChunkStageOperands over the chunk's images (image_noise_scale set)
    tables,  # _ChunkStageTables
    mstep,  # _ChunkMstepCarry after every slot's blocks (noise and norm partials already scaled)
    min_diff2,
    class_log_z,
    best_log_score,
    max_posterior,
    best_cell_index,
    rows,
    spec,
):
    """Fold one tilt chunk into the resident statistics, particle terms once and image terms per image.

    The SPA fold (resident_pass2._accumulate_chunk_image_terms) with RELION's subtomogram rules
    (acc_ml_optimiser_impl.h storeWeightedSums): the offset, sumw, rotation and score/pose terms are the
    particle's; the image power, the noise shells and the norm residual are summed over its images with
    1 / n_images (:4903-4905, :4923-4924, :4945-4950); the scale sums are summed over its images as they
    are (:4907-4912). The noise and norm block partials arrive already scaled (image_noise_scale).
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

    # --- 3. image power: each image with its particle's mass / n_images ----
    safe_unit = jnp.minimum(image_unit_local, jnp.int32(unit_capacity - 1))
    image_mass = jnp.where(valid_image, unit_mass[safe_unit] * scale, jnp.zeros((), unit_mass.dtype))
    weighted_img_shells, weighted_img_per_image = rp._weighted_image_power_shells_and_per_image_core(
        operands.processed_image_half,
        tables.shell_indices_half,
        image_mass,
        jnp.where(valid_image, operands.relion_norm_high_shell * scale, 0.0),
        valid_image,
        shell_count=n_shells,
        norm_unweighted_shell_cutoff=config.norm_unweighted_shell_cutoff,
        include_unweighted_high_shell=config.include_unweighted_high_shell,
        disable_cuda_binning=config.disable_cuda_binning,
        deterministic_norm_reduction=config.deterministic_norm_reduction,
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
            group_img_shells, _ = rp._weighted_image_power_shells_and_per_image_core(
                operands.processed_image_half,
                tables.shell_indices_half,
                jnp.where(in_group, image_mass, jnp.zeros((), image_mass.dtype)),
                jnp.where(in_group, operands.relion_norm_high_shell * scale, 0.0),
                in_group,
                shell_count=n_shells,
                norm_unweighted_shell_cutoff=config.norm_unweighted_shell_cutoff,
                include_unweighted_high_shell=config.include_unweighted_high_shell,
                disable_cuda_binning=config.disable_cuda_binning,
                deterministic_norm_reduction=config.deterministic_norm_reduction,
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
    log_score_offset = (-jnp.asarray(min_diff2)).astype(jnp.float64)
    best = jnp.asarray(best_log_score, dtype=jnp.float64)
    finite = jnp.isfinite(best)
    absolute = jnp.asarray(class_log_z, dtype=jnp.float64) + log_score_offset
    neg_inf = jnp.asarray(-jnp.inf, dtype=jnp.float64)
    best_local_rot = (best_cell_index // jnp.int64(n_fine_trans)).astype(jnp.int32)
    best_chunk_row = jnp.clip(
        rows.image_row_start + best_local_rot.astype(jnp.int64), 0, jnp.int64(int(spec.row_capacity) - 1)
    ).astype(jnp.int32)
    best_fine_rot = jnp.asarray(rows.row_fine_rot, dtype=jnp.int64)[best_chunk_row]
    best_cell_values = best_fine_rot * jnp.int64(n_fine_trans) + best_cell_index % jnp.int64(n_fine_trans)
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
            jnp.asarray(max_posterior, dtype=stats.max_posterior.dtype), mode="drop"
        ),
        best_cell=stats.best_cell.at[unit_slot].set(best_cell_values, mode="drop"),
        score_log_z=stats.score_log_z.at[unit_slot].set(jnp.where(finite, absolute, neg_inf), mode="drop"),
        best_local_rot=stats.best_local_rot.at[unit_slot].set(best_local_rot, mode="drop"),
        invalid_best_rows=stats.invalid_best_rows,
        classes=stats.classes,
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
