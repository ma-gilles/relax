"""Tilt images of the resident pass 2's units (cryo-ET, S4.2).

The posterior unit of the resident pass 2 is a particle; for subtomograms it owns several
tilt images (docs/development/resident_segments.md, "The scoring side"). A capacity chunk
holds a contiguous range of units, so it also holds the contiguous range of their images.
:func:`relax.fine_pass.resident_tilt_layout.chunk_tilt_layout` enumerates, for each image slot ``s``, which chunk image scores
each hypothesis row: the ``s``-th visible tilt of the row's unit, in RELION's ``img_id``
order (frame order, ``tomo_input.relion_image_geometry``). The scorer and the M-step loop
over ``s`` and add each image's contribution in that order, as RELION's ``img_id`` loops
do (acc_ml_optimiser_impl.h:1490, :2959).

Single-particle data are the one-image case: ``unit_image_offsets = arange(n + 1)``.

The chunk's layout is ``resident_tilt_layout``'s, its slot M-step program ``resident_tilt_mstep``'s; this
module runs a chunk (:func:`run_tilt_chunk`) and sizes it.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from relax.fine_pass.projection_blocks import project_rows_by_class
from relax.fine_pass.resident_tilt_layout import ChunkTiltLayout, chunk_tilt_layout, tilt_slot_rotations
from relax.fine_pass.resident_tilt_mstep import (
    MstepTranslations,
    mstep_translation_blocks,
    slot_mstep_tables,
    slot_tables_block,
    tilt_mstep_program,
    tilt_mstep_tile_bytes,
    tilt_mstep_units,
    unit_mstep_translations,
)
from relax.fine_pass.wavg import (
    replace_low_shell_noise_with_relion_wavg_direct_residual_jnp,
    weighted_image_power_from_shells,
)
from relax.runtime.shape_buckets import pow2_floor

logger = logging.getLogger(__name__)


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
    projections made once) when they fit or the budget is unknown; else the slots split evenly into as few
    blocks as hold at most the count that fits (at least one slot each). The block count, not the free bytes,
    sets the block size, so the passes of a run keep the same slot blocks and their programs (the largest
    count that fits gave the M-step program a new slot axis in nearly every pass: 30 + 11, 31 + 10, 32 + 9).
    """

    need = int(max_rows) * int(slot_row_bytes)
    if budget_bytes is None or need <= 0 or need * slot_capacity <= int(budget_bytes) // 2:
        return slot_capacity
    fit = int(min(slot_capacity, max(1, (int(budget_bytes) // 2) // need)))
    n_blocks = -(-int(slot_capacity) // fit)
    return -(-int(slot_capacity) // n_blocks)


def tilt_lone_row_bytes(n_fine_trans: int, *, slot_capacity: int, n_accumulators: int = 1) -> int:
    """Device bytes each row of a row-blocked lone particle keeps beyond its block (:func:`run_tilt_chunk`).

    A particle's chunk holds ``rows x image slots x pixels`` projection values at once (each row is projected
    once per tilt image); when that does not fit for a particle run alone, its rows are projected, scored and
    backprojected ``lone_block_rows`` at a time, and only these stay for all of its rows:

    - its ``T`` posterior cells, as the SPA lone chunk counts them (``resident_chunk_memory.lone_chunk_row_bytes``),
      and its posterior at the particle's own translations (at most ``T`` float32 cells) per accumulator;
    - per image slot, the row's float32 3x3 matrix ``inv(L_i A_r)`` and its int32 chunk image.
    """

    from relax.fine_pass.resident_chunk_memory import lone_chunk_row_bytes

    return (
        lone_chunk_row_bytes(int(n_fine_trans))
        + int(n_fine_trans) * 4 * max(int(n_accumulators), 1)
        + int(slot_capacity) * (36 + 4)
    )


class TiltMstepCensus:
    """A pass's tilt M-step translation blocks, logged once per pass (relax#27).

    Each chunk adds its particles' ``k`` (:func:`unit_mstep_translations`), the translations one block
    holds and its block count (:func:`mstep_translation_blocks`). More than one block is where each
    row's partials are merged before the adjoint; the line records whether a run reached it.
    """

    def __init__(self):
        self.chunks: list[tuple[int, int, int]] = []

    def add(self, k: int, block: int, n_blocks: int) -> None:
        self.chunks.append((int(k), int(block), int(n_blocks)))

    def log(self) -> None:
        if not self.chunks:
            return
        k, block, _ = zip(*self.chunks)
        multi = sum(1 for _, _, n in self.chunks if n > 1)
        logger.info(
            "Tilt M-step translation blocks: largest k %d, smallest block %d translations, multi-block chunks %d/%d",
            max(k),
            min(block),
            multi,
            len(self.chunks),
        )


class TiltChunkHost(NamedTuple):
    """A tilt chunk's host-only preparation (:func:`prepare_tilt_chunk_host`): its materialized rows, its image layout
    and every (image slot, row) matrix ``inv(L_i A_r)`` (``[S * C_R, 3, 3]``)."""

    host: dict
    layout: ChunkTiltLayout
    slot_matrices: np.ndarray


def prepare_tilt_chunk_host(chunk, *, tables, tilt: TiltPassInputs) -> TiltChunkHost:
    """The host NumPy part of :func:`run_tilt_chunk` that depends on the plan alone, not on the device.

    The chunk loop runs it for the next chunk on a host thread while the current chunk's programs run
    (resident_pass2): the slot matrices are float64 host arithmetic per (image, row), 1.5 s of host time per
    hn_k2_s0150 VDAM iteration (relax#115).
    """

    from relax.fine_pass import resident_chunk_programs

    _block_tables, host, _local_chunk = resident_chunk_programs.chunk_host_rows(tables, chunk)
    n_slots = int(tilt.slot_capacity)
    layout = chunk_tilt_layout(
        tilt.unit_image_offsets,
        unit_start=int(chunk.image_start),
        n_valid_units=int(chunk.n_valid_images),
        row_unit_local=np.asarray(host["row_image_local"]),
        n_valid_rows=chunk.n_valid_rows,
        image_capacity=tilt_chunk_image_capacity(int(chunk.image_capacity), n_slots),
        slot_capacity=n_slots,
    )
    row_fine_rot = np.asarray(host["row_fine_rot"], dtype=np.int64)
    row_eulers = (
        np.zeros((row_fine_rot.size, 3))
        if tilt.fine_source_eulers is None
        else np.asarray(tilt.fine_source_eulers, dtype=np.float64)[row_fine_rot]
    )
    slot_matrices = tilt_slot_rotations(
        layout, row_eulers, tilt.image_left, row_spa_matrices=np.asarray(tilt.fine_rotations)[row_fine_rot]
    )
    return TiltChunkHost(host=host, layout=layout, slot_matrices=slot_matrices)


def run_tilt_chunk(
    chunk,
    *,
    tables,
    tilt: TiltPassInputs,
    resident_operands,
    project_rotations,
    base_tables,
    n_fine_trans: int,
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
    mstep_census: TiltMstepCensus | None = None,
    row_partial=None,
    lone_block_rows: int | None = None,
    lean_lone: bool = False,
    prepared: TiltChunkHost | None = None,
):
    """Score, weight and backproject one chunk of particles; return ``(Ft_y_total, Ft_ctf_total, stats)``.

    ``prepared`` is the chunk's :func:`prepare_tilt_chunk_host`, built ahead by the chunk loop, or None (built here).

    ``row_partial`` is the pass's partial accumulator pair (``half_volume_mstep.RowPartial``), into which
    the M-step program scatters its rows when the pass has one.

    ``Ft_y_total``/``Ft_ctf_total`` are the pass's BPref pairs, one per accumulator slot (class +
    K * group, resident_pass2); ``resident_operands`` hold the half's images from
    ``operand_image_start`` on: the whole half (0), or only this chunk's images when the half's
    operands do not fit (resident_pass2).

    - Rows are the chunk's particle hypotheses (unit-major, as the candidate tables hold them).
    - Each row is scored by every image of its particle, image slot by image slot, with that image's
      matrix ``inv(L_i A_r)`` (a chunk-local projection cache laid out ``slot * C_R + row``), phases,
      CTF and noise, and the diff2 summed per row (``tilt_image_rows_raw_diff2``). The minimum, the weights
      and the significance are per particle (RELION's img_id loop, acc_ml_optimiser_impl.h:2282).
    - The M-step visits the slots again: each (image, row) backprojects with the particle's posterior,
      the image's matrix and phases; the noise and norm sums carry 1 / n_images, the scale sums and
      the backprojection do not (docs in PLAN.md, "S4.2 statistics semantics").
    - The M-step visits each particle's own translations with posterior mass (:func:`unit_mstep_translations`):
      a row's posterior and its image's phases are taken at its particle's ``k`` translations, so the
      translated tiles cover only those (the dropped terms carry exactly zero posterior).
    - Memory: the translated Wavg tiles of one slot are ``[C_U, T_b, P_rect]`` (the slot view), for blocks
      of ``T_b`` of the particles' ``k`` translations, sized to fit ``tile_budget_bytes``
      (:func:`tilt_mstep_tile_bytes`, :func:`mstep_translation_blocks`); ``mstep_census`` records each
      chunk's ``k`` and blocks (:class:`TiltMstepCensus`). Every translation block repeats the M-step
      block bodies of the slot's rows, so a block holds as many translations as fit (relax#82).
    - K>1 classes (RELION subtomogram Class3D): a particle's rows are class-major (the candidate
      tables' class layout), each (slot, row) projects from its class's reference, the posterior
      segment is the particle's over every class, and each accumulator slot's rows backproject into
      its own BPref pair, their scale sums under the class's mask (``fold_class_scale_sums``).
    - ``slot_block`` (:func:`tilt_projection_slot_block`): when the (slot, row) projections of every slot
      do not fit, they are made a block of slots at a time, for the scores (the running diff2 carried
      across blocks, so its slot order is kept) and again for the M-step.
    - ``lone_block_rows`` (resident_pass2's plan, :func:`tilt_lone_row_bytes`): a particle run alone whose
      rows' projections do not fit even one slot at a time (a flat coarse posterior keeps every fine
      rotation: 294,912 rows at HEALPix 3) is projected and scored that many rows at a time, keeping only
      the rows' ``[C_R, T]`` diff2; its posterior is formed over all of its rows as before, and the M-step
      projects and backprojects its rows with posterior mass, that many at a time. Each row's diff2 keeps its
      slot order and the posterior is the one-call chunk's; only the grouping of the projector calls and of
      the M-step blocks changes. Its M-step tables and translated tiles hold its one unit; any other chunk's
      hold its valid units, rounded up to a power of two (:func:`tilt_mstep_units`).
    - ``lean_lone`` (resident_pass2's plan where the particle's full posterior arrays do not fit,
      :func:`~relax.fine_pass.resident_tilt_lone.lean_lone_row_bytes`): such a particle's scores are built in
      place a block of rows at a time and its posterior comes from the reconstruction-only handler, the same
      values in 13 instead of 33 bytes per cell (:mod:`relax.fine_pass.resident_tilt_lone`, relax#118).
    The per-stage (non-jitted) path.
    """

    from relax.fine_pass import (
        resident_chunk_posterior,
        resident_chunk_programs,
        resident_chunk_records,
        resident_chunk_stages,
        resident_mstep_blocks,
        resident_tilt_lone,
    )
    from relax.fine_pass.policy import ResidentConfigurationUnsupported
    from relax.fine_pass.resident_candidates import expand_chunk_mask_jnp
    from relax.fine_pass.resident_operands import gather_resident_chunk_operands
    from relax.fine_pass.resident_scoring import tilt_image_rows_raw_diff2, tilt_rows_scores_from_raw

    lone = lone_block_rows is not None
    if lean_lone and not lone:
        raise ValueError("the lean form is a lone particle's: lean_lone needs lone_block_rows")
    if lone and int(chunk.n_valid_images) != 1:
        raise ValueError(f"a chunk run in row blocks holds one particle, not {int(chunk.n_valid_images)}")
    if lone and (bool(spec_kwargs.get("firstiter_cc")) or int(chunk.row_capacity) % int(lone_block_rows)):
        raise ResidentConfigurationUnsupported(
            "a particle past the largest row class runs in row blocks that divide its rows, which the "
            "first-iteration cross-correlation pass does not implement"
        )

    unit_capacity = int(chunk.image_capacity)
    n_slots = int(tilt.slot_capacity)
    image_capacity = tilt_chunk_image_capacity(unit_capacity, n_slots)

    n_accumulators = len(Ft_y_total)
    if n_accumulators != int(tables.n_slots):
        raise ValueError(f"{n_accumulators} BPref pairs for the tables' {int(tables.n_slots)} accumulator slots")
    rows = resident_chunk_programs.make_chunk_row_arrays(tables, chunk, n_fine_trans, place=resident_chunk_records.PLACE_ON_DEVICE, n_fine_rot=n_fine_rot)
    if prepared is None:
        prepared = prepare_tilt_chunk_host(chunk, tables=tables, tilt=tilt)
    host, layout, slot_matrices = prepared
    valid_images = layout.image_ids >= 0
    image_slots = np.where(valid_images, layout.image_ids, -1).astype(np.int32)
    operand_slots = np.where(valid_images, layout.image_ids - int(operand_image_start), -1).astype(np.int32)
    safe_images = np.where(valid_images, layout.image_ids, 0)

    # --- projections of every (slot, row): RELION's host inv(L_i A_r) (prepared.slot_matrices) ------
    row_class = np.zeros(chunk.row_capacity, dtype=np.int64)
    if tables.n_classes > 1:
        row_class[:chunk.n_valid_rows] = np.asarray(host["row_class"], dtype=np.int64)[:chunk.n_valid_rows]
    entry_class = np.tile(row_class, n_slots)
    slot_block = n_slots if slot_block is None else max(1, min(int(slot_block), n_slots))
    slot_blocks = [(start, min(start + slot_block, n_slots)) for start in range(0, n_slots, slot_block)]

    def project_block(start, stop):
        entries = slice(start * chunk.row_capacity, stop * chunk.row_capacity)
        return project_slot_rows(project_rotations, slot_matrices[entries], entry_class[entries], n_classes=tables.n_classes)

    def block_entries(start, stop, row_ids):
        # Cache entries of slots start:stop for the chunk rows ``row_ids``, slot-major: ``slot * n + row``.
        return (np.arange(start, stop)[:, None] * chunk.row_capacity + np.asarray(row_ids)[None, :]).reshape(-1)

    def project_rows(start, stop, row_ids):
        entries = block_entries(start, stop, row_ids)
        return project_slot_rows(project_rotations, slot_matrices[entries], entry_class[entries], n_classes=tables.n_classes)

    # One block: the chunk's projections are made once, for the scores and the M-step.
    whole = project_block(0, n_slots) if len(slot_blocks) == 1 and not lone else None
    score_cache, recon_cache, recon_abs2_cache = (None, None, None) if whole is None else whole

    # --- per-image operands: the SPA gather, with each image's own phases ---
    image_angles = jnp.asarray(np.asarray(tilt.image_angles, dtype=np.float32)[safe_images])
    # The translated Wavg tile is [images, T, P_rect]; with a subtomogram's 3D grid (thousands of
    # translations) it is built per image slot in the M-step below, for C_U images at a time, never
    # for the chunk's C_U * S images. The chunk-wide gather takes one zero translation.
    recon = gather_resident_chunk_operands(
        resident_operands,
        operand_slots,
        translation_angles=np.zeros((1, 2), dtype=np.float32),
        rect_indices=rect_indices_device,
        exact_positions=exact_positions_device,
        image_shape=image_shape,
    )
    # Each chunk image's unshifted Wavg window (the resident operands keep only that window of the
    # image); the slot views translate it with each image's own phases.
    from relax.fine_pass.resident_operands import gather_rows

    chunk_wavg_window = gather_rows(
        resident_operands.wavg_image_rect,
        jnp.asarray(np.maximum(operand_slots, 0), dtype=jnp.int32),
        jnp.asarray(operand_slots >= 0),
    )
    noise_scale = np.where(valid_images, np.asarray(tilt.image_noise_scale, dtype=np.float32)[safe_images], 0.0)
    operands = resident_chunk_programs.make_chunk_stage_operands(recon, None)._replace(image_noise_scale=jnp.asarray(noise_scale))
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
                slot_matrices[start * chunk.row_capacity : stop * chunk.row_capacity],
                dtype=base_tables.mstep_grid.dtype,
            ),
            translation_angles=image_angles,
            cache_slot_fine_rot=None,
        )

    stage_tables = block_tables(0, n_slots, (score_cache, recon_cache, recon_abs2_cache))
    spec = resident_chunk_programs.make_chunk_program_spec(
        row_capacity=chunk.row_capacity, image_capacity=image_capacity, n_fine_trans=n_fine_trans, **spec_kwargs
    )

    # --- scoring and the per-particle posterior -----------------------------
    row_is_valid = jnp.arange(chunk.row_capacity, dtype=jnp.int32) < rows.n_valid_rows
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
            row_capacity=chunk.row_capacity,
            unit_capacity=unit_capacity,
            block_rows=int(spec.mstep_block_rows),
            tile_budget_bytes=int(tile_budget_bytes),
        )
        posterior = resident_chunk_posterior.winner_take_all_posterior(
            scored,
            rows,
            image_capacity=unit_capacity,
            row_is_valid=row_is_valid,
            kernel_row_image_ids=kernel_row_image_ids,
        )
    else:
        initial_diff2 = (
            jnp.zeros((image_capacity,), dtype=jnp.float32)
            if operands.highres_xi2_half is None
            else jnp.asarray(operands.highres_xi2_half, dtype=jnp.float32)
        )

        def raw_diff2_of_rows(rows_of, project):
            # The rows' diff2 summed over every slot in slot order, their projections made a slot block at a time.
            running = None
            for start, stop in slot_blocks:
                block_score_cache = score_cache if whole is not None else project(start, stop)[0]
                running = tilt_image_rows_raw_diff2(
                    block_score_cache,
                    slot_image_ids[start:stop, rows_of],
                    operands.score_input,
                    operands.corr_img_score,
                    initial_diff2,
                    candidate_mask[rows_of],
                    row_is_valid[rows_of],
                    half_weights=base_tables.half_weights,
                    image_translation_angles=image_angles,
                    full_to_compact=base_tables.full_to_compact,
                    logical_current_size=jnp.asarray(spec.current_size, dtype=jnp.int32),
                    running=running,
                )
                del block_score_cache
            return running

        if lean_lone:
            scored = resident_tilt_lone.lean_lone_scores(
                lambda row_start: raw_diff2_of_rows(
                    slice(row_start, row_start + int(lone_block_rows)),
                    lambda start, stop, row_ids=np.arange(row_start, row_start + int(lone_block_rows)): project_rows(
                        start, stop, row_ids
                    ),
                ),
                row_capacity=chunk.row_capacity,
                block_rows=int(lone_block_rows),
                n_valid_rows=chunk.n_valid_rows,
                n_fine_trans=n_fine_trans,
                row_unit_local=rows.row_image_local,
                row_log_prior=rows.row_log_prior,
                unit_translation_prior=jnp.asarray(unit_prior),
                candidate_mask=candidate_mask,
                row_is_valid=row_is_valid,
                unit_capacity=unit_capacity,
            )
        elif not lone:
            raw_sum = raw_diff2_of_rows(slice(None), project_block)
        else:
            raw_blocks = []
            for row_start in range(0, chunk.row_capacity, int(lone_block_rows)):
                if row_start >= chunk.n_valid_rows:
                    # Padding rows are never scored: their running sum is the zeros it starts from.
                    raw_blocks.append(jnp.zeros((int(lone_block_rows), n_fine_trans), dtype=jnp.float32))
                    continue
                row_ids = np.arange(row_start, row_start + int(lone_block_rows))
                raw_blocks.append(
                    raw_diff2_of_rows(
                        slice(row_start, row_start + int(lone_block_rows)),
                        lambda start, stop, row_ids=row_ids: project_rows(start, stop, row_ids),
                    )
                )
            raw_sum = jnp.concatenate(raw_blocks, axis=0)
            del raw_blocks
        if not lean_lone:
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
        posterior = resident_chunk_posterior.chunk_posterior_from_scores(
            scored,
            rows,
            stage_tables,
            spec=resident_chunk_programs.make_chunk_program_spec(
                row_capacity=chunk.row_capacity, image_capacity=unit_capacity, n_fine_trans=n_fine_trans, **spec_kwargs
            ),
            row_is_valid=row_is_valid,
            kernel_row_image_ids=kernel_row_image_ids,
            reconstruction_only=lean_lone,
        )
        if lean_lone:
            del scored  # the particle's scores: the posterior is all the M-step reads
    row_posterior = posterior.row_posterior

    # --- M-step, one pass per image slot ------------------------------------
    # Each slot backprojects the C_U images that are its units' s-th images (a slot view of the chunk's
    # operands, indexed by unit), and its per-image partials land on those images' chunk rows.
    # The carry's dtypes follow the projections' (one slot's when they are made per block).
    mstep = resident_mstep_blocks.initial_mstep_carry(
        Ft_y_total[0],
        Ft_ctf_total[0],
        operands,
        stage_tables
        if whole is not None
        else block_tables(0, 1, project_rows(0, 1, np.arange(int(lone_block_rows))) if lone else project_block(0, 1)),
        spec=spec,
    )
    # Only each particle's translations with posterior mass enter the M-step: a subtomogram's 3D grid
    # has thousands, a particle a handful of significant samples, and the Wavg rectangle is
    # [images, T, P_rect] (gathered per row, [rows, T, P_rect]). Dropped translations carry exactly
    # zero posterior.
    # A lone particle's M-step block is at most its row block (a lean particle's can be below the row classes).
    block_rows = min(int(spec.mstep_block_rows), int(lone_block_rows)) if lone else int(spec.mstep_block_rows)
    host_posterior = np.asarray(row_posterior)[:chunk.n_valid_rows]  # sliced on the host: no program per row count
    # Rows without posterior mass (RELION's non-significant samples) add exact zeros; the M-step
    # blocks visit only the rows with mass (row_has_mass), gathered to the front of each slot's order.
    row_has_mass = np.any(host_posterior > 0.0, axis=1)
    host_row_unit = np.asarray(host["row_image_local"], dtype=np.int64)
    unit_translations = unit_mstep_translations(
        host_posterior, host_row_unit[:chunk.n_valid_rows], unit_capacity=unit_capacity, n_fine_trans=n_fine_trans
    )
    n_unit_trans = int(unit_translations.index.shape[1])
    # The rows' posterior at their particle's translations, [C_R, k]; padded entries are zero.
    row_unit_device = np.zeros(chunk.row_capacity, dtype=np.int64)
    row_unit_device[:chunk.n_valid_rows] = host_row_unit[:chunk.n_valid_rows]
    unit_posterior = jnp.where(
        jnp.asarray(unit_translations.valid[row_unit_device]),
        jnp.take_along_axis(row_posterior, jnp.asarray(unit_translations.index[row_unit_device], dtype=jnp.int32), axis=1),
        jnp.zeros((), row_posterior.dtype),
    )
    unit_kept = MstepTranslations(index=np.arange(n_unit_trans, dtype=np.int64), valid=np.ones(n_unit_trans, dtype=bool))
    # The M-step's units: the chunk's valid ones (tilt_mstep_units), or the one particle of a chunk run in row
    # blocks (unit 0).
    mstep_units = 1 if lone else tilt_mstep_units(int(chunk.n_valid_images), unit_capacity=unit_capacity)
    bytes_per_translation = tilt_mstep_tile_bytes(
        mstep_units,
        n_rect_pixels=int(rect_indices_device.shape[0]),
        n_exact_pixels=int(exact_positions_device.shape[0]),
    )
    translation_blocks = mstep_translation_blocks(
        unit_kept, bytes_per_translation=bytes_per_translation, tile_budget_bytes=int(tile_budget_bytes)
    )
    # Several blocks merge each row's partials in a [C_R, recon pixels] pair before its one adjoint
    # (relax#27); the pair's bytes come out of the tiles' budget.
    merge_row_partials = len(translation_blocks) > 1
    # The M-step program's rows: the chunk's, or one block of a lone particle's rows with posterior mass.
    mstep_row_capacity = int(lone_block_rows) if lone else int(chunk.row_capacity)
    if merge_row_partials:
        row_sum_bytes = (
            mstep_row_capacity
            * int(spec.n_recon_pixels)
            * (np.dtype(Ft_y_total[0].dtype).itemsize + np.dtype(Ft_ctf_total[0].dtype).itemsize)
        )
        translation_blocks = mstep_translation_blocks(
            unit_kept,
            bytes_per_translation=bytes_per_translation,
            tile_budget_bytes=max(int(tile_budget_bytes) - row_sum_bytes, 0),
        )
    if mstep_census is not None:
        mstep_census.add(n_unit_trans, translation_blocks[0].index.size, len(translation_blocks))
    slot_spec = resident_chunk_programs.make_chunk_program_spec(
        row_capacity=mstep_row_capacity,
        image_capacity=mstep_units,
        n_fine_trans=int(translation_blocks[0].index.size),
        presum_adjoint=merge_row_partials,
        **dict(spec_kwargs, mstep_block_rows=block_rows),
    )
    unit_slot_images = _unit_slot_images(layout, unit_capacity=unit_capacity, slot_capacity=n_slots)[:, :mstep_units]
    kept_index = jnp.asarray(np.stack([kept.index for kept in translation_blocks]), dtype=jnp.int32)
    kept_valid = jnp.asarray(np.stack([kept.valid for kept in translation_blocks]))
    # Each accumulator slot (class + K * group) backprojects its own rows into its BPref pair; the
    # per-image partials add over the slots (_place_slot_partials), the scale sums under each class's
    # mask (RELION keeps XA/AA per class, acc_ml_optimiser_impl.h:4893-4912).
    row_accumulator = np.zeros(chunk.row_capacity, dtype=np.int64)
    if n_accumulators > 1:
        row_accumulator[:chunk.n_valid_rows] = np.asarray(host["row_slot"], dtype=np.int64)[:chunk.n_valid_rows]
    in_accumulator = [row_accumulator == accumulator for accumulator in range(n_accumulators)]
    accumulator_posteriors = [
        unit_posterior
        if n_accumulators == 1
        else jnp.where(jnp.asarray(in_accumulator[accumulator])[:, None], unit_posterior, jnp.zeros((), unit_posterior.dtype))
        for accumulator in range(n_accumulators)
    ]

    def slot_tables(accumulator, row_ids=None):
        # The chunk's rows, or a lone particle's block ``row_ids`` (chunk rows, padded past its valid ones).
        n_rows = chunk.n_valid_rows if row_ids is None else int(np.count_nonzero(row_ids >= 0))
        rows_of = slice(None) if row_ids is None else np.maximum(row_ids, 0)
        return slot_mstep_tables(
            unit_slot_images,
            row_unit=host_row_unit[rows_of],
            row_has_mass=(row_has_mass_padded & in_accumulator[accumulator])[rows_of][:n_rows],
            n_valid_rows=n_rows,
            row_capacity=mstep_row_capacity,
            block_rows=block_rows,
            image_angles=np.asarray(tilt.image_angles, dtype=np.float32),
            layout_image_ids=np.asarray(layout.image_ids),
            unit_translations=unit_translations.index[:mstep_units],
            translation_blocks=translation_blocks,
            image_capacity=image_capacity,
        )

    row_has_mass_padded = np.zeros(chunk.row_capacity, dtype=bool)
    row_has_mass_padded[:chunk.n_valid_rows] = row_has_mass
    Ft_y_out, Ft_ctf_out = list(Ft_y_total), list(Ft_ctf_total)

    def mstep_slot_block(mstep, tables_b, slots_of, posteriors):
        # One slot block's M-step for every accumulator, in accumulator order.
        for accumulator in range(n_accumulators):
            mstep, *partial_pair = tilt_mstep_program(
                (
                    mstep._replace(Ft_y=Ft_y_out[accumulator], Ft_ctf=Ft_ctf_out[accumulator]),
                    *(() if row_partial is None else ((row_partial.y, row_partial.ctf),)),
                ),
                slots_of(accumulator),
                operands,
                tables_b,
                chunk_wavg_window,
                posteriors[accumulator],
                kept_index,
                kept_valid,
                rect_indices_device,
                jnp.asarray(exact_positions_device, dtype=jnp.int32),
                slot_spec=slot_spec,
                image_shape=tuple(int(n) for n in image_shape),
            )
            if partial_pair:
                row_partial.y, row_partial.ctf = partial_pair[0]
            if mstep.scale_xa_per_image is not None:
                mstep = resident_chunk_stages.fold_class_scale_sums(
                    mstep, stage_tables.wavg_scale_pixel_mask, resident_chunk_records.device_int32(accumulator % tables.n_classes)
                )
            Ft_y_out[accumulator], Ft_ctf_out[accumulator] = mstep.Ft_y, mstep.Ft_ctf
        return mstep

    if not lone:
        accumulator_slots = [slot_tables(accumulator) for accumulator in range(n_accumulators)]
        # Each accumulator's slots are visited in slot order (block by block); an image is one slot's, so its
        # per-image partials add over the accumulators in the same order with or without blocks.
        for start, stop in slot_blocks:
            tables_b = stage_tables if whole is not None else block_tables(start, stop, project_block(start, stop))
            mstep = mstep_slot_block(
                mstep,
                tables_b,
                lambda accumulator: accumulator_slots[accumulator]
                if whole is not None
                else slot_tables_block(accumulator_slots[accumulator], start, stop),
                accumulator_posteriors,
            )
            del tables_b
    else:
        # Only the rows with posterior mass are projected again, a block of them at a time: each block is the
        # M-step of a chunk holding those rows alone (its tables, posterior and projections are theirs).
        mass_rows = np.flatnonzero(row_has_mass_padded)
        for block_start in range(0, mass_rows.size, mstep_row_capacity):
            row_ids = np.full(mstep_row_capacity, -1, dtype=np.int64)
            block = mass_rows[block_start : block_start + mstep_row_capacity]
            row_ids[: block.size] = block
            safe_rows = np.maximum(row_ids, 0)
            in_block = jnp.asarray(row_ids >= 0)[:, None]
            block_posteriors = [
                jnp.where(in_block, posterior_of[jnp.asarray(safe_rows, dtype=jnp.int32)], jnp.zeros((), posterior_of.dtype))
                for posterior_of in accumulator_posteriors
            ]
            block_slots = [slot_tables(accumulator, row_ids) for accumulator in range(n_accumulators)]
            for start, stop in slot_blocks:
                tables_b = base_tables._replace(
                    **dict(
                        zip(
                            ("projection_score_cache", "projection_recon_cache", "projection_recon_abs2_cache"),
                            project_rows(start, stop, safe_rows),
                        )
                    ),
                    mstep_grid=jnp.asarray(
                        slot_matrices[block_entries(start, stop, safe_rows)], dtype=base_tables.mstep_grid.dtype
                    ),
                    translation_angles=image_angles,
                    cache_slot_fine_rot=None,
                )
                mstep = mstep_slot_block(
                    mstep,
                    tables_b,
                    lambda accumulator: slot_tables_block(block_slots[accumulator], start, stop),
                    block_posteriors,
                )
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
        row_fine_rot=jnp.asarray(row_fine_rot_device(host, chunk.row_capacity)),
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
    (``relion_cuda_fine_normalized_cc_score``); the float32 running sum keeps the slot order.
    ``min_diff2`` is the particle's ``0.5 |X_i|^2`` summed over its images (the SPA evidence offset).
    """

    from relax.cuda import translate as cuda_translate
    from relax.fine_pass.resident_scoring import ResidentChunkScores
    from relax.fine_pass.scoring import relion_cuda_fine_normalized_cc_score

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
                tiles = cuda_translate.relion_translate_score_f32(
                    score_input[safe], block_angles[safe], pixels, shape
                ).reshape(safe.shape[0], index.size, -1)
                images = jnp.asarray(np.where(slot_image_ids[slot] >= 0, slot_image_ids[slot], 0), dtype=jnp.int32)
                has_image = jnp.asarray(slot_image_ids[slot] >= 0) & row_is_valid
                projections = jax.lax.dynamic_slice_in_dim(cache, (slot - start) * row_capacity, row_capacity, axis=0)

                def block_cc(first, projections=projections, tiles=tiles, images=images):
                    rows = jax.lax.dynamic_slice_in_dim(row_unit, first, block_rows)
                    row_images = jax.lax.dynamic_slice_in_dim(images, first, block_rows)
                    return relion_cuda_fine_normalized_cc_score(
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


def _unit_slot_images(layout: ChunkTiltLayout, *, unit_capacity: int, slot_capacity: int) -> np.ndarray:
    """``[S, C_U]`` chunk image of each unit's s-th image, -1 past its images or on a padded unit."""

    n_images = int(layout.n_valid_images)
    unit_of_image = np.asarray(layout.image_unit_local[:n_images], dtype=np.int64)
    counts = np.bincount(unit_of_image, minlength=int(unit_capacity))[: int(unit_capacity)]
    first = np.concatenate([[0], np.cumsum(counts)[:-1]])
    slots = np.arange(int(slot_capacity))[:, None]
    return np.where(slots < counts[None, :], first[None, :] + slots, -1).astype(np.int64)


def row_fine_rot_device(host, row_capacity: int) -> np.ndarray:
    """The chunk rows' global fine rotations, padded rows at 0."""

    out = np.zeros(int(row_capacity), dtype=np.int32)
    values = np.asarray(host["row_fine_rot"], dtype=np.int32)
    out[: values.size] = values[: int(row_capacity)]
    return out


@partial(jax.jit, static_argnames=("spec",))
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
    operands,  # ChunkStageOperands over the chunk's images (image_noise_scale set)
    tables,  # ChunkStageTables
    mstep,  # ChunkMstepCarry after every slot's blocks (noise and norm partials already scaled)
    posterior,  # resident_chunk_records.ChunkPosterior over the chunk's particles
    rows,
    spec,
):
    """Fold one tilt chunk into the resident statistics, particle terms once and image terms per image.

    The SPA fold (resident_image_terms.accumulate_chunk_image_terms) with RELION's subtomogram rules
    (acc_ml_optimiser_impl.h storeWeightedSums): the offset, sumw, rotation and score/pose terms are the
    particle's; the image power, the noise shells and the norm residual are summed over its images with
    1 / n_images (:3490-3491, :3512-3516); the scale sums are summed over its images as they
    are (:3474-3479). The noise and norm block partials arrive already scaled (image_noise_scale).
    With K>1 the class terms are the particle's (``resident_image_terms.fold_class_axis``) and the scale
    sums arrive folded class by class (``mstep.scale_xa_per_image``).

    One compiled program per chunk shape, the optics groups' noise shells vmapped over the groups. Run
    eagerly with a Python loop over the groups it dispatched each group's operations one by one while the
    GPU waited: 13.6% of a real tilt-series final iteration's pass-2 host samples, with 130 optics groups.
    Compiled, XLA fuses and orders the same operations its own way, so the sums differ from the eager
    ones in float32 rounding, not bitwise.
    """

    from relax.fine_pass import resident_image_terms
    from relax.fine_pass.resident_statistics import ResidentStatistics, drop_index, segment_sum_by_image

    config = spec.stats_config
    unit_capacity = int(unit_ids.shape[0])
    valid_unit = unit_ids >= 0
    unit_slot = drop_index(unit_ids, int(config.image_capacity))
    valid_image = image_ids >= 0
    scale = jnp.where(valid_image, jnp.asarray(operands.image_noise_scale, dtype=jnp.float32), jnp.float32(0.0))

    # --- 1/2. offset and support mass: once per particle --------------------
    translation_posterior = segment_sum_by_image(
        row_posterior, row_unit_local, unit_capacity
    )
    sigma2_offset = stats.sigma2_offset
    if unit_translation_sqdist_ang is not None:
        sigma2_offset = sigma2_offset + jnp.sum(
            jnp.where(valid_unit[:, None], translation_posterior.astype(jnp.float64), 0.0)
            * jnp.asarray(unit_translation_sqdist_ang, dtype=jnp.float64)
        )
    unit_mass = jnp.where(
        valid_unit, jnp.sum(translation_posterior, axis=1), jnp.zeros((), translation_posterior.dtype)
    )
    if config.n_optics_groups == 1:
        sumw = stats.sumw + jnp.sum(unit_mass.astype(jnp.float64))
    else:
        sumw = stats.sumw + jax.ops.segment_sum(
            unit_mass.astype(jnp.float64), unit_optics_groups, num_segments=config.n_optics_groups
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
    if config.n_optics_groups == 1:
        residual_shells, image_power_shells = replace_low_shell_noise_with_relion_wavg_direct_residual_jnp(
            jnp.asarray(mstep.noise_shells, dtype=jnp.float64),
            weighted_img_shells.astype(jnp.float64),
            direct_residual,
            tables.wavg_shell_indices,
            exclusive_shell_stop=int(config.direct_noise_exclusive_shell_stop),
            shell_count=config.n_shells,
        )
    else:
        image_optics = jnp.asarray(operands.optics_groups, dtype=jnp.int32)

        def group_shells(group, group_noise_shells):
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
            return replace_low_shell_noise_with_relion_wavg_direct_residual_jnp(
                jnp.asarray(group_noise_shells, dtype=jnp.float64),
                group_img_shells.astype(jnp.float64),
                jnp.where(in_group[:, None], direct_residual, jnp.float32(0.0)),
                tables.wavg_shell_indices,
                exclusive_shell_stop=int(config.direct_noise_exclusive_shell_stop),
                shell_count=config.n_shells,
            )

        residual_shells, image_power_shells = jax.vmap(group_shells)(
            jnp.arange(config.n_optics_groups, dtype=jnp.int32), mstep.noise_shells
        )

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
        group_slot = drop_index(operands.group_ids, int(config.n_scale_groups))
        keep = valid_image & (jnp.asarray(operands.group_ids, dtype=jnp.int32) >= 0)
        scale_xa = scale_xa.at[group_slot].add(jnp.where(keep, xa, 0.0), mode="drop")
        scale_aa = scale_aa.at[group_slot].add(jnp.where(keep, aa, 0.0), mode="drop")

    # --- 9. rotation posterior sums: the particle's -------------------------
    probs_sum_t = jnp.sum(row_posterior, axis=-1)
    coarse_slot = drop_index(row_coarse_rot, int(config.n_coarse_rot))
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
    best_local_rot = (best_cell_index // jnp.int64(config.n_fine_trans)).astype(jnp.int32)
    last_row = jnp.int64(int(spec.row_capacity) - 1)
    best_chunk_row = jnp.clip(rows.image_row_start + best_local_rot.astype(jnp.int64), 0, last_row).astype(jnp.int32)
    row_fine_rot = jnp.asarray(row_fine_rot, dtype=jnp.int64)
    n_t = jnp.int64(config.n_fine_trans)
    best_cell_values = row_fine_rot[best_chunk_row] * n_t + best_cell_index % n_t

    # --- 11. the class axis (K>1): the particle's --------------------------
    classes = stats.classes
    if classes is not None:
        class_cell = jnp.asarray(posterior.classes.best_cell_index, dtype=jnp.int64)
        class_best_row = jnp.clip(rows.classes.segment_row_start + class_cell // n_t, 0, last_row).astype(jnp.int32)
        classes = resident_image_terms.fold_class_axis(
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
    rows = sorted({max(int(min_rows), pow2_floor(max(int(r) // slots, 1))) for r in row_ladder})
    units = sorted({max(1, int(b)) for b in image_ladder})
    return tuple(rows), tuple(units)
