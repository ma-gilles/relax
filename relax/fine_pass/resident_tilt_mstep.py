"""The slot M-step program of the resident tilt chunk (cryo-ET, S4.2).

A tilt chunk's M-step runs per block of image slots: ``slot_mstep_tables`` builds each slot's unit, row and
translation tables (``SlotMstepTables``), ``tilt_mstep_program`` gathers each slot's per-image operands
(``_SLOT_VIEW_FIELDS``), translates them, sums every row's translation-block partials and places the slot's
partials in the chunk's (``_SLOT_PARTIAL_FIELDS``). The translations a unit keeps (``unit_mstep_translations``)
are cut into blocks that fit the tile budget (``mstep_translation_blocks``, ``tilt_mstep_tile_bytes``).
``resident_tilts.run_tilt_chunk`` calls them; this module imports nothing from it.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from relax.reconstruction.half_volume_mstep import blocks_per_flush, flushed_block_loop
from relax.runtime.shape_buckets import pow2_ceil, pow2_floor

# relion_translate_sum.cuh kMinRowsPerTableImage: a translate-and-sum launch with at least this many rows per
# image builds its images' shifted [images, T, recon pixels] float4 table.
TRANSLATE_SUM_MIN_ROWS_PER_TABLE_IMAGE = 64


def tilt_mstep_tile_bytes(
    n_units: int, *, n_rect_pixels: int, n_exact_pixels: int, n_recon_pixels: int = 0, block_rows: int = 0
) -> int:
    """Device bytes one M-step translation adds to a slot's translated Wavg tiles (:func:`tilt_mstep_program`).

    The tiles are per unit, ``[units, T, pixels]``: the translated rectangle (complex64) and its copy with the
    padded units zeroed (:func:`_slot_view_translate`), its exact-radius gather, and the rectangle's float32
    power and the product it is built from (``add_wavg_rectangle_image_power``). The M-step kernels read them
    by each row's image; no tile has a row axis.

    An M-step block of ``block_rows`` rows with at least :data:`TRANSLATE_SUM_MIN_ROWS_PER_TABLE_IMAGE` per unit
    also has the translate-and-sum's shifted table, ``n_recon_pixels`` float4 per unit and translation, beside
    them: XLA's scratch for the call, from the same pool (relax#120).
    """

    tiles = int(n_units) * (3 * int(n_rect_pixels) + int(n_exact_pixels)) * 8
    table = int(n_units) * int(n_recon_pixels) * 16
    return tiles + (table if int(block_rows) >= TRANSLATE_SUM_MIN_ROWS_PER_TABLE_IMAGE * int(n_units) else 0)


def tilt_mstep_units(n_valid_units: int, *, unit_capacity: int) -> int:
    """Units a tilt chunk's M-step tables and translated tiles hold (:func:`run_tilt_chunk`, relax#102).

    The chunk's valid units rounded up to a power of two, at most its unit capacity: the padded units carry no
    image, so sizing the tiles at the capacity (32 in a final iteration) only shrank the translation blocks
    of a chunk holding few particles (:func:`tilt_mstep_tile_bytes`) and multiplied its M-step loop
    iterations. The power of two bounds the M-step programs to one per power of two up to the capacity.
    Fewer blocks change the order of each row's partial sums over translations, not their terms.
    """

    return min(int(unit_capacity), pow2_ceil(int(n_valid_units)))


def slot_tables_block(slots: "SlotMstepTables", start: int, stop: int) -> "SlotMstepTables":
    """Slots ``start:stop`` of :func:`slot_mstep_tables`, renumbered from 0 (their block's projections)."""

    block = slots._replace(**{name: getattr(slots, name)[start:stop] for name in slots._fields})
    return block._replace(slot=jnp.arange(stop - start, dtype=jnp.int32))


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


def slot_mstep_tables(
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
    """Host tables for :func:`tilt_mstep_program`: one transfer per chunk instead of several per slot.

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
def tilt_mstep_program(
    state,
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
    translation block once (as the SPA chunk does, ``add_chunk_wavg_image_power``) and puts its
    per-image partials on those images' chunk rows. ``row_posterior`` ``[C_R, k]`` is each row's posterior at
    its unit's ``k`` translations (:func:`unit_mstep_translations`), and ``slots.angles`` the units' phases
    there; ``kept_index``/``kept_valid`` are the translation blocks' ``[K, T_b]`` positions in those ``k``
    and validity: positions outside a block (and padding) carry zero posterior there.

    With more than one translation block (``slot_spec.presum_adjoint``), a row's weight arrives as one
    partial per block. Adjoined block by block, a partial below half an ulp of a voxel's float32 total
    is lost (relax#27: DC and low shells at et09 lost ~1e-6 of their weight, more with more blocks). So
    the blocks add each row's partials into a ``[C_R, recon pixels]`` pair at the row's own scale, and
    each row is backprojected once, as in one block (and in RELION, which sums an orientation's
    translations before it backprojects).

    ``state`` is ``(mstep,)`` or ``(mstep, (partial_y, partial_ctf))`` and is returned in the same form. With
    the pass's zeroed partial pair, the rows are scattered into it and it is added into the volumes every
    ``blocks_per_flush`` blocks and at the end of every slot (relax#65): a row is one tilt image here, so
    the flush counts tilt images.
    """

    from relax.fine_pass import resident_block_programs, resident_chunk_records, resident_mstep_blocks

    row_capacity = int(row_posterior.shape[0])
    block_rows = int(slot_spec.mstep_block_rows)
    merge_row_partials = bool(slot_spec.presum_adjoint)
    fields = {name: getattr(operands, name) for name in _SLOT_VIEW_FIELDS}
    flush_blocks = blocks_per_flush(block_rows)
    mstep, *partial_pair = state

    def one_slot(scan_carry, xs):
        full, *pair = scan_carry
        slot, safe, valid, target, order, active, units, n_blocks, angles = xs
        ordered_posterior = row_posterior[order]
        gathered, window = _slot_view_gather(fields, wavg_window, safe, valid)
        kernel_row_image_ids = jnp.where(active, units, jnp.int32(-1))
        row_fine_rot = slot * jnp.int32(row_capacity) + order
        Ft_y, Ft_ctf = full.Ft_y, full.Ft_ctf
        # With a partial pair the blocks scatter into it, and the loops carry the slot's volumes beside it.
        flush_blocks_into_volumes = bool(pair) and not merge_row_partials
        if flush_blocks_into_volumes:
            Ft_y, Ft_ctf = pair[0]
        if merge_row_partials:
            # Each row's partials, by its visiting position, until the one adjoint below.
            n_recon = int(slot_spec.n_recon_pixels)
            Ft_y = jnp.zeros((row_capacity, n_recon), dtype=full.Ft_y.dtype)
            Ft_ctf = jnp.zeros((row_capacity, n_recon), dtype=full.Ft_ctf.dtype)
        # The block stages' dtypes follow from the gathered operands alone (resident_mstep_blocks._mstep_block_operand_dtypes).
        slot_carry = resident_mstep_blocks.initial_mstep_carry(
            Ft_y,
            Ft_ctf,
            operands._replace(**gathered),
            stage_tables._replace(translation_angles=angles[0]),
            spec=slot_spec,
        )

        # One translation block at a time: a loop iteration's translated tiles are dead before the next
        # block translates its own, so the program holds one block's tiles, as mstep_translation_blocks
        # sizes them. Unrolled blocks let XLA schedule every block's translation first (all tiles live).
        def translation_block(k, loop_carry):
            slot_carry, *volumes = loop_carry
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
            blocks = resident_chunk_records.MstepBlockInputs(
                row_image_local=units,
                kernel_row_image_ids=kernel_row_image_ids,
                row_posterior=jnp.where(active[:, None], block_posterior, jnp.zeros((), block_posterior.dtype)),
                row_fine_rot=row_fine_rot,
                projections=None,
                row_sum_ids=jnp.arange(row_capacity, dtype=jnp.int32) if merge_row_partials else None,
            )

            def block(i, carry):
                return resident_mstep_blocks.resident_mstep_block_at(
                    i * jnp.int32(block_rows),
                    blocks,
                    slot_operands,
                    slot_tables,
                    carry,
                    spec=slot_spec,
                )

            if flush_blocks_into_volumes:
                slot_carry, *volumes = flushed_block_loop(
                    n_blocks,
                    block,
                    slot_carry,
                    *volumes,
                    flush_blocks=flush_blocks,
                    get_pair=lambda carry: (carry.Ft_y, carry.Ft_ctf),
                    set_pair=lambda carry, y, ctf: carry._replace(Ft_y=y, Ft_ctf=ctf),
                )
            else:
                slot_carry = jax.lax.fori_loop(0, n_blocks, block, slot_carry)
            slot_carry = resident_block_programs.add_chunk_wavg_image_power(
                slot_carry,
                slot_operands,
                slot_tables,
                blocks.row_posterior,
                blocks.kernel_row_image_ids,
                spec=slot_spec,
            )
            return (slot_carry, *volumes)

        slot_carry, *volumes = jax.lax.fori_loop(
            0,
            int(kept_index.shape[0]),
            translation_block,
            (slot_carry, full.Ft_y, full.Ft_ctf) if flush_blocks_into_volumes else (slot_carry,),
        )
        Ft_y, Ft_ctf = slot_carry.Ft_y, slot_carry.Ft_ctf
        if flush_blocks_into_volumes:
            pair = [(Ft_y, Ft_ctf)]
            Ft_y, Ft_ctf = volumes
        if merge_row_partials:

            def adjoint(i, volumes):
                def take(values):
                    return jax.lax.dynamic_slice_in_dim(values, i * jnp.int32(block_rows), block_rows, axis=0)

                rotations = resident_mstep_blocks.cached_block_projections(stage_tables, take(row_fine_rot))[2]
                return resident_mstep_blocks.backproject_block_rows(
                    take(slot_carry.Ft_y),
                    take(slot_carry.Ft_ctf),
                    rotations,
                    *volumes,
                    tables=stage_tables,
                    spec=slot_spec,
                )

            if pair:
                pair_out, Ft_y, Ft_ctf = flushed_block_loop(
                    n_blocks,
                    adjoint,
                    pair[0],
                    full.Ft_y,
                    full.Ft_ctf,
                    flush_blocks=flush_blocks,
                    get_pair=lambda volumes: volumes,
                    set_pair=lambda volumes, y, ctf: (y, ctf),
                )
                pair = [pair_out]
            else:
                Ft_y, Ft_ctf = jax.lax.fori_loop(0, n_blocks, adjoint, (full.Ft_y, full.Ft_ctf))
        placed = _place_slot_partials(
            {name: getattr(full, name) for name in _SLOT_PARTIAL_FIELDS},
            {name: getattr(slot_carry, name) for name in _SLOT_PARTIAL_FIELDS},
            full.noise_shells,
            slot_carry.noise_shells,
            target,
        )
        return (full._replace(Ft_y=Ft_y, Ft_ctf=Ft_ctf, **placed), *pair), None

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
    state, _ = jax.lax.scan(one_slot, (mstep, *partial_pair), xs)
    return state


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
    k = max(int(minimum), pow2_ceil(largest))
    if k >= n_fine_trans:
        k = n_fine_trans
    # Translations with mass first, each set ascending (a stable sort of "no mass").
    index = np.argsort(~unit_mass, axis=1, kind="stable")[:, :k].astype(np.int64)
    valid = np.arange(k)[None, :] < counts[:, None]
    last = np.take_along_axis(index, np.maximum(counts - 1, 0)[:, None], axis=1)
    index = np.where(valid, index, np.where(counts[:, None] > 0, last, 0))
    return UnitMstepTranslations(index=index, valid=valid)


# The flat-row translate-and-sum keeps a block's posterior rows and active-translation list in under
# 48 KB of shared memory: 8 T bytes at one row per block for tilt images, whose angle tables stay in
# global memory (relion_translate_sum.cuh).
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

    fit = min(max(int(tile_budget_bytes) // max(int(bytes_per_translation), 1), 1), TILT_MSTEP_MAX_TRANSLATIONS)
    if kept.index.size <= fit:
        return (kept,)
    block = pow2_floor(fit)
    out = []
    for start in range(0, kept.index.size, block):
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

    from relax.fine_pass.resident_operands import gather_rows

    gathered = {
        name: gather_rows(values, safe, valid, fill=_SLOT_VIEW_FILL.get(name, 0)) for name, values in fields.items()
    }
    return gathered, gather_rows(wavg_window, safe, valid)


def _slot_view_translate(window, valid, angles, rect_indices, exact_positions, *, image_shape):
    """The slot's translated Wavg rectangle ``[C, T, W]``, each image with its own phases (one launch)."""

    from relax.fine_pass.wavg import relion_cuda_translate_wavg_norm_window

    translated = relion_cuda_translate_wavg_norm_window(window, angles, rect_indices, image_shape)
    translated = jnp.where(valid[:, None, None], translated, jnp.zeros((), translated.dtype))
    return translated, translated[:, :, exact_positions]


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
