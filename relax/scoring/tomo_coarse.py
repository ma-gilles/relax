"""Coarse pass (pass 1) of subtomogram particles: each tilt image scored, summed per particle.

RELION's GPU path scores a subtomogram one tilt image at a time (the ``img_id`` loop of
``getAllSquaredDifferencesCoarse``, acc_ml_optimiser_impl.h:1190-1402). Every image has
its own scorer matrices (``make_eulers_3D`` with the image's ``Aproj`` as the left matrix,
:1086-1116) and its own phase table (``Aproj[:2]`` times the 3D trial shift plus the
rounded old offset, :1214-1240). Each image first adds its ``highres_Xi2 / 2`` to the
particle's running diff2 and then its pixel sums (:1289-1296, diff2.cuh:186-188), so
the significance cut sees one diff2 per particle hypothesis.

relax reuses the SPA pieces: the RELION CUDA preprocessing with no pre-shift and no norm
correction (RELION neither translates nor normalises a tomo image, :429-476), the exact
coarse operands, RELION's score translation, and the SPA coarse GEMM scorer (RELION's
``d0 + 0.5 sum w |p - y|^2`` as two real-packed float32 GEMMs) on each image's own projections,
a block of a batch's images per call. The GEMM expansion rounds differently from RELION's
direct square, so the rotations with a sample within the expansion's error bound of a particle's
significance cut (or of its smallest diff2) are scored again by the fused direct-square kernel before the
cut is taken (:mod:`relax.scoring.exact_cut`). The tests compare the GEMM scorer with
that kernel scored one image at a time (``oracles.tomo_coarse.tilt_image_coarse_diff2``).

The GEMM block is ``tomo_coarse_gemm``'s; the ``--firstiter_cc`` winner pass, a different contract, is
``tomo_coarse_cc``'s. This module keeps the layout, the operands and the significance pass with its budgets.
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from relax.runtime import memory_budget
from relax.runtime.shape_buckets import pow2_ceil, pow2_floor
from relax.scoring.exact_cut import (
    direct_rows_diff2,
    first_rows,
    float32_unit,
    put_rows,
    row_capacity,
    undecided_rotations,
)
from relax.scoring.tomo_coarse_gemm import coarse_gemm_frame, coarse_gemm_slot_block, slot_block_error

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CoarseScoreLayout:
    """The coarse score window and its fused-projector lookup, shared by every image of a pass.

    ``image_current_size`` is the iteration's current size (RELION's ``image_current_size``, the pass-2 size): the
    diff2 constant counts the image power beyond it, not beyond the coarse ``current_size``. None when the two agree.
    """

    image_shape: tuple
    current_size: int
    score_indices_np: np.ndarray
    score_active_mask: jnp.ndarray
    full_to_compact: jnp.ndarray
    half_weights: jnp.ndarray
    image_current_size: int | None = None


def coarse_score_layout(
    image_shape, current_size, *, half_spectrum_scoring: bool, square_window: bool, image_current_size=None
) -> CoarseScoreLayout:
    """The SPA coarse pass's score window (significance.py:1105-1123, 1680-1726) for one current size;
    ``image_current_size`` is the iteration's (:class:`CoarseScoreLayout`)."""

    from relax.fourier.fourier_window import make_fourier_window_spec
    from relax.fourier.half_spectrum import make_scoring_half_image_weights
    from relax.scoring.coarse_layout import coarse_gaussian_fused_logical_lookup, plan_coarse_gaussian_square_layout

    image_shape = tuple(int(size) for size in image_shape)
    n_half = image_shape[0] * (image_shape[1] // 2 + 1)
    window = make_fourier_window_spec(
        image_shape, current_size, n_half, square=square_window, include_recon_window=False
    )
    active = (
        np.arange(n_half, dtype=np.int32)
        if window.score_indices_np is None
        else np.asarray(window.score_indices_np, dtype=np.int32)
    )
    layout = plan_coarse_gaussian_square_layout(
        image_shape, int(current_size), active, stable_fourier_window_shapes=False, stable_window_quantum=None
    )
    return CoarseScoreLayout(
        image_shape=image_shape,
        current_size=int(current_size),
        score_indices_np=np.asarray(layout.score_indices_np, dtype=np.int32),
        score_active_mask=jnp.asarray(layout.score_active_mask_np, dtype=jnp.bool_),
        full_to_compact=coarse_gaussian_fused_logical_lookup(
            jnp.asarray(layout.full_to_compact_np, dtype=jnp.int32), layout, current_size=int(current_size)
        ),
        half_weights=make_scoring_half_image_weights(
            image_shape, relion_half_sum=half_spectrum_scoring, exclude_relion_redundant_x0=True
        ),
        image_current_size=None if image_current_size is None else int(image_current_size),
    )


def tilt_image_coarse_operands(
    experiment_dataset,
    image_indices,
    layout: CoarseScoreLayout,
    *,
    noise_variance_half,
    optics_group_ids=None,
    scale_corrections=None,
    score_with_masked_images: bool = True,
    fetched=None,
):
    """Corrected images, pixel weights and initial diff2 of tilt images: the SPA exact coarse operands.

    ``noise_variance_half`` is one half-pixel spectrum or a ``[G, P]`` table with
    ``optics_group_ids`` per dataset image. No pre-shift and no norm correction: RELION
    neither translates nor normalises a tomo image (acc_ml_optimiser_impl.h:429-476);
    ``scale_corrections`` (per dataset image) still divide the image and weight the
    CTF, as for SPA (:1244-1263). ``fetched`` is the images' host read
    (:func:`relax.io.batch_fetch.fetch_indexed_batch`) when it was made ahead.
    """

    from relax.io.batch_fetch import fetch_indexed_batch
    from relax.relion.ctf import relion_exact_ctf_half_from_source_star
    from relax.relion.image_preprocessing import prepare_batch_preprocess_operands
    from relax.relion.optics_groups import noise_rows
    from relax.scoring.coarse_operands import (
        process_relion_exact_coarse_half_image,
        relion_exact_coarse_operands,
    )

    image_indices = np.asarray(image_indices, dtype=np.int64)
    # One vectorized host read (the dataset's batch iterator collates image by image, 15x slower).
    batch_data, _ctf_params, fetched = fetch_indexed_batch(experiment_dataset, image_indices) if fetched is None else fetched
    if not np.array_equal(np.asarray(fetched), image_indices):
        raise RuntimeError("the dataset returned the tilt images in another order")
    batch_data = np.asarray(batch_data)
    relion_cuda, _shifts, _corr, batch_scale, preprocess_kwargs = prepare_batch_preprocess_operands(
        experiment_dataset, batch_data, image_indices, scale_corrections=scale_corrections
    )
    if not relion_cuda:
        raise RuntimeError("the subtomogram coarse pass needs the RELION CUDA image preprocessing")
    # The image indices demodulate odd aberrations, as RELION does for every tilt image (ml_optimiser.cpp:6418).
    processed = process_relion_exact_coarse_half_image(
        experiment_dataset,
        batch_data,
        score_with_masked_images,
        relion_preprocess_kwargs=preprocess_kwargs,
        image_indices=image_indices,
    )
    # On the device, where the operands are built: a host copy waited for the queued coarse batches.
    ctf = relion_exact_ctf_half_from_source_star(
        experiment_dataset, image_indices, layout.image_shape, pixel_indices=layout.score_indices_np
    )
    unshifted, pixel_weight = relion_exact_coarse_operands(
        jnp.asarray(ctf, dtype=jnp.float64),
        jnp.asarray(batch_scale, dtype=jnp.float32),
        processed,
        jnp.asarray(layout.score_indices_np),
        layout.score_active_mask,
        noise_rows(noise_variance_half, optics_group_ids, image_indices),
        layout.half_weights,
        image_shape=layout.image_shape,
        use_float64_scoring=False,
        scale_corrections_enabled=scale_corrections is not None,
    )
    return unshifted, pixel_weight, jnp.asarray(highres_xi2_half(processed, layout), dtype=jnp.float32)


def highres_xi2_half(processed, layout: CoarseScoreLayout):
    """Each image's coarse diff2 constant: half of RELION's powerClass Xi2, the image power at and beyond
    ``image_current_size / 2 + 1``, the iteration's size (``layout.image_current_size``, else its ``current_size``).

    RELION takes it once per particle for both passes (acc_ml_optimiser_impl.h:1200-1265), so the coarse size does
    not enter; it is 0 when that size is the full box. One program per box, the size a runtime scalar.
    """

    from relax.fine_pass.scoring import relion_cuda_powerclass_highres_xi2_half

    size = layout.current_size if layout.image_current_size is None else layout.image_current_size
    return relion_cuda_powerclass_highres_xi2_half(
        processed, image_shape=layout.image_shape, current_size=int(layout.image_shape[0]), runtime_current_size=jnp.int32(size)
    )


# Bytes of per-image projections ([images, R, P]) and coarse diff2 ([images, R, T] float32) one call holds, at most
# (_coarse_pass_budget caps it by what the device can still hand out).
_COARSE_BATCH_BYTES = 2 << 30

# Bytes of particles' summed diff2 ([P, K * R, T] float32) one significance call cuts, at most
# (_flush_plan bounds the flush by what the device can still hand out).
_SIGNIFICANCE_BATCH_BYTES = 512 << 20


# Each class's projector texture, kept from pass to pass and refilled with the pass's projector: the compiled
# coarse programs capture the texture's native handle, so a new texture per pass compiled every slot-block
# program again (1,061 compiles, 220 s of a 200-iteration et09_box64 subtomogram run). Class index -> (texture,
# a concrete result of its last use, which a refill or close must wait for).
_CLASS_TEXTURES: dict = {}


def _coarse_capacity_texture(
    projector_half, layout: CoarseScoreLayout, *, model_max_r: int, padding_factor: int, class_index: int
):
    """The class's persistent projector texture, refilled with ``projector_half``, when the half-storage kernel
    serves it, else ``None``."""

    from relax.projection.projection import relion_capacity_texture_serves

    projector_half = jnp.asarray(projector_half)
    if jax.default_backend() != "gpu" or not relion_capacity_texture_serves(
        projector_half,
        r_max=int(model_max_r),
        padding_factor=int(padding_factor),
        projector_output_size=int(layout.current_size),
        relion_texture_interp=True,
    ):
        return None
    from relax.cuda.projector import RelionCapacityHalfTextureF32

    texture, completion = _CLASS_TEXTURES.pop(int(class_index), (None, None))
    if (
        texture is not None
        and not texture.closed
        and texture.shape == tuple(int(n) for n in projector_half.shape)
        and texture.padding_factor == int(padding_factor)
        and texture.device in projector_half.devices()
    ):
        texture.refresh_after(projector_half, completion, logical_r_max=int(model_max_r))
    else:
        if texture is not None:
            texture.close_after(completion)
        texture = RelionCapacityHalfTextureF32(
            projector_half, int(model_max_r), padding_factor=int(padding_factor), reusable_staging=True
        )
    _CLASS_TEXTURES[int(class_index)] = (texture, jnp.zeros((), jnp.float32))
    return texture


@partial(jax.jit, static_argnames=("capacity", "row_length"))
def _significant_cells(mask, *, capacity: int, row_length: int):
    """Flat ids (row-major over ``mask`` ``[P, N]``) of its true cells, ascending, padded to ``capacity`` with -1.

    The cells are found in two steps, the rows of ``row_length`` cells holding one and then the cells of the first
    ``capacity`` such rows: one ``nonzero`` over every cell scattered all ``P * N`` of them, 122 ms per et09 flush
    of 29 particles on an A100 (``N`` = 36864 rotations x 123 translations).
    """

    rows = mask.reshape(-1, row_length)
    hit = jnp.any(rows, axis=1)
    # At most every row holds a cell: a larger capacity only pads ``held``, whose nonzero scans all its cells.
    row_capacity = min(capacity, rows.shape[0])
    row_ids = jnp.nonzero(hit, size=row_capacity, fill_value=0)[0]
    held = rows[row_ids] & (jnp.arange(row_capacity) < jnp.sum(hit))[:, None]
    cell = jnp.nonzero(held.reshape(-1), size=capacity, fill_value=-1)[0]
    return jnp.where(cell >= 0, row_ids[cell // row_length] * row_length + cell % row_length, -1)


# Rotations per particle whose samples the adaptive-fraction rule sorts (exact_cut.undecided_rotations).
_SORTED_ROTATIONS = 256

# Undecided rotations per particle scored again in one go; a flush with a particle holding more takes a larger size.
_UNDECIDED_ROTATIONS = 8


# The share of the GEMM error bound (relax.scoring.exact_cut.gemm_error_bound) the cut's margin takes. The bound is
# the worst case, (4P + 11) u times the largest term, while the measured GEMM - direct-square differences are 1.6e3
# to 2.6e4 times smaller (10th to 50th percentile per particle over a w2_09 box-192 Refine3D, relax#33 dumps, job
# 15178008). On hard-noise subtomogram VDAM K=1 (cell 13, one iteration from RELION it190, job 15206974) a tenth of
# it leaves 79,880 rows undecided instead of 56.8M and flips none of the 22,782 significant cells against the full
# bound (GEMM scores alone flip 26); the tenth still leaves a factor of 160 over the largest observed difference.
_GEMM_ERROR_MARGIN_SHARE = 0.1


@partial(jax.jit, static_argnames=("max_significants", "adaptive_fraction", "sorted_rotations"))
def _near_cut_rows(
    diff2, values, mask, cutoff_count, error, n_slots,
    *, max_significants: int, adaptive_fraction: float, sorted_rotations: int | None = _SORTED_ROTATIONS,
):
    """:func:`relax.scoring.exact_cut.undecided_rotations` of particles from their summed GEMM diff2 ``[P, K * R, T]``.

    The particle's bound is ``_GEMM_ERROR_MARGIN_SHARE`` of its images' GEMM bounds ``error`` plus one float32 unit
    of its largest diff2 per image for the two slot-order sums and four for forming the log weights.
    """

    scored = jnp.isfinite(values).reshape(diff2.shape)
    largest = jnp.max(jnp.where(scored, diff2, 0.0), axis=(1, 2))
    smallest = jnp.min(jnp.where(scored, diff2, jnp.inf), axis=(1, 2))
    margin = 2.0 * (_GEMM_ERROR_MARGIN_SHARE * error + (n_slots.astype(jnp.float32) + 4.0) * float32_unit(largest))
    return undecided_rotations(
        values,
        mask,
        cutoff_count,
        margin,
        jnp.where(scored, diff2, jnp.inf),
        smallest,
        n_trans=int(diff2.shape[2]),
        max_significants=max_significants,
        adaptive_fraction=adaptive_fraction,
        sorted_rotations=sorted_rotations,
    )


def _particle_near_cut_rows(cut_inputs, full_sort, *, max_significants: int, adaptive_fraction: float):
    """Each particle's undecided rotations (:func:`_near_cut_rows` of ``cut_inputs``), their largest count and the
    particles the sort of the leading ``_SORTED_ROTATIONS`` rotations leaves unresolved. The particles marked in
    ``full_sort`` (``[P]``, or None) take their rows from every rotation sorted: a particle's rows depend on its own
    values alone, not on the flush it is cut in."""

    static = dict(max_significants=max_significants, adaptive_fraction=adaptive_fraction)
    rows, _most, unresolved = _near_cut_rows(*cut_inputs, **static, sorted_rotations=_SORTED_ROTATIONS)
    if full_sort is not None:
        every, _most, _unresolved = _near_cut_rows(*cut_inputs, **static, sorted_rotations=None)
        rows = jnp.where(full_sort[:, None], every, rows)
        unresolved = unresolved & ~full_sort
    return rows, jnp.max(jnp.sum(rows, axis=1)), unresolved


# Scorer matrices built per call of the exact cut's rotation kernel: the call builds every pairing of its
# particles' images and rotations, of which each particle's own are kept.
_UNDECIDED_ROTATION_CALL_BYTES = 64 << 20


def _undecided_scorer_rotations(
    ids,
    use,
    rotation_counts,
    unit_rotation_rows,
    eulers_f32,
    particle_left,
    random_perturbation,
    angular_sampling_deg,
    *,
    r_pad: int,
    slots: int,
    pass1_rotations,
):
    """The scorer matrices ``[P, S, M, 3, 3]`` float32 of the particles' undecided rotations ``ids`` ``[P, M]``
    (device arrays, where ``use``), built on the device without reading ``ids`` back.

    A rotation's matrix for a tilt image is ``pass1_rotations`` of its coarse Euler row with the image's left
    matrix (``particle_left[p]`` ``[S_p, 3, 3]``) and the pass's perturbation matrix
    (:func:`relax.sampling.device_rotations.pass1_rotations_left_f32`, the kernel :func:`_batch_scoring_rotations`
    uses), so the values are those the GEMM pass scored; a padded rotation repeats the particle's last one and a local
    search's rows index its own rotations (``unit_rotation_rows[p]``). The kernel builds every pairing of its
    Euler rows and left matrices, so the particles go in groups, each particle keeping its own pairings.
    Padded slots and particles without an undecided rotation are zero. The perturbation enters the program as its
    float32 matrix, not as a static float: every VDAM iteration draws a new one, which compiled the program again
    for every shape in every iteration (relax#115).
    """

    from relax.sampling.device_rotations import pass1_right_matrix

    n_particles, capacity = (int(n) for n in ids.shape)
    table = None
    if unit_rotation_rows is not None:
        table = np.zeros((n_particles, max(int(np.max(rotation_counts)), 1)), dtype=np.int32)
        for p, rows in enumerate(unit_rotation_rows):
            table[p, : rows.size] = rows
        table = jnp.asarray(table)
    left = np.tile(np.eye(3, dtype=np.float32), (n_particles, slots, 1, 1))  # a padded slot's matrices are zeroed
    n_images = np.zeros(n_particles, dtype=np.int32)
    for p, images in enumerate(particle_left):
        left[p, : len(images)] = np.asarray(images, dtype=np.float32)
        n_images[p] = len(images)
    right = pass1_right_matrix(random_perturbation, angular_sampling_deg)
    return _lay_out_undecided_rotations(
        ids,
        use,
        jnp.asarray(np.asarray(rotation_counts, dtype=np.int32)),
        table,
        eulers_f32,
        jnp.asarray(left),
        jnp.asarray(n_images),
        None if right is None else jnp.asarray(np.asarray(right, dtype=np.float32)),
        r_pad=int(r_pad),
        group=max(1, int(np.sqrt(_UNDECIDED_ROTATION_CALL_BYTES / (36.0 * slots * capacity)))),
        pass1_rotations=pass1_rotations,
    )


@partial(
    jax.jit,
    static_argnames=("r_pad", "group", "pass1_rotations"),
)
def _lay_out_undecided_rotations(
    ids, use, counts, table, eulers_f32, left, n_images, right_f32, *, r_pad: int, group: int, pass1_rotations,
):
    """:func:`_undecided_scorer_rotations` as one program: ``left`` ``[P, S, 3, 3]`` float32 per slot, ``table``
    ``[P, R_max]`` a local search's own rotations (``None``: the grid's), ``right_f32`` the float32 perturbation
    matrix (``None``: none)."""

    n_particles, capacity = ids.shape
    slots = left.shape[1]
    grid_rows = jnp.minimum(jnp.where(use, ids % r_pad, 0), counts[:, None] - 1)
    if table is not None:
        grid_rows = jnp.take_along_axis(table, grid_rows, axis=1)
    particle_eulers = eulers_f32[grid_rows]  # [P, M, 3]
    parts = []
    for start in range(0, n_particles, group):
        stop = min(start + group, n_particles)
        g = stop - start
        built = jnp.asarray(
            pass1_rotations(particle_eulers[start:stop].reshape(-1, 3), right_f32, left[start:stop].reshape(-1, 3, 3)),
            dtype=jnp.float32,
        ).reshape(g, slots, g, capacity, 3, 3)
        own = jnp.arange(g)
        parts.append(built[own, :, own])  # [g, S, M, 3, 3]
    rotations = parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=0)
    keep = (jnp.arange(slots)[None, :] < n_images[:, None])[:, :, None] & jnp.any(use, axis=1)[:, None, None]
    return jnp.where(keep[..., None, None], rotations, jnp.zeros((), jnp.float32))


@partial(jax.jit, static_argnames=("r_pad", "current_size", "physical_image_size", "model_max_r", "padding_factor"))
def _exact_rows_diff2(
    diff2, ids, use, row_rotations, projectors_full, unshifted, pixel_weight, initial_diff2, translation_angles,
    full_to_compact, *, r_pad: int, current_size: int, physical_image_size: int, model_max_r: int, padding_factor: int,
):
    """Particles' diff2 ``[P, K * R_pad, T]`` with their undecided rotations ``ids`` ``[P, M]`` (where ``use``)
    scored by the direct square against their class's projector, as one program.

    ``row_rotations`` ``[P, S, M, 3, 3]`` are those rotations' scorer matrices for each tilt image; the other
    operands are the particles' (``[P, S, pixels]``, ``[P, S]``, ``[P, S, T, 2]``).
    """

    for class_index, projector_full in enumerate(projectors_full):
        own = use & (ids // r_pad == class_index)
        exact = direct_rows_diff2(
            projector_full,
            row_rotations,
            unshifted,
            pixel_weight,
            initial_diff2,
            translation_angles,
            full_to_compact,
            current_size=current_size,
            physical_image_size=physical_image_size,
            model_max_r=model_max_r,
            padding_factor=padding_factor,
        )
        diff2 = put_rows(diff2, ids, own, exact)
    return diff2


@partial(jax.jit, static_argnames=("capacity",))
def _take_rescore_group(rows, exact, operands, take, n, *, capacity: int):
    """The rows, operands and diff2 of a rescore group's flush particles ``take`` (``[n_pad]``, the first ``n``
    real), its undecided rotations (``first_rows`` at ``capacity``) and which of them it scores (none on padding).
    One program in place of the group's eager gathers, each compiled for every new group shape (relax#115)."""

    ids, use = first_rows(rows[take], capacity=capacity)
    use = use & (jnp.arange(take.shape[0]) < n)[:, None]
    return ids, use, tuple(value[take] for value in operands), exact[take]


@jax.jit
def _put_rescore_group(exact, scored, group_exact, group_use, take, n):
    """``exact`` with a rescore group's diff2 written back to its real particles, and ``scored`` with their counts
    of rotations scored again; the padding writes past the end, which the scatters drop."""

    back = jnp.where(jnp.arange(take.shape[0]) < n, take, exact.shape[0])
    group_scored = jnp.sum(group_use, axis=1, dtype=jnp.int32)
    return exact.at[back].set(group_exact, mode="drop"), scored.at[back].set(group_scored, mode="drop")


@jax.jit
def _exact_cut_check(scored, unresolved, most, n_real):
    """A flush's exact-cut scalars ``(most, unresolved, rows, particles)``: the largest count of undecided rotations,
    the real particles the leading sort left unresolved, and the rotations scored again and the particles that had
    any (a padded flush's copies are not counted)."""

    real_particle = jnp.arange(scored.shape[0]) < n_real
    real = jnp.where(real_particle, scored, 0)
    n_unresolved = jnp.sum(unresolved & real_particle, dtype=jnp.int32)
    return most, n_unresolved, jnp.sum(real, dtype=jnp.int32), jnp.sum(real > 0, dtype=jnp.int32)


def _coarse_batches(
    rotation_counts,
    *,
    n_slots: int,
    n_trans: int,
    n_pixels: int = 0,
    projection_bytes_per_pixel: int = 8,
    budget_bytes: int = _COARSE_BATCH_BYTES,
):
    """Batches of consecutive particles for the coarse GEMM scorer: ``(units, R_pad, P_pad, slot_block)``.

    One shape for the whole pass, so one program compiles: ``R_pad`` is the largest rotation count
    rounded up to a multiple of 128, and ``P_pad`` particles of
    ``slot_block`` slots per call keep each image's ``R_pad`` projected rows of ``n_pixels`` pixels at
    ``projection_bytes_per_pixel`` (:func:`_coarse_projection_bytes_per_pixel`) and its ``R_pad * T`` float32
    diff2 within ``budget_bytes`` (a particle whose images exceed it is scored ``slot_block`` slots at a time).
    """

    counts = np.asarray(rotation_counts, dtype=np.int64)
    if counts.size == 0:
        return []
    r_pad = -(-int(np.max(counts)) // 128) * 128
    per_image = r_pad * (int(n_trans) * 4 + int(n_pixels) * int(projection_bytes_per_pixel))
    slot_block = int(min(n_slots, max(1, int(budget_bytes) // per_image)))
    p_pad = max(1, int(budget_bytes) // (int(n_slots) * per_image)) if slot_block == n_slots else 1
    p_pad = min(p_pad, int(counts.size))
    return [
        (np.arange(start, min(start + p_pad, counts.size)), r_pad, p_pad, slot_block)
        for start in range(0, int(counts.size), p_pad)
    ]


_OPERAND_IMAGE_BATCH = 1024


# Bytes of coarse image operands (unshifted, pixel weight, initial diff2) held at once: the pass
# prepares them for a block of consecutive particles at a time (a box-320 half's all at once was
# 15 GiB, the final all-data pass at the full box; etbench w2_10_box320).
_COARSE_OPERAND_BLOCK_BYTES = 4 << 30


class CoarsePassBudget(NamedTuple):
    """Device bytes of one coarse pass: its operand block, its GEMM batch and its flushes (with their cuts)."""

    operand_block: int
    batch: int
    flush: int


def _coarse_pass_budget(resident_bytes: int) -> CoarsePassBudget:
    """One budget for the coarse pass, read once: three quarters of what the device can still hand out, less the
    pass's ``resident_bytes`` not yet allocated (each class's full projector for the exact cut).

    The operand block and the GEMM batch take at most a quarter of it each, within their fixed caps; the flushes
    take the rest. The three parts were each a quarter of the free memory read at its own call, which with the
    uncounted operands and cut copies of the flushes ran 16 and 40 GB cards out of memory (relax#28). On an 80 GB
    card the block and the batch keep their fixed caps.
    """

    available = memory_budget.current_device_available_bytes()
    if available is None:
        return CoarsePassBudget(_COARSE_OPERAND_BLOCK_BYTES, _COARSE_BATCH_BYTES, 1 << 62)
    usable = 0.75 * max(float(available) - float(resident_bytes), 0.0)
    operand_block = int(min(_COARSE_OPERAND_BLOCK_BYTES, max(256 << 20, 0.25 * usable)))
    batch = int(min(_COARSE_BATCH_BYTES, max(256 << 20, 0.25 * usable)))
    return CoarsePassBudget(operand_block, batch, int(max(256 << 20, usable - operand_block - batch)))


# A flush's cut, in copies of its summed diff2 live beside the held flushes (the significance's log weights, mask
# and weights, the near-cut search's tables and sort, then the exact scores and their significance). Measured on
# et09 at iteration 100 (H100), every cut on the full sort (sorted_rotations=None, job 15134533): the coarse pass's
# device peak (live bytes plus the near-cut program's temporaries) grew by 167 MB per flush particle between
# flushes of 29 and 7, 6.6 diff2 copies beyond the held flushes and the operand copy (4.1 with the 256-rotation
# sort, job 15131889).
_CUT_DIFF2_COPIES = 7

# The share of a flush's budget left for scoring its undecided rotations again (their scorer matrices and
# per-image direct squares scale with the rotations, not with the diff2): _rescore_rows_per_call.
_RESCORE_SHARE = 0.125


def _flush_plan(
    diff2_bytes: int, operand_bytes: int, *, batch_particles: int, flush_bytes: int, describe: str
) -> tuple[int, int, int, int]:
    """Particles one flush cuts, its rescore bytes, and the diff2 and device bounds it was taken from.

    The flush is whole coarse batches of ``batch_particles`` with at most ``_SIGNIFICANCE_BATCH_BYTES`` of summed
    diff2, and its device bytes within ``flush_bytes`` (``_coarse_pass_budget``). Per particle a flush holds its
    summed diff2 (``diff2_bytes``), its significant-cell mask (a quarter of that) and its operands
    (``operand_bytes``) until it is collected, so two flushes' are held at once; the operands are concatenated once
    more while it is dispatched, and its cut works on ``_CUT_DIFF2_COPIES`` more copies of the diff2, one more while
    its undecided rotations are scored again in groups (each group's diff2 replaces the last). The rescore
    of the undecided rotations takes ``_RESCORE_SHARE`` of ``flush_bytes``. Counting the diff2 alone ran a 16 GB
    V100 out of memory at box 192 with 41 tilts, where the operands dominate (relax#28). The device bound is rounded
    down to a power of two of particles, so the free bytes at plan time give few flush sizes and few programs.
    """

    by_diff2 = max(1, _SIGNIFICANCE_BATCH_BYTES // max(diff2_bytes, 1))
    # The rescore's groups each write a new diff2 while the previous group's is alive: one copy beyond the cut's.
    held = 2 * (diff2_bytes + diff2_bytes // 4 + operand_bytes) + operand_bytes + (_CUT_DIFF2_COPIES + 1) * diff2_bytes
    rescore_bytes = int(_RESCORE_SHARE * int(flush_bytes))
    by_device = (int(flush_bytes) - rescore_bytes) // max(held, 1)
    if by_device < 1:
        raise MemoryError(
            f"The tomo coarse pass needs {held / 2**30:.2f} GiB of device memory per particle ({describe}) and this "
            f"GPU can give its flushes {(int(flush_bytes) - rescore_bytes) / 2**30:.2f} GiB: run on a GPU with "
            "more memory or with a smaller box."
        )
    power = pow2_floor(by_device)
    batch = max(int(batch_particles), 1)
    return min(by_diff2, max(batch, power // batch * batch)), rescore_bytes, by_diff2, int(by_device)


def _rescore_rows_per_call(particles: int, slots: int, n_trans: int, rescore_bytes: int, capacity: int) -> int:
    """Undecided rotations per particle scored again in one call, a power of two within ``rescore_bytes``.

    Per particle and rotation a call holds its slots' scorer matrices (36 B each), their direct squares and the
    translation blocks' concatenation (twice ``n_trans`` float32 per slot), and the particle's sum and the
    flush's updated rows (twice ``n_trans`` float32). A flat K > 1 posterior leaves thousands of rotations
    undecided, whose matrices alone were gigabytes in one call.
    """

    per_row = int(particles) * (int(slots) * (36 + 2 * int(n_trans) * 4) + 2 * int(n_trans) * 4)
    rows = max(1, int(rescore_bytes) // max(per_row, 1))
    return min(int(capacity), pow2_floor(rows))


def _coarse_projection_bytes_per_pixel(
    class_textures, projector_dtype, *, box_size, current_size, model_max_r
) -> int:
    """Device bytes one projected score pixel takes in :func:`coarse_gemm_slot_block`.

    The texture's packed float32 ``[Re | Im]`` rows take 8 bytes. The JAX projection the pass falls back to
    (no texture: a complex128 projector, or a window the packed rows do not serve) holds its rows in the
    projector's complex dtype and a temporary of the same size: 6.42 GiB for 164 images x 4608 rotations x
    (81 translations x 4 B + 278 pixels x 32 B) with a complex128 projector (Polar 413469).
    """

    from relax.projection.projection import relion_coarse_packed_rows_serve

    if all(t is not None for t in class_textures) and relion_coarse_packed_rows_serve(
        int(box_size), int(current_size), int(model_max_r)
    ):
        return 8
    return 2 * int(np.dtype(projector_dtype).itemsize)


@partial(jax.jit, donate_argnums=(0,))
def _put_rows(buffer, values, start):
    """``buffer`` with ``values`` written at row ``start``, in place (the buffer is donated)."""

    return jax.lax.dynamic_update_slice_in_dim(buffer, values.astype(buffer.dtype), start, axis=0)


def _all_image_coarse_operands(
    experiment_dataset,
    image_start: int,
    image_stop: int,
    layout,
    *,
    noise_variance_half,
    optics_group_ids,
    scale_corrections,
):
    """:func:`tilt_image_coarse_operands` of dataset images ``image_start:image_stop``, in batches of ``_OPERAND_IMAGE_BATCH``.

    The last batch is padded with repeats of its last image, so every batch runs the same programs.
    The batches are written into buffers of whole batches (rows past ``image_stop - image_start`` are
    padding), so the block is held once, not once in parts and once concatenated.
    """

    from relax.io.batch_fetch import fetch_indexed_batch, prefetched_batches

    n_images = int(image_stop) - int(image_start)
    n_rows = -(-n_images // _OPERAND_IMAGE_BATCH) * _OPERAND_IMAGE_BATCH
    starts = range(0, n_images, _OPERAND_IMAGE_BATCH)
    batches = []
    for offset in starts:
        indices = np.arange(
            int(image_start) + offset, min(int(image_start) + offset + _OPERAND_IMAGE_BATCH, int(image_stop))
        )
        batches.append(np.concatenate([indices, np.full(_OPERAND_IMAGE_BATCH - indices.size, indices[-1])]))
    buffers = None
    # The host image reads run one batch ahead, on a worker thread (a local pass read its blocks on the main
    # thread, with the device idle: 22 s of a healpix-8 local iteration of the real tilt-series case, py-spy).
    reads = (fetch_indexed_batch(experiment_dataset, padded) for padded in batches)
    with prefetched_batches(reads, depth=1) as fetched_batches:
        for offset, padded, fetched in zip(starts, batches, fetched_batches, strict=True):
            values = tilt_image_coarse_operands(
                experiment_dataset,
                padded,
                layout,
                noise_variance_half=noise_variance_half,
                optics_group_ids=optics_group_ids,
                scale_corrections=scale_corrections,
                fetched=fetched,
            )
            if buffers is None:
                buffers = [jnp.zeros((n_rows,) + v.shape[1:], dtype=v.dtype) for v in values]
            buffers = [_put_rows(b, v, jnp.int32(offset)) for b, v in zip(buffers, values)]
    return tuple(buffers)


class SharedFrames(NamedTuple):
    """The tilt frames a global coarse pass shares (:func:`shared_frames`): images with one projection left matrix."""

    image_key: np.ndarray  # [I] each image's frame
    key_rank: np.ndarray  # [F] each frame's place in an order that every particle's slots follow
    key_image: np.ndarray  # [F] an image of each frame
    unit_order: np.ndarray  # [U] the particles, those sharing frames (one tomogram's) next to each other


def shared_frames(image_left, unit_image_offsets) -> SharedFrames | None:
    """The frames of the tilt images with bitwise-equal projection left matrices, when they hold two images each on
    average, else None.

    The images of one tomogram's tilt frame have one left matrix (``Aproj`` with the optics group's
    magnification), so their coarse scorer matrices and projections are the same: the pass projects each frame once
    for a group of particles. A particle's images add in its slot (``img_id``) order, so the frames are ranked in an
    order every particle's slots follow (None when there is none, or a particle has two images in one frame).
    """

    offsets = np.asarray(unit_image_offsets, dtype=np.int64)
    # The matrices' bits: -0.0 and 0.0 are different frames.
    left = np.ascontiguousarray(np.asarray(image_left, dtype=np.float64).reshape(-1, 9)).view(np.uint64)
    n_images, n_units = left.shape[0], offsets.size - 1
    if n_images == 0:
        return None
    _rows, key_image, image_key = np.unique(left, axis=0, return_index=True, return_inverse=True)
    image_key = image_key.reshape(-1).astype(np.int64)
    n_keys = int(key_image.size)
    if n_images < 2 * n_keys:
        return None
    unit_of_image = np.repeat(np.arange(n_units), np.diff(offsets))
    if np.unique(unit_of_image * n_keys + image_key).size != n_images:
        return None  # a particle with two images in one frame
    # Each particle's consecutive slots: its frame order. Kahn's order over them, the earliest image first on ties.
    same = unit_of_image[1:] == unit_of_image[:-1]
    edges = np.unique(np.stack([image_key[:-1][same], image_key[1:][same]], axis=1), axis=0)
    indegree = np.bincount(edges[:, 1], minlength=n_keys)
    successors = np.split(edges[:, 1], np.cumsum(np.bincount(edges[:, 0], minlength=n_keys))[:-1])

    ready = [(int(key_image[k]), k) for k in np.flatnonzero(indegree == 0)]
    heapq.heapify(ready)
    key_rank = np.full(n_keys, -1, dtype=np.int64)
    rank = 0
    while ready:
        _first, key = heapq.heappop(ready)
        key_rank[key] = rank
        rank += 1
        for nxt in successors[key]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                heapq.heappush(ready, (int(key_image[nxt]), int(nxt)))
    if rank != n_keys:
        return None  # two particles' slots order two frames both ways
    # The frames a particle's slots link are one component (one tomogram's): its particles go together.
    parent = np.arange(n_keys)

    def root(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for a, b in edges:
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    component = np.asarray([root(k) for k in range(n_keys)])[image_key[offsets[:-1]]] if n_units else np.zeros(0, int)
    first_unit = {}
    for unit, c in enumerate(component):
        first_unit.setdefault(int(c), unit)
    unit_order = np.lexsort((np.arange(n_units), np.asarray([first_unit[int(c)] for c in component])))
    return SharedFrames(image_key, key_rank, key_image.astype(np.int64), unit_order.astype(np.int64))


def _operand_batches(image_indices):
    """``image_indices`` in batches of ``_OPERAND_IMAGE_BATCH``, the last padded with its last image (so every
    batch runs the same programs)."""

    image_indices = np.asarray(image_indices, dtype=np.int64)
    return [
        np.concatenate([indices, np.full(_OPERAND_IMAGE_BATCH - indices.size, indices[-1])])
        for indices in (
            image_indices[start : start + _OPERAND_IMAGE_BATCH]
            for start in range(0, image_indices.size, _OPERAND_IMAGE_BATCH)
        )
    ]


def _indexed_image_coarse_operands(
    experiment_dataset, image_indices, layout, *, noise_variance_half, optics_group_ids, scale_corrections, fetched=None
):
    """:func:`tilt_image_coarse_operands` of the dataset images ``image_indices``, in the batches of
    :func:`_operand_batches`, rows past the images included (repeats of the last one): whole batches give the
    programs that read them few shapes. ``fetched`` holds the host reads when they were made ahead."""

    parts = []
    for i, padded in enumerate(_operand_batches(image_indices)):
        parts.append(
            tilt_image_coarse_operands(
                experiment_dataset,
                padded,
                layout,
                noise_variance_half=noise_variance_half,
                optics_group_ids=optics_group_ids,
                scale_corrections=scale_corrections,
                fetched=None if fetched is None else fetched[i],
            )
        )
    return tuple(values[0] if len(parts) == 1 else jnp.concatenate(values, axis=0) for values in zip(*parts))


def _group_image_reads(experiment_dataset, plan, offsets):
    """Each frame group's host image reads (:func:`relax.io.batch_fetch.fetch_indexed_batch`), one list of its
    operand batches per group, in group order; nothing without a plan."""

    from relax.io.batch_fetch import fetch_indexed_batch

    for units in [] if plan is None else plan.groups:
        images = np.concatenate([np.arange(offsets[unit], offsets[unit + 1]) for unit in units])
        yield [fetch_indexed_batch(experiment_dataset, padded) for padded in _operand_batches(images)]


@jax.jit
def _lay_out_frame_rotations(built, rot):
    """``built[:, rot]`` ``[F, R_pad, 3, 3]``: each frame's rotations at the pass's padded rotation rows."""

    return built[:, rot]


def particle_coarse_significance(
    particle_diff2,
    rotation_log_prior,
    translation_log_prior,
    *,
    adaptive_fraction,
    max_significants,
):
    """RELION's coarse weights and significance of particles from their summed diff2 ``[P, R, T]``.

    The subtomogram coarse pass converts the particle's summed diff2 exactly as the SPA pass
    converts one image's (convertAllSquaredDifferencesToWeights, acc_ml_optimiser_impl.h:2245-2345):
    the log weight is ``log prior + min_diff2 - diff2`` in float32, then RELION's sort, scan and
    tail cut. ``rotation_log_prior`` is ``[R]``, ``[P, R]`` (each particle's own, a local search) or
    ``None``, and ``translation_log_prior`` ``[P, T]``, the particle's 3D offset prior. The SPA
    posterior primitive is reused, so the two paths cannot drift apart. Returns its statistics dict
    (``mask [P, R * T]``, ``n_significant``, ``pmax``, ``winner``, ...) and the ``log_weights`` it cut.
    """

    from relax.sampling.oversampling import relion_cuda_f32_coarse_log_weights
    from relax.scoring.coarse_publication import posterior_statistics

    # RELION's left-to-right float32 order pdf_orientation + pdf_offset + min_diff2 - diff2
    # (cuda_kernel_weights_exponent_coarse). Adding the priors to the absolute scores first and the
    # min_diff2 offset afterwards rounds differently and can move near-tie cells across the
    # significance cut (relax.sampling.oversampling.relion_cuda_f32_coarse_log_weights).
    raw = -jnp.asarray(particle_diff2, dtype=jnp.float32)
    n_particles = int(raw.shape[0])
    rotation_prior = (
        jnp.zeros(raw.shape[1], jnp.float32) if rotation_log_prior is None else jnp.asarray(rotation_log_prior)
    )
    values = relion_cuda_f32_coarse_log_weights(
        raw, rotation_prior, jnp.asarray(translation_log_prior, jnp.float32)
    ).reshape(n_particles, -1)
    # The log weights already carry min_diff2: no further offset.
    raw_max = jnp.zeros(n_particles, jnp.float32)
    statistics = posterior_statistics(
        values,
        raw_max,
        None,
        adaptive_fraction=float(adaptive_fraction),
        max_significants=None if max_significants is None or int(max_significants) <= 0 else int(max_significants),
        tie_score_ulps=0,
    )
    return {**statistics, "log_weights": values}


def _batch_scoring_rotations(
    units,
    offsets,
    image_left,
    coarse_eulers_deg,
    unit_rotation_rows,
    random_perturbation,
    angular_sampling_deg,
    *,
    p_pad: int,
    slots: int,
    r_pad: int,
    pass1_rotations,
) -> jax.Array:
    """A coarse batch's scorer matrices on the device, ``[P_pad, S, R_pad, 3, 3]`` float32.

    Every tilt image has its own matrices, ``make_eulers_3D`` with its left matrix
    (:func:`relax.sampling.device_rotations.relion_adaptive_pass1_rotations`). A global search builds the whole
    batch's images in one call; a local search builds each particle's own rotations. A particle's
    rotations beyond its own count repeat its last one, and padded slots and particles are zero.
    """

    from relax.sampling import tomo_particles

    counts = np.diff(offsets)[np.asarray(units)]
    slot = np.arange(slots)[None, :]
    if unit_rotation_rows is None:
        images = np.concatenate([np.arange(offsets[unit], offsets[unit + 1]) for unit in units])
        left, _applies = tomo_particles.relion_left_matrices(image_left[images])
        built = jnp.asarray(
            pass1_rotations(coarse_eulers_deg, random_perturbation, angular_sampling_deg, left_matrices=left),
            dtype=jnp.float32,
        )
        n_rot = int(built.shape[1])
        # Each slot's image row in ``built`` (padded slots read row 0 and are zeroed below).
        rows = np.zeros((p_pad, slots), dtype=np.int64)
        rows[: len(units)] = np.where(slot < counts[:, None], (np.cumsum(counts) - counts)[:, None] + slot, 0)
        rot = np.minimum(np.arange(r_pad), n_rot - 1)
        full_valid = np.zeros((p_pad, slots), dtype=bool)
        full_valid[: len(units)] = slot < counts[:, None]
        return _lay_out_rotations(built, jnp.asarray(rows), jnp.asarray(rot), jnp.asarray(full_valid))
    else:
        # Each particle's rotations at the pass's shape: its rows padded to R_pad with its last one, its images'
        # left matrices padded to S slots with the identity, so the rotation kernel takes one shape for the pass
        # (it builds each (image, row) matrix on its own; padded entries are zeroed below). Particles' own counts
        # of tilts and rotations compiled a call and its eager layout per pair (~8 dispatches per particle).
        eulers = np.asarray(coarse_eulers_deg)
        left = np.tile(np.eye(3), (len(units), slots, 1, 1))
        built = []
        for p, (unit, unit_rows) in enumerate(zip(units, unit_rotation_rows)):
            unit_rows = np.asarray(unit_rows)
            left[p, : counts[p]] = tomo_particles.relion_left_matrices(image_left[offsets[unit] : offsets[unit + 1]])[0]
            built.append(
                pass1_rotations(
                    eulers[unit_rows[np.minimum(np.arange(r_pad), unit_rows.size - 1)]],
                    random_perturbation,
                    angular_sampling_deg,
                    left_matrices=left[p],
                )
            )
        valid = np.zeros((p_pad, slots), dtype=bool)
        valid[: len(units)] = slot < counts[:, None]
        return _stack_particle_rotations(tuple(built), jnp.asarray(valid))


@jax.jit
def _stack_particle_rotations(built, valid):
    """The particles' ``[S, R_pad, 3, 3]`` matrices as one ``[P_pad, S, R_pad, 3, 3]`` float32 array, zero in invalid
    slots and padded particles (``valid`` ``[P_pad, S]``), as one device program."""

    built = tuple(jnp.asarray(b, dtype=jnp.float32) for b in built)
    stacked = jnp.stack(built + (built[0],) * (valid.shape[0] - len(built)))
    return jnp.where(valid[:, :, None, None, None], stacked, jnp.zeros((), jnp.float32))


@jax.jit
def _lay_out_rotations(built, rows, rot, valid):
    """``built[rows, rot]`` with invalid slots zeroed, as one device program (one output buffer)."""

    laid_out = built[rows[:, :, None], rot[None, None, :]]
    return jnp.where(valid[:, :, None, None, None], laid_out, jnp.zeros((), laid_out.dtype))


class TiltCoarseInputs(NamedTuple):
    """The tilt images' geometry and the pass's rotations, as :func:`particle_coarse_supports` takes them."""

    offsets: np.ndarray
    image_left: np.ndarray
    image_projections: np.ndarray
    old_offsets: np.ndarray  # RELION's rounded old offsets per particle (tomo_particles.relion_gpu_old_offsets)
    coarse_translations_px: np.ndarray
    box_size: int
    coarse_eulers_deg: np.ndarray
    random_perturbation: float
    angular_sampling_deg: float


class CoarseScorer(NamedTuple):
    """Each class's projector and texture, the score pixels and the static settings of the GEMM programs."""

    projectors: tuple
    textures: list
    score_indices: jax.Array
    statics: dict


def _fill_tilt_slots(units, tilt: TiltCoarseInputs, starts, *, n_pad, slots, n_trans):
    """The batch's per-slot translation angles ``[n_pad, S, T, 2]`` float32, image indices ``[n_pad, S]`` int64 and
    validity ``[n_pad, S]`` bool: particle ``p``'s tilt images fill its first slots, indexed from ``starts[p]``;
    padded slots and particles are zero and invalid."""

    from relax.sampling import tomo_particles

    angles = np.zeros((n_pad, slots, n_trans, 2), dtype=np.float32)
    image_index = np.zeros((n_pad, slots), dtype=np.int64)
    image_valid = np.zeros((n_pad, slots), dtype=bool)
    for p, (unit, start) in enumerate(zip(units, starts)):
        images = np.arange(tilt.offsets[unit], tilt.offsets[unit + 1])
        angles[p, : images.size] = tomo_particles.tilt_translation_angles(
            tilt.coarse_translations_px,
            tilt.old_offsets[unit : unit + 1],
            tilt.image_projections[images],
            np.zeros(images.size, int),
            tilt.box_size,
        )
        image_index[p, : images.size] = np.arange(start, start + images.size)
        image_valid[p, : images.size] = True
    return angles, image_index, image_valid


@jax.jit
def _gather_slot_operands(unshifted, weight, initial, index, image_valid, angles):
    """The batch's padded device operands ``(unshifted, weight, initial, angles)`` ``[n_pad, S, ...]``: the operand
    rows at ``index``, zero in invalid slots, as one device program (eagerly, four dispatches and a transfer per
    batch: ~10 ms a particle of a local pass's host time, one particle per batch)."""

    valid = jnp.asarray(image_valid)
    batch_unshifted = jnp.where(valid[..., None], unshifted[index], jnp.zeros((), unshifted.dtype))
    batch_weight = jnp.where(valid[..., None], weight[index], jnp.zeros((), weight.dtype))
    batch_initial = jnp.where(valid, initial[index], jnp.zeros((), initial.dtype))
    return batch_unshifted, batch_weight, batch_initial, jnp.asarray(angles)


def _scored_batch(class_totals, error, image_valid, batch_operands, n, total):
    """A scored batch's result for its ``n`` real particles: diff2 ``[n, K * R_pad, T]`` (class-major along the
    rotation axis), error bounds, image counts, operands and the last total."""

    # K>1: [P, K * R_pad, T], class-major along the rotation axis.
    diff2 = class_totals[0] if len(class_totals) == 1 else jnp.concatenate(class_totals, axis=1)
    operands = tuple(operand[:n] for operand in batch_operands)
    return diff2, error[:n], image_valid[:n].sum(axis=1), operands, total


def _score_slot_batch(
    units, tilt: TiltCoarseInputs, operands_for, scorer: CoarseScorer, *, r_pad, p_pad, slots, slot_block,
    slot_offsets, unit_rotations,
):
    """A coarse batch's per-particle diff2 ``[P, K * R_pad, T]``, GEMM error bounds ``[P]``, image counts, operands
    (``[P, S, ...]``, as the exact cut takes them) and its last total, by the slot-block programs.

    The batch's operands are padded to ``[P_pad, S, R_pad, ...]``: padded rotations repeat the particle's last one,
    padded slots and particles carry zero weight (their diff2 adds zeros and is not read). The scorer matrices are
    built and laid out on the device (:func:`_batch_scoring_rotations`); ``operands_for(units)`` gives the operand
    block holding the batch's images; ``unit_rotations`` are a local search's rows per particle (None: the grid's).
    """

    from relax.sampling.device_rotations import relion_adaptive_pass1_rotations

    n_trans = int(np.asarray(tilt.coarse_translations_px).shape[0])
    rotations = _batch_scoring_rotations(
        units,
        tilt.offsets,
        tilt.image_left,
        tilt.coarse_eulers_deg,
        unit_rotations,
        tilt.random_perturbation,
        tilt.angular_sampling_deg,
        p_pad=p_pad,
        slots=slots,
        r_pad=r_pad,
        pass1_rotations=relion_adaptive_pass1_rotations,
    )
    angles, image_index, image_valid = _fill_tilt_slots(
        units, tilt, tilt.offsets[units], n_pad=p_pad, slots=slots, n_trans=n_trans
    )
    block_start, _block_stop, unshifted, weight, initial = operands_for(units)
    index = jnp.asarray(np.where(image_valid, image_index - block_start, 0))
    batch_operands = _gather_slot_operands(unshifted, weight, initial, index, image_valid, angles)
    batch_unshifted, batch_weight, batch_initial, batch_angles = batch_operands
    class_totals, error, total = [], None, None
    for projector, texture in zip(scorer.projectors, scorer.textures):
        total = jnp.zeros((p_pad, n_trans, r_pad), dtype=jnp.float32)
        class_error = jnp.zeros(p_pad, dtype=jnp.float32)
        for first in range(0, slots, slot_block):
            total, class_error = coarse_gemm_slot_block(
                total,
                class_error,
                projector,
                rotations,
                batch_unshifted,
                batch_weight,
                batch_initial,
                batch_angles,
                scorer.score_indices,
                slot_offsets[first],
                count=int(min(slot_block, slots - first)),
                texture=texture,
                **scorer.statics,
            )
        class_totals.append(total[: units.size].swapaxes(1, 2))
        error = class_error if error is None else jnp.maximum(error, class_error)
    # The block and the batch's padded operands are freed on return: only operands_for and the pending particles
    # keep operands, so a new block is allocated after the old one is freed.
    return _scored_batch(class_totals, error, image_valid, batch_operands, units.size, total)


class FrameGroupPlan(NamedTuple):
    """A global coarse pass's frame-shared schedule (:func:`_frame_group_plan`)."""

    frames: SharedFrames
    groups: list  # the particles of each group, one flush each, one tomogram's together
    group_size: int  # the flush size every group's arrays are padded to
    images_per_call: int  # a frame's images scored per program call
    slot_block: int  # the slot blocks the per-image programs sum the error bounds in
    frame_ids: list  # device int32 scalars: a group's frame positions


def _frame_group_plan(
    tilt: TiltCoarseInputs, *, flush_size, slots, slot_block, r_pad, n_trans, n_score_pixels, projection_bytes, batch_bytes
) -> FrameGroupPlan | None:
    """The frame-shared schedule of a global pass whose images share frames (:func:`shared_frames`), else None.

    The particles go in groups of one flush, one tomogram's together; each frame of a group is projected once and
    its images scored ``images_per_call`` at a time, as many as the batch budget holds (``batch_bytes``: a frame's
    projections at ``projection_bytes`` per score pixel and, per image, its diff2, scores and cross terms).
    """

    frames = shared_frames(tilt.image_left, tilt.offsets)
    if frames is None:
        return None
    per_image = n_trans * r_pad * 4 * 3 + n_trans * n_score_pixels * 8 * 2
    per_frame = r_pad * n_score_pixels * projection_bytes * 3 // 2
    images_per_call = min(flush_size, max(1, (batch_bytes - per_frame) // per_image))
    n_units = tilt.offsets.size - 1
    n_frames = pow2_ceil(min(frames.key_rank.size, slots * flush_size))
    logger.info(
        "Coarse pass shares each tilt frame's projections: %d images in %d frames; particles in groups of %d, up to "
        "%d images per frame call", int(frames.image_key.size), int(frames.key_rank.size), flush_size, images_per_call,
    )
    return FrameGroupPlan(
        frames,
        [frames.unit_order[start : start + flush_size] for start in range(0, n_units, flush_size)],
        int(flush_size),
        images_per_call,
        int(slot_block),
        [jax.device_put(np.int32(i)) for i in range(n_frames)],
    )


def _frame_group_calls(plan: FrameGroupPlan, group_frames, image_frame, entry_unit, entry_slot):
    """The group's frame program calls, frames in slot order: ``(frame position, unit index, slot index)``, each
    call's images padded to a power of two (at most ``images_per_call``) with the group size as their unit."""

    calls = []
    for f, frame in enumerate(group_frames):
        entries = np.flatnonzero(image_frame == frame)
        for start in range(0, entries.size, plan.images_per_call):
            chunk = entries[start : start + plan.images_per_call]
            n_pad = min(pow2_ceil(int(chunk.size)), plan.images_per_call)
            unit_index = np.full(n_pad, plan.group_size, dtype=np.int32)
            slot_index = np.zeros(n_pad, dtype=np.int32)
            unit_index[: chunk.size] = entry_unit[chunk]
            slot_index[: chunk.size] = entry_slot[chunk]
            calls.append((plan.frame_ids[f], jnp.asarray(unit_index), jnp.asarray(slot_index)))
    return calls


def _score_frame_group(units, plan: FrameGroupPlan, tilt: TiltCoarseInputs, operands_of, scorer: CoarseScorer, *, r_pad, slots):
    """A group's per-particle diff2 ``[G, K * R_pad, T]``, GEMM error bounds ``[G]``, image counts, operands
    (``[G, S, ...]``, as the exact cut takes them) and its last total, scored frame by frame.

    ``operands_of(images)`` gives the dataset images' coarse operands (:func:`_indexed_image_coarse_operands`).
    Each frame's scorer matrices are built once (``make_eulers_3D`` with the frame's left matrix) and its
    projections made once per call of :func:`relax.scoring.tomo_coarse_gemm.coarse_gemm_frame`; the error bounds
    add in the slot blocks of the per-image programs (:func:`relax.scoring.tomo_coarse_gemm.slot_block_error`).
    """

    from relax.sampling import tomo_particles
    from relax.sampling.device_rotations import relion_adaptive_pass1_rotations

    offsets, g_pad = tilt.offsets, plan.group_size
    n_trans = int(np.asarray(tilt.coarse_translations_px).shape[0])
    images = np.concatenate([np.arange(offsets[unit], offsets[unit + 1]) for unit in units])
    unshifted, weight, initial = operands_of(images)
    # Each particle's slots index its images in ``images``, which lists the particles' images in order.
    counts = offsets[units + 1] - offsets[units]
    angles, image_index, image_valid = _fill_tilt_slots(
        units, tilt, np.cumsum(counts) - counts, n_pad=g_pad, slots=slots, n_trans=n_trans
    )
    # Each image's particle and slot, in the order of ``images``.
    entry_unit, entry_slot = (np.asarray(axis, dtype=np.int32) for axis in np.nonzero(image_valid))
    batch_operands = _gather_slot_operands(
        unshifted, weight, initial, jnp.asarray(image_index), image_valid, angles
    )
    batch_unshifted, batch_weight, batch_initial, batch_angles = batch_operands
    # The group's frames in an order every particle's slots follow, each frame's scorer matrices built once.
    image_frame = plan.frames.image_key[images]
    group_frames = np.unique(image_frame)
    group_frames = group_frames[np.argsort(plan.frames.key_rank[group_frames], kind="stable")]
    # The frames' left matrices padded to a power of two with the last one, so the builds take few shapes.
    n_frames = pow2_ceil(int(group_frames.size))
    frame_images = plan.frames.key_image[group_frames]
    frame_images = np.concatenate([frame_images, np.repeat(frame_images[-1:], n_frames - frame_images.size)])
    left, _applies = tomo_particles.relion_left_matrices(tilt.image_left[frame_images])
    built = jnp.asarray(
        relion_adaptive_pass1_rotations(
            tilt.coarse_eulers_deg, tilt.random_perturbation, tilt.angular_sampling_deg, left_matrices=left
        ),
        dtype=jnp.float32,
    )
    frame_rotations = _lay_out_frame_rotations(built, jnp.asarray(np.minimum(np.arange(r_pad), int(built.shape[1]) - 1)))
    calls = _frame_group_calls(plan, group_frames, image_frame, entry_unit, entry_slot)
    class_totals, error, total = [], None, None
    for projector, texture in zip(scorer.projectors, scorer.textures):
        total = jnp.zeros((g_pad, n_trans, r_pad), dtype=jnp.float32)
        bounds = jnp.zeros((g_pad, slots), dtype=jnp.float32)
        for frame_id, unit_index, slot_index in calls:
            total, bounds = coarse_gemm_frame(
                total, bounds, projector, frame_rotations, frame_id, batch_unshifted, batch_weight, batch_initial,
                batch_angles, scorer.score_indices, unit_index, slot_index, texture=texture, **scorer.statics,
            )
        class_error = slot_block_error(bounds, slot_block=plan.slot_block)
        class_totals.append(total[: units.size].swapaxes(1, 2))
        error = class_error if error is None else jnp.maximum(error, class_error)
    return _scored_batch(class_totals, error, image_valid, batch_operands, units.size, total)


def particle_coarse_supports(
    experiment_dataset,
    *,
    unit_image_offsets,
    image_projections,
    image_left,
    unit_old_offsets_px,
    coarse_eulers_deg,
    random_perturbation,
    angular_sampling_deg,
    coarse_translations_px,
    projector_half,
    layout: CoarseScoreLayout,
    noise_variance_half,
    rotation_log_prior,
    unit_translation_log_prior,
    adaptive_fraction,
    max_significants,
    model_max_r: int,
    padding_factor: int,
    box_size: int,
    optics_group_ids=None,
    scale_corrections=None,
    unit_rotation_ids=None,
    unit_rotation_log_priors=None,
    empty_classes=(),
):
    """Each particle's coarse significant samples, ``rot * T + t`` int32 ids per unit, its coarse Pmax and its
    significant-sample count as RELION records it.

    RELION's GPU coarse pass for a subtomogram (acc_ml_optimiser_impl.h:1190-1402): every tilt image is
    scored with its own device matrices (``make_eulers_3D`` with the image's left matrix ``image_left``,
    its ``Aproj`` with the optics group's magnification, :func:`relax.steps.tomo_half.tilt_left_matrices`;
    :func:`relax.sampling.device_rotations.relion_adaptive_pass1_rotations`), its phases for the 3D trial shifts of ``Aproj``
    (``image_projections``) plus
    the rounded old offset (:func:`relax.sampling.tomo_particles.tilt_translation_angles`) and its own
    CTF and noise; the images' diff2 is summed in ``img_id`` order and the particle's weights are cut
    once (:func:`particle_coarse_significance`). Dataset images ``unit_image_offsets[u]:[u+1]`` are
    particle ``u``'s, in ``img_id`` order.

    A local search scores each particle over its own rotations only: ``unit_rotation_ids[u]`` (ascending
    rows of ``coarse_eulers_deg``) with the log prior ``unit_rotation_log_priors[u]`` (RELION's
    orientations with a nonzero prior, selectOrientationsWithNonZeroPriorProbability); the returned
    ids keep indexing ``coarse_eulers_deg``.

    Class3D (K>1) passes a tuple of K ``projector_full`` and ``rotation_log_prior`` ``[K, R]`` (each class's
    ``log pdf_class`` folded into its direction prior): every image is scored against each class, and the
    particle's weights over (class, rotation, translation) are cut jointly, as RELION's
    convertAllSquaredDifferencesToWeights sorts all classes' weights of the particle together
    (acc_ml_optimiser_impl.h:2245-2345). The supports are then a list per class.

    Class3D under an orientational prior (``--sigma_ang``, ``--sigma_tilt``) passes the K projectors with the
    particles' own rotations: RELION selects one orientation list per particle and scores every class on it,
    and its orientation weight is then ``directions_prior * psi_prior`` alone, without ``pdf_class`` or
    ``pdf_direction`` (acc_ml_optimiser_impl.h:1960-1972). ``empty_classes`` are the classes it skips
    (``pdf_class == 0``): they keep no sample.
    The count is RELION's ``filteredSize - thresholdIdx`` (acc_ml_optimiser_impl.h:3236): a sample tied with the
    cut weight below the cut's rank is significant but not counted.
    """

    from relax.cuda.preprocess import deferred_relion_preprocess_checks
    from relax.io.batch_fetch import prefetched_batches
    from relax.projection.projection import relion_projector_half_to_texture_full
    from relax.sampling import tomo_particles
    from relax.sampling.device_rotations import pass1_rotations_left_f32

    class_projectors = tuple(projector_half) if isinstance(projector_half, (tuple, list)) else (projector_half,)
    n_classes = len(class_projectors)
    if n_classes > 1 and unit_rotation_ids is None and (
        rotation_log_prior is None or np.ndim(rotation_log_prior) != 2
    ):
        raise ValueError("a global K-class coarse pass takes one rotation log prior per class, [K, R]")
    empty_classes = tuple(int(k) for k in empty_classes)
    if empty_classes and (unit_rotation_ids is None or n_classes == 1):
        raise ValueError("a global pass carries an empty class in its rotation log prior")
    scored_classes = [k for k in range(n_classes) if k not in empty_classes]
    offsets = np.asarray(unit_image_offsets, dtype=np.int64)
    image_projections = np.asarray(image_projections, dtype=np.float64)
    image_left = np.asarray(image_left, dtype=np.float64)
    old = tomo_particles.relion_gpu_old_offsets(np.asarray(unit_old_offsets_px, dtype=np.float64))
    local = unit_rotation_ids is not None
    if local != (unit_rotation_log_priors is not None) or (local and rotation_log_prior is not None):
        raise ValueError("a local search takes each particle's rotations and priors, and no shared prior")
    coarse_eulers_deg = np.asarray(coarse_eulers_deg)
    n_coarse_trans = int(np.asarray(coarse_translations_px).shape[0])
    n_units = int(offsets.size - 1)
    slots = int(np.max(np.diff(offsets))) if n_units else 1
    # The tilt images' coarse operands, in fixed image batches (one preprocessing program), prepared for
    # a block of consecutive particles at a time (_COARSE_OPERAND_BLOCK_BYTES); the coarse batches are
    # consecutive particles, so each batch reads one block.
    operand_bytes_per_image = int(layout.score_indices_np.size) * (8 + 4) + 4
    operand_block = None  # (first image, image stop, unshifted, weight, initial)

    def operands_for(units):
        nonlocal operand_block
        lo, hi = int(offsets[units[0]]), int(offsets[units[-1] + 1])
        if operand_block is None or lo < operand_block[0] or hi > operand_block[1]:
            operand_block = None
            stop_unit = int(np.searchsorted(offsets, lo + block_images, side="right")) - 1
            stop = int(offsets[max(stop_unit, int(units[-1]) + 1)])
            operand_block = (lo, stop) + tuple(
                _all_image_coarse_operands(
                    experiment_dataset,
                    lo,
                    stop,
                    layout,
                    noise_variance_half=noise_variance_half,
                    optics_group_ids=optics_group_ids,
                    scale_corrections=scale_corrections,
                )
            )
        return operand_block

    unit_rotations = [None if not local else np.asarray(unit_rotation_ids[u], dtype=np.int64) for u in range(n_units)]
    for u, rows in enumerate(unit_rotations):
        if local and (rows.size == 0 or np.any(np.diff(rows) <= 0)):
            raise ValueError(f"particle {u}'s local rotations must be ascending and non-empty")
    rotation_counts = [coarse_eulers_deg.shape[0] if rows is None else rows.size for rows in unit_rotations]
    # Each class's projector texture, staged once for the pass (the SPA coarse path's capacity texture).
    class_textures = [
        _coarse_capacity_texture(
            class_projector, layout, model_max_r=int(model_max_r), padding_factor=int(padding_factor), class_index=k
        )
        for k, class_projector in enumerate(class_projectors)
    ]
    # One budget for the pass, less the full projectors the exact cut builds for each class on first use (complex64
    # cubes kept for the pass, and one more cube and the cast half while one is built).
    cube_bytes = 8 * int(np.prod(jax.eval_shape(relion_projector_half_to_texture_full, class_projectors[0]).shape))
    budget = _coarse_pass_budget((n_classes + 1) * cube_bytes + 8 * int(np.prod(np.shape(class_projectors[0]))))
    block_images = max(1, budget.operand_block // operand_bytes_per_image)
    batches = _coarse_batches(
        rotation_counts,
        n_slots=slots,
        n_trans=n_coarse_trans,
        n_pixels=int(layout.score_indices_np.size),
        projection_bytes_per_pixel=_coarse_projection_bytes_per_pixel(
            class_textures,
            jnp.asarray(class_projectors[0]).dtype,
            box_size=int(layout.image_shape[0]),
            current_size=int(layout.current_size),
            model_max_r=int(model_max_r),
        ),
        budget_bytes=budget.batch,
    )
    score_indices = jnp.asarray(layout.score_indices_np, dtype=jnp.int32)
    r_pad_all = batches[0][1] if batches else 0
    # The significance of several batches' particles runs as one call, [P_sig, R_pad * T] values
    # within the batch budget; a particle's padded rotations carry a -inf prior and are never significant.
    p_pad_all = batches[0][2] if batches else 1
    n_score_pixels = int(layout.score_indices_np.size)
    significance_batch, rescore_bytes, by_diff2, by_device = _flush_plan(
        n_classes * r_pad_all * n_coarse_trans * 4,
        slots * (n_score_pixels * (8 + 4) + 4 + n_coarse_trans * 2 * 4),  # unshifted, weight, initial, phases
        batch_particles=p_pad_all,
        flush_bytes=budget.flush,
        describe=(
            f"box {int(layout.image_shape[0])}, current size {int(layout.current_size)}, {slots} tilts, "
            f"{n_classes} x {r_pad_all} rotations x {n_coarse_trans} translations"
        ),
    )
    # A flush holds whole batches, the first that reach significance_batch particles, and never more than the pass.
    flush_size = min(-(-significance_batch // p_pad_all) * p_pad_all, n_units)
    logger.info(
        "Coarse flushes of %d particles for %d (%d per coarse batch; bounds %d by the diff2 cap, %d by the device's "
        "%.2f GiB flush budget)", flush_size, n_units, p_pad_all, by_diff2, by_device, budget.flush / 2**30,
    )
    class_full = [None] * n_classes  # each class's full projector for the direct-square kernel, built on first use
    eulers_f32 = jnp.asarray(np.asarray(coarse_eulers_deg, dtype=np.float32))  # the exact cut's rotations, gathered on the device
    rescored, n_scored, full_sorted = [], 0, []
    supports, pmax_by_unit, count_by_unit = [{} for _ in range(n_classes)], {}, {}
    last_total = None
    pending = []
    # The last flush's device results, read back by the next flush: by then the GPU holds the batches
    # queued after them, so the readback waits for nothing and the host's bookkeeping overlaps them.
    in_flight = None
    # The flush's compaction size, a guess from the previous flush's significant cells (2x margin); a
    # flush whose cells overflow it is compacted again at its exact size.
    cells_capacity = 0

    def rotation_prior_of(batch_units):
        if local:
            # The particle's own prior for every scored class, class-major (class k at k * R_pad), padding at -inf.
            rotation_prior = np.full((batch_units.size, n_classes, r_pad_all), -np.inf, dtype=np.float32)
            for i, unit in enumerate(batch_units):
                rotation_prior[i, scored_classes, : int(rotation_counts[unit])] = np.asarray(
                    unit_rotation_log_priors[unit], np.float32
                )
            return rotation_prior.reshape(batch_units.size, -1)
        if n_classes > 1:
            # The classes' rotations side by side, class-major (class k at k * R_pad), padding at -inf.
            rotation_prior = np.full((n_classes, r_pad_all), -np.inf, dtype=np.float32)
            rotation_prior[:, : coarse_eulers_deg.shape[0]] = np.asarray(rotation_log_prior, np.float32)
            return rotation_prior.reshape(-1)
        if r_pad_all > coarse_eulers_deg.shape[0]:
            rotation_prior = np.full(r_pad_all, -np.inf, dtype=np.float32)
            rotation_prior[: coarse_eulers_deg.shape[0]] = (
                0.0 if rotation_log_prior is None else np.asarray(rotation_log_prior, np.float32)
            )
            return rotation_prior
        return rotation_log_prior

    def significance(batch_units, diff2):
        return particle_coarse_significance(
            diff2,
            rotation_prior_of(batch_units),
            np.asarray(unit_translation_log_prior, dtype=np.float32)[batch_units],
            adaptive_fraction=adaptive_fraction,
            max_significants=max_significants,
        )

    def exact_cut(units_all, diff2, error, n_images, operands, n_real, *, capacity, full_sort=None, groups=()):
        """The flushed particles' cut on exact scores: the rotations the GEMM rounding leaves undecided at a
        particle's cut (max_significants or the adaptive fraction) and at its smallest diff2 are scored again by
        the direct-square kernel, and the cut is taken on those values (:mod:`relax.scoring.exact_cut`).

        Everything stays on the device: the undecided rotations are compacted to ``capacity`` per particle and
        their scorer matrices built there (:func:`_undecided_scorer_rotations`). Returns the cut and the device
        scalars ``(most, unresolved, rows, particles)``; the caller reads them later and takes the cut again
        (``exact_cut_again``) when a particle held more than ``capacity`` undecided rotations or the fraction
        rule needs every rotation of a particle sorted (``full_sort``). ``groups`` are ``(positions, size)``: those
        flush particles' undecided rotations are scored again at that size (each particle's own, a group of
        particles at a time), so a particle with many does not size every particle's rescore.
        """

        stats = significance(units_all, diff2)
        cut_inputs = (diff2, stats["log_weights"], stats["mask"], stats["cutoff_count"], error, n_images)
        rows, most, unresolved = _particle_near_cut_rows(cut_inputs, full_sort, **cut_static)
        ids, use = first_rows(rows, capacity=capacity)
        for class_index, class_projector in enumerate(class_projectors):
            if class_full[class_index] is None:
                # Cast first (the embedding only moves values): one complex64 cube, not a complex128 one and its cast.
                class_full[class_index] = relion_projector_half_to_texture_full(
                    jnp.asarray(class_projector).astype(jnp.complex64)
                )
        # The undecided rotations go in groups that fit the flush's rescore bytes; a rotation's direct square does
        # not depend on the others of its call.
        per_call = _rescore_rows_per_call(units_all.size, slots, n_coarse_trans, rescore_bytes, capacity)
        exact = diff2
        for first in range(0, capacity, per_call):
            chunk = slice(first, first + per_call)
            exact = rescore(units_all, exact, ids[:, chunk], use[:, chunk], operands)
        scored = jnp.sum(use, axis=1, dtype=jnp.int32)  # each particle's rotations scored again
        for positions, group_capacity in groups:
            exact, scored = rescore_group(units_all, exact, scored, rows, operands, positions, group_capacity)
        check = _exact_cut_check(scored, unresolved, most, jnp.int32(n_real))
        return significance(units_all, exact), check

    def rescore(units_all, diff2, ids, use, operands):
        """``diff2`` with the rotations ``ids`` (where ``use``) scored again by the direct square."""

        row_rotations = _undecided_scorer_rotations(
            ids,
            use,
            np.asarray(rotation_counts, dtype=np.int64)[units_all],
            None if not local else [unit_rotations[unit] for unit in units_all],
            eulers_f32,
            [image_left[offsets[unit] : offsets[unit + 1]] for unit in units_all],
            random_perturbation,
            angular_sampling_deg,
            r_pad=int(r_pad_all),
            slots=slots,
            pass1_rotations=pass1_rotations_left_f32,
        )
        unshifted, weight, initial, angles = operands
        return _exact_rows_diff2(
            diff2,
            ids,
            use,
            row_rotations,
            tuple(class_full),
            unshifted,
            weight,
            initial,
            angles,
            layout.full_to_compact,
            r_pad=int(r_pad_all),
            current_size=int(layout.current_size),
            physical_image_size=int(layout.image_shape[0]),
            model_max_r=int(model_max_r),
            padding_factor=int(padding_factor),
        )

    def rescore_group(units_all, exact, scored, rows, operands, positions, group_capacity):
        """``exact`` with the undecided rotations of the flush particles ``positions`` scored again, up to
        ``group_capacity`` each, and ``scored`` with their counts of rotations scored again. The group is padded to
        a power of two of particles (the padding scores nothing and writes nothing back)."""

        n = int(positions.size)
        n_pad = pow2_ceil(n)
        take = np.concatenate([positions, np.full(n_pad - n, positions[-1])]).astype(np.int64)
        take_device, n_device = jnp.asarray(take, dtype=jnp.int32), jnp.int32(n)
        group_ids, group_use, group_operands, group_exact = _take_rescore_group(
            rows, exact, operands, take_device, n_device, capacity=group_capacity
        )
        per_call = _rescore_rows_per_call(n_pad, slots, n_coarse_trans, rescore_bytes, group_capacity)
        for first in range(0, group_capacity, per_call):
            chunk = slice(first, first + per_call)
            group_exact = rescore(
                units_all[take], group_exact, group_ids[:, chunk], group_use[:, chunk], group_operands
            )
        return _put_rescore_group(exact, scored, group_exact, group_use, take_device, n_device)

    def exact_cut_again(cut_operands, unresolved):
        """The flush's exact cut at the size its undecided rotations need (read from the device): every rotation
        sorted for the particles whose fraction rule needed it (when ``unresolved`` counts any), and room for the
        particle with the most."""

        units_all, diff2, error, n_images, operands, _n_real = cut_operands
        stats = significance(units_all, diff2)
        cut_inputs = (diff2, stats["log_weights"], stats["mask"], stats["cutoff_count"], error, n_images)
        full_sort = _particle_near_cut_rows(cut_inputs, None, **cut_static)[2] if unresolved else None
        rows, _most, _ = _particle_near_cut_rows(cut_inputs, full_sort, **cut_static)
        # Every particle is scored again at the first cut's size (on its rows with the full sort where it needs
        # it); a real particle with more undecided rotations again with its group, the particles whose counts
        # round up to the same power of two.
        counts = np.asarray(jnp.sum(rows, axis=1, dtype=jnp.int32))[:_n_real]
        again = counts > _UNDECIDED_ROTATIONS
        sizes = np.array([row_capacity(int(c), rows.shape[1], least=_UNDECIDED_ROTATIONS) for c in counts])
        groups = tuple(
            (np.flatnonzero(again & (sizes == size)), int(size)) for size in np.unique(sizes[again])
        )
        return exact_cut(*cut_operands, capacity=_UNDECIDED_ROTATIONS, full_sort=full_sort, groups=groups)

    def compacted(stats, n_real):
        # The significant cells are compacted on the device; only their ids come back (the dense mask is
        # K * R * T booleans per particle). A padded flush's copies' cells are dropped.
        mask = stats["mask"] & (jnp.arange(stats["mask"].shape[0]) < n_real)[:, None]
        counts = jnp.sum(mask, axis=1, dtype=jnp.int32)
        flat = (
            _significant_cells(mask, capacity=cells_capacity, row_length=n_coarse_trans) if cells_capacity else None
        )
        for value in (counts, flat, stats["pmax"], stats["cutoff_count"]):
            if value is not None:
                value.copy_to_host_async()
        return mask, counts, flat, stats["pmax"], stats["cutoff_count"]

    def dispatch():
        units_all = np.concatenate([u for u, *_ in pending])
        diff2 = jnp.concatenate([t for _, t, *_ in pending], axis=0)
        error = jnp.concatenate([e for _, _, e, *_ in pending], axis=0)
        n_images = jnp.asarray(np.concatenate([n for _, _, _, n, _ in pending]))
        operands = [o for *_, o in pending]
        pending.clear()
        # Every flush of the pass has the same number of particles, so its programs compile once per pass shape: a
        # shorter (last) flush repeats its first particle, whose copies are cut alike and not read back.
        n_real = int(units_all.size)
        padding = flush_size - n_real
        if padding > 0:
            first = np.zeros(padding, dtype=np.int64)
            units_all = np.concatenate([units_all, units_all[first]])
            diff2 = jnp.concatenate([diff2, jnp.repeat(diff2[:1], padding, axis=0)], axis=0)
            error = jnp.concatenate([error, jnp.repeat(error[:1], padding, axis=0)], axis=0)
            n_images = jnp.concatenate([n_images, jnp.repeat(n_images[:1], padding, axis=0)], axis=0)
            operands.append(tuple(jnp.repeat(value[:1], padding, axis=0) for value in operands[0]))
        # One copy of the flush's operands, held until it is collected; the batches' slices are freed with the list.
        operands = tuple(jnp.concatenate(parts, axis=0) for parts in zip(*operands))
        cut_operands = (units_all, diff2, error, n_images, operands, n_real)
        stats, check = exact_cut(*cut_operands, capacity=_UNDECIDED_ROTATIONS)
        for value in check:
            value.copy_to_host_async()
        # The flush's GEMM scores and operands stay until it is collected, in case its cut is taken again.
        return cut_operands, check, compacted(stats, n_real)

    def collect(cut_operands, check, result):
        nonlocal cells_capacity
        n_real = cut_operands[5]
        units_all = cut_operands[0][:n_real]
        most, unresolved, n_rows, n_particles = (int(value) for value in check)
        if unresolved:
            full_sorted.append(unresolved)
        if unresolved or most > _UNDECIDED_ROTATIONS:
            stats, check = exact_cut_again(cut_operands, unresolved)
            most, unresolved, n_rows, n_particles = (int(value) for value in check)
            result = compacted(stats, n_real)
        if n_rows:
            rescored.append((n_rows, n_particles))
        mask, counts, flat, pmax, cutoff_count = result
        counts = np.asarray(counts, dtype=np.int64)
        n_significant = int(counts.sum())
        if flat is None or n_significant > int(flat.shape[0]):
            flat = _significant_cells(
                mask, capacity=pow2_ceil(n_significant), row_length=n_coarse_trans
            )
        # The next flush's guess follows this flush, not the largest so far: one flat-posterior particle's
        # millions of cells kept every later flush compacting at its size (0.9 s per flush, relax#98's real case).
        cells_capacity = pow2_ceil(2 * n_significant, minimum=2)
        flat = np.asarray(flat, dtype=np.int64)[:n_significant]
        particle_pmax = np.asarray(pmax, dtype=np.float64)
        particle_count = np.asarray(cutoff_count, dtype=np.int64)
        cell_offsets = np.concatenate([[0], np.cumsum(counts)])
        n_cells = int(mask.shape[1])
        for i, unit in enumerate(units_all):
            all_cells = flat[cell_offsets[i] : cell_offsets[i + 1]] - i * n_cells
            cell_class = all_cells // (r_pad_all * n_coarse_trans)
            for class_index in range(n_classes):
                cells = all_cells[cell_class == class_index] - class_index * r_pad_all * n_coarse_trans
                if cells.size and int(cells[-1]) // n_coarse_trans >= int(rotation_counts[unit]):
                    raise RuntimeError(f"particle {unit}: a padded coarse rotation came out significant")
                if local:
                    cells = unit_rotations[unit][cells // n_coarse_trans] * n_coarse_trans + cells % n_coarse_trans
                supports[class_index][int(unit)] = cells.astype(np.int32)
            pmax_by_unit[int(unit)] = float(particle_pmax[i])
            count_by_unit[int(unit)] = int(particle_count[i])

    def flush(*, last=False):
        """Dispatch the pending particles' significance, then read back the previous flush's."""

        nonlocal in_flight
        dispatched = dispatch() if pending else None
        if in_flight is not None:
            collect(*in_flight)
        in_flight = dispatched
        if last and in_flight is not None:
            collect(*in_flight)
            in_flight = None

    cut_static = dict(
        max_significants=0 if max_significants is None else max(int(max_significants), 0),
        adaptive_fraction=float(adaptive_fraction),
    )
    # The slot-block program's traced slot offsets, on the device once per pass: a host scalar argument is a
    # host-to-device copy on every call, and a late pass makes tens of thousands of calls.
    slot_offsets = [jax.device_put(np.int32(first)) for first in range(slots)]
    tilt = TiltCoarseInputs(
        offsets, image_left, image_projections, old, coarse_translations_px, box_size, coarse_eulers_deg,
        random_perturbation, angular_sampling_deg,
    )
    scorer = CoarseScorer(class_projectors, class_textures, score_indices, dict(
        image_shape=tuple(layout.image_shape), current_size=int(layout.current_size), model_max_r=int(model_max_r),
        padding_factor=int(padding_factor),
    ))
    plan = None if local or not batches else _frame_group_plan(
        tilt, flush_size=flush_size, slots=slots, slot_block=int(batches[0][3]), r_pad=r_pad_all,
        n_trans=n_coarse_trans, n_score_pixels=n_score_pixels, batch_bytes=budget.batch,
        projection_bytes=_coarse_projection_bytes_per_pixel(
            class_textures, jnp.asarray(class_projectors[0]).dtype, box_size=int(layout.image_shape[0]),
            current_size=int(layout.current_size), model_max_r=int(model_max_r),
        ),
    )
    operands_of = partial(
        _indexed_image_coarse_operands, experiment_dataset, layout=layout, noise_variance_half=noise_variance_half,
        optics_group_ids=optics_group_ids, scale_corrections=scale_corrections,
    )
    # The image preprocess kernel's finite check is read at the end of the pass, as pass 1 reads it (run_pass1):
    # read at each operand batch, it waited for the device's queued work inside every operand block's
    # preparation (22 read-backs per block, the device idle meanwhile). An invalid image still stops the pass,
    # with the same failure naming the image, before its supports are used.
    with deferred_relion_preprocess_checks("tilt coarse pass") as preprocess_checks:
        # The groups' host image reads run one group ahead, on a worker thread.
        with prefetched_batches(_group_image_reads(experiment_dataset, plan, offsets), depth=1) as reads:
            for units, fetched in zip([] if plan is None else plan.groups, reads):
                preprocess_checks.at(f"particle group {int(units[0])}-{int(units[-1])}")
                diff2, error, n_valid, operands, last_total = _score_frame_group(
                    units, plan, tilt, partial(operands_of, fetched=fetched), scorer, r_pad=r_pad_all, slots=slots
                )
                n_scored += int(units.size) * n_classes * r_pad_all * n_coarse_trans
                pending.append((units, diff2, error, n_valid, operands))
                flush()
        for units, r_pad, p_pad, slot_block in batches if plan is None else ():
            preprocess_checks.at(f"particles {int(units[0])}-{int(units[-1])}")
            diff2, error, n_valid, operands, last_total = _score_slot_batch(
                units, tilt, operands_for, scorer, r_pad=r_pad, p_pad=p_pad, slots=slots, slot_block=slot_block,
                slot_offsets=slot_offsets, unit_rotations=None if not local else [unit_rotations[unit] for unit in units],
            )
            n_scored += int(units.size) * n_classes * r_pad * n_coarse_trans
            # The flush cuts these particles, and scores their undecided rotations again from these operands.
            pending.append((units, diff2, error, n_valid, operands))
            if sum(int(u.size) for u, *_ in pending) >= significance_batch:
                flush()
        flush(last=True)
    if rescored:
        n_rows, n_particles = (int(sum(int(v) for v in column)) for column in zip(*rescored))
        logger.info(
            "Coarse cut: %d of %d particles had rotations within the GEMM error bound of their cut or of their "
            "smallest diff2; %d rotations scored again by the direct square (%.3g%% of the pass's samples); %d "
            "particles needed every rotation sorted", n_particles, n_units, n_rows,
            100.0 * n_rows * n_coarse_trans / max(n_scored, 1), sum(full_sorted),
        )
    for k, texture in enumerate(class_textures):
        if texture is not None and last_total is not None:
            # The flush read every batch's significance back, so the last projection has completed; the
            # texture stays for the next pass (_coarse_capacity_texture), which refills it after this result.
            _CLASS_TEXTURES[k] = (texture, last_total)
    class_supports = [[supports[k][u] for u in range(n_units)] for k in range(n_classes)]
    return class_supports[0] if n_classes == 1 else class_supports, np.asarray(
        [pmax_by_unit[u] for u in range(n_units)], dtype=np.float64
    ), np.asarray([count_by_unit[u] for u in range(n_units)], dtype=np.int32)
