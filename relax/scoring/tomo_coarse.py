"""Coarse pass (pass 1) of subtomogram particles: each tilt image scored, summed per particle (S4.2).

RELION's GPU path scores a subtomogram one tilt image at a time (the ``img_id`` loop of
``getAllSquaredDifferencesCoarse``, acc_ml_optimiser_impl.h:1190-1402). Every image has
its own scorer matrices (``make_eulers_3D`` with the image's ``Aproj`` as the left matrix,
:1086-1116) and its own phase table (``Aproj[:2]`` times the 3D trial shift plus the
rounded old offset, :1214-1240). Each image first adds its ``highres_Xi2 / 2`` to the
particle's running diff2 and then its pixel sums (:1289-1296, diff2.cuh:186-188), so
the significance cut sees one diff2 per particle hypothesis.

relax reuses the SPA pieces: the RELION CUDA preprocessing with no pre-shift and no norm
correction (RELION neither translates nor normalises a tomo image, :429-476), the exact
coarse operands, and the fused coarse projector with each image's own poses, all images of
a batch of particles in one launch. The fused projector takes at most 128 translations, so
each image is scored in translation chunks;
RELION runs one 1024-thread block for 515 translations. The chunks change only the
float32 order of the pixel sums, the same class as RELION's own atomic lane order.
See PLAN.md "S4.2 implementation ladder" in the cryo-ET coordination directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

# The fused coarse projector's translation capacity (one 128-thread block).
_FUSED_TRANSLATION_CAPACITY = 128


@dataclass(frozen=True)
class CoarseScoreLayout:
    """The coarse score window and its fused-projector lookup, shared by every image of a pass."""

    image_shape: tuple
    current_size: int
    score_indices_np: np.ndarray
    score_active_mask: jnp.ndarray
    full_to_compact: jnp.ndarray
    half_weights: jnp.ndarray


def coarse_score_layout(
    image_shape, current_size, *, half_spectrum_scoring: bool, square_window: bool
) -> CoarseScoreLayout:
    """The SPA coarse pass's score window (significance.py:1105-1123, 1680-1726) for one current size."""

    from relax.helpers.fourier_window import make_fourier_window_spec
    from relax.helpers.half_spectrum import make_scoring_half_image_weights
    from relax.scoring.significance import _coarse_gaussian_fused_logical_lookup, _plan_coarse_gaussian_square_layout

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
    layout = _plan_coarse_gaussian_square_layout(
        image_shape, int(current_size), active, stable_fourier_window_shapes=False
    )
    return CoarseScoreLayout(
        image_shape=image_shape,
        current_size=int(current_size),
        score_indices_np=np.asarray(layout.score_indices_np, dtype=np.int32),
        score_active_mask=jnp.asarray(layout.score_active_mask_np, dtype=jnp.bool_),
        full_to_compact=_coarse_gaussian_fused_logical_lookup(
            jnp.asarray(layout.full_to_compact_np, dtype=jnp.int32), layout, current_size=int(current_size)
        ),
        half_weights=make_scoring_half_image_weights(
            image_shape, relion_half_sum=half_spectrum_scoring, exclude_relion_redundant_x0=True
        ),
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
):
    """Corrected images, pixel weights and initial diff2 of tilt images: the SPA exact coarse operands.

    ``noise_variance_half`` is one half-pixel spectrum or a ``[G, P]`` table with
    ``optics_group_ids`` per dataset image. No pre-shift and no norm correction: RELION
    neither translates nor normalises a tomo image (acc_ml_optimiser_impl.h:429-476);
    ``scale_corrections`` (per dataset image) still divide the image and weight the
    CTF, as for SPA (:1244-1263).
    """

    from relax.helpers.batch_fetch import fetch_indexed_batch
    from relax.helpers.optics_noise import noise_rows
    from relax.helpers.preprocessing import prepare_batch_preprocess_operands
    from relax.relion.relion_coarse_operands import (
        _process_relion_exact_coarse_half_image,
        _relion_exact_coarse_operands,
    )
    from relax.relion.relion_ctf import _relion_exact_ctf_half_from_source_star_host
    from relax.sparse_pass2.sparse_pass2_scoring import _relion_cuda_powerclass_highres_xi2_half

    image_indices = np.asarray(image_indices, dtype=np.int64)
    # One vectorized host read (the dataset's batch iterator collates image by image, 15x slower).
    batch_data, _ctf_params, fetched = fetch_indexed_batch(experiment_dataset, image_indices)
    if not np.array_equal(np.asarray(fetched), image_indices):
        raise RuntimeError("the dataset returned the tilt images in another order")
    batch_data = np.asarray(batch_data)
    relion_cuda, _shifts, _corr, batch_scale, preprocess_kwargs = prepare_batch_preprocess_operands(
        experiment_dataset, batch_data, image_indices, scale_corrections=scale_corrections
    )
    if not relion_cuda:
        raise RuntimeError("the subtomogram coarse pass needs the RELION CUDA image preprocessing")
    processed = _process_relion_exact_coarse_half_image(
        experiment_dataset, batch_data, score_with_masked_images, relion_preprocess_kwargs=preprocess_kwargs
    )
    ctf = _relion_exact_ctf_half_from_source_star_host(
        experiment_dataset, image_indices, layout.image_shape, pixel_indices=layout.score_indices_np
    )
    unshifted, pixel_weight = _relion_exact_coarse_operands(
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
    initial = _relion_cuda_powerclass_highres_xi2_half(
        processed, image_shape=layout.image_shape, current_size=layout.current_size
    )
    return unshifted, pixel_weight, jnp.asarray(initial, dtype=jnp.float32)


def tilt_image_coarse_diff2(
    projector_full,
    rotations,
    unshifted,
    pixel_weight,
    initial_diff2,
    translation_angles,
    layout: CoarseScoreLayout,
    *,
    model_max_r: int,
    padding_factor: int,
    canonical_reduction: bool = True,
):
    """One tilt image's coarse diff2 ``[R, T]`` with its own matrices ``[R, 3, 3]`` and phases ``[T, 2]``."""

    from relax.cuda import kernels as em_cuda_kernels

    translation_angles = jnp.asarray(translation_angles, dtype=jnp.float32)
    chunks = [
        em_cuda_kernels.relion_coarse_diff2_projector_f32(
            projector_full,
            jnp.asarray(rotations, dtype=jnp.float32),
            jnp.asarray(unshifted, dtype=jnp.complex64)[None],
            translation_angles[start : start + _FUSED_TRANSLATION_CAPACITY],
            jnp.asarray(pixel_weight, dtype=jnp.float32)[None],
            jnp.asarray(initial_diff2, dtype=jnp.float32).reshape(1),
            layout.full_to_compact,
            current_size=layout.current_size,
            physical_image_size=int(layout.image_shape[0]),
            model_max_r=int(model_max_r),
            padding_factor=int(padding_factor),
            canonical_reduction=canonical_reduction,
        )[0]
        for start in range(0, int(translation_angles.shape[0]), _FUSED_TRANSLATION_CAPACITY)
    ]
    return jnp.concatenate(chunks, axis=1)


def particle_coarse_diff2(image_diff2_in_slot_order):
    """A particle's coarse diff2: its images' diff2 added in slot (``img_id``) order, float32."""

    total = None
    for image_diff2 in image_diff2_in_slot_order:
        image_diff2 = jnp.asarray(image_diff2, dtype=jnp.float32)
        total = image_diff2 if total is None else total + image_diff2
    return total


# Bytes of per-image coarse diff2 ([images, R, T] float32) one batched call materialises.
_COARSE_BATCH_BYTES = 512 << 20


@partial(jax.jit, static_argnames=("current_size", "physical_image_size", "model_max_r", "padding_factor", "n_chunks"))
def _particles_coarse_diff2(
    total,
    projector_full,
    rotations,
    unshifted,
    pixel_weight,
    initial_diff2,
    translation_angles,
    full_to_compact,
    *,
    current_size: int,
    physical_image_size: int,
    model_max_r: int,
    padding_factor: int,
    n_chunks: int,
):
    """``total`` ``[P, R, T]`` plus the particles' images' coarse diff2, added in slot order.

    ``rotations`` ``[P, S, R, 3, 3]``, ``unshifted``/``pixel_weight`` ``[P, S, pixels]``, ``initial_diff2``
    ``[P, S]`` and ``translation_angles`` ``[P, S, T, 2]`` are the particles' tilt images (or a block of
    their slots) with each image's own poses; every image of the call is scored in one launch per
    translation chunk (:func:`relax.cuda.kernels.relion_coarse_diff2_projector_per_image_f32`, the
    per-image arithmetic of :func:`tilt_image_coarse_diff2`). A padded slot or particle has zero
    pixel weight and zero initial diff2, so it adds exact zeros.
    """

    from relax.cuda import kernels as em_cuda_kernels

    n_particles, n_slots, n_rot = (int(n) for n in rotations.shape[:3])
    capacity = _FUSED_TRANSLATION_CAPACITY

    def flat(values):
        return values.reshape((n_particles * n_slots,) + values.shape[2:])

    image_diff2 = jnp.concatenate(
        [
            em_cuda_kernels.relion_coarse_diff2_projector_per_image_f32(
                projector_full,
                flat(rotations),
                flat(unshifted),
                flat(translation_angles[:, :, chunk * capacity : (chunk + 1) * capacity]),
                flat(pixel_weight),
                flat(initial_diff2),
                full_to_compact,
                current_size=current_size,
                physical_image_size=physical_image_size,
                model_max_r=model_max_r,
                padding_factor=padding_factor,
            ).reshape(n_particles, n_slots, n_rot, -1)
            for chunk in range(n_chunks)
        ],
        axis=3,
    )
    for slot in range(n_slots):
        total = total + image_diff2[:, slot]
    return total


def _coarse_batches(rotation_counts, *, n_slots: int, n_trans: int, budget_bytes: int = _COARSE_BATCH_BYTES):
    """Batches of consecutive particles for :func:`_particles_coarse_diff2`: ``(units, R_pad, P_pad, slot_block)``.

    One shape for the whole pass, so one program compiles: ``R_pad`` is the largest rotation count
    rounded up to a multiple of 128 (the kernel's main segment), and ``P_pad`` particles of
    ``slot_block`` slots per call keep ``P_pad * slot_block * R_pad * T`` float32 within ``budget_bytes``
    (a particle whose images exceed it is scored ``slot_block`` slots at a time).
    """

    counts = np.asarray(rotation_counts, dtype=np.int64)
    if counts.size == 0:
        return []
    r_pad = -(-int(np.max(counts)) // 128) * 128
    per_image = r_pad * int(n_trans) * 4
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

    The last batch is padded with repeats of its last image (dropped after), so every batch runs
    the same programs.
    """

    parts = ([], [], [])
    for start in range(int(image_start), int(image_stop), _OPERAND_IMAGE_BATCH):
        indices = np.arange(start, min(start + _OPERAND_IMAGE_BATCH, int(image_stop)))
        n_valid = indices.size
        padded = np.concatenate([indices, np.full(_OPERAND_IMAGE_BATCH - n_valid, indices[-1])])
        for part, values in zip(
            parts,
            tilt_image_coarse_operands(
                experiment_dataset,
                padded,
                layout,
                noise_variance_half=noise_variance_half,
                optics_group_ids=optics_group_ids,
                scale_corrections=scale_corrections,
            ),
        ):
            part.append(values[:n_valid])
    return tuple(jnp.concatenate(part, axis=0) for part in parts)


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
    (``mask [P, R * T]``, ``n_significant``, ``pmax``, ``winner``, ...).
    """

    from relax.scoring.coarse_publication import _dense_prior_scores, _posterior_statistics

    raw = -jnp.asarray(particle_diff2, dtype=jnp.float32)
    n_particles = int(raw.shape[0])
    if rotation_log_prior is not None and np.ndim(rotation_log_prior) == 2:
        # _dense_prior_scores' additions in its order, with a per-particle rotation prior.
        values = raw + jnp.float32(0.0)
        values = values + jnp.asarray(rotation_log_prior, jnp.float32)[:, :, None]
        values = values + jnp.asarray(translation_log_prior, jnp.float32)[:, None, :]
        values = values.reshape(n_particles, -1)
    else:
        values = _dense_prior_scores(
            raw, jnp.float32(0.0), rotation_log_prior, translation_log_prior, jnp.int32(n_particles)
        )
    raw_max = jnp.max(raw.reshape(n_particles, -1), axis=1)
    return _posterior_statistics(
        values,
        raw_max,
        None,
        adaptive_fraction=float(adaptive_fraction),
        max_significants=None if max_significants is None or int(max_significants) <= 0 else int(max_significants),
        tie_score_ulps=0,
    )


def particle_coarse_supports(
    experiment_dataset,
    *,
    unit_image_offsets,
    image_projections,
    unit_old_offsets_px,
    coarse_eulers_deg,
    random_perturbation,
    angular_sampling_deg,
    coarse_translations_px,
    projector_full,
    layout: CoarseScoreLayout,
    noise_variance_half,
    rotation_log_prior,
    unit_translation_log_prior,
    adaptive_fraction,
    max_significants,
    model_max_r: int,
    padding_factor: int,
    image_size: int,
    optics_group_ids=None,
    scale_corrections=None,
    unit_rotation_ids=None,
    unit_rotation_log_priors=None,
):
    """Each particle's coarse significant samples, ``rot * T + t`` int32 ids per unit, and its coarse Pmax.

    RELION's GPU coarse pass for a subtomogram (acc_ml_optimiser_impl.h:1190-1402): every tilt image is
    scored with its own device matrices (``make_eulers_3D`` with the image's ``Aproj``,
    :func:`relax.sampling._relion_adaptive_pass1_rotations`), its phases for the 3D trial shifts plus
    the rounded old offset (:func:`relax.refinement.tomo_particles.tilt_translation_angles`) and its own
    CTF and noise; the images' diff2 is summed in ``img_id`` order and the particle's weights are cut
    once (:func:`particle_coarse_significance`). Dataset images ``unit_image_offsets[u]:[u+1]`` are
    particle ``u``'s, in ``img_id`` order.

    A local search scores each particle over its own rotations only: ``unit_rotation_ids[u]`` (ascending
    rows of ``coarse_eulers_deg``) with the log prior ``unit_rotation_log_priors[u]`` (RELION's
    orientations with a nonzero prior, selectOrientationsWithNonZeroPriorProbability); the returned
    ids keep indexing ``coarse_eulers_deg``.
    """

    from relax.refinement import tomo_particles
    from relax.sampling import _relion_adaptive_pass1_rotations

    offsets = np.asarray(unit_image_offsets, dtype=np.int64)
    image_projections = np.asarray(image_projections, dtype=np.float64)
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
    block_images = max(1, _COARSE_OPERAND_BLOCK_BYTES // operand_bytes_per_image)
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

    n_chunks = -(-n_coarse_trans // _FUSED_TRANSLATION_CAPACITY)
    unit_rotations = [None if not local else np.asarray(unit_rotation_ids[u], dtype=np.int64) for u in range(n_units)]
    for u, rows in enumerate(unit_rotations):
        if local and (rows.size == 0 or np.any(np.diff(rows) <= 0)):
            raise ValueError(f"particle {u}'s local rotations must be ascending and non-empty")
    rotation_counts = [coarse_eulers_deg.shape[0] if rows is None else rows.size for rows in unit_rotations]
    batches = _coarse_batches(rotation_counts, n_slots=slots, n_trans=n_coarse_trans)
    r_pad_all = batches[0][1] if batches else 0
    # The significance of several batches' particles runs as one call, [P_sig, R_pad * T] values
    # within the batch budget; a particle's padded rotations carry a -inf prior and are never significant.
    significance_batch = max(1, _COARSE_BATCH_BYTES // max(r_pad_all * n_coarse_trans * 4, 1))
    supports, pmax_by_unit = {}, {}
    pending = []

    def flush():
        if not pending:
            return
        units_all = np.concatenate([u for u, _ in pending])
        totals = jnp.concatenate([t for _, t in pending], axis=0)
        pending.clear()
        if local:
            rotation_prior = np.full((units_all.size, r_pad_all), -np.inf, dtype=np.float32)
            for i, unit in enumerate(units_all):
                rotation_prior[i, : int(rotation_counts[unit])] = np.asarray(unit_rotation_log_priors[unit], np.float32)
        elif r_pad_all > coarse_eulers_deg.shape[0]:
            rotation_prior = np.full(r_pad_all, -np.inf, dtype=np.float32)
            rotation_prior[: coarse_eulers_deg.shape[0]] = (
                0.0 if rotation_log_prior is None else np.asarray(rotation_log_prior, np.float32)
            )
        else:
            rotation_prior = rotation_log_prior
        stats = particle_coarse_significance(
            totals,
            rotation_prior,
            np.asarray(unit_translation_log_prior, dtype=np.float32)[units_all],
            adaptive_fraction=adaptive_fraction,
            max_significants=max_significants,
        )
        masks, particle_pmax = np.asarray(stats["mask"]), np.asarray(stats["pmax"], dtype=np.float64)
        for i, unit in enumerate(units_all):
            cells = np.flatnonzero(masks[i])
            if cells.size and int(cells[-1]) // n_coarse_trans >= int(rotation_counts[unit]):
                raise RuntimeError(f"particle {unit}: a padded coarse rotation came out significant")
            if local:
                cells = unit_rotations[unit][cells // n_coarse_trans] * n_coarse_trans + cells % n_coarse_trans
            supports[int(unit)] = cells.astype(np.int32)
            pmax_by_unit[int(unit)] = float(particle_pmax[i])

    for units, r_pad, p_pad, slot_block in batches:
        # Host operands of the batch, padded to [P_pad, S, R_pad, ...]: padded rotations repeat the particle's
        # last one, padded slots and particles carry zero weight (their diff2 adds zeros and is not read).
        rotations = np.zeros((p_pad, slots, r_pad, 3, 3), dtype=np.float32)
        angles = np.zeros((p_pad, slots, n_coarse_trans, 2), dtype=np.float32)
        image_index = np.zeros((p_pad, slots), dtype=np.int64)
        image_valid = np.zeros((p_pad, slots), dtype=bool)
        for p, unit in enumerate(units):
            images = np.arange(offsets[unit], offsets[unit + 1])
            left, _applies = tomo_particles.relion_left_matrices(image_projections[images])
            unit_rot = _relion_adaptive_pass1_rotations(
                coarse_eulers_deg if not local else coarse_eulers_deg[unit_rotations[unit]],
                random_perturbation,
                angular_sampling_deg,
                left_matrices=left,
            )
            n_rot = int(rotation_counts[unit])
            rotations[p, : images.size, :n_rot] = np.asarray(unit_rot, dtype=np.float32)
            rotations[p, : images.size, n_rot:] = np.asarray(unit_rot, dtype=np.float32)[:, -1:]
            angles[p, : images.size] = tomo_particles.tilt_translation_angles(
                coarse_translations_px,
                old[unit : unit + 1],
                image_projections[images],
                np.zeros(images.size, int),
                image_size,
            )
            image_index[p, : images.size] = images
            image_valid[p, : images.size] = True
        block_start, _block_stop, unshifted, weight, initial = operands_for(units)
        index = jnp.asarray(np.where(image_valid, image_index - block_start, 0))
        valid = jnp.asarray(image_valid)
        batch_unshifted = jnp.where(valid[..., None], unshifted[index], jnp.zeros((), unshifted.dtype))
        batch_weight = jnp.where(valid[..., None], weight[index], jnp.zeros((), weight.dtype))
        batch_initial = jnp.where(valid, initial[index], jnp.zeros((), initial.dtype))
        total = jnp.zeros((p_pad, r_pad, n_coarse_trans), dtype=jnp.float32)
        for first in range(0, slots, slot_block):
            block = slice(first, first + slot_block)
            total = _particles_coarse_diff2(
                total,
                projector_full,
                jnp.asarray(rotations[:, block]),
                batch_unshifted[:, block],
                batch_weight[:, block],
                batch_initial[:, block],
                jnp.asarray(angles[:, block]),
                layout.full_to_compact,
                current_size=int(layout.current_size),
                physical_image_size=int(layout.image_shape[0]),
                model_max_r=int(model_max_r),
                padding_factor=int(padding_factor),
                n_chunks=int(n_chunks),
            )
        pending.append((units, total[: units.size]))
        if sum(int(u.size) for u, _ in pending) >= significance_batch:
            flush()
    flush()
    return [supports[u] for u in range(n_units)], np.asarray(
        [pmax_by_unit[u] for u in range(n_units)], dtype=np.float64
    )
