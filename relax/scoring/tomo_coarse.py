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
coarse operands, and the fused coarse projector, called once per image. The fused
projector takes at most 128 translations, so each image is scored in translation chunks;
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

    from relax.helpers.optics_noise import noise_rows
    from relax.helpers.preprocessing import prepare_batch_preprocess_operands
    from relax.relion.relion_coarse_operands import (
        _process_relion_exact_coarse_half_image,
        _relion_exact_coarse_operands,
    )
    from relax.relion.relion_ctf import _relion_exact_ctf_half_from_source_star_host
    from relax.sparse_pass2.sparse_pass2_scoring import _relion_cuda_powerclass_highres_xi2_half

    image_indices = np.asarray(image_indices, dtype=np.int64)
    (batch_data, *_rest, fetched) = next(
        iter(experiment_dataset.iter_batches(int(image_indices.size), indices=image_indices, by_image=True))
    )
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


@partial(
    jax.jit, static_argnames=("current_size", "physical_image_size", "model_max_r", "padding_factor", "n_chunks")
)
def _particle_coarse_diff2_scan(
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
    """One particle's coarse diff2 ``[R, T]``: its images' diff2 (:func:`tilt_image_coarse_diff2`) added in slot order.

    One program per (slots, rotations, translations): the images run in a device loop, not one
    host dispatch per image and translation chunk. A padded slot has zero pixel weight and zero
    initial diff2, so it adds exact zeros.
    """

    from relax.cuda import kernels as em_cuda_kernels

    capacity = _FUSED_TRANSLATION_CAPACITY

    def image_diff2(rotation, image, weight, initial, angles):
        return jnp.concatenate(
            [
                em_cuda_kernels.relion_coarse_diff2_projector_f32(
                    projector_full,
                    rotation,
                    image[None],
                    angles[chunk * capacity : (chunk + 1) * capacity],
                    weight[None],
                    initial.reshape(1),
                    full_to_compact,
                    current_size=current_size,
                    physical_image_size=physical_image_size,
                    model_max_r=model_max_r,
                    padding_factor=padding_factor,
                    canonical_reduction=True,
                )[0]
                for chunk in range(n_chunks)
            ],
            axis=1,
        )

    def body(total, xs):
        return total + image_diff2(*xs), None

    n_rot, n_trans = rotations.shape[1], translation_angles.shape[1]
    total, _ = jax.lax.scan(
        body,
        jnp.zeros((n_rot, n_trans), dtype=jnp.float32),
        (rotations, unshifted, pixel_weight, initial_diff2, translation_angles),
    )
    return total


_OPERAND_IMAGE_BATCH = 1024


def _all_image_coarse_operands(
    experiment_dataset, n_images: int, layout, *, noise_variance_half, optics_group_ids, scale_corrections
):
    """:func:`tilt_image_coarse_operands` of every dataset image, in batches of ``_OPERAND_IMAGE_BATCH``.

    The last batch is padded with repeats of its last image (dropped after), so every batch runs
    the same programs.
    """

    parts = ([], [], [])
    for start in range(0, int(n_images), _OPERAND_IMAGE_BATCH):
        indices = np.arange(start, min(start + _OPERAND_IMAGE_BATCH, int(n_images)))
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
    tail cut. ``rotation_log_prior`` is ``[R]`` (or ``None``) and ``translation_log_prior``
    ``[P, T]``, the particle's 3D offset prior. The SPA posterior primitive is reused, so the two
    paths cannot drift apart. Returns its statistics dict (``mask [P, R * T]``, ``n_significant``,
    ``pmax``, ``winner``, ...).
    """

    from relax.scoring.coarse_publication import _dense_prior_scores, _posterior_statistics

    raw = -jnp.asarray(particle_diff2, dtype=jnp.float32)
    n_particles = int(raw.shape[0])
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
    supports, pmax = [], np.zeros(offsets.size - 1, dtype=np.float64)
    local = unit_rotation_ids is not None
    if local != (unit_rotation_log_priors is not None) or (local and rotation_log_prior is not None):
        raise ValueError("a local search takes each particle's rotations and priors, and no shared prior")
    coarse_eulers_deg = np.asarray(coarse_eulers_deg)
    n_coarse_trans = int(np.asarray(coarse_translations_px).shape[0])
    n_units = int(offsets.size - 1)
    slots = int(np.max(np.diff(offsets))) if n_units else 1
    # Every tilt image's coarse operands, in fixed image batches (one preprocessing program).
    unshifted, weight, initial = _all_image_coarse_operands(
        experiment_dataset,
        int(offsets[-1]),
        layout,
        noise_variance_half=noise_variance_half,
        optics_group_ids=optics_group_ids,
        scale_corrections=scale_corrections,
    )
    n_chunks = -(-n_coarse_trans // _FUSED_TRANSLATION_CAPACITY)
    for unit in range(n_units):
        images = np.arange(offsets[unit], offsets[unit + 1])
        left, _applies = tomo_particles.relion_left_matrices(image_projections[images])
        unit_rotations = None if not local else np.asarray(unit_rotation_ids[unit], dtype=np.int64)
        if local and (unit_rotations.size == 0 or np.any(np.diff(unit_rotations) <= 0)):
            raise ValueError(f"particle {unit}'s local rotations must be ascending and non-empty")
        rotations = _relion_adaptive_pass1_rotations(
            coarse_eulers_deg if not local else coarse_eulers_deg[unit_rotations],
            random_perturbation,
            angular_sampling_deg,
            left_matrices=left,
        )
        angles = tomo_particles.tilt_translation_angles(
            coarse_translations_px,
            old[unit : unit + 1],
            image_projections[images],
            np.zeros(images.size, int),
            image_size,
        )
        pad = slots - images.size
        diff2 = _particle_coarse_diff2_scan(
            projector_full,
            jnp.asarray(np.pad(np.asarray(rotations, dtype=np.float32), ((0, pad), (0, 0), (0, 0), (0, 0)))),
            jnp.pad(unshifted[images], ((0, pad), (0, 0))),
            jnp.pad(weight[images], ((0, pad), (0, 0))),
            jnp.pad(initial[images], (0, pad)),
            jnp.asarray(np.pad(angles, ((0, pad), (0, 0), (0, 0)))),
            layout.full_to_compact,
            current_size=int(layout.current_size),
            physical_image_size=int(layout.image_shape[0]),
            model_max_r=int(model_max_r),
            padding_factor=int(padding_factor),
            n_chunks=int(n_chunks),
        )
        stats = particle_coarse_significance(
            diff2[None],
            rotation_log_prior if not local else np.asarray(unit_rotation_log_priors[unit], dtype=np.float32),
            np.asarray(unit_translation_log_prior, dtype=np.float32)[unit : unit + 1],
            adaptive_fraction=adaptive_fraction,
            max_significants=max_significants,
        )
        cells = np.flatnonzero(np.asarray(stats["mask"][0]))
        if local:
            cells = unit_rotations[cells // n_coarse_trans] * n_coarse_trans + cells % n_coarse_trans
        supports.append(cells.astype(np.int32))
        pmax[unit] = float(stats["pmax"][0])
    return supports, pmax
