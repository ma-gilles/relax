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
coarse operands, RELION's score translation, and the SPA coarse GEMM scorer (RELION's
``d0 + 0.5 sum w |p - y|^2`` as two real-packed float32 GEMMs) on each image's own projections,
a block of a batch's images per call. The GEMM expansion rounds differently from RELION's
direct square, so the rotations with a sample within the expansion's error bound of a particle's
significance cut (or of its smallest diff2) are scored again by the fused direct-square kernel before the
cut is taken (:mod:`relax.scoring.exact_cut`). :func:`tilt_image_coarse_diff2` (that kernel, one image) is
the reference the tests compare the GEMM scorer with.
See PLAN.md "S4.2 implementation ladder" in the cryo-ET coordination directory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from relax.scoring.exact_cut import (
    FUSED_TRANSLATION_CAPACITY as _FUSED_TRANSLATION_CAPACITY,
)
from relax.scoring.exact_cut import (
    direct_rows_diff2,
    first_rows,
    float32_unit,
    gemm_error_bound,
    put_rows,
    row_capacity,
    undecided_rotations,
)

logger = logging.getLogger(__name__)


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
    # The image indices demodulate odd aberrations, as RELION does for every tilt image (ml_optimiser.cpp:6418).
    processed = _process_relion_exact_coarse_half_image(
        experiment_dataset,
        batch_data,
        score_with_masked_images,
        relion_preprocess_kwargs=preprocess_kwargs,
        image_indices=image_indices,
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


# Bytes of per-image projections ([images, R, P]) and coarse diff2 ([images, R, T] float32) one call holds, at most
# (_coarse_batch_bytes caps it by what the device can still hand out).
_COARSE_BATCH_BYTES = 2 << 30

# Bytes of particles' summed diff2 ([P, K * R, T] float32) one significance call cuts.
_SIGNIFICANCE_BATCH_BYTES = 512 << 20


def _coarse_gemm_projections(
    projector_half, rotations, layout: CoarseScoreLayout, *, model_max_r: int, padding_factor: int, texture=None
):
    """The score-window rows of a RELION ``PPref`` half at ``rotations`` ``[N, 3, 3]`` (see
    :func:`_score_window_projections` for the two layouts).

    The projection the SPA coarse GEMM scorer reads (significance.py ``_project_relion_compact_score_rows``):
    RELION's texture interpolation with the coarse diff2 kernel's row rule, in ``layout``'s score pixels.
    ``texture`` is the half's :class:`relax.cuda.kernels.RelionCapacityHalfTextureF32`, staged once per pass.
    """

    return _score_window_projections(
        projector_half,
        rotations,
        jnp.asarray(layout.score_indices_np, dtype=jnp.int32),
        image_shape=tuple(layout.image_shape),
        current_size=int(layout.current_size),
        model_max_r=int(model_max_r),
        padding_factor=int(padding_factor),
        texture=texture,
    )


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

    from relax.helpers.projection import relion_capacity_texture_serves

    projector_half = jnp.asarray(projector_half)
    if jax.default_backend() != "gpu" or not relion_capacity_texture_serves(
        projector_half,
        r_max=int(model_max_r),
        padding_factor=int(padding_factor),
        projector_output_size=int(layout.current_size),
        relion_texture_interp=True,
    ):
        return None
    from relax.cuda.kernels import RelionCapacityHalfTextureF32

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


def _score_window_projections(
    projector_half, rotations, score_indices, *, image_shape, current_size, model_max_r, padding_factor, texture
):
    """Score-window projections: float32 ``[N, 2 P]`` packed ``[Re | Im]`` straight from ``texture`` where it
    serves, else complex64 ``[N, P]``; the GEMM scorer takes either."""

    from relax.helpers.projection import (
        compute_relion_projector_projections_block,
        project_relion_coarse_packed_rows,
        relion_coarse_packed_rows_serve,
    )

    if texture is not None and relion_coarse_packed_rows_serve(image_shape[0], current_size, model_max_r):
        return project_relion_coarse_packed_rows(
            texture, rotations, score_indices, image_shape=image_shape, projector_output_size=int(current_size)
        )
    projected, _abs2 = compute_relion_projector_projections_block(
        projector_half,
        rotations,
        image_shape,
        r_max=int(model_max_r),
        padding_factor=int(padding_factor),
        return_abs2=False,
        centered_rows=True,
        dense_scale=True,
        projector_output_size=int(current_size),
        pixel_indices=score_indices,
        relion_texture_interp=True,
        relion_kernel="coarse",
        capacity_texture=texture,
    )
    return projected


@partial(jax.jit, static_argnames=("image_shape",))
def _images_coarse_gemm_diff2(projected, unshifted, pixel_weight, initial_diff2, translation_angles, score_indices, *, image_shape):
    """Coarse diff2 ``[N, T, R]`` (translation-major) of ``N`` tilt images with their own projections ``[N, R, P]``
    and phases ``[N, T, 2]``.

    Each image is scored by the SPA coarse GEMM scorer (:func:`relax.scoring.scoring._relion_coarse_gaussian_gemm_scores_jit`,
    RELION's ``d0 + 0.5 sum w |p - y|^2`` as two real-packed float32 GEMMs) on its shifted pixels from RELION's
    score translation; a zero-weight padded image scores zero.
    """

    from relax.cuda import kernels as em_cuda_kernels
    from relax.scoring.scoring import _relion_coarse_gaussian_gemm_scores_jit

    n_images, n_trans = (int(n) for n in translation_angles.shape[:2])
    # Every image's translations in one launch, each with its own phases.
    shifted_all = em_cuda_kernels.relion_translate_score_f32(
        unshifted, translation_angles, score_indices, image_shape
    ).reshape(n_images, n_trans, -1)

    def one(reference, shifted, weight, initial):
        scores = _relion_coarse_gaussian_gemm_scores_jit(
            reference,
            None,
            shifted[None],
            weight[None],
            initial.reshape(1),
            jnp.int32(1),
            n_images=1,
            n_trans=n_trans,
            image_shape=image_shape,
            volume_shape=(1, 1, 1),
            translation_major=True,
        )
        return -scores[0]

    return jax.vmap(one)(projected, shifted_all, pixel_weight, initial_diff2)


def _images_gemm_error_bound(projected, unshifted, pixel_weight, initial_diff2):
    """:func:`relax.scoring.exact_cut.gemm_error_bound` ``[N]`` of images with their own projections ``[N, R, P]``
    (complex, or ``[N, R, 2 P]`` packed ``[Re | Im]``): each image's largest model energy over its rotations."""

    weight = pixel_weight.astype(jnp.float32)
    if jnp.iscomplexobj(projected):
        squares = projected.real * projected.real + projected.imag * projected.imag
    else:
        n_pixels = projected.shape[-1] // 2
        squares = projected[..., :n_pixels] * projected[..., :n_pixels] + projected[..., n_pixels:] * projected[..., n_pixels:]
    model_energy = jnp.max(jnp.sum(squares * weight[:, None, :], axis=-1), axis=1)
    image_energy = jnp.sum((unshifted.real * unshifted.real + unshifted.imag * unshifted.imag) * weight, axis=-1)
    return gemm_error_bound(model_energy, image_energy, initial_diff2, int(weight.shape[-1]))


@partial(
    jax.jit,
    static_argnames=("count", "image_shape", "current_size", "model_max_r", "padding_factor", "texture"),
)
def _coarse_gemm_slot_block(
    total, error, projector_half, rotations, unshifted, pixel_weight, initial_diff2, translation_angles, score_indices,
    first, *, count, image_shape, current_size, model_max_r, padding_factor, texture,
):
    """One batch's slots ``first:first + count`` in one program: each image's projections, its GEMM diff2,
    added in slot order, and the images' GEMM error bounds (:func:`_images_gemm_error_bound`) added to
    ``error`` ``[P]``.

    ``rotations`` ``[P, S, R, 3, 3]``, ``unshifted``/``pixel_weight`` ``[P, S, pixels]``, ``initial_diff2``
    ``[P, S]`` and ``translation_angles`` ``[P, S, T, 2]`` are the whole batch's; ``total`` ``[P, T, R]`` is
    translation-major, the GEMM's layout (the per-cell values do not depend on it), and is transposed once per
    batch rather than once per image.
    """

    def block(x):  # slots first:first + count; ``first`` is traced, so one program serves every block of a batch
        return jax.lax.dynamic_slice_in_dim(x, first, int(count), axis=1)

    rotations, unshifted, pixel_weight = block(rotations), block(unshifted), block(pixel_weight)
    initial_diff2, translation_angles = block(initial_diff2), block(translation_angles)
    n_particles, n_slots, n_rot = (int(n) for n in rotations.shape[:3])
    n_images = n_particles * n_slots
    projected = _score_window_projections(
        projector_half,
        rotations.reshape(n_images * n_rot, 3, 3),
        score_indices,
        image_shape=image_shape,
        current_size=current_size,
        model_max_r=model_max_r,
        padding_factor=padding_factor,
        texture=texture,
    ).reshape(n_images, n_rot, -1)
    image_diff2 = _images_coarse_gemm_diff2(
        projected,
        unshifted.reshape(n_images, -1),
        pixel_weight.reshape(n_images, -1),
        initial_diff2.reshape(n_images),
        translation_angles.reshape(n_images, translation_angles.shape[2], 2),
        score_indices,
        image_shape=image_shape,
    )
    bound = _images_gemm_error_bound(
        projected, unshifted.reshape(n_images, -1), pixel_weight.reshape(n_images, -1), initial_diff2.reshape(n_images)
    )
    return (
        _add_image_diff2_in_slot_order(total, image_diff2.reshape(n_particles, n_slots, -1, n_rot)),
        error + jnp.sum(bound.reshape(n_particles, n_slots), axis=1),
    )


@jax.jit
def _add_image_diff2_in_slot_order(total, image_diff2):
    """``total`` plus the particles' images' diff2 ``[P, S, ...]``, slot by slot (``img_id`` order)."""

    for slot in range(int(image_diff2.shape[1])):
        total = total + image_diff2[:, slot]
    return total


@partial(jax.jit, static_argnames=("capacity",))
def _significant_cells(mask, *, capacity: int):
    """Flat ids (row-major over ``mask`` ``[P, N]``) of its true cells, ascending, padded to ``capacity``."""

    return jnp.nonzero(mask.reshape(-1), size=capacity, fill_value=-1)[0]


# Rotations per particle whose samples the adaptive-fraction rule sorts (exact_cut.undecided_rotations).
_SORTED_ROTATIONS = 256

# Undecided rotations per particle scored again in one go; a flush with a particle holding more takes a larger size.
_UNDECIDED_ROTATIONS = 8


@partial(jax.jit, static_argnames=("max_significants", "adaptive_fraction", "sorted_rotations"))
def _near_cut_rows(
    diff2, values, mask, cutoff_count, error, n_slots,
    *, max_significants: int, adaptive_fraction: float, sorted_rotations: int | None = _SORTED_ROTATIONS,
):
    """:func:`relax.scoring.exact_cut.undecided_rotations` of particles from their summed GEMM diff2 ``[P, K * R, T]``.

    The particle's bound is its images' GEMM bounds ``error`` plus one float32 unit of its largest diff2 per image
    for the two slot-order sums and four for forming the log weights.
    """

    scored = jnp.isfinite(values).reshape(diff2.shape)
    largest = jnp.max(jnp.where(scored, diff2, 0.0), axis=(1, 2))
    smallest = jnp.min(jnp.where(scored, diff2, jnp.inf), axis=(1, 2))
    margin = 2.0 * (error + (n_slots.astype(jnp.float32) + 4.0) * float32_unit(largest))
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


def _device_share_bytes(cap: int) -> int:
    """At most ``cap`` and a quarter of what the device can still hand out (``cap`` when unknown)."""

    from relax.sparse_pass2.sparse_pass2_budget import (
        _device_free_memory_bytes,
        _jax_allocator_free_memory_bytes,
        _jax_allocator_pool_free_bytes,
        device_available_bytes,
    )

    available = device_available_bytes(
        _device_free_memory_bytes(), _jax_allocator_free_memory_bytes(), _jax_allocator_pool_free_bytes()
    )
    if available is None:
        return int(cap)
    return int(min(int(cap), max(256 << 20, 0.25 * float(available))))


def _coarse_operand_block_bytes() -> int:
    """At most ``_COARSE_OPERAND_BLOCK_BYTES`` and a quarter of what the device can still hand out.

    The coarse pass runs next to the refinement's resident state; a fixed 4 GiB block did not fit
    beside it at iteration 12 of a 5k-particle box-256 run (etbench w2_02_n5k, 3.09 GiB refused).
    """

    return _device_share_bytes(_COARSE_OPERAND_BLOCK_BYTES)


def _coarse_batch_bytes() -> int:
    """At most ``_COARSE_BATCH_BYTES`` and a quarter of what the device can still hand out.

    A fixed 2 GiB batch, next to half 1's resident state on a 16 GB P100, asked for a 6.42 GiB program
    buffer and ran out of memory (Polar 413469, cryoet_s1 iteration 1, half 2).
    """

    return _device_share_bytes(_COARSE_BATCH_BYTES)


def _coarse_projection_bytes_per_pixel(
    class_textures, projector_dtype, *, image_size, current_size, model_max_r
) -> int:
    """Device bytes one projected score pixel takes in :func:`_coarse_gemm_slot_block`.

    The texture's packed float32 ``[Re | Im]`` rows take 8 bytes. The JAX projection the pass falls back to
    (no texture: a complex128 projector, or a window the packed rows do not serve) holds its rows in the
    projector's complex dtype and a temporary of the same size: 6.42 GiB for 164 images x 4608 rotations x
    (81 translations x 4 B + 278 pixels x 32 B) with a complex128 projector (Polar 413469).
    """

    from relax.helpers.projection import relion_coarse_packed_rows_serve

    if all(t is not None for t in class_textures) and relion_coarse_packed_rows_serve(
        int(image_size), int(current_size), int(model_max_r)
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

    n_images = int(image_stop) - int(image_start)
    n_rows = -(-n_images // _OPERAND_IMAGE_BATCH) * _OPERAND_IMAGE_BATCH
    buffers = None
    for offset in range(0, n_images, _OPERAND_IMAGE_BATCH):
        indices = np.arange(
            int(image_start) + offset, min(int(image_start) + offset + _OPERAND_IMAGE_BATCH, int(image_stop))
        )
        padded = np.concatenate([indices, np.full(_OPERAND_IMAGE_BATCH - indices.size, indices[-1])])
        values = tilt_image_coarse_operands(
            experiment_dataset,
            padded,
            layout,
            noise_variance_half=noise_variance_half,
            optics_group_ids=optics_group_ids,
            scale_corrections=scale_corrections,
        )
        if buffers is None:
            buffers = [jnp.zeros((n_rows,) + v.shape[1:], dtype=v.dtype) for v in values]
        buffers = [_put_rows(b, v, jnp.int32(offset)) for b, v in zip(buffers, values)]
    return tuple(buffers)


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

    from relax.helpers.oversampling import relion_cuda_f32_coarse_log_weights
    from relax.scoring.coarse_publication import _posterior_statistics

    # RELION's left-to-right float32 order pdf_orientation + pdf_offset + min_diff2 - diff2
    # (cuda_kernel_weights_exponent_coarse). Adding the priors to the absolute scores first and the
    # min_diff2 offset afterwards rounds differently and can move near-tie cells across the
    # significance cut (relax.helpers.oversampling.relion_cuda_f32_coarse_log_weights).
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
    statistics = _posterior_statistics(
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
    (:func:`relax.sampling._relion_adaptive_pass1_rotations`). A global search builds the whole
    batch's images in one call; a local search builds each particle's own rotations. A particle's
    rotations beyond its own count repeat its last one, and padded slots and particles are zero.
    """

    from relax.refinement import tomo_particles

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
        per_unit = []
        for unit, unit_rows in zip(units, unit_rotation_rows):
            images = np.arange(offsets[unit], offsets[unit + 1])
            left, _applies = tomo_particles.relion_left_matrices(image_left[images])
            built = jnp.asarray(
                pass1_rotations(coarse_eulers_deg[unit_rows], random_perturbation, angular_sampling_deg, left_matrices=left),
                dtype=jnp.float32,
            )
            rot = jnp.asarray(np.minimum(np.arange(r_pad), int(built.shape[1]) - 1))
            per_unit.append(jnp.pad(built[:, rot], ((0, slots - images.size), (0, 0), (0, 0), (0, 0))))
        # jnp.pad zero-fills the padded slots; padded particles are zeros.
        return jnp.stack(per_unit + [jnp.zeros_like(per_unit[0])] * (p_pad - len(units)))


@jax.jit
def _lay_out_rotations(built, rows, rot, valid):
    """``built[rows, rot]`` with invalid slots zeroed, as one device program (one output buffer)."""

    laid_out = built[rows[:, :, None], rot[None, None, :]]
    return jnp.where(valid[:, :, None, None, None], laid_out, jnp.zeros((), laid_out.dtype))


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
    image_size: int,
    optics_group_ids=None,
    scale_corrections=None,
    unit_rotation_ids=None,
    unit_rotation_log_priors=None,
):
    """Each particle's coarse significant samples, ``rot * T + t`` int32 ids per unit, and its coarse Pmax.

    RELION's GPU coarse pass for a subtomogram (acc_ml_optimiser_impl.h:1190-1402): every tilt image is
    scored with its own device matrices (``make_eulers_3D`` with the image's left matrix ``image_left``,
    its ``Aproj`` with the optics group's magnification, :func:`relax.refinement.tomo_half.tilt_left_matrices`;
    :func:`relax.sampling._relion_adaptive_pass1_rotations`), its phases for the 3D trial shifts of ``Aproj``
    (``image_projections``) plus
    the rounded old offset (:func:`relax.refinement.tomo_particles.tilt_translation_angles`) and its own
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
    """

    from relax.helpers.projection import relion_projector_half_to_texture_full
    from relax.refinement import tomo_particles
    from relax.sampling import _relion_adaptive_pass1_rotations

    class_projectors = tuple(projector_half) if isinstance(projector_half, (tuple, list)) else (projector_half,)
    n_classes = len(class_projectors)
    if n_classes > 1 and (
        unit_rotation_ids is not None or rotation_log_prior is None or np.ndim(rotation_log_prior) != 2
    ):
        raise ValueError("a K-class coarse pass is a global search with one rotation log prior per class, [K, R]")
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
    block_images = max(1, _coarse_operand_block_bytes() // operand_bytes_per_image)
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
    batches = _coarse_batches(
        rotation_counts,
        n_slots=slots,
        n_trans=n_coarse_trans,
        n_pixels=int(layout.score_indices_np.size),
        projection_bytes_per_pixel=_coarse_projection_bytes_per_pixel(
            class_textures,
            jnp.asarray(class_projectors[0]).dtype,
            image_size=int(layout.image_shape[0]),
            current_size=int(layout.current_size),
            model_max_r=int(model_max_r),
        ),
        budget_bytes=_coarse_batch_bytes(),
    )
    score_indices = jnp.asarray(layout.score_indices_np, dtype=jnp.int32)
    r_pad_all = batches[0][1] if batches else 0
    # The significance of several batches' particles runs as one call, [P_sig, R_pad * T] values
    # within the batch budget; a particle's padded rotations carry a -inf prior and are never significant.
    significance_batch = max(1, _SIGNIFICANCE_BATCH_BYTES // max(n_classes * r_pad_all * n_coarse_trans * 4, 1))
    class_full = [None] * n_classes  # each class's full projector for the direct-square kernel, built on first use
    rescored, n_scored = [], 0
    supports, pmax_by_unit = [{} for _ in range(n_classes)], {}
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
            rotation_prior = np.full((batch_units.size, r_pad_all), -np.inf, dtype=np.float32)
            for i, unit in enumerate(batch_units):
                rotation_prior[i, : int(rotation_counts[unit])] = np.asarray(unit_rotation_log_priors[unit], np.float32)
            return rotation_prior
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

    def exact_cut(units_all, diff2, stats, error, n_images, operands):
        """The flushed particles' cut on exact scores: the rotations the GEMM rounding leaves undecided at a
        particle's cut (max_significants or the adaptive fraction) and at its smallest diff2 are scored again by
        the direct-square kernel, and the cut is taken on those values (:mod:`relax.scoring.exact_cut`).

        One read for the flush: which rotations. Their scorer matrices are built for those rotations alone.
        """

        cut_inputs = (diff2, stats["log_weights"], stats["mask"], stats["cutoff_count"], error, n_images)
        rows, most, unresolved = _near_cut_rows(*cut_inputs, **cut_static)
        ids, use = first_rows(rows, capacity=_UNDECIDED_ROTATIONS)
        ids, use, most, unresolved = jax.device_get((ids, use, most, unresolved))
        if int(unresolved) or int(most) > _UNDECIDED_ROTATIONS:
            if int(unresolved):
                # A particle's weight outside its sorted rotations reaches its cut: the rule on every sample.
                rows, most, _ = _near_cut_rows(*cut_inputs, **{**cut_static, "sorted_rotations": None})
            ids, use = jax.device_get(first_rows(rows, capacity=row_capacity(int(most), rows.shape[1], least=8)))
        if not use.any():
            return stats
        for class_index, class_projector in enumerate(class_projectors):
            if class_full[class_index] is None:
                class_full[class_index] = relion_projector_half_to_texture_full(class_projector).astype(jnp.complex64)
        row_rotations = np.zeros((units_all.size, slots, ids.shape[1], 3, 3), dtype=np.float32)
        for p in np.flatnonzero(use.any(axis=1)):
            unit = int(units_all[p])
            images = np.arange(offsets[unit], offsets[unit + 1])
            left, _applies = tomo_particles.relion_left_matrices(image_left[images])
            # A padded rotation repeats the particle's last one (_batch_scoring_rotations).
            grid_rows = np.minimum(np.where(use[p], ids[p] % r_pad_all, 0), int(rotation_counts[unit]) - 1)
            if local:
                grid_rows = unit_rotations[unit][grid_rows]
            row_rotations[p, : images.size] = np.asarray(
                _relion_adaptive_pass1_rotations(
                    coarse_eulers_deg[grid_rows], random_perturbation, angular_sampling_deg, left_matrices=left
                ),
                dtype=np.float32,
            )
        unshifted, weight, initial, angles = (jnp.concatenate(parts, axis=0) for parts in zip(*operands))
        diff2 = _exact_rows_diff2(
            diff2,
            jnp.asarray(ids),
            jnp.asarray(use),
            jnp.asarray(row_rotations),
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
        rescored.append((int(use.sum()), int(use.any(axis=1).sum())))
        return significance(units_all, diff2)

    def dispatch():
        units_all = np.concatenate([u for u, *_ in pending])
        diff2 = jnp.concatenate([t for _, t, *_ in pending], axis=0)
        error = jnp.concatenate([e for _, _, e, *_ in pending], axis=0)
        n_images = jnp.asarray(np.concatenate([n for _, _, _, n, _ in pending]))
        operands = [o for *_, o in pending]
        pending.clear()
        stats = exact_cut(units_all, diff2, significance(units_all, diff2), error, n_images, operands)
        # The significant cells are compacted on the device; only their ids come back (the dense mask is
        # K * R * T booleans per particle).
        counts = jnp.sum(stats["mask"], axis=1, dtype=jnp.int32)
        flat = _significant_cells(stats["mask"], capacity=cells_capacity) if cells_capacity else None
        for value in (counts, flat, stats["pmax"]):
            if value is not None:
                value.copy_to_host_async()
        return units_all, stats["mask"], counts, flat, stats["pmax"]

    def collect(units_all, mask, counts, flat, pmax):
        nonlocal cells_capacity
        counts = np.asarray(counts, dtype=np.int64)
        n_significant = int(counts.sum())
        if flat is None or n_significant > int(flat.shape[0]):
            flat = _significant_cells(mask, capacity=max(1, 1 << (max(n_significant, 1) - 1).bit_length()))
        cells_capacity = max(int(cells_capacity), 1 << max(2 * n_significant - 1, 1).bit_length())
        flat = np.asarray(flat, dtype=np.int64)[:n_significant]
        particle_pmax = np.asarray(pmax, dtype=np.float64)
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
    for units, r_pad, p_pad, slot_block in batches:
        # Operands of the batch, padded to [P_pad, S, R_pad, ...]: padded rotations repeat the particle's
        # last one, padded slots and particles carry zero weight (their diff2 adds zeros and is not read).
        # The scorer matrices are built and laid out on the device (_batch_scoring_rotations).
        angles = np.zeros((p_pad, slots, n_coarse_trans, 2), dtype=np.float32)
        image_index = np.zeros((p_pad, slots), dtype=np.int64)
        image_valid = np.zeros((p_pad, slots), dtype=bool)
        rotations = _batch_scoring_rotations(
            units,
            offsets,
            image_left,
            coarse_eulers_deg,
            None if not local else [unit_rotations[unit] for unit in units],
            random_perturbation,
            angular_sampling_deg,
            p_pad=p_pad,
            slots=slots,
            r_pad=r_pad,
            pass1_rotations=_relion_adaptive_pass1_rotations,
        )
        for p, unit in enumerate(units):
            images = np.arange(offsets[unit], offsets[unit + 1])
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
        batch_angles = jnp.asarray(angles)
        class_totals, error = [], None
        for class_index, class_projector in enumerate(class_projectors):
            total = jnp.zeros((p_pad, n_coarse_trans, r_pad), dtype=jnp.float32)
            class_error = jnp.zeros(p_pad, dtype=jnp.float32)
            for first in range(0, slots, slot_block):
                total, class_error = _coarse_gemm_slot_block(
                    total,
                    class_error,
                    class_projector,
                    rotations,
                    batch_unshifted,
                    batch_weight,
                    batch_initial,
                    batch_angles,
                    score_indices,
                    slot_offsets[first],
                    count=int(min(slot_block, slots - first)),
                    image_shape=tuple(layout.image_shape),
                    current_size=int(layout.current_size),
                    model_max_r=int(model_max_r),
                    padding_factor=int(padding_factor),
                    texture=class_textures[class_index],
                )
            class_totals.append(total[: units.size].swapaxes(1, 2))
            error = class_error if error is None else jnp.maximum(error, class_error)
            last_total = total
        # K>1: [P, K * R_pad, T], class-major along the rotation axis.
        diff2 = class_totals[0] if n_classes == 1 else jnp.concatenate(class_totals, axis=1)
        n_scored += int(units.size) * n_classes * r_pad * n_coarse_trans
        # The flush cuts these particles, and scores their undecided rotations again from these operands.
        pending.append(
            (
                units,
                diff2,
                error[: units.size],
                image_valid[: units.size].sum(axis=1),
                (
                    batch_unshifted[: units.size], batch_weight[: units.size], batch_initial[: units.size],
                    batch_angles[: units.size],
                ),
            )
        )
        # Only operands_for and the pending particles keep operands, so a new block is allocated after the old one is freed.
        del unshifted, weight, initial, batch_unshifted, batch_weight, batch_initial
        if sum(int(u.size) for u, *_ in pending) >= significance_batch:
            flush()
    flush(last=True)
    if rescored:
        n_rows, n_particles = (int(sum(int(v) for v in column)) for column in zip(*rescored))
        logger.info(
            "Coarse cut: %d of %d particles had rotations within the GEMM error bound of their cut or of their "
            "smallest diff2; %d rotations scored again by the direct square (%.3g%% of the pass's samples)",
            n_particles, n_units, n_rows, 100.0 * n_rows * n_coarse_trans / max(n_scored, 1),
        )
    for k, texture in enumerate(class_textures):
        if texture is not None and last_total is not None:
            # The flush read every batch's significance back, so the last projection has completed; the
            # texture stays for the next pass (_coarse_capacity_texture), which refills it after this result.
            _CLASS_TEXTURES[k] = (texture, last_total)
    class_supports = [[supports[k][u] for u in range(n_units)] for k in range(n_classes)]
    return class_supports[0] if n_classes == 1 else class_supports, np.asarray(
        [pmax_by_unit[u] for u in range(n_units)], dtype=np.float64
    )


# ---------------------------------------------------------------------------
# --firstiter_cc: the coarse normalized-CC winner of each particle
# ---------------------------------------------------------------------------


def tilt_image_cc_coarse_operands(experiment_dataset, image_indices, window_indices, half_weights, *, scale_corrections=None):
    """Tilt images' ``--firstiter_cc`` coarse operands: the SPA exact CC operands, without pre-shift or norm.

    The unshifted corrected score image and the CC pixel weight ``corr_img * half_weights`` in the
    score window (significance.py's exact normalized-CC coarse pass: ``assemble_relion_cc_coarse_operands``
    of the per-image RELION FFT, RFLOAT CTF and ``1 / sum |X|^2``). RELION neither translates nor
    normalises a tomo image (acc_ml_optimiser_impl.h:429-476).
    """

    from relax.helpers.batch_fetch import fetch_indexed_batch
    from relax.helpers.preprocessing import prepare_batch_preprocess_operands
    from relax.relion.relion_coarse_operands import (
        _process_relion_exact_coarse_half_image,
        _relion_cc_inverse_power_from_processed,
        assemble_relion_cc_coarse_operands,
    )
    from relax.relion.relion_ctf import _relion_exact_ctf_half_from_source_star

    image_indices = np.asarray(image_indices, dtype=np.int64)
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
        experiment_dataset, batch_data, True, relion_preprocess_kwargs=preprocess_kwargs, image_indices=image_indices
    )
    window = jnp.asarray(window_indices, dtype=jnp.int32)
    operands = assemble_relion_cc_coarse_operands(
        processed,
        _relion_exact_ctf_half_from_source_star(experiment_dataset, image_indices, experiment_dataset.image_shape),
        _relion_cc_inverse_power_from_processed(processed, window),
        jnp.asarray(batch_scale, dtype=jnp.float32),
        phase_factors=None,
        window_indices=window,
        scale_corrections_enabled=scale_corrections is not None,
    )
    return (
        jnp.asarray(operands.windowed_unshifted, dtype=jnp.complex64),
        jnp.asarray(operands.windowed_corr_img * jnp.asarray(half_weights)[window], dtype=jnp.float32),
    )


@jax.jit
def _add_tilt_image_cc_diff2(running_diff2, projections, shifted, pixel_weight):
    """``running_diff2`` ``[R, T]`` plus one tilt image's coarse CC term, RELION's way.

    ``cuda_kernel_diff2_CC_coarse`` has each of its 128 threads atomically add
    ``-X / (128 sqrt(A))`` to the particle's diff2, which already holds the earlier images' terms;
    ``X`` and ``A`` are the SPA coarse CC GEMM terms (scoring._relion_coarse_gemm_terms).
    """

    from relax.scoring.scoring import _relion_coarse_gemm_terms

    n_trans = int(shifted.shape[0])
    cross, model_energy, _, _ = _relion_coarse_gemm_terms(
        projections, shifted[None], pixel_weight[None], 1, n_images=1, n_trans=n_trans, wide=jnp.float32
    )
    contribution = jnp.asarray(cross[0].T, dtype=jnp.float32) / (
        jnp.float32(128.0) * jnp.sqrt(jnp.maximum(model_energy[0][:, None], jnp.float32(1e-30)))
    )
    return jax.lax.fori_loop(0, 128, lambda _, acc: acc - contribution, running_diff2, unroll=True)


def particle_coarse_cc_winners(
    experiment_dataset,
    *,
    unit_image_offsets,
    image_projections,
    image_left,
    unit_old_offsets_px,
    coarse_eulers_deg,
    relion_order,
    random_perturbation,
    angular_sampling_deg,
    coarse_translations_px,
    relion_projector_half,
    relion_projector_r_max: int,
    padding_factor: int,
    coarse_size: int,
    image_size: int,
    scale_corrections=None,
):
    """Each particle's coarse ``--firstiter_cc`` winner: the cell ``rotation * T + translation`` and its CC.

    RELION's first CC iteration sums every tilt image's normalized CC into the particle's diff2
    (acc_ml_optimiser_impl.h:1290-1347, no Xi2 offset and no priors) and keeps the minimum alone
    (convertAllSquaredDifferencesToWeights' CC branch); pass 2 then scores that sample's children.
    Each image is scored with its own coarse matrices ``L R`` (``image_left``) and phases, as the Gaussian pass
    (:func:`particle_coarse_supports`). ``relion_order`` ``[R]`` is each coarse rotation's position in
    RELION's orientation order: the first maximum in that order wins, as RELION's ordered minimum.
    ``scale_corrections`` are per dataset image.
    """

    from relax.cuda import kernels as em_cuda_kernels
    from relax.helpers.fourier_window import make_fourier_window_spec
    from relax.helpers.half_spectrum import make_scoring_half_image_weights
    from relax.helpers.projection import compute_relion_projector_projections_block
    from relax.refinement import tomo_particles
    from relax.sampling import _relion_adaptive_pass1_rotations

    image_shape = tuple(int(n) for n in experiment_dataset.image_shape)
    n_half = image_shape[0] * (image_shape[1] // 2 + 1)
    # RELION's CC kernels sum every element of the windowed FFTW image, the DC and the redundant x=0
    # column included, and Xi2 is its full power (ml_optimiser.cpp:6846-6855): the square current-size
    # crop of the CC pass (sparse_pass2_window's normalized_cc window, _pass2_half_weights).
    window = np.asarray(
        make_fourier_window_spec(
            image_shape,
            int(coarse_size),
            n_half,
            square=False,
            score_square=True,
            score_include_dc=True,
            include_recon_window=False,
        ).score_indices_np,
        dtype=np.int32,
    )
    half_weights = make_scoring_half_image_weights(image_shape, relion_half_sum=True, exclude_relion_redundant_x0=False)
    offsets = np.asarray(unit_image_offsets, dtype=np.int64)
    image_projections = np.asarray(image_projections, dtype=np.float64)
    image_left = np.asarray(image_left, dtype=np.float64)
    old = tomo_particles.relion_gpu_old_offsets(np.asarray(unit_old_offsets_px, dtype=np.float64))
    coarse_eulers_deg = np.asarray(coarse_eulers_deg)
    n_rot = int(coarse_eulers_deg.shape[0])
    n_trans = int(np.asarray(coarse_translations_px).shape[0])
    relion_rank = np.asarray(relion_order, dtype=np.int64).reshape(-1)
    if relion_rank.shape != (n_rot,):
        raise ValueError("relion_order needs one position per coarse rotation")
    # Cells in RELION's orientation-major order, so the first maximum is RELION's.
    relion_cells = (np.argsort(relion_rank, kind="stable")[:, None] * n_trans + np.arange(n_trans)[None, :]).reshape(-1)
    window_device = jnp.asarray(window)
    n_units = int(offsets.size - 1)
    winners = np.zeros(n_units, dtype=np.int64)
    best = np.zeros(n_units, dtype=np.float64)
    for unit in range(n_units):
        images = np.arange(offsets[unit], offsets[unit + 1])
        unshifted, pixel_weight = tilt_image_cc_coarse_operands(
            experiment_dataset,
            images,
            window,
            half_weights,
            scale_corrections=None if scale_corrections is None else np.asarray(scale_corrections)[images],
        )
        left, _applies = tomo_particles.relion_left_matrices(image_left[images])
        rotations = np.asarray(
            _relion_adaptive_pass1_rotations(
                coarse_eulers_deg, random_perturbation, angular_sampling_deg, left_matrices=left
            ),
            dtype=np.float32,
        )
        angles = tomo_particles.tilt_translation_angles(
            coarse_translations_px, old[unit : unit + 1], image_projections[images], np.zeros(images.size, int), image_size
        )
        running = jnp.zeros((n_rot, n_trans), dtype=jnp.float32)
        for slot in range(images.size):
            projections, _ = compute_relion_projector_projections_block(
                relion_projector_half,
                jnp.asarray(rotations[slot]),
                image_shape,
                r_max=int(relion_projector_r_max),
                padding_factor=int(padding_factor),
                centered_rows=True,
                dense_scale=True,
                relion_texture_interp=True,
                relion_kernel="coarse",
                projector_output_size=int(coarse_size),
            )
            shifted = em_cuda_kernels.relion_translate_score_f32(
                unshifted[slot : slot + 1], jnp.asarray(angles[slot], dtype=jnp.float32), window_device, image_shape
            )
            running = _add_tilt_image_cc_diff2(
                running, jnp.asarray(projections, dtype=jnp.complex64)[:, window_device], shifted, pixel_weight[slot]
            )
        scores = -np.asarray(running, dtype=np.float32).reshape(-1)
        position = int(np.argmax(scores[relion_cells]))
        winners[unit] = int(relion_cells[position])
        best[unit] = float(scores[winners[unit]])
    return winners, best
