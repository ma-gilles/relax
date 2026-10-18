"""The GEMM diff2 block of the subtomogram coarse pass.

``coarse_gemm_slot_block`` scores one block of image slots: each tilt image's window projections
(``_score_window_projections``) against its coarse operands as one GEMM per image with its rounding bound
(``_images_coarse_gemm_diff2``), the images' diff2 added per particle in slot order
(``_add_image_diff2_in_slot_order``). ``coarse_gemm_frame`` scores the images of a particle group that share one
projection left matrix (one tilt frame of one tomogram): the frame's projections once, the images' cross terms as
one GEMM, each image's model energy as its own product, as ``_images_coarse_gemm_diff2`` scores each image. The
significance pass (``tomo_coarse.particle_coarse_supports``) calls them and rescores the rows the bound leaves near
the cut; this module imports nothing from ``tomo_coarse``.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from relax.scoring.exact_cut import gemm_error_bound


def _score_window_projections(
    projector_half, rotations, score_indices, *, image_shape, current_size, model_max_r, padding_factor, texture
):
    """Score-window projections: float32 ``[N, 2 P]`` packed ``[Re | Im]`` straight from ``texture`` where it
    serves, else complex64 ``[N, P]``; the GEMM scorer takes either."""

    from relax.projection.projection import (
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
    and phases ``[N, T, 2]``, and each image's GEMM error bound ``[N]``.

    Each image is scored by the SPA coarse GEMM scorer (:func:`relax.scoring.coarse_kernels.relion_coarse_gaussian_gemm_scores_jit`,
    RELION's ``d0 + 0.5 sum w |p - y|^2`` as two real-packed float32 GEMMs) on its shifted pixels from RELION's
    score translation; a zero-weight padded image scores zero. The bound
    (:func:`relax.scoring.exact_cut.gemm_error_bound`) takes the image's largest model energy over its rotations from
    the scorer's own model-energy GEMM, and its image energy from its unshifted pixels.
    """

    from relax.cuda import translate as cuda_translate
    from relax.scoring.coarse_kernels import relion_coarse_gaussian_gemm_scores_jit

    n_images, n_trans = (int(n) for n in translation_angles.shape[:2])
    # Every image's translations in one launch, each with its own phases.
    shifted_all = cuda_translate.relion_translate_score_f32(
        unshifted, translation_angles, score_indices, image_shape
    ).reshape(n_images, n_trans, -1)

    def one(reference, shifted, weight, initial):
        scores, model_energy = relion_coarse_gaussian_gemm_scores_jit(
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
            return_model_energy=True,
        )
        return -scores[0], jnp.max(model_energy[0])

    diff2, model_energy = jax.vmap(one)(projected, shifted_all, pixel_weight, initial_diff2)
    weight = pixel_weight.astype(jnp.float32)
    image_energy = jnp.sum((unshifted.real * unshifted.real + unshifted.imag * unshifted.imag) * weight, axis=-1)
    return diff2, gemm_error_bound(model_energy, image_energy, initial_diff2, int(weight.shape[-1]))


@partial(
    jax.jit,
    static_argnames=("count", "image_shape", "current_size", "model_max_r", "padding_factor", "texture"),
)
def coarse_gemm_slot_block(
    total, error, projector_half, rotations, unshifted, pixel_weight, initial_diff2, translation_angles, score_indices,
    first, *, count, image_shape, current_size, model_max_r, padding_factor, texture,
):
    """One batch's slots ``first:first + count`` in one program: each image's projections, its GEMM diff2,
    added in slot order, and the images' GEMM error bounds (:func:`_images_coarse_gemm_diff2`) added to
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
    image_diff2, bound = _images_coarse_gemm_diff2(
        projected,
        unshifted.reshape(n_images, -1),
        pixel_weight.reshape(n_images, -1),
        initial_diff2.reshape(n_images),
        translation_angles.reshape(n_images, translation_angles.shape[2], 2),
        score_indices,
        image_shape=image_shape,
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


@partial(jax.jit, static_argnames=("image_shape",))
def _frame_images_coarse_gemm_diff2(
    projected, unshifted, pixel_weight, initial_diff2, translation_angles, score_indices, *, image_shape
):
    """:func:`_images_coarse_gemm_diff2` of ``N`` tilt images that share the projections ``projected`` ``[R, 2 P]``
    (packed ``[Re | Im]``, or complex ``[R, P]``): diff2 ``[N, T, R]`` and each image's GEMM error bound ``[N]``.

    The cross terms of all the images are one GEMM against the shared rows; the model energy ``A`` is each image's
    own product of its weights with the rows' ``|p|^2`` (formed once), as the per-image scorer forms it, and the
    scores are assembled as :func:`relax.scoring.coarse_kernels.relion_coarse_gaussian_gemm_scores_jit` assembles
    them. A zero-weight image scores zero.
    """

    from relax.cuda import translate as cuda_translate
    from relax.scoring.coarse_kernels import relion_coarse_gemm_terms

    n_images, n_trans = (int(n) for n in translation_angles.shape[:2])
    shifted_all = cuda_translate.relion_translate_score_f32(
        unshifted, translation_angles, score_indices, image_shape
    ).reshape(n_images, n_trans, -1)
    wide = jnp.float32
    cross, _shared_energy, image_energy, active = relion_coarse_gemm_terms(
        projected, shifted_all, pixel_weight, jnp.int32(n_images), n_images=n_images, n_trans=n_trans, wide=wide
    )
    if jnp.iscomplexobj(projected):
        reference_re, reference_im = projected.real.astype(wide), projected.imag.astype(wide)
    else:
        n_pixels = projected.shape[-1] // 2
        reference_re, reference_im = projected[..., :n_pixels].astype(wide), projected[..., n_pixels:].astype(wide)
    abs2_rows = (reference_re * reference_re + reference_im * reference_im).T
    weight = jnp.where(active[:, None], pixel_weight, 0).astype(wide)
    # One image's weights at a time: the per-image scorer's product (a GEMM over the images rounds otherwise).
    model_energy = jax.lax.map(
        lambda w: jax.lax.dot(
            w[None], abs2_rows, precision=jax.lax.DotAlgorithmPreset.F32_F32_F32, preferred_element_type=wide
        )[0],
        weight,
    )
    initial = jnp.where(active, initial_diff2, 0).astype(wide)
    half = jnp.asarray(0.5, dtype=wide)
    scores = cross - half * model_energy[:, None, :] - half * image_energy[:, :, None] - initial[:, None, None]
    scores = jnp.where(active[:, None, None], scores, 0).astype(pixel_weight.dtype)
    weight32 = pixel_weight.astype(jnp.float32)
    unshifted_energy = jnp.sum((unshifted.real * unshifted.real + unshifted.imag * unshifted.imag) * weight32, axis=-1)
    return -scores, gemm_error_bound(
        jnp.max(model_energy, axis=1), unshifted_energy, initial_diff2, int(weight32.shape[-1])
    )


@partial(
    jax.jit,
    static_argnames=("image_shape", "current_size", "model_max_r", "padding_factor", "texture"),
    donate_argnums=(0, 1),
)
def coarse_gemm_frame(
    total, bounds, projector_half, frame_rotations, frame, unshifted, pixel_weight, initial_diff2, translation_angles,
    score_indices, unit_index, slot_index, *, image_shape, current_size, model_max_r, padding_factor, texture,
):
    """The images ``(unit_index, slot_index)`` of a particle group that share frame ``frame``'s projection left
    matrix, in one program: the frame's projections (its rotations ``frame_rotations[frame]`` ``[R, 3, 3]``) once,
    the images' diff2 (:func:`_frame_images_coarse_gemm_diff2`) added to their particles' ``total`` ``[G, T, R]``
    and their GEMM error bounds put at ``bounds[unit, slot]`` ``[G, S]``.

    The operands are the group's ``[G, S, ...]``; a particle has at most one image per frame, so each total takes
    one addition, and the frames go in every particle's slot order (``img_id``), so a particle's images add in that
    order. An entry with ``unit_index == G`` is padding and changes nothing.
    """

    n_group = int(total.shape[0])
    rotations = jax.lax.dynamic_index_in_dim(frame_rotations, frame, axis=0, keepdims=False)
    projected = _score_window_projections(
        projector_half,
        rotations,
        score_indices,
        image_shape=image_shape,
        current_size=current_size,
        model_max_r=model_max_r,
        padding_factor=padding_factor,
        texture=texture,
    )
    unit = jnp.minimum(unit_index, n_group - 1)
    real = unit_index < n_group
    image_diff2, bound = _frame_images_coarse_gemm_diff2(
        projected,
        unshifted[unit, slot_index],
        jnp.where(real[:, None], pixel_weight[unit, slot_index], 0),
        jnp.where(real, initial_diff2[unit, slot_index], 0),
        translation_angles[unit, slot_index],
        score_indices,
        image_shape=image_shape,
    )
    total = total.at[unit_index].add(image_diff2, mode="drop")
    return total, bounds.at[unit_index, slot_index].set(bound, mode="drop")


@partial(jax.jit, static_argnames=("slot_block",))
def slot_block_error(bounds, *, slot_block: int):
    """The particles' GEMM error bounds ``[G]`` from their images' ``bounds`` ``[G, S]``, summed as the slot-block
    programs sum them (:func:`coarse_gemm_slot_block`): block by block, each block's images first."""

    n_particles, n_slots = (int(n) for n in bounds.shape)
    error = jnp.zeros(n_particles, jnp.float32)
    for first in range(0, n_slots, int(slot_block)):
        count = min(int(slot_block), n_slots - first)
        error = error + jnp.sum(bounds[:, first : first + count].reshape(n_particles, count), axis=1)
    return error
