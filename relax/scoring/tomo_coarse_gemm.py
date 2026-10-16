"""The GEMM diff2 block of the subtomogram coarse pass.

``coarse_gemm_slot_block`` scores one block of image slots: each tilt image's window projections
(``_score_window_projections``) against its coarse operands as one GEMM per image with its rounding bound
(``_images_coarse_gemm_diff2``), the images' diff2 added per particle in slot order
(``_add_image_diff2_in_slot_order``). The significance pass (``tomo_coarse.particle_coarse_supports``) calls it
and rescores the rows the bound leaves near the cut; this module imports nothing from ``tomo_coarse``.
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
