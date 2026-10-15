"""The PPCA kernels of librelax_cuda.so: the moment scatter and the streamed window projection, latent epilogue and
posterior preparation.

Each wrapper registers the library's targets through ``library.ensure_ffi()`` before its first FFI call.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from recovar import cuda_backproject

from relax.cuda import library


def ppca_moment_scatter_f32(
    lhs_images: jax.Array,
    rhs_parts: jax.Array,
    projections: jax.Array,
    pixel_indices: jax.Array,
    rotation_matrices: jax.Array,
    *,
    image_shape,
    volume_shape,
    max_r,
    metric_trace: bool,
) -> tuple[jax.Array, jax.Array]:
    """PPCA residuals of one rotation block and their backprojection with the metric images.

    ``lhs_images`` ``(tri(P), R, F)`` packed upper LHS images and ``rhs_parts``
    ``(P, R, 2F)`` RHS images (real parts, then imaginary parts) on the window
    ``pixel_indices`` of the half image (``-1`` pads: no contribution); ``projections`` ``(P, R, 2F)`` on the same window in the
    same planar layout (:func:`ppca_window_project_f32`); rotations ``(R, 3, 3)``. Returns ``(groups, V, 32)`` half volumes holding the
    metric channels (the packed LHS, or its trace with ``metric_trace``) and the real and
    imaginary parts of the residual ``R_p - sum_q L_pq A_q`` (channel ``c`` is lane ``c % 32``
    of group ``c // 32``), and the noise-power correction ``(F,)``
    ``sum_r [sum_pq L_pq Re(conj A_p A_q) - 2 sum_p Re(R_p conj A_p)]`` of
    :func:`relax.ppca_refinement.residual_statistics.residual_statistics_from_moment_images`.
    Each channel is recovar's linear-interpolation ``batch_backproject_indexed`` (half volume,
    half image, no RELION x-fold, unit upsampling) into a zero volume, up to float32 order.
    """

    library.ensure_ffi()
    lhs_images, rhs_parts, projections = (jnp.asarray(x) for x in (lhs_images, rhs_parts, projections))
    if any(x.dtype != jnp.float32 for x in (lhs_images, rhs_parts, projections)):
        raise ValueError("ppca_moment_scatter_f32 expects float32 LHS/RHS images and planar projections")
    kw, _, _ = cuda_backproject.ffi_kwargs(image_shape, volume_shape, 1, True, True, max_r, None)
    if int(kw["upsampling"]) != 1:
        raise ValueError("ppca_moment_scatter_f32 supports unit upsampling only")
    rot6 = cuda_backproject.rot_to_compact(jnp.asarray(rotation_matrices), jnp.float32)
    P = rhs_parts.shape[0]
    channels = (1 if metric_trace else P * (P + 1) // 2) + 2 * P
    N0, N1, N2 = (int(n) for n in volume_shape)
    out_types = (
        jax.ShapeDtypeStruct((-(-channels // 32), N0 * N1 * (N2 // 2 + 1), 32), jnp.float32),
        jax.ShapeDtypeStruct((lhs_images.shape[2],), jnp.float32),
    )
    return jax.ffi.ffi_call(library.TARGET_PPCA_MOMENT_SCATTER_F32, out_types, vmap_method="sequential")(
        lhs_images,
        rhs_parts,
        projections,
        jnp.asarray(pixel_indices, jnp.int32).reshape(-1),
        rot6,
        image_h=kw["image_h"],
        image_w=kw["image_w"],
        full_image_w=kw["full_image_w"],
        N0=kw["N0"],
        N1=kw["N1"],
        N2=kw["N2"],
        max_r2_x4=kw["max_r2_x4"],
        metric_trace=np.int64(int(metric_trace)),
    )


def ppca_window_project_f32(
    volume_voxel_major: jax.Array,
    pixel_indices: jax.Array,
    rotation_matrices: jax.Array,
    *,
    image_shape,
    volume_shape,
    max_r,
    with_products: bool,
) -> tuple[jax.Array, jax.Array]:
    """Windowed augmented projections of one rotation block, planar, with their packed pair products.

    ``volume_voxel_major`` ``(V, P)`` complex64 holds the ``P`` half volumes voxel-major;
    ``pixel_indices`` ``(F,)`` window the half image (``-1`` pads: zero output); rotations ``(R, 3, 3)``. Returns
    ``(P, R, 2F)`` float32 with the real parts of each component's windowed projection, then its
    imaginary parts, and, with ``with_products``, ``(tri(P), R, F)`` ``Re(conj(A_i) A_j)`` for the
    packed upper pairs (``np.triu_indices`` order), else an empty ``(0,)`` array. Each projection is
    recovar's linear-interpolation ``batch_slice_volume`` (half volume, half image, unit
    upsampling) at the windowed pixels, up to float32 association.
    """

    library.ensure_ffi()
    volume_voxel_major = jnp.asarray(volume_voxel_major)
    if volume_voxel_major.dtype != jnp.complex64:
        raise ValueError("ppca_window_project_f32 expects complex64 volumes")
    kw, _, _ = cuda_backproject.ffi_kwargs(image_shape, volume_shape, 1, True, True, max_r, None)
    if int(kw["upsampling"]) != 1:
        raise ValueError("ppca_window_project_f32 supports unit upsampling only")
    rot6 = cuda_backproject.rot_to_compact(jnp.asarray(rotation_matrices), jnp.float32)
    pixel_indices = jnp.asarray(pixel_indices, jnp.int32).reshape(-1)
    P, R, F = volume_voxel_major.shape[1], rot6.shape[0], pixel_indices.shape[0]
    out_types = (
        jax.ShapeDtypeStruct((P, R, 2 * F), jnp.float32),
        jax.ShapeDtypeStruct((P * (P + 1) // 2, R, F) if with_products else (0,), jnp.float32),
    )
    return jax.ffi.ffi_call(library.TARGET_PPCA_WINDOW_PROJECT_F32, out_types, vmap_method="sequential")(
        volume_voxel_major,
        rot6,
        pixel_indices,
        image_h=kw["image_h"],
        image_w=kw["image_w"],
        full_image_w=kw["full_image_w"],
        N0=kw["N0"],
        N1=kw["N1"],
        N2=kw["N2"],
        max_r2_x4=kw["max_r2_x4"],
        with_products=np.int64(int(with_products)),
    )


def ppca_latent_epilogue_f32(inner, gram, rows, prior_tables, start, kept):
    """Pass-1 pose scores, latent means and covariances of one rotation block, in place in ``kept``.

    ``inner`` ``(P, R, ld)`` holds ``Re<A_p, Y_bt>`` at column ``b T + t`` (``ld >= B T``: the
    score GEMM's aligned shift axis; the padding is ignored) and ``gram`` ``(tri(P), R, B)`` the
    CTF/noise-weighted ``Re(conj(A_i) A_j)``; ``rows`` ``(R,)`` the block's fine rows and
    ``prior_tables`` ``(rotation_parent, rotation_log_prior, translation_parent,
    translation_log_prior, coarse_mask)`` the pose log-prior of
    :func:`relax.ppca_refinement.full_row_stream.full_row_pose_log_prior`. ``kept`` is
    ``(score, mean, covariance, part_max, part_arg, part_sum)`` of shapes ``(cap, B, T)``,
    ``(q, cap, B, T)``, ``(cap, B, tri(q))`` and ``(cap, B)``; rows ``start .. start + R`` are
    overwritten (the buffers are aliased, so a donated ``kept`` is updated in place) with the
    formulation of :func:`relax.ppca_refinement.full_row_stream._latent_block` and each
    (rotation, image)'s maximum score, first maximizing translation and
    ``sum_t exp(score - maximum)``. Latent rank at most 16.
    """

    library.ensure_ffi()
    out_types = tuple(jax.ShapeDtypeStruct(x.shape, x.dtype) for x in kept)
    first = 9
    return jax.ffi.ffi_call(
        library.TARGET_PPCA_LATENT_EPILOGUE_F32,
        out_types,
        input_output_aliases={first + i: i for i in range(len(kept))},
        vmap_method="sequential",
    )(
        inner,
        gram,
        jnp.asarray(rows, jnp.int32),
        *prior_tables,
        jnp.reshape(jnp.asarray(start, jnp.int32), (1,)),
        *kept,
    )


def ppca_posterior_prep_f32(kept_score, kept_mean, kept_covariance, center, centered_logZ, shift_squared, start,
                            *, block_size: int, row_length: int | None = None):
    """Pass-2 posterior weights and moment sums of the kept rows ``start .. start + block_size``.

    With ``gamma = exp((score - center) - centered_logZ)`` per pose, returns the weights
    ``(P, R, row_length)`` ``[gamma, gamma E z_1, ...]`` at column ``b T + t``, zero in the
    padding up to ``row_length`` (default ``B T``; the RHS GEMM's aligned shift axis), the packed
    upper translation sums
    ``(tri(P), R, B)`` of ``gamma E[a_i a_j]`` for ``a = [1, z]``
    (:func:`relax.ppca_refinement.full_row_stream.second_moment_sums`), the partial diagnostics
    ``(2, R, B)`` (``-sum_t gamma log-gamma`` over ``gamma > 0`` and ``sum_t gamma |shift|^2``) and
    the count ``(R, B)`` of poses with ``gamma > 1e-3``.
    """

    library.ensure_ffi()
    cap, B, T = kept_score.shape
    q = kept_mean.shape[0]
    P, R = q + 1, int(block_size)
    row_length = B * T if row_length is None else int(row_length)
    if row_length < B * T:
        raise ValueError("row_length must hold every (image, translation) column")
    out_types = (
        jax.ShapeDtypeStruct((P, R, row_length), jnp.float32),
        jax.ShapeDtypeStruct((P * (P + 1) // 2, R, B), jnp.float32),
        jax.ShapeDtypeStruct((2, R, B), jnp.float32),
        jax.ShapeDtypeStruct((R, B), jnp.int32),
    )
    return jax.ffi.ffi_call(library.TARGET_PPCA_POSTERIOR_PREP_F32, out_types, vmap_method="sequential")(
        kept_score,
        kept_mean,
        kept_covariance,
        center,
        centered_logZ,
        shift_squared,
        jnp.reshape(jnp.asarray(start, jnp.int32), (1,)),
        block_size=np.int64(R),
    )
