"""Shared adjoint-slice wrappers for dense and local EM engines."""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from recovar import core


class ReferenceSphereClip(NamedTuple):
    """The M-step clip when the image radius alone is not RELION's backprojection support.

    RELION backprojects an image pixel ``k`` when its weight is nonzero, which bounds the
    image radius by the rounded support ``ROUND(|k|) <= image_current_size / 2`` (the
    resolution pointers zero ``Minvsigma2`` beyond it, ml_optimiser.cpp:6894-6901), and when
    its rotated, reference-grid radius ``|A^-1 k|`` is inside the model's r_max
    (acc/cuda/cuda_kernels/BP.cuh:322; BackProjector::backproject2Dto3D). recovar's kernel
    clips the image radius (and repeats the cut on the rotated radius), so two cases arise:

    - The projection matrix is ``s^-1`` times an orthogonal matrix (one grid, or an optics
      group on another pixel size or box): ``|A^-1 k| = |k| / s``, so the image clip at
      ``r_max * s`` is RELION's rule pixel for pixel. ``reference_radius`` is None.
    - Anisotropic magnification (``A = inv(M3) Aproj R``, ``M`` with unequal singular
      values): ``|A^-1 k| = |M k|`` varies with the direction of ``k``, so no image radius
      reproduces the rotated cut. The adjoint then keeps the rounded image support
      (``image_radius = r_max + 1/2``) and masks each row's pixels by ``|A^-1 k| <= r_max``
      (``reference_radius``); see docs/math/sparse_projection_radius.md.

    The reference padding is not implied by the image shape and the radius, and is given
    explicitly.
    """

    image_radius: float
    upsampling: int
    reference_radius: float | None = None


# Relative singular-value spread below which a magnification matrix is a scaled rotation.
_ISOTROPIC_MAGNIFICATION_TOLERANCE = 1e-9


def magnification_is_anisotropic(mag_matrices) -> bool:
    """Whether any optics group's 2x2 ``rlnMagMat`` stretches some directions more than others.

    Decided once from the optics table, never from a pass's rotations: a scaled rotation
    (equal singular values, identity included) keeps the image-radius clip exact.
    """

    import numpy as np

    for matrix in mag_matrices:
        singular = np.linalg.svd(np.asarray(matrix, dtype=np.float64).reshape(2, 2), compute_uv=False)
        if singular[0] - singular[1] > _ISOTROPIC_MAGNIFICATION_TOLERANCE * singular[0]:
            return True
    return False


def mstep_adjoint_max_r(volume_current_size, image_radius, padding_factor, *, anisotropic_magnification=False):
    """The adjoint ``max_r``: r_max on one grid, a :class:`ReferenceSphereClip` otherwise.

    ``anisotropic_magnification`` (:func:`magnification_is_anisotropic` of the images' optics
    groups) selects the rotated-radius mask; it is defined for images on the reference grid.
    """

    r_max = float(int(volume_current_size) // 2)
    if anisotropic_magnification:
        if image_radius is not None:
            raise NotImplementedError(
                "anisotropic magnification on another pixel size or box than the reference is not supported"
            )
        return ReferenceSphereClip(r_max + 0.5, int(padding_factor), r_max)
    if image_radius is None:
        return r_max
    return ReferenceSphereClip(float(image_radius), int(padding_factor))


def _recovar_clip_kwargs(max_r):
    if isinstance(max_r, ReferenceSphereClip):
        return {"max_r": max_r.image_radius, "upsampling": max_r.upsampling}
    return {"max_r": max_r}


def rotated_radius_mask(window_indices, rotations_block, image_shape, reference_radius):
    """``[rows, pixels]``: 1 where a row's pixel lies inside RELION's rotated sphere, else 0.

    ``window_indices`` are FFTW half-image pixels (rows ``0 .. N/2`` nonnegative ``y``, the rest
    ``y - N``; column ``x``), as the x-half adjoint reads them. The adjoint kernel places pixel
    ``(x, y)`` of row ``r`` at ``x B[r, 0] + y B[r, 1]`` for the row's matrix ``B``
    (recovar.cuda_backproject packs rows 1 and 0), which is RELION's ``A^-1 (x, y, 0)``.
    ``reference_radius`` may be traced.
    """

    height = int(image_shape[0])
    half_width = int(image_shape[1]) // 2 + 1
    indices = jnp.asarray(window_indices, dtype=jnp.int32)
    row = indices // half_width
    x = (indices % half_width).astype(rotations_block.dtype)
    y = jnp.where(row < half_width, row, row - height).astype(rotations_block.dtype)
    coords = x[None, :, None] * rotations_block[:, None, 0, :] + y[None, :, None] * rotations_block[:, None, 1, :]
    radius = jnp.asarray(reference_radius, dtype=rotations_block.dtype)
    return (jnp.sum(coords * coords, axis=-1) <= radius * radius).astype(rotations_block.dtype)


@partial(jax.jit, static_argnums=(4, 5, 6, 7, 8, 9, 10))
def adjoint_slice_volume_windowed(
    windowed_half,
    window_indices,
    rotations_block,
    volume,
    image_shape,
    volume_shape,
    disc_type,
    half_image,
    half_volume=False,
    max_r=None,
    relion_x_half=False,
    runtime_max_r=None,
):
    """Scatter a windowed half-spectrum into a full half-grid and adjoint-slice.

    ``runtime_max_r`` (a traced scalar, not a key) clips at RELION's radius while
    the static ``max_r`` sizes a larger capacity volume
    (recovar ``backproject_indexed``).
    """

    extra = {} if runtime_max_r is None else {"runtime_max_r": runtime_max_r}
    if isinstance(max_r, ReferenceSphereClip) and max_r.reference_radius is not None:
        if not relion_x_half:
            raise NotImplementedError("the rotated-radius mask reads the RELION x-half window layout")
        # A runtime radius is the logical image radius r_max + 1/2 of a larger physical class.
        reference_radius = max_r.reference_radius if runtime_max_r is None else runtime_max_r - 0.5
        mask = rotated_radius_mask(window_indices, rotations_block, image_shape, reference_radius)
        windowed_half = windowed_half * mask.astype(windowed_half.real.dtype)
    return core.adjoint_slice_volume_indexed(
        windowed_half,
        window_indices,
        rotations_block,
        image_shape,
        volume_shape,
        disc_type,
        volume=volume,
        half_image=half_image,
        half_volume=half_volume,
        relion_x_half=relion_x_half,
        **_recovar_clip_kwargs(max_r),
        **extra,
    )


@partial(jax.jit, static_argnums=(4, 5, 6, 7, 8, 9, 10))
def _batch_adjoint_slice_volume_windowed(
    windowed_halves,
    window_indices,
    rotations_block,
    volumes,
    image_shape,
    volume_shape,
    disc_type,
    half_image,
    half_volume=False,
    max_r=None,
    relion_x_half=False,
):
    """Batched indexed adjoint-slice for windowed half-spectrum blocks."""

    if isinstance(max_r, ReferenceSphereClip) and max_r.reference_radius is not None:
        raise NotImplementedError("the batched adjoint has no rotated-radius mask (anisotropic magnification)")
    return core.batch_adjoint_slice_volume_indexed(
        windowed_halves,
        window_indices,
        rotations_block,
        image_shape,
        volume_shape,
        disc_type,
        volumes=volumes,
        half_image=half_image,
        half_volume=half_volume,
        relion_x_half=relion_x_half,
        **_recovar_clip_kwargs(max_r),
    )


def batch_adjoint_slice_volume_maybe_windowed(
    half_blocks,
    window_indices,
    rotations_block,
    volumes,
    image_shape,
    volume_shape,
    disc_type,
    half_image,
    half_volume=False,
    *,
    use_window: bool,
    max_r=None,
    relion_x_half=False,
):
    """Batched adjoint-slice either full half-grids or indexed Fourier windows."""

    if use_window or relion_x_half:
        if window_indices is None:
            n_half = int(image_shape[0]) * (int(image_shape[1]) // 2 + 1)
            window_indices = jnp.arange(n_half, dtype=jnp.int32)
        return _batch_adjoint_slice_volume_windowed(
            half_blocks,
            window_indices,
            rotations_block,
            volumes,
            image_shape,
            volume_shape,
            disc_type,
            half_image,
            half_volume,
            max_r,
            relion_x_half,
        )
    return batch_adjoint_slice_volume_half(
        half_blocks,
        rotations_block,
        volumes,
        image_shape,
        volume_shape,
        disc_type,
        half_image,
        half_volume,
    )


@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7))
def adjoint_slice_volume_half(
    half_block,
    rotations_block,
    volume,
    image_shape,
    volume_shape,
    disc_type,
    half_image,
    half_volume=False,
):
    """Adjoint-slice a half-spectrum block into the volume accumulator."""

    return core.adjoint_slice_volume(
        half_block,
        rotations_block,
        image_shape,
        volume_shape,
        disc_type,
        volume=volume,
        half_image=half_image,
        half_volume=half_volume,
    )


@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7))
def batch_adjoint_slice_volume_half(
    half_blocks,
    rotations_block,
    volumes,
    image_shape,
    volume_shape,
    disc_type,
    half_image,
    half_volume=False,
):
    """Batched adjoint-slice half-spectrum blocks into volume accumulators."""

    return core.batch_adjoint_slice_volume(
        half_blocks,
        rotations_block,
        image_shape,
        volume_shape,
        disc_type,
        volumes=volumes,
        half_image=half_image,
        half_volume=half_volume,
    )
