"""Shared adjoint-slice wrappers for dense and local EM engines."""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from recovar import core


class ReferenceSphereClip(NamedTuple):
    """The M-step clip for images on another grid than the reference.

    RELION keeps a backprojected sample when its rotated, reference-grid radius is
    inside the model's r_max (acc/cuda/cuda_kernels/BP.cuh:322). recovar's kernel
    clips the image radius, which is the same test for unit rotations; an optics
    group on another pixel size or box backprojects with rotations scaled by 1/s,
    so its image radius is r_max * s. The reference padding is then no longer
    implied by the image shape and that radius, and is given explicitly.
    """

    image_radius: float
    upsampling: int


def mstep_adjoint_max_r(volume_current_size, image_radius, padding_factor):
    """The adjoint ``max_r``: r_max on one grid, a :class:`ReferenceSphereClip` on another."""

    if image_radius is None:
        return float(int(volume_current_size) // 2)
    return ReferenceSphereClip(float(image_radius), int(padding_factor))


def _recovar_clip_kwargs(max_r):
    if isinstance(max_r, ReferenceSphereClip):
        return {"max_r": max_r.image_radius, "upsampling": max_r.upsampling}
    return {"max_r": max_r}


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
):
    """Scatter a windowed half-spectrum into a full half-grid and adjoint-slice."""

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
    )


# Same arithmetic, explicit consumption of the accumulator for single-owner updates.
adjoint_slice_volume_windowed_donating = jax.jit(
    adjoint_slice_volume_windowed.__wrapped__,
    static_argnums=(4, 5, 6, 7, 8, 9, 10),
    donate_argnums=(3,),
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


def adjoint_slice_volume_maybe_windowed(
    half_block,
    window_indices,
    rotations_block,
    volume,
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
    """Adjoint-slice either a full half-grid or an indexed Fourier window."""

    if use_window or relion_x_half:
        if window_indices is None:
            n_half = int(image_shape[0]) * (int(image_shape[1]) // 2 + 1)
            window_indices = jnp.arange(n_half, dtype=jnp.int32)
        return adjoint_slice_volume_windowed(
            half_block,
            window_indices,
            rotations_block,
            volume,
            image_shape,
            volume_shape,
            disc_type,
            half_image,
            half_volume,
            max_r,
            relion_x_half,
        )
    return adjoint_slice_volume_half(
        half_block,
        rotations_block,
        volume,
        image_shape,
        volume_shape,
        disc_type,
        half_image,
        half_volume,
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
