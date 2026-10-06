"""Projection and noise primitives shared by dense/global and local EM paths."""

from __future__ import annotations

import logging
import math
import os
from functools import partial
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from recovar import core
from recovar.cuda_backproject import cuda_available as _cuda_projection_available

from relax.helpers.env_flags import parse_env_strict_flag
from relax.helpers.half_spectrum import bin_shell_values_jax
from relax.helpers.optics_noise import image_rotation_rows

logger = logging.getLogger(__name__)

DEFAULT_PROJECTION_MAX_R = object()
_RELION_PROJECTOR_TEXTURE_ENV = "RELAX_RELION_PROJECTOR_TEXTURE_INTERP"


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def project_relion_projector_half_spectrum(
    volume_relion_half,
    rotations_block,
    image_shape,
    r_max: int,
    padding_factor: int = 1,
    relion_acc_double_floorf_quirk: bool = False,
):
    """Forward-project RELION Projector storage into full half-image layout.

    ``volume_relion_half`` is RELION's ``Projector::data`` array, not
    recovar's centered full Fourier volume. This path is used by InitialModel
    parity code where RELION's pass-1/pass-2 scores must consume the exact
    ``PPref`` representation.

    ``relion_acc_double_floorf_quirk`` reproduces RELION's GPU-accelerated
    projector's float32-narrowing floor (see ``recovar.core.relion_project``
    module docstring); it is off by default.
    """

    from relax.relion.relion_project import relion_project_half

    image_size = int(image_shape[0])

    def project_one(R):
        return relion_project_half(
            volume_relion_half,
            R,
            image_size,
            int(r_max),
            int(padding_factor),
            relion_acc_double_floorf_quirk,
        )

    proj_fftw = jax.vmap(project_one)(rotations_block)

    return proj_fftw.reshape((rotations_block.shape[0], -1))


def _relion_projector_fftw_block(
    volume_relion_half,
    rotations_block,
    image_size: int,
    r_max: int,
    padding_factor: int,
    projector_output_size: int | None,
    relion_acc_double_floorf_quirk: bool,
):
    """Project one rotation block through RELION's Projector in FFTW row order.

    RELION projects onto a ``2 * r_max`` (or the requested) square that never
    exceeds the image; the scorer's rotation matrices are transposed at this
    handoff. Returns the ``(rotations, rows, x_half)`` FFTW-ordered block and
    the projector image size the centered-row and indexed projectors map from.
    """

    projector_image_size = int(r_max) * 2 if projector_output_size is None else int(projector_output_size)
    if projector_image_size <= 0 or projector_image_size > image_size:
        projector_image_size = image_size
    projector_rotations = jnp.swapaxes(rotations_block, -1, -2)
    proj_fftw = project_relion_projector_half_spectrum(
        volume_relion_half,
        projector_rotations,
        (projector_image_size, projector_image_size),
        int(r_max),
        int(padding_factor),
        relion_acc_double_floorf_quirk,
    ).reshape((rotations_block.shape[0], projector_image_size, projector_image_size // 2 + 1))
    return proj_fftw, projector_image_size


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
def project_relion_projector_half_spectrum_centered_rows(
    volume_relion_half,
    rotations_block,
    image_shape,
    r_max: int,
    padding_factor: int = 1,
    projector_output_size: int | None = None,
    relion_acc_double_floorf_quirk: bool = False,
) -> jnp.ndarray:
    """Project RELION ``PPref`` data and return recovar-centered row order.

    ``relion_project_half`` is a raw port of RELION's Projector and consumes
    RELION's FFTW-row projector matrix directly.  The dense E-step scorer
    supplies the same rotation matrices used by RECOVAR's centered half-spectrum
    scoring path, whose image-plane convention is the transpose at this
    handoff.  Apply that conversion here, then shift rows into RECOVAR's
    centered order.
    """

    image_size = int(image_shape[0])
    proj_fftw, projector_image_size = _relion_projector_fftw_block(
        volume_relion_half,
        rotations_block,
        image_size,
        r_max,
        padding_factor,
        projector_output_size,
        relion_acc_double_floorf_quirk,
    )
    if projector_image_size == image_size:
        row_order = jnp.fft.fftshift(jnp.arange(image_size, dtype=jnp.int32))
        return proj_fftw[:, row_order, :].reshape((rotations_block.shape[0], -1))

    # Placing the crop into the full centred half image as
    # ``zeros.at[:, full_indices].set(crop)`` is a scatter along the pixel axis,
    # and XLA lowers that scatter as a while loop with one trip per scattered
    # column: per trip a ``[rotations, 1]`` dynamic-slice and a
    # ``[rotations, full_pixels]`` dynamic-update-slice, plus the induction and
    # predicate kernels.  At the hp3 resident cache that is 4324 trips for every
    # rotation block and about 1.6 M launches per half (P4-C,
    # em_p4c_hp3_kernel_census_20260920), for work that moves each value once.
    #
    # The same placement read backwards is a gather: every full pixel takes at
    # most one crop element, because ``full_indices`` is injective, so for each
    # destination we can compute its source directly and fill the rest with
    # zeros.  Validity is decided by the round trip rather than by re-deriving
    # the row bounds, so the two paths agree by construction and the values
    # moved are identical, not merely equal.
    crop_x_half = projector_image_size // 2 + 1
    full_x_half = image_size // 2 + 1
    full_pixels = jnp.arange(image_size * full_x_half, dtype=jnp.int32)
    full_rows_of = full_pixels // full_x_half
    full_cols_of = full_pixels - full_rows_of * full_x_half
    crop_ky_of = full_rows_of - image_size // 2
    crop_rows_of = jnp.where(crop_ky_of >= 0, crop_ky_of, crop_ky_of + projector_image_size)
    # Clip so the gather stays in bounds; the round-trip check below discards
    # whatever the clip invented.
    safe_rows = jnp.clip(crop_rows_of, 0, projector_image_size - 1)
    safe_cols = jnp.clip(full_cols_of, 0, crop_x_half - 1)
    source = safe_rows * crop_x_half + safe_cols
    round_trip_ky = jnp.where(
        safe_rows <= projector_image_size // 2,
        safe_rows,
        safe_rows - projector_image_size,
    )
    covered = (round_trip_ky + image_size // 2) * full_x_half + safe_cols == full_pixels
    gathered = proj_fftw.reshape((rotations_block.shape[0], -1))[:, source]
    return jnp.where(covered[None, :], gathered, jnp.zeros((), dtype=proj_fftw.dtype))


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 7))
def project_relion_projector_half_spectrum_centered_rows_at_indices(
    volume_relion_half,
    rotations_block,
    image_shape,
    r_max: int,
    padding_factor: int = 1,
    projector_output_size: int | None = None,
    pixel_indices=None,
    relion_acc_double_floorf_quirk: bool = False,
) -> jnp.ndarray:
    """Project RELION ``PPref`` data and gather centered-row half-image pixels.

    This is equivalent to ``project_relion_projector_half_spectrum_centered_rows(
    ...)[..., pixel_indices]`` but avoids building the full centered half-image
    when RELION's current image size only needs a cropped Fourier window.
    """

    image_size = int(image_shape[0])
    proj_fftw, projector_image_size = _relion_projector_fftw_block(
        volume_relion_half,
        rotations_block,
        image_size,
        r_max,
        padding_factor,
        projector_output_size,
        relion_acc_double_floorf_quirk,
    )

    indices = jnp.asarray(pixel_indices, dtype=jnp.int32)
    full_x_half = image_size // 2 + 1
    full_rows = indices // full_x_half
    cols = indices - full_rows * full_x_half
    if projector_image_size == image_size:
        row_order = jnp.fft.fftshift(jnp.arange(image_size, dtype=jnp.int32))
        projector_rows = row_order[full_rows]
    else:
        ky = full_rows - image_size // 2
        projector_rows = jnp.where(ky >= 0, ky, ky + projector_image_size)
    projector_x_half = projector_image_size // 2 + 1
    projector_flat_indices = projector_rows * projector_x_half + cols
    return proj_fftw.reshape((rotations_block.shape[0], -1))[:, projector_flat_indices]


def centered_relion_projector_crop_mask(pixel_indices, *, image_shape, projector_output_size: int) -> np.ndarray:
    """Which centered full-image half-spectrum pixels lie in the RELION projector crop.

    The crop's row zero is the even box's Nyquist row (+N/2), then rows
    -N/2+1 .. N/2-1 (:func:`_texture_centered_crop_to_full`); a pixel outside
    the crop is zero in the full row. See ``docs/math/em_projector_indices.md``
    for full-box Nyquist labeling.
    """

    indices = np.asarray(pixel_indices, dtype=np.int64)
    image_size = int(image_shape[0])
    full_x_half = image_size // 2 + 1
    rows = indices // full_x_half
    cols = indices - rows * full_x_half
    ky = rows - image_size // 2
    projector_size = int(projector_output_size)
    projector_x_half = projector_size // 2 + 1
    min_ky = -(projector_size // 2 - 1)
    max_ky = projector_size // 2
    if projector_size == image_size:
        # The full even box stores its positive Nyquist row at centered row zero,
        # matching the texture gather. Smaller crops exclude that physical row.
        ky = np.where(rows == 0, max_ky, ky)
    return (
        (indices >= 0)
        & (indices < image_size * full_x_half)
        & (ky >= min_ky)
        & (ky <= max_ky)
        & (cols >= 0)
        & (cols < projector_x_half)
    )


def _host_pixel_indices(pixel_indices) -> bool:
    """Whether per-call pixel indices are host values, which are validated on every call.

    Device indices come from a producer that builds them valid, the resident
    passes' :func:`~relax.sparse_pass2.sparse_pass2_projection_blocks.projection_window_union`
    (it keeps only in-crop pixels); reading them back to validate synchronized
    every projector call, about 70 s per half of the EMPIAR-10202 pass-1 probe
    (py-spy, bigbox 14747301). Traced indices cannot be read.
    """

    return not isinstance(pixel_indices, (jax.Array, jax.core.Tracer))


def _validate_centered_relion_projector_pixel_indices(
    pixel_indices,
    *,
    image_shape,
    projector_output_size: int,
) -> None:
    """Validate compact indices against the RELION projector crop (:func:`centered_relion_projector_crop_mask`)."""

    indices = np.asarray(pixel_indices, dtype=np.int64)
    if indices.size == 0:
        return
    valid = centered_relion_projector_crop_mask(
        indices, image_shape=image_shape, projector_output_size=projector_output_size
    )
    if not np.all(valid):
        bad = indices[~valid][:8].tolist()
        raise ValueError(
            "centered RELION projector compact indices exceed projector crop "
            f"(image_shape={tuple(image_shape)}, projector_output_size={int(projector_output_size)}, "
            f"bad_indices={bad})"
        )


RELION_KERNELS = ("coarse", "fine")


def relion_kernel_zero_rows(image_size: int, projector_output_size: int, r_max, relion_kernel: str = "fine"):
    """Centred half-image pixels whose reference RELION's accelerated kernels set to zero, or ``None``.

    ``AccProjectorKernel::makeKernel`` clamps ``maxR = min(PPref.r_max, imgX - 1)``
    (acc/acc_projectorkernel_impl.h:301-310), and the kernels relabel the FFTW row
    index ``i`` of a pixel before projecting it:

    - the coarse diff2 kernel wraps at ``maxR``: ``y = i - imgY`` for ``i > maxR``
      (acc/cuda/cuda_kernels/diff2.cuh:86-90);
    - the fine diff2 and weighted-sum kernels keep ``y = i - imgY`` only for
      ``i >= imgY - maxR`` and move every other row beyond ``maxR`` to
      ``x = maxR`` (diff2.cuh:688-694, wavg.cuh:81-86).

    This changes nothing for an image window within the model sphere
    (``imgY / 2 <= r_max``). For an image on a coarser grid than the reference
    (scale ``1 < s < sqrt(2)``, :mod:`relax.helpers.optics_scale`) the window is
    wider, and each relabelled pixel lands outside the rotated sphere. RELION
    projects zeros there: rows ``maxR < label <= imgY/2`` for the coarse kernel,
    rows ``|label| > maxR`` for the fine kernels. The coarse kernel projects the
    Nyquist row at ``-imgY/2``; its ``x = 0`` pixel stays inside the sphere only if
    ``imgY/2`` equals ``s * maxR`` to within the integer radius test, which
    this mask does not reproduce. For ``s >= sqrt(2)`` the moved fine pixel can
    fall inside the sphere; :func:`relax.refinement.optics_shapes.make_shape_classes`
    refuses those grids. For unscaled rotations every one of these rows is already
    outside the sphere, so either rule changes nothing there.

    ``r_max`` is the model radius, or a traced runtime radius (projector-capacity route).
    Returns a boolean mask over the ``image_size x (image_size // 2 + 1)`` centred grid
    (true where RELION's reference is zero), or ``None`` when a static radius leaves no
    row affected.
    """

    if relion_kernel not in RELION_KERNELS:
        raise ValueError(f"relion_kernel must be one of {RELION_KERNELS}, got {relion_kernel!r}")
    size = int(projector_output_size)
    if isinstance(r_max, (int, np.integer)):
        if int(r_max) <= 0 or size // 2 <= int(r_max):
            return None
        max_r = int(r_max)
    else:
        max_r = jnp.minimum(jnp.asarray(r_max, jnp.int32), size // 2)
    n = int(image_size)
    label = np.arange(n) - n // 2
    if size == n:
        # The full even box keeps the positive Nyquist row at centred row zero.
        label[0] = n // 2
    label = jnp.asarray(label, jnp.int32)
    zero = label > max_r if relion_kernel == "coarse" else jnp.abs(label) > max_r
    return jnp.broadcast_to(zero[:, None], (n, n // 2 + 1)).reshape(-1)


class RelionCoarseRelabel(NamedTuple):
    """The pixels RELION's coarse diff2 kernel projects and shifts at a relabelled row.

    ``positions`` index the caller's pixel axis (entries of its ``pixel_indices``, or the
    centred ``image_size x (image_size // 2 + 1)`` grid). ``grid_indices`` are the same
    pixels' centred indices in a ``grid_size`` square grid whose rows hold the relabelled
    labels, so the unchanged projector and translate kernels evaluate them there.
    ``row_shift`` is the relabelled minus the caller's row label, ``-window`` for every pixel.
    """

    positions: np.ndarray
    grid_indices: np.ndarray
    grid_size: int
    row_shift: int


def relion_coarse_relabel(image_size: int, window: int, r_max: int, pixel_indices=None):
    """RELION's coarse diff2 row relabelling for a window wider than the model sphere, or ``None``.

    The coarse kernel reads FFTW row ``i`` of a ``window``-sized image as ``y = i`` for
    ``i <= maxR`` and ``y = i - window`` beyond, for the projection and the image shift alike
    (acc/cuda/cuda_kernels/diff2.cuh:86-90, 163-164), with ``maxR = min(PPref.r_max, window / 2)``
    (acc/acc_projectorkernel_impl.h:301-310); relax's fused coarse kernel does the same
    (relion_coarse_diff2_projector_body.inc). In relax's centred layout that moves the
    positive rows ``maxR < label <= window / 2`` to ``label - window``: the window's Nyquist
    row ``+window / 2`` (FFTW row ``window / 2``) becomes ``-window / 2``; for a window at the
    full box that row is centred row 0. Returns ``None`` when ``window / 2 <= maxR`` (no row moves).
    """

    n = int(image_size)
    window = int(window)
    max_r = min(int(r_max), window // 2)
    if window // 2 <= max_r:
        return None
    half_width = n // 2 + 1
    labels = np.arange(n) - n // 2
    if window == n:
        labels[0] = n // 2
    moved = (labels > max_r) & (labels <= window // 2)
    if pixel_indices is None:
        rows, cols = np.nonzero(moved[:, None] & (np.arange(half_width)[None, :] <= window // 2))
        positions = rows * half_width + cols
    else:
        indices = np.asarray(pixel_indices, dtype=np.int64).reshape(-1)
        positions = np.nonzero(moved[indices // half_width])[0]
        rows, cols = indices[positions] // half_width, indices[positions] % half_width
    new_labels = labels[rows] - window
    grid_size = 2 * int(np.max(np.abs(new_labels), initial=0)) + 2
    grid_indices = (new_labels + grid_size // 2) * (grid_size // 2 + 1) + cols
    return RelionCoarseRelabel(
        positions=positions.astype(np.int32),
        grid_indices=grid_indices.astype(np.int32),
        grid_size=int(grid_size),
        row_shift=-window,
    )


def relion_projector_half_to_texture_full(volume_relion_half: jax.Array) -> jax.Array:
    """Embed RELION ``Projector::data[z,y,x>=0]`` for CUDA texture staging.

    The CUDA texture projector only stages the non-negative model-x half from
    the centered full volume.  Consequently the negative-x half can remain
    zero: RELION handles negative projected x by flipping all coordinates and
    conjugating the sampled positive-x value.
    """

    volume_relion_half = jnp.asarray(volume_relion_half)
    pad_z, pad_y, half_x = volume_relion_half.shape
    if pad_z != pad_y or pad_z % 2 != 1 or half_x != pad_z // 2 + 1:
        raise ValueError(
            "RELION texture projection expects odd Projector::data shape "
            f"(pad, pad, pad//2+1), got {volume_relion_half.shape}",
        )
    center = pad_z // 2
    full = jnp.zeros((pad_z, pad_z, pad_z), dtype=volume_relion_half.dtype)
    return full.at[center:, :, :].set(jnp.transpose(volume_relion_half, (2, 1, 0)))


_TEXTURE_FALLBACK_REPORTED: set[str] = set()


def _relion_projector_texture_enabled(
    volume_relion_half,
    *,
    r_max: int,
    padding_factor: int,
    enabled: bool | None = None,
) -> bool:
    if enabled is None:
        if not parse_env_strict_flag(_RELION_PROJECTOR_TEXTURE_ENV, default=True):
            return False
    elif not bool(enabled):
        return False
    shape = tuple(int(value) for value in volume_relion_half.shape)
    expected_pad = 2 * (int(float(padding_factor) * float(r_max) + 0.5) + 1) + 1
    cuda_ok = _cuda_projection_available()
    dtype_ok = jnp.dtype(volume_relion_half.dtype) == jnp.dtype(jnp.complex64)
    shape_ok = len(shape) == 3 and shape == (expected_pad, expected_pad, expected_pad // 2 + 1)
    enabled_now = cuda_ok and dtype_ok and shape_ok
    if not enabled_now:
        # Falling back here replaces one texture-projector call with a vmapped
        # JAX projection whose per-row dispatch dominates pass-2 host time, so
        # report the reason once per distinct cause instead of failing silent.
        reason = (
            f"cuda={cuda_ok} dtype={volume_relion_half.dtype} (want complex64) "
            f"shape={shape} (want {(expected_pad, expected_pad, expected_pad // 2 + 1)}) "
            f"r_max={int(r_max)} padding_factor={int(padding_factor)}"
        )
        if reason not in _TEXTURE_FALLBACK_REPORTED:
            _TEXTURE_FALLBACK_REPORTED.add(reason)
            logger.warning(
                "RELION texture projector unavailable; using the vmapped JAX "
                "projection fallback: %s",
                reason,
            )
    return enabled_now


def _host_relion_projector_texture_enabled(
    projector_half, *, r_max, padding_factor, allow_float32_cast=False, enabled=None,
):
    """Check upload geometry using the fine consumer's effective dtype.

    Planning inspects metadata only. The fine dispatcher performs the existing
    c128-to-c64 production cast on host; coarse callers retain their source.
    """
    if not (
        isinstance(projector_half, np.ndarray)
        and projector_half.ndim == 3 and projector_half.flags.c_contiguous
        and r_max is not None
    ):
        return False
    dtype = projector_half.dtype
    if allow_float32_cast and dtype == np.dtype(np.complex128):
        dtype = np.dtype(np.complex64)
    return bool(dtype == np.dtype(np.complex64) and _relion_projector_texture_enabled(
        SimpleNamespace(shape=projector_half.shape, dtype=dtype),
        r_max=int(r_max), padding_factor=int(padding_factor), enabled=enabled,
    ))


def _texture_centered_crop_to_full(
    projection_crop,
    *,
    image_shape,
    projector_output_size: int,
    mask_current_image_disk: bool = False,
    current_image_mask_size=None,
):
    """Scatter a centered even-size CUDA projection into the full image box."""

    return _texture_centered_crop_to_full_jit(
        projection_crop,
        None if current_image_mask_size is None else jnp.asarray(current_image_mask_size, dtype=jnp.int32),
        image_size=int(image_shape[0]),
        projector_output_size=int(projector_output_size),
        mask_current_image_disk=bool(mask_current_image_disk),
    )


@partial(
    jax.jit,
    static_argnames=("image_size", "projector_output_size", "mask_current_image_disk"),
)
def _texture_centered_crop_to_full_jit(
    projection_crop,
    current_image_mask_size,
    *,
    image_size: int,
    projector_output_size: int,
    mask_current_image_disk: bool,
):
    image_size = int(image_size)
    crop_size = int(projector_output_size)
    crop = projection_crop.reshape((projection_crop.shape[0], crop_size, crop_size // 2 + 1))
    crop_rows = jnp.arange(crop_size, dtype=jnp.int32)
    crop_ky = jnp.where(crop_rows == 0, crop_size // 2, crop_rows - crop_size // 2)
    crop_cols = jnp.arange(crop_size // 2 + 1, dtype=jnp.int32)
    output_radius = (
        jnp.asarray(crop_size, dtype=jnp.int32)
        if current_image_mask_size is None
        else jnp.asarray(current_image_mask_size, dtype=jnp.int32)
    ) // jnp.int32(2)
    output_disk = crop_ky[:, None] ** 2 + crop_cols[None, :] ** 2 <= output_radius**2
    # RELION clips projections to min(PPref.mdlMaxR, image_half_width-1).
    # The texture kernel already enforces the PPref/model sphere; apply the
    # independent current-image disk here before embedding the crop.
    if mask_current_image_disk:
        crop = jnp.where(output_disk[None, :, :], crop, jnp.zeros((), dtype=crop.dtype))
    if crop_size == image_size:
        return crop.reshape((projection_crop.shape[0], -1))
    # Row zero is the even-box Nyquist row (+N/2 == -N/2); remaining rows
    # proceed from -N/2+1 through +N/2-1 in centered order.
    full_rows = crop_ky + image_size // 2
    full_indices = (full_rows[:, None] * (image_size // 2 + 1) + crop_cols[None, :]).reshape(-1)
    full = jnp.zeros(
        (projection_crop.shape[0], image_size * (image_size // 2 + 1)),
        dtype=projection_crop.dtype,
    )
    return full.at[:, full_indices].set(crop.reshape((projection_crop.shape[0], -1)))


def _centered_crop_indices(pixel_indices, *, image_size: int, crop_size: int):
    """Flat crop-grid indices of centred full-image half pixels, with their signed row ``ky`` and column."""

    full_x_half = image_size // 2 + 1
    crop_x_half = crop_size // 2 + 1
    indices = jnp.asarray(pixel_indices, dtype=jnp.int32)
    full_rows = indices // full_x_half
    cols = indices - full_rows * full_x_half

    if crop_size == image_size:
        crop_rows = full_rows
        ky = jnp.where(
            full_rows == 0,
            crop_size // 2,
            full_rows - crop_size // 2,
        )
    else:
        ky = full_rows - image_size // 2
        crop_rows = jnp.where(
            ky == crop_size // 2,
            0,
            ky + crop_size // 2,
        )
    return crop_rows * crop_x_half + cols, ky, cols


@partial(jax.jit, static_argnames=("image_shape", "projector_output_size", "mask_current_image_disk"))
def _texture_centered_crop_at_indices(
    projection_crop,
    pixel_indices,
    *,
    image_shape,
    projector_output_size: int,
    mask_current_image_disk: bool = False,
    current_image_mask_size=None,
):
    """Gather centered full-image pixels directly from a CUDA projection crop.

    One program per shape: called eagerly, its index arithmetic, gather and mask
    compiled about eight primitive programs for every new window size.
    """

    crop_size = int(projector_output_size)
    crop_indices, ky, cols = _centered_crop_indices(pixel_indices, image_size=int(image_shape[0]), crop_size=crop_size)
    selected = projection_crop.reshape((projection_crop.shape[0], -1))[:, crop_indices]
    if not mask_current_image_disk:
        return selected
    output_radius = (
        jnp.asarray(crop_size, dtype=jnp.int32)
        if current_image_mask_size is None
        else jnp.asarray(current_image_mask_size, dtype=jnp.int32)
    ) // jnp.int32(2)
    output_disk = ky * ky + cols * cols <= output_radius**2
    return jnp.where(output_disk[None, :], selected, jnp.zeros((), dtype=selected.dtype))


_HALF_STORAGE_MAX_ROTATIONS = 65535


def _half_storage_projection(volume_relion_half, *, r_max, padding_factor, projector_output_size) -> bool:
    """Whether a fixed-radius texture projection of this slab takes the half-storage kernel."""

    return (
        int(padding_factor) in (1, 2)
        and int(r_max) > 0
        and 0 < int(projector_output_size) <= 4096
        and int(projector_output_size) % 2 == 0
        and volume_relion_half.dtype == jnp.complex64
        and tuple(volume_relion_half.shape) == (
            2 * int(r_max) * int(padding_factor) + 3,
            2 * int(r_max) * int(padding_factor) + 3,
            int(r_max) * int(padding_factor) + 2,
        )
        # The staging fill kernel indexes texels with int.
        and int(np.prod(volume_relion_half.shape)) <= np.iinfo(np.int32).max
    )


def relion_capacity_texture_serves(
    volume_relion_half, *, r_max, padding_factor, projector_output_size, relion_texture_interp=None
) -> bool:
    """Whether :class:`~relax.cuda.kernels.RelionCapacityHalfTextureF32` can serve this slab's projections.

    It serves exactly the fixed-radius projections that
    :func:`compute_relion_projector_projections_block` sends to the
    half-storage kernel: the RELION texture projector is enabled and the slab
    has the half-storage geometry.
    """

    return _relion_projector_texture_enabled(
        volume_relion_half, r_max=int(r_max), padding_factor=int(padding_factor), enabled=relion_texture_interp
    ) and _half_storage_projection(
        volume_relion_half,
        r_max=r_max,
        padding_factor=padding_factor,
        projector_output_size=projector_output_size,
    )


def _project_relion_projector_texture(
    volume_relion_half,
    rotations_block,
    image_shape,
    *,
    r_max: int,
    projector_output_size: int,
    mask_current_image_disk: bool = False,
    current_image_mask_size=None,
    pixel_indices=None,
    runtime_r_max=None,
    padding_factor=1,
    image_r_max=None,
    persistent_texture=None,
    capacity_texture=None,
):
    """Project one RELION ``PPref`` block with RELION's CUDA texture arithmetic.

    Eligible F32 inputs use direct half storage; see
    ``docs/development/em_half_texture_staging.md`` for eligibility and gates.
    Radius ownership is specified in ``docs/math/sparse_projection_radius.md``.
    """

    # Native texture kernels own the rotated, integer-truncated radius test.
    # Capacity buffers may exceed the active image; pass its radius into the
    # kernel rather than applying an exact source-pixel disk after projection.
    if (
        not mask_current_image_disk and persistent_texture is None
        and image_r_max is None and current_image_mask_size is not None
    ):
        image_r_max = jnp.asarray(current_image_mask_size, dtype=jnp.int32) // jnp.int32(2)
    if persistent_texture is not None:
        if current_image_mask_size is not None:
            raise ValueError("persistent texture requires a fixed current-image radius")
        if runtime_r_max is not None or image_r_max is not None:
            raise ValueError("persistent texture requires a fixed model and image radius")
        from relax.cuda.kernels import relion_projector_persistent_half_texture_f32
        projection_crop = relion_projector_persistent_half_texture_f32(
            persistent_texture, jnp.asarray(rotations_block, dtype=jnp.float32),
            current_size=int(projector_output_size), padding_factor=int(padding_factor),
            projector_max_r=int(r_max),
        )
    elif image_r_max is not None:
        if mask_current_image_disk:
            raise ValueError("rotated image radius cannot be combined with an exact image-disk mask")
        from relax.cuda.kernels import project_relion_half_capacity

        projection_crop = project_relion_half_capacity(
            volume_relion_half, rotations_block,
            jnp.asarray(r_max, jnp.int32) if runtime_r_max is None else runtime_r_max,
            image_shape=(int(projector_output_size), int(projector_output_size)),
            padding_factor=int(padding_factor), image_r_max=image_r_max,
        )
    elif (
        runtime_r_max is None
        and _half_storage_projection(
            volume_relion_half,
            r_max=r_max,
            padding_factor=padding_factor,
            projector_output_size=projector_output_size,
        )
        and rotations_block.dtype == jnp.float32
        and 0 < rotations_block.shape[0]
    ):
        # The half-storage kernel stages the same texels without a cubic
        # transpose/zero-fill buffer. Output image extent is independent of
        # model radius; keep crop, mask, gather and scaling unchanged below.
        from relax.cuda.kernels import project_relion_half_capacity

        image_radius_kwargs = (
            {"image_r_max": jnp.asarray(projector_output_size // 2, jnp.int32)}
            if not mask_current_image_disk else {}
        )
        if capacity_texture is not None:
            # The same staging and kernel, with the texture staged once per pass.
            if (
                tuple(capacity_texture.shape) != tuple(volume_relion_half.shape)
                or capacity_texture.logical_r_max != int(r_max)
                or capacity_texture.padding_factor != int(padding_factor)
            ):
                raise ValueError("the capacity projector texture does not hold this projector")
            project = partial(
                capacity_texture.project,
                image_shape=(int(projector_output_size), int(projector_output_size)),
                **image_radius_kwargs,
            )
            capacity_texture = None
        else:

            def project(rotations):
                return project_relion_half_capacity(
                    volume_relion_half, rotations, jnp.asarray(r_max, jnp.int32),
                    image_shape=(int(projector_output_size), int(projector_output_size)),
                    padding_factor=int(padding_factor),
                    **image_radius_kwargs,
                )

        # The kernel takes at most _HALF_STORAGE_MAX_ROTATIONS rotations per launch.
        n_rotations = int(rotations_block.shape[0])
        if n_rotations <= _HALF_STORAGE_MAX_ROTATIONS:
            projection_crop = project(rotations_block)
        else:
            projection_crop = jnp.concatenate(
                [
                    project(rotations_block[start : start + _HALF_STORAGE_MAX_ROTATIONS])
                    for start in range(0, n_rotations, _HALF_STORAGE_MAX_ROTATIONS)
                ],
                axis=0,
            )
    elif (
        runtime_r_max is None and int(r_max) > 0 and int(padding_factor) > 0
        and volume_relion_half.dtype == jnp.complex64
        and rotations_block.dtype == jnp.float32
        and volume_relion_half.shape == (
            2 * int(r_max) * int(padding_factor) + 3,
            2 * int(r_max) * int(padding_factor) + 3,
            int(r_max) * int(padding_factor) + 2,
        )
    ):
        # Geometry the half-storage kernel does not take (an odd output size, a
        # padding factor other than 1 or 2); every slab size it takes goes there.
        from relax.cuda.kernels import relion_projector_half_texture_f32
        projection_crop = relion_projector_half_texture_f32(
            volume_relion_half, rotations_block,
            current_size=int(projector_output_size), padding_factor=int(padding_factor),
            projector_max_r=int(r_max),
        )
    elif runtime_r_max is None:
        projector_full = relion_projector_half_to_texture_full(volume_relion_half)
        pad_size = int(projector_full.shape[0])
        projection_crop = project_half_spectrum(
            projector_full.reshape(-1),
            rotations_block,
            (int(projector_output_size), int(projector_output_size)),
            (pad_size, pad_size, pad_size),
            "linear_interp",
            max_r=float(r_max),
            relion_texture_interp=True,
        )
    else:
        from relax.cuda.kernels import project_relion_half_capacity

        projection_crop = project_relion_half_capacity(
            volume_relion_half, rotations_block, runtime_r_max,
            image_shape=(int(projector_output_size), int(projector_output_size)),
            padding_factor=int(padding_factor),
            **({"image_r_max": jnp.asarray(projector_output_size // 2, jnp.int32)}
               if not mask_current_image_disk else {}),
        )
    if capacity_texture is not None:
        raise ValueError("a capacity projector texture serves only the fixed-radius half-storage projection")
    if pixel_indices is not None:
        return _texture_centered_crop_at_indices(
            projection_crop,
            pixel_indices,
            image_shape=tuple(int(v) for v in image_shape),
            projector_output_size=int(projector_output_size),
            mask_current_image_disk=bool(mask_current_image_disk),
            current_image_mask_size=current_image_mask_size,
        )
    return _texture_centered_crop_to_full(
        projection_crop,
        image_shape=image_shape,
        projector_output_size=int(projector_output_size),
        mask_current_image_disk=bool(mask_current_image_disk),
        current_image_mask_size=current_image_mask_size,
    )


def compute_relion_projector_projections_block(
    volume_relion_half,
    rotations_block,
    image_shape,
    *,
    r_max: int,
    padding_factor: int = 1,
    return_abs2: bool = True,
    centered_rows: bool = False,
    dense_scale: bool = False,
    projector_output_size: int | None = None,
    pixel_indices=None,
    relion_texture_interp: bool | None = None,
    relion_acc_double_floorf_quirk: bool = False,
    mask_current_image_disk: bool = False,
    current_image_mask_size=None,
    projector_capacity: bool = False,
    runtime_r_max=None,
    image_r_max=None,
    persistent_texture=None,
    capacity_texture=None,
    relion_kernel: str = "fine",
):
    """Project precomputed RELION ``PPref`` data for one rotation block.

    Strict parity defaults to RELION's CUDA texture interpolator when the
    custom CUDA projector is available.  Set
    ``RELAX_RELION_PROJECTOR_TEXTURE_INTERP=0`` to force the manual/JAX
    diagnostic fallback.

    ``relion_acc_double_floorf_quirk`` only applies to that manual/JAX
    fallback (the texture path is float32-only hardware interpolation, unrelated
    to this quirk); see ``recovar.core.relion_project`` module docstring.

    ``relion_kernel`` ("fine" or "coarse") names the RELION kernel whose row rule
    applies when the image window is wider than the model sphere, as for an
    image on a coarser grid than the reference (:func:`relion_kernel_zero_rows`).
    Callers emulating RELION's coarse diff2 kernel pass "coarse".
    """

    image_size = int(image_shape[0])
    resolved_output_size = int(r_max) * 2 if projector_output_size is None else int(projector_output_size)
    if resolved_output_size <= 0 or resolved_output_size > image_size:
        resolved_output_size = image_size
    zero_rows = relion_kernel_zero_rows(
        image_size, resolved_output_size, runtime_r_max if projector_capacity else int(r_max), relion_kernel
    )
    coarse_relabel = None
    coarse_window = None
    if relion_kernel == "coarse" and zero_rows is not None and not isinstance(rotations_block, jax.core.Tracer):
        from relax.helpers.optics_scale import coarse_rows_wrap_inside

        # RELION's window is the active current size, which a stable-shape caller passes as
        # its runtime mask size.
        coarse_window = resolved_output_size
        if current_image_mask_size is not None and not isinstance(current_image_mask_size, jax.core.Tracer):
            coarse_window = int(current_image_mask_size)
        # Other-grid rotations carry 1 / s (applyScaleDifference). Rows can wrap inside the
        # sphere only for a window wider than 2 r_max, so only then is the scale read (a
        # device rotation block costs a host sync per block).
        if coarse_window // 2 > int(r_max) and coarse_rows_wrap_inside(
            coarse_window,
            int(r_max),
            1.0 / float(np.linalg.norm(np.asarray(rotations_block[0, 0], dtype=np.float64))),
        ):
            unsupported = (
                not centered_rows or projector_capacity or runtime_r_max is not None or mask_current_image_disk
                or persistent_texture is not None or capacity_texture is not None
                or isinstance(current_image_mask_size, jax.core.Tracer)
            )
            if unsupported:
                raise NotImplementedError(
                    f"a {coarse_window} px coarse window lies between 2 r_max and 2 s r_max "
                    f"(r_max {int(r_max)}): RELION's coarse kernel projects its relabelled outer rows there "
                    "(diff2.cuh:86-90); this projection route (a runtime radius, an image-disk mask or "
                    "uncentred rows) does not reproduce them"
                )
            coarse_relabel = relion_coarse_relabel(image_size, coarse_window, int(r_max), pixel_indices)
    if persistent_texture is not None:
        if projector_capacity or runtime_r_max is not None or image_r_max is not None:
            raise ValueError("persistent texture cannot use the runtime capacity/radius route")
        if relion_texture_interp is False:
            raise ValueError("persistent texture cannot use manual projection")
        use_texture = True
    elif projector_capacity:
        if (
            int(r_max) != 0 or runtime_r_max is None or not centered_rows
            or pixel_indices is None or projector_output_size is None
            or not 0 < int(projector_output_size) <= image_size
            or int(projector_output_size) % 2
        ):
            raise ValueError(
                "projector capacity requires compact centered pixels, valid explicit "
                "output size, runtime radius and static radius sentinel 0"
            )
        if relion_texture_interp is False:
            raise ValueError("projector capacity cannot use the manual projection fallback")
        use_texture = True
    else:
        use_texture = _relion_projector_texture_enabled(
            volume_relion_half,
            r_max=int(r_max),
            padding_factor=int(padding_factor),
            enabled=relion_texture_interp,
        )

    if use_texture:
        if not centered_rows and pixel_indices is not None:
            raise ValueError("pixel_indices are only supported with centered_rows=True")
        if pixel_indices is not None and _host_pixel_indices(pixel_indices):
            _validate_centered_relion_projector_pixel_indices(
                pixel_indices,
                image_shape=image_shape,
                projector_output_size=resolved_output_size,
            )
        texture_kwargs = {
            "r_max": int(r_max),
            "projector_output_size": resolved_output_size,
        }
        if persistent_texture is not None:
            texture_kwargs["persistent_texture"] = persistent_texture
        if capacity_texture is not None:
            texture_kwargs["capacity_texture"] = capacity_texture
        if int(padding_factor) != 1:
            texture_kwargs["padding_factor"] = int(padding_factor)
        if image_r_max is not None:
            texture_kwargs["image_r_max"] = image_r_max
            texture_kwargs["padding_factor"] = int(padding_factor)
        if centered_rows and pixel_indices is not None:
            texture_kwargs["pixel_indices"] = pixel_indices
        if mask_current_image_disk:
            texture_kwargs["mask_current_image_disk"] = True
        if current_image_mask_size is not None:
            texture_kwargs["current_image_mask_size"] = current_image_mask_size
        if projector_capacity:
            texture_kwargs["runtime_r_max"] = runtime_r_max
            texture_kwargs["padding_factor"] = int(padding_factor)
        proj_centered = _project_relion_projector_texture(
            volume_relion_half,
            rotations_block,
            image_shape,
            **texture_kwargs,
        )
        if centered_rows:
            proj_half = proj_centered
        else:
            proj_half = jnp.fft.ifftshift(
                proj_centered.reshape((proj_centered.shape[0], image_size, image_size // 2 + 1)),
                axes=1,
            ).reshape((proj_centered.shape[0], -1))

    elif capacity_texture is not None:
        raise RuntimeError("a capacity projector texture requires the RELION texture projector")
    elif current_image_mask_size is not None or image_r_max is not None:
        raise RuntimeError(
            "a runtime current-image projection mask requires the RELION "
            "texture projector",
        )
    elif pixel_indices is not None:
        if not centered_rows:
            raise ValueError("pixel_indices are only supported with centered_rows=True")
        if resolved_output_size < image_size and _host_pixel_indices(pixel_indices):
            _validate_centered_relion_projector_pixel_indices(
                pixel_indices,
                image_shape=image_shape,
                projector_output_size=resolved_output_size,
            )
        proj_half = project_relion_projector_half_spectrum_centered_rows_at_indices(
            volume_relion_half,
            rotations_block,
            image_shape,
            int(r_max),
            int(padding_factor),
            projector_output_size,
            pixel_indices,
            relion_acc_double_floorf_quirk,
        )
    elif centered_rows:
        proj_half = project_relion_projector_half_spectrum_centered_rows(
            volume_relion_half,
            rotations_block,
            image_shape,
            int(r_max),
            int(padding_factor),
            projector_output_size,
            relion_acc_double_floorf_quirk,
        )
    else:
        proj_half = project_relion_projector_half_spectrum(
            volume_relion_half,
            rotations_block,
            image_shape,
            int(r_max),
            int(padding_factor),
            relion_acc_double_floorf_quirk,
        )
    if zero_rows is not None:
        if not centered_rows:
            n = int(image_shape[0])
            zero_rows = jnp.fft.ifftshift(zero_rows.reshape(n, n // 2 + 1), axes=0).reshape(-1)
        if centered_rows and pixel_indices is not None:
            zero_rows = zero_rows[jnp.asarray(pixel_indices)]
        proj_half = jnp.where(zero_rows, jnp.zeros((), proj_half.dtype), proj_half)
    if coarse_relabel is not None and coarse_relabel.positions.size:
        # RELION projects those rows at their relabelled coordinates: the same projector on
        # a grid that holds them, with the kernel's maxR radius test.
        grid = int(coarse_relabel.grid_size)
        max_r = min(int(r_max), int(coarse_window) // 2)
        if use_texture:
            moved = _project_relion_projector_texture(
                volume_relion_half,
                rotations_block,
                (grid, grid),
                r_max=int(r_max),
                projector_output_size=grid,
                padding_factor=int(padding_factor),
                pixel_indices=coarse_relabel.grid_indices,
                image_r_max=jnp.asarray(max_r, dtype=jnp.int32),
            )
        else:
            moved = project_relion_projector_half_spectrum_centered_rows_at_indices(
                volume_relion_half,
                rotations_block,
                (grid, grid),
                max_r,
                int(padding_factor),
                grid,
                coarse_relabel.grid_indices,
                relion_acc_double_floorf_quirk,
            )
        proj_half = proj_half.at[:, jnp.asarray(coarse_relabel.positions)].set(moved.astype(proj_half.dtype))
    if dense_scale:
        proj_half = proj_half * _dense_means_scale(int(image_shape[0]))
    proj_abs2_half = jnp.abs(proj_half) ** 2 if return_abs2 else None
    return proj_half, proj_abs2_half


def _dense_means_scale(image_size: int):
    """The dense-means projection scale, ``-N^2`` unless ``RELAX_DENSE_MEANS_SCALE`` selects ``N2``."""

    token = (os.environ.get("RELAX_DENSE_MEANS_SCALE") or "-N2").strip()
    scale = {"-N2": -(image_size**2), "N2": float(image_size**2)}.get(token)
    if scale is None:
        raise ValueError(f"Unsupported RELAX_DENSE_MEANS_SCALE={token!r}")
    return scale


def relion_coarse_packed_rows_serve(image_size: int, projector_output_size: int, r_max: int) -> bool:
    """Whether :func:`project_relion_coarse_packed_rows` reproduces this coarse projection: no coarse-kernel
    row of the window is zeroed or relabelled (the window lies within the model sphere)."""

    return relion_kernel_zero_rows(int(image_size), int(projector_output_size), int(r_max), "coarse") is None


def project_relion_coarse_packed_rows(
    capacity_texture, rotations_block, pixel_indices, *, image_shape, projector_output_size: int
):
    """F32 ``[N, 2 P]``: the dense-scaled coarse projections ``p`` that
    :func:`compute_relion_projector_projections_block` returns (``centered_rows``, ``dense_scale``,
    ``pixel_indices``, ``relion_kernel="coarse"``, ``capacity_texture``) at ``pixel_indices`` ``[P]``, packed
    as ``[Re p | Im p]``, the coarse GEMM scorer's reference operand, in one kernel (no crop gather, scale or
    split). Only where :func:`relion_coarse_packed_rows_serve` holds.
    """

    crop_size = int(projector_output_size)
    crop_indices, _ky, _cols = _centered_crop_indices(pixel_indices, image_size=int(image_shape[0]), crop_size=crop_size)
    # One launch for any number of rotations (no per-launch chunks to concatenate).
    return capacity_texture.project_compact_packed(
        rotations_block,
        crop_indices,
        image_shape=(crop_size, crop_size),
        image_r_max=jnp.asarray(crop_size // 2, jnp.int32),
        scale=float(_dense_means_scale(int(image_shape[0]))),
    )


def project_half_spectrum(
    volume,
    rotations_block,
    image_shape,
    volume_shape,
    disc_type,
    *,
    half_volume: bool = False,
    max_r=DEFAULT_PROJECTION_MAX_R,
    relion_texture_interp: bool = True,
    force_jax: bool = False,
):
    """Forward-slice one rotation block into half-spectrum image layout."""
    if force_jax:
        order = core.decide_order(disc_type)
        if order > 1:
            raise ValueError("force_jax projection is only supported for nearest/linear interpolation")
        from recovar.core import relion_interp

        resolved_max_r = core._default_max_r(image_shape) if max_r is DEFAULT_PROJECTION_MAX_R else max_r
        return relion_interp.project(
            volume,
            rotations_block,
            image_shape,
            volume_shape,
            order=order,
            half_volume=half_volume,
            half_image=True,
            max_r=resolved_max_r,
        )

    kwargs = {
        "half_image": True,
        "relion_texture_interp": relion_texture_interp,
    }
    if half_volume:
        kwargs["half_volume"] = True
    if max_r is not DEFAULT_PROJECTION_MAX_R:
        kwargs["max_r"] = max_r
    return core.slice_volume(
        volume,
        rotations_block,
        image_shape,
        volume_shape,
        disc_type,
        **kwargs,
    )


def compute_projections_block(
    volume,
    rotations_block,
    image_shape,
    volume_shape,
    disc_type,
    *,
    max_r=DEFAULT_PROJECTION_MAX_R,
    return_abs2: bool = True,
    relion_texture_interp: bool = True,
    force_jax: bool = False,
):
    """Forward-slice one rotation block and optionally compute ``|proj|^2``.

    Dense scoring and noise accumulation need ``|proj|^2`` repeatedly enough to
    materialize it. Exact-local paths can pass ``return_abs2=False`` and compute
    norms on demand when that saves memory.

    This path slices RECOVAR's centered-grid volume. RELION projector options
    belong to ``compute_relion_projector_projections_block``.
    """
    proj_half = project_half_spectrum(
        volume,
        rotations_block,
        image_shape,
        volume_shape,
        disc_type,
        max_r=max_r,
        relion_texture_interp=relion_texture_interp,
        force_jax=force_jax,
    )
    proj_abs2_half = jnp.abs(proj_half) ** 2 if return_abs2 else None
    return proj_half, proj_abs2_half


@partial(jax.jit, static_argnums=(6, 7))
def compute_noise_block(
    proj_half,
    proj_abs2_half,
    summed_masked,
    ctf_probs,
    noise_variance_half,
    shell_indices,
    shell_count,
    return_split: bool = True,
):
    """Accumulate RELION-style posterior-weighted noise for one rotation block.

    Uses the decomposition::

        E_w[|CTF*proj - img|^2] = E_w[|CTF*proj|^2] - 2*Re(E_w[conj(img)*CTF*proj]) + |img|^2
                                 =     A2            -           2*XA                  + P_img

    ``P_img`` is handled by the caller (image-only, no rotation dependence).
    This function computes the ``A2 - 2*XA`` contribution from one rotation
    block, binned to resolution shells. Inputs are un-Hermitian-weighted packed
    half spectra because RELION's noise update bins over its FFTW half-plane
    convention directly.

    The returned per-shell blocks keep whatever real dtype ``a2``/``xa``
    naturally promote to from the inputs (float32 if all inputs are float32,
    float64 if any is float64) -- no explicit cast is applied. Callers
    accumulate many of these blocks (one per rotation block/microbatch,
    across every particle) into a single running sigma2_noise sum; an
    unconditional float32 cast here used to truncate every block before that
    accumulation, compounding error across the whole reduction, unlike
    RELION's own ``wsum_model.sigma2_noise``, which stays RFLOAT (double
    under double-precision builds) through the entire per-particle
    accumulation. Pass float64 inputs (matching ``use_float64_scoring``/
    ``use_float64_projections``) to get float64 accumulation here too.
    """
    ctf_has_mass = ctf_probs != 0.0
    ctf_probs_raw = jnp.where(ctf_has_mass, ctf_probs * noise_variance_half, 0.0)
    a2_terms = jnp.where(ctf_has_mass, proj_abs2_half * ctf_probs_raw, 0.0)
    a2 = jnp.sum(a2_terms, axis=0)

    cross_terms = jnp.where(summed_masked != 0.0, proj_half * jnp.conj(summed_masked), 0.0)
    cross = jnp.sum(cross_terms, axis=0)
    xa = jnp.where(cross.real != 0.0, noise_variance_half * cross.real, 0.0)
    block_noise = a2 - 2.0 * xa

    noise_shells = bin_shell_values_jax(block_noise, shell_indices, shell_count)
    if not return_split:
        zeros = jnp.zeros(shell_count, dtype=block_noise.dtype)
        return noise_shells, zeros, zeros
    a2_shells = bin_shell_values_jax(a2, shell_indices, shell_count)
    xa_shells = bin_shell_values_jax(xa, shell_indices, shell_count)
    return noise_shells, a2_shells, xa_shells


def compute_noise_block_per_optics_group(
    proj_half,
    proj_abs2_half,
    summed_masked,
    ctf_probs,
    noise_table,
    row_optics_groups,
    shell_indices,
    shell_count,
):
    """:func:`compute_noise_block` with one noise spectrum per optics group.

    Row ``r`` carries its image's group ``row_optics_groups[r]`` and is weighted by
    that group's row of ``noise_table`` ``[G, P]``. The ``A2 - 2*XA`` terms are summed
    per group, statement for statement as the one-group function sums them over all
    rows (``A2`` per row, then the group's noise times its summed cross term), and
    binned to ``[G, shell_count]``: RELION's ``wsum_model.sigma2_noise[optics_group]``.
    """
    n_groups = int(noise_table.shape[0])
    noise_rows = noise_table[row_optics_groups]
    ctf_has_mass = ctf_probs != 0.0
    ctf_probs_raw = jnp.where(ctf_has_mass, ctf_probs * noise_rows, 0.0)
    a2_terms = jnp.where(ctf_has_mass, proj_abs2_half * ctf_probs_raw, 0.0)
    # Each group's rows are summed by a masked row reduction, the one-group
    # function's jnp.sum(axis=0) per group. A segment_sum here is a scatter-add of
    # every row onto G rows: thousands of atomics per address, 47 ms per 8192-row
    # M-step block (87% of the multishape Class3D iteration-2 kernel time).
    member = row_optics_groups[:, None] == jnp.arange(n_groups, dtype=row_optics_groups.dtype)[None, :]
    a2 = jnp.sum(jnp.where(member[:, :, None], a2_terms[:, None, :], 0.0), axis=0)

    cross_terms = jnp.where(summed_masked != 0.0, proj_half * jnp.conj(summed_masked), 0.0)
    cross = jnp.sum(jnp.where(member[:, :, None], cross_terms[:, None, :], 0.0), axis=0)
    xa = jnp.where(cross.real != 0.0, noise_table * cross.real, 0.0)
    block_noise = a2 - 2.0 * xa
    return jax.vmap(lambda values: bin_shell_values_jax(values, shell_indices, shell_count))(block_noise)


@jax.jit
def compute_norm_residual_per_image(
    proj_half,
    proj_abs2_half,
    summed_masked,
    ctf_probs,
    noise_variance_half,
):
    """Return RELION norm-correction residual terms per image.

    This is the same ``A2 - 2*XA`` contribution as :func:`compute_noise_block`,
    but summed per image instead of binned over shells.  The caller adds the
    image-power term once per image.

    Like ``compute_noise_block``, keeps whatever real dtype the inputs
    naturally promote to -- no explicit float32 cast, which used to
    truncate this per-image residual before the caller's own accumulation
    even when the inputs were already float64 (double-precision scoring).
    """

    ctf_has_mass = ctf_probs != 0.0
    ctf_probs_raw = jnp.where(ctf_has_mass, ctf_probs * image_rotation_rows(noise_variance_half), 0.0)
    a2_terms = jnp.where(ctf_has_mass, proj_abs2_half * ctf_probs_raw, 0.0)
    a2_per_image = jnp.sum(a2_terms, axis=(1, 2))

    cross_terms = jnp.where(summed_masked != 0.0, proj_half * jnp.conj(summed_masked), 0.0)
    xa_terms = image_rotation_rows(noise_variance_half) * cross_terms.real
    xa_per_image = jnp.sum(xa_terms, axis=(1, 2))
    return a2_per_image - 2.0 * xa_per_image


def relion_scale_correction_pixel_mask(data_vs_prior, shell_indices, *, n_shells=None):
    """Return RELION's ``data_vs_prior > 3`` scale-statistic pixel mask."""

    indices = jnp.asarray(shell_indices)
    if n_shells is None:
        n_shells = int(np.asarray(data_vs_prior).size) if data_vs_prior is not None else int(np.max(indices))
    dvp = None if data_vs_prior is None else jnp.asarray(data_vs_prior)
    if dvp is not None and dvp.size == 0:
        return jnp.zeros((int(indices.size),), dtype=bool)
    return _scale_correction_pixel_mask(dvp, indices, n_shells=int(n_shells))


@partial(jax.jit, static_argnames=("n_shells",))
def _scale_correction_pixel_mask(data_vs_prior, shell_indices, *, n_shells):
    # One program per pixel count: eager, the mask was about ten single-primitive
    # programs compiled again at every new current size.
    indices = shell_indices.astype(jnp.int32).reshape(-1)
    valid_shell = (indices >= 0) & (indices < n_shells)
    if data_vs_prior is None:
        return valid_shell
    dvp = data_vs_prior.reshape(-1)
    safe_indices = jnp.clip(indices, 0, dvp.size - 1)
    return valid_shell & (indices < dvp.size) & (dvp[safe_indices] > 3.0)


def compact_relion_projector_half_for_centered_indices(
    projector_half,
    pixel_indices,
    image_shape,
    *,
    r_max: int,
    padding_factor: int,
):
    """Materialize the smallest host PPref slab covering score pixels.

    RELION's first-iteration normalized-CC pass scores a small square Fourier
    window even when ``Projector::data`` was built for the full image box.
    Copy only the centered y/z region and nonnegative-x prefix that can be
    sampled by those pixels, retaining the standard one-voxel interpolation
    halo.  The returned radius is strictly larger than every consumed image
    radius, so the compact texture's model-sphere cutoff remains inactive for
    the complete score window.

    This helper intentionally accepts a NumPy host array.  Compacting after an
    eager JAX transfer would leave the full projector resident on the device,
    defeating the memory bound this operation provides.
    """

    if not isinstance(projector_half, np.ndarray):
        raise TypeError("RELION projector compaction requires a NumPy host array")
    if projector_half.ndim != 3:
        raise ValueError(
            "RELION projector compaction expects one (z, y, x-half) slab, "
            f"got {projector_half.shape}",
        )
    r_max = int(r_max)
    padding_factor = int(padding_factor)
    if r_max <= 0 or padding_factor <= 0:
        raise ValueError(
            f"r_max and padding_factor must be positive, got {r_max} and {padding_factor}",
        )
    padded_r_max = r_max * padding_factor
    expected_size = 2 * (padded_r_max + 1) + 1
    expected_shape = (expected_size, expected_size, padded_r_max + 2)
    if projector_half.shape != expected_shape:
        raise ValueError(
            "RELION projector shape does not match r_max/padding_factor: "
            f"got {projector_half.shape}, expected {expected_shape}",
        )

    image_height, image_width = (int(value) for value in image_shape)
    if (
        image_height <= 0
        or image_width <= 0
        or image_height != image_width
        or image_height % 2
        or image_width % 2
    ):
        raise ValueError(f"expected a positive even square image shape, got {image_shape}")
    indices = np.asarray(pixel_indices, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        raise ValueError("RELION projector compaction requires at least one pixel index")
    image_half_width = image_width // 2 + 1
    if np.any(indices < 0) or np.any(indices >= image_height * image_half_width):
        raise ValueError("RELION projector compaction pixel indices exceed the half image")
    rows = indices // image_half_width
    columns = indices - rows * image_half_width
    centered_rows = rows - image_height // 2
    max_radius_squared = int(np.max(centered_rows * centered_rows + columns * columns))
    # ``isqrt(max_r2) + 1`` is a strict radius bound, including when the
    # farthest score pixel lies exactly on an integer-radius shell.
    compact_r_max = min(r_max, max(1, math.isqrt(max_radius_squared) + 1))
    if compact_r_max == r_max:
        return projector_half, r_max

    compact_padded_r_max = compact_r_max * padding_factor
    source_center = padded_r_max + 1
    compact_center = compact_padded_r_max + 1
    start = source_center - compact_center
    stop = source_center + compact_center + 1
    compact = np.ascontiguousarray(
        projector_half[
            start:stop,
            start:stop,
            : compact_padded_r_max + 2,
        ],
    )
    expected_compact_shape = (
        2 * compact_center + 1,
        2 * compact_center + 1,
        compact_padded_r_max + 2,
    )
    if compact.shape != expected_compact_shape:
        raise RuntimeError(
            "internal RELION projector compaction shape mismatch: "
            f"got {compact.shape}, expected {expected_compact_shape}",
        )
    return compact, compact_r_max
