"""What a numbered reconstruction does to its maps after the solve: RELION's initial low-pass filter,
solvent flattening (and its mask), and the sign alignment of a K=1 map to its reference."""

from __future__ import annotations

import functools
import logging
import math
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from recovar.core import fourier_transform_utils, mask

from relax.helpers.xla_memory_reserve import device_fits
from relax.reconstruction import regularization_relion
from relax.relion.reference_initialization import initial_low_pass_filter_references

if TYPE_CHECKING:
    from relax.refinement.numbered_reconstruction import ReconstructionSettings

logger = logging.getLogger(__name__)


def _apply_relion_initial_lowpass_filter(
    volume_ft_flat, volume_shape, voxel_size, ini_high_angstrom, filter_edgewidth
):
    """Apply RELION's ``initialLowPassFilterReferences`` to a full Fourier volume."""
    if ini_high_angstrom is None or float(ini_high_angstrom) <= 0.0:
        return volume_ft_flat
    # RELION filters Iref in double on the host. Both transforms run there too, on the CPU backend, and only the
    # filtered map goes back to the device in its dtype: on the device the inverse transform's copies and the
    # complex128 forward transform held seven maps, 1.34 GiB requests at box 448, which a 16 GB V100 could not
    # place in iteration 1 (relax#47).
    dtype = volume_ft_flat.dtype
    with jax.default_device("cpu"):
        original = jnp.asarray(np.asarray(volume_ft_flat)).reshape(volume_shape)
        volume_real = np.real(np.asarray(fourier_transform_utils.get_idft3(original))).astype(np.float64, copy=False)
        del original
        filtered_real = initial_low_pass_filter_references(
            volume_real[None, ...],
            box_size=int(volume_shape[0]),
            pixel_size=float(voxel_size),
            ini_high_ang=float(ini_high_angstrom),
            filter_edgewidth=float(filter_edgewidth),
        )[0]
        del volume_real
        filtered_ft = np.asarray(fourier_transform_utils.get_dft3(jnp.asarray(filtered_real)).astype(dtype))
    return jnp.asarray(filtered_ft.reshape(-1))


@functools.partial(jax.jit, static_argnames=("volume_shape",))
def _centered_real_overlap(volume_ft_flat, reference_ft_flat, *, volume_shape):
    """The dot product of the two volumes' real parts in real space, each less its mean, on the device."""

    vol_real = jnp.real(fourier_transform_utils.get_idft3(volume_ft_flat.reshape(volume_shape)))
    ref_real = jnp.real(fourier_transform_utils.get_idft3(reference_ft_flat.reshape(volume_shape)))
    return jnp.sum((ref_real - jnp.mean(ref_real)) * (vol_real - jnp.mean(vol_real)))


# The sign overlap program holds about three complex64 maps (the two inverse transforms and a temporary): one 2.01
# GiB request at box 448 (relax#49).
_SIGN_OVERLAP_MAPS = 3.0


def _sign_overlap_exceeds_device_headroom(volume_shape) -> bool:
    """Whether the sign overlap's working set (``_SIGN_OVERLAP_MAPS`` complex64 maps, one program request: 2.01 GiB at
    box 448 failed in a fragmented 16 GB pool, relax#49) does not fit the device now (``device_fits``)."""

    return not device_fits(_SIGN_OVERLAP_MAPS * int(np.prod(volume_shape)) * 8)


def _align_fourier_volume_sign_to_reference(volume_ft_flat, reference_ft_flat, volume_shape):
    """Keep reconstructed volumes on the same real-space sign branch as the reference.

    Only the overlap's sign is read back: the volumes' inverse transforms, centring and dot product stay on the
    device (copying both real volumes to the host in float64 took 7.6 s of a 10k-particle box-256 auto-refine).
    """
    if reference_ft_flat is None:
        return volume_ft_flat, False
    shape = tuple(int(n) for n in volume_shape)
    if _sign_overlap_exceeds_device_headroom(shape):
        # The same program on the CPU backend: at the end of a box-448 K=1 M-step on a 16 GB card the device held
        # both halves' accumulators and unregularized maps, and the program's 2.01 GiB could not be placed (relax#49).
        logger.info("K=1 sign overlap on the CPU backend: shape=%s", shape)
        cpu = jax.devices("cpu")[0]
        with jax.default_device(cpu):
            overlap = float(
                _centered_real_overlap(
                    jax.device_put(volume_ft_flat, cpu), jax.device_put(reference_ft_flat, cpu), volume_shape=shape
                )
            )
    else:
        overlap = float(
            _centered_real_overlap(jnp.asarray(volume_ft_flat), jnp.asarray(reference_ft_flat), volume_shape=shape)
        )
    if overlap < 0.0:
        return -volume_ft_flat, True
    return volume_ft_flat, False


def _solvent_flatten_requested(settings: ReconstructionSettings) -> bool:
    """Return whether new references are solvent-flattened: a user mask or a particle diameter is set."""
    if settings.solvent_mask is not None:
        return True
    return settings.particle_diameter_angstrom is not None and settings.particle_diameter_angstrom > 0


def _numbered_solvent_mask(settings: ReconstructionSettings, *, dtype):
    """The solvent mask in ``dtype``: the user mask, else the soft sphere of the particle diameter.

    RELION's solventFlatten multiplies each reference by the user mask as read (no soft edge
    added), or by the cosine-edged sphere when there is none (ml_optimiser.cpp:5506-5590).
    """
    if settings.solvent_mask is not None:
        return jnp.asarray(settings.solvent_mask, dtype=dtype)
    flatten_radius = settings.particle_diameter_angstrom / (2.0 * settings.voxel_size)
    return _make_relion_solvent_mask(
        settings.volume_shape,
        radius=flatten_radius,
        radius_p=flatten_radius + settings.width_mask_edge,
        dtype=dtype,
    )


@functools.partial(jax.jit, donate_argnums=0)
def _set_class_row(class_maps, row, class_idx):
    """``class_maps`` with row ``class_idx`` replaced by ``row``, in place (the stack is donated)."""
    return class_maps.at[class_idx].set(row)


def _lowpass_class_stack(class_maps, settings: ReconstructionSettings, n_classes):
    """Apply the first-CC initial low-pass to every class, in place (``class_maps`` is consumed).

    One class is filtered at a time and written back into the stack, so the stack and one class's temporaries
    are live. Building the classes in a list and stacking them held the input, the classes and the new stack
    together: 1.23 GiB each for Class3D K=3 at box 380 (relax#45).
    """
    for class_idx in range(n_classes):
        row = _apply_relion_initial_lowpass_filter(
            class_maps[class_idx],
            settings.volume_shape,
            settings.voxel_size,
            settings.first_iteration_lowpass_angstrom,
            filter_edgewidth=settings.fmask_edge,
        )
        class_maps = _set_class_row(class_maps, row, class_idx)
        del row
    return class_maps


@functools.partial(jax.jit, static_argnames=("volume_shape",))
def _flatten_volume(volume_ft_flat, solvent_mask, *, volume_shape):
    """RELION's solventFlatten of one flat Fourier map as one program: inverse transform, mask, forward transform."""
    vol_real = fourier_transform_utils.get_idft3(volume_ft_flat.reshape(volume_shape))
    return fourier_transform_utils.get_dft3(vol_real * solvent_mask).reshape(-1)


@functools.partial(jax.jit, static_argnames=("volume_shape",), donate_argnums=0)
def _flatten_class_row(class_maps, solvent_mask, class_idx, *, volume_shape):
    """``class_maps`` with class ``class_idx`` solvent-flattened, in place (the stack is donated)."""
    return class_maps.at[class_idx].set(_flatten_volume(class_maps[class_idx], solvent_mask, volume_shape=volume_shape))


def _flatten_class_stack(class_maps, solvent_mask, volume_shape, n_classes):
    """Solvent-flatten every class on the device with one mask, in place (``class_maps`` is consumed).

    One program per class keeps the stack plus about two class maps of transforms live (measured at box 256:
    2.0 maps; the eager statements held 5.0, and collecting the classes in a list then stacking them 11.8).
    """
    volume_shape = tuple(int(n) for n in volume_shape)
    for class_idx in range(n_classes):
        class_maps = _flatten_class_row(class_maps, solvent_mask, class_idx, volume_shape=volume_shape)
    return class_maps


def _log_first_cc_lowpass(settings: ReconstructionSettings) -> None:
    """Report the first-CC initial low-pass once all slots are postprocessed."""
    if settings.first_iteration_lowpass_angstrom is not None:
        logger.info(
            "RELION iter-1 CC emulation: reapplying ini_high low-pass filter at %.2f A",
            float(settings.first_iteration_lowpass_angstrom),
        )


def align_k1_volume_signs(means, previous_means, unregularized_means, volume_shape) -> None:
    """Align K=1 means and matching diagnostic maps to previous references, in place (``means[k]`` and
    ``unregularized_means[k]`` are replaced)."""

    for k in range(2):
        means[k], sign_flipped = _align_fourier_volume_sign_to_reference(
            means[k],
            previous_means[k],
            volume_shape,
        )
        if sign_flipped and unregularized_means[k] is not None:
            unregularized_means[k] = -unregularized_means[k]
        if sign_flipped:
            logger.info("Aligned half-%d volume sign to the previous reference", k + 1)


_LARGE_RELION_SOLVENT_MASK_COORDINATE_BYTES_LIMIT = 2 * 1024**3


def _relion_solvent_mask_unfused_coordinate_bytes(volume_shape) -> int:
    """Estimate the promoted coordinate stack used by ``raised_cosine_mask``."""

    return math.prod(int(size) for size in volume_shape) * 3 * np.dtype(np.float64).itemsize


def _large_relion_solvent_mask_uses_compiled_builder(volume_shape) -> bool:
    """Return whether the unfused solvent-mask coordinate stack is too large."""

    return (
        _relion_solvent_mask_unfused_coordinate_bytes(volume_shape) > _LARGE_RELION_SOLVENT_MASK_COORDINATE_BYTES_LIMIT
    )


@functools.cache
def _compiled_relion_solvent_mask(volume_shape, *, dtype=None):
    """Cache a shape-specialized builder, without retaining a mask array."""

    volume_shape = tuple(int(size) for size in volume_shape)

    @jax.jit
    def build(radius, radius_p, offset):
        return mask.raised_cosine_mask(
            volume_shape,
            radius=radius,
            radius_p=radius_p,
            offset=offset,
            dtype=dtype,
        )

    return build


def _make_relion_solvent_mask(volume_shape, *, radius, radius_p, dtype):
    """Build a RELION solvent mask, centred, without materializing a giant coordinate stack."""

    volume_shape = tuple(int(size) for size in volume_shape)
    if not _large_relion_solvent_mask_uses_compiled_builder(volume_shape):
        return mask.raised_cosine_mask(
            volume_shape,
            radius=radius,
            radius_p=radius_p,
            offset=jnp.zeros(3),
            dtype=dtype,
        )

    estimated_bytes = _relion_solvent_mask_unfused_coordinate_bytes(volume_shape)
    logger.info(
        "RELION box-scale solvent mask fused construction: shape=%s estimated_unfused_coordinate_bytes=%d",
        volume_shape,
        estimated_bytes,
    )
    solvent_mask = _compiled_relion_solvent_mask(volume_shape, dtype=dtype)(
        radius,
        radius_p,
        jnp.zeros(3),
    )
    solvent_mask.block_until_ready()
    logger.info(
        "RELION box-scale solvent mask ready: shape=%s dtype=%s",
        volume_shape,
        solvent_mask.dtype,
    )
    return solvent_mask


def _apply_relion_solvent_flatten_k1(
    volume_ft_flat,
    solvent_mask,
    volume_shape,
):
    """Apply the K=1 solvent mask and host-stage box-scale FFT results."""

    # One program (_flatten_volume): the eager statements held four box-size shift copies and transforms at once, and
    # at box 448 on a 16 GB card the fifth could not be placed (relax#49).
    flattened = _flatten_volume(volume_ft_flat, solvent_mask, volume_shape=tuple(int(n) for n in volume_shape))
    if not _large_relion_solvent_mask_uses_compiled_builder(volume_shape):
        return flattened

    # JAX dispatch is asynchronous.  At box 800, even after the first half's
    # real-space volume and mask are released, retaining its complex128 FFT
    # output leaves too little room for the second normalized FFT allocation.
    # Finish the exact existing FFT, copy its completed bits to host memory,
    # and release all three device buffers before starting the next half.  The
    # transform arithmetic, normalization, dtype, and flattened shape remain
    # unchanged; only the storage owner crosses the device/host boundary.
    flattened.block_until_ready()
    flattened_host = np.array(jax.device_get(flattened), copy=True, order="C")
    regularization_relion.delete_device_array(flattened)
    regularization_relion.delete_device_array(solvent_mask)
    return flattened_host
