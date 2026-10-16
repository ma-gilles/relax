"""RELION's low-resolution join of the two half maps (``--low_resol_join_halves``).

``join_halves_at_low_resolution`` averages the two halves' Fourier coefficients below the join resolution and
writes the average into both, on the device or, for a large map, through the host
(``LowResolutionJoinRoute``: forced, or above ``LOW_RESOLUTION_JOIN_HOST_FALLBACK_MIN_ELEMENTS`` elements).
The definitions moved verbatim from ``relax.reconstruction.regularization``, whose shell statistics they do not
use; the generic device release ``regularization.delete_device_array`` stays there.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar.reconstruction.regularization import logger

from relax.reconstruction.regularization import delete_device_array
from relax.relion.macros import relion_round
from relax.runtime.xla_memory_reserve import device_fits

LOW_RESOLUTION_JOIN_HOST_FALLBACK_MIN_ELEMENTS = 200_000_000


@dataclass(frozen=True)
class LowResolutionJoinRoute:
    """Where the low-resolution half join runs; no effect on its values (``ReconstructionPrograms.lowres_join``).

    ``forced_host`` (``RELAX_LOWRES_JOIN_HOST_FALLBACK``, ``env_flags.parse_env_auto_flag``): True on the host,
    False on the device, None by size; ``host_min_elements`` (``RELAX_LOWRES_JOIN_HOST_FALLBACK_MIN_ELEMENTS``,
    default :data:`LOW_RESOLUTION_JOIN_HOST_FALLBACK_MIN_ELEMENTS`): the accumulator size from which it runs on
    the host.
    """

    forced_host: bool | None
    host_min_elements: int


def _unscaled_fft_frequency_grid_np(n):
    half = int(n) // 2
    return np.arange(-half, int(n) - half, dtype=np.int64)


def _unscaled_rfft_frequency_grid_np(n):
    return np.arange(0, int(n) // 2 + 1, dtype=np.int64)


@functools.lru_cache(maxsize=64)
def _low_resolution_join_flat_indices(volume_shape, half_layout, lowres_r2_max):
    """Return flat Fourier voxels inside RELION's low-resolution join sphere."""
    volume_shape = tuple(int(s) for s in volume_shape)
    if half_layout:
        layout_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(volume_shape)
        x_coords = _unscaled_rfft_frequency_grid_np(volume_shape[2])
    else:
        layout_shape = volume_shape
        x_coords = _unscaled_fft_frequency_grid_np(volume_shape[2])

    z_coords = _unscaled_fft_frequency_grid_np(volume_shape[0])
    y_coords = _unscaled_fft_frequency_grid_np(volume_shape[1])
    y2 = y_coords * y_coords
    x2 = x_coords * x_coords
    ny = int(layout_shape[1])
    nx = int(layout_shape[2])

    flat_chunks = []
    for zi, z_coord in enumerate(z_coords):
        remaining_after_z = lowres_r2_max - int(z_coord * z_coord)
        if remaining_after_z < 0:
            continue
        y_indices = np.nonzero(y2 <= remaining_after_z)[0]
        z_offset = zi * ny * nx
        for yi in y_indices:
            remaining_after_y = remaining_after_z - int(y2[yi])
            x_indices = np.nonzero(x2 <= remaining_after_y)[0]
            if x_indices.size:
                flat_chunks.append((z_offset + int(yi) * nx + x_indices).astype(np.int32, copy=False))

    if not flat_chunks:
        return np.empty((0,), dtype=np.int32)
    return np.concatenate(flat_chunks).astype(np.int32, copy=False)


def _low_resolution_join_host_fallback_enabled_for_size(
    values_size, join_size, *, route: LowResolutionJoinRoute, itemsize: int = 8
):
    """Whether the low-resolution half join runs on the host.

    Physically large accumulators (the host-staged reconstruction's grids) always do. Otherwise the device join
    holds both joined copies of a half pair (``2 * values_size * itemsize`` bytes), and it runs on the host when
    that does not fit the device (``xla_memory_reserve.device_fits``): at box 448 on a 16 GB card the 710 MiB
    scatter could not be placed in a fragmented pool (relax#49).
    """
    if route.forced_host is not None:
        return route.forced_host
    if int(join_size) >= int(values_size):
        return False
    return int(values_size) >= route.host_min_elements or not device_fits(2 * int(values_size) * int(itemsize))


def _low_resolution_join_host_fallback_enabled(values_0, flat_indices, route: LowResolutionJoinRoute):
    return _low_resolution_join_host_fallback_enabled_for_size(
        np.size(values_0), np.size(flat_indices), route=route, itemsize=np.dtype(values_0.dtype).itemsize
    )


@functools.partial(jax.jit, donate_argnums=(0,))
def _scatter_joined_values_into_first_device(values_0, flat_indices, joined_values):
    """Write host-computed join values while reusing the first device buffer."""

    joined = values_0.reshape(-1).at[flat_indices].set(
        joined_values,
        indices_are_sorted=True,
        unique_indices=True,
    )
    return joined.reshape(values_0.shape)


def _join_half_pair_at_indices_host(
    values_0,
    values_1,
    flat_indices,
    *,
    preserve_inputs=True,
    retain_first_device=False,
):
    values_0_np = np.asarray(jax.device_get(values_0))
    values_1_np = np.asarray(jax.device_get(values_1))
    if preserve_inputs or not values_0_np.flags.writeable:
        values_0_np = np.array(values_0_np, copy=True)
    if preserve_inputs or not values_1_np.flags.writeable:
        values_1_np = np.array(values_1_np, copy=True)
    flat_indices_np = np.asarray(jax.device_get(flat_indices), dtype=np.intp)

    values_0_flat = values_0_np.reshape(-1)
    values_1_flat = values_1_np.reshape(-1)
    half_scalar = np.asarray(0.5, dtype=values_0_flat.real.dtype)
    if int(flat_indices_np.size) >= int(values_0_flat.size):
        average_at_join = ((values_0_flat + values_1_flat) * half_scalar).astype(
            values_0_flat.dtype,
            copy=False,
        )
        values_0_np = average_at_join.reshape(values_0_np.shape)
        values_1_np = average_at_join.reshape(values_1_np.shape).copy()
    else:
        average_at_join = (
            (values_0_flat[flat_indices_np] + values_1_flat[flat_indices_np]) * half_scalar
        ).astype(values_0_flat.dtype, copy=False)
        values_0_flat[flat_indices_np] = average_at_join
        values_1_flat[flat_indices_np] = average_at_join

    retained_first_device = None
    if retain_first_device and not preserve_inputs:
        if isinstance(values_0, np.ndarray):
            logger.info(
                "Low-resolution half-join reserving joined first numerator on device: "
                "elements=%d joined=%d dtype=%s",
                int(values_0_flat.size),
                int(flat_indices_np.size),
                values_0_np.dtype,
            )
            retained_first_device = jnp.asarray(values_0_np)
            retained_first_device.block_until_ready()
        else:
            logger.info(
                "Low-resolution half-join retaining first numerator device buffer: "
                "elements=%d joined=%d dtype=%s",
                int(values_0_flat.size),
                int(flat_indices_np.size),
                values_0.dtype,
            )
            retained_first_device = _scatter_joined_values_into_first_device(
                values_0,
                jnp.asarray(flat_indices_np, dtype=jnp.int32),
                jnp.asarray(average_at_join),
            )
    else:
        delete_device_array(values_0)
    delete_device_array(values_1)
    return values_0_np, values_1_np, retained_first_device


def _join_half_pair_at_indices(values_0, values_1, flat_indices, route: LowResolutionJoinRoute):
    values_0_flat = values_0.reshape(-1)
    values_1_flat = values_1.reshape(-1)
    if int(flat_indices.size) >= int(values_0_flat.size):
        average = 0.5 * (values_0_flat + values_1_flat)
        return average.reshape(values_0.shape), average.reshape(values_1.shape)

    if _low_resolution_join_host_fallback_enabled(values_0_flat, flat_indices, route):
        logger.info(
            "Low-resolution half-join using host fallback: elements=%d joined=%d dtype=%s",
            int(values_0_flat.size),
            int(flat_indices.size),
            values_0.dtype,
        )
        joined_0, joined_1, _ = _join_half_pair_at_indices_host(
            values_0,
            values_1,
            flat_indices,
        )
        return joined_0, joined_1

    average_at_join = 0.5 * (values_0_flat[flat_indices] + values_1_flat[flat_indices])
    joined_0 = values_0_flat.at[flat_indices].set(average_at_join)
    joined_1 = values_1_flat.at[flat_indices].set(average_at_join)
    return joined_0.reshape(values_0.shape), joined_1.reshape(values_1.shape)


def join_halves_at_low_resolution(
    Ft_y_0,
    Ft_y_1,
    Ft_ctf_0,
    Ft_ctf_1,
    volume_shape,
    voxel_size,
    box_size,
    low_resol_join_halves_angstrom,
    current_resolution_angstrom=None,
    padding_factor=None,
    preserve_inputs=True,
    return_retained_first_numerator=False,
    *,
    route: LowResolutionJoinRoute,
):
    """RELION's ``--low_resol_join_halves`` operation on Fourier accumulators (on the host or the device, as
    ``route``, ``ReconstructionPrograms.lowres_join``, decides).

    Mirrors ``MlOptimiserMpi::joinTwoHalvesAtLowResolution`` in
    ``relion/src/ml_optimiser_mpi.cpp:3112-3219``: at low resolutions where
    the half-set reconstructions are unreliably independent, RELION
    averages the **backprojection accumulators** (``data`` ↔ ``Ft_y`` and
    ``weight`` ↔ ``Ft_ctf``) of the two halves at every Fourier voxel
    inside a low-resolution sphere, then writes the average back into both
    halves before doing the Wiener solve. This forces the iter's two
    half-maps to share their low-frequency content, preventing the
    half-sets from diverging in orientation space at SNR-poor low shells.

    The joining radius (in shells) is set by the LARGER (lower-frequency)
    of:
        - ``low_resol_join_halves_angstrom`` (the user/GUI default 40 Å), and
        - ``current_resolution_angstrom`` (the iter's resolution estimate)
    so that the joining radius never exceeds the actual resolution of the
    map (which would join shells where the FSC is genuinely high).
    Concretely:

    .. code-block:: text

        myres = max(low_resol_join_halves_angstrom, current_resolution_angstrom)
        lowres_r_max = ceil(box_size * voxel_size / myres)

    matching ``ml_optimiser_mpi.cpp:3122-3123``:

    .. code-block:: cpp

        RFLOAT myres = XMIPP_MAX(low_resol_join_halves, 1./mymodel.current_resolution);
        int lowres_r_max = CEIL(mymodel.ori_size * mymodel.pixel_size / myres);

    Parameters
    ----------
    Ft_y_0, Ft_y_1 : array (volume_size,) complex
        Per-half ``Pᵀy`` (numerator of the Wiener filter) accumulators
        from the M-step, in centered Fourier order, flattened.
    Ft_ctf_0, Ft_ctf_1 : array (volume_size,) float
        Per-half ``Pᵀ(CTF² / σ²)`` (denominator) accumulators.
    volume_shape : tuple of 3 ints
        Shape of the centered Fourier volume.
    voxel_size : float
        Voxel size in Angstroms (image pixel size in real space).
    box_size : int
        Real-space grid edge length, ``ori_size`` in RELION terms.
    low_resol_join_halves_angstrom : float
        The user-set joining resolution (RELION's ``--low_resol_join_halves``).
        Pass ``<= 0`` to disable; the function then returns the inputs
        unchanged.
    current_resolution_angstrom : float or None
        The current iteration's resolution estimate in Angstroms. The
        joining radius is the LOWER frequency (LARGER Å) of this and
        ``low_resol_join_halves_angstrom``. Pass ``None`` (the default)
        to ignore (equivalent to passing ``+inf``).
    padding_factor : int or None
        RELION backprojector padding factor used to convert native-shell
        join radii to accumulator-space coordinates. If omitted, falls back
        to the legacy shape-based inference, which is only reliable for full
        padded accumulators and not current-size BPref grids.
    preserve_inputs : bool
        Keep the four input accumulators unchanged. Numbered EM iterations
        may set this to ``False`` after saving pre-join diagnostics; writable
        host arrays then update only the joined entries in existing storage,
        as RELION does. Final all-data reconstruction leaves this enabled
        because its unfiltered half maps retain the pre-join accumulators.
    return_retained_first_numerator : bool
        Internal K=1 memory option. When the large host fallback receives a
        device-resident first numerator with ``preserve_inputs=False``, append
        a fifth return value containing that buffer after donating it to an
        exact sparse scatter of the host-computed joined entries. The four
        ordinary returns remain host arrays for FSC/tau2. The default keeps the
        public four-value API.

    Returns
    -------
    (Ft_y_0_joined, Ft_y_1_joined, Ft_ctf_0_joined, Ft_ctf_1_joined)
        Accumulators with the low-resolution shells averaged. Outside the
        joining sphere they are identical to the inputs. With
        ``preserve_inputs=False``, writable host inputs may be returned and
        updated in place. If ``return_retained_first_numerator=True``, the
        tuple has a fifth entry as described above, or ``None`` when no device
        buffer was retained.
    """

    def _format_result(values, retained_first_numerator=None):
        if return_retained_first_numerator:
            return (*values, retained_first_numerator)
        return values

    if low_resol_join_halves_angstrom is None or low_resol_join_halves_angstrom <= 0:
        return _format_result((Ft_y_0, Ft_y_1, Ft_ctf_0, Ft_ctf_1))

    # Effective joining resolution: the larger (lower-frequency) of
    # low_resol_join_halves and current_resolution.
    myres = float(low_resol_join_halves_angstrom)
    if current_resolution_angstrom is not None and np.isfinite(current_resolution_angstrom):
        myres = max(myres, float(current_resolution_angstrom))

    lowres_r_max = int(np.ceil(box_size * voxel_size / myres))
    if lowres_r_max <= 0:
        return _format_result((Ft_y_0, Ft_y_1, Ft_ctf_0, Ft_ctf_1))

    # RELION BackProjector::getLowResDataAndWeight / setLowResDataAndWeight
    # uses squared coordinates, not rounded shell labels:
    #   lowres_r2_max = ROUND(padding_factor * lowres_r_max)^2
    #   if (k*k + i*i + j*j <= lowres_r2_max) ...
    # Using rounded radial shells joins extra boundary voxels and changes the
    # downsampled half-map FSC near the 40 A join cutoff.
    if padding_factor is None:
        pf = volume_shape[0] // box_size if volume_shape[0] > box_size else 1
    else:
        pf = int(padding_factor)
        if pf <= 0:
            raise ValueError(f"padding_factor must be positive, got {padding_factor}")
    lowres_r_max_padded = relion_round(float(pf) * lowres_r_max)
    lowres_r2_max = lowres_r_max_padded * lowres_r_max_padded

    volume_shape = tuple(int(s) for s in volume_shape)
    half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(volume_shape)
    full_size = int(np.prod(volume_shape))
    half_size = int(np.prod(half_shape))
    ft_y_size = int(np.size(Ft_y_0))
    if ft_y_size == full_size:
        half_layout = False
    elif ft_y_size == half_size:
        half_layout = True
    else:
        raise ValueError(
            f"Could not infer Fourier layout for join_halves_at_low_resolution with shape {np.shape(Ft_y_0)} "
            f"and volume_shape={volume_shape}"
        )

    join_indices_np = _low_resolution_join_flat_indices(volume_shape, half_layout, lowres_r2_max)
    if join_indices_np.size == 0:
        return _format_result((Ft_y_0, Ft_y_1, Ft_ctf_0, Ft_ctf_1))

    max_input_size = max(
        ft_y_size,
        int(np.size(Ft_y_1)),
        int(np.size(Ft_ctf_0)),
        int(np.size(Ft_ctf_1)),
    )
    # The host/device decision counts the physical accumulator grid, the unit the
    # reconstruction's large-grid boundary uses (numbered_reconstruction.
    # _should_host_stage_large_relion_ifft): a packed half of a physically large
    # grid stores about half its voxels. Counting stored elements moved the host
    # half accumulators of a 611^3 grid (114M elements, 228M voxels) back to the
    # device, and the reconstruction then took its monolithic device 1600^3
    # inverse FFT and failed to create the cuFFT plan (EMPIAR-10202 it3, 14456981).
    if _low_resolution_join_host_fallback_enabled_for_size(
        max(max_input_size, full_size), join_indices_np.size, route=route,
    ):
        retain_first_device = (
            return_retained_first_numerator
            and not preserve_inputs
            and padding_factor is not None
            and int(volume_shape[0]) > int(box_size) * int(padding_factor)
        )
        logger.info(
            "Low-resolution half join using host fallback: size=%d join_voxels=%d "
            "preserve_inputs=%s retain_first_device=%s",
            max_input_size,
            int(join_indices_np.size),
            bool(preserve_inputs),
            bool(retain_first_device),
        )
        (
            Ft_y_0_joined,
            Ft_y_1_joined,
            retained_first_numerator,
        ) = _join_half_pair_at_indices_host(
            Ft_y_0,
            Ft_y_1,
            join_indices_np,
            preserve_inputs=preserve_inputs,
            retain_first_device=retain_first_device,
        )
        Ft_ctf_0_joined, Ft_ctf_1_joined, _ = _join_half_pair_at_indices_host(
            Ft_ctf_0,
            Ft_ctf_1,
            join_indices_np,
            preserve_inputs=preserve_inputs,
        )
        return _format_result(
            (Ft_y_0_joined, Ft_y_1_joined, Ft_ctf_0_joined, Ft_ctf_1_joined),
            retained_first_numerator,
        )

    Ft_y_0_arr = jnp.asarray(Ft_y_0)
    Ft_y_1_arr = jnp.asarray(Ft_y_1)
    Ft_ctf_0_arr = jnp.asarray(Ft_ctf_0)
    Ft_ctf_1_arr = jnp.asarray(Ft_ctf_1)
    join_indices = jnp.asarray(join_indices_np)

    Ft_y_0_joined, Ft_y_1_joined = _join_half_pair_at_indices(Ft_y_0_arr, Ft_y_1_arr, join_indices, route)
    Ft_ctf_0_joined, Ft_ctf_1_joined = _join_half_pair_at_indices(Ft_ctf_0_arr, Ft_ctf_1_arr, join_indices, route)

    return _format_result((Ft_y_0_joined, Ft_y_1_joined, Ft_ctf_0_joined, Ft_ctf_1_joined))
