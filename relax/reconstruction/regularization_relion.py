"""EM-only symbols relocated from ``recovar.reconstruction.regularization`` (relax split P1, pure move).

Function bodies are AST-identical to their originals at Q (tag q-reconcile-20260923); only the module
path changed.  See pr179_coordination/relax_split_plan_20260923/PLAN.md.
"""

import functools
import os
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar import jax_config
from recovar.reconstruction.regularization import (  # noqa: F401  (staying helpers and shared loader state)
    logger,
)

from relax.helpers.env_flags import parse_env_auto_flag
from relax.helpers.shells import shell_of_radius
from relax.helpers.xla_memory_reserve import device_fits
from relax.relion.macros import relion_round, relion_round_array

_RELION_SHELL_STATS_DEVICE_REDUCTION_MAX_VOXELS = 200_000_000
# Elements of one host shell-statistics block (_numpy_bincount_shell_stats): about 0.5 GiB of temporaries.
_SHELL_STATS_BLOCK = 1 << 24
# Device bytes per padded-grid voxel of the device shell statistics: the two halves' weights, their combination and
# its float64 copy (8 B each), and the radius, shell-index and mask grids of _padded_shell_sums_device.
_SHELL_STATS_DEVICE_BYTES_PER_VOXEL = 48
# Largest share of what the device can still hand out that the device shell statistics may take. The final tau2 runs
# beside the final all-data pass's accumulators and the XLA pool's fragmentation, and the estimate above counts the
# named arrays only, not the reductions' temporaries; half leaves room for both.
_SHELL_STATS_DEVICE_SHARE = 0.5
# `shell_rounding` ("round" is RELION ROUND) as a relax.helpers.shells rule.
_SHELL_RULE = {"round": "half_up", "floor": "floor"}


def _shell_stats_on_host(n_voxels: int) -> bool:
    """Whether the padded-grid shell statistics of ``n_voxels`` reduce on the host instead of the device.

    On the host past ``_RELION_SHELL_STATS_DEVICE_REDUCTION_MAX_VOXELS``, or when their device arrays would take more
    than ``_SHELL_STATS_DEVICE_SHARE`` of what the device can still hand out: a box-256 final tau2 at padding 2
    (512^3 voxels) asked for another 1.02 GiB on a 16 GB card after the final all-data pass and ran out of memory
    (A100 emulating 16 GB, relax c8ac3e6). The two reductions sum the same shells; the routing changes only where
    they run.
    """

    if int(n_voxels) > _RELION_SHELL_STATS_DEVICE_REDUCTION_MAX_VOXELS:
        return True
    from relax.sparse_pass2.sparse_pass2_budget import (
        _device_free_memory_bytes,
        _jax_allocator_free_memory_bytes,
        _jax_allocator_pool_free_bytes,
        device_available_bytes,
    )

    if jax.default_backend() != "gpu":
        return False
    available = device_available_bytes(
        _device_free_memory_bytes(), _jax_allocator_free_memory_bytes(), _jax_allocator_pool_free_bytes()
    )
    return available is not None and (
        int(n_voxels) * _SHELL_STATS_DEVICE_BYTES_PER_VOXEL > _SHELL_STATS_DEVICE_SHARE * float(available)
    )


_LOW_RESOLUTION_JOIN_HOST_FALLBACK_MIN_ELEMENTS = 200_000_000


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


def _low_resolution_join_host_fallback_enabled_for_size(values_size, join_size, *, itemsize: int = 8):
    """Whether the low-resolution half join runs on the host.

    Physically large accumulators (the host-staged reconstruction's grids) always do. Otherwise the device join
    holds both joined copies of a half pair (``2 * values_size * itemsize`` bytes), and it runs on the host when
    that does not fit the device (``xla_memory_reserve.device_fits``): at box 448 on a 16 GB card the 710 MiB
    scatter could not be placed in a fragmented pool (relax#49).
    """
    forced = parse_env_auto_flag("RELAX_LOWRES_JOIN_HOST_FALLBACK", logger=logger)
    if forced is not None:
        return forced
    if int(join_size) >= int(values_size):
        return False
    threshold = int(
        os.environ.get(
            "RELAX_LOWRES_JOIN_HOST_FALLBACK_MIN_ELEMENTS",
            _LOW_RESOLUTION_JOIN_HOST_FALLBACK_MIN_ELEMENTS,
        )
    )
    return int(values_size) >= threshold or not device_fits(2 * int(values_size) * int(itemsize))


def _low_resolution_join_host_fallback_enabled(values_0, flat_indices):
    return _low_resolution_join_host_fallback_enabled_for_size(
        np.size(values_0), np.size(flat_indices), itemsize=np.dtype(values_0.dtype).itemsize
    )


def delete_device_array(value):
    """Release a completed JAX buffer even when another dead handle survives."""

    delete = getattr(value, "delete", None)
    if callable(delete):
        try:
            delete()
        except RuntimeError:
            # Donation invalidates the input handle when the output aliases it.
            pass


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


def _join_half_pair_at_indices(values_0, values_1, flat_indices):
    values_0_flat = values_0.reshape(-1)
    values_1_flat = values_1.reshape(-1)
    if int(flat_indices.size) >= int(values_0_flat.size):
        average = 0.5 * (values_0_flat + values_1_flat)
        return average.reshape(values_0.shape), average.reshape(values_1.shape)

    if _low_resolution_join_host_fallback_enabled(values_0_flat, flat_indices):
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


def compute_relion_tau2_from_iref_power_spectrum(
    Iref_padded_fourier,
    volume_shape,
    *,
    padding_factor=1,
    current_size=None,
    return_details=False,
    projector_power_spectrum=None,
    shell_pair_counting="relion",
):
    """Compute RELION-style tau2 from a previous Iref Fourier volume.

    Mirrors RELION's ``Projector::computeFourierTransformMap`` path by
    converting the reference back to real space and taking the power spectrum
    of relax's projector setup of it. The returned
    spectrum is expanded back into RECOVAR's centered Fourier layout.

    The output is in the same Fourier-amplitude scale RECOVAR uses for
    ``mean_variance`` / Wiener regularization. ``current_size`` optionally
    clips the spectrum to the same resolution limit RELION uses when updating
    the projector map.

    ``projector_power_spectrum`` is the power spectrum the scoring projector
    setup of this same reference and padding already computed
    (:func:`relax.relion.relion_projector_setup.reference_to_relion_projector_half_maps_and_power`,
    which reproduces ``computeFourierTransformMap`` for a projector of 2-D
    images). RELION computes the projector and its spectrum in one
    ``computeFourierTransformMap`` call per class; passing it skips a second,
    host, transform, and ``Iref_padded_fourier`` is then not read. That
    projector scales the transform by ``normfft = pf^3 * ori_size``
    (``data_dim=2``) where RELION's ``ReferenceTau2`` uses the ``data_dim=3``
    scale (``normfft = pf^3``), so its power is ``ori_size^2`` times larger and
    is divided back here. Without it, the same projector setup is built here,
    with ``shell_pair_counting`` (a supplied spectrum carries its own).
    """

    volume_shape = tuple(int(s) for s in volume_shape)
    current_size = None if current_size is None else int(current_size)
    if projector_power_spectrum is not None:
        relion_power_spectrum = np.asarray(projector_power_spectrum, dtype=np.float64) / float(volume_shape[0]) ** 2
    else:
        from recovar.core import fourier_transform_utils as _ftu

        from relax.relion.relion_projector_setup import reference_to_relion_projector_half_maps_and_power

        vol_ft = jnp.asarray(Iref_padded_fourier).reshape(volume_shape)
        vol_real = np.asarray(_ftu.get_idft3(vol_ft).real, dtype=np.float64)
        _, projector_power, _ = reference_to_relion_projector_half_maps_and_power(
            vol_real[None],
            current_size=-1 if current_size is None else current_size,
            padding_factor=int(padding_factor),
            shell_pair_counting=shell_pair_counting,
        )
        relion_power_spectrum = projector_power[0] / float(volume_shape[0]) ** 2

    # RELION stores ReferenceTau2 on the projector's shell-average scale:
    # getSpectrum(..., POWER_SPECTRUM) is normalized by the padded FFT volume
    # and then multiplied by the projector's normfft/2 factor. For the
    # padding/FFT convention RECOVAR uses here, that lands on an
    # ``ori_size^2 * padding_factor^3 / 8`` scale.
    norm_scale = float(volume_shape[0] ** 2 * (int(padding_factor) ** 3) / 8.0)
    tau2_shells = (np.asarray(relion_power_spectrum, dtype=np.float64) * norm_scale).astype(jnp.float32)
    radial_distances = (
        fourier_transform_utils.get_grid_of_radial_distances(
            volume_shape,
            scaled=False,
            frequency_shift=0,
        )
        .astype(int)
        .reshape(-1)
    )
    radial_distances = jnp.minimum(radial_distances, volume_shape[0] // 2)
    tau2 = tau2_shells[radial_distances]
    if not return_details:
        return tau2
    details = {
        "tau2_shells": tau2_shells,
        "shell_sum": tau2_shells.astype(jnp.float32),
        "shell_count": jnp.ones_like(tau2_shells, dtype=jnp.float32),
    }
    return tau2, details


def _centered_full_half_axis_mask(shape, axis, dtype):
    """1 on the stored half-complex axis of a centered full volume (RELION iterates only it)."""

    size = int(shape[axis])
    idx = jnp.arange(size)
    keep = idx >= size // 2
    if size % 2 == 0:
        keep = keep | (idx == 0)
    keep = keep.astype(dtype)
    mask_shape = [1] * len(shape)
    mask_shape[axis] = size
    return jnp.broadcast_to(keep.reshape(mask_shape), shape)


def _relion_x_half_multiplicity_in_native_half(volume_shape, axis):
    """How many of RELION's stored entries each entry of a native packed half stands for.

    RELION's shell loops visit every stored entry of its half, ``k_axis >= 0`` along ``axis``, once. A native
    packed half keeps the last axis' non-negative frequencies instead, and holds, for each of RELION's
    entries, either that entry or its Hermitian mate. Where the last-axis frequency has a distinct mate (not
    0, not an even axis' Nyquist), an entry stands for one of RELION's entries, or for two when its mate
    lies in RELION's half as well: the ``k_axis = 0`` plane, and an even axis' Nyquist plane. On the last
    axis' own-mate planes both mates are stored, and only those RELION stores count. The entries standing
    for two take the value of one of them for both, so the plane must be Hermitian, as RELION's
    ``enforceHermitianSymmetry`` leaves ``k_axis = 0``.

    Returns int8 counts (0, 1 or 2) that broadcast over the packed half of ``volume_shape``.
    """

    n_axis = int(volume_shape[axis])
    n_last = int(volume_shape[-1])
    coords = np.arange(-(n_axis // 2), n_axis - n_axis // 2, dtype=np.int64)
    nyquist = (coords == -(n_axis // 2)) if n_axis % 2 == 0 else np.zeros(n_axis, dtype=bool)
    stored = (coords >= 0) | nyquist
    own_mate_plane = (coords == 0) | nyquist
    last = np.arange(n_last // 2 + 1, dtype=np.int64)
    last_own_mate = (last == 0) | ((n_last % 2 == 0) & (last == n_last // 2))
    multiplicity = np.where(last_own_mate[None, :], stored[:, None], 1 + own_mate_plane[:, None]).astype(np.int8)
    shape = [1, 1, n_last // 2 + 1]
    shape[axis] = n_axis
    return multiplicity.reshape(shape)


# How a 3-D shell statistic counts the Hermitian pairs of a stored half volume: "relion" visits every
# stored entry once, which counts the pairs on the stored axis' zero plane twice (both mates are stored
# there); "once" counts every pair once. See docs/math/relion_consistency_options.md.
SHELL_PAIR_COUNTINGS = ("relion", "once")


def _require_shell_pair_counting(shell_pair_counting) -> bool:
    """Whether every Hermitian pair counts once; an unknown name is an error."""

    if shell_pair_counting not in SHELL_PAIR_COUNTINGS:
        raise ValueError(f"shell_pair_counting must be one of {SHELL_PAIR_COUNTINGS}, got {shell_pair_counting!r}")
    return shell_pair_counting == "once"


def _own_mate_planes(size, *, half_axis):
    """Indices along one Fourier axis of the planes that are their own Hermitian mirror.

    Frequency 0, and the Nyquist frequency of an even axis: ``size // 2`` and 0 on a centered full
    axis, 0 and ``size // 2`` on the non-negative half of an rfft axis (``half_axis``).
    """

    zero = 0 if half_axis else size // 2
    if size % 2:
        return (zero,)
    return (zero, size // 2 if half_axis else 0)


def _pair_once_weights(volume_shape, *, is_half_layout, full_half_axis):
    """Weight of each stored entry of a Fourier volume when every Hermitian pair counts once.

    A sum with these weights is half the sum over the whole Fourier grid, in every layout: an entry
    whose mate is not stored stands for its pair (weight 1); on a plane that is its own mirror both
    mates are stored and each takes 1/2, as does an entry that is its own mate (the origin). A full
    layout keeps RELION's stored half along ``full_half_axis`` (weight 0 elsewhere); a packed half
    is weighted on its own stored axis, the last, whatever ``full_half_axis`` says.

    Returns float64 weights that broadcast over the layout.
    """

    axis = 2 if is_half_layout else int(full_half_axis)
    size = int(volume_shape[axis])
    if is_half_layout:
        weights = np.ones(size // 2 + 1, dtype=np.float64)
    else:
        coords = np.arange(-(size // 2), size - size // 2)
        weights = ((coords >= 0) | ((size % 2 == 0) & (coords == -(size // 2)))).astype(np.float64)
    weights[list(_own_mate_planes(size, half_axis=is_half_layout))] = 0.5
    shape = [1, 1, 1]
    shape[axis] = weights.shape[0]
    return weights.reshape(shape)


@functools.partial(
    jax.jit,
    static_argnames=(
        "radial_volume_shape",
        "radial_shape",
        "is_half_layout",
        "full_half_axis",
        "padding_factor",
        "max_r_pad",
        "shell_rounding",
        "ori_half",
        "n_shells",
        "pair_once",
    ),
)
def _padded_shell_sums_device(
    weight,
    *,
    radial_volume_shape,
    radial_shape,
    is_half_layout,
    full_half_axis,
    padding_factor,
    max_r_pad,
    shell_rounding,
    ori_half,
    n_shells,
    pair_once=False,
):
    """The padded-grid shell sums of :func:`compute_relion_weight_shell_stats` on the device.

    One program per grid: eager, the radial grid, masks and bincounts were
    about thirty single-primitive programs compiled again at every new
    reconstruction size. The statements are the eager path's; the radial grid
    is sums of squared integers, exact in any association.
    """

    radial_fn = (
        fourier_transform_utils.get_grid_of_radial_distances_real
        if is_half_layout
        else fourier_transform_utils.get_grid_of_radial_distances
    )
    padded_dist = radial_fn(
        radial_volume_shape,
        scaled=False,
        frequency_shift=0,
        rounded=False,
    ).reshape(-1)
    if max_r_pad is None:
        radius_included = jnp.ones_like(padded_dist, dtype=jnp.float64)
    else:
        radius_included = (padded_dist * padded_dist < float(max_r_pad * max_r_pad)).astype(jnp.float64)
    rounded = shell_of_radius(
        padded_dist,
        rule=_SHELL_RULE[shell_rounding],
        padding_factor=padding_factor,
        index_dtype=jnp.int32,
        xp=jnp,
    )
    shell_index = jnp.minimum(rounded, ori_half)
    if pair_once:
        pair_weights = _pair_once_weights(
            radial_volume_shape, is_half_layout=is_half_layout, full_half_axis=full_half_axis
        )
        included = radius_included * jnp.broadcast_to(jnp.asarray(pair_weights), radial_shape).reshape(-1)
    elif is_half_layout:
        included = radius_included
        if full_half_axis != 2:
            # A native half repacked from RELION x-half storage: RELION's half is along full_half_axis.
            multiplicity = _relion_x_half_multiplicity_in_native_half(radial_volume_shape, full_half_axis)
            included = included * jnp.broadcast_to(
                jnp.asarray(multiplicity, dtype=jnp.float64),
                radial_shape,
            ).reshape(-1)
    else:
        # RELION iterates the stored half-complex axis only. For native RECOVAR
        # full volumes that axis is last; for full volumes expanded from RELION
        # x-half storage and transposed to public layout it is axis 0.
        included = _centered_full_half_axis_mask(radial_shape, full_half_axis, jnp.float64).reshape(-1) * radius_included
    shell_sum = jnp.bincount(shell_index, weights=weight * included, length=n_shells).astype(jnp.float64)
    shell_count = jnp.bincount(shell_index, weights=included, length=n_shells).astype(jnp.float64)
    return shell_sum, shell_count


@jax.jit
def _real_float64(weight):
    # One program per weight shape: eager, the real part and the cast were two.
    return weight.real.astype(jnp.float64)


def compute_relion_weight_shell_stats(
    weight,
    volume_shape,
    *,
    padding_factor=1,
    r_max=None,
    shell_rounding="round",
    full_half_axis=-1,
    accumulator_volume_shape=None,
    shell_pair_counting="relion",
):
    """Match RELION's shell-wise weight averaging for tau2 diagnostics.

    Parameters
    ----------
    weight : array-like
        Combined Fourier weight volume (typically ``(Ft_ctf_0 + Ft_ctf_1) / 2``).
        Accepts flat or grid-shaped centered-full arrays, or packed
        half-volume arrays on the same grid.
    volume_shape : tuple[int, int, int]
        Native reconstruction shape ``(N, N, N)``.
    padding_factor : int
        Fourier padding factor. When ``> 1``, ``weight`` must live on the
        padded grid ``(pf*N)^3``.
    r_max : float or None
        RELION reconstruction support radius in native Fourier pixels.  When
        provided, match ``BackProjector::updateSSNRarrays`` by averaging only
        padded voxels with ``r2 < ROUND(r_max * padding_factor)^2``.
    shell_rounding : {"round", "floor"}
        Shell binning rule. RELION's SSNR/tau2 update path uses ``round``
        while the Wiener reconstruct / current-size path uses ``floor``.
    full_half_axis : {-3, -2, -1, 0, 1, 2}
        Axis that corresponds to the RELION half-complex packed dimension.
        Native RECOVAR volumes use the last axis (default). Volumes expanded
        or repacked from RELION x-half storage and transposed into RECOVAR
        public layout use axis 0, as a full volume or as a packed half: a
        packed half then counts its entries as RELION's half along that axis
        does (:func:`_relion_x_half_multiplicity_in_native_half`).
    shell_pair_counting : {"relion", "once"}
        ``"relion"`` visits every stored entry once, as ``updateSSNRarrays``
        does, so the Hermitian pairs on the stored axis' zero plane count
        twice. ``"once"`` counts every pair once
        (:func:`_pair_once_weights`); sums and counts are then half the
        full-grid ones in every layout, and the plane must be Hermitian.

    Returns
    -------
    dict
        ``shell_sum``, ``shell_count``, and ``avg_weight_shells`` arrays with
        RELION-matching shell indexing.
    """
    volume_shape = tuple(int(s) for s in volume_shape)
    ori_half = volume_shape[0] // 2
    n_shells = ori_half + 1

    grid_shape = (
        tuple(d * padding_factor for d in volume_shape)
        if accumulator_volume_shape is None
        else tuple(int(s) for s in accumulator_volume_shape)
    )
    native_full_size = int(np.prod(volume_shape))
    half_grid_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(grid_shape)
    full_size = int(np.prod(grid_shape))
    half_size = int(np.prod(half_grid_shape))
    weight_size = int(np.size(weight))
    force_host_shell_stats = (
        padding_factor > 1 and weight_size in {full_size, half_size} and _shell_stats_on_host(weight_size)
    )
    weight_arr = np.asarray(weight).real if force_host_shell_stats else _real_float64(jnp.asarray(weight))
    native_layout = False
    if weight_size == full_size:
        is_half_layout = False
        # Keep shell-stat inputs in RECOVAR's centered grid convention so a
        # full volume and its packed-half view average the same voxels.
        # RELION backprojector FSC has its own axis conversion below.
        weight = weight_arr.reshape(-1)
        relion_grid_shape = grid_shape
    elif weight_size == half_size:
        is_half_layout = True
        weight = weight_arr.reshape(-1)
        relion_grid_shape = grid_shape
    elif padding_factor > 1 and weight_size == native_full_size:
        # Some callers still pass native-grid weights while requesting a
        # padded-shell scaling factor. Treat those as native full-layout input
        # and only apply the oversampling correction to the output.
        native_layout = True
        is_half_layout = False
        weight = weight_arr.reshape(-1)
        relion_grid_shape = volume_shape
    else:
        raise ValueError(
            f"Expected full or half Fourier weight with {full_size} or {half_size} voxels for "
            f"volume_shape={volume_shape} and padding_factor={padding_factor}, got {weight_size}"
        )

    if shell_rounding not in {"round", "floor"}:
        raise ValueError(f"shell_rounding must be 'round' or 'floor', got {shell_rounding!r}")
    if full_half_axis < 0:
        full_half_axis += 3
    if full_half_axis not in {0, 1, 2}:
        raise ValueError(f"full_half_axis must identify one Fourier axis, got {full_half_axis!r}")
    pair_once = _require_shell_pair_counting(shell_pair_counting)

    shell_sum_np = None
    shell_count_np = None

    def _centered_full_half_axis_mask_np(shape, axis):
        size = int(shape[axis])
        coords = np.arange(-(size // 2), size - size // 2, dtype=np.int64)
        keep = coords >= 0
        if size % 2 == 0:
            keep = keep | (coords == -(size // 2))
        mask_shape = [1] * len(shape)
        mask_shape[axis] = size
        return np.broadcast_to(keep.reshape(mask_shape), shape)

    def _numpy_bincount_shell_stats(labels, values, mask):
        """Per shell, the sum and the count of the masked ``values``: ``np.bincount`` over the flattened arrays.

        Accumulated in blocks of :data:`_SHELL_STATS_BLOCK` elements, so the int64 labels and the selected
        labels and values exist for one block at a time (three box-scale copies before: +20 GiB at EMPIAR-10202's
        padded box, relax#39). Each block's ``bincount`` starts from the running sums: they are its first
        weights, at labels ``0..n-1``, and ``bincount`` adds its weights in input order, so every shell's sum
        continues exactly where the previous block left it and equals the single call's, bit for bit.
        """

        labels_np = np.asarray(labels).reshape(-1)
        values_np = np.asarray(values).reshape(-1)
        mask_np = np.asarray(mask, dtype=bool).reshape(-1)
        n_bins = max(int(n_shells), int(labels_np.max()) + 1 if labels_np.size else 0)
        bins = np.arange(n_bins, dtype=np.int64)
        sums = np.zeros(n_bins, dtype=np.float64)
        counts = np.zeros(n_bins, dtype=np.int64)
        for start in range(0, labels_np.size, _SHELL_STATS_BLOCK):
            block = slice(start, start + _SHELL_STATS_BLOCK)
            included = mask_np[block]
            block_labels = labels_np[block][included].astype(np.int64, copy=False)
            block_values = values_np[block][included].astype(np.float64, copy=False)
            sums = np.bincount(
                np.concatenate([bins, block_labels]), weights=np.concatenate([sums, block_values]), minlength=n_bins
            )
            counts += np.bincount(block_labels, minlength=n_bins)
        return sums[:n_shells], counts.astype(np.float64)[:n_shells]

    if padding_factor > 1 and not native_layout:
        if is_half_layout:
            radial_shape = half_grid_shape
            radial_fn = fourier_transform_utils.get_grid_of_radial_distances_real
            radial_volume_shape = grid_shape
        else:
            radial_shape = relion_grid_shape
            radial_fn = fourier_transform_utils.get_grid_of_radial_distances
            radial_volume_shape = relion_grid_shape

        if force_host_shell_stats or int(np.prod(radial_shape)) > _RELION_SHELL_STATS_DEVICE_REDUCTION_MAX_VOXELS:
            coords = [
                np.arange(-(int(s) // 2), int(s) - int(s) // 2, dtype=np.float32)
                for s in radial_volume_shape[:-1]
            ]
            if is_half_layout:
                coords.append(np.arange(0, int(radial_volume_shape[-1]) // 2 + 1, dtype=np.float32))
            else:
                last_dim = int(radial_volume_shape[-1])
                coords.append(np.arange(-(last_dim // 2), last_dim - last_dim // 2, dtype=np.float32))
            radial_sq = np.zeros(tuple(radial_shape), dtype=np.float32)
            for axis, grid_axis in enumerate(coords):
                shape = [1] * len(radial_shape)
                shape[axis] = grid_axis.shape[0]
                radial_sq += grid_axis.reshape(shape) ** 2
            padded_dist_np = np.sqrt(radial_sq).astype(np.float32, copy=False)
            del radial_sq
            if r_max is None:
                radius_included_np = np.ones_like(padded_dist_np, dtype=bool)
            else:
                max_r_pad = relion_round(float(r_max) * padding_factor)
                radius_included_np = padded_dist_np * padded_dist_np < float(max_r_pad * max_r_pad)
            shell_index_np = shell_of_radius(
                padded_dist_np,
                rule=_SHELL_RULE[shell_rounding],
                padding_factor=padding_factor,
                index_dtype=np.int32,
            )
            shell_index_np = np.minimum(shell_index_np, ori_half)
            multiplicity_np = None
            if not is_half_layout:
                radius_included_np = radius_included_np & _centered_full_half_axis_mask_np(
                    radial_shape,
                    full_half_axis,
                )
            elif full_half_axis != 2 and not pair_once:
                multiplicity_np = _relion_x_half_multiplicity_in_native_half(radial_volume_shape, full_half_axis)
                radius_included_np &= multiplicity_np > 0
            weight_grid_np = np.asarray(weight).reshape(radial_shape)
            shell_sum_np, shell_count_np = _numpy_bincount_shell_stats(
                shell_index_np,
                weight_grid_np,
                radius_included_np,
            )
            if multiplicity_np is not None:
                # The entries that stand for two of RELION's lie on a few planes; count those once more.
                other_axes = tuple(a for a in range(3) if a != full_half_axis)
                for plane in np.flatnonzero(np.any(multiplicity_np == 2, axis=other_axes)):
                    plane_index = tuple(int(plane) if a == full_half_axis else slice(None) for a in range(3))
                    twice = np.broadcast_to(multiplicity_np == 2, radial_shape)[plane_index]
                    plane_sum, plane_count = _numpy_bincount_shell_stats(
                        shell_index_np[plane_index],
                        weight_grid_np[plane_index],
                        radius_included_np[plane_index] & twice,
                    )
                    shell_sum_np = shell_sum_np + plane_sum
                    shell_count_np = shell_count_np + plane_count
            if pair_once:
                # Both mates of the stored axis' own-mirror planes were counted; take half of each back.
                stored_axis = 2 if is_half_layout else full_half_axis
                for plane in _own_mate_planes(radial_volume_shape[stored_axis], half_axis=is_half_layout):
                    plane_index = tuple(int(plane) if a == stored_axis else slice(None) for a in range(3))
                    plane_sum, plane_count = _numpy_bincount_shell_stats(
                        shell_index_np[plane_index],
                        weight_grid_np[plane_index],
                        radius_included_np[plane_index],
                    )
                    shell_sum_np = shell_sum_np - 0.5 * plane_sum
                    shell_count_np = shell_count_np - 0.5 * plane_count
        else:
            shell_sum, shell_count = _padded_shell_sums_device(
                weight,
                radial_volume_shape=tuple(int(v) for v in radial_volume_shape),
                radial_shape=tuple(int(v) for v in radial_shape),
                is_half_layout=bool(is_half_layout),
                full_half_axis=full_half_axis,
                padding_factor=int(padding_factor),
                max_r_pad=(
                    None
                    if r_max is None
                    else relion_round(float(r_max) * padding_factor)
                ),
                shell_rounding=shell_rounding,
                ori_half=int(ori_half),
                n_shells=int(n_shells),
                pair_once=pair_once,
            )
            avg_weight = jnp.where(shell_count > 0, shell_sum / shell_count, 0.0)
            return {
                "shell_sum": shell_sum,
                "shell_count": shell_count,
                "avg_weight_shells": avg_weight,
            }
    else:
        radial_fn = (
            fourier_transform_utils.get_grid_of_radial_distances_real
            if is_half_layout
            else fourier_transform_utils.get_grid_of_radial_distances
        )
        radial_raw = radial_fn(
            volume_shape,
            scaled=False,
            frequency_shift=0,
            rounded=False,
        ).reshape(-1)
        shell_index = shell_of_radius(radial_raw, rule=_SHELL_RULE[shell_rounding], index_dtype=jnp.int32, xp=jnp)
        shell_index = jnp.minimum(shell_index, ori_half)
        if r_max is None:
            included = jnp.ones(weight.shape[0], dtype=jnp.float64)
        else:
            max_r_native = relion_round(float(r_max))
            included = (radial_raw * radial_raw < float(max_r_native * max_r_native)).astype(jnp.float64)
        if pair_once:
            pair_weights = _pair_once_weights(volume_shape, is_half_layout=is_half_layout, full_half_axis=full_half_axis)
            included = included * jnp.broadcast_to(
                jnp.asarray(pair_weights),
                fourier_transform_utils.volume_shape_to_half_volume_shape(volume_shape) if is_half_layout else volume_shape,
            ).reshape(-1)
        elif not is_half_layout:
            included = included * _centered_full_half_axis_mask(
                relion_grid_shape,
                full_half_axis,
                jnp.float64,
            ).reshape(-1)
        elif full_half_axis != 2:
            multiplicity = _relion_x_half_multiplicity_in_native_half(volume_shape, full_half_axis)
            included = included * jnp.broadcast_to(
                jnp.asarray(multiplicity, dtype=jnp.float64),
                fourier_transform_utils.volume_shape_to_half_volume_shape(volume_shape),
            ).reshape(-1)

    if shell_sum_np is None:
        shell_sum = jnp.bincount(shell_index, weights=weight * included, length=n_shells).astype(jnp.float64)
        shell_count = jnp.bincount(shell_index, weights=included, length=n_shells).astype(jnp.float64)
    else:
        shell_sum = jnp.asarray(shell_sum_np, dtype=jnp.float64)
        shell_count = jnp.asarray(shell_count_np, dtype=jnp.float64)
    avg_weight = jnp.where(shell_count > 0, shell_sum / shell_count, 0.0)
    return {
        "shell_sum": shell_sum,
        "shell_count": shell_count,
        "avg_weight_shells": avg_weight,
    }


def compute_relion_tau2_from_weights(
    Ft_ctf_0,
    Ft_ctf_1,
    fsc,
    volume_shape,
    *,
    tau2_fudge=1.0,
    padding_factor=1,
    r_max=None,
    is_whole_instead_of_half=False,
    return_details=False,
    full_half_axis=-1,
    accumulator_volume_shape=None,
    weight_combination="average",
    output_dtype=jnp.float32,
    shell_pair_counting="relion",
):
    """Compute tau2 from CTF weights and external FSC (RELION's updateSSNRarrays).

    RELION computes tau2 = SSNR * sigma2 where:
    - SSNR = fsc / (1 - fsc) * tau2_fudge
    - sigma2 = count_per_shell / (pf³ * sum_weight_per_shell)
      which is the inverse of the average weight per shell

    When padding_factor > 1, Ft_ctf arrays are at (pf*N)³ or the packed
    half-volume equivalent, while volume_shape is the native (N,N,N).
    Shell averages are computed at native resolution (clamping padded radial
    indices to ori_size/2), matching RELION's updateSSNRarrays which uses
    ``ires = MIN(ires, ori_size/2)``. Output tau2 is at native N³ resolution.
    r_max : float or None
        Reconstruction support radius. RELION ignores padded weights with
        ``r2 >= ROUND(r_max * padding_factor)^2`` in
        ``BackProjector::updateSSNRarrays``; pass the current iteration
        ``current_size // 2`` for auto-refine parity.
    weight_combination : {"average", "sum"}
        How to combine the two input weight arrays before shell averaging.
        Numbered split-half iterations pass one half twice, so the default
        average preserves legacy behavior. RELION's final joined all-data
        iteration first adds the two half BackProjectors, then calls
        ``updateSSNRarrays`` on the combined BackProjector; pass ``"sum"`` for
        that path.
    output_dtype : dtype
        RELION ``RFLOAT`` precision for the weight combination, FSC, and
        returned tau2 arrays. Defaults to float32 for compatibility; the
        dense double-precision refinement path passes float64.
    shell_pair_counting : {"relion", "once"}
        How the shell average of the weight counts Hermitian pairs
        (:func:`compute_relion_weight_shell_stats`).
    """
    prior_dtype = jnp.dtype(output_dtype)
    if prior_dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
        raise ValueError(f"output_dtype must be float32 or float64, got {output_dtype}")
    host_prior_dtype = np.dtype(prior_dtype)
    if weight_combination not in {"average", "sum"}:
        raise ValueError(
            "weight_combination must be 'average' or 'sum', "
            f"got {weight_combination!r}"
        )

    padded_shape = (
        tuple(int(s) * int(padding_factor) for s in volume_shape)
        if accumulator_volume_shape is None
        else tuple(int(s) for s in accumulator_volume_shape)
    )
    half_padded_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(padded_shape)
    large_weight_size = max(int(np.prod(padded_shape)), int(np.prod(half_padded_shape)))
    padded_sizes = {int(np.prod(padded_shape)), int(np.prod(half_padded_shape))}
    use_host_shell_stats = (
        int(padding_factor) > 1
        and max(int(np.size(Ft_ctf_0)), int(np.size(Ft_ctf_1))) in padded_sizes
        and _shell_stats_on_host(large_weight_size)
    )
    if use_host_shell_stats:
        # One ufunc casts each element and adds in host_prior_dtype, as converting both
        # halves first would, without the two converted copies (2 x 16.5 GB of host
        # memory at EMPIAR-10202's box 800, float32 weights to float64).
        H_comb = np.add(np.asarray(Ft_ctf_0).real, np.asarray(Ft_ctf_1).real, dtype=host_prior_dtype)
        if weight_combination == "average":
            H_comb *= host_prior_dtype.type(0.5)
    else:
        H0 = jnp.asarray(Ft_ctf_0).real.astype(prior_dtype)
        H1 = jnp.asarray(Ft_ctf_1).real.astype(prior_dtype)
        H_comb = H0 + H1
        if weight_combination == "average":
            H_comb = H_comb / jnp.asarray(2.0, dtype=prior_dtype)
    shell_stats = compute_relion_weight_shell_stats(
        H_comb,
        volume_shape,
        padding_factor=padding_factor,
        r_max=r_max,
        full_half_axis=full_half_axis,
        accumulator_volume_shape=accumulator_volume_shape,
        shell_pair_counting=shell_pair_counting,
    )
    shell_sum = shell_stats["shell_sum"]
    shell_count = shell_stats["shell_count"]
    bottom_avg = shell_stats["avg_weight_shells"]

    n_shells = bottom_avg.shape[0]

    # Compute SSNR in float64 to avoid catastrophic cancellation in
    # 1 - fsc when fsc is clamped near 0.999 (float32 loses ~3 digits).
    fsc_raw = jnp.asarray(fsc, dtype=jnp.float64)
    fsc_indices = jnp.minimum(jnp.arange(n_shells), fsc_raw.shape[0] - 1)
    fsc_arr = fsc_raw[fsc_indices]
    epsilon = jax_config.FSC_ZERO_THRESHOLD
    fsc_clamped = jnp.maximum(fsc_arr, epsilon)
    if is_whole_instead_of_half:
        fsc_clamped = jnp.sqrt(2.0 * fsc_clamped / (fsc_clamped + 1.0))
    fsc_clamped = jnp.minimum(fsc_clamped, 1.0 - epsilon)
    ssnr = fsc_clamped / (1.0 - fsc_clamped) * tau2_fudge

    # RELION backprojector.cpp:1061,1075 — updateSSNRarrays multiplies each
    # weight by oversampling_correction = pf³ before shell-averaging, because
    # padding dilutes the per-voxel weight by that factor.  Match here.
    oversampling_correction = padding_factor**3
    sigma2_shells = jnp.where(bottom_avg > 0, 1.0 / (oversampling_correction * bottom_avg), 0.0)
    prior_avg = jnp.where(bottom_avg > 0, ssnr * sigma2_shells, jax_config.EPSILON)
    prior_avg = prior_avg.astype(prior_dtype)

    radial_distances = (
        fourier_transform_utils.get_grid_of_radial_distances(volume_shape, scaled=False, frequency_shift=0)
        .astype(int)
        .reshape(-1)
    )
    prior = prior_avg[radial_distances]
    if not return_details:
        return prior, fsc_clamped

    details = {
        "prior_shells": prior_avg,
        "sigma2_shells": sigma2_shells.astype(prior_dtype),
        "avg_weight_shells": bottom_avg.astype(prior_dtype),
        "shell_sum": shell_sum,
        "shell_count": shell_count,
        "fsc_shells": fsc_clamped,
        "ssnr_shells": ssnr.astype(prior_dtype),
        "oversampling_correction": jnp.asarray(oversampling_correction, dtype=prior_dtype),
        "is_whole_instead_of_half": jnp.asarray(bool(is_whole_instead_of_half)),
        "weight_combination": np.asarray(str(weight_combination)),
    }
    return prior, fsc_clamped, details


# Packed-half accumulators of every size take the streamed reduction: it visits
# the coefficients in the expanded path's order (tests assert the two FSCs equal)
# and is faster at every measured size, because it never expands four padded
# full cubes on the host: 0.12 vs 0.27 s at a 185^3 accumulator, 0.29 vs 0.64 s
# at 259^3 and 2.2 vs 9.8 s at 515^3, the 10097 final all-data pass (login node,
# 8 threads). The environment variable keeps the expanded path selectable.
_RELION_FSC_PACKED_STREAM_MIN_ELEMENTS = 0


def _relion_fsc_packed_stream_enabled(half_size):
    threshold = int(
        os.environ.get(
            "RELAX_RELION_FSC_PACKED_STREAM_MIN_ELEMENTS",
            _RELION_FSC_PACKED_STREAM_MIN_ELEMENTS,
        )
    )
    return int(half_size) >= threshold


def _compute_relion_fsc_from_packed_half_streamed(
    Ft_y_0,
    Ft_y_1,
    Ft_ctf_0,
    Ft_ctf_1,
    volume_shape,
    padded_shape,
    *,
    padding_factor,
    r_max,
    pair_once=False,
):
    """Reduce native packed-half BPref arrays without expanding full cubes.

    The public packed layout stores the last Fourier axis as an RFFT half,
    while RELION's logical compact x axis corresponds to public axis zero.
    The legacy implementation first rebuilt four full padded cubes and then
    built six more full coordinate/rounded-coordinate cubes.  At box 800
    that path requires hundreds of GiB of host memory.

    This implementation visits source coefficients in the same logical
    ``(relion_z, relion_y, relion_x)`` order as the expanded implementation.
    Source z planes that round to one native z coordinate are reduced
    together, so every downsampled Fourier cell is completed in exactly one
    bounded slab.  The final valid native cells are retained in canonical
    z/y/x order and passed to one shell ``bincount`` per statistic; this
    avoids changing the last-stage floating-point reduction topology.
    ``pair_once`` weights the cells of RELION's ``x = 0`` plane, where both
    Hermitian mates are stored, by 1/2 in the three shell sums.
    """

    volume_shape = tuple(int(value) for value in volume_shape)
    padded_shape = tuple(int(value) for value in padded_shape)
    half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(padded_shape)
    data0_half = np.asarray(Ft_y_0).reshape(half_shape)
    data1_half = np.asarray(Ft_y_1).reshape(half_shape)
    weight0_half = np.asarray(Ft_ctf_0).reshape(half_shape)
    weight1_half = np.asarray(Ft_ctf_1).reshape(half_shape)

    n = int(volume_shape[0])
    pf = int(padding_factor)
    half = n // 2
    max_shell = half if r_max is None else int(r_max)
    down_radius = max_shell + 1
    down_size = 2 * down_radius + 1
    down_xsize = down_size // 2 + 1
    shell_count = half + 1

    axes = [
        np.asarray(
            fourier_transform_utils.get_1d_frequency_grid(size, scaled=False),
            dtype=np.float64,
        )
        for size in padded_shape
    ]
    rounded_relion_z = relion_round_array(axes[1] / pf).astype(np.int64)
    rounded_relion_y = relion_round_array(axes[2] / pf).astype(np.int64)
    rounded_relion_x = relion_round_array(axes[0] / pf).astype(np.int64)
    source_z_indices = np.flatnonzero(
        (rounded_relion_z >= -down_radius) & (rounded_relion_z <= down_radius)
    ).astype(np.intp, copy=False)
    source_y_indices = np.flatnonzero(
        (rounded_relion_y >= -down_radius) & (rounded_relion_y <= down_radius)
    ).astype(np.intp, copy=False)
    source_x_indices = np.flatnonzero(
        (rounded_relion_x >= 0) & (rounded_relion_x < down_xsize)
    ).astype(np.intp, copy=False)

    n0, n1, n2 = padded_shape
    ic2 = n2 // 2
    if n2 % 2 == 0:
        packed_indices = np.concatenate(
            [
                np.arange(ic2, n2, dtype=np.intp),
                np.asarray([0], dtype=np.intp),
            ]
        )
        redundant_indices = np.arange(1, ic2, dtype=np.intp)
    else:
        packed_indices = np.arange(ic2, n2, dtype=np.intp)
        redundant_indices = np.arange(0, ic2, dtype=np.intp)

    full_to_half = np.full(n2, -1, dtype=np.intp)
    full_to_half[packed_indices] = np.arange(packed_indices.size, dtype=np.intp)
    full_to_half[redundant_indices] = ic2 - redundant_indices
    if np.any(full_to_half < 0):
        raise RuntimeError(
            f"Could not map packed half axis for padded shape {padded_shape}"
        )
    is_redundant = np.zeros(n2, dtype=bool)
    is_redundant[redundant_indices] = True
    partner_i0 = (
        (n0 - (n0 % 2) - np.arange(n0, dtype=np.intp)) % n0
    ).astype(np.intp, copy=False)
    partner_i1 = (
        (n1 - (n1 % 2) - np.arange(n1, dtype=np.intp)) % n1
    ).astype(np.intp, copy=False)

    rounded_y_selected = rounded_relion_y[source_y_indices]
    rounded_x_selected = rounded_relion_x[source_x_indices]
    local_labels_one_z = (
        (rounded_y_selected[:, None] + down_radius) * down_xsize
        + rounded_x_selected[None, :]
    ).reshape(-1)
    local_size = down_size * down_xsize

    target_y = np.arange(-down_radius, down_radius + 1, dtype=np.float64)
    target_x = np.arange(0, down_xsize, dtype=np.float64)
    target_radius_sq_yx = target_y[:, None] ** 2 + target_x[None, :] ** 2
    valid_count_by_z = np.asarray(
        [
            np.count_nonzero(target_radius_sq_yx + float(z * z) <= float(max_shell * max_shell))
            for z in range(-down_radius, down_radius + 1)
        ],
        dtype=np.int64,
    )
    valid_count = int(np.sum(valid_count_by_z, dtype=np.int64))
    avg0_valid = np.empty(valid_count, dtype=np.complex128)
    avg1_valid = np.empty(valid_count, dtype=np.complex128)
    shell_labels = np.empty(valid_count, dtype=np.int64)
    pair_weights = np.empty(valid_count, dtype=np.float64) if pair_once else None

    def _gather_full_slab(half_grid, z_indices, *, conjugate):
        slab = np.empty(
            (z_indices.size, source_y_indices.size, source_x_indices.size),
            dtype=half_grid.dtype,
        )
        direct_positions = np.flatnonzero(~is_redundant[source_y_indices])
        if direct_positions.size:
            source_half_indices = full_to_half[source_y_indices[direct_positions]]
            direct = half_grid[
                np.ix_(source_x_indices, z_indices, source_half_indices)
            ].transpose(1, 2, 0)
            slab[:, direct_positions, :] = direct
        redundant_positions = np.flatnonzero(is_redundant[source_y_indices])
        if redundant_positions.size:
            source_half_indices = full_to_half[source_y_indices[redundant_positions]]
            mirrored = half_grid[
                np.ix_(
                    partner_i0[source_x_indices],
                    partner_i1[z_indices],
                    source_half_indices,
                )
            ].transpose(1, 2, 0)
            if conjugate:
                mirrored = np.conj(mirrored)
            slab[:, redundant_positions, :] = mirrored
        return slab.reshape(-1)

    def _downsample_one_half(data_half, weight_half, z_indices, labels):
        data_values = _gather_full_slab(data_half, z_indices, conjugate=True)
        weight_values = _gather_full_slab(weight_half, z_indices, conjugate=True).real
        sum_weight = np.bincount(labels, weights=weight_values, minlength=local_size)
        sum_real = np.bincount(labels, weights=data_values.real, minlength=local_size)
        sum_imag = np.bincount(labels, weights=data_values.imag, minlength=local_size)
        average = sum_real + 1j * sum_imag
        nonzero = sum_weight > 0.0
        average[nonzero] /= sum_weight[nonzero]
        average[~nonzero] = 0.0
        return average.reshape((down_size, down_xsize))

    cursor = 0
    for offset, target_z in enumerate(range(-down_radius, down_radius + 1)):
        count = int(valid_count_by_z[offset])
        if count == 0:
            continue
        z_indices = source_z_indices[rounded_relion_z[source_z_indices] == target_z]
        target_valid = target_radius_sq_yx + float(target_z * target_z) <= float(max_shell * max_shell)
        target_shells = relion_round_array(
            np.sqrt(target_radius_sq_yx + float(target_z * target_z))
        ).astype(np.int64)[target_valid]
        if z_indices.size:
            labels = np.tile(local_labels_one_z, z_indices.size)
            avg0 = _downsample_one_half(data0_half, weight0_half, z_indices, labels)
            avg1 = _downsample_one_half(data1_half, weight1_half, z_indices, labels)
            avg0_valid[cursor : cursor + count] = avg0[target_valid]
            avg1_valid[cursor : cursor + count] = avg1[target_valid]
        else:
            avg0_valid[cursor : cursor + count] = 0.0
            avg1_valid[cursor : cursor + count] = 0.0
        shell_labels[cursor : cursor + count] = target_shells
        if pair_once:
            pair_weights[cursor : cursor + count] = np.broadcast_to(
                np.where(target_x == 0.0, 0.5, 1.0)[None, :], target_valid.shape
            )[target_valid]
        cursor += count
    if cursor != valid_count:
        raise RuntimeError(
            f"Streamed RELION FSC filled {cursor} valid cells; expected {valid_count}"
        )

    cross_terms = (np.conj(avg0_valid) * avg1_valid).real
    power0_terms = np.abs(avg0_valid) ** 2
    power1_terms = np.abs(avg1_valid) ** 2
    if pair_once:
        cross_terms, power0_terms, power1_terms = (
            terms * pair_weights for terms in (cross_terms, power0_terms, power1_terms)
        )
    numerator = np.bincount(
        shell_labels,
        weights=cross_terms,
        minlength=shell_count,
    )
    denom0 = np.bincount(
        shell_labels,
        weights=power0_terms,
        minlength=shell_count,
    )
    denom1 = np.bincount(
        shell_labels,
        weights=power1_terms,
        minlength=shell_count,
    )
    fsc = np.zeros(shell_count, dtype=np.float64)
    nonzero = (denom0 * denom1) > 0.0
    fsc[nonzero] = numerator[nonzero] / np.sqrt(denom0[nonzero] * denom1[nonzero])
    fsc[0] = 1.0
    return fsc, numerator, denom0, denom1


@functools.lru_cache(maxsize=4)
def _relion_fsc_downsample_labels(padded_shape, pf, max_shell):
    """``BackProjector::getDownsampledAverage``'s native-grid bins for one accumulator shape.

    They depend only on the shape, the padding factor and the FSC radius, and
    computing them (three padded-grid meshgrids and RELION roundings) took
    about 0.3 s per call, twice an iteration on the 5k K=1 run. Returns the
    padded-grid mask of voxels that land on the downsampled grid, their
    flattened downsampled labels, and the downsampled grid's radius and sizes.
    Read-only: the arrays are shared between calls.
    """

    axes = [
        np.asarray(fourier_transform_utils.get_1d_frequency_grid(s, scaled=False), dtype=np.float64)
        for s in padded_shape
    ]
    # RECOVAR stores centered Fourier volumes as (z, y, x), but the RELION
    # BackProjector's compact half-axis is its logical x coordinate.  In the
    # dense EM rotation convention that logical RELION x corresponds to
    # RECOVAR axis 0 after the CUDA rotation-row swap.  Interpret the saved
    # full accumulator as (relion_y, relion_x, relion_z) before mirroring
    # BackProjector::getDownsampledAverage.  This is source-level layout
    # emulation; the shell bins are rotationally invariant, but the compact
    # half-axis selection is not.
    relion_z, relion_y, relion_x = np.meshgrid(axes[1], axes[2], axes[0], indexing="ij")
    dz = relion_round_array(relion_z / pf).astype(np.int64)
    dy = relion_round_array(relion_y / pf).astype(np.int64)
    dx = relion_round_array(relion_x / pf).astype(np.int64)

    down_radius = max_shell + 1
    down_size = 2 * down_radius + 1
    down_xsize = down_size // 2 + 1
    valid = (
        (dz >= -down_radius)
        & (dz <= down_radius)
        & (dy >= -down_radius)
        & (dy <= down_radius)
        & (dx >= 0)
        & (dx < down_xsize)
    )
    labels = ((dz[valid] + down_radius) * down_size + (dy[valid] + down_radius)) * down_xsize + dx[valid]
    labels = labels.reshape(-1)
    valid.setflags(write=False)
    labels.setflags(write=False)
    return valid, labels, down_radius, down_size, down_xsize


def _packed_half_of_hermitian_full(values, padded_shape):
    """The packed-half layout of a Hermitian full centered volume (host copy).

    The inverse of the expansion in :func:`compute_relion_fsc_from_backprojector`:
    the last axis keeps its centered non-negative frequencies, with the Nyquist
    column of an even axis last, as ``half_volume_to_full_volume`` lays it out.
    """

    full = np.asarray(values).reshape(padded_shape)
    n2 = int(padded_shape[2])
    ic2 = n2 // 2
    if n2 % 2 == 0:
        packed_idx = np.concatenate([np.arange(ic2, n2, dtype=np.int64), np.asarray([0], dtype=np.int64)])
    else:
        packed_idx = np.arange(ic2, n2, dtype=np.int64)
    return np.ascontiguousarray(full[:, :, packed_idx]).reshape(-1)


def compute_relion_fsc_from_backprojector(
    Ft_y_0,
    Ft_y_1,
    Ft_ctf_0,
    Ft_ctf_1,
    volume_shape,
    *,
    padding_factor=1,
    r_max=None,
    accumulator_volume_shape=None,
    output_dtype=jnp.float32,
    full_is_hermitian=False,
    shell_pair_counting="relion",
):
    """Compute RELION's gold-standard FSC from backprojector accumulators.

    RELION does not compute the auto-refine FSC used by ``updateSSNRarrays``
    from reconstructed unregularized maps. In
    ``BackProjector::getDownsampledAverage`` it rounds each padded
    backprojector Fourier voxel onto the native grid, divides accumulated
    complex data by accumulated weight, and then
    ``calculateDownSampledFourierShellCorrelation`` bins the two half averages
    with ``ROUND(R)``. RELION's ``ROUND`` is round-half-away-from-zero, not
    NumPy's banker rounding. This helper mirrors that path for centered full
    Fourier arrays used by RECOVAR's dense single-volume M-step. ``output_dtype``
    selects the RELION ``RFLOAT`` precision of the returned FSC.

    ``full_is_hermitian`` declares full-layout inputs to be exact Hermitian
    expansions of packed halves (the public layout of the x-half
    BackProjectors). Their packed half then carries every value the full path
    reads, so the streamed packed-half reduction runs on it instead of the
    full cubes: 2.2 s instead of 6.4 s at a 515^3 accumulator, equal FSC.

    ``shell_pair_counting`` ``"relion"`` sums every stored cell of the
    downsampled half, as ``calculateDownSampledFourierShellCorrelation`` does,
    so the Hermitian pairs of its ``x = 0`` plane count twice; ``"once"``
    weights that plane by 1/2 in the numerator and both denominators.
    """

    pair_once = _require_shell_pair_counting(shell_pair_counting)

    volume_shape = tuple(int(s) for s in volume_shape)
    if len(volume_shape) != 3 or len(set(volume_shape)) != 1:
        raise ValueError(f"Expected cubic 3-D volume_shape, got {volume_shape}")
    n = volume_shape[0]
    pf = int(padding_factor)
    if pf <= 0:
        raise ValueError(f"padding_factor must be positive, got {padding_factor}")
    padded_shape = (
        tuple(d * pf for d in volume_shape)
        if accumulator_volume_shape is None
        else tuple(int(s) for s in accumulator_volume_shape)
    )
    full_size = int(np.prod(padded_shape))
    half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(padded_shape)
    half_size = int(np.prod(half_shape))

    fsc_dtype = jnp.dtype(output_dtype)
    if fsc_dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
        raise ValueError(f"output_dtype must be float32 or float64, got {output_dtype}")
    input_sizes = tuple(
        int(np.size(value)) for value in (Ft_y_0, Ft_y_1, Ft_ctf_0, Ft_ctf_1)
    )
    if full_is_hermitian and half_size < full_size and input_sizes == (full_size,) * 4:
        Ft_y_0, Ft_y_1, Ft_ctf_0, Ft_ctf_1 = (
            _packed_half_of_hermitian_full(value, padded_shape)
            for value in (Ft_y_0, Ft_y_1, Ft_ctf_0, Ft_ctf_1)
        )
        input_sizes = (half_size,) * 4
    dump_dir = os.environ.get("RELAX_MSTEP_FSC_DUMP_DIR")
    dump_avg = bool(dump_dir) and os.environ.get(
        "RELAX_MSTEP_FSC_DUMP_AVG", ""
    ).lower() in {"1", "true", "yes", "on"}
    packed_stream_eligible = (
        half_size < full_size
        and input_sizes == (half_size, half_size, half_size, half_size)
        and _relion_fsc_packed_stream_enabled(half_size)
    )
    if packed_stream_eligible and dump_avg:
        logger.warning(
            "RELION backprojector average-grid dump requested; using the legacy "
            "full-expansion FSC diagnostic path"
        )
    if packed_stream_eligible and not dump_avg:
        logger.info(
            "RELION backprojector FSC using streamed packed-half reduction: "
            "shape=%s half_elements=%d",
            padded_shape,
            half_size,
        )
        fsc, numerator, denom0, denom1 = _compute_relion_fsc_from_packed_half_streamed(
            Ft_y_0,
            Ft_y_1,
            Ft_ctf_0,
            Ft_ctf_1,
            volume_shape,
            padded_shape,
            padding_factor=pf,
            r_max=r_max,
            pair_once=pair_once,
        )
        if dump_dir:
            pathlib.Path(dump_dir).mkdir(parents=True, exist_ok=True)
            tag = os.environ.get("RELAX_MSTEP_FSC_DUMP_TAG", "recovar")
            np.savetxt(
                pathlib.Path(dump_dir) / f"{tag}_downsampled_fsc.txt",
                np.column_stack(
                    [np.arange(fsc.size), numerator, denom0, denom1, fsc]
                ),
                header="shell num den1 den2 fsc",
            )
        return jnp.asarray(fsc, dtype=fsc_dtype)

    def _packed_half_to_full_numpy(arr_np):
        """Expand RECOVAR's centered packed half-volume layout on host."""
        half_grid = np.asarray(arr_np).reshape(half_shape)
        n0, n1, n2 = padded_shape
        ic2 = n2 // 2
        if n2 % 2 == 0:
            packed_idx = np.concatenate([np.arange(ic2, n2, dtype=np.int64), np.asarray([0], dtype=np.int64)])
            redundant = np.arange(1, ic2, dtype=np.int64)
        else:
            packed_idx = np.arange(ic2, n2, dtype=np.int64)
            redundant = np.arange(0, ic2, dtype=np.int64)

        full_grid = np.zeros(padded_shape, dtype=half_grid.dtype)
        full_grid[:, :, packed_idx] = half_grid
        if redundant.size:
            partner_i0 = (n0 - (n0 % 2) - np.arange(n0, dtype=np.int64)) % n0
            partner_i1 = (n1 - (n1 % 2) - np.arange(n1, dtype=np.int64)) % n1
            conj_partner = np.conj(half_grid[partner_i0[:, None], partner_i1[None, :], :])
            source_cols = ic2 - redundant
            full_grid[:, :, redundant] = conj_partner[:, :, source_cols]
        return full_grid

    def _as_padded_full(arr, name):
        arr_np = np.asarray(arr)
        if arr_np.size == full_size:
            return arr_np.reshape(padded_shape)
        if arr_np.size == half_size:
            return _packed_half_to_full_numpy(arr_np)
        raise ValueError(
            f"{name} must be centered full Fourier data with {full_size} entries or packed half "
            f"Fourier data with {half_size} entries for volume_shape={volume_shape}, "
            f"padding_factor={padding_factor}; got {arr_np.size}"
        )

    data0 = _as_padded_full(Ft_y_0, "Ft_y_0")
    data1 = _as_padded_full(Ft_y_1, "Ft_y_1")
    weight0 = _as_padded_full(Ft_ctf_0, "Ft_ctf_0").real
    weight1 = _as_padded_full(Ft_ctf_1, "Ft_ctf_1").real

    data0 = np.transpose(data0, (1, 2, 0))
    data1 = np.transpose(data1, (1, 2, 0))
    weight0 = np.transpose(weight0, (1, 2, 0))
    weight1 = np.transpose(weight1, (1, 2, 0))

    half = n // 2
    max_shell = half if r_max is None else int(r_max)
    valid, labels, down_radius, down_size, down_xsize = _relion_fsc_downsample_labels(
        padded_shape, pf, max_shell
    )
    minlength = down_size * down_size * down_xsize

    def _downsample_average(data, weight):
        weight_flat = weight[valid].reshape(-1)
        data_flat = data[valid].reshape(-1)
        sum_weight = np.bincount(labels, weights=weight_flat, minlength=minlength)
        sum_real = np.bincount(labels, weights=data_flat.real, minlength=minlength)
        sum_imag = np.bincount(labels, weights=data_flat.imag, minlength=minlength)
        avg = sum_real + 1j * sum_imag
        nonzero = sum_weight > 0.0
        avg[nonzero] /= sum_weight[nonzero]
        avg[~nonzero] = 0.0
        return avg.reshape((down_size, down_size, down_xsize)), sum_weight.reshape((down_size, down_size, down_xsize))

    avg0, down_weight0 = _downsample_average(data0, weight0)
    avg1, down_weight1 = _downsample_average(data1, weight1)

    z_axis = np.arange(-down_radius, down_radius + 1, dtype=np.float64)
    y_axis = np.arange(-down_radius, down_radius + 1, dtype=np.float64)
    x_axis = np.arange(0, down_xsize, dtype=np.float64)
    rz, ry, rx = np.meshgrid(z_axis, y_axis, x_axis, indexing="ij")
    radius = np.sqrt(rz * rz + ry * ry + rx * rx)
    shell = shell_of_radius(radius, rule="half_up")
    shell_count = half + 1
    # RELION's calculateDownSampledFourierShellCorrelation bins by ROUND(R),
    # but first skips samples with exact native radius R > r_max.
    shell_valid = radius <= float(max_shell)
    shell_labels = shell[shell_valid].reshape(-1)
    avg0_flat = avg0[shell_valid].reshape(-1)
    avg1_flat = avg1[shell_valid].reshape(-1)

    cross_terms = (np.conj(avg0_flat) * avg1_flat).real
    power0_terms = np.abs(avg0_flat) ** 2
    power1_terms = np.abs(avg1_flat) ** 2
    if pair_once:
        pair_weights = np.where(rx[shell_valid].reshape(-1) == 0.0, 0.5, 1.0)
        cross_terms, power0_terms, power1_terms = (
            terms * pair_weights for terms in (cross_terms, power0_terms, power1_terms)
        )
    numerator = np.bincount(shell_labels, weights=cross_terms, minlength=shell_count)
    denom0 = np.bincount(shell_labels, weights=power0_terms, minlength=shell_count)
    denom1 = np.bincount(shell_labels, weights=power1_terms, minlength=shell_count)
    fsc = np.zeros(shell_count, dtype=np.float64)
    nonzero = (denom0 * denom1) > 0.0
    fsc[nonzero] = numerator[nonzero] / np.sqrt(denom0[nonzero] * denom1[nonzero])
    fsc[0] = 1.0

    dump_dir = os.environ.get("RELAX_MSTEP_FSC_DUMP_DIR")
    if dump_dir:
        pathlib.Path(dump_dir).mkdir(parents=True, exist_ok=True)
        tag = os.environ.get("RELAX_MSTEP_FSC_DUMP_TAG", "recovar")
        np.savetxt(
            pathlib.Path(dump_dir) / f"{tag}_downsampled_fsc.txt",
            np.column_stack([np.arange(shell_count), numerator, denom0, denom1, fsc]),
            header="shell num den1 den2 fsc",
        )
        if os.environ.get("RELAX_MSTEP_FSC_DUMP_AVG", "").lower() in {"1", "true", "yes", "on"}:
            coords = np.column_stack(
                [
                    rz[shell_valid].reshape(-1).astype(np.int64),
                    ry[shell_valid].reshape(-1).astype(np.int64),
                    rx[shell_valid].reshape(-1).astype(np.int64),
                ]
            )
            for suffix, avg, down_weight in (
                ("half1", avg0, down_weight0),
                ("half2", avg1, down_weight1),
            ):
                avg_flat = avg[shell_valid].reshape(-1)
                np.savetxt(
                    pathlib.Path(dump_dir) / f"{tag}_downsampled_avg_{suffix}.txt",
                    np.column_stack(
                        [
                            coords,
                            avg_flat.real,
                            avg_flat.imag,
                            down_weight[shell_valid].reshape(-1),
                        ]
                    ),
                    header=(
                        f"xsize {down_xsize} ysize {down_size} zsize {down_size} "
                        f"xinit 0 yinit {-down_radius} zinit {-down_radius}\n"
                        "k i j real imag weight"
                    ),
                )
    fsc_dtype = jnp.dtype(output_dtype)
    if fsc_dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
        raise ValueError(f"output_dtype must be float32 or float64, got {output_dtype}")
    return jnp.asarray(fsc, dtype=fsc_dtype)


def compute_data_vs_prior(
    Ft_ctf,
    tau2,
    volume_shape,
    padding_factor=1,
    tau2_fudge=1.0,
    current_size=None,
    full_half_axis=-1,
    accumulator_volume_shape=None,
    shell_pair_counting="relion",
):
    """Compute RELION's data_vs_prior ratio per radial shell.

    RELION determines the effective resolution from the shell where
    ``data_vs_prior`` drops below 1.0, rather than from FSC < 0.143.

    The ratio is defined as::

        data_vs_prior[ires] = avg_Fweight[ires] * tau2_fudge * tau2[ires] * padding_factor**3

    where ``avg_Fweight`` is the shell-averaged Fourier weight from
    backprojection (the real part of ``Ft_ctf``), and ``tau2`` is the
    spectral signal prior.

    Parameters
    ----------
    Ft_ctf : jnp.ndarray
        Fourier-space CTF weight array in either centered full-volume or
        packed half-volume layout. The real part gives the per-voxel
        weight (sum of CTF^2 / noise).
    tau2 : jnp.ndarray, shape (n_shells,)
        Spectral signal prior (one value per radial shell).
    volume_shape : tuple of int
        3-D volume dimensions, e.g. ``(N, N, N)``.
    padding_factor : int or float
        Oversampling / padding factor (1 for no padding).
    tau2_fudge : float
        RELION's ``--tau2_fudge`` parameter (default 1.0).
    current_size : int or None
        Optional current image size. When provided, shells beyond
        ``current_size // 2`` are zeroed to match RELION's current-resolution
        truncation during growth updates.
    shell_pair_counting : {"relion", "once"}
        How the shell average of the weight counts Hermitian pairs
        (:func:`compute_relion_weight_shell_stats`).

    Returns
    -------
    jnp.ndarray, shape (n_shells,)
        Per-shell data_vs_prior ratio.
    """
    avg_weight = compute_relion_weight_shell_stats(
        Ft_ctf,
        volume_shape,
        padding_factor=padding_factor,
        r_max=current_size // 2 if current_size is not None else None,
        shell_rounding="round",
        full_half_axis=full_half_axis,
        accumulator_volume_shape=accumulator_volume_shape,
        shell_pair_counting=shell_pair_counting,
    )["avg_weight_shells"].astype(jnp.asarray(tau2).dtype)
    tau2 = jnp.asarray(tau2)
    if tau2.shape[0] != avg_weight.shape[0]:
        shell_count = min(int(tau2.shape[0]), int(avg_weight.shape[0]))
        avg_weight = avg_weight[:shell_count]
        tau2 = tau2[:shell_count]
    oversampling_correction = padding_factor**3
    data_vs_prior = avg_weight * tau2_fudge * tau2 * oversampling_correction
    if current_size is not None:
        shell_limit = min(int(current_size) // 2, int(data_vs_prior.shape[0]) - 1)
        if shell_limit + 1 < int(data_vs_prior.shape[0]):
            # A traced limit: the slice update compiled again at every new current size.
            data_vs_prior = jnp.where(
                jnp.arange(int(data_vs_prior.shape[0])) <= jnp.int32(shell_limit),
                data_vs_prior,
                jnp.zeros((), data_vs_prior.dtype),
            )
    return data_vs_prior


# RELION's --minres_map default (relion/src/ml_optimiser.cpp:1298): the
# smallest shell updateCurrentResolution may report, and the lowest shells
# that MAP reconstruction leaves without the Wiener prior term.
RELION_MINRES_MAP = 5


def resolution_from_data_vs_prior(
    data_vs_prior,
    *,
    box_size=None,
    allow_high_res_recovery=False,
    recovery_margin_shells=3,
    minres_map=RELION_MINRES_MAP,
):
    """RELION ``updateCurrentResolution`` shell for one data_vs_prior curve.

    Mirrors relion/src/ml_optimiser.cpp:6776-6819. Scans shells
    ``1 <= ires < ori_size // 2`` (skipping DC) and returns the last shell
    before ``data_vs_prior < 1``; a curve that never drops returns
    ``ori_size // 2 - 1``, one shell below Nyquist, as RELION does.

    When ``allow_high_res_recovery`` is enabled, mimic RELION's check for
    split-half auto-refine: scanning down from ``ori_size // 2 - 1``, if the
    curve rises above 1.0 again more than ``recovery_margin_shells`` shells
    past the first crossing, keep the later shell. This handles the
    phase-randomization / tight-mask artefact check in RELION.

    The result is never below ``minres_map`` (``maxres = XMIPP_MAX(maxres,
    minres_map)``, ml_optimiser.cpp:6819). RELION floors the class maximum;
    flooring each class first gives the same maximum.

    Parameters
    ----------
    data_vs_prior : array-like, shape (n_shells,)
        Per-shell data_vs_prior ratio from :func:`compute_data_vs_prior`.
    box_size : int, optional
        Box size. Defaults to ``2 * (n_shells - 1)``, RELION's
        ``ori_size / 2 + 1`` shell layout.
    allow_high_res_recovery : bool, optional
        Enable RELION's high-resolution recheck.
    recovery_margin_shells : int, optional
        Minimum number of shells by which the recovered high-resolution shell
        must exceed the first crossing.
    minres_map : int, optional
        RELION ``--minres_map``.

    Returns
    -------
    int
        Shell index of the resolution limit.
    """
    dvp = np.asarray(data_vs_prior)
    limit = len(dvp) - 1 if box_size is None else min(int(box_size) // 2, len(dvp))
    ires = 1
    while ires < limit and not dvp[ires] < 1.0:
        ires += 1
    maxres = ires - 1

    if allow_high_res_recovery:
        recovered = maxres
        for ires2 in range(limit - 1, maxres - 1, -1):
            if dvp[ires2] > 1.0:
                recovered = ires2
                break
        if recovered > maxres + int(recovery_margin_shells):
            maxres = recovered

    return max(maxres, int(minres_map))


def fsc_to_relion_ssnr(fsc, tau2_fudge=1.0, is_whole_instead_of_half=False):
    """Convert an FSC curve to RELION's data-vs-prior / SSNR curve.

    In gold-standard auto-refine, RELION updates ``data_vs_prior`` from the
    half-map FSC by converting each shell's FSC into an SSNR value. The shell
    where this curve drops below ``1`` is the same shell where the FSC drops
    below ``0.5``.
    """
    fsc = jnp.asarray(fsc)
    epsilon = jax_config.FSC_ZERO_THRESHOLD
    myfsc = jnp.maximum(fsc, epsilon)
    if is_whole_instead_of_half:
        myfsc = jnp.sqrt(2.0 * myfsc / (myfsc + 1.0))
    myfsc = jnp.minimum(myfsc, 1.0 - epsilon)
    return tau2_fudge * myfsc / (1.0 - myfsc)


def first_shell_below_threshold(values, threshold):
    """Return the first shell index below ``threshold``.

    RELION's shell scans start at shell 1 (skipping DC). If no shell drops
    below the threshold, return the last available shell.
    """
    arr = np.asarray(values)
    for i in range(1, len(arr)):
        if arr[i] < threshold:
            return i
    return len(arr) - 1


def compute_relion_incr_size_from_fsc(fsc, default=10):
    """RELION auto-refine shell-growth heuristic from the current FSC curve.

    RELION enlarges ``incr_size`` to at least ``fsc0143 - fsc05 + 5`` after the
    half-map comparison, where ``fsc05`` and ``fsc0143`` are the first shells
    where the FSC drops below 0.5 and 0.143, respectively.
    """
    fsc05 = first_shell_below_threshold(fsc, 0.5)
    fsc0143 = first_shell_below_threshold(fsc, 0.143)
    return max(int(default), int(fsc0143 - fsc05 + 5))


def update_relion_growth_state_from_fsc(
    fsc,
    current_size,
    *,
    incr_size=10,
    has_high_fsc_at_limit=False,
):
    """Update RELION's sticky current-size growth state from the FSC curve.

    RELION keeps ``incr_size`` as a non-decreasing value across iterations and
    only flips ``has_high_fsc_at_limit`` from false to true once. This helper
    mirrors the MPI auto-refine update in ``ml_optimiser_mpi.cpp``.
    """
    fsc = np.asarray(fsc)
    next_incr_size = compute_relion_incr_size_from_fsc(fsc, default=int(incr_size))

    if len(fsc) == 0:
        return next_incr_size, bool(has_high_fsc_at_limit)

    limit_shell = min(max(int(current_size) // 2 - 1, 0), len(fsc) - 1)
    high_fsc_now = bool(float(fsc[limit_shell]) > 0.2)
    return next_incr_size, bool(has_high_fsc_at_limit or high_fsc_now)


def compute_current_size_relion(resolution_shell, box_size, ave_Pmax=0.0, has_high_fsc_at_limit=False, incr_size=10):
    """Compute the next current_size using RELION's growth logic.

    RELION grows current_size beyond the current resolution limit.  If
    the average maximum posterior probability (``ave_Pmax``) exceeds 0.1
    AND the FSC is still high at the resolution limit, the jump is 25%
    of ``ori_size / 2`` (aggressive growth).  Otherwise the jump is
    ``incr_size`` shells (conservative).

    The result is clamped to ``ori_size``.

    Parameters
    ----------
    resolution_shell : int
        Current resolution shell index (e.g. from
        :func:`resolution_from_data_vs_prior` or FSC-based estimate).
    box_size : int
        Original image size in pixels (diameter, e.g. 128).
    ave_Pmax : float
        Average of the per-image maximum posterior probability.
        Typical range 0-1; early iterations have low values.
    has_high_fsc_at_limit : bool
        True if the FSC is still significantly above 0 at the current
        resolution limit (indicating the data supports higher resolution).
    incr_size : int
        Default shell increment when conditions for aggressive growth
        are not met.

    Returns
    -------
    int
        New current_size in pixels (diameter).
    """
    maxres = resolution_shell
    if ave_Pmax > 0.1 and has_high_fsc_at_limit:
        maxres += round(0.25 * box_size / 2)
    else:
        maxres += incr_size
    return min(2 * maxres, box_size)


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
):
    """RELION's ``--low_resol_join_halves`` operation on Fourier accumulators.

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
    # reconstruction's large-grid boundary uses (mean_helpers.
    # _should_host_stage_large_relion_ifft): a packed half of a physically large
    # grid stores about half its voxels. Counting stored elements moved the host
    # half accumulators of a 611^3 grid (114M elements, 228M voxels) back to the
    # device, and the reconstruction then took its monolithic device 1600^3
    # inverse FFT and failed to create the cuFFT plan (EMPIAR-10202 it3, 14456981).
    if _low_resolution_join_host_fallback_enabled_for_size(
        max(max_input_size, full_size), join_indices_np.size
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

    Ft_y_0_joined, Ft_y_1_joined = _join_half_pair_at_indices(Ft_y_0_arr, Ft_y_1_arr, join_indices)
    Ft_ctf_0_joined, Ft_ctf_1_joined = _join_half_pair_at_indices(Ft_ctf_0_arr, Ft_ctf_1_arr, join_indices)

    return _format_result((Ft_y_0_joined, Ft_y_1_joined, Ft_ctf_0_joined, Ft_ctf_1_joined))
