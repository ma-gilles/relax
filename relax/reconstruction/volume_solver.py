"""The RELION-style volume solver: one regularized (or unregularized) reconstruction from a half's
accumulators, with the large-box routes that stage its padded inverse FFT on the host."""

from __future__ import annotations

import functools
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from recovar.core import fourier_transform_utils

from relax.helpers.xla_memory_reserve import SINGLE_WORKING_SET_LIMIT_SHARE, device_fits
from relax.reconstruction import regularization_relion

if TYPE_CHECKING:
    from relax.refinement.refinement_options import ReconstructionPrograms

logger = logging.getLogger(__name__)


_LARGE_IRFFT_TRANSFORM_SIZE_LIMIT = np.iinfo(np.int32).max


def _stable_reconstruction_class(current_size, vol_shape, padding_factor, accumulator_volume_shape, tau_is_1d):
    """``(physical current size, physical accumulator shape)`` of the reconstruction class, or None.

    With the resident engine's stable Fourier windows on, RELION's
    reconstruction runs in one class for the whole run, the full box: the
    current-size accumulator is zero-padded to the full-box cube and
    ``post_process_from_filter_v2`` takes the box as its static bound and
    RELION's size as the traced ``logical_current_size``, so one program serves
    every iteration (recovar 8633a2a24). The padded voxels lie outside every
    logical support, so the reconstruction is the logical one. The inverse
    transform already runs on the padded full box at every size, so the larger
    accumulator adds only elementwise work, and no early iteration needs more
    memory than the final full-box one.
    """

    from relax.helpers.half_volume_mstep import relion_backprojector_volume_shape

    if current_size is None or accumulator_volume_shape is None or not tau_is_1d:
        return None
    box_size = int(vol_shape[0])
    logical = int(current_size)
    if logical <= 0 or logical > box_size:
        return None
    logical_shape = tuple(
        int(v) for v in relion_backprojector_volume_shape(vol_shape, padding_factor, current_size=logical)
    )
    if logical_shape != tuple(int(v) for v in accumulator_volume_shape):
        return None
    physical_shape = tuple(
        int(v) for v in relion_backprojector_volume_shape(vol_shape, padding_factor, current_size=box_size)
    )
    return box_size, physical_shape


def _stable_unregularized_class(vol_shape, padding_factor, accumulator_volume_shape, tau, current_size):
    """The full-box accumulator shape an unregularized reconstruction runs in, or None.

    Without a prior and without ``current_size`` (the unregularized half maps
    and class means), the accumulator's own size sets the Wiener mask radius
    and the floor's clamp shell. Zero-padded to the full-box cube with that
    size passed as recovar's traced ``logical_accumulator_size``, every
    iteration's accumulator runs in one program instead of one per size; the
    padded voxels lie outside the logical support.
    """

    from relax.helpers.half_volume_mstep import relion_backprojector_volume_shape

    if tau is not None or current_size is not None or accumulator_volume_shape is None:
        return None
    logical_shape = tuple(int(v) for v in accumulator_volume_shape)
    physical_shape = tuple(int(v) for v in relion_backprojector_volume_shape(vol_shape, padding_factor, current_size=int(vol_shape[0])))
    size = logical_shape[0]
    if len(set(logical_shape)) != 1 or size % 2 == 0 or size >= physical_shape[0]:
        return None
    return physical_shape


@functools.partial(jax.jit, static_argnames=("logical_shape", "physical_shape"))
def _pad_accumulators_to_class(Ft_ctf, Ft_y, logical_shape, physical_shape):
    """Both accumulators zero-padded to the class cube, in one program (eager: four per new size)."""

    return (
        _pad_accumulator_to_class(Ft_ctf, logical_shape, physical_shape),
        _pad_accumulator_to_class(Ft_y, logical_shape, physical_shape),
    )


def _pad_accumulator_to_class(values, logical_shape, physical_shape):
    """Zero-pad a centered full or packed-half (x, y, z>=0) accumulator to a larger odd cube."""

    logical_size, physical_size = int(logical_shape[0]), int(physical_shape[0])
    pad = (physical_size - logical_size) // 2
    values = jnp.asarray(values)
    flat = values.ndim == 1
    full = int(values.size) == logical_size**3
    grid = values.reshape((logical_size,) * 3 if full else (logical_size, logical_size, logical_size // 2 + 1))
    widths = [(pad, pad), (pad, pad), (pad, pad) if full else (0, physical_size // 2 - logical_size // 2)]
    padded = jnp.pad(grid, widths)
    return padded.reshape(-1) if flat else padded


# RECOVAR's name of each gridding-correction window (``post_process_from_filter_v2``).
_RECOVAR_GRIDDING_CORRECT = {"radial": "radial", "separable": "square"}


def _reconstruct_volume_eager(
    Ft_ctf,
    Ft_y,
    vol_shape,
    padding_factor,
    tau,
    tau2_fudge,
    projection_padding_factor,
    use_spherical_mask=True,
    grid_correct=True,
    minres_map=0,
    current_size=None,
    return_real_space=False,
    accumulator_volume_shape=None,
    tau_is_1d=False,
    preserve_output_precision=False,
    relion_filter_scale=None,
    retained_device_numerator=None,
    gridding_kernel="radial",
    *,
    programs: ReconstructionPrograms,
):
    """Eager RELION-style reconstruction from full or half Fourier accumulators.

    ``programs`` (the run's ``ScoringVariants.reconstruction``) chooses the stable full-box class and where the
    padded inverse FFT runs; it changes no value.

    This keeps the reconstruction boundary out of a single monolithic JIT while
    letting the local exact path keep its accumulators in packed half-volume
    layout until the final iDFT boundary. ``gridding_kernel`` is the real-space
    correction window: RELION's ``"radial"`` one or the ``"separable"`` per-axis product.
    relax always leaves ``use_spherical_mask``, ``grid_correct`` and ``return_real_space`` at their
    defaults; the parameters remain for scripts/replay_bpref_contribution_bundle.py and
    scripts/run_k_class_parity.py, which set them.
    """
    gridding_correct = _RECOVAR_GRIDDING_CORRECT[gridding_kernel]
    from recovar.reconstruction import relion_functions

    from relax.reconstruction import relion_functions_relion

    padded_shape = relion_functions._relion_reconstruction_padded_shape(vol_shape, padding_factor)
    # A compact accumulator whose FFTW pad the device cannot hold (_relion_pad_exceeds_device_working_set) is solved
    # on the CPU backend from its repack on: after a pass the 16 GB pool could not place even the 0.69 GiB repacked
    # half at box 448 (relax#49). The accumulators go back to the device if the reconstruction is not host-staged.
    cpu_accumulators = not relion_functions._large_grid_postprocess_is_physically_large(
        int(np.prod(accumulator_volume_shape or [int(vol_shape[0]) * int(padding_factor)] * 3))
    ) and _relion_pad_exceeds_device_working_set(padded_shape)
    if cpu_accumulators:
        cpu = jax.devices("cpu")[0]
        Ft_ctf, Ft_y = (jax.device_put(value, cpu) if isinstance(value, jax.Array) else value for value in (Ft_ctf, Ft_y))
        with jax.default_device(cpu):
            Ft_ctf, Ft_y = _pack_compact_full_accumulators_for_large_relion_ifft(
                Ft_ctf, Ft_y, vol_shape, padding_factor, accumulator_volume_shape, relion_functions
            )
    else:
        Ft_ctf, Ft_y = _pack_compact_full_accumulators_for_large_relion_ifft(
            Ft_ctf,
            Ft_y,
            vol_shape,
            padding_factor,
            accumulator_volume_shape,
            relion_functions,
        )
    stable_class = (
        _stable_reconstruction_class(current_size, vol_shape, padding_factor, accumulator_volume_shape, tau_is_1d)
        if programs.stable_windows
        else None
    )
    unregularized_class = (
        _stable_unregularized_class(vol_shape, padding_factor, accumulator_volume_shape, tau, current_size)
        if programs.stable_windows
        else None
    )
    postprocess_args = (Ft_ctf, Ft_y, vol_shape, padding_factor)
    postprocess_kwargs = dict(
        tau=tau,
        kernel="triangular",
        use_spherical_mask=use_spherical_mask,
        grid_correct=grid_correct,
        gridding_correct=gridding_correct,
        kernel_width=1,
        tau2_fudge=tau2_fudge,
        gridding_padding_factor=projection_padding_factor,
        minres_map=minres_map,
        current_size=current_size,
        return_real_space=return_real_space,
        accumulator_volume_shape=accumulator_volume_shape,
        tau_is_1d=tau_is_1d,
        preserve_output_precision=preserve_output_precision,
        # EM transform precision follows its accumulation precision, not the
        # deliberate RFLOAT denominator/gridding operands. Complex128 inputs
        # retain the diagnostic transform path.
        fft_compute_dtype=jnp.result_type(Ft_y.dtype, jnp.complex64),
        relion_filter_scale=relion_filter_scale,
    )
    host_stage_large_ifft = _should_host_stage_large_relion_ifft(
        Ft_ctf,
        Ft_y,
        vol_shape,
        padding_factor,
        accumulator_volume_shape,
        relion_functions,
    )
    if cpu_accumulators and not host_stage_large_ifft:
        Ft_ctf, Ft_y = (jax.device_put(value, jax.devices()[0]) for value in (Ft_ctf, Ft_y))
    if retained_device_numerator is not None and not host_stage_large_ifft:
        raise ValueError(
            "A retained device numerator is only valid for the large host-staged RELION reconstruction path"
        )
    if not host_stage_large_ifft:
        if stable_class is not None:
            physical_size, physical_shape = stable_class
            postprocess_args = (
                *_pad_accumulators_to_class(
                    Ft_ctf,
                    Ft_y,
                    tuple(int(v) for v in accumulator_volume_shape),
                    tuple(int(v) for v in physical_shape),
                ),
                vol_shape,
                padding_factor,
            )
            postprocess_kwargs = dict(
                postprocess_kwargs,
                current_size=int(physical_size),
                accumulator_volume_shape=physical_shape,
                logical_current_size=jnp.int32(int(current_size)),
            )
        elif unregularized_class is not None:
            logical_shape = tuple(int(v) for v in accumulator_volume_shape)
            postprocess_args = (
                *_pad_accumulators_to_class(Ft_ctf, Ft_y, logical_shape, unregularized_class),
                vol_shape,
                padding_factor,
            )
            postprocess_kwargs = dict(
                postprocess_kwargs,
                accumulator_volume_shape=unregularized_class,
                logical_accumulator_size=jnp.int32(logical_shape[0]),
            )
        result = relion_functions.post_process_from_filter_v2(
            *postprocess_args,
            **postprocess_kwargs,
        )
        reconstruction_shape = relion_functions._relion_reconstruction_padded_shape(
            vol_shape,
            padding_factor,
        )
        if _large_irfft_requires_explicit_normalization(reconstruction_shape):
            # XLA's built-in ``norm='backward'`` normalization overflows its
            # signed-int32 transform-size product at 1600^3 and silently omits
            # the reciprocal.  Physically large device accumulators take this
            # monolithic branch to avoid overlapping another box-scale device
            # buffer, so apply the reciprocal to the completed result in a
            # separate donating executable.
            logger.info(
                "RELION large inverse-FFT normalization boundary: "
                "reconstruction_shape=%s transform_size=%d "
                "implementation=jax_monolithic_dynamic_scale",
                reconstruction_shape,
                math.prod(reconstruction_shape),
            )
            result = _apply_large_irfft_scale(result, reconstruction_shape)
        return result

    accumulator_shape = _accumulator_shape_or_default(vol_shape, padding_factor, accumulator_volume_shape)
    reconstruction_shape = relion_functions._relion_reconstruction_padded_shape(
        vol_shape,
        padding_factor,
    )
    packed_half_bytes = int(
        np.prod(fourier_transform_utils.volume_shape_to_half_volume_shape(reconstruction_shape))
        * np.dtype(np.complex64).itemsize
    )
    logger.info(
        "RELION split pre-IFFT host boundary: accumulator_shape=%s reconstruction_shape=%s packed_half_bytes=%d",
        accumulator_shape,
        reconstruction_shape,
        packed_half_bytes,
    )
    if accumulator_shape[0] > reconstruction_shape[0]:
        # The original Stage-A executable combined denominator
        # regularization and complex division.  Although donation aliases its
        # numerator to the output, the regularization needs several
        # box-scale float32 temporaries.  Keep the complex64 numerator on the
        # host while those temporaries are live, then stage it only for the
        # zero-temporary donating divide.
        stage_a_filter = jnp.asarray(Ft_ctf)
        stage_a_filter.block_until_ready()
        if np.dtype(stage_a_filter.dtype) != np.dtype(np.float32):
            raise TypeError(
                f"Large RELION Stage A requires a float32 filter for exact donation, got {stage_a_filter.dtype}"
            )
        logger.info(
            "RELION Stage A staging host filter for donating regularization: shape=%s dtype=%s",
            tuple(stage_a_filter.shape),
            stage_a_filter.dtype,
        )
        regularized_filter_device = relion_functions_relion.regularize_large_relion_half_filter_donate_ctf(
            stage_a_filter,
            tau,
            vol_shape,
            padding_factor,
            tau2_fudge,
            minres_map,
            current_size,
            accumulator_shape,
            tau_is_1d,
            relion_filter_scale,
        )
        regularized_filter_device.block_until_ready()
        filter_input_donated = _device_array_is_deleted(stage_a_filter)
        if filter_input_donated is not True:
            regularization_relion.delete_device_array(regularized_filter_device)
            regularization_relion.delete_device_array(stage_a_filter)
            raise RuntimeError("Large RELION Stage-A regularization did not donate its float32 filter input")
        if np.dtype(regularized_filter_device.dtype) != np.dtype(np.float32):
            regularization_relion.delete_device_array(regularized_filter_device)
            raise TypeError(
                f"Large RELION Stage-A regularization must return float32, got {regularized_filter_device.dtype}"
            )
        logger.info(
            "RELION Stage A regularization complete: filter_input_donated=%s",
            filter_input_donated,
        )

        if retained_device_numerator is not None:
            if tuple(retained_device_numerator.shape) != tuple(Ft_y.shape):
                raise ValueError(
                    "Retained device numerator shape does not match the host numerator: "
                    f"{tuple(retained_device_numerator.shape)} != {tuple(Ft_y.shape)}"
                )
            if np.dtype(retained_device_numerator.dtype) != np.dtype(Ft_y.dtype):
                raise ValueError(
                    "Retained device numerator dtype does not match the host numerator: "
                    f"{retained_device_numerator.dtype} != {Ft_y.dtype}"
                )
            stage_a_numerator = retained_device_numerator
            logger.info(
                "RELION Stage A reusing retained half-0 device numerator: shape=%s dtype=%s",
                tuple(retained_device_numerator.shape),
                retained_device_numerator.dtype,
            )
        else:
            # A NumPy argument passed directly to a donate_argnums JIT is first
            # staged by dispatch, but that transient input cannot be donated.
            # Materialise an explicit JAX array so half 2 can alias its 15-GiB
            # Stage-A output into the staged numerator just as half 1 aliases
            # the retained low-resolution-join buffer.
            stage_a_numerator = jnp.asarray(Ft_y)
            stage_a_numerator.block_until_ready()
            if isinstance(Ft_y, np.ndarray):
                logger.info(
                    "RELION Stage A staging host numerator for donation: shape=%s dtype=%s",
                    tuple(stage_a_numerator.shape),
                    stage_a_numerator.dtype,
                )
        wiener_half_device = relion_functions_relion.divide_large_relion_half_numerator_donate_numerator(
            stage_a_numerator,
            regularized_filter_device,
            padding_factor,
            current_size,
            accumulator_shape,
        )
        wiener_half_device.block_until_ready()
        wiener_half_host = np.asarray(jax.device_get(wiener_half_device)).reshape(
            fourier_transform_utils.volume_shape_to_half_volume_shape(accumulator_shape),
        )
        regularization_relion.delete_device_array(wiener_half_device)
        regularization_relion.delete_device_array(stage_a_numerator)
        regularization_relion.delete_device_array(regularized_filter_device)
        regularization_relion.delete_device_array(stage_a_filter)
        del wiener_half_device
        del stage_a_numerator
        del regularized_filter_device
        del stage_a_filter

        fftw_half_host = _crop_relion_wiener_half_to_fftw_host(
            wiener_half_host,
            accumulator_shape,
            reconstruction_shape,
            relion_functions,
        )
        del wiener_half_host
    else:
        if retained_device_numerator is not None:
            raise ValueError(
                "A retained device numerator is only valid for the crop branch of the "
                "large host-staged RELION reconstruction path"
            )
        if relion_functions._large_grid_postprocess_is_physically_large(
            int(np.prod(accumulator_shape)),
        ):
            # A physically large accumulator that is still smaller than the
            # inverse-FFT grid (EMPIAR-10202 from about current size 410: 823^3
            # to 1163^3 against 1600^3) returns its Wiener half before the pad,
            # and the pad into the 16.4-GB FFTW half runs on the host, as RELION
            # windows its reconstruction on the CPU. Padding on the device built
            # the zero grid and its scattered copy side by side (32.8 GB live in
            # the census, bigbox 14468686) on top of the resident state.
            wiener_half_device = relion_functions.post_process_from_filter_v2(
                *postprocess_args,
                **postprocess_kwargs,
                input_half_volume=True,
                return_wiener_half_before_window=True,
            )
            wiener_half_device.block_until_ready()
            wiener_half_host = np.asarray(jax.device_get(wiener_half_device))
            regularization_relion.delete_device_array(wiener_half_device)
            del wiener_half_device
            logger.info(
                "RELION pre-window Wiener half padded on the host: accumulator_shape=%s reconstruction_shape=%s",
                accumulator_shape,
                reconstruction_shape,
            )
            fftw_half_host = _pad_relion_wiener_half_to_fftw_host(
                wiener_half_host,
                accumulator_shape,
                reconstruction_shape,
                relion_functions,
            )
            del wiener_half_host
        elif _relion_pad_exceeds_device_working_set(reconstruction_shape):
            # The same Wiener solve and pad on the CPU backend: the device would need the zero FFTW half and its
            # scattered copy as single blocks (relax#49: at box 380 a 1.64 GiB FFTW half could not be placed in
            # a fragmented 16 GB pool; at box 448 2.69 GiB did not fit). Only the small accumulators move.
            cpu = jax.devices("cpu")[0]

            def on_cpu(value):
                return jax.device_put(value, cpu) if isinstance(value, jax.Array) else value

            with jax.default_device(cpu):
                fftw_half_host = np.asarray(
                    relion_functions.post_process_from_filter_v2(
                        *[on_cpu(value) for value in postprocess_args],
                        **{key: on_cpu(value) for key, value in postprocess_kwargs.items()},
                        input_half_volume=True,
                        return_fftw_half_before_ifft=True,
                    )
                )
            logger.info(
                "RELION Wiener solve and FFTW pad on the CPU backend: accumulator_shape=%s reconstruction_shape=%s",
                accumulator_shape,
                reconstruction_shape,
            )
        else:
            fftw_half_device = relion_functions.post_process_from_filter_v2(
                *postprocess_args,
                **postprocess_kwargs,
                input_half_volume=True,
                return_fftw_half_before_ifft=True,
            )
            fftw_half_device.block_until_ready()
            fftw_half_host = np.asarray(jax.device_get(fftw_half_device))
            del fftw_half_device

    explicit_irfft_normalization = _large_irfft_requires_explicit_normalization(
        reconstruction_shape,
    )
    host_irfft = _large_relion_host_irfft_enabled(reconstruction_shape, forced=programs.host_irfft)
    if host_irfft:
        logger.info(
            "RELION padded inverse FFT using host scipy.fft: reconstruction_shape=%s "
            "output_shape=%s input_bytes=%d workers=%d",
            reconstruction_shape,
            tuple(int(size) for size in vol_shape),
            int(fftw_half_host.nbytes),
            programs.host_fft_workers,
        )
        unpadded_real_host = _host_irfft_and_center_crop(
            fftw_half_host,
            reconstruction_shape,
            vol_shape,
            workers=programs.host_fft_workers,
        )
        del fftw_half_host
        result = relion_functions_relion.finish_large_relion_postprocess_from_unpadded_real(
            unpadded_real_host,
            vol_shape,
            padding_factor,
            kernel="triangular",
            use_spherical_mask=use_spherical_mask,
            grid_correct=grid_correct,
            gridding_correct=gridding_correct,
            kernel_width=1,
            return_real_space=return_real_space,
            gridding_padding_factor=projection_padding_factor,
        )
    else:
        result = relion_functions_relion.finish_large_relion_postprocess_from_fftw_half(
            fftw_half_host,
            vol_shape,
            padding_factor,
            kernel="triangular",
            use_spherical_mask=use_spherical_mask,
            grid_correct=grid_correct,
            gridding_correct=gridding_correct,
            kernel_width=1,
            return_real_space=return_real_space,
            gridding_padding_factor=projection_padding_factor,
        )
    if explicit_irfft_normalization:
        logger.info(
            "RELION large inverse-FFT normalization boundary: "
            "reconstruction_shape=%s transform_size=%d implementation=%s",
            reconstruction_shape,
            math.prod(reconstruction_shape),
            "scipy_host_backward" if host_irfft else "jax_dynamic_scale",
        )
    if explicit_irfft_normalization and not host_irfft:
        # The inverse-FFT executable has completed; see _apply_large_irfft_scale.
        result = _apply_large_irfft_scale(result, reconstruction_shape)
    return result


def _large_irfft_requires_explicit_normalization(volume_shape) -> bool:
    """Return whether XLA's inverse-FFT normalization exceeds int32 range."""

    return math.prod(int(size) for size in volume_shape) > _LARGE_IRFFT_TRANSFORM_SIZE_LIMIT


# Padding the Wiener half into the FFTW half on the device holds the zero FFTW half and its scattered copy side by
# side (bigbox 14468686): two halves.
_DEVICE_PAD_HALVES = 2.0


# The device inverse FFT's working set over its packed half: the half itself, the real output (twice the half's
# float32 count, about one more half in bytes) and the cuFFT work area, measured at 1.63 GiB for the 760^3 half of
# 1.76 GB (0.93 of it; relax#40), so three halves.
_DEVICE_IRFFT_HALVES = 3.0


def _device_allocator_limit_bytes() -> int | None:
    """The JAX allocator's byte limit on the first GPU, or None off GPU."""

    devices = [device for device in jax.devices() if getattr(device, "platform", "") in {"gpu", "cuda"}]
    if not devices:
        return None
    stats = devices[0].memory_stats() or {}
    limit = stats.get("bytes_limit")
    return None if limit is None else int(limit)


def _relion_pad_exceeds_device_working_set(reconstruction_shape, *, allocator_limit_bytes: int | None = None) -> bool:
    """Whether the Wiener solve and pad into the FFTW half of ``reconstruction_shape`` should run on the CPU.

    When the pad's device working set (``_DEVICE_PAD_HALVES`` packed complex64 halves) does not fit the device
    (``xla_memory_reserve.device_fits``; the limit is ``allocator_limit_bytes``, read from the device when None).
    Off GPU it is False.
    """

    limit = _device_allocator_limit_bytes() if allocator_limit_bytes is None else int(allocator_limit_bytes)
    if limit is None:
        return False
    half_bytes = int(np.prod(fourier_transform_utils.volume_shape_to_half_volume_shape(reconstruction_shape))) * 8
    return not device_fits(int(_DEVICE_PAD_HALVES * half_bytes), pool_limit_bytes=limit)


def _large_relion_host_irfft_enabled(
    volume_shape, *, forced: bool | None, allocator_limit_bytes: int | None = None
) -> bool:
    """Return whether a padded RELION inverse FFT should execute on the host.

    ``forced`` (``ReconstructionPrograms.host_irfft``) decides when not None. Otherwise automatically when the
    transform's int32 size product overflows, or when its device working set (``_DEVICE_IRFFT_HALVES`` packed
    complex64 halves) does not fit the device (``xla_memory_reserve.device_fits``; the limit is
    ``allocator_limit_bytes``, read from the device when None): a cuFFT work area that cannot be found aborts the
    process instead of raising.
    """

    if forced is not None:
        return forced
    if _large_irfft_requires_explicit_normalization(volume_shape):
        return True
    limit = _device_allocator_limit_bytes() if allocator_limit_bytes is None else int(allocator_limit_bytes)
    if limit is None:
        return False
    half_bytes = int(np.prod(fourier_transform_utils.volume_shape_to_half_volume_shape(volume_shape))) * 8
    working_set = _DEVICE_IRFFT_HALVES * half_bytes
    if device_fits(int(working_set), pool_limit_bytes=limit):
        return False
    logger.info(
        "RELION padded inverse FFT on the host: its device working set %.2f GiB does not fit the %.2f GiB allocator "
        "(device_fits: %.0f%% of the limit, the allocator's headroom and largest block; relax#40)",
        working_set / 2**30, limit / 2**30, 100 * SINGLE_WORKING_SET_LIMIT_SHARE,
    )
    return True


# Complex128 elements of one slab of the host inverse FFT (1 GiB; smaller slabs ran slower at 1600^3).
_HOST_IRFFT_SLAB_ELEMENTS = 1 << 26


def _host_irfft_and_center_crop(
    fftw_half,
    reconstruction_shape,
    output_shape,
    *,
    workers: int,
):
    """RELION's padded inverse FFT of a reconstruction, keeping only its centre crop, in slabs and in double.

    ``fftw_half`` is in raw FFTW order. RELION transforms the padded volume in ``RFLOAT`` (double in its CPU
    build, ``DoublePrec_CPU``; BackProjector::windowToOridimRealSpace) and crops it to the output box. The
    separable inverse runs the same way here, without the whole padded real volume: for a slab of x columns at a
    time the (z, y) inverse transform runs in complex128 and only the crop's z and y rows are kept; the x
    inverse then runs on those rows a z slab at a time and keeps the crop's x. The strided slab copies run on
    ``workers`` threads, as the transforms do. The crop indices combine
    ``ifftshift`` with RELION's spatial unpadding. At box 800 (1600^3 padded) the transient beside the input is
    about 10 GiB instead of scipy's whole-volume ``irfftn`` working set (41-64 GiB measured, relax#39). Returns the
    float32 crop, C-contiguous.
    """

    from scipy import fft as scipy_fft

    reconstruction_shape = tuple(int(size) for size in reconstruction_shape)
    output_shape = tuple(int(size) for size in output_shape)
    if len(reconstruction_shape) != 3 or len(output_shape) != 3:
        raise ValueError(
            "RELION host inverse FFT requires three-dimensional shapes, got "
            f"reconstruction={reconstruction_shape} output={output_shape}"
        )
    if any(output > reconstruction for output, reconstruction in zip(output_shape, reconstruction_shape)):
        raise ValueError(
            "RELION host inverse FFT crop cannot exceed its reconstruction: "
            f"reconstruction={reconstruction_shape} output={output_shape}"
        )

    expected_half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(
        reconstruction_shape,
    )
    fftw_half = np.asarray(fftw_half, order="C").reshape(expected_half_shape)
    raw_indices = []
    for reconstruction, output in zip(reconstruction_shape, output_shape):
        padding_width = reconstruction - output
        pad_before = padding_width // 2
        centered_indices = np.arange(pad_before, pad_before + output, dtype=np.intp)
        # ``np.fft.ifftshift`` takes centered output index ``i`` from raw
        # input index ``i + floor(N / 2)``.  Keep the explicit floor because
        # the distinction matters for odd reconstruction sizes.
        raw_indices.append((centered_indices + reconstruction // 2) % reconstruction)
    z_runs, y_runs, x_runs = (_contiguous_runs(indices) for indices in raw_indices)
    n_z, n_y, n_x_half = expected_half_shape
    workers = max(1, int(workers))

    def in_parallel(function, n):
        """``function(start, stop)`` over ``workers`` contiguous parts of ``range(n)``: the slab copies are
        strided NumPy assignments, which release the GIL."""
        step = -(-n // workers)
        list(pool.map(lambda i: function(i * step, min(n, (i + 1) * step)), range(workers)))

    def copy_runs(destination, source, runs, axis):
        for source_start, source_stop, destination_start in runs:
            index = [slice(None)] * destination.ndim
            index[axis] = slice(destination_start, destination_start + source_stop - source_start)
            source_index = [slice(None)] * source.ndim
            source_index[axis] = slice(source_start, source_stop)
            destination[tuple(index)] = source[tuple(source_index)]

    zy = np.empty((output_shape[0], output_shape[1], n_x_half), dtype=np.complex128)
    cropped = np.empty(output_shape, dtype=np.float32)
    x_step = max(1, min(n_x_half, _HOST_IRFFT_SLAB_ELEMENTS // (n_z * n_y)))
    slab_buffer = np.empty((n_z, n_y, x_step), dtype=np.complex128)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # (z, y) inverse per slab of x columns, kept at the crop's z and y rows.
        for x0 in range(0, n_x_half, x_step):
            x1 = min(n_x_half, x0 + x_step)
            slab = slab_buffer[:, :, : x1 - x0]

            def gather(z0, z1, slab=slab, x0=x0, x1=x1):
                slab[z0:z1] = fftw_half[z0:z1, :, x0:x1]

            in_parallel(gather, n_z)
            slab = scipy_fft.ifftn(slab, axes=(0, 1), norm="backward", overwrite_x=True, workers=workers)

            def keep(z0, z1, slab=slab, x0=x0, x1=x1):
                for source_start, source_stop, destination_start in z_runs:
                    lo = max(z0, destination_start)
                    hi = min(z1, destination_start + source_stop - source_start)
                    if lo < hi:
                        rows = slab[source_start + lo - destination_start : source_start + hi - destination_start]
                        copy_runs(zy[lo:hi, :, x0:x1], rows, y_runs, axis=1)

            in_parallel(keep, output_shape[0])
            del slab
        del slab_buffer

        # x inverse per slab of z rows, kept at the crop's x.
        z_step = max(1, _HOST_IRFFT_SLAB_ELEMENTS // (output_shape[1] * reconstruction_shape[2]))
        for z0 in range(0, output_shape[0], z_step):
            z1 = min(output_shape[0], z0 + z_step)
            rows = scipy_fft.irfft(zy[z0:z1], n=reconstruction_shape[2], axis=2, norm="backward", workers=workers)
            copy_runs(cropped[z0:z1], rows, x_runs, axis=2)
            del rows
    return cropped


def _contiguous_runs(indices):
    """``[(source_start, source_stop, destination_start)]`` covering ``indices`` (crop index ``i`` reads
    ``indices[i]``) as runs of consecutive source indices; a wrapped centre crop is two runs."""

    breaks = np.flatnonzero(np.diff(indices) != 1) + 1
    starts = np.concatenate([[0], breaks])
    stops = np.concatenate([breaks, [indices.size]])
    return [(int(indices[a]), int(indices[b - 1]) + 1, int(a)) for a, b in zip(starts, stops)]


def _accumulator_shape_or_default(vol_shape, padding_factor, accumulator_volume_shape) -> tuple:
    """The backprojector accumulator's full shape: ``accumulator_volume_shape``, else the padded box cubed."""
    return (
        tuple(3 * [int(vol_shape[0]) * int(padding_factor)])
        if accumulator_volume_shape is None
        else tuple(int(s) for s in accumulator_volume_shape)
    )


def _apply_large_irfft_scale(result, reconstruction_shape):
    """Apply the inverse FFT's 1 / N in a separate donating executable.

    XLA's built-in ``norm='backward'`` normalization overflows its signed-int32 transform-size product at
    1600^3 and silently omits the reciprocal; donation keeps this correction memory-neutral for box-scale maps.
    """
    inverse_transform_scale = jnp.asarray(
        np.float32(1.0 / float(math.prod(reconstruction_shape))),
    )
    return _normalize_large_irfft_result_donate(
        result,
        inverse_transform_scale,
    )


@functools.partial(jax.jit, donate_argnums=(0,))
def _normalize_large_irfft_result_donate(result, inverse_transform_scale):
    """Normalize a large raw inverse FFT in a separate donating executable."""

    return result * inverse_transform_scale


def _should_host_stage_large_relion_ifft(
    Ft_ctf,
    Ft_y,
    vol_shape,
    padding_factor,
    accumulator_volume_shape,
    relion_functions,
):
    """Return whether eager reconstruction should cross the padded-iFFT host boundary."""

    accumulator_shape = _accumulator_shape_or_default(vol_shape, padding_factor, accumulator_volume_shape)
    reconstruction_shape = relion_functions._relion_reconstruction_padded_shape(
        vol_shape,
        padding_factor,
    )
    if accumulator_shape == reconstruction_shape:
        return False
    half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(accumulator_shape)
    half_size = int(np.prod(half_shape))

    def _is_packed_half(array):
        return tuple(array.shape) == half_shape or (array.ndim == 1 and int(array.size) == half_size)

    # The split is governed by the inverse-FFT grid, not the current-size
    # accumulator. Early box-scale iterations can have a compact accumulator
    # but still pad to a 1600^3 transform whose built-in normalization
    # overflows XLA's signed-int32 transform-size product. Compact device
    # accumulators can safely form and host-stage that one padded boundary.
    # Large accumulators still require the existing earlier host offload so
    # their storage cannot overlap the padded inverse-FFT workspace.
    accumulator_is_large = relion_functions._large_grid_postprocess_is_physically_large(
        int(np.prod(accumulator_shape)),
    )
    inputs_are_host = isinstance(Ft_ctf, np.ndarray) and isinstance(Ft_y, np.ndarray)
    if not _is_packed_half(Ft_ctf) or not _is_packed_half(Ft_y):
        return False
    if accumulator_is_large and not inputs_are_host:
        return False
    return relion_functions._large_grid_postprocess_single_precision_enabled(
        int(np.prod(reconstruction_shape)),
    )


def _pack_compact_full_accumulators_for_large_relion_ifft(
    Ft_ctf,
    Ft_y,
    vol_shape,
    padding_factor,
    accumulator_volume_shape,
    relion_functions,
):
    """Losslessly repack compact full accumulators before a giant padded iFFT.

    The RELION x-half M-step keeps the historical RECOVAR full-volume public
    contract on normal-sized accumulator grids. At large box sizes an early
    iteration can therefore reach reconstruction with compact full Hermitian
    arrays even though its padded inverse-FFT grid is giant. Repack only this
    compact/full case so it can use the packed pre-iFFT host boundary without
    changing the public M-step contract or the large-accumulator offload path.
    """

    accumulator_shape = _accumulator_shape_or_default(vol_shape, padding_factor, accumulator_volume_shape)
    reconstruction_shape = relion_functions._relion_reconstruction_padded_shape(
        vol_shape,
        padding_factor,
    )
    if accumulator_shape == reconstruction_shape:
        return Ft_ctf, Ft_y

    accumulator_voxels = int(np.prod(accumulator_shape))
    reconstruction_voxels = int(np.prod(reconstruction_shape))
    # The accumulator decision is physical: forcing single precision on a
    # compact grid must not misclassify it as too large to repack.  The
    # reconstruction still needs the single-precision large-grid path because
    # this boundary stages a complex64 packed half-volume.
    if relion_functions._large_grid_postprocess_is_physically_large(
        accumulator_voxels,
    ) or not relion_functions._large_grid_postprocess_single_precision_enabled(
        reconstruction_voxels,
    ):
        return Ft_ctf, Ft_y

    def _is_full(array):
        return tuple(array.shape) == accumulator_shape or (array.ndim == 1 and int(array.size) == accumulator_voxels)

    if not _is_full(Ft_ctf) or not _is_full(Ft_y):
        return Ft_ctf, Ft_y

    logger.info(
        "RELION giant-iFFT compact full-to-half repack: accumulator_shape=%s reconstruction_shape=%s",
        accumulator_shape,
        reconstruction_shape,
    )
    return (
        fourier_transform_utils.full_volume_to_half_volume(
            Ft_ctf,
            accumulator_shape,
        ).reshape(-1),
        fourier_transform_utils.full_volume_to_half_volume(
            Ft_y,
            accumulator_shape,
        ).reshape(-1),
    )


def _crop_relion_wiener_half_to_fftw_host(
    wiener_half,
    accumulator_shape,
    reconstruction_shape,
    relion_functions,
):
    """Crop a centered packed half-volume into raw FFTW order on the host."""

    accumulator_shape = tuple(int(s) for s in accumulator_shape)
    reconstruction_shape = tuple(int(s) for s in reconstruction_shape)
    accumulator_half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(
        accumulator_shape,
    )
    wiener_half = np.asarray(wiener_half).reshape(accumulator_half_shape)
    centered_axis_idx = relion_functions._relion_centered_axis_take_indices(
        accumulator_shape[0],
        reconstruction_shape[0],
    )
    raw_axis_idx = np.fft.ifftshift(centered_axis_idx)
    col_idx = np.arange(reconstruction_shape[-1] // 2 + 1, dtype=np.int32)
    return wiener_half[np.ix_(raw_axis_idx, raw_axis_idx, col_idx)]


def _pad_relion_wiener_half_to_fftw_host(
    wiener_half,
    accumulator_shape,
    reconstruction_shape,
    relion_functions,
):
    """Pad a centered packed Wiener half into raw FFTW order on the host.

    The host counterpart of recovar's ``_relion_pad_centered_half_fourier_to_fftw``
    (the padding branch of ``post_process_from_filter_v2``): the same support
    sphere is zeroed and the same placement is written, so the result equals the
    device pad exactly.
    """

    accumulator_shape = tuple(int(s) for s in accumulator_shape)
    reconstruction_shape = tuple(int(s) for s in reconstruction_shape)
    old_dim, new_dim = accumulator_shape[0], reconstruction_shape[0]
    if len(set(accumulator_shape)) != 1 or len(set(reconstruction_shape)) != 1 or new_dim <= old_dim:
        raise ValueError(
            f"host Wiener padding needs cubic shapes with a larger target, got {accumulator_shape} -> "
            f"{reconstruction_shape}"
        )
    old_half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(accumulator_shape)
    new_half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(reconstruction_shape)
    wiener_half = np.asarray(wiener_half).reshape(old_half_shape)
    freq = relion_functions._relion_centered_axis_fftw_frequencies(old_dim).astype(np.int64)
    raw_axis_idx = np.where(freq >= 0, freq, new_dim + freq)
    col_freq = np.arange(old_half_shape[-1], dtype=np.int64)
    max_r2 = int(old_half_shape[-1] - 1) ** 2
    out = np.zeros(new_half_shape, dtype=wiener_half.dtype)
    freq2 = freq * freq
    for i0 in range(old_dim):
        # One plane at a time keeps the support mask small (old_dim^2 x cols).
        support = (freq2[i0] + freq2[:, None] + (col_freq * col_freq)[None, :]) <= max_r2
        plane = np.where(support, wiener_half[i0], np.zeros((), dtype=wiener_half.dtype))
        out[raw_axis_idx[i0][None, None], raw_axis_idx[:, None], col_freq[None, :]] = plane
    return out


def _device_array_is_deleted(value):
    """Return JAX's deletion state when the device-array API exposes it."""

    is_deleted = getattr(value, "is_deleted", None)
    return bool(is_deleted()) if callable(is_deleted) else None


def _finish_host_staged_reconstruction(result, *accumulators):
    """Finish a host-staged reconstruction before dispatching the next half.

    Large RELION accumulators are moved to NumPy before reconstruction so the
    device only needs one half's inputs and FFT workspace at a time.  JAX
    dispatch is asynchronous, so wait here before the loop starts the other
    half; otherwise both padded FFT workspaces can overlap despite the host
    staging boundary.
    """

    if any(isinstance(accumulator, np.ndarray) for accumulator in accumulators):
        result.block_until_ready()
    return result
