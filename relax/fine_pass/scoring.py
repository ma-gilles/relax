"""Scoring helpers of the sparse pass 2.

RELION's accelerated-GPU fine diff2 lane reduction and pixel weights, the normalized cross-correlation operands,
RELION's powerClass noise operands and the full-to-compact window lookup the resident
scoring stage (``resident_scoring``) uses.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from relax.fourier.half_spectrum import bin_shell_values_jax, make_shell_indices_half
from relax.numerics.shells import shell_of_radius_sq

_RELION_FINE_DIFF2_FUSED_FFI_ENV = "RELAX_RELION_FINE_DIFF2_FUSED_FFI"


_RELION_CUDA_FINE_REF3D_BLOCK_SIZE = 256


def _relion_cuda_fine_reduce_lanes(lanes):
    """Reduce the 256 shared-memory lanes used by RELION CUDA REF3D fine diff2.

    Preserves the caller's dtype (promoted against float32 as a floor) instead
    of forcing float32, so a caller that has already promoted its operands to
    float64 (to match a ``ACC_DOUBLE_PRECISION`` RELION oracle) is not
    silently narrowed back down here.
    """

    lanes = jnp.asarray(lanes, dtype=jnp.result_type(lanes, jnp.float32))
    if lanes.shape[-1] != _RELION_CUDA_FINE_REF3D_BLOCK_SIZE:
        raise ValueError(
            "RELION CUDA fine reduction needs exactly "
            f"{_RELION_CUDA_FINE_REF3D_BLOCK_SIZE} lanes, got {lanes.shape[-1]}"
        )
    for width in (128, 64, 32, 16, 8, 4, 2, 1):
        lanes = lanes[..., :width] + lanes[..., width : 2 * width]
    return lanes[..., 0]


def _relion_cuda_fine_full_to_compact_lookup(image_shape, current_size, compact_indices):
    """Map RELION's full current-size packed pixel order to compact score rows."""

    box_size = int(image_shape[0])
    current_size = box_size if current_size is None else int(current_size)
    original_half_width = int(image_shape[1]) // 2 + 1
    current_half_width = current_size // 2 + 1
    compact_indices = np.asarray(compact_indices, dtype=np.int64).reshape(-1)
    centered_rows = compact_indices // original_half_width
    columns = compact_indices % original_half_width
    ky = centered_rows - box_size // 2
    fftw_rows = np.where(ky < 0, ky + current_size, ky)
    if np.any(fftw_rows < 0) or np.any(fftw_rows >= current_size):
        raise ValueError("compact score indices contain rows outside the RELION current-size crop")
    if np.any(columns < 0) or np.any(columns >= current_half_width):
        raise ValueError("compact score indices contain columns outside the RELION current-size crop")
    relion_flat_indices = fftw_rows * current_half_width + columns
    if np.unique(relion_flat_indices).size != relion_flat_indices.size:
        raise ValueError("compact score indices do not map uniquely into RELION's current-size layout")
    lookup = np.full(current_size * current_half_width, -1, dtype=np.int32)
    lookup[relion_flat_indices] = np.arange(compact_indices.size, dtype=np.int32)
    return lookup


def _relion_cuda_fine_normalized_cc_score(
    reference,
    shifted_score,
    score_weight,
    half_weights,
    relion_full_to_compact=None,
):
    """Reproduce RELION CUDA's 256-lane fine normalized-CC reduction.

    The pinned ``cuda_kernel_diff2_CC_fine<REF3D=true>`` accumulates numerator
    and reference norm over pixels ``tid + pass * 256`` in RELION's XFLOAT,
    then uses the same shared-memory tree as fine Gaussian ``diff2``. XFLOAT is
    float32 in RELION's default accelerated build but float64 whenever
    ``ACC_DOUBLE_PRECISION`` is set (our double-precision oracle build); this
    reduction follows the caller's operand dtype instead of hardcoding
    float32, so it matches whichever precision the RELION oracle actually
    used. RECOVAR stores the score window in centered compact order, so
    ``relion_full_to_compact`` restores RELION's packed current-size FFTW
    pixel order before accumulation.
    """

    complex_dtype = jnp.result_type(reference, shifted_score, jnp.complex64)
    real_dtype = jnp.result_type(score_weight, half_weights, jnp.float32)
    reference = jnp.asarray(reference, dtype=complex_dtype)
    shifted_score = jnp.asarray(shifted_score, dtype=complex_dtype)
    score_weight = jnp.asarray(score_weight, dtype=real_dtype)
    half_weights = jnp.asarray(half_weights, dtype=real_dtype)
    n_values = int(reference.shape[-1])
    if (
        shifted_score.shape[-1] != n_values
        or score_weight.shape[-1] != n_values
        or half_weights.shape != (n_values,)
    ):
        raise ValueError(
            "RELION CUDA fine normalized-CC operands must have the same pixel count: "
            f"reference={reference.shape[-1]}, shifted={shifted_score.shape[-1]}, "
            f"score_weight={score_weight.shape[-1]}, half_weights={half_weights.shape}"
        )
    numerator_shape = jnp.broadcast_shapes(
        reference.shape[:-1], shifted_score.shape[:-1], score_weight.shape[:-1]
    )
    norm_shape = jnp.broadcast_shapes(reference.shape[:-1], score_weight.shape[:-1])
    if n_values == 0:
        return jnp.full(numerator_shape, -jnp.inf, dtype=real_dtype)

    if relion_full_to_compact is None:
        relion_full_to_compact = jnp.arange(n_values, dtype=jnp.int32)
    else:
        relion_full_to_compact = jnp.asarray(relion_full_to_compact, dtype=jnp.int32)
        if relion_full_to_compact.ndim != 1:
            raise ValueError(
                "RELION full-to-compact lookup must be one-dimensional, got "
                f"{relion_full_to_compact.shape}"
            )

    full_image_size = int(relion_full_to_compact.shape[0])
    block_size = _RELION_CUDA_FINE_REF3D_BLOCK_SIZE
    n_passes = (full_image_size + block_size - 1) // block_size
    padded_size = n_passes * block_size
    relion_full_to_compact = jnp.pad(
        relion_full_to_compact,
        [(0, padded_size - full_image_size)],
        constant_values=-1,
    )
    numerator_lanes = jnp.zeros(numerator_shape + (block_size,), dtype=real_dtype)
    norm_lanes = jnp.zeros(norm_shape + (block_size,), dtype=real_dtype)

    def accumulate_pass(pass_index, lane_values):
        numerator, norm = lane_values
        start = pass_index * block_size
        compact_rows = jax.lax.dynamic_slice_in_dim(
            relion_full_to_compact, start, block_size, axis=-1
        )
        valid_pixel = compact_rows >= 0
        safe_rows = jnp.where(valid_pixel, compact_rows, 0)
        ref_pass = jnp.take(reference, safe_rows, axis=-1)
        shifted_pass = jnp.take(shifted_score, safe_rows, axis=-1)
        score_weight_pass = jnp.take(score_weight, safe_rows, axis=-1)
        half_weight_pass = jnp.take(half_weights, safe_rows, axis=-1)
        numerator_terms = (
            ref_pass.real * shifted_pass.real + ref_pass.imag * shifted_pass.imag
        ) * score_weight_pass * half_weight_pass
        norm_terms = (
            ref_pass.real * ref_pass.real + ref_pass.imag * ref_pass.imag
        ) * score_weight_pass * half_weight_pass
        zero = jnp.asarray(0.0, dtype=real_dtype)
        numerator_terms = jnp.where(valid_pixel, numerator_terms, zero)
        norm_terms = jnp.where(valid_pixel, norm_terms, zero)
        return numerator + numerator_terms, norm + norm_terms

    numerator_lanes, norm_lanes = jax.lax.fori_loop(
        0,
        n_passes,
        accumulate_pass,
        (numerator_lanes, norm_lanes),
    )
    numerator = _relion_cuda_fine_reduce_lanes(numerator_lanes)
    norm = _relion_cuda_fine_reduce_lanes(norm_lanes)
    return numerator / jnp.sqrt(
        jnp.maximum(norm, jnp.asarray(1e-30, dtype=real_dtype))
    )


def _relion_cuda_fine_pixel_weights(corr_img_score, half_weights):
    """Form RELION XFLOAT pixel weights in the active ACC precision."""

    real_dtype = jnp.result_type(corr_img_score, half_weights, jnp.float32)
    return jnp.asarray(corr_img_score, dtype=real_dtype) * jnp.asarray(
        half_weights, dtype=real_dtype
    )


def _relion_cuda_corr_img_from_rfloat_ctf(
    inverse_noise,
    ctf_rfloat,
    scale=None,
    *,
    output_dtype=jnp.float32,
):
    """Form XFLOAT ``corr_img`` after RELION's RFLOAT CTF square.

    The deployed mixed-precision build stores ``Minvsigma2`` and ``corr_img``
    as float32 (XFLOAT), but evaluates the CTF and ``CTF * CTF`` as float64
    (RFLOAT).  The compound multiplication promotes Minvsigma2 to float64 and
    casts the product back to float32 before the optional float32 scale square.
    """

    output_dtype = jnp.dtype(output_dtype)
    if output_dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
        raise TypeError(f"output_dtype must be float32 or float64, got {output_dtype}")
    inverse_noise_rfloat = jnp.asarray(inverse_noise, dtype=output_dtype).astype(jnp.float64)
    ctf_rfloat = jnp.asarray(ctf_rfloat, dtype=jnp.float64)
    ctf_squared_rfloat = jax.lax.optimization_barrier(ctf_rfloat * ctf_rfloat)
    corr_img = jax.lax.optimization_barrier(
        inverse_noise_rfloat * ctf_squared_rfloat
    ).astype(output_dtype)
    if scale is not None:
        scale = jnp.asarray(scale, dtype=output_dtype)
        scale_squared = jax.lax.optimization_barrier(scale * scale)
        corr_img = corr_img * scale_squared
    return corr_img


def _relion_cuda_corr_img_from_native_noise_variance(
    noise_variance,
    ctf_rfloat,
    image_shape,
    scale=None,
    *,
    output_dtype=jnp.float32,
):
    """Form score-unit ``corr_img`` with RELION's native-FFT cast order.

    RECOVAR stores the noise variance in its normalized-FFT units, larger than
    RELION's variance by ``N**4``.  Reciprocating that value into float32 and
    then applying the compensating Fourier scale is algebraically correct but
    changes ``Minvsigma2`` by one ULP on real parity fixtures.  RELION first
    reciprocates its native-unit binary64 variance into XFLOAT, forms the
    CTF-square product, and only then does RECOVAR need to convert the completed
    XFLOAT operand back to normalized-FFT score units. ``output_dtype`` selects
    the score dtype of that final conversion (float32 production, float64 for
    double-precision scoring); the RELION-side casts above it are unchanged.
    """

    output_dtype = jnp.dtype(output_dtype)
    if output_dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
        raise TypeError(f"output_dtype must be float32 or float64, got {output_dtype}")
    box_size = int(image_shape[0])
    native_fourier_scale_rfloat = jnp.asarray(box_size**4, dtype=jnp.float64)
    native_corr_img = _relion_cuda_native_corr_img_from_noise_variance(
        noise_variance,
        ctf_rfloat,
        image_shape,
        scale,
    )
    # XLA's float32 division may lower to a reciprocal multiply and differs
    # from correctly rounded division by one ULP.  This conversion is not a
    # RELION operation, so perform it in binary64 and cast once to preserve the
    # native XFLOAT operand under RECOVAR's Fourier normalization.
    return (
        native_corr_img.astype(jnp.float64) / native_fourier_scale_rfloat
    ).astype(output_dtype)


def _relion_cuda_native_corr_img_from_noise_variance(
    noise_variance,
    ctf_rfloat,
    image_shape,
    scale=None,
):
    """Return RELION's native-unit XFLOAT ``corr_img`` before FFT rescaling."""

    box_size = int(image_shape[0])
    if tuple(image_shape) != (box_size, box_size):
        raise ValueError(f"RELION corr_img requires a square image, got {image_shape}")
    native_fourier_scale_rfloat = jnp.asarray(box_size**4, dtype=jnp.float64)
    native_variance = (
        jnp.asarray(noise_variance, dtype=jnp.float64)
        / native_fourier_scale_rfloat
    )
    native_inverse_noise = jnp.reciprocal(native_variance).astype(jnp.float32)
    return _relion_cuda_corr_img_from_rfloat_ctf(
        native_inverse_noise,
        ctf_rfloat,
        scale,
    )


def _relion_native_fine_units_enabled(
    *,
    fresh_k1_guard,
    use_exact_relion_gaussian,
    use_float64_scoring,
    has_ctf_rfloat,
) -> bool:
    """Whether pass 2 scores its fine diff2 in RELION's native FFT units.

    A fresh K=1 exact-Gaussian float32 pass with RELION's RFLOAT CTF operand
    (final Q aa0eccbfd4). The compact engine and the device-resident driver
    both key on this one condition; ``fresh_k1_guard`` is the pass's
    ``source_faithful_spectrum_norm`` argument before any environment
    resolution.
    """

    return bool(
        fresh_k1_guard
        and use_exact_relion_gaussian
        and not use_float64_scoring
        and has_ctf_rfloat
    )


def _relion_native_score_corr_img(
    noise_variance_rows,
    ctf_rfloat,
    image_shape,
    scale=None,
    *,
    zero_dc: bool,
):
    """RELION's native-unit fine-score ``corr_img`` over the whole half spectrum.

    ``noise_variance_rows`` is ``[1, P]`` (one shared spectrum) or ``[B, P]``
    per-image rows, in RECOVAR units. The value is RELION's XFLOAT operand
    before any N**-4 conversion
    (:func:`_relion_cuda_native_corr_img_from_noise_variance`); with
    ``zero_dc`` (half-spectrum scoring) the origin is zero, as in RELION's
    ``Minvsigma2`` at the full box. The caller windows it to the score window.
    """

    corr_img = _relion_cuda_native_corr_img_from_noise_variance(
        noise_variance_rows,
        ctf_rfloat,
        image_shape,
        scale,
    )
    if zero_dc:
        dc_mask = make_shell_indices_half(image_shape) == 0
        corr_img = jnp.where(dc_mask[None, :], 0.0, corr_img).astype(jnp.float32)
    return corr_img


@partial(jax.jit, static_argnames=("fft_size",))
def _relion_native_fine_units(values, fft_size):
    """Convert a complex fine-score operand to RELION's unnormalised FFT units.

    RELION evaluates the fine diff2 in native units.  RECOVAR's shifted image
    and projected reference carry a factor ``fft_size = N**2`` and its score
    ``corr_img`` a factor ``N**-4``; the factors cancel over the reals but not
    in float32 pixel products when ``N**2`` is not a power of two.  Score with
    the complex operands divided by ``fft_size`` and with the native
    ``corr_img`` of :func:`_relion_cuda_native_corr_img_from_noise_variance`.

    XLA's complex64 division by a real scalar is not correctly rounded, and
    its bits change with the device and the fusion context.  Divide each real
    and imaginary part in binary64 and round once to float32 instead.  See
    ``docs/math/relion_refinement_algorithm.md`` section 3.
    """

    values = jnp.asarray(values, dtype=jnp.complex64)
    native_fft_scale = jnp.asarray(fft_size, dtype=jnp.float64)
    real = (jnp.real(values).astype(jnp.float64) / native_fft_scale).astype(jnp.float32)
    imag = (jnp.imag(values).astype(jnp.float64) / native_fft_scale).astype(jnp.float32)
    return jax.lax.complex(real, imag)


def _relion_cuda_pixel_correction_from_rfloat_ctf(
    scale,
    ctf_rfloat,
    *,
    output_dtype=jnp.float32,
):
    """Form RELION's XFLOAT score-image correction from an RFLOAT CTF."""

    output_dtype = jnp.dtype(output_dtype)
    if output_dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
        raise TypeError(f"output_dtype must be float32 or float64, got {output_dtype}")
    scale = jnp.asarray(scale, dtype=output_dtype)
    ctf_rfloat = jnp.asarray(ctf_rfloat, dtype=jnp.float64)
    pixel_correction = jax.lax.optimization_barrier(jnp.reciprocal(scale))
    corrected = jax.lax.optimization_barrier(
        pixel_correction.astype(jnp.float64) / ctf_rfloat
    ).astype(output_dtype)
    return jnp.where(jnp.abs(ctf_rfloat) > 1e-8, corrected, pixel_correction)


_RELION_CUDA_POWERCLASS_BLOCK_SIZE = 128


class _RelionPowerClassOperands(NamedTuple):
    """RELION ``powerClass`` operands of one flattened centred rfft image batch."""

    relion_image: jnp.ndarray
    real_dtype: object
    image_height: int
    image_width: int
    half_width: int
    resolution_limit: object
    shell: np.ndarray
    valid: np.ndarray


def _relion_powerclass_resolution_limit(current_size, runtime_current_size, *, image_width):
    """RELION's first high-resolution shell, ``current_size / 2 + 1`` (static or traced)."""

    if runtime_current_size is None:
        if current_size is None:
            current_size = image_width
        return int(current_size) // 2 + 1
    return jnp.asarray(runtime_current_size, dtype=jnp.int32) // 2 + 1


def _relion_powerclass_packed_image(processed_score_half, *, image_shape, dtype=None):
    """Repack centred rfft images into RELION's unshifted ``Faux`` layout.

    RELION's packed rows are ``0, +1, ..., +Nyquist, -Nyquist+1, ..., -1`` and
    its FFT amplitudes are smaller than RECOVAR's by the real-space pixel
    count. ``dtype`` forces the complex working precision (the native kernels
    take complex64); otherwise complex128 inputs keep double precision.
    Returns ``(relion_image, real_dtype, image_height, image_width, half_width)``.
    """

    image_height = int(image_shape[0])
    image_width = int(image_shape[1])
    if image_height != image_width:
        raise ValueError(f"RELION powerClass parity requires square images, got {image_shape}")
    half_width = image_width // 2 + 1
    if dtype is None:
        processed_score_half = jnp.asarray(processed_score_half)
        complex_dtype = (
            jnp.complex128
            if processed_score_half.dtype == jnp.dtype(jnp.complex128)
            else jnp.complex64
        )
        processed_score_half = processed_score_half.astype(complex_dtype)
    else:
        complex_dtype = dtype
        processed_score_half = jnp.asarray(processed_score_half, dtype=complex_dtype)
    real_dtype = jnp.float64 if complex_dtype == jnp.complex128 else jnp.float32
    if processed_score_half.ndim != 2 or processed_score_half.shape[-1] != image_height * half_width:
        raise ValueError(
            "RELION powerClass input must be flattened centred rfft images, got "
            f"{processed_score_half.shape} for image_shape={image_shape}"
        )
    relion_image = jnp.roll(
        processed_score_half.reshape((-1, image_height, half_width)),
        -(image_height // 2),
        axis=1,
    ).reshape((processed_score_half.shape[0], -1))
    relion_image = relion_image / jnp.asarray(image_height * image_width, dtype=real_dtype)
    return relion_image, real_dtype, image_height, image_width, half_width


def _relion_powerclass_operands(processed_score_half, *, image_shape, current_size, runtime_current_size):
    """RELION ``powerClass`` operands for the JAX reproductions.

    Besides the packed image this supplies the integer shell of every packed
    pixel (CUDA ``__float2int_rn(sqrtf(...))``, or the double variant under
    ``ACC_DOUBLE_PRECISION``) and the kernel's pixel validity: shells 1 to
    ``half_width - 1`` excluding the redundant negative-row DC column.
    """

    relion_image, real_dtype, image_height, image_width, half_width = _relion_powerclass_packed_image(
        processed_score_half, image_shape=image_shape
    )
    resolution_limit = _relion_powerclass_resolution_limit(
        current_size, runtime_current_size, image_width=image_width
    )
    rows = np.arange(image_height, dtype=np.int32)[:, None]
    columns = np.arange(half_width, dtype=np.int32)[None, :]
    signed_rows = np.where(rows < half_width, rows, rows - image_height)
    radius_squared = columns * columns + signed_rows * signed_rows
    shell_real_dtype = np.float64 if real_dtype == jnp.float64 else np.float32
    shell = shell_of_radius_sq(radius_squared, rule="half_even", real_dtype=shell_real_dtype, index_dtype=np.int32)
    valid = (
        (shell > 0)
        & (shell < half_width)
        & ~((columns == 0) & (signed_rows < 0))
    ).reshape(-1)
    return _RelionPowerClassOperands(
        relion_image, real_dtype, image_height, image_width, half_width, resolution_limit, shell, valid
    )


@partial(jax.jit, static_argnames=("image_shape", "current_size"))
def relion_cuda_powerclass_highres_xi2_half(
    processed_score_half,
    *,
    image_shape,
    current_size,
    runtime_current_size=None,
):
    """Reproduce the class-power high-resolution image tail used by fine diff2.

    RELION's CUDA ``powerClass`` kernel (``cuda_kernels/helper.cuh``) bins the
    unshifted, unnormalised ``Faux`` image in XFLOAT, reduces each contiguous
    128-pixel block with a shared-memory tree, and atomically accumulates bins
    at or above ``current_size / 2 + 1``. XFLOAT is float32 normally and
    float64 under ``ACC_DOUBLE_PRECISION``. ``diff2_fine`` then adds half of
    that scalar to every fine-search hypothesis (``cuda_kernels/diff2.cuh``).

    RECOVAR stores the y axis centred and its FFT amplitudes are larger by the
    real-space pixel count. Convert both conventions before reproducing the
    matching XFLOAT power and block reduction. The final cross-block
    accumulation uses ascending block order; RELION's atomic arrival order is
    not specified, so its last bit may vary between launches while the
    per-block arithmetic is fixed.
    """

    operands = _relion_powerclass_operands(
        processed_score_half,
        image_shape=image_shape,
        current_size=current_size,
        runtime_current_size=runtime_current_size,
    )
    relion_image = operands.relion_image
    real_dtype = operands.real_dtype
    valid = jnp.asarray(operands.valid) & (
        jnp.asarray(operands.shell.reshape(-1), dtype=jnp.int32) >= operands.resolution_limit
    )

    power = relion_image.real * relion_image.real
    power = jax.lax.optimization_barrier(power)
    power = power + relion_image.imag * relion_image.imag
    power = jnp.where(jnp.asarray(valid)[None, :], power, jnp.asarray(0.0, dtype=real_dtype))

    block_size = _RELION_CUDA_POWERCLASS_BLOCK_SIZE
    n_blocks = (power.shape[-1] + block_size - 1) // block_size
    power = jnp.pad(power, ((0, 0), (0, n_blocks * block_size - power.shape[-1])))
    block_lanes = power.reshape((power.shape[0], n_blocks, block_size))
    for width in (64, 32, 16, 8, 4, 2, 1):
        block_lanes = block_lanes[..., :width] + block_lanes[..., width : 2 * width]
        block_lanes = jax.lax.optimization_barrier(block_lanes)
    block_sums = block_lanes[..., 0]

    def add_block(block_index, total):
        total = total + block_sums[:, block_index]
        return jax.lax.optimization_barrier(total)

    highres_xi2 = jax.lax.fori_loop(
        0,
        n_blocks,
        add_block,
        jnp.zeros((relion_image.shape[0],), dtype=real_dtype),
    )
    return highres_xi2 * jnp.asarray(0.5, dtype=real_dtype)


def _relion_powerclass_highres_xi2_half_to_norm_units(highres_xi2_half, image_shape):
    """Convert RELION's half-Xi2 FFT units to RECOVAR norm N^4 units."""

    image_height = int(image_shape[0])
    image_width = int(image_shape[1])
    highres = jnp.asarray(highres_xi2_half)
    return _powerclass_norm_units_jit(highres, float((image_height * image_width) ** 2))


@partial(jax.jit, static_argnums=(1,))
def _powerclass_norm_units_jit(highres, pixel_count_sq: float):
    """Convert norm units in one program, preserving the rounding barrier."""
    highres = highres * jnp.asarray(2.0, dtype=highres.dtype)
    highres = jax.lax.optimization_barrier(highres)
    return highres * jnp.asarray(pixel_count_sq, dtype=highres.dtype)


@partial(jax.jit, static_argnames=("image_shape", "current_size"))
def _relion_cuda_powerclass_spectrum_highres_norm_units(
    processed_score_half,
    *,
    image_shape,
    current_size,
    runtime_current_size=None,
):
    """Reproduce the high-shell norm term from RELION's power spectrum.

    RELION's ``powerClass`` kernel produces two independently reduced values:
    a block-tree ``highres_Xi2`` scalar used by fine scoring, and an
    atomically binned shell spectrum.  Norm correction consumes the latter,
    summing its high shells sequentially in host RFLOAT.  These reductions are
    numerically distinct, so the fine-score scalar cannot be reused here.
    """

    operands = _relion_powerclass_operands(
        processed_score_half,
        image_shape=image_shape,
        current_size=current_size,
        runtime_current_size=runtime_current_size,
    )
    relion_image = operands.relion_image
    image_height, image_width, half_width = operands.image_height, operands.image_width, operands.half_width
    resolution_limit = operands.resolution_limit
    shell = np.where(operands.valid, operands.shell.reshape(-1), half_width).astype(np.int32)

    power = relion_image.real * relion_image.real
    power = jax.lax.optimization_barrier(power)
    power = power + relion_image.imag * relion_image.imag
    spectrum = jax.vmap(
        lambda row: bin_shell_values_jax(row, jnp.asarray(shell), half_width)
    )(power)

    # RELION copies the float32 spectrum to the host and adds the selected
    # shells into an RFLOAT accumulator in increasing shell order.
    def add_shell(shell_index, total):
        return total + spectrum[:, shell_index].astype(jnp.float64)

    high_shell = jax.lax.fori_loop(
        resolution_limit,
        half_width,
        add_shell,
        jnp.zeros((relion_image.shape[0],), dtype=jnp.float64),
    )
    return high_shell * jnp.asarray((image_height * image_width) ** 2, dtype=jnp.float64)


def relion_powerclass_noise_presence(
    *,
    use_exact_relion_gaussian,
    accumulate_noise,
    current_size,
):
    """Which ``powerClass`` terms a pass-2 batch will produce, from the flags alone.

    :func:`_relion_powerclass_noise_terms` decides this while it has the image
    array in hand. A caller that has to know before any image exists -- the
    compile-ahead warm-up, which describes the operand set to lower a program
    against -- asks here instead, so there is one statement of the rule rather
    than a copy that can fall out of step with it.

    Returns ``(has_highres_xi2, has_norm_high_shell)``.
    """

    has_xi2 = bool(use_exact_relion_gaussian or (accumulate_noise and current_size is not None))
    has_norm = bool(accumulate_noise and current_size is not None and has_xi2)
    return has_xi2, has_norm


def relion_powerclass_noise_dtypes(
    *,
    real_dtype,
    source_faithful_spectrum_norm,
):
    """Dtypes of the two ``powerClass`` terms, from the image dtype and the mode.

    ``highres_Xi2`` keeps the image's own real dtype. The norm high-shell term
    does too, *except* in source-faithful mode, where
    :func:`_relion_cuda_powerclass_spectrum_highres_norm_units` accumulates the
    selected shells in float64 to follow RELION's host-side sum. Production runs
    float32 images with source-faithful normalization on, so in production these
    two terms have different dtypes; predicting one from the other is wrong, and
    was, until the resident driver's own after-the-fact check said so on
    2026-09-20.

    Returns ``(highres_xi2_dtype, norm_high_shell_dtype)``.
    """

    xi2_dtype = jnp.dtype(real_dtype)
    if source_faithful_spectrum_norm:
        return xi2_dtype, jnp.dtype(jnp.float64)
    return xi2_dtype, xi2_dtype


def _relion_powerclass_noise_terms(
    processed_score_half_for_noise,
    *,
    image_shape,
    current_size,
    use_exact_relion_gaussian,
    accumulate_noise,
    source_faithful_spectrum_norm,
):
    """RELION ``powerClass`` terms one sparse pass-2 batch needs.

    Exact fine Gaussian scoring adds half of the block-tree ``highres_Xi2``
    to every hypothesis, and norm correction at a current size consumes the
    high-shell power: the atomically binned spectrum in source-faithful mode,
    otherwise the same ``highres_Xi2`` converted to RECOVAR's N^4 units
    (``ml_optimiser.cpp`` ``storeWeightedSums``). Returns
    ``(highres_xi2_half, norm_high_shell)`` with ``None`` for terms the batch
    does not need.
    """

    wants_xi2, _ = relion_powerclass_noise_presence(
        use_exact_relion_gaussian=use_exact_relion_gaussian,
        accumulate_noise=accumulate_noise,
        current_size=current_size,
    )
    # The current size reaches the kernels as a traced scalar, so one program per
    # box serves every current size (the first high shell is the only thing it
    # sets); the static key is the box.
    size_kwargs = (
        {"current_size": None}
        if current_size is None
        else {"current_size": int(image_shape[0]), "runtime_current_size": jnp.int32(int(current_size))}
    )
    relion_highres_xi2_half = None
    if wants_xi2:
        relion_highres_xi2_half = relion_cuda_powerclass_highres_xi2_half(
            processed_score_half_for_noise,
            image_shape=image_shape,
            **size_kwargs,
        )
    if accumulate_noise and current_size is not None and relion_highres_xi2_half is not None:
        if source_faithful_spectrum_norm:
            relion_norm_high_shell = _relion_cuda_powerclass_spectrum_highres_norm_units(
                processed_score_half_for_noise,
                image_shape=image_shape,
                **size_kwargs,
            )
        else:
            relion_norm_high_shell = _relion_powerclass_highres_xi2_half_to_norm_units(
                relion_highres_xi2_half,
                image_shape,
            )
    else:
        relion_norm_high_shell = None
    return relion_highres_xi2_half, relion_norm_high_shell


