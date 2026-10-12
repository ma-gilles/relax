"""EM-only symbols relocated from ``recovar.cuda_backproject`` (relax split P1, pure move).

Function bodies are AST-identical to their originals at Q (tag q-reconcile-20260923); only the module
path changed.  See pr179_coordination/relax_split_plan_20260923/PLAN.md.
"""
from __future__ import annotations

import functools
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from recovar.cuda_backproject import (
    custom_cuda_requested,
)
from recovar.cuda_backproject import (
    rot_to_compact as _rot_to_compact,
)

from relax.cuda import library


@functools.partial(jax.jit, static_argnums=(3,))
def relion_translate_score_f32(
    images: jax.Array,
    translation_angles: jax.Array,
    pixel_indices: jax.Array,
    image_shape: Tuple[int, int],
) -> jax.Array:
    """Translate score images with RELION's accelerated float32 arithmetic.

    ``translation_angles`` contains RELION's per-translation ``(tx, ty)``
    radians, ``[T, 2]`` shared by the batch or ``[B, T, 2]`` per image (one
    launch for a subtomogram's tilt images, each with its own phases).
    ``pixel_indices`` use RECOVAR's centered half-spectrum layout.
    The CUDA primitive evaluates ``sincosf(x*tx + y*ty)`` and the explicit
    real/imaginary products used by RELION's fine Gaussian scorer. The output
    is flattened in image-major, translation-major order to match
    :func:`apply_half_translation_phases`.

    This is a strict CUDA primitive. Callers that support a non-CUDA fallback
    must select it before invoking this function.
    """

    if images.dtype != jnp.complex64:
        raise TypeError(f"images must be complex64, got {images.dtype}")
    if translation_angles.dtype != jnp.float32:
        raise TypeError(
            f"translation_angles must be float32, got {translation_angles.dtype}"
        )
    if pixel_indices.dtype != jnp.int32:
        raise TypeError(f"pixel_indices must be int32, got {pixel_indices.dtype}")
    if images.ndim != 2:
        raise ValueError(f"images must have shape (batch, pixels), got {images.shape}")
    if not (
        (translation_angles.ndim == 2 and translation_angles.shape[1:] == (2,))
        or (translation_angles.ndim == 3 and translation_angles.shape[0] == images.shape[0]
            and translation_angles.shape[2:] == (2,))
    ):
        raise ValueError(
            "translation_angles must have shape (translations, 2) or (batch, translations, 2), got "
            f"{translation_angles.shape}"
        )
    if pixel_indices.shape != (images.shape[1],):
        raise ValueError(
            f"pixel_indices must have shape ({images.shape[1]},), got "
            f"{pixel_indices.shape}"
        )
    if len(image_shape) != 2 or any(int(size) <= 0 for size in image_shape):
        raise ValueError(f"image_shape must contain two positive sizes, got {image_shape}")
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION score translation requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError(
            "RELION score translation was explicitly requested but custom CUDA is disabled"
        )
    library._ensure_ffi()

    image_h, image_w = (int(size) for size in image_shape)
    half_width = image_w // 2 + 1
    out_type = jax.ShapeDtypeStruct(
        (
            images.shape[0] * translation_angles.shape[-2],
            images.shape[1],
        ),
        jnp.complex64,
    )
    return jax.ffi.ffi_call(
        library._TARGET_RELION_TRANSLATE_SCORE_F32,
        out_type,
        vmap_method="sequential",
    )(
        images,
        translation_angles,
        pixel_indices,
        image_h=np.int64(image_h),
        image_half_width=np.int64(half_width),
    )


@functools.partial(jax.jit, static_argnums=(3,))
def relion_translate_score_f64(
    images: jax.Array,
    translation_angles: jax.Array,
    pixel_indices: jax.Array,
    image_shape: Tuple[int, int],
) -> jax.Array:
    """Translate score/M-step images with RELION's CUDA double arithmetic."""

    if images.dtype != jnp.complex128:
        raise TypeError(f"images must be complex128, got {images.dtype}")
    if translation_angles.dtype != jnp.float64:
        raise TypeError(
            f"translation_angles must be float64, got {translation_angles.dtype}"
        )
    if pixel_indices.dtype != jnp.int32:
        raise TypeError(f"pixel_indices must be int32, got {pixel_indices.dtype}")
    if images.ndim != 2:
        raise ValueError(f"images must have shape (batch, pixels), got {images.shape}")
    if translation_angles.ndim != 2 or translation_angles.shape[1:] != (2,):
        raise ValueError(
            "translation_angles must have shape (translations, 2), got "
            f"{translation_angles.shape}"
        )
    if pixel_indices.shape != (images.shape[1],):
        raise ValueError(
            f"pixel_indices must have shape ({images.shape[1]},), got "
            f"{pixel_indices.shape}"
        )
    if len(image_shape) != 2 or any(int(size) <= 0 for size in image_shape):
        raise ValueError(f"image_shape must contain two positive sizes, got {image_shape}")
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION score translation requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError(
            "RELION score translation was explicitly requested but custom CUDA is disabled"
        )
    library._ensure_ffi()

    image_h, image_w = (int(size) for size in image_shape)
    half_width = image_w // 2 + 1
    out_type = jax.ShapeDtypeStruct(
        (
            images.shape[0] * translation_angles.shape[0],
            images.shape[1],
        ),
        jnp.complex128,
    )
    return jax.ffi.ffi_call(
        library._TARGET_RELION_TRANSLATE_SCORE_F64,
        out_type,
        vmap_method="sequential",
    )(
        images,
        translation_angles,
        pixel_indices,
        image_h=np.int64(image_h),
        image_half_width=np.int64(half_width),
    )


def _relion_posterior_rows_f32(target: str, operation: str, operand_name: str, values, operand):
    """One RELION float32 posterior kernel on a 1-D ``values`` and a float32 scalar ``operand`` (traced)."""

    if values.dtype != jnp.float32:
        raise TypeError(f"values must be float32, got {values.dtype}")
    if values.ndim != 1 or values.shape[0] < 1:
        raise ValueError(f"values must be a nonempty 1-D array, got {values.shape}")
    if operand.dtype != jnp.float32 or operand.ndim != 0:
        raise TypeError(f"{operand_name} must be a float32 scalar, got {operand.dtype} {operand.shape}")
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"RELION float32 {operation} requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError(f"RELION float32 {operation} was requested but custom CUDA is disabled")
    library._ensure_ffi()

    output_type = jax.ShapeDtypeStruct(values.shape, jnp.float32)
    return jax.ffi.ffi_call(
        target,
        output_type,
        vmap_method="sequential",
    )(values, operand)


def _relion_posterior_batched_f32(target: str, operation: str, operand_name: str, values, operand):
    """One RELION float32 posterior kernel on a ``values`` matrix and one float32 ``operand`` per row (traced)."""

    if values.dtype != jnp.float32 or values.ndim != 2:
        raise TypeError(f"values must be a float32 matrix, got {values.dtype} {values.shape}")
    if values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError(f"values must be nonempty, got {values.shape}")
    if operand.dtype != jnp.float32 or operand.shape != (values.shape[0],):
        raise TypeError(
            f"{operand_name} must be one float32 scalar per row, got "
            f"{operand.dtype} {operand.shape} for values {values.shape}"
        )
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"RELION batched float32 {operation} requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError(
            f"RELION batched float32 {operation} was requested but custom CUDA is disabled"
        )
    library._ensure_ffi()

    return jax.ffi.ffi_call(
        target,
        jax.ShapeDtypeStruct(values.shape, jnp.float32),
    )(values, operand)


@jax.jit
def relion_exponentiate_f32(values: jax.Array, add: jax.Array) -> jax.Array:
    """Apply RELION's fine-weight ``expf(value + add)`` CUDA kernel."""

    return _relion_posterior_rows_f32(library._TARGET_RELION_EXPONENTIATE_F32, "exponentiation", "add", values, add)


@jax.jit
def relion_exponentiate_batched_f32(
    values: jax.Array,
    add: jax.Array,
) -> jax.Array:
    """Apply RELION's float32 posterior exponentiation to a row batch."""

    return _relion_posterior_batched_f32(
        library._TARGET_RELION_EXPONENTIATE_BATCHED_F32, "exponentiation", "add", values, add
    )


@jax.jit
def relion_divide_f32(values: jax.Array, divisor: jax.Array) -> jax.Array:
    """Apply RELION's CUDA ``float / float`` posterior normalization."""

    return _relion_posterior_rows_f32(library._TARGET_RELION_DIVIDE_F32, "division", "divisor", values, divisor)


@jax.jit
def relion_divide_batched_f32(
    values: jax.Array,
    divisor: jax.Array,
) -> jax.Array:
    """Apply RELION float32 division to all posterior rows in one FFI call."""

    return _relion_posterior_batched_f32(
        library._TARGET_RELION_DIVIDE_BATCHED_F32, "division", "divisor", values, divisor
    )


@jax.jit
def relion_cub_sort_scan_f32(values: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Sort and inclusively scan one float32 vector with RELION's CUB calls.

    A test reference: relax's cuts use :func:`relion_coarse_cut_f32`.

    This is a strict diagnostic primitive for the coarse-significance boundary.
    RELION invokes ``cub::DeviceRadixSort::SortKeys`` followed by
    ``cub::DeviceScan::InclusiveSum`` on each particle's positive weights.
    Keeping both intermediate arrays observable lets parity tests identify a
    sort discrepancy separately from a scan discrepancy.
    """

    if values.dtype != jnp.float32:
        raise TypeError(f"values must be float32, got {values.dtype}")
    if values.ndim != 1 or values.shape[0] < 1:
        raise ValueError(f"values must be a nonempty 1-D array, got {values.shape}")
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION CUB sort/scan requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError("RELION CUB sort/scan was requested but custom CUDA is disabled")
    library._ensure_ffi()

    output_type = jax.ShapeDtypeStruct(values.shape, jnp.float32)
    return jax.ffi.ffi_call(
        library._TARGET_RELION_CUB_SORT_SCAN_F32,
        (output_type, output_type),
        vmap_method="sequential",
    )(values)


@jax.jit
def relion_cub_sort_scan_batched_f32(
    values: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Run the exact RELION CUB sort/scan row order in one FFI call.

    A test reference: relax's cuts use :func:`relion_coarse_cut_f32`.

    Rows remain serialized on the caller's XLA stream. The implementation
    reuses one stream-ordered scratch allocation, so this changes dispatch and
    allocation topology without changing any row's CUB arithmetic. Rows must be
    posterior weights (zero or positive, no NaN): only the positive weights are
    radix-sorted and the zeros are written in front of them.
    """

    if values.dtype != jnp.float32 or values.ndim != 2:
        raise TypeError(f"values must be a float32 matrix, got {values.dtype} {values.shape}")
    if values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError(f"values must be nonempty, got {values.shape}")
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION batched CUB sort/scan requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError(
            "RELION batched CUB sort/scan was requested but custom CUDA is disabled"
        )
    library._ensure_ffi()

    output_type = jax.ShapeDtypeStruct(values.shape, jnp.float32)
    return jax.ffi.ffi_call(
        library._TARGET_RELION_CUB_SORT_SCAN_BATCHED_F32,
        (output_type, output_type),
    )(values)


def relion_coarse_cut_f32(
    weights: jax.Array,
    *,
    adaptive_fraction: float,
    max_significants: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """RELION's coarse significance cut of posterior-weight rows, by radix select.

    ``weights`` are a batch of rows of zero or positive float32 weights. Returns,
    per row, the float32 sum of the positive weights, the cut weight (the first
    ascending positive weight whose cumulative sum exceeds RELION's float32 tail
    target ``(1 - adaptive_fraction) * sum``, cumulative sums in float64), and the
    number of positive samples from the cut upwards (RELION's pre-tie rank), with
    RELION's ``maximum_significants`` floor when ``max_significants > 0``. A row
    without positive weights returns zeros.
    """

    if weights.dtype != jnp.float32 or weights.ndim != 2:
        raise TypeError(f"weights must be a float32 matrix, got {weights.dtype} {weights.shape}")
    if weights.shape[0] < 1 or weights.shape[1] < 1:
        raise ValueError(f"weights must be nonempty, got {weights.shape}")
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION coarse cut requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError("RELION coarse cut was requested but custom CUDA is disabled")
    library._ensure_ffi()

    rows = weights.shape[0]
    return jax.ffi.ffi_call(
        library._TARGET_RELION_COARSE_CUT_F32,
        (
            jax.ShapeDtypeStruct((rows,), jnp.float32),
            jax.ShapeDtypeStruct((rows,), jnp.float32),
            jax.ShapeDtypeStruct((rows,), jnp.int32),
        ),
    )(weights, fraction=np.float32(adaptive_fraction), maxsig=np.int64(max_significants))


def _require_sparse_pass2_cuda_backend(label: str) -> None:
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"{label} requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError(f"{label} was requested but custom CUDA is disabled")


# Byte size of the device-side per-row state record of SparsePass2PosteriorF32.
_SPARSE_PASS2_ROW_STATE_BYTES = 40


# Byte size of the device-side per-segment state record of
# SparsePass2SegmentedPosteriorF32: the rectangular RowState plus the segment
# extent (valid flag, explicit padding, begin, cell count).
_SPARSE_PASS2_SEGMENT_STATE_BYTES = _SPARSE_PASS2_ROW_STATE_BYTES + 24


def _sparse_pass2_segment_geometry(
    scores: jax.Array, segment_offsets: jax.Array, n_valid_images: jax.Array
) -> tuple[int, int]:
    """Validate a segmented pass-2 operand set and return ``(cells, segments)``.

    ``scores`` holds ``cells`` candidate scores in image order; image ``i`` owns
    ``scores.reshape(-1)[segment_offsets[i]:segment_offsets[i + 1]]``.  Offsets
    are cell indices, must be nondecreasing and must stay within ``[0, cells]``;
    cells covered by no segment are treated as padding.
    """

    if scores.dtype != jnp.float32:
        raise TypeError(f"scores must be float32, got {scores.dtype}")
    if scores.ndim < 1 or any(int(d) < 1 for d in scores.shape):
        raise ValueError(f"scores must be a nonempty array, got {scores.shape}")
    cells = int(np.prod(scores.shape))
    if cells > np.iinfo(np.int32).max:
        raise ValueError(f"scores {scores.shape} exceed the int32 cell-offset range")
    if segment_offsets.dtype != jnp.int32:
        raise TypeError(f"segment_offsets must be int32, got {segment_offsets.dtype}")
    if segment_offsets.ndim != 1 or int(segment_offsets.shape[0]) < 2:
        raise ValueError(
            "segment_offsets must have shape (n_segments + 1,) with n_segments >= 1, "
            f"got {segment_offsets.shape}"
        )
    segments = int(segment_offsets.shape[0]) - 1
    if n_valid_images.dtype != jnp.int32:
        raise TypeError(f"n_valid_images must be int32, got {n_valid_images.dtype}")
    if int(np.prod(n_valid_images.shape)) != 1:
        raise ValueError(
            f"n_valid_images must hold exactly one value, got {n_valid_images.shape}"
        )
    return cells, segments


@jax.jit
def sparse_pass2_segmented_log_z_f64(
    scores: jax.Array,
    segment_offsets: jax.Array,
    n_valid_images: jax.Array,
) -> jax.Array:
    """Per-segment float64 log-sum-exp of flat float32 sparse pass-2 scores.

    Segmented form of :func:`sparse_pass2_log_z_f64`: segments without a finite
    score, empty segments and segments at or beyond ``n_valid_images`` return
    ``-inf``, which is what the rectangular handler returns for an all ``-inf``
    row.  Flattening a rectangular row into a segment reproduces that row's
    value bitwise.  This handler runs entirely on the device.
    """

    _, segments = _sparse_pass2_segment_geometry(scores, segment_offsets, n_valid_images)
    _require_sparse_pass2_cuda_backend("Segmented sparse pass-2 log-Z")
    library._ensure_optional_ffi(library._TARGET_SPARSE_PASS2_SEGMENTED_LOG_Z_F64)
    return jax.ffi.ffi_call(
        library._TARGET_SPARSE_PASS2_SEGMENTED_LOG_Z_F64,
        jax.ShapeDtypeStruct((segments,), jnp.float64),
    )(scores, segment_offsets, n_valid_images.astype(jnp.int32).reshape(()))


@functools.partial(
    jax.jit,
    static_argnames=(
        "adaptive_fraction",
        "keep_all",
        "use_external_sum_weight",
        "return_scratch",
    ),
)
def sparse_pass2_segmented_posterior_f32(
    scores: jax.Array,
    segment_offsets: jax.Array,
    n_valid_images: jax.Array,
    log_z: jax.Array,
    external_sum_weight: jax.Array,
    *,
    adaptive_fraction: float,
    keep_all: bool,
    use_external_sum_weight: bool,
    return_scratch: bool = False,
) -> tuple[jax.Array, ...]:
    """Segmented form of :func:`sparse_pass2_posterior_f32`.

    ``scores`` is a flat cell array whose segment ``i`` is
    ``[segment_offsets[i], segment_offsets[i + 1])``; the per-segment operands
    and outputs replace the rectangular per-row ones.  Returns ``(log_z,
    best_log_score, best_cell_index, max_posterior, probs, normalized_weights,
    reconstruction_probs, mask, n_significant, sum_weight, threshold)``.
    ``best_cell_index`` is relative to the segment, so it equals the rectangular
    ``best_argmax`` of the row a segment was flattened from; add
    ``segment_offsets[i]`` for the flat cell.  Per-cell outputs keep the shape
    of ``scores``; cells covered by no segment, empty segments and segments at
    or beyond ``n_valid_images`` carry the values the rectangular handler
    produces for an all ``-inf`` row.

    Launch geometry is capacity-only: every kernel is launched at the static
    segment count from ``segment_offsets.shape`` and ``n_valid_images`` is read on
    the device alone, so nothing on the host depends on the chunk's occupancy and
    the handler has no host round trip. The significance boundary is RELION's cut
    with float64 cumulative sums, by radix select, one block per segment (the
    definition of :func:`relion_coarse_cut_f32`).

    ``return_scratch`` appends the handler's raw weights to the result.
    """

    _, segments = _sparse_pass2_segment_geometry(
        scores, segment_offsets, n_valid_images
    )
    if log_z.dtype != jnp.float64 or log_z.shape != (segments,):
        raise TypeError(
            f"log_z must be float64 with shape ({segments},), got {log_z.dtype} {log_z.shape}"
        )
    if external_sum_weight.dtype != jnp.float32 or external_sum_weight.shape != (segments,):
        raise TypeError(
            f"external_sum_weight must be float32 with shape ({segments},), got "
            f"{external_sum_weight.dtype} {external_sum_weight.shape}"
        )
    if type(keep_all) is not bool or type(use_external_sum_weight) is not bool:
        raise TypeError("keep_all and use_external_sum_weight must be static Python bools")
    if type(return_scratch) is not bool:
        raise TypeError("return_scratch must be a static Python bool")
    _require_sparse_pass2_cuda_backend("Segmented sparse pass-2 posterior")
    library._ensure_optional_ffi(library._TARGET_SPARSE_PASS2_SEGMENTED_POSTERIOR_F32)
    full = scores.shape
    outputs = (
        jax.ShapeDtypeStruct((segments,), jnp.float64),  # log_z
        jax.ShapeDtypeStruct((segments,), jnp.float32),  # best_log_score
        jax.ShapeDtypeStruct((segments,), jnp.int64),    # best_cell_index
        jax.ShapeDtypeStruct((segments,), jnp.float32),  # max_posterior
        jax.ShapeDtypeStruct(full, jnp.float64),         # probs
        jax.ShapeDtypeStruct(full, jnp.float32),         # normalized_weights
        jax.ShapeDtypeStruct(full, jnp.float32),         # reconstruction_probs
        jax.ShapeDtypeStruct(full, jnp.bool_),           # mask
        jax.ShapeDtypeStruct((segments,), jnp.int32),    # n_significant
        jax.ShapeDtypeStruct((segments,), jnp.float32),  # sum_weight
        jax.ShapeDtypeStruct((segments,), jnp.float32),  # threshold
        jax.ShapeDtypeStruct(full, jnp.float32),         # raw weights (scratch)
        jax.ShapeDtypeStruct(
            (segments, _SPARSE_PASS2_SEGMENT_STATE_BYTES), jnp.uint8
        ),                                               # segment state
    )
    result = jax.ffi.ffi_call(library._TARGET_SPARSE_PASS2_SEGMENTED_POSTERIOR_F32, outputs)(
        scores,
        segment_offsets,
        n_valid_images.astype(jnp.int32).reshape(()),
        log_z,
        external_sum_weight,
        adaptive_fraction=np.float32(adaptive_fraction),
        keep_all=np.int64(keep_all),
        use_external_sum_weight=np.int64(use_external_sum_weight),
    )
    return tuple(result[:12] if return_scratch else result[:11])


@functools.partial(jax.jit, static_argnums=(4,))
def relion_translate_bpref_f32(
    images: jax.Array,
    weighted_ctf: jax.Array,
    translation_angles: jax.Array,
    pixel_indices: jax.Array,
    image_shape: Tuple[int, int],
) -> jax.Array:
    """Translate raw images, then apply BPref weights in RELION float32 order.

    RELION evaluates ``sincosf`` and the explicit complex translation before
    multiplying each translated real/imaginary component by ``weighted_ctf``.
    Keeping this as one CUDA primitive avoids the extra rounding boundary in
    RECOVAR's algebraically equivalent weight-then-translate path.
    """

    if images.dtype != jnp.complex64:
        raise TypeError(f"images must be complex64, got {images.dtype}")
    if weighted_ctf.dtype != jnp.float32:
        raise TypeError(f"weighted_ctf must be float32, got {weighted_ctf.dtype}")
    if translation_angles.dtype != jnp.float32:
        raise TypeError(
            f"translation_angles must be float32, got {translation_angles.dtype}"
        )
    if pixel_indices.dtype != jnp.int32:
        raise TypeError(f"pixel_indices must be int32, got {pixel_indices.dtype}")
    if images.ndim != 2:
        raise ValueError(f"images must have shape (batch, pixels), got {images.shape}")
    if weighted_ctf.shape != images.shape:
        raise ValueError(
            f"weighted_ctf must have shape {images.shape}, got {weighted_ctf.shape}"
        )
    if translation_angles.ndim != 2 or translation_angles.shape[1:] != (2,):
        raise ValueError(
            "translation_angles must have shape (translations, 2), got "
            f"{translation_angles.shape}"
        )
    if pixel_indices.shape != (images.shape[1],):
        raise ValueError(
            f"pixel_indices must have shape ({images.shape[1]},), got "
            f"{pixel_indices.shape}"
        )
    if len(image_shape) != 2 or any(int(size) <= 0 for size in image_shape):
        raise ValueError(f"image_shape must contain two positive sizes, got {image_shape}")
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION BPref translation requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError(
            "RELION BPref translation was explicitly requested but custom CUDA is disabled"
        )
    library._ensure_ffi()

    image_h, image_w = (int(size) for size in image_shape)
    half_width = image_w // 2 + 1
    out_type = jax.ShapeDtypeStruct(
        (
            images.shape[0] * translation_angles.shape[0],
            images.shape[1],
        ),
        jnp.complex64,
    )
    return jax.ffi.ffi_call(
        library._TARGET_RELION_TRANSLATE_BPREF_F32,
        out_type,
        vmap_method="sequential",
    )(
        images,
        weighted_ctf,
        translation_angles,
        pixel_indices,
        image_h=np.int64(image_h),
        image_half_width=np.int64(half_width),
    )


@functools.partial(jax.jit, static_argnums=(4,))
def relion_translate_bpref_f64(
    images: jax.Array,
    weighted_ctf: jax.Array,
    translation_angles: jax.Array,
    pixel_indices: jax.Array,
    image_shape: Tuple[int, int],
) -> jax.Array:
    """Translate raw images, then apply BPref weights in RELION double order."""

    if images.dtype != jnp.complex128:
        raise TypeError(f"images must be complex128, got {images.dtype}")
    if weighted_ctf.dtype != jnp.float64:
        raise TypeError(f"weighted_ctf must be float64, got {weighted_ctf.dtype}")
    if translation_angles.dtype != jnp.float64:
        raise TypeError(
            f"translation_angles must be float64, got {translation_angles.dtype}"
        )
    if pixel_indices.dtype != jnp.int32:
        raise TypeError(f"pixel_indices must be int32, got {pixel_indices.dtype}")
    if images.ndim != 2:
        raise ValueError(f"images must have shape (batch, pixels), got {images.shape}")
    if weighted_ctf.shape != images.shape:
        raise ValueError(
            f"weighted_ctf must have shape {images.shape}, got {weighted_ctf.shape}"
        )
    if translation_angles.ndim != 2 or translation_angles.shape[1:] != (2,):
        raise ValueError(
            "translation_angles must have shape (translations, 2), got "
            f"{translation_angles.shape}"
        )
    if pixel_indices.shape != (images.shape[1],):
        raise ValueError(
            f"pixel_indices must have shape ({images.shape[1]},), got "
            f"{pixel_indices.shape}"
        )
    if len(image_shape) != 2 or any(int(size) <= 0 for size in image_shape):
        raise ValueError(f"image_shape must contain two positive sizes, got {image_shape}")
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION BPref translation requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError(
            "RELION BPref translation was explicitly requested but custom CUDA is disabled"
        )
    library._ensure_ffi()

    image_h, image_w = (int(size) for size in image_shape)
    half_width = image_w // 2 + 1
    out_type = jax.ShapeDtypeStruct(
        (
            images.shape[0] * translation_angles.shape[0],
            images.shape[1],
        ),
        jnp.complex128,
    )
    return jax.ffi.ffi_call(
        library._TARGET_RELION_TRANSLATE_BPREF_F64,
        out_type,
        vmap_method="sequential",
    )(
        images,
        weighted_ctf,
        translation_angles,
        pixel_indices,
        image_h=np.int64(image_h),
        image_half_width=np.int64(half_width),
    )


def _validate_relion_fine_diff2_inputs(
    reference: jax.Array,
    shifted_image: jax.Array,
    weight: jax.Array,
    full_to_compact: jax.Array,
    *,
    real_dtype=jnp.float32,
) -> None:
    real_dtype = jnp.dtype(real_dtype)
    complex_dtype = jnp.dtype(jnp.complex128 if real_dtype == jnp.float64 else jnp.complex64)
    if reference.dtype != complex_dtype:
        raise TypeError(f"reference must be {complex_dtype.name}, got {reference.dtype}")
    if shifted_image.dtype != complex_dtype:
        raise TypeError(f"shifted_image must be {complex_dtype.name}, got {shifted_image.dtype}")
    if weight.dtype != real_dtype:
        raise TypeError(f"weight must be {real_dtype.name}, got {weight.dtype}")
    if full_to_compact.dtype != jnp.int32:
        raise TypeError(
            f"full_to_compact must be int32, got {full_to_compact.dtype}"
        )
    if full_to_compact.ndim != 1 or full_to_compact.shape[0] <= 0:
        raise ValueError(
            "full_to_compact must be a nonempty rank-1 array, got "
            f"{full_to_compact.shape}"
        )
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION fine diff2 requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError(
            "RELION fine diff2 was explicitly requested but custom CUDA is disabled"
        )
    library._ensure_ffi()


@jax.jit
def relion_coarse_diff2_rectangular_f32(
    reference: jax.Array,
    shifted_image: jax.Array,
    weight: jax.Array,
    initial_diff2: jax.Array,
    full_to_compact: jax.Array,
) -> jax.Array:
    """Evaluate RELION's coarse Gaussian CUDA reduction topology.

    Shapes are ``reference=(R,N)``, ``shifted_image=(B,T,N)``,
    ``weight=(B,N)``, ``initial_diff2=(B,)``, and
    ``full_to_compact=(F,)``. The kernel uses RELION's 128-thread,
    16-orientation blocks and its native per-translation lane assignment. The
    lane partials are combined with CUDA atomics on top of the supplied
    high-resolution image term, as in RELION's coarse scorer.
    """

    _validate_relion_fine_diff2_inputs(
        reference,
        shifted_image,
        weight,
        full_to_compact,
    )
    if initial_diff2.dtype != jnp.float32:
        raise TypeError(
            f"initial_diff2 must be float32, got {initial_diff2.dtype}"
        )
    if reference.ndim != 2 or shifted_image.ndim != 3 or weight.ndim != 2:
        raise ValueError(
            "rectangular coarse diff2 expects reference rank 2, shifted rank "
            f"3, and weight rank 2, got {reference.shape}, "
            f"{shifted_image.shape}, {weight.shape}"
        )
    if (
        shifted_image.shape[0] != weight.shape[0]
        or reference.shape[1] != shifted_image.shape[2]
        or reference.shape[1] != weight.shape[1]
        or reference.shape[0] <= 0
        or reference.shape[1] <= 0
        or shifted_image.shape[0] <= 0
        or shifted_image.shape[1] <= 0
        or shifted_image.shape[1] > 128
        or initial_diff2.shape != (shifted_image.shape[0],)
    ):
        raise ValueError(
            "rectangular coarse diff2 operands have inconsistent shapes or "
            "more than 128 translations: "
            f"{reference.shape}, {shifted_image.shape}, {weight.shape}"
        )
    out_type = jax.ShapeDtypeStruct(
        (shifted_image.shape[0], reference.shape[0], shifted_image.shape[1]),
        jnp.float32,
    )
    return jax.ffi.ffi_call(
        library._TARGET_RELION_COARSE_DIFF2_RECTANGULAR_F32,
        out_type,
        vmap_method="sequential",
    )(reference, shifted_image, weight, initial_diff2, full_to_compact)


@jax.jit
def relion_coarse_normalized_cc_pairs_f32(
    shifted_image: jax.Array,
    score_weight: jax.Array,
    reference: jax.Array,
    half_weights: jax.Array,
    packed_to_compact: jax.Array,
) -> jax.Array:
    """Evaluate bounded coarse normalized-CC pairs with RELION CUDA arithmetic.

    The first three operands have shape ``(B,C,N)`` and the output has shape
    ``(B,C)``. One 128-thread CUDA block evaluates each candidate, preserving
    RELION's operand contraction, lane tree, ``sqrtf``, and 128 identical
    atomic additions. ``packed_to_compact`` restores current-size FFTW order.
    """

    for name, value, dtype in (
        ("shifted_image", shifted_image, jnp.complex64),
        ("score_weight", score_weight, jnp.float32),
        ("reference", reference, jnp.complex64),
        ("half_weights", half_weights, jnp.float32),
        ("packed_to_compact", packed_to_compact, jnp.int32),
    ):
        if value.dtype != dtype:
            raise TypeError(f"{name} must be {dtype}, got {value.dtype}")
    if (
        shifted_image.ndim != 3
        or score_weight.shape != shifted_image.shape
        or reference.shape != shifted_image.shape
        or half_weights.shape != (shifted_image.shape[-1],)
        or packed_to_compact.ndim != 1
        or packed_to_compact.shape[0] <= 0
        or shifted_image.shape[0] <= 0
        or shifted_image.shape[1] <= 0
        or shifted_image.shape[2] <= 0
    ):
        raise ValueError(
            "RELION coarse normalized-CC pairs have inconsistent shapes: "
            f"{shifted_image.shape}, {score_weight.shape}, {reference.shape}, "
            f"{half_weights.shape}, {packed_to_compact.shape}",
        )
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION coarse normalized-CC pairs require a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError(
            "RELION coarse normalized-CC pairs require the custom CUDA extension",
        )
    library._ensure_ffi()
    out_type = jax.ShapeDtypeStruct(shifted_image.shape[:-1], jnp.float32)
    return jax.ffi.ffi_call(
        library._TARGET_RELION_COARSE_NORMALIZED_CC_PAIRS_F32,
        out_type,
        vmap_method="sequential",
    )(
        shifted_image,
        score_weight,
        reference,
        half_weights,
        packed_to_compact,
    )


@functools.partial(jax.jit, static_argnums=(6, 7, 8, 9))
def relion_coarse_normalized_cc_native_texture_pairs_f32(
    projector_full: jax.Array,
    rotation_matrices: jax.Array,
    shifted_image: jax.Array,
    score_weight: jax.Array,
    half_weights: jax.Array,
    packed_to_compact: jax.Array,
    current_size: int,
    padding_factor: int,
    projector_max_r: int,
    return_components: bool = False,
    translation_angles: jax.Array | None = None,
    numerator_weight: jax.Array | None = None,
) -> jax.Array:
    """Evaluate bounded normalized-CC pairs from RELION's CUDA texture.

    Each row is one candidate. Its projection is sampled from
    ``projector_full`` inside the same 128-thread CUDA block that contracts the
    normalized-CC numerator and reference norm. When ``translation_angles`` is
    supplied, the image is translated in that same block with RELION's
    coarse-kernel ``sincosf`` arithmetic. This avoids materializing a RECOVAR
    projection or translated image at a different arithmetic boundary.
    """

    projector_full = jnp.asarray(projector_full)
    rotation_matrices = jnp.asarray(rotation_matrices)
    shifted_image = jnp.asarray(shifted_image)
    score_weight = jnp.asarray(score_weight)
    half_weights = jnp.asarray(half_weights)
    packed_to_compact = jnp.asarray(packed_to_compact)
    if translation_angles is None:
        translation_angles = jnp.zeros(
            (shifted_image.shape[0], 2),
            dtype=jnp.float32,
        )
    else:
        translation_angles = jnp.asarray(translation_angles)
    if numerator_weight is None:
        numerator_weight = jnp.ones_like(score_weight, dtype=jnp.float32)
    else:
        numerator_weight = jnp.asarray(numerator_weight)
    for name, value, dtype in (
        ("projector_full", projector_full, jnp.complex64),
        ("rotation_matrices", rotation_matrices, jnp.float32),
        ("shifted_image", shifted_image, jnp.complex64),
        ("score_weight", score_weight, jnp.float32),
        ("half_weights", half_weights, jnp.float32),
        ("packed_to_compact", packed_to_compact, jnp.int32),
        ("translation_angles", translation_angles, jnp.float32),
        ("numerator_weight", numerator_weight, jnp.float32),
    ):
        if value.dtype != dtype:
            raise TypeError(f"{name} must be {dtype}, got {value.dtype}")
    if (
        projector_full.ndim != 3
        or projector_full.shape[0] <= 0
        or projector_full.shape[1:]
        != (projector_full.shape[0], projector_full.shape[0])
        or rotation_matrices.ndim != 3
        or rotation_matrices.shape[1:] != (3, 3)
        or rotation_matrices.shape[0] <= 0
        or shifted_image.ndim != 2
        or shifted_image.shape[0] != rotation_matrices.shape[0]
        or shifted_image.shape[1] <= 0
        or score_weight.shape != shifted_image.shape
        or translation_angles.shape != (shifted_image.shape[0], 2)
        or numerator_weight.shape != shifted_image.shape
        or half_weights.shape != (shifted_image.shape[1],)
        or packed_to_compact.shape
        != (int(current_size) * (int(current_size) // 2 + 1),)
        or int(current_size) <= 0
        or int(padding_factor) <= 0
        or int(projector_max_r) <= 0
    ):
        raise ValueError(
            "native texture normalized-CC pair operands have inconsistent shapes: "
            f"projector={projector_full.shape}, rotations={rotation_matrices.shape}, "
            f"image={shifted_image.shape}, weight={score_weight.shape}, "
            f"half_weights={half_weights.shape}, lookup={packed_to_compact.shape}",
        )
    if jax.default_backend() != "gpu":
        raise RuntimeError(
            "RELION native texture normalized-CC pairs require a JAX GPU backend"
        )
    if not custom_cuda_requested():
        raise RuntimeError(
            "RELION native texture normalized-CC pairs require the custom CUDA extension"
        )
    library._ensure_ffi()
    eulers = jnp.swapaxes(rotation_matrices, -1, -2).reshape(
        rotation_matrices.shape[0], 9
    )
    out_type = jax.ShapeDtypeStruct((shifted_image.shape[0], 3), jnp.float32)
    components = jax.ffi.ffi_call(
        library._TARGET_RELION_COARSE_NORMALIZED_CC_NATIVE_TEXTURE_PAIRS_F32,
        out_type,
        vmap_method="sequential",
    )(
        projector_full,
        eulers,
        shifted_image,
        translation_angles,
        score_weight,
        numerator_weight,
        half_weights,
        packed_to_compact,
        current_size=np.int64(current_size),
        padding_factor=np.int64(padding_factor),
        projector_max_r=np.int64(projector_max_r),
    )
    return components if return_components else components[:, 0]


def _prepare_relion_coarse_diff2_projector_f32(
    projector_full: jax.Array,
    rotation_matrices: jax.Array,
    images: jax.Array,
    translation_angles: jax.Array,
    weight: jax.Array,
    initial_diff2: jax.Array,
    full_to_compact: jax.Array,
    *,
    current_size: int,
    physical_image_size: int,
    model_max_r: int,
) -> tuple[jax.Array, jax.ShapeDtypeStruct]:
    """Validate and prepare the one shared fused coarse-projector ABI."""
    if projector_full.dtype != jnp.complex64 or projector_full.ndim != 3:
        raise TypeError(
            "projector_full must be a rank-3 complex64 array, got "
            f"{projector_full.shape} {projector_full.dtype}"
        )
    if images.dtype != jnp.complex64 or images.ndim != 2:
        raise TypeError(
            f"images must be rank-2 complex64, got {images.shape} {images.dtype}"
        )
    if rotation_matrices.dtype != jnp.float32 or rotation_matrices.ndim != 3:
        raise TypeError(
            "rotation_matrices must be rank-3 float32, got "
            f"{rotation_matrices.shape} {rotation_matrices.dtype}"
        )
    if rotation_matrices.shape[1:] != (3, 3):
        raise ValueError(
            f"rotation_matrices must end in (3, 3), got {rotation_matrices.shape}"
        )
    if translation_angles.dtype != jnp.float32 or translation_angles.ndim != 2:
        raise TypeError(
            "translation_angles must be rank-2 float32, got "
            f"{translation_angles.shape} {translation_angles.dtype}"
        )
    if translation_angles.shape[1] != 2 or translation_angles.shape[0] > 128:
        raise ValueError(
            "translation_angles must have shape (T, 2) with T <= 128, got "
            f"{translation_angles.shape}"
        )
    if weight.dtype != jnp.float32 or weight.shape != images.shape:
        raise TypeError(
            f"weight must be float32 with shape {images.shape}, got {weight.shape} {weight.dtype}"
        )
    if initial_diff2.dtype != jnp.float32 or initial_diff2.shape != (images.shape[0],):
        raise TypeError(
            "initial_diff2 must be float32 with one value per image, got "
            f"{initial_diff2.shape} {initial_diff2.dtype}"
        )
    if full_to_compact.dtype != jnp.int32 or full_to_compact.shape != (
        int(current_size) * (int(current_size) // 2 + 1),
    ):
        raise TypeError(
            "full_to_compact must be the current-size packed int32 lookup, got "
            f"{full_to_compact.shape} {full_to_compact.dtype}"
        )
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION fused coarse projector requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError(
            "RELION fused coarse projector was requested but custom CUDA is disabled"
        )
    library._ensure_ffi()
    compact_rotations = _rot_to_compact(rotation_matrices, jnp.float32)
    out_type = jax.ShapeDtypeStruct(
        (images.shape[0], rotation_matrices.shape[0], translation_angles.shape[0]),
        jnp.float32,
    )
    return compact_rotations, out_type


def _validate_relion_coarse_single_lane_canonical(
    translation_count: int,
    *,
    canonical_reduction: bool,
    single_lane_canonical: bool,
) -> None:
    """Fail closed unless one CUDA thread owns each coarse translation."""

    if not single_lane_canonical:
        return
    if not canonical_reduction:
        raise ValueError(
            "single_lane_canonical=True requires canonical_reduction=True",
        )
    if not 65 <= int(translation_count) <= 128:
        raise ValueError(
            "single_lane_canonical=True requires 65--128 translations, got "
            f"{translation_count}",
        )


def _validate_relion_coarse_prehalf_weight(
    *,
    canonical_reduction: bool,
    single_lane_canonical: bool,
    prehalf_weight: bool,
) -> None:
    """Restrict the prehalved-weight experiment to native atomics."""

    if prehalf_weight and (canonical_reduction or single_lane_canonical):
        raise ValueError(
            "prehalf_weight=True requires canonical_reduction=False and single_lane_canonical=False",
        )


@functools.partial(
    jax.jit,
    static_argnames=(
        "current_size",
        "physical_image_size",
        "model_max_r",
        "padding_factor",
        "canonical_reduction",
        "single_lane_canonical",
        "prehalf_weight",
    ),
)
def relion_coarse_diff2_projector_f32(
    projector_full: jax.Array,
    rotation_matrices: jax.Array,
    images: jax.Array,
    translation_angles: jax.Array,
    weight: jax.Array,
    initial_diff2: jax.Array,
    full_to_compact: jax.Array,
    *,
    current_size: int,
    physical_image_size: int,
    model_max_r: int,
    padding_factor: int = 1,
    canonical_reduction: bool = False,
    single_lane_canonical: bool = False,
    prehalf_weight: bool = False,
) -> jax.Array:
    """Run the parity-locked shared RELION fused coarse projector.

    ``canonical_reduction`` replaces the final atomic lane admission with a
    fixed lane-index sum.  It retains the same projector, interpolation,
    translation, pixel traversal, and per-lane arithmetic and is intended for
    source-order parity qualification at marginal adaptive cutoffs.

    ``single_lane_canonical`` is a default-off compile-time specialization for
    65--128 translations, where one CUDA thread is the sole contributor to
    each translation.  It preserves the canonical initial-plus-lane addition
    while omitting the generic lane staging buffer.

    ``prehalf_weight`` is a default-off native-atomic experiment that matches
    RELION's coarse source order by staging ``weight * 0.5`` once per pixel
    instead of multiplying every orientation's square sum by ``0.5``.  It is
    not supported by either canonical-reduction specialization.
    """

    _validate_relion_coarse_single_lane_canonical(
        translation_angles.shape[0],
        canonical_reduction=canonical_reduction,
        single_lane_canonical=single_lane_canonical,
    )
    _validate_relion_coarse_prehalf_weight(
        canonical_reduction=canonical_reduction,
        single_lane_canonical=single_lane_canonical,
        prehalf_weight=prehalf_weight,
    )

    compact_rotations, out_type = _prepare_relion_coarse_diff2_projector_f32(
        projector_full,
        rotation_matrices,
        images,
        translation_angles,
        weight,
        initial_diff2,
        full_to_compact,
        current_size=current_size,
        physical_image_size=physical_image_size,
        model_max_r=model_max_r,
    )
    return jax.ffi.ffi_call(
        library._TARGET_RELION_COARSE_DIFF2_PROJECTOR_F32,
        out_type,
        vmap_method="sequential",
    )(
        projector_full,
        compact_rotations,
        images,
        translation_angles,
        weight,
        initial_diff2,
        full_to_compact,
        current_size=np.int64(current_size),
        physical_image_size=np.int64(physical_image_size),
        model_max_r=np.int64(model_max_r),
        padding_factor=np.int64(int(padding_factor)),
        canonical_reduction=np.int64(bool(canonical_reduction)),
        single_lane_canonical=np.int64(bool(single_lane_canonical)),
        prehalf_weight=np.int64(bool(prehalf_weight)),
    )


@functools.partial(
    jax.jit,
    static_argnames=("current_size", "physical_image_size", "model_max_r", "padding_factor"),
)
def relion_coarse_diff2_projector_per_image_f32(
    projector_full: jax.Array,
    rotation_matrices: jax.Array,
    images: jax.Array,
    translation_angles: jax.Array,
    weight: jax.Array,
    initial_diff2: jax.Array,
    full_to_compact: jax.Array,
    *,
    current_size: int,
    physical_image_size: int,
    model_max_r: int,
    padding_factor: int = 1,
) -> jax.Array:
    """:func:`relion_coarse_diff2_projector_f32` with one pose set per image, canonical reduction.

    Image ``b`` is scored with its own rotations ``rotation_matrices[b]`` ``[R, 3, 3]`` and phases
    ``translation_angles[b]`` ``[T, 2]`` (``T <= 128``), as a subtomogram's tilt images each have their
    own ``Aproj``; returns ``[B, R, T]``. Each image's diff2 is the per-image call's (same kernel
    arithmetic and lane reduction); the texture is staged once and all images run in one launch.
    """

    n_images, n_rotations = int(rotation_matrices.shape[0]), int(rotation_matrices.shape[1])
    if rotation_matrices.shape[2:] != (3, 3) or translation_angles.ndim != 3 or translation_angles.shape[2] != 2:
        raise ValueError(
            f"per-image poses are [B, R, 3, 3] and [B, T, 2], got {rotation_matrices.shape} {translation_angles.shape}"
        )
    if images.shape[0] != n_images or translation_angles.shape[0] != n_images:
        raise ValueError("rotations, phases and images must have one entry per image")
    compact_rotations, _ = _prepare_relion_coarse_diff2_projector_f32(
        projector_full,
        rotation_matrices.reshape(n_images * n_rotations, 3, 3),
        images,
        translation_angles[0],
        weight,
        initial_diff2,
        full_to_compact,
        current_size=current_size,
        physical_image_size=physical_image_size,
        model_max_r=model_max_r,
    )
    return jax.ffi.ffi_call(
        library._TARGET_RELION_COARSE_DIFF2_PROJECTOR_PER_IMAGE_F32,
        jax.ShapeDtypeStruct((n_images, n_rotations, int(translation_angles.shape[1])), jnp.float32),
        vmap_method="sequential",
    )(
        projector_full,
        compact_rotations.reshape(n_images, n_rotations, 6),
        images,
        translation_angles,
        weight,
        initial_diff2,
        full_to_compact,
        current_size=np.int64(current_size),
        physical_image_size=np.int64(physical_image_size),
        model_max_r=np.int64(model_max_r),
        padding_factor=np.int64(int(padding_factor)),
    )


def _validate_relion_fine_rectangular_shapes(reference, shifted_image, weight):
    """Validate the rectangular candidate layout shared by both precisions."""
    if reference.ndim != 3 or shifted_image.ndim != 3 or weight.ndim != 2:
        raise ValueError(
            "rectangular fine diff2 expects reference/shifted rank 3 and "
            f"weight rank 2, got {reference.shape}, {shifted_image.shape}, "
            f"{weight.shape}"
        )
    if (
        reference.shape[0] != shifted_image.shape[0]
        or reference.shape[0] != weight.shape[0]
        or reference.shape[2] != shifted_image.shape[2]
        or reference.shape[2] != weight.shape[1]
        or reference.shape[0] <= 0
        or reference.shape[1] <= 0
        or shifted_image.shape[1] <= 0
        or reference.shape[2] <= 0
    ):
        raise ValueError(
            "rectangular fine diff2 operands have inconsistent shapes: "
            f"{reference.shape}, {shifted_image.shape}, {weight.shape}"
        )


def _validate_relion_fine_pairs_shapes(reference, shifted_image, weight):
    """Validate the pairs candidate layout shared by both precisions."""
    if reference.ndim != 3 or shifted_image.ndim != 3 or weight.ndim != 2:
        raise ValueError(
            "pair fine diff2 expects reference/shifted rank 3 and weight rank "
            f"2, got {reference.shape}, {shifted_image.shape}, {weight.shape}"
        )
    if (
        reference.shape != shifted_image.shape
        or reference.shape[0] != weight.shape[0]
        or reference.shape[2] != weight.shape[1]
        or reference.shape[0] <= 0
        or reference.shape[1] <= 0
        or reference.shape[2] <= 0
    ):
        raise ValueError(
            "pair fine diff2 operands have inconsistent shapes: "
            f"{reference.shape}, {shifted_image.shape}, {weight.shape}"
        )


@jax.jit
def relion_fine_diff2_rectangular_f32(
    reference: jax.Array,
    shifted_image: jax.Array,
    weight: jax.Array,
    full_to_compact: jax.Array,
    initial_diff2: jax.Array | None = None,
) -> jax.Array:
    """Evaluate RELION's SM80 fine Gaussian tree on a rectangular grid.

    Shapes are ``reference=(B,R,N)``, ``shifted_image=(B,T,N)``,
    ``weight=(B,N)``, ``full_to_compact=(F,)``, and ``initial_diff2=(B,)``.
    One CUDA block evaluates each output hypothesis ``(B,R,T)`` with the
    production 256-lane topology, explicit binary32 FMA rounding boundaries,
    and RELION's final in-kernel ``sum + sum_init`` operation.
    """

    _validate_relion_fine_diff2_inputs(
        reference,
        shifted_image,
        weight,
        full_to_compact,
    )
    _validate_relion_fine_rectangular_shapes(reference, shifted_image, weight)
    if initial_diff2 is None:
        initial_diff2 = jnp.zeros((reference.shape[0],), dtype=jnp.float32)
    else:
        initial_diff2 = jnp.asarray(initial_diff2)
    if initial_diff2.dtype != jnp.float32 or initial_diff2.shape != (
        reference.shape[0],
    ):
        raise ValueError(
            "rectangular fine diff2 initial_diff2 must be float32 with shape "
            f"({reference.shape[0]},), got {initial_diff2.shape} {initial_diff2.dtype}"
        )
    out_type = jax.ShapeDtypeStruct(
        (reference.shape[0], reference.shape[1], shifted_image.shape[1]),
        jnp.float32,
    )
    return jax.ffi.ffi_call(
        library._TARGET_RELION_FINE_DIFF2_RECTANGULAR_F32,
        out_type,
        vmap_method="sequential",
    )(reference, shifted_image, weight, initial_diff2, full_to_compact)


def relion_fine_diff2_rectangular_masked_supported() -> bool:
    """Return whether the loaded library exports the masked fine diff2 target.

    The masked kernel is an optional ABI, so a library built before it exists
    stays loadable; callers fall back to the unmasked rectangular kernel.
    """

    try:
        library._ensure_ffi()
        symbol_name = library._OPTIONAL_FFI_REGISTRATIONS[
            library._TARGET_RELION_FINE_DIFF2_RECTANGULAR_MASKED_F32
        ][0]
        return getattr(library._get_lib(), symbol_name, None) is not None
    except Exception:
        return False


@jax.jit
def relion_fine_diff2_rectangular_masked_f32(
    reference: jax.Array,
    shifted_image: jax.Array,
    weight: jax.Array,
    full_to_compact: jax.Array,
    candidate_mask: jax.Array,
    initial_diff2: jax.Array | None = None,
) -> jax.Array:
    """Rectangular fine diff2 that skips cells whose candidate mask is False.

    Same operands and output layout as :func:`relion_fine_diff2_rectangular_f32`
    plus ``candidate_mask=(B,R,T)`` bool.  Masked cells return 0 without any
    pixel work, mirroring RELION's fine pass, which only evaluates significant
    (orientation, translation) pairs (``makeJobsForDiff2Fine``).  Valid cells
    are bitwise identical to the unmasked kernel.  Optional FFI target: it
    requires a library built with ``RelionFineDiff2RectangularMaskedF32``.
    """

    _validate_relion_fine_diff2_inputs(
        reference,
        shifted_image,
        weight,
        full_to_compact,
    )
    _validate_relion_fine_rectangular_shapes(reference, shifted_image, weight)
    candidate_mask = jnp.asarray(candidate_mask)
    expected = (reference.shape[0], reference.shape[1], shifted_image.shape[1])
    if candidate_mask.shape != expected or candidate_mask.dtype != jnp.bool_:
        raise ValueError(
            "masked rectangular fine diff2 candidate_mask must be bool with shape "
            f"{expected}, got {candidate_mask.shape} {candidate_mask.dtype}"
        )
    if initial_diff2 is None:
        initial_diff2 = jnp.zeros((reference.shape[0],), dtype=jnp.float32)
    else:
        initial_diff2 = jnp.asarray(initial_diff2)
    if initial_diff2.dtype != jnp.float32 or initial_diff2.shape != (
        reference.shape[0],
    ):
        raise ValueError(
            "masked rectangular fine diff2 initial_diff2 must be float32 with shape "
            f"({reference.shape[0]},), got {initial_diff2.shape} {initial_diff2.dtype}"
        )
    library._ensure_optional_ffi(library._TARGET_RELION_FINE_DIFF2_RECTANGULAR_MASKED_F32)
    out_type = jax.ShapeDtypeStruct(expected, jnp.float32)
    return jax.ffi.ffi_call(
        library._TARGET_RELION_FINE_DIFF2_RECTANGULAR_MASKED_F32,
        out_type,
        vmap_method="sequential",
    )(reference, shifted_image, weight, initial_diff2, full_to_compact, candidate_mask)


def _prepare_relion_fine_diff2_fused_translate_flat_rows_operands(
    reference: jax.Array,
    row_image_ids: jax.Array,
    image: jax.Array,
    translation_angles: jax.Array,
    weight: jax.Array,
    full_to_compact: jax.Array,
    initial_diff2: jax.Array | None = None,
) -> tuple[jax.Array, ...]:
    reference = jnp.asarray(reference)
    row_image_ids = jnp.asarray(row_image_ids)
    image = jnp.asarray(image)
    translation_angles = jnp.asarray(translation_angles)
    weight = jnp.asarray(weight)
    full_to_compact = jnp.asarray(full_to_compact)
    if reference.dtype != jnp.complex64 or image.dtype != jnp.complex64:
        raise TypeError(
            "flat-row RELION fine diff2 reference/image must be complex64, got "
            f"{reference.dtype} and {image.dtype}"
        )
    if row_image_ids.dtype != jnp.int32:
        raise TypeError(
            "flat-row RELION fine diff2 row_image_ids must be int32, got "
            f"{row_image_ids.dtype}"
        )
    if translation_angles.dtype != jnp.float32 or weight.dtype != jnp.float32:
        raise TypeError(
            "flat-row RELION fine diff2 angles/weight must be float32, got "
            f"{translation_angles.dtype} and {weight.dtype}"
        )
    if full_to_compact.dtype != jnp.int32:
        raise TypeError(
            "flat-row RELION fine diff2 lookup must be int32, got "
            f"{full_to_compact.dtype}"
        )
    if (
        reference.ndim != 2
        or row_image_ids.shape != (reference.shape[0],)
        or image.ndim != 2
        or translation_angles.ndim not in (2, 3)
        or translation_angles.shape[-1] != 2
        # [T, 2] shared, or [B, T, 2] one table per image (tilt images).
        or (translation_angles.ndim == 3 and translation_angles.shape[0] != image.shape[0])
        or weight.shape != image.shape
        or reference.shape[1] != image.shape[1]
        or reference.shape[0] <= 0
        or reference.shape[1] <= 0
        or image.shape[0] <= 0
        or translation_angles.shape[-2] <= 0
    ):
        raise ValueError(
            "flat-row RELION fine diff2 operands have inconsistent shapes: "
            f"{reference.shape}, {row_image_ids.shape}, {image.shape}, "
            f"{translation_angles.shape}, {weight.shape}, {full_to_compact.shape}"
        )
    if initial_diff2 is None:
        initial_diff2 = jnp.zeros((image.shape[0],), dtype=jnp.float32)
    else:
        initial_diff2 = jnp.asarray(initial_diff2)
    if initial_diff2.dtype != jnp.float32 or initial_diff2.shape != (
        image.shape[0],
    ):
        raise ValueError(
            "flat-row RELION fine diff2 initial_diff2 must be float32 with shape "
            f"({image.shape[0]},), got {initial_diff2.shape} {initial_diff2.dtype}"
        )
    return (
        reference,
        row_image_ids,
        image,
        translation_angles,
        weight,
        full_to_compact,
        initial_diff2,
    )


@jax.jit
def relion_fine_diff2_fused_translate_runtime_flat_rows_f32(
    reference: jax.Array,
    row_image_ids: jax.Array,
    image: jax.Array,
    translation_angles: jax.Array,
    weight: jax.Array,
    full_to_compact: jax.Array,
    logical_current_size: jax.Array,
    initial_diff2: jax.Array | None = None,
    translation_chunk_live: jax.Array | None = None,
) -> jax.Array:
    """Evaluate packed rotation rows inside a fixed physical pixel capacity.

    The scalar logical size controls the exact RELION rectangle traversal;
    physical-only score rows remain inert.

    ``translation_chunk_live`` (bool ``[rows, ceil(T / 4)]``, see
    :func:`relion_fine_diff2_translation_chunk_live`) names the chunks of four
    translations that hold a candidate cell; the kernel skips the others and
    writes their cells as ``+inf``. A traversed chunk computes each of its
    translations as without it, so the cells of live chunks are unchanged.
    """

    (
        reference,
        row_image_ids,
        image,
        translation_angles,
        weight,
        full_to_compact,
        initial_diff2,
    ) = _prepare_relion_fine_diff2_fused_translate_flat_rows_operands(
        reference,
        row_image_ids,
        image,
        translation_angles,
        weight,
        full_to_compact,
        initial_diff2,
    )
    logical_current_size = jnp.asarray(logical_current_size, dtype=jnp.int32)
    if logical_current_size.shape != ():
        raise ValueError(
            "runtime flat-row RELION fine diff2 logical_current_size must be a scalar, "
            f"got {logical_current_size.shape}"
        )
    if jax.default_backend() != "gpu":
        raise RuntimeError(
            "runtime flat-row fused RELION fine diff2 requires a JAX GPU backend"
        )
    if not custom_cuda_requested():
        raise RuntimeError(
            "runtime flat-row fused RELION fine diff2 was explicitly requested "
            "but custom CUDA is disabled"
        )
    library._ensure_ffi()

    out_type = jax.ShapeDtypeStruct(
        (reference.shape[0], translation_angles.shape[-2]),
        jnp.float32,
    )
    operands = (
        reference,
        row_image_ids,
        image,
        translation_angles,
        weight,
        initial_diff2,
        full_to_compact,
        logical_current_size,
    )
    if translation_chunk_live is None:
        return jax.ffi.ffi_call(
            library._TARGET_RELION_FINE_DIFF2_FUSED_TRANSLATE_RUNTIME_FLAT_ROWS_F32,
            out_type,
            vmap_method="sequential",
        )(*operands)
    expected_live_shape = (
        reference.shape[0],
        # [T, 2] shared or [B, T, 2] per image (tilt images): T is the second-to-last axis.
        -(-int(translation_angles.shape[-2]) // RELION_FINE_DIFF2_TRANSLATION_CHUNK),
    )
    if tuple(translation_chunk_live.shape) != expected_live_shape:
        raise ValueError(
            f"translation_chunk_live must have shape {expected_live_shape}, "
            f"got {tuple(translation_chunk_live.shape)}"
        )
    library._ensure_optional_ffi(library._TARGET_RELION_FINE_DIFF2_FUSED_TRANSLATE_RUNTIME_MASKED_FLAT_ROWS_F32)
    return jax.ffi.ffi_call(
        library._TARGET_RELION_FINE_DIFF2_FUSED_TRANSLATE_RUNTIME_MASKED_FLAT_ROWS_F32,
        out_type,
        vmap_method="sequential",
    )(*operands, jnp.asarray(translation_chunk_live, dtype=jnp.uint8))


# Translations per block of the fused fine-diff2 kernels (kRelionFineDiff2Ref3dJobChunk).
RELION_FINE_DIFF2_TRANSLATION_CHUNK = 4


def relion_fine_diff2_translation_chunk_live(candidate_mask: jax.Array) -> jax.Array:
    """Which chunks of four translations of each row hold a candidate cell.

    ``candidate_mask`` is bool ``[rows, T]``; the result is bool
    ``[rows, ceil(T / 4)]``, the kernels' translation blocking.
    """

    candidate_mask = jnp.asarray(candidate_mask, dtype=bool)
    n_rows, n_translations = candidate_mask.shape
    chunk = RELION_FINE_DIFF2_TRANSLATION_CHUNK
    n_chunks = -(-int(n_translations) // chunk)
    padded = jnp.pad(candidate_mask, ((0, 0), (0, n_chunks * chunk - int(n_translations))))
    return jnp.any(padded.reshape(n_rows, n_chunks, chunk), axis=-1)


@jax.jit
def relion_fine_diff2_pairs_f32(
    reference: jax.Array,
    shifted_image: jax.Array,
    weight: jax.Array,
    full_to_compact: jax.Array,
) -> jax.Array:
    """Evaluate RELION's SM80 fine Gaussian tree for compact candidate pairs.

    Shapes are ``reference=(B,P,N)``, ``shifted_image=(B,P,N)``,
    ``weight=(B,N)``, and ``full_to_compact=(F,)``; output is ``(B,P)``.
    """

    _validate_relion_fine_diff2_inputs(
        reference,
        shifted_image,
        weight,
        full_to_compact,
    )
    _validate_relion_fine_pairs_shapes(reference, shifted_image, weight)
    out_type = jax.ShapeDtypeStruct(reference.shape[:2], jnp.float32)
    return jax.ffi.ffi_call(
        library._TARGET_RELION_FINE_DIFF2_PAIRS_F32,
        out_type,
        vmap_method="sequential",
    )(reference, shifted_image, weight, full_to_compact)


@jax.jit
def relion_fine_diff2_rectangular_f64(
    reference: jax.Array,
    shifted_image: jax.Array,
    weight: jax.Array,
    full_to_compact: jax.Array,
) -> jax.Array:
    """ACC_DOUBLE_PRECISION specialization of the rectangular fine tree."""

    _validate_relion_fine_diff2_inputs(
        reference, shifted_image, weight, full_to_compact,
        real_dtype=jnp.float64,
    )
    _validate_relion_fine_rectangular_shapes(reference, shifted_image, weight)
    out_type = jax.ShapeDtypeStruct(
        (reference.shape[0], reference.shape[1], shifted_image.shape[1]),
        jnp.float64,
    )
    return jax.ffi.ffi_call(
        library._TARGET_RELION_FINE_DIFF2_RECTANGULAR_F64,
        out_type,
        vmap_method="sequential",
    )(reference, shifted_image, weight, full_to_compact)


@jax.jit
def relion_fine_diff2_pairs_f64(
    reference: jax.Array,
    shifted_image: jax.Array,
    weight: jax.Array,
    full_to_compact: jax.Array,
) -> jax.Array:
    """ACC_DOUBLE_PRECISION specialization of the compact-pair fine tree."""

    _validate_relion_fine_diff2_inputs(
        reference, shifted_image, weight, full_to_compact,
        real_dtype=jnp.float64,
    )
    _validate_relion_fine_pairs_shapes(reference, shifted_image, weight)
    out_type = jax.ShapeDtypeStruct(reference.shape[:2], jnp.float64)
    return jax.ffi.ffi_call(
        library._TARGET_RELION_FINE_DIFF2_PAIRS_F64,
        out_type,
        vmap_method="sequential",
    )(reference, shifted_image, weight, full_to_compact)


@jax.jit
def relion_wavg_rotation_atomic_runtime_triplet_add_f32(
    terms: jax.Array,
    accumulator: jax.Array,
    logical_pixel_count: jax.Array,
) -> jax.Array:
    """Run Wavg atomics over a logical prefix of fixed-capacity storage."""

    library._ensure_ffi()
    terms = jnp.asarray(terms)
    accumulator = jnp.asarray(accumulator)
    logical_pixel_count = jnp.asarray(logical_pixel_count, dtype=jnp.int32)
    if terms.dtype != jnp.float32 or terms.ndim != 4 or terms.shape[-1] != 3:
        raise ValueError(
            "runtime RELION Wavg triplets require float32 [batch,rotation,pixel,3] terms"
        )
    if accumulator.dtype != jnp.float32 or accumulator.shape != (
        terms.shape[0],
        terms.shape[2],
        3,
    ):
        raise ValueError(
            "runtime RELION Wavg triplets require a matching [batch,pixel,3] accumulator"
        )
    if logical_pixel_count.shape != ():
        raise ValueError("logical_pixel_count must be an int32 scalar")
    output_type = jax.ShapeDtypeStruct(accumulator.shape, jnp.float32)
    return jax.ffi.ffi_call(
        library._TARGET_RELION_WAVG_ROTATION_ATOMIC_RUNTIME_TRIPLET_ADD_F32,
        output_type,
        input_output_aliases={1: 0},
        vmap_method="sequential",
    )(terms, accumulator, logical_pixel_count)


@jax.jit
def relion_wavg_sequential_runtime_flat_rows_triplet_f32(
    projections: jax.Array,
    row_image_ids: jax.Array,
    raw_ctf: jax.Array,
    scale: jax.Array,
    shifted_images: jax.Array,
    posterior: jax.Array,
    logical_pixel_count: jax.Array,
) -> jax.Array:
    """Accumulate Wavg triplets over packed candidate rows.

    ``projections`` is ``(Q,P)`` and ``posterior`` is ``(Q,T)``; every row's
    image/CTF/scale/shifted-image address comes from ``row_image_ids`` ``(Q,)``
    instead of the rectangular ``[B,R]`` grid, the same substitution the
    flat-row fine scorer makes.  The CUDA implementation shares the rectangular
    kernel's per-pixel binary32 arithmetic and its translation-storage order, so
    a flattened rectangular problem returns bitwise identical triplets.  Rows
    whose id is negative are padding: they read nothing and stay zero.  The
    output is ``(Q,P,3)`` holding ``[XA, AA, diff2]``.
    """

    projections = jnp.asarray(projections)
    row_image_ids = jnp.asarray(row_image_ids)
    raw_ctf = jnp.asarray(raw_ctf)
    scale = jnp.asarray(scale)
    shifted_images = jnp.asarray(shifted_images)
    posterior = jnp.asarray(posterior)
    logical_pixel_count = jnp.asarray(logical_pixel_count, dtype=jnp.int32)
    if projections.dtype != jnp.complex64 or projections.ndim != 2:
        raise ValueError(
            "flat-row RELION Wavg expects complex64 projections[Q,P]"
        )
    row_count, pixel_capacity = projections.shape
    if row_image_ids.dtype != jnp.int32 or row_image_ids.shape != (row_count,):
        raise ValueError("flat-row RELION Wavg expects int32 row_image_ids[Q]")
    if raw_ctf.dtype != jnp.float32 or raw_ctf.ndim != 2 or (
        raw_ctf.shape[1] != pixel_capacity
    ):
        raise ValueError("flat-row RELION Wavg expects float32 raw_ctf[B,P]")
    batch_size = raw_ctf.shape[0]
    if scale.dtype != jnp.float32 or scale.shape != (batch_size,):
        raise ValueError("flat-row RELION Wavg expects float32 scale[B]")
    if (
        shifted_images.dtype != jnp.complex64
        or shifted_images.ndim != 3
        or shifted_images.shape[0] != batch_size
        or shifted_images.shape[2] != pixel_capacity
    ):
        raise ValueError(
            "flat-row RELION Wavg expects complex64 shifted_images[B,T,P]"
        )
    if posterior.dtype != jnp.float32 or posterior.shape != (
        row_count,
        shifted_images.shape[1],
    ):
        raise ValueError("flat-row RELION Wavg expects float32 posterior[Q,T]")
    if logical_pixel_count.shape != ():
        raise ValueError("logical_pixel_count must be an int32 scalar")
    if jax.default_backend() != "gpu":
        raise RuntimeError("flat-row RELION Wavg requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError("flat-row RELION Wavg requires custom CUDA")
    library._ensure_optional_ffi(
        library._TARGET_RELION_WAVG_SEQUENTIAL_RUNTIME_FLAT_ROWS_TRIPLET_F32
    )
    output_type = jax.ShapeDtypeStruct((row_count, pixel_capacity, 3), jnp.float32)
    return jax.ffi.ffi_call(
        library._TARGET_RELION_WAVG_SEQUENTIAL_RUNTIME_FLAT_ROWS_TRIPLET_F32,
        output_type,
        vmap_method="sequential",
    )(
        projections,
        row_image_ids,
        raw_ctf,
        scale,
        shifted_images,
        posterior,
        logical_pixel_count,
    )


@jax.jit
def relion_wavg_rotation_atomic_runtime_flat_rows_triplet_add_f32(
    terms: jax.Array,
    row_image_ids: jax.Array,
    accumulator: jax.Array,
    logical_pixel_count: jax.Array,
) -> jax.Array:
    """Atomically add packed-row Wavg triplets into per-image accumulators.

    ``terms`` is ``(Q,P,3)``, ``row_image_ids`` ``(Q,)`` and ``accumulator``
    ``(B,P,3)``.  The launch keeps one block per row, so a flattened
    rectangular problem issues exactly the same multiset of per-cell atomic
    adds as :func:`relion_wavg_rotation_atomic_runtime_triplet_add_f32`, and
    the linear block index matches that kernel's ``rotation + image *
    rotation_count``.  The order in which those adds land is hardware
    scheduled, so the accumulated float32 sums agree bitwise only when the
    summands are exactly representable; the rectangular kernel does not
    reproduce itself bitwise either at realistic rotation counts.  Rows whose
    id is negative contribute nothing.
    """

    terms = jnp.asarray(terms)
    row_image_ids = jnp.asarray(row_image_ids)
    accumulator = jnp.asarray(accumulator)
    logical_pixel_count = jnp.asarray(logical_pixel_count, dtype=jnp.int32)
    if terms.dtype != jnp.float32 or terms.ndim != 3 or terms.shape[-1] != 3:
        raise ValueError(
            "flat-row RELION Wavg atomics require float32 [row, pixel, 3] terms"
        )
    row_count, pixel_capacity = terms.shape[0], terms.shape[1]
    if row_image_ids.dtype != jnp.int32 or row_image_ids.shape != (row_count,):
        raise ValueError(
            "flat-row RELION Wavg atomics require int32 row_image_ids[Q]"
        )
    if (
        accumulator.dtype != jnp.float32
        or accumulator.ndim != 3
        or accumulator.shape[1] != pixel_capacity
        or accumulator.shape[2] != 3
    ):
        raise ValueError(
            "flat-row RELION Wavg atomics require a float32 [batch,pixel,3] accumulator"
        )
    if logical_pixel_count.shape != ():
        raise ValueError("logical_pixel_count must be an int32 scalar")
    library._ensure_optional_ffi(
        library._TARGET_RELION_WAVG_ROTATION_ATOMIC_RUNTIME_FLAT_ROWS_TRIPLET_ADD_F32
    )
    output_type = jax.ShapeDtypeStruct(accumulator.shape, jnp.float32)
    return jax.ffi.ffi_call(
        library._TARGET_RELION_WAVG_ROTATION_ATOMIC_RUNTIME_FLAT_ROWS_TRIPLET_ADD_F32,
        output_type,
        input_output_aliases={2: 0},
        vmap_method="sequential",
    )(terms, row_image_ids, accumulator, logical_pixel_count)


def relion_wavg_exact_atomic_flat_rows_triplet_add_f32(
    exact_terms: jax.Array,
    row_image_ids: jax.Array,
    exact_positions: jax.Array,
    accumulator: jax.Array,
    logical_pixel_count: jax.Array,
) -> jax.Array:
    """:func:`relion_wavg_rotation_atomic_runtime_flat_rows_triplet_add_f32` on the exact pixels only.

    ``exact_terms`` is ``(Q, P_exact, 3)`` and ``exact_positions`` ``(P_exact,)``
    their positions in the ``(B, P_rect, 3)`` accumulator's rectangle; the
    rectangle's other positions would have received zeros, which change no
    accumulator value. Each image's run of rows is summed in row order and
    added with one atomic per cell, as in the rectangle form. Positions at or
    past ``logical_pixel_count`` are skipped.
    """

    exact_terms = jnp.asarray(exact_terms)
    row_image_ids = jnp.asarray(row_image_ids)
    exact_positions = jnp.asarray(exact_positions)
    accumulator = jnp.asarray(accumulator)
    logical_pixel_count = jnp.asarray(logical_pixel_count, dtype=jnp.int32)
    if exact_terms.dtype != jnp.float32 or exact_terms.ndim != 3 or exact_terms.shape[-1] != 3:
        raise ValueError("exact-position RELION Wavg atomics require float32 [row, exact pixel, 3] terms")
    row_count, n_exact = exact_terms.shape[0], exact_terms.shape[1]
    if row_image_ids.dtype != jnp.int32 or row_image_ids.shape != (row_count,):
        raise ValueError("exact-position RELION Wavg atomics require int32 row_image_ids[Q]")
    if exact_positions.dtype != jnp.int32 or exact_positions.shape != (n_exact,):
        raise ValueError("exact-position RELION Wavg atomics require int32 exact_positions[P_exact]")
    if (
        accumulator.dtype != jnp.float32
        or accumulator.ndim != 3
        or accumulator.shape[2] != 3
        or accumulator.shape[1] < n_exact
    ):
        raise ValueError("exact-position RELION Wavg atomics require a float32 [batch,rect pixel,3] accumulator")
    if logical_pixel_count.shape != ():
        raise ValueError("logical_pixel_count must be an int32 scalar")
    library._ensure_optional_ffi(library._TARGET_RELION_WAVG_EXACT_ATOMIC_FLAT_ROWS_TRIPLET_ADD_F32)
    output_type = jax.ShapeDtypeStruct(accumulator.shape, jnp.float32)
    return jax.ffi.ffi_call(
        library._TARGET_RELION_WAVG_EXACT_ATOMIC_FLAT_ROWS_TRIPLET_ADD_F32,
        output_type,
        input_output_aliases={3: 0},
        vmap_method="sequential",
    )(exact_terms, row_image_ids, exact_positions, accumulator, logical_pixel_count)


@functools.partial(
    jax.jit, static_argnames=("image_shape", "rows_per_block")
)
def relion_translate_sum_flat_rows_f32(
    recon_image: jax.Array,
    noise_image: jax.Array,
    row_image_ids: jax.Array,
    posterior: jax.Array,
    translation_angles: jax.Array,
    pixel_indices: jax.Array,
    n_valid_rows: jax.Array,
    logical_pixel_count: jax.Array,
    recon_weight: jax.Array | None = None,
    ctf2_over_nv: jax.Array | None = None,
    *,
    image_shape: Tuple[int, int],
    rows_per_block: int = 0,
) -> tuple[jax.Array, ...]:
    """Translate and posterior-weight two per-image operands over packed rows.

    This is the flat-row form of the resident M-step's weighted sums. For every
    row ``r`` with image ``row_image_ids[r]`` and every reconstruction pixel
    ``p`` it returns

    ``summed[r, p]        = sum_t posterior[r, t] * shift_t(recon_image[id, p])``
    ``summed_masked[r, p] = sum_t posterior[r, t] * shift_t(noise_image[id, p])``
    ``probs_sum_t[r]      = sum_t posterior[r, t]``

    The translation is applied inside the reduction with the phase and complex
    rotation of :func:`relion_translate_score_f32`, so the kernel replaces a
    pre-shifted ``[images, translations, pixels]`` tile and its contraction
    without ever materialising the tile. ``pixel_indices`` are RECOVAR's
    centered packed-half indices, the same operand
    :func:`relion_translate_score_f32` takes.

    ``recon_weight`` switches the first operand to the exact RELION BPref
    convention, the one ``_prepare_bucket_io`` uses when
    ``relion_exact_bpref_operands`` is selected: pass the raw BPref image as
    ``recon_image`` and the weighted CTF as ``recon_weight``, and the kernel
    reproduces :func:`relion_translate_bpref_f32`, whose imaginary component
    and post-rotation weighting round differently from the score primitive.
    The noise operand always uses the score convention, which is how
    ``shifted_score_half_with_dc`` is built in both modes.

    ``n_valid_rows`` and ``logical_pixel_count`` are device int32 scalars, so a
    capacity-shaped chunk keeps one traced program: rows at or past
    ``n_valid_rows``, rows whose image id is negative, and pixels at or past
    ``logical_pixel_count`` are written as zeros and read nothing. The sum over
    translations is sequential in increasing ``t`` with a rounded multiply and a
    rounded add; the XLA path it replaces contracts the same products at
    ``Precision.HIGHEST``, so the two agree to a few float32 ulp and exactly at
    ``T == 1``.

    ``ctf2_over_nv`` adds the M-step block's fourth output. Supplied, the call
    returns ``(summed, summed_masked, probs_sum_t, ctf_probs)`` with

    ``ctf_probs[r, p] = probs_sum_t[r] != 0 ? probs_sum_t[r] * ctf2[id, p] : 0``

    which is :func:`relax.fine_pass.local_backprojection.compute_local_ctf_sums_from_probs_sum_t`
    term for term, including its predicate on the mass rather than on the
    product, and bitwise against it on the same ``probs_sum_t``. Omitted, the
    call returns the three-output form and reads no CTF operand.

    ``rows_per_block`` selects how many packed rows share one ``sincosf``
    evaluation. Zero picks the kernel's default; it changes performance only,
    never the arithmetic.
    """

    recon_image = jnp.asarray(recon_image)
    noise_image = jnp.asarray(noise_image)
    row_image_ids = jnp.asarray(row_image_ids)
    posterior = jnp.asarray(posterior)
    translation_angles = jnp.asarray(translation_angles)
    pixel_indices = jnp.asarray(pixel_indices)
    n_valid_rows = jnp.asarray(n_valid_rows, dtype=jnp.int32)
    logical_pixel_count = jnp.asarray(logical_pixel_count, dtype=jnp.int32)
    if recon_image.dtype != jnp.complex64 or noise_image.dtype != jnp.complex64:
        raise TypeError(
            "flat-row translate-and-sum expects complex64 image operands, got "
            f"{recon_image.dtype} and {noise_image.dtype}"
        )
    if recon_image.ndim != 2 or noise_image.shape != recon_image.shape:
        raise ValueError(
            "flat-row translate-and-sum expects matching [batch, pixels] image "
            f"operands, got {recon_image.shape} and {noise_image.shape}"
        )
    batch_size, pixel_capacity = (int(size) for size in recon_image.shape)
    if row_image_ids.dtype != jnp.int32 or row_image_ids.ndim != 1:
        raise ValueError(
            "flat-row translate-and-sum expects int32 row_image_ids[Q]"
        )
    row_count = int(row_image_ids.shape[0])
    # [T, 2] shared by every image, or [B, T, 2], one table per image (tilt images; one row per block).
    per_image_angles = translation_angles.ndim == 3
    if translation_angles.dtype != jnp.float32 or translation_angles.ndim not in (2, 3) or (
        translation_angles.shape[-1] != 2
        or (per_image_angles and int(translation_angles.shape[0]) != batch_size)
    ):
        raise ValueError(
            "flat-row translate-and-sum expects float32 translation_angles[T,2] or [B,T,2]"
        )
    n_trans = int(translation_angles.shape[-2])
    if posterior.dtype != jnp.float32 or posterior.shape != (row_count, n_trans):
        raise ValueError(
            "flat-row translate-and-sum expects float32 posterior[Q,T], got "
            f"{posterior.shape} {posterior.dtype}"
        )
    if pixel_indices.dtype != jnp.int32 or pixel_indices.shape != (
        pixel_capacity,
    ):
        raise ValueError(
            "flat-row translate-and-sum expects int32 pixel_indices[P], got "
            f"{pixel_indices.shape} {pixel_indices.dtype}"
        )
    if n_valid_rows.shape != () or logical_pixel_count.shape != ():
        raise ValueError(
            "n_valid_rows and logical_pixel_count must be int32 scalars"
        )
    if len(image_shape) != 2 or any(int(size) <= 0 for size in image_shape):
        raise ValueError(
            f"image_shape must contain two positive sizes, got {image_shape}"
        )
    if int(rows_per_block) not in (0, 1, 2, 4, 8):
        raise ValueError(
            f"rows_per_block must be 0, 1, 2, 4 or 8, got {rows_per_block}"
        )
    bpref_recon = recon_weight is not None
    if bpref_recon:
        recon_weight = jnp.asarray(recon_weight)
        if recon_weight.dtype != jnp.float32 or recon_weight.shape != (
            batch_size,
            pixel_capacity,
        ):
            raise ValueError(
                "flat-row translate-and-sum expects float32 recon_weight[B,P], "
                f"got {recon_weight.shape} {recon_weight.dtype}"
            )
    else:
        recon_weight = jnp.zeros((1, 1), dtype=jnp.float32)
    write_ctf_probs = ctf2_over_nv is not None
    if write_ctf_probs:
        ctf2_over_nv = jnp.asarray(ctf2_over_nv)
        if ctf2_over_nv.dtype != jnp.float32 or ctf2_over_nv.shape != (
            batch_size,
            pixel_capacity,
        ):
            raise ValueError(
                "flat-row translate-and-sum expects float32 ctf2_over_nv[B,P], "
                f"got {ctf2_over_nv.shape} {ctf2_over_nv.dtype}"
            )
    else:
        ctf2_over_nv = jnp.zeros((1, 1), dtype=jnp.float32)
    if batch_size <= 0 or pixel_capacity <= 0 or row_count <= 0 or n_trans <= 0:
        raise ValueError(
            "flat-row translate-and-sum operands must be non-empty, got "
            f"batch={batch_size} pixels={pixel_capacity} rows={row_count} "
            f"translations={n_trans}"
        )
    if jax.default_backend() != "gpu":
        raise RuntimeError(
            "flat-row translate-and-sum requires a JAX GPU backend"
        )
    if not custom_cuda_requested():
        raise RuntimeError(
            "flat-row translate-and-sum was explicitly requested but custom "
            "CUDA is disabled"
        )
    library._ensure_optional_ffi(library._TARGET_RELION_TRANSLATE_SUM_FLAT_ROWS_F32)

    image_h, image_w = (int(size) for size in image_shape)
    outputs = (
        jax.ShapeDtypeStruct((row_count, pixel_capacity), jnp.complex64),
        jax.ShapeDtypeStruct((row_count, pixel_capacity), jnp.complex64),
        jax.ShapeDtypeStruct((row_count,), jnp.float32),
        jax.ShapeDtypeStruct(
            (row_count, pixel_capacity) if write_ctf_probs else (1, 1),
            jnp.float32,
        ),
    )
    results = jax.ffi.ffi_call(
        library._TARGET_RELION_TRANSLATE_SUM_FLAT_ROWS_F32,
        outputs,
        vmap_method="sequential",
    )(
        recon_image,
        noise_image,
        recon_weight,
        ctf2_over_nv,
        row_image_ids,
        posterior,
        translation_angles,
        pixel_indices,
        n_valid_rows,
        logical_pixel_count,
        image_h=np.int64(image_h),
        image_half_width=np.int64(image_w // 2 + 1),
        rows_per_block=np.int64(int(rows_per_block)),
        bpref_recon=np.int64(int(bpref_recon)),
        write_ctf_probs=np.int64(int(write_ctf_probs)),
    )
    # The unused fourth buffer is a 1x1 placeholder; the three-output form
    # never exposes it.
    return tuple(results) if write_ctf_probs else tuple(results[:3])


def sparse_pass2_segmented_supported() -> bool:
    """Return whether the loaded library exports both segmented pass-2 targets.

    The segmented handlers are an optional ABI, so a library built before them
    stays loadable and callers fall back to the rectangular handlers.
    """

    try:
        library._ensure_ffi()
        lib = library._get_lib()
        return all(
            getattr(lib, library._OPTIONAL_FFI_REGISTRATIONS[target][0], None) is not None
            for target in (
                library._TARGET_SPARSE_PASS2_SEGMENTED_LOG_Z_F64,
                library._TARGET_SPARSE_PASS2_SEGMENTED_POSTERIOR_F32,
            )
        )
    except Exception:
        return False


def relion_wavg_sequential_runtime_flat_rows_triplet_f32_supported() -> bool:
    """Return whether the loaded library exports the flat-row Wavg target."""

    return library._optional_target_supported(
        library._TARGET_RELION_WAVG_SEQUENTIAL_RUNTIME_FLAT_ROWS_TRIPLET_F32
    )


def relion_wavg_rotation_atomic_runtime_flat_rows_triplet_add_f32_supported() -> bool:
    """Return whether the loaded library exports the flat-row Wavg atomics."""

    return library._optional_target_supported(
        library._TARGET_RELION_WAVG_ROTATION_ATOMIC_RUNTIME_FLAT_ROWS_TRIPLET_ADD_F32
    )
