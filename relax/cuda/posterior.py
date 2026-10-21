"""RELION posterior kernels on the GPU: the batched f32 exponentiate and divide and the coarse cut of the
significant-sample posterior, and the segmented sparse pass-2 log-Z and posterior.

Each wrapper registers the library targets through ``library.ensure_ffi()`` or
``library.ensure_optional_ffi(target)`` before its first FFI call; the segmented pass-2 targets are an optional
ABI. The per-row kernels and the batched CUB sort-scan the tests compare these with are in
``tests/oracles/cuda_posterior.py``.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from recovar.cuda_backproject import (
    custom_cuda_requested,
)

from relax.cuda import library


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
    library.ensure_ffi()

    return jax.ffi.ffi_call(
        target,
        jax.ShapeDtypeStruct(values.shape, jnp.float32),
    )(values, operand)


@jax.jit
def relion_exponentiate_batched_f32(
    values: jax.Array,
    add: jax.Array,
) -> jax.Array:
    """Apply RELION's float32 posterior exponentiation to a row batch."""

    return _relion_posterior_batched_f32(
        library.TARGET_RELION_EXPONENTIATE_BATCHED_F32, "exponentiation", "add", values, add
    )


@jax.jit
def relion_divide_batched_f32(
    values: jax.Array,
    divisor: jax.Array,
) -> jax.Array:
    """Apply RELION float32 division to all posterior rows in one FFI call."""

    return _relion_posterior_batched_f32(
        library.TARGET_RELION_DIVIDE_BATCHED_F32, "division", "divisor", values, divisor
    )


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
    library.ensure_ffi()

    rows = weights.shape[0]
    return jax.ffi.ffi_call(
        library.TARGET_RELION_COARSE_CUT_F32,
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


# Byte size of the device-side per-row state record (RowState) of the sparse pass-2 posterior.
_SPARSE_PASS2_ROW_STATE_BYTES = 40


# Byte size of the device-side per-segment state record of
# SparsePass2SegmentedPosteriorF32: the per-row RowState plus the segment
# extent (valid flag, explicit padding, begin, cell count) and the significance
# cut's 256-bin histogram and state (BlockCut, 3112 bytes; the handler asserts the size).
_SPARSE_PASS2_SEGMENT_STATE_BYTES = _SPARSE_PASS2_ROW_STATE_BYTES + 24 + 3112


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

    Segments without a finite score, empty segments and segments at or beyond
    ``n_valid_images`` return ``-inf``, the log-sum-exp of an all ``-inf`` row.
    This handler runs entirely on the device.
    """

    _, segments = _sparse_pass2_segment_geometry(scores, segment_offsets, n_valid_images)
    _require_sparse_pass2_cuda_backend("Segmented sparse pass-2 log-Z")
    library.ensure_optional_ffi(library.TARGET_SPARSE_PASS2_SEGMENTED_LOG_Z_F64)
    return jax.ffi.ffi_call(
        library.TARGET_SPARSE_PASS2_SEGMENTED_LOG_Z_F64,
        jax.ShapeDtypeStruct((segments,), jnp.float64),
    )(scores, segment_offsets, n_valid_images.astype(jnp.int32).reshape(()))


def _segmented_posterior_segments(
    scores, segment_offsets, n_valid_images, log_z, external_sum_weight, keep_all, use_external_sum_weight
) -> int:
    """The segment count, after the operand checks the two segmented posterior handlers share."""

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
    return segments


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
    """Fused normalization and RELION float32 fine posterior over segments of flat sparse pass-2 scores.

    ``scores`` is a flat cell array whose segment ``i`` is
    ``[segment_offsets[i], segment_offsets[i + 1])``; operands and outputs are
    per segment.  Returns ``(log_z,
    best_log_score, best_cell_index, max_posterior, probs, normalized_weights,
    reconstruction_probs, mask, n_significant, sum_weight, threshold)``.
    ``best_cell_index`` is relative to the segment; add
    ``segment_offsets[i]`` for the flat cell.  Per-cell outputs keep the shape
    of ``scores``; cells covered by no segment, empty segments and segments at
    or beyond ``n_valid_images`` carry the values of an all ``-inf`` row.

    Launch geometry is capacity-only: every kernel is launched at the static
    segment count from ``segment_offsets.shape`` and ``n_valid_images`` is read on
    the device alone, so nothing on the host depends on the chunk's occupancy and
    the handler has no host round trip. The significance boundary is RELION's cut
    with float64 cumulative sums, by radix select (the definition of
    :func:`relion_coarse_cut_f32`), each segment's histogram added over the
    blocks its cells are split into (the grid's segment-count and cell-count
    are static, so the split is too).

    ``return_scratch`` appends the handler's raw weights to the result.
    """

    segments = _segmented_posterior_segments(
        scores, segment_offsets, n_valid_images, log_z, external_sum_weight, keep_all, use_external_sum_weight
    )
    if type(return_scratch) is not bool:
        raise TypeError("return_scratch must be a static Python bool")
    _require_sparse_pass2_cuda_backend("Segmented sparse pass-2 posterior")
    library.ensure_optional_ffi(library.TARGET_SPARSE_PASS2_SEGMENTED_POSTERIOR_F32)
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
    result = jax.ffi.ffi_call(library.TARGET_SPARSE_PASS2_SEGMENTED_POSTERIOR_F32, outputs)(
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


@functools.partial(
    jax.jit,
    static_argnames=("adaptive_fraction", "keep_all", "use_external_sum_weight"),
)
def sparse_pass2_segmented_reconstruction_f32(
    scores: jax.Array,
    segment_offsets: jax.Array,
    n_valid_images: jax.Array,
    log_z: jax.Array,
    external_sum_weight: jax.Array,
    *,
    adaptive_fraction: float,
    keep_all: bool,
    use_external_sum_weight: bool,
) -> tuple[jax.Array, ...]:
    """:func:`sparse_pass2_segmented_posterior_f32` without ``probs``, ``normalized_weights`` and ``mask``.

    Returns ``(log_z, best_log_score, best_cell_index, max_posterior, reconstruction_probs, n_significant,
    sum_weight, threshold)``, each bitwise the full handler's: the same kernels run in the same order and skip
    only the three per-cell outputs (SparsePass2SegmentedReconstructionF32). The handler holds 12 bytes per cell
    (scores, reconstruction and raw weights) instead of 29, for a particle too large for the full posterior
    (relax#118).
    """

    segments = _segmented_posterior_segments(
        scores, segment_offsets, n_valid_images, log_z, external_sum_weight, keep_all, use_external_sum_weight
    )
    _require_sparse_pass2_cuda_backend("Reconstruction-only segmented sparse pass-2 posterior")
    library.ensure_optional_ffi(library.TARGET_SPARSE_PASS2_SEGMENTED_RECONSTRUCTION_F32)
    full = scores.shape
    outputs = (
        jax.ShapeDtypeStruct((segments,), jnp.float64),  # log_z
        jax.ShapeDtypeStruct((segments,), jnp.float32),  # best_log_score
        jax.ShapeDtypeStruct((segments,), jnp.int64),    # best_cell_index
        jax.ShapeDtypeStruct((segments,), jnp.float32),  # max_posterior
        jax.ShapeDtypeStruct(full, jnp.float32),         # reconstruction_probs
        jax.ShapeDtypeStruct((segments,), jnp.int32),    # n_significant
        jax.ShapeDtypeStruct((segments,), jnp.float32),  # sum_weight
        jax.ShapeDtypeStruct((segments,), jnp.float32),  # threshold
        jax.ShapeDtypeStruct(full, jnp.float32),         # raw weights (scratch)
        jax.ShapeDtypeStruct(
            (segments, _SPARSE_PASS2_SEGMENT_STATE_BYTES), jnp.uint8
        ),                                               # segment state
    )
    result = jax.ffi.ffi_call(library.TARGET_SPARSE_PASS2_SEGMENTED_RECONSTRUCTION_F32, outputs)(
        scores,
        segment_offsets,
        n_valid_images.astype(jnp.int32).reshape(()),
        log_z,
        external_sum_weight,
        adaptive_fraction=np.float32(adaptive_fraction),
        keep_all=np.int64(keep_all),
        use_external_sum_weight=np.int64(use_external_sum_weight),
    )
    return tuple(result[:8])
