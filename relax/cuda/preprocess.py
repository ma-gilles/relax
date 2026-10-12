"""RELION image preprocessing on the GPU and its deferred finite-value checks.

relion_preprocess_real_f32 normalizes, soft-masks and Fourier-transforms a batch of real images with the
librelax_cuda.so preprocessing kernels; the deferred-check scopes queue its device-side invalid-image counts and
raise at a drain point instead of synchronizing every batch. Importing this module registers
relion_preprocess_real_f32 as the relion_cuda image backend of recovar.data_io.image_backends (relax split seam
S2), so a module that needs that backend imports this one. Each wrapper registers the library targets through
``library.ensure_ffi()`` before its first FFI call.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
from recovar.cuda_backproject import (
    custom_cuda_requested,
)
from recovar.data_io import image_backends as _image_backends

from relax.cuda import library
from relax.runtime.env_flags import parse_env_binary_flag

_RELION_PREPROCESS_BLOCK_SIZE = 128      # kRelionPreprocessBlockSize in cuda_backproject.cu


_RELION_SOFTMASK_BLOCKS = 128            # kRelionSoftMaskBlocks


_RELION_PREPROCESS_CUB_TEMP_BYTES = 1 << 16  # budget for one small cub::DeviceReduce::Sum


def _relion_preprocess_workspace_bytes(
    batch_size: int,
    *,
    native_lane_reduction: bool,
    native_atomic_reduction: bool,
) -> int:
    """Byte size of the soft-mask workspace output, mirroring the C++ scratch layout.

    Per image: two primary sum arrays (one entry per soft-mask block, or per
    lane for the native-atomic tree) plus, for the lane tree, two lane arrays
    ahead of the block-by-lane partials; then two floats of reduced sums per
    image; then the CUB temporary budget.  Each region is 256-byte aligned.
    The handler recomputes its own layout and fails closed if this is short.
    """

    if native_lane_reduction:
        per_image = 2 * _RELION_SOFTMASK_BLOCKS * _RELION_PREPROCESS_BLOCK_SIZE + 2 * _RELION_PREPROCESS_BLOCK_SIZE
    elif native_atomic_reduction:
        per_image = 2 * _RELION_PREPROCESS_BLOCK_SIZE
    else:
        per_image = 2 * _RELION_SOFTMASK_BLOCKS

    def _align(nbytes: int) -> int:
        return (nbytes + 255) // 256 * 256

    return (
        _align(int(batch_size) * per_image * 4)
        + _align(2 * int(batch_size) * 4)
        + _align(_RELION_PREPROCESS_CUB_TEMP_BYTES)
    )


def _relion_preprocess_real_f32_impl(
    images: jax.Array,
    normalization_factors: jax.Array,
    integer_shifts: jax.Array,
    radius: float,
    cosine_width: float,
    apply_mask: bool = True,
    native_lane_reduction: bool = False,
    native_atomic_reduction: bool = False,
    host_check: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Traced body of :func:`relion_preprocess_real_f32`; also returns the invalid-image count."""

    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION CUDA preprocessing requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError("RELION CUDA preprocessing was explicitly requested but custom CUDA is disabled")
    library.ensure_ffi()
    if images.dtype != jnp.float32:
        raise TypeError(f"images must be float32, got {images.dtype}")
    if normalization_factors.dtype != jnp.float32:
        raise TypeError(f"normalization_factors must be float32, got {normalization_factors.dtype}")
    if integer_shifts.dtype != jnp.int32:
        raise TypeError(f"integer_shifts must be int32, got {integer_shifts.dtype}")
    if images.ndim != 3 or images.shape[-2] != images.shape[-1]:
        raise ValueError(f"images must have shape (batch, D, D), got {images.shape}")
    batch_size = images.shape[0]
    if normalization_factors.shape != (batch_size,):
        raise ValueError(f"normalization_factors must have shape ({batch_size},), got {normalization_factors.shape}")
    if integer_shifts.shape != (batch_size, 2):
        raise ValueError(f"integer_shifts must have shape ({batch_size}, 2), got {integer_shifts.shape}")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError(f"radius must be finite and positive, got {radius}")
    if not np.isfinite(cosine_width) or cosine_width <= 0.0:
        raise ValueError(f"cosine_width must be finite and positive, got {cosine_width}")
    if native_lane_reduction and native_atomic_reduction:
        raise ValueError("native lane and native atomic reductions are mutually exclusive")

    out_type = jax.ShapeDtypeStruct(images.shape, jnp.float32)
    # [number of invalid images, smallest batch position of one] (relion_preprocess.cuh).
    count_type = jax.ShapeDtypeStruct((2,), jnp.int32)
    workspace_type = jax.ShapeDtypeStruct(
        (
            _relion_preprocess_workspace_bytes(
                batch_size,
                native_lane_reduction=native_lane_reduction,
                native_atomic_reduction=native_atomic_reduction,
            ),
        ),
        jnp.uint8,
    )
    target = (
        library.TARGET_RELION_PREPROCESS_REAL_F32_NATIVE_ATOMIC
        if native_atomic_reduction
        else (
            library.TARGET_RELION_PREPROCESS_REAL_F32_NATIVE_LANE
            if native_lane_reduction
            else library.TARGET_RELION_PREPROCESS_REAL_F32
        )
    )
    # Result order matches the handler binding: images, masked, workspace, count.
    normalized_shifted, masked, _workspace, invalid_count = jax.ffi.ffi_call(
        target,
        (out_type, out_type, workspace_type, count_type),
        vmap_method="sequential",
    )(
        images,
        normalization_factors,
        integer_shifts,
        radius=np.float32(radius),
        cosine_width=np.float32(cosine_width),
        apply_mask=np.int64(int(apply_mask)),
        host_check=np.int64(int(bool(host_check))),
    )
    return normalized_shifted, masked, invalid_count


_relion_preprocess_real_f32_jit = jax.jit(
    _relion_preprocess_real_f32_impl, static_argnums=(3, 4, 5, 6, 7, 8)
)


def relion_preprocess_real_f32(
    images: jax.Array,
    normalization_factors: jax.Array,
    integer_shifts: jax.Array,
    radius: float,
    cosine_width: float,
    apply_mask: bool = True,
    native_lane_reduction: bool = False,
    native_atomic_reduction: bool = False,
    deferred_finite_check: bool | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Apply RELION's accelerated float32 real-space preprocessing.

    Returns ``(normalized_shifted, masked)`` so captured operand tests can
    gate both stored RELION boundaries.  The implementation preserves
    RELION's separate float32 normalization and zero-filled translation and
    CUDA ``sqrtf``/``cospif`` mask arithmetic.  The default uses RECOVAR's
    accepted deterministic block-first addition tree.  The diagnostic-only
    ``native_lane_reduction`` mode instead deterministically reproduces the
    native observer's lane-across-blocks tree before its final CUB sum. The
    diagnostic ``native_atomic_reduction`` mode reproduces RELION's actual
    schedule-dependent atomic lane accumulation.

    The soft-mask launch is batched: the background sums stay on the device
    and the fill kernel forms the same float32 quotient there, so the masked
    images are bit-identical to the former per-image launch.  Default failure
    semantics are unchanged: a mask with no exterior texel fails closed with
    ``CUDA: invalid argument``, and an invalid image (a non-finite pixel
    anywhere, inside the particle mask included, or a non-positive or
    non-finite soft-mask background) raises after one per-call read-back of
    the device's invalid-image count, naming the image
    (:func:`note_relion_preprocess_batch`). The count is taken on the
    preprocessed images, so a non-finite pixel that the integer shift moves
    out of the frame is not an invalid image. A call without a mask reads
    nothing back: its count is queued and read when the queue drains.

    ``deferred_finite_check`` (default: on inside a
    :func:`deferred_relion_preprocess_checks` block, else from
    ``RELAX_RELION_PREPROCESS_DEFERRED_CHECK``, off) removes that per-call
    read-back and stream synchronization: the device counts invalid images
    into a small array that is queued on the open block, which reads it when
    the block ends, or else for :func:`drain_relion_preprocess_checks`, which
    the iteration loop calls after each E-step.  Until that read, an invalid
    image carries NaN in its masked exterior; the read then raises the same
    failure, later.
    """

    if deferred_finite_check is None:
        deferred_finite_check = bool(_RELION_PREPROCESS_DEFERRED_SCOPES) or relion_preprocess_deferred_check_requested()
    # A Python queue cannot retain status tracers from the local big JIT.
    # Keep the native fail-closed check inside that compiled execution.
    eager = type(jax.core.trace_ctx.trace).__name__ == "EvalTrace"
    if not eager:
        deferred_finite_check = False
    normalized_shifted, masked, invalid_count = _relion_preprocess_real_f32_jit(
        images,
        normalization_factors,
        integer_shifts,
        radius,
        cosine_width,
        apply_mask,
        native_lane_reduction,
        native_atomic_reduction,
        not eager,
    )
    batch_images = _take_noted_batch(int(images.shape[0])) if eager else None
    if deferred_finite_check or (eager and not apply_mask):
        # Without a mask the call never waited on the device: its count joins the queue the E-step drains.
        _queue_relion_preprocess_check(invalid_count, batch_images)
    elif eager:
        # The same per-call read-back the kernel's own check made, with the image named.
        count, first = (int(v) for v in np.asarray(_count_and_first(invalid_count)))
        if count:
            raise RuntimeError(
                f"RELION CUDA preprocessing: {_INVALID_IMAGE.format(count=count)}; "
                f"first: {_invalid_image_name(first, batch_images)}"
            )
    return normalized_shifted, masked


_INVALID_IMAGE = "{count} image(s) had a non-finite pixel or a non-positive or non-finite soft-mask background"


# The dataset images of the batch being preprocessed (relax.relion.image_preprocessing.prepare_batch_preprocess_operands).
_RELION_PREPROCESS_BATCH_IMAGES: np.ndarray | None = None


def note_relion_preprocess_batch(image_indices) -> None:
    """Name the dataset images of the next ``relion_preprocess_real_f32`` calls for their failure messages."""

    global _RELION_PREPROCESS_BATCH_IMAGES
    _RELION_PREPROCESS_BATCH_IMAGES = None if image_indices is None else np.asarray(image_indices).reshape(-1)


def _take_noted_batch(batch_size: int) -> np.ndarray | None:
    """The noted dataset images when they are this call's batch (one note serves one call), else None."""

    global _RELION_PREPROCESS_BATCH_IMAGES
    images, _RELION_PREPROCESS_BATCH_IMAGES = _RELION_PREPROCESS_BATCH_IMAGES, None
    return images if images is not None and images.size == batch_size else None


def _invalid_image_name(position: int, images: np.ndarray | None) -> str:
    """``position`` of a preprocess call's batch, with its dataset image when the batch was noted."""

    if images is None or not 0 <= position < images.size:
        return f"position {position} of the batch"
    return f"dataset image {int(images[position])} (position {position} of the batch)"


def _count_and_first(invalid_count: jax.Array) -> jax.Array:
    """``[count, first position]`` of one call; a bare count has no position (-1)."""

    flat = invalid_count.reshape(-1)
    return flat[:2] if flat.shape[0] >= 2 else jnp.concatenate([flat, jnp.full((1,), -1, flat.dtype)])


# Tests bypass the JIT through ``__wrapped__`` to exercise the eager guards.
relion_preprocess_real_f32.__wrapped__ = _relion_preprocess_real_f32_impl


RELION_PREPROCESS_DEFERRED_CHECK_ENV = "RELAX_RELION_PREPROCESS_DEFERRED_CHECK"


_RELION_PREPROCESS_PENDING_CHECKS: list[jax.Array] = []


_RELION_PREPROCESS_PENDING_LOCK = threading.Lock()


_RELION_PREPROCESS_PENDING_LIMIT = 1024


def relion_preprocess_deferred_check_requested() -> bool:
    """Strict 0/1 read of ``RELAX_RELION_PREPROCESS_DEFERRED_CHECK`` (unset is off)."""

    return parse_env_binary_flag(RELION_PREPROCESS_DEFERRED_CHECK_ENV)


class DeferredPreprocessChecks:
    """The soft-mask checks queued inside one :func:`deferred_relion_preprocess_checks` block.

    ``at(label)`` names the batch the next preprocess calls belong to; the failure message lists
    the labels of the batches that held an invalid image (the kernel reports a count per call, not
    which images of the call).
    """

    def __init__(self, where: str):
        self.where = str(where)
        self.label = "unlabelled batch"
        self.pending: list[tuple[str, np.ndarray | None, jax.Array]] = []

    def at(self, label: str) -> None:
        self.label = str(label)

    def queue(self, invalid_count: jax.Array, batch_images: np.ndarray | None = None) -> None:
        self.pending.append((self.label, batch_images, _count_and_first(invalid_count)))
        if len(self.pending) >= _RELION_PREPROCESS_PENDING_LIMIT:
            self.raise_if_invalid()

    def raise_if_invalid(self) -> int:
        """Wait on the queued checks; raise naming every batch with an invalid image. Returns the number read."""

        pending, self.pending = self.pending, []
        if not pending:
            return 0
        counts = np.asarray(jnp.stack([count for _, _, count in pending])).astype(np.int64)
        if int(counts[:, 0].sum()):
            batches = "; ".join(
                f"{label}: {int(count)} image(s), first {_invalid_image_name(int(first), images)}"
                for (label, images, _), (count, first) in zip(pending, counts)
                if count
            )
            raise RuntimeError(
                f"RELION CUDA preprocessing (deferred check, {self.where}): "
                f"{_INVALID_IMAGE.format(count=int(counts[:, 0].sum()))}, in {batches}"
            )
        return len(pending)


# Open :func:`deferred_relion_preprocess_checks` scopes, innermost last (the batch loops run on one thread).
_RELION_PREPROCESS_DEFERRED_SCOPES: list[DeferredPreprocessChecks] = []


@contextmanager
def deferred_relion_preprocess_checks(where: str):
    """Defer the soft-mask check of every ``relion_preprocess_real_f32`` call in the block to its end.

    A batch loop that keeps the device busy while the host prepares the next batch cannot afford the
    synchronous check: its read-back waits for everything queued before it. Inside the block the calls
    queue their invalid-image counts on the yielded :class:`DeferredPreprocessChecks` (the loop names
    each batch with ``at``), and leaving the block normally reads them and raises for an invalid image,
    naming ``where`` and the batches: the same failure, at the end of the loop instead of at the image's
    batch. The queue belongs to the block: an exception from the block propagates unchanged and the
    block's unread checks are dropped with it, so they cannot be reported against a later pass.
    """

    checks = DeferredPreprocessChecks(where)
    _RELION_PREPROCESS_DEFERRED_SCOPES.append(checks)
    try:
        yield checks
    finally:
        _RELION_PREPROCESS_DEFERRED_SCOPES.pop()
    checks.raise_if_invalid()


def _queue_relion_preprocess_check(invalid_count: jax.Array, batch_images: np.ndarray | None = None) -> None:
    """Queue one device invalid-image count: on the innermost open scope, else on the process queue
    (``RELAX_RELION_PREPROCESS_DEFERRED_CHECK``), which drains when full."""

    if _RELION_PREPROCESS_DEFERRED_SCOPES:
        _RELION_PREPROCESS_DEFERRED_SCOPES[-1].queue(invalid_count, batch_images)
        return
    with _RELION_PREPROCESS_PENDING_LOCK:
        _RELION_PREPROCESS_PENDING_CHECKS.append(_count_and_first(invalid_count))
        full = len(_RELION_PREPROCESS_PENDING_CHECKS) >= _RELION_PREPROCESS_PENDING_LIMIT
    if full:
        drain_relion_preprocess_checks()


def drain_relion_preprocess_checks() -> int:
    """Wait on every queued deferred soft-mask check and fail closed on invalid images.

    Returns the number of drained checks.  Raises ``RuntimeError`` when any
    batch contained an image whose soft-mask background weight was
    non-positive or whose sums were non-finite; this is the deferred form of
    the ``CUDA: invalid argument`` failure of the synchronous check.
    """

    with _RELION_PREPROCESS_PENDING_LOCK:
        pending = list(_RELION_PREPROCESS_PENDING_CHECKS)
        _RELION_PREPROCESS_PENDING_CHECKS.clear()
    if not pending:
        return 0
    counts = np.asarray(jnp.stack(pending)).astype(np.int64)[:, 0]
    invalid_images = int(counts.sum())
    if invalid_images:
        raise RuntimeError(
            f"RELION CUDA preprocessing (deferred check): {_INVALID_IMAGE.format(count=invalid_images)}, "
            f"in {int((counts > 0).sum())} batch(es)"
        )
    return len(pending)


@contextmanager
def expectation_relion_preprocess_checks():
    """The process queue's checks belong to one expectation: read when it ends, dropped when it raises.

    Leaving the block normally drains the queue (:func:`drain_relion_preprocess_checks`). An exception
    propagates unchanged and the queued checks are dropped with it, so a later run or test in the same
    process never reads a failed expectation's counts.
    """

    try:
        yield
    except BaseException:
        with _RELION_PREPROCESS_PENDING_LOCK:
            _RELION_PREPROCESS_PENDING_CHECKS.clear()
        raise
    drain_relion_preprocess_checks()


# Seam S2 (relax split): the relion_cuda image backend in recovar.data_io.image_backends calls
# RELION's CUDA preprocessing through this registration instead of importing EM code.
_image_backends.register_relion_cuda_preprocessor(relion_preprocess_real_f32)
