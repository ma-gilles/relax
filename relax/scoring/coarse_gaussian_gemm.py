"""Coarse Gaussian GEMM backend of the K-class significance pass.

The env-gated GEMM macro and its opt-ins, the device-memory budgets and
projection cache, the hybrid image-batch plan and its request validation,
and the hybrid batch itself. ``significance`` selects this backend once per
pass and calls it per image batch.
"""

import operator
import os
from typing import NamedTuple

import jax
import numpy as np

from relax.helpers import projection_cache as projection_cache_helpers
from relax.helpers.env_flags import parse_env_strict_flag

_COARSE_GAUSSIAN_GEMM_MAX_PROJECTED_TRANSIENT_GB_ENV = (
    "RECOVAR_COARSE_GAUSSIAN_GEMM_MAX_PROJECTED_TRANSIENT_GB"
)


COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_ENV = (
    "RECOVAR_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE"
)


_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_MAX_GB_ENV = (
    "RECOVAR_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_MAX_GB"
)


_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_DEFAULT_MAX_GB = 4.0


_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_CHUNK_ROWS = 4_608


_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_ROW_ALIGNMENT = 16



_K1_RELION_F32_COARSE_SUPPORT_ENV = "RECOVAR_K1_RELION_F32_COARSE_SUPPORT"


def coarse_gaussian_gemm_projection_cache_enabled(
    *,
    default: bool = False,
) -> bool:
    """Resolve the explicit call-scoped coarse-projection cache toggle."""

    return parse_env_strict_flag(COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_ENV, default=default)


class CoarseGaussianGemmResources(NamedTuple):
    """Conservative device-transient accounting for one GEMM rotation block."""

    full_centered_projection_bytes: int
    compact_projection_bytes: int
    compact_projection_abs2_bytes: int
    predicted_peak_projection_bytes: int
    projected_transient_budget_bytes: int


def coarse_gaussian_gemm_projected_transient_budget_bytes(
    *,
    default_gb: float = 2.0,
) -> int:
    """Return the explicit conservative projection-transient budget."""

    token = os.environ.get(
        _COARSE_GAUSSIAN_GEMM_MAX_PROJECTED_TRANSIENT_GB_ENV,
        str(default_gb),
    ).strip()
    try:
        budget_gb = float(token)
    except ValueError as error:
        raise ValueError(
            f"{_COARSE_GAUSSIAN_GEMM_MAX_PROJECTED_TRANSIENT_GB_ENV} must be "
            f"a finite positive number, got {token!r}",
        ) from error
    if not np.isfinite(budget_gb) or budget_gb <= 0.0:
        raise ValueError(
            f"{_COARSE_GAUSSIAN_GEMM_MAX_PROJECTED_TRANSIENT_GB_ENV} must be "
            f"a finite positive number, got {token!r}",
        )
    return int(budget_gb * 1024**3)


_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_DEVICE_FRACTION = 0.2


def coarse_gaussian_gemm_projection_cache_budget_bytes(
    *,
    default_gb: float | None = None,
) -> int:
    """Return the call-scoped cache budget.

    Without an explicit budget the cache may use a fifth of the GPU's memory
    (16 GB on an 80 GB H100/A100); off-GPU it keeps the fixed 4 GB default.
    """

    if default_gb is None:
        default_gb = _COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_DEFAULT_MAX_GB
        devices = jax.local_devices()
        if devices and devices[0].platform == "gpu":
            stats = devices[0].memory_stats() or {}
            if "bytes_limit" in stats:
                default_gb = (
                    _COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_DEVICE_FRACTION
                    * float(stats["bytes_limit"])
                    / 1024**3
                )
    token = os.environ.get(
        _COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_MAX_GB_ENV,
        str(default_gb),
    ).strip()
    try:
        budget_gb = float(token)
    except ValueError as error:
        raise ValueError(
            f"{_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_MAX_GB_ENV} must be "
            f"a finite positive number, got {token!r}",
        ) from error
    if not np.isfinite(budget_gb) or budget_gb <= 0.0:
        raise ValueError(
            f"{_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_MAX_GB_ENV} must be "
            f"a finite positive number, got {token!r}",
        )
    return int(budget_gb * 1024**3)


def coarse_gaussian_gemm_projection_row_bytes(*, image_shape, compact_pixel_count: int) -> int:
    """Projector transient per rotation row, as ``coarse_gaussian_gemm_resources`` counts it."""

    image_height, image_width = (int(value) for value in image_shape)
    full_count = image_height * (image_width // 2 + 1)
    compact_count = int(compact_pixel_count)
    return (
        full_count * np.dtype(np.complex64).itemsize
        + compact_count * np.dtype(np.complex64).itemsize
        + compact_count * np.dtype(np.float32).itemsize
    )


def coarse_gaussian_gemm_fit_rotation_block_size(
    requested_rows: int,
    *,
    image_shape,
    compact_pixel_count: int,
    budget_bytes: int,
    row_alignment: int = 16,
) -> int:
    """Largest rotation block, at most ``requested_rows``, whose projector transient fits the budget.

    Large boxes at fine HEALPix orders would otherwise exceed the transient
    budget with the caller's block (10097 10k K=1 at HEALPix 3: 9.9 GB for
    5,000 rows). Blocks only split the rotation axis of one image batch, so
    every score is unchanged. The block keeps a 16-row alignment when that
    fits and never drops below one row.
    """

    requested = operator.index(requested_rows)
    if requested <= 0:
        raise ValueError("requested rotation block must be positive")
    row_bytes = coarse_gaussian_gemm_projection_row_bytes(
        image_shape=image_shape,
        compact_pixel_count=compact_pixel_count,
    )
    fit = int(budget_bytes) // row_bytes
    if fit >= requested:
        return requested
    if fit >= row_alignment:
        return fit // row_alignment * row_alignment
    return max(1, fit)


def coarse_gaussian_gemm_cached_block_rows(
    n_rotations: int,
    *,
    image_batch_size: int,
    n_translations: int,
    compact_pixel_count: int,
    budget_bytes: int,
    row_alignment: int = 16,
) -> int:
    """Rotation rows per GEMM block when cached projections feed the scorer.

    Without a projector transient the block is bounded by the GEMM's own
    per-row temporaries: the float32 reference components and their packed
    copy (about six floats per compact pixel) and the image-batch cross, its
    transpose and the scores (three floats per image-translation lane). The
    rows are balanced over the fewest blocks that fit the budget.
    """

    rows = operator.index(n_rotations)
    if rows <= 0:
        raise ValueError("coarse GEMM needs at least one rotation")
    row_bytes = 4 * (6 * int(compact_pixel_count) + 3 * int(image_batch_size) * int(n_translations))
    fit = max(row_alignment, int(budget_bytes) // row_bytes // row_alignment * row_alignment)
    if fit >= rows:
        return rows
    blocks = -(-rows // fit)
    balanced = -(-rows // blocks)
    return -(-balanced // row_alignment) * row_alignment


def coarse_gaussian_gemm_resources(
    *,
    rotation_block_size: int,
    image_shape,
    compact_pixel_count: int,
    budget_bytes: int,
) -> CoarseGaussianGemmResources:
    """Estimate and gate projection buffers before the first GPU launch.

    The mature texture projector currently creates a full physical centered
    half-image before gathering the current-size compact pixels.  Count that
    buffer plus the compact complex projection and its real abs2 companion.
    The pixel-index table is deliberately kept as a NumPy host array, so this
    path performs no device-to-host index materialization.
    """

    block_size = int(rotation_block_size)
    image_height, image_width = (int(value) for value in image_shape)
    compact_count = int(compact_pixel_count)
    if block_size <= 0 or image_height <= 0 or image_width <= 0 or compact_count <= 0:
        raise ValueError("coarse GEMM resource dimensions must all be positive")
    full_count = image_height * (image_width // 2 + 1)
    full_bytes = block_size * full_count * np.dtype(np.complex64).itemsize
    compact_bytes = block_size * compact_count * np.dtype(np.complex64).itemsize
    abs2_bytes = block_size * compact_count * np.dtype(np.float32).itemsize
    predicted_peak = full_bytes + compact_bytes + abs2_bytes
    resources = CoarseGaussianGemmResources(
        full_centered_projection_bytes=int(full_bytes),
        compact_projection_bytes=int(compact_bytes),
        compact_projection_abs2_bytes=int(abs2_bytes),
        predicted_peak_projection_bytes=int(predicted_peak),
        projected_transient_budget_bytes=int(budget_bytes),
    )
    if resources.predicted_peak_projection_bytes > resources.projected_transient_budget_bytes:
        raise MemoryError(
            "coarse GEMM predicted projection transient exceeds "
            f"{_COARSE_GAUSSIAN_GEMM_MAX_PROJECTED_TRANSIENT_GB_ENV}: "
            f"{resources.predicted_peak_projection_bytes} > "
            f"{resources.projected_transient_budget_bytes} bytes",
        )
    return resources


def validate_coarse_gaussian_gemm_projection_cache_request(
    *,
    n_rotations: int,
    relion_projector_dtype,
) -> None:
    """Fail closed unless the cache serves the exact RELION projector rows.

    The cache holds exactly the complex64 rows the coarse GEMMs' projector
    returns, one table per class, so any class count is served unchanged.
    """

    prefix = f"{COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_ENV}=1 requires"
    if int(n_rotations) <= 0 or int(n_rotations) % int(
        _COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_ROW_ALIGNMENT
    ):
        raise ValueError(
            f"{prefix} a positive rotation count divisible by "
            f"{_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_ROW_ALIGNMENT}, "
            f"got {int(n_rotations)}",
        )
    if relion_projector_dtype is None or np.dtype(relion_projector_dtype) != np.dtype(
        np.complex64
    ):
        raise TypeError(
            f"{prefix} a complex64 RELION projector, got "
            f"{relion_projector_dtype}",
        )


def plan_coarse_gaussian_gemm_projection_cache(
    *,
    n_classes: int = 1,
    n_rotations: int,
    compact_pixel_count: int,
    image_shape,
    budget_bytes: int,
) -> projection_cache_helpers.ProjectionCachePlan:
    """Plan one conservative C64 cache table per class in 4,608-row chunks."""

    image_height, image_width = (int(value) for value in image_shape)
    return projection_cache_helpers.plan_projection_cache(
        table_count=int(n_classes),
        row_count=int(n_rotations),
        pixel_count=int(compact_pixel_count),
        cache_dtype=np.complex64,
        requested_max_chunk_rows=(
            _COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_CHUNK_ROWS
        ),
        row_alignment=_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_ROW_ALIGNMENT,
        transient_specs=(
            projection_cache_helpers.ProjectionCacheTransientSpec(
                name="full_centered_projection",
                elements_per_row=image_height * (image_width // 2 + 1),
                dtype=np.complex64,
            ),
        ),
        budget_bytes=int(budget_bytes),
        # Job 13332001 observed donation aliasing on one H100 lowering.  The
        # production admission remains conservative and correct if that
        # informational hardware-specific observation does not generalize.
        destination_alias_proven=False,
    )
