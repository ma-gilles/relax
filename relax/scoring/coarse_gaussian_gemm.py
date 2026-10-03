"""Coarse Gaussian GEMM backend of the K-class significance pass.

The env-gated GEMM macro and its opt-ins, the device-memory budgets and
projection cache, the hybrid image-batch plan and its request validation,
and the hybrid batch itself. ``significance`` selects this backend once per
pass and calls it per image batch.
"""

import operator
import os
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from relax.helpers import projection_cache as projection_cache_helpers
from relax.helpers.env_flags import parse_env_strict_flag
from relax.scoring.scoring import (
    _relion_coarse_gaussian_gemm_scores,
)

_COARSE_GAUSSIAN_GEMM_MACRO_ENV = "RECOVAR_COARSE_GAUSSIAN_GEMM_MACRO"


_COARSE_GAUSSIAN_GEMM_MAX_PROJECTED_TRANSIENT_GB_ENV = (
    "RECOVAR_COARSE_GAUSSIAN_GEMM_MAX_PROJECTED_TRANSIENT_GB"
)


_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_ENV = (
    "RECOVAR_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE"
)


_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_MAX_GB_ENV = (
    "RECOVAR_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_MAX_GB"
)


_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_DEFAULT_MAX_GB = 4.0


_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_CHUNK_ROWS = 4_608


_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_ROW_ALIGNMENT = 16


_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_ALIAS_EVIDENCE_JOB = 13_332_001


_K1_RELION_EXACT_COARSE_OPERANDS_ENV = "RECOVAR_K1_RELION_EXACT_COARSE_OPERANDS"


_K1_RELION_F32_COARSE_SUPPORT_ENV = "RECOVAR_K1_RELION_F32_COARSE_SUPPORT"


def _coarse_gaussian_gemm_macro_enabled(*, default: bool = False) -> bool:
    """Whether one coarse projection feeds the shared multi-image GEMMs.

    The exact-arithmetic objective is unchanged, but expanding the direct
    square changes operation order and can amplify cancellation.  Keep it
    default-off until same-H100 repeats establish stable numerical noise,
    unchanged discrete choices/support and quality, and a material end-to-end
    speedup.
    """

    return parse_env_strict_flag(_COARSE_GAUSSIAN_GEMM_MACRO_ENV, default=default)


def _coarse_gaussian_gemm_projection_cache_enabled(
    *,
    default: bool = False,
) -> bool:
    """Resolve the explicit call-scoped coarse-projection cache toggle."""

    return parse_env_strict_flag(_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_ENV, default=default)


class CoarseGaussianGemmResources(NamedTuple):
    """Conservative device-transient accounting for one GEMM rotation block."""

    full_centered_projection_bytes: int
    compact_projection_bytes: int
    compact_projection_abs2_bytes: int
    predicted_peak_projection_bytes: int
    projected_transient_budget_bytes: int


def _coarse_gaussian_gemm_projected_transient_budget_bytes(
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


def _coarse_gaussian_gemm_projection_cache_budget_bytes(
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


def _coarse_gaussian_gemm_projection_row_bytes(*, image_shape, compact_pixel_count: int) -> int:
    """Projector transient per rotation row, as ``_coarse_gaussian_gemm_resources`` counts it."""

    image_height, image_width = (int(value) for value in image_shape)
    full_count = image_height * (image_width // 2 + 1)
    compact_count = int(compact_pixel_count)
    return (
        full_count * np.dtype(np.complex64).itemsize
        + compact_count * np.dtype(np.complex64).itemsize
        + compact_count * np.dtype(np.float32).itemsize
    )


def _coarse_gaussian_gemm_fit_rotation_block_size(
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
    row_bytes = _coarse_gaussian_gemm_projection_row_bytes(
        image_shape=image_shape,
        compact_pixel_count=compact_pixel_count,
    )
    fit = int(budget_bytes) // row_bytes
    if fit >= requested:
        return requested
    if fit >= row_alignment:
        return fit // row_alignment * row_alignment
    return max(1, fit)


def _coarse_gaussian_gemm_cached_block_rows(
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


def _coarse_gaussian_gemm_resources(
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


def _validate_coarse_gaussian_gemm_projection_cache_request(
    *,
    macro_enabled: bool,
    n_rotations: int,
    coarse_gaussian_ffi_enabled: bool,
    exact_coarse_operands_enabled: bool,
    use_relion_projector: bool,
    relion_texture_interp_enabled: bool,
    use_float64_scoring: bool,
    relion_projector_dtype,
) -> None:
    """Fail closed unless the cache serves the exact RELION projector rows.

    The cache holds exactly the complex64 rows the macro's projector callback
    returns, one table per class, so any class count is served unchanged.
    """

    prefix = f"{_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_ENV}=1 requires"
    if not macro_enabled:
        raise ValueError(
            f"{prefix} {_COARSE_GAUSSIAN_GEMM_MACRO_ENV}=1",
        )
    if int(n_rotations) <= 0 or int(n_rotations) % int(
        _COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_ROW_ALIGNMENT
    ):
        raise ValueError(
            f"{prefix} a positive rotation count divisible by "
            f"{_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_ROW_ALIGNMENT}, "
            f"got {int(n_rotations)}",
        )
    if not coarse_gaussian_ffi_enabled:
        raise ValueError(
            f"{prefix} the exact RELION coarse Gaussian FFI path",
        )
    if not exact_coarse_operands_enabled:
        raise ValueError(
            f"{prefix} {_K1_RELION_EXACT_COARSE_OPERANDS_ENV}=1",
        )
    if not use_relion_projector or not relion_texture_interp_enabled:
        raise ValueError(
            f"{prefix} the supplied RELION texture projector",
        )
    if use_float64_scoring:
        raise ValueError(f"{prefix} production float32/complex64 scoring")
    if relion_projector_dtype is None or np.dtype(relion_projector_dtype) != np.dtype(
        np.complex64
    ):
        raise TypeError(
            f"{prefix} a complex64 RELION projector, got "
            f"{relion_projector_dtype}",
        )


def _plan_coarse_gaussian_gemm_projection_cache(
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


def _coarse_gaussian_gemm_projection_cache_stats(plan, *, enabled: bool):
    """Describe conservative admission and narrowly scoped alias evidence."""

    h100_alias_evidence_applies = bool(
        plan.cache_shape == (1, 36_864, 5_100)
        and plan.chunk_rows
        == _COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_CHUNK_ROWS
    )
    return {
        "enabled": bool(enabled),
        "scope": "one significance call",
        "cache_shape": tuple(int(value) for value in plan.cache_shape),
        "cache_dtype": plan.cache_dtype.name,
        "stores_projection_abs2": False,
        "chunk_rows": int(plan.chunk_rows),
        "chunk_count": int(plan.chunk_count_per_table),
        "retained_bytes": int(plan.retained_bytes),
        "conservative_predicted_peak_bytes": int(plan.predicted_peak_bytes),
        "budget_bytes": int(plan.budget_bytes),
        "admission_destination_alias_proven": bool(
            plan.destination_alias_proven
        ),
        # Informational only: deterministic job 13332001 observed donated
        # insert aliasing for exactly (1, 36864, 5100) with 4608-row chunks
        # on one H100. Admission always reserves a non-aliased copy.
        "h100_alias_evidence_applies_to_plan": h100_alias_evidence_applies,
        "h100_alias_evidence_cache_shape": (1, 36_864, 5_100),
        "h100_alias_evidence_chunk_rows": int(
            _COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_CHUNK_ROWS
        ),
        "h100_observed_donated_insert_alias": (
            True if h100_alias_evidence_applies else None
        ),
        "h100_alias_evidence_job_id": int(
            _COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_ALIAS_EVIDENCE_JOB
        ),
        "h100_observed_alias_peak_bytes": (
            int(plan.predicted_peak_bytes - plan.destination_copy_bytes)
            if h100_alias_evidence_applies
            else None
        ),
        "h100_alias_evidence_used_for_admission": False,
    }


def _project_coarse_gaussian_gemm_projection_cache_block_once(
    cache,
    class_index,
    mean_for_proj,
    rotations_block,
    *,
    rotation_start: int,
):
    """Serve one existing macro block from C64 cache without reprojection."""

    del mean_for_proj
    cache = jnp.asarray(cache)
    if cache.ndim != 3 or np.dtype(cache.dtype) != np.dtype(np.complex64):
        raise TypeError(
            "coarse GEMM projection cache must have shape "
            "(table, rotation, pixel) and dtype complex64",
        )
    table_index = int(class_index)
    if table_index < 0 or table_index >= int(cache.shape[0]):
        raise IndexError(
            f"coarse GEMM projection-cache table {table_index} is out of range",
        )
    start = int(rotation_start)
    requested_rows = int(rotations_block.shape[0])
    if start < 0 or requested_rows <= 0 or start >= int(cache.shape[1]):
        raise IndexError(
            "coarse GEMM projection-cache block must start inside the cache "
            "and contain at least one row",
        )
    stop = min(start + requested_rows, int(cache.shape[1]))
    return _projection_cache_block(cache, table_index, start, rows=stop - start, padded_rows=requested_rows)


@partial(jax.jit, static_argnames=("rows", "padded_rows"))
def _projection_cache_block(cache, table_index, start, *, rows, padded_rows):
    """One cached block and its squared modulus in one program.

    The block is a slice of one table, zero-padded to ``padded_rows`` rows: the
    shared significance loop masks those physical tail rows to -inf, and zero
    padding keeps its fixed score-block shape without projecting synthetic
    identity rotations or changing any valid cached row. The squared modulus
    is the same C64 expression as the projector callback's; the promoted
    certificate ignores this companion and forms its two component squares in
    FP64. Eager, the slice, pad and square were four programs per class and
    rotation block, and the device idled between them (K15 50k, nsys 14693142).
    """

    projected_reference = jax.lax.dynamic_slice_in_dim(cache[table_index], start, rows, axis=0)
    if padded_rows > rows:
        projected_reference = jnp.pad(projected_reference, ((0, padded_rows - rows), (0, 0)))
    return projected_reference, jnp.abs(projected_reference) ** 2


def _score_relion_coarse_gaussian_gemm_macro(
    project_block_once,
    class_index,
    mean_for_proj,
    rotations_block,
    shifted_corrected,
    pixel_weight,
    initial_diff2,
    actual_image_count,
    *,
    image_shape,
    volume_shape,
):
    """Project once, then score every physical image lane through shared GEMMs."""

    projected_reference, projected_reference_abs2 = project_block_once(
        class_index,
        mean_for_proj,
        rotations_block,
    )
    return _relion_coarse_gaussian_gemm_scores(
        projected_reference,
        projected_reference_abs2,
        shifted_corrected,
        pixel_weight,
        initial_diff2,
        actual_image_count,
        image_shape=image_shape,
        volume_shape=volume_shape,
    )
