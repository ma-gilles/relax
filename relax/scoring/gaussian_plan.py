"""Planning the Gaussian route of pass 1 before its first batch: the scored rows, the GEMM's resources, the cache."""

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from relax.scoring.coarse_gaussian_gemm import (
    _COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_ENV,
    _coarse_gaussian_gemm_cached_block_rows,
    _coarse_gaussian_gemm_fit_rotation_block_size,
    _coarse_gaussian_gemm_projected_transient_budget_bytes,
    _coarse_gaussian_gemm_projection_cache_budget_bytes,
    _coarse_gaussian_gemm_projection_cache_enabled,
    _coarse_gaussian_gemm_projection_cache_stats,
    _coarse_gaussian_gemm_projection_row_bytes,
    _coarse_gaussian_gemm_resources,
    _plan_coarse_gaussian_gemm_projection_cache,
    _validate_coarse_gaussian_gemm_projection_cache_request,
)
from relax.scoring.coarse_layout import CoarseGaussianSquareLayout, plan_coarse_gaussian_square_layout
from relax.scoring.coarse_publication import coarse_square_layout_metadata
from relax.scoring.pass1_results import PassShape

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CoarseGaussianPlan:
    """What the coarse Gaussian GEMM of a pass is planned to run on; fixed before the first batch.

    ``square_layout`` is the compact table of RELION's square crop of the current size (inside a stable physical
    capacity when the pass asks for stable shapes). ``score_indices_np`` / ``score_indices`` are its rows of the half
    spectrum (int32, on the host and the device), ``score_active_mask`` marks the logical rows among them, and
    ``projector_output_size`` is the physical size the projector crops to. ``powerclass`` is RELION's high-resolution
    image power kernel. ``resource_estimate`` is the GEMM's memory estimate. ``projection_cache_plan`` is the plan of
    the cached C64 projections, ``None`` when the pass keeps no cache. ``rotation_block_size`` is the rotation block
    the pass runs with: the caller's, fitted down to the projector transient budget, or grown to what the temporaries
    allow when the projections are cached.
    """

    square_layout: CoarseGaussianSquareLayout
    score_indices_np: np.ndarray
    score_indices: Any
    score_active_mask: Any
    projector_output_size: int
    powerclass: Callable
    resource_estimate: Any
    projection_cache_plan: Any
    rotation_block_size: int


def _custom_cuda_ready() -> bool:
    """Whether JAX runs on a GPU with RECOVAR's custom CUDA library loaded (pass 1's FFI kernels)."""

    from recovar import cuda_backproject

    return jax.default_backend() == "gpu" and cuda_backproject.cuda_available()


def _projection_cache_requested(shape: PassShape, relion_projector_half) -> tuple[bool, bool]:
    """Whether the pass asks for the projection cache, and whether the environment asked for it explicitly.

    An explicit request fails closed; the default cache quietly stands down wherever its contract does not hold.
    """

    requested = _coarse_gaussian_gemm_projection_cache_enabled(default=True)
    explicit = _COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_ENV in os.environ
    if requested:
        try:
            _validate_coarse_gaussian_gemm_projection_cache_request(
                n_rotations=shape.n_rot,
                # The dtype of the class stack: indexing a device array for it dispatched a slice per pass.
                relion_projector_dtype=(
                    relion_projector_half.dtype
                    if hasattr(relion_projector_half, "dtype")
                    else relion_projector_half[0].dtype
                ),
            )
        except (ValueError, TypeError) as reason:
            if explicit:
                raise
            logger.info("coarse GEMM projection cache off: %s", reason)
            requested = False
    return requested, explicit


def plan_coarse_gaussian(
    shape: PassShape,
    window_score_indices_np,
    relion_projector_half,
    *,
    rotation_block_size: int,
    image_batch_size: int,
    stable_fourier_window_shapes: bool,
) -> CoarseGaussianPlan:
    """Plan the Gaussian route of a pass over ``shape``, scoring the rows ``window_score_indices_np`` (``None``: all).

    Refuses more than 128 translations and a backend without the custom CUDA library. Logs the plan.
    """

    cache_requested, cache_explicit = _projection_cache_requested(shape, relion_projector_half)
    if shape.n_trans > 128:
        raise ValueError(f"the coarse GEMM scorer supports at most 128 translations, got {shape.n_trans}")
    from relax.sparse_pass2.sparse_pass2_scoring import (
        _relion_cuda_powerclass_highres_xi2_half,
    )

    if not _custom_cuda_ready():
        raise RuntimeError("pass 1 scores RELION's exact coarse operands and needs the custom CUDA backend")
    active_score_indices_np = (
        np.arange(shape.n_half, dtype=np.int32)
        if window_score_indices_np is None
        else np.asarray(window_score_indices_np, dtype=np.int32)
    )
    square_layout = plan_coarse_gaussian_square_layout(
        shape.image_shape,
        shape.score_size,
        active_score_indices_np,
        stable_fourier_window_shapes=bool(stable_fourier_window_shapes),
    )
    square_score_count = square_layout.physical_square_count
    score_indices_np = np.asarray(
        square_layout.score_indices_np,
        dtype=np.int32,
    )
    transient_budget = _coarse_gaussian_gemm_projected_transient_budget_bytes()
    # The plain GEMM splits the rotation axis until the projector
    # transient fits.
    fitted_block_size = _coarse_gaussian_gemm_fit_rotation_block_size(
        int(rotation_block_size),
        image_shape=shape.image_shape,
        compact_pixel_count=int(square_score_count),
        budget_bytes=transient_budget,
    )
    if fitted_block_size != int(rotation_block_size):
        logger.info(
            "coarse GEMM rotation block %d -> %d rows to fit the %d-byte "
            "projector transient budget",
            int(rotation_block_size),
            fitted_block_size,
            transient_budget,
        )
        rotation_block_size = fitted_block_size
    transient_budget = max(
        transient_budget,
        int(rotation_block_size)
        * _coarse_gaussian_gemm_projection_row_bytes(
            image_shape=shape.image_shape,
            compact_pixel_count=int(square_score_count),
        ),
    )
    resource_estimate = _coarse_gaussian_gemm_resources(
        rotation_block_size=int(rotation_block_size),
        image_shape=shape.image_shape,
        compact_pixel_count=int(square_score_count),
        budget_bytes=transient_budget,
    )
    projection_cache_plan = None
    if cache_requested:
        projection_cache_plan = _plan_coarse_gaussian_gemm_projection_cache(
            n_classes=shape.n_classes,
            n_rotations=shape.n_rot,
            compact_pixel_count=int(square_score_count),
            image_shape=shape.image_shape,
            budget_bytes=_coarse_gaussian_gemm_projection_cache_budget_bytes(),
        )
        if not (projection_cache_plan.admitted or cache_explicit):
            logger.info(
                "coarse GEMM projection cache off: %s",
                projection_cache_plan.admission_reason,
            )
            projection_cache_plan = None
    if projection_cache_plan is not None:
        # Cached projections need no per-block projector transient, so the
        # GEMM block grows to what its own temporaries allow (usually every
        # rotation): each image batch runs one score, prior and reduction
        # program per class and block instead of one per 5,000 rows.
        rotation_block_size = _coarse_gaussian_gemm_cached_block_rows(
            shape.n_rot,
            image_batch_size=int(image_batch_size),
            n_translations=int(shape.n_trans),
            compact_pixel_count=int(square_score_count),
            budget_bytes=_coarse_gaussian_gemm_projected_transient_budget_bytes(),
        )
    logger.info(
        "Coarse pass on RELION's exact operands (coarse GEMMs): classes=%d rotations=%d "
        "current_size=%d physical_size=%d square_pixels=%d image_lanes=%d translations=%d stable_shapes=%s",
        shape.n_classes,
        shape.n_rot,
        shape.score_size,
        square_layout.physical_current_size,
        square_score_count,
        int(image_batch_size),
        shape.n_trans,
        bool(stable_fourier_window_shapes),
    )
    return CoarseGaussianPlan(
        square_layout=square_layout,
        score_indices_np=score_indices_np,
        score_indices=jnp.asarray(score_indices_np, dtype=jnp.int32),
        score_active_mask=jnp.asarray(square_layout.score_active_mask_np, dtype=jnp.bool_),
        projector_output_size=square_layout.physical_current_size,
        powerclass=_relion_cuda_powerclass_highres_xi2_half,
        resource_estimate=resource_estimate,
        projection_cache_plan=projection_cache_plan,
        rotation_block_size=int(rotation_block_size),
    )


def coarse_gaussian_report(plan: CoarseGaussianPlan, *, stable_fourier_window_shapes: bool) -> dict:
    """The Gaussian route's entries of the pass's ``full_stats``: the square layout, the GEMM resources and the cache.

    The cache entry is present when the plan has a cache (it is built whenever it is planned).
    """

    report = {
        "coarse_gaussian_square_layout": coarse_square_layout_metadata(
            plan.square_layout,
            stable_fourier_window_shapes=stable_fourier_window_shapes,
        ),
        "coarse_gaussian_gemm_resources": {
            field: int(value) for field, value in plan.resource_estimate._asdict().items()
        },
    }
    if plan.projection_cache_plan is not None:
        report["coarse_gaussian_gemm_projection_cache"] = _coarse_gaussian_gemm_projection_cache_stats(
            plan.projection_cache_plan,
            enabled=True,
        )
    return report
