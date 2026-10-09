"""The score route of one pass 1, by contract: the Gaussian GEMM's plan or the normalized CC's.

A pass scores with one of the two, decided once from ``score_mode``. :func:`plan_gaussian_route` plans the Gaussian GEMM
(its scored rows, resources and projection cache) and :func:`plan_cc_route` the normalized CC (its operands and the tree
rescore). Both return the same :class:`RoutePlan`, which is what the stages after the route read.
"""

import logging
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from relax.relion.relion_coarse_operands import k1_relion_f32_coarse_support_enabled
from relax.scoring.coarse_projector import CompactRows
from relax.scoring.gaussian_plan import coarse_gaussian_report, plan_coarse_gaussian
from relax.scoring.pass1_operands import CcOperandPlan, GaussianOperandPlan
from relax.scoring.pass1_request import Pass1Request
from relax.scoring.pass1_results import PassShape
from relax.scoring.pass1_window import ScoringWindow
from relax.scoring.tree_rescore import TreeRescoreGeometry, plan_tree_rescore, require_tree_rescore_call

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoutePlan:
    """What a route decides before the pass's first batch, for the stages that follow it.

    ``operand_plan`` prepares every batch's score operands. ``score_kind`` is the kernels' static ``"gaussian"`` or
    ``"normalized_cc"`` and ``executed_backend`` names the scorer in ``full_stats``. ``float32_support`` says that the
    pass forms its support on RELION's float32 route (the Gaussian route only). ``rotation_block_size`` is the rotations
    per block the pass runs with. ``compact_rows`` are the rows the projector returns (``None``: the full crop).
    ``projection_cache_plan`` is the plan of the cached projections (``None``: the pass keeps no cache) and
    ``tree_rescore_plan`` the plan of the tree rescore (``None`` unless the pass rescores). ``report`` is the route's
    entries of ``full_stats``.
    """

    operand_plan: Any
    score_kind: str
    executed_backend: str
    float32_support: bool
    rotation_block_size: int
    compact_rows: CompactRows | None
    projection_cache_plan: Any
    tree_rescore_plan: Any
    report: dict


def plan_gaussian_route(
    request: Pass1Request,
    shape: PassShape,
    window: ScoringWindow,
    relion_projector_half,
    translations_source,
    *,
    rotation_block_size: int,
    image_batch_size: int,
    kernel_window: int | None,
) -> RoutePlan:
    """Plan the Gaussian GEMM route over ``shape``, scoring the rows of ``window``.

    ``relion_projector_half`` is the class-stacked projector and ``translations_source`` the translations the scores
    shift by. ``kernel_window`` is the rows RELION's coarse kernel shifts at a relabelled position (``None``: none).
    """

    gaussian_plan = plan_coarse_gaussian(
        shape,
        window.window_spec.score_indices_np,
        relion_projector_half,
        rotation_block_size=rotation_block_size,
        image_batch_size=image_batch_size,
        stable_fourier_window_shapes=request.stable_fourier_window_shapes,
    )
    return RoutePlan(
        operand_plan=GaussianOperandPlan(
            experiment_dataset=request.experiment_dataset,
            gaussian_plan=gaussian_plan,
            image_shape=shape.image_shape,
            half_weights=window.half_weights,
            translations_source=translations_source,
            relion_translation_angle_scale=request.relion_translation_angle_scale,
            score_with_masked_images=request.score_with_masked_images,
            nyquist_column_counting=request.nyquist_column_counting,
            scale_corrections_enabled=request.scale_corrections is not None,
            use_float64_scoring=request.use_float64_scoring,
            stable_fourier_window_shapes=request.stable_fourier_window_shapes,
            score_size=shape.score_size,
            current_size=request.current_size,
            coarse_kernel_window=kernel_window,
            coarse_kernel_r_max=None if kernel_window is None else int(request.relion_projector_r_max),
        ),
        score_kind="gaussian",
        executed_backend="gemm_macro",
        float32_support=k1_relion_f32_coarse_support_enabled(default=True),
        rotation_block_size=gaussian_plan.rotation_block_size,
        compact_rows=CompactRows(gaussian_plan.score_indices_np, gaussian_plan.projector_output_size),
        projection_cache_plan=gaussian_plan.projection_cache_plan,
        tree_rescore_plan=None,
        report=coarse_gaussian_report(
            gaussian_plan, stable_fourier_window_shapes=request.stable_fourier_window_shapes
        ),
    )


def plan_cc_route(
    request: Pass1Request,
    shape: PassShape,
    window: ScoringWindow,
    relion_projector_half,
    translations_source,
    *,
    rotation_block_size: int,
    coarse_healpix_order: int | None,
    coarse_rotation_ids,
) -> RoutePlan:
    """Plan the normalized-CC route (``--firstiter_cc``) over ``shape``, scoring the rows of ``window``.

    RELION's exact coarse operands: the tree rescore's per-image FFT, RFLOAT CTF and corr_img operands, translated
    with RELION's sincosf for every translation and scored by the coarse GEMMs
    (``relion_coarse_normalized_cc_gemm_scores_jit``). With ``tree_rescore_max_margin`` the route also plans the
    top-two rescore of the ambiguous winners; ``coarse_healpix_order`` and ``coarse_rotation_ids`` place its candidates
    in RELION's coarse grid.
    """

    from relax.sparse_pass2.sparse_pass2_bucket_io import relion_translation_angles_f32

    score_indices = jnp.asarray(
        np.arange(shape.n_half)
        if window.window_spec.score_indices_np is None
        else window.window_spec.score_indices_np,
        dtype=jnp.int32,
    )
    translation_angles = jnp.asarray(
        relion_translation_angles_f32(
            translations_source,
            shape.image_shape,
            angle_scale=request.relion_translation_angle_scale,
        ),
        dtype=jnp.float32,
    )
    logger.info(
        "RELION normalized-CC coarse pass on the exact operands: classes=%d current_size=%d "
        "score_pixels=%d translations=%d",
        shape.n_classes,
        shape.score_size,
        int(score_indices.shape[0]),
        shape.n_trans,
    )
    # The margin is a run option, while only iteration 1 uses normalized CC.
    # Later Gaussian iterations must remain unaffected.
    tree_rescore_plan = None
    if request.tree_rescore_max_margin is not None:
        require_tree_rescore_call(n_classes=shape.n_classes, return_class_best=request.return_class_best)
        tree_rescore_plan = plan_tree_rescore(
            max_margin=request.tree_rescore_max_margin,
            relion_projector_half=relion_projector_half,
            image_shape=shape.image_shape,
            n_half=shape.n_half,
            score_indices_np=window.window_spec.score_indices_np,
            translation_angles=translation_angles,
            geometry=TreeRescoreGeometry(
                half_weights=window.score_half_weights,
                rotations=request.rotations,
                n_trans=shape.n_trans,
                score_size=shape.score_size,
                padding_factor=request.projection_padding_factor,
                projector_max_r=request.relion_projector_r_max,
                coarse_healpix_order=coarse_healpix_order,
                coarse_rotation_ids=coarse_rotation_ids,
                symmetry_label=request.symmetry_label,
            ),
        )
    return RoutePlan(
        operand_plan=CcOperandPlan(
            experiment_dataset=request.experiment_dataset,
            image_shape=shape.image_shape,
            image_pre_shifts=request.image_pre_shifts,
            window_indices=window.window_indices,
            score_indices=score_indices,
            score_half_weights=window.score_half_weights,
            support_power_weights=window.score_half_weights if window.cc_gaussian_support else None,
            translation_angles=translation_angles,
            n_trans=shape.n_trans,
            score_with_masked_images=request.score_with_masked_images,
            scale_corrections_enabled=request.scale_corrections is not None,
        ),
        score_kind="normalized_cc",
        executed_backend="exact_cc_gemm",
        float32_support=False,
        rotation_block_size=rotation_block_size,
        compact_rows=(
            CompactRows(window.window_spec.score_indices_np, shape.score_size) if window.use_window else None
        ),
        projection_cache_plan=None,
        tree_rescore_plan=tree_rescore_plan,
        report={},
    )
