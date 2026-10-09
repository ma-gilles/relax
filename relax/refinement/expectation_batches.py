"""Refinement expectation batches and their run-scoped memory planner."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace

import numpy as np

from relax.classification.k_class_inputs import _select_projector_half_for_class
from relax.helpers.batch_planning import (
    _RELION_EM_COMPACT_K1_FIXED_BASE_GB,
    _estimate_relion_em_batch_sizes,
    _plan_adaptive_dense_batch_sizes,
    _safe_dense_k_class_rotation_block_size,
    _safe_firstiter_cc_image_batch_size,
)
from relax.helpers.dtype_policy import DensePrecisionPolicy
from relax.helpers.half_volume_mstep import half_volume_accumulator_shape, relion_backprojector_volume_shape
from relax.helpers.projection import _host_relion_projector_texture_enabled
from relax.refinement import optics_shapes
from relax.refinement.refinement_options import ExecutionOptions
from relax.relion.geometry import PROJECTION_PADDING_FACTOR, RECONSTRUCTION_PADDING_FACTOR
from relax.sparse_pass2 import firstiter_bpref, sparse_pass2_budget


@dataclass(frozen=True, eq=False, kw_only=True)
class BatchPlanner:
    """Run requests and fixed grids; query available memory when a pass is planned."""

    requested: ExecutionOptions
    image_shape: tuple
    volume_shape: tuple
    n_classes: int
    # The run's dense precision (RefinementOptions.precision).
    precision: DensePrecisionPolicy
    log: object

    def __call__(
        self, n_rot, n_trans, *, classes=None, image_shape_for_batch=None,
        current_size_for_batch=None, compact_k1_relion_layout=False,
        compact_k1_relion_score_bpref_overlap=False,
        model_current_size_for_batch=None, score_projector_staging_bytes=0,
        windowed_translation=False,
    ):
        """Reduce batch sizes using the selected phase's pending allocations."""
        use_float64_scoring_for_batch = bool(
            self.precision.use_float64_scoring
            or os.environ.get("RELAX_DIAGNOSTIC_FLOAT64_PASS2_ITERATIONS", "").strip()
        )
        runtime_free_memory_gb = None
        if compact_k1_relion_layout:
            # CUDA textures allocate outside XLA's reusable pool, so the physical reading bounds them; under a
            # pool limit (XLA_PYTHON_CLIENT_MEM_FRACTION, or a pool holding most of the card) the run may only
            # use what the allocator can still hand out, which is the smaller reading there (relax#44).
            physical = sparse_pass2_budget._device_free_memory_bytes()
            in_pool = sparse_pass2_budget.device_available_bytes(
                physical,
                sparse_pass2_budget._jax_allocator_free_memory_bytes(),
                sparse_pass2_budget._jax_allocator_pool_free_bytes(),
            )
            readings = [float(value) for value in (physical, in_pool) if value is not None]
            free_bytes = min(readings) if readings else None
            self.log.info(
                "Compact K1 texture budget: physical free %s GB, allocator available %s GB, used %s GB",
                "n/a" if physical is None else f"{physical / 1e9:.2f}",
                "n/a" if in_pool is None else f"{in_pool / 1e9:.2f}",
                "n/a" if free_bytes is None else f"{free_bytes / 1e9:.2f}",
            )
            if free_bytes is not None and free_bytes > 0:
                runtime_free_memory_gb = free_bytes / 1e9
        plan = _estimate_relion_em_batch_sizes(
            requested_image_batch_size=self.requested.image_batch_size,
            requested_rotation_block_size=self.requested.rotation_block_size,
            n_rot=n_rot,
            n_trans=n_trans,
            image_shape=image_shape_for_batch or self.image_shape,
            volume_shape=self.volume_shape,
            padding_factor=RECONSTRUCTION_PADDING_FACTOR,
            n_classes=self.n_classes if classes is None else classes,
            current_size=current_size_for_batch,
            use_float64_scoring=use_float64_scoring_for_batch,
            compact_k1_relion_layout=compact_k1_relion_layout,
            compact_k1_relion_score_bpref_overlap=compact_k1_relion_score_bpref_overlap,
            model_current_size=model_current_size_for_batch,
            runtime_free_memory_gb=runtime_free_memory_gb,
            score_projector_staging_bytes=score_projector_staging_bytes,
            windowed_translation=windowed_translation,
        )
        plan.log_adjustment(
            requested_image_batch_size=self.requested.image_batch_size,
            requested_rotation_block_size=self.requested.rotation_block_size,
            n_rot=n_rot,
            n_trans=n_trans,
            n_classes=self.n_classes if classes is None else classes,
            logger=self.log,
        )
        return plan.image_batch_size, plan.rotation_block_size


@dataclass(frozen=True, kw_only=True)
class ShapeClassBatchSizes:
    """One shape class's adaptive dense batch sizes, planned from the class's own box and sizes."""

    k_class_image_batch_size: int
    k_class_rotation_block_size: int
    significance_image_batch_size: int
    significance_rotation_block_size: int


def _class_adaptive_batch_overrides(
    half, *, plan, cs_for_engine, coarse_cs, coarse_sizing
) -> tuple[ShapeClassBatchSizes, ...]:
    """Adaptive dense batch sizes planned per shape class, from the class's own box and sizes."""

    overrides = []
    for shape_class in half.classes:
        class_cs, class_coarse = optics_shapes.class_adaptive_sizes(
            shape_class, cs_for_engine, coarse_cs, coarse_sizing
        )
        class_plan = plan(image_shape=shape_class.dataset.image_shape, cs_for_engine=class_cs, coarse_cs=class_coarse)
        overrides.append(
            ShapeClassBatchSizes(
                k_class_image_batch_size=class_plan.pass2_image_batch_size,
                k_class_rotation_block_size=class_plan.pass2_rotation_block_size,
                significance_image_batch_size=class_plan.significance_image_batch_size,
                significance_rotation_block_size=class_plan.significance_rotation_block_size,
            )
        )
    return tuple(overrides)


@dataclass(frozen=True)
class HalfBatchPlan:
    """Coarse/fine sizes and staging-aware callbacks for one half's expectation."""

    safe_batch_sizes: object
    significance_safe_batch_sizes: object
    fine_image_batch_size: int
    fine_rotation_block_size: int
    coarse_image_batch_size: int | None
    coarse_rotation_block_size: int | None
    class_overrides: tuple[ShapeClassBatchSizes, ...] | None


def prepare_half_batches(
    dataset,
    projector,
    *,
    planner: BatchPlanner,
    rotations,
    translations,
    cs_for_engine,
    coarse_cs,
    model_current_size_for_engine,
    use_adaptive,
    use_local,
    relion_firstiter_cc_this_iter,
    firstiter_cc_tree_rescore_max_margin,
    firstiter_winner_take_all_this_iter,
    source_faithful_spectrum_norm,
    preserve_bpref_particle_order,
    bpref_device_signature_active,
    use_relion_x_half_mstep: bool,
    multi_shape_halves,
    coarse_sizing,
) -> HalfBatchPlan:
    """Resolve compact staging and coarse/fine batches before one half scores.

    Grids are inspected only in the branches that need them. Dense scoring uses
    the safe-size callbacks and the coarse/fine sizes; local scoring sizes its own tiles.
    """
    batching = planner.requested
    n_classes = planner.n_classes
    k_class_enabled = n_classes > 1
    volume_shape = planner.volume_shape
    k_class_image_batch_size = batching.image_batch_size
    dense_k_class_rotation_block_size = batching.rotation_block_size
    significance_image_batch_size = None
    significance_rotation_block_size = None
    safe_batch_sizes_for_half = planner
    significance_safe_batch_sizes_for_half = planner
    projector_half = _select_projector_half_for_class(
        None if projector is None else projector.data, 0, n_classes,
    )
    compact_precision = not (
        planner.precision.use_float64_scoring
        or planner.precision.use_float64_projections
        or os.environ.get("RELAX_DIAGNOSTIC_FLOAT64_PASS2_ITERATIONS", "").strip()
    )
    if (
        use_adaptive and not use_local and not k_class_enabled
        and relion_firstiter_cc_this_iter and compact_precision
        and _host_relion_projector_texture_enabled(
            projector_half, r_max=None if projector is None else projector.r_max,
            padding_factor=PROJECTION_PADDING_FACTOR, allow_float32_cast=True,
        )
    ):
        model_size = int(volume_shape[0] if model_current_size_for_engine is None
                         else model_current_size_for_engine)
        recon_shape = relion_backprojector_volume_shape(
            volume_shape, RECONSTRUCTION_PADDING_FACTOR, current_size=model_size,
        )
        decision = firstiter_bpref._relion_firstiter_compact_batch_planning_decision(
            source_faithful_spectrum_norm=source_faithful_spectrum_norm,
            winner_take_all=firstiter_winner_take_all_this_iter,
            preserve_bpref_particle_order=preserve_bpref_particle_order,
            use_relion_x_half_mstep=use_relion_x_half_mstep,
            projector_half=SimpleNamespace(shape=projector_half.shape, dtype=np.dtype(np.complex64)),
            score_complex_dtype=np.complex64,
            recon_volume_size=int(np.prod(half_volume_accumulator_shape(recon_shape))),
            bpref_device_signature_active=bpref_device_signature_active,
            fixed_base_bytes=int(_RELION_EM_COMPACT_K1_FIXED_BASE_GB * 1e9),
        )
        if decision.enabled:
            safe_batch_sizes_for_half = partial(
                planner, compact_k1_relion_layout=True,
                model_current_size_for_batch=model_size,
            )
            from relax.scoring.pass1_plan import global_pass1_relion_projector_texture_enabled
            if (projector_half.dtype == np.dtype(np.complex64)
                and global_pass1_relion_projector_texture_enabled()
                and firstiter_cc_tree_rescore_max_margin is None):
                significance_safe_batch_sizes_for_half = partial(
                    safe_batch_sizes_for_half,
                    score_projector_staging_bytes=int(projector_half.nbytes),
                )
            planner.log.info(
                "Compact firstiter K1 batch planning: model_size=%d deferred=%s coarse_staging=%d",
                model_size, decision.deferred_firstiter_bpref, projector_half.nbytes,
            )
    if use_adaptive:
        adaptive_batch_plan = _plan_adaptive_dense_batch_sizes(
            n_rot=rotations.shape[0],
            n_trans=translations.shape[0],
            n_classes=n_classes,
            image_shape=dataset.image_shape,
            cs_for_engine=cs_for_engine,
            coarse_cs=coarse_cs,
            significance_safe_batch_sizes=significance_safe_batch_sizes_for_half,
            safe_batch_sizes=safe_batch_sizes_for_half,
        )
        k_class_image_batch_size = adaptive_batch_plan.pass2_image_batch_size
        dense_k_class_rotation_block_size = adaptive_batch_plan.pass2_rotation_block_size
        significance_image_batch_size = adaptive_batch_plan.significance_image_batch_size
        significance_rotation_block_size = adaptive_batch_plan.significance_rotation_block_size
        class_batch_overrides = None
        if multi_shape_halves:
            # Each shape class is planned for its own image box and sizes.
            class_batch_overrides = _class_adaptive_batch_overrides(
                dataset,
                plan=partial(
                    _plan_adaptive_dense_batch_sizes,
                    n_rot=rotations.shape[0],
                    n_trans=translations.shape[0],
                    n_classes=n_classes,
                    significance_safe_batch_sizes=significance_safe_batch_sizes_for_half,
                    safe_batch_sizes=safe_batch_sizes_for_half,
                ),
                cs_for_engine=cs_for_engine,
                coarse_cs=coarse_cs,
                coarse_sizing=coarse_sizing,
            )
    elif k_class_enabled:
        k_class_image_batch_size, dense_k_class_rotation_block_size = planner(
            rotations.shape[0],
            translations.shape[0],
            classes=n_classes,
            image_shape_for_batch=dataset.image_shape,
            current_size_for_batch=cs_for_engine,
        )
        k_class_image_batch_size = min(
            k_class_image_batch_size,
            _safe_firstiter_cc_image_batch_size(
                translations.shape[0],
                dataset.image_shape,
            ),
        )
        dense_k_class_rotation_block_size = min(
            dense_k_class_rotation_block_size,
            _safe_dense_k_class_rotation_block_size(
                translations.shape[0],
                k_class_image_batch_size,
            ),
        )
    if k_class_enabled:
        if k_class_image_batch_size != batching.image_batch_size:
            planner.log.info(
                "STRICT-PARITY: clamping dense K-class image_batch_size from %d to %d",
                batching.image_batch_size,
                k_class_image_batch_size,
            )
        if dense_k_class_rotation_block_size != batching.rotation_block_size:
            planner.log.info(
                "STRICT-PARITY: clamping dense K-class rotation_block_size from %d to %d",
                batching.rotation_block_size,
                dense_k_class_rotation_block_size,
            )
        if use_adaptive and (
            significance_image_batch_size != k_class_image_batch_size
            or significance_rotation_block_size != dense_k_class_rotation_block_size
        ):
            planner.log.info(
                "RELION adaptive pass-1 significance batch sizing: "
                "image_batch_size=%d rotation_block_size=%d "
                "(pass2 image_batch_size=%d rotation_block_size=%d, "
                "coarse_current_size=%s fine_current_size=%s)",
                significance_image_batch_size,
                significance_rotation_block_size,
                k_class_image_batch_size,
                dense_k_class_rotation_block_size,
                coarse_cs,
                cs_for_engine,
            )
    return HalfBatchPlan(
        class_overrides=(
            class_batch_overrides if use_adaptive else None
        ),
        safe_batch_sizes=safe_batch_sizes_for_half,
        significance_safe_batch_sizes=significance_safe_batch_sizes_for_half,
        fine_image_batch_size=k_class_image_batch_size,
        fine_rotation_block_size=dense_k_class_rotation_block_size,
        coarse_image_batch_size=significance_image_batch_size,
        coarse_rotation_block_size=significance_rotation_block_size,
    )
