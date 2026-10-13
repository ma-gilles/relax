"""Refinement expectation batches and their run-scoped memory planner."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace

import numpy as np

from relax.classification.k_class_inputs import select_projector_half_for_class
from relax.numerics.precision import DensePrecisionPolicy
from relax.projection.projection import host_relion_projector_texture_enabled
from relax.reconstruction.half_volume_mstep import half_volume_accumulator_shape, relion_backprojector_volume_shape
from relax.refinement import firstiter_bpref
from relax.refinement.refinement_options import ExecutionOptions
from relax.relion.geometry import PROJECTION_PADDING_FACTOR, RECONSTRUCTION_PADDING_FACTOR
from relax.runtime import memory_budget
from relax.runtime.batch_planning import (
    RELION_EM_COMPACT_K1_FIXED_BASE_GB,
    estimate_relion_em_batch_sizes,
    plan_adaptive_dense_batch_sizes,
)
from relax.runtime.engine_tuning import EngineTuning
from relax.steps import shape_class_scoring


@dataclass(frozen=True, eq=False, kw_only=True)
class BatchPlanner:
    """Run requests and fixed grids; query available memory when a pass is planned."""

    requested: ExecutionOptions
    image_shape: tuple
    volume_shape: tuple
    n_classes: int
    # The run's dense precision (RefinementOptions.precision).
    precision: DensePrecisionPolicy
    # How the engines execute (RefinementOptions.environment.engine).
    engine_tuning: EngineTuning
    log: object

    def __call__(
        self, n_rot, n_trans, *, classes=None, image_shape_for_batch=None,
        current_size_for_batch=None, compact_k1_relion_layout=False,
        compact_k1_relion_score_bpref_overlap=False,
        compact_accumulator_shape_for_batch=None, score_projector_staging_bytes=0,
        windowed_translation=False, diagnostic_float64_pass2=False,
    ):
        """Reduce batch sizes using the selected phase's pending allocations."""
        use_float64_scoring_for_batch = self.precision.use_float64_scoring or diagnostic_float64_pass2
        runtime_free_memory_gb = None
        if compact_k1_relion_layout:
            # CUDA textures allocate outside XLA's reusable pool, so the physical reading bounds them; under a
            # pool limit (XLA_PYTHON_CLIENT_MEM_FRACTION, or a pool holding most of the card) the run may only
            # use what the allocator can still hand out, which is the smaller reading there (relax#44).
            physical = memory_budget._device_free_memory_bytes()
            in_pool = memory_budget.device_available_bytes(
                physical,
                memory_budget._jax_allocator_free_memory_bytes(),
                memory_budget._jax_allocator_pool_free_bytes(),
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
        plan = estimate_relion_em_batch_sizes(
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
            compact_accumulator_shape=compact_accumulator_shape_for_batch,
            runtime_free_memory_gb=runtime_free_memory_gb,
            score_projector_staging_bytes=score_projector_staging_bytes,
            windowed_translation=windowed_translation,
            projection_fraction=self.engine_tuning.em_batch_projection_fraction,
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

    def for_expectation(self, *, diagnostic_float64_pass2: bool):
        """The planner of one expectation: itself, or float64 sizes when the scoring side runs this iteration's
        pass 2 in float64 (``precision._diagnostic_float64_pass2_matches`` of the same iteration)."""
        return partial(self, diagnostic_float64_pass2=True) if diagnostic_float64_pass2 else self


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
        class_cs, class_coarse = shape_class_scoring.class_adaptive_sizes(
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
    use_relion_x_half_mstep: bool,
    multi_shape_halves,
    coarse_sizing,
    diagnostic_float64_pass2: bool,
    bpref_half: firstiter_bpref.NumberedBPrefHalf,
) -> HalfBatchPlan:
    """Resolve compact staging and coarse/fine batches before one half scores.

    Grids are inspected only in the branches that need them. Dense scoring uses
    the safe-size callbacks and the coarse/fine sizes; local scoring sizes its own tiles.
    ``diagnostic_float64_pass2``: this iteration's pass 2 scores in float64 (a listed iteration of
    RELAX_DIAGNOSTIC_FLOAT64_PASS2_ITERATIONS), so its plan is sized for float64 and is not compact.
    ``bpref_half`` is the numbered half being planned, for the first-iteration BPref gates.
    """
    batching = planner.requested
    n_classes = planner.n_classes
    k_class_enabled = n_classes > 1
    volume_shape = planner.volume_shape
    k_class_image_batch_size = batching.image_batch_size
    dense_k_class_rotation_block_size = batching.rotation_block_size
    significance_image_batch_size = None
    significance_rotation_block_size = None
    safe_batch_sizes_for_half = planner.for_expectation(diagnostic_float64_pass2=diagnostic_float64_pass2)
    significance_safe_batch_sizes_for_half = safe_batch_sizes_for_half
    projector_half = select_projector_half_for_class(
        None if projector is None else projector.data, 0, n_classes,
    )
    compact_precision = not (
        planner.precision.use_float64_scoring
        or planner.precision.use_float64_projections
        or diagnostic_float64_pass2
    )
    if (
        use_adaptive and not use_local and not k_class_enabled
        and relion_firstiter_cc_this_iter and compact_precision
        and host_relion_projector_texture_enabled(
            projector_half, r_max=None if projector is None else projector.r_max,
            padding_factor=PROJECTION_PADDING_FACTOR, allow_float32_cast=True,
        )
    ):
        model_size = int(volume_shape[0] if model_current_size_for_engine is None
                         else model_current_size_for_engine)
        recon_shape = relion_backprojector_volume_shape(
            volume_shape, RECONSTRUCTION_PADDING_FACTOR, current_size=model_size,
        )
        compact_accumulator_shape = half_volume_accumulator_shape(recon_shape)
        decision = firstiter_bpref.relion_firstiter_compact_batch_planning_decision(
            source_faithful_spectrum_norm=source_faithful_spectrum_norm,
            winner_take_all=firstiter_winner_take_all_this_iter,
            preserve_bpref_particle_order=preserve_bpref_particle_order,
            use_relion_x_half_mstep=use_relion_x_half_mstep,
            projector_half=SimpleNamespace(shape=projector_half.shape, dtype=np.dtype(np.complex64)),
            score_complex_dtype=np.complex64,
            recon_volume_size=int(np.prod(compact_accumulator_shape)),
            fixed_base_bytes=int(RELION_EM_COMPACT_K1_FIXED_BASE_GB * 1e9),
            half=bpref_half,
        )
        if decision.enabled:
            if model_size <= 0:
                raise ValueError(f"model_current_size must be positive, got {model_size}")
            safe_batch_sizes_for_half = partial(
                planner, compact_k1_relion_layout=True,
                compact_accumulator_shape_for_batch=compact_accumulator_shape,
            )
            if projector_half.dtype == np.dtype(np.complex64) and firstiter_cc_tree_rescore_max_margin is None:
                significance_safe_batch_sizes_for_half = partial(
                    safe_batch_sizes_for_half,
                    score_projector_staging_bytes=int(projector_half.nbytes),
                )
            planner.log.info(
                "Compact firstiter K1 batch planning: model_size=%d deferred=%s coarse_staging=%d",
                model_size, decision.deferred_firstiter_bpref, projector_half.nbytes,
            )
    if use_adaptive:
        adaptive_batch_plan = plan_adaptive_dense_batch_sizes(
            n_rot=rotations.shape[0],
            n_trans=translations.shape[0],
            n_classes=n_classes,
            image_shape=dataset.image_shape,
            cs_for_engine=cs_for_engine,
            coarse_cs=coarse_cs,
            significance_safe_batch_sizes=significance_safe_batch_sizes_for_half,
            safe_batch_sizes=safe_batch_sizes_for_half,
            firstiter_cc_recon_complex_budget=planner.engine_tuning.firstiter_cc_recon_complex_budget,
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
                    plan_adaptive_dense_batch_sizes,
                    n_rot=rotations.shape[0],
                    n_trans=translations.shape[0],
                    n_classes=n_classes,
                    significance_safe_batch_sizes=significance_safe_batch_sizes_for_half,
                    safe_batch_sizes=safe_batch_sizes_for_half,
                    firstiter_cc_recon_complex_budget=planner.engine_tuning.firstiter_cc_recon_complex_budget,
                ),
                cs_for_engine=cs_for_engine,
                coarse_cs=coarse_cs,
                coarse_sizing=coarse_sizing,
            )
    # A non-adaptive half is given no batch sizes from this plan: its fine sizes stay the requested ones.
    if k_class_enabled and use_adaptive:
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
        if (
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
