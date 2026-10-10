"""Dispatch dense scoring for one refinement half-set.

The iteration controller owns scheduling, state transitions, result-container
lifetime and device offloading. These adapters choose existing engine routes,
prepare per-half arguments and populate caller-owned result slots. A diagnostic
BPref scope surrounds each half's scoring call.
"""

import logging
from dataclasses import dataclass, replace

import jax.numpy as jnp
import numpy as np
from recovar import utils

from relax.classification.k_class import run_dense_k_class_em_adaptive
from relax.classification.k_class_results import KClassEMResult
from relax.cuda import kernels as em_cuda_kernels
from relax.diagnostics import parity_dump as _parity_dump
from relax.refinement import scoring_policy
from relax.refinement.half_inputs import HalfScoringData
from relax.refinement.precision import DensePrecisionPolicy
from relax.refinement.score_outputs import (
    HalfScoreResult,
    _collapse_fine_pose_assignments_to_coarse,
    _collapse_single_class_stats_to_coarse,
    _select_single_class_accumulator,
    class_em_to_half_result,
)
from relax.refinement.scoring_policy import (
    RELION_ADAPTIVE_FRACTION,
    RELION_FOURIER_WINDOW_SQUARE,
)
from relax.refinement.shape_class_scoring import OpticsSpec, engine_projection_inputs, reference_grid_kwargs
from relax.relion.geometry import (
    PROJECTION_PADDING_FACTOR,
    RECONSTRUCTION_PADDING_FACTOR,
)
from relax.relion.optics_aberrations import (
    dataset_magnification_is_anisotropic,
    dataset_projection_magnification,
    reported_rotations,
)
from relax.runtime.batch_planning import (
    _plan_kclass_adaptive_grid_batch_sizes,
    safe_dense_k_class_rotation_block_size,
    safe_firstiter_cc_image_batch_size,
)
from relax.sampling import (
    apply_relion_translation_perturbation,
    project_rows,
)
from relax.sampling.oversampling import (
    AdaptivePass2Grids,
    build_adaptive_pass2_grids,
    prepare_adaptive_pass2_grids,
    project_pass2_rotations,
)
from relax.sampling.symmetry import canonicalize_rotational_symmetry

logger = logging.getLogger("relax.dense.half_scoring")


def expand_significant_samples_to_full_parent_translations(
    significant_sample_indices,
    n_parent_translations: int,
):
    """Expand significant parent rotation ids to every parent translation."""

    if n_parent_translations <= 0:
        raise ValueError(f"n_parent_translations must be positive, got {n_parent_translations}")
    expanded = []
    parent_translations = np.arange(n_parent_translations, dtype=np.int64)
    for samples in significant_sample_indices:
        if samples is None:
            expanded.append(None)
            continue
        samples_np = np.asarray(samples, dtype=np.int64).reshape(-1)
        if samples_np.size == 0:
            expanded.append(samples_np)
            continue
        parent_rotations = np.unique(samples_np // n_parent_translations).astype(np.int64, copy=False)
        expanded_samples = (parent_rotations[:, None] * n_parent_translations + parent_translations[None, :]).reshape(
            -1
        )
        expanded.append(expanded_samples.astype(np.int64, copy=False))
    return expanded


@dataclass(frozen=True, kw_only=True)
class DenseSamplingSpec:
    """Resolved dense sampling grids, oversampling and reconstruction sizes."""

    effective_rotations: object
    current_translations: object
    base_translations: object
    current_healpix_order: int
    oversampling_order: int
    translation_step: float
    random_perturbation: float
    image_window_size: int | None
    coarse_engine: str
    model_support_size: int | None = None
    # --strict_highres_exp: the weighted sums' image size, above image_window_size (None: image_window_size).
    wsum_current_size: int | None = None
    coarse_angular_step_deg: float | None = None
    coarse_rotation_ids: object | None = None
    coarse_scoring_rotations: object | None = None
    symmetry: str
    # RELION's device-built pass-1 source of ``effective_rotations`` / ``coarse_scoring_rotations``
    # (None: host-built rows); images on another grid or magnified rebuild their rows from it.
    effective_device_source: object | None = None
    coarse_scoring_device_source: object | None = None

    def rotation_source(self, adaptive_oversampling: int, symmetry: str, *, coarse_scoring: bool = False) -> dict:
        """Provenance of :meth:`pass2_grids` rows for :func:`project_pass2_rotations`; ``coarse_scoring``
        when the scored coarse rows are ``coarse_scoring_rotations``."""
        return {
            "coarse_healpix_order": self.current_healpix_order,
            "adaptive_oversampling": adaptive_oversampling,
            "random_perturbation": self.random_perturbation,
            "coarse_rotation_ids": self.coarse_rotation_ids,
            "symmetry": symmetry,
            "coarse_device_source": (
                self.coarse_scoring_device_source if coarse_scoring else self.effective_device_source
            ),
            "grid_device_source": self.effective_device_source,
        }

    def pass2_grids(self, *, adaptive_oversampling: int, symmetry: str) -> AdaptivePass2Grids:
        """The adaptive pass-1 and pass-2 trial grids of these sampling settings."""
        return prepare_adaptive_pass2_grids(
            self.effective_rotations,
            self.current_translations,
            self.base_translations,
            healpix_order=self.current_healpix_order,
            adaptive_oversampling=adaptive_oversampling,
            translation_step=self.translation_step,
            random_perturbation=self.random_perturbation,
            coarse_rotation_ids=self.coarse_rotation_ids,
            symmetry=symmetry,
        )


@dataclass(frozen=True, kw_only=True)
class DensePriorSpec:
    """Direction, translation and class priors for one dense half."""

    rotation_log_prior_k: object
    class_rotation_log_prior_k: object
    translation_log_prior: object
    translation_search_base: object
    trans_prior_center_for_engine: object
    class_log_priors: object


@dataclass(frozen=True, kw_only=True)
class DenseBatchPolicy:
    """Existing batch planners and route-specific overrides."""

    image_batch_size: int
    safe_batch_sizes: object
    max_significants: int | None
    significance_safe_batch_sizes: object | None
    k_class_image_batch_size_override: int | None
    k_class_rotation_block_size_override: int | None
    significance_image_batch_size_override: int | None
    significance_rotation_block_size_override: int | None
    # Multi-shape halves: each shape class's own batch sizes (expectation_batches.ShapeClassBatchSizes).
    class_batch_overrides: tuple | None


@dataclass(frozen=True, kw_only=True)
class DenseVariantPolicy:
    """Explicit dense, adaptive and first-iteration route selections.

    ``coarse_window_size`` and ``fine_window_size`` are the adaptive passes' image windows on every
    iteration (None: the engine's current size; the fine one is None on a non-adaptive pass).
    """

    score_mode: str
    winner_take_all: bool
    k_class_enabled: bool
    firstiter_cc: bool
    coarse_window_size: int | None
    fine_window_size: int | None
    # RELION --skip_align: classify at each particle's stored pose (relax.refinement.given_poses).
    skip_align: bool


@dataclass(frozen=True, kw_only=True)
class DenseExecutionPolicy:
    """Interpolation, ablation and dense diagnostic settings."""

    disc_type: str
    return_best_pose_details: bool
    bpref_device_signature_active: bool
    debug_iteration: int | None
    diagnostic_float64_pass2: bool
    preserve_bpref_particle_order: bool
    source_faithful_spectrum_norm: bool
    relion_translation_angle_scale: float
    # The K=1 --firstiter_cc coarse-tree top-2 rescore margin (RelionParityOptions); None is off.
    firstiter_cc_tree_rescore_max_margin: float | None
    # RelionConsistencyOptions: the support of the first-iteration normalized CC, and how the
    # per-image sums count the Hermitian pairs of the full-size Nyquist column.
    firstiter_cc_support: str
    nyquist_column_counting: str
    # ScoringVariants.relion_x_half_mstep for this half's class count: RELION's x-half M-step accumulators.
    relion_x_half_mstep: bool
    # The run's dense precision (RefinementOptions.precision): engine precision and the pose dtype.
    precision: DensePrecisionPolicy


def _score_adaptive_kclass_dense(
    half: HalfScoringData,
    sampling: DenseSamplingSpec,
    priors: DensePriorSpec,
    batching: DenseBatchPolicy,
    variant: DenseVariantPolicy,
    execution: DenseExecutionPolicy,
    optics: OpticsSpec,
    em_kwargs,
    symmetry,
) -> tuple[object, AdaptivePass2Grids]:
    """Run the ordinary adaptive K-class engine and return its trial grids."""

    adaptive_os = int(sampling.oversampling_order)
    coarse_current_size = variant.coarse_window_size
    fine_current_size = variant.fine_window_size
    if adaptive_os <= 0:
        coarse_current_size = sampling.image_window_size
        fine_current_size = sampling.image_window_size
        logger.info(
            "RELION K-class scale groups at oversampling 0: routing the single pass through the "
            "adaptive engine (current_size=%s)",
            sampling.image_window_size,
        )
    pass2_grids = sampling.pass2_grids(adaptive_oversampling=adaptive_os, symmetry=symmetry)
    adaptive_em_kwargs = dict(em_kwargs)
    n_classes = int(np.shape(half.reference)[0]) if np.ndim(half.reference) >= 2 else 1
    grid_batch_plan = _plan_kclass_adaptive_grid_batch_sizes(
        coarse_rotations=pass2_grids.coarse_rotations,
        coarse_translations=pass2_grids.coarse_translations,
        fine_rotations=pass2_grids.fine_rotations,
        fine_translations=pass2_grids.fine_translations,
        n_classes=n_classes,
        image_shape=half.particles.dataset.image_shape,
        coarse_current_size=coarse_current_size,
        fine_current_size=fine_current_size,
        safe_batch_sizes=batching.safe_batch_sizes,
        significance_safe_batch_sizes=batching.significance_safe_batch_sizes,
    )
    adaptive_em_kwargs["image_batch_size"] = grid_batch_plan.pass2_image_batch_size
    adaptive_em_kwargs["rotation_block_size"] = grid_batch_plan.pass2_rotation_block_size
    logger.info(
        "RELION adaptive K-class grid batch sizing: "
        "coarse image_batch_size=%d rotation_block_size=%d; "
        "fine image_batch_size=%d rotation_block_size=%d",
        grid_batch_plan.significance_image_batch_size,
        grid_batch_plan.significance_rotation_block_size,
        adaptive_em_kwargs["image_batch_size"],
        adaptive_em_kwargs["rotation_block_size"],
    )
    adaptive_em_kwargs["sparse_pass2"] = True
    logger.info(
        "RELION adaptive K-class routing through run_dense_k_class_em_adaptive "
        "(oversampling=%d, pass2_backend=sparse, fine_mstep_prune=True)",
        adaptive_os,
    )
    # Images on another grid (applyScaleDifference) or magnified (applyAnisoMag): the
    # projection and backprojection matrices carry it.
    magnification, grid_kwargs = engine_projection_inputs(
        half.particles.dataset,
        scale=optics.projection_scale,
        reference_current_size=optics.reference_current_size,
    )
    projected_coarse, projected_fine, projected_mstep = project_pass2_rotations(
        pass2_grids.coarse_rotations,
        pass2_grids.fine_rotations,
        pass2_grids.fine_mstep_rotations,
        scale=optics.projection_scale,
        magnification=magnification,
        **sampling.rotation_source(adaptive_os, symmetry),
    )
    adaptive_em_kwargs.update(grid_kwargs)
    result = run_dense_k_class_em_adaptive(
        half.particles.dataset,
        half.reference,
        half.mean_variance,
        half.noise_variance,
        projected_coarse,
        pass2_grids.coarse_translations,
        projected_fine,
        pass2_grids.fine_translations,
        pass2_grids.rotation_parent_map,
        pass2_grids.translation_parent_map,
        execution.disc_type,
        significance_image_batch_size=grid_batch_plan.significance_image_batch_size,
        significance_rotation_block_size=grid_batch_plan.significance_rotation_block_size,
        coarse_current_size=coarse_current_size,
        fine_current_size=fine_current_size,
        oversampling_order=adaptive_os,
        image_seed_classes=half.image_seed_classes,
        class_log_priors=priors.class_log_priors,
        accumulate_noise=True,
        adaptive_fraction=RELION_ADAPTIVE_FRACTION,
        max_significants=(-1 if batching.max_significants is None else int(batching.max_significants)),
        relion_fine_mstep_prune=True,
        coarse_healpix_order=int(sampling.current_healpix_order),
        fine_mstep_rotations_override=projected_mstep,
        return_best_pose_details=execution.return_best_pose_details,
        bpref_device_signature_active=execution.bpref_device_signature_active,
        debug_iteration=execution.debug_iteration,
        **adaptive_em_kwargs,
    )
    return result, pass2_grids


def _score_kclass_at_given_poses(
    half: HalfScoringData,
    sampling: DenseSamplingSpec,
    priors: DensePriorSpec,
    batching: DenseBatchPolicy,
    execution: DenseExecutionPolicy,
    optics: OpticsSpec,
    em_kwargs,
    symmetry,
    *,
    firstiter_cc: bool = False,
):
    """RELION ``--skip_align``: the K-class pass 2 at each particle's stored pose, no pass 1.

    Each particle's one sample is its stored orientation and what is left of its stored offset
    after the rounded offset the image pre-shift already applies (``priors.translation_search_base``,
    RELION's ``old_offset - ROUND(old_offset)``, ml_optimiser.cpp:4684-4720); pass 2 translates the
    particle's images by that remainder (``given_image_translations``). The posterior is over the
    classes alone (``pdf_orientation = pdf_class``, :8454; one translation makes the normalised
    ``pdf_offset`` 1). ``wsum_sigma2_offset`` still sums ``|prior - old_offset|^2`` over the whole
    offset (:8690), so the remainder moves the engine's sigma-offset center. Returns the engine
    result with the stored Euler angles as its best-pose angles, and the grids it scored.

    ``firstiter_cc`` is RELION's ``--firstiter_cc`` iteration at the given poses: RELION accepts it with
    ``--skip_align``, collapses the sample range to the particle's own pose (ml_optimiser.cpp:4989) and
    scores it by normalized cross-correlation (exp_local_sqrtXi2, :8021) against the one reference
    (:4389-4402), its one sample taking the whole weight; the scale and sigma skips of that iteration
    apply as in the searched one (:6190, :6316, :6385).
    """

    from relax.refinement.given_poses import given_pose_grids

    particles = half.particles
    if particles.rotation_eulers is None or particles.translations is None:
        raise ValueError("--skip_align classifies at the input poses: the particle STAR needs angles and offsets")
    eulers = np.asarray(particles.rotation_eulers, dtype=np.float64)
    stored = np.asarray(particles.translations, dtype=np.float64)
    base = np.zeros_like(stored) if priors.translation_search_base is None else np.asarray(priors.translation_search_base)
    pose_dtype = execution.precision.rotation_real_dtype
    grids = given_pose_grids(
        np.asarray(utils.R_from_relion(eulers, degrees=True), dtype=pose_dtype),
        # RELION's offsets are RFLOAT: the remainder stays double until pass 2 forms its phase.
        stored - np.asarray(base, dtype=np.float64),
        symmetry=symmetry,
    )
    n_classes = int(np.shape(half.reference)[0])
    plan = _plan_kclass_adaptive_grid_batch_sizes(
        coarse_rotations=grids.rotations[:1],
        coarse_translations=grids.translations[:1],
        fine_rotations=grids.rotations[:1],
        fine_translations=grids.translations[:1],
        n_classes=n_classes,
        image_shape=particles.dataset.image_shape,
        coarse_current_size=sampling.image_window_size,
        fine_current_size=sampling.image_window_size,
        safe_batch_sizes=batching.safe_batch_sizes,
        significance_safe_batch_sizes=batching.significance_safe_batch_sizes,
    )
    engine_kwargs = dict(em_kwargs)
    engine_kwargs["image_batch_size"] = plan.pass2_image_batch_size
    engine_kwargs["rotation_block_size"] = plan.pass2_rotation_block_size
    engine_kwargs["sparse_pass2"] = True
    # No orientational or translational prior and no perturbation (ml_optimiser.cpp:2619-2626).
    for name in ("rotation_log_prior", "class_rotation_log_prior", "translation_log_prior", "rotation_translation_mask"):
        engine_kwargs.pop(name, None)
    centers = engine_kwargs.get("translation_prior_centers")
    if centers is not None:
        # The engine's squared distance is |translation - center|^2 at the zero translation.
        centers = np.broadcast_to(np.asarray(centers, dtype=np.float64), grids.image_translations.shape)
        engine_kwargs["translation_prior_centers"] = (centers - grids.image_translations).astype(pose_dtype)
    magnification, grid_kwargs = engine_projection_inputs(
        particles.dataset,
        scale=optics.projection_scale,
        reference_current_size=optics.reference_current_size,
    )
    # The given poses are host-built rows (RELION's matrices of the stored RFLOAT angles).
    projected_rotations = project_rows(
        grids.rotations,
        optics.projection_scale,
        magnification,
        host_rows=lambda: (
            given_pose_grids(
                utils.R_from_relion(eulers, degrees=True).astype(np.float64), stored, symmetry=symmetry
            ).rotations
        ),
        what="given-pose rows",
    )
    engine_kwargs.update(grid_kwargs)
    logger.info(
        "RELION --skip_align: %d particles scored against %d classes at their stored poses (HEALPix order %d holds the list)",
        eulers.shape[0], n_classes, grids.healpix_order,
    )
    result = run_dense_k_class_em_adaptive(
        particles.dataset,
        half.reference,
        half.mean_variance,
        half.noise_variance,
        projected_rotations,
        grids.translations,
        projected_rotations,
        grids.translations,
        grids.rotation_parent_map,
        grids.translation_parent_map,
        execution.disc_type,
        class_log_priors=priors.class_log_priors,
        accumulate_noise=True,
        coarse_current_size=sampling.image_window_size,
        fine_current_size=sampling.image_window_size,
        coarse_healpix_order=grids.healpix_order,
        oversampling_order=0,
        relion_fine_mstep_prune=True,
        return_best_pose_details=True,
        bpref_device_signature_active=execution.bpref_device_signature_active,
        debug_iteration=execution.debug_iteration,
        image_seed_classes=half.image_seed_classes,
        firstiter_cc_pass2_only_best_coarse=bool(firstiter_cc),
        given_supports=grids.supports,
        given_image_translations=grids.image_translations,
        **engine_kwargs,
    )
    return result, grids


def _given_pose_collapse(given):
    """``(rotation parent map, translation parent map, fine translations)`` for the coarse ids of a given-pose pass.

    The list of stored poses is not the sampling grid: its rotation and translation ids collapse onto the
    grid's first sample (RELION does not read pdf_direction under --skip_align either), and the poses go
    back as the explicit best-pose details (:func:`_keep_given_poses`).
    """

    return (
        np.zeros_like(given.rotation_parent_map),
        np.zeros_like(given.translation_parent_map),
        int(given.translations.shape[0]),
    )


def _keep_given_poses(score_result, half, given) -> None:
    """A given-pose pass returns each particle's pose as it came and leaves the direction prior alone."""

    # RELION leaves pdf_direction as it is under --skip_align (ml_optimiser.cpp:5198, :9133): no rotation
    # mass, so the loop keeps the direction prior.
    score_result.classes = replace(score_result.classes, rotation_mass=None)
    # The stored angles go back as they came, not through a matrix round trip.
    score_result.best_pose_rotation_eulers = np.asarray(half.particles.rotation_eulers, dtype=np.float64)
    # The engine scored the zero translation of pre-translated images: the offset is the image's own
    # remainder, relative to the pre-shift as the search grid's translations are.
    score_result.best_pose_translations = np.asarray(
        given.image_translations, dtype=score_result.best_pose_translations.dtype
    )


def _score_adaptive_k1_dense(
    half: HalfScoringData,
    sampling: DenseSamplingSpec,
    priors: DensePriorSpec,
    batching: DenseBatchPolicy,
    variant: DenseVariantPolicy,
    execution: DenseExecutionPolicy,
    optics: OpticsSpec,
    base_em_kwargs,
    *,
    symmetry,
) -> tuple[object, AdaptivePass2Grids]:
    """Run the ordinary adaptive K=1 engine and return its trial grids."""

    adaptive_os = int(sampling.oversampling_order)
    coarse_current_size = variant.coarse_window_size
    fine_current_size = variant.fine_window_size
    if adaptive_os <= 0:
        # The sparse engine supplies group statistics and per-particle BPref
        # launches even for a single pass on the current grid.
        coarse_current_size = sampling.image_window_size
        fine_current_size = sampling.image_window_size
        logger.info(
            "RELION K=1 group statistics or BPref order at oversampling 0: single pass through the "
            "adaptive engine (current_size=%s)",
            sampling.image_window_size,
        )
    if symmetry != "C1" and not execution.relion_x_half_mstep:
        raise RuntimeError(
            f"{symmetry} reconstruction requires RELION x-half BPref accumulation; "
            "RELAX_K1_RELION_X_HALF_MSTEP=0, CPU-only execution, or disabled "
            "custom CUDA is unsupported for non-C1 symmetry"
        )
    means_single = jnp.asarray(half.reference)[None, :]
    pass2_grids = sampling.pass2_grids(adaptive_oversampling=adaptive_os, symmetry=symmetry)
    adaptive_em_kwargs = dict(base_em_kwargs)
    adaptive_em_kwargs["sparse_pass2"] = True
    logger.info(
        "RELION adaptive K=1 routing through run_dense_k_class_em_adaptive "
        "(oversampling=%d, pass2_backend=sparse, "
        "fine_mstep_prune=True, relion_x_half_mstep=%s, supplied_ppref=%s, "
        "engine_ppref=%s)",
        adaptive_os,
        bool(execution.relion_x_half_mstep),
        half.projector is not None,
        adaptive_em_kwargs.get("relion_projector_half") is not None,
    )
    magnification, grid_kwargs = engine_projection_inputs(
        half.particles.dataset,
        scale=optics.projection_scale,
        reference_current_size=optics.reference_current_size,
    )
    coarse_scoring = (
        sampling.coarse_scoring_rotations is not None
        and adaptive_os == 0
        and execution.relion_x_half_mstep
        and variant.score_mode == "gaussian"
        and not execution.diagnostic_float64_pass2
    )
    projected_coarse, projected_fine, projected_mstep = project_pass2_rotations(
        sampling.coarse_scoring_rotations if coarse_scoring else pass2_grids.coarse_rotations,
        pass2_grids.fine_rotations,
        pass2_grids.fine_mstep_rotations,
        scale=optics.projection_scale,
        magnification=magnification,
        **sampling.rotation_source(adaptive_os, symmetry, coarse_scoring=coarse_scoring),
    )
    adaptive_em_kwargs.update(grid_kwargs)
    k1_adaptive_result = run_dense_k_class_em_adaptive(
        half.particles.dataset,
        means_single,
        half.mean_variance,
        half.noise_variance,
        projected_coarse,
        pass2_grids.coarse_translations,
        projected_fine,
        pass2_grids.fine_translations,
        pass2_grids.rotation_parent_map,
        pass2_grids.translation_parent_map,
        execution.disc_type,
        pass2_use_float64_scoring=True if execution.diagnostic_float64_pass2 else None,
        pass2_use_float64_projections=True if execution.diagnostic_float64_pass2 else None,
        coarse_translation_phase_source=pass2_grids.coarse_translation_phase_source,
        significance_image_batch_size=batching.significance_image_batch_size_override,
        significance_rotation_block_size=batching.significance_rotation_block_size_override,
        coarse_current_size=coarse_current_size,
        fine_current_size=fine_current_size,
        oversampling_order=adaptive_os,
        class_log_priors=priors.class_log_priors,
        accumulate_noise=True,
        adaptive_fraction=RELION_ADAPTIVE_FRACTION,
        max_significants=(-1 if batching.max_significants is None else int(batching.max_significants)),
        relion_fine_mstep_prune=True,
        coarse_healpix_order=int(sampling.current_healpix_order),
        fine_mstep_rotations_override=projected_mstep,
        return_best_pose_details=execution.return_best_pose_details,
        bpref_device_signature_active=execution.bpref_device_signature_active,
        debug_iteration=execution.debug_iteration,
        **adaptive_em_kwargs,
    )
    return k1_adaptive_result, pass2_grids


def _score_half_dense_one_shape(
    half: HalfScoringData,
    sampling: DenseSamplingSpec,
    priors: DensePriorSpec,
    batching: DenseBatchPolicy,
    variant: DenseVariantPolicy,
    execution: DenseExecutionPolicy,
    optics: OpticsSpec,
) -> HalfScoreResult:
    """Dense (non-local-search) E+M scoring for one half-set: its statistics, class summaries and poses.

    ``half.particles.optics_group_ids`` gives each image's row of a
    per-optics-group ``noise_variance_k`` table
    (:mod:`relax.relion.optics_noise`); only the K=1 adaptive route carries it,
    every other engine refuses it.

    ``optics.projection_scale`` and ``reference_current_size`` describe
    images on another grid than the reference (one shape class of
    :mod:`relax.refinement.optics_shapes`): the projection and backprojection
    matrices are divided by the scale and the backprojector keeps the reference
    model size; reported poses stay unscaled.

    ``batching.safe_batch_sizes`` plans against memory available at invocation.
    """

    from relax.sampling.symmetry import canonicalize_rotational_symmetry

    symmetry = canonicalize_rotational_symmetry(sampling.symmetry)
    if symmetry != "C1" and int(sampling.oversampling_order) <= 0 and sampling.coarse_engine != "gemm_dense":
        raise NotImplementedError(
            f"{symmetry} reconstruction at adaptive oversampling 0 is unsupported; "
            "use adaptive oversampling >= 1 or the exact-local RELION x-half M-step"
        )
    safe_ibs, safe_rbs = batching.safe_batch_sizes(
        sampling.effective_rotations.shape[0],
        sampling.current_translations.shape[0],
        current_size_for_batch=sampling.image_window_size,
    )
    if half.image_seed_classes is not None and not variant.k_class_enabled:
        raise NotImplementedError("a seed iteration runs on the adaptive or first-iteration CC K-class route")
    if execution.diagnostic_float64_pass2:
        logger.info(
            "Diagnostic genuine-float64 adaptive pass 2 at iteration %d; pass 1 and prior boundaries remain f32",
            int(execution.debug_iteration),
        )
    # The engine keywords of every dense route. A keyword at the engine's own default is the same call as an
    # omitted one (each is read with that default), so every keyword is passed.
    em_kwargs = {
        "score_with_masked_images": True,
        "half_spectrum_scoring": True,
        "projection_padding_factor": PROJECTION_PADDING_FACTOR,
        "reconstruction_padding_factor": RECONSTRUCTION_PADDING_FACTOR,
        "use_float64_scoring": execution.precision.use_float64_scoring,
        "use_float64_projections": execution.precision.use_float64_projections,
        "relion_exact_fine_gaussian": scoring_policy.RELION_EXACT_FINE_GAUSSIAN,
        "do_gridding_correction": True,
        "square_window": RELION_FOURIER_WINDOW_SQUARE,
        "image_batch_size": (
            safe_ibs if batching.k_class_image_batch_size_override is None
            else batching.k_class_image_batch_size_override
        ),
        "rotation_block_size": (
            safe_rbs if batching.k_class_rotation_block_size_override is None
            else batching.k_class_rotation_block_size_override
        ),
        "current_size": sampling.image_window_size,
        # A class direction prior replaces the shared one.
        "rotation_log_prior": (
            priors.rotation_log_prior_k if priors.class_rotation_log_prior_k is None else None
        ),
        "class_rotation_log_prior": priors.class_rotation_log_prior_k,
        "translation_log_prior": priors.translation_log_prior,
        "image_corrections": half.particles.image_corrections,
        "scale_corrections": half.particles.scale_corrections,
        "group_ids": half.scale_group_ids,
        "scale_correction_group_count": half.scale_group_count,
        "scale_correction_data_vs_prior": half.scale_correction_data_vs_prior,
        "image_pre_shifts": priors.translation_search_base,
        "translation_prior_centers": priors.trans_prior_center_for_engine,
        "relion_firstiter_score_mode": variant.score_mode,
        "relion_firstiter_winner_take_all": variant.winner_take_all,
        "coarse_engine": sampling.coarse_engine,
        "symmetry_label": symmetry,
        "optics_group_ids": half.particles.optics_group_ids,
        "reconstruction_current_size": sampling.model_support_size,
        "wsum_current_size": sampling.wsum_current_size,
        "preserve_bpref_particle_order": execution.preserve_bpref_particle_order,
        "source_faithful_spectrum_norm": execution.source_faithful_spectrum_norm,
        "firstiter_cc_tree_rescore_max_margin": execution.firstiter_cc_tree_rescore_max_margin,
        "relion_translation_angle_scale": float(execution.relion_translation_angle_scale),
        "firstiter_cc_support": execution.firstiter_cc_support,
        "nyquist_column_counting": execution.nyquist_column_counting,
        "relion_projector_half": None if half.projector is None else half.projector.data,
        "relion_projector_r_max": None if half.projector is None else half.projector.r_max,
        "mstep_relion_x_half": execution.relion_x_half_mstep,
        "relion_half_volume_mstep": False,
    }
    logger.info(
        "Dense half-set projector handoff: supplied_ppref=%s state_oversampling=%d",
        half.projector is not None,
        int(sampling.oversampling_order),
    )
    if variant.k_class_enabled:
        magnification = dataset_projection_magnification(half.particles.dataset)
        # K-class uses RELION's x-half BackProjector accumulator layout by
        # default, matching the K=1 parity path. The explicit selector can
        # still choose the dense full-volume path.
        if symmetry != "C1" and not execution.relion_x_half_mstep:
            raise RuntimeError(f"{symmetry} requires sparse RELION x-half BPref reconstruction")
        k_class_mstep_full_half_axis_this_score = None
        rot_pmap_for_collapse = None
        trans_pmap_for_collapse = None
        n_trans_fine_for_collapse = None
        adaptive_os_local = 0
        # STRICT-PARITY: at iter 1 with --firstiter_cc, route through the
        # adaptive 2-pass engine with normalized-CC scoring. Pass 2 retains the
        # oversampled children of the single best coarse class/pose, matching
        # RELION's firstiter-CC binarized coarse support.
        if variant.skip_align:
            k_class_result, given = _score_kclass_at_given_poses(
                half, sampling, priors, batching, execution, optics, em_kwargs, symmetry,
                firstiter_cc=variant.firstiter_cc,
            )
            rot_pmap_for_collapse, trans_pmap_for_collapse, n_trans_fine_for_collapse = _given_pose_collapse(given)
            k_class_mstep_full_half_axis_this_score = k_class_result.mstep_full_half_axis
        elif variant.firstiter_cc:
            adaptive_os_local = int(sampling.oversampling_order)
            firstiter = _score_kclass_firstiter_cc_pass2(
                half,
                sampling,
                priors,
                batching,
                variant,
                execution,
                projection_scale=optics.projection_scale,
                magnification=magnification,
                em_kwargs={
                    **em_kwargs,
                    **reference_grid_kwargs(optics.reference_current_size, optics.projection_scale, magnification),
                },
            )
            k_class_result = firstiter.result
            rot_pmap_for_collapse = firstiter.rotation_parent_map
            trans_pmap_for_collapse = firstiter.translation_parent_map
            n_trans_fine_for_collapse = firstiter.n_fine_translations
            k_class_mstep_full_half_axis_this_score = k_class_result.mstep_full_half_axis
        else:
            k_class_result, pass2_grids = _score_adaptive_kclass_dense(
                half,
                sampling,
                priors,
                batching,
                variant,
                execution,
                optics,
                em_kwargs,
                symmetry,
            )
            adaptive_os_local = int(sampling.oversampling_order)
            rot_pmap_for_collapse = pass2_grids.rotation_parent_map
            trans_pmap_for_collapse = pass2_grids.translation_parent_map
            n_trans_fine_for_collapse = pass2_grids.n_fine_translations
            k_class_mstep_full_half_axis_this_score = k_class_result.mstep_full_half_axis
        if optics.projection_scale != 1.0 or magnification is not None:
            # Poses are reported unscaled and unmagnified; only projection used the transformed matrices.
            def unmagnified(rotations):
                return None if rotations is None else reported_rotations(rotations, optics.projection_scale, magnification)

            per_class = k_class_result.per_class_best_pose_rotations
            k_class_result = k_class_result._replace(
                best_pose_rotations=unmagnified(k_class_result.best_pose_rotations),
                per_class_best_pose_rotations=None if per_class is None else tuple(map(unmagnified, per_class)),
            )
        score_result = class_em_to_half_result(
            k_class_result,
            effective_rotations=given.rotations if variant.skip_align else sampling.effective_rotations,
            rot_pmap_for_collapse=rot_pmap_for_collapse,
            adaptive_os_local=adaptive_os_local,
            require_best_pose_details=execution.return_best_pose_details or variant.skip_align,
            pose_dtype=execution.precision.rotation_real_dtype,
        )
        if variant.skip_align:
            _keep_given_poses(score_result, half, given)
        score_result.coarse_ha = _collapse_fine_pose_assignments_to_coarse(
            score_result.ha,
            rot_parent_map=rot_pmap_for_collapse,
            trans_parent_map=trans_pmap_for_collapse,
            n_trans_coarse=sampling.current_translations.shape[0],
            n_trans_fine=n_trans_fine_for_collapse,
        )
        score_result.significant_counts = (
            None
            if k_class_result.significant_counts is None
            else np.asarray(k_class_result.significant_counts, dtype=np.int32)
        )
        score_result.profile_summary = k_class_result.profile_summary
        score_result.mstep_full_half_axis = k_class_mstep_full_half_axis_this_score
        score_result.mstep_accumulator_shape = getattr(k_class_result, "mstep_accumulator_shape", None)
        return score_result

    adaptive_os_local = int(sampling.oversampling_order)
    rot_pmap_for_collapse = None
    trans_pmap_for_collapse = None
    n_trans_fine_for_collapse = None
    fine_rotations_for_pose = None
    if variant.firstiter_cc:
        if symmetry != "C1" and not execution.relion_x_half_mstep:
            raise RuntimeError(
                f"{symmetry} reconstruction requires RELION x-half BPref accumulation; "
                "RELAX_K1_RELION_X_HALF_MSTEP=0, CPU-only execution, or disabled "
                "custom CUDA is unsupported for non-C1 symmetry"
            )
        # Images on another grid (applyScaleDifference) or magnified (applyAnisoMag).
        magnification = dataset_projection_magnification(half.particles.dataset)
        firstiter = _score_kclass_firstiter_cc_pass2(
            half,
            sampling,
            priors,
            batching,
            variant,
            execution,
            projection_scale=optics.projection_scale,
            magnification=magnification,
            em_kwargs={
                **em_kwargs,
                **reference_grid_kwargs(optics.reference_current_size, optics.projection_scale, magnification),
            },
        )
        k1_adaptive_result = firstiter.result
        rot_pmap_for_collapse = firstiter.rotation_parent_map
        trans_pmap_for_collapse = firstiter.translation_parent_map
        n_trans_fine_for_collapse = firstiter.n_fine_translations
    else:
        k1_adaptive_result, pass2_grids = _score_adaptive_k1_dense(
            half,
            sampling,
            priors,
            batching,
            variant,
            execution,
            optics,
            em_kwargs,
            symmetry=symmetry,
        )
        rot_pmap_for_collapse = pass2_grids.rotation_parent_map
        trans_pmap_for_collapse = pass2_grids.translation_parent_map
        n_trans_fine_for_collapse = pass2_grids.n_fine_translations
        fine_rotations_for_pose = pass2_grids.fine_rotations
    ha_k = np.asarray(k1_adaptive_result.pose_assignments, dtype=np.int32)
    Ft_y_k = _select_single_class_accumulator(k1_adaptive_result.Ft_y, label="Ft_y")
    Ft_ctf_k = _select_single_class_accumulator(k1_adaptive_result.Ft_ctf, label="Ft_ctf")
    em_stats_k = _collapse_single_class_stats_to_coarse(
        k1_adaptive_result.stats,
        rot_parent_map=rot_pmap_for_collapse,
        n_rot_coarse=sampling.effective_rotations.shape[0],
        dtype=execution.precision.rotation_real_dtype,
    )
    noise_stats_k = k1_adaptive_result.aggregate_noise_stats
    if noise_stats_k is None and k1_adaptive_result.noise_stats is not None:
        noise_stats_k = k1_adaptive_result.noise_stats[0]
    if noise_stats_k is None:
        raise RuntimeError("K=1 adaptive path did not return noise statistics")
    coarse_ha_k = _collapse_fine_pose_assignments_to_coarse(
        ha_k,
        rot_parent_map=rot_pmap_for_collapse,
        trans_parent_map=trans_pmap_for_collapse,
        n_trans_coarse=sampling.current_translations.shape[0],
        n_trans_fine=n_trans_fine_for_collapse,
    )
    if execution.return_best_pose_details:
        if k1_adaptive_result.best_pose_rotations is None or k1_adaptive_result.best_pose_translations is None:
            raise RuntimeError("K=1 adaptive path did not return best pose details")
        pose_dtype = execution.precision.rotation_real_dtype
        best_rots = np.asarray(k1_adaptive_result.best_pose_rotations, dtype=pose_dtype)
        magnification = dataset_projection_magnification(half.particles.dataset)
        if optics.projection_scale != 1.0 or magnification is not None:
            # Poses are reported unscaled and unmagnified; only projection used the transformed matrices.
            best_rots = np.asarray(
                reported_rotations(best_rots, optics.projection_scale, magnification), dtype=pose_dtype
            )
        best_eulers = (
            np.asarray(k1_adaptive_result.best_pose_eulers_deg, dtype=np.float64)
            if k1_adaptive_result.best_pose_eulers_deg is not None
            else utils.R_to_relion(best_rots, degrees=True).astype(pose_dtype)
        )
        best_translations = np.asarray(
            k1_adaptive_result.best_pose_translations, dtype=pose_dtype
        )
    else:
        best_rots = best_eulers = best_translations = None
    if fine_rotations_for_pose is None and rot_pmap_for_collapse is not None:
        fine_rotations_for_pose = sampling.pass2_grids(
            adaptive_oversampling=adaptive_os_local, symmetry=symmetry
        ).fine_rotations
    fine_rotation_eulers_for_pose = None
    if fine_rotations_for_pose is not None and _parity_dump.is_active():
        fine_rotation_eulers_for_pose = utils.R_to_relion(
            np.asarray(fine_rotations_for_pose, dtype=np.float32),
            degrees=True,
        ).astype(np.float32)
    return HalfScoreResult(
        ha=ha_k,
        Ft_y=Ft_y_k,
        Ft_ctf=Ft_ctf_k,
        em_stats=em_stats_k,
        noise_stats=noise_stats_k,
        best_pose_rotations=best_rots,
        best_pose_rotation_eulers=best_eulers,
        best_pose_translations=best_translations,
        coarse_ha=coarse_ha_k,
        pose_rotations=fine_rotations_for_pose,
        pose_rotation_eulers=fine_rotation_eulers_for_pose,
        significant_counts=(
            None
            if k1_adaptive_result.significant_counts is None
            else np.asarray(k1_adaptive_result.significant_counts, dtype=np.int32)
        ),
        profile_summary=k1_adaptive_result.profile_summary,
        mstep_full_half_axis=k1_adaptive_result.mstep_full_half_axis,
        mstep_accumulator_shape=getattr(k1_adaptive_result, "mstep_accumulator_shape", None),
    )


@dataclass(frozen=True, kw_only=True)
class DenseShapeOwners:
    """The dense scoring owners one shape class of a multi-shape half changes; execution is shared."""

    half: HalfScoringData
    sampling: DenseSamplingSpec
    priors: DensePriorSpec
    batching: DenseBatchPolicy
    variant: DenseVariantPolicy
    optics: OpticsSpec


def _class_batching(batching: DenseBatchPolicy, class_index: int) -> DenseBatchPolicy:
    """The batch policy of one shape class: its own planned sizes when the half has them."""
    if batching.class_batch_overrides is None:
        return replace(batching, class_batch_overrides=None)
    sizes = batching.class_batch_overrides[class_index]
    return replace(
        batching,
        class_batch_overrides=None,
        k_class_image_batch_size_override=sizes.k_class_image_batch_size,
        k_class_rotation_block_size_override=sizes.k_class_rotation_block_size,
        significance_image_batch_size_override=sizes.significance_image_batch_size,
        significance_rotation_block_size_override=sizes.significance_rotation_block_size,
    )


def _dense_owners_for_shape(
    half: HalfScoringData,
    sampling: DenseSamplingSpec,
    priors: DensePriorSpec,
    batching: DenseBatchPolicy,
    variant: DenseVariantPolicy,
    optics: OpticsSpec,
    shape_class,
    class_index: int,
) -> DenseShapeOwners:
    """Derive one shape class without changing the shared scoring owners."""

    from relax.refinement import shape_class_scoring

    shape_values = shape_class_scoring.class_kwargs(
        {
            "experiment_dataset": half.particles.dataset,
            "image_corrections_k": half.particles.image_corrections,
            "scale_corrections_k": half.particles.scale_corrections,
            "group_ids_k": half.scale_group_ids,
            "optics_group_ids_k": half.particles.optics_group_ids,
            "image_seed_classes": half.image_seed_classes,
            "rotation_log_prior_k": priors.rotation_log_prior_k,
            "class_rotation_log_prior_k": priors.class_rotation_log_prior_k,
            "translation_log_prior": priors.translation_log_prior,
            "translation_search_base": priors.translation_search_base,
            "trans_prior_center_for_engine": priors.trans_prior_center_for_engine,
            "current_translations": sampling.current_translations,
            "base_translations": sampling.base_translations,
            "translation_step": sampling.translation_step,
            "cs_for_engine": sampling.image_window_size,
            "model_current_size_for_engine": sampling.model_support_size,
            "firstiter_coarse_current_size": variant.coarse_window_size,
            "firstiter_fine_current_size": variant.fine_window_size,
            "coarse_sizing": optics.coarse_sizing,
        },
        shape_class,
        half.particles.dataset.n_units,
    )
    if optics.class_translations is not None:
        translations = optics.class_translations[class_index]
        shape_values["translation_search_base"] = translations.search_base
        shape_values["trans_prior_center_for_engine"] = translations.engine_prior_center
        if translations.log_prior is not None:
            shape_values["translation_log_prior"] = translations.log_prior

    return DenseShapeOwners(
        half=replace(
            half,
            particles=replace(
                half.particles,
                dataset=shape_values["experiment_dataset"],
                optics_group_ids=shape_values["optics_group_ids_k"],
                image_corrections=shape_values["image_corrections_k"],
                scale_corrections=shape_values["scale_corrections_k"],
            ),
            noise_variance=shape_class_scoring.class_noise_table(
                optics.noise_radial_k,
                shape_class,
                int(half.particles.dataset.image_shape[0]),
            ),
            scale_group_ids=shape_values["group_ids_k"],
            image_seed_classes=shape_values["image_seed_classes"],
        ),
        sampling=replace(
            sampling,
            current_translations=shape_values["current_translations"],
            base_translations=shape_values["base_translations"],
            translation_step=shape_values["translation_step"],
            image_window_size=shape_values["cs_for_engine"],
            model_support_size=shape_values["model_current_size_for_engine"],
        ),
        priors=replace(
            priors,
            rotation_log_prior_k=shape_values["rotation_log_prior_k"],
            class_rotation_log_prior_k=shape_values["class_rotation_log_prior_k"],
            translation_log_prior=shape_values["translation_log_prior"],
            translation_search_base=shape_values["translation_search_base"],
            trans_prior_center_for_engine=shape_values["trans_prior_center_for_engine"],
        ),
        batching=_class_batching(batching, class_index),
        variant=replace(
            variant,
            coarse_window_size=shape_values["firstiter_coarse_current_size"],
            fine_window_size=shape_values["firstiter_fine_current_size"],
        ),
        optics=replace(
            optics,
            noise_radial_k=None,
            class_translations=None,
            projection_scale=shape_values["projection_scale"],
            reference_current_size=shape_values["reference_current_size"],
        ),
    )


def require_multi_shape_inputs(half: HalfScoringData, optics: OpticsSpec) -> None:
    """A half with several image shapes needs reference-shell noise spectra and each image's optics group."""
    if optics.noise_radial_k is None:
        raise ValueError("a half with several image shapes needs reference-shell noise spectra")
    if half.particles.optics_group_ids is None:
        raise ValueError("a half with several image shapes needs each image's optics group")


def merge_shape_class_results(results, experiment_half) -> HalfScoreResult:
    """One result for a multi-shape half from its shape classes' results, on the reference box."""
    from relax.refinement import shape_class_scoring

    return shape_class_scoring.merge_class_results(
        results,
        experiment_half.classes,
        experiment_half.n_units,
        int(experiment_half.image_shape[0]),
    )


def _score_half_dense(
    half: HalfScoringData,
    sampling: DenseSamplingSpec,
    priors: DensePriorSpec,
    batching: DenseBatchPolicy,
    variant: DenseVariantPolicy,
    execution: DenseExecutionPolicy,
    optics: OpticsSpec,
) -> HalfScoreResult:
    """Dense E+M scoring for one half; several image shapes run per shape class."""

    from relax.refinement import optics_shapes

    experiment_half = half.particles.dataset
    if not isinstance(experiment_half, optics_shapes.MultiShapeHalf):
        sampling, optics = single_shape_reconstruction_grid(experiment_half, sampling, optics)
        return _score_half_dense_one_shape(half, sampling, priors, batching, variant, execution, optics)
    require_multi_shape_inputs(half, optics)
    if batching.class_batch_overrides is not None and len(batching.class_batch_overrides) != len(experiment_half.classes):
        raise ValueError("class_batch_overrides needs one entry per shape class")

    results = []
    for index, shape_class in enumerate(experiment_half.classes):
        owners = _dense_owners_for_shape(half, sampling, priors, batching, variant, optics, shape_class, index)
        result = _score_half_dense_one_shape(
            owners.half,
            owners.sampling,
            owners.priors,
            owners.batching,
            owners.variant,
            execution,
            owners.optics,
        )
        if not variant.k_class_enabled:
            # K1 shape merging retains common statistics, not class-prior summaries.
            result.classes = None
        results.append(result)
    return merge_shape_class_results(results, experiment_half)


def _score_half_dense_in_bpref_scope(
    half: HalfScoringData,
    sampling: DenseSamplingSpec,
    priors: DensePriorSpec,
    batching: DenseBatchPolicy,
    variant: DenseVariantPolicy,
    execution: DenseExecutionPolicy,
    optics: OpticsSpec,
) -> HalfScoreResult:
    """Keep all authoritative dense-half work outside diagnostic CUDA scope."""

    with em_cuda_kernels.bpref_device_signature_scope(False):
        return _score_half_dense(half, sampling, priors, batching, variant, execution, optics)


def single_shape_reconstruction_grid(dataset, sampling, optics: OpticsSpec):
    """``(sampling, optics)`` of a one-shape half (dense or local sampling) the engine clips at the reference sphere.

    With a scale difference (``optics.projection_scale`` is not 1, even by the float32 rounding of the
    reference header's pixel) RELION's remap gives the images a current size above the model's, and
    ``sampling.model_support_size`` arrives as the model's. Without magnification the engine builds
    the exact-radius window of that size, which is RELION's support, and nothing changes here. With
    anisotropic magnification (:func:`relax.relion.optics_aberrations.dataset_magnification_is_anisotropic`,
    the predicate of the engines' ``reference_sphere_clip``) the engine keeps a rounded window and the
    kernel clips: the window is then the model support on the image grid and the reference keeps the
    model's size, as for a shape class (:func:`relax.refinement.shape_class_scoring.class_kwargs`, the same
    ``group_current_size``). A rounded window of the model's size on the larger image crop is one the
    Wavg rectangle refuses (relax#69).
    """

    from relax.relion import optics_scale

    reference_size = sampling.model_support_size
    if reference_size is None or optics.projection_scale == 1.0 or not dataset_magnification_is_anisotropic(dataset):
        return sampling, optics
    window = optics_scale.group_current_size(reference_size, int(dataset.image_shape[0]), optics.projection_scale)
    return replace(sampling, model_support_size=window), replace(optics, reference_current_size=int(reference_size))


@dataclass(frozen=True, kw_only=True)
class FirstIterCCPass2:
    """The engine's result of a first-iteration CC dispatch and the trial-grid maps it ran on."""

    result: KClassEMResult
    # Each fine rotation's and fine translation's coarse parent.
    rotation_parent_map: np.ndarray
    translation_parent_map: np.ndarray
    n_fine_translations: int


def _score_kclass_firstiter_cc_pass2(
    half: HalfScoringData,
    sampling: DenseSamplingSpec,
    priors: DensePriorSpec,
    batching: DenseBatchPolicy,
    variant: DenseVariantPolicy,
    execution: DenseExecutionPolicy,
    *,
    projection_scale: float,
    magnification,
    em_kwargs: dict,
) -> FirstIterCCPass2:
    """RELION iter-1 ``--firstiter_cc`` adaptive two-pass dispatch.

    Build coarse/fine grids and invoke the K-class engine with normalized-CC
    scoring through the global coarse winner subset. The winner-take-all
    policy applies to M-step support as well as reported Pmax.

    The K-class and K=1 adaptive scoring branches share this dispatcher, with the images' projection scale
    and magnification and their engine keywords. A K-class half scores its class stack on its coarse rotation
    ids; a K=1 half scores its one reference as a one-class
    stack on every coarse rotation. ``em_kwargs`` are the dense route's engine keywords; this dispatch reads
    only their batch sizes (``image_batch_size``, ``rotation_block_size``) and gives the engine a copy with
    clamped batch sizes; the caller's dictionary is not changed. The coarse/fine sizes are
    ``variant.coarse_window_size`` and ``fine_window_size`` (None: the engine's ``current_size``). The oversampling order used is
    ``sampling.oversampling_order``.
    """

    mean = half.reference if variant.k_class_enabled else jnp.asarray(half.reference)[None, :]
    # The coarse rotations a K-class half scores (None: all of them).
    coarse_ids = sampling.coarse_rotation_ids if variant.k_class_enabled else None
    if not variant.k_class_enabled:
        log_label = "K=1 "
    else:
        log_label = "" if variant.fine_window_size is not None else "(non-adaptive site) "
    point_group = canonicalize_rotational_symmetry(sampling.symmetry)
    adaptive_os_local = int(sampling.oversampling_order)
    (
        coarse_rot,
        coarse_trans,
        fine_rot,
        fine_trans,
        rot_pmap,
        trans_pmap,
        fine_mstep_rot,
    ) = build_adaptive_pass2_grids(
        sampling.effective_rotations,
        sampling.current_translations,
        sampling.base_translations,
        int(sampling.current_healpix_order),
        adaptive_os_local,
        float(sampling.translation_step),
        sampling.random_perturbation,
        return_mstep_rotations=True,
        coarse_rotation_ids=coarse_ids,
        symmetry=point_group,
    )
    coarse_rot, fine_rot, fine_mstep_rot = project_pass2_rotations(
        coarse_rot,
        fine_rot,
        fine_mstep_rot,
        scale=float(projection_scale),
        magnification=magnification,
        coarse_healpix_order=int(sampling.current_healpix_order),
        adaptive_oversampling=adaptive_os_local,
        random_perturbation=sampling.random_perturbation,
        coarse_rotation_ids=coarse_ids,
        symmetry=point_group,
        coarse_device_source=sampling.effective_device_source,
        grid_device_source=sampling.effective_device_source,
    )
    coarse_translation_phase_source = apply_relion_translation_perturbation(
        np.asarray(sampling.base_translations, dtype=np.float64),
        float(sampling.random_perturbation),
        float(sampling.translation_step),
    )
    n_classes = int(np.shape(mean)[0]) if np.ndim(mean) >= 2 else 1
    firstiter_significance_image_batch_size = None
    firstiter_significance_rotation_block_size = None
    if point_group != "C1" and sampling.coarse_engine != "gemm_dense":
        if not execution.relion_x_half_mstep:
            raise RuntimeError(f"{point_group} requires sparse RELION x-half BPref reconstruction")
    if batching.safe_batch_sizes is not None:
        batch_plan = _plan_kclass_adaptive_grid_batch_sizes(
            coarse_rotations=coarse_rot,
            coarse_translations=coarse_trans,
            fine_rotations=fine_rot,
            fine_translations=fine_trans,
            n_classes=n_classes,
            image_shape=half.particles.dataset.image_shape,
            coarse_current_size=(
                variant.coarse_window_size
                if variant.coarse_window_size is not None
                else sampling.image_window_size
            ),
            fine_current_size=(
                variant.fine_window_size
                if variant.fine_window_size is not None
                else sampling.image_window_size
            ),
            safe_batch_sizes=batching.safe_batch_sizes,
            significance_safe_batch_sizes=batching.significance_safe_batch_sizes,
        )
        requested_firstiter_image_batch_size = int(em_kwargs["image_batch_size"])
        firstiter_image_batch_size = min(
            requested_firstiter_image_batch_size,
            safe_firstiter_cc_image_batch_size(
                fine_trans.shape[0],
                half.particles.dataset.image_shape,
            ),
        )
        firstiter_rotation_block_size = min(
            int(em_kwargs["rotation_block_size"]),
            safe_dense_k_class_rotation_block_size(
                fine_trans.shape[0],
                firstiter_image_batch_size,
            ),
        )
        firstiter_significance_image_batch_size = batch_plan.significance_image_batch_size
        firstiter_significance_rotation_block_size = batch_plan.significance_rotation_block_size
        logger.info(
            "STRICT-PARITY: iter-1 K-class adaptive batch sizing "
            "coarse image_batch_size=%d rotation_block_size=%d; "
            "fine image_batch_size=%d rotation_block_size=%d (sparse pass2)",
            firstiter_significance_image_batch_size,
            firstiter_significance_rotation_block_size,
            firstiter_image_batch_size,
            firstiter_rotation_block_size,
        )
    else:
        requested_firstiter_image_batch_size = int(em_kwargs["image_batch_size"])
        firstiter_image_batch_size = min(
            requested_firstiter_image_batch_size,
            safe_firstiter_cc_image_batch_size(
                fine_trans.shape[0],
                half.particles.dataset.image_shape,
            ),
        )
        firstiter_rotation_block_size = int(em_kwargs["rotation_block_size"])
        if firstiter_image_batch_size != requested_firstiter_image_batch_size:
            logger.info(
                "STRICT-PARITY: clamping iter-1 winner-take-all image_batch_size from %d to %d",
                requested_firstiter_image_batch_size,
                firstiter_image_batch_size,
            )
    firstiter_em_kwargs = dict(em_kwargs)
    firstiter_em_kwargs["image_batch_size"] = firstiter_image_batch_size
    firstiter_em_kwargs["rotation_block_size"] = firstiter_rotation_block_size
    firstiter_em_kwargs["sparse_pass2"] = True
    logger.info(
        "STRICT-PARITY %srouting iter-1 K-class through sparse run_dense_k_class_em_adaptive "
        "(oversampling=%d, relion_x_half_mstep=%s, best_coarse_subset=True)",
        log_label,
        adaptive_os_local,
        execution.relion_x_half_mstep,
    )
    k_class_result = run_dense_k_class_em_adaptive(
        half.particles.dataset,
        mean,
        half.mean_variance,
        half.noise_variance,
        coarse_rot,
        coarse_trans,
        fine_rot,
        fine_trans,
        rot_pmap,
        trans_pmap,
        execution.disc_type,
        class_log_priors=priors.class_log_priors,
        accumulate_noise=True,
        return_best_pose_details=True,
        firstiter_cc_pass2_only_best_coarse=True,
        image_seed_classes=half.image_seed_classes,
        relion_fine_mstep_prune=True,
        significance_image_batch_size=firstiter_significance_image_batch_size,
        significance_rotation_block_size=firstiter_significance_rotation_block_size,
        coarse_healpix_order=int(sampling.current_healpix_order),
        coarse_rotation_ids=coarse_ids,
        oversampling_order=int(adaptive_os_local),
        fine_mstep_rotations_override=(
            fine_mstep_rot
        ),
        bpref_device_signature_active=execution.bpref_device_signature_active,
        debug_iteration=execution.debug_iteration,
        coarse_translation_phase_source=(
            coarse_translation_phase_source if n_classes == 1 else None
        ),
        coarse_current_size=variant.coarse_window_size,
        fine_current_size=variant.fine_window_size,
        **firstiter_em_kwargs,
    )
    return FirstIterCCPass2(
        result=k_class_result,
        rotation_parent_map=rot_pmap,
        translation_parent_map=trans_pmap,
        n_fine_translations=int(fine_trans.shape[0]),
    )
