"""Dispatch dense and local scoring for one refinement half-set.

The iteration controller owns scheduling, state transitions, result-container
lifetime and device offloading. These adapters choose existing engine routes,
prepare per-half arguments and populate caller-owned result slots. Diagnostic
BPref scopes surround the same calls as before; no array copies are introduced
by the ownership boundary.
"""

import logging
import os
from dataclasses import dataclass, field, replace
from typing import Literal

import jax.numpy as jnp
import numpy as np
from recovar import utils

from relax.classification.k_class import run_dense_k_class_em_adaptive
from relax.cuda import kernels as em_cuda_kernels
from relax.dense import scoring_policy
from relax.dense.score_outputs import (
    HalfScoreResult,
    _collapse_fine_pose_assignments_to_coarse,
    _collapse_single_class_stats_to_coarse,
    _select_single_class_accumulator,
    class_em_to_half_result,
)
from relax.dense.scoring_policy import (
    _K1_RELION_X_HALF_MSTEP_ENV,
    _LOCAL_ADAPTIVE_PASS2_DENOMINATOR_SUPPORT_ENV,
    _LOCAL_ADAPTIVE_PASS2_FULL_PARENT_ENV,
    _LOCAL_ADAPTIVE_PASS2_ROTATION_ONLY_ENV,
    RELION_ACC_DOUBLE_FLOORF_QUIRK,
    RELION_ADAPTIVE_FRACTION,
    RELION_FOURIER_WINDOW_SQUARE,
    _dense_global_scoring_dtype,
    _k1_relion_x_half_mstep_enabled,
    _k_class_relion_x_half_mstep_enabled,
    local_precision,
)
from relax.diagnostics import parity_dump as _parity_dump
from relax.diagnostics.local_debug import log_local_adaptive_support, log_local_denominator_support
from relax.helpers.batch_planning import _plan_kclass_adaptive_grid_batch_sizes
from relax.helpers.half_volume_mstep import relion_backprojector_volume_shape
from relax.helpers.oversampling import AdaptivePass2Grids, prepare_adaptive_pass2_grids
from relax.local.local_layout import (
    build_local_adaptive_pass2_hypothesis_layout,
    build_local_hypothesis_layout,
    expand_local_layout_classes,
    restrict_local_layout_classes,
)
from relax.refinement.firstiter_cc import (
    FirstIterCCBatching,
    FirstIterCCData,
    FirstIterCCExecution,
    FirstIterCCGridSpec,
    FirstIterCCPolicy,
    _score_kclass_firstiter_cc_pass2,
)
from relax.refinement.half_inputs import HalfSet
from relax.refinement.local_sampling import LocalSampling
from relax.refinement.local_search_iteration import (
    LocalSearchData,
    LocalSearchGridSpec,
    LocalSearchKernelPolicy,
    LocalSearchSupportPolicy,
    _run_local_search_iteration,
)
from relax.refinement.optics_shapes import (
    OpticsSpec,
    engine_projection_inputs,
    reconstruction_image_radius,
    reference_grid_kwargs,
)
from relax.refinement.ports import RunObserver
from relax.refinement.projector_preparation import PreparedProjector
from relax.relion.geometry import (
    PROJECTION_PADDING_FACTOR,
    RECONSTRUCTION_PADDING_FACTOR,
)
from relax.relion.optics_aberrations import dataset_projection_magnification, projection_rotations, reported_rotations
from relax.sampling import (
    build_local_search_grid_metadata,
    relion_angular_sampling_deg,
)

logger = logging.getLogger("relax.dense.half_scoring")


def _expand_significant_samples_to_full_parent_translations(
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






def _adaptive_engine_common_kwargs(
    pass2_grids: AdaptivePass2Grids,
    priors: "DensePriorSpec",
    batching: "DenseBatchPolicy",
    sampling: "DenseSamplingSpec",
    execution: "DenseExecutionPolicy",
) -> dict:
    """Owner-derived keywords shared by both adaptive dense routes.

    Both routes accumulate noise, keep RELION's adaptive significance fraction
    and prune the fine M-step rotations of the sparse pass 2. Route-local
    batch sizes, Fourier sizes and oversampling order stay beside each engine
    call rather than being hidden in a one-call plan.
    """

    return dict(
        class_log_priors=priors.class_log_priors,
        accumulate_noise=True,
        adaptive_fraction=RELION_ADAPTIVE_FRACTION,
        max_significants=(-1 if batching.max_significants is None else int(batching.max_significants)),
        relion_fine_mstep_prune=True,
        coarse_healpix_order=int(sampling.current_healpix_order),
        fine_mstep_rotations_override=pass2_grids.fine_mstep_rotations,
        return_best_pose_details=execution.return_best_pose_details,
        bpref_device_signature_active=execution.bpref_device_signature_active,
        debug_iteration=execution.debug_iteration,
    )


@dataclass(frozen=True, kw_only=True)
class HalfScoringData:
    """Persistent particle half and the model operands for one expectation."""

    particles: HalfSet
    reference: object
    noise_variance: object
    noise_radial: object | None = None
    mean_variance: object | None = None
    projector: PreparedProjector | None = None
    scale_group_ids: object | None = None
    scale_group_count: int | None = None
    scale_correction_data_vs_prior: object | None = None
    image_seed_classes: object | None = None


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
    cs_for_engine: int | None
    coarse_engine: str = "auto"
    model_current_size_for_engine: int | None = None
    # --strict_highres_exp: the weighted sums' image size, above cs_for_engine (None: cs_for_engine).
    wsum_current_size_for_engine: int | None = None
    coarse_angular_step_deg: float | None = None
    coarse_rotation_ids: object | None = None
    coarse_scoring_rotations: object | None = None
    symmetry: str = "C1"

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
    significance_safe_batch_sizes: object | None = None
    k_class_image_batch_size_override: int | None = None
    k_class_rotation_block_size_override: int | None = None
    significance_image_batch_size_override: int | None = None
    significance_rotation_block_size_override: int | None = None
    class_batch_overrides: tuple[dict, ...] | None = None


@dataclass(frozen=True, kw_only=True)
class DenseVariantPolicy:
    """Explicit dense, adaptive and first-iteration route selections."""

    firstiter_score_mode_this_iter: str
    firstiter_winner_take_all_this_iter: bool
    k_class_enabled: bool
    relion_firstiter_cc_this_iter: bool
    firstiter_coarse_current_size: int | None = None
    firstiter_fine_current_size: int | None = None
    firstiter_log_label: str = "(non-adaptive site) "
    # RELION --skip_align: classify at each particle's stored pose (relax.classification.given_poses).
    skip_align: bool = False


@dataclass(frozen=True, kw_only=True)
class DenseExecutionPolicy:
    """Interpolation, ablation and dense diagnostic settings."""

    disc_type: str
    disable_adjoint_y: bool
    disable_adjoint_ctf: bool
    return_best_pose_details: bool = True
    bpref_device_signature_active: bool = False
    debug_iteration: int | None = None
    diagnostic_float64_pass2: bool = False
    preserve_bpref_particle_order: bool = False
    source_faithful_spectrum_norm: bool = False
    relion_translation_angle_scale: float = 1.0
    # The K=1 --firstiter_cc coarse-tree top-2 rescore margin (RelionParityOptions); None is off.
    firstiter_cc_tree_rescore_max_margin: float | None = None
    # RelionConsistencyOptions: the support of the first-iteration normalized CC, and how the
    # per-image sums count the Hermitian pairs of the full-size Nyquist column.
    firstiter_cc_support: str = "relion"
    nyquist_column_counting: str = "relion"


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
):
    """Run the ordinary adaptive K-class engine and return its trial grids."""

    adaptive_os = int(sampling.oversampling_order)
    coarse_current_size = variant.firstiter_coarse_current_size
    fine_current_size = variant.firstiter_fine_current_size
    if adaptive_os <= 0:
        coarse_current_size = sampling.cs_for_engine
        fine_current_size = sampling.cs_for_engine
        logger.info(
            "RELION K-class scale groups at oversampling 0: routing the single pass through the "
            "adaptive engine (current_size=%s)",
            sampling.cs_for_engine,
        )
    pass2_grids = sampling.pass2_grids(adaptive_oversampling=adaptive_os, symmetry=symmetry)
    adaptive_em_kwargs = dict(em_kwargs)
    n_classes = (
        int(np.asarray(half.reference).shape[0])
        if np.asarray(half.reference).ndim >= 2
        else 1
    )
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
    common_kwargs = _adaptive_engine_common_kwargs(pass2_grids, priors, batching, sampling, execution)
    # Images on another grid (applyScaleDifference) or magnified (applyAnisoMag): the
    # projection and backprojection matrices carry it.
    projected, grid_kwargs = engine_projection_inputs(
        half.particles.dataset,
        scale=optics.projection_scale,
        reference_current_size=optics.reference_current_size,
        rotations={
            "coarse": pass2_grids.coarse_rotations,
            "fine": pass2_grids.fine_rotations,
            "mstep": common_kwargs["fine_mstep_rotations_override"],
        },
    )
    adaptive_em_kwargs.update(grid_kwargs)
    common_kwargs["fine_mstep_rotations_override"] = projected["mstep"]
    result = run_dense_k_class_em_adaptive(
        half.particles.dataset,
        half.reference,
        half.mean_variance,
        half.noise_variance,
        projected["coarse"],
        pass2_grids.coarse_translations,
        projected["fine"],
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
        **common_kwargs,
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

    from relax.classification.given_poses import given_pose_grids

    particles = half.particles
    if particles.rotation_eulers is None or particles.translations is None:
        raise ValueError("--skip_align classifies at the input poses: the particle STAR needs angles and offsets")
    eulers = np.asarray(particles.rotation_eulers, dtype=np.float64)
    stored = np.asarray(particles.translations, dtype=np.float64)
    base = np.zeros_like(stored) if priors.translation_search_base is None else np.asarray(priors.translation_search_base)
    pose_dtype = _dense_global_scoring_dtype()
    grids = given_pose_grids(
        np.asarray(utils.R_from_relion(eulers, degrees=True), dtype=pose_dtype),
        # RELION's offsets are RFLOAT: the remainder stays double until pass 2 forms its phase.
        stored - np.asarray(base, dtype=np.float64),
        symmetry=symmetry,
    )
    n_classes = int(np.asarray(half.reference).shape[0])
    plan = _plan_kclass_adaptive_grid_batch_sizes(
        coarse_rotations=grids.rotations[:1],
        coarse_translations=grids.translations[:1],
        fine_rotations=grids.rotations[:1],
        fine_translations=grids.translations[:1],
        n_classes=n_classes,
        image_shape=particles.dataset.image_shape,
        coarse_current_size=sampling.cs_for_engine,
        fine_current_size=sampling.cs_for_engine,
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
    projected, grid_kwargs = engine_projection_inputs(
        particles.dataset,
        scale=optics.projection_scale,
        reference_current_size=optics.reference_current_size,
        rotations={"fine": grids.rotations},
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
        projected["fine"],
        grids.translations,
        projected["fine"],
        grids.translations,
        grids.rotation_parent_map,
        grids.translation_parent_map,
        execution.disc_type,
        class_log_priors=priors.class_log_priors,
        accumulate_noise=True,
        coarse_current_size=sampling.cs_for_engine,
        fine_current_size=sampling.cs_for_engine,
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
):
    """Run the ordinary adaptive K=1 engine and return its trial grids."""

    adaptive_os = int(sampling.oversampling_order)
    coarse_current_size = variant.firstiter_coarse_current_size
    fine_current_size = variant.firstiter_fine_current_size
    if adaptive_os <= 0:
        # The sparse engine supplies group statistics and per-particle BPref
        # launches even for a single pass on the current grid.
        coarse_current_size = sampling.cs_for_engine
        fine_current_size = sampling.cs_for_engine
        logger.info(
            "RELION K=1 group statistics or BPref order at oversampling 0: single pass through the "
            "adaptive engine (current_size=%s)",
            sampling.cs_for_engine,
        )
    relion_x_half_mstep = _k1_relion_x_half_mstep_enabled()
    if symmetry != "C1" and not relion_x_half_mstep:
        raise RuntimeError(
            f"{symmetry} reconstruction requires RELION x-half BPref accumulation; "
            "RELAX_K1_RELION_X_HALF_MSTEP=0, CPU-only execution, or disabled "
            "custom CUDA is unsupported for non-C1 symmetry"
        )
    means_single = jnp.asarray(half.reference)[None, :]
    pass2_grids = sampling.pass2_grids(adaptive_oversampling=adaptive_os, symmetry=symmetry)
    adaptive_em_kwargs = dict(base_em_kwargs)
    adaptive_em_kwargs["sparse_pass2"] = True
    if half.scale_group_ids is not None:
        adaptive_em_kwargs["group_ids"] = half.scale_group_ids
    if relion_x_half_mstep:
        adaptive_em_kwargs["mstep_relion_x_half"] = True
    logger.info(
        "RELION adaptive K=1 routing through run_dense_k_class_em_adaptive "
        "(oversampling=%d, pass2_backend=sparse, "
        "fine_mstep_prune=True, relion_x_half_mstep=%s, supplied_ppref=%s, "
        "engine_ppref=%s)",
        adaptive_os,
        bool(relion_x_half_mstep),
        half.projector is not None,
        adaptive_em_kwargs.get("relion_projector_half") is not None,
    )
    common_kwargs = _adaptive_engine_common_kwargs(pass2_grids, priors, batching, sampling, execution)
    projected, grid_kwargs = engine_projection_inputs(
        half.particles.dataset,
        scale=optics.projection_scale,
        reference_current_size=optics.reference_current_size,
        rotations={
            "coarse": (
                sampling.coarse_scoring_rotations
                if sampling.coarse_scoring_rotations is not None
                and adaptive_os == 0
                and relion_x_half_mstep
                and variant.firstiter_score_mode_this_iter == "gaussian"
                and not execution.diagnostic_float64_pass2
                else pass2_grids.coarse_rotations
            ),
            "fine": pass2_grids.fine_rotations,
            "mstep": common_kwargs["fine_mstep_rotations_override"],
        },
    )
    adaptive_em_kwargs.update(grid_kwargs)
    common_kwargs["fine_mstep_rotations_override"] = projected["mstep"]
    k1_adaptive_result = run_dense_k_class_em_adaptive(
        half.particles.dataset,
        means_single,
        half.mean_variance,
        half.noise_variance,
        projected["coarse"],
        pass2_grids.coarse_translations,
        projected["fine"],
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
        **common_kwargs,
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
    """Dense (non-local-search) E+M scoring for one half-set.

    The signature exposes per-half data, sampling, priors, batching, route
    selection, execution controls and optics adaptations as cohesive owners.
    Stable fields are read through those owners; only values changed by route
    planning become local variables.

    ``half.particles.optics_group_ids`` gives each image's row of a
    per-optics-group ``noise_variance_k`` table
    (:mod:`relax.helpers.optics_noise`); only the K=1 adaptive route carries it,
    every other engine refuses it.

    ``optics.projection_scale`` and ``reference_current_size`` describe
    images on another grid than the reference (one shape class of
    :mod:`relax.refinement.optics_shapes`): the projection and backprojection
    matrices are divided by the scale and the backprojector keeps the reference
    model size; reported poses stay unscaled.

    Used by both the single-pass (``else``) and adaptive-2-pass
    (``elif use_adaptive``) branches of the half-set loop. The two modes
    differ in four places, all controlled by explicit policy fields:

    1. ``k_class_image_batch_size_override`` /
       ``k_class_rotation_block_size_override`` — adaptive overrides
       em_kwargs ibs/rbs to K-class values before firstiter_cc check.
    2. ``significance_*_override`` — adaptive pass 1 may use a smaller
       Fourier window than pass 2, so it needs its own memory-sized batches.
    3. ``firstiter_coarse_current_size`` / ``firstiter_fine_current_size``
       — adaptive passes ``coarse_cs`` / ``cs_for_engine`` through to the
       adaptive 2-pass engine; single-pass omits them.
    4. ``firstiter_log_label`` — single-pass uses
       ``"(non-adaptive site) "`` for the routing log message.

    Returns common statistics, class summaries and explicit poses together.
    The controller records them at its existing half-update boundary.
    ``batching.safe_batch_sizes`` plans against memory available at invocation.
    """

    from relax.symmetry import canonicalize_rotational_symmetry

    symmetry = canonicalize_rotational_symmetry(sampling.symmetry)
    if symmetry != "C1" and int(sampling.oversampling_order) <= 0 and sampling.coarse_engine != "gemm_dense":
        raise NotImplementedError(
            f"{symmetry} reconstruction at adaptive oversampling 0 is unsupported; "
            "use adaptive oversampling >= 1 or the exact-local RELION x-half M-step"
        )
    safe_ibs, safe_rbs = batching.safe_batch_sizes(
        sampling.effective_rotations.shape[0],
        sampling.current_translations.shape[0],
        current_size_for_batch=sampling.cs_for_engine,
    )
    em_kwargs = {
        "score_with_masked_images": True,
        "half_spectrum_scoring": True,
        "projection_padding_factor": PROJECTION_PADDING_FACTOR,
        "reconstruction_padding_factor": RECONSTRUCTION_PADDING_FACTOR,
        "use_float64_scoring": scoring_policy.DENSE_PRECISION.use_float64_scoring,
        "use_float64_projections": scoring_policy.DENSE_PRECISION.use_float64_projections,
        "relion_exact_fine_gaussian": scoring_policy.RELION_EXACT_FINE_GAUSSIAN,
        "do_gridding_correction": True,
        "square_window": RELION_FOURIER_WINDOW_SQUARE,
        "image_batch_size": safe_ibs,
        "rotation_block_size": safe_rbs,
        "current_size": sampling.cs_for_engine,
        "rotation_log_prior": priors.rotation_log_prior_k,
        "translation_log_prior": priors.translation_log_prior,
        "image_corrections": half.particles.image_corrections,
        "scale_corrections": half.particles.scale_corrections,
        "group_ids": half.scale_group_ids,
        "scale_correction_group_count": half.scale_group_count,
        "scale_correction_data_vs_prior": half.scale_correction_data_vs_prior,
        "image_pre_shifts": priors.translation_search_base,
        "translation_prior_centers": priors.trans_prior_center_for_engine,
        "relion_firstiter_score_mode": variant.firstiter_score_mode_this_iter,
        "relion_firstiter_winner_take_all": variant.firstiter_winner_take_all_this_iter,
    }
    if sampling.coarse_engine != "auto":
        em_kwargs["coarse_engine"] = sampling.coarse_engine
    if symmetry != "C1":
        em_kwargs["symmetry_label"] = symmetry
    if half.particles.optics_group_ids is not None:
        em_kwargs["optics_group_ids"] = half.particles.optics_group_ids
    if sampling.model_current_size_for_engine is not None:
        em_kwargs["reconstruction_current_size"] = sampling.model_current_size_for_engine
    if sampling.wsum_current_size_for_engine is not None:
        em_kwargs["wsum_current_size"] = sampling.wsum_current_size_for_engine
    if half.image_seed_classes is not None and not variant.k_class_enabled:
        raise NotImplementedError("a seed iteration runs on the adaptive or first-iteration CC K-class route")
    if execution.preserve_bpref_particle_order:
        em_kwargs["preserve_bpref_particle_order"] = True
    if execution.source_faithful_spectrum_norm:
        em_kwargs["source_faithful_spectrum_norm"] = True
    if execution.firstiter_cc_tree_rescore_max_margin is not None:
        em_kwargs["firstiter_cc_tree_rescore_max_margin"] = float(execution.firstiter_cc_tree_rescore_max_margin)
    if float(execution.relion_translation_angle_scale) != 1.0:
        em_kwargs["relion_translation_angle_scale"] = float(execution.relion_translation_angle_scale)
    if execution.firstiter_cc_support != "relion":
        em_kwargs["firstiter_cc_support"] = execution.firstiter_cc_support
    if execution.nyquist_column_counting != "relion":
        em_kwargs["nyquist_column_counting"] = execution.nyquist_column_counting
    if execution.diagnostic_float64_pass2:
        logger.info(
            "Diagnostic genuine-float64 adaptive pass 2 at iteration %d; pass 1 and prior boundaries remain f32",
            int(execution.debug_iteration),
        )
    if batching.k_class_image_batch_size_override is not None:
        em_kwargs["image_batch_size"] = batching.k_class_image_batch_size_override
    if batching.k_class_rotation_block_size_override is not None:
        em_kwargs["rotation_block_size"] = batching.k_class_rotation_block_size_override
    if priors.class_rotation_log_prior_k is not None:
        em_kwargs["rotation_log_prior"] = None
        em_kwargs["class_rotation_log_prior"] = priors.class_rotation_log_prior_k
    if half.projector is not None:
        em_kwargs["relion_projector_half"] = half.projector.data
        em_kwargs["relion_projector_r_max"] = half.projector.r_max
    logger.info(
        "Dense half-set projector handoff: supplied_ppref=%s state_oversampling=%d",
        half.projector is not None,
        int(sampling.oversampling_order),
    )
    if variant.relion_firstiter_cc_this_iter:
        # Shared first-iteration inputs; means, layouts and pose IDs remain route-specific.
        firstiter_data = FirstIterCCData(
                logger=logger,
                experiment_dataset=half.particles.dataset,
                mean=half.reference,
                mean_variance=half.mean_variance,
                noise_variance=half.noise_variance,
                image_shape=half.particles.dataset.image_shape,
                image_seed_classes=half.image_seed_classes,
        )
        firstiter_grid = FirstIterCCGridSpec(
                effective_rotations=sampling.effective_rotations,
                current_translations=sampling.current_translations,
                base_translations=sampling.base_translations,
                current_healpix_order=sampling.current_healpix_order,
                oversampling_order=sampling.oversampling_order,
                translation_step=sampling.translation_step,
                random_perturbation=sampling.random_perturbation,
                symmetry=symmetry,
        )
        firstiter_policy = FirstIterCCPolicy(
                disc_type=execution.disc_type,
                class_log_priors=priors.class_log_priors,
        )
        firstiter_batching = FirstIterCCBatching(
                image_batch_size=batching.image_batch_size,
                em_kwargs=em_kwargs,
                safe_batch_sizes=batching.safe_batch_sizes,
                significance_safe_batch_sizes=batching.significance_safe_batch_sizes,
                coarse_current_size=variant.firstiter_coarse_current_size,
                fine_current_size=variant.firstiter_fine_current_size,
        )
        firstiter_execution = FirstIterCCExecution(
                log_label=variant.firstiter_log_label,
                bpref_device_signature_active=execution.bpref_device_signature_active,
                debug_iteration=execution.debug_iteration,
        )

    if variant.k_class_enabled:
        if execution.disable_adjoint_y or execution.disable_adjoint_ctf:
            raise NotImplementedError("K-class refine does not support adjoint ablation flags")
        magnification = dataset_projection_magnification(half.particles.dataset)
        # K-class uses RELION's x-half BackProjector accumulator layout by
        # default, matching the K=1 parity path. The explicit selector can
        # still choose the dense full-volume path.
        k_class_relion_x_half_mstep = _k_class_relion_x_half_mstep_enabled()
        if symmetry != "C1" and not k_class_relion_x_half_mstep:
            raise RuntimeError(f"{symmetry} requires sparse RELION x-half BPref reconstruction")
        em_kwargs["mstep_relion_x_half"] = bool(k_class_relion_x_half_mstep)
        em_kwargs["relion_half_volume_mstep"] = False
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
                firstiter_cc=variant.relion_firstiter_cc_this_iter,
            )
            rot_pmap_for_collapse, trans_pmap_for_collapse, n_trans_fine_for_collapse = _given_pose_collapse(given)
            k_class_mstep_full_half_axis_this_score = k_class_result.mstep_full_half_axis
        elif variant.relion_firstiter_cc_this_iter:
            adaptive_os_local = int(sampling.oversampling_order)
            (
                k_class_result,
                rot_pmap_for_collapse,
                trans_pmap_for_collapse,
                n_trans_fine_for_collapse,
            ) = _score_kclass_firstiter_cc_pass2(
                firstiter_data,
                replace(
                    firstiter_grid,
                    coarse_rotation_ids=sampling.coarse_rotation_ids,
                    projection_rotations=(
                        None
                        if magnification is None and optics.projection_scale == 1.0
                        else lambda rotations: projection_rotations(rotations, optics.projection_scale, magnification)
                    ),
                ),
                firstiter_policy,
                replace(
                    firstiter_batching,
                    em_kwargs={
                        **em_kwargs,
                        **reference_grid_kwargs(optics.reference_current_size, optics.projection_scale),
                    },
                ),
                firstiter_execution,
            )
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
            pose_dtype=_dense_global_scoring_dtype(),
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

    if execution.disable_adjoint_y or execution.disable_adjoint_ctf:
        raise NotImplementedError("K=1 adaptive oversampling does not support adjoint ablation flags")
    adaptive_os_local = int(sampling.oversampling_order)
    rot_pmap_for_collapse = None
    trans_pmap_for_collapse = None
    n_trans_fine_for_collapse = None
    fine_rotations_for_pose = None
    if variant.relion_firstiter_cc_this_iter:
        k1_relion_x_half_mstep = _k1_relion_x_half_mstep_enabled()
        if symmetry != "C1" and not k1_relion_x_half_mstep:
            raise RuntimeError(
                f"{symmetry} reconstruction requires RELION x-half BPref accumulation; "
                "RELAX_K1_RELION_X_HALF_MSTEP=0, CPU-only execution, or disabled "
                "custom CUDA is unsupported for non-C1 symmetry"
            )
        (
            k1_adaptive_result,
            rot_pmap_for_collapse,
            trans_pmap_for_collapse,
            n_trans_fine_for_collapse,
        ) = _score_kclass_firstiter_cc_pass2(
            replace(firstiter_data, mean=jnp.asarray(half.reference)[None, :]),
            replace(
                firstiter_grid,
                # Images on another grid (applyScaleDifference) or magnified (applyAnisoMag).
                projection_rotations=lambda rotations: projection_rotations(
                    rotations,
                    optics.projection_scale,
                    dataset_projection_magnification(half.particles.dataset),
                ),
            ),
            firstiter_policy,
            replace(
                firstiter_batching,
                em_kwargs={
                    **em_kwargs,
                    **({"mstep_relion_x_half": True} if k1_relion_x_half_mstep else {}),
                    **reference_grid_kwargs(optics.reference_current_size, optics.projection_scale),
                },
            ),
            replace(firstiter_execution, log_label="K=1 "),
        )
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
        dtype=_dense_global_scoring_dtype(),
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
        pose_dtype = _dense_global_scoring_dtype()
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



def _dense_owners_for_shape(
    half: HalfScoringData,
    sampling: DenseSamplingSpec,
    priors: DensePriorSpec,
    batching: DenseBatchPolicy,
    variant: DenseVariantPolicy,
    execution: DenseExecutionPolicy,
    optics: OpticsSpec,
    shape_class,
    class_index: int,
) -> tuple:
    """Derive one shape class without changing the shared scoring owners."""

    from relax.refinement import optics_shapes

    shape_values = optics_shapes.class_kwargs(
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
            "cs_for_engine": sampling.cs_for_engine,
            "model_current_size_for_engine": sampling.model_current_size_for_engine,
            "firstiter_coarse_current_size": variant.firstiter_coarse_current_size,
            "firstiter_fine_current_size": variant.firstiter_fine_current_size,
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

    batch_overrides = {} if batching.class_batch_overrides is None else batching.class_batch_overrides[class_index]
    return (
        replace(
            half,
            particles=replace(
                half.particles,
                dataset=shape_values["experiment_dataset"],
                optics_group_ids=shape_values["optics_group_ids_k"],
                image_corrections=shape_values["image_corrections_k"],
                scale_corrections=shape_values["scale_corrections_k"],
            ),
            noise_variance=optics_shapes.class_noise_table(
                optics.noise_radial_k,
                shape_class,
                int(half.particles.dataset.image_shape[0]),
            ),
            scale_group_ids=shape_values["group_ids_k"],
            image_seed_classes=shape_values["image_seed_classes"],
        ),
        replace(
            sampling,
            current_translations=shape_values["current_translations"],
            base_translations=shape_values["base_translations"],
            translation_step=shape_values["translation_step"],
            cs_for_engine=shape_values["cs_for_engine"],
            model_current_size_for_engine=shape_values["model_current_size_for_engine"],
        ),
        replace(
            priors,
            rotation_log_prior_k=shape_values["rotation_log_prior_k"],
            class_rotation_log_prior_k=shape_values["class_rotation_log_prior_k"],
            translation_log_prior=shape_values["translation_log_prior"],
            translation_search_base=shape_values["translation_search_base"],
            trans_prior_center_for_engine=shape_values["trans_prior_center_for_engine"],
        ),
        replace(batching, class_batch_overrides=None, **batch_overrides),
        replace(
            variant,
            firstiter_coarse_current_size=shape_values["firstiter_coarse_current_size"],
            firstiter_fine_current_size=shape_values["firstiter_fine_current_size"],
        ),
        execution,
        replace(
            optics,
            noise_radial_k=None,
            class_translations=None,
            projection_scale=shape_values["projection_scale"],
            reference_current_size=shape_values["reference_current_size"],
        ),
    )


def _require_multi_shape_inputs(half: HalfScoringData, optics: OpticsSpec) -> None:
    """A half with several image shapes needs reference-shell noise spectra and each image's optics group."""
    if optics.noise_radial_k is None:
        raise ValueError("a half with several image shapes needs reference-shell noise spectra")
    if half.particles.optics_group_ids is None:
        raise ValueError("a half with several image shapes needs each image's optics group")


def _merge_shape_class_results(results, experiment_half) -> HalfScoreResult:
    """One result for a multi-shape half from its shape classes' results, on the reference box."""
    from relax.refinement import optics_shapes

    return optics_shapes.merge_class_results(
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
        return _score_half_dense_one_shape(half, sampling, priors, batching, variant, execution, optics)
    _require_multi_shape_inputs(half, optics)
    if batching.class_batch_overrides is not None and len(batching.class_batch_overrides) != len(experiment_half.classes):
        raise ValueError("class_batch_overrides needs one entry per shape class")

    results = []
    for index, shape_class in enumerate(experiment_half.classes):
        result = _score_half_dense_one_shape(
            *_dense_owners_for_shape(
                half,
                sampling,
                priors,
                batching,
                variant,
                execution,
                optics,
                shape_class,
                index,
            )
        )
        if not variant.k_class_enabled:
            # K1 shape merging retains common statistics, not class-prior summaries.
            result.classes = None
        results.append(result)
    return _merge_shape_class_results(results, experiment_half)


@dataclass(frozen=True, kw_only=True)
class LocalPriorSpec:
    """Translation-prior operands and their grid-selection policy."""

    trans_prior_center: object
    trans_prior_center_for_engine: object
    current_sigma_offset_angstrom: float
    translation_search_base: object
    local_search_translation_prior_mode: str
    replay_prior_translations: object | None = None


@dataclass(frozen=True, kw_only=True)
class LocalBatchPolicy:
    """Exact-local posterior support control."""

    max_significants: int | None


@dataclass(frozen=True, kw_only=True)
class LocalExecutionPolicy:
    """Interpolation and reconstruction controls."""

    disc_type: str
    disable_adjoint_y: bool
    disable_adjoint_ctf: bool
    source_faithful_spectrum_norm: bool = False
    relion_translation_angle_scale: float = 1.0
    # RelionConsistencyOptions.nyquist_column_counting (see DenseExecutionPolicy).
    nyquist_column_counting: str = "relion"


@dataclass(frozen=True, kw_only=True)
class LocalDiagnosticPolicy:
    """Profiling, debug capture and score-only controls; ``observer`` receives each collected profile."""

    iteration: int
    collect_local_search_profile: bool
    diagnostic_score_only: bool
    local_profile_history: object
    debug_iteration: int | None = None
    bpref_device_signature_active: bool = False
    adaptive_pass2_full_parent: bool = False
    adaptive_pass2_rotation_only: bool = False
    adaptive_pass2_denominator_mode: Literal["full_parent", "rotation_only"] | None = None
    observer: RunObserver = field(default_factory=RunObserver)


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


def _local_translation_prior_reference_translations(
    *,
    current_translations,
    base_translations,
    replay_prior_translations,
    dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, str, bool]:
    """Choose a local-search translation-prior grid compatible with scoring."""

    if replay_prior_translations is not None:
        candidate = replay_prior_translations
        source = "replay"
    else:
        candidate = base_translations
        source = "base"

    if candidate.shape == current_translations.shape:
        return np.asarray(candidate, dtype=dtype), source, False
    if base_translations.shape == current_translations.shape:
        return np.asarray(base_translations, dtype=dtype), "base", True
    return np.asarray(current_translations, dtype=dtype), "current", True


def _relion_coarse_significant_counts(significant_sample_indices):
    """Count explicit retained pass-1 samples using RELION metadata semantics."""

    if any(indices is None for indices in significant_sample_indices):
        return None
    return np.asarray(
        [np.asarray(indices).size for indices in significant_sample_indices],
        dtype=np.int32,
    )


def _prepare_local_adaptive_pass2_support(
    parent_layout,
    significant_sample_indices,
    sampling: LocalSampling,
    diagnostics: LocalDiagnosticPolicy,
    parent_order: int,
    fine_layout_dtype,
):
    """Derive fine and diagnostic support from retained parent samples."""

    pruned_parent_significant_sample_indices = significant_sample_indices
    # RELION's rlnNrOfSignificantSamples records the number of retained
    # coarse hypotheses from pass 1, not the number of fine hypotheses used
    # for reconstruction in pass 2. Preserve this before any diagnostic
    # expansion of the pass-2 parent support.
    relion_significant_counts = _relion_coarse_significant_counts(
        pruned_parent_significant_sample_indices
    )
    if relion_significant_counts is None:
        logger.warning(
            "RELION local adaptive pass 1 did not return explicit retained support; "
            "rlnNrOfSignificantSamples-compatible counts are unavailable"
        )

    parent_mode = "full_parent" if diagnostics.adaptive_pass2_full_parent else "pruned_parent"
    if diagnostics.adaptive_pass2_full_parent:
        significant_sample_indices = [None] * len(significant_sample_indices)
        logger.info(
            "RELION local adaptive pass 2: expanding all parent samples; set %s=0 for pruned-parent support",
            _LOCAL_ADAPTIVE_PASS2_FULL_PARENT_ENV,
        )
    elif diagnostics.adaptive_pass2_rotation_only:
        significant_sample_indices = _expand_significant_samples_to_full_parent_translations(
            significant_sample_indices,
            int(sampling.translations.shape[0]),
        )
        parent_mode = "significant_rotation_full_translation"
        logger.info(
            "RELION local adaptive pass 2 diagnostic: expanding significant parent rotations to all "
            "parent translations via %s=1",
            _LOCAL_ADAPTIVE_PASS2_ROTATION_ONLY_ENV,
        )

    layout_kwargs = dict(
        oversampling_order=int(sampling.search.oversampling_order),
        random_perturbation=float(sampling.perturbation),
        dtype=fine_layout_dtype,
        symmetry=sampling.search.symmetry,
    )
    pass2_layout = build_local_adaptive_pass2_hypothesis_layout(
        parent_layout,
        significant_sample_indices,
        parent_order,
        **layout_kwargs,
    )

    denominator_layout = None
    denominator_mode = diagnostics.adaptive_pass2_denominator_mode
    if denominator_mode is not None:
        if denominator_mode == "full_parent":
            denominator_significant_sample_indices = [None] * len(
                pruned_parent_significant_sample_indices
            )
        else:
            denominator_significant_sample_indices = (
                _expand_significant_samples_to_full_parent_translations(
                    pruned_parent_significant_sample_indices,
                    int(sampling.translations.shape[0]),
                )
            )
        denominator_layout = build_local_adaptive_pass2_hypothesis_layout(
            parent_layout,
            denominator_significant_sample_indices,
            parent_order,
            **layout_kwargs,
        )
        log_local_denominator_support(
            logger,
            denominator_layout,
            denominator_mode,
            _LOCAL_ADAPTIVE_PASS2_DENOMINATOR_SUPPORT_ENV,
        )
    log_local_adaptive_support(
        logger,
        parent_layout,
        significant_sample_indices,
        sampling.translations,
        pass2_layout,
    )
    return pass2_layout, relion_significant_counts, denominator_layout, parent_mode


def _build_local_adaptive_parent_layout(
    half: HalfScoringData,
    sampling: LocalSampling,
    priors: LocalPriorSpec,
    translation_prior_reference_translations,
    layout_dtype,
):
    """Build the coarse parent layout used by exact-local adaptive pass 1."""

    parent_prior_translations = priors.trans_prior_center
    if parent_prior_translations is None:
        parent_prior_translations = np.zeros(
            (
                np.asarray(half.particles.rotation_eulers).shape[0],
                sampling.translations.shape[1],
            ),
            dtype=layout_dtype,
        )
    parent_order = sampling.search.parent_order
    if parent_order < 0:
        raise ValueError(
            "local_search_order must be >= local_parent_oversampling_order; "
            f"got {sampling.search.healpix_order} and "
            f"{sampling.search.oversampling_order}",
        )
    parent_grid_metadata = build_local_search_grid_metadata(
        parent_order,
        symmetry=sampling.search.symmetry,
    )
    parent_layout = build_local_hypothesis_layout(
        half.particles.rotation_eulers,
        None,
        sampling.search.sigma_rot,
        sampling.search.sigma_psi,
        parent_order,
        sampling.translations,
        parent_prior_translations,
        priors.current_sigma_offset_angstrom,
        None,
        half.particles.dataset.voxel_size,
        grid_metadata=parent_grid_metadata,
        translation_prior_reference_translations=translation_prior_reference_translations,
        rotation_log_prior=None,
        rotation_grid_random_perturbation=sampling.perturbation,
        rotation_grid_angular_sampling_deg=relion_angular_sampling_deg(
            parent_order,
            adaptive_oversampling=0,
        ),
        dtype=layout_dtype,
    )
    return parent_layout, parent_order


def _local_owners_for_shape(
    half: HalfScoringData,
    sampling: LocalSampling,
    priors: LocalPriorSpec,
    batching: LocalBatchPolicy,
    execution: LocalExecutionPolicy,
    diagnostics: LocalDiagnosticPolicy,
    optics: OpticsSpec,
    shape_class,
    class_index: int,
) -> tuple:
    """Derive one exact-local shape class through the optics owner."""

    from relax.refinement import optics_shapes

    shape_values = optics_shapes.class_kwargs(
        {
            "experiment_dataset": half.particles.dataset,
            "previous_best_rotation_eulers_k": half.particles.rotation_eulers,
            "image_corrections_k": half.particles.image_corrections,
            "scale_corrections_k": half.particles.scale_corrections,
            "group_ids_k": half.scale_group_ids,
            "optics_group_ids_k": half.particles.optics_group_ids,
            "current_translations": sampling.translations,
            "base_translations": sampling.base_translations,
            "cs_for_engine": sampling.image_window_size,
            "model_current_size_for_engine": sampling.model_support_size,
            "local_pass1_current_size": sampling.coarse_image_window_size,
            "trans_prior_center": priors.trans_prior_center,
            "trans_prior_center_for_engine": priors.trans_prior_center_for_engine,
            "translation_search_base": priors.translation_search_base,
            "replay_prior_translations": priors.replay_prior_translations,
            "coarse_sizing": optics.coarse_sizing,
        },
        shape_class,
        half.particles.dataset.n_units,
    )
    if optics.class_translations is not None:
        translations = optics.class_translations[class_index]
        shape_values["translation_search_base"] = translations.search_base
        shape_values["trans_prior_center"] = translations.local_prior_center
        shape_values["trans_prior_center_for_engine"] = translations.engine_prior_center
    return (
        replace(
            half,
            particles=replace(
                half.particles,
                dataset=shape_values["experiment_dataset"],
                optics_group_ids=shape_values["optics_group_ids_k"],
                rotation_eulers=shape_values["previous_best_rotation_eulers_k"],
                image_corrections=shape_values["image_corrections_k"],
                scale_corrections=shape_values["scale_corrections_k"],
            ),
            noise_variance=optics_shapes.class_noise_table(
                optics.noise_radial_k,
                shape_class,
                int(half.particles.dataset.image_shape[0]),
            ),
            scale_group_ids=shape_values["group_ids_k"],
        ),
        replace(
            sampling,
            translations=shape_values["current_translations"],
            base_translations=shape_values["base_translations"],
            image_window_size=shape_values["cs_for_engine"],
            model_support_size=shape_values["model_current_size_for_engine"],
            coarse_image_window_size=shape_values["local_pass1_current_size"],
        ),
        replace(
            priors,
            trans_prior_center=shape_values["trans_prior_center"],
            trans_prior_center_for_engine=shape_values["trans_prior_center_for_engine"],
            translation_search_base=shape_values["translation_search_base"],
            replay_prior_translations=shape_values["replay_prior_translations"],
        ),
        batching,
        execution,
        diagnostics,
        replace(
            optics,
            noise_radial_k=None,
            class_translations=None,
            projection_scale=shape_values["projection_scale"],
            reference_current_size=shape_values["reference_current_size"],
        ),
    )


def _score_half_local(
    half: HalfScoringData,
    sampling: LocalSampling,
    priors: LocalPriorSpec,
    batching: LocalBatchPolicy,
    execution: LocalExecutionPolicy,
    diagnostics: LocalDiagnosticPolicy,
    optics: OpticsSpec,
) -> HalfScoreResult:
    """Exact-local scoring for one half; several shapes run per shape class."""

    from relax.refinement import optics_shapes

    experiment_half = half.particles.dataset
    if not isinstance(experiment_half, optics_shapes.MultiShapeHalf):
        return _score_half_local_one_shape(half, sampling, priors, batching, execution, diagnostics, optics)
    _require_multi_shape_inputs(half, optics)
    optics_shapes.require_exact_local_parent_windows(
        {
            "experiment_dataset": experiment_half,
            "cs_for_engine": sampling.image_window_size,
            "local_pass1_current_size": sampling.coarse_image_window_size,
            "coarse_sizing": optics.coarse_sizing,
        }
    )
    results = [
        _score_half_local_one_shape(
            *_local_owners_for_shape(
                half,
                sampling,
                priors,
                batching,
                execution,
                diagnostics,
                optics,
                shape_class,
                index,
            )
        )
        for index, shape_class in enumerate(experiment_half.classes)
    ]
    return _merge_shape_class_results(results, experiment_half)


def _score_half_local_one_shape(
    half: HalfScoringData,
    sampling: LocalSampling,
    priors: LocalPriorSpec,
    batching: LocalBatchPolicy,
    execution: LocalExecutionPolicy,
    diagnostics: LocalDiagnosticPolicy,
    optics: OpticsSpec,
) -> HalfScoreResult:
    """Local-search E+M scoring for one half-set.

    Stable inputs remain under their data, sampling, prior, batching,
    execution, diagnostic and optics owners. Only route-derived values become
    locals inside this function.

    Routes through ``_run_local_search_iteration``. Local searches are K=1 only: Class3D keeps
    global searches, as RELION switches to local searches from the HEALPix order
    only under auto-refine (ml_optimiser.cpp:2541-2565, 3936-3938).

    Caller handles ``noise_stats_per_half[k]``, ``pose_rotations[k] = None``,
    and ``coarse_ha[k] = ha_k`` from the returned ``HalfScoreResult``.
    """

    # RELION's convertAllSquaredDifferencesToWeights uses mymodel.pdf_direction
    # only when orientational_prior_mode == NOPRIOR. Local searches run through
    # PRIOR_ROTTILT_PSI and score the local direction/psi priors in the
    # hypothesis layout, so adding the learned global direction prior here
    # biases both support selection and final weights.

    reconstruction_current_size_for_engine = (
        sampling.image_window_size
        if sampling.model_support_size is None
        else sampling.model_support_size
    )
    # Class3D local searches (--sigma_ang) score every class at each particle's local orientations.
    projector_slabs = None if half.projector is None else half.projector.data
    n_classes = int(np.shape(projector_slabs)[0]) if projector_slabs is not None and np.ndim(projector_slabs) == 4 else 1

    local_debug_iteration = (
        diagnostics.iteration + 1 if diagnostics.debug_iteration is None else int(diagnostics.debug_iteration)
    )
    parent_precision = local_precision(local_debug_iteration, pass_index=1)
    fine_precision = local_precision(local_debug_iteration, pass_index=2)
    # Adaptive pass-2 (fine, oversampled) hypothesis layout precision; see
    # ``parent_local_layout_dtype`` below for the matching pass-1 value.
    fine_local_layout_dtype = fine_precision.rotation_real_dtype
    if fine_precision.use_float64_scoring or fine_precision.use_float64_projections:
        logger.info(
            "Local-search precision iteration %d: pass1 scoring/projections=%s/%s pass2 scoring/projections=%s/%s",
            local_debug_iteration,
            parent_precision.use_float64_scoring,
            parent_precision.use_float64_projections,
            fine_precision.use_float64_scoring,
            fine_precision.use_float64_projections,
        )

    # Keep the local-search hypothesis grid genuinely double precision end to
    # end when either flag requests it; default stays float32 to match
    # RELION's accelerated-GPU precision.
    parent_local_layout_dtype = parent_precision.rotation_real_dtype
    if priors.local_search_translation_prior_mode == "coarse":
        translation_prior_reference_translations, prior_grid_source, prior_grid_shape_mismatch = (
            _local_translation_prior_reference_translations(
                current_translations=sampling.translations,
                base_translations=sampling.base_translations,
                replay_prior_translations=priors.replay_prior_translations,
                dtype=parent_local_layout_dtype,
            )
        )
        if prior_grid_shape_mismatch:
            logger.warning(
                "RELION mode: local translation prior grid from replay/base did not match scoring grid; "
                "using %s grid shape=%s for scoring grid shape=%s",
                prior_grid_source,
                translation_prior_reference_translations.shape,
                sampling.translations.shape,
            )
        logger.info(
            "RELION mode: local translation prior uses coarse %s grid (n=%d) while scoring perturbed translations",
            prior_grid_source,
            translation_prior_reference_translations.shape[0],
        )
    else:
        translation_prior_reference_translations = np.asarray(
            sampling.translations, dtype=parent_local_layout_dtype
        )
    if int(sampling.search.oversampling_order) > 0:
        logger.info(
            "RELION local search: expanding translations by oversampling_order=%d (coarse n=%d -> fine n=%d)",
            int(sampling.search.oversampling_order),
            int(sampling.translations.shape[0]),
            int(sampling.translations.shape[0]) * 4 ** int(sampling.search.oversampling_order),
        )
    # Shared owners for one typed pass. Parent, denominator and final execution
    # derive their intentional differences with ``replace`` below.
    local_data = LocalSearchData(
            experiment_dataset=half.particles.dataset,
            mean=half.reference,
            noise_variance=half.noise_variance,
            image_corrections=half.particles.image_corrections,
            scale_corrections=half.particles.scale_corrections,
            group_ids=half.scale_group_ids,
            scale_correction_group_count=half.scale_group_count,
            scale_correction_data_vs_prior=half.scale_correction_data_vs_prior,
            image_pre_shifts=priors.translation_search_base,
            optics_group_ids=half.particles.optics_group_ids,
    )
    local_grid = LocalSearchGridSpec(
            prior_rotations=half.particles.rotation_eulers,
            rotation_grid_rotations=sampling.rotations,
            healpix_order=sampling.search.healpix_order,
            sigma_rot=sampling.search.sigma_rot,
            sigma_psi=sampling.search.sigma_psi,
            translations=sampling.translations,
            prior_translations=priors.trans_prior_center,
            sigma_offset_angstrom=priors.current_sigma_offset_angstrom,
            translation_prior_reference_translations=translation_prior_reference_translations,
            translation_prior_centers=priors.trans_prior_center_for_engine,
            rotation_grid_random_perturbation=sampling.perturbation,
            rotation_grid_angular_sampling_deg=sampling.angular_step_deg,
            local_parent_oversampling_order=sampling.search.oversampling_order,
            rotation_grid_mstep_rotations=sampling.mstep_rotations,
            generate_relion_mstep_rotations=True,
            symmetry=sampling.search.symmetry,
            n_classes=n_classes,
            image_seed_classes=half.image_seed_classes if n_classes > 1 else None,
    )
    local_kernel = LocalSearchKernelPolicy(
            disc_type=execution.disc_type,
            current_size=sampling.image_window_size,
            reconstruction_current_size=reconstruction_current_size_for_engine,
            projection_padding_factor=PROJECTION_PADDING_FACTOR,
            reconstruction_padding_factor=RECONSTRUCTION_PADDING_FACTOR,
            square_window=RELION_FOURIER_WINDOW_SQUARE,
            half_spectrum_scoring=True,
            relion_projector_half=None if half.projector is None else half.projector.data,
            relion_projector_r_max=None if half.projector is None else half.projector.r_max,
            source_faithful_spectrum_norm=execution.source_faithful_spectrum_norm,
            relion_translation_angle_scale=float(execution.relion_translation_angle_scale),
            projection_scale=float(optics.projection_scale),
            reconstruction_volume_current_size=optics.reference_current_size,
            reconstruction_image_radius=reconstruction_image_radius(
                optics.reference_current_size,
                optics.projection_scale,
            ),
            nyquist_column_counting=execution.nyquist_column_counting,
            wsum_current_size=sampling.wsum_current_size,
    )
    local_support = LocalSearchSupportPolicy(
            disable_adjoint_y=execution.disable_adjoint_y,
            disable_adjoint_ctf=execution.disable_adjoint_ctf,
            adaptive_fraction=RELION_ADAPTIVE_FRACTION,
            max_significants=batching.max_significants,
            return_profile=diagnostics.collect_local_search_profile,
    )
    pass2_layout = None
    relion_significant_counts_k = None
    local_adaptive_pass2_parent_mode = "none"
    local_adaptive_pass2_denominator_layout = None
    local_normalization_log_evidence = None
    if int(sampling.search.oversampling_order) > 0:
        parent_layout, parent_order = _build_local_adaptive_parent_layout(
            half,
            sampling,
            priors,
            translation_prior_reference_translations,
            parent_local_layout_dtype,
        )
        if n_classes > 1:
            # Class3D: pass 1 and pass 2 share the class-expanded parents, so the significant samples
            # carry their class into pass 2's support.
            parent_layout = expand_local_layout_classes(parent_layout, n_classes)
            if half.image_seed_classes is not None:
                parent_layout = restrict_local_layout_classes(parent_layout, half.image_seed_classes)
        parent_local_rot_max = (
            int(np.max(np.asarray(parent_layout.rotation_counts, dtype=np.int64)))
            if int(np.asarray(parent_layout.rotation_counts).size)
            else 1
        )
        logger.info(
            "RELION local adaptive pass 1: parent_order=%d local_rot_max=%d n_trans=%d current_size=%s",
            parent_order,
            parent_local_rot_max,
            int(sampling.translations.shape[0]),
            sampling.coarse_image_window_size,
        )
        logger.info("RELION local adaptive pass 1: using manual supplied-PPref interpolation")
        parent_outputs = _run_local_search_iteration(
            local_data,
            replace(
                local_grid,
                rotation_grid_rotations=None,
                healpix_order=parent_order,
                pass2_layout=parent_layout,
                rotation_grid_random_perturbation=0.0,
                rotation_grid_angular_sampling_deg=None,
                local_parent_oversampling_order=0,
                rotation_grid_mstep_rotations=None,
                generate_relion_mstep_rotations=False,
            ),
            replace(
                local_kernel,
                current_size=sampling.coarse_image_window_size,
                reconstruction_current_size=None,
                wsum_current_size=None,
                use_float64_scoring=parent_precision.use_float64_scoring,
                use_float64_projections=parent_precision.use_float64_projections,
                relion_exact_score_translation=bool(
                    scoring_policy.RELION_EXACT_FINE_GAUSSIAN and not parent_precision.use_float64_scoring
                ),
                # RELION's GPU pass 1 projects through the projector texture, as the
                # global pass 1 and pass 2 do.
                projection_relion_texture_interp=None,
                projection_relion_acc_double_floorf_quirk=RELION_ACC_DOUBLE_FLOORF_QUIRK,
                # RELION pass 1 uses the coarse diff2 kernel's row rule.
                projection_relion_kernel="coarse",
            ),
            replace(
                local_support,
                disable_adjoint_y=True,
                disable_adjoint_ctf=True,
                reconstruct_significant_only=True,
                return_reconstruction_sample_indices=True,
                apply_max_significants_to_support=True,
                score_only=True,
                return_profile=True,
            ),
        )
        parent_profile = parent_outputs.profile_summary
        significant_sample_indices = parent_profile["reconstruction_sample_indices_by_image"]
        (
            pass2_layout,
            relion_significant_counts_k,
            local_adaptive_pass2_denominator_layout,
            local_adaptive_pass2_parent_mode,
        ) = _prepare_local_adaptive_pass2_support(
            parent_layout,
            significant_sample_indices,
            sampling,
            diagnostics,
            parent_order,
            fine_local_layout_dtype,
        )
    local_relion_x_half_mstep = _k1_relion_x_half_mstep_enabled()
    if diagnostics.diagnostic_score_only:
        local_relion_x_half_mstep = False
    if sampling.search.symmetry != "C1" and not diagnostics.diagnostic_score_only and not local_relion_x_half_mstep:
        raise RuntimeError(
            f"{sampling.search.symmetry} exact-local reconstruction requires RELION x-half BPref "
            f"accumulation; {_K1_RELION_X_HALF_MSTEP_ENV}=0, CPU-only execution, or disabled custom CUDA "
            "is unsupported for non-C1 symmetry"
        )
    if local_relion_x_half_mstep:
        logger.info(
            "RELION local K=1 M-step: using x-half BPref-layout backprojection",
        )
    if local_adaptive_pass2_denominator_layout is not None:
        logger.info("RELION local adaptive pass 2 diagnostic: running score-only broad-denominator probe")
        local_debug_env_names = [
            name
            for name in os.environ
            if name.startswith("RELAX_LOCAL_SCORE_DUMP_")
            or name.startswith("RELAX_LOCAL_FUSED_POSTERIOR_DUMP_")
            or name.startswith("RELAX_LOCAL_NOISE_COMPONENT_DUMP_")
        ]
        saved_local_debug_env = {name: os.environ.pop(name) for name in local_debug_env_names}
        try:
            denominator_outputs = _run_local_search_iteration(
                local_data,
                replace(
                    local_grid,
                    pass2_layout=local_adaptive_pass2_denominator_layout,
                    rotation_grid_angular_sampling_deg=relion_angular_sampling_deg(
                        sampling.search.healpix_order,
                        adaptive_oversampling=0,
                    ),
                    local_parent_oversampling_order=0,
                    rotation_grid_mstep_rotations=None,
                    generate_relion_mstep_rotations=False,
                ),
                replace(
                    local_kernel,
                    accumulate_noise=False,
                    use_float64_scoring=fine_precision.use_float64_scoring,
                    use_float64_projections=fine_precision.use_float64_projections,
                    relion_exact_score_translation=bool(
                        scoring_policy.RELION_EXACT_FINE_GAUSSIAN and not fine_precision.use_float64_scoring
                    ),
                ),
                replace(
                    local_support,
                    disable_adjoint_y=True,
                    disable_adjoint_ctf=True,
                    reconstruct_significant_only=False,
                    score_only=True,
                    return_profile=False,
                ),
            )
        finally:
            os.environ.update(saved_local_debug_env)
        denominator_stats = denominator_outputs.relion_stats
        local_normalization_log_evidence = np.asarray(
            denominator_stats.log_evidence_per_image,
            dtype=np.float64,
        )
        logger.info(
            "RELION local adaptive pass 2 diagnostic: broad-denominator evidence ready (finite=%d/%d)",
            int(np.count_nonzero(np.isfinite(local_normalization_log_evidence))),
            int(local_normalization_log_evidence.size),
        )
    # RELION's accelerated local-search loop still executes the symbolic
    # second pass when adaptive_oversampling == 0. In that case
    # convertAllSquaredDifferencesToWeights sets significant_weight to the
    # minimum fine-pass weight, so storeWeightedSums keeps all local
    # candidates. Do not apply the 0.999 significant-support prune on
    # this os0 path.
    local_reconstruct_significant_only = int(sampling.search.oversampling_order) > 0
    local_accumulate_noise = not diagnostics.diagnostic_score_only
    local_disable_adjoint_y = bool(execution.disable_adjoint_y or diagnostics.diagnostic_score_only)
    local_disable_adjoint_ctf = bool(execution.disable_adjoint_ctf or diagnostics.diagnostic_score_only)
    logger.info(
        "RELION local fine pass 2: supplied-PPref interpolation follows "
        "RELAX_RELION_PROJECTOR_TEXTURE_INTERP (default texture)"
    )
    local_outputs = _run_local_search_iteration(
        local_data,
        replace(local_grid, pass2_layout=pass2_layout),
        replace(
            local_kernel,
            accumulate_noise=local_accumulate_noise,
            use_float64_scoring=fine_precision.use_float64_scoring,
            use_float64_projections=fine_precision.use_float64_projections,
            relion_exact_score_translation=bool(
                scoring_policy.RELION_EXACT_FINE_GAUSSIAN and not fine_precision.use_float64_scoring
            ),
            # RELION local execution is intentionally hybrid: parent pass 1
            # uses manual supplied-PPref projection, while fine pass 2 follows
            # the user-switchable texture default.
            projection_relion_texture_interp=None,
            projection_relion_acc_double_floorf_quirk=RELION_ACC_DOUBLE_FLOORF_QUIRK,
        ),
        replace(
            local_support,
            mstep_relion_x_half=local_relion_x_half_mstep,
            disable_adjoint_y=local_disable_adjoint_y,
            disable_adjoint_ctf=local_disable_adjoint_ctf,
            reconstruct_significant_only=local_reconstruct_significant_only,
            return_best_pose_details=True,
            normalization_log_evidence=local_normalization_log_evidence,
            stats_use_reconstruction_probs=local_reconstruct_significant_only,
            score_only=diagnostics.diagnostic_score_only,
        ),
    )
    Ft_y_k = local_outputs.Ft_y
    Ft_ctf_k = local_outputs.Ft_ctf
    ha_k = local_outputs.hard_assignment
    best_rots_k = local_outputs.best_pose_rotations
    best_trans_k = local_outputs.best_pose_translations
    em_stats_k = local_outputs.relion_stats
    noise_stats_k = local_outputs.noise_stats
    if diagnostics.collect_local_search_profile:
        local_profile_k = local_outputs.profile_summary
        profile_row = dict(local_profile_k)
        profile_row["iteration"] = np.int32(diagnostics.iteration)
        profile_row["half_index"] = np.int32(half.particles.index)
        profile_row["local_adaptive_pass2_parent_mode"] = local_adaptive_pass2_parent_mode
        profile_row["local_adaptive_pass2_full_parent"] = np.bool_(local_adaptive_pass2_parent_mode == "full_parent")
        profile_row["diagnostic_score_only"] = np.bool_(diagnostics.diagnostic_score_only)
        diagnostics.local_profile_history.append(profile_row)
        diagnostics.observer.local_search_profile(diagnostics.iteration, half.particles.index, local_profile_k)
    # Must match the current-size BPref grid allocated by the local engine above; downstream
    # join/reconstruct calls infer layout from this shape.
    mstep_accumulator_shape = (
        relion_backprojector_volume_shape(
            half.particles.dataset.volume_shape,
            RECONSTRUCTION_PADDING_FACTOR,
            # Images on another grid fill the backprojector at the reference model size.
            current_size=(
                reconstruction_current_size_for_engine
                if optics.reference_current_size is None
                else optics.reference_current_size
            ),
        )
        if local_relion_x_half_mstep
        else None
    )
    if local_outputs.class_pass is not None:
        return _class_local_half_result(
            local_outputs.class_pass,
            n_classes=n_classes,
            significant_counts=relion_significant_counts_k,
            mstep_accumulator_shape=mstep_accumulator_shape,
        )
    pose_dtype = _dense_global_scoring_dtype()
    best_rots = np.asarray(best_rots_k, dtype=pose_dtype)
    best_eulers = (
        np.asarray(local_outputs.best_pose_eulers_deg, dtype=np.float64)
        if local_outputs.best_pose_eulers_deg is not None
        else utils.R_to_relion(np.asarray(best_rots_k), degrees=True).astype(pose_dtype)
    )
    best_translations = np.asarray(best_trans_k, dtype=pose_dtype)
    return HalfScoreResult(
        ha=ha_k,
        Ft_y=Ft_y_k,
        Ft_ctf=Ft_ctf_k,
        em_stats=em_stats_k,
        noise_stats=noise_stats_k,
        best_pose_rotations=best_rots,
        best_pose_rotation_eulers=best_eulers,
        best_pose_translations=best_translations,
        significant_counts=relion_significant_counts_k,
        mstep_full_half_axis=0 if local_relion_x_half_mstep else None,
        mstep_accumulator_shape=mstep_accumulator_shape,
    )


def _class_local_half_result(class_pass, *, n_classes: int, significant_counts, mstep_accumulator_shape):
    """A Class3D local pass as the half's K-class result, as the subtomogram K-class pass adapts its own.

    ``class_pass`` is the resident local engine's ``ResidentKClassPass2Output``; its class rotation sums
    are over the layout's posterior grid, which is the direction-prior grid of the local search.
    """

    from relax.classification.k_class import _class_segmented_em_result

    k_class_result = _class_segmented_em_result(
        class_pass,
        n_classes=n_classes,
        class_posterior_sums_from_noise=True,
        return_profile=False,
        host_accumulators=True,
        mstep_full_half_axis=0,
        mstep_accumulator_shape=mstep_accumulator_shape,
    )
    score_result = class_em_to_half_result(
        k_class_result,
        effective_rotations=np.zeros((int(np.shape(class_pass.class_rotation_posterior_sums)[1]), 0)),
        rot_pmap_for_collapse=None,
        adaptive_os_local=0,
        require_best_pose_details=True,
        pose_dtype=_dense_global_scoring_dtype(),
    )
    score_result.ha = np.asarray(score_result.ha, dtype=np.int32)
    score_result.significant_counts = significant_counts
    score_result.mstep_full_half_axis = 0
    score_result.mstep_accumulator_shape = mstep_accumulator_shape
    return score_result


def _score_half_local_in_bpref_scope(
    half: HalfScoringData,
    sampling: LocalSampling,
    priors: LocalPriorSpec,
    batching: LocalBatchPolicy,
    execution: LocalExecutionPolicy,
    diagnostics: LocalDiagnosticPolicy,
    optics: OpticsSpec,
) -> HalfScoreResult:
    """Run local scoring with device-capture flags disabled or fail closed."""

    if diagnostics.bpref_device_signature_active:
        raise RuntimeError("BPref device signature capture is supported only by sparse adaptive pass 2")
    with em_cuda_kernels.bpref_device_signature_scope(False):
        return _score_half_local(half, sampling, priors, batching, execution, diagnostics, optics)
