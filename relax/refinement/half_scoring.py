"""Dispatch dense and local scoring for one refinement half-set.

The iteration controller owns scheduling, state transitions, result-container
lifetime and device offloading. These adapters choose existing engine routes,
prepare per-half arguments and populate caller-owned result slots. Diagnostic
BPref scopes surround the same calls as before; no array copies are introduced
by the ownership boundary.
"""

import logging
import os
from dataclasses import dataclass, replace
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from recovar import utils

from relax.classification.k_class import (
    _sparse_pass2_selected,
    run_dense_k_class_em,
    run_dense_k_class_em_adaptive,
)
from relax.cuda import kernels as em_cuda_kernels
from relax.dense.em_engine import run_em
from relax.dense.score_outputs import (
    HalfScoreResult,
    PerHalfOutputs,
    _collapse_fine_pose_assignments_to_coarse,
    _collapse_single_class_stats_to_coarse,
    _scatter_dense_k_class_result,
    _select_single_class_accumulator,
)
from relax.dense.scoring_policy import (
    _DENSE_EM_STATIC_KWARGS,
    _K1_RELION_X_HALF_MSTEP_ENV,
    _LOCAL_ADAPTIVE_PASS2_DENOMINATOR_SUPPORT_ENV,
    _LOCAL_ADAPTIVE_PASS2_FULL_PARENT_ENV,
    _LOCAL_ADAPTIVE_PASS2_ROTATION_ONLY_ENV,
    PADDING_FACTOR,
    PROJECTION_PADDING_FACTOR,
    RELION_ACC_DOUBLE_FLOORF_QUIRK,
    RELION_ADAPTIVE_FRACTION,
    RELION_FOURIER_WINDOW_SQUARE,
    _dense_global_scoring_dtype,
    _k1_relion_x_half_mstep_enabled,
    _k1_skip_significance_pruning_enabled,
    _k_class_relion_x_half_mstep_enabled,
)
from relax.diagnostics import parity_dump as _parity_dump
from relax.diagnostics.local_debug import log_local_adaptive_support, log_local_denominator_support
from relax.helpers.batch_planning import _plan_kclass_adaptive_grid_batch_sizes
from relax.helpers.half_volume_mstep import relion_backprojector_volume_shape
from relax.helpers.oversampling import build_adaptive_pass2_grids
from relax.helpers.preprocessing import uses_relion_cuda_image_preprocessing
from relax.local.local_layout import build_local_adaptive_pass2_hypothesis_layout, build_local_hypothesis_layout
from relax.refinement.firstiter_cc import (
    FirstIterCCBatching,
    FirstIterCCData,
    FirstIterCCExecution,
    FirstIterCCGridSpec,
    FirstIterCCPolicy,
    _score_kclass_firstiter_cc_pass2,
)
from relax.refinement.local_search_iteration import (
    LocalSearchBatchPolicy,
    LocalSearchData,
    LocalSearchDiagnosticPolicy,
    LocalSearchGridSpec,
    LocalSearchKernelPolicy,
    LocalSearchSupportPolicy,
    _run_local_search_iteration,
)
from relax.refinement.optics_shapes import engine_projection_inputs, reconstruction_image_radius, reference_grid_kwargs
from relax.relion.optics_aberrations import dataset_projection_magnification, projection_rotations, reported_rotations
from relax.sampling import (
    apply_relion_translation_perturbation,
    build_local_search_grid_metadata,
    relion_angular_sampling_deg,
    rotation_grid_size,
)
from relax.sparse_pass2.engine_record import warn_deprecated_engine

logger = logging.getLogger("relax.dense.half_scoring")


def _expand_significant_samples_to_full_parent_translations(
    significant_sample_indices,
    n_parent_translations: int,
):
    """Expand significant parent rotation ids to every parent translation."""

    n_parent_translations = int(n_parent_translations)
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


class _AdaptivePass2Grids(NamedTuple):
    """Coarse/fine trial grids of one adaptive-oversampling expectation."""

    coarse_rotations: np.ndarray
    coarse_translations: np.ndarray
    fine_rotations: np.ndarray
    fine_translations: np.ndarray
    rotation_parent_map: np.ndarray
    translation_parent_map: np.ndarray
    fine_mstep_rotations: np.ndarray
    coarse_translation_phase_source: np.ndarray
    n_fine_translations: int


def _adaptive_pass2_grids(
    effective_rotations,
    current_translations,
    base_translations,
    *,
    healpix_order,
    adaptive_oversampling,
    translation_step,
    random_perturbation,
    coarse_rotation_ids,
    symmetry: str = "C1",
) -> _AdaptivePass2Grids:
    """Materialize RELION's two-pass trial grids for the dense adaptive engine.

    Pass 1 scores the perturbed coarse grid; pass 2 scores the oversampled
    children of the significant coarse candidates with parent maps back to
    the coarse grid, and the exact M-step rotations of the fine grid. The
    coarse translation phases come from the host-double base grid under the
    same SamplingPerturbation. The K=1 and K-class dense routes share this rule.
    """

    (
        coarse_rot,
        coarse_trans,
        fine_rot,
        fine_trans,
        rot_pmap,
        trans_pmap,
        fine_mstep_rot,
    ) = build_adaptive_pass2_grids(
        effective_rotations,
        current_translations,
        base_translations,
        int(healpix_order),
        adaptive_oversampling,
        float(translation_step),
        random_perturbation,
        return_mstep_rotations=True,
        **({"coarse_rotation_ids": coarse_rotation_ids} if coarse_rotation_ids is not None else {}),
        **({"symmetry": symmetry} if symmetry != "C1" else {}),
    )
    coarse_translation_phase_source = apply_relion_translation_perturbation(
        np.asarray(base_translations, dtype=np.float64),
        float(random_perturbation),
        float(translation_step),
    )
    return _AdaptivePass2Grids(
        coarse_rot,
        coarse_trans,
        fine_rot,
        fine_trans,
        rot_pmap,
        trans_pmap,
        fine_mstep_rot,
        coarse_translation_phase_source,
        int(fine_trans.shape[0]),
    )


def _dense_uses_adaptive_engine(adaptive_oversampling, group_ids) -> bool:
    """Whether dense scoring (K=1 or K-class) runs through the adaptive/sparse engine.

    RELION accumulates the group-scale sufficient statistics (``XA``/``AA``)
    and the per-particle norm-correction residuals (``ml_optimiser.cpp``
    ``storeWeightedSums``) in every expectation pass, whatever the oversampling
    order. Only RECOVAR's adaptive/sparse engine accumulates them, so scoring
    with RELION scale groups uses that engine at the requested oversampling
    order, including 0, where its single coarse pass on the current grid is
    RELION's single pass with significance pruning. Without scale groups,
    oversampling 0 keeps the direct dense engine.
    """

    return int(adaptive_oversampling) > 0 or group_ids is not None


def _adaptive_engine_common_kwargs(
    pass2_grids: _AdaptivePass2Grids,
    priors: "DensePriorSpec",
    batching: "DenseBatchPolicy",
    sampling: "DenseSamplingSpec",
    execution: "DenseExecutionPolicy",
    *,
    sparse_pass2,
    full_grid_mstep: bool = False,
) -> dict:
    """Owner-derived keywords shared by both adaptive dense routes.

    Both routes accumulate noise, keep RELION's adaptive significance fraction
    and prune the fine M-step rotations only when pass 2 is sparse. Route-local
    batch sizes, Fourier sizes and oversampling order stay beside each engine
    call rather than being hidden in a one-call plan.
    """

    return dict(
        class_log_priors=priors.class_log_priors,
        accumulate_noise=True,
        adaptive_fraction=RELION_ADAPTIVE_FRACTION,
        max_significants=(-1 if batching.max_significants is None else int(batching.max_significants)),
        relion_fine_mstep_prune=bool(sparse_pass2),
        coarse_healpix_order=int(sampling.current_healpix_order),
        fine_mstep_rotations_override=(pass2_grids.fine_mstep_rotations if sparse_pass2 or full_grid_mstep else None),
        return_best_pose_details=execution.return_best_pose_details,
        bpref_device_signature_active=execution.bpref_device_signature_active,
        debug_iteration=execution.debug_iteration,
    )


def _coarse_pose_assignments(ha, *, rot_parent_map, trans_parent_map, n_trans_coarse, n_trans_fine):
    """Collapse fine pose assignments onto the coarse grid; ``None`` when no fine pass ran."""

    if trans_parent_map is None or n_trans_fine is None:
        return None
    return _collapse_fine_pose_assignments_to_coarse(
        ha,
        rot_parent_map=rot_parent_map,
        trans_parent_map=trans_parent_map,
        n_trans_coarse=n_trans_coarse,
        n_trans_fine=n_trans_fine,
    )


@dataclass(frozen=True, kw_only=True)
class DenseHalfData:
    """Per-half arrays and output slots retained by a dense scoring call."""

    k: int
    experiment_dataset: object
    means_k: object
    mean_variance: object
    noise_variance_k: object
    image_corrections_k: object
    scale_corrections_k: object
    outputs: PerHalfOutputs
    group_ids_k: object | None = None
    group_count_k: object | None = None
    scale_correction_data_vs_prior: object | None = None
    optics_group_ids_k: object | None = None
    # RELION's seed iteration of a Class3D run from one reference: each image's class
    # (k_class_inputs.seed_iteration_supports); the adaptive K-class route takes it.
    image_seed_classes: object | None = None


@dataclass(frozen=True, kw_only=True)
class DenseSamplingSpec:
    """Sampling grids, reconstruction sizes and state for one dense half."""

    effective_rotations: object
    current_translations: object
    base_translations: object
    current_healpix_order: int
    state: object
    random_perturbation: float
    disc_type: str
    cs_for_engine: int | None
    coarse_engine: str = "auto"
    model_current_size_for_engine: int | None = None
    coarse_rotation_ids: object | None = None
    coarse_scoring_rotations: object | None = None
    symmetry: str = "C1"


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
    firstiter_updates_em_kwargs_ibs: bool = False


@dataclass(frozen=True, kw_only=True)
class DenseExecutionPolicy:
    """Ablation, projector and diagnostic choices for dense scoring."""

    disable_adjoint_y: bool
    disable_adjoint_ctf: bool
    relion_projector_half: object | None = None
    relion_projector_r_max: int | None = None
    return_best_pose_details: bool = True
    bpref_device_signature_active: bool = False
    debug_iteration: int | None = None
    diagnostic_float64_pass2: bool = False
    preserve_bpref_particle_order: bool = False
    source_faithful_spectrum_norm: bool = False
    relion_translation_angle_scale: float = 1.0
    # The K=1 --firstiter_cc coarse-tree top-2 rescore margin (RelionParityOptions); None is off.
    firstiter_cc_tree_rescore_max_margin: float | None = None


@dataclass(frozen=True, kw_only=True)
class DenseOpticsSpec:
    """Multi-shape adaptations, absent for an ordinary single-shape half."""

    noise_radial_k: object | None = None
    coarse_sizing: tuple[float, float | None] | None = None
    class_batch_overrides: tuple[dict, ...] | None = None
    class_translation_overrides: tuple[dict, ...] | None = None
    projection_scale: float = 1.0
    reference_current_size: int | None = None


def _score_direct_k1_dense(
    half: DenseHalfData,
    sampling: DenseSamplingSpec,
    execution: DenseExecutionPolicy,
    em_kwargs,
) -> HalfScoreResult:
    """Run the legacy single-pass K=1 dense engine."""

    if half.optics_group_ids_k is not None:
        raise NotImplementedError(
            "the single-pass dense engine keeps one optics group's noise spectrum"
        )
    if dataset_projection_magnification(half.experiment_dataset) is not None:
        raise NotImplementedError("the single-pass dense engine does not implement anisotropic magnification")
    # Scale groups never reach the direct dense engine: see
    # _dense_uses_adaptive_engine.
    warn_deprecated_engine(
        "dense",
        "global",
        "a K=1 pass at oversampling 0 without scale groups takes the direct dense engine",
    )
    direct_em_kwargs = dict(em_kwargs)
    direct_em_kwargs.pop("group_ids", None)
    direct_em_kwargs.pop("scale_correction_group_count", None)
    direct_em_kwargs.pop("scale_correction_data_vs_prior", None)
    # Exact fine-Gaussian scoring is implemented only by sparse pass 2. This
    # branch is the single dense pass used when adaptive oversampling is off.
    direct_em_kwargs.pop("relion_exact_fine_gaussian", None)
    direct_em_kwargs.pop("reconstruction_current_size", None)
    # A single dense pass projects the means itself; it has no pass 1 to read the projector.
    direct_em_kwargs.pop("relion_projector_half", None)
    direct_em_kwargs.pop("relion_projector_r_max", None)
    em_result = run_em(
        half.experiment_dataset,
        half.means_k,
        half.mean_variance,
        half.noise_variance_k,
        sampling.effective_rotations,
        sampling.current_translations,
        sampling.disc_type,
        return_stats=True,
        accumulate_noise=True,
        disable_adjoint_y=execution.disable_adjoint_y,
        disable_adjoint_ctf=execution.disable_adjoint_ctf,
        **direct_em_kwargs,
    )
    return HalfScoreResult(
        ha=em_result.hard_assignments,
        Ft_y=em_result.Ft_y,
        Ft_ctf=em_result.Ft_ctf,
        em_stats=em_result.stats,
        noise_stats=em_result.noise_stats,
        mstep_accumulator_shape=None,
    )


def _score_direct_kclass_dense(
    half: DenseHalfData,
    sampling: DenseSamplingSpec,
    priors: DensePriorSpec,
    execution: DenseExecutionPolicy,
    optics: DenseOpticsSpec,
    em_kwargs,
):
    """Run the legacy single-pass K-class dense engine."""

    if optics.projection_scale != 1.0 or optics.reference_current_size is not None:
        raise NotImplementedError("the single-pass dense engine keeps the images on the reference grid")
    if dataset_projection_magnification(half.experiment_dataset) is not None:
        raise NotImplementedError("the single-pass dense engine does not implement anisotropic magnification")
    warn_deprecated_engine(
        "dense",
        "global",
        "a K-class pass at oversampling 0 without scale groups takes the direct dense engine",
    )
    dense_em_kwargs = dict(em_kwargs)
    # The direct dense K-class wrapper delegates to run_em, which does not
    # implement RELION x-half accumulators. Keep that branch on its historical
    # layout and avoid tagging its full-volume output as x-half-expanded.
    dense_em_kwargs.pop("mstep_relion_x_half", None)
    dense_em_kwargs.pop("group_ids", None)
    dense_em_kwargs.pop("scale_correction_group_count", None)
    dense_em_kwargs.pop("scale_correction_data_vs_prior", None)
    # Exact fine-Gaussian scoring is implemented only by sparse pass 2. A
    # non-adaptive dense iteration has no fine pass to select.
    dense_em_kwargs.pop("relion_exact_fine_gaussian", None)
    dense_em_kwargs.pop("reconstruction_current_size", None)
    dense_em_kwargs.pop("relion_projector_half", None)
    dense_em_kwargs.pop("relion_projector_r_max", None)
    return run_dense_k_class_em(
        half.experiment_dataset,
        half.means_k,
        half.mean_variance,
        half.noise_variance_k,
        sampling.effective_rotations,
        sampling.current_translations,
        sampling.disc_type,
        class_log_priors=priors.class_log_priors,
        accumulate_noise=True,
        return_best_pose_details=execution.return_best_pose_details,
        **dense_em_kwargs,
    )


def _score_adaptive_kclass_dense(
    half: DenseHalfData,
    sampling: DenseSamplingSpec,
    priors: DensePriorSpec,
    batching: DenseBatchPolicy,
    variant: DenseVariantPolicy,
    execution: DenseExecutionPolicy,
    optics: DenseOpticsSpec,
    em_kwargs,
    symmetry,
):
    """Run the ordinary adaptive K-class engine and return its trial grids."""

    adaptive_os = int(sampling.state.adaptive_oversampling)
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
    pass2_grids = _adaptive_pass2_grids(
        sampling.effective_rotations,
        sampling.current_translations,
        sampling.base_translations,
        healpix_order=sampling.current_healpix_order,
        adaptive_oversampling=adaptive_os,
        translation_step=sampling.state.translation_step,
        random_perturbation=sampling.random_perturbation,
        coarse_rotation_ids=sampling.coarse_rotation_ids,
        **({"symmetry": symmetry} if symmetry != "C1" else {}),
    )
    adaptive_em_kwargs = dict(em_kwargs)
    n_classes = (
        int(np.asarray(half.means_k).shape[0])
        if np.asarray(half.means_k).ndim >= 2
        else 1
    )
    grid_batch_plan = _plan_kclass_adaptive_grid_batch_sizes(
        coarse_rotations=pass2_grids.coarse_rotations,
        coarse_translations=pass2_grids.coarse_translations,
        fine_rotations=pass2_grids.fine_rotations,
        fine_translations=pass2_grids.fine_translations,
        n_classes=n_classes,
        image_shape=half.experiment_dataset.image_shape,
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
    sparse_pass2 = _sparse_pass2_selected("RELAX_K_CLASS_DENSE_PASS2")
    if symmetry != "C1" and not sparse_pass2 and sampling.coarse_engine != "gemm_dense":
        raise RuntimeError(f"{symmetry} requires sparse RELION x-half BPref reconstruction")
    adaptive_em_kwargs["sparse_pass2"] = sparse_pass2
    # Class3D pass 1 scores RELION's exact coarse operands; a normalized-CC
    # pass keeps the generic scorer, which the exact path leaves dormant.
    adaptive_em_kwargs["relion_exact_coarse"] = uses_relion_cuda_image_preprocessing(
        half.experiment_dataset
    )
    logger.info(
        "RELION adaptive K-class routing through run_dense_k_class_em_adaptive "
        "(oversampling=%d, pass2_backend=%s, fine_mstep_prune=%s)",
        adaptive_os,
        "sparse" if sparse_pass2 else "dense",
        bool(sparse_pass2),
    )
    common_kwargs = _adaptive_engine_common_kwargs(
        pass2_grids,
        priors,
        batching,
        sampling,
        execution,
        sparse_pass2=sparse_pass2,
        full_grid_mstep=sampling.coarse_engine == "gemm_dense",
    )
    # Images on another grid (applyScaleDifference) or magnified (applyAnisoMag): the
    # projection and backprojection matrices carry it.
    projected, grid_kwargs = engine_projection_inputs(
        half.experiment_dataset,
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
        half.experiment_dataset,
        half.means_k,
        half.mean_variance,
        half.noise_variance_k,
        projected["coarse"],
        pass2_grids.coarse_translations,
        projected["fine"],
        pass2_grids.fine_translations,
        pass2_grids.rotation_parent_map,
        pass2_grids.translation_parent_map,
        sampling.disc_type,
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


def _score_adaptive_k1_dense(
    half: DenseHalfData,
    sampling: DenseSamplingSpec,
    priors: DensePriorSpec,
    batching: DenseBatchPolicy,
    variant: DenseVariantPolicy,
    execution: DenseExecutionPolicy,
    optics: DenseOpticsSpec,
    base_em_kwargs,
    *,
    symmetry,
):
    """Run the ordinary adaptive K=1 engine and return its trial grids."""

    adaptive_os = int(sampling.state.adaptive_oversampling)
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
    means_single = jnp.asarray(half.means_k)[None, :]
    pass2_grids = _adaptive_pass2_grids(
        sampling.effective_rotations,
        sampling.current_translations,
        sampling.base_translations,
        healpix_order=sampling.current_healpix_order,
        adaptive_oversampling=adaptive_os,
        translation_step=sampling.state.translation_step,
        random_perturbation=sampling.random_perturbation,
        coarse_rotation_ids=sampling.coarse_rotation_ids,
        **({"symmetry": symmetry} if symmetry != "C1" else {}),
    )
    adaptive_em_kwargs = dict(base_em_kwargs)
    sparse_pass2 = _sparse_pass2_selected("RELAX_K1_DENSE_PASS2")
    if symmetry != "C1" and not sparse_pass2 and sampling.coarse_engine != "gemm_dense":
        raise RuntimeError(f"{symmetry} requires sparse RELION x-half BPref reconstruction")
    skip_significance_pruning = _k1_skip_significance_pruning_enabled()
    adaptive_em_kwargs["sparse_pass2"] = sparse_pass2
    # Every K=1 start scores RELION's exact coarse operands, as Class3D and
    # VDAM do; a fresh start requires the RELION CUDA preprocessing anyway.
    adaptive_em_kwargs["relion_exact_coarse"] = bool(
        execution.preserve_bpref_particle_order
        or uses_relion_cuda_image_preprocessing(half.experiment_dataset)
    )
    if half.group_ids_k is not None:
        adaptive_em_kwargs["group_ids"] = half.group_ids_k
    if relion_x_half_mstep:
        adaptive_em_kwargs["mstep_relion_x_half"] = True
    logger.info(
        "RELION adaptive K=1 routing through run_dense_k_class_em_adaptive "
        "(oversampling=%d, pass2_backend=%s, skip_significance_pruning=%s, "
        "fine_mstep_prune=%s, relion_x_half_mstep=%s, supplied_ppref=%s, "
        "engine_ppref=%s)",
        adaptive_os,
        "sparse" if sparse_pass2 else "dense",
        bool(skip_significance_pruning),
        bool(sparse_pass2),
        bool(relion_x_half_mstep),
        execution.relion_projector_half is not None,
        adaptive_em_kwargs.get("relion_projector_half") is not None,
    )
    common_kwargs = _adaptive_engine_common_kwargs(
        pass2_grids,
        priors,
        batching,
        sampling,
        execution,
        sparse_pass2=sparse_pass2,
        full_grid_mstep=sampling.coarse_engine == "gemm_dense",
    )
    projected, grid_kwargs = engine_projection_inputs(
        half.experiment_dataset,
        scale=optics.projection_scale,
        reference_current_size=optics.reference_current_size,
        rotations={
            "coarse": (
                sampling.coarse_scoring_rotations
                if sampling.coarse_scoring_rotations is not None
                and adaptive_os == 0
                and sparse_pass2
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
        half.experiment_dataset,
        means_single,
        half.mean_variance,
        half.noise_variance_k,
        projected["coarse"],
        pass2_grids.coarse_translations,
        projected["fine"],
        pass2_grids.fine_translations,
        pass2_grids.rotation_parent_map,
        pass2_grids.translation_parent_map,
        sampling.disc_type,
        skip_significance_pruning=skip_significance_pruning,
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
    half: DenseHalfData,
    sampling: DenseSamplingSpec,
    priors: DensePriorSpec,
    batching: DenseBatchPolicy,
    variant: DenseVariantPolicy,
    execution: DenseExecutionPolicy,
    optics: DenseOpticsSpec,
) -> HalfScoreResult:
    """Dense (non-local-search) E+M scoring for one half-set.

    The signature exposes per-half data, sampling, priors, batching, route
    selection, execution controls and optics adaptations as cohesive owners.
    Stable fields are read through those owners; only values changed by route
    planning become local variables.

    ``half.optics_group_ids_k`` gives each image's row of a
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
    differ in five places, all controlled by explicit policy fields:

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
    5. ``firstiter_updates_em_kwargs_ibs`` — adaptive overrides
       em_kwargs["image_batch_size"] with the firstiter clamp; single-pass
       leaves em_kwargs untouched.

    Stores K-class summaries and explicit best poses in ``half.outputs``. The
    caller records the common payload from the returned ``HalfScoreResult``.
    ``batching.safe_batch_sizes`` is the closure-bound batch sizer from
    ``refine_single_volume``.
    """

    # These values are refined by route-specific planning below. All other
    # stable values retain their owning specification object.
    firstiter_coarse_current_size = variant.firstiter_coarse_current_size
    firstiter_fine_current_size = variant.firstiter_fine_current_size

    from relax.symmetry import canonicalize_rotational_symmetry

    symmetry = canonicalize_rotational_symmetry(sampling.symmetry)
    if symmetry != "C1" and int(sampling.state.adaptive_oversampling) <= 0 and sampling.coarse_engine != "gemm_dense":
        raise NotImplementedError(
            f"{symmetry} non-adaptive dense reconstruction is unsupported; "
            "use the adaptive sparse or exact-local RELION x-half M-step"
        )
    safe_ibs, safe_rbs = batching.safe_batch_sizes(
        sampling.effective_rotations.shape[0],
        sampling.current_translations.shape[0],
        current_size_for_batch=sampling.cs_for_engine,
    )
    em_kwargs = {
        **_DENSE_EM_STATIC_KWARGS,
        "image_batch_size": safe_ibs,
        "rotation_block_size": safe_rbs,
        "current_size": sampling.cs_for_engine,
        "rotation_log_prior": priors.rotation_log_prior_k,
        "translation_log_prior": priors.translation_log_prior,
        "image_corrections": half.image_corrections_k,
        "scale_corrections": half.scale_corrections_k,
        "group_ids": half.group_ids_k,
        "scale_correction_group_count": half.group_count_k,
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
    if half.optics_group_ids_k is not None:
        em_kwargs["optics_group_ids"] = half.optics_group_ids_k
    if sampling.model_current_size_for_engine is not None:
        em_kwargs["reconstruction_current_size"] = sampling.model_current_size_for_engine
    if execution.preserve_bpref_particle_order and variant.k_class_enabled:
        raise ValueError("RELION BPref particle-order preservation is K=1-only")
    if half.image_seed_classes is not None and (
        not variant.k_class_enabled
        or not (
            variant.relion_firstiter_cc_this_iter
            or _dense_uses_adaptive_engine(sampling.state.adaptive_oversampling, half.group_ids_k)
        )
    ):
        raise NotImplementedError("a seed iteration runs on the adaptive or first-iteration CC K-class route")
    if execution.preserve_bpref_particle_order:
        em_kwargs["preserve_bpref_particle_order"] = True
    if execution.source_faithful_spectrum_norm:
        em_kwargs["source_faithful_spectrum_norm"] = True
    if execution.firstiter_cc_tree_rescore_max_margin is not None:
        em_kwargs["firstiter_cc_tree_rescore_max_margin"] = float(execution.firstiter_cc_tree_rescore_max_margin)
    if float(execution.relion_translation_angle_scale) != 1.0:
        if variant.k_class_enabled:
            raise ValueError("the RELION model/optics translation-angle scale is K=1-only")
        em_kwargs["relion_translation_angle_scale"] = float(execution.relion_translation_angle_scale)
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
    if execution.relion_projector_half is not None:
        em_kwargs["relion_projector_half"] = execution.relion_projector_half
        em_kwargs["relion_projector_r_max"] = execution.relion_projector_r_max
    logger.info(
        "Dense half-set projector handoff: supplied_ppref=%s state_oversampling=%d",
        execution.relion_projector_half is not None,
        int(sampling.state.adaptive_oversampling),
    )
    if variant.relion_firstiter_cc_this_iter:
        # Shared first-iteration inputs; means, layouts and pose IDs remain route-specific.
        firstiter_data = FirstIterCCData(
                logger=logger,
                experiment_dataset=half.experiment_dataset,
                mean=half.means_k,
                mean_variance=half.mean_variance,
                noise_variance=half.noise_variance_k,
                image_shape=half.experiment_dataset.image_shape,
                image_seed_classes=half.image_seed_classes,
        )
        firstiter_grid = FirstIterCCGridSpec(
                effective_rotations=sampling.effective_rotations,
                current_translations=sampling.current_translations,
                base_translations=sampling.base_translations,
                current_healpix_order=sampling.current_healpix_order,
                state=sampling.state,
                random_perturbation=sampling.random_perturbation,
                symmetry=symmetry,
        )
        firstiter_policy = FirstIterCCPolicy(
                disc_type=sampling.disc_type,
                class_log_priors=priors.class_log_priors,
        )
        firstiter_batching = FirstIterCCBatching(
                image_batch_size=batching.image_batch_size,
                em_kwargs=em_kwargs,
                safe_batch_sizes=batching.safe_batch_sizes,
                significance_safe_batch_sizes=batching.significance_safe_batch_sizes,
                coarse_current_size=firstiter_coarse_current_size,
                fine_current_size=firstiter_fine_current_size,
        )
        firstiter_execution = FirstIterCCExecution(
                log_label=variant.firstiter_log_label,
                update_em_kwargs_image_batch_size=variant.firstiter_updates_em_kwargs_ibs,
                bpref_device_signature_active=execution.bpref_device_signature_active,
                debug_iteration=execution.debug_iteration,
        )

    if variant.k_class_enabled:
        if execution.disable_adjoint_y or execution.disable_adjoint_ctf:
            raise NotImplementedError("K-class refine does not support adjoint ablation flags")
        magnification = dataset_projection_magnification(half.experiment_dataset)
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
        fine_rotations_for_pose = None
        adaptive_os_local = 0
        # STRICT-PARITY: at iter 1 with --firstiter_cc, route through the
        # adaptive 2-pass engine with normalized-CC scoring. Pass 2 retains the
        # oversampled children of the single best coarse class/pose, matching
        # RELION's firstiter-CC binarized coarse support.
        if variant.relion_firstiter_cc_this_iter:
            (
                k_class_result,
                rot_pmap_for_collapse,
                trans_pmap_for_collapse,
                n_trans_fine_for_collapse,
                adaptive_os_local,
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
        elif sampling.coarse_engine != "auto" or _dense_uses_adaptive_engine(sampling.state.adaptive_oversampling, half.group_ids_k):
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
            adaptive_os_local = int(sampling.state.adaptive_oversampling)
            rot_pmap_for_collapse = pass2_grids.rotation_parent_map
            trans_pmap_for_collapse = pass2_grids.translation_parent_map
            n_trans_fine_for_collapse = pass2_grids.n_fine_translations
            k_class_mstep_full_half_axis_this_score = k_class_result.mstep_full_half_axis
        else:
            k_class_result = _score_direct_kclass_dense(
                half,
                sampling,
                priors,
                execution,
                optics,
                em_kwargs,
            )
            k_class_mstep_full_half_axis_this_score = None
        if optics.projection_scale != 1.0 or magnification is not None:
            # Poses are reported unscaled and unmagnified; only projection used the transformed matrices.
            def unmagnified(rotations):
                return None if rotations is None else reported_rotations(rotations, optics.projection_scale, magnification)

            per_class = k_class_result.per_class_best_pose_rotations
            k_class_result = k_class_result._replace(
                best_pose_rotations=unmagnified(k_class_result.best_pose_rotations),
                per_class_best_pose_rotations=None if per_class is None else tuple(map(unmagnified, per_class)),
            )
        ha_k, Ft_y_k, Ft_ctf_k, em_stats_k, noise_stats_k = _scatter_dense_k_class_result(
            k_class_result,
            k=half.k,
            effective_rotations=sampling.effective_rotations,
            rot_pmap_for_collapse=rot_pmap_for_collapse,
            adaptive_os_local=adaptive_os_local,
            outputs=half.outputs,
            require_best_pose_details=execution.return_best_pose_details,
            pose_dtype=_dense_global_scoring_dtype(),
        )
        coarse_ha_k = _coarse_pose_assignments(
            ha_k,
            rot_parent_map=rot_pmap_for_collapse,
            trans_parent_map=trans_pmap_for_collapse,
            n_trans_coarse=sampling.current_translations.shape[0],
            n_trans_fine=n_trans_fine_for_collapse,
        )
        return HalfScoreResult(
            ha=ha_k,
            Ft_y=Ft_y_k,
            Ft_ctf=Ft_ctf_k,
            em_stats=em_stats_k,
            noise_stats=noise_stats_k,
            coarse_ha=coarse_ha_k,
            significant_counts=(
                None
                if k_class_result.significant_counts is None
                else np.asarray(k_class_result.significant_counts, dtype=np.int32)
            ),
            profile_summary=k_class_result.profile_summary,
            mstep_full_half_axis=k_class_mstep_full_half_axis_this_score,
            mstep_accumulator_shape=getattr(k_class_result, "mstep_accumulator_shape", None),
        )

    if sampling.coarse_engine != "auto" or execution.preserve_bpref_particle_order or _dense_uses_adaptive_engine(
        sampling.state.adaptive_oversampling, half.group_ids_k
    ):
        if execution.disable_adjoint_y or execution.disable_adjoint_ctf:
            raise NotImplementedError("K=1 adaptive oversampling does not support adjoint ablation flags")
        adaptive_os_local = int(sampling.state.adaptive_oversampling)
        rot_pmap_for_collapse = None
        trans_pmap_for_collapse = None
        n_trans_fine_for_collapse = None
        fine_rotations_for_pose = None
        if variant.relion_firstiter_cc_this_iter:
            if adaptive_os_local <= 0:
                # The sparse engine supplies group statistics and per-particle
                # BPref launches even for a single pass on the current grid.
                firstiter_coarse_current_size = sampling.cs_for_engine
                firstiter_fine_current_size = sampling.cs_for_engine
                logger.info(
                    "RELION K=1 group statistics or BPref order at oversampling 0: single pass through the "
                    "adaptive engine (current_size=%s)",
                    sampling.cs_for_engine,
                )
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
                adaptive_os_local,
            ) = _score_kclass_firstiter_cc_pass2(
                replace(firstiter_data, mean=jnp.asarray(half.means_k)[None, :]),
                replace(
                    firstiter_grid,
                    # Images on another grid (applyScaleDifference) or magnified (applyAnisoMag).
                    projection_rotations=lambda rotations: projection_rotations(
                        rotations,
                        optics.projection_scale,
                        dataset_projection_magnification(half.experiment_dataset),
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
        coarse_ha_k = _coarse_pose_assignments(
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
            magnification = dataset_projection_magnification(half.experiment_dataset)
            if optics.projection_scale != 1.0 or magnification is not None:
                # Poses are reported unscaled and unmagnified; only projection used the transformed matrices.
                best_rots = np.asarray(
                    reported_rotations(best_rots, optics.projection_scale, magnification), dtype=pose_dtype
                )
            half.outputs.best_pose_rotations[half.k] = best_rots
            half.outputs.best_pose_rotation_eulers[half.k] = (
                np.asarray(k1_adaptive_result.best_pose_eulers_deg, dtype=np.float64)
                if k1_adaptive_result.best_pose_eulers_deg is not None
                else utils.R_to_relion(best_rots, degrees=True).astype(pose_dtype)
            )
            half.outputs.best_pose_translations[half.k] = np.asarray(
                k1_adaptive_result.best_pose_translations, dtype=pose_dtype
            )
        if fine_rotations_for_pose is None and rot_pmap_for_collapse is not None:
            fine_rotations_for_pose = _adaptive_pass2_grids(
                sampling.effective_rotations,
                sampling.current_translations,
                sampling.base_translations,
                healpix_order=sampling.current_healpix_order,
                adaptive_oversampling=adaptive_os_local,
                translation_step=sampling.state.translation_step,
                random_perturbation=sampling.random_perturbation,
                coarse_rotation_ids=sampling.coarse_rotation_ids,
                **({"symmetry": symmetry} if symmetry != "C1" else {}),
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
            best_pose_rotations=half.outputs.best_pose_rotations[half.k],
            best_pose_rotation_eulers=half.outputs.best_pose_rotation_eulers[half.k],
            best_pose_translations=half.outputs.best_pose_translations[half.k],
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

    return _score_direct_k1_dense(half, sampling, execution, em_kwargs)


def _dense_owners_for_shape(
    half: DenseHalfData,
    sampling: DenseSamplingSpec,
    priors: DensePriorSpec,
    batching: DenseBatchPolicy,
    variant: DenseVariantPolicy,
    execution: DenseExecutionPolicy,
    optics: DenseOpticsSpec,
    shape_class,
    class_index: int,
) -> tuple:
    """Derive one shape class without changing the shared scoring owners."""

    from relax.refinement import optics_shapes

    shape_values = optics_shapes.class_kwargs(
        {
            "experiment_dataset": half.experiment_dataset,
            "image_corrections_k": half.image_corrections_k,
            "scale_corrections_k": half.scale_corrections_k,
            "group_ids_k": half.group_ids_k,
            "optics_group_ids_k": half.optics_group_ids_k,
            "image_seed_classes": half.image_seed_classes,
            "rotation_log_prior_k": priors.rotation_log_prior_k,
            "class_rotation_log_prior_k": priors.class_rotation_log_prior_k,
            "translation_log_prior": priors.translation_log_prior,
            "translation_search_base": priors.translation_search_base,
            "trans_prior_center_for_engine": priors.trans_prior_center_for_engine,
            "current_translations": sampling.current_translations,
            "base_translations": sampling.base_translations,
            "state": sampling.state,
            "cs_for_engine": sampling.cs_for_engine,
            "model_current_size_for_engine": sampling.model_current_size_for_engine,
            "firstiter_coarse_current_size": variant.firstiter_coarse_current_size,
            "firstiter_fine_current_size": variant.firstiter_fine_current_size,
            "coarse_sizing": optics.coarse_sizing,
        },
        shape_class,
        half.experiment_dataset.n_units,
    )
    if optics.class_translation_overrides is not None:
        shape_values.update(
            {
                key: value
                for key, value in optics.class_translation_overrides[class_index].items()
                if key in shape_values
            }
        )

    batch_overrides = {} if optics.class_batch_overrides is None else optics.class_batch_overrides[class_index]
    return (
        replace(
            half,
            experiment_dataset=shape_values["experiment_dataset"],
            noise_variance_k=optics_shapes.class_noise_table(
                optics.noise_radial_k,
                shape_class,
                int(half.experiment_dataset.image_shape[0]),
            ),
            image_corrections_k=shape_values["image_corrections_k"],
            scale_corrections_k=shape_values["scale_corrections_k"],
            group_ids_k=shape_values["group_ids_k"],
            optics_group_ids_k=shape_values["optics_group_ids_k"],
            image_seed_classes=shape_values["image_seed_classes"],
            outputs=PerHalfOutputs(),
        ),
        replace(
            sampling,
            current_translations=shape_values["current_translations"],
            base_translations=shape_values["base_translations"],
            state=shape_values["state"],
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
        replace(batching, **batch_overrides),
        replace(
            variant,
            firstiter_coarse_current_size=shape_values["firstiter_coarse_current_size"],
            firstiter_fine_current_size=shape_values["firstiter_fine_current_size"],
        ),
        execution,
        replace(
            optics,
            noise_radial_k=None,
            class_batch_overrides=None,
            class_translation_overrides=None,
            projection_scale=shape_values["projection_scale"],
            reference_current_size=shape_values["reference_current_size"],
        ),
    )


def _score_half_dense(
    half: DenseHalfData,
    sampling: DenseSamplingSpec,
    priors: DensePriorSpec,
    batching: DenseBatchPolicy,
    variant: DenseVariantPolicy,
    execution: DenseExecutionPolicy,
    optics: DenseOpticsSpec,
) -> HalfScoreResult:
    """Dense E+M scoring for one half; several image shapes run per shape class."""

    from relax.refinement import optics_shapes

    experiment_half = half.experiment_dataset
    if not isinstance(experiment_half, optics_shapes.MultiShapeHalf):
        return _score_half_dense_one_shape(half, sampling, priors, batching, variant, execution, optics)
    if optics.noise_radial_k is None:
        raise ValueError("a half with several image shapes needs reference-shell noise spectra")
    if half.optics_group_ids_k is None:
        raise ValueError("a half with several image shapes needs each image's optics group")
    if optics.class_batch_overrides is not None and len(optics.class_batch_overrides) != len(experiment_half.classes):
        raise ValueError("class_batch_overrides needs one entry per shape class")

    owners = [
        _dense_owners_for_shape(
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
        for index, shape_class in enumerate(experiment_half.classes)
    ]
    results = [_score_half_dense_one_shape(*class_owners) for class_owners in owners]
    merged = optics_shapes.merge_class_results(
        results,
        experiment_half.classes,
        experiment_half.n_units,
        int(experiment_half.image_shape[0]),
    )
    if variant.k_class_enabled:
        # The K-class routes report their summaries and best poses in the outputs only.
        optics_shapes.merge_k_class_outputs(
            half.outputs,
            half.k,
            [class_owners[0].outputs for class_owners in owners],
            experiment_half.classes,
            experiment_half.n_units,
            int(experiment_half.image_shape[0]),
        )
        return merged
    half.outputs.best_pose_rotations[half.k] = merged.best_pose_rotations
    half.outputs.best_pose_rotation_eulers[half.k] = merged.best_pose_rotation_eulers
    half.outputs.best_pose_translations[half.k] = merged.best_pose_translations
    return merged


@dataclass(frozen=True, kw_only=True)
class LocalHalfData:
    """Per-half arrays and caller-owned outputs for exact-local scoring."""

    k: int
    experiment_dataset: object
    means_k: object
    noise_variance_k: object
    previous_best_rotation_eulers_k: object
    image_corrections_k: object
    scale_corrections_k: object
    outputs: PerHalfOutputs
    group_ids_k: object | None = None
    group_count_k: object | None = None
    scale_correction_data_vs_prior: object | None = None
    optics_group_ids_k: object | None = None


@dataclass(frozen=True, kw_only=True)
class LocalSamplingSpec:
    """Local orientation/translation grids and reconstruction sizes."""

    local_search_rotations: object
    local_search_order: int
    sigma_rot: object
    sigma_psi: object
    current_translations: object
    base_translations: object
    disc_type: str
    cs_for_engine: int | None
    local_pass1_current_size: int | None
    local_search_random_perturbation: float
    local_search_angular_sampling_deg: float
    local_parent_oversampling_order: int
    local_search_mstep_rotations: object | None = None
    model_current_size_for_engine: int | None = None
    symmetry: str = "C1"


@dataclass(frozen=True, kw_only=True)
class LocalPriorSpec:
    """Translation-prior operands and their grid-selection policy."""

    trans_prior_center: object
    trans_prior_center_for_engine: object
    current_sigma_offset_angstrom: float
    translation_search_base: object
    local_search_translation_prior_mode: str
    replay_prior_translations: object


@dataclass(frozen=True, kw_only=True)
class LocalBatchPolicy:
    """Exact-local support and batch planning controls."""

    max_significants: int | None
    safe_batch_sizes: object


@dataclass(frozen=True, kw_only=True)
class LocalExecutionPolicy:
    """Projector, reconstruction and source-compatibility controls."""

    disable_adjoint_y: bool
    disable_adjoint_ctf: bool
    relion_projector_half: object | None = None
    relion_projector_r_max: int | None = None
    source_faithful_spectrum_norm: bool = False
    relion_translation_angle_scale: float = 1.0


@dataclass(frozen=True, kw_only=True)
class LocalDiagnosticPolicy:
    """Profiling, debug capture and score-only controls."""

    iteration: int
    save_intermediates_dir: object
    collect_local_search_profile: bool
    diagnostic_score_only: bool
    local_profile_history: object
    debug_iteration: int | None = None
    bpref_device_signature_active: bool = False
    parent_use_float64_scoring: bool = False
    parent_use_float64_projections: bool = False
    fine_use_float64_scoring: bool = False
    fine_use_float64_projections: bool = False
    adaptive_pass2_full_parent: bool = False
    adaptive_pass2_rotation_only: bool = False
    adaptive_pass2_denominator_mode: str | None = None


@dataclass(frozen=True, kw_only=True)
class LocalOpticsSpec:
    """Multi-shape adaptations, absent for an ordinary single-shape half."""

    noise_radial_k: object | None = None
    coarse_sizing: tuple[float, float | None] | None = None
    class_translation_overrides: tuple[dict, ...] | None = None
    projection_scale: float = 1.0
    reference_current_size: int | None = None


def _score_half_dense_in_bpref_scope(
    half: DenseHalfData,
    sampling: DenseSamplingSpec,
    priors: DensePriorSpec,
    batching: DenseBatchPolicy,
    variant: DenseVariantPolicy,
    execution: DenseExecutionPolicy,
    optics: DenseOpticsSpec,
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

    current = np.asarray(current_translations, dtype=dtype)
    base = np.asarray(base_translations, dtype=dtype)
    if replay_prior_translations is not None:
        candidate = np.asarray(replay_prior_translations, dtype=dtype)
        source = "replay"
    else:
        candidate = base
        source = "base"

    if candidate.shape == current.shape:
        return candidate, source, False
    if base.shape == current.shape:
        return base, "base", True
    return current, "current", True


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
    sampling: LocalSamplingSpec,
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
            int(sampling.current_translations.shape[0]),
        )
        parent_mode = "significant_rotation_full_translation"
        logger.info(
            "RELION local adaptive pass 2 diagnostic: expanding significant parent rotations to all "
            "parent translations via %s=1",
            _LOCAL_ADAPTIVE_PASS2_ROTATION_ONLY_ENV,
        )

    layout_kwargs = dict(
        oversampling_order=int(sampling.local_parent_oversampling_order),
        random_perturbation=float(sampling.local_search_random_perturbation),
        dtype=fine_layout_dtype,
        **({"symmetry": sampling.symmetry} if sampling.symmetry != "C1" else {}),
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
        elif denominator_mode == "rotation_only":
            denominator_significant_sample_indices = (
                _expand_significant_samples_to_full_parent_translations(
                    pruned_parent_significant_sample_indices,
                    int(sampling.current_translations.shape[0]),
                )
            )
        else:  # Defensive only; parser restricts values.
            raise AssertionError(f"unexpected denominator mode {denominator_mode!r}")
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
        sampling.current_translations,
        pass2_layout,
    )
    return pass2_layout, relion_significant_counts, denominator_layout, parent_mode


def _build_local_adaptive_parent_layout(
    half: LocalHalfData,
    sampling: LocalSamplingSpec,
    priors: LocalPriorSpec,
    translation_prior_reference_translations,
    layout_dtype,
):
    """Build the coarse parent layout used by exact-local adaptive pass 1."""

    parent_prior_translations = priors.trans_prior_center
    if parent_prior_translations is None:
        parent_prior_translations = np.zeros(
            (
                np.asarray(half.previous_best_rotation_eulers_k).shape[0],
                np.asarray(sampling.current_translations).shape[1],
            ),
            dtype=layout_dtype,
        )
    parent_order = int(sampling.local_search_order) - int(
        sampling.local_parent_oversampling_order
    )
    if parent_order < 0:
        raise ValueError(
            "local_search_order must be >= local_parent_oversampling_order; "
            f"got {sampling.local_search_order} and "
            f"{sampling.local_parent_oversampling_order}",
        )
    parent_grid_metadata = build_local_search_grid_metadata(
        parent_order,
        **({"symmetry": sampling.symmetry} if sampling.symmetry != "C1" else {}),
    )
    parent_layout = build_local_hypothesis_layout(
        half.previous_best_rotation_eulers_k,
        None,
        sampling.sigma_rot,
        sampling.sigma_psi,
        parent_order,
        sampling.current_translations,
        parent_prior_translations,
        priors.current_sigma_offset_angstrom,
        None,
        half.experiment_dataset.voxel_size,
        grid_metadata=parent_grid_metadata,
        translation_prior_reference_translations=translation_prior_reference_translations,
        rotation_log_prior=None,
        rotation_grid_random_perturbation=sampling.local_search_random_perturbation,
        rotation_grid_angular_sampling_deg=relion_angular_sampling_deg(
            parent_order,
            adaptive_oversampling=0,
        ),
        dtype=layout_dtype,
    )
    return parent_layout, parent_order


def _local_owners_for_shape(
    half: LocalHalfData,
    sampling: LocalSamplingSpec,
    priors: LocalPriorSpec,
    batching: LocalBatchPolicy,
    execution: LocalExecutionPolicy,
    diagnostics: LocalDiagnosticPolicy,
    optics: LocalOpticsSpec,
    shape_class,
    class_index: int,
) -> tuple:
    """Derive one exact-local shape class through the optics owner."""

    from relax.refinement import optics_shapes

    shape_values = optics_shapes.class_kwargs(
        {
            "experiment_dataset": half.experiment_dataset,
            "previous_best_rotation_eulers_k": half.previous_best_rotation_eulers_k,
            "image_corrections_k": half.image_corrections_k,
            "scale_corrections_k": half.scale_corrections_k,
            "group_ids_k": half.group_ids_k,
            "optics_group_ids_k": half.optics_group_ids_k,
            "current_translations": sampling.current_translations,
            "base_translations": sampling.base_translations,
            "cs_for_engine": sampling.cs_for_engine,
            "model_current_size_for_engine": sampling.model_current_size_for_engine,
            "local_pass1_current_size": sampling.local_pass1_current_size,
            "trans_prior_center": priors.trans_prior_center,
            "trans_prior_center_for_engine": priors.trans_prior_center_for_engine,
            "translation_search_base": priors.translation_search_base,
            "replay_prior_translations": priors.replay_prior_translations,
            "coarse_sizing": optics.coarse_sizing,
        },
        shape_class,
        half.experiment_dataset.n_units,
    )
    if optics.class_translation_overrides is not None:
        shape_values.update(
            {
                key: value
                for key, value in optics.class_translation_overrides[class_index].items()
                if key in shape_values
            }
        )
    return (
        replace(
            half,
            experiment_dataset=shape_values["experiment_dataset"],
            noise_variance_k=optics_shapes.class_noise_table(
                optics.noise_radial_k,
                shape_class,
                int(half.experiment_dataset.image_shape[0]),
            ),
            previous_best_rotation_eulers_k=shape_values["previous_best_rotation_eulers_k"],
            image_corrections_k=shape_values["image_corrections_k"],
            scale_corrections_k=shape_values["scale_corrections_k"],
            group_ids_k=shape_values["group_ids_k"],
            optics_group_ids_k=shape_values["optics_group_ids_k"],
            outputs=PerHalfOutputs(),
        ),
        replace(
            sampling,
            current_translations=shape_values["current_translations"],
            base_translations=shape_values["base_translations"],
            cs_for_engine=shape_values["cs_for_engine"],
            model_current_size_for_engine=shape_values["model_current_size_for_engine"],
            local_pass1_current_size=shape_values["local_pass1_current_size"],
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
            class_translation_overrides=None,
            projection_scale=shape_values["projection_scale"],
            reference_current_size=shape_values["reference_current_size"],
        ),
    )


def _score_half_local(
    half: LocalHalfData,
    sampling: LocalSamplingSpec,
    priors: LocalPriorSpec,
    batching: LocalBatchPolicy,
    execution: LocalExecutionPolicy,
    diagnostics: LocalDiagnosticPolicy,
    optics: LocalOpticsSpec,
) -> HalfScoreResult:
    """Exact-local scoring for one half; several shapes run per shape class."""

    from relax.refinement import optics_shapes

    experiment_half = half.experiment_dataset
    if not isinstance(experiment_half, optics_shapes.MultiShapeHalf):
        return _score_half_local_one_shape(half, sampling, priors, batching, execution, diagnostics, optics)
    if optics.noise_radial_k is None:
        raise ValueError("a half with several image shapes needs reference-shell noise spectra")
    if half.optics_group_ids_k is None:
        raise ValueError("a half with several image shapes needs each image's optics group")
    optics_shapes.require_exact_local_parent_windows(
        {
            "experiment_dataset": experiment_half,
            "cs_for_engine": sampling.cs_for_engine,
            "local_pass1_current_size": sampling.local_pass1_current_size,
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
    merged = optics_shapes.merge_class_results(
        results,
        experiment_half.classes,
        experiment_half.n_units,
        int(experiment_half.image_shape[0]),
    )
    half.outputs.best_pose_rotations[half.k] = merged.best_pose_rotations
    half.outputs.best_pose_rotation_eulers[half.k] = merged.best_pose_rotation_eulers
    half.outputs.best_pose_translations[half.k] = merged.best_pose_translations
    return merged


def _score_half_local_one_shape(
    half: LocalHalfData,
    sampling: LocalSamplingSpec,
    priors: LocalPriorSpec,
    batching: LocalBatchPolicy,
    execution: LocalExecutionPolicy,
    diagnostics: LocalDiagnosticPolicy,
    optics: LocalOpticsSpec,
) -> HalfScoreResult:
    """Local-search E+M scoring for one half-set.

    Stable inputs remain under their data, sampling, prior, batching,
    execution, diagnostic and optics owners. Only route-derived values become
    locals inside this function.

    Sizes the per-chunk M-step batches against the cone-restricted
    rotation count (not the full HEALPix grid) so chunk_size doesn't
    collapse at high HEALPix orders. Routes through
    ``_run_local_search_iteration``. Local searches are K=1 only: Class3D keeps
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
        sampling.cs_for_engine
        if sampling.model_current_size_for_engine is None
        else sampling.model_current_size_for_engine
    )

    # For local search the per-chunk M-step only sees the cone-restricted
    # rotation set (typically a few thousand rotations per image with high
    # overlap across the chunk) rather than the full ~10⁶-rotation grid at
    # healpix order 5+. Estimate per-image cone size from
    #     fraction = (sigma_cutoff * sigma_rot / pi)^2
    # (spherical cap area as a fraction of full SO(3) volume; good to
    # within ~30% for reasonable cones). Use that for an effective rotation
    # count equal to ``chunk_size * cone_size`` with a 2x safety factor.
    cone_radius = 3.0 * float(sampling.sigma_rot)  # sigma_cutoff=3.0
    cone_fraction = max(
        (cone_radius / float(np.pi)) ** 2,
        1.0 / float(rotation_grid_size(sampling.local_search_order)),
    )
    est_cone_rots = int(np.ceil(rotation_grid_size(sampling.local_search_order) * cone_fraction))
    eff_n_rot = max(64, 2 * est_cone_rots)
    local_n_trans = int(sampling.current_translations.shape[0])
    if int(sampling.local_parent_oversampling_order) > 0:
        local_n_trans *= int(4 ** int(sampling.local_parent_oversampling_order))
    local_debug_iteration = (
        diagnostics.iteration + 1 if diagnostics.debug_iteration is None else int(diagnostics.debug_iteration)
    )
    parent_use_float64_scoring = diagnostics.parent_use_float64_scoring
    parent_use_float64_projections = diagnostics.parent_use_float64_projections
    fine_use_float64_scoring = diagnostics.fine_use_float64_scoring
    fine_use_float64_projections = diagnostics.fine_use_float64_projections
    # Adaptive pass-2 (fine, oversampled) hypothesis layout precision; see
    # ``parent_local_layout_dtype`` below for the matching pass-1 value.
    fine_local_layout_dtype = np.float64 if (fine_use_float64_scoring or fine_use_float64_projections) else np.float32
    if fine_use_float64_scoring or fine_use_float64_projections:
        logger.info(
            "Local-search precision iteration %d: pass1 scoring/projections=%s/%s pass2 scoring/projections=%s/%s",
            local_debug_iteration,
            parent_use_float64_scoring,
            parent_use_float64_projections,
            fine_use_float64_scoring,
            fine_use_float64_projections,
        )

    safe_ibs, safe_rbs = batching.safe_batch_sizes(
        eff_n_rot,
        local_n_trans,
        image_shape_for_batch=half.experiment_dataset.image_shape,
        current_size_for_batch=sampling.cs_for_engine,
    )
    logger.info(
        "Local search batch sizing: cone_radius=%.3f rad (%.2f deg), est_cone_rots=%d, eff_n_rot=%d "
        "n_trans=%d → image_batch_size=%d, rotation_block_size=%d",
        cone_radius,
        np.rad2deg(cone_radius),
        est_cone_rots,
        eff_n_rot,
        local_n_trans,
        safe_ibs,
        safe_rbs,
    )
    # Keep the local-search hypothesis grid genuinely double precision end to
    # end when either flag requests it; default stays float32 to match
    # RELION's accelerated-GPU precision.
    parent_local_layout_dtype = (
        np.float64 if (parent_use_float64_scoring or parent_use_float64_projections) else np.float32
    )
    translation_prior_reference_translations = np.asarray(
        sampling.current_translations, dtype=parent_local_layout_dtype
    )
    if priors.local_search_translation_prior_mode == "coarse":
        translation_prior_reference_translations, prior_grid_source, prior_grid_shape_mismatch = (
            _local_translation_prior_reference_translations(
                current_translations=sampling.current_translations,
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
                np.asarray(sampling.current_translations).shape,
            )
        logger.info(
            "RELION mode: local translation prior uses coarse %s grid (n=%d) while scoring perturbed translations",
            prior_grid_source,
            translation_prior_reference_translations.shape[0],
        )
    if int(sampling.local_parent_oversampling_order) > 0:
        logger.info(
            "RELION local search: expanding translations by oversampling_order=%d (coarse n=%d -> fine n=%d)",
            int(sampling.local_parent_oversampling_order),
            int(sampling.current_translations.shape[0]),
            int(local_n_trans),
        )
    # Shared owners for one typed pass. Parent, denominator and final execution
    # derive their intentional differences with ``replace`` below.
    local_data = LocalSearchData(
            experiment_dataset=half.experiment_dataset,
            mean=half.means_k,
            noise_variance=half.noise_variance_k,
            image_corrections=half.image_corrections_k,
            scale_corrections=half.scale_corrections_k,
            group_ids=half.group_ids_k,
            scale_correction_group_count=half.group_count_k,
            scale_correction_data_vs_prior=half.scale_correction_data_vs_prior,
            image_pre_shifts=priors.translation_search_base,
            optics_group_ids=half.optics_group_ids_k,
    )
    local_grid = LocalSearchGridSpec(
            prior_rotations=half.previous_best_rotation_eulers_k,
            rotation_grid_rotations=sampling.local_search_rotations,
            healpix_order=sampling.local_search_order,
            sigma_rot=sampling.sigma_rot,
            sigma_psi=sampling.sigma_psi,
            translations=sampling.current_translations,
            prior_translations=priors.trans_prior_center,
            sigma_offset_angstrom=priors.current_sigma_offset_angstrom,
            translation_prior_reference_translations=translation_prior_reference_translations,
            translation_prior_centers=priors.trans_prior_center_for_engine,
            rotation_grid_random_perturbation=sampling.local_search_random_perturbation,
            rotation_grid_angular_sampling_deg=sampling.local_search_angular_sampling_deg,
            local_parent_oversampling_order=sampling.local_parent_oversampling_order,
            rotation_grid_mstep_rotations=sampling.local_search_mstep_rotations,
            generate_relion_mstep_rotations=True,
            symmetry=sampling.symmetry,
    )
    local_batching = LocalSearchBatchPolicy(
            image_batch_size=safe_ibs,
            rotation_block_size=safe_rbs,
            batch_size_planner=batching.safe_batch_sizes,
    )
    local_kernel = LocalSearchKernelPolicy(
            disc_type=sampling.disc_type,
            current_size=sampling.cs_for_engine,
            reconstruction_current_size=reconstruction_current_size_for_engine,
            projection_padding_factor=PROJECTION_PADDING_FACTOR,
            reconstruction_padding_factor=PADDING_FACTOR,
            do_gridding_correction=True,
            square_window=RELION_FOURIER_WINDOW_SQUARE,
            half_spectrum_scoring=True,
            relion_projector_half=execution.relion_projector_half,
            relion_projector_r_max=execution.relion_projector_r_max,
            source_faithful_spectrum_norm=execution.source_faithful_spectrum_norm,
            relion_translation_angle_scale=float(execution.relion_translation_angle_scale),
            projection_scale=float(optics.projection_scale),
            reconstruction_volume_current_size=optics.reference_current_size,
            reconstruction_image_radius=reconstruction_image_radius(
                optics.reference_current_size,
                optics.projection_scale,
            ),
    )
    local_support = LocalSearchSupportPolicy(
            disable_adjoint_y=execution.disable_adjoint_y,
            disable_adjoint_ctf=execution.disable_adjoint_ctf,
            adaptive_fraction=RELION_ADAPTIVE_FRACTION,
            max_significants=batching.max_significants,
    )
    local_diagnostics = LocalSearchDiagnosticPolicy(
            return_profile=diagnostics.collect_local_search_profile,
            debug_iteration=local_debug_iteration,
            debug_pass_label="pass2_final",
    )
    pass2_layout = None
    relion_significant_counts_k = None
    local_adaptive_pass2_parent_mode = "none"
    local_adaptive_pass2_denominator_layout = None
    local_normalization_log_evidence = None
    if int(sampling.local_parent_oversampling_order) > 0:
        parent_layout, parent_order = _build_local_adaptive_parent_layout(
            half,
            sampling,
            priors,
            translation_prior_reference_translations,
            parent_local_layout_dtype,
        )
        parent_local_rot_max = (
            int(np.max(np.asarray(parent_layout.rotation_counts, dtype=np.int64)))
            if int(np.asarray(parent_layout.rotation_counts).size)
            else 1
        )
        parent_ibs, parent_rbs = batching.safe_batch_sizes(
            max(64, parent_local_rot_max),
            int(sampling.current_translations.shape[0]),
        )
        logger.info(
            "RELION local adaptive pass 1: parent_order=%d local_rot_max=%d n_trans=%d current_size=%s",
            parent_order,
            parent_local_rot_max,
            int(sampling.current_translations.shape[0]),
            sampling.local_pass1_current_size,
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
            replace(local_batching, image_batch_size=parent_ibs, rotation_block_size=parent_rbs),
            replace(
                local_kernel,
                current_size=sampling.local_pass1_current_size,
                reconstruction_current_size=None,
                use_float64_scoring=parent_use_float64_scoring,
                use_float64_projections=parent_use_float64_projections,
                relion_exact_score_translation=bool(
                    _DENSE_EM_STATIC_KWARGS["relion_exact_fine_gaussian"] and not parent_use_float64_scoring
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
            ),
            replace(local_diagnostics, return_profile=True, debug_pass_label="pass1_parent"),
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
    if sampling.symmetry != "C1" and not diagnostics.diagnostic_score_only and not local_relion_x_half_mstep:
        raise RuntimeError(
            f"{sampling.symmetry} exact-local reconstruction requires RELION x-half BPref "
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
                        sampling.local_search_order,
                        adaptive_oversampling=0,
                    ),
                    local_parent_oversampling_order=0,
                    rotation_grid_mstep_rotations=None,
                    generate_relion_mstep_rotations=False,
                ),
                local_batching,
                replace(
                    local_kernel,
                    accumulate_noise=False,
                    use_float64_scoring=fine_use_float64_scoring,
                    use_float64_projections=fine_use_float64_projections,
                    relion_exact_score_translation=bool(
                        _DENSE_EM_STATIC_KWARGS["relion_exact_fine_gaussian"] and not fine_use_float64_scoring
                    ),
                ),
                replace(
                    local_support,
                    disable_adjoint_y=True,
                    disable_adjoint_ctf=True,
                    reconstruct_significant_only=False,
                    score_only=True,
                ),
                replace(local_diagnostics, return_profile=False, debug_iteration=None, debug_pass_label=None),
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
    local_reconstruct_significant_only = int(sampling.local_parent_oversampling_order) > 0
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
        local_batching,
        replace(
            local_kernel,
            accumulate_noise=local_accumulate_noise,
            use_float64_scoring=fine_use_float64_scoring,
            use_float64_projections=fine_use_float64_projections,
            relion_exact_score_translation=bool(
                _DENSE_EM_STATIC_KWARGS["relion_exact_fine_gaussian"] and not fine_use_float64_scoring
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
        local_diagnostics,
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
        profile_row["half_index"] = np.int32(half.k)
        profile_row["local_adaptive_pass2_parent_mode"] = local_adaptive_pass2_parent_mode
        profile_row["local_adaptive_pass2_full_parent"] = np.bool_(local_adaptive_pass2_parent_mode == "full_parent")
        profile_row["diagnostic_score_only"] = np.bool_(diagnostics.diagnostic_score_only)
        diagnostics.local_profile_history.append(profile_row)
        if diagnostics.save_intermediates_dir is not None:
            np.savez_compressed(
                os.path.join(
                    diagnostics.save_intermediates_dir,
                    f"it{diagnostics.iteration:03d}_half{half.k + 1}_local_profile.npz",
                ),
                **local_profile_k,
            )
    pose_dtype = _dense_global_scoring_dtype()
    half.outputs.best_pose_rotations[half.k] = np.asarray(best_rots_k, dtype=pose_dtype)
    half.outputs.best_pose_rotation_eulers[half.k] = (
        np.asarray(local_outputs.best_pose_eulers_deg, dtype=np.float64)
        if local_outputs.best_pose_eulers_deg is not None
        else utils.R_to_relion(np.asarray(best_rots_k), degrees=True).astype(pose_dtype)
    )
    half.outputs.best_pose_translations[half.k] = np.asarray(best_trans_k, dtype=pose_dtype)
    return HalfScoreResult(
        ha=ha_k,
        Ft_y=Ft_y_k,
        Ft_ctf=Ft_ctf_k,
        em_stats=em_stats_k,
        noise_stats=noise_stats_k,
        best_pose_rotations=half.outputs.best_pose_rotations[half.k],
        best_pose_rotation_eulers=half.outputs.best_pose_rotation_eulers[half.k],
        best_pose_translations=half.outputs.best_pose_translations[half.k],
        significant_counts=relion_significant_counts_k,
        mstep_full_half_axis=0 if local_relion_x_half_mstep else None,
        mstep_accumulator_shape=(
            # Must match the current-size BPref grid allocated by the local
            # engine above; downstream join/reconstruct calls infer layout
            # from this shape.
            relion_backprojector_volume_shape(
                half.experiment_dataset.volume_shape,
                PADDING_FACTOR,
                # Images on another grid fill the backprojector at the reference model size.
                current_size=(
                    reconstruction_current_size_for_engine
                    if optics.reference_current_size is None
                    else optics.reference_current_size
                ),
            )
            if local_relion_x_half_mstep
            else None
        ),
    )


def _score_half_local_in_bpref_scope(
    half: LocalHalfData,
    sampling: LocalSamplingSpec,
    priors: LocalPriorSpec,
    batching: LocalBatchPolicy,
    execution: LocalExecutionPolicy,
    diagnostics: LocalDiagnosticPolicy,
    optics: LocalOpticsSpec,
) -> HalfScoreResult:
    """Run local scoring with device-capture flags disabled or fail closed."""

    if diagnostics.bpref_device_signature_active:
        raise RuntimeError("BPref device signature capture is supported only by sparse adaptive pass 2")
    with em_cuda_kernels.bpref_device_signature_scope(False):
        return _score_half_local(half, sampling, priors, batching, execution, diagnostics, optics)
