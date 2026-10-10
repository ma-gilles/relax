"""Exact-local scoring of one refinement half-set: the local search's records, its adaptive pass-2 layouts and
the dispatch of a half (one image shape, or one search per shape class) to the local-search iteration. A
diagnostic BPref scope surrounds each half's scoring call."""

import logging
from dataclasses import dataclass, replace

import numpy as np
from recovar import utils

from relax.cuda import kernels as em_cuda_kernels
from relax.dense import scoring_policy
from relax.dense.score_outputs import (
    HalfScoreResult,
    class_em_to_half_result,
)
from relax.dense.scoring_policy import (
    RELION_ACC_DOUBLE_FLOORF_QUIRK,
    RELION_ADAPTIVE_FRACTION,
    RELION_FOURIER_WINDOW_SQUARE,
)
from relax.diagnostics import local_debug
from relax.helpers.dtype_policy import DensePrecisionPolicy
from relax.helpers.half_volume_mstep import relion_backprojector_volume_shape
from relax.local.local_layout import (
    build_local_adaptive_pass2_hypothesis_layout,
    build_local_hypothesis_layout,
    drop_local_layout_classes,
    expand_local_layout_classes,
    restrict_local_layout_classes,
)
from relax.refinement.dense_half import (
    expand_significant_samples_to_full_parent_translations,
    merge_shape_class_results,
    require_multi_shape_inputs,
    single_shape_reconstruction_grid,
)
from relax.refinement.half_inputs import HalfScoringData
from relax.refinement.local_sampling import LocalSampling
from relax.refinement.local_search_iteration import (
    LocalClassSearchResult,
    LocalSearchGridSpec,
    _run_local_search_iteration,
)
from relax.refinement.ports import ExpectationProbe
from relax.refinement.refinement_options import LocalAdaptivePass2Support
from relax.refinement.shape_class_scoring import OpticsSpec, reconstruction_image_radius
from relax.relion.geometry import (
    PROJECTION_PADDING_FACTOR,
    RECONSTRUCTION_PADDING_FACTOR,
)
from relax.relion.optics_aberrations import (
    dataset_projection_magnification,
)
from relax.sampling import (
    build_local_search_grid_metadata,
    relion_angular_sampling_deg,
)
from relax.sparse_pass2.local_search_records import (
    LocalSearchData,
    LocalSearchKernelPolicy,
    LocalSearchSupportPolicy,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class LocalPriorSpec:
    """Translation-prior operands and their grid-selection policy."""

    trans_prior_center: object
    trans_prior_center_for_engine: object
    current_sigma_offset_angstrom: float
    translation_search_base: object
    replay_prior_translations: object | None = None
    # Class3D: the class log priors. A local search's weights carry no class prior, as RELION's, but a
    # class at -inf (pdf_class == 0) is left out of the search.
    class_log_priors: object | None = None


@dataclass(frozen=True, kw_only=True)
class LocalBatchPolicy:
    """Exact-local posterior support control."""

    max_significants: int | None


@dataclass(frozen=True, kw_only=True)
class LocalExecutionPolicy:
    """Interpolation and reconstruction controls."""

    disc_type: str
    source_faithful_spectrum_norm: bool
    relion_translation_angle_scale: float
    # RelionConsistencyOptions.nyquist_column_counting (see DenseExecutionPolicy).
    nyquist_column_counting: str
    # ScoringVariants: the K=1 x-half M-step, and adaptive pass 2's support (off without oversampling).
    relion_x_half_mstep: bool
    adaptive_pass2: LocalAdaptivePass2Support
    # The run's dense precision (pass 1 and the pose dtype), and pass 2's, which a float64 diagnostic of this
    # iteration may widen (scoring_policy.local_precision); resolved by the caller.
    precision: DensePrecisionPolicy
    fine_precision: DensePrecisionPolicy
    # --stop_after_local_search_score_only: score without backprojection, noise accumulation or x-half M-step.
    score_only: bool
    # RELION's --firstiter_cc iteration when the search is local from iteration 1 (--sigma_ang).
    firstiter_cc: bool


@dataclass(frozen=True, kw_only=True)
class LocalDiagnosticPolicy:
    """Profiling, debug capture and score-only controls; ``probe`` receives each collected profile."""

    iteration: int
    collect_local_search_profile: bool
    local_profile_history: object
    debug_iteration: int | None
    bpref_device_signature_active: bool
    probe: ExpectationProbe


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


@dataclass(frozen=True, kw_only=True)
class LocalPass2Layouts:
    """Adaptive local pass 2's hypothesis layouts, derived from pass 1's retained parent samples."""

    # The fine (oversampled) hypotheses pass 2 scores.
    layout: object
    # RELION's rlnNrOfSignificantSamples: retained pass-1 samples per image (int32), or None when pass 1 did
    # not return explicit support.
    significant_counts: np.ndarray | None
    # The broader support of the diagnostic denominator probe (support.denominator_mode), or None.
    denominator_layout: object | None


def _prepare_local_adaptive_pass2_support(
    parent_layout,
    significant_sample_indices,
    sampling: LocalSampling,
    support: LocalAdaptivePass2Support,
    parent_order: int,
    fine_layout_dtype,
) -> LocalPass2Layouts:
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

    if support.full_parent:
        significant_sample_indices = [None] * len(significant_sample_indices)
        logger.info(
            "RELION local adaptive pass 2: expanding all parent samples; set %s=0 for pruned-parent support",
            scoring_policy.LOCAL_ADAPTIVE_PASS2_FULL_PARENT_ENV,
        )
    elif support.rotation_only:
        significant_sample_indices = expand_significant_samples_to_full_parent_translations(
            significant_sample_indices,
            int(sampling.translations.shape[0]),
        )
        logger.info(
            "RELION local adaptive pass 2 diagnostic: expanding significant parent rotations to all "
            "parent translations via %s=1",
            scoring_policy.LOCAL_ADAPTIVE_PASS2_ROTATION_ONLY_ENV,
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
    if support.denominator_mode is not None:
        if support.denominator_mode == "full_parent":
            denominator_significant_sample_indices = [None] * len(
                pruned_parent_significant_sample_indices
            )
        else:
            denominator_significant_sample_indices = (
                expand_significant_samples_to_full_parent_translations(
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
        local_debug.log_local_denominator_support(
            logger,
            denominator_layout,
            support.denominator_mode,
            scoring_policy.LOCAL_ADAPTIVE_PASS2_DENOMINATOR_SUPPORT_ENV,
        )
    local_debug.log_local_adaptive_support(
        logger,
        parent_layout,
        significant_sample_indices,
        sampling.translations,
        pass2_layout,
    )
    return LocalPass2Layouts(
        layout=pass2_layout,
        significant_counts=relion_significant_counts,
        denominator_layout=denominator_layout,
    )


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


@dataclass(frozen=True, kw_only=True)
class LocalShapeOwners:
    """The exact-local scoring owners one shape class changes; batching, execution and diagnostics are shared."""

    half: HalfScoringData
    sampling: LocalSampling
    priors: LocalPriorSpec
    optics: OpticsSpec


def _local_owners_for_shape(
    half: HalfScoringData,
    sampling: LocalSampling,
    priors: LocalPriorSpec,
    optics: OpticsSpec,
    shape_class,
    class_index: int,
) -> LocalShapeOwners:
    """Derive one exact-local shape class through the optics owner."""

    from relax.refinement import shape_class_scoring

    shape_values = shape_class_scoring.class_kwargs(
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
    return LocalShapeOwners(
        half=replace(
            half,
            particles=replace(
                half.particles,
                dataset=shape_values["experiment_dataset"],
                optics_group_ids=shape_values["optics_group_ids_k"],
                rotation_eulers=shape_values["previous_best_rotation_eulers_k"],
                image_corrections=shape_values["image_corrections_k"],
                scale_corrections=shape_values["scale_corrections_k"],
            ),
            noise_variance=shape_class_scoring.class_noise_table(
                optics.noise_radial_k,
                shape_class,
                int(half.particles.dataset.image_shape[0]),
            ),
            scale_group_ids=shape_values["group_ids_k"],
        ),
        sampling=replace(
            sampling,
            translations=shape_values["current_translations"],
            base_translations=shape_values["base_translations"],
            image_window_size=shape_values["cs_for_engine"],
            model_support_size=shape_values["model_current_size_for_engine"],
            coarse_image_window_size=shape_values["local_pass1_current_size"],
        ),
        priors=replace(
            priors,
            trans_prior_center=shape_values["trans_prior_center"],
            trans_prior_center_for_engine=shape_values["trans_prior_center_for_engine"],
            translation_search_base=shape_values["translation_search_base"],
            replay_prior_translations=shape_values["replay_prior_translations"],
        ),
        optics=replace(
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

    from relax.refinement import optics_shapes, shape_class_scoring

    experiment_half = half.particles.dataset
    if not isinstance(experiment_half, optics_shapes.MultiShapeHalf):
        sampling, optics = single_shape_reconstruction_grid(experiment_half, sampling, optics)
        return _score_half_local_one_shape(half, sampling, priors, batching, execution, diagnostics, optics)
    require_multi_shape_inputs(half, optics)
    shape_class_scoring.require_exact_local_parent_windows(
        {
            "experiment_dataset": experiment_half,
            "cs_for_engine": sampling.image_window_size,
            "local_pass1_current_size": sampling.coarse_image_window_size,
            "coarse_sizing": optics.coarse_sizing,
        }
    )
    results = []
    for index, shape_class in enumerate(experiment_half.classes):
        owners = _local_owners_for_shape(half, sampling, priors, optics, shape_class, index)
        results.append(
            _score_half_local_one_shape(
                owners.half, owners.sampling, owners.priors, batching, execution, diagnostics, owners.optics
            )
        )
    return merge_shape_class_results(results, experiment_half)


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

    Routes through ``_run_local_search_iteration``. Local searches are K=1 only: Class3D keeps
    global searches, as RELION switches to local searches from the HEALPix order
    only under auto-refine (ml_optimiser.cpp:2541-2565, 3936-3938).
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
    empty_classes = (
        ()
        if n_classes == 1 or priors.class_log_priors is None
        else tuple(int(k) for k in np.flatnonzero(np.isneginf(np.asarray(priors.class_log_priors, dtype=np.float64))))
    )

    local_debug_iteration = (
        diagnostics.iteration + 1 if diagnostics.debug_iteration is None else int(diagnostics.debug_iteration)
    )
    # Adaptive pass-2 (fine, oversampled) hypothesis layout precision; see
    # ``parent_local_layout_dtype`` below for the matching pass-1 value.
    fine_local_layout_dtype = execution.fine_precision.rotation_real_dtype
    if execution.fine_precision.use_float64_scoring or execution.fine_precision.use_float64_projections:
        logger.info(
            "Local-search precision iteration %d: pass1 scoring/projections=%s/%s pass2 scoring/projections=%s/%s",
            local_debug_iteration,
            execution.precision.use_float64_scoring,
            execution.precision.use_float64_projections,
            execution.fine_precision.use_float64_scoring,
            execution.fine_precision.use_float64_projections,
        )

    # Keep the local-search hypothesis grid genuinely double precision end to
    # end when either flag requests it; default stays float32 to match
    # RELION's accelerated-GPU precision.
    parent_local_layout_dtype = execution.precision.rotation_real_dtype
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
    if int(sampling.search.oversampling_order) > 0:
        logger.info(
            "RELION local search: expanding translations by oversampling_order=%d (coarse n=%d -> fine n=%d)",
            int(sampling.search.oversampling_order),
            int(sampling.translations.shape[0]),
            int(sampling.translations.shape[0]) * 4 ** int(sampling.search.oversampling_order),
        )
    # One pass's operands; the parent, denominator and final passes ``replace`` the fields they change.
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
            empty_classes=empty_classes,
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
            # A class on the reference grid (no reference size) keeps the engine's own bound and reads no optics.
            reconstruction_image_radius=(
                None
                if optics.reference_current_size is None
                else reconstruction_image_radius(
                    optics.reference_current_size,
                    optics.projection_scale,
                    dataset_projection_magnification(half.particles.dataset),
                )
            ),
            nyquist_column_counting=execution.nyquist_column_counting,
            wsum_current_size=sampling.wsum_current_size,
            firstiter_cc=execution.firstiter_cc,
    )
    local_support = LocalSearchSupportPolicy(
            disable_adjoint_y=False,
            disable_adjoint_ctf=False,
            adaptive_fraction=RELION_ADAPTIVE_FRACTION,
            max_significants=batching.max_significants,
            return_profile=diagnostics.collect_local_search_profile,
    )
    pass2 = LocalPass2Layouts(layout=None, significant_counts=None, denominator_layout=None)
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
            parent_layout = drop_local_layout_classes(parent_layout, empty_classes)
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
                use_float64_scoring=execution.precision.use_float64_scoring,
                use_float64_projections=execution.precision.use_float64_projections,
                relion_exact_score_translation=bool(
                    scoring_policy.RELION_EXACT_FINE_GAUSSIAN and not execution.precision.use_float64_scoring
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
        pass2 = _prepare_local_adaptive_pass2_support(
            parent_layout,
            significant_sample_indices,
            sampling,
            execution.adaptive_pass2,
            parent_order,
            fine_local_layout_dtype,
        )
    local_relion_x_half_mstep = execution.relion_x_half_mstep
    if execution.score_only:
        local_relion_x_half_mstep = False
    if sampling.search.symmetry != "C1" and not execution.score_only and not local_relion_x_half_mstep:
        raise RuntimeError(
            f"{sampling.search.symmetry} exact-local reconstruction requires RELION x-half BPref "
            f"accumulation; {scoring_policy.K1_RELION_X_HALF_MSTEP_ENV}=0, CPU-only execution, or disabled custom CUDA "
            "is unsupported for non-C1 symmetry"
        )
    if local_relion_x_half_mstep:
        logger.info(
            "RELION local K=1 M-step: using x-half BPref-layout backprojection",
        )
    if pass2.denominator_layout is not None:
        logger.info("RELION local adaptive pass 2 diagnostic: running score-only broad-denominator probe")
        denominator_outputs = _run_local_search_iteration(
            local_data,
            replace(
                local_grid,
                pass2_layout=pass2.denominator_layout,
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
                use_float64_scoring=execution.fine_precision.use_float64_scoring,
                use_float64_projections=execution.fine_precision.use_float64_projections,
                relion_exact_score_translation=bool(
                    scoring_policy.RELION_EXACT_FINE_GAUSSIAN and not execution.fine_precision.use_float64_scoring
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
    local_accumulate_noise = not execution.score_only
    logger.info(
        "RELION local fine pass 2: supplied-PPref interpolation follows "
        "RELAX_RELION_PROJECTOR_TEXTURE_INTERP (default texture)"
    )
    local_outputs = _run_local_search_iteration(
        local_data,
        replace(local_grid, pass2_layout=pass2.layout),
        replace(
            local_kernel,
            accumulate_noise=local_accumulate_noise,
            use_float64_scoring=execution.fine_precision.use_float64_scoring,
            use_float64_projections=execution.fine_precision.use_float64_projections,
            relion_exact_score_translation=bool(
                scoring_policy.RELION_EXACT_FINE_GAUSSIAN and not execution.fine_precision.use_float64_scoring
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
            # A score-only pass has no M-step.
            disable_adjoint_y=execution.score_only,
            disable_adjoint_ctf=execution.score_only,
            reconstruct_significant_only=local_reconstruct_significant_only,
            return_best_pose_details=True,
            normalization_log_evidence=local_normalization_log_evidence,
            stats_use_reconstruction_probs=local_reconstruct_significant_only,
            score_only=execution.score_only,
        ),
    )
    if diagnostics.collect_local_search_profile:
        local_profile_k = local_outputs.profile_summary
        profile_row = dict(local_profile_k)
        profile_row["iteration"] = np.int32(diagnostics.iteration)
        profile_row["half_index"] = np.int32(half.particles.index)
        parent_mode = (
            execution.adaptive_pass2.parent_mode if int(sampling.search.oversampling_order) > 0 else "none"
        )
        profile_row["local_adaptive_pass2_parent_mode"] = parent_mode
        profile_row["local_adaptive_pass2_full_parent"] = np.bool_(parent_mode == "full_parent")
        profile_row["diagnostic_score_only"] = np.bool_(execution.score_only)
        diagnostics.local_profile_history.append(profile_row)
        diagnostics.probe.local_search_profile(diagnostics.iteration, half.particles.index, local_profile_k)
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
    if isinstance(local_outputs, LocalClassSearchResult):
        return _class_local_half_result(
            local_outputs,
            n_classes=n_classes,
            significant_counts=pass2.significant_counts,
            mstep_accumulator_shape=mstep_accumulator_shape,
            pose_dtype=execution.precision.rotation_real_dtype,
        )
    pose_dtype = execution.precision.rotation_real_dtype
    best_rots = np.asarray(local_outputs.best_pose_rotations, dtype=pose_dtype)
    best_eulers = (
        np.asarray(local_outputs.best_pose_eulers_deg, dtype=np.float64)
        if local_outputs.best_pose_eulers_deg is not None
        else utils.R_to_relion(np.asarray(local_outputs.best_pose_rotations), degrees=True).astype(pose_dtype)
    )
    best_translations = np.asarray(local_outputs.best_pose_translations, dtype=pose_dtype)
    return HalfScoreResult(
        ha=local_outputs.hard_assignment,
        Ft_y=local_outputs.Ft_y,
        Ft_ctf=local_outputs.Ft_ctf,
        em_stats=local_outputs.relion_stats,
        noise_stats=local_outputs.noise_stats,
        best_pose_rotations=best_rots,
        best_pose_rotation_eulers=best_eulers,
        best_pose_translations=best_translations,
        significant_counts=pass2.significant_counts,
        mstep_full_half_axis=0 if local_relion_x_half_mstep else None,
        mstep_accumulator_shape=mstep_accumulator_shape,
    )


def _class_local_half_result(
    outputs: LocalClassSearchResult, *, n_classes: int, significant_counts, mstep_accumulator_shape, pose_dtype
):
    """A Class3D local pass as the half's K-class result, as the subtomogram K-class pass adapts its own.

    ``outputs.class_pass`` is the resident local engine's ``ResidentKClassPass2Output``; its class rotation
    sums are over the layout's posterior grid, which is the direction-prior grid of the local search.
    """

    from relax.classification.k_class import _class_segmented_em_result

    k_class_result = _class_segmented_em_result(
        outputs.class_pass,
        n_classes=n_classes,
        class_posterior_sums_from_noise=True,
        return_profile=False,
        host_accumulators=True,
        mstep_full_half_axis=0,
        mstep_accumulator_shape=mstep_accumulator_shape,
    )
    score_result = class_em_to_half_result(
        k_class_result,
        effective_rotations=np.zeros((int(np.shape(outputs.class_pass.class_rotation_posterior_sums)[1]), 0)),
        rot_pmap_for_collapse=None,
        adaptive_os_local=0,
        require_best_pose_details=True,
        pose_dtype=pose_dtype,
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
