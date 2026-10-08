"""Build and execute one exact local-search iteration.

Construct image-specific pose neighborhoods, dispatch the single-class or
K-class kernel, and return named statistics to the
refinement controller. Dependencies are imported from their owning modules.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass

import numpy as np

from relax.helpers.types import NoiseStats, RelionStats
from relax.local.local_layout import (
    build_local_hypothesis_layout,
    expand_local_layout_classes,
    restrict_local_layout_classes,
)
from relax.relion.optics_aberrations import (
    dataset_projection_magnification,
    projection_rotations,
    reported_rotations,
)
from relax.sampling import build_local_search_grid_metadata
from relax.sparse_pass2.engine_record import record_pass_engine
from relax.sparse_pass2.resident_local_pass2 import compute_local_search_resident

logger = logging.getLogger("relax.local.local_search_iteration")


# Mirror iteration_loop's constant locally so the helper has a stable home.
EXACT_LOCAL_PRECOMPUTE_FINE_GRID_MAX_ROTATIONS = 3_000_000


def _precompute_exact_local_fine_grid_enabled(healpix_order: int, symmetry: str = "C1") -> bool:
    """Return whether exact local search should materialize the fine grid once."""
    from relax.sampling import rotation_grid_size

    return rotation_grid_size(int(healpix_order), symmetry) <= EXACT_LOCAL_PRECOMPUTE_FINE_GRID_MAX_ROTATIONS


@dataclass
class _LocalSearchIterationResult:
    Ft_y: object
    Ft_ctf: object
    hard_assignment: object
    relion_stats: RelionStats
    noise_stats: NoiseStats | None = None
    profile_summary: dict | None = None
    best_pose_rotations: object | None = None
    best_pose_translations: object | None = None
    best_pose_eulers_deg: np.ndarray | None = None
    # Class3D fine pass: the engine's class-segmented output (``ResidentKClassPass2Output``); the
    # K=1 fields above are then None.
    class_pass: object | None = None


@dataclass(frozen=True, kw_only=True)
class LocalSearchData:
    """Images, reference and per-image corrections for one local pass."""

    experiment_dataset: object
    mean: object
    noise_variance: object
    image_corrections: object | None = None
    scale_corrections: object | None = None
    group_ids: object | None = None
    scale_correction_group_count: object | None = None
    scale_correction_data_vs_prior: object | None = None
    image_pre_shifts: object | None = None
    optics_group_ids: object | None = None


@dataclass(frozen=True, kw_only=True)
class LocalSearchGridSpec:
    """Pose neighborhoods, priors and optional prebuilt pass-2 layout."""

    prior_rotations: object
    rotation_grid_rotations: object
    healpix_order: int
    sigma_rot: object
    sigma_psi: object
    translations: object
    prior_translations: object
    sigma_offset_angstrom: float
    translation_prior_reference_translations: object | None = None
    pass2_layout: object | None = None
    translation_prior_centers: object | None = None
    rotation_grid_random_perturbation: float = 0.0
    rotation_grid_angular_sampling_deg: float | None = None
    local_parent_oversampling_order: int = 0
    rotation_grid_mstep_rotations: object | None = None
    generate_relion_mstep_rotations: bool = False
    symmetry: str = "C1"
    # Class3D: each image's rows are scored against every class (expand_local_layout_classes), or, in
    # the first iteration from one reference, against each image's seed class only.
    n_classes: int = 1
    image_seed_classes: object | None = None


@dataclass(frozen=True, kw_only=True)
class LocalSearchKernelPolicy:
    """Numerical kernel, Fourier window, projector and optics choices."""

    disc_type: str
    current_size: int | None
    reconstruction_current_size: int | None = None
    accumulate_noise: bool = False
    projection_padding_factor: int = 1
    reconstruction_padding_factor: int = 1
    use_float64_scoring: bool = False
    use_float64_projections: bool = False
    square_window: bool = False
    half_spectrum_scoring: bool = False
    relion_exact_score_translation: bool = False
    projection_relion_texture_interp: bool | None = False
    projection_relion_acc_double_floorf_quirk: bool = False
    projection_relion_kernel: str = "fine"
    relion_projector_half: object | None = None
    relion_projector_r_max: int | None = None
    source_faithful_spectrum_norm: bool = False
    relion_translation_angle_scale: float = 1.0
    projection_scale: float = 1.0
    reconstruction_volume_current_size: int | None = None
    reconstruction_image_radius: float | None = None
    # RelionConsistencyOptions.nyquist_column_counting: how the per-image sums count the Hermitian
    # pairs of the full-size Nyquist column.
    nyquist_column_counting: str = "relion"
    # --strict_highres_exp: the fine pass's weighted-sum image size (None: current_size).
    wsum_current_size: int | None = None


@dataclass(frozen=True, kw_only=True)
class LocalSearchSupportPolicy:
    """Posterior support, accumulation and returned-detail choices."""

    mstep_relion_x_half: bool = False
    disable_adjoint_y: bool = False
    disable_adjoint_ctf: bool = False
    adaptive_fraction: float = 0.999
    max_significants: int | None = -1
    reconstruct_significant_only: bool = True
    return_best_pose_details: bool = False
    normalization_log_evidence: object | None = None
    return_reconstruction_sample_indices: bool = False
    apply_max_significants_to_support: bool = False
    stats_use_reconstruction_probs: bool = False
    score_only: bool = False
    return_profile: bool = False


def _run_local_search_iteration(
    data: LocalSearchData,
    grid: LocalSearchGridSpec,
    kernel: LocalSearchKernelPolicy,
    support: LocalSearchSupportPolicy,
) -> _LocalSearchIterationResult:
    """Run local search on the device-resident engine and return named halfset statistics and pose fields.

    Optional fields are None when their corresponding return flags are disabled.
    Arrays retain the engine's layouts and identities; profile metadata is copied
    and augmented with this wrapper's timings.
    """
    prior_rotations = grid.prior_rotations
    prior_translations = grid.prior_translations
    # Keep the local-search hypothesis grid (rotations/translations/priors)
    # genuinely double precision end to end when either flag requests it;
    # default stays float32 to match RELION's accelerated-GPU precision.
    local_layout_dtype = np.float64 if (kernel.use_float64_scoring or kernel.use_float64_projections) else np.float32
    # The prior angles stay as given (float64): RELION selects local supports from them in double.
    prior_rotations = np.asarray(prior_rotations)
    if prior_rotations.ndim == 3:
        n_prior = prior_rotations.shape[0]
    elif prior_rotations.ndim == 2 and prior_rotations.shape[1] == 3:
        n_prior = prior_rotations.shape[0]
    else:
        raise ValueError(f"prior_rotations must have shape (n,3,3) or (n,3), got {prior_rotations.shape}")
    if prior_translations is None:
        prior_translations = np.zeros(
            (n_prior, np.asarray(grid.translations).shape[1]),
            dtype=local_layout_dtype,
        )
    else:
        prior_translations = np.asarray(prior_translations, dtype=local_layout_dtype).reshape(
            -1,
            np.asarray(grid.translations).shape[1],
        )

    if grid.pass2_layout is None:
        metadata_t0 = time.time()
        # RELION local priors remain factorized in canonical direction/psi index
        # space even when the scored trial rotations have been perturbed.
        local_grid_metadata = build_local_search_grid_metadata(
            grid.healpix_order, symmetry=grid.symmetry
        )
        metadata_build_time = time.time() - metadata_t0

        layout_t0 = time.time()
        layout_kwargs = {}
        if int(grid.local_parent_oversampling_order) > 0:
            layout_kwargs["local_parent_oversampling_order"] = int(grid.local_parent_oversampling_order)
        if grid.rotation_grid_mstep_rotations is not None:
            layout_kwargs["rotation_grid_mstep_rotations"] = grid.rotation_grid_mstep_rotations
        if bool(grid.generate_relion_mstep_rotations):
            layout_kwargs["generate_relion_mstep_rotations"] = True
        local_layout = build_local_hypothesis_layout(
            prior_rotations,
            grid.rotation_grid_rotations,
            grid.sigma_rot,
            grid.sigma_psi,
            grid.healpix_order,
            grid.translations,
            prior_translations,
            grid.sigma_offset_angstrom,
            # Match the grouped RELION-mode path: local translation priors use the
            # learned/model sigma, not the older range/3 override.
            None,
            data.experiment_dataset.voxel_size,
            grid_metadata=local_grid_metadata,
            translation_prior_reference_translations=grid.translation_prior_reference_translations,
            rotation_log_prior=None,
            rotation_grid_random_perturbation=grid.rotation_grid_random_perturbation,
            rotation_grid_angular_sampling_deg=grid.rotation_grid_angular_sampling_deg,
            dtype=local_layout_dtype,
            **layout_kwargs,
        )
        selector_time = time.time() - layout_t0
    else:
        local_layout = grid.pass2_layout
        metadata_build_time = 0.0
        selector_time = 0.0

    # Class3D local searches (--sigma_ang) score each image's rows against every class.
    if int(grid.n_classes) > 1 and int(getattr(local_layout, "n_classes", 1)) == 1:
        local_layout = expand_local_layout_classes(local_layout, int(grid.n_classes))
        if grid.image_seed_classes is not None:
            local_layout = restrict_local_layout_classes(local_layout, grid.image_seed_classes)
    local_n_classes = int(getattr(local_layout, "n_classes", 1))
    magnification = dataset_projection_magnification(data.experiment_dataset)
    if kernel.projection_scale != 1.0 or magnification is not None:
        # Images on another grid than the reference (RELION applyScaleDifference) or with an
        # anisotropic magnification (applyAnisoMag): only the projection and backprojection
        # matrices are transformed; priors and reported poses are not.
        local_layout = dataclasses.replace(
            local_layout,
            rotations_flat=projection_rotations(local_layout.rotations_flat, kernel.projection_scale, magnification),
            mstep_rotations_flat=projection_rotations(
                local_layout.mstep_rotations_flat, kernel.projection_scale, magnification
            ),
        )

    # A regular iteration at the full box hands current_size=None (full-box support;
    # MS2 box 512 it25, bench 14641044); the resident driver scores RELION's radial
    # window at the box for it, as for the final all-data pass's explicit box size.
    # The device-resident local pass is relax's one local engine, for the fine pass 2
    # and for RELION's pass-1 parent probe (score-only: significant samples, no
    # M-step); a configuration it does not implement is an error
    # (ResidentConfigurationUnsupported).
    logger.info(
        "running the device-resident local %s (its capacity plan is sized from the projection byte budget)",
        "pass-1 parent probe" if support.score_only else "fine pass 2",
    )
    engine_outputs = compute_local_search_resident(
        data.experiment_dataset,
        data.mean,
        data.noise_variance,
        local_layout,
        kernel.disc_type,
        current_size=kernel.current_size,
        reconstruction_current_size=kernel.reconstruction_current_size,
        wsum_current_size=kernel.wsum_current_size,
        accumulate_noise=kernel.accumulate_noise,
        projection_padding_factor=kernel.projection_padding_factor,
        reconstruction_padding_factor=kernel.reconstruction_padding_factor,
        half_spectrum_scoring=kernel.half_spectrum_scoring,
        relion_exact_score_translation=kernel.relion_exact_score_translation,
        projection_relion_texture_interp=kernel.projection_relion_texture_interp,
        projection_relion_acc_double_floorf_quirk=kernel.projection_relion_acc_double_floorf_quirk,
        projection_relion_kernel=kernel.projection_relion_kernel,
        relion_projector_half=kernel.relion_projector_half,
        relion_projector_r_max=kernel.relion_projector_r_max,
        use_float64_scoring=kernel.use_float64_scoring,
        use_float64_projections=kernel.use_float64_projections,
        square_window=kernel.square_window,
        image_corrections=data.image_corrections,
        scale_corrections=data.scale_corrections,
        group_ids=data.group_ids,
        scale_correction_group_count=data.scale_correction_group_count,
        scale_correction_data_vs_prior=data.scale_correction_data_vs_prior,
        image_pre_shifts=data.image_pre_shifts,
        mstep_relion_x_half=support.mstep_relion_x_half,
        disable_adjoint_y=support.disable_adjoint_y,
        disable_adjoint_ctf=support.disable_adjoint_ctf,
        reconstruct_significant_only=support.reconstruct_significant_only,
        adaptive_fraction=support.adaptive_fraction,
        max_significants=support.max_significants if support.apply_max_significants_to_support else -1,
        return_best_pose_details=support.return_best_pose_details,
        return_reconstruction_sample_indices=support.return_reconstruction_sample_indices,
        return_profile=support.return_profile,
        stats_use_reconstruction_probs=support.stats_use_reconstruction_probs,
        translation_prior_centers=grid.translation_prior_centers,
        normalization_log_evidence=support.normalization_log_evidence,
        source_faithful_spectrum_norm=kernel.source_faithful_spectrum_norm,
        relion_translation_angle_scale=kernel.relion_translation_angle_scale,
        score_only=support.score_only,
        optics_group_ids=data.optics_group_ids,
        reconstruction_volume_current_size=kernel.reconstruction_volume_current_size,
        symmetry_label=grid.symmetry,
        reconstruction_image_radius=kernel.reconstruction_image_radius,
        nyquist_column_counting=kernel.nyquist_column_counting,
    )
    record_pass_engine("local_probe" if support.score_only else "local", "resident")
    if local_n_classes > 1 and not support.score_only:
        if kernel.projection_scale != 1.0 or magnification is not None:
            engine_outputs = engine_outputs._replace(
                per_class_best_pose_rotations=tuple(
                    reported_rotations(rotations, kernel.projection_scale, magnification)
                    for rotations in engine_outputs.per_class_best_pose_rotations
                )
            )
        return _LocalSearchIterationResult(
            Ft_y=None,
            Ft_ctf=None,
            hard_assignment=None,
            relion_stats=engine_outputs.stats,
            noise_stats=engine_outputs.noise_stats,
            profile_summary=engine_outputs.profile if support.return_profile else None,
            class_pass=engine_outputs,
        )
    result = _LocalSearchIterationResult(
        Ft_y=engine_outputs.Ft_y,
        Ft_ctf=engine_outputs.Ft_ctf,
        hard_assignment=engine_outputs.hard_assignments,
        relion_stats=engine_outputs.stats,
        noise_stats=engine_outputs.noise_stats,
        profile_summary=engine_outputs.profile if support.return_profile else None,
        best_pose_rotations=(
            engine_outputs.best_pose_rotations
            if (kernel.projection_scale == 1.0 and magnification is None) or engine_outputs.best_pose_rotations is None
            else reported_rotations(engine_outputs.best_pose_rotations, kernel.projection_scale, magnification)
        ),
        best_pose_translations=engine_outputs.best_pose_translations,
        best_pose_eulers_deg=engine_outputs.best_pose_eulers_deg,
    )

    if support.return_profile and result.profile_summary is not None:
        result.profile_summary = dict(result.profile_summary)
        result.profile_summary["metadata_build_time_s"] = np.float64(metadata_build_time)
        result.profile_summary["selector_time_s"] = np.float64(selector_time)

    return result
