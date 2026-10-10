"""Build and execute one exact local-search iteration.

Construct image-specific pose neighborhoods, dispatch the single-class or
K-class kernel, and return named statistics to the
refinement controller. Dependencies are imported from their owning modules.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass

import numpy as np

from relax.helpers.timing import Stopwatch
from relax.helpers.types import NoiseStats, RelionStats
from relax.local.local_layout import (
    build_local_hypothesis_layout,
    drop_local_layout_classes,
    expand_local_layout_classes,
    local_layout_host_rotations,
    restrict_local_layout_classes,
)
from relax.relion.optics_aberrations import (
    dataset_projection_magnification,
    reported_rotations,
)
from relax.sampling import build_local_search_grid_metadata, project_rows
from relax.sparse_pass2.engine_record import record_pass_engine
from relax.sparse_pass2.local_search_records import (
    LocalSearchData,
    LocalSearchKernelPolicy,
    LocalSearchSupportPolicy,
)
from relax.sparse_pass2.resident_local_pass2 import compute_local_search_resident

logger = logging.getLogger("relax.local.local_search_iteration")


@dataclass(frozen=True, kw_only=True)
class LocalSearchResult:
    """One K=1 local pass: the half's accumulators, statistics and best poses (unscaled, unmagnified)."""

    Ft_y: object
    Ft_ctf: object
    hard_assignment: object
    relion_stats: RelionStats
    noise_stats: NoiseStats | None = None
    profile_summary: dict | None = None
    best_pose_rotations: object | None = None
    best_pose_translations: object | None = None
    best_pose_eulers_deg: np.ndarray | None = None


@dataclass(frozen=True, kw_only=True)
class LocalClassSearchResult:
    """One Class3D local fine pass: the engine's class-segmented output and its statistics."""

    # The resident engine's ``ResidentKClassPass2Output`` (per-class accumulators, sums and best poses).
    class_pass: object
    relion_stats: RelionStats
    noise_stats: NoiseStats | None
    profile_summary: dict | None


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
    # Classes at zero weight (RELION's pdf_class == 0): not scored (drop_local_layout_classes).
    empty_classes: tuple = ()


def _run_local_search_iteration(
    data: LocalSearchData,
    grid: LocalSearchGridSpec,
    kernel: LocalSearchKernelPolicy,
    support: LocalSearchSupportPolicy,
) -> LocalSearchResult | LocalClassSearchResult:
    """Run local search on the device-resident engine and return named halfset statistics and pose fields.

    A Class3D fine pass (several classes, not score-only) returns :class:`LocalClassSearchResult`, every
    other pass :class:`LocalSearchResult`.

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
        metadata_clock = Stopwatch()
        # RELION local priors remain factorized in canonical direction/psi index
        # space even when the scored trial rotations have been perturbed.
        local_grid_metadata = build_local_search_grid_metadata(
            grid.healpix_order, symmetry=grid.symmetry
        )
        metadata_build_time = metadata_clock.seconds

        layout_clock = Stopwatch()
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
            local_parent_oversampling_order=grid.local_parent_oversampling_order,
            rotation_grid_mstep_rotations=grid.rotation_grid_mstep_rotations,
            generate_relion_mstep_rotations=grid.generate_relion_mstep_rotations,
            dtype=local_layout_dtype,
        )
        selector_time = layout_clock.seconds
    else:
        local_layout = grid.pass2_layout
        metadata_build_time = 0.0
        selector_time = 0.0

    # Class3D local searches (--sigma_ang) score each image's rows against every class.
    if int(grid.n_classes) > 1 and local_layout.n_classes == 1:
        local_layout = expand_local_layout_classes(local_layout, int(grid.n_classes))
        if grid.image_seed_classes is not None:
            local_layout = restrict_local_layout_classes(local_layout, grid.image_seed_classes)
        local_layout = drop_local_layout_classes(local_layout, grid.empty_classes)
    local_n_classes = local_layout.n_classes
    magnification = dataset_projection_magnification(data.experiment_dataset)
    if kernel.projection_scale != 1.0 or magnification is not None:
        # Images on another grid than the reference (RELION applyScaleDifference) or with an
        # anisotropic magnification (applyAnisoMag): only the projection and backprojection
        # matrices are transformed; priors and reported poses are not. The local rows are host-built
        # rows, composed from their float64 matrices (RELION's generateEulerMatrices rule).
        def host_rows(mstep: bool):
            return lambda: local_layout_host_rotations(local_layout, mstep=mstep)

        local_layout = dataclasses.replace(
            local_layout,
            rotations_flat=project_rows(
                local_layout.rotations_flat,
                kernel.projection_scale,
                magnification,
                host_rows=host_rows(False),
                what="local rows",
            ),
            mstep_rotations_flat=project_rows(
                local_layout.mstep_rotations_flat,
                kernel.projection_scale,
                magnification,
                host_rows=host_rows(True),
                what="local M-step rows",
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
        data,
        local_layout,
        kernel,
        support,
        translation_prior_centers=grid.translation_prior_centers,
        symmetry_label=grid.symmetry,
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
        return LocalClassSearchResult(
            class_pass=engine_outputs,
            relion_stats=engine_outputs.stats,
            noise_stats=engine_outputs.noise_stats,
            profile_summary=engine_outputs.profile if support.return_profile else None,
        )
    profile_summary = engine_outputs.profile if support.return_profile else None
    if support.return_profile and profile_summary is not None:
        profile_summary = dict(profile_summary)
        profile_summary["metadata_build_time_s"] = np.float64(metadata_build_time)
        profile_summary["selector_time_s"] = np.float64(selector_time)
    return LocalSearchResult(
        Ft_y=engine_outputs.Ft_y,
        Ft_ctf=engine_outputs.Ft_ctf,
        hard_assignment=engine_outputs.hard_assignments,
        relion_stats=engine_outputs.stats,
        noise_stats=engine_outputs.noise_stats,
        profile_summary=profile_summary,
        best_pose_rotations=(
            engine_outputs.best_pose_rotations
            if (kernel.projection_scale == 1.0 and magnification is None) or engine_outputs.best_pose_rotations is None
            else reported_rotations(engine_outputs.best_pose_rotations, kernel.projection_scale, magnification)
        ),
        best_pose_translations=engine_outputs.best_pose_translations,
        best_pose_eulers_deg=engine_outputs.best_pose_eulers_deg,
    )
