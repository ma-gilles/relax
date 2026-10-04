"""Prepare local-search grids and pass sizes for refinement."""

from dataclasses import dataclass

import numpy as np

from relax import sampling
from relax.helpers.convergence import healpix_angular_step
from relax.helpers.resolution import ImageGeometry, relion_local_pass1_current_size
from relax.refinement.local_search_iteration import _precompute_exact_local_fine_grid_enabled


@dataclass(frozen=True, kw_only=True)
class LocalSearchSettings:
    """Fine-grid order, parent expansion and angular prior widths in radians."""

    healpix_order: int
    oversampling_order: int
    sigma_rot: float
    sigma_psi: float
    symmetry: str = "C1"

    @property
    def parent_order(self) -> int:
        return self.healpix_order - self.oversampling_order


@dataclass(frozen=True, kw_only=True)
class LocalSampling:
    """Local orientation/translation grids and reconstruction sizes."""

    search: LocalSearchSettings
    rotations: object | None
    translations: object
    base_translations: object
    image_window_size: int | None
    coarse_image_window_size: int | None
    perturbation: float
    angular_step_deg: float | None
    mstep_rotations: object | None = None
    model_support_size: int | None = None
    coarse_angular_step_deg: float | None = None
    rotation_eulers: object | None = None


def prepare_numbered_local_sampling(
    search: LocalSearchSettings,
    grid: sampling.TrialGrid,
    optics,
    *,
    base_translations,
    image_window_size: int | None,
    model_support_size: int | None,
    base_healpix_order: int,
    coarse_size_healpix_order: int,
    perturbation: float,
    particle_diameter_angstrom: float | None,
    log,
) -> LocalSampling:
    """Resolve a numbered local grid and its pre-update pass-1 window.

    The supplied trial rows may be capped below the fine search order. Parent
    expansion defers that fine grid; its window uses the preceding sizing order.
    Reads from ``optics`` (the run's ``RunOptics``), only when parents are expanded: the first optics
    group's pixel and box size, else the model pixel size and the image box.
    """
    coarse_image_window_size = image_window_size
    coarse_angular_step_deg = None
    if grid.rotations.shape[0] != sampling.rotation_grid_size(search.healpix_order, symmetry=search.symmetry):
        log.info(
            "Using lazy fine local-search grid: order=%d (%d rotations) from capped base order=%d",
            search.healpix_order,
            sampling.rotation_grid_size(search.healpix_order, symmetry=search.symmetry),
            base_healpix_order,
        )
        angular_step_deg = sampling.relion_angular_sampling_deg(search.healpix_order, adaptive_oversampling=0)
        if search.oversampling_order == 0 and _precompute_exact_local_fine_grid_enabled(
            search.healpix_order, symmetry=search.symmetry,
        ):
            rotations, rotation_eulers, mstep_rotations = sampling._exact_local_fine_grid(
                healpix_order=search.healpix_order,
                angular_sampling_deg=angular_step_deg,
                random_perturbation=float(perturbation),
                symmetry=search.symmetry,
            )
            deferred_perturbation = 0.0
        else:
            rotations = rotation_eulers = mstep_rotations = None
            deferred_perturbation = float(perturbation)
            if search.oversampling_order > 0:
                log.info(
                    "RELION local search: expanding selected coarse parents by oversampling_order=%d",
                    int(search.oversampling_order),
                )
                parent_order = search.healpix_order - int(search.oversampling_order)
                coarse_angular_step_deg = healpix_angular_step(coarse_size_healpix_order)
                coarse_image_window_size = relion_local_pass1_current_size(
                    pre_update_healpix_order=coarse_size_healpix_order,
                    pixel_size=(
                        float(optics.optics_pixel_sizes[0])
                        if optics.optics_pixel_sizes is not None
                        else optics.model_pixel_size
                    ),
                    ori_size=(
                        int(optics.optics_image_sizes[0])
                        if optics.optics_image_sizes is not None
                        else optics.image_geometry.box_size
                    ),
                    particle_diameter=particle_diameter_angstrom,
                    current_size=image_window_size,
                )
                log.info(
                    "Local adaptive oversampling: pass 1 at coarse_size=%s, "
                    "pass 2 at current_size=%s (size_order=%d, parent_order=%d, oversampling=%d)",
                    coarse_image_window_size,
                    image_window_size,
                    coarse_size_healpix_order,
                    parent_order,
                    int(search.oversampling_order),
                )
    else:
        rotations = grid.rotations
        rotation_eulers = None
        mstep_rotations = sampling._local_search_mstep_rotations(
            grid.mstep_rotations, grid.rotation_eulers, search.healpix_order, symmetry=search.symmetry,
        )
        deferred_perturbation = 0.0
        angular_step_deg = None
    log.info(
        "Local search (batched exact): fine_order=%d, sigma_rot=%.4f rad (%.2f deg), sigma_psi=%.4f rad",
        search.healpix_order,
        search.sigma_rot,
        np.rad2deg(search.sigma_rot),
        search.sigma_psi,
    )
    return LocalSampling(
        search=search,
        rotations=rotations,
        rotation_eulers=rotation_eulers,
        mstep_rotations=mstep_rotations,
        translations=grid.translations,
        base_translations=base_translations,
        image_window_size=image_window_size,
        model_support_size=model_support_size,
        coarse_image_window_size=coarse_image_window_size,
        coarse_angular_step_deg=coarse_angular_step_deg,
        perturbation=deferred_perturbation,
        angular_step_deg=angular_step_deg,
    )


def prepare_final_local_sampling(
    search: LocalSearchSettings,
    image_geometry: ImageGeometry,
    *,
    translations,
    base_translations,
    image_window_size: int,
    particle_diameter_angstrom: float | None,
    perturbation: float | None,
    rotation_dtype,
) -> LocalSampling:
    """Prepare final local sampling; None perturbation differs from applied zero."""
    angular_step_deg = sampling.relion_angular_sampling_deg(search.healpix_order, adaptive_oversampling=0)

    if search.oversampling_order > 0:
        coarse_angular_step_deg = healpix_angular_step(search.parent_order)
        coarse_image_window_size = relion_local_pass1_current_size(
            pre_update_healpix_order=search.parent_order,
            pixel_size=image_geometry.pixel_size_angstrom,
            ori_size=image_geometry.box_size,
            particle_diameter=particle_diameter_angstrom,
            current_size=image_window_size,
        )
    else:
        coarse_image_window_size = image_window_size
        coarse_angular_step_deg = None

    if search.oversampling_order == 0 and _precompute_exact_local_fine_grid_enabled(search.healpix_order, symmetry=search.symmetry):
        fine_rotations, _, fine_mstep_rotations = sampling._exact_local_fine_grid(
            healpix_order=search.healpix_order,
            angular_sampling_deg=angular_step_deg,
            random_perturbation=perturbation,
            dtype=rotation_dtype,
            symmetry=search.symmetry,
        )
        deferred_perturbation = 0.0
    else:
        fine_rotations = fine_mstep_rotations = None
        deferred_perturbation = 0.0 if perturbation is None else perturbation

    return LocalSampling(
        search=search,
        rotations=fine_rotations,
        mstep_rotations=fine_mstep_rotations,
        translations=translations,
        base_translations=base_translations,
        image_window_size=image_window_size,
        coarse_image_window_size=coarse_image_window_size,
        perturbation=deferred_perturbation,
        angular_step_deg=angular_step_deg,
        coarse_angular_step_deg=coarse_angular_step_deg,
    )
