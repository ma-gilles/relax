"""Prepare local-search grids and pass sizes for refinement."""

import logging
from dataclasses import dataclass

import numpy as np

from relax import sampling
from relax.helpers.convergence import healpix_angular_step
from relax.helpers.orientation_priors import relion_local_search_sigmas
from relax.helpers.resolution import ImageGeometry, relion_local_pass1_current_size
from relax.refinement.iteration_planning import ExpectationWindows, plan_expectation_windows
from relax.refinement.local_search_iteration import _precompute_exact_local_fine_grid_enabled

# The numbered controller's log: its operations log under its name wherever they live.
logger = logging.getLogger("relax.refinement.iteration_loop")


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


@dataclass(frozen=True)
class ExpectationSampling:
    """Where and how finely one numbered expectation scores; built once per iteration, read-only.

    ``windows`` are the model and image Fourier widths. ``local`` is the local-search sampling, None for
    a global search.
    """

    windows: ExpectationWindows
    local: LocalSampling | None


def plan_expectation_sampling(
    trial_grid: sampling.TrialGrid,
    coarse_grids,
    state,
    options,
    optics,
    *,
    current_size: int,
    use_local: bool,
    perturbation: float,
    coarse_size_healpix_order: int,
) -> ExpectationSampling:
    """Plan the Fourier windows and, for a local search, the local sampling of one numbered expectation.

    ``use_local`` is the trajectory's decision to search locally; ``perturbation`` is this iteration's
    sampling perturbation and ``coarse_size_healpix_order`` the order that sized pass 1 before the
    sampling advanced. In order: the expectation windows, then the local-search prior widths and sampling.
    Reads from ``coarse_grids``: ``base_translations`` and the rotation grid's order; from ``state``:
    ``healpix_order``, ``adaptive_oversampling``, ``sigma_rot`` and ``sigma_psi``; from ``options``:
    ``symmetry.point_group`` and ``schedule.particle_diameter_ang``; ``optics`` as the window plan and the
    local sampling read it.
    """
    windows = plan_expectation_windows(current_size, optics, log=logger)
    sigma_rot, sigma_psi = relion_local_search_sigmas(state, use_local=use_local)

    # Angular step behind this iteration's pass-1 coarse size, when RELION's
    # adaptive formula sets it (shape classes recompute their own from it).
    if use_local:
        local_sampling = prepare_numbered_local_sampling(
            LocalSearchSettings(
                healpix_order=state.healpix_order + state.adaptive_oversampling,
                oversampling_order=int(state.adaptive_oversampling) if state.adaptive_oversampling > 0 else 0,
                sigma_rot=sigma_rot,
                sigma_psi=sigma_psi,
                symmetry=options.symmetry.point_group,
            ),
            trial_grid,
            optics,
            base_translations=coarse_grids.base_translations,
            image_window_size=windows.image_window_size,
            model_support_size=windows.model_window_size,
            base_healpix_order=coarse_grids.rotation_grid.healpix_order,
            coarse_size_healpix_order=coarse_size_healpix_order,
            perturbation=perturbation,
            particle_diameter_angstrom=options.schedule.particle_diameter_ang,
            log=logger,
        )
    else:
        local_sampling = None
    return ExpectationSampling(windows, local_sampling)
