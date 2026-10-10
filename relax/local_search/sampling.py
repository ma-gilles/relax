"""Prepare local-search grids and pass sizes for refinement."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from relax import sampling
from relax.fourier.resolution import relion_local_pass1_current_size
from relax.helpers.convergence import direction_prior_healpix_order_for_scoring, healpix_angular_step
from relax.refinement.image_size_plans import ExpectationWindows, RunOptics, plan_expectation_windows
from relax.refinement.iteration_planning import IterationCarry
from relax.refinement.trial_grids import (
    CoarseGrids,
    builds_coarse_pass1_rotations,
    coarse_pass1_rotations,
    iteration_trial_grid,
)
from relax.sampling.orientation_priors import relion_direction_log_priors, relion_local_search_sigmas

if TYPE_CHECKING:
    from relax.refinement.refinement_options import RefinementOptions
    from relax.refinement.setup_checks import RunContext

# The numbered controller's log: its operations log under its name wherever they live.
logger = logging.getLogger("relax.refinement.iteration_loop")

EXACT_LOCAL_PRECOMPUTE_FINE_GRID_MAX_ROTATIONS = 3_000_000


def _precompute_exact_local_fine_grid_enabled(healpix_order: int, symmetry: str) -> bool:
    """Return whether exact local search should materialize the fine grid once."""

    return sampling.rotation_grid_size(healpix_order, symmetry) <= EXACT_LOCAL_PRECOMPUTE_FINE_GRID_MAX_ROTATIONS


@dataclass(frozen=True, kw_only=True)
class LocalSearchSettings:
    """Fine-grid order, parent expansion and angular prior widths in radians."""

    healpix_order: int
    oversampling_order: int
    sigma_rot: float
    sigma_psi: float
    symmetry: str

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
    # --strict_highres_exp: the fine pass's weighted-sum image size, above image_window_size; None when equal.
    wsum_current_size: int | None = None

    @property
    def symmetry(self) -> str:
        return self.search.symmetry


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
    strict_pass1: bool,
    wsum_current_size: int | None,
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
        logger.info(
            "Using lazy fine local-search grid: order=%d (%d rotations) from capped base order=%d",
            search.healpix_order,
            sampling.rotation_grid_size(search.healpix_order, symmetry=search.symmetry),
            base_healpix_order,
        )
        angular_step_deg = sampling.relion_angular_sampling_deg(search.healpix_order, adaptive_oversampling=0)
        if search.oversampling_order == 0 and _precompute_exact_local_fine_grid_enabled(
            search.healpix_order, symmetry=search.symmetry,
        ):
            rotations, rotation_eulers, mstep_rotations = sampling.exact_local_fine_grid(
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
                logger.info(
                    "RELION local search: expanding selected coarse parents by oversampling_order=%d",
                    int(search.oversampling_order),
                )
                parent_order = search.healpix_order - int(search.oversampling_order)
                coarse_angular_step_deg = healpix_angular_step(coarse_size_healpix_order)
                # --strict_highres_exp: pass 1 scores at the E-step cap, as pass 2 does.
                first_pixel_size, first_box = optics.first_optics_group_geometry()
                coarse_image_window_size = image_window_size if strict_pass1 else relion_local_pass1_current_size(
                    pre_update_healpix_order=coarse_size_healpix_order,
                    pixel_size=first_pixel_size,
                    box_size=first_box,
                    particle_diameter=particle_diameter_angstrom,
                    current_size=image_window_size,
                )
                logger.info(
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
        mstep_rotations = sampling.local_search_mstep_rotations(
            grid.mstep_rotations, grid.rotation_eulers, search.healpix_order, symmetry=search.symmetry,
        )
        deferred_perturbation = 0.0
        angular_step_deg = None
    logger.info(
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
        wsum_current_size=wsum_current_size,
    )


def prepare_final_local_sampling(
    search: LocalSearchSettings,
    optics: RunOptics,
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
        first_pixel_size, first_box = optics.first_optics_group_geometry()
        coarse_image_window_size = relion_local_pass1_current_size(
            pre_update_healpix_order=search.parent_order,
            pixel_size=first_pixel_size,
            box_size=first_box,
            particle_diameter=particle_diameter_angstrom,
            current_size=image_window_size,
        )
    else:
        coarse_image_window_size = image_window_size
        coarse_angular_step_deg = None

    if search.oversampling_order == 0 and _precompute_exact_local_fine_grid_enabled(search.healpix_order, symmetry=search.symmetry):
        fine_rotations, _, fine_mstep_rotations = sampling.exact_local_fine_grid(
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
    windows = plan_expectation_windows(
        current_size, optics, log=logger, strict_highres_exp_angstrom=options.adaptive.strict_highres_exp_angstrom
    )
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
            image_window_size=windows.score_window_size,
            model_support_size=windows.engine_model_window_size,
            strict_pass1=windows.score_size is not None,
            wsum_current_size=windows.wsum_size_for_engine,
            base_healpix_order=coarse_grids.rotation_grid.healpix_order,
            coarse_size_healpix_order=coarse_size_healpix_order,
            perturbation=perturbation,
            particle_diameter_angstrom=options.schedule.particle_diameter_ang,
        )
    else:
        local_sampling = None
    return ExpectationSampling(windows, local_sampling)


def _should_use_adaptive_search(state, options: RefinementOptions, *, use_local: bool, n_rotations: int) -> bool:
    """Keep non-C1 refinement on its supported sparse/x-half route.

    Reads ``state.adaptive_oversampling`` and ``options.symmetry.point_group``.

    Small C1 grids may use the direct dense path. Point-group symmetry cannot:
    symmetry reduction itself can make a valid grid smaller than that cutoff
    (O has 12 coarse rotations at HEALPix order 1, I1 fewer), while its
    scoring and reconstruction still require the adaptive RELION x-half path.
    Ported from final Q 22efd8065.
    """

    if int(state.adaptive_oversampling) <= 0 or bool(use_local):
        return False
    return int(n_rotations) > 16 or str(options.symmetry.point_group).upper() != "C1"


@dataclass(frozen=True, kw_only=True)
class NumberedSamplingPlan:
    """A numbered expectation's sampling, planned from the carried grids, state, perturbation and priors.

    ``coarse_grids`` are the carried grids with the iteration's perturbed translations (the controller installs
    them); ``adaptive_pass1`` the adaptive pass-1 rotations and their device source (None where no pass 1 is
    built); ``direction_log_priors`` each half's direction log priors at ``direction_prior_healpix_order``;
    ``use_adaptive`` whether the halves take the two-pass adaptive route.
    """

    coarse_grids: CoarseGrids
    trial_grid: sampling.TrialGrid
    adaptive_pass1: tuple | None
    sampling_plan: ExpectationSampling
    direction_prior_healpix_order: int
    direction_log_priors: list
    use_adaptive: bool
    # The input source's choices and the adaptive pass-1 width, installed by the loop after its ports: the
    # rotation ids the scorer reads (None: the grid's own), the angular step and size of the adaptive pass 1
    # (None off the adaptive route) and the translations a replayed sampling centres the local translation
    # prior on (None natively).
    scoring_rotation_ids: object = None
    coarse_angular_step_deg: float | None = None
    coarse_cs: int | None = None
    replay_prior_translations: object = None


def plan_numbered_sampling(
    ctx: RunContext,
    carry: IterationCarry,
    options: RefinementOptions,
    *,
    first_iteration,
    replayed_sampling_healpix_order: int | None,
    coarse_size_healpix_order: int,
    current_size: int,
    sealed_sampling_state,
    log,
) -> NumberedSamplingPlan:
    """Plan a numbered expectation's sampling: the perturbed trial grid, the adaptive pass-1 rotations, the
    expectation windows and local sampling, the direction log priors and the adaptive route.

    Reads from ``carry``: ``coarse_grids``, ``state`` (its sampling, local-search and oversampling fields),
    ``random_perturbation`` (already this iteration's) and ``direction_priors``; from ``ctx``: ``optics`` and
    ``scoring_dtype``. ``replayed_sampling_healpix_order`` is the order of a replayed sampling (its angular step
    then scales the perturbation; None: the grid's, when ``parity.perturb_factor`` is on);
    ``sealed_sampling_state`` a frozen boundary's captured sampling (None natively).
    """
    use_local = carry.state.do_local_search
    # The HEALPix order whose angular step scales the perturbation: a replayed sampling's (RELION's grid
    # order; the run's may be capped at the exhaustive-grid order), else the grid's; None: none applies.
    perturbation_order = (
        replayed_sampling_healpix_order
        if replayed_sampling_healpix_order is not None
        else carry.coarse_grids.rotation_grid.healpix_order if options.parity.perturb_factor > 0 else None
    )
    trial_grid = iteration_trial_grid(
        carry.coarse_grids, carry.state, options, carry.random_perturbation, perturbation_order=perturbation_order,
        sealed_grid=sealed_sampling_state is not None, dtype=ctx.scoring_dtype,
    )
    coarse_grids = replace(carry.coarse_grids, translations=trial_grid.translations)
    adaptive_pass1 = None
    if builds_coarse_pass1_rotations(carry.state, options, first_iteration, use_local=use_local):
        adaptive_pass1 = coarse_pass1_rotations(
            coarse_grids.rotation_grid, carry.random_perturbation, options,
            perturbation_order=perturbation_order, dtype=ctx.scoring_dtype, log=log,
        )
    # First-iteration CC scores the full translation grid before choosing
    # its single winning pose (ml_optimiser.cpp:9181-9207).
    sampling_plan = plan_expectation_sampling(
        trial_grid, coarse_grids, carry.state, options, ctx.optics, current_size=current_size,
        use_local=use_local, perturbation=carry.random_perturbation,
        coarse_size_healpix_order=coarse_size_healpix_order,
    )
    direction_prior_healpix_order = direction_prior_healpix_order_for_scoring(
        carry.state, use_local=use_local, grid_healpix_order=coarse_grids.rotation_grid.healpix_order,
        local_search_order=sampling_plan.local.search.healpix_order if use_local else None,
    )
    return NumberedSamplingPlan(
        coarse_grids=coarse_grids,
        trial_grid=trial_grid,
        adaptive_pass1=adaptive_pass1,
        sampling_plan=sampling_plan,
        direction_prior_healpix_order=direction_prior_healpix_order,
        direction_log_priors=relion_direction_log_priors(
            carry.direction_priors, options, use_local=use_local, scoring_healpix_order=direction_prior_healpix_order,
            sealed_sampling_state=sealed_sampling_state, dtype=ctx.scoring_dtype, log=log,
        ),
        use_adaptive=_should_use_adaptive_search(
            carry.state, options, use_local=use_local, n_rotations=trial_grid.rotations.shape[0],
        ),
    )
