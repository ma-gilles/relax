"""Host-side plans for one standard refinement iteration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from relax import sampling
from relax.dense.scoring_policy import _dense_global_scoring_dtype
from relax.diagnostics.frozen_boundary import _restore_diagnostic_frozen_boundary_state
from relax.diagnostics.relion_replay import (
    _perturbation_restart_state_iteration,
    _resolve_replay_random_perturbation,
    _restore_convergence_state_from_replay_restart,
    _sealed_sampling_base_grids,
)
from relax.helpers.convergence import RefinementState, healpix_angular_step
from relax.helpers.fourier_window import quantize_current_size
from relax.helpers.resolution import (
    ImageGeometry,
    _bootstrap_current_size_relion,
    _firstiter_cc_scheduling_resolution_shell,
    _truncate_data_vs_prior_for_current_size,
    _truncate_fsc_for_current_size_growth,
    bootstrap_current_size_from_ini_high_relion,
    clamp_relion_coarse_image_size,
    compute_coarse_image_size,
    initialize_resolution_from_firstiter_ini_high,
    initialize_resolution_from_fsc,
    initialize_resolution_from_ini_high,
    relion_optics_image_current_sizes,
)
from relax.reconstruction.regularization_relion import (
    compute_current_size_relion,
    fsc_to_relion_ssnr,
    resolution_from_data_vs_prior,
    update_relion_growth_state_from_fsc,
)
from relax.refinement.iteration_snapshot import validate_resume_snapshot as _validate_resume_snapshot

if TYPE_CHECKING:
    from relax.refinement.refinement_options import RefinementOptions, RefinementSchedule, RelionParityOptions


def resolve_numbered_perturbation(
    previous_perturbation: float,
    parity: RelionParityOptions,
    *,
    iteration: int,
    init_relion_iteration: int,
    replay_metadata,
    replay_dir: str | None,
    rng,
    log: logging.Logger,
) -> float:
    """Resolve sealed, STAR-replayed or native perturbation, in that order.

    Only native sampling advances the run RNG. Replay may instead reconstruct
    the seeded sequence from a restart iteration. See
    ``docs/math/relion_refinement_algorithm.md#2-sampling-grids-and-units``.
    """
    if replay_metadata is not None:
        if replay_metadata.get("sealed_v3", False):
            perturbation = float(replay_metadata["random_perturbation"])
            source = "sealed_frozen_boundary_v3"
        else:
            relion_iteration = init_relion_iteration + iteration + 1
            restart_iteration = _perturbation_restart_state_iteration(
                parity.perturb_replay_restart_state_iterations, relion_iteration,
            )
            perturbation, source = _resolve_replay_random_perturbation(
                star_value=float(replay_metadata["random_perturbation"]),
                perturbation_factor=float(replay_metadata["perturbation_factor"]),
                relion_iteration=relion_iteration,
                replay_dir=str(replay_dir),
                replay_prefix=parity.perturb_replay_relion_prefix,
                explicit_seed=parity.perturb_seed,
                precision_mode=str(parity.perturb_replay_precision),
                restart_state_iteration=restart_iteration,
            )
        log.info(
            "Perturbation replay: iter=%d rp=%+.12g pf=%.3f relion_hp_order=%d source=%s",
            iteration + 1, perturbation,
            float(replay_metadata["perturbation_factor"]),
            int(replay_metadata["healpix_order"]), source,
        )
        return perturbation
    if not parity.perturb_factor > 0:
        return previous_perturbation
    relion_iteration = init_relion_iteration + iteration + 1
    perturbation, seed = sampling._advance_relion_perturbation(
        previous_perturbation,
        perturb_factor=parity.perturb_factor,
        perturb_seed=parity.perturb_seed,
        relion_iteration=relion_iteration,
        rng=rng,
    )
    if seed is not None:
        log.info(
            "Perturbation advance: iter=%d relion_iter=%d seed=%d rp=%+.5f",
            iteration + 1, relion_iteration, seed, perturbation,
        )
    else:
        log.info("Perturbation advance: iter=%d rp=%+.5f", iteration + 1, perturbation)
    return perturbation


@dataclass(frozen=True, kw_only=True)
class ExpectationWindows:
    """Model and particle Fourier widths for one numbered expectation."""

    model_size: int
    image_size: int
    image_box_size: int

    @property
    def image_window_size(self) -> int | None:
        return self.image_size if self.image_size < self.image_box_size else None

    @property
    def model_window_size(self) -> int | None:
        # Engines otherwise reuse the image cutoff for a None model window.
        if self.model_size < self.image_box_size or self.model_size != self.image_size:
            return self.model_size
        return None


def plan_expectation_windows(
    current_size: int,
    image_geometry: ImageGeometry,
    *,
    model_pixel_size: float,
    optics_image_sizes,
    optics_pixel_sizes,
    log: logging.Logger,
) -> ExpectationWindows:
    """Remap single-shape optics while keeping model support independent.

    Shape-class callers omit optics image sizes: each class remaps its own
    support. See ``docs/math/relion_refinement_algorithm.md#5-accumulation-reconstruction-and-parameter-updates``.
    """
    image_size = current_size
    if optics_image_sizes is not None:
        remapped = relion_optics_image_current_sizes(
            current_size,
            model_ori_size=image_geometry.box_size,
            model_pixel_size=model_pixel_size,
            optics_image_sizes=optics_image_sizes,
            optics_pixel_sizes=optics_pixel_sizes,
        )
        unique_sizes = np.unique(remapped)
        if unique_sizes.size != 1:
            raise NotImplementedError(
                "K=1 parity currently requires all optics groups to share one remapped "
                f"image current size; got {remapped.tolist()}",
            )
        image_size = int(unique_sizes[0])
    if image_size != current_size:
        log.info(
            "RELION optics current-size remap: model_current_size=%d "
            "image_current_size=%d model_pixel_size=%.9g",
            current_size, image_size, model_pixel_size,
        )
    return ExpectationWindows(
        model_size=current_size,
        image_size=image_size,
        image_box_size=image_geometry.box_size,
    )


@dataclass(frozen=True, kw_only=True)
class CoarseImageSize:
    """Pass-1 width and the pre-update angular step used to size it."""

    size: int
    angular_step_deg: float


def plan_adaptive_image_size(
    pre_update_healpix_order: int,
    windows: ExpectationWindows,
    image_geometry: ImageGeometry,
    *,
    particle_diameter_angstrom: float | None,
    optics_image_sizes,
    optics_pixel_sizes,
    sealed_sampling_state,
    log: logging.Logger,
) -> CoarseImageSize:
    """Size pass 1 from incoming sampling, then admit an exact sealed width.

    The fine grid can already have advanced to another order. See
    ``docs/math/relion_refinement_algorithm.md#5-accumulation-reconstruction-and-parameter-updates``.
    """
    angular_step_deg = healpix_angular_step(pre_update_healpix_order)
    coarse_size = compute_coarse_image_size(
        angular_step_deg,
        float(optics_pixel_sizes[0]) if optics_pixel_sizes is not None else image_geometry.pixel_size_angstrom,
        int(optics_image_sizes[0]) if optics_image_sizes is not None else image_geometry.box_size,
        particle_diameter=particle_diameter_angstrom,
    )
    coarse_size = clamp_relion_coarse_image_size(
        coarse_size,
        windows.image_size if windows.image_window_size is not None else None,
        image_geometry.box_size,
    )
    if sealed_sampling_state is not None:
        coarse_size = int(sealed_sampling_state["coarse_size"])
        if coarse_size > windows.model_size:
            raise ValueError(
                "sealed sampling coarse_size exceeds active current_size: "
                f"coarse={coarse_size} current={windows.model_size}"
            )
        log.info("Frozen-boundary v3 directly owns adaptive pass-1 coarse_size=%d", coarse_size)
    return CoarseImageSize(size=coarse_size, angular_step_deg=angular_step_deg)


def initialize_refinement_state(
    options: RefinementOptions,
    image_geometry: ImageGeometry,
    *,
    subtomogram: bool,
    dtype,
) -> RefinementState:
    """Resolve startup sampling/convergence state before initial grid construction.

    Replay restart takes precedence over fresh FSC/ini_high initialization;
    diagnostic frozen fields and a validated continuation are applied afterwards.
    See ``docs/math/relion_refinement_algorithm.md#startup-sampling-state``.
    """
    schedule = options.schedule
    parity = options.parity
    init_relion_iteration = schedule.init_relion_iteration
    n_classes = int(options.k_class.n_classes)
    grid_size = image_geometry.box_size
    state = RefinementState(
        iteration=0,
        healpix_order=schedule.init_healpix_order,
        adaptive_oversampling=options.adaptive.adaptive_oversampling,
        translation_range=schedule.init_translation_range,
        translation_step=schedule.init_translation_step,
        max_healpix_order=schedule.max_healpix_order,
        auto_local_healpix_order=options.local_search.auto_local_healpix_order,
        # Class3D (K>1) never switches to local searches from the HEALPix order.
        auto_sampling=not (n_classes > 1),
        current_resolution=float("inf"),
        voxel_size_angstrom=image_geometry.pixel_size_angstrom,
        particle_diameter_angstrom=float(schedule.particle_diameter_ang or 0.0),
        subtomogram=subtomogram,
    )
    # RELION's convergence counters are not initialized against an infinite
    # previous resolution.  They resume from the previous optimiser/model STAR
    # in replay mode, or from the initial FSC/ini_high state in a fresh run.
    if (
        options.debug.sealed_sampling_state is None
        and parity.perturb_replay_relion_dir is not None
        and int(init_relion_iteration) > 0
    ):
        _restore_convergence_state_from_replay_restart(state, options)
    elif schedule.init_fsc is not None:
        initialize_resolution_from_fsc(
            state, options, grid_size=grid_size, voxel_size=image_geometry.pixel_size_angstrom,
            dtype=dtype,
        )
    elif init_relion_iteration == 0 and parity.relion_firstiter_ini_high_angstrom is not None:
        initialize_resolution_from_firstiter_ini_high(state, options, grid_size=grid_size, voxel_size=image_geometry.pixel_size_angstrom)
    elif init_relion_iteration == 0 and schedule.ini_high_angstrom is not None:
        initialize_resolution_from_ini_high(
            state, schedule.ini_high_angstrom, grid_size=grid_size, voxel_size=image_geometry.pixel_size_angstrom
        )
    if options.replay.init_refinement_state_fields is not None:
        _restore_diagnostic_frozen_boundary_state(state, options)
    # A continuation (RELION --continue) starts from the run files' sampling state; the
    # rest of its snapshot is installed just before the loop.
    resume = options.checkpoint.resume
    if resume is not None:
        _validate_resume_snapshot(
            resume,
            init_relion_iteration=init_relion_iteration,
            n_classes=n_classes,
            grid_size=grid_size,
            options=options,
        )
        state = resume.refinement_state(state)
    return state


@dataclass(frozen=True)
class InitialCoarseGrids:
    """Exhaustive coarse trial grid selected for refinement startup."""

    rotation_grid: sampling.RotationGrid
    base_translations: np.ndarray
    translations: jnp.ndarray


def build_initial_coarse_grids(
    healpix_order,
    translations,
    *,
    translation_range,
    translation_step,
    n_classes,
    voxel_size,
    symmetry="C1",
) -> InitialCoarseGrids:
    """Pair RELION's canonical initial rotations with its translation grid."""

    dtype = _dense_global_scoring_dtype()
    rotation_grid = sampling.relion_scoring_rotation_grid(
        healpix_order,
        dtype=dtype,
        symmetry=symmetry,
    )
    if translations is None:
        translations = sampling._relion_base_translation_grid(
            translation_range,
            translation_step,
            n_classes=n_classes,
            voxel_size=voxel_size,
        )
    base_translations = np.asarray(translations, dtype=np.float64)
    return InitialCoarseGrids(
        rotation_grid=rotation_grid,
        base_translations=base_translations,
        translations=jnp.asarray(translations, dtype=dtype),
    )


def build_sealed_initial_coarse_grids(
    sealed_sampling_state,
    *,
    initialized_healpix_order,
    voxel_size,
    symmetry: str,
    log: logging.Logger,
) -> InitialCoarseGrids:
    """Materialize and validate a schema-v3 sealed initial sampling grid."""

    rotations, rotation_eulers, current_translations = _sealed_sampling_base_grids(
        sealed_sampling_state,
        voxel_size_angstrom=voxel_size,
        dtype=_dense_global_scoring_dtype(),
    )
    healpix_order = int(sealed_sampling_state["healpix_order_original"])
    if healpix_order != int(initialized_healpix_order):
        raise ValueError(
            "sealed sampling HEALPix order does not match initialized boundary: "
            f"sealed={healpix_order} init={initialized_healpix_order}"
        )
    log.info(
        "Frozen-boundary v3 directly materialized %d Euler rows and %d translations",
        int(rotation_eulers.shape[0]),
        int(current_translations.shape[0]),
    )
    return InitialCoarseGrids(
        rotation_grid=sampling.RotationGrid(
            rotations=rotations, rotation_eulers=rotation_eulers,
            healpix_order=healpix_order, symmetry=symmetry,
        ),
        base_translations=np.asarray(current_translations, dtype=np.float64),
        translations=current_translations,
    )


@dataclass(frozen=True)
class ImageSizeUpdate:
    """Initial image width and any FSC-derived scheduling/growth update."""

    size: int
    data_vs_prior: np.ndarray | None
    incr_size: int
    has_high_fsc_at_limit: bool


@dataclass(frozen=True)
class HalfmapImageSize(ImageSizeUpdate):
    """Split-half growth decision, including its resolution and pre-quantized size."""

    resolution_shell: int
    raw_size: int


@dataclass(frozen=True)
class ClassImageSize:
    """Class3D growth decision with the class curves used by its diagnostic capture."""

    size: int
    resolution_shell: int
    raw_size: int
    resolution_shells_per_class: np.ndarray
    raw_data_vs_prior: np.ndarray
    data_vs_prior: np.ndarray


def plan_initial_image_size(
    schedule: RefinementSchedule,
    *,
    parity: RelionParityOptions,
    grid_size,
    pixel_size_angstrom,
    incr_size,
    has_high_fsc_at_limit,
    dtype,
    log: logging.Logger,
) -> ImageSizeUpdate:
    """Resolve startup ini_high, initial FSC or bootstrap width in that order.

    See ``docs/math/relion_refinement_algorithm.md`` for image-size scheduling.
    """
    if schedule.init_relion_iteration == 0:
        seeded_cs = bootstrap_current_size_from_ini_high_relion(
            grid_size,
            pixel_size_angstrom,
            parity.relion_firstiter_ini_high_angstrom,
            incr_size=incr_size,
        )
    else:
        seeded_cs = None
    if seeded_cs is not None:
        current_size = int(seeded_cs)
        data_vs_prior_iter = None
        log.info(
            "RELION init bootstrap: seeding iter-1 current_size from ini_high=%.2f A -> %d",
            float(parity.relion_firstiter_ini_high_angstrom),
            current_size,
        )
    elif schedule.init_fsc is not None:
        prev_cs = int(schedule.init_current_size)
        fsc_prev = _truncate_fsc_for_current_size_growth(
            schedule.init_fsc,
            current_size=prev_cs,
            grid_size=grid_size,
            dtype=dtype,
        )
        data_vs_prior_iter = np.asarray(
            fsc_to_relion_ssnr(fsc_prev, tau2_fudge=parity.tau2_fudge),
        )
        res_shell = resolution_from_data_vs_prior(
            data_vs_prior_iter,
            ori_size=grid_size,
            allow_high_res_recovery=True,
        )
        incr_size, has_high_fsc_at_limit = update_relion_growth_state_from_fsc(
            fsc_prev,
            prev_cs,
            incr_size=incr_size,
            has_high_fsc_at_limit=has_high_fsc_at_limit,
        )
        _init_pmax = float(schedule.init_ave_Pmax) if schedule.init_ave_Pmax is not None else 0.0
        raw_cs = compute_current_size_relion(
            res_shell,
            grid_size,
            ave_Pmax=_init_pmax,
            has_high_fsc_at_limit=has_high_fsc_at_limit,
            incr_size=incr_size,
        )
        current_size = quantize_current_size(raw_cs, ori_size=grid_size)
    else:
        current_size = _bootstrap_current_size_relion(schedule.init_current_size, grid_size)
        data_vs_prior_iter = None

    return ImageSizeUpdate(
        size=current_size, data_vs_prior=data_vs_prior_iter,
        incr_size=incr_size, has_high_fsc_at_limit=has_high_fsc_at_limit,
    )


def plan_class_image_size(
    data_vs_prior,
    *,
    previous_size,
    grid_size,
    pixel_size_angstrom,
    incr_size,
    ave_pmax,
    completed_relion_iteration,
    parity: RelionParityOptions,
    dtype,
    log: logging.Logger,
) -> ClassImageSize:
    """Plan Class3D support from its best class's truncated prior curve.

    See ``docs/math/relion_refinement_algorithm.md`` for image-size scheduling.
    """
    if data_vs_prior is None:
        raise RuntimeError("K-class current-size scheduling requires a previous data_vs_prior curve")
    data_vs_prior_prev_raw = np.asarray(
        data_vs_prior,
        dtype=dtype,
    ).copy()
    data_vs_prior_prev = _truncate_data_vs_prior_for_current_size(
        data_vs_prior_prev_raw,
        current_size=previous_size,
        grid_size=grid_size,
        dtype=dtype,
    )
    per_class_res_shell = np.asarray(
        [
            resolution_from_data_vs_prior(dvp_class, ori_size=grid_size, allow_high_res_recovery=False)
            for dvp_class in np.asarray(data_vs_prior_prev)
        ],
        dtype=np.int32,
    )
    res_shell = int(np.max(per_class_res_shell))
    scheduling_res_shell = _firstiter_cc_scheduling_resolution_shell(
        res_shell,
        emulate_relion_firstiter_cc=parity.emulate_relion_firstiter_cc,
        ini_high_angstrom=parity.relion_firstiter_ini_high_angstrom,
        relion_iteration=completed_relion_iteration,
        grid_size=grid_size,
        voxel_size=pixel_size_angstrom,
    )
    if scheduling_res_shell != res_shell:
        res_shell = scheduling_res_shell
        log.info(
            "RELION firstiter_cc scheduling: using ini_high=%.2f A shell %d "
            "for next K-class current_size",
            float(parity.relion_firstiter_ini_high_angstrom),
            int(res_shell),
        )
    raw_cs = compute_current_size_relion(
        res_shell,
        grid_size,
        ave_Pmax=ave_pmax,
        has_high_fsc_at_limit=False,
        incr_size=incr_size,
    )
    computed_cs = quantize_current_size(raw_cs, ori_size=grid_size)

    return ClassImageSize(
        size=computed_cs, resolution_shell=res_shell, raw_size=raw_cs,
        resolution_shells_per_class=per_class_res_shell,
        raw_data_vs_prior=data_vs_prior_prev_raw, data_vs_prior=data_vs_prior_prev,
    )


def plan_halfmap_image_size(
    fsc_history,
    *,
    growth_fsc_history,
    restart,
    data_vs_prior,
    previous_size,
    grid_size,
    pixel_size_angstrom,
    incr_size,
    has_high_fsc_at_limit,
    ave_pmax,
    completed_relion_iteration,
    parity: RelionParityOptions,
    dtype,
    log: logging.Logger,
) -> HalfmapImageSize:
    """Plan split-half support from the previous iteration's curve and RELION's growth latch.

    ``data_vs_prior`` is the curve the previous iteration published (or the
    continued snapshot's); the raw FSC is read only as the growth fallback.

    See ``docs/math/relion_refinement_algorithm.md`` for image-size scheduling.
    """
    fsc_prev_raw = np.asarray(
        fsc_history[-1] if fsc_history else restart.fsc,
        dtype=dtype,
    ).copy()
    fsc_prev_for_growth = _truncate_fsc_for_current_size_growth(
        growth_fsc_history[-1]
        if growth_fsc_history
        else (fsc_prev_raw if restart is None or restart.fsc_for_growth is None else restart.fsc_for_growth),
        current_size=previous_size,
        grid_size=grid_size,
        dtype=dtype,
    )

    data_vs_prior_iter = _truncate_data_vs_prior_for_current_size(
        data_vs_prior,
        current_size=previous_size,
        grid_size=grid_size,
        dtype=dtype,
    )
    res_shell = resolution_from_data_vs_prior(
        data_vs_prior_iter,
        ori_size=grid_size,
        allow_high_res_recovery=True,
    )
    incr_size, has_high_fsc_at_limit = update_relion_growth_state_from_fsc(
        fsc_prev_for_growth,
        previous_size,
        incr_size=incr_size,
        has_high_fsc_at_limit=has_high_fsc_at_limit,
    )
    scheduling_res_shell = _firstiter_cc_scheduling_resolution_shell(
        res_shell,
        emulate_relion_firstiter_cc=parity.emulate_relion_firstiter_cc,
        ini_high_angstrom=parity.relion_firstiter_ini_high_angstrom,
        relion_iteration=completed_relion_iteration,
        grid_size=grid_size,
        voxel_size=pixel_size_angstrom,
    )
    if scheduling_res_shell != res_shell:
        res_shell = scheduling_res_shell
        log.info(
            "RELION firstiter_cc scheduling: using ini_high=%.2f A shell %d for next current_size",
            float(parity.relion_firstiter_ini_high_angstrom),
            int(res_shell),
        )

    raw_cs = compute_current_size_relion(
        res_shell,
        grid_size,
        ave_Pmax=ave_pmax,
        has_high_fsc_at_limit=has_high_fsc_at_limit,
        incr_size=incr_size,
    )
    current_size = quantize_current_size(raw_cs, ori_size=grid_size)

    return HalfmapImageSize(
        size=current_size, data_vs_prior=data_vs_prior_iter,
        incr_size=incr_size, has_high_fsc_at_limit=has_high_fsc_at_limit,
        resolution_shell=res_shell, raw_size=raw_cs,
    )
