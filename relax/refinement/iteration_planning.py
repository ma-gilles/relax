"""Host-side plans for one standard refinement iteration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
import numpy as np

from relax import sampling
from relax.dense import scoring_policy
from relax.dense.scoring_policy import _dense_global_scoring_dtype
from relax.diagnostics.frozen_boundary import _restore_diagnostic_frozen_boundary_state
from relax.diagnostics.relion_replay import (
    _perturbation_restart_state_iteration,
    _resolve_replay_random_perturbation,
    _restore_convergence_state_from_replay_restart,
    _sealed_sampling_base_grids,
)
from relax.helpers.convergence import RefinementState, _exhaustive_grid_order_for_state, healpix_angular_step
from relax.helpers.fourier_window import quantize_current_size
from relax.helpers.resolution import (
    ImageGeometry,
    _bootstrap_current_size_relion,
    _firstiter_cc_scheduling_resolution_shell,
    _zero_shells_past_current_size,
    bootstrap_current_size_from_ini_high_relion,
    clamp_relion_coarse_image_size,
    class_resolution_shells,
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
from relax.sampling import _relion_adaptive_pass1_rotations

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


class FirstIterationPolicy(NamedTuple):
    """How one numbered iteration scores and reconstructs: only a run's first iteration departs."""

    relion_firstiter_cc: bool
    score_mode: str
    winner_take_all: bool


def first_iteration_policy(parity: RelionParityOptions, *, init_relion_iteration, iteration) -> FirstIterationPolicy:
    """RELION's first-iteration CC emulation, the score mode and whether one pose takes all the weight.

    Normalised cross-correlation scoring and hard reconstruction apply to iteration 1 of a run that
    starts at RELION iteration 0, under ``--firstiter_cc`` emulation or the two diagnostic options.
    """
    relion_firstiter_cc_this_iter = bool(
        parity.emulate_relion_firstiter_cc and init_relion_iteration == 0 and iteration == 0
    )
    first_iter_normalized_cc_this_iter = bool(
        parity.first_iteration_score_mode == "normalized_cc" and init_relion_iteration == 0 and iteration == 0
    )
    first_iter_hard_reconstruction_this_iter = bool(
        parity.first_iteration_reconstruction_mode == "hard" and init_relion_iteration == 0 and iteration == 0
    )
    firstiter_score_mode_this_iter = (
        "normalized_cc" if (relion_firstiter_cc_this_iter or first_iter_normalized_cc_this_iter) else "gaussian"
    )
    firstiter_winner_take_all_this_iter = bool(
        relion_firstiter_cc_this_iter or first_iter_hard_reconstruction_this_iter
    )
    return FirstIterationPolicy(
        relion_firstiter_cc_this_iter, firstiter_score_mode_this_iter, firstiter_winner_take_all_this_iter,
    )


@dataclass(frozen=True)
class CoarseGrids:
    """The exhaustive coarse trial grid: built at start-up, rebuilt by ``refresh_coarse_grids``.

    ``translations`` is the device grid the expectation scores; the numbered loop replaces it with the
    iteration's perturbed copy, while ``base_translations`` keeps the unperturbed host coordinates.
    """

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
) -> CoarseGrids:
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
    return CoarseGrids(
        rotation_grid=rotation_grid,
        base_translations=base_translations,
        translations=jnp.asarray(translations, dtype=dtype),
    )


def refresh_coarse_grids(
    grids: CoarseGrids,
    state: RefinementState,
    *,
    n_classes,
    voxel_size,
    symmetry: str,
    dtype,
    replay_translations: bool,
    log: logging.Logger,
) -> CoarseGrids:
    """The exhaustive coarse grids of ``state``'s sampling, rebuilt where they changed.

    A new HEALPix order rebuilds the rotation grid (up to the exhaustive-grid cap) and the translation
    grid. With ``replay_translations`` a replayed translation range or step rebuilds the translation
    grid alone. ``grids.translations`` may be a perturbed copy; a rebuild replaces it with the base grid.
    """
    current_rotation_grid = grids.rotation_grid
    base_translations = grids.base_translations
    current_translations = grids.translations
    if state.healpix_order != current_rotation_grid.healpix_order:
        new_order = _exhaustive_grid_order_for_state(state)
        if new_order != current_rotation_grid.healpix_order:
            log.info(
                "Regenerating rotation grid: order %d -> %d",
                current_rotation_grid.healpix_order,
                new_order,
            )
            current_rotation_grid = sampling.relion_scoring_rotation_grid(
                new_order, dtype=dtype,
                symmetry=symmetry,
            )
        else:
            log.info(
                "Angular step refined to order %d (exhaustive grid stays at order %d — local search handles finer sampling)",
                state.healpix_order,
                current_rotation_grid.healpix_order,
            )

        # Regenerate translation grid based on updated parameters
        base_translations = sampling._relion_base_translation_grid(
            state.translation_range,
            state.translation_step,
            n_classes=n_classes,
            voxel_size=voxel_size,
        )
        current_translations = jnp.asarray(base_translations, dtype=dtype)
        log.info(
            "New grid: %d rotations, %d translations (range=%.1f, step=%.1f)",
            current_rotation_grid.rotations.shape[0],
            current_translations.shape[0],
            state.translation_range,
            state.translation_step,
        )
    elif replay_translations:
        # Translation params may have changed under replay without an
        # hp_order bump. Regenerate the translation grid to match RELION.
        _new_t_source = sampling._relion_base_translation_grid(
            state.translation_range,
            state.translation_step,
            n_classes=n_classes,
            voxel_size=voxel_size,
        )
        _new_t = jnp.asarray(_new_t_source, dtype=dtype)
        if _new_t.shape != base_translations.shape or not jnp.allclose(
            _new_t,
            np.asarray(base_translations, dtype=dtype),
        ):
            current_translations = _new_t
            base_translations = _new_t_source
            log.info(
                "Replay: regenerated translation grid: %d translations (range=%.2f px, step=%.2f px)",
                current_translations.shape[0],
                state.translation_range,
                state.translation_step,
            )
    return CoarseGrids(current_rotation_grid, base_translations, current_translations)


def perturbed_trial_grid(
    rotation_eulers,
    base_translations,
    random_perturbation: float,
    *,
    grid_healpix_order: int,
    replay_metadata,
    translation_step,
    use_grid_eulers: bool,
    symmetry: str,
    dtype,
) -> sampling.TrialGrid:
    """This iteration's trial grid: the coarse grid under RELION's sampling perturbation.

    ``rotation_eulers`` are the coarse grid's working Euler rows and ``base_translations`` its
    unperturbed host translations. The angular sampling that scales the perturbation is the grid's
    HEALPix order, or the replayed order when ``replay_metadata`` supplies one.
    """
    # Use RELION's actual hp_order when replaying (recovar's current
    # grid order may be capped at MAX_FULL_GRID_ORDER=4 for memory).
    _angsamp_order = int(replay_metadata["healpix_order"]) if replay_metadata is not None else grid_healpix_order
    angsamp_deg = sampling.relion_angular_sampling_deg(_angsamp_order, adaptive_oversampling=0)
    return sampling._perturbed_trial_grid(
        rotation_eulers=rotation_eulers,
        mstep_source_eulers=sampling._relion_mstep_source_eulers(
            rotation_eulers,
            _angsamp_order,
            use_grid_eulers=use_grid_eulers,
            symmetry=symmetry,
        ),
        base_translations=base_translations,
        translation_step=float(translation_step),
        random_perturbation=random_perturbation,
        angular_sampling_deg=angsamp_deg,
        dtype=dtype,
    )


def coarse_pass1_rotations(
    source_eulers,
    random_perturbation: float,
    *,
    grid_healpix_order: int,
    replay_metadata,
    perturb_factor: float,
    log: logging.Logger,
):
    """RELION's device-built rotations for the pass-1 coarse scorer, or None where the host grid serves.

    ``source_eulers`` are the unperturbed coarse Euler rows (float64). The perturbation applies only
    when the trial grid is perturbed: under replay, or with a positive ``perturb_factor``.
    """
    adaptive_pass1_order = (
        int(replay_metadata["healpix_order"])
        if replay_metadata is not None
        else int(grid_healpix_order)
    )
    adaptive_pass1_use_float64 = bool(scoring_policy.DENSE_PRECISION.use_float64_scoring)
    adaptive_pass1_rotations = _relion_adaptive_pass1_rotations(
        source_eulers,
        random_perturbation if (replay_metadata is not None or perturb_factor > 0) else 0.0,
        sampling.relion_angular_sampling_deg(adaptive_pass1_order, adaptive_oversampling=0),
        use_float64=adaptive_pass1_use_float64,
    )
    if adaptive_pass1_rotations is not None:
        log.info(
            "RELION adaptive pass 1: using %s-built coarse scorer rotations; "
            "fine/M-step rotations remain host-generated",
            "double-precision CUDA" if adaptive_pass1_use_float64 else "CUDA",
        )
    return adaptive_pass1_rotations


def build_sealed_initial_coarse_grids(
    sealed_sampling_state,
    *,
    initialized_healpix_order,
    voxel_size,
    symmetry: str,
    log: logging.Logger,
) -> CoarseGrids:
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
    return CoarseGrids(
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
        fsc_prev = _zero_shells_past_current_size(
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
    data_vs_prior_prev_raw = np.asarray(
        data_vs_prior,
        dtype=dtype,
    ).copy()
    data_vs_prior_prev = _zero_shells_past_current_size(
        data_vs_prior_prev_raw,
        current_size=previous_size,
        grid_size=grid_size,
        dtype=dtype,
    )
    per_class_res_shell = np.asarray(
        class_resolution_shells(data_vs_prior_prev, grid_size=grid_size),
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
    fsc_prev_for_growth = _zero_shells_past_current_size(
        growth_fsc_history[-1]
        if growth_fsc_history
        else (fsc_prev_raw if restart is None or restart.fsc_for_growth is None else restart.fsc_for_growth),
        current_size=previous_size,
        grid_size=grid_size,
        dtype=dtype,
    )

    data_vs_prior_iter = _zero_shells_past_current_size(
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
