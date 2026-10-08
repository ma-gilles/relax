"""Host-side plans for one standard refinement iteration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
import numpy as np

from relax import sampling
from relax.helpers.convergence import RefinementState, _exhaustive_grid_order_for_state, healpix_angular_step
from relax.helpers.fourier_window import quantize_current_size
from relax.helpers.resolution import (
    ImageGeometry,
    _bootstrap_current_size_relion,
    _firstiter_cc_scheduling_resolution_shell,
    _zero_shells_past_current_size,
    bootstrap_current_size_from_ini_high_relion,
    class_resolution_shells,
    initialize_resolution_from_firstiter_ini_high,
    initialize_resolution_from_fsc,
    initialize_resolution_from_ini_high,
    relion_coarse_image_size,
    relion_optics_image_current_sizes,
)
from relax.reconstruction.regularization_relion import (
    compute_current_size_relion,
    fsc_to_relion_ssnr,
    resolution_from_data_vs_prior,
    update_relion_growth_state_from_fsc,
)
from relax.refinement.iteration_snapshot import validate_resume_snapshot
from relax.refinement.ports import InputSource
from relax.sampling import _relion_adaptive_pass1_rotations

if TYPE_CHECKING:
    from relax.refinement.refinement_options import RefinementOptions, RelionParityOptions


def initial_random_perturbation(options: RefinementOptions, *, log: logging.Logger) -> float:
    """A fresh run's sampling perturbation before its first iteration: RELION's value at
    ``schedule.init_relion_iteration`` from ``parity.perturb_seed`` (logged), else 0.0.

    RELION applies a random rigid rotation of the entire SO(3) trial grid at each iteration
    (SamplingPerturbation, healpix_sampling.cpp:167-174): A -> A @ R_perturb with R_perturb =
    R_from_relion([m,m,m]) and m = random_perturbation * angular_sampling; the perturbation advances per
    iteration (``resolve_numbered_perturbation``). A replay reads _rlnSamplingPerturbInstance from RELION's
    sampling.star instead (an input source's).
    """
    if options.parity.perturb_factor > 0 and options.parity.perturb_seed is not None:
        random_perturbation = sampling.relion_sampling_perturbation_for_iteration(
            options.parity.perturb_factor,
            options.parity.perturb_seed,
            options.schedule.init_relion_iteration,
        )
        log.info(
            "Perturbation init: relion_iter=%d random_seed=%d rp=%+.5f",
            int(options.schedule.init_relion_iteration),
            int(options.parity.perturb_seed),
            random_perturbation,
        )
    else:
        random_perturbation = 0.0
    return random_perturbation


def resolve_numbered_perturbation(
    previous_perturbation: float,
    options: RefinementOptions,
    *,
    iteration: int,
    rng,
    log: logging.Logger,
) -> float:
    """The run's own sampling perturbation of this iteration: the native advance of the run RNG.

    A RELION run's or a sealed sampling state's perturbation is an input source's
    (``InputSource.random_perturbation``). Reads from ``options``: ``schedule.init_relion_iteration``;
    ``parity.perturb_factor`` and ``perturb_seed``. See
    ``docs/math/relion_refinement_algorithm.md#2-sampling-grids-and-units``.
    """
    parity = options.parity
    init_relion_iteration = options.schedule.init_relion_iteration
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
class RunOptics:
    """The run's image and optics geometry: built once before the first iteration, never modified.

    ``optics_image_sizes`` and ``optics_pixel_sizes`` are RELION's per-optics-group box and pixel sizes
    (both None when the particles share the model's grid); ``multi_shape_halves`` says the halves hold
    several image shapes, each remapping its own support.
    """

    image_geometry: ImageGeometry
    model_pixel_size: float
    optics_image_sizes: np.ndarray | None
    optics_pixel_sizes: np.ndarray | None
    multi_shape_halves: bool

    def first_optics_group_geometry(self) -> tuple[float, int]:
        """The first optics group's (STAR pixel size, box): the image geometry when groups give none.

        RELION sizes its E-step windows from ``remap_sizes * ori_size * mymodel.pixel_size``, the optics
        group's box times its image pixel size, never the model pixel size (ml_optimiser.cpp:6917-6941).
        """
        return (
            float(self.optics_pixel_sizes[0])
            if self.optics_pixel_sizes is not None
            else self.image_geometry.pixel_size_angstrom,
            int(self.optics_image_sizes[0]) if self.optics_image_sizes is not None else self.image_geometry.box_size,
        )


@dataclass(frozen=True, kw_only=True)
class ExpectationWindows:
    """Model and particle Fourier widths for one numbered expectation."""

    model_size: int
    image_current_size: int
    image_box_size: int
    # --strict_highres_exp: the E-step's image size, below image_current_size; None scores at image_current_size.
    score_size: int | None = None

    @property
    def image_window_size(self) -> int | None:
        return self.image_current_size if self.image_current_size < self.image_box_size else None

    @property
    def score_window_size(self) -> int | None:
        """The E-step's scoring window: the --strict_highres_exp size, else the image window."""
        if self.score_size is None:
            return self.image_window_size
        return self.score_size if self.score_size < self.image_box_size else None

    @property
    def wsum_size_for_engine(self) -> int | None:
        """The weighted sums' image size an engine needs when the E-step scores below it, else None."""
        if self.score_size is not None and self.score_size < self.image_current_size:
            return int(self.image_current_size)
        return None

    @property
    def engine_model_window_size(self) -> int | None:
        """The M-step window an engine reconstructs at; explicit whenever the E-step scores below it."""
        if self.score_size is not None and self.score_size < self.image_current_size:
            return self.model_size
        return self.model_window_size

    @property
    def model_window_size(self) -> int | None:
        # Engines otherwise reuse the image cutoff for a None model window.
        if self.model_size < self.image_box_size or self.model_size != self.image_current_size:
            return self.model_size
        return None


def plan_expectation_windows(
    current_size: int, optics: RunOptics, *, log: logging.Logger, strict_highres_exp_angstrom: float | None = None
) -> ExpectationWindows:
    """Remap single-shape optics while keeping model support independent.

    Shape classes are not remapped here: each class remaps its own support. Reads every field of
    ``optics``. ``strict_highres_exp_angstrom`` (RELION --strict_highres_exp) caps the E-step's size at
    :func:`relion_strict_highres_image_size`. See
    ``docs/math/relion_refinement_algorithm.md#5-accumulation-reconstruction-and-parameter-updates``.
    """
    image_current_size = _optics_image_current_size(current_size, optics)
    if image_current_size != current_size:
        log.info(
            "RELION optics current-size remap: model_current_size=%d "
            "image_current_size=%d model_pixel_size=%.9g",
            current_size, image_current_size, optics.model_pixel_size,
        )
    score_size = None
    if strict_highres_exp_angstrom is not None:
        limit = _strict_highres_limit(optics, strict_highres_exp_angstrom)
        score_size = min(limit, image_current_size)
        log.info(
            "RELION --strict_highres_exp %.3f A: E-step size %d (limit %d, image current size %d)",
            strict_highres_exp_angstrom, score_size, limit, image_current_size,
        )
    return ExpectationWindows(
        model_size=current_size,
        image_current_size=image_current_size,
        image_box_size=optics.image_geometry.box_size,
        score_size=score_size,
    )


def _optics_image_current_size(current_size: int, optics: RunOptics) -> int:
    """The images' current size for the model's ``current_size``: RELION's optics remap on single-shape optics
    groups (which must agree on one size), else ``current_size``."""
    optics_image_sizes = None if optics.multi_shape_halves else optics.optics_image_sizes
    if optics_image_sizes is None:
        return current_size
    remapped = relion_optics_image_current_sizes(
        current_size,
        model_box_size=optics.image_geometry.box_size,
        model_pixel_size=optics.model_pixel_size,
        optics_image_sizes=optics_image_sizes,
        optics_pixel_sizes=optics.optics_pixel_sizes,
    )
    unique_sizes = np.unique(remapped)
    if unique_sizes.size != 1:
        raise NotImplementedError(
            "K=1 parity currently requires all optics groups to share one remapped "
            f"image current size; got {remapped.tolist()}",
        )
    return int(unique_sizes[0])


def _strict_highres_limit(optics: RunOptics, strict_highres_exp_angstrom: float) -> int:
    """The --strict_highres_exp image size of the first optics group (single-shape optics only)."""
    if optics.multi_shape_halves:
        raise NotImplementedError("--strict_highres_exp with optics groups on several image shapes")
    return relion_strict_highres_image_size(*optics.first_optics_group_geometry(), strict_highres_exp_angstrom)


def strict_e_step_size(current_size: int, optics: RunOptics, options: RefinementOptions) -> int | None:
    """The --strict_highres_exp E-step size at this current size (the expected accuracy's size), or None when off."""
    limit = options.adaptive.strict_highres_exp_angstrom
    if limit is None:
        return None
    return min(_strict_highres_limit(optics, limit), _optics_image_current_size(current_size, optics))


def relion_strict_highres_image_size(pixel_size: float, box_size: int, limit_angstrom: float) -> int:
    """RELION's --strict_highres_exp image size, ``2 * ROUND(box * pixel / limit)`` (ml_optimiser.cpp:5755-5758).

    ``remap_sizes * ori_size * pixel_size`` is the optics group's box times its pixel size. ROUND rounds
    half away from zero.
    """
    value = float(box_size) * float(pixel_size) / float(limit_angstrom)
    return 2 * int(np.floor(value + 0.5))


@dataclass(frozen=True, kw_only=True)
class CoarseImageSize:
    """Pass-1 width and the pre-update angular step used to size it."""

    size: int
    angular_step_deg: float


def plan_adaptive_image_size(
    pre_update_healpix_order: int,
    windows: ExpectationWindows,
    optics: RunOptics,
    options: RefinementOptions,
    *,
    log: logging.Logger,
) -> CoarseImageSize:
    """Size pass 1 from incoming sampling (a sealed sampling state's exact width is the input source's,
    ``InputSource.adaptive_coarse_size``).

    The fine grid can already have advanced to another order. Reads from ``optics``: the image geometry
    and the first optics group's box and pixel size; from ``options``: ``schedule.particle_diameter_ang``.
    See ``docs/math/relion_refinement_algorithm.md#5-accumulation-reconstruction-and-parameter-updates``.
    """
    image_geometry = optics.image_geometry
    angular_step_deg = healpix_angular_step(pre_update_healpix_order)
    if windows.score_size is not None:
        # --strict_highres_exp replaces the angular rule: pass 1 scores at the E-step cap as well.
        coarse_size = int(windows.score_size)
    else:
        coarse_size = relion_coarse_image_size(
            angular_step_deg,
            *optics.first_optics_group_geometry(),
            particle_diameter=options.schedule.particle_diameter_ang,
            current_size=windows.image_current_size if windows.image_window_size is not None else None,
            clamp_box_size=image_geometry.box_size,
        )
    return CoarseImageSize(size=coarse_size, angular_step_deg=angular_step_deg)


def initialize_refinement_state(
    options: RefinementOptions,
    image_geometry: ImageGeometry,
    *,
    subtomogram: bool,
    dtype,
    source: InputSource,
) -> RefinementState:
    """Resolve startup sampling/convergence state before initial grid construction.

    A replay restart's convergence counters (``source.restore_convergence_state``, the native source has
    none) take precedence over fresh FSC/ini_high initialization;
    diagnostic frozen fields and a validated continuation are applied afterwards.
    See ``docs/math/relion_refinement_algorithm.md#startup-sampling-state``.
    """
    schedule = options.schedule
    parity = options.parity
    init_relion_iteration = schedule.init_relion_iteration
    # RELION --sigma_ang turns the orientational prior on from iteration 1 at any HEALPix order, with that
    # width for rot, tilt and psi (ml_optimiser.cpp parseInitial); without it a local start uses 2 x step.
    sigma_ang_deg = options.local_search.sigma_ang_deg
    sigma_ang = 0.0 if sigma_ang_deg is None else float(np.deg2rad(sigma_ang_deg))
    state = RefinementState(
        iteration=0,
        healpix_order=schedule.init_healpix_order,
        adaptive_oversampling=options.adaptive.adaptive_oversampling,
        translation_range=schedule.init_translation_range,
        translation_step=schedule.init_translation_step,
        max_healpix_order=schedule.max_healpix_order,
        auto_local_healpix_order=options.local_search.auto_local_healpix_order,
        # Class3D (K>1) never switches to local searches from the HEALPix order.
        auto_sampling=not (options.k_class.n_classes > 1),
        voxel_size_angstrom=image_geometry.pixel_size_angstrom,
        particle_diameter_angstrom=float(schedule.particle_diameter_ang or 0.0),
        subtomogram=subtomogram,
        do_local_search=sigma_ang_deg is not None,
        sigma_rot=sigma_ang,
        sigma_psi=sigma_ang,
    )
    # RELION's convergence counters are not initialized against an infinite
    # previous resolution.  They resume from the previous optimiser/model STAR
    # in replay mode, or from the initial FSC/ini_high state in a fresh run.
    if source.restore_convergence_state(state):
        pass
    elif schedule.init_fsc is not None:
        initialize_resolution_from_fsc(
            state, options, box_size=image_geometry.box_size, voxel_size=image_geometry.pixel_size_angstrom,
            dtype=dtype,
        )
    elif init_relion_iteration == 0 and parity.relion_firstiter_ini_high_angstrom is not None:
        initialize_resolution_from_firstiter_ini_high(
            state, options, box_size=image_geometry.box_size, voxel_size=image_geometry.pixel_size_angstrom
        )
    elif init_relion_iteration == 0 and schedule.ini_high_angstrom is not None:
        initialize_resolution_from_ini_high(
            state, schedule.ini_high_angstrom, box_size=image_geometry.box_size,
            voxel_size=image_geometry.pixel_size_angstrom,
        )
    source.restore_boundary_state(state)
    # A continuation (RELION --continue) starts from the run files' sampling state; the
    # rest of its snapshot is installed just before the loop.
    resume = options.checkpoint.resume
    if resume is not None:
        validate_resume_snapshot(
            resume,
            init_relion_iteration=init_relion_iteration,
            n_classes=options.k_class.n_classes,
            box_size=image_geometry.box_size,
            options=options,
            replays_relion_trajectory=source.replays_relion_trajectory,
            starts_from_frozen_boundary=source.starts_from_frozen_boundary,
            swaps_state=source.swaps_state,
        )
        state = resume.refinement_state(state)
    return state


def resolve_current_size(
    image_size_plan,
    options: RefinementOptions,
    *,
    previous_size,
    incr_size,
    has_high_fsc_at_limit,
    ave_pmax,
    iteration: int,
    box_size,
    log: logging.Logger,
) -> int:
    """This iteration's current size: the planned size quantised, or the size an oracle schedule gives.

    ``previous_size`` is the preceding iteration's size, None for a run's first iteration (which logs no
    decision). Reads from ``options``: ``adaptive.relion_current_sizes``, per-iteration sizes that replace
    the plan (past their end the last one holds), and ``schedule.init_current_size``, which a non-positive
    entry means.
    """
    oracle_sizes = options.adaptive.relion_current_sizes
    current_size = quantize_current_size(image_size_plan.size, box_size=box_size)
    if previous_size is not None:
        log.info(
            "RELION current-size decision: iter=%d prev=%d res_shell=%d "
            "incr_size=%d high_fsc_at_limit=%s ave_Pmax=%.6f raw=%d quantized=%d",
            iteration + 1,
            int(previous_size),
            int(image_size_plan.resolution_shell),
            int(incr_size),
            bool(has_high_fsc_at_limit),
            float(ave_pmax),
            int(image_size_plan.raw_size),
            int(current_size),
        )
    if oracle_sizes is not None:
        if iteration < len(oracle_sizes):
            oracle_cs = int(oracle_sizes[iteration])
        else:
            oracle_cs = int(oracle_sizes[-1])
        if oracle_cs <= 0:
            oracle_cs = int(options.schedule.init_current_size)
        current_size = quantize_current_size(oracle_cs, box_size=box_size)
        log.info(
            "Current-size oracle: iteration %d using current_size=%d",
            iteration + 1,
            current_size,
        )
    return current_size


class FirstIterationPolicy(NamedTuple):
    """How one numbered iteration scores and reconstructs: only a run's first iteration departs."""

    relion_firstiter_cc: bool
    score_mode: str
    winner_take_all: bool


def first_iteration_policy(options: RefinementOptions, *, iteration) -> FirstIterationPolicy:
    """RELION's first-iteration CC emulation, the score mode and whether one pose takes all the weight.

    Normalised cross-correlation scoring and hard reconstruction apply to iteration 1 of a run that
    starts at RELION iteration 0, under ``--firstiter_cc`` emulation or the two diagnostic options.
    Reads from ``options``: ``schedule.init_relion_iteration``; ``parity.emulate_relion_firstiter_cc``,
    ``first_iteration_score_mode`` and ``first_iteration_reconstruction_mode``.
    """
    parity = options.parity
    init_relion_iteration = options.schedule.init_relion_iteration
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


class ClassSeeding(NamedTuple):
    """Whether a numbered iteration of a Class3D start from one reference scores one class or seeds classes."""

    # The first-iteration CC of a seeded start scores class 0 alone; its model is then copied to every class.
    single_class_iteration: bool
    # The random class of each particle this iteration scores against (None: no seeding this iteration).
    seed_classes: object | None


def class_seeding(options: RefinementOptions, *, continued: bool, iteration: int) -> ClassSeeding:
    """RELION's Class3D from one reference scores each particle against one random class in its first
    iteration (do_generate_seeds, ml_optimiser.cpp:4626-4633, :4880-4898). With --firstiter_cc the first
    iteration scores class 0 alone (CC), its model is then copied to every class, and the random classes are
    seeded in the second iteration. A continued run (``continued``) or one past RELION iteration 0 seeds none.
    """
    seeded_start = (
        options.k_class.first_iteration_seed_classes is not None
        and not continued
        and int(options.schedule.init_relion_iteration) == 0
    )
    seed_after_cc = seeded_start and bool(options.parity.emulate_relion_firstiter_cc)
    seed_iteration = seeded_start and iteration == (1 if seed_after_cc else 0)
    return ClassSeeding(
        single_class_iteration=seed_after_cc and iteration == 0,
        seed_classes=options.k_class.first_iteration_seed_classes if seed_iteration else None,
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
    dtype,
) -> CoarseGrids:
    """Pair RELION's canonical initial rotations with its translation grid, in the scoring ``dtype``."""

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
    options: RefinementOptions,
    *,
    voxel_size,
    dtype,
    log: logging.Logger,
) -> CoarseGrids:
    """The exhaustive coarse grids of ``state``'s sampling, rebuilt where they changed.

    A new HEALPix order rebuilds the rotation grid (up to the exhaustive-grid cap) and the translation
    grid (a replayed range or step at the same order is the input source's, ``InputSource.coarse_grids``).
    ``grids.translations`` may be a perturbed copy; a rebuild replaces it with the base grid. Reads from ``state``: ``healpix_order`` (and the fields the
    exhaustive-grid cap reads), ``translation_range`` and ``translation_step``; from ``options``:
    ``k_class.n_classes`` and ``symmetry.point_group``.
    """
    symmetry = options.symmetry.point_group
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
            n_classes=options.k_class.n_classes,
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
    return CoarseGrids(current_rotation_grid, base_translations, current_translations)


def iteration_trial_grid(
    grids: CoarseGrids,
    state: RefinementState,
    options: RefinementOptions,
    random_perturbation: float,
    *,
    perturbation_order: int | None,
    sealed_grid: bool,
    dtype,
) -> sampling.TrialGrid:
    """This iteration's trial grid: the coarse grid, under RELION's sampling perturbation where one applies.

    ``perturbation_order`` is the HEALPix order whose angular sampling scales the perturbation, or None when
    none applies: then the coarse rotations and the grid's current translations are the trial grid, without
    exact M-step rotations. Reads from ``grids``: the rotation grid (rotations, Euler rows),
    ``base_translations`` (the unperturbed host translations) and ``translations``; ``state.translation_step``;
    from ``options``: ``symmetry.point_group``. A sealed grid (``sealed_grid``) keeps its own Euler rows.
    """
    rotation_grid = grids.rotation_grid
    rotation_eulers = np.asarray(
        rotation_grid.rotation_eulers,
        dtype=dtype,
    )
    if perturbation_order is None:
        return sampling.TrialGrid(
            rotations=rotation_grid.rotations,
            rotation_eulers=rotation_eulers,
            mstep_rotations=None,
            translations=grids.translations,
        )
    angsamp_deg = sampling.relion_angular_sampling_deg(perturbation_order, adaptive_oversampling=0)
    return sampling._perturbed_trial_grid(
        rotation_eulers=rotation_eulers,
        mstep_source_eulers=sampling._relion_mstep_source_eulers(
            rotation_eulers,
            perturbation_order,
            use_grid_eulers=sealed_grid,
            symmetry=options.symmetry.point_group,
        ),
        base_translations=grids.base_translations,
        translation_step=float(state.translation_step),
        random_perturbation=random_perturbation,
        angular_sampling_deg=angsamp_deg,
        dtype=dtype,
    )


def coarse_pass1_rotations(
    rotation_grid: sampling.RotationGrid,
    random_perturbation: float,
    options: RefinementOptions,
    *,
    perturbation_order: int | None,
    dtype,
    log: logging.Logger,
):
    """RELION's device-built rotations for the pass-1 coarse scorer, or None where the host grid serves.

    The source is the unperturbed coarse Euler rows of ``rotation_grid``, in the scoring ``dtype`` and
    then widened to float64. The perturbation applies only when the trial grid is perturbed
    (``perturbation_order``, the order whose angular step scales it; None: unperturbed, at the grid's order).
    """
    source_eulers = np.asarray(np.asarray(rotation_grid.rotation_eulers, dtype=dtype), dtype=np.float64)
    adaptive_pass1_order = perturbation_order if perturbation_order is not None else int(rotation_grid.healpix_order)
    adaptive_pass1_use_float64 = options.precision.use_float64_scoring
    adaptive_pass1_rotations = _relion_adaptive_pass1_rotations(
        source_eulers,
        random_perturbation if perturbation_order is not None else 0.0,
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
    options: RefinementOptions,
    *,
    box_size,
    pixel_size_angstrom,
    incr_size,
    has_high_fsc_at_limit,
    dtype,
    log: logging.Logger,
) -> ImageSizeUpdate:
    """Resolve startup ini_high, initial FSC or bootstrap width in that order.

    Reads from ``options``: ``schedule.init_relion_iteration``, ``init_fsc``, ``init_current_size`` and
    ``init_ave_Pmax``; ``parity.relion_firstiter_ini_high_angstrom`` and ``tau2_fudge``.
    See ``docs/math/relion_refinement_algorithm.md`` for image-size scheduling.
    """
    schedule = options.schedule
    parity = options.parity
    if schedule.init_relion_iteration == 0:
        seeded_cs = bootstrap_current_size_from_ini_high_relion(
            box_size,
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
            box_size=box_size,
            dtype=dtype,
        )
        data_vs_prior_iter = np.asarray(
            fsc_to_relion_ssnr(fsc_prev, tau2_fudge=parity.tau2_fudge),
        )
        res_shell = resolution_from_data_vs_prior(
            data_vs_prior_iter,
            box_size=box_size,
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
            box_size,
            ave_Pmax=_init_pmax,
            has_high_fsc_at_limit=has_high_fsc_at_limit,
            incr_size=incr_size,
        )
        current_size = quantize_current_size(raw_cs, box_size=box_size)
    else:
        current_size = _bootstrap_current_size_relion(schedule.init_current_size, box_size)
        data_vs_prior_iter = None

    return ImageSizeUpdate(
        size=current_size, data_vs_prior=data_vs_prior_iter,
        incr_size=incr_size, has_high_fsc_at_limit=has_high_fsc_at_limit,
    )


def _firstiter_cc_resolution_shell(
    res_shell: int,
    *,
    size_name: str,
    box_size,
    pixel_size_angstrom,
    completed_relion_iteration,
    parity: RelionParityOptions,
    log: logging.Logger,
) -> int:
    """The shell both growth plans schedule from: RELION's firstiter_cc override of ``res_shell``, logged
    naming ``size_name``."""
    scheduling_res_shell = _firstiter_cc_scheduling_resolution_shell(
        res_shell,
        emulate_relion_firstiter_cc=parity.emulate_relion_firstiter_cc,
        ini_high_angstrom=parity.relion_firstiter_ini_high_angstrom,
        relion_iteration=completed_relion_iteration,
        box_size=box_size,
        voxel_size=pixel_size_angstrom,
    )
    if scheduling_res_shell != res_shell:
        res_shell = scheduling_res_shell
        log.info(
            "RELION firstiter_cc scheduling: using ini_high=%.2f A shell %d for next " + size_name,
            float(parity.relion_firstiter_ini_high_angstrom),
            int(res_shell),
        )
    return res_shell


def _quantized_current_size(res_shell, *, box_size, ave_pmax, has_high_fsc_at_limit, incr_size) -> tuple[int, int]:
    """RELION's next current size from the scheduling shell: ``(raw, quantized)``."""
    raw_cs = compute_current_size_relion(
        res_shell,
        box_size,
        ave_Pmax=ave_pmax,
        has_high_fsc_at_limit=has_high_fsc_at_limit,
        incr_size=incr_size,
    )
    return raw_cs, quantize_current_size(raw_cs, box_size=box_size)


def plan_class_image_size(
    data_vs_prior,
    *,
    previous_size,
    box_size,
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
        box_size=box_size,
        dtype=dtype,
    )
    per_class_res_shell = np.asarray(
        class_resolution_shells(data_vs_prior_prev, box_size=box_size),
        dtype=np.int32,
    )
    res_shell = _firstiter_cc_resolution_shell(
        int(np.max(per_class_res_shell)),
        size_name="K-class current_size",
        box_size=box_size,
        pixel_size_angstrom=pixel_size_angstrom,
        completed_relion_iteration=completed_relion_iteration,
        parity=parity,
        log=log,
    )
    raw_cs, computed_cs = _quantized_current_size(
        res_shell, box_size=box_size, ave_pmax=ave_pmax, has_high_fsc_at_limit=False, incr_size=incr_size
    )

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
    box_size,
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
    if growth_fsc_history:
        fsc_for_growth = growth_fsc_history[-1]
    elif restart is not None and restart.fsc_for_growth is not None:
        fsc_for_growth = restart.fsc_for_growth
    else:
        # The raw FSC: this run's last one, else the continued snapshot's.
        fsc_for_growth = fsc_history[-1] if fsc_history else restart.fsc
    fsc_prev_for_growth = _zero_shells_past_current_size(
        fsc_for_growth,
        current_size=previous_size,
        box_size=box_size,
        dtype=dtype,
    )

    data_vs_prior_iter = _zero_shells_past_current_size(
        data_vs_prior,
        current_size=previous_size,
        box_size=box_size,
        dtype=dtype,
    )
    res_shell = resolution_from_data_vs_prior(
        data_vs_prior_iter,
        box_size=box_size,
        allow_high_res_recovery=True,
    )
    incr_size, has_high_fsc_at_limit = update_relion_growth_state_from_fsc(
        fsc_prev_for_growth,
        previous_size,
        incr_size=incr_size,
        has_high_fsc_at_limit=has_high_fsc_at_limit,
    )
    res_shell = _firstiter_cc_resolution_shell(
        res_shell,
        size_name="current_size",
        box_size=box_size,
        pixel_size_angstrom=pixel_size_angstrom,
        completed_relion_iteration=completed_relion_iteration,
        parity=parity,
        log=log,
    )
    raw_cs, current_size = _quantized_current_size(
        res_shell,
        box_size=box_size,
        ave_pmax=ave_pmax,
        has_high_fsc_at_limit=has_high_fsc_at_limit,
        incr_size=incr_size,
    )

    return HalfmapImageSize(
        size=current_size, data_vs_prior=data_vs_prior_iter,
        incr_size=incr_size, has_high_fsc_at_limit=has_high_fsc_at_limit,
        resolution_shell=res_shell, raw_size=raw_cs,
    )
