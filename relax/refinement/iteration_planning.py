"""A numbered iteration's records and start-up state: the start-up
``RefinementState``, the first-iteration and seeding policies and what one iteration carries to the next
(image sizes: ``image_size_plans.py``; trial grids and their perturbation: ``trial_grids.py``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from relax.helpers.convergence import RefinementState
from relax.helpers.expected_accuracy import PublishedAccuracy
from relax.helpers.resolution import (
    ImageGeometry,
    initialize_resolution_from_firstiter_ini_high,
    initialize_resolution_from_fsc,
    initialize_resolution_from_ini_high,
)
from relax.refinement.iteration_snapshot import validate_resume_snapshot
from relax.refinement.ports import InputSource

if TYPE_CHECKING:
    from relax.refinement.refinement_options import RefinementOptions


def initialize_refinement_state(
    options: RefinementOptions,
    image_geometry: ImageGeometry,
    *,
    subtomogram: bool,
    dtype,
    source: InputSource,
    hands_reference_real: bool = False,
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
    elif options.start.init_fsc is not None:
        initialize_resolution_from_fsc(
            state, options, box_size=image_geometry.box_size, voxel_size=image_geometry.pixel_size_angstrom,
            dtype=dtype,
        )
    elif init_relion_iteration == 0 and parity.relion_firstiter_ini_high_angstrom is not None:
        initialize_resolution_from_firstiter_ini_high(
            state, options, box_size=image_geometry.box_size, voxel_size=image_geometry.pixel_size_angstrom
        )
    elif init_relion_iteration == 0 and options.start.ini_high_angstrom is not None:
        initialize_resolution_from_ini_high(
            state, options.start.ini_high_angstrom, box_size=image_geometry.box_size,
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
            hands_reference_real=hands_reference_real,
        )
        state = resume.refinement_state(state)
    return state


class FirstIterationPolicy(NamedTuple):
    """How one numbered iteration scores and reconstructs: only a run's first iteration departs."""

    relion_firstiter_cc: bool
    score_mode: str
    winner_take_all: bool


def first_iteration_policy(options: RefinementOptions, *, iteration) -> FirstIterationPolicy:
    """RELION's first-iteration CC emulation, the score mode and whether one pose takes all the weight.

    Normalised cross-correlation scoring and hard reconstruction apply to iteration 1 of a run that
    starts at RELION iteration 0, under ``--firstiter_cc`` emulation.
    Reads from ``options``: ``schedule.init_relion_iteration``, ``parity.emulate_relion_firstiter_cc``.
    """
    parity = options.parity
    init_relion_iteration = options.schedule.init_relion_iteration
    relion_firstiter_cc_this_iter = bool(
        parity.emulate_relion_firstiter_cc and init_relion_iteration == 0 and iteration == 0
    )
    return FirstIterationPolicy(
        relion_firstiter_cc_this_iter,
        "normalized_cc" if relion_firstiter_cc_this_iter else "gaussian",
        relion_firstiter_cc_this_iter,
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


@dataclass(frozen=True, kw_only=True)
class NumberedIteration:
    """One numbered iteration's identity and the decisions taken before its expectation: the loop's 0-based
    index, RELION's iteration number, whether a completed iteration (or a snapshot) precedes it, its
    first-iteration policy, its Class3D seeding and its current size."""

    iteration: int
    numbered_relion_iteration: int
    has_previous_iteration: bool
    first_iteration: FirstIterationPolicy
    seeding: ClassSeeding
    # The image size the iteration plans its windows from, after the input source's replacements.
    current_size: int


@dataclass(frozen=True, kw_only=True)
class IterationCarry:
    """What a numbered iteration reads from the iterations before it (or from the start-up), and leaves for the
    ones after it and the final pass.

    The controller replaces it with ``dataclasses.replace`` after each statement that changes a field. Frozen,
    not deeply immutable: ``state`` (the ``RefinementState``) is updated in place by the input source's
    ``numbered_state`` and ``swapped_state`` and by the controller's convergence latch and accuracy install,
    and the ``direction_priors`` list's entries are replaced in place (the input source's ``numbered_state``,
    the learned priors, the Class3D copy). The reference model, the halves and the history are not carried: the
    controller holds them, and they are updated in place.
    """

    # The RefinementState: sampling, accuracy and convergence counters.
    state: object
    # The class weights and their log priors (one class for K=1).
    class_mixture: object
    # Each half's DirectionPrior.
    direction_priors: list
    noise_model: object
    # The translation prior widths (SigmaOffset).
    sigma_offset: object
    # The coarse rotation grid and the base and current translations (CoarseGrids).
    coarse_grids: object
    random_perturbation: float
    # The HEALPix order of the replayed sampling STAR the next expectation sizes its coarse pass from (None:
    # the run's own sampling), and whether this iteration samples natively (no STAR, no sealed state).
    replay_saved_healpix_order: int | None
    native_sampling_boundary: bool
    # RELION's current-size growth state.
    relion_incr_size: int
    relion_has_high_fsc_at_limit: bool
    # The data-vs-prior curve the next image-size plan and the scale gate read (None before the first).
    previous_data_vs_prior_for_scheduling: object
    # Each half's coarse-grid and class assignments of the preceding iteration, for the change statistics.
    previous_assignments: list
    previous_class_assignments: list
    # Each half's best rotations of the preceding iteration (RELION's orientation-change metric).
    previous_best_rotations: list
    # This iteration's per-half class and hard assignments (the next accuracy estimate, the final pass, the
    # result).
    class_assignments: list
    hard_assignments: list
    published_accuracy: PublishedAccuracy
