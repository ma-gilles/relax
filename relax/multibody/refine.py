"""RELION multi-body refinement: the controller.

``relion_refine --continue <consensus run_itNNN_optimiser.star> --multibody_masks <bodies.star>`` refines every
body of every particle around the consensus pose, starting each body's two half maps from the consensus half
maps. The iteration follows ``MlOptimiserMpi::iterate`` (ml_optimiser_mpi.cpp at f2c1a38) with its multi-body
branches:

1. at the top, convergence (``checkConvergence``): on convergence every body with a prior is refined again and the
   last iteration joins the halves at the full image size;
2. the image size from the best resolution over the bodies (``updateImageSizeAndResolutionPointers``);
3. each body's masked, centred projector (``setFourierTransformMaps``) and its expected accuracy
   (``calculateExpectedAngularErrors``: one per body, ``acc_rot`` their minimum);
4. the sampling update (``updateAngularSampling``), which keeps a body fixed once the angular step is below 90% of its
   accuracy, and latches fine-enough sampling when every body is fixed;
5. the expectation, body after body (:mod:`relax.multibody.estep`), skipping fixed bodies;
6. per body, the low-resolution half join, the solvent-corrected FSC in its mask, the prior and the two half maps
   (:mod:`relax.multibody.mstep`), and RELION's image-size growth state from that FSC;
7. the noise from every body's statistics; norm, group scales and the offset prior width stay the consensus's;
8. the hidden-variable changes over all refined bodies, and the resolution: the best over the bodies, which gains
   only when one body gains.

Out of scope and refused: more than one class or optics group, helical, subtomograms, CTF premultiplication,
``--reconstruct_subtracted_bodies`` off, Blush and SGD.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
from recovar.core import fourier_transform_utils as ftu
from recovar.utils.helpers import load_relion_volume

from relax.multibody import geometry, run_files
from relax.multibody.bodies import RESIDUAL_IDENTITY_EULER, Body, read_bodies, subtraction_masks
from relax.multibody.checkpoint import RelionCheckpoint, read_relion_checkpoint
from relax.multibody.estep import HalfParticles, score_body_half
from relax.multibody.mstep import maximize_body, maximize_body_final
from relax.multibody.subtraction import BodyPoses, ParticleFrames, body_projector_reference
from relax.numerics.relion_random import init_random_generator
from relax.reconstruction.resolution_criteria import (
    compute_current_size_relion,
    resolution_from_data_vs_prior,
    update_relion_growth_state_from_fsc,
)
from relax.refinement import noise_updates
from relax.refinement.iteration_planning import ClassSeeding, FirstIterationPolicy, NumberedIteration
from relax.refinement.local_sampling import plan_expectation_sampling
from relax.refinement.refinement_state import (
    MAX_NR_ITER_WO_LARGE_HIDDEN_VARIABLE_CHANGES,
    MAX_NR_ITER_WO_RESOL_GAIN,
    RefinementState,
    effective_angular_step,
    refine_angular_sampling,
    resolution_triggers_angular_refinement,
)
from relax.refinement.trial_grids import build_initial_coarse_grids, iteration_trial_grid, refresh_coarse_grids
from relax.relion.macros import relion_round
from relax.sampling.healpix import euler_angles_to_matrix
from relax.sampling.perturbation import advance_relion_perturbation_from_seed
from relax.steps.convergence import compute_relion_offset_changes_angstrom, compute_relion_orientation_changes
from relax.steps.expected_accuracy import estimate_relion_expected_accuracy
from relax.steps.projector_preparation import ProjectorFiles, prepare_scoring_projector
from relax.types import make_noise_stats, total_sumw

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class MultibodyCommand:
    """The multi-body options of the RELION command (``getCommandsMultiBodyJob``), in its units."""

    body_star: Path
    healpix_order: int
    auto_local_healpix_order: int
    offset_range_px: float
    offset_step_px: float
    oversampling: int
    reconstruct_subtracted_bodies: bool
    max_iter: int
    # --sigma_ang (degrees): replaces the consensus rlnSigmaPriorRot/Tilt/PsiAngle (parseContinue,
    # ml_optimiser.cpp:368-373); None keeps them, as the GUI's command does.
    sigma_ang_deg: float | None = None
    # --multibody_norm_overlap: overlapping masks are normalised where they sum above one, instead of RELION's
    # default extra projectors for the overlaps (relax.multibody.bodies.subtraction_masks).
    norm_overlap: bool = False
    # --pad: None keeps the consensus's rlnPaddingFactor (ml_optimiser.cpp:413).
    padding_factor: int | None = None


@dataclass
class BodyTrack:
    """One body's state between iterations: its half maps (recovar Fourier, stored frame), half-1 data-vs-prior
    curve, FSC, accuracies, whether it is kept fixed, and every particle's residual pose."""

    maps: list
    data_vs_prior: np.ndarray
    fsc: np.ndarray
    tau2: list
    acc_rot: float
    acc_trans: float
    keep_fixed: bool
    poses: BodyPoses


@dataclass
class MultibodyState:
    """The run state RELION carries from one multi-body iteration to the next."""

    iteration: int  # the last completed RELION iteration
    bodies: list  # BodyTrack per body
    noise: object  # relax.refinement.noise_updates.NoiseModel
    sampling: RefinementState
    perturbation: float
    current_resolution_inv: float  # 1 / Angstrom, RELION's mymodel.current_resolution
    best_resol_thus_far: float
    incr_size: int
    has_high_fsc_at_limit: bool
    ave_pmax: float
    current_size: int
    # Each particle's stored Pmax (RELION's METADATA_PMAX), which only body 1's expectation writes.
    particle_pmax: np.ndarray = None
    has_converged: bool = False
    history: list = field(default_factory=list)


@dataclass(frozen=True)
class MultibodyRun:
    """What does not change during a run."""

    checkpoint: RelionCheckpoint
    command: MultibodyCommand
    bodies: list  # Body per body
    dataset: object
    halves: tuple  # HalfParticles per half
    pixel_size: float
    box_size: int
    particle_diameter: float
    width_mask_edge: float
    low_resolution_join: float
    random_seed: int
    perturb_factor: float
    do_ctf_correction: bool
    options: object  # RefinementOptions (precision, engine, variants and scoring choices)
    optics: object  # RunOptics
    settings: object  # ReconstructionSettings without a solvent flatten
    phase_stream: object  # RELION's MPI leader rand() stream
    output: Path
    subtraction_masks: list = None  # [target][other] RELION-frame mask the other body is subtracted with


def _floats(column) -> np.ndarray:
    return np.asarray(column, dtype=np.float64)


def _fourier(real) -> np.ndarray:
    return np.asarray(ftu.get_dft3(jnp.asarray(np.asarray(real, dtype=np.float32))).reshape(-1))


def _padding_factor(checkpoint: RelionCheckpoint, command: MultibodyCommand) -> int:
    """--pad, else the consensus's; the multi-body start may change it (parseContinue, ml_optimiser.cpp:413-424)."""

    padding = command.padding_factor
    return int(float(checkpoint.models[0].general["rlnPaddingFactor"]) if padding is None else padding)


def _require_supported(checkpoint: RelionCheckpoint, command: MultibodyCommand) -> None:
    general = checkpoint.optimiser
    model = checkpoint.models[0].general
    checks = {
        "one class": int(model["rlnNrClasses"]) == 1,
        "C1 symmetry": str(checkpoint.sampling.get("rlnSymmetryGroup", "C1")).upper() == "C1",
        "one optics group": int(model["rlnNrOpticsGroups"]) == 1 and len(checkpoint.optics) == 1,
        "a consensus (not multi-body) checkpoint": int(model.get("rlnNrBodies", 1)) == 1,
        "--zero_mask": int(general["rlnDoZeroMask"]) == 1,
        "auto-refine": int(general["rlnDoAutoRefine"]) == 1,
        "no Blush": int(general.get("rlnDoBlush", 0)) == 0,
        "no gradient refinement": int(general.get("rlnDoGradientRefine", 0)) == 0,
        "--reconstruct_subtracted_bodies": command.reconstruct_subtracted_bodies,
        "adaptive oversampling 1": command.oversampling == 1,
        "local searches from the start": command.healpix_order >= command.auto_local_healpix_order,
        "2D particles": int(model["rlnDataDimensionality"]) == 2,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise NotImplementedError(f"relax multibody supports only: {', '.join(failed)}")
    if _padding_factor(checkpoint, command) not in (1, 2):
        raise NotImplementedError("relax multibody supports padding factors 1 and 2")


def start_from_consensus(checkpoint_path, command: MultibodyCommand, options, *, output, relion_cwd=None):
    """RELION's multi-body start from a consensus checkpoint: ``MlOptimiser::initialise`` with
    ``do_initialise_bodies`` (ml_optimiser.cpp:2122-2190 at f2c1a38). Returns ``(MultibodyRun, MultibodyState)``."""

    from recovar.data_io.cryoem_dataset import load_dataset

    from relax.refinement.image_size_plans import RunOptics
    from relax.refinement.setup_checks import ImageGeometry, reconstruction_settings_for_run
    from relax.steps.half_inputs import configure_half_image_preprocessing

    checkpoint = read_relion_checkpoint(checkpoint_path, relion_cwd=relion_cwd)
    _require_supported(checkpoint, command)
    general = checkpoint.optimiser
    models = checkpoint.models
    pixel_size = float(models[0].general["rlnPixelSize"])
    box_size = int(models[0].general["rlnOriginalImageSize"])
    bodies = read_bodies(
        command.body_star, pixel_size=pixel_size, norm_overlap=command.norm_overlap, paths_from=relion_cwd
    )
    if len(bodies) < 2:
        raise ValueError("a multi-body refinement needs at least two bodies")

    data_star = checkpoint.optimiser_star.parent.parent / general["rlnExperimentalDataStarFile"]
    dataset = load_dataset(
        str(data_star), datadir=str(checkpoint.optimiser_star.parent.parent), lazy=True,
        dtype=np.complex64, absent_angles_zero=True,
    )
    particles = checkpoint.particles
    # Experiment::read numbers the groups by first appearance of rlnGroupName and ignores rlnGroupNumber
    # (exp_model.cpp:931-967); the model's scales are indexed by that number (ml_model.cpp:360).
    group_numbers = (
        pd.factorize(particles["rlnGroupName"])[0] + 1
        if "rlnGroupName" in particles
        else np.asarray(particles["rlnGroupNumber"], dtype=np.int64)
    )
    subsets = np.asarray(particles["rlnRandomSubset"], dtype=np.int64)
    halves = []
    for h in range(2):
        rows = np.flatnonzero(subsets == h + 1)
        model = models[h]
        avg_norm = float(model.general["rlnNormCorrectionAverage"])
        norm = _floats(particles["rlnNormCorrection"])[rows]
        scale = model.group_scale[group_numbers[rows] - 1]
        frames = ParticleFrames(
            consensus=geometry.consensus_matrices(
                np.stack([_floats(particles[c])[rows] for c in ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi")], axis=1)
            ),
            offsets_px=np.stack([_floats(particles[c])[rows] for c in ("rlnOriginXAngst", "rlnOriginYAngst")], axis=1)
            / pixel_size,
            norm_factors=avg_norm / norm,
        )
        halves.append(HalfParticles(
            index=h, rows=rows, frames=frames, image_corrections=(avg_norm / norm) * scale,
            scale_corrections=scale, optics_group_ids=None, group_ids=group_numbers[rows] - 1,
            group_count=int(model.group_scale.size),
        ))

    particle_diameter = float(general["rlnParticleDiameter"])
    width_mask_edge = float(general["rlnWidthMaskEdge"])
    configure_half_image_preprocessing(
        [dataset], pixel_size_angstrom=pixel_size, particle_diameter_angstrom=particle_diameter,
        width_mask_edge_px=width_mask_edge, fourier_backend=options.parity.image_fourier_backend,
        source_faithful_spectrum_norm=options.parity.preserve_bpref_particle_order, log=logger,
    )
    image_geometry = ImageGeometry(image_shape=dataset.image_shape, pixel_size_angstrom=pixel_size)
    optics = RunOptics(
        image_geometry=image_geometry, model_pixel_size=pixel_size, optics_image_sizes=None,
        optics_pixel_sizes=None, multi_shape_halves=False,
    )
    settings = reconstruction_settings_for_run(
        replace(
            options,
            schedule=replace(options.schedule, particle_diameter_ang=None, width_mask_edge_px=width_mask_edge),
            parity=replace(options.parity, tau2_fudge=float(models[0].general["rlnTau2FudgeFactor"])),
        ),
        image_geometry, dataset.volume_shape, options.consistency,
    )
    padding = _padding_factor(checkpoint, command)
    settings = replace(settings, padding_factor=padding, projection_padding_factor=padding)
    random_seed = int(general["rlnRandomSeed"])
    # The consensus run's particle diameter sizes the pass-1 window (image_coarse_size); the reconstruction
    # settings above leave it out because RELION does not flatten the body maps.
    options = replace(options, schedule=replace(options.schedule, particle_diameter_ang=particle_diameter))
    run = MultibodyRun(
        checkpoint=checkpoint, command=command, bodies=bodies, dataset=dataset, halves=tuple(halves),
        subtraction_masks=subtraction_masks(bodies, norm_overlap=command.norm_overlap),
        pixel_size=pixel_size, box_size=box_size, particle_diameter=particle_diameter,
        width_mask_edge=width_mask_edge, low_resolution_join=float(general["rlnJoinHalvesUntilThisResolution"]),
        random_seed=random_seed, perturb_factor=float(checkpoint.sampling["rlnSamplingPerturbFactor"]),
        do_ctf_correction=int(general.get("rlnDoCorrectCtf", 1)) == 1, options=options, optics=optics,
        settings=settings,
        # The leader seeds once (ml_optimiser_mpi.cpp:827) and its stream randomises the FSC phases.
        phase_stream=init_random_generator(random_seed), output=Path(output),
    )
    return run, _initial_state(run)


def _initial_state(run: MultibodyRun) -> MultibodyState:
    checkpoint = run.checkpoint
    models = checkpoint.models
    frame = float(run.box_size) ** 4  # relax spectra are RELION's times ori_size**4 (run_files.py)
    n_particles = len(checkpoint.particles)
    half_maps = [_fourier(load_relion_volume(str(paths[0]))) for paths in checkpoint.half_map_paths]
    body_references = [
        None if body.reference_path is None else _fourier(load_relion_volume(str(body.reference_path)))
        for body in run.bodies
    ]
    class_shells = [model.class_shells[0] for model in models]
    tracks = [
        BodyTrack(
            # Every body's stored reference starts as the consensus half map (initialiseBodies copies Iref[0]), or,
            # in both halves, as its rlnBodyReferenceName map (ml_model.cpp:1343-1358).
            maps=[(m if reference is None else reference).copy() for m in half_maps],
            data_vs_prior=class_shells[0]["rlnSsnrMap"].copy(),
            fsc=class_shells[0]["rlnGoldStandardFsc"].copy(),
            tau2=[shells["rlnReferenceTau2"] * frame for shells in class_shells],
            acc_rot=float(models[0].classes["rlnAccuracyRotations"][0]),
            acc_trans=float(models[0].classes["rlnAccuracyTranslationsAngst"][0]),
            keep_fixed=body.keep_fixed,
            poses=BodyPoses(
                eulers_deg=np.tile(np.asarray(RESIDUAL_IDENTITY_EULER), (n_particles, 1)),
                offsets_px=np.zeros((n_particles, 2)),
            ),
        )
        for body, reference in zip(run.bodies, body_references)
    ]
    # Reproduces RELION issue #7 (https://github.com/ma-gilles/relax/issues/7): the first follower (half 1, which
    # read the half-1 model) broadcasts its sigma2_noise to every rank after initialiseGeneral
    # (ml_optimiser_mpi.cpp:748-758), also on a continuation, so both halves start from half 1's noise, as
    # RELION's run_it000_half2_model.star shows. relax's own K=1 --continue keeps each half's noise instead
    # (relax/refinement/run_files.py); a multi-body run always starts from a consensus, so it copies RELION.
    noise = noise_updates.noise_model_from_shells(
        [models[0].sigma2_noise[0] * frame for _ in models], run.dataset.image_shape
    )
    command = run.command
    # RELION multiplies the continuation's --offset_range and --offset_step by the pixel size twice: in
    # parseContinue (ml_optimiser.cpp:349-358) and again in initialiseGeneral for iter 0 (:2347-2354), which the
    # multi-body start resets to. The sampling is therefore pixel_size times wider than the command asks: RELION
    # issue https://github.com/ma-gilles/relax/issues/132, reproduced for parity.
    offset_range_px = command.offset_range_px * run.pixel_size
    offset_step_px = command.offset_step_px * run.pixel_size
    sigma_rot = np.deg2rad(float(models[0].general["rlnSigmaPriorRotAngle"]))
    sigma_psi = np.deg2rad(float(models[0].general["rlnSigmaPriorPsiAngle"]))
    if command.sigma_ang_deg is not None:
        sigma_rot = sigma_psi = np.deg2rad(float(command.sigma_ang_deg))
    sampling = RefinementState(
        healpix_order=command.healpix_order,
        adaptive_oversampling=command.oversampling,
        translation_range=offset_range_px,
        translation_step=offset_step_px,
        auto_local_healpix_order=command.auto_local_healpix_order,
        do_local_search=True,
        sigma_rot=sigma_rot,
        sigma_psi=sigma_psi,
        voxel_size_angstrom=run.pixel_size,
        particle_diameter_angstrom=run.particle_diameter,
        has_fine_enough_angular_sampling=bool(int(checkpoint.optimiser.get("rlnHasFineEnoughAngularSampling", 0))),
        acc_rot=float(checkpoint.optimiser["rlnOverallAccuracyRotations"]),
        acc_trans=float(checkpoint.optimiser["rlnOverallAccuracyTranslationsAngst"]),
    )
    # HealpixSampling::read keeps the consensus perturbation and initialise advances it once from the C library's
    # default stream (seed 1) before any init_random_generator call (healpix_sampling.cpp:128, :167-174).
    perturbation = advance_relion_perturbation_from_seed(
        float(checkpoint.sampling["rlnSamplingPerturbInstance"]), run.perturb_factor, seed=1
    )
    state = MultibodyState(
        iteration=0,
        bodies=tracks,
        noise=noise,
        sampling=sampling,
        perturbation=perturbation,
        current_resolution_inv=1.0 / float(models[0].general["rlnCurrentResolution"]),
        best_resol_thus_far=float(checkpoint.optimiser["rlnBestResolutionThusFar"]),
        incr_size=int(checkpoint.optimiser["rlnIncrementImageSize"]),
        has_high_fsc_at_limit=bool(int(checkpoint.optimiser["rlnHasHighFscAtResolLimit"])),
        ave_pmax=float(models[0].general["rlnAveragePmax"]),
        current_size=int(models[0].general["rlnCurrentImageSize"]),
        particle_pmax=_floats(checkpoint.particles["rlnMaxValueProbDistribution"]).copy(),
    )
    return state


# --- scheduling --------------------------------------------------------------------------------------------------


def update_current_resolution(state: MultibodyState, run: MultibodyRun) -> MultibodyState:
    """``updateCurrentResolution`` (ml_optimiser.cpp:5577-5677): every body's resolution from its curve, the best
    over the bodies, and a stall only when no body gains."""

    no_gain = 0
    best = 0.0
    nr_iter_wo_resol_gain = state.sampling.nr_iter_wo_resol_gain
    best_thus_far = state.best_resol_thus_far
    for track in state.bodies:
        maxres = resolution_from_data_vs_prior(
            track.data_vs_prior, box_size=run.box_size, allow_high_res_recovery=True
        )
        newres = float(maxres) / (run.box_size * run.pixel_size)
        best = max(best, newres)
        if newres <= state.current_resolution_inv + 0.0001:
            no_gain += 1
        else:
            nr_iter_wo_resol_gain = 0
        best_thus_far = max(best_thus_far, newres)
    if no_gain == len(state.bodies):
        nr_iter_wo_resol_gain += 1
    return replace(
        state,
        current_resolution_inv=best,
        best_resol_thus_far=best_thus_far,
        sampling=replace(
            state.sampling, nr_iter_wo_resol_gain=nr_iter_wo_resol_gain, current_resolution=1.0 / best,
        ),
    )


def image_current_size(state: MultibodyState, run: MultibodyRun) -> int:
    """``updateImageSizeAndResolutionPointers``: the size from the run's current resolution."""

    if state.has_converged:
        return run.box_size
    maxres = int(state.current_resolution_inv * run.box_size * run.pixel_size)  # getPixelFromResolution
    return int(compute_current_size_relion(
        maxres, run.box_size, ave_Pmax=state.ave_pmax, has_high_fsc_at_limit=state.has_high_fsc_at_limit,
        incr_size=state.incr_size,
    ))


def converged(state: MultibodyState) -> bool:
    """``checkConvergence`` (ml_optimiser.cpp:10135-10175)."""

    sampling = state.sampling
    return bool(
        sampling.has_fine_enough_angular_sampling
        and sampling.nr_iter_wo_resol_gain >= MAX_NR_ITER_WO_RESOL_GAIN
        and sampling.nr_iter_wo_large_hidden_variable_changes >= MAX_NR_ITER_WO_LARGE_HIDDEN_VARIABLE_CHANGES
    )


def update_angular_sampling(state: MultibodyState) -> MultibodyState:
    """``updateAngularSampling`` (ml_optimiser.cpp:9686-9960) with its multi-body branch: a body whose accuracy is
    worse than the angular step by 10% is kept fixed, and the sampling is fine enough once every body is."""

    sampling = state.sampling
    old_step = effective_angular_step(sampling.healpix_order, sampling.adaptive_oversampling)
    proceed_resolution = (
        resolution_triggers_angular_refinement(sampling) or sampling.nr_iter_wo_resol_gain >= MAX_NR_ITER_WO_RESOL_GAIN
    )
    proceed_hidden = sampling.nr_iter_wo_large_hidden_variable_changes >= MAX_NR_ITER_WO_LARGE_HIDDEN_VARIABLE_CHANGES
    if not (proceed_resolution and proceed_hidden):
        return state
    tracks = []
    all_done = True
    for track in state.bodies:
        if old_step < 0.90 * track.acc_rot and not state.has_converged:
            logger.info("Body with rotational accuracy %.3f deg will be kept fixed", track.acc_rot)
            tracks.append(replace(track, keep_fixed=True))
        else:
            all_done = False
            tracks.append(track)
    if all_done or old_step < 0.75 * sampling.acc_rot:
        return replace(state, bodies=tracks, sampling=replace(sampling, has_fine_enough_angular_sampling=True))
    return replace(state, bodies=tracks, sampling=refine_angular_sampling(replace(sampling, has_fine_enough_angular_sampling=False)))


def update_hidden_variable_changes(state: MultibodyState, previous_poses, run: MultibodyRun, refined) -> MultibodyState:
    """``monitorHiddenVariableChanges`` and ``updateOverallChangesInHiddenVariables`` (ml_optimiser.cpp:9155-9287):
    the mean residual-pose change over every refined body and particle, judged against the MPI leader's sampling
    (fixed at the start, ``RefinementState.mpi_leader_*``)."""

    sampling = state.sampling
    rot_means, offset_squares = [], []
    for b in refined:
        new, old = state.bodies[b].poses, previous_poses[b]
        rot_means.append(compute_relion_orientation_changes(
            euler_angles_to_matrix(new.eulers_deg), euler_angles_to_matrix(old.eulers_deg)
        ))
        offset_squares.append(compute_relion_offset_changes_angstrom(new.offsets_px, old.offsets_px, run.pixel_size) ** 2)
    if refined:
        orientations = float(np.mean(rot_means))
        offsets = float(np.sqrt(np.mean(offset_squares)))
    else:
        orientations = offsets = 0.0
    classes = 0.0
    ratio_orient = orientations / sampling.mpi_leader_hidden_variable_angular_step_deg
    ratio_trans = offsets / sampling.mpi_leader_hidden_variable_translation_step_angstrom
    small = (
        1.03 * classes >= sampling.smallest_changes_optimal_classes
        and (ratio_trans < 0.40 or 1.03 * offsets >= sampling.smallest_changes_optimal_offsets_angstrom)
        and (ratio_orient < 0.40 or 1.03 * orientations >= sampling.smallest_changes_optimal_orientations)
    )
    return replace(state, sampling=replace(
        sampling,
        current_changes_optimal_orientations=orientations,
        current_changes_optimal_offsets_angstrom=offsets,
        current_changes_optimal_classes=classes,
        nr_iter_wo_large_hidden_variable_changes=(
            sampling.nr_iter_wo_large_hidden_variable_changes + 1 if small else 0
        ),
        smallest_changes_optimal_classes=min(sampling.smallest_changes_optimal_classes, float(relion_round(classes))),
        smallest_changes_optimal_offsets_angstrom=min(sampling.smallest_changes_optimal_offsets_angstrom, offsets),
        smallest_changes_optimal_orientations=min(sampling.smallest_changes_optimal_orientations, orientations),
    ))


# --- one iteration -----------------------------------------------------------------------------------------------


def half_projectors(body_maps, run: MultibodyRun, half: int, current_size: int):
    """Each body's masked, centred map of half ``half`` (recovar Fourier) and its projector at ``current_size``;
    ``body_maps[b][half]`` is body ``b``'s stored-frame half map (recovar Fourier, flat)."""

    references, projectors = [], []
    for maps, body in zip(body_maps, run.bodies):
        fourier, projector = _body_projector(maps[half], body, run, current_size)
        references.append(fourier)
        projectors.append(projector)
    return references, projectors


def _body_projector(map_ft, body: Body, run: MultibodyRun, current_size: int, mask=None):
    real = body_projector_reference(map_ft, body, run.dataset.volume_shape, mask=mask)
    fourier = _fourier(real)
    return fourier, prepare_scoring_projector(
        fourier, volume_shape=run.dataset.volume_shape, current_size=current_size,
        padding_factor=run.settings.projection_padding_factor, n_classes=1, real_references=real[None],
        gridding_kernel=run.settings.gridding_kernel, files=ProjectorFiles(),
        stable_window_quantum=run.options.environment.engine.stable_window_quantum,
    )


def target_projector_sets(body_maps, run: MultibodyRun, half: int, current_size: int, projectors) -> list:
    """For each target body, the projectors its images are prepared with: ``sets[i][i]`` is body ``i``'s own
    (scoring) projector and ``sets[i][o]`` the projector body ``o`` is subtracted with, an extra one when the two
    bodies' masks overlap (``run.subtraction_masks``, RELION's ``pointer_body_overlap``)."""

    n = len(run.bodies)
    sets = []
    for i in range(n):
        row = []
        for o in range(n):
            mask = None if o == i else run.subtraction_masks[i][o]
            if mask is None or mask is run.bodies[o].mask:
                row.append(projectors[o])
            else:
                row.append(_body_projector(body_maps[o][half], run.bodies[o], run, current_size, mask=mask)[1])
        sets.append(row)
    return sets


def _expected_accuracy(state: MultibodyState, run: MultibodyRun, references_half1, current_size: int, iteration: int):
    """``calculateExpectedAngularErrors`` on half 1: every body's masked centred map at the consensus poses of the
    first 100 particles of half 1 in RELION's order of this iteration."""

    half = run.halves[0]
    if run.random_seed == 0:
        order = np.arange(half.rows.size)  # relion_refine skips the shuffle with seed 0 (iterate, ml_optimiser_mpi.cpp)
    else:
        from relax.sampling.expected_accuracy import relion_auto_refine_half_orders

        subsets = np.asarray(run.checkpoint.particles["rlnRandomSubset"], dtype=np.int64)
        order = np.searchsorted(half.rows, relion_auto_refine_half_orders(subsets, run.random_seed, iteration)[0])
    eulers = np.stack(
        [_floats(run.checkpoint.particles[c])[half.rows] for c in ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi")], axis=1
    )
    accuracy = estimate_relion_expected_accuracy(
        reference_fourier=np.stack(references_half1),
        volume_shape=tuple(run.dataset.volume_shape),
        best_eulers_deg=eulers,
        class_ids=np.zeros(half.rows.size, dtype=np.int32),
        class_weights=np.ones(len(run.bodies)),
        sigma2_noise_native=np.asarray(state.noise.radial_per_half[0], dtype=np.float64),
        dataset=run.dataset.subset(half.rows),
        trial_order_local=order,
        current_image_size=current_size,
        padding_factor=run.settings.projection_padding_factor,
        sigma2_fudge=1.0,
        random_seed=run.random_seed,
        random_seed_particle_ids=half.rows[order],
        do_ctf_correction=run.do_ctf_correction,
        device_memory_override=run.options.environment.engine.device_memory_bytes,
        stable_window_quantum=run.options.environment.engine.stable_window_quantum,
    )
    return accuracy


def _local_sampling(state: MultibodyState, run: MultibodyRun, current_size: int, coarse_grids, coarse_size_order: int):
    trial_grid = iteration_trial_grid(
        coarse_grids, state.sampling, run.options, state.perturbation,
        perturbation_order=coarse_grids.rotation_grid.healpix_order, sealed_grid=False,
        dtype=run.options.precision.rotation_real_dtype,
    )
    coarse_grids = replace(coarse_grids, translations=trial_grid.translations)
    plan = plan_expectation_sampling(
        trial_grid, coarse_grids, state.sampling, run.options, run.optics, current_size=current_size,
        use_local=True, perturbation=state.perturbation, coarse_size_healpix_order=coarse_size_order,
    )
    return plan.local


def _summed_noise_stats(stats):
    from relax.refinement.noise_updates import _combined_noise_stats

    return _combined_noise_stats(stats) if stats else make_noise_stats(
        wsum_sigma2_noise=0.0, wsum_img_power=0.0, wsum_sigma2_offset=0.0, sumw=0.0
    )


@dataclass(frozen=True)
class IterationInputs:
    """What an iteration's expectation reads besides the state: its number, the image size, the HEALPix order that
    sizes pass 1 (the order before this iteration's sampling update), and each half's masked, centred body maps
    (recovar Fourier) and their projectors."""

    iteration: int
    current_size: int
    coarse_size_order: int
    references: tuple
    projectors: tuple
    target_projectors: tuple = None  # [half][target][body]: target_projector_sets


def body_projectors(state: MultibodyState, run: MultibodyRun, iteration: int, current_size: int, coarse_size_order: int):
    """``setFourierTransformMaps`` for both halves: :class:`IterationInputs` of ``state``'s maps."""

    maps = [track.maps for track in state.bodies]
    references, projectors = zip(*(half_projectors(maps, run, h, current_size) for h in range(2)))
    return IterationInputs(
        iteration=iteration, current_size=current_size, coarse_size_order=coarse_size_order,
        references=references, projectors=projectors,
        target_projectors=tuple(target_projector_sets(maps, run, h, current_size, projectors[h]) for h in range(2)),
    )


def plan_iteration(state: MultibodyState, run: MultibodyRun):
    """The decisions RELION takes before an expectation (``iterate`` up to ``expectation``'s loop): convergence, the
    image size, the projectors, the expected accuracies, the sampling update (which may keep bodies fixed) and this
    iteration's perturbation. Returns the state with them and the :class:`IterationInputs`."""

    iteration = state.iteration + 1
    if iteration > 1 and converged(state):
        logger.info("Multi-body refinement has converged: the last iteration joins the halves at the full size")
        # checkConvergence: every body with a prior is refined again in the last iteration (ml_optimiser.cpp:10162).
        state = replace(state, has_converged=True, bodies=[
            replace(track, keep_fixed=body.keep_fixed) for track, body in zip(state.bodies, run.bodies)
        ])
    current_size = image_current_size(state, run)
    inputs = body_projectors(state, run, iteration, current_size, state.sampling.healpix_order)
    # calculateExpectedAngularErrors with the projectors of half 1.
    accuracy = _expected_accuracy(state, run, inputs.references[0], current_size, iteration)
    tracks = [
        replace(track, acc_rot=float(accuracy.acc_rot_per_class[b]), acc_trans=float(accuracy.acc_trans_per_class_angstrom[b]))
        for b, track in enumerate(state.bodies)
    ]
    state = replace(state, bodies=tracks, sampling=replace(
        state.sampling, acc_rot=float(accuracy.acc_rot), acc_trans=float(accuracy.acc_trans_angstrom)
    ))
    logger.info(
        "Multi-body iteration %d expected accuracy: angles %.4f deg, offsets %.4f A; per body %s",
        iteration, accuracy.acc_rot, accuracy.acc_trans_angstrom,
        [(round(t.acc_rot, 4), round(t.acc_trans, 4)) for t in tracks],
    )
    if iteration > 1:
        state = update_angular_sampling(state)
    perturbation = advance_relion_perturbation_from_seed(state.perturbation, run.perturb_factor, seed=run.random_seed + iteration)
    return replace(state, perturbation=perturbation, current_size=current_size), inputs


def run_iteration(state: MultibodyState, run: MultibodyRun, coarse_grids):
    """One RELION multi-body iteration from ``state``; returns the new state and the coarse grids."""

    state, inputs = plan_iteration(state, run)
    return execute_iteration(state, run, inputs, coarse_grids)


def execute_iteration(state: MultibodyState, run: MultibodyRun, inputs: IterationInputs, coarse_grids):
    """An iteration's expectation and maximisation with the decisions ``state`` holds (``plan_iteration``'s, or a
    replayed RELION iteration's)."""

    iteration, current_size = inputs.iteration, inputs.current_size
    references, projectors = inputs.references, inputs.projectors
    coarse_grids = refresh_coarse_grids(
        coarse_grids, state.sampling, run.options, voxel_size=run.pixel_size,
        dtype=run.options.precision.rotation_real_dtype, log=logger,
    )
    refined = [b for b, track in enumerate(state.bodies) if not track.keep_fixed]
    logger.info(
        "Multi-body iteration %d: current_size=%d healpix_order=%d offset_range=%.4f px refined bodies=%s",
        iteration, current_size, state.sampling.healpix_order, state.sampling.translation_range,
        [b + 1 for b in refined],
    )
    if not refined:
        # Every body is fixed: RELION's expectation and maximisation return at once (ml_optimiser_mpi.cpp:1128-1136,
        # 2358-2365); the resolution is re-evaluated as usual.
        state = update_current_resolution(replace(state, iteration=iteration), run)
        for b, track in enumerate(state.bodies):
            run_files.write_fixed_body_maps(run.output, iteration, b, track.maps, run)
        run_files.write_iteration(run.output, iteration, state, run)
        return state, coarse_grids

    sampling = _local_sampling(state, run, current_size, coarse_grids, inputs.coarse_size_order)
    previous_poses = [track.poses for track in state.bodies]
    offset_range_angstrom = state.sampling.translation_range * run.pixel_size

    def score(b, half, half_poses):
        h = half.index
        return score_body_half(
            b, run.bodies, dataset=run.dataset, particles=half, poses=half_poses,
            projectors=list(inputs.target_projectors[h][b]), reference=references[h][b],
            noise_variance=state.noise.variance_per_half[h], noise_radial=state.noise.radial_per_half[h],
            data_vs_prior=state.bodies[b].data_vs_prior, sampling=sampling, offset_range_angstrom=offset_range_angstrom, current_size=current_size,
            padding_factor=run.settings.projection_padding_factor, options=run.options, iteration=iteration - 1,
        )

    results, poses = score_bodies(run.halves, previous_poses, refined, score)

    if state.has_converged:
        return _final_maximization(state, run, results, poses, refined, current_size), coarse_grids

    # Maximisation, one body at a time (ml_optimiser_mpi.cpp:2347-2704), then the shared parameters.
    this_iteration = NumberedIteration(
        iteration=iteration - 1, numbered_relion_iteration=iteration, has_previous_iteration=True,
        first_iteration=FirstIterationPolicy(relion_firstiter_cc=False, score_mode="gaussian", winner_take_all=False),
        seeding=ClassSeeding(single_class_iteration=False, seed_classes=None), current_size=current_size,
    )
    tracks = list(state.bodies)
    incr_size, has_high = state.incr_size, state.has_high_fsc_at_limit
    for b in refined:
        pair = results[b]
        shapes = [r.mstep_accumulator_shape for r in pair if r.mstep_accumulator_shape is not None]
        accumulator_shape = shapes[0] if shapes else tuple(run.settings.padding_factor * d for d in run.dataset.volume_shape)
        body_mstep = maximize_body(
            run.bodies[b], [r.Ft_y for r in pair], [r.Ft_ctf for r in pair], run.settings,
            current_size=current_size, accumulator_shape=accumulator_shape,
            full_half_axes=[r.mstep_full_half_axis for r in pair],
            low_resolution_join_angstrom=run.low_resolution_join,
            current_resolution_angstrom=1.0 / state.current_resolution_inv,
            phase_stream=run.phase_stream, this_iteration=this_iteration,
            scoring_dtype=run.options.precision.rotation_real_dtype,
        )
        incr_size, has_high = update_relion_growth_state_from_fsc(
            body_mstep.fsc, current_size, incr_size=incr_size, has_high_fsc_at_limit=has_high
        )
        details = body_mstep.prior.details_per_half
        tracks[b] = replace(
            tracks[b], maps=body_mstep.maps, fsc=body_mstep.fsc,
            data_vs_prior=np.asarray(details[0]["ssnr_shells"], dtype=np.float64),
            tau2=[np.asarray(d["prior_shells"], dtype=np.float64) for d in details],
            poses=poses[b],
        )
        run_files.write_body_maps(run.output, iteration, b, body_mstep, run)
    for b, track in enumerate(tracks):
        if b not in refined:
            run_files.write_fixed_body_maps(run.output, iteration, b, track.maps, run)

    # maximizationOtherParameters: the noise from every refined body's sums; norm, scales and sigma2_offset are
    # not updated in a multi-body refinement (ml_optimiser.cpp:5111-5121, 5219).
    noise_stats = [_summed_noise_stats([results[b][h].noise_stats for b in refined]) for h in range(2)]
    summed_shell_count = run.options.consistency.noise_shell_count == "summed"
    noise = noise_updates.update_k1_posterior_noise_variance(
        noise_stats, state.noise, run.dataset.image_shape, firstiter_cc=False, ctf_premultiplied=False,
        # RELION divides each shell by its full-image pixel count (Npix_per_shell), so the shell at the current
        # size's Nyquist, partly summed, dips; the "summed" consistency option divides by the pixels summed.
        summed_current_size=current_size if summed_shell_count else None,
        nyquist_column_counting=run.options.consistency.nyquist_column_counting,
    ).model
    # ave_Pmax (maximizationOtherParameters, ml_optimiser.cpp:5298) is half 1's sum of each particle's stored Pmax,
    # added once per refined body, over its summed posterior mass; only body 1 stores a Pmax
    # (acc_ml_optimiser_impl.h:2921-2930, 3581), so another body adds body 1's, from this or an earlier iteration.
    particle_pmax = state.particle_pmax.copy()
    if 0 in refined:
        for h, half in enumerate(run.halves):
            particle_pmax[half.rows] = np.asarray(results[0][h].em_stats.max_posterior_per_image, dtype=np.float64)
    half1 = run.halves[0]
    ave_pmax = len(refined) * float(np.sum(particle_pmax[half1.rows])) / sum(
        total_sumw(results[b][0].noise_stats.sumw) for b in refined
    )
    state = replace(
        state, iteration=iteration, bodies=tracks, noise=noise, incr_size=incr_size, has_high_fsc_at_limit=has_high,
        ave_pmax=ave_pmax, particle_pmax=particle_pmax,
    )
    state = update_hidden_variable_changes(state, previous_poses, run, refined)
    state = update_current_resolution(state, run)
    run_files.write_iteration(run.output, iteration, state, run)
    return state, coarse_grids


def score_bodies(halves, poses, refined, score):
    """The expectation of every refined body on both halves, in RELION's order: within a half, body after body, each
    scored against the poses the bodies before it were just given (``accDoExpectationOneParticle``). ``score(b,
    half, half_poses)`` scores body ``b`` on ``half`` with every body's poses of that half's particles and returns a
    :class:`relax.multibody.estep.BodyScore`. Returns ``{b: [half 1 result, half 2 result]}`` and every body's new
    poses."""

    poses = list(poses)
    results = {b: [None, None] for b in refined}
    for half in halves:
        half_poses = [BodyPoses(p.eulers_deg[half.rows], p.offsets_px[half.rows]) for p in poses]
        for b in refined:
            scored = score(b, half, list(half_poses))
            half_poses[b] = scored.poses
            results[b][half.index] = scored.result
        for b in refined:
            eulers, offsets = poses[b].eulers_deg.copy(), poses[b].offsets_px.copy()
            eulers[half.rows], offsets[half.rows] = half_poses[b].eulers_deg, half_poses[b].offsets_px
            poses[b] = BodyPoses(eulers, offsets)
    return results, poses


def _final_maximization(state: MultibodyState, run: MultibodyRun, results, poses, refined, current_size) -> MultibodyState:
    """The last iteration's maximisation: each body's two halves joined (``combineWeightedSumsTwoRandomHalves``)
    and reconstructed to the full size; the run's files without an iteration number."""

    tracks = list(state.bodies)
    for b in refined:
        pair = results[b]
        shapes = [r.mstep_accumulator_shape for r in pair if r.mstep_accumulator_shape is not None]
        axes = [r.mstep_full_half_axis for r in pair if r.mstep_full_half_axis is not None]
        final = maximize_body_final(
            run.bodies[b], [r.Ft_y for r in pair], [r.Ft_ctf for r in pair], run.settings,
            current_size=current_size,
            accumulator_shape=shapes[0] if shapes else tuple(run.settings.padding_factor * d for d in run.dataset.volume_shape),
            full_half_axis=axes[0] if axes else -1,
            low_resolution_join_angstrom=run.low_resolution_join,
            current_resolution_angstrom=1.0 / state.current_resolution_inv,
            phase_stream=run.phase_stream, scoring_dtype=run.options.precision.rotation_real_dtype,
        )
        tracks[b] = replace(tracks[b], maps=[final.map, final.map], fsc=final.fsc, poses=poses[b])
        run_files.write_final_body_maps(run.output, b, final, run)
    state = replace(state, iteration=state.iteration + 1, bodies=tracks)
    run_files.write_iteration(run.output, state.iteration, state, run, final=True)
    return state


def refine(run: MultibodyRun, state: MultibodyState) -> MultibodyState:
    """Iterate until RELION's convergence and its last, halves-joined iteration (or ``max_iter``)."""

    run_files.write_iteration(run.output, 0, state, run)
    # iterate() starts with updateCurrentResolution (ml_optimiser_mpi.cpp:3854).
    state = update_current_resolution(state, run)
    coarse_grids = build_initial_coarse_grids(
        state.sampling.healpix_order, None, translation_range=state.sampling.translation_range,
        translation_step=state.sampling.translation_step, n_classes=1, voxel_size=run.pixel_size,
        dtype=run.options.precision.rotation_real_dtype,
    )
    while state.iteration < run.command.max_iter:
        state, coarse_grids = run_iteration(state, run, coarse_grids)
        if state.has_converged:
            break
    return state
