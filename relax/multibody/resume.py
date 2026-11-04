"""Continue a multi-body refinement from a written iteration, as ``relion_refine --continue run_itNNN_optimiser.star``.

RELION's continuation reads the iteration's files and nothing else, so it carries what they hold and no more:

- the maps are the written half maps (single precision), the poses the data STAR's (six decimals), the curves,
  noise and perturbation the STAR files' values;
- the leader's ``rand()`` stream starts again from ``init_random_generator(random_seed)``
  (ml_optimiser_mpi.cpp:965), so the phase randomisation of the next FSCs is not the uninterrupted run's;
- ``iterate()`` begins with ``updateCurrentResolution`` (ml_optimiser_mpi.cpp:4028), which the iteration already
  applied before writing its files, so the stall counter can step once more than in an uninterrupted run.

relax does the same: :func:`continue_refinement` builds the run from the consensus checkpoint as a fresh start does
(the data, halves and random streams of ``relax multibody``), takes the state from the iteration files
(:func:`iteration_state`) and runs :func:`relax.multibody.refine.refine` from there without rewriting iteration 0.
The files may be relax's or RELION's own multi-body iteration.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import starfile
from recovar.utils.helpers import load_relion_volume

from relax.multibody import refine
from relax.multibody.subtraction import BodyPoses
from relax.refinement import noise_updates
from relax.sampling.healpix import angular_sampling_deg
from relax.star.blocks import read_star_blocks

logger = logging.getLogger(__name__)


def iteration_state(run: refine.MultibodyRun, optimiser_star) -> refine.MultibodyState:
    """The multi-body state an iteration's files hold: RELION's (``rlnModelStarFile`` names its models) or relax's."""

    optimiser_star = Path(optimiser_star)
    optimiser = _general(optimiser_star)
    if "rlnModelStarFile" in optimiser:
        from relax.multibody.replay import relion_multibody_state

        return relion_multibody_state(run, optimiser_star)
    return _relax_state(run, optimiser_star, optimiser)


def _general(path: Path) -> dict:
    """An optimiser STAR's general block as scalars (relax writes it as a one-row loop, RELION as pairs)."""

    block = read_star_blocks(path)["optimiser_general"]
    return {key: (value[0] if isinstance(value, (list, tuple)) else value) for key, value in block.items()}


def _float(table, key, default=None) -> float:
    if key not in table:
        if default is None:
            raise KeyError(f"{key} is missing")
        return default
    value = table[key]
    return float(value.iloc[0] if hasattr(value, "iloc") else value)


def _relax_state(run: refine.MultibodyRun, optimiser_star: Path, optimiser: dict) -> refine.MultibodyState:
    stem = str(optimiser_star)[: -len("_optimiser.star")]
    iteration = int(optimiser["rlnCurrentIteration"])
    model = starfile.read(f"{stem}_model.star", always_dict=True)
    data = starfile.read(f"{stem}_data.star", always_dict=True)
    frame = float(run.box_size) ** 4
    start = refine._initial_state(run)
    general, bodies = model["model_general"], model["model_bodies"]
    tracks = []
    for b, track0 in enumerate(start.bodies):
        shells = model[f"model_body_{b + 1}"]
        table = data[f"images_body_{b + 1}"]
        tracks.append(replace(
            track0,
            maps=[
                refine._fourier(load_relion_volume(f"{stem}_half{h + 1}_body{b + 1:03d}.mrc")) for h in range(2)
            ],
            data_vs_prior=shells["rlnSsnrMap"].to_numpy(np.float64),
            fsc=shells["rlnGoldStandardFsc"].to_numpy(np.float64),
            tau2=[shells[f"rlnReferenceTau2Half{h + 1}"].to_numpy(np.float64) * frame for h in range(2)],
            acc_rot=float(bodies["rlnAccuracyRotations"].iloc[b]),
            acc_trans=float(bodies["rlnAccuracyTranslationsAngst"].iloc[b]),
            keep_fixed=bool(int(bodies["rlnBodyKeepFixed"].iloc[b])),
            poses=BodyPoses(
                eulers_deg=table[["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]].to_numpy(np.float64),
                offsets_px=table[["rlnOriginXAngst", "rlnOriginYAngst"]].to_numpy(np.float64) / run.pixel_size,
            ),
            fourier_coverage=(
                shells["rlnFourierCompleteness"].to_numpy(np.float64) if "rlnFourierCompleteness" in shells else None
            ),
        ))
    noise = noise_updates.noise_model_from_shells(
        [model[f"model_noise_half{h + 1}"]["rlnSigma2Noise"].to_numpy(np.float64) * frame for h in range(2)],
        run.dataset.image_shape,
    )

    healpix_order = int(optimiser["rlnHealpixOrder"])
    sampling0 = start.sampling
    current_resolution = _float(general, "rlnCurrentResolution")
    earlier = _earlier_optimisers(optimiser_star, iteration)
    if "rlnSigmaPriorRotAngle" in optimiser:
        sigma_rot = np.deg2rad(float(optimiser["rlnSigmaPriorRotAngle"]))
        sigma_psi = np.deg2rad(float(optimiser["rlnSigmaPriorPsiAngle"]))
    elif healpix_order > sampling0.healpix_order and healpix_order >= sampling0.auto_local_healpix_order:
        # Files from before relax wrote the prior: refine_angular_sampling set it when it raised the order,
        # sqrt(2 * 2) times the new step over 2^oversampling (refinement_state.py).
        sigma_rot = sigma_psi = 2.0 * np.deg2rad(angular_sampling_deg(healpix_order) / 2**sampling0.adaptive_oversampling)
    else:
        sigma_rot, sigma_psi = sampling0.sigma_rot, sampling0.sigma_psi
    smallest = {
        key: float(optimiser[key]) if key in optimiser else min(
            [initial] + [float(o["rlnChangesOptimalOrientations" if "Orient" in key else "rlnChangesOptimalOffsets"])
                         for o in earlier]
        )
        for key, initial in (
            ("rlnSmallestChangesOrientations", sampling0.smallest_changes_optimal_orientations),
            ("rlnSmallestChangesOffsets", sampling0.smallest_changes_optimal_offsets_angstrom),
        )
    }
    if "rlnBestResolutionThusFar" in optimiser:
        best_resol_thus_far = float(optimiser["rlnBestResolutionThusFar"])
    else:
        resolutions = [_float(starfile.read(str(o["_path"])[: -len("_optimiser.star")] + "_model.star",
                                            always_dict=True)["model_general"], "rlnCurrentResolution")
                       for o in earlier]
        best_resol_thus_far = max([start.best_resol_thus_far] + [1.0 / r for r in resolutions])
    sampling = replace(
        sampling0,
        iteration=iteration,
        healpix_order=healpix_order,
        angular_step=angular_sampling_deg(healpix_order),
        translation_range=float(optimiser["rlnOffsetRange"]) / run.pixel_size,
        translation_step=float(optimiser["rlnOffsetStep"]) / run.pixel_size,
        current_resolution=current_resolution,
        previous_resolution=float(optimiser.get("rlnPreviousResolution", current_resolution)),
        nr_iter_wo_resol_gain=int(optimiser["rlnNumberOfIterWithoutResolutionGain"]),
        nr_iter_wo_large_hidden_variable_changes=int(optimiser["rlnNumberOfIterWithoutChangingAssignments"]),
        has_fine_enough_angular_sampling=bool(int(optimiser["rlnHasFineEnoughAngularSampling"])),
        current_changes_optimal_orientations=float(optimiser["rlnChangesOptimalOrientations"]),
        current_changes_optimal_offsets_angstrom=float(optimiser["rlnChangesOptimalOffsets"]),
        smallest_changes_optimal_orientations=smallest["rlnSmallestChangesOrientations"],
        smallest_changes_optimal_offsets_angstrom=smallest["rlnSmallestChangesOffsets"],
        # Files from before relax wrote it: a multi-body iteration's class changes are 0, so after iteration 1 the
        # smallest is min(start, round(0)) = 0 (refine.py); the start value would fail every "small changes" test.
        smallest_changes_optimal_classes=float(
            optimiser.get("rlnSmallestChangesClasses", sampling0.smallest_changes_optimal_classes if iteration < 1 else 0.0)
        ),
        acc_rot=float(optimiser["rlnOverallAccuracyRotations"]),
        acc_trans=float(optimiser["rlnOverallAccuracyTranslationsAngst"]),
        sigma_rot=float(sigma_rot),
        sigma_psi=float(sigma_psi),
        do_local_search=sampling0.do_local_search or healpix_order >= sampling0.auto_local_healpix_order,
        # RELION's leader takes its hidden-variable steps from the sampling it reads (0 recomputes them).
        mpi_leader_hidden_variable_angular_step_deg=0.0,
        mpi_leader_hidden_variable_translation_step_angstrom=0.0,
    )
    return refine.MultibodyState(
        iteration=iteration,
        bodies=tracks,
        noise=noise,
        sampling=sampling,
        perturbation=float(optimiser["rlnSamplingPerturbInstance"]),
        current_resolution_inv=1.0 / current_resolution,
        best_resol_thus_far=best_resol_thus_far,
        incr_size=int(optimiser["rlnIncrementImageSize"]),
        has_high_fsc_at_limit=bool(int(optimiser["rlnHasHighFscAtResolLimit"])),
        ave_pmax=_float(general, "rlnAveragePmax"),
        current_size=int(_float(general, "rlnCurrentImageSize")),
        particle_pmax=data["particles"]["rlnMaxValueProbDistribution"].to_numpy(np.float64),
        has_converged=bool(int(optimiser["rlnHasConverged"])),
    )


def _earlier_optimisers(optimiser_star: Path, iteration: int) -> list:
    """relax's optimiser tables of iterations 1..``iteration`` in the same directory (for files that lack a field)."""

    tables = []
    for k in range(1, iteration + 1):
        path = optimiser_star.parent / f"run_it{k:03d}_optimiser.star"
        if path.exists():
            table = _general(path)
            table["_path"] = path
            tables.append(table)
    return tables


def continue_refinement(run: refine.MultibodyRun, optimiser_star, *, as_uninterrupted: bool = False) -> refine.MultibodyState:
    """Continue a multi-body refinement after a written iteration.

    By default as ``relion_refine --continue`` does: the first follower's (half 1's) noise on every rank, the
    leader's hidden-variable steps from the sampling it reads, a new ``rand()`` stream and a second resolution
    update (relax issue #7 and the module notes). ``as_uninterrupted`` instead resumes the run the files came from:
    each half's noise, the start's leader steps, the stream past the draws the files record, and no second update;
    from relax's files with ``rlnRelaxPhaseDraws`` it repeats the uninterrupted run up to the files' rounding.
    """

    state = iteration_state(run, optimiser_star)
    if state.has_converged:
        raise ValueError(f"{optimiser_star} is the last iteration of a converged run; there is nothing to continue")
    radial = [np.asarray(r) for r in state.noise.radial_per_half]
    if as_uninterrupted:
        start = refine._initial_state(run).sampling
        state = replace(state, sampling=replace(
            state.sampling,
            mpi_leader_hidden_variable_angular_step_deg=start.mpi_leader_hidden_variable_angular_step_deg,
            mpi_leader_hidden_variable_translation_step_angstrom=start.mpi_leader_hidden_variable_translation_step_angstrom,
        ))
        draws = int(float(_general(Path(optimiser_star)).get("rlnRelaxPhaseDraws", -1)))
        if draws >= 0:
            run.phase_stream.rand_array(draws)
        else:
            logger.warning("%s records no rand() draws: the next FSCs' phase randomisation is not the run's", optimiser_star)
    else:
        state = replace(state, noise=noise_updates.noise_model_from_shells([radial[0], radial[0]], run.dataset.image_shape))
    logger.info(
        "Continuing the multi-body refinement after iteration %d from %s (%s)", state.iteration, optimiser_star,
        "as the uninterrupted run" if as_uninterrupted else "as relion_refine --continue",
    )
    return refine.refine(run, state, write_start=False, update_at_start=not as_uninterrupted)
