"""Fixed-state replay of one RELION multi-body iteration (a diagnostic, not part of a refinement).

The state after RELION's iteration ``N - 1`` (maps, body poses, noise and curves from its ``run_it<N-1>`` files)
and the decisions RELION took for iteration ``N`` (image size, sampling, perturbation, fixed bodies, from its
``run_it<N>`` files) go through relax's expectation and maximisation (:func:`relax.multibody.refine.execute_iteration`),
so relax's iteration ``N`` can be compared with RELION's from identical inputs. A refinement never reads RELION's
later files: :func:`relax.multibody.refine.refine` evolves its own state from the consensus checkpoint.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from recovar.utils.helpers import load_relion_volume

from relax.multibody import refine
from relax.multibody.checkpoint import read_relion_checkpoint
from relax.multibody.subtraction import BodyPoses
from relax.reconstruction.solvent_mask import _fftw_shells
from relax.refinement import noise_updates
from relax.refinement.trial_grids import build_initial_coarse_grids
from relax.relion.macros import relion_round
from relax.sampling.perturbation import advance_relion_perturbation_from_seed


def relion_multibody_state(run: refine.MultibodyRun, optimiser_star) -> refine.MultibodyState:
    """The multi-body state RELION wrote at the end of an iteration (``run_itNNN_optimiser.star``)."""

    checkpoint = read_relion_checkpoint(optimiser_star, relion_cwd=Path(optimiser_star).resolve().parent.parent)
    models = checkpoint.models
    frame = float(run.box_size) ** 4
    tracks = []
    for b, table in enumerate(checkpoint.body_tables):
        shells = [model.class_shells[b] for model in models]
        tracks.append(refine.BodyTrack(
            maps=[refine._fourier(load_relion_volume(str(checkpoint.half_map_paths[h][b]))) for h in range(2)],
            data_vs_prior=shells[0]["rlnSsnrMap"].copy(),
            fsc=shells[0]["rlnGoldStandardFsc"].copy(),
            tau2=[s["rlnReferenceTau2"] * frame for s in shells],
            acc_rot=float(models[0].classes["rlnAccuracyRotations"][b]),
            acc_trans=float(models[0].classes["rlnAccuracyTranslationsAngst"][b]),
            keep_fixed=bool(int(models[0].classes["rlnBodyKeepFixed"][b])),
            poses=BodyPoses(
                eulers_deg=table[["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]].to_numpy(np.float64),
                offsets_px=table[["rlnOriginXAngst", "rlnOriginYAngst"]].to_numpy(np.float64) / run.pixel_size,
            ),
        ))
    general = models[0].general
    optimiser = checkpoint.optimiser
    sampling = replace(
        refine._initial_state(run).sampling,
        healpix_order=int(checkpoint.sampling["rlnHealpixOrder"]),
        translation_range=float(checkpoint.sampling["rlnOffsetRange"]) / run.pixel_size,
        translation_step=float(checkpoint.sampling["rlnOffsetStep"]) / run.pixel_size,
        sigma_rot=np.deg2rad(float(general["rlnSigmaPriorRotAngle"])),
        sigma_psi=np.deg2rad(float(general["rlnSigmaPriorPsiAngle"])),
        nr_iter_wo_resol_gain=int(optimiser["rlnNumberOfIterWithoutResolutionGain"]),
        nr_iter_wo_large_hidden_variable_changes=int(optimiser["rlnNumberOfIterWithoutChangingAssignments"]),
        current_changes_optimal_orientations=float(optimiser["rlnChangesOptimalOrientations"]),
        current_changes_optimal_offsets_angstrom=float(optimiser["rlnChangesOptimalOffsets"]),
        smallest_changes_optimal_orientations=float(optimiser["rlnSmallestChangesOrientations"]),
        smallest_changes_optimal_offsets_angstrom=float(optimiser["rlnSmallestChangesOffsets"]),
        acc_rot=float(optimiser["rlnOverallAccuracyRotations"]),
        acc_trans=float(optimiser["rlnOverallAccuracyTranslationsAngst"]),
    )
    return refine.MultibodyState(
        iteration=checkpoint.iteration,
        bodies=tracks,
        noise=noise_updates.noise_model_from_shells([m.sigma2_noise[0] * frame for m in models], run.dataset.image_shape),
        sampling=sampling,
        perturbation=float(checkpoint.sampling["rlnSamplingPerturbInstance"]),
        current_resolution_inv=1.0 / float(general["rlnCurrentResolution"]),
        best_resol_thus_far=float(optimiser["rlnBestResolutionThusFar"]),
        incr_size=int(optimiser["rlnIncrementImageSize"]),
        has_high_fsc_at_limit=bool(int(optimiser["rlnHasHighFscAtResolLimit"])),
        ave_pmax=float(general["rlnAveragePmax"]),
        current_size=int(general["rlnCurrentImageSize"]),
        particle_pmax=np.asarray(checkpoint.particles["rlnMaxValueProbDistribution"], dtype=np.float64),
    )


def _exact_perturbation(run: refine.MultibodyRun, iteration: int) -> float:
    """The iteration's perturbation at full precision. RELION's star holds 6 digits, and at --pad 1 the last bits of
    the rows it builds decide which pixels are scored (:func:`relax.multibody.layout._device_parent_rows`)."""

    perturbation = refine._initial_state(run).perturbation
    for k in range(1, iteration + 1):
        perturbation = advance_relion_perturbation_from_seed(perturbation, run.perturb_factor, seed=run.random_seed + k)
    return perturbation


def relion_randomize_shells(stdout_log, box_size: int, pixel_size: float) -> list[int]:
    """The ``randomize_at`` shell of each solvent-corrected FSC of a RELION run, in order, from its log."""

    shells = []
    for line in Path(stdout_log).read_text(errors="ignore").splitlines():
        if "randomize phases beyond:" in line:
            angstrom = float(line.split("beyond:")[1].split()[0])
            shells.append(relion_round(box_size * pixel_size / angstrom))
    return shells


def advance_phase_stream(stream, shells, box_size: int) -> None:
    """Put the leader's ``rand()`` stream past the FSCs already drawn: each one draws a value per FFTW half voxel at
    ``|k| >= shell`` for each half map (:func:`relax.reconstruction.solvent_mask.randomize_phases_beyond`)."""

    _, radius_sq = _fftw_shells(box_size)
    for shell in shells:
        stream.rand_array(2 * int(np.count_nonzero(radius_sq >= shell * shell)))


def replay_iteration(run: refine.MultibodyRun, relion_dir, iteration: int) -> refine.MultibodyState:
    """relax's iteration ``iteration`` from RELION's state after ``iteration - 1`` and RELION's decisions for it."""

    relion_dir = Path(relion_dir)
    log = relion_dir / "stdout.log"
    if log.exists():
        shells = relion_randomize_shells(log, run.box_size, run.pixel_size)
        refined = [
            sum(not track.keep_fixed for track in relion_multibody_state(run, relion_dir / f"run_it{k:03d}_optimiser.star").bodies)
            for k in range(1, iteration)
        ]
        advance_phase_stream(run.phase_stream, shells[: sum(refined)], run.box_size)
    before = relion_multibody_state(run, relion_dir / f"run_it{iteration - 1:03d}_optimiser.star")
    after = relion_multibody_state(run, relion_dir / f"run_it{iteration:03d}_optimiser.star")
    perturbation = _exact_perturbation(run, iteration)
    if not np.isclose(perturbation, after.perturbation, rtol=0.0, atol=1e-4):
        raise ValueError(f"perturbation {perturbation} does not reproduce RELION's {after.perturbation}")
    start = refine._initial_state(run).sampling
    # The star rounds the offset grid too; before RELION first narrows it, the start's exact values apply.
    same_grid = np.isclose(after.sampling.translation_range, start.translation_range, rtol=1e-5)
    state = replace(
        before,
        sampling=replace(
            before.sampling,
            healpix_order=after.sampling.healpix_order,
            translation_range=start.translation_range if same_grid else after.sampling.translation_range,
            translation_step=start.translation_step if same_grid else after.sampling.translation_step,
            sigma_rot=after.sampling.sigma_rot,
            sigma_psi=after.sampling.sigma_psi,
        ),
        perturbation=perturbation,
        current_size=after.current_size,
        bodies=[replace(track, keep_fixed=done.keep_fixed) for track, done in zip(before.bodies, after.bodies)],
    )
    inputs = refine.body_projectors(state, run, iteration, after.current_size, before.sampling.healpix_order)
    coarse_grids = build_initial_coarse_grids(
        state.sampling.healpix_order, None, translation_range=state.sampling.translation_range,
        translation_step=state.sampling.translation_step, n_classes=1, voxel_size=run.pixel_size,
        dtype=run.options.precision.rotation_real_dtype,
    )
    state, _ = refine.execute_iteration(state, run, inputs, coarse_grids)
    return state
