"""The multi-body run's output files, named as relion_refine names them.

Per iteration ``NNN``: ``run_itNNN_half<h>_body<BBB>.mrc`` and ``..._unfil.mrc`` (stored frame, RELION map
convention), ``run_itNNN_data.star`` with the consensus particle table and one ``images_body_<b>`` table of residual
poses per body (``Experiment::write``), and ``run_itNNN_model.star`` with each body's shell curves and the noise. The
last, halves-joined iteration writes ``run_body<BBB>.mrc``, ``run_half<h>_body<BBB>_unfil.mrc`` and ``run_data.star``.
These files are outputs; a multi-body run is not continued from them. ``run_itNNN_model.star`` is relax's summary of
the iteration; the last iteration's ``run_model.star`` is a RELION model STAR (``MlModel::write``), which
``relion_flex_analyse`` reads with ``run_data.star``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import starfile

from relax.io.map_io import write_map


def _stem(output: Path, iteration: int | None) -> Path:
    return output / ("run" if iteration is None else f"run_it{iteration:03d}")


def _real(map_ft, volume_shape):
    import jax.numpy as jnp
    from recovar.core import fourier_transform_utils as ftu

    return np.asarray(ftu.get_idft3(jnp.asarray(map_ft).reshape(volume_shape)).real, dtype=np.float32)


def write_body_maps(output, iteration: int, body_index: int, mstep, run) -> None:
    """One body's half maps and unfiltered half maps of a numbered iteration."""

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stem = _stem(output, iteration)
    for h in range(2):
        write_map(f"{stem}_half{h + 1}_body{body_index + 1:03d}.mrc", _real(mstep.maps[h], run.dataset.volume_shape), run.pixel_size)
        write_map(f"{stem}_half{h + 1}_body{body_index + 1:03d}_unfil.mrc", mstep.unfiltered_maps[h], run.pixel_size)


def write_fixed_body_maps(output, iteration: int, body_index: int, maps, run) -> None:
    """A fixed body's half maps of a numbered iteration: unchanged, written as RELION writes every body's maps each
    iteration (only a reconstructed body also has ``_unfil`` maps)."""

    stem = _stem(Path(output), iteration)
    Path(output).mkdir(parents=True, exist_ok=True)
    for h in range(2):
        write_map(f"{stem}_half{h + 1}_body{body_index + 1:03d}.mrc", _real(maps[h], run.dataset.volume_shape), run.pixel_size)


def write_final_body_maps(output, body_index: int, final, run) -> None:
    """One body's joined map and unfiltered half maps of the last iteration."""

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_map(output / f"run_body{body_index + 1:03d}.mrc", _real(final.map, run.dataset.volume_shape), run.pixel_size)
    for h in range(2):
        write_map(output / f"run_half{h + 1}_body{body_index + 1:03d}_unfil.mrc", final.unfiltered_maps[h], run.pixel_size)


def body_tables(state, run) -> dict:
    """The ``images_body_<b>`` tables: residual offsets (Angstrom) and Euler angles of every particle."""

    norm = np.asarray(run.checkpoint.particles["rlnNormCorrection"], dtype=np.float64)
    tables = {}
    for b, track in enumerate(state.bodies):
        tables[f"images_body_{b + 1}"] = pd.DataFrame({
            "rlnOriginXAngst": track.poses.offsets_px[:, 0] * run.pixel_size,
            "rlnOriginYAngst": track.poses.offsets_px[:, 1] * run.pixel_size,
            "rlnAngleRot": track.poses.eulers_deg[:, 0],
            "rlnAngleTilt": track.poses.eulers_deg[:, 1],
            "rlnAnglePsi": track.poses.eulers_deg[:, 2],
            "rlnNormCorrection": norm,
        })
    return tables


def write_data(path, state, run) -> None:
    particles = run.checkpoint.particles.copy()
    particles["rlnMaxValueProbDistribution"] = state.particle_pmax
    starfile.write(
        {"optics": run.checkpoint.optics, "particles": particles, **body_tables(state, run)},
        path, overwrite=True, float_format="%.6f",
    )


def relion_estimated_resolution(data_vs_prior, box_size: int, pixel_size: float) -> float:
    """A body's ``rlnEstimatedResolution`` in Angstrom: the last shell, from DC, before its data_vs_prior first
    drops below 1 (``MlModel::calculateTotalFourierCoverage``, ml_model.cpp:1645-1657); infinite at shell 0."""

    maxres = 0
    for ires, value in enumerate(np.asarray(data_vs_prior, dtype=np.float64).reshape(-1)):
        if value < 1.0:
            break
        maxres = ires
    return float("inf") if maxres == 0 else float(box_size) * float(pixel_size) / maxres


def relion_total_fourier_coverage(data_vs_prior, coverage) -> float:
    """A body's ``rlnOverallFourierCompleteness``: its per-shell coverage averaged over the voxels of a cube of side
    ``2 maxres - 1`` within radius ``maxres`` (``int`` of the distance), with ``maxres`` the estimated-resolution
    shell (``MlModel::calculateTotalFourierCoverage``, ml_model.cpp:1644-1672). 1.0 before any coverage exists."""

    if coverage is None:
        return 1.0
    coverage = np.asarray(coverage, dtype=np.float64).reshape(-1)
    maxres = 0
    for ires, value in enumerate(np.asarray(data_vs_prior, dtype=np.float64).reshape(-1)):
        if value < 1.0:
            break
        maxres = ires
    window = 2 * maxres - 1
    if window <= 0:
        return 0.0
    axis = np.arange(-(window // 2), window - window // 2)
    r = np.sqrt((axis[:, None, None] ** 2 + axis[None, :, None] ** 2 + axis[None, None, :] ** 2).astype(np.float64))
    r = r.astype(np.int64)[r.astype(np.int64) <= maxres]
    return float(np.pad(coverage, (0, max(0, maxres + 1 - coverage.size)))[r].sum() / r.size)


def write_iteration(output, iteration: int, state, run, *, final: bool = False) -> None:
    """The data and model STAR files of an iteration (``final``: the last one, without an iteration number)."""

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stem = _stem(output, None if final else iteration)
    write_data(f"{stem}_data.star", state, run)
    if final:
        write_relion_model(output, state, run)
    else:
        _write_model_summary(stem, state, run)
    _write_optimiser(stem, iteration, state, run)


def _write_model_summary(stem, state, run) -> None:
    frame = float(run.box_size) ** 4
    n_shells = run.box_size // 2 + 1
    shells = np.arange(n_shells)
    general = pd.DataFrame({
        "rlnCurrentResolution": [1.0 / state.current_resolution_inv],
        "rlnCurrentImageSize": [state.current_size],
        "rlnNrBodies": [len(state.bodies)],
        "rlnAveragePmax": [state.ave_pmax],
        "rlnPixelSize": [run.pixel_size],
        "rlnOriginalImageSize": [run.box_size],
    })
    blocks = {"model_general": general, "model_bodies": pd.DataFrame({
        "rlnAccuracyRotations": [t.acc_rot for t in state.bodies],
        "rlnAccuracyTranslationsAngst": [t.acc_trans for t in state.bodies],
        "rlnBodyKeepFixed": [int(t.keep_fixed) for t in state.bodies],
        "rlnEstimatedResolution": [
            relion_estimated_resolution(t.data_vs_prior, run.box_size, run.pixel_size) for t in state.bodies
        ],
        "rlnOverallFourierCompleteness": [
            relion_total_fourier_coverage(t.data_vs_prior, t.fourier_coverage) for t in state.bodies
        ],
    })}
    for b, track in enumerate(state.bodies):
        table = {"rlnSpectralIndex": shells, "rlnResolution": shells / (run.box_size * run.pixel_size)}
        coverage = np.ones(n_shells) if track.fourier_coverage is None else track.fourier_coverage
        for label, values in (
            ("rlnSsnrMap", track.data_vs_prior), ("rlnGoldStandardFsc", track.fsc), ("rlnFourierCompleteness", coverage),
        ):
            values = np.asarray(values, dtype=np.float64).reshape(-1)
            table[label] = np.pad(values, (0, max(0, n_shells - values.size)))[:n_shells]
        for h in range(2):
            tau2 = np.asarray(track.tau2[h], dtype=np.float64).reshape(-1) / frame
            table[f"rlnReferenceTau2Half{h + 1}"] = np.pad(tau2, (0, max(0, n_shells - tau2.size)))[:n_shells]
        blocks[f"model_body_{b + 1}"] = pd.DataFrame(table)
    for h in range(2):
        noise = np.asarray(state.noise.radial_per_half[h], dtype=np.float64).reshape(-1) / frame
        blocks[f"model_noise_half{h + 1}"] = pd.DataFrame({
            "rlnSpectralIndex": np.arange(noise.size), "rlnSigma2Noise": noise,
        })
    starfile.write(blocks, f"{stem}_model.star", overwrite=True, float_format="%.8g")


def _write_optimiser(stem, iteration: int, state, run) -> None:
    sampling = state.sampling
    starfile.write({"optimiser_general": pd.DataFrame({
        "rlnCurrentIteration": [iteration],
        "rlnHealpixOrder": [sampling.healpix_order],
        "rlnOffsetRange": [sampling.translation_range * run.pixel_size],
        "rlnOffsetStep": [sampling.translation_step * run.pixel_size],
        "rlnSamplingPerturbInstance": [state.perturbation],
        "rlnOverallAccuracyRotations": [sampling.acc_rot],
        "rlnOverallAccuracyTranslationsAngst": [sampling.acc_trans],
        "rlnNumberOfIterWithoutResolutionGain": [sampling.nr_iter_wo_resol_gain],
        "rlnNumberOfIterWithoutChangingAssignments": [sampling.nr_iter_wo_large_hidden_variable_changes],
        "rlnChangesOptimalOrientations": [sampling.current_changes_optimal_orientations],
        "rlnChangesOptimalOffsets": [sampling.current_changes_optimal_offsets_angstrom],
        "rlnHasFineEnoughAngularSampling": [int(sampling.has_fine_enough_angular_sampling)],
        "rlnIncrementImageSize": [state.incr_size],
        "rlnHasHighFscAtResolLimit": [int(state.has_high_fsc_at_limit)],
        "rlnHasConverged": [int(state.has_converged)],
        # What a continuation needs beyond the above (relax.multibody.resume), as RELION's optimiser and model hold.
        "rlnBestResolutionThusFar": [state.best_resol_thus_far],
        "rlnSmallestChangesOrientations": [sampling.smallest_changes_optimal_orientations],
        "rlnSmallestChangesOffsets": [sampling.smallest_changes_optimal_offsets_angstrom],
        "rlnSmallestChangesClasses": [sampling.smallest_changes_optimal_classes],
        "rlnSigmaPriorRotAngle": [float(np.rad2deg(sampling.sigma_rot))],
        "rlnSigmaPriorPsiAngle": [float(np.rad2deg(sampling.sigma_psi))],
        "rlnPreviousResolution": [sampling.previous_resolution],
        # The leader's rand() draws so far (the FSCs' phase randomisation): an exact resume continues the stream.
        "rlnRelaxPhaseDraws": [int(getattr(run.phase_stream, "draws", -1))],
    })}, f"{stem}_optimiser.star", overwrite=True, float_format="%.8g")


def write_relion_model(output, state, run) -> None:
    """The last iteration's ``run_model.star`` in RELION's format: the consensus half-1 model's general values,
    groups and orientation distribution, with each body's map, accuracies, rotate direction, freezing and shell
    curves, and the half-1 noise. Curves relax does not keep (completeness, orientability) are written as RELION
    leaves them unset."""

    output = Path(output)
    template = starfile.read(run.checkpoint.model_paths[0], always_dict=True)
    frame = float(run.box_size) ** 4
    general = dict(template["model_general"])
    general.update({
        "rlnCurrentResolution": 1.0 / state.current_resolution_inv, "rlnCurrentImageSize": state.current_size,
        "rlnNrBodies": len(state.bodies), "rlnAveragePmax": state.ave_pmax,
        **{f"rlnSigmaPrior{a}Angle": float(np.rad2deg(state.sampling.sigma_rot if a != "Psi" else state.sampling.sigma_psi))
           for a in ("Rot", "Tilt", "Psi")},
    })
    blocks = {"model_general": general}
    n_shells = run.box_size // 2 + 1
    shells = np.arange(n_shells)
    resolution = shells / (run.box_size * run.pixel_size)
    blocks["model_bodies"] = pd.DataFrame({
        "rlnReferenceImage": [str(output / f"run_body{b + 1:03d}.mrc") for b in range(len(state.bodies))],
        "rlnClassDistribution": [1.0] * len(state.bodies),
        "rlnAccuracyRotations": [t.acc_rot for t in state.bodies],
        "rlnAccuracyTranslationsAngst": [t.acc_trans for t in state.bodies],
        "rlnEstimatedResolution": [
            relion_estimated_resolution(t.data_vs_prior, run.box_size, run.pixel_size) for t in state.bodies
        ],
        "rlnOverallFourierCompleteness": [
            relion_total_fourier_coverage(t.data_vs_prior, t.fourier_coverage) for t in state.bodies
        ],
        "rlnBodyRotateDirectionX": [float(body.orientation[2, 0]) for body in run.bodies],
        "rlnBodyRotateDirectionY": [float(body.orientation[2, 1]) for body in run.bodies],
        "rlnBodyRotateDirectionZ": [float(body.orientation[2, 2]) for body in run.bodies],
        "rlnBodyKeepFixed": [int(t.keep_fixed) for t in state.bodies],
    })

    def padded(values):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        return np.pad(values, (0, max(0, n_shells - values.size)))[:n_shells]

    for b, track in enumerate(state.bodies):
        tau2 = padded(track.tau2[0]) / frame
        blocks[f"model_body_{b + 1}"] = pd.DataFrame({
            "rlnSpectralIndex": shells, "rlnResolution": resolution,
            "rlnAngstromResolution": np.where(shells > 0, 1.0 / np.maximum(resolution, 1e-12), 999.0),
            "rlnSsnrMap": padded(track.data_vs_prior), "rlnGoldStandardFsc": padded(track.fsc),
            "rlnFourierCompleteness": np.ones(n_shells) if track.fourier_coverage is None else padded(track.fourier_coverage), "rlnReferenceSigma2": np.zeros(n_shells),
            "rlnReferenceTau2": tau2, "rlnSpectralOrientabilityContribution": np.zeros(n_shells),
        })
    blocks["model_groups"] = template["model_groups"]
    noise = np.asarray(state.noise.radial_per_half[0], dtype=np.float64).reshape(-1) / frame
    blocks["model_optics_group_1"] = pd.DataFrame({
        "rlnSpectralIndex": np.arange(noise.size), "rlnResolution": np.arange(noise.size) / (run.box_size * run.pixel_size),
        "rlnSigma2Noise": noise,
    })
    for b in range(len(state.bodies)):
        blocks[f"model_pdf_orient_body_{b + 1}"] = template["model_pdf_orient_class_1"]
    starfile.write(blocks, output / "run_model.star", overwrite=True, float_format="%.8g")
