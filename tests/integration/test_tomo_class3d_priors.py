"""RELION's orientational priors through subtomogram Class3D (GPU, end to end; relax#85).

RELION's Class3D selects one orientation list per particle and scores every class on it: under ``--sigma_ang``
a local search around the particle's current angles (the input STAR's in iteration 1, the previous iteration's
after), under ``--sigma_tilt`` alone every direction whose tilt is near the particle's ``rlnAngleTiltPrior``
with every in-plane angle (ml_optimiser.cpp:5970-6013, healpix_sampling.cpp:733-879). The RELION 5 subtomogram
tutorial's Class3D command is the first case.

A tiny RELION 5 2D-stack project is simulated with orientations spread over the sphere, and its particle STAR is
given the same wrong angles on every particle and ``rlnAngleTiltPrior 90``. What the priors do is geometric, so
the checks need no tolerance of their own:

- ``--sigma_ang 10``: a particle keeps the coarse directions within 3 x 10 degrees of its centre direction and
  the in-plane angles within 3 x 10 degrees of its centre's; its assigned orientation is an oversampled child
  of a kept one, at most one coarse step (15 degrees at HEALPix order 2) away, moved by RELION's sampling
  perturbation of at most half a step. So in every iteration the assigned direction lies within 30 + 15 + 7.5
  degrees of the previous one, and the in-plane angle likewise. The two windows are separate: the rotation as a
  whole may move further.
- ``--sigma_tilt 10``: every assigned tilt lies within the same 52.5 degrees of 90.
- A control without either option searches globally and leaves both bands; if it did not, the test would prove
  nothing and says so.

That relax selects RELION's orientations and weights is tested against a transcription of RELION's code on the
CPU (tests/unit/test_tilt_prior_selection.py, test_sigma_ang_local_search.py); that the K-class passes cut each
particle's own orientations over the scored classes, in tests/unit/test_tomo_coarse.py and
test_compact_coarse_grid.py; against RELION itself, iteration by iteration, it was measured on the tutorial's
Class3D command (relax#85).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import mrcfile
import numpy as np
import pytest
from conftest import gpu_subprocess_env
from helpers.relion_tomo_form import add_particle_column, drop_loop_column

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.gpu]

GRID, VOXEL = 32, 4.25
N_PARTICLES, N_CLASSES, N_ITERATIONS = 60, 2, 2
SIGMA = 10.0
INPUT_ANGLES = (30.0, 90.0, 45.0)  # rot, tilt, psi written on every particle
# 3 sigma, one coarse step at HEALPix order 2 and half a step of sampling perturbation.
MAX_DEVIATION = 3.0 * SIGMA + 15.0 + 7.5
ANGLES = ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]


def _volume():
    x = (np.arange(GRID) - GRID / 2) * VOXEL
    zz, yy, xx = np.meshgrid(x, x, x, indexing="ij")
    return (
        np.exp(-((xx - 10) ** 2 + yy**2 + zz**2) / 60)
        + 0.7 * np.exp(-(xx**2 + (yy + 12) ** 2 + zz**2) / 40)
        + 0.5 * np.exp(-(xx**2 + yy**2 + (zz - 14) ** 2) / 30)
    )


def _class3d(project: Path, reference: Path, out: Path, *extra):
    command = [
        sys.executable, "-m", "relax.commands.class3d",
        "--data_dir", str(project), "--output", str(out), "--init_volume", str(reference),
        "--n_classes", str(N_CLASSES), "--init_resolution", "20", "--particle_diameter_ang", "120", "--sym", "C1",
        "--healpix_order", "2", "--offset_range", "3", "--offset_step", "1",
        "--seed", "1", "--max_iter", str(N_ITERATIONS), "--no-firstiter_cc", *extra,
    ]  # fmt: skip
    log = out.parent / f"{out.name}.log"
    with open(log, "w") as handle:
        result = subprocess.run(command, env=gpu_subprocess_env(), stdout=handle, stderr=subprocess.STDOUT)
    assert result.returncode == 0, log.read_text()[-4000:]
    return out


def _direction_change_deg(before, after):
    def unit(rot, tilt):
        rot, tilt = np.radians(rot), np.radians(tilt)
        return np.stack([np.sin(tilt) * np.cos(rot), np.sin(tilt) * np.sin(rot), np.cos(tilt)], axis=-1)

    cosine = np.sum(unit(before[:, 0], before[:, 1]) * unit(after[:, 0], after[:, 1]), axis=-1)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _psi_change_deg(before, after):
    return np.abs((after[:, 2] - before[:, 2] + 180.0) % 360.0 - 180.0)


def test_class3d_keeps_every_particle_inside_its_orientational_prior(tmp_path: Path):
    import starfile
    from recovar import utils
    from recovar.simulation import relion_tomo

    with mrcfile.new(tmp_path / "vol0000.mrc", overwrite=True) as mrc:
        mrc.set_data(_volume().astype(np.float32))
        mrc.voxel_size = VOXEL
    project = tmp_path / "project"
    relion_tomo.generate_relion5_tomo_dataset(
        str(project), str(tmp_path / "vol"), VOXEL, n_particles=N_PARTICLES, grid_size=GRID, n_tomograms=2,
        max_tilt=30.0, tilt_step=10.0, tomogram_size=(512, 512, 128), snr=0.2, seed=5,
        atomic_solvent_correction=False,
    )  # fmt: skip
    particles = project / "particles.star"
    text = particles.read_text()
    for label, value in zip(ANGLES, INPUT_ANGLES, strict=True):
        text = add_particle_column(drop_loop_column(text, "particles", f"_{label}"), f"_{label}", [f"{value:.6f}"] * N_PARTICLES)
    particles.write_text(add_particle_column(text, "_rlnAngleTiltPrior", ["90.000000"] * N_PARTICLES))
    reference = tmp_path / "reference_init_relion.mrc"
    utils.write_relion_mrc(str(reference), -np.transpose(_volume(), (2, 1, 0)).astype(np.float32), voxel_size=VOXEL)

    local = _class3d(project, reference, tmp_path / "run_sigma_ang", "--sigma_ang", str(SIGMA))
    tilted = _class3d(project, reference, tmp_path / "run_sigma_tilt", "--sigma_tilt", str(SIGMA))
    control = _class3d(project, reference, tmp_path / "run_control")

    def assigned(run, iteration):
        table = starfile.read(run / f"run_it{iteration:03d}_data.star", always_dict=True)["particles"]
        assert len(table) == N_PARTICLES and set(table["rlnClassNumber"].astype(int)) <= set(range(1, N_CLASSES + 1))
        return table.set_index("rlnTomoParticleName")[ANGLES].astype(float)

    def prior_fields(run, iteration):
        general = starfile.read(run / f"run_it{iteration:03d}_model.star", always_dict=True)["model_general"]
        return [
            float(np.asarray(general[key]).ravel()[0])
            for key in ("rlnOrientationalPriorMode", "rlnSigmaPriorRotAngle", "rlnSigmaPriorTiltAngle", "rlnSigmaPriorPsiAngle")
        ]

    start = assigned(local, 1).copy()
    start[ANGLES] = INPUT_ANGLES
    before = start
    for iteration in range(1, N_ITERATIONS + 1):
        after = assigned(local, iteration).loc[before.index]
        direction = _direction_change_deg(before.to_numpy(), after.to_numpy())
        psi = _psi_change_deg(before.to_numpy(), after.to_numpy())
        assert direction.max() <= MAX_DEVIATION, f"--sigma_ang iteration {iteration}: direction moved {direction.max():.2f} deg"
        assert psi.max() <= MAX_DEVIATION, f"--sigma_ang iteration {iteration}: in-plane angle moved {psi.max():.2f} deg"
        assert prior_fields(local, iteration) == [1.0, SIGMA, SIGMA, SIGMA]
        before = after
        tilt = np.abs(assigned(tilted, iteration)["rlnAngleTilt"].to_numpy() - 90.0)
        assert tilt.max() <= MAX_DEVIATION, f"--sigma_tilt iteration {iteration}: tilt {tilt.max():.2f} deg from the prior"
        assert prior_fields(tilted, iteration) == [1.0, 0.0, SIGMA, 0.0]
        for run in (local, tilted):
            for k in range(1, N_CLASSES + 1):
                class_map = np.asarray(mrcfile.open(run / f"run_it{iteration:03d}_class{k:03d}.mrc").data, dtype=np.float64)
                assert class_map.shape == (GRID,) * 3 and np.isfinite(class_map).all()
    # The control searches globally from the same wrong angles: it must leave both bands.
    found = assigned(control, 1).loc[start.index]
    outside_local = int(
        (
            (_direction_change_deg(start.to_numpy(), found.to_numpy()) > MAX_DEVIATION)
            | (_psi_change_deg(start.to_numpy(), found.to_numpy()) > MAX_DEVIATION)
        ).sum()
    )
    outside_tilt = int((np.abs(found["rlnAngleTilt"].to_numpy() - 90.0) > MAX_DEVIATION).sum())
    print(f"control without a prior: {outside_local} of {N_PARTICLES} particles outside the --sigma_ang band, {outside_tilt} outside the --sigma_tilt band")
    assert outside_local > 0 and outside_tilt > 0, "the control stays inside a band: this project cannot show the prior acting"
    assert prior_fields(control, 1)[0] == 0.0
