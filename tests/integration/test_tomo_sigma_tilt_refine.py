"""RELION's ``--sigma_tilt`` through subtomogram auto-refine (GPU, end to end; relax#84).

A tiny RELION 5 2D-stack project with random orientations is given ``rlnAngleTiltPrior 90`` on every particle
(particles picked on a lattice, as in RELION's subtomogram tutorial) and refined with ``--sigma_tilt 10`` and,
as a control, without it.

What the prior does is geometric, so the checks need no tolerance of their own. A particle keeps the coarse
directions within 3 x 10 degrees of its prior tilt; its assigned orientation is an oversampled child of a kept
direction, at most one coarse step (15 degrees at HEALPix order 2) from it, moved by RELION's sampling
perturbation of at most half a step. So every assigned tilt lies within 30 + 15 + 7.5 degrees of 90. The true
tilts are spread over the sphere, so the control run assigns tilts outside that band; if it did not, the test
would prove nothing and says so. The model STARs carry RELION's prior fields, and the particles' prior column
is written back unchanged.

The run with the prior also runs the final all-data iteration straight after its two iterations
(``RELAX_FINAL_ALL_DATA_AFTER_MAX_ITER``), at HEALPix order 2, below ``--auto_local_healpix_order 4``: the
state a run is in when it converges before the local searches start. RELION's final iteration keeps the
prior (convergence changes neither the prior nor the sampling), so the final pass must run under the tilt
prior and give finite half maps.

The control run does the same without the prior (relax#86): RELION then has no orientational prior in its
final iteration either, which is the global search of the numbered iterations weighted by the direction
distribution they learned. The final pass must run as that global search and give finite half maps.

That relax selects RELION's orientations and weights is tested against a transcription of RELION's code on the
CPU (tests/unit/test_tilt_prior_selection.py); against RELION itself, iteration by iteration, it was measured
on a 1000-particle fixture (relax#84).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import mrcfile
import numpy as np
import pytest
from conftest import gpu_subprocess_env
from helpers.relion_tomo_form import add_particle_column

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.gpu]

GRID, VOXEL = 32, 4.25
N_PARTICLES = 60
SIGMA_TILT, PRIOR_TILT = 10.0, 90.0
# 3 sigma, one coarse step at HEALPix order 2 and half a step of sampling perturbation.
MAX_TILT_DEVIATION = 3.0 * SIGMA_TILT + 15.0 + 7.5


def _volume():
    x = (np.arange(GRID) - GRID / 2) * VOXEL
    zz, yy, xx = np.meshgrid(x, x, x, indexing="ij")
    return (
        np.exp(-((xx - 10) ** 2 + yy**2 + zz**2) / 60)
        + 0.7 * np.exp(-(xx**2 + (yy + 12) ** 2 + zz**2) / 40)
        + 0.5 * np.exp(-(xx**2 + yy**2 + (zz - 14) ** 2) / 30)
    )


def _refine(project: Path, reference: Path, out: Path, *extra, final_pass: bool = False):
    command = [
        sys.executable, "-m", "relax.commands.refine",
        "--data_dir", str(project), "--output", str(out), "--init_volume", str(reference),
        "--init_resolution", "20", "--particle_diameter_ang", "120", "--sym", "C1",
        "--healpix_order", "2", "--auto_local_healpix_order", "4", "--offset_range", "3", "--offset_step", "1",
        "--seed", "1", "--max_iter", "2", "--no-firstiter_cc", *extra,
    ]  # fmt: skip
    log = out.parent / f"{out.name}.log"
    with open(log, "w") as handle:
        env = dict(gpu_subprocess_env(), RELAX_FINAL_ALL_DATA_AFTER_MAX_ITER="1" if final_pass else "0")
        result = subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
    assert result.returncode == 0, log.read_text()[-4000:]
    return out


def test_sigma_tilt_keeps_every_particle_near_its_prior_tilt(tmp_path: Path):
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
    particles.write_text(add_particle_column(particles.read_text(), "_rlnAngleTiltPrior", [f"{PRIOR_TILT:.6f}"] * N_PARTICLES))
    reference = tmp_path / "reference_init_relion.mrc"
    utils.write_relion_mrc(str(reference), -np.transpose(_volume(), (2, 1, 0)).astype(np.float32), voxel_size=VOXEL)

    with_prior = _refine(
        project, reference, tmp_path / "run_sigma_tilt", "--sigma_tilt", str(SIGMA_TILT), final_pass=True
    )
    control = _refine(project, reference, tmp_path / "run_control", final_pass=True)

    def assigned(run, iteration):
        return starfile.read(run / f"run_it{iteration:03d}_data.star", always_dict=True)["particles"]

    for iteration in (1, 2):
        table = assigned(with_prior, iteration)
        assert len(table) == N_PARTICLES
        deviation = np.abs(table["rlnAngleTilt"].to_numpy(dtype=float) - PRIOR_TILT)
        assert deviation.max() <= MAX_TILT_DEVIATION, f"iteration {iteration}: tilt {deviation.max():.2f} deg from the prior"
        np.testing.assert_array_equal(table["rlnAngleTiltPrior"].to_numpy(dtype=float), PRIOR_TILT)
        for half in (1, 2):
            general = starfile.read(with_prior / f"run_it{iteration:03d}_half{half}_model.star", always_dict=True)["model_general"]
            fields = {key: float(np.asarray(general[key]).ravel()[0]) for key in (
                "rlnOrientationalPriorMode", "rlnSigmaPriorRotAngle", "rlnSigmaPriorTiltAngle", "rlnSigmaPriorPsiAngle",
            )}  # fmt: skip
            assert fields == {
                "rlnOrientationalPriorMode": 1.0, "rlnSigmaPriorRotAngle": 0.0,
                "rlnSigmaPriorTiltAngle": SIGMA_TILT, "rlnSigmaPriorPsiAngle": 0.0,
            }  # fmt: skip
    # The final all-data iteration ran to its end for both halves, under the prior (the pass logs which).
    log_text = (tmp_path / "run_sigma_tilt.log").read_text()
    for half in (1, 2):
        assert f"RELION final all-data half-{half} done" in log_text
        final_map = np.asarray(mrcfile.open(with_prior / f"final_half{half}_unfil.mrc").data, dtype=np.float64)
        assert final_map.shape == (GRID,) * 3 and np.isfinite(final_map).all() and np.abs(final_map).max() > 0.0
    assert "final all-data iteration under the --sigma_tilt prior" in log_text
    # Without the prior the final all-data iteration is a global search under the learned direction distribution.
    control_final_log = (tmp_path / "run_control.log").read_text().split("=== RELION final all-data Nyquist iteration ===")[1]
    assert "final all-data iteration under the --sigma_tilt prior" not in control_final_log
    for half in (1, 2):
        start = re.search(rf"RELION final all-data half-{half} start: .*", control_final_log).group(0)
        assert f"current_size={GRID} healpix_order=2 " in start and start.endswith("local_search=False"), start
        assert f"Using learned global direction prior half-{half}: 192 directions at healpix_order=2" in control_final_log
        assert f"RELION final all-data half-{half} done" in control_final_log
        final_map = np.asarray(mrcfile.open(control / f"final_half{half}_unfil.mrc").data, dtype=np.float64)
        assert final_map.shape == (GRID,) * 3 and np.isfinite(final_map).all() and np.abs(final_map).max() > 0.0
    outside = int((np.abs(assigned(control, 1)["rlnAngleTilt"].to_numpy(dtype=float) - PRIOR_TILT) > MAX_TILT_DEVIATION).sum())
    print(f"control without the prior: {outside} of {N_PARTICLES} particles assigned outside the band")
    assert outside > 0, "the control assigns no tilt outside the band: this project cannot show the prior acting"
    general = starfile.read(control / "run_it001_half1_model.star", always_dict=True)["model_general"]
    assert float(np.asarray(general["rlnOrientationalPriorMode"]).ravel()[0]) == 0.0
