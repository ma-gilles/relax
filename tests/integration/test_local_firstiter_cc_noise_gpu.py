"""A search local from iteration 1 keeps RELION's --firstiter_cc iteration-1 noise and scale (relax#51).

RELION's CC iteration updates neither sigma2_noise nor the scale corrections
(ml_optimiser.cpp:6190, 6316), and its weights are one-hot (Pmax 1). relax applies
that after scoring, in the shared iteration loop; this test catches a route change
that sends the local (--sigma_ang) path around it. The same 300 particles run one
CC iteration local from their poses and one global; both must keep the initial
noise spectra and unit scales, and the local one must report Pmax 1.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import starfile
from conftest import gpu_subprocess_env
from helpers.coarse_engine_selection import run_selected_command
from helpers.em_fixtures import fixture_root, require_fixture_sets

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

N_PARTICLES = 300


def _project(tmp_path: Path) -> Path:
    require_fixture_sets("k1_5k128_data")
    source = fixture_root("k1_5k128_data")
    project = tmp_path / "project"
    project.mkdir()
    star = starfile.read(source / "particles.star", always_dict=True)
    starfile.write({"optics": star["optics"], "particles": star["particles"].iloc[:N_PARTICLES]}, project / "particles.star")
    stack = str(star["particles"]["rlnImageName"].iloc[0]).split("@")[1]
    for name in (stack, "reference_init.mrc", "reference_init_relion.mrc"):
        os.symlink(source / name, project / name)
    return project


def _refine(project: Path, output: Path, *search: str):
    command = [
        sys.executable, "-m", "relax.commands.refine", "--data_dir", str(project), "--output", str(output),
        "--max_iter", "1", "--offset_range", "3.0", "--offset_step", "1.0", "--seed", "1775735620",
        "--firstiter_cc", "--particle_diameter_ang", "544", "--apply-initial-lowpass", "--init_resolution", "30.0",
        *search,
    ]
    proc = run_selected_command(command, capture_output=True, text=True, env=gpu_subprocess_env())
    assert proc.returncode == 0, f"stdout:\n{proc.stdout[-4000:]}\nstderr:\n{proc.stderr[-6000:]}"


def _model(output: Path, half: int):
    return starfile.read(output / f"run_it001_half{half}_model.star", always_dict=True)


def test_local_firstiter_cc_keeps_the_noise_and_scale_and_takes_the_winner(tmp_path):
    project = _project(tmp_path)
    local, global_ = tmp_path / "local", tmp_path / "global"
    _refine(project, local, "--healpix_order", "3", "--auto_local_healpix_order", "3", "--sigma_ang", "3")
    _refine(project, global_, "--healpix_order", "2")
    for half in (1, 2):
        got, ref = _model(local, half), _model(global_, half)
        noise_blocks = [name for name in ref if name.startswith("model_optics_group_")]
        assert noise_blocks
        for name in noise_blocks:
            np.testing.assert_allclose(
                np.asarray(got[name]["rlnSigma2Noise"], dtype=np.float64),
                np.asarray(ref[name]["rlnSigma2Noise"], dtype=np.float64),
                rtol=1e-6,
            )
        np.testing.assert_allclose(np.asarray(got["model_groups"]["rlnGroupScaleCorrection"], dtype=np.float64), 1.0)
        assert float(got["model_general"]["rlnAveragePmax"]) == pytest.approx(1.0)
