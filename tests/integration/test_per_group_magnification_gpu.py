"""Optics groups with different magnification matrices each score as a run of that group alone (relax#48).

RELION applies each optics group's ``rlnMagMat`` to its own images' projection and backprojection
matrices (``applyAnisoMag``). relax splits such groups into shape classes, one matrix each. Here 200
particles of the K=1 5k/128 fixture are given two anisotropic matrices by optics group; the
``--firstiter_cc`` iteration (one-hot CC winner from the initial reference, no noise or norm
dependence) must pick for every particle the pose the same command picks when that particle's group
is the only one in the STAR.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import starfile
from conftest import gpu_subprocess_env
from helpers.coarse_engine_selection import run_selected_command
from helpers.em_fixtures import fixture_root, require_fixture_sets

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

N_PARTICLES = 200
MAGS = {1: (1.012, 0.006, -0.004, 0.991), 2: (0.987, -0.010, -0.009, 1.006)}
MAG_COLUMNS = ("rlnMagMat00", "rlnMagMat01", "rlnMagMat10", "rlnMagMat11")


def _project(tmp_path: Path, name: str, groups: tuple[int, ...]) -> Path:
    """The fixture's first particles, alternating optics groups 1 and 2, keeping only ``groups``."""

    source = fixture_root("k1_5k128_data")
    project = tmp_path / name
    project.mkdir()
    star = starfile.read(source / "particles.star", always_dict=True)
    particles = star["particles"].iloc[:N_PARTICLES].copy()
    particles["rlnOpticsGroup"] = 1 + np.arange(N_PARTICLES) % 2
    particles = particles[particles["rlnOpticsGroup"].isin(groups)].copy()
    row = star["optics"].iloc[0]
    optics = pd.DataFrame([row] * len(groups)).reset_index(drop=True)
    optics["rlnOpticsGroup"] = np.arange(1, len(groups) + 1)
    optics["rlnOpticsGroupName"] = [f"opticsGroup{g}" for g in groups]
    for column, values in zip(MAG_COLUMNS, zip(*(MAGS[g] for g in groups))):
        optics[column] = values
    particles["rlnOpticsGroup"] = particles["rlnOpticsGroup"].map({g: i + 1 for i, g in enumerate(groups)})
    starfile.write({"optics": optics, "particles": particles}, project / "particles.star")
    stack = str(particles["rlnImageName"].iloc[0]).split("@")[1]
    for entry in (stack, "reference_init.mrc", "reference_init_relion.mrc"):
        os.symlink(source / entry, project / entry)
    return project


def _iteration_one_poses(project: Path, output: Path):
    command = [
        sys.executable, "-m", "relax.commands.refine", "--data_dir", str(project), "--output", str(output),
        "--max_iter", "1", "--healpix_order", "2", "--offset_range", "3.0", "--offset_step", "1.0",
        "--seed", "1775735620", "--firstiter_cc", "--particle_diameter_ang", "544",
        "--apply-initial-lowpass", "--init_resolution", "30.0",
    ]
    proc = run_selected_command(command, capture_output=True, text=True, env=gpu_subprocess_env())
    assert proc.returncode == 0, f"stdout:\n{proc.stdout[-4000:]}\nstderr:\n{proc.stderr[-6000:]}"
    particles = starfile.read(output / "run_it001_data.star", always_dict=True)["particles"]
    columns = ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi", "rlnOriginXAngst", "rlnOriginYAngst"]
    return particles.set_index("rlnImageName")[columns].astype(np.float64)


def test_each_magnification_group_scores_as_its_own_run(tmp_path):
    require_fixture_sets("k1_5k128_data")
    both = _iteration_one_poses(_project(tmp_path, "both", (1, 2)), tmp_path / "out_both")
    alone = pd.concat(
        [_iteration_one_poses(_project(tmp_path, f"group{g}", (g,)), tmp_path / f"out_{g}") for g in (1, 2)]
    )
    assert len(both) == N_PARTICLES and set(both.index) == set(alone.index)
    np.testing.assert_allclose(both.to_numpy(), alone.loc[both.index].to_numpy(), atol=1e-4, rtol=0)
