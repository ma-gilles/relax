"""A split-halves refinement whose reference is named ``half1`` scores half set 2 against ``half2`` (GPU; relax#93).

RELION's MPI refinement gives each half-set rank its own ``--ref`` (``FileName::getHalf``,
ml_optimiser_mpi.cpp:726-731). Here the ``half2`` map is the ``half1`` map turned 90 degrees about z, so after
one global iteration half set 1's single particles sit near their simulated orientations and half set 2's about 90
degrees away; with one shared reference both halves would sit near them. Subtomograms: against the same run with one
reference for both halves, half set 1 is unchanged and half set 2 moves.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import starfile
from conftest import gpu_subprocess_env

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.gpu]

GRID, VOXEL = 32, 4.25


def _euler_matrix(rot, tilt, psi):
    a, b, g = np.deg2rad(rot), np.deg2rad(tilt), np.deg2rad(psi)
    ca, cb, cg, sa, sb, sg = np.cos(a), np.cos(b), np.cos(g), np.sin(a), np.sin(b), np.sin(g)
    return np.stack(
        [
            np.stack([cg * cb * ca - sg * sa, cg * cb * sa + sg * ca, -cg * sb], -1),
            np.stack([-sg * cb * ca - cg * sa, -sg * cb * sa + cg * ca, sg * sb], -1),
            np.stack([sb * ca, sb * sa, cb], -1),
        ],
        -2,
    )


def _angles(table):
    return _euler_matrix(
        *(np.asarray(table[c], dtype=np.float64) for c in ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"))
    )


def _distance_deg(a, b):
    trace = np.einsum("nij,nij->n", a, b)
    return np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))


def _half_set_references(source: Path, directory: Path) -> Path:
    """``reference_relion_half1.mrc`` (a copy of ``source``) and ``reference_relion_half2.mrc`` (it turned 90 degrees
    about z: the map's (z, y, x) array rotated in its y-x plane)."""
    half1 = directory / "reference_relion_half1.mrc"
    shutil.copyfile(source, half1)
    half2 = directory / "reference_relion_half2.mrc"
    shutil.copyfile(source, half2)
    with mrcfile.open(half2, mode="r+") as handle:
        handle.set_data(np.ascontiguousarray(np.rot90(np.asarray(handle.data), k=1, axes=(1, 2))))
    return half1


def _refine(project: Path, reference: Path, out: Path, extra: list[str]) -> str:
    command = [
        sys.executable, "-m", "relax.commands.refine", "--data_dir", str(project), "--output", str(out),
        "--init_volume", str(reference), "--sym", "C1", "--max_iter", "1", "--no-firstiter_cc", "--seed", "1",
        *extra,
    ]  # fmt: skip
    log = out.with_suffix(".log")
    with open(log, "w") as handle:
        result = subprocess.run(command, env=gpu_subprocess_env(), stdout=handle, stderr=subprocess.STDOUT)
    text = log.read_text()
    assert result.returncode == 0, text[-4000:]
    return text


def _assert_half_set_2_followed_its_reference(out: Path, truth, key: str, text: str, reference: Path):
    assert f"half 1 reads {reference}, half 2 reads {str(reference).replace('half1', 'half2')}" in text
    got = starfile.read(out / "run_it001_data.star", always_dict=True)["particles"].set_index(key)
    truth = truth.set_index(key).loc[got.index]
    distance = _distance_deg(_angles(got), _angles(truth))
    half = np.asarray(got["rlnRandomSubset"], dtype=np.int64)
    assert np.median(distance[half == 1]) < 30.0, np.median(distance[half == 1])
    assert np.median(distance[half == 2]) > 60.0, np.median(distance[half == 2])


def test_single_particle_half_set_2_scores_against_the_half2_reference(tmp_path):
    from helpers.em_fixtures import fixture_root, require_fixture_sets

    require_fixture_sets("k1_5k128_data")
    source = fixture_root("k1_5k128_data")
    project = tmp_path / "project"
    project.mkdir()
    star = starfile.read(source / "particles.star", always_dict=True)
    particles = star["particles"].iloc[:300]
    starfile.write({"optics": star["optics"], "particles": particles}, project / "particles.star")
    os.symlink(source / str(particles["rlnImageName"].iloc[0]).split("@")[1], project / "particles.128.mrcs")
    reference = _half_set_references(source / "reference_init_relion.mrc", tmp_path)
    out = tmp_path / "run"
    text = _refine(
        project, reference, out,
        ["--healpix_order", "2", "--offset_range", "3", "--offset_step", "1", "--particle_diameter_ang", "544",
         "--init_resolution", "30"],
    )  # fmt: skip
    _assert_half_set_2_followed_its_reference(out, particles, "rlnImageName", text, reference)


def test_subtomogram_half_set_2_scores_against_the_half2_reference(tmp_path):
    import jax.numpy as jnp
    from recovar import utils
    from recovar.core import fourier_transform_utils as ftu
    from recovar.simulation import relion_tomo, simulator

    x = (np.arange(GRID) - GRID / 2) * VOXEL
    zz, yy, xx = np.meshgrid(x, x, x, indexing="ij")
    volume = (
        np.exp(-((xx - 10) ** 2 + yy**2 + zz**2) / 60)
        + 0.7 * np.exp(-(xx**2 + (yy + 12) ** 2 + zz**2) / 40)
        + 0.5 * np.exp(-(xx**2 + yy**2 + (zz - 14) ** 2) / 30)
    )
    with mrcfile.new(tmp_path / "vol0000.mrc", overwrite=True) as mrc:
        mrc.set_data(volume.astype(np.float32))
        mrc.voxel_size = VOXEL
    project = tmp_path / "project"
    relion_tomo.generate_relion5_tomo_dataset(
        str(project), str(tmp_path / "vol"), VOXEL, n_particles=60, grid_size=GRID, n_tomograms=2,
        optics_groups=[{"voltage": 300.0, "cs": 2.7, "amp_contrast": 0.1, "noise_scale": 1.0}],
        max_tilt=30.0, tilt_step=10.0, tomogram_size=(512, 512, 128), snr=1.0, seed=5,
        atomic_solvent_correction=False,
    )  # fmt: skip
    loaded = simulator.load_volumes_from_folder(str(tmp_path / "vol"), GRID, True, normalize=False)[0]
    simulated = tmp_path / "simulated_relion.mrc"
    utils.write_relion_mrc(
        str(simulated), np.real(np.asarray(ftu.get_idft3(jnp.asarray(loaded).reshape((GRID,) * 3)))), voxel_size=VOXEL
    )
    reference = _half_set_references(simulated, tmp_path)
    shared = tmp_path / "reference_relion_shared.mrc"
    shutil.copyfile(reference, shared)
    command = [
        "--healpix_order", "2", "--auto_local_healpix_order", "4", "--offset_range", "3", "--offset_step", "1",
        "--particle_diameter_ang", "120", "--init_resolution", "20",
    ]  # fmt: skip
    text = _refine(project, reference, tmp_path / "per_half", command)
    assert f"half 1 reads {reference}, half 2 reads {str(reference).replace('half1', 'half2')}" in text
    _refine(project, shared, tmp_path / "shared", command)
    # The simulated angles are not recovered from 60 particles in one iteration, so the control is the same run
    # with one reference for both halves: half set 1 is unchanged, half set 2 moves.
    per_half, control = (
        starfile.read(tmp_path / run / "run_it001_data.star", always_dict=True)["particles"].set_index(
            "rlnTomoParticleName"
        )
        for run in ("per_half", "shared")
    )
    control = control.loc[per_half.index]
    distance = _distance_deg(_angles(per_half), _angles(control))
    half = np.asarray(per_half["rlnRandomSubset"], dtype=np.int64)
    assert np.all(distance[half == 1] < 1e-3), distance[half == 1].max()
    assert np.median(distance[half == 2]) > 30.0, np.median(distance[half == 2])
