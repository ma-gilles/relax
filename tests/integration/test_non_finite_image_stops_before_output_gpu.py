"""A non-finite pixel inside the particle mask stops every workflow before it writes a map (relax#16).

The preprocess kernel's finite check used to read only the soft-mask background, so a NaN inside the mask
entered the E-step: refine raised after a whole iteration, VDAM and Class3D in iteration 1, each from a
different sum and none naming the image. Each case here runs the public command on a copy of a fixture
with one NaN at an image centre and requires the failure to name the image and to leave no map behind.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import starfile
from conftest import gpu_subprocess_env
from helpers.coarse_engine_selection import run_selected_command
from helpers.em_fixtures import fixture_root, require_fixture_sets

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

N_PARTICLES = 300
BAD_IMAGE = 7


def _spa_project(tmp_path: Path, fixture: str, extra_files: tuple[str, ...]) -> Path:
    """The fixture's first ``N_PARTICLES`` particles, with a NaN at the centre of image ``BAD_IMAGE``."""

    require_fixture_sets(fixture)
    source = fixture_root(fixture)
    project = tmp_path / "project"
    project.mkdir()
    star = starfile.read(source / "particles.star", always_dict=True)
    particles = star["particles"].iloc[:N_PARTICLES].copy()
    names = particles["rlnImageName"].str.split("@", expand=True)
    assert names[0].astype(int).tolist() == list(range(1, N_PARTICLES + 1)) and names[1].nunique() == 1
    with mrcfile.mmap(source / names[1].iloc[0], permissive=True) as stack:
        images = np.array(stack.data[:N_PARTICLES], dtype=np.float32)
        voxel = float(stack.voxel_size.x)
    centre = images.shape[-1] // 2
    images[BAD_IMAGE, centre, centre] = np.nan
    with mrcfile.new(project / names[1].iloc[0]) as stack:
        stack.set_data(images)
        stack.voxel_size = voxel
    starfile.write({"optics": star["optics"], "particles": particles}, project / "particles.star")
    for name in extra_files:
        os.symlink(source / name, project / name)
    return project


def _tomo_project(tmp_path: Path, particle_row: int, tilt: int) -> tuple[Path, str]:
    """The subtomogram fixture with a NaN at the centre of one tilt image of one particle's stack."""

    require_fixture_sets("cryoet_s1_offsets_data")
    source = fixture_root("cryoet_s1_offsets_data")
    project = tmp_path / "project"
    project.mkdir()
    particles = starfile.read(source / "particles.star", always_dict=True)["particles"]
    bad_stack = str(particles["rlnImageName"].iloc[particle_row])
    name = str(particles["rlnTomoParticleName"].iloc[particle_row])
    for entry in source.iterdir():
        if entry.name != bad_stack.split("/")[0]:
            os.symlink(entry, project / entry.name)
    for stack in particles["rlnImageName"].astype(str):
        target = project / stack
        target.parent.mkdir(parents=True, exist_ok=True)
        if stack != bad_stack:
            os.symlink(source / stack, target)
    with mrcfile.open(source / bad_stack, permissive=True) as stack:
        images = np.array(stack.data, dtype=np.float32)
        voxel = float(stack.voxel_size.x)
    centre = images.shape[-1] // 2
    images[tilt, centre, centre] = np.nan
    with mrcfile.new(project / bad_stack) as stack:
        stack.set_data(images)
        stack.voxel_size = voxel
    return project, name


def _assert_stops_before_output(command: list[str], output: Path, *expected: str) -> str:
    proc = run_selected_command(command, capture_output=True, text=True, env=gpu_subprocess_env())
    log = f"stdout:\n{proc.stdout[-4000:]}\nstderr:\n{proc.stderr[-6000:]}"
    assert proc.returncode != 0, f"the command finished on an image with a NaN pixel\n{log}"
    for text in ("non-finite pixel", *expected):
        assert text in proc.stderr, f"{text!r} is not in the failure\n{log}"
    maps = sorted(str(path.relative_to(output)) for path in output.rglob("*.mrc")) if output.exists() else []
    assert not maps, f"maps were written before the failure: {maps}\n{log}"
    return proc.stderr


def test_refine_stops_on_a_nan_inside_the_mask(tmp_path):
    project = _spa_project(tmp_path, "k1_5k128_data", ("reference_init.mrc", "reference_init_relion.mrc"))
    output = tmp_path / "out"
    _assert_stops_before_output(
        [
            sys.executable,
            "-m",
            "relax.commands.refine",
            "--data_dir",
            str(project),
            "--output",
            str(output),
            "--max_iter",
            "3",
            "--healpix_order",
            "3",
            "--offset_range",
            "3.0",
            "--offset_step",
            "1.0",
            "--adaptive_oversampling",
            "1",
            "--tau2_fudge",
            "1.0",
            "--seed",
            "1775735620",
            "--no-firstiter_cc",
            "--particle_diameter_ang",
            "544",
            "--apply-initial-lowpass",
            "--init_resolution",
            "30.0",
        ],
        output,
        f"dataset image {BAD_IMAGE} ",
    )


def test_class3d_stops_on_a_nan_inside_the_mask(tmp_path):
    project = _spa_project(
        tmp_path,
        "k2_5k128_data",
        (
            "reference_init_class001_relion.mrc",
            "reference_init_class002_relion.mrc",
            "reference_init_classes_relion.star",
        ),
    )
    output = tmp_path / "out"
    _assert_stops_before_output(
        [
            sys.executable,
            "-m",
            "relax.commands.class3d",
            "--data_dir",
            str(project),
            "--output",
            str(output),
            "--n_classes",
            "2",
            "--max_iter",
            "3",
            "--healpix_order",
            "2",
            "--offset_range",
            "5",
            "--offset_step",
            "2",
            "--init_resolution",
            "60.0",
            "--tau2_fudge",
            "4.0",
            "--ref_star",
            str(project / "reference_init_classes_relion.star"),
            "--particle_diameter_ang",
            "200",
            "--seed",
            "29",
            "--firstiter_cc",
            "--apply-initial-lowpass",
        ],
        output,
        f"dataset image {BAD_IMAGE} ",
    )


def test_vdam_stops_on_a_nan_inside_the_mask(tmp_path):
    project = _spa_project(tmp_path, "k1_5k128_data", ())
    output = tmp_path / "out"
    output.mkdir()
    _assert_stops_before_output(
        [
            sys.executable,
            "-m",
            "relax.commands.initial_model",
            "--i",
            str(project / "particles.star"),
            "--datadir",
            str(project),
            "--o",
            str(output / "run"),
            "--nr_iter",
            "20",
            "--padding_factor",
            "1",
            "--K",
            "1",
            "--sym",
            "C1",
            "--particle_diameter",
            "150",
            "--oversampling",
            "1",
            "--healpix_order",
            "1",
            "--offset_range",
            "6",
            "--offset_step",
            "2",
            "--tau2_fudge",
            "4",
            "--random_seed",
            "29",
        ],
        output,
    )


@pytest.mark.parametrize(
    "particle_row", [0, 900], ids=["in_the_startup_noise_estimate", "first_met_by_the_coarse_pass"]
)
def test_subtomogram_refine_stops_on_a_nan_inside_the_mask(tmp_path, particle_row):
    project, name = _tomo_project(tmp_path, particle_row, tilt=20)
    # Whichever check meets the image first names it: the start-up noise estimate by particle, the preprocess
    # kernel by the image's index in the per-tilt dataset.
    output = tmp_path / "out"
    stderr = _assert_stops_before_output(
        [
            sys.executable,
            "-m",
            "relax.commands.refine",
            "--data_dir",
            str(project),
            "--output",
            str(output),
            "--init_volume",
            str(project / "reference_init_relion.mrc"),
            "--init_resolution",
            "40",
            "--particle_diameter_ang",
            "400",
            "--sym",
            "C1",
            "--healpix_order",
            "2",
            "--offset_range",
            "5",
            "--offset_step",
            "1",
            "--seed",
            "20260926",
            "--no-firstiter_cc",
            "--max_iter",
            "3",
        ],
        output,
    )
    assert f"particle {name} " in stderr or "dataset image " in stderr
