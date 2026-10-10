"""--scratch_dir stages compact optics-group stacks; every output STAR keeps the input's image names.

With ``--scratch_dir`` single particles are read through a scratch copy of the particle STAR whose
``_rlnImageName`` points into one compact stack per optics group (relax.io.particle_io, as RELION's
``copyParticlesToScratch``). The input STAR stays the only source of image names for outputs, so a
scratch name in a written STAR would be a leak. Refine3D runs on the K1 5k/128 fixture (RELION run
files are written for one image shape, ``--write-iteration-every 1``) and VDAM InitialModel on the S3b
fixture (600 particles, two optics groups on two image shapes), two iterations each with
``--scratch_dir``; every ``*_data.star`` they write must name exactly the input's images. An execution
check.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
import starfile
from conftest import gpu_subprocess_env
from helpers.em_fixtures import fixture_root, require_fixture_sets

FIXTURE_DIR = fixture_root("multioptics_s3b_600_data")
K1_FIXTURE_DIR = fixture_root("k1_5k128_data")


def _input_names(fixture_dir):
    particles = starfile.read(fixture_dir / "particles.star", always_dict=True)["particles"]
    return sorted(str(n) for n in particles["rlnImageName"])


def _assert_outputs_keep_names(out, log, fixture_dir):
    assert "optics-group stack(s)" in log, "the run did not stage compact optics-group stacks"
    stars = sorted(out.rglob("*_data.star"))
    assert stars, f"no data.star written under {out}"
    expected = _input_names(fixture_dir)
    for star in stars:
        names = sorted(str(n) for n in starfile.read(star, always_dict=True)["particles"]["rlnImageName"])
        assert names == expected, f"{star} does not name the input's images"


@pytest.mark.gpu
@pytest.mark.integration
def test_refine_with_scratch_dir_writes_the_input_image_names(tmp_path):
    require_fixture_sets("k1_5k128_data")
    out = tmp_path / "refine"
    out.mkdir()
    local = tmp_path / "local"
    local.mkdir()
    cmd = [
        sys.executable,
        "-c",
        "import logging, sys; logging.basicConfig(level=logging.INFO); "
        "from relax.refinement.full_refinement import run_from_command_line; sys.argv[0] = 'relax refine'; "
        "run_from_command_line('refine')",
        "--data_dir",
        str(K1_FIXTURE_DIR),
        "--output",
        str(out),
        "--max_iter",
        "2",
        "--seed",
        "42",
        "--no-firstiter_cc",
        "--write-iteration-every",
        "1",
        "--scratch_dir",
        str(local),
        "--keep_free_scratch",
        "0",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=gpu_subprocess_env())
    assert proc.returncode == 0, f"relax refine exited {proc.returncode}\nstderr:\n{proc.stderr[-6000:]}"
    _assert_outputs_keep_names(out, proc.stdout + proc.stderr, K1_FIXTURE_DIR)
    assert list(local.iterdir()) == [], "the scratch copy was not removed at exit"


@pytest.mark.gpu
@pytest.mark.integration
def test_vdam_with_scratch_dir_writes_the_input_image_names(tmp_path):
    require_fixture_sets("multioptics_s3b_600_data")
    out = tmp_path / "vdam"
    out.mkdir()
    local = tmp_path / "local"
    local.mkdir()
    cmd = [
        sys.executable,
        "-c",
        "import logging, sys; logging.basicConfig(level=logging.INFO); "
        "from relax.commands.initial_model import main; sys.exit(main(sys.argv[1:]))",
        "--i",
        str(FIXTURE_DIR / "particles.star"),
        "--datadir",
        str(FIXTURE_DIR),
        "--o",
        str(out / "run"),
        "--nr_iter",
        "2",
        "--grad_write_iter",
        "1",
        "--K",
        "1",
        "--sym",
        "C1",
        "--particle_diameter",
        "200",
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
        "1",
        "--gpu",
        "0",
        "--scratch_dir",
        str(local),
        "--keep_free_scratch",
        "0",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=gpu_subprocess_env())
    assert proc.returncode == 0, f"relax initial_model exited {proc.returncode}\nstderr:\n{proc.stderr[-6000:]}"
    _assert_outputs_keep_names(out, proc.stdout + proc.stderr, FIXTURE_DIR)
    assert list(local.iterdir()) == [], "the scratch copy was not removed at exit"
