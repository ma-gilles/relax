"""VDAM InitialModel on optics groups of two image shapes, with the two classes' windows apart.

The S3b fixture (600 particles; 128 px at 4.25 A and 112 px at 5.44 A) runs ``relax
initial_model`` for three iterations at a fixed Fourier radius of 50 (current size 100 on
the 128-px model grid, 10.88 A). The 112-px group's Nyquist is 10.88 A, so its pass runs at
its box and keeps the logical window, while the model-grid group runs a stable physical
window (100 -> 120). Each shape class's E-step result is merged by
relax.refinement.shape_class_scoring.merge_k_class_engine_results, which sums the classes'
BPref cubes; the multishape seed runs of af9e5870 failed there after iteration 110 when
the model-grid class returned its physical cube (job 15071919). This reaches that state in
the first iteration. An execution check; quality is a benchmark run against RELION.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
import starfile
from conftest import gpu_subprocess_env
from helpers.em_fixtures import fixture_root, require_fixture_sets

FIXTURE_DIR = fixture_root("multioptics_s3b_600_data")
N_ITER = 3


@pytest.mark.gpu
@pytest.mark.integration
def test_vdam_two_image_shapes_merges_classes_on_different_windows(tmp_path):
    require_fixture_sets("multioptics_s3b_600_data")
    out = tmp_path / "vdam_two_shapes"
    out.mkdir()
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
        str(N_ITER),
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
        "--fourier-radius-schedule",
        f"50x{N_ITER}",
        "--gpu",
        "0",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=gpu_subprocess_env())
    assert proc.returncode == 0, f"relax initial_model exited {proc.returncode}\nstderr:\n{proc.stderr[-6000:]}"
    log = proc.stdout + proc.stderr
    # Both window regimes must have run, or the test no longer covers the merge it guards.
    assert "physical class 120" in log, "the model-grid class did not run a stable physical window"
    assert "not applicable (current size at the box)" in log, "the 112-px class did not run at its box"

    n_images = len(starfile.read(FIXTURE_DIR / "particles.star")["particles"])
    data = starfile.read(out / f"run_it{N_ITER:03d}_data.star", always_dict=True)["particles"]
    assert len(data) == n_images
    assert np.all(np.isfinite(data[["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]].to_numpy(dtype=np.float64)))
    pmax = data["rlnMaxValueProbDistribution"].to_numpy(dtype=np.float64)
    assert np.all((pmax > 0.0) & (pmax <= 1.0))
