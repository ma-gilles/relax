"""Class3D K=2 on optics groups of two image shapes, end to end, for a few iterations.

The S3b fixture (600 particles; 128 px at 4.25 A and 112 px at 5.44 A, scale 1.12) runs
the default ``relax class3d`` command: iteration 1 is the first-iteration CC route,
iterations 2-4 the adaptive Gaussian route, each half scored once per shape class and
merged (relax.refinement.optics_shapes). Every per-image and per-class input and output
of those routes is sliced and merged on the real code path, so a slicing or merge error
fails here in minutes. This is an execution check; the quality comparison against stock
RELION on the CPU is a benchmark run.
"""

from __future__ import annotations

import sys

import mrcfile
import numpy as np
import pytest
import starfile
from conftest import gpu_subprocess_env
from helpers.coarse_engine_selection import run_selected_command
from helpers.em_fixtures import fixture_root, require_fixture_sets

FIXTURE_DIR = fixture_root("multioptics_s3b_600_data")
N_ITER = 4


@pytest.mark.gpu
@pytest.mark.integration
def test_class3d_k2_two_image_shapes_runs_and_writes_every_particle(tmp_path):
    require_fixture_sets("multioptics_s3b_600_data")
    output_dir = tmp_path / "class3d_k2_two_shapes"
    cmd = [
        sys.executable,
        "-m",
        "relax.commands.class3d",
        "--data_dir",
        str(FIXTURE_DIR),
        "--output",
        str(output_dir),
        "--init_volume",
        str(FIXTURE_DIR / "reference_init_relion_greyscale.mrc"),
        "--n_classes",
        "2",
        "--max_iter",
        str(N_ITER),
        "--init_resolution",
        "30",
        "--healpix_order",
        "2",
        "--offset_range",
        "5",
        "--offset_step",
        "1",
        "--seed",
        "1",
    ]
    proc = run_selected_command(cmd, capture_output=True, text=True, env=gpu_subprocess_env())
    assert proc.returncode == 0, (
        f"relax class3d exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )

    # Runs on several image shapes write no RELION run files; the results archive holds
    # every image's state in particle-STAR order.
    n_images = len(starfile.read(FIXTURE_DIR / "particles.star")["particles"])
    last = N_ITER - 1
    with np.load(output_dir / "refinement_results.npz", allow_pickle=True) as results:
        classes = np.asarray(results[f"class_assignments_by_image_iter_{last:03d}"])
        eulers = np.asarray(results["best_rotation_eulers_final_by_image"], dtype=np.float64)
        translations = np.asarray(results["best_translations_final_by_image"], dtype=np.float64)
        pmax = np.asarray(results[f"pmax_per_image_by_image_iter_{last:03d}"], dtype=np.float64)
        significant = np.asarray(results[f"sig_counts_by_image_iter_{last:03d}"])
    assert classes.shape == (n_images,) and set(np.unique(classes)) <= {0, 1}
    assert eulers.shape == (n_images, 3) and np.all(np.isfinite(eulers))
    assert translations.shape == (n_images, 2) and np.all(np.isfinite(translations))
    assert np.all((pmax > 0.0) & (pmax <= 1.0))
    assert np.all(significant >= 1)
    for c in (1, 2):
        with mrcfile.open(output_dir / f"final_class{c:03d}.mrc", permissive=True) as handle:
            volume = np.asarray(handle.data, dtype=np.float64)
        assert volume.shape == (128, 128, 128)
        assert np.all(np.isfinite(volume)) and np.std(volume) > 0.0
