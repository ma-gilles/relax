"""Subtomograms on Refine3D's global, non-adaptive route refuse with the scorer's message (GPU).

``--adaptive_oversampling 0`` with ``--auto_local_healpix_order`` above the run's order keeps the numbered
iterations global and single-pass, which the subtomogram scorer does not support ("subtomogram particles run
RELION's adaptive two-pass E-step only"). Before d35cf21e the controller read the adaptive pass-1 size on that
route, which only the adaptive route binds, so the run died with UnboundLocalError (iteration 0) or would have
scored a later iteration with an earlier iteration's size. It must reach the refusal.

No RELION subtomogram run is among the parity fixtures; the supported subtomogram routes are checked against
the simulation in test_tomo_aberrations_refine.py.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import mrcfile
import numpy as np
import pytest
from conftest import gpu_subprocess_env

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.gpu]

GRID, VOXEL = 32, 4.25


def _volume():
    x = (np.arange(GRID) - GRID / 2) * VOXEL
    zz, yy, xx = np.meshgrid(x, x, x, indexing="ij")
    return (
        np.exp(-((xx - 10) ** 2 + yy**2 + zz**2) / 60)
        + 0.7 * np.exp(-(xx**2 + (yy + 12) ** 2 + zz**2) / 40)
        + 0.5 * np.exp(-(xx**2 + yy**2 + (zz - 14) ** 2) / 30)
    )


def test_subtomograms_on_the_global_non_adaptive_route_refuse(tmp_path: Path):
    import jax.numpy as jnp
    from recovar import utils
    from recovar.core import fourier_transform_utils as ftu
    from recovar.simulation import relion_tomo, simulator

    with mrcfile.new(tmp_path / "vol0000.mrc", overwrite=True) as mrc:
        mrc.set_data(_volume().astype(np.float32))
        mrc.voxel_size = VOXEL
    project = tmp_path / "project"
    relion_tomo.generate_relion5_tomo_dataset(
        str(project),
        str(tmp_path / "vol"),
        VOXEL,
        n_particles=60,
        grid_size=GRID,
        n_tomograms=2,
        optics_groups=[{"voltage": 300.0, "cs": 2.7, "amp_contrast": 0.1, "noise_scale": 1.0}],
        max_tilt=30.0,
        tilt_step=10.0,
        tomogram_size=(512, 512, 128),
        snr=0.2,
        seed=5,
        atomic_solvent_correction=False,
    )
    loaded = simulator.load_volumes_from_folder(str(tmp_path / "vol"), GRID, True, normalize=False)[0]
    reference = tmp_path / "reference_relion.mrc"
    utils.write_relion_mrc(
        str(reference), np.real(np.asarray(ftu.get_idft3(jnp.asarray(loaded).reshape((GRID,) * 3)))), voxel_size=VOXEL
    )
    out = tmp_path / "run"
    command = [
        sys.executable, "-m", "relax.commands.refine",
        "--data_dir", str(project), "--output", str(out), "--init_volume", str(reference),
        "--init_resolution", "20", "--particle_diameter_ang", "120", "--sym", "C1",
        "--healpix_order", "2", "--auto_local_healpix_order", "6", "--adaptive_oversampling", "0",
        "--offset_range", "3", "--offset_step", "1", "--seed", "1", "--max_iter", "3", "--no-firstiter_cc",
    ]  # fmt: skip
    log = tmp_path / "refine.log"
    env = dict(gpu_subprocess_env(), RELAX_FINAL_ALL_DATA_AFTER_MAX_ITER="1")
    with open(log, "w") as handle:
        result = subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
    text = log.read_text()
    assert result.returncode != 0, "the unsupported route ran"
    assert "UnboundLocalError" not in text, text[-4000:]
    assert "subtomogram particles run RELION's adaptive two-pass E-step only" in text, text[-4000:]
