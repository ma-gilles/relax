"""--solvent_mask and --solvent_correct_fsc through relax refine (GPU, end to end, a tiny subtomogram project).

The user mask covers one blob of three. RELION flattens each new reference with it after the M-step and records it
in the optimiser STAR; with --solvent_correct_fsc the reported gold-standard FSC is the masked, corrected curve,
which on this volume differs from the unmasked one.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import starfile
from conftest import gpu_subprocess_env

pytestmark = [pytest.mark.integration, pytest.mark.slow]

GRID, VOXEL = 32, 4.25


def _volume():
    x = (np.arange(GRID) - GRID / 2) * VOXEL
    zz, yy, xx = np.meshgrid(x, x, x, indexing="ij")
    return (
        np.exp(-((xx - 10) ** 2 + yy**2 + zz**2) / 60)
        + 0.7 * np.exp(-(xx**2 + (yy + 12) ** 2 + zz**2) / 40)
        + 0.5 * np.exp(-(xx**2 + yy**2 + (zz - 14) ** 2) / 30)
    )


def _refine(project: Path, reference: Path, out: Path, extra: list[str]) -> None:
    command = [
        sys.executable, "-m", "relax.commands.refine",
        "--data_dir", str(project), "--output", str(out), "--init_volume", str(reference),
        "--init_resolution", "20", "--particle_diameter_ang", "120", "--sym", "C1",
        "--healpix_order", "2", "--auto_local_healpix_order", "2", "--offset_range", "3", "--offset_step", "1",
        "--seed", "1", "--max_iter", "2", "--no-firstiter_cc", *extra,
    ]  # fmt: skip
    log = out.with_suffix(".log")
    env = dict(gpu_subprocess_env(), RELAX_FINAL_ALL_DATA_AFTER_MAX_ITER="1")
    with open(log, "w") as handle:
        result = subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
    assert result.returncode == 0, log.read_text()[-4000:]


@pytest.mark.gpu
def test_solvent_mask_flattens_and_corrects_the_fsc(tmp_path):
    import jax.numpy as jnp
    from recovar import utils
    from recovar.core import fourier_transform_utils as ftu
    from recovar.simulation import relion_tomo, simulator

    with mrcfile.new(tmp_path / "vol0000.mrc", overwrite=True) as mrc:
        mrc.set_data(_volume().astype(np.float32))
        mrc.voxel_size = VOXEL
    project = tmp_path / "project"
    relion_tomo.generate_relion5_tomo_dataset(
        str(project), str(tmp_path / "vol"), VOXEL, n_particles=60, grid_size=GRID, n_tomograms=2,
        optics_groups=[{"voltage": 300.0, "cs": 2.7, "amp_contrast": 0.1, "noise_scale": 1.0}],
        max_tilt=30.0, tilt_step=10.0, tomogram_size=(512, 512, 128), snr=0.2, seed=5,
        atomic_solvent_correction=False,
    )  # fmt: skip
    loaded = simulator.load_volumes_from_folder(str(tmp_path / "vol"), GRID, True, normalize=False)[0]
    reference = tmp_path / "reference_relion.mrc"
    utils.write_relion_mrc(
        str(reference), np.real(np.asarray(ftu.get_idft3(jnp.asarray(loaded).reshape((GRID,) * 3)))), voxel_size=VOXEL
    )
    # A hard box mask over one corner of the map file (RELION's axis order): a region the flatten must zero.
    mask = np.zeros((GRID,) * 3, dtype=np.float32)
    mask[4:28, 4:28, 4:28] = 1.0
    mask[:, :, :12] = 0.0
    mask_path = tmp_path / "mask.mrc"
    with mrcfile.new(mask_path) as mrc:
        mrc.set_data(mask)
        mrc.voxel_size = VOXEL

    masked, plain = tmp_path / "masked", tmp_path / "plain"
    _refine(project, reference, masked, ["--solvent_mask", str(mask_path), "--solvent_correct_fsc"])
    _refine(project, reference, plain, [])

    it1 = np.asarray(mrcfile.open(masked / "run_it001_half1_class001.mrc").data, dtype=np.float64)
    assert np.abs(it1[mask == 0]).max() < 1e-6 * np.abs(it1).max()
    general = starfile.read(masked / "run_it001_optimiser.star", always_dict=True)["optimiser_general"]

    def value(key):
        return general[key].iloc[0] if hasattr(general[key], "iloc") else general[key]

    assert str(value("rlnSolventMaskName")) == str(mask_path)
    assert bool(int(value("rlnDoSolventFscCorrection")))

    def fsc(run):
        return np.asarray(starfile.read(run / "run_it002_half1_model.star")["model_class_1"]["rlnGoldStandardFsc"])

    assert np.abs(fsc(masked) - fsc(plain)).max() > 1e-3
    assert (masked / "run_it002_half1_class001_unfil.mrc").exists()
