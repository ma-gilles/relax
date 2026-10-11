"""A subtomogram project in the STAR forms RELION's own tools write, through Refine3D (GPU, end to end; relax#79).

``relax refine`` did not recognise RELION 5's own ``relion_tomo_subtomo --stack2d`` particle STAR (its
``data_general`` is a list of label-value pairs, not a loop), and behind that the reader needed labels RELION
leaves out (``rlnRandomSubset``; ``rlnTomoXTilt`` and ``rlnCtfScalefactor`` at their defaults). Every project
relax was tested on came from RECOVAR's simulator, which writes none of those forms.

A tiny CTF-premultiplied project (RELION 5's extraction default) is simulated, rewritten in RELION's forms
(tests/helpers/relion_tomo_form.py: list ``data_general``, no ``rlnRandomSubset``, no ``rlnTomoXTilt``) and
refined by ``relax refine`` through three iterations and the final all-data iteration. The run must finish,
draw two non-empty halves itself, write every particle to its data STARs with the subtomogram flag, and give
finite, non-zero half maps.

No map is compared with a run on the simulated forms: two runs of this 60-particle refinement differ by
about 1e-4 in their final half maps on the same inputs (H100, 2026-10-11, job 15358869: 1.13e-4), so a map
bound here would only measure the GPU's summation order. That the two forms hand the engine identical
inputs is asserted exactly, on the CPU, in tests/unit/test_tomo_relion_star_forms.py (per-tilt rows, the
loaded dataset's arrays and every pixel), together with RELION's own STAR text from a real project.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import mrcfile
import numpy as np
import pytest
from conftest import gpu_subprocess_env
from helpers.relion_tomo_form import write_relion_form

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.gpu]

GRID, VOXEL = 32, 4.25
N_PARTICLES = 60


def _volume():
    x = (np.arange(GRID) - GRID / 2) * VOXEL
    zz, yy, xx = np.meshgrid(x, x, x, indexing="ij")
    return (
        np.exp(-((xx - 10) ** 2 + yy**2 + zz**2) / 60)
        + 0.7 * np.exp(-(xx**2 + (yy + 12) ** 2 + zz**2) / 40)
        + 0.5 * np.exp(-(xx**2 + yy**2 + (zz - 14) ** 2) / 30)
    )


def test_subtomograms_in_relions_star_forms_refine(tmp_path: Path):
    import starfile
    from recovar import utils
    from recovar.simulation import relion_tomo

    from relax.refinement.tomo_half import is_relion5_2d_stack_star

    with mrcfile.new(tmp_path / "vol0000.mrc", overwrite=True) as mrc:
        mrc.set_data(_volume().astype(np.float32))
        mrc.voxel_size = VOXEL
    simulated = tmp_path / "simulated"
    relion_tomo.generate_relion5_tomo_dataset(
        str(simulated), str(tmp_path / "vol"), VOXEL, n_particles=N_PARTICLES, grid_size=GRID, n_tomograms=2,
        max_tilt=30.0, tilt_step=10.0, tomogram_size=(512, 512, 128), snr=0.2, seed=5,
        premultiplied_ctf=True, atomic_solvent_correction=False,
    )  # fmt: skip
    reference = tmp_path / "reference_init_relion.mrc"
    utils.write_relion_mrc(str(reference), -np.transpose(_volume(), (2, 1, 0)).astype(np.float32), voxel_size=VOXEL)
    project = write_relion_form(simulated, tmp_path / "relion_form")
    assert "_rlnRandomSubset" not in (project / "particles.star").read_text()
    names = set(starfile.read(project / "particles.star", always_dict=True)["particles"]["rlnTomoParticleName"])
    assert len(names) == N_PARTICLES

    out = tmp_path / "run"
    command = [
        sys.executable, "-m", "relax.commands.refine",
        "--data_dir", str(project), "--output", str(out), "--init_volume", str(reference),
        "--init_resolution", "20", "--particle_diameter_ang", "120", "--sym", "C1",
        "--healpix_order", "2", "--auto_local_healpix_order", "2", "--offset_range", "3", "--offset_step", "1",
        "--seed", "1", "--max_iter", "3", "--no-firstiter_cc",
    ]  # fmt: skip
    log = tmp_path / "refine.log"
    env = dict(gpu_subprocess_env(), RELAX_FINAL_ALL_DATA_AFTER_MAX_ITER="1")
    with open(log, "w") as handle:
        result = subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
    assert result.returncode == 0, log.read_text()[-4000:]

    data_stars = sorted(out.glob("run_it*_data.star"))
    assert len(data_stars) >= 3, [path.name for path in data_stars]
    for path in data_stars:
        particles = starfile.read(path, always_dict=True)["particles"]
        assert set(particles["rlnTomoParticleName"]) == names and len(particles) == N_PARTICLES, path.name
        assert set(particles["rlnRandomSubset"].astype(int)) == {1, 2}, path.name
        assert is_relion5_2d_stack_star(path), path.name
    for h in (1, 2):
        half = np.asarray(mrcfile.open(out / f"final_half{h}_unfil.mrc").data, dtype=np.float64)
        assert half.shape == (GRID,) * 3 and np.isfinite(half).all() and np.abs(half).max() > 0.0, h
