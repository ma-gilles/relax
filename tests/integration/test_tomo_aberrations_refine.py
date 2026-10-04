"""Beam tilt / odd Zernike and even Zernike subtomograms through Refine3D (GPU, end to end).

A tiny RELION 5 2D-stack project is simulated with one strong aberration in its optics group and refined by
``relax refine`` twice: from the project as written, and from a copy whose optics table has the aberration
column removed (what a program that ignores the feature sees). relion_refine demodulates the odd phase of
every tilt image and puts the even terms in every tilt image's CTF; relax does the same through the per-tilt
STAR (:mod:`relax.relion.tomo_input`, :mod:`relax.relion.optics_aberrations`). The aberrations are far
stronger than a microscope's (rms phase 0.6-0.7 rad at shell 6 and 1.2-1.3 rad at shell 8 of the 32-pixel box,
8.5 A Nyquist at shell 16), so that ignoring them costs correlation in the shells a 60-particle run resolves.

Checked on the final merged map against the simulated volume: the mean Fourier shell correlation over
shells 5-12 is high with the column and clearly lower without it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import mrcfile
import numpy as np
import pytest
from conftest import gpu_subprocess_env

pytestmark = [pytest.mark.integration, pytest.mark.slow]

GRID, VOXEL = 32, 4.25
SHELLS = range(5, 13)
ABERRATIONS = {
    "odd": ("odd_zernike", [0, 0, 7200, 0, 0, -5400], "_rlnOddZernike"),
    "even": ("even_zernike", [0, 300, 0, 400, 0, 0, 0, 0, 0], "_rlnEvenZernike"),
}
# Mean GT FSC over shells 5-12 (H100, 2026-10-04, job 14992440): odd 0.588 with the column and 0.331 without it
# (per shell 8 / 9 / 10: 0.80 / 0.65 / 0.41 against 0.38 / -0.02 / -0.17); even 0.607 and 0.381 (0.78 / 0.66 / 0.48
# against 0.45 / 0.16 / -0.05). With aberrations acting only near Nyquist the two differed by 0.002 and 0.015 (job
# 14986240), which is why these are strong at shells 6-10. The bounds leave about half of each gap.
MIN_FSC_WITH = {"odd": 0.48, "even": 0.50}
MIN_GAIN = {"odd": 0.12, "even": 0.11}


def _volume():
    x = (np.arange(GRID) - GRID / 2) * VOXEL
    zz, yy, xx = np.meshgrid(x, x, x, indexing="ij")
    return (
        np.exp(-((xx - 10) ** 2 + yy**2 + zz**2) / 60)
        + 0.7 * np.exp(-(xx**2 + (yy + 12) ** 2 + zz**2) / 40)
        + 0.5 * np.exp(-(xx**2 + yy**2 + (zz - 14) ** 2) / 30)
    )


def _without_optics_column(project: Path, label: str) -> Path:
    """A copy of the project whose particle STAR optics table lacks ``label``."""

    stripped = project.with_name(project.name + "_ignored")
    stripped.mkdir()
    for entry in project.iterdir():
        if entry.name != "particles.star":
            (stripped / entry.name).symlink_to(entry)
    lines, labels, in_optics = [], [], False
    for line in (project / "particles.star").read_text().splitlines():
        if line.startswith("data_"):
            in_optics, labels = line.startswith("data_optics"), []
        if in_optics and line.startswith("_rln"):
            labels.append(line.split()[0])
            if line.split()[0] == label:
                continue
        elif in_optics and labels and line.strip() and not line.startswith(("loop_", "data_", "#")):
            values = line.split()
            del values[labels.index(label)]
            line = " ".join(values)
        lines.append(line)
    (stripped / "particles.star").write_text("\n".join(lines) + "\n")
    return stripped


def _refine(project: Path, reference: Path) -> np.ndarray:
    out = project.with_name("run_" + project.name)
    command = [
        sys.executable, "-m", "relax.commands.refine",
        "--data_dir", str(project), "--output", str(out), "--init_volume", str(reference),
        "--init_resolution", "20", "--particle_diameter_ang", "120", "--sym", "C1",
        "--healpix_order", "2", "--auto_local_healpix_order", "2", "--offset_range", "3", "--offset_step", "1",
        "--seed", "1", "--max_iter", "3", "--no-firstiter_cc",
    ]  # fmt: skip
    log = project.with_name(project.name + ".log")
    env = dict(gpu_subprocess_env(), RELAX_FINAL_ALL_DATA_AFTER_MAX_ITER="1")
    with open(log, "w") as handle:
        result = subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
    assert result.returncode == 0, log.read_text()[-4000:]
    return np.fft.fftn(np.asarray(mrcfile.open(out / "final_merged.mrc").data, dtype=np.float64))


def _mean_fsc(map_ft, reference_ft) -> float:
    k = np.fft.fftfreq(GRID) * GRID
    kz, ky, kx = np.meshgrid(k, k, k, indexing="ij")
    shells = np.round(np.sqrt(kx**2 + ky**2 + kz**2)).astype(int)
    values = []
    for s in SHELLS:
        a, b = map_ft[shells == s], reference_ft[shells == s]
        values.append(float(np.real(np.vdot(a, b)) / np.sqrt(np.vdot(a, a).real * np.vdot(b, b).real)))
    print("per-shell GT FSC:", " ".join(f"{s}:{v:.2f}" for s, v in zip(SHELLS, values)))
    return float(np.mean(values))


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["odd", "even"])
def test_aberrated_subtomograms_refine_with_their_optics(tmp_path, kind):
    import jax.numpy as jnp
    from recovar import utils
    from recovar.core import fourier_transform_utils as ftu
    from recovar.simulation import relion_tomo, simulator

    key, coefficients, label = ABERRATIONS[kind]
    with mrcfile.new(tmp_path / "vol0000.mrc", overwrite=True) as mrc:
        mrc.set_data(_volume().astype(np.float32))
        mrc.voxel_size = VOXEL
    project = tmp_path / kind
    relion_tomo.generate_relion5_tomo_dataset(
        str(project),
        str(tmp_path / "vol"),
        VOXEL,
        n_particles=60,
        grid_size=GRID,
        n_tomograms=2,
        optics_groups=[{"voltage": 300.0, "cs": 2.7, "amp_contrast": 0.1, "noise_scale": 1.0, key: coefficients}],
        max_tilt=30.0,
        tilt_step=10.0,
        tomogram_size=(512, 512, 128),
        snr=0.2,
        seed=5,
        atomic_solvent_correction=False,
    )
    # The reference is the volume as the simulator loads it, written in RELION's file frame: the ground truth of
    # the project's poses, and the start of both refinements.
    loaded = simulator.load_volumes_from_folder(str(tmp_path / "vol"), GRID, True, normalize=False)[0]
    reference = tmp_path / "reference_relion.mrc"
    utils.write_relion_mrc(
        str(reference), np.real(np.asarray(ftu.get_idft3(jnp.asarray(loaded).reshape((GRID,) * 3)))), voxel_size=VOXEL
    )
    reference_ft = np.fft.fftn(np.asarray(mrcfile.open(reference).data, dtype=np.float64))
    with_optics = _mean_fsc(_refine(project, reference), reference_ft)
    ignored = _mean_fsc(_refine(_without_optics_column(project, label), reference), reference_ft)
    print(f"{kind}: mean GT FSC shells {SHELLS.start}-{SHELLS.stop - 1}: {with_optics:.3f}, column removed {ignored:.3f}")
    assert with_optics > MIN_FSC_WITH[kind], f"{kind}: mean GT FSC {with_optics:.3f}"
    assert with_optics - ignored > MIN_GAIN[kind], f"{kind}: {with_optics:.3f} against {ignored:.3f} without the column"
