"""CTF-premultiplied subtomograms through Refine3D and its final all-data iteration (GPU, end to end).

A tiny RELION 5 2D-stack project is simulated twice from one seed, as ordinary and as CTF-premultiplied
tilt images (``rlnCtfDataAreCtfPremultiplied 1``, the default of RELION 5's extraction), and refined by
``relax refine`` for a few iterations and the final all-data iteration
(``RELAX_FINAL_ALL_DATA_AFTER_MAX_ITER``). The final all-data iteration reconstructs to Nyquist without a
split-half prior, so a wrong backprojected CTF weight shows there first. On 2026-10-03 a tilt slot read the
chunk's u-th image's premultiplied CTF weight for its unit u (resident_tilts._SLOT_VIEW_FIELDS): the final
half maps of a premultiplied project were uncorrelated (half-half FSC -0.26 at low resolution) and 1e6 times
too strong, while every split-half iteration looked plausible.

Checked on the unfiltered final half maps of the premultiplied project: the half-half FSC of the lowest
shells (the bug fails shells 1 and 3), and as a guard the shell power against the ordinary project's final half maps (the same particles and poses;
the images differ by the CTF and its per-image normalisation, which leaves the CTF-corrected maps within a
small factor of each other).
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
# Half-half FSC of the premultiplied project's final half maps (H100, 2026-10-03): fixed (b6b741a, job 14937368)
# 0.942 / 0.872 / 0.761 at shells 1-3; with the slot view missing the premultiplied CTF weight (the bug, job
# 14937369) 0.595 / 0.746 / 0.346. The ordinary project gives 0.99 / 0.98 / 0.97. Premultiplied / ordinary
# power 2.3-2.8 fixed, 3.2-4.8 with the bug; on the 10k-image fixture the bug reached 1e6, so the power bound
# is only a guard.
MIN_HALF_FSC = {1: 0.8, 2: 0.6, 3: 0.6}
MAX_POWER_RATIO = 30.0


def _volume():
    x = (np.arange(GRID) - GRID / 2) * VOXEL
    zz, yy, xx = np.meshgrid(x, x, x, indexing="ij")
    return (
        np.exp(-((xx - 10) ** 2 + yy**2 + zz**2) / 60)
        + 0.7 * np.exp(-(xx**2 + (yy + 12) ** 2 + zz**2) / 40)
        + 0.5 * np.exp(-(xx**2 + yy**2 + (zz - 14) ** 2) / 30)
    )


def _shells():
    k = np.fft.fftfreq(GRID) * GRID
    kz, ky, kx = np.meshgrid(k, k, k, indexing="ij")
    return np.round(np.sqrt(kx**2 + ky**2 + kz**2)).astype(int)


def _refine(tmp_path, premultiplied: bool):
    from recovar import utils
    from recovar.simulation import relion_tomo

    project = tmp_path / ("premultiplied" if premultiplied else "ordinary")
    relion_tomo.generate_relion5_tomo_dataset(
        str(project),
        str(tmp_path / "vol"),
        VOXEL,
        n_particles=60,
        grid_size=GRID,
        n_tomograms=2,
        max_tilt=30.0,
        tilt_step=10.0,
        tomogram_size=(512, 512, 128),
        snr=0.2,
        seed=5,
        premultiplied_ctf=premultiplied,
        atomic_solvent_correction=False,
    )
    utils.write_relion_mrc(
        str(project / "reference_init_relion.mrc"),
        -np.transpose(_volume(), (2, 1, 0)).astype(np.float32),
        voxel_size=VOXEL,
    )
    out = tmp_path / f"run_{project.name}"
    env = dict(gpu_subprocess_env(), RELAX_FINAL_ALL_DATA_AFTER_MAX_ITER="1")
    command = [
        sys.executable,
        "-m",
        "relax.commands.refine",
        "--data_dir",
        str(project),
        "--output",
        str(out),
        "--init_volume",
        str(project / "reference_init_relion.mrc"),
        "--init_resolution",
        "20",
        "--particle_diameter_ang",
        "120",
        "--sym",
        "C1",
        "--healpix_order",
        "2",
        "--auto_local_healpix_order",
        "2",
        "--offset_range",
        "3",
        "--offset_step",
        "1",
        "--seed",
        "1",
        "--max_iter",
        "3",
        "--no-firstiter_cc",
    ]
    log = out.parent / f"{project.name}.log"
    with open(log, "w") as handle:
        result = subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
    assert result.returncode == 0, Path(log).read_text()[-4000:]
    halves = [np.asarray(mrcfile.open(out / f"final_half{h}_unfil.mrc").data, dtype=np.float64) for h in (1, 2)]
    return [np.fft.fftn(half) for half in halves]


@pytest.mark.gpu
def test_premultiplied_subtomograms_final_half_maps(tmp_path):
    with mrcfile.new(tmp_path / "vol0000.mrc", overwrite=True) as mrc:
        mrc.set_data(_volume().astype(np.float32))
        mrc.voxel_size = VOXEL
    shells = _shells()
    ordinary = _refine(tmp_path, premultiplied=False)
    premultiplied = _refine(tmp_path, premultiplied=True)
    for s, min_fsc in MIN_HALF_FSC.items():
        mask = shells == s
        a, b = premultiplied[0][mask], premultiplied[1][mask]
        fsc = float(np.real(np.vdot(a, b)) / np.sqrt(np.vdot(a, a).real * np.vdot(b, b).real))
        power = np.mean([np.mean(np.abs(h[mask]) ** 2) for h in premultiplied])
        reference = np.mean([np.mean(np.abs(h[mask]) ** 2) for h in ordinary])
        print(f"shell {s}: premultiplied half-half FSC {fsc:.3f}, power / ordinary {power / reference:.3g}")
        assert fsc > min_fsc, f"shell {s}: half-half FSC {fsc:.3f}"
        assert 1.0 / MAX_POWER_RATIO < power / reference < MAX_POWER_RATIO, (
            f"shell {s}: power ratio {power / reference:.3g}"
        )
