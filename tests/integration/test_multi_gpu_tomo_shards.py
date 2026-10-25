"""``relax refine --gpu 0:1:2[:3]`` on subtomograms: every rank scores shards of both halves (relax_cli,
relax.refinement.remote_halves.ShardPool, relax.steps.tomo_shards).

Needs three or four GPUs; run as its own job (``--gres=gpu:4``), not in a tier. The data are a tiny simulated
RELION 5 subtomogram project. A shard changes which particles one call scores, not how any particle is scored,
so the run must give the one-GPU run's first iteration: the same data STAR, bitwise (but the norm correction,
which divides by the half's GPU-summed image power), and maps that differ only by the order of the half's sums.
The two iterations and the final all-data pass all go through the shards.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import starfile
from conftest import gpu_subprocess_env

pytestmark = [pytest.mark.gpu, pytest.mark.integration]

GRID, VOXEL = 32, 4.25
N_PARTICLES = 120
# As tests/integration/test_two_gpu_refine.py: a few times the one-GPU repeat floor of maps and norm corrections.
MAX_MAP_REL_L2 = 1e-5
MAX_NORM_CORRECTION_REL = 1e-6
# The --firstiter_cc iteration's rlnLogLikeliContribution is a float32 sum, and two one-GPU runs already differ in the
# last bit of some rows (1,291 of 18,466 and 3,762 of 10,633 rows between same-code one-GPU repeats, sig89b and memw,
# 2026-10-25). Here the 3-GPU run differed in 7 of 120 rows by at most 1.25e-7 relative (one float32 ulp at
# |LL| ~2.7e5; job 16207525). The bound is two float32 ulps.
MAX_CC_LOG_LIKELIHOOD_REL = 2.0**-22


def _visible_gpus(env) -> int:
    visible = env.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return len([device for device in visible.split(",") if device.strip()])
    listed = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, env=env).stdout
    return len([line for line in listed.splitlines() if line.startswith("GPU ")])


@pytest.fixture
def shard_env():
    env = gpu_subprocess_env()
    gpus = _visible_gpus(env)
    if gpus < 3:
        pytest.skip("subtomogram shards need three or four visible GPUs")
    return env, min(gpus, 4)


def _volume():
    x = (np.arange(GRID) - GRID / 2) * VOXEL
    zz, yy, xx = np.meshgrid(x, x, x, indexing="ij")
    return (
        np.exp(-((xx - 10) ** 2 + yy**2 + zz**2) / 60)
        + 0.7 * np.exp(-(xx**2 + (yy + 12) ** 2 + zz**2) / 40)
        + 0.5 * np.exp(-(xx**2 + yy**2 + (zz - 14) ** 2) / 30)
    )


def _project(tmp_path: Path):
    from recovar import utils
    from recovar.simulation import relion_tomo

    with mrcfile.new(tmp_path / "vol0000.mrc", overwrite=True) as mrc:
        mrc.set_data(_volume().astype(np.float32))
        mrc.voxel_size = VOXEL
    project = tmp_path / "project"
    relion_tomo.generate_relion5_tomo_dataset(
        str(project), str(tmp_path / "vol"), VOXEL, n_particles=N_PARTICLES, grid_size=GRID, n_tomograms=2,
        max_tilt=30.0, tilt_step=10.0, tomogram_size=(512, 512, 128), snr=0.2, seed=5,
        atomic_solvent_correction=False,
    )  # fmt: skip
    reference = tmp_path / "reference_init_relion.mrc"
    utils.write_relion_mrc(str(reference), -np.transpose(_volume(), (2, 1, 0)).astype(np.float32), voxel_size=VOXEL)
    return project, reference


def _refine(env, project: Path, reference: Path, out: Path, *extra: str) -> str:
    command = [
        sys.executable, "-m", "relax_cli", "refine",
        "--data_dir", str(project), "--output", str(out), "--init_volume", str(reference),
        "--init_resolution", "20", "--particle_diameter_ang", "120", "--sym", "C1",
        "--healpix_order", "2", "--auto_local_healpix_order", "4", "--offset_range", "3", "--offset_step", "1",
        "--seed", "1", "--max_iter", "2", *extra,
    ]  # fmt: skip
    proc = subprocess.run(
        command, capture_output=True, text=True, env=dict(env, RELAX_FINAL_ALL_DATA_AFTER_MAX_ITER="1"), timeout=1800
    )
    assert proc.returncode == 0, f"exit {proc.returncode}\n{proc.stdout[-3000:]}\n{proc.stderr[-6000:]}"
    return proc.stdout + proc.stderr


def _map(path: Path) -> np.ndarray:
    with mrcfile.open(path, permissive=True) as handle:
        return np.asarray(handle.data, dtype=np.float64)


@pytest.mark.parametrize("firstiter_cc", [False, True], ids=["ml", "firstiter-cc"])
def test_shards_on_every_gpu_give_the_one_gpu_first_iteration(tmp_path, shard_env, firstiter_cc):
    """With --firstiter_cc the first iteration has no coarse pass: each half is scored whole on one rank
    (TomoHalfJob.cut), the second iteration and the final pass in shards."""

    env, gpus = shard_env
    project, reference = _project(tmp_path)
    one, many = tmp_path / "one", tmp_path / "many"
    cc = "--firstiter_cc" if firstiter_cc else "--no-firstiter_cc"
    _refine(env, project, reference, one, cc)
    log = _refine(env, project, reference, many, cc, "--gpu", ":".join(str(gpu) for gpu in range(gpus)))
    assert f"Multi-GPU run: {gpus} ranks connected" in log
    # Iterations 1 and 2 and the final pass.
    counts = re.findall(rf"Shards of halves 1,2: (\S+) shards \(\S+ alone\) on {gpus} ranks", log)
    assert len(counts) == 3, log[-4000:]
    if firstiter_cc:
        assert counts[0] == "1+1", counts  # the CC iteration: each half whole
    for rank in range(1, gpus):
        assert re.search(rf"rank {rank} exited with status 0; peak resident memory", log)
        assert not list((many / f".rank{rank}").glob("run_it*")), f"rank {rank} wrote run files"

    ours = starfile.read(one / "run_it001_data.star", always_dict=True)["particles"]
    theirs = starfile.read(many / "run_it001_data.star", always_dict=True)["particles"]
    differing = [column for column in ours.columns if not ours[column].equals(theirs[column])]
    # Every column bitwise but these, each within its one-GPU repeat floor.
    bounded = {"rlnNormCorrection": MAX_NORM_CORRECTION_REL}
    if firstiter_cc:
        bounded["rlnLogLikeliContribution"] = MAX_CC_LOG_LIKELIHOOD_REL
    assert set(differing) <= set(bounded), differing
    for column, bound in bounded.items():
        a, b = (frame[column].to_numpy(dtype=np.float64) for frame in (ours, theirs))
        assert np.max(np.abs(a - b) / np.abs(a)) < bound, column
    for name in ("run_it001_half1_class001.mrc", "run_it001_half2_class001.mrc"):
        a, b = _map(one / name), _map(many / name)
        assert np.linalg.norm(a - b) / np.linalg.norm(a) < MAX_MAP_REL_L2, name
    for out in (one, many):
        assert np.all(np.isfinite(_map(out / "final_merged.mrc")))
