"""``relax refine --gpu 0:1``: one process per GPU, one half set each (relax_cli, relax.refinement.remote_halves).

The runs go through ``python -m relax_cli``, the console script that starts the ranks.

Needs two GPUs; run as its own job (``--gres=gpu:2``), not in a tier. The data are the first few hundred particles
of the K1 5k/128 fixture. A two-GPU run must give the one-GPU run's first iteration: the same data STAR, bitwise
(each half's particles are scored with the one-GPU arithmetic), and maps that differ only by GPU run-to-run noise.
A rank that dies must end the run: the launcher stops the other rank and exits non-zero.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import starfile
from conftest import gpu_subprocess_env
from helpers.em_fixtures import fixture_root, require_fixture_sets

pytestmark = [pytest.mark.gpu, pytest.mark.integration]

N_PARTICLES = 300
REFINE_ARGS = (
    "--offset_range", "3.0", "--offset_step", "1.0", "--healpix_order", "2", "--seed", "1775735620",
    "--particle_diameter_ang", "544", "--init_resolution", "30.0",
)  # fmt: skip
# Same-code GPU repeats differ by device atomics: on this case two one-GPU runs give half maps 2.3e-7 relL2 apart and
# norm corrections 1.2e-7 relative (em_work/relax_mgpu_20261022/ab/repeat_k1_300_16008281, H100, 2026-10-22). The
# bounds are a few times that floor, far below what a wrong or missing half gives.
MAX_MAP_REL_L2 = 1e-5
MAX_NORM_CORRECTION_REL = 1e-6


def _visible_gpus(env) -> int:
    """The GPUs this test's processes may use: CUDA_VISIBLE_DEVICES when set (nvidia-smi ignores it), else every
    GPU nvidia-smi lists. A tier item runs on one GPU, so this test skips there."""

    visible = env.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return len([device for device in visible.split(",") if device.strip()])
    listed = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, env=env).stdout
    return len([line for line in listed.splitlines() if line.startswith("GPU ")])


@pytest.fixture
def two_gpu_env():
    env = gpu_subprocess_env()
    if _visible_gpus(env) < 2:
        pytest.skip("relax refine --gpu 0:1 needs two visible GPUs")
    return env


def _project(tmp_path: Path) -> Path:
    require_fixture_sets("k1_5k128_data")
    source = fixture_root("k1_5k128_data")
    project = tmp_path / "project"
    project.mkdir()
    star = starfile.read(source / "particles.star", always_dict=True)
    starfile.write(
        {"optics": star["optics"], "particles": star["particles"].iloc[:N_PARTICLES]}, project / "particles.star"
    )
    stack = str(star["particles"]["rlnImageName"].iloc[0]).split("@")[1]
    for name in (stack, "reference_init_relion.mrc"):
        os.symlink(source / name, project / name)
    return project


def _refine(env, project: Path, out: Path, *extra: str) -> subprocess.CompletedProcess:
    command = [sys.executable, "-m", "relax_cli", "refine", "--data_dir", str(project), "--output", str(out),
               "--max_iter", "1", *REFINE_ARGS, *extra]  # fmt: skip
    proc = subprocess.run(command, capture_output=True, text=True, env=env, timeout=1800)
    assert proc.returncode == 0, f"exit {proc.returncode}\n{proc.stdout[-3000:]}\n{proc.stderr[-6000:]}"
    return proc


def _map(path: Path) -> np.ndarray:
    with mrcfile.open(path, permissive=True) as handle:
        return np.asarray(handle.data, dtype=np.float64)


def test_two_gpus_give_the_one_gpu_first_iteration(tmp_path, two_gpu_env):
    project = _project(tmp_path)
    one, two = tmp_path / "one", tmp_path / "two"
    env = dict(two_gpu_env, RELAX_FINAL_ALL_DATA_AFTER_MAX_ITER="1")  # iteration 1, then the final pass
    _refine(env, project, one)
    proc = _refine(env, project, two, "--gpu", "0:1")
    log = proc.stdout + proc.stderr
    assert "Multi-GPU run: 2 ranks connected" in log
    assert len(re.findall(r"Half 2 scored on rank 1", log)) == 2  # iteration 1 and the final pass
    assert re.search(r"rank 1 exited with status 0; peak resident memory", log)
    assert (two / ".rank1" / "rank1.log").is_file()
    assert not list((two / ".rank1").glob("run_it*")), "rank 1 wrote run files"

    ours = starfile.read(one / "run_it001_data.star", always_dict=True)["particles"]
    theirs = starfile.read(two / "run_it001_data.star", always_dict=True)["particles"]
    # Every column bitwise but the norm correction, which two one-GPU runs of single particles also differ in at
    # 1.2e-7 relative (it divides by the half's GPU-summed image power; the same 1.2e-7 between one and two GPUs here).
    differing = [column for column in ours.columns if not ours[column].equals(theirs[column])]
    assert differing in ([], ["rlnNormCorrection"]), differing
    norm_a, norm_b = (frame["rlnNormCorrection"].to_numpy(dtype=np.float64) for frame in (ours, theirs))
    assert np.max(np.abs(norm_a - norm_b) / np.abs(norm_a)) < MAX_NORM_CORRECTION_REL
    for name in ("run_it001_half1_class001.mrc", "run_it001_half2_class001.mrc"):
        a, b = _map(one / name), _map(two / name)
        assert np.linalg.norm(a - b) / np.linalg.norm(a) < MAX_MAP_REL_L2, name
    # The final pass starts from maps that already differ by that noise, so its outputs are checked for presence
    # only (one-GPU repeats of this case: final map 1.6e-5 relL2).
    for out in (one, two):
        assert np.all(np.isfinite(_map(out / "final_merged.mrc")))


def test_a_killed_rank_ends_the_run(tmp_path, two_gpu_env):
    project = _project(tmp_path)
    out = tmp_path / "killed"
    command = [sys.executable, "-m", "relax_cli", "refine", "--data_dir", str(project), "--output", str(out),
               "--max_iter", "3", *REFINE_ARGS, "--gpu", "0:1"]  # fmt: skip
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=two_gpu_env)
    pids = {}
    for line in proc.stdout:
        found = re.search(r"relax: rank (\d) on CUDA device \S+ \(pid (\d+)\)", line)
        if found:
            pids[int(found.group(1))] = int(found.group(2))
        if "ranks connected" in line:
            break
    assert set(pids) == {0, 1}
    time.sleep(5)
    os.kill(pids[1], signal.SIGKILL)
    start = time.monotonic()
    rest = proc.communicate(timeout=120)[0]
    assert proc.returncode != 0, rest[-3000:]
    assert time.monotonic() - start < 60
    assert "rank 1 exited with status 137" in rest
    for pid in pids.values():
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
