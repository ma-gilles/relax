"""Production CLI modes no other tier item runs, each through the installed ``relax`` dispatcher.

Every run goes through the console entry point (``[project.scripts] relax =
relax.command_line:main_commands``), as users start relax, instead of ``python -m
relax.commands.<cmd>``: ``relax refine --preread_images``, ``relax refine --continue``,
``relax class3d --init_class_volumes``, ``relax initial_model --K 2`` and ``relax
ppca_initial_model --i`` (with and without ``--oversampling 1``). The data are the first few
hundred particles of the K1 and K2 5k/128 fixtures, a few iterations each. These are execution checks: each run must complete and
write sane outputs (files present, maps finite and nonzero, STAR and npz files readable,
the iteration counts asked for); quality is measured by the parity and benchmark items.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import starfile
from conftest import gpu_subprocess_env
from helpers.em_fixtures import fixture_root, require_fixture_sets

pytestmark = [pytest.mark.gpu, pytest.mark.integration]

N_PARTICLES = 300
# The K1 fixture's RELION command values (tests/integration/test_local_firstiter_cc_noise_gpu.py).
REFINE_ARGS = (
    "--offset_range", "3.0", "--offset_step", "1.0", "--healpix_order", "2", "--seed", "1775735620",
    "--particle_diameter_ang", "544", "--init_resolution", "30.0",
)


def _relax() -> str:
    """The installed console script of the interpreter running the tests."""
    script = Path(sys.executable).with_name("relax")
    assert script.is_file(), f"no relax console script beside {sys.executable} (pip install -e . / pixi install)"
    return str(script)


def _run(*args: str, timeout: float = 1200) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        [_relax(), *args], capture_output=True, text=True, env=gpu_subprocess_env(), timeout=timeout
    )
    assert proc.returncode == 0, (
        f"relax {' '.join(args)} exited {proc.returncode}\nstdout:\n{proc.stdout[-4000:]}\nstderr:\n{proc.stderr[-6000:]}"
    )
    return proc


def _project(tmp_path: Path, fixture: str, links: tuple[str, ...] = ()) -> Path:
    """A data directory holding the fixture's first N_PARTICLES particles; the stack and ``links`` are symlinked."""
    require_fixture_sets(fixture)
    source = fixture_root(fixture)
    project = tmp_path / f"{fixture}_project"
    project.mkdir()
    star = starfile.read(source / "particles.star", always_dict=True)
    starfile.write({"optics": star["optics"], "particles": star["particles"].iloc[:N_PARTICLES]}, project / "particles.star")
    stack = str(star["particles"]["rlnImageName"].iloc[0]).split("@")[1]
    for name in (stack, *links):
        os.symlink(source / name, project / name)
    return project


def _map(path: Path, box: int = 128) -> np.ndarray:
    assert path.is_file(), f"missing map {path}"
    with mrcfile.open(path, permissive=True) as handle:
        volume = np.asarray(handle.data, dtype=np.float64)
    assert volume.shape == (box, box, box), f"{path}: shape {volume.shape}"
    assert np.all(np.isfinite(volume)) and np.std(volume) > 0.0, f"{path}: non-finite or constant"
    return volume


def _particles(path: Path) -> "starfile.pd.DataFrame":
    particles = starfile.read(path, always_dict=True)["particles"]
    assert len(particles) == N_PARTICLES, f"{path}: {len(particles)} particles, expected {N_PARTICLES}"
    angles = particles[["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]].to_numpy(dtype=np.float64)
    assert np.all(np.isfinite(angles))
    return particles


def _refine_outputs(out: Path, *, last_iteration: int) -> None:
    """The run files of the last numbered iteration, the final maps and the results archive (a run stopped at
    --max_iter writes its final maps without the all-data pass, as relion_refine)."""
    _particles(out / f"run_it{last_iteration:03d}_data.star")
    assert not (out / f"run_it{last_iteration + 1:03d}_data.star").exists()
    for half in (1, 2):
        _map(out / f"run_it{last_iteration:03d}_half{half}_class001.mrc")
        _map(out / f"final_half{half}.mrc")
    _map(out / "final_merged.mrc")
    with np.load(out / "refinement_results.npz", allow_pickle=True) as results:
        assert results.files


def test_dispatcher_lists_commands_and_dispatches_help():
    """``relax`` with no or an unknown command lists the commands and exits 1; ``relax <cmd> --help`` reaches the command."""
    env = gpu_subprocess_env()
    for argv in ([], ["no_such_command"]):
        proc = subprocess.run([_relax(), *argv], capture_output=True, text=True, env=env, timeout=120)
        assert proc.returncode == 1, proc.stderr
        listed = proc.stderr.split("Available commands:", 1)[1].split()
        assert {"refine", "class3d", "initial_model", "ppca_initial_model"} <= set(listed)
    for command, option in (("refine", "--preread_images"), ("class3d", "--init_class_volumes"),
                            ("initial_model", "--nr-iter"), ("ppca_initial_model", "--particle-diameter")):
        proc = subprocess.run([_relax(), command, "--help"], capture_output=True, text=True, env=env, timeout=300)
        assert proc.returncode == 0, f"relax {command} --help exited {proc.returncode}\n{proc.stderr[-4000:]}"
        assert option in proc.stdout, f"relax {command} --help does not show {option}"


def test_refine_preread_images(tmp_path):
    """``relax refine --preread_images``: two iterations with the stack read into memory."""
    project = _project(tmp_path, "k1_5k128_data", ("reference_init_relion.mrc",))
    out = tmp_path / "refine_preread"
    _run("refine", "--data_dir", str(project), "--output", str(out), "--max_iter", "2", *REFINE_ARGS, "--preread_images")
    _refine_outputs(out, last_iteration=2)


def test_refine_continue(tmp_path):
    """``relax refine --continue``: run iteration 1, then continue from its optimiser to iteration 2."""
    project = _project(tmp_path, "k1_5k128_data", ("reference_init_relion.mrc",))
    out = tmp_path / "refine_continue"
    common = ("--data_dir", str(project), "--output", str(out), *REFINE_ARGS)
    _run("refine", *common, "--max_iter", "1")
    optimiser = out / "run_it001_optimiser.star"
    assert optimiser.is_file()
    first = {p.name: p.stat().st_mtime_ns for p in out.glob("run_it001_*")}
    assert not (out / "run_it002_data.star").exists()
    _run("refine", *common, "--max_iter", "2", "--continue", str(optimiser))
    # The continued run starts at iteration 2: iteration 1's files are not rewritten, the final maps are.
    assert {p.name: p.stat().st_mtime_ns for p in out.glob("run_it001_*")} == first
    assert (out / "final_merged.mrc").stat().st_mtime_ns > max(first.values())
    _refine_outputs(out, last_iteration=2)


def test_class3d_init_class_volumes(tmp_path):
    """``relax class3d --init_class_volumes`` with two per-class maps the data directory does not hold."""
    project = _project(tmp_path, "k2_5k128_data")
    source = fixture_root("k2_5k128_data")
    volumes = []
    for k in (1, 2):
        # Not the default <data_dir>/reference_init_class00K_relion.mrc names: only the option can supply them.
        target = tmp_path / f"start_k{k}_relion.mrc"
        shutil.copyfile(source / f"reference_init_class00{k}_relion.mrc", target)
        volumes.append(str(target))
    out = tmp_path / "class3d_init_volumes"
    _run("class3d", "--data_dir", str(project), "--output", str(out), "--n_classes", "2",
         "--init_class_volumes", ",".join(volumes), "--max_iter", "2", "--healpix_order", "2",
         "--offset_range", "3", "--offset_step", "1", "--init_resolution", "30", "--seed", "1")
    particles = _particles(out / "run_it002_data.star")
    assert set(particles["rlnClassNumber"].astype(int)) <= {1, 2}
    for k in (1, 2):
        _map(out / f"run_it002_class00{k}.mrc")
    model = starfile.read(out / "run_it002_model.star", always_dict=True)["model_classes"]
    distribution = model["rlnClassDistribution"].to_numpy(dtype=np.float64)
    assert len(distribution) == 2 and np.isclose(distribution.sum(), 1.0, atol=1e-4)


def test_initial_model_two_classes(tmp_path):
    """``relax initial_model --K 2``: three VDAM iterations writing both classes."""
    project = _project(tmp_path, "k2_5k128_data")
    out = tmp_path / "initial_model_k2"
    out.mkdir()
    n_iter = 3
    _run("initial_model", "--i", str(project / "particles.star"), "--datadir", str(project), "--o", str(out / "run"),
         "--nr_iter", str(n_iter), "--grad_write_iter", "1", "--K", "2", "--sym", "C1", "--particle_diameter", "400",
         "--oversampling", "1", "--healpix_order", "1", "--offset_range", "6", "--offset_step", "2",
         "--tau2_fudge", "4", "--random_seed", "1", "--gpu", "0")
    particles = _particles(out / f"run_it{n_iter:03d}_data.star")
    assert set(particles["rlnClassNumber"].astype(int)) <= {1, 2}
    for k in (1, 2):
        _map(out / f"run_it{n_iter:03d}_class00{k}.mrc")
    assert not (out / f"run_it{n_iter + 1:03d}_data.star").exists()


@pytest.mark.parametrize("oversampling", [0, 1], ids=["full_grid", "oversampling_1"])
def test_ppca_initial_model_single_particles(tmp_path, oversampling):
    """``relax ppca_initial_model --i particles.star``: three short updates, checkpoints and the final embeddings;
    with ``--oversampling 1`` every update runs RELION's two passes (the GPU stream pads its shift phases to the
    GEMM window)."""
    project = _project(tmp_path, "k1_5k128_data")
    out = tmp_path / "ppca"
    n_iter, q = 3, 2
    _run("ppca_initial_model", "--i", str(project / "particles.star"), "--particle-diameter", "400", "-o", str(out),
         "--q", str(q), "--iterations", str(n_iter), "--stages", "[[1, 4, 1]]", "--image-batch-size", "100",
         "--oversampling", str(oversampling))
    with open(out / "iterations.jsonl") as stream:
        updates = [json.loads(line) for line in stream]
    assert len(updates) == n_iter
    assert all(("oversampling" in update) == bool(oversampling) for update in updates)
    checkpoints = sorted(out.glob("checkpoint_*.npz"))
    assert [p.name for p in checkpoints] == [f"checkpoint_{i:04d}.npz" for i in range(n_iter + 1)]
    with np.load(checkpoints[-1], allow_pickle=False) as arrays:
        for name in ("theta", "noise"):
            values = np.asarray(arrays[name])
            assert values.size and np.all(np.isfinite(values)), f"{name}: empty or non-finite"
        assert np.any(np.asarray(arrays["theta"]) != 0)
    with np.load(out / "embeddings.npz", allow_pickle=False) as arrays:
        assert np.array_equal(np.sort(arrays["particle_ids"]), np.arange(N_PARTICLES))
        z = np.asarray(arrays["z"])
        assert z.shape == (N_PARTICLES, q) and np.all(np.isfinite(z))
        for name in ("log_evidence", "max_posterior"):
            values = np.asarray(arrays[name])
            assert values.shape == (N_PARTICLES,) and np.all(np.isfinite(values))
