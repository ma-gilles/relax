"""Same-checkpoint continuation and honest reporting of numerical discrepancies."""

import dataclasses
import json
import os
import subprocess
from unittest.mock import patch

import jax
import numpy as np
import pytest
from conftest import ROOT, repo_python_command, repo_subprocess_env
from helpers.float_compare import assert_trees_match

from relax.ppca_initial_class3d import checkpoint, iteration_loop
from relax.ppca_initial_class3d.tests.test_mixture_controller import spa_data, tiny_config, tomo_data
from relax.ppca_initial_class3d.tests.validate_k2_cli import (
    FORK_ITERATION,
    ITERATIONS,
    STOP_ITERATION,
    compare_run_artifacts,
    comparison_details,
    digest,
    fork_checkpoint_run,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("tomography", [False, True], ids=["spa", "two-noise-group-et"])
def test_k2_same_checkpoint_stop_resume_cpu(tmp_path, tomography):
    """Keep CPU execution separate from a parent that already activated CUDA."""
    env = repo_subprocess_env()
    env.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", RECOVAR_DISABLE_CUDA="1")
    for name in ("JAX_PLATFORM_NAME", "RECOVAR_CUDA_LIB", "RELAX_CUDA_LIB",
                 "RELAX_ENABLE_CUSTOM_CUDA", "RELAX_REQUIRE_CUSTOM_CUDA_FOR_TESTS"):
        env.pop(name, None)
    env["PYTHONPATH"] = str(ROOT / "tests") + os.pathsep + env["PYTHONPATH"]
    command = repo_python_command("-c", """
import sys
from pathlib import Path
from relax.ppca_initial_class3d.tests.test_mixture_restart import _check_cpu_restart
_check_cpu_restart(Path(sys.argv[1]), sys.argv[2] == "tomo")
""", str(tmp_path), "tomo" if tomography else "spa")
    child = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    assert child.returncode == 0, child.stdout + child.stderr


def _check_cpu_restart(tmp_path, tomography):
    """Isolate serialization/reload from independent GPU initialization/reductions."""
    assert jax.default_backend() == "cpu", "Restart serialization check must run in a CPU-only process"
    with jax.default_device(jax.devices("cpu")[0]):
        data = tomo_data() if tomography else spa_data(tmp_path / "inputs")
        config = dataclasses.replace(tiny_config(2), iterations=ITERATIONS, shift_range=1, shift_step=1)
        identity = {"case": "same-checkpoint", "tomography": tomography}
        prefix, direct, resumed = (tmp_path / name for name in ("prefix", "direct", "resumed"))
        iteration_loop.run(data, config, prefix, identity, 6., stop_after=FORK_ITERATION)
        shared = prefix / f"checkpoint_{FORK_ITERATION:04d}.npz"
        original_digest = digest(shared)
        direct_start = fork_checkpoint_run(prefix, direct, FORK_ITERATION)
        resumed_start = fork_checkpoint_run(prefix, resumed, FORK_ITERATION)

        def must_load_saved_state(*args, **kwargs):
            raise AssertionError("A same-checkpoint continuation must not initialize a fresh model")

        with patch.object(iteration_loop, "initialize_state", must_load_saved_state):
            iteration_loop.run(data, config, direct, identity, 6., resume=direct_start)
            stopped = iteration_loop.run(data, config, resumed, identity, 6.,
                                         resume=resumed_start, stop_after=STOP_ITERATION)
            require_checkpoint = resumed / f"checkpoint_{STOP_ITERATION:04d}.npz"
            assert stopped.iteration == STOP_ITERATION
            assert not (resumed / "assignments.npz").exists()
            restored = checkpoint.load(require_checkpoint, config, identity)
            assert_trees_match(dataclasses.asdict(restored), dataclasses.asdict(stopped))
            iteration_loop.run(data, config, resumed, identity, 6., resume=require_checkpoint)
        report = compare_run_artifacts(resumed, direct, ITERATIONS)
        assert report["passed"], json.dumps(report, indent=2)
        assert digest(shared) == original_digest == digest(direct_start) == digest(resumed_start)
        for path in (direct, resumed):
            records = [json.loads(line) for line in (path / "iterations.jsonl").read_text().splitlines()]
            assert [record["iteration"] for record in records] == list(range(1, ITERATIONS + 1))
        if tomography:
            assert data.n_noise_groups == 2
            assert restored.noise.shape[0] == 2


def test_comparison_report_retains_strict_failure():
    desired = np.asarray([1., 2.], np.float32)
    actual = desired.copy()
    actual[0] += np.float32(1e-3)
    result = comparison_details(actual, desired)
    assert not result["passed"]
    assert "rtol 1e-06" in result["reason"]
    assert result["max_abs"] > 1e-4
    assert comparison_details(desired, desired)["passed"]
    assert not comparison_details(np.asarray([0, 1]), np.asarray([0, 0]))["passed"]


def test_checkpoint_fork_copies_only_existing_prefix_and_refuses_overwrite(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    for iteration in range(4):
        np.savez(source / f"checkpoint_{iteration:04d}.npz",
                 metadata=json.dumps({"iteration": iteration}), theta=np.zeros(2, np.complex64))
    (source / "iterations.jsonl").write_text("".join(json.dumps({"iteration": i}) + "\n" for i in range(1, 4)))
    (source / "assignments.npz").write_bytes(b"not part of the prefix")
    copied = fork_checkpoint_run(source, target, 2)
    assert digest(copied) == digest(source / "checkpoint_0002.npz")
    assert not (target / "checkpoint_0003.npz").exists()
    assert not (target / "assignments.npz").exists()
    assert [json.loads(line)["iteration"] for line in (target / "iterations.jsonl").read_text().splitlines()] == [1, 2]
    with pytest.raises(FileExistsError):
        fork_checkpoint_run(source, target, 2)
