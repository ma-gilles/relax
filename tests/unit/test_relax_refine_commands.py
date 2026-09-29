"""``relax refine`` and ``relax class3d`` parse and dispatch to the full-refinement driver."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from relax import command_line
from relax.refinement import full_refinement

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_relax_lists_the_refinement_commands(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["relax"])
    with pytest.raises(SystemExit):
        command_line.main_commands()
    listed = capsys.readouterr().err.split()
    assert "refine" in listed and "class3d" in listed


@pytest.mark.parametrize("command", ["refine", "class3d"])
def test_help_parses_the_refinement_options(monkeypatch, capsys, command):
    # The driver points JAX's live config at recovar's compilation cache first; keep this process's config.
    monkeypatch.setattr(full_refinement, "activate_recovar_compilation_cache", lambda: None)
    monkeypatch.setattr(sys, "argv", ["relax", command, "--help"])
    with pytest.raises(SystemExit) as excinfo:
        command_line.main_commands()
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "--data_dir" in out and "--n_classes" in out


@pytest.mark.parametrize("command", ["refine", "class3d"])
def test_command_dispatches_to_the_driver_with_its_arguments(monkeypatch, command):
    seen = {}

    def fake_main(command=None):
        seen["command"] = command
        seen["argv"] = list(sys.argv)

    monkeypatch.setattr(full_refinement, "main", fake_main)
    monkeypatch.setattr(sys, "argv", ["relax", command, "--data_dir", "d", "--output", "o"])
    command_line.main_commands()
    assert seen == {"command": command, "argv": ["relax", "--data_dir", "d", "--output", "o"]}


@pytest.mark.parametrize(
    ("command", "n_classes", "accepted"),
    [("refine", 1, True), ("refine", 4, False), ("class3d", 4, True), ("class3d", 2, True), ("class3d", 1, False)],
)
def test_each_command_runs_only_its_class_count(command, n_classes, accepted):
    if accepted:
        full_refinement._require_command_n_classes(command, n_classes)
    else:
        with pytest.raises(SystemExit, match="relax (refine|class3d)"):
            full_refinement._require_command_n_classes(command, n_classes)


def test_requested_diagnostic_dump_stop_exits_zero(monkeypatch):
    class SignificanceDumpComplete(RuntimeError):
        pass

    def stop(command=None):
        raise SignificanceDumpComplete("target reached")

    monkeypatch.setattr(full_refinement, "main", stop)
    monkeypatch.setenv("RELAX_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET", "1")
    with pytest.raises(SystemExit) as excinfo:
        full_refinement.run_from_command_line("refine")
    assert excinfo.value.code == 0
    monkeypatch.delenv("RELAX_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET")
    with pytest.raises(SignificanceDumpComplete):
        full_refinement.run_from_command_line("refine")


def test_python_m_class3d_rejects_a_missing_class_count(tmp_path):
    """The module form runs the same driver and refuses K=1 before reading any input."""

    env = {**os.environ, "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": "", "PYTHONNOUSERSITE": "1"}
    proc = subprocess.run(
        [sys.executable, "-m", "relax.commands.class3d", "--data_dir", str(tmp_path), "--output", str(tmp_path)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode != 0
    assert "relax class3d needs --n_classes K with K >= 2" in proc.stderr
