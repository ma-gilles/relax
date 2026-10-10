"""Package-contained tests of the unified CLI and unchanged single-model path."""

import dataclasses
import json
import sys
from argparse import Namespace

import pytest

from relax.commands import ppca_initial_model as command
from relax.ppca_initial_class3d import images, runner
from relax.ppca_initial_class3d.config import Config as MixtureConfig
from relax.ppca_initial_model import iteration_loop
from relax.ppca_initial_model.config import Config

pytestmark = pytest.mark.unit


def test_pre_extension_namespace_keeps_original_single_model_dispatch():
    from relax.ppca_initial_class3d.command import dispatch

    assert dispatch(Namespace()) is False


@pytest.mark.parametrize("k", [None, 1, 2])
@pytest.mark.parametrize("tomography", [False, True], ids=["spa", "tomo"])
def test_unified_command_selects_controller_and_preserves_single_identity(tmp_path, monkeypatch, k, tomography):
    monkeypatch.setenv("SLURM_JOB_ID", "unit-test")
    data, calls = object(), []
    identity = {"input_hash": "unchanged"}
    monkeypatch.setattr(command, "load_training", lambda *_args: (data, {"particle_diameter_ang": 6}, dict(identity)))
    monkeypatch.setattr(command, "load_tilt_training", lambda *_args: (data, dict(identity)))
    # Tilt inputs default to the zero mask, whose loader lives in the package (images.py).
    monkeypatch.setattr(images, "load_tilt_training", lambda *_args, **_kw: (data, dict(identity)))
    monkeypatch.setattr(command, "training_image_reading", lambda *_args: (True, {"preread": True}))
    monkeypatch.setattr(command, "source_identity", lambda: {"files": {}})
    monkeypatch.setattr(iteration_loop, "run", lambda *args, **kw: calls.append(("single", args, kw)))
    monkeypatch.setattr(runner, "run_iterations", lambda *args, **kw: calls.append(("mixture", args, kw)))
    inputs = ["--ios", "ios.star", "--particle-diameter", "6"] if tomography else ["training.json"]
    argv = [*inputs, "-o", str(tmp_path), "--q", "1", "--iterations", "2",
            "--ppca-gemm-precision", "fp32", "--ppca-preread-images", "off",
            "--ppca-pass2-mass-floor", "0", "--stop-after", "1"]
    if k is not None:
        argv += ["--K", str(k)]
    if k == 2:
        argv += ["--class-pseudocount", "0.5"]
    command.main(argv)
    assert len(calls) == 1
    mode, args, kwargs = calls[0]
    assert mode == ("mixture" if k == 2 else "single")
    assert args[0] is data and args[2] == tmp_path and args[4] == 6
    assert kwargs == {"resume": None, "stop_after": 1, "stop_file": None}
    config, actual_identity = args[1], args[3]
    assert config.q == 1 and config.iterations == 2
    assert config.gemm_precision == "fp32" and config.preread_images == "off" and config.pass2_mass_floor == 0
    receipt = json.loads((tmp_path / "run.json").read_text())
    if k == 2:
        assert type(config) is MixtureConfig and config.n_classes == 2 and config.class_pseudocount == .5
        assert config.auto_sampling is tomography
        assert config.shift_step == Config().shift_step
        assert receipt["command"] == "ppca_initial_model"
        assert actual_identity["particle_diameter_ang"] == 6
        assert "relax/ppca_initial_class3d/runner.py" in actual_identity["source"]["files"]
        assert "relax/commands/ppca_initial_class3d.py" not in actual_identity["source"]["files"]
    else:
        assert type(config) is Config
        assert "n_classes" not in dataclasses.asdict(config)
        assert actual_identity == {**identity, "source": {"files": {}}}
        assert set(receipt) == {"config", "identity", "image_reading"}


@pytest.mark.parametrize("options", [
    ["--K", "0"], ["--K", "-1"], ["--K", "1", "--class-pseudocount", "1"],
    ["--K", "2", "--oversampling-start", "0"], ["--K", "2", "--full-grid", "--oversampling-start", "3"],
    ["--K", "2", "--optimizer", "momentum_sgd"],
    ["--K", "2", "--no-stream-coarse-recompute"],
    ["--K", "2", "--tomogram-batches", "--balanced-stochastic-halves", "--stochastic-batch-size", "4",
     "--stochastic-all-iterations"],
    ["--K", "2", "--fine-image-tile-size", "2"], ["--K", "2", "--sgd-learning-rate", "0.1"],
])
def test_invalid_or_unsupported_class_options_fail_before_input_loading(tmp_path, monkeypatch, options):
    monkeypatch.setenv("SLURM_JOB_ID", "unit-test")
    monkeypatch.setattr(command, "load_training", lambda *_args: pytest.fail("invalid options loaded data"))
    with pytest.raises(ValueError):
        command.main(["training.json", "-o", str(tmp_path / "new_output"), *options])
    assert not (tmp_path / "new_output").exists()


@pytest.mark.parametrize("tomography", [False, True], ids=["spa", "tomo"])
@pytest.mark.parametrize("sampling_flag,expected", [
    (None, None), ("--auto-sampling", True), ("--no-auto-sampling", False),
])
def test_sampling_default_resolves_at_mixture_input_boundary(monkeypatch, tomography, sampling_flag, expected):
    calls = []
    monkeypatch.setattr(runner, "run", lambda args, config: calls.append(config))
    inputs = ["--ios", "ios.star", "--particle-diameter", "6"] if tomography else ["training.json"]
    argv = [*inputs, "-o", "unused-output", "--K", "2"]
    if sampling_flag is not None:
        argv.append(sampling_flag)
    command.main(argv)
    assert len(calls) == 1
    assert calls[0].auto_sampling is (tomography if expected is None else expected)
    assert calls[0].shift_step == Config().shift_step


def test_explicit_shift_step_one_remains_available(monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "run", lambda args, config: calls.append(config))
    command.main(["--ios", "ios.star", "--particle-diameter", "6", "-o", "unused-output",
                  "--K", "2", "--no-auto-sampling", "--shift-step", "1"])
    assert len(calls) == 1
    assert not calls[0].auto_sampling
    assert calls[0].shift_step == 1


def test_standalone_class3d_command_is_not_discoverable(monkeypatch, capsys):
    from relax.command_line import main_commands

    monkeypatch.setattr(sys, "argv", ["relax", "ppca_initial_class3d"])
    with pytest.raises(SystemExit) as error:
        main_commands()
    assert error.value.code == 1
    text = capsys.readouterr().err
    assert "Command 'ppca_initial_class3d' not found." in text
    assert "  ppca_initial_class3d\n" not in text
    assert "  ppca_initial_model\n" in text
