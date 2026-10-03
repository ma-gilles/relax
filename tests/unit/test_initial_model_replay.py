"""Iteration-zero replay frames, source precedence and controller ownership."""

import ast
import logging
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.diagnostics import initial_model_replay as replay


def _write_model(path, *, n_classes=1, sigma2=None, prior_dtype=np.float64, prior_column="rlnReferenceTau2", fudge="_rlnTau2FudgeFactor 1.75"):
    if sigma2 is None:
        sigma2 = np.asarray([1.01, 1.23, 1.57, 1.88, 2.31], dtype=np.float64)
    sections = ["data_model_general", fudge, "_rlnSigmaOffsetsAngst 8.125", ""]
    if sigma2 is not False:
        sections += ["data_model_optics_group_1", "loop_", "_rlnSpectralIndex #1", "_rlnSigma2Noise #2"]
        sections += [f"{index} {value:.17g}" for index, value in enumerate(sigma2)]
    prior_shells = []
    for index in range(n_classes):
        shells = np.asarray((np.arange(5) + 0.123456789) * 0.7**(index + 1), dtype=prior_dtype)
        prior_shells.append(shells)
        sections += ["", f"data_model_class_{index + 1}", "loop_", "_rlnSpectralIndex #1", f"_{prior_column} #2"]
        sections += [f"{shell} {value:.17g}" for shell, value in enumerate(shells)]
    path.write_text("\n".join(sections) + "\n")
    return prior_shells


def _radial_pixels(shells, shape):
    coordinates = np.indices(shape) - np.asarray(shape).reshape((-1,) + (1,) * len(shape)) // 2
    # Startup radial expansion uses nearest-integer Fourier shells.
    shells_by_pixel = np.rint(np.sqrt(np.sum(coordinates**2, axis=0))).astype(int).reshape(-1)
    return np.asarray(shells)[np.minimum(shells_by_pixel, len(shells) - 1)]


@pytest.mark.unit
@pytest.mark.parametrize("n_classes,source", [(1, "shared"), (1, "half-specific"), (4, "shared")])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("override", ["star", "npz", "live", "npz-and-live"])
def test_noise_frames_override_precedence_and_half1_broadcast(tmp_path, n_classes, source, dtype, override):
    first = np.asarray([1.01, 1.23, 1.57, 1.88, 2.31], dtype=dtype)
    if source == "shared":
        _write_model(tmp_path / "run_it000_model.star", n_classes=n_classes, sigma2=first)
    else:
        _write_model(tmp_path / "run_it000_half1_model.star", sigma2=first)
        _write_model(tmp_path / "run_it000_half2_model.star", sigma2=first + 3.0)
    model = replay.read_initial_model(tmp_path, n_classes=n_classes)
    frame_scale = 8**4
    explicit = np.asarray([2.34, 2.79, 3.16, 3.73, 4.19], dtype=dtype) * frame_scale
    live = np.asarray([5.01, 5.47, 5.89, 6.31, 6.83], dtype=dtype)
    noise = replay.prepare_noise(
        model,
        grid_size=8,
        image_shape=(8, 8),
        explicit_noise_radial=explicit if "npz" in override else None,
        live_sigma2=live if "live" in override else None,
        log=logging.getLogger(__name__),
    )
    expected = (
        np.asarray(explicit, dtype=np.float64) / frame_scale
        if "npz" in override
        else np.asarray(live if "live" in override else first, dtype=np.float64)
    )
    assert model.source == source
    assert model.reference is model.models[0]
    assert noise.frame_scale == frame_scale
    assert_matches(noise.sigma2_per_model[0], expected)
    assert noise.sigma2_per_model[0].dtype == np.float64
    variance = noise.variance if source == "shared" else noise.variance[0]
    assert_matches(variance, _radial_pixels(expected * frame_scale, (8, 8)))
    assert np.asarray(variance).dtype == np.float64
    if source == "half-specific":
        assert_matches(noise.sigma2_per_model[1], expected)
        assert noise.sigma2_per_model[1] is not noise.sigma2_per_model[0]
        assert_matches(noise.variance[1], variance)


@pytest.mark.unit
@pytest.mark.parametrize("n_classes", [1, 4])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("column", ["rlnReferenceTau2", "rlnReferenceSigma2"])
def test_tau2_shell_frame_and_class_layout(tmp_path, n_classes, dtype, column):
    shells = _write_model(tmp_path / "run_it000_model.star", n_classes=n_classes, prior_dtype=dtype, prior_column=column)
    model = replay.read_initial_model(tmp_path, n_classes=n_classes)
    variance = replay.prepare_prior(model, n_classes=n_classes, grid_size=8, volume_shape=(8, 8, 8))
    expected = np.stack([_radial_pixels(np.asarray(row, dtype=np.float64) * 8**4, (8, 8, 8)) for row in shells])
    if n_classes == 1:
        expected = expected[0]
    assert variance.shape == expected.shape
    assert np.asarray(variance).dtype == np.float64
    assert_matches(variance, expected)


@pytest.mark.unit
def test_half_specific_prior_uses_the_half1_reference_model(tmp_path):
    first = _write_model(tmp_path / "run_it000_half1_model.star", prior_dtype=np.float64)
    _write_model(tmp_path / "run_it000_half2_model.star", prior_dtype=np.float64)
    model = replay.read_initial_model(tmp_path, n_classes=1)
    model.models[1].tables["model_class_1"]["rlnReferenceTau2"] *= 7.0
    variance = replay.prepare_prior(model, n_classes=1, grid_size=8, volume_shape=(8, 8, 8))
    assert_matches(variance, _radial_pixels(first[0] * 8**4, (8, 8, 8)))


@pytest.mark.unit
def test_reference_tau2_takes_precedence_over_reference_sigma2(tmp_path):
    shells = _write_model(tmp_path / "run_it000_model.star")
    model = replay.read_initial_model(tmp_path, n_classes=1)
    model.reference.tables["model_class_1"]["rlnReferenceSigma2"] = np.asarray(shells[0]) * 9.0
    variance = replay.prepare_prior(model, n_classes=1, grid_size=8, volume_shape=(8, 8, 8))
    assert_matches(variance, _radial_pixels(shells[0] * 8**4, (8, 8, 8)))


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_fudge,optimiser,expected_fudge,expected_offset",
    [
        ("_rlnTau2FudgeFactor 1.75", "_rlnTau2FudgeArg 3.25\n_rlnSigmaOffsetsAngst 0.42", 1.75, 0.42),
        ("_rlnTau2FudgeArg -1", "_rlnTau2FudgeArg 3.25\n_rlnSigmaOffsetsAngst 0.42", 3.25, 0.42),
        ("", "_rlnTau2FudgeFactor 2.75", 2.75, None),
        ("_rlnTau2FudgeFactor -1", "_rlnTau2FudgeArg 3.25", -1.0, None),
        ("_rlnTau2FudgeFactor 1.75", None, 1.75, None),
        ("_rlnTau2FudgeArg -1", None, None, None),
    ],
)
def test_model_fudge_precedence_and_optimiser_only_offset(tmp_path, model_fudge, optimiser, expected_fudge, expected_offset):
    _write_model(tmp_path / "run_it000_model.star", fudge=model_fudge)
    if optimiser is not None:
        (tmp_path / "run_it000_optimiser.star").write_text(optimiser + "\n")
    model = replay.read_initial_model(tmp_path, n_classes=1)
    controls = replay.read_controls(model, log=logging.getLogger(__name__))
    for value, expected in [(controls.tau2_fudge, expected_fudge), (controls.sigma_offset_angstrom, expected_offset)]:
        if expected is None:
            assert value is None
        else:
            assert_matches(value, expected)


@pytest.mark.unit
@pytest.mark.parametrize("override", ["npz", "live"])
def test_missing_model_noise_still_refuses_with_an_override(tmp_path, override):
    _write_model(tmp_path / "run_it000_model.star", sigma2=False)
    model = replay.read_initial_model(tmp_path, n_classes=1)
    with pytest.raises(ValueError, match="missing rlnSigma2Noise"):
        replay.prepare_noise(
            model,
            grid_size=8,
            image_shape=(8, 8),
            explicit_noise_radial=np.ones(5) if override == "npz" else None,
            live_sigma2=np.ones(5) if override == "live" else None,
            log=logging.getLogger(__name__),
        )


def _controller_replay_code():
    source = Path(__file__).resolve().parents[2] / "relax/refinement/full_refinement.py"
    tree = ast.parse(source.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    block = next(
        node for node in main.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "args.relion_init_dir is not None and frozen_boundary is None"
    )
    return compile(ast.Module(body=[block], type_ignores=[]), str(source), "exec")


@pytest.mark.unit
@pytest.mark.parametrize("n_classes", [1, 4])
def test_controller_installs_arrays_before_reporting_and_drops_temporary_owner(n_classes):
    code = _controller_replay_code()
    namespace = dict(
        args=SimpleNamespace(relion_init_dir="model", n_classes=n_classes),
        frozen_boundary=None,
        ds=SimpleNamespace(grid_size=8, image_shape=(8, 8), volume_shape=(8, 8, 8)),
        initial_noise_radial=None,
        relion_live_initial_sigma2=None,
        noise_variance=np.empty(64),
        mean_variance=np.empty(512),
    )
    old_noise = weakref.ref(namespace["noise_variance"])
    old_prior = weakref.ref(namespace["mean_variance"])
    events = []

    def read(directory, *, n_classes):
        events.append("read")
        return "model"

    def noise(model, **kwargs):
        assert old_noise() is not None
        events.append("noise")
        return replay.NoiseReplay(np.zeros(64), [np.ones(5)], 8**4)

    def report(noise, *, log):
        assert namespace["noise_variance"] is noise.variance
        assert old_noise() is None
        events.append("noise reported")

    def prior(model, **kwargs):
        assert "replayed_noise" not in namespace
        assert old_prior() is not None
        events.append("prior")
        return np.zeros(512)

    def controls(model, *, log):
        assert old_prior() is None
        events.append("controls")
        return replay.InitialModelControls(1.75, 0.42)

    class Log:
        def info(self, *args):
            assert old_prior() is None
            events.append("prior reported")

    namespace.update(
        initial_model_replay=SimpleNamespace(read_initial_model=read, prepare_noise=noise, log_noise_source=report, prepare_prior=prior, read_controls=controls),
        logger=Log(),
    )
    exec(code, namespace)
    assert events == ["read", "noise", "noise reported", "prior", "prior reported", "controls"]
    assert_matches(namespace["relion_init_tau2_fudge"], 1.75)
    assert_matches(namespace["relion_init_sigma_offset_angstrom"], 0.42)


@pytest.mark.unit
@pytest.mark.parametrize("directory,frozen", [(None, None), ("model", object())])
def test_controller_does_not_read_replay_inputs_outside_its_gate(directory, frozen):
    namespace = dict(args=SimpleNamespace(relion_init_dir=directory), frozen_boundary=frozen)
    exec(_controller_replay_code(), namespace)
    assert "initial_model" not in namespace
