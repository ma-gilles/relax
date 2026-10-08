"""Iteration-zero replay frames, source precedence and controller ownership."""

import logging

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.parity import initial_model_replay as replay


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
        box_size=8,
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
    variance = replay.prepare_prior(model, n_classes=n_classes, box_size=8, volume_shape=(8, 8, 8))
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
    variance = replay.prepare_prior(model, n_classes=1, box_size=8, volume_shape=(8, 8, 8))
    assert_matches(variance, _radial_pixels(first[0] * 8**4, (8, 8, 8)))


@pytest.mark.unit
def test_reference_tau2_takes_precedence_over_reference_sigma2(tmp_path):
    shells = _write_model(tmp_path / "run_it000_model.star")
    model = replay.read_initial_model(tmp_path, n_classes=1)
    model.reference.tables["model_class_1"]["rlnReferenceSigma2"] = np.asarray(shells[0]) * 9.0
    variance = replay.prepare_prior(model, n_classes=1, box_size=8, volume_shape=(8, 8, 8))
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
            box_size=8,
            image_shape=(8, 8),
            explicit_noise_radial=np.ones(5) if override == "npz" else None,
            live_sigma2=np.ones(5) if override == "live" else None,
            log=logging.getLogger(__name__),
        )


def _relion_init_run(monkeypatch, tmp_path, n_classes, trace=None, **stand_ins):
    """Main up to the controller with --relion_init_dir: the half-set STAR is the input STAR, and the run_it000
    reads are stood in for (a model with a 256-pixel noise image and a 4096-voxel prior)."""
    from helpers.tiny_main import controller_inputs, write_tiny_data_dir

    from relax.parity import relion_replay

    data = write_tiny_data_dir(tmp_path / "data", n_classes=n_classes,
                               extra_columns={"rlnRandomSubset": np.arange(12) % 2 + 1})
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "run_it000_data.star").write_text((data / "particles.star").read_text())
    def overrides(relion_dir, half1_rows, half2_rows, max_iter, **kwargs):
        # run_it000's input origins, which a fresh Class3D start keeps.
        first = {"previous_best_translations": [np.zeros((len(half1_rows), 2)), np.zeros((len(half2_rows), 2))]}
        return [first] + [None] * int(max_iter)

    monkeypatch.setattr(relion_replay, "_build_replay_iteration_overrides", overrides)
    defaults = dict(
        read_initial_model=lambda directory, *, n_classes: "model",
        prepare_noise=lambda model, **kwargs: replay.NoiseReplay(np.full(256, 2.0, np.float32), [np.ones(9)], 16**4),
        log_noise_source=lambda noise, *, log: None,
        prepare_prior=lambda model, **kwargs: np.full(4096 if n_classes == 1 else (n_classes, 4096), 3.0),
        read_controls=lambda model, *, log: replay.InitialModelControls(1.75, 0.42),
    )
    for name, function in {**defaults, **stand_ins}.items():
        monkeypatch.setattr(replay, name, function)
        if trace is not None:
            trace.wrap(replay, name)
    # A non-MPI RELION oracle for Class3D: single-process group scales, no dispatch schedule.
    command = ["refine"] if n_classes == 1 else ["class3d", "--n_classes", str(n_classes), "--relion-scale-followers", "0"]
    return controller_inputs(monkeypatch, tmp_path, command[0], *command[1:], "--relion_init_dir", tmp_path / "model",
                             "--relion_half_sets", "<DATA>/particles.star", n_classes=n_classes, data=data)


@pytest.mark.unit
@pytest.mark.parametrize("n_classes", [1, 4])
def test_controller_installs_arrays_before_reporting_and_drops_temporary_owner(monkeypatch, tmp_path, n_classes):
    """--relion_init_dir: RELION's iteration-0 noise, then prior, then controls; each replaced start-up array
    is released when its replacement is installed, and the replayed values reach the controller."""
    from helpers.tiny_main import main_frame_arrays
    from helpers.tiny_refinement import CallTrace

    held = {}

    def prior(model, **kwargs):
        held["before_prior"] = main_frame_arrays()
        held["prior"] = np.full(4096 if n_classes == 1 else (n_classes, 4096), 3.0)
        return held["prior"]

    def controls(model, *, log):
        # The start-up prior main bootstrapped from the reference's power: one real value per voxel. Reading
        # main's locals again refreshes the frame's snapshot of them, which held the replaced prior.
        main_frame_arrays()
        alive = [reference() for reference in held["before_prior"] if reference() is not None]
        assert not [value for value in alive if value is not held["prior"] and np.ndim(value) == 1
                    and np.size(value) == 4096 and not np.iscomplexobj(value)]
        return replay.InitialModelControls(1.75, 0.42)

    trace = CallTrace(monkeypatch)
    inputs = _relion_init_run(monkeypatch, tmp_path, n_classes, trace=trace, prepare_prior=prior, read_controls=controls)
    assert trace.labels() == ["read_initial_model", "prepare_noise", "log_noise_source", "prepare_prior", "read_controls"]
    assert_matches(np.asarray(inputs["init_mean_variance"]), np.asarray(trace.calls("prepare_prior")[0].result))
    assert_matches(np.asarray(inputs["init_noise_variance"]), np.full(256, 2.0, np.float32))
    assert_matches(inputs["options"].parity.tau2_fudge, 1.75)
    assert_matches(inputs["options"].schedule.init_translation_sigma_angstrom, 0.42)


@pytest.mark.unit
def test_controller_does_not_read_replay_inputs_outside_its_gate(monkeypatch, tmp_path):
    from helpers.tiny_main import controller_inputs
    from helpers.tiny_refinement import CallTrace

    def refuse(*args, **kwargs):
        raise AssertionError("run_it000 read without --relion_init_dir")

    trace = CallTrace(monkeypatch)
    for name in ("read_initial_model", "prepare_noise", "prepare_prior", "read_controls"):
        monkeypatch.setattr(replay, name, refuse)
        trace.wrap(replay, name)
    controller_inputs(monkeypatch, tmp_path, "refine")
    assert trace.labels() == []
