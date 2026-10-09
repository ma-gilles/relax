"""RELION's start-up noise estimate is the only initial-noise estimator of a refinement."""

import argparse
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.refinement import command_options, startup_noise
from relax.refinement import full_refinement as driver
from relax.refinement.half_inputs import HalfPair

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('option', ['--initial-noise-bootstrap', '--initial_noise_cache_dir'])
def test_cli_has_no_other_noise_estimator(monkeypatch, capsys, option):
    monkeypatch.setattr(sys, 'argv', ['run_full_refinement.py', '--data_dir', 'd', '--output', 'o', option, 'pipeline'])
    with pytest.raises(SystemExit):
        command_options.parse_refinement_args()
    assert 'unrecognized arguments' in capsys.readouterr().err


def test_parsed_defaults_never_enable_full_state_replay(monkeypatch):
    class Parsed(BaseException):
        pass

    original = argparse.ArgumentParser.parse_args

    def stop_after_parse(parser, *args, **kwargs):
        parsed = original(parser, *args, **kwargs)
        assert parsed.relion_init_dir is None
        assert parsed.perturb_replay_relion_dir is None
        raise Parsed

    monkeypatch.setattr(argparse.ArgumentParser, 'parse_args', stop_after_parse)
    monkeypatch.setattr(sys, 'argv', ['run_full_refinement.py', '--data_dir', 'd', '--output', 'o'])
    with pytest.raises(Parsed):
        driver.main()


def _args(**overrides):
    return SimpleNamespace(**(dict(n_classes=1, init_relion_iteration=0,
        perturb_replay_relion_dir=None, relion_init_dir=None,
        init_noise_from_npz=None, relion_half_sets="particles.star") | overrides))


def test_startup_noise_preserves_order_and_float32_boundary(monkeypatch):
    dataset = SimpleNamespace(grid_size=8)
    rows = np.array([4, 1, 2], dtype=np.int64)
    optics = np.array([7, 7, 7], dtype=np.int64)
    sigma = np.array([[.5, .4, .3, .2, .1]], dtype=np.float64)
    calls = []

    def compute(ds, **kwargs):
        assert ds is dataset
        assert_matches(kwargs['source_rows'], rows)
        assert_matches(kwargs['optics_group_ids'], optics)
        assert kwargs['particle_diameter_ang'] == 12
        assert kwargs['width_mask_edge_px'] == 3
        assert kwargs['image_pixel_size'] == 1.25
        calls.append(kwargs)
        return sigma

    monkeypatch.setattr(startup_noise, 'estimate_startup_sigma2', compute)
    radial, noise = startup_noise.prepare_startup_noise(
        dataset, source_rows=rows,
        optics_group_ids=optics, mask_params=(12., 3),
        optics_pixel_sizes=np.array([1.25]), output_dtype=np.float32, pair_counting="relion")
    assert noise.half2 is noise.half1
    noise = noise.half1
    assert len(calls) == 1
    assert radial.dtype == np.float64 and noise.dtype == np.float32
    assert_matches(radial, sigma[0] * 8**4)
    assert_matches(noise, startup_noise.scoring_noise_from_sigma2(
        sigma[0], box_size=8, output_dtype=np.float32))


def test_startup_noise_float64_output_for_double_scoring(monkeypatch):
    sigma = np.array([[.5, .4, .3, .2, .1]], dtype=np.float64)
    monkeypatch.setattr(startup_noise, 'estimate_startup_sigma2', lambda ds, **kwargs: sigma)
    _radial, noise = startup_noise.prepare_startup_noise(
        SimpleNamespace(grid_size=8),         source_rows=np.arange(3), optics_group_ids=np.ones(3), mask_params=(12., 3),
        optics_pixel_sizes=np.array([1.25]), output_dtype=np.float64, pair_counting="relion")
    assert noise.half1.dtype == np.float64






@pytest.mark.parametrize('overrides', [
    dict(source_rows=None),
    dict(optics_group_ids=None), dict(mask_params=None), dict(optics_pixel_sizes=None),
    dict(optics_pixel_sizes=np.array([1.25, 1.3])),
])
def test_startup_noise_rejects_missing_or_unsupported_inputs(overrides):
    params = dict(source_rows=np.arange(3),
                  optics_group_ids=np.ones(3), mask_params=(12., 3),
                  optics_pixel_sizes=np.array([1.25])) | overrides
    with pytest.raises(ValueError, match='start-up noise'):
        startup_noise.prepare_startup_noise(SimpleNamespace(grid_size=8),
            **params, output_dtype=np.float32, pair_counting="relion")


def test_class3d_startup_noise_has_one_spectrum_per_optics_group(monkeypatch):
    # Class3D keeps one sigma2_noise per optics group, as K=1 does (subtomogram Class3D, two groups).
    monkeypatch.setattr(
        startup_noise, 'estimate_startup_sigma2',
        lambda ds, **kwargs: np.array([[.5, .4, .3, .2, .1], [.6, .5, .4, .3, .2]], dtype=np.float64),
    )
    radial, noise = startup_noise.prepare_startup_noise(SimpleNamespace(grid_size=8),
                source_rows=np.arange(3), optics_group_ids=np.array([1, 2, 1]), mask_params=(12., 3),
        optics_pixel_sizes=np.array([1.25, 1.25]), output_dtype=np.float32, pair_counting="relion")
    assert radial.shape[0] == 2 and noise.half1.shape[0] == 2


def test_class3d_startup_noise_takes_unsplit_order(monkeypatch):
    rows = np.array([2, 0, 1], dtype=np.int64)
    seen = {}

    def compute(ds, **kwargs):
        seen.update(kwargs)
        return np.array([[.5, .4, .3, .2, .1]], dtype=np.float64)

    monkeypatch.setattr(startup_noise, 'estimate_startup_sigma2', compute)
    startup_noise.prepare_startup_noise(
        SimpleNamespace(grid_size=8), source_rows=rows, optics_group_ids=np.ones(3, dtype=np.int64),
        mask_params=(12., 3), optics_pixel_sizes=np.array([1.25]), output_dtype=np.float32, pair_counting="relion")
    assert_matches(seen['source_rows'], rows)


def test_class3d_noise_layout_is_the_micrograph_sorted_input_order():
    import pandas as pd

    # relion_refine stable-sorts rlnMicrographName byte-wise (exp_model.cpp:900-901);
    # Class3D keeps that order unsplit for the startup noise.
    particles = pd.DataFrame({
        'rlnImageName': ['1@s.mrcs', '2@s.mrcs', '3@s.mrcs', '4@s.mrcs'],
        'rlnMicrographName': ['2', '10', '1', '10'],
        'rlnOpticsGroup': [1, 1, 2, 1],
    })
    rows, optics = startup_noise.class3d_noise_order(particles)
    assert_matches(rows, [2, 1, 3, 0])
    assert_matches(optics, [2, 1, 1, 1])


@pytest.mark.parametrize("source", ["fresh", "replay", "loaded", "model"])
def test_cli_selects_noise_source_before_image_estimation(source, monkeypatch, tmp_path):
    """The command estimates the start-up noise from the images only when nothing else supplies it (a fresh
    or a STAR-replayed start); an archive's spectrum or RELION's iteration-0 model replaces the estimate.

    A frozen boundary has no CPU run; K=1 without half sets is refused before the noise is chosen.
    """
    from helpers.tiny_main import controller_inputs, run_tiny_main, write_tiny_data_dir
    from helpers.tiny_refinement import CallTrace, write_replay_dir

    from relax.parity import initial_model_replay, relion_replay

    # The consistency option is refused with RELION-seeded or replayed state: only a fresh start sets it.
    pair_counting = "once" if source == "fresh" else "relion"
    arguments = ["--initial_noise_pair_counting", pair_counting]
    data = None
    if source == "loaded":
        archive = run_tiny_main(monkeypatch, tmp_path, "refine", "--max_iter", "1", output="first")
        arguments += ["--init_noise_from_npz", archive / "refinement_results.npz"]
    if source in {"replay", "model"}:
        data = write_tiny_data_dir(tmp_path / "data", extra_columns={"rlnRandomSubset": np.arange(12) % 2 + 1})
        arguments += ["--relion_half_sets", "<DATA>/particles.star"]
        monkeypatch.setattr(relion_replay, "_build_replay_iteration_overrides",
                            lambda *args, **kwargs: [None] * (int(args[3]) + 1))
    if source == "replay":
        arguments += ["--perturb_replay_relion_dir", write_replay_dir(tmp_path / "relion", max_iter=2)]
    replayed_noise = np.full(256, 2.0, np.float32)
    if source == "model":
        (tmp_path / "model").mkdir()
        arguments += ["--relion_init_dir", tmp_path / "model"]
        for name, function in dict(
            read_initial_model=lambda directory, *, n_classes: "model",
            prepare_noise=lambda model, **kwargs: initial_model_replay.NoiseReplay(HalfPair.shared(replayed_noise), [np.ones(9)], 16**4),
            log_noise_source=lambda noise, *, log: None,
            prepare_prior=lambda model, **kwargs: np.full(4096, 3.0),
            read_controls=lambda model, *, log: initial_model_replay.InitialModelControls(1.0, 10.0),
        ).items():
            monkeypatch.setattr(initial_model_replay, name, function)
    trace = CallTrace(monkeypatch).wrap(startup_noise, "prepare_startup_noise", "estimate")
    inputs = controller_inputs(monkeypatch, tmp_path / "run", "refine", *arguments, data=data)
    estimates = trace.calls("estimate")
    assert len(estimates) == int(source in {"fresh", "replay"})
    if estimates:
        assert estimates[0].kwargs["output_dtype"] == np.float32
        assert estimates[0].kwargs["pair_counting"] == pair_counting  # the command's option reaches the estimate
        assert_matches(np.asarray(inputs["init_noise_variance"]), np.asarray(estimates[0].result.pixel_variance))
    elif source == "model":
        assert_matches(np.asarray(inputs["init_noise_variance"].half1), replayed_noise)
