"""RELION's start-up noise estimate is the only initial-noise estimator of a refinement."""

import argparse
import ast
import logging
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.refinement import command_options, startup_noise
from relax.refinement import full_refinement as driver

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
        optics_pixel_sizes=np.array([1.25]))
    assert len(calls) == 1
    assert radial.dtype == np.float64 and noise.dtype == np.float32
    assert_matches(radial, sigma[0] * 8**4)
    assert_matches(noise, startup_noise.scoring_noise_from_sigma2(
        sigma[0], grid_size=8, output_dtype=np.float32))


def test_startup_noise_float64_output_for_double_scoring(monkeypatch):
    sigma = np.array([[.5, .4, .3, .2, .1]], dtype=np.float64)
    monkeypatch.setattr(startup_noise, 'estimate_startup_sigma2', lambda ds, **kwargs: sigma)
    _radial, noise = startup_noise.prepare_startup_noise(
        SimpleNamespace(grid_size=8),         source_rows=np.arange(3), optics_group_ids=np.ones(3), mask_params=(12., 3),
        optics_pixel_sizes=np.array([1.25]), output_dtype=np.float64)
    assert noise.dtype == np.float64






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
            **params)


def test_class3d_startup_noise_has_one_spectrum_per_optics_group(monkeypatch):
    # Class3D keeps one sigma2_noise per optics group, as K=1 does (subtomogram Class3D, two groups).
    monkeypatch.setattr(
        startup_noise, 'estimate_startup_sigma2',
        lambda ds, **kwargs: np.array([[.5, .4, .3, .2, .1], [.6, .5, .4, .3, .2]], dtype=np.float64),
    )
    radial, noise = startup_noise.prepare_startup_noise(SimpleNamespace(grid_size=8),
                source_rows=np.arange(3), optics_group_ids=np.array([1, 2, 1]), mask_params=(12., 3),
        optics_pixel_sizes=np.array([1.25, 1.25]))
    assert radial.shape[0] == 2 and noise.shape[0] == 2


def test_class3d_startup_noise_takes_unsplit_order(monkeypatch):
    rows = np.array([2, 0, 1], dtype=np.int64)
    seen = {}

    def compute(ds, **kwargs):
        seen.update(kwargs)
        return np.array([[.5, .4, .3, .2, .1]], dtype=np.float64)

    monkeypatch.setattr(startup_noise, 'estimate_startup_sigma2', compute)
    startup_noise.prepare_startup_noise(
        SimpleNamespace(grid_size=8), source_rows=rows, optics_group_ids=np.ones(3, dtype=np.int64),
        mask_params=(12., 3), optics_pixel_sizes=np.array([1.25]))
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


@pytest.mark.parametrize("source", ["fresh", "replay", "missing_half_sets", "loaded", "frozen", "model"])
def test_cli_selects_noise_source_before_image_estimation(source, monkeypatch):
    tree = ast.parse(driver.Path(driver.__file__).read_text())
    call_name = "startup_noise.prepare_startup_noise"
    block = next(n for n in ast.walk(tree)
                 if isinstance(n, ast.If) and ast.unparse(n.test) == "frozen_boundary is not None"
                 and any(isinstance(c, ast.Call) and ast.unparse(c.func) == call_name for c in ast.walk(n)))
    calls = []
    spectrum = np.arange(1, 6, dtype=np.float64)
    pixels = np.ones(64, dtype=np.float32)

    def prepare(dataset, **kwargs):
        calls.append(kwargs)
        assert kwargs["output_dtype"] == np.float32
        assert kwargs["pair_counting"] == "once"  # the command's consistency option reaches the estimate
        return startup_noise.StartupNoise(radial=spectrum, pixel_variance=pixels)

    monkeypatch.setattr(startup_noise, "prepare_startup_noise", prepare)
    args = _args(relion_half_sets=None if source == "missing_half_sets" else "particles.star",
                 init_noise_from_npz="noise.npz" if source == "loaded" else None,
                 relion_init_dir="run" if source == "model" else None,
                 perturb_replay_relion_dir="replay" if source == "replay" else None)
    frozen = SimpleNamespace(noise_radial_per_half=[spectrum, spectrum], source_dir="frozen") if source == "frozen" else None
    namespace = dict(vars(driver))
    namespace.update(args=args, frozen_boundary=frozen, resume_snapshot=None,
        ds=SimpleNamespace(grid_size=8, image_shape=(8, 8)),
        relion_fresh_initial_noise_source_rows=np.arange(3),
        relion_fresh_initial_noise_optics_group_ids=np.ones(3),
        relion_mask_params=(12., 3), relion_optics_pixel_sizes=np.array([1.25]),
        class3d_noise_optics_pixel_sizes=None, _double_image_preprocessing=False,
        consistency_options=SimpleNamespace(initial_noise_pair_counting="once"),
        logger=logging.getLogger(__name__),
        frozen_boundary_cli=SimpleNamespace(expand_boundary_noise=lambda noise, shape: pixels),
        iteration_history=SimpleNamespace(_load_init_noise_radial_npz=lambda path, iteration:
                                        {"noise_radial": spectrum, "iteration": "000"}),
        recon_noise=SimpleNamespace(make_radial_noise=lambda noise, shape: pixels))
    args.init_noise_iter = "last"
    program = compile(ast.fix_missing_locations(ast.Module(body=[block], type_ignores=[])), driver.__file__, "exec")
    if source == "missing_half_sets":
        with pytest.raises(ValueError, match="start-up noise"):
            exec(program, namespace)
    else:
        exec(program, namespace)
        if source == "model":
            assert namespace["initial_noise_radial"] is None and namespace["noise_variance"] is None
        else:
            assert_matches(namespace["initial_noise_radial"], spectrum)
            assert_matches(namespace["noise_variance"], pixels)
    assert len(calls) == int(source in {"fresh", "replay"})
