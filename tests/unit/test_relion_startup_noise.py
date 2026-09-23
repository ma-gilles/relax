"""RELION's start-up noise estimate is the only initial-noise estimator of a refinement."""

import argparse
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_full_refinement as driver

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('option', ['--initial-noise-bootstrap', '--initial_noise_cache_dir'])
def test_cli_has_no_other_noise_estimator(monkeypatch, capsys, option):
    monkeypatch.setattr(sys, 'argv', ['run_full_refinement.py', option, 'pipeline'])
    with pytest.raises(SystemExit):
        driver._parse_args()
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
    monkeypatch.setattr(sys, 'argv', ['run_full_refinement.py'])
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
        np.testing.assert_array_equal(kwargs['source_rows'], rows)
        np.testing.assert_array_equal(kwargs['optics_group_ids'], optics)
        assert kwargs['particle_diameter_ang'] == 12
        assert kwargs['width_mask_edge_px'] == 3
        assert kwargs['image_pixel_size'] == 1.25
        calls.append(kwargs)
        return sigma

    monkeypatch.setattr(driver, '_compute_relion_fresh_k1_initial_sigma2', compute)
    radial, noise = driver._compute_relion_startup_noise(
        dataset, args=_args(), frozen_boundary=None, source_rows=rows,
        optics_group_ids=optics, mask_params=(12., 3),
        optics_pixel_sizes=np.array([1.25]))
    assert len(calls) == 1
    assert radial.dtype == np.float64 and noise.dtype == np.float32
    np.testing.assert_array_equal(radial, sigma[0] * 8**4)
    np.testing.assert_array_equal(noise, driver._relion_sigma2_to_native_noise_variance(
        sigma[0], grid_size=8, output_dtype=np.float32))


def test_startup_noise_float64_output_for_double_scoring(monkeypatch):
    sigma = np.array([[.5, .4, .3, .2, .1]], dtype=np.float64)
    monkeypatch.setattr(driver, '_compute_relion_fresh_k1_initial_sigma2', lambda ds, **kwargs: sigma)
    _radial, noise = driver._compute_relion_startup_noise(
        SimpleNamespace(grid_size=8), args=_args(), frozen_boundary=None,
        source_rows=np.arange(3), optics_group_ids=np.ones(3), mask_params=(12., 3),
        optics_pixel_sizes=np.array([1.25]), output_dtype=np.float64)
    assert noise.dtype == np.float64


@pytest.mark.parametrize('overrides', [
    dict(init_relion_iteration=1), dict(perturb_replay_relion_dir='replay'),
])
def test_startup_noise_starts_replays_without_their_model_noise(monkeypatch, overrides):
    # A replay injects RELION's model noise only from its first loaded state on;
    # the start of a fresh replay is RELION's estimate from the images.
    sigma = np.array([[.5, .4, .3, .2, .1]], dtype=np.float64)
    monkeypatch.setattr(driver, '_compute_relion_fresh_k1_initial_sigma2', lambda ds, **kwargs: sigma)
    radial, _noise = driver._compute_relion_startup_noise(
        SimpleNamespace(grid_size=8), args=_args(**overrides), frozen_boundary=None,
        source_rows=np.arange(3), optics_group_ids=np.ones(3), mask_params=(12., 3),
        optics_pixel_sizes=np.array([1.25]))
    np.testing.assert_array_equal(radial, sigma[0] * 8**4)


@pytest.mark.parametrize('overrides', [
    dict(init_noise_from_npz='noise.npz'), dict(relion_half_sets=None),
])
def test_startup_noise_rejects_loaded_noise_and_missing_half_sets(overrides):
    with pytest.raises(ValueError, match='start-up noise'):
        driver._compute_relion_startup_noise(SimpleNamespace(grid_size=8),
            args=_args(**overrides), frozen_boundary=None,
            source_rows=np.arange(3), optics_group_ids=np.ones(3), mask_params=(12., 3),
            optics_pixel_sizes=np.array([1.25]))


@pytest.mark.parametrize('overrides', [
    dict(frozen_boundary=object()), dict(source_rows=None),
    dict(optics_group_ids=None), dict(mask_params=None),
    dict(optics_group_ids=np.array([1, 2, 1])), dict(optics_pixel_sizes=None),
    dict(optics_pixel_sizes=np.array([1.25, 1.3])),
])
def test_startup_noise_rejects_missing_or_unsupported_inputs(overrides):
    params = dict(frozen_boundary=None, source_rows=np.arange(3),
                  optics_group_ids=np.ones(3), mask_params=(12., 3),
                  optics_pixel_sizes=np.array([1.25])) | overrides
    with pytest.raises(ValueError, match='start-up noise'):
        driver._compute_relion_startup_noise(SimpleNamespace(grid_size=8),
            args=_args(), **params)


def test_class3d_startup_noise_takes_unsplit_order(monkeypatch):
    rows = np.array([2, 0, 1], dtype=np.int64)
    seen = {}

    def compute(ds, **kwargs):
        seen.update(kwargs)
        return np.array([[.5, .4, .3, .2, .1]], dtype=np.float64)

    monkeypatch.setattr(driver, '_compute_relion_fresh_k1_initial_sigma2', compute)
    driver._compute_relion_startup_noise(
        SimpleNamespace(grid_size=8), args=_args(n_classes=4, relion_half_sets=None),
        frozen_boundary=None, source_rows=rows, optics_group_ids=np.ones(3, dtype=np.int64),
        mask_params=(12., 3), optics_pixel_sizes=np.array([1.25]))
    np.testing.assert_array_equal(seen['source_rows'], rows)


def test_class3d_noise_layout_is_the_micrograph_sorted_input_order():
    import pandas as pd

    # relion_refine stable-sorts rlnMicrographName byte-wise (exp_model.cpp:900-901);
    # Class3D keeps that order unsplit for the startup noise.
    particles = pd.DataFrame({
        'rlnImageName': ['1@s.mrcs', '2@s.mrcs', '3@s.mrcs', '4@s.mrcs'],
        'rlnMicrographName': ['2', '10', '1', '10'],
        'rlnOpticsGroup': [1, 1, 2, 1],
    })
    rows, optics = driver._relion_class3d_initial_noise_layout(particles)
    np.testing.assert_array_equal(rows, [2, 1, 3, 0])
    np.testing.assert_array_equal(optics, [2, 1, 1, 1])
