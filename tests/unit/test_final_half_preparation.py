"""Final priors retain their half and image-pixel frame."""

from dataclasses import replace

import numpy as np
import pytest
from helpers.float_compare import assert_matches
from test_numbered_expectation import numbered_inputs

from relax.refinement.finalization import prepare_final_half
from relax.refinement.trial_grids import FinalSampling, FinalSamplingSettings
from relax.sampling.orientation_priors import DirectionPrior

pytestmark = pytest.mark.unit


def final_inputs():
    half, phase, inputs = numbered_inputs()
    grid = phase.grid
    settings = FinalSamplingSettings(
        relion_iteration=4, grid_order=0, perturbation_order=0,
        translation_range=3., translation_step=1., pixel_size_angstrom=1.25,
        perturbation_factor=.5, perturbation=.125,
    )
    sampling = FinalSampling(settings=settings, base_rotations=grid.rotations,
                             base_translations=grid.translations, grid=grid)
    kwargs = dict(
        image_geometry=inputs['image_geometry'], sigma_offset_angstrom=2.,
        noise_radial=half.noise_radial, direction_prior=DirectionPrior(None, None),
        n_classes=1, use_local=False, coarse_angular_step_deg=None,
        particle_diameter_angstrom=3., sealed_sampling_state=None,
        symmetry='C1', dtype=np.float32, projection_scale=1.,
    )
    return half.particles, sampling, kwargs


@pytest.mark.parametrize('local', [False, True])
def test_final_cold_center_keeps_flat_prior_and_zero_engine_center(local):
    half, sampling, kwargs = final_inputs()
    kwargs.update(use_local=local, coarse_angular_step_deg=15. if local else None)
    prepared = prepare_final_half(half, sampling, **kwargs)
    assert prepared.translations.prior_center is None
    assert prepared.translations.local_prior_center is None
    assert_matches(prepared.translations.engine_prior_center, np.zeros(2, dtype=np.float32))
    assert_matches(prepared.translation_log_prior, np.zeros(2, dtype=np.float32))
    assert prepared.directions.rotation_log_prior is None
    assert prepared.directions.class_rotation_log_prior is None


def test_final_half_keeps_score_and_sigma_centers_in_their_distinct_frames():
    half, sampling, kwargs = final_inputs()
    stored = np.array([[.6, -1.2], [2.1, .7]], dtype=np.float64)
    half = replace(half, translations=stored)
    prepared = prepare_final_half(half, sampling, **kwargs)
    score_center = np.array([[-.8, .8], [-1.6, -.8]], dtype=np.float32)
    sigma_center = np.array([[-1., 1.], [-2., -1.]], dtype=np.float32)
    assert_matches(prepared.translations.prior_center, score_center)
    assert_matches(prepared.translations.local_prior_center, score_center)
    assert_matches(prepared.translations.engine_prior_center, sigma_center)
    assert prepared.translations.prior_center is not prepared.translations.local_prior_center
    assert prepared.translations.engine_prior_center is not prepared.translations.prior_center
    assert half.translations is stored
