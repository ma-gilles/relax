"""--offset_prior_units: RELION's Angstrom/pixel mix of the score offset prior and the consistent Gaussian."""

from __future__ import annotations

import numpy as np
import pytest

from helpers.float_compare import assert_matches
from relax.sampling.orientation_priors import (
    make_relion_translation_log_prior,
    pdf_offset_score_sigma,
    relion_translation_prior_center,
)

pytestmark = pytest.mark.unit


def test_consistent_offset_prior_is_a_gaussian_of_sigma_angstrom_on_the_whole_offset():
    pixel, sigma = 4.25, 12.0
    previous = np.array([[3.0, -2.0]])
    translations = np.array([[0.0, 0.0], [1.0, 0.0], [-2.0, 1.0]])
    center = relion_translation_prior_center(previous, pixel, dtype=np.float64, consistent_units=True)
    log_prior = make_relion_translation_log_prior(
        translations,
        pixel,
        pdf_offset_score_sigma(sigma, pixel, consistent_units=True),
        center,
        dtype=np.float64,
    )[0]
    offsets_angstrom = (previous + translations) * pixel
    expected = -0.5 * np.sum(offsets_angstrom**2, axis=-1) / sigma**2
    assert_matches(log_prior - log_prior[0], expected - expected[0])


def test_relion_offset_prior_units_are_unchanged():
    pixel, sigma = 4.25, 12.0
    previous = np.array([[3.0, -2.0]])
    assert pdf_offset_score_sigma(sigma, pixel, consistent_units=False) == sigma
    assert_matches(
        relion_translation_prior_center(previous, pixel, dtype=np.float64),
        -previous / pixel,
    )
