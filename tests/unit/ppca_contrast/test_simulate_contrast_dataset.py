"""The contrast stream of the contrast-spread dataset simulator."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "ppca_contrast"))
from simulate_contrast_dataset import draw_contrast  # noqa: E402

pytestmark = pytest.mark.unit


def test_contrast_is_reproducible_truncated_and_centred():
    first = draw_contrast(20_000, 0.3, 0.2, seed=5)
    again = draw_contrast(20_000, 0.3, 0.2, seed=5)
    np.testing.assert_array_equal(first, again)
    assert first.dtype == np.float32 and first.min() >= 0.2
    assert abs(first.mean() - 1.0) < 0.01 and abs(first.std() - 0.3) < 0.01


def test_zero_spread_is_exactly_unit_contrast():
    np.testing.assert_array_equal(draw_contrast(100, 0.0, 0.2, seed=1), np.ones(100, np.float32))


def test_lognormal_contrast_has_unit_mean_the_requested_sd_and_is_skewed():
    contrast = draw_contrast(200_000, 0.3, 0.2, seed=2, distribution="lognormal")
    assert contrast.min() > 0 and abs(contrast.mean() - 1.0) < 0.005 and abs(contrast.std() - 0.3) < 0.005
    assert np.mean((contrast - contrast.mean()) ** 3) > 0  # right-skewed
    with pytest.raises(ValueError, match="unknown contrast distribution"):
        draw_contrast(10, 0.3, 0.2, seed=1, distribution="uniform")
