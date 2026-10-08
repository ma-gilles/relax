"""The offset prior is RELION's double-precision pdf_offset, stored as float32."""

from __future__ import annotations

import numpy as np
import pytest

from relax.helpers.orientation_priors import make_relion_translation_log_prior

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("seed", range(6))
def test_float32_offset_prior_is_relions_double_value_rounded_once(seed):
    # acc_ml_optimiser_impl.h:3060-3094: per axis (old + t - prior)^2 / (-2 sigma2) in double, times
    # pixel_size^2, stored as XFLOAT. One float32 rounding of that value is within 6e-8 of it per entry; a
    # float32 computation drifted up to 7.6e-6 on near-zero entries.
    rng = np.random.default_rng(seed)
    pixel = float(rng.choice([1.0, 1.35, 2.6, 3.4, 4.25, 0.885]))
    sigma = float(rng.uniform(2.0, 30.0))
    step = float(rng.choice([1.0, 2.0, 0.5]))
    grid = np.arange(-6.0, 6.0001, step)
    translations_px = np.stack(np.meshgrid(grid, grid, indexing="ij"), -1).reshape(-1, 2)
    raw = rng.uniform(-8.0, 8.0, (200, 2))
    old_px = np.where(raw > 0, np.trunc(raw + 0.5), np.trunc(raw - 0.5))
    offset = old_px[:, None, :] + translations_px[None] * pixel
    relion = (offset[..., 0] ** 2 / (-2.0 * sigma**2) + offset[..., 1] ** 2 / (-2.0 * sigma**2)) * pixel * pixel
    got = make_relion_translation_log_prior(translations_px, pixel, sigma, -old_px / pixel, dtype=np.float32)
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, relion, rtol=1e-6, atol=0.0)
