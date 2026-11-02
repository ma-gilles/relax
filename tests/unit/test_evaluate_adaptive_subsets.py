"""The truth a subset reconstruction is scored against."""

import numpy as np

from scripts.evaluate_adaptive_subsets import amplitude_slope, inside_nyquist_sphere


def test_nyquist_sphere_keeps_what_a_reconstruction_can_hold_and_removes_the_corners():
    n = 32
    x, y, z = np.meshgrid(*[np.arange(n)] * 3, indexing="ij")
    inside = np.cos(2 * np.pi * 7 * x / n) + np.sin(2 * np.pi * (5 * y + 3 * z) / n)
    corner = np.cos(2 * np.pi * (14 * x + 14 * y + 14 * z) / n)
    np.testing.assert_allclose(inside_nyquist_sphere(inside + corner), inside, atol=1e-10)


def test_missing_corner_power_lowers_the_slope_by_its_share_of_the_power():
    n = 32
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(n, n, n))
    limited = inside_nyquist_sphere(truth)
    share = np.sum(limited**2) / np.sum(truth**2)
    np.testing.assert_allclose(amplitude_slope(limited, truth, np.ones_like(truth)), share)
    np.testing.assert_allclose(amplitude_slope(limited, limited, np.ones_like(truth)), 1.0)
