"""RELION's per-particle dLL (relax.relion.log_likelihood), against a hand-built case (relax#88)."""

import numpy as np
import pytest

from relax.relion.ctf import fftw_shell_labels
from relax.relion.log_likelihood import (
    relion_firstiter_cc_log_likelihood_contributions,
    relion_log_likelihood_contributions,
)

pytestmark = pytest.mark.unit


def test_dll_subtracts_logsigma2_over_the_current_size_pixels_once_per_image():
    """``log(sum_weight) - min_diff2 - nr_images * sum log(2 pi sigma2[ires])`` over the FFTW half pixels of the
    current size with ires > 0 (acc_ml_optimiser_impl.h:3555-3575)."""

    box, current = 16, 10
    sigma2 = np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]])
    shells = fftw_shell_labels(box, current, centered_rows=False)
    shells = shells[shells > 0]
    expected_logsigma2 = [float(np.sum(np.log(2 * np.pi * sigma2[g][shells]))) for g in (0, 1)]
    evidence = np.array([100.0, 50.0, -20.0])
    groups = np.array([0, 1, 0])
    n_images = np.array([1, 1, 3])
    got = relion_log_likelihood_contributions(
        evidence, sigma2_noise=sigma2, groups=groups, n_images=n_images, box_size=box, current_size=current
    )
    want = evidence - n_images * np.asarray(expected_logsigma2)[groups]
    np.testing.assert_allclose(got, want, rtol=1e-14)


def test_cross_correlation_dll_is_minus_min_diff2_per_image():
    np.testing.assert_array_equal(
        relion_firstiter_cc_log_likelihood_contributions(np.array([6.0, -3.0]), np.array([1, 3])), [6.0, -1.0]
    )
