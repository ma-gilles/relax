"""scripts/lib/highres_refinement.py reduces its translation priors with the library's log-sum-exp (deep2 DU-D10)."""

import numpy as np
import pytest

from relax.classification import k_class_results
from scripts.lib import highres_refinement

pytestmark = pytest.mark.unit


def _script_copy(values, axis):
    """The script's former private copy: float64 arithmetic, non-finite entries dropped."""
    values = np.asarray(values, dtype=np.float64)
    max_value = np.max(values, axis=axis, keepdims=True)
    safe_max = np.where(np.isfinite(max_value), max_value, 0.0)
    summed = np.sum(np.exp(np.where(np.isfinite(values), values - safe_max, -np.inf)), axis=axis)
    with np.errstate(divide="ignore"):
        return np.squeeze(safe_max, axis=axis) + np.log(summed)


def test_the_script_has_no_log_sum_exp_of_its_own():
    assert highres_refinement._logsumexp_np is k_class_results._logsumexp_np


def test_the_library_reduction_gives_the_scripts_numbers_for_log_priors():
    """Log priors are finite or -inf: on those, in float64, the two reductions agree to the last bit."""
    rng = np.random.default_rng(7)
    priors = rng.normal(scale=30.0, size=(5, 11)).astype(np.float32)
    priors[1, 2] = priors[:, 4] = -np.inf  # one excluded entry, one fully excluded column
    # As the script calls it: the float32 priors, reduced in float64.
    reduced = k_class_results._logsumexp_np(np.asarray(priors, dtype=np.float32).astype(np.float64), axis=0)
    expected = _script_copy(priors, axis=0)
    assert reduced.dtype == expected.dtype == np.float64
    assert np.array_equal(reduced, expected)
    assert reduced[4] == -np.inf and np.all(np.isfinite(np.delete(reduced, 4)))
    # Where they differ: the library does not drop a +inf entry.
    assert k_class_results._logsumexp_np(np.array([0.0, np.inf]), axis=0) == np.inf
