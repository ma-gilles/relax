"""The truth mean of a fixture and the sign scripts/write_ppca_truth_mean.py gives it from the data."""

import json

import numpy as np
import pytest
from helpers.float_compare import assert_matches
from recovar.utils.helpers import write_mrc

from scripts import write_ppca_truth_mean as truth_mean

pytestmark = pytest.mark.unit


def test_the_truth_mean_weights_the_states_by_their_particle_counts(tmp_path):
    """Two 4-pixel states of 1 and 3 particles average to a quarter of the first plus three quarters of the second."""
    states = np.random.default_rng(3).normal(size=(2, 4, 4, 4)).astype(np.float32)
    for k, state in enumerate(states):
        write_mrc(str(tmp_path / f"state{k}.mrc"), state, voxel_size=2.0)
    (tmp_path / "manifest.json").write_text(json.dumps({"counts": [1, 3]}))
    assert_matches(truth_mean.truth_mean_volume(tmp_path), 0.25 * states[0] + 0.75 * states[1])


def test_the_band_correlation_is_plus_or_minus_one_for_equal_or_opposite_volumes_and_ignores_other_shells():
    """The correlation is taken over the shells up to the last one only."""
    rng = np.random.default_rng(4)
    a = rng.normal(size=12) + 1j * rng.normal(size=12)
    shells = np.repeat([0, 1, 2, 3], 3)
    assert truth_mean.band_correlation(a, a, shells, 3) == pytest.approx(1.0)
    assert truth_mean.band_correlation(-a, a, shells, 3) == pytest.approx(-1.0)
    other = a.copy()
    other[shells == 3] = 5 * np.conj(a[shells == 3])  # differs only in the last shell
    assert truth_mean.band_correlation(other, a, shells, 2) == pytest.approx(1.0)
    assert truth_mean.band_correlation(other, a, shells, 3) < 0.99


@pytest.mark.parametrize(("correlation", "sign"), [(0.96, 1), (-0.96, -1), (0.5, 1), (-0.5, -1)])
def test_the_sign_follows_a_strong_correlation(correlation, sign):
    """A reconstruction that agrees (or is opposite) gives the map its sign."""
    assert truth_mean.mean_sign(correlation) == sign


@pytest.mark.parametrize("correlation", [0.0, 0.3, -0.49, float("nan")])
def test_a_weak_or_missing_correlation_is_refused(correlation):
    """No sign is guessed when the particles and the maps do not line up."""
    with pytest.raises(ValueError, match="do not align"):
        truth_mean.mean_sign(correlation)
