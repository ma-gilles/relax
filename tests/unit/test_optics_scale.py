"""RELION S3b size and noise-shell rules for an optics group on another pixel size and box."""

import numpy as np
import pytest

from relax.helpers import optics_scale


@pytest.mark.unit
def test_sizes_follow_relion_formulas():
    # Group 2 of the S3b fixture: 112 px at 5.44 A against the 128 px, 4.25 A reference.
    s = optics_scale.scale_difference(112, 5.44, 128, 4.25)
    assert np.isclose(s, 112 * 5.44 / (128 * 4.25))
    assert optics_scale.group_current_size(64, 112, s) == 2 * int(np.ceil(0.5 * s * 64))
    assert optics_scale.group_current_size(128, 112, s) == 112  # never above the group's box
    current = optics_scale.group_current_size(64, 112, s)
    assert optics_scale.group_coarse_size(20.0, current, 112, s) == min(2 * int(np.ceil(s * 20.0)), current)


@pytest.mark.unit
def test_noise_shell_remap_both_ways():
    s = 1.25
    reference = np.arange(10.0) + 1.0
    group = optics_scale.group_noise_from_reference(reference, 14, s)
    expected_index = np.floor(np.arange(14) / s + 0.5).astype(int)
    for i, j in enumerate(expected_index):
        assert group[i] == (reference[j] if j < 10 else 0.0)
    sums = optics_scale.add_group_shells_to_reference(np.zeros(10), np.ones(14), s)
    np.testing.assert_array_equal(sums, np.bincount(expected_index[expected_index < 10], minlength=10))
