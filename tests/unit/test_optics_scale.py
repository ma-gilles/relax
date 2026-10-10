"""RELION S3b size and noise-shell rules for an optics group on another pixel size and box."""

import math

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.relion import optics_scale


# Moved from relax/relion/optics_scale.py (PLAN e1): no relax module uses it, only this test file.
def group_coarse_size(coarse_resolution_pixels, current_size_g, box_size, scale, max_coarse_size=None) -> int:
    """``image_coarse_size[g]`` for adaptive oversampling (``ml_optimiser.cpp:5761-5777``).

    ``coarse_resolution_pixels`` is ``pixel_size_ref * ori_size / coarse_resolution``, the
    reference-grid Fourier radius RELION's coarse pass needs; the group's size is that
    radius scaled by ``s_g``, capped by the scaled ``max_coarse_size`` and by the group's
    current size.
    """

    size = 2 * math.ceil(float(scale) * float(coarse_resolution_pixels))
    cap = int(box_size) if max_coarse_size is None or max_coarse_size <= 0 else int(float(scale) * max_coarse_size)
    return int(min(size, cap, int(current_size_g)))


@pytest.mark.unit
def test_sizes_follow_relion_formulas():
    # Group 2 of the S3b fixture: 112 px at 5.44 A against the 128 px, 4.25 A reference.
    s = optics_scale.scale_difference(112, 5.44, 128, 4.25)
    assert np.isclose(s, 112 * 5.44 / (128 * 4.25))
    assert optics_scale.group_current_size(64, 112, s) == 2 * int(np.ceil(0.5 * s * 64))
    assert optics_scale.group_current_size(128, 112, s) == 112  # never above the group's box
    current = optics_scale.group_current_size(64, 112, s)
    assert group_coarse_size(20.0, current, 112, s) == min(2 * int(np.ceil(s * 20.0)), current)


@pytest.mark.unit
def test_noise_shell_remap_both_ways():
    s = 1.25
    reference = np.arange(10.0) + 1.0
    group = optics_scale.group_noise_from_reference(reference, 14, s)
    expected_index = np.floor(np.arange(14) / s + 0.5).astype(int)
    for i, j in enumerate(expected_index):
        assert group[i] == (reference[j] if j < 10 else 0.0)
    sums = optics_scale.add_group_shells_to_reference(np.zeros(10), np.ones(14), s)
    assert_matches(sums, np.bincount(expected_index[expected_index < 10], minlength=10))
