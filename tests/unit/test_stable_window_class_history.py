"""A refinement's resident passes reuse a physical window class they already ran."""

import numpy as np
import pytest

from relax.helpers.fourier_window import make_stable_fourier_window_shape_plan
from relax.sparse_pass2.resident_pass2 import _stable_window_physical_class, stable_window_class_history

pytestmark = pytest.mark.unit


def test_a_new_class_reuses_a_run_class_at_most_one_quantum_larger():
    assert _stable_window_physical_class(128, 80, 8) is None  # outside a refinement
    with stable_window_class_history():
        # The 5k K=1 run's current sizes: 38, 58, 82, then 80, 82, 84.
        chosen = [_stable_window_physical_class(128, size, 8) for size in (38, 58, 82, 80, 82, 84, 64, 40)]
        assert chosen == [40, 64, 88, 88, 88, 88, 64, 40]
        # Another box or quantum keeps its own history.
        assert _stable_window_physical_class(256, 80, 8) == 80
        assert _stable_window_physical_class(128, 80, 16) == 80
    with stable_window_class_history():
        assert _stable_window_physical_class(128, 80, 8) == 80  # a new refinement starts empty


def test_an_explicit_physical_class_keeps_the_logical_window():
    image_shape, n_half = (64, 64), 64 * 33
    default = make_stable_fourier_window_shape_plan(image_shape, 40, n_half, enabled=True, quantum=8)
    larger = make_stable_fourier_window_shape_plan(
        image_shape, 40, n_half, enabled=True, quantum=8, physical_current_size=48
    )
    assert default.physical_current_size == 40
    assert larger.physical_current_size == larger.physical_reconstruction_current_size == 48
    np.testing.assert_array_equal(larger.logical_spec.score_indices_np, default.logical_spec.score_indices_np)
    np.testing.assert_array_equal(larger.logical_spec.recon_indices_np, default.logical_spec.recon_indices_np)
    assert larger.physical_score_pixels > default.physical_score_pixels
    for bad in (38, 45, 64):
        with pytest.raises(ValueError, match="explicit physical class"):
            make_stable_fourier_window_shape_plan(
                image_shape, 40, n_half, enabled=True, quantum=8, physical_current_size=bad
            )
    with pytest.raises(ValueError, match="explicit physical class"):
        make_stable_fourier_window_shape_plan(
            image_shape, 40, n_half, reconstruction_current_size=36, enabled=True, quantum=8,
            physical_current_size=48,
        )
