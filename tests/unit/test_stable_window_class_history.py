"""A refinement's resident passes reuse a physical window class they already ran."""

import numpy as np
import pytest

from relax.fine_pass.resident_pass2 import _stable_window_physical_class, stable_window_class_history
from relax.fourier.fourier_window import (
    DEFAULT_STABLE_FOURIER_WINDOW_QUANTUM,
    STABLE_FOURIER_WINDOW_QUANTUM_ENV,
    VDAM_STABLE_FOURIER_WINDOW_QUANTUM,
    make_stable_fourier_window_shape_plan,
    stable_fourier_window_quantum,
)

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


def test_a_refinement_runs_on_its_own_quantum(monkeypatch):
    monkeypatch.delenv(STABLE_FOURIER_WINDOW_QUANTUM_ENV, raising=False)
    assert stable_fourier_window_quantum() == DEFAULT_STABLE_FOURIER_WINDOW_QUANTUM
    with stable_window_class_history(quantum=VDAM_STABLE_FOURIER_WINDOW_QUANTUM):
        quantum = stable_fourier_window_quantum()
        assert quantum == VDAM_STABLE_FOURIER_WINDOW_QUANTUM == 24
        # The 10k/256 VDAM ramp: 56 at iteration 1, then 30, 66, 82, 104, 114.
        chosen = [_stable_window_physical_class(256, size, quantum) for size in (56, 30, 66, 82, 104, 114)]
        assert chosen == [72, 72, 72, 96, 120, 120]
        monkeypatch.setenv(STABLE_FOURIER_WINDOW_QUANTUM_ENV, "16")
        assert stable_fourier_window_quantum() == 16  # the environment overrides
        monkeypatch.delenv(STABLE_FOURIER_WINDOW_QUANTUM_ENV)
    assert stable_fourier_window_quantum() == DEFAULT_STABLE_FOURIER_WINDOW_QUANTUM
    with stable_window_class_history():
        assert stable_fourier_window_quantum() == DEFAULT_STABLE_FOURIER_WINDOW_QUANTUM
    for bad in (0, 7):
        with pytest.raises(ValueError, match="even integer"):
            with stable_window_class_history(quantum=bad):
                pass


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
