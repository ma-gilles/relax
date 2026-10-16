"""A refinement's resident passes reuse a physical window class they already ran."""

import numpy as np
import pytest

from relax.fine_pass.resident_stable_windows import _stable_window_physical_class, stable_window_class_history
from relax.fourier.fourier_window import make_stable_fourier_window_shape_plan
from relax.runtime.engine_tuning import (
    DEFAULT_STABLE_FOURIER_WINDOW_QUANTUM,
    VDAM_STABLE_FOURIER_WINDOW_QUANTUM,
    EngineTuning,
)
from relax.vdam.native_options import VdamEnvironment

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


def test_a_size_below_the_box_never_reuses_the_box_class():
    """A final pass at the box records the box class through its reference-model cube (images on another
    grid); a pass at 126 of the 128 box then keeps its own class 126 (EMPIAR-10299 final iteration, relax#95)."""

    with stable_window_class_history():
        assert _stable_window_physical_class(128, 128, 8) == 128
        assert _stable_window_physical_class(128, 126, 8) == 126
        assert _stable_window_physical_class(128, 122, 8) == 126
        make_stable_fourier_window_shape_plan(
            (128, 128), 126, 128 * 65, enabled=True, quantum=8, physical_current_size=126
        )


def test_a_refinement_runs_on_its_own_quantum(monkeypatch):
    """Each command resolves its own quantum once, where it reads the environment: Class3D and auto-refine 8,
    VDAM 24; the passes receive it (EngineTuning.stable_window_quantum)."""
    from relax.runtime.engine_tuning import STABLE_FOURIER_WINDOW_QUANTUM_ENV

    monkeypatch.delenv(STABLE_FOURIER_WINDOW_QUANTUM_ENV, raising=False)
    assert EngineTuning().stable_window_quantum == EngineTuning.from_environ().stable_window_quantum
    assert EngineTuning.from_environ().stable_window_quantum == DEFAULT_STABLE_FOURIER_WINDOW_QUANTUM == 8
    quantum = VdamEnvironment.from_environ().engine.stable_window_quantum
    assert quantum == VdamEnvironment().engine.stable_window_quantum == VDAM_STABLE_FOURIER_WINDOW_QUANTUM == 24
    with stable_window_class_history():
        # The 10k/256 VDAM ramp: 56 at iteration 1, then 30, 66, 82, 104, 114.
        chosen = [_stable_window_physical_class(256, size, quantum) for size in (56, 30, 66, 82, 104, 114)]
        assert chosen == [72, 72, 72, 96, 120, 120]


def test_the_quantum_override_is_read_once_at_the_boundary(monkeypatch):
    """RELAX_RELION_VDAM_STABLE_FOURIER_WINDOW_QUANTUM: unset or blank is the command's own quantum; a set value
    replaces it for every command; odd, small and text values are refused, as before."""
    from relax.runtime.engine_tuning import STABLE_FOURIER_WINDOW_QUANTUM_ENV

    for raw in (None, " "):
        if raw is None:
            monkeypatch.delenv(STABLE_FOURIER_WINDOW_QUANTUM_ENV, raising=False)
        else:
            monkeypatch.setenv(STABLE_FOURIER_WINDOW_QUANTUM_ENV, raw)
        assert EngineTuning.from_environ().stable_window_quantum == DEFAULT_STABLE_FOURIER_WINDOW_QUANTUM
        assert EngineTuning.from_environ(stable_window_quantum=24).stable_window_quantum == 24
    monkeypatch.setenv(STABLE_FOURIER_WINDOW_QUANTUM_ENV, " 16 ")
    assert EngineTuning.from_environ().stable_window_quantum == 16
    assert VdamEnvironment.from_environ().engine.stable_window_quantum == 16
    for bad in ("7", "0", "eight"):
        monkeypatch.setenv(STABLE_FOURIER_WINDOW_QUANTUM_ENV, bad)
        with pytest.raises(ValueError, match="must be an even integer >= 2"):
            EngineTuning.from_environ()


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
