import pytest

from relax.ppca_initial_model.vdam_controls import VdamPilotControls
from relax.vdam.native_options import NativeInitialModelOptions
from relax.vdam.schedules import default_subset_sizes_for_3d_initial_model

pytestmark = pytest.mark.unit


def test_unset_controls_are_native_vdam():
    assert VdamPilotControls.from_values() is None
    assert VdamPilotControls.from_values(stop_file="stop") == VdamPilotControls(stop_file="stop")


def test_invalid_caps_are_rejected():
    with pytest.raises(ValueError, match="stochastic_batch_size"):
        VdamPilotControls(stochastic_batch_size=0)
    with pytest.raises(ValueError, match="max_fourier_radius"):
        VdamPilotControls(max_fourier_radius=0)
    with pytest.raises(ValueError, match="max_healpix_order"):
        NativeInitialModelOptions(
            fn_img="particles.star",
            outputname="out/run",
            healpix_order=1,
            pilot_controls=VdamPilotControls(max_healpix_order=0),
        ).validate_run()


def test_fixed_subset_replaces_both_gradient_subsets():
    assert VdamPilotControls(stochastic_batch_size=200).subset_sizes(5000) == (200, 200)
    assert VdamPilotControls(stochastic_batch_size=200).subset_sizes(150) == (150, 150)
    assert VdamPilotControls(max_fourier_radius=8).subset_sizes(5000) == (
        default_subset_sizes_for_3d_initial_model(5000)
    )


def test_caps_and_stop_file(tmp_path):
    assert VdamPilotControls(max_fourier_radius=8).cap_current_size(60) == 16
    assert VdamPilotControls().cap_current_size(60) == 60
    controls = VdamPilotControls(stop_file=str(tmp_path / "stop"))
    assert not controls.stop_requested()
    controls.check_completed(3, 10)
    (tmp_path / "stop").touch()
    assert controls.stop_requested()
    controls.check_completed(10, 10)
    with pytest.raises(RuntimeError, match="saved iteration 3 before final output"):
        controls.check_completed(3, 10)


def test_vdam_batch_step_factor():
    from relax.ppca_initial_model.config import Config
    from relax.vdam.schedules import (
        DEFAULT_GRAD_EM_ITERS,
        compute_phase_lengths,
        compute_subset_size,
        default_subset_sizes_for_3d_initial_model,
    )

    # VDAM's own schedule: the factor is exactly one on every iteration.
    native = Config()
    assert all(native.step_factor(it, n) == 1.0 for n in (399, 20_000, 100_000) for it in range(1, 201))
    # A fixed batch below VDAM's subset scales the step by its share of the scheduled subset.
    pinned = Config(iterations=6000, stochastic_batch_size=300, stochastic_all_iterations=True)
    phases = compute_phase_lengths(6000)
    first, last = default_subset_sizes_for_3d_initial_model(100_000)
    scheduled = compute_subset_size(4001, phases, first, last, 100_000, 6000, grad_em_iters=DEFAULT_GRAD_EM_ITERS)
    assert pinned.step_factor(4001, 100_000) == pytest.approx(300 / scheduled)
    assert pinned.step_factor(6000, 100_000) == pytest.approx(300 / 100_000)
    # The final all-particle update keeps VDAM's step; a batch above the subset never raises it.
    assert Config(iterations=6000, stochastic_batch_size=300).step_factor(6000, 100_000) == 1.0
    assert Config(iterations=6000, stochastic_batch_size=50_000).step_factor(4001, 100_000) == 1.0


def test_vdam_refuses_iteration_counts_with_a_nan_tau2_fudge():
    import numpy as np

    from relax.ppca_initial_model.config import Config

    # RELION's tau2-fudge sigmoid length grad_inbetween_iter // 4 is 0 for 4 and 5 iterations, so the fudge at
    # iteration 1 is NaN and every VDAM gate (rho = 2 * fudge * signal / disagreement) would be NaN.
    for iterations in (4, 5):
        with pytest.raises(
            ValueError, match=r"tau2-fudge schedule for \d+ iterations is not finite at iteration\(s\) \[1\]"
        ):
            Config(iterations=iterations)
        # Momentum SGD does not read the fudge.
        Config(iterations=iterations, optimizer="momentum_sgd")
    # Every accepted VDAM count schedules a finite fudge on every iteration.
    for iterations in (1, 2, 3, 6, 7, 50, 200):
        config = Config(iterations=iterations)
        assert all(np.isfinite(config.schedule(it, 2000)[2]) for it in range(1, iterations + 1))
