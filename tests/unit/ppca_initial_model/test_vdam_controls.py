import pytest

from relax.ppca_initial_model.vdam_controls import VdamPilotControls
from relax.vdam import dense_adapter
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


def test_native_projector_setup_stays_float64():
    assert dense_adapter._projector_setup_dtype("native", "float32") == "float64"
    assert dense_adapter._projector_setup_dtype("jax", "float32") == "float32"
    assert dense_adapter._projector_setup_dtype("jax", "float64") == "float64"
