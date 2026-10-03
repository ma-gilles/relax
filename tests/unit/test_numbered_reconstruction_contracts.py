"""Mode operands and ordered postprocessing of numbered reference maps."""


import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.diagnostics import reconstruction as reconstruction_diagnostics
from relax.refinement import mean_helpers

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("n_classes", [1, 4], ids=["half-maps", "class-stack"])
@pytest.mark.parametrize("tau_is_1d", [False, True], ids=["volume-prior", "shell-prior"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["production-f32", "diagnostic-f64"])
@pytest.mark.parametrize(
    ("first_cc", "flatten_solvent"),
    [(False, False), (True, True), (False, True), (True, False)],
    ids=["unmodified", "first-cc-solvent", "solvent-only", "first-cc-only"],
)
def test_numbered_reconstruction_preserves_mode_operands_and_operation_order(
    monkeypatch, n_classes, tau_is_1d, dtype, first_cc, flatten_solvent,
):
    # The low-pass resolution is a run setting; after iteration 1 it stays set
    # and only the per-iteration first-CC flag turns the filter off.
    settings = mean_helpers.ReconstructionSettings(
        grid_size=2, voxel_size=np.float32(1.3), volume_shape=(2, 2, 2),
        padding_factor=2, projection_padding_factor=1, minres_map=0,
        width_mask_edge=5, fmask_edge=2, tau2_fudge=1,
        particle_diameter_angstrom=np.float32(3.7) if flatten_solvent else None,
        first_iteration_lowpass_angstrom=20 if first_cc or flatten_solvent else None,
    )
    count = 2 if n_classes == 1 else n_classes
    numerators = [np.full(8, k + 1, dtype=np.complex64) for k in range(count)]
    denominators = [np.full(8, k + 2, dtype=dtype) for k in range(count)]
    priors = [np.full(2 if tau_is_1d else 8, k + 3, dtype=dtype) for k in range(count)]
    retained = object()
    events, solve_calls, masks, captures = [], [], [], []

    def solve(denominator, numerator, *args, **kwargs):
        index = len(solve_calls)
        assert denominator is denominators[index]
        assert numerator is numerators[index]
        assert kwargs["tau_is_1d"] is tau_is_1d
        assert kwargs["accumulator_volume_shape"] == (4, 4, 4)
        assert_matches(np.asarray(kwargs["tau"]), priors[index])
        if n_classes == 1:
            assert kwargs["tau"].dtype == jnp.float64
            assert kwargs["preserve_output_precision"] is True
            assert kwargs.get("retained_device_numerator") is (retained if index == 0 else None)
        else:
            assert kwargs["tau"] is priors[index]
            assert "preserve_output_precision" not in kwargs
            assert "retained_device_numerator" not in kwargs
        solve_calls.append(kwargs)
        events.append("solve")
        complex_dtype = np.complex64 if dtype == np.float32 else np.complex128
        return jnp.asarray(numerator, dtype=complex_dtype)

    def capture(value, **kwargs):
        assert len(solve_calls) == count
        captures.append(value)
        assert kwargs["half_index"] == len(captures) - 1
        assert kwargs["n_classes"] == n_classes
        events.append("capture")

    def lowpass(value, *_args, **_kwargs):
        events.append("filter")
        return value * 2

    def mask(_shape, **kwargs):
        events.append("mask")
        masks.append(kwargs)
        assert kwargs["dtype"] == dtype
        return jnp.ones((2, 2, 2), dtype=dtype)

    def flatten(value, _mask, _shape, *, half_index):
        events.append("flatten")
        assert half_index == len(masks) - 1
        return value

    monkeypatch.setenv("RELAX_PREMASK_DUMP_DIR", "captured-by-test")
    monkeypatch.setattr(mean_helpers, "_reconstruct_volume_eager", solve)
    monkeypatch.setattr(mean_helpers, "_finish_host_staged_reconstruction", lambda value, *_args: value)
    monkeypatch.setattr(reconstruction_diagnostics, "write_premask_mean", capture)
    monkeypatch.setattr(mean_helpers, "_apply_relion_initial_lowpass_filter", lowpass)
    monkeypatch.setattr(mean_helpers, "_make_relion_solvent_mask", mask)
    monkeypatch.setattr(mean_helpers, "_apply_relion_solvent_flatten_k1", flatten)
    monkeypatch.setattr(mean_helpers, "_large_relion_solvent_mask_uses_compiled_builder", lambda _shape: False)
    monkeypatch.setattr(mean_helpers.gc, "collect", lambda: None)
    monkeypatch.setattr(mean_helpers.fourier_transform_utils, "get_idft3", lambda value: events.append("ifft") or value)
    monkeypatch.setattr(mean_helpers.fourier_transform_utils, "get_dft3", lambda value: events.append("fft") or value)
    common = dict(iteration=0, current_size=2, accumulator_volume_shape=(4, 4, 4),
                  tau_is_1d=tau_is_1d, relion_firstiter_cc_this_iter=first_cc)
    if n_classes == 1:
        result = mean_helpers.reconstruct_numbered_k1_halfmaps(
            numerators, denominators, priors, settings, retained_first_numerator=retained, **common,
        )
        slot_events = ["capture"] + ["filter"] * first_cc + ["mask", "flatten"] * flatten_solvent
        expected = numerators
    else:
        result = mean_helpers.reconstruct_numbered_class_maps(
            numerators, denominators, priors, settings, n_classes=n_classes, **common,
        )
        slot_events = (["capture"] + ["filter"] * n_classes * first_cc
                       + (["mask"] + ["ifft", "fft"] * n_classes) * flatten_solvent)
        expected = [np.stack(numerators)] * 2
        assert captures[0] is captures[1]
        assert (result[0] is result[1]) is (not (first_cc or flatten_solvent))
    assert events == ["solve"] * count + slot_events * 2
    for value, original in zip(result, expected, strict=True):
        assert_matches(np.asarray(value), np.asarray(original) * (2 if first_cc else 1))
    assert len(masks) == (2 if flatten_solvent else 0)
    if flatten_solvent:
        radius = (float(settings.particle_diameter_angstrom) / (2.0 * float(settings.voxel_size))
                  if n_classes == 1 else settings.particle_diameter_angstrom / (2.0 * settings.voxel_size))
        for kwargs in masks:
            assert type(kwargs["radius"]) is type(radius)
            assert_matches(kwargs["radius"], radius)
