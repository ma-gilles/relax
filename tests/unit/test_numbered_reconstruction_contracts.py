"""Mode operands and ordered postprocessing of numbered reference maps."""


import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.diagnostics import reconstruction as reconstruction_diagnostics
from relax.refinement import mean_helpers

pytestmark = pytest.mark.unit

DTYPES = pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["production-f32", "diagnostic-f64"])
POSTPROCESSING = pytest.mark.parametrize(
    ("first_cc", "flatten_solvent"),
    [(False, False), (True, True), (False, True), (True, False)],
    ids=["unmodified", "first-cc-solvent", "solvent-only", "first-cc-only"],
)
COMMON = dict(iteration=0, current_size=2, accumulator_volume_shape=(4, 4, 4))


def _settings(first_cc, flatten_solvent):
    # The low-pass resolution is a run setting; after iteration 1 it stays set
    # and only the per-iteration first-CC flag turns the filter off. The
    # settings turn float32 scalars into Python floats, so both operations
    # compute the mask radius in double precision.
    return mean_helpers.ReconstructionSettings(
        grid_size=2, voxel_size=np.float32(1.3), volume_shape=(2, 2, 2),
        padding_factor=2, projection_padding_factor=1, minres_map=0,
        width_mask_edge=5, fmask_edge=2, tau2_fudge=1,
        particle_diameter_angstrom=np.float32(3.7) if flatten_solvent else None,
        first_iteration_lowpass_angstrom=20 if first_cc or flatten_solvent else None,
        premask_dump_dir="captured-by-test",
    )


def _operands(count, dtype):
    numerators = [np.full(8, k + 1, dtype=np.complex64) for k in range(count)]
    denominators = [np.full(8, k + 2, dtype=dtype) for k in range(count)]
    priors = [np.full(2, k + 3, dtype=dtype) for k in range(count)]
    return numerators, denominators, priors


class _Recorder:
    """Fakes of the steps both operations share; each records its call in order."""

    def __init__(self, monkeypatch, *, dtype, solve_count, n_classes):
        self.events, self.solve_calls, self.masks, self.captures = [], [], [], []
        self.dtype = dtype

        def capture(value, **kwargs):
            assert len(self.solve_calls) == solve_count
            self.captures.append(value)
            assert kwargs["half_index"] == len(self.captures) - 1
            assert kwargs["n_classes"] == n_classes
            self.events.append("capture")

        def lowpass(value, *_args, **_kwargs):
            self.events.append("filter")
            return value * 2

        def mask(_shape, **kwargs):
            self.events.append("mask")
            self.masks.append(kwargs)
            assert kwargs["dtype"] == dtype
            return jnp.ones((2, 2, 2), dtype=dtype)

        def flatten(value, _mask, _shape, *, half_index):
            self.events.append("flatten")
            assert half_index == len(self.masks) - 1
            return value

        monkeypatch.setattr(mean_helpers, "_finish_host_staged_reconstruction", lambda value, *_args: value)
        monkeypatch.setattr(reconstruction_diagnostics, "write_premask_mean", capture)
        monkeypatch.setattr(mean_helpers, "_apply_relion_initial_lowpass_filter", lowpass)
        monkeypatch.setattr(mean_helpers, "_make_relion_solvent_mask", mask)
        monkeypatch.setattr(mean_helpers, "_apply_relion_solvent_flatten_k1", flatten)
        monkeypatch.setattr(mean_helpers, "_large_relion_solvent_mask_uses_compiled_builder", lambda _shape: False)
        monkeypatch.setattr(mean_helpers.gc, "collect", lambda: None)
        monkeypatch.setattr(
            mean_helpers.fourier_transform_utils, "get_idft3", lambda value: self.events.append("ifft") or value
        )
        monkeypatch.setattr(
            mean_helpers.fourier_transform_utils, "get_dft3", lambda value: self.events.append("fft") or value
        )

    def solved(self, numerator, kwargs):
        """Record one solve and return its map at the complex dtype of the run."""
        self.solve_calls.append(kwargs)
        self.events.append("solve")
        complex_dtype = np.complex64 if self.dtype == np.float32 else np.complex128
        return jnp.asarray(numerator, dtype=complex_dtype)


def _assert_solve_operands(kwargs, denominator, numerator, expected_denominator, expected_numerator, prior):
    assert denominator is expected_denominator
    assert numerator is expected_numerator
    assert kwargs["tau_is_1d"] is True
    assert kwargs["accumulator_volume_shape"] == (4, 4, 4)
    assert_matches(np.asarray(kwargs["tau"]), prior)


def _assert_results_and_mask_radius(result, expected, settings, masks, *, first_cc, flatten_solvent):
    for value, original in zip(result, expected, strict=True):
        assert_matches(np.asarray(value), np.asarray(original) * (2 if first_cc else 1))
    assert len(masks) == (2 if flatten_solvent else 0)
    assert type(settings.voxel_size) is float
    if flatten_solvent:
        assert type(settings.particle_diameter_angstrom) is float
        radius = float(np.float32(3.7)) / (2.0 * float(np.float32(1.3)))
        for kwargs in masks:
            assert type(kwargs["radius"]) is float
            assert kwargs["radius"] == radius
            assert kwargs["radius_p"] == radius + 5


@DTYPES
@POSTPROCESSING
def test_numbered_k1_halfmaps_preserve_operands_and_operation_order(monkeypatch, dtype, first_cc, flatten_solvent):
    settings = _settings(first_cc, flatten_solvent)
    numerators, denominators, priors = _operands(2, dtype)
    retained = object()
    record = _Recorder(monkeypatch, dtype=dtype, solve_count=2, n_classes=1)

    def solve(denominator, numerator, *args, **kwargs):
        index = len(record.solve_calls)
        _assert_solve_operands(kwargs, denominator, numerator, denominators[index], numerators[index], priors[index])
        assert kwargs["tau"].dtype == jnp.float64
        assert kwargs["preserve_output_precision"] is True
        assert kwargs.get("retained_device_numerator") is (retained if index == 0 else None)
        return record.solved(numerator, kwargs)

    monkeypatch.setattr(mean_helpers, "_reconstruct_volume_eager", solve)
    result = mean_helpers.reconstruct_numbered_k1_halfmaps(
        numerators, denominators, priors, settings, retained_first_numerator=retained,
        relion_firstiter_cc_this_iter=first_cc, **COMMON,
    )
    slot_events = ["capture"] + ["filter"] * first_cc + ["mask", "flatten"] * flatten_solvent
    assert record.events == ["solve"] * 2 + slot_events * 2
    _assert_results_and_mask_radius(
        result, numerators, settings, record.masks, first_cc=first_cc, flatten_solvent=flatten_solvent,
    )


@DTYPES
@POSTPROCESSING
def test_numbered_class_maps_preserve_operands_and_operation_order(monkeypatch, dtype, first_cc, flatten_solvent):
    n_classes = 4
    settings = _settings(first_cc, flatten_solvent)
    numerators, denominators, priors = _operands(n_classes, dtype)
    record = _Recorder(monkeypatch, dtype=dtype, solve_count=n_classes, n_classes=n_classes)

    def solve(denominator, numerator, *args, **kwargs):
        index = len(record.solve_calls)
        _assert_solve_operands(kwargs, denominator, numerator, denominators[index], numerators[index], priors[index])
        assert kwargs["tau"] is priors[index]
        assert "preserve_output_precision" not in kwargs
        assert "retained_device_numerator" not in kwargs
        return record.solved(numerator, kwargs)

    monkeypatch.setattr(mean_helpers, "_reconstruct_volume_eager", solve)
    result = mean_helpers.reconstruct_numbered_class_maps(
        numerators, denominators, priors, settings, n_classes=n_classes,
        relion_firstiter_cc_this_iter=first_cc, **COMMON,
    )
    slot_events = (["capture"] + ["filter"] * n_classes * first_cc
                   + (["mask"] + ["ifft", "fft"] * n_classes) * flatten_solvent)
    assert record.captures[0] is record.captures[1]
    assert (result[0] is result[1]) is (not (first_cc or flatten_solvent))
    assert record.events == ["solve"] * n_classes + slot_events * 2
    _assert_results_and_mask_radius(
        result, [np.stack(numerators)] * 2, settings, record.masks, first_cc=first_cc, flatten_solvent=flatten_solvent,
    )
