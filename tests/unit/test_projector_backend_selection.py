"""The projector setup has one implementation, the device transform; RELION's is the oracle.

The dtype the caller asks for is honoured, so refinement's complex128 slab is not
silently narrowed to the complex64 the InitialModel consumer takes.
"""

from __future__ import annotations

import numpy as np
import pytest

from relax.refinement.refinement_options import RefinementOptions


def _build(**kwargs):
    from relax.relion.relion_projector_setup import (
        reference_to_relion_projector_half_maps_and_power,
    )

    rng = np.random.default_rng(0)
    reference = rng.standard_normal((1, 32, 32, 32)).astype(np.float64)
    return reference_to_relion_projector_half_maps_and_power(reference, current_size=16, **kwargs)


def test_refinement_has_no_projector_backend_option():
    assert not hasattr(RefinementOptions(), "projector_setup_backend")


def test_setup_keeps_its_complex64_default():
    """The InitialModel consumer's behaviour must not move."""
    slab, _power, _r_max = _build(padding_factor=2)
    assert np.asarray(slab).dtype == np.complex64


def test_an_explicit_dtype_is_honoured():
    """Refinement asks for complex128 and must get it."""
    slab, _power, _r_max = _build(padding_factor=2, projector_data_dtype="complex128")
    assert np.asarray(slab).dtype == np.complex128


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"padding_factor": 3}, "padding factor"), ({"padding_factor": 2, "interpolator": 0}, "trilinear")],
)
def test_unsupported_geometry_is_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _build(**kwargs)


@pytest.mark.parametrize("padding_factor", [1, 2])
def test_setup_agrees_with_relion_to_double_precision(padding_factor):
    """Not bitwise across FFT implementations, but within a few ulp (RELION through the binding)."""
    pytest.importorskip("relax.relion_bind._relion_bind_core")
    from relax.diagnostics.native_projector_setup import native_reference_to_relion_projector_half_maps_and_power

    rng = np.random.default_rng(0)
    reference = rng.standard_normal((1, 32, 32, 32)).astype(np.float64)
    native, native_power, native_r = native_reference_to_relion_projector_half_maps_and_power(
        reference, current_size=16, padding_factor=padding_factor
    )
    ours, power, r_max = _build(padding_factor=padding_factor, projector_data_dtype="complex128")
    assert native.shape == ours.shape
    assert int(native_r) == int(r_max)
    scale = max(float(np.abs(native).max()), np.finfo(np.float64).tiny)
    assert float(np.abs(np.asarray(ours) - native).max()) / scale < 1e-12
    power_scale = max(float(np.abs(native_power).max()), np.finfo(np.float64).tiny)
    assert float(np.abs(np.asarray(power) - native_power).max()) / power_scale < 1e-11


def test_the_projector_cache_round_trips(tmp_path, monkeypatch):
    from relax.refinement.projector_preparation import _relion_projector_half_maps_for_scoring

    monkeypatch.setenv("RELAX_RELION_PROJECTOR_CACHE_DIR", str(tmp_path))
    rng = np.random.default_rng(1)
    reference = rng.standard_normal((1, 32, 32, 32)).astype(np.float64)
    # means_k is validated even when real_references supplies the volume.
    flat_ft = np.zeros((1, 32**3), dtype=np.complex128)
    kwargs = dict(volume_shape=(32, 32, 32), current_size=16, padding_factor=2, n_classes=1, real_references=reference)
    built, _r, _p = _relion_projector_half_maps_for_scoring(flat_ft, **kwargs)
    assert len(sorted(tmp_path.glob("projector_*.npz"))) == 1
    cached, _r, _p = _relion_projector_half_maps_for_scoring(flat_ft, **kwargs)
    np.testing.assert_array_equal(np.asarray(cached), np.asarray(built))
