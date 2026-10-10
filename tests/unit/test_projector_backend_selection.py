"""The projector setup has one implementation, the device transform; RELION's is the oracle.

The dtype the caller asks for is honoured, so refinement's complex128 slab is not
silently narrowed to the complex64 the InitialModel consumer takes.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers.float_compare import assert_matches
from helpers.run_options import stand_in


def _build(**kwargs):
    from relax.relion.projector_setup import (
        reference_to_relion_projector_half_maps_and_power,
    )

    rng = np.random.default_rng(0)
    reference = rng.standard_normal((1, 32, 32, 32)).astype(np.float64)
    return reference_to_relion_projector_half_maps_and_power(reference, current_size=16, **kwargs)


def test_refinement_has_no_projector_backend_option():
    assert not hasattr(stand_in.options(), "projector_setup_backend")


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
    from scripts.lib.native_projector_setup import native_reference_to_relion_projector_half_maps_and_power

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
    from relax.refinement.projector_preparation import prepare_scoring_projector

    monkeypatch.setenv("RELAX_RELION_PROJECTOR_CACHE_DIR", str(tmp_path))
    rng = np.random.default_rng(1)
    reference = rng.standard_normal((1, 32, 32, 32)).astype(np.float64)
    # Fourier references are validated even when real_references supplies the volume.
    flat_ft = np.zeros((1, 32**3), dtype=np.complex128)
    kwargs = dict(volume_shape=(32, 32, 32), current_size=16, padding_factor=2, n_classes=1, real_references=reference)
    built = prepare_scoring_projector(flat_ft, **kwargs)
    assert len(sorted(tmp_path.glob("projector_*.npz"))) == 1
    cached = prepare_scoring_projector(flat_ft, **kwargs)
    assert_matches(cached.data, built.data)
    assert cached.r_max == built.r_max
    assert_matches(cached.power_spectrum, built.power_spectrum)


@pytest.mark.parametrize("current_size,cached_size", [(4, 4), (None, 8)])
def test_accuracy_projector_reuse_bypasses_conversion_cache_and_dump(
    monkeypatch, tmp_path, current_size, cached_size,
):
    from relax.refinement.projector_preparation import (
        PreparedProjector,
        ProjectorReuse,
        prepare_scoring_projector,
    )

    class UnconvertibleReference:
        def __array__(self, *args, **kwargs):
            pytest.fail("reused references were converted")

    references = UnconvertibleReference()
    projector = PreparedProjector(
        data=np.ones((1, 3, 3, 2), dtype=np.complex128),
        r_max=2,
        power_spectrum=np.ones((1, 3), dtype=np.float64),
    )
    reusable = ProjectorReuse(
        references=references, current_size=cached_size, image_box_size=8,
        projector=projector,
    )
    monkeypatch.setenv("RELAX_RELION_PROJECTOR_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("RELAX_RELION_PROJECTOR_DUMP_DIR", str(tmp_path / "dump"))
    result = prepare_scoring_projector(
        references, volume_shape=(4, 4, 4), current_size=current_size,
        padding_factor=2, n_classes=1, reusable=reusable,
        real_references=UnconvertibleReference(), dump_label="reuse",
    )

    assert result is projector
    assert result.power_spectrum is projector.power_spectrum
    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "dump").exists()


@pytest.mark.parametrize("n_classes", [1, 4])
@pytest.mark.parametrize(
    "same_reference,current_size,cached_size,transform_size",
    [(False, 4, 4, 4), (True, 2, 4, 2), (True, None, 4, 4)],
)
def test_accuracy_projector_reuse_misses_build_the_requested_transform(
    monkeypatch, n_classes, same_reference, current_size, cached_size, transform_size,
):
    import relax.relion.projector_setup as setup
    from relax.refinement.projector_preparation import (
        PreparedProjector,
        ProjectorReuse,
        prepare_scoring_projector,
    )

    monkeypatch.delenv("RELAX_RELION_PROJECTOR_CACHE_DIR", raising=False)
    monkeypatch.delenv("RELAX_RELION_PROJECTOR_DUMP_DIR", raising=False)
    references = np.zeros((n_classes, 4**3), dtype=np.complex64)
    real_references = np.arange(n_classes * 4**3, dtype=np.float64).reshape(n_classes, 4, 4, 4)
    cached = PreparedProjector(data=object(), r_max=2)
    reusable = ProjectorReuse(
        references=references if same_reference else references.copy(),
        current_size=cached_size, image_box_size=8, projector=cached,
    )
    calls = []

    def transform(real, *, current_size, padding_factor, projector_data_dtype, gridding_kernel):
        assert real is real_references and gridding_kernel == "radial"
        calls.append((current_size, padding_factor, projector_data_dtype))
        return (
            np.ones((n_classes, 3, 3, 2), dtype=np.complex128),
            np.ones((n_classes, 3), dtype=np.float64),
            current_size // 2,
        )

    monkeypatch.setattr(setup, "reference_to_relion_projector_half_maps_and_power", transform)
    result = prepare_scoring_projector(
        references, volume_shape=(4, 4, 4), current_size=current_size,
        padding_factor=2, n_classes=n_classes, reusable=reusable,
        real_references=real_references,
    )

    assert calls == [(transform_size, 2, "complex128")]
    assert result is not cached
    assert result.r_max == transform_size // 2
    assert result.data.dtype == np.complex128
    assert result.power_spectrum.dtype == np.float64
