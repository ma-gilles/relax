"""Real native-oracle tests of the shared fixed-capacity projector setup."""

import json

import jax
import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.relion.relion_projector_setup import setup_relion_projector

pytestmark = pytest.mark.unit


def _crop(projector, r_max, padding):
    size = 2 * (padding * r_max + 1) + 1
    start = projector.shape[0] // 2 - size // 2
    return projector[start : start + size, start : start + size, : size // 2 + 1]


def _relative_metrics(left, right):
    left, right = np.asarray(left, dtype=np.complex128), np.asarray(right, dtype=np.complex128)
    delta = left - right
    tiny = np.finfo(np.float64).tiny
    return {
        "relative_l2": float(
            np.linalg.norm(delta.ravel()) / max(np.linalg.norm(left.ravel()), np.linalg.norm(right.ravel()), tiny)
        ),
        "relative_max": float(np.max(np.abs(delta)) / max(np.max(np.abs(left)), np.max(np.abs(right)), tiny)),
        "relative_mean_abs": float(np.mean(np.abs(delta)) / max(np.mean(np.abs(left)), np.mean(np.abs(right)), tiny)),
    }


def _existing_float32_policy(control1, candidate1, candidate2, control2):
    # Existing static-key ABBA policy: control repeat <=4eps-f32; paired and
    # candidate-repeat metrics <=min(control repeat+4eps-f32,8eps-f32).
    # Reproduced here so this unit test does not depend on a scratch analyzer.
    floor = 4 * np.finfo(np.float32).eps
    control = _relative_metrics(control1, control2)
    assert all(value <= floor for value in control.values())
    panels = [
        _relative_metrics(control1, candidate1),
        _relative_metrics(control2, candidate2),
        _relative_metrics(candidate1, candidate2),
    ]
    for panel in panels:
        assert all(
            value <= np.nextafter(min(control[key] + floor, 2 * floor), np.inf) for key, value in panel.items()
        ), panel
    return panels


@pytest.mark.requires_relion_bind
@pytest.mark.parametrize("size", [8, 16])
@pytest.mark.parametrize("padding", [1, 2])
@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("gridding", [False, True])
def test_native_projector_and_power_fp64_and_consumer_cast(size, padding, full, gridding):
    from relax.relion_bind import _relion_bind_core as bind

    reference = np.random.default_rng(61).normal(size=(size,) * 3).astype(np.float64)
    before = reference.copy()
    radius = size // (2 if full else 4)
    controls = [
        bind.compute_fourier_transform_map(reference, size, padding, 1, 2 * radius, gridding, 2) for _ in range(2)
    ]
    candidates = [
        setup_relion_projector(reference, np.int32(radius), box_size=size, padding_factor=padding, do_gridding=gridding)
        for _ in range(2)
    ]
    records = {}
    for field in (0, 1):
        controls_field = [np.asarray(value[field]) for value in controls]
        candidates_field = [np.asarray(value[field]) for value in candidates]
        if field == 0:
            candidates_field = [_crop(value, radius, padding) for value in candidates_field]
            assert candidates_field[0].dtype == np.complex128
            assert_matches(candidates_field[0] == 0, controls_field[0] == 0)
        else:
            assert candidates_field[0].dtype == np.float64
        metrics = _relative_metrics(controls_field[0], candidates_field[0])
        # Existing native projector FP64 contract (test_e1_padding_parity.py).
        # More than five orders tighter than the four-eps-f32 consumer panel.
        assert all(value < 1e-12 for value in metrics.values()), metrics
        cast_dtype = np.complex64 if field == 0 else np.float32
        records[str(field)] = {
            "fp64": metrics,
            "cast_panel": _existing_float32_policy(
                controls_field[0].astype(cast_dtype),
                candidates_field[0].astype(cast_dtype),
                candidates_field[1].astype(cast_dtype),
                controls_field[1].astype(cast_dtype),
            ),
        }
    assert_matches(reference, before)
    print(json.dumps({"size": size, "padding": padding, "full": full, "gridding": gridding, "metrics": records}))


def test_radius_and_gridding_reuse_one_compiled_shape():
    setup_relion_projector.clear_cache()
    reference = np.ones((8,) * 3, dtype=np.float64)
    outputs = []
    for radius, corrected in [(0, True), (2, True), (4, False), (-1, False)]:
        result = setup_relion_projector(
            reference, np.int32(radius), box_size=8, padding_factor=2, do_gridding=corrected
        )
        jax.block_until_ready(result)
        outputs.append(result)
    assert setup_relion_projector._cache_size() == 1
    assert all(value[0].shape == (19, 19, 10) and value[1].shape == (5,) for value in outputs)
    assert_matches(outputs[2][0], outputs[3][0])


def test_corrected_projector_float32_keeps_compute_precision():
    reference = np.random.default_rng(73).normal(size=(8, 8, 8)).astype(np.float32)
    lowered = setup_relion_projector.lower(
        reference, np.int32(4), box_size=8, do_gridding=True,
        compute_dtype=np.float32,
    )
    assert "f64" not in str(lowered.compiler_ir(dialect="stablehlo"))
    projector, power = setup_relion_projector(
        reference, np.int32(4), box_size=8, do_gridding=True,
        compute_dtype=np.float32,
    )
    assert projector.dtype == np.complex64
    assert power.dtype == np.float32
    assert np.all(np.isfinite(np.asarray(projector)))
    assert np.all(np.isfinite(np.asarray(power)))


@pytest.mark.requires_relion_bind
def test_positive_nyquist_only_and_inclusive_sphere():
    from relax.relion_bind import _relion_bind_core as bind

    reference = np.zeros((8,) * 3, dtype=np.float64)
    reference[4, 4, 4] = 1.0
    candidate, power = setup_relion_projector(reference, np.int32(4), box_size=8, do_gridding=False)
    actual = np.asarray(candidate)
    native, native_power, *_ = bind.compute_fourier_transform_map(reference, 8, 1, 1, 8, False, 2)
    assert_matches(actual, native)
    np.testing.assert_allclose(power, native_power, rtol=1e-12, atol=0)
    center = actual.shape[0] // 2
    assert actual[center + 4, center, 0] != 0
    assert actual[center - 4, center, 0] == 0
    assert actual[center, center + 4, 0] != 0
    assert actual[center, center - 4, 0] == 0
    assert actual[center, center, 4] != 0
    assert actual[center + 4, center + 1, 0] == 0


def _float64_fft_bound(size, padding):
    """Derived float64 bound between two evaluations of the padded transform.

    Each evaluation is three one-dimensional FFTs of length M; a radix-2 FFT
    carries a relative error of about log2(M) eps64 per pass (Higham, Accuracy
    and Stability of Numerical Algorithms, 2nd ed., section 24.1), and the
    gridding division and pf^3 N scale each add one rounding. The difference of
    two such evaluations is bounded by the sum of both errors.
    """

    per_evaluation = (3 * np.log2(padding * size) + 2) * np.finfo(np.float64).eps
    return 2 * per_evaluation


def _float64_power_bound(size, padding):
    """The shell power: squares double the slab bound; shell sums over the capacity
    grid add the derived sqrt(n) eps64 accumulation-order term."""

    capacity = padding * size + 3
    voxels = capacity * capacity * (capacity // 2 + 1)
    return 2 * _float64_fft_bound(size, padding) + np.sqrt(voxels) * np.finfo(np.float64).eps


@pytest.mark.requires_relion_bind
@pytest.mark.parametrize("size", [8, 16])
@pytest.mark.parametrize("padding", [1, 2])
@pytest.mark.parametrize("full", [False, True])
def test_host_window_build_meets_the_derived_float64_bound(size, padding, full):
    """The host wrapper's build (window at r_max) against RELION's own computeFourierTransformMap."""

    from relax.relion.relion_projector_setup import setup_relion_projector_on_host
    from relax.relion_bind import _relion_bind_core as bind

    reference = np.random.default_rng(62).normal(size=(size,) * 3).astype(np.float64)
    radius = size // (2 if full else 4)
    native_slab, native_power, *_ = bind.compute_fourier_transform_map(
        reference, size, padding, 1, 2 * radius, True, 2
    )
    slab, power = setup_relion_projector_on_host(reference, radius, box_size=size, padding_factor=padding)
    assert slab.dtype == np.complex128 and slab.shape == np.asarray(native_slab).shape
    assert_matches(slab == 0, np.asarray(native_slab) == 0)
    bound = _float64_fft_bound(size, padding)
    for control, candidate, limit in (
        (native_slab, slab, bound),
        (native_power, power, _float64_power_bound(size, padding)),
    ):
        metrics = _relative_metrics(control, candidate)
        assert all(value < limit for value in metrics.values()), (metrics, limit)
    capacity_slab, capacity_power = setup_relion_projector(
        reference, np.int32(radius), box_size=size, padding_factor=padding
    )
    metrics = _relative_metrics(_crop(np.asarray(capacity_slab), radius, padding), slab)
    assert all(value < bound for value in metrics.values()), (metrics, bound)


def test_host_window_build_is_chunking_invariant():
    from relax.relion.relion_projector_setup import setup_relion_projector_on_host

    reference = np.random.default_rng(63).normal(size=(16,) * 3).astype(np.float64)
    whole = setup_relion_projector_on_host(reference, 6, box_size=16, padding_factor=2)
    chunked = setup_relion_projector_on_host(reference, 6, box_size=16, padding_factor=2, chunk_bytes=1)
    # Each one-dimensional FFT sees the same row, so the slab is unchanged; the
    # shell power is summed per chunk, in a different order.
    assert_matches(chunked[0], whole[0])
    assert all(value < _float64_power_bound(16, 2) for value in _relative_metrics(whole[1], chunked[1]).values())


@pytest.mark.requires_relion_bind
def test_jax_backend_builds_on_the_device_at_every_size(monkeypatch):
    """The JAX backend has one device build: the window core, dispatched chunk by chunk."""

    from recovar.utils.helpers import recovar_volume_to_relion

    from relax.relion import relion_projector_setup as setup
    from relax.relion_bind import _relion_bind_core as bind

    called = []
    real = setup.setup_relion_projector_on_host
    monkeypatch.setattr(setup, "setup_relion_projector_on_host", lambda *a, **k: called.append(1) or real(*a, **k))
    monkeypatch.setattr(setup, "_CHUNK_BYTES", 1)
    reference = np.random.default_rng(64).normal(size=(1, 16, 16, 16)).astype(np.float64)
    slab, power, r_max = setup.reference_to_relion_projector_half_maps_and_power(
        reference, current_size=12, padding_factor=2, projector_data_dtype="complex128"
    )
    assert called == [1]
    native, native_power, *_ = bind.compute_fourier_transform_map(
        np.asarray(recovar_volume_to_relion(reference[0]), dtype=np.float64), 16, 2, 1, 12, True, 2
    )
    assert int(r_max) == 6 and slab.dtype == np.complex128
    bound = _float64_fft_bound(16, 2)
    assert all(value < bound for value in _relative_metrics(native, slab[0]).values())
    assert all(value < _float64_power_bound(16, 2) for value in _relative_metrics(native_power, power[0]).values())


def test_host_build_reuses_programs_inside_a_stable_window_class():
    """Radii whose current sizes share a stable window class (quantum 8) share compiled programs."""

    from relax.relion import relion_projector_setup as setup

    reference = np.random.default_rng(65).normal(size=(32,) * 3).astype(np.float64)
    programs = (setup._transform_xy, setup._transform_z, setup._mask_and_shell_power)
    setup.setup_relion_projector_on_host(reference, 5, box_size=32, padding_factor=2)
    compiled = [program._cache_size() for program in programs]
    slab, power = setup.setup_relion_projector_on_host(reference, 6, box_size=32, padding_factor=2)
    assert [program._cache_size() for program in programs] == compiled
    # Current sizes 10 and 12 both run in class 16; the slab is still cropped to radius 6.
    assert slab.shape == (27, 27, 14)
    exact = setup._build_projector_window(
        setup.gridding_correct_volume_real(reference, 32, 2), 6, 32, 2, 6, to_host=True
    )
    assert_matches(slab, exact[0])
    assert all(value < _float64_power_bound(32, 2) for value in _relative_metrics(exact[1], power).values())
