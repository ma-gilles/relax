"""Native-oracle coverage of the VDAM device projector adapter boundary."""


import numpy as np
import pytest
from helpers.vdam import relative_metrics

from relax.diagnostics.native_projector_setup import (
    native_reference_to_relion_projector_half_maps,
    native_reference_to_relion_projector_half_maps_and_power,
)
from relax.relion import relion_projector_setup
from relax.vdam import dense_adapter as adapter
from relax.vdam.init import initialise_denovo_state
from recovar.utils.helpers import recovar_volume_to_relion
from helpers.float_compare import assert_matches

pytestmark = pytest.mark.unit


def _assert_existing_consumer_policy(control1, candidate1, candidate2, control2):
    # Immutable static-key ABBA policy: control repeat <= 4eps-f32, paired
    # and candidate-repeat <= min(control repeat + 4eps-f32, 8eps-f32).
    floor = 4 * np.finfo(np.float32).eps
    repeat = relative_metrics(control1, control2)
    assert np.all(repeat <= floor)
    limit = np.nextafter(np.minimum(repeat + floor, 2 * floor), np.inf)
    for left, right in [(control1, candidate1), (control2, candidate2), (candidate1, candidate2)]:
        metrics = relative_metrics(left, right)
        assert np.all(metrics <= limit), metrics


@pytest.mark.requires_relion_bind
@pytest.mark.parametrize("size", [8, 16])
@pytest.mark.parametrize("padding", [1, 2])
@pytest.mark.parametrize("current", ["negative", "zero", "one", "partial_odd", "full", "oversize"])
def test_adapter_native_radius_layout_frame_and_consumer_policy(size, padding, current, monkeypatch):
    from relax.relion import relion_projector_setup as setup
    from relax.relion_bind import _relion_bind_core as bind

    current_size = {
        "negative": -1,
        "zero": 0,
        "one": 1,
        "partial_odd": size // 2 + 1,
        "full": size,
        "oversize": size + 4,
    }[current]
    references = np.random.default_rng(29).normal(size=(2, size, size, size))
    before = references.copy()
    original_setup = setup.setup_relion_projector_on_host
    raw = []

    def capture(*args, **kwargs):
        result = original_setup(*args, **kwargs)
        raw.append(result)
        return result

    monkeypatch.setattr(setup, "setup_relion_projector_on_host", capture)
    kwargs = dict(current_size=current_size, padding_factor=padding)
    controls = [
        native_reference_to_relion_projector_half_maps_and_power(references, **kwargs)
        for _ in range(2)
    ]
    candidates = [
        relion_projector_setup.reference_to_relion_projector_half_maps_and_power(references, **kwargs)
        for _ in range(2)
    ]
    assert len(raw) == 4  # The native controls never enter the device helper.
    assert candidates[0][2] == controls[0][2]
    assert candidates[0][0].shape == controls[0][0].shape
    assert candidates[0][0].dtype == np.complex64
    assert candidates[0][1].dtype == np.float64
    assert_matches(candidates[0][0] == 0, controls[0][0] == 0)
    for field in (0, 1):
        _assert_existing_consumer_policy(
            controls[0][field], candidates[0][field], candidates[1][field], controls[1][field]
        )
    # FP64 companion checks the same setup before the consumer cast, against
    # the real native oracle under its existing 1e-12 projector contract.
    for reference, (full, power) in zip(references, raw):
        native = bind.compute_fourier_transform_map(
            np.asarray(recovar_volume_to_relion(reference), np.float64),
            size,
            padding,
            1,
            current_size,
            True,
            2,
        )
        logical_size = native[0].shape[0]
        start = full.shape[0] // 2 - logical_size // 2
        cropped = full[start : start + logical_size, start : start + logical_size, : native[0].shape[2]]
        assert np.all(relative_metrics(native[0], cropped) < 1e-12)
        assert np.all(relative_metrics(native[1], power) < 1e-12)
    assert_matches(references, before)


@pytest.mark.parametrize("size,padding,interpolator", [(8, 1, 0), (8, 3, 1), (9, 1, 1)])
def test_unsupported_projector_geometry_is_refused(size, padding, interpolator):
    refs = np.random.default_rng(31).normal(size=(1, size, size, size))
    with pytest.raises(ValueError, match="projector setup"):
        relion_projector_setup.reference_to_relion_projector_half_maps_and_power(
            refs, current_size=size, padding_factor=padding, interpolator=interpolator
        )


@pytest.mark.requires_relion_bind
def test_vdam_config_uses_the_device_projector_state_default_size_and_dump(monkeypatch, tmp_path):
    state = initialise_denovo_state(ori_size=8, pixel_size=1.0, K=1, nr_iter=2, n_directions=3, pseudo_halfsets=True)
    state.Iref = np.random.default_rng(33).normal(size=state.Iref.shape)
    state.current_size = 0
    config = adapter.DenseInitialModelEstepConfig(
        noise_variance=np.ones(5),
        rotations=np.eye(3)[None],
        translations=np.zeros((1, 2)),
        relion_projector_frame=True,
    )
    assert not hasattr(config, "projector_setup_backend")

    def native_inputs():
        half, r_max = native_reference_to_relion_projector_half_maps(state.Iref, current_size=8, padding_factor=1)
        return adapter._finish_relion_projector_class_inputs(state, 1, half, r_max)

    natives = [native_inputs() for _ in range(2)]
    candidate2 = adapter._resolve_class_inputs(state, config)
    monkeypatch.setenv(adapter._RELION_PROJECTOR_DUMP_DIR_ENV, str(tmp_path))
    candidate = adapter._resolve_class_inputs(state, config)
    assert candidate[3] == natives[0][1] == 4
    assert candidate[2].dtype == np.complex64
    _assert_existing_consumer_policy(natives[0][0], candidate[2], candidate2[2], natives[1][0])
    with np.load(tmp_path / "iter000_relion_projector_half.npz") as dumped:
        assert_matches(dumped["projector_half"], candidate[2])
        assert int(dumped["current_size"]) == 8
    inputs, power = adapter.prepare_relion_projector_class_inputs_and_power(state, padding_factor=1)
    assert_matches(inputs[0], candidate[2])
    assert power.shape == (1, 5)


def test_float32_projector_route(monkeypatch):
    refs = np.random.default_rng(41).normal(size=(1, 8, 8, 8)).astype(np.float32)
    captured = []
    original = relion_projector_setup.setup_relion_projector_on_host

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(relion_projector_setup, "setup_relion_projector_on_host", capture)
    halves, power, radius = relion_projector_setup.reference_to_relion_projector_half_maps_and_power(
        refs, current_size=8, compute_dtype=np.float32,
    )
    assert radius == 4
    assert halves.dtype == np.complex64
    assert captured[0][0].dtype == np.complex64
    assert captured[0][1].dtype == np.float32
    assert power.dtype == np.float64  # tau2 host metadata keeps its precision.
    with pytest.raises(ValueError, match="trilinear"):
        relion_projector_setup.reference_to_relion_projector_half_maps_and_power(
            refs, current_size=8, interpolator=0, compute_dtype=np.float32,
        )
