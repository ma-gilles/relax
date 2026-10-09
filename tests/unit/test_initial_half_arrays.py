"""Initial half/class layouts preserve canonical values and array ownership."""

import logging

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.refinement.half_inputs import HalfPair
from relax.refinement.mean_helpers import initialize_class_reference_model, initialize_reference_model
from relax.refinement.projector_preparation import (
    prepare_initial_real_references,
)

pytestmark = pytest.mark.unit
LOG = logging.getLogger(__name__)
SHAPE = (3, 3, 3)


@pytest.mark.parametrize("classes", [1, 2, 4])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("layout", ["shared", "pair"])
def test_real_reference_half_class_layout_and_aliases(classes, dtype, layout):
    source = (np.arange(classes * 27).reshape((classes,) + SHAPE) / 17).astype(dtype)
    if layout == "shared":
        value = source[0] if classes == 1 else source
        expected = [source, source]
    else:
        value = (source.copy(), -source)
        expected = list(value)
    result = prepare_initial_real_references(
        HalfPair.shared(value) if layout == "shared" else HalfPair(*value),
        volume_shape=SHAPE, n_classes=classes, init_relion_iteration=0, log=LOG,
    )
    for actual, wanted in zip(result, expected):
        assert actual.dtype == np.float64
        assert_matches(actual, wanted)
    if layout == "shared":
        assert result[0] is result[1]
    if dtype == np.float64:
        for index in range(2):
            origin = value[index] if layout == "pair" else value
            assert np.shares_memory(result[index], origin)


def test_absent_real_reference_keeps_fourier_fallback(caplog):
    result = prepare_initial_real_references(None, volume_shape=SHAPE, n_classes=4, init_relion_iteration=0, log=LOG)
    assert result == [None, None]
    assert not caplog.records


@pytest.mark.parametrize("classes", [1, 2])
def test_a_half_stacked_array_is_refused(classes):
    # The producers hand a HalfPair of one shared map or stack, or of each half's own; a half axis on an array
    # is not a layout any of them makes.
    value = HalfPair.shared(np.zeros((2,) + ((classes,) if classes > 1 else ()) + SHAPE))
    with pytest.raises(ValueError, match="init_reference_real must be"):
        prepare_initial_real_references(value, volume_shape=SHAPE, n_classes=classes, init_relion_iteration=0, log=LOG)


def test_real_reference_handoff_rejects_resumed_run():
    # A mid-trajectory replay (--init_relion_iteration 10 --firstiter_cc) used to take the
    # handoff and score its first iteration against the low-passed start-up map.
    value = HalfPair.shared(np.ones(SHAPE, dtype=np.float64))
    with pytest.raises(ValueError, match="--firstiter_cc"):
        prepare_initial_real_references(value, volume_shape=SHAPE, n_classes=1, init_relion_iteration=10, log=LOG)


def test_resumed_run_without_handoff_keeps_fourier_fallback():
    # Replays that never request the handoff (frozen boundaries, run_multi_iter_parity) stay valid.
    result = prepare_initial_real_references(None, volume_shape=SHAPE, n_classes=1, init_relion_iteration=10, log=LOG)
    assert result == [None, None]


@pytest.mark.parametrize(
    "value,classes",
    [
        (HalfPair.shared(np.zeros(27)), 1), (HalfPair.shared(np.zeros(SHAPE)), 4),
        (HalfPair(np.zeros(SHAPE), np.zeros(SHAPE)), 2),
    ],
)
def test_incompatible_real_reference_fails_without_broadcast(value, classes):
    with pytest.raises(ValueError, match="init_reference_real must be"):
        prepare_initial_real_references(value, volume_shape=SHAPE, n_classes=classes, init_relion_iteration=0, log=LOG)


@pytest.mark.parametrize("classes", [False, True])
def test_shared_tau2_keeps_original_object(classes):
    initial = jnp.asarray([1.0, 3.0], dtype=jnp.float32)
    model = (
        initialize_class_reference_model([None, None], initial, use_per_half_mean_variance=False)
        if classes
        else initialize_reference_model([None, None], initial, use_per_half_mean_variance=False, dtype=np.float32, log=LOG)
    )
    assert model.tau2 is initial and model.tau2_per_half is None
    assert model.half_tau2(0) is initial and model.half_tau2(1) is initial


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_half_tau2_values_and_existing_promoted_average(dtype):
    source = np.array([[1, 2**24, 3], [2**-20, -(2**24), 5]], dtype=dtype)
    initial = jnp.asarray(source)
    model = initialize_reference_model(
        [None, None], initial, use_per_half_mean_variance=True, dtype=np.float32, log=LOG
    )
    shared, halves = model.tau2, model.tau2_per_half
    expected = ((source[0].astype(np.float64) + source[1].astype(np.float64)) * 0.5).astype(np.float32)
    assert shared.dtype == np.float32
    assert_matches(shared, expected)
    for index, half in enumerate(halves):
        assert half.dtype == dtype
        assert_matches(half, source[index])


@pytest.mark.parametrize(
    "shape,kclass,message",
    [((2, 3), True, "only for K=1"), ((3,), False, "leading half axis 2"), ((3, 2), False, "leading half axis 2")],
)
def test_half_tau2_rejects_unsupported_inputs(shape, kclass, message):
    with pytest.raises(ValueError, match=message):
        if kclass:
            initialize_class_reference_model([None, None], jnp.ones(shape), use_per_half_mean_variance=True)
        else:
            initialize_reference_model([None, None], jnp.ones(shape), use_per_half_mean_variance=True, dtype=np.float32, log=LOG)
