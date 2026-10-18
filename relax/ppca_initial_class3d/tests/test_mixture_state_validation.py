"""State health checks reject corruption without host copies of model volumes."""

from copy import deepcopy
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from relax.ppca_initial_class3d import state as state_module
from relax.ppca_initial_class3d.config import Config
from relax.ppca_initial_class3d.state import State, validate_state
from relax.ppca_initial_model.update import empty_moments


def _valid_state():
    config = Config(n_classes=2, q=2, iterations=12)
    theta = jnp.ones((2, 32, 3), jnp.complex64)
    state = State(
        theta, tuple(empty_moments(model) for model in theta), jnp.ones((2, 5), jnp.float32),
        np.asarray([.3, .7], np.float32), 3, np.arange(8), {}, 2.0, 4, {},
        np.ones(12, np.float32), 0,
    )
    return state, config


def _with_moment(state, index=0, **changes):
    moments = list(state.moments)
    moments[index] = replace(moments[index], **changes)
    return replace(state, moments=tuple(moments))


def test_validation_transfers_only_reduced_flags_not_model_arrays(monkeypatch):
    state, config = _valid_state()
    validate_state(state, config)  # compile reductions before inspecting transfers
    large = [state.theta, state.noise]
    large.extend(array for moment in state.moments for array in (moment.first, moment.second))
    large_ids = {id(array) for array in large}
    original_asarray = np.asarray
    original_get = jax.device_get
    transfers = []

    def refuse_large_asarray(array, *args, **kwargs):
        assert id(array) not in large_ids, "validation copied a model-sized array to the host"
        return original_asarray(array, *args, **kwargs)

    def record_scalar_get(value):
        leaves = jax.tree_util.tree_leaves(value)
        for array in leaves:
            assert np.prod(array.shape, dtype=np.int64) <= 2
            assert array.dtype == jnp.bool_
        transfers.extend(leaves)
        return original_get(value)

    monkeypatch.setattr(state_module.np, "asarray", refuse_large_asarray)
    monkeypatch.setattr(state_module.jax, "device_get", record_scalar_get)
    validate_state(state, config)
    assert len(transfers) == config.n_classes + 2


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_validation_rejects_nonfinite_model_and_noise(value):
    state, config = _valid_state()
    with pytest.raises(ValueError, match="Nonfinite mixture PPCA model"):
        validate_state(replace(state, theta=state.theta.at[1, 0, 0].set(value)), config)
    with pytest.raises(ValueError, match="Mixture noise must be finite and positive"):
        validate_state(replace(state, noise=state.noise.at[0, 0].set(value)), config)


def test_validation_rejects_nonfinite_imaginary_model_component():
    state, config = _valid_state()
    value = jax.lax.complex(jnp.float32(1), jnp.float32(np.inf))
    with pytest.raises(ValueError, match="Nonfinite mixture PPCA model"):
        validate_state(replace(state, theta=state.theta.at[1, 0, 0].set(value)), config)


@pytest.mark.parametrize("field", ["first", "second"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_validation_rejects_nonfinite_optimizer_fields(field, value):
    state, config = _valid_state()
    array = getattr(state.moments[1], field)
    corrupt = array.at[(0,) * array.ndim].set(value)
    with pytest.raises(ValueError, match="Nonfinite mixture optimizer state"):
        validate_state(_with_moment(state, 1, **{field: corrupt}), config)


def test_validation_keeps_zero_second_moments_and_rejects_negative_values():
    state, config = _valid_state()
    validate_state(_with_moment(state, second=jnp.zeros_like(state.moments[0].second)), config)
    with pytest.raises(ValueError, match="Negative mixture second moments"):
        validate_state(_with_moment(state, second=state.moments[0].second.at[0, 0].set(-1)), config)


@pytest.mark.parametrize("value", [0, -1])
def test_validation_rejects_nonpositive_noise(value):
    state, config = _valid_state()
    with pytest.raises(ValueError, match="Mixture noise must be finite and positive"):
        validate_state(replace(state, noise=state.noise.at[0, 0].set(value)), config)


def test_validation_preserves_component_failure_order():
    state, config = _valid_state()
    state = _with_moment(state, first=state.moments[0].first.at[0, 0, 0].set(np.nan))
    state = _with_moment(state, 1, second=jnp.ones((1,), jnp.float32))
    with pytest.raises(ValueError, match="Nonfinite mixture optimizer state"):
        validate_state(state, config)


def test_validation_preserves_shape_and_dtype_checks():
    state, config = _valid_state()
    with pytest.raises(ValueError, match="Mixture checkpoint/model K or q mismatch"):
        validate_state(replace(state, theta=state.theta[:, :, :2]), config)
    with pytest.raises(TypeError, match="Mixture PPCA requires production complex64/float32"):
        validate_state(replace(state, theta=np.ones(state.theta.shape, np.complex128)), config)
    with pytest.raises(ValueError, match="Every mixture component needs its own optimizer state"):
        validate_state(replace(state, moments=state.moments[:1]), config)
    with pytest.raises(ValueError, match="Mixture optimizer shape mismatch"):
        validate_state(_with_moment(state, second=jnp.ones((1,), jnp.float32)), config)
    with pytest.raises(TypeError, match="Mixture optimizer requires complex64/float32"):
        validate_state(_with_moment(state, second=np.ones((32, 2), np.float64)), config)
    with pytest.raises(TypeError, match="Mixture optimizer requires complex64/float32"):
        validate_state(_with_moment(state, initialized=jnp.zeros((2, 32), jnp.int32)), config)


def test_validation_preserves_controller_prior_and_schedule_checks():
    state, config = _valid_state()
    for prior in (np.asarray([0, 1]), np.asarray([np.nan, 1]), np.ones(3)):
        with pytest.raises(ValueError, match="Mixture class priors must be finite and positive"):
            validate_state(replace(state, class_prior=prior), config)
    with pytest.raises(ValueError, match="Mixture class priors must sum to one"):
        validate_state(replace(state, class_prior=np.asarray([.3, .6])), config)
    with pytest.raises(ValueError, match="Mixture offset variance must be finite and positive"):
        validate_state(replace(state, offset_variance=0), config)
    with pytest.raises(ValueError, match="Mixture checkpoint iteration is outside the configured schedule"):
        validate_state(replace(state, iteration=config.iterations + 1), config)
    for prior in (np.asarray([1, np.nan]), np.asarray([1, -1]), np.zeros(2), np.ones((1, 2))):
        with pytest.raises(ValueError, match="Mixture angular prior is invalid"):
            validate_state(replace(state, direction_prior=prior), config)


def test_tomo_initialization_preserves_models_but_estimates_shared_noise_once(monkeypatch):
    from helpers.float_compare import assert_matches
    from recovar.core import fourier_transform_utils as ftu

    from relax.ppca_initial_class3d import initialization as mixture_initialization
    from relax.ppca_initial_model import tomo

    real = np.random.default_rng(41).normal(size=(8, 8, 8)).astype(np.float32)
    half = ftu.get_dft2_real(jnp.asarray(real)).reshape(8, -1)
    angle = np.deg2rad(30)
    tilted = np.asarray([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]])

    def read(ids):
        return real[ids], half[ids], jnp.ones((len(ids), 40), jnp.float32)

    particles = tomo.TiltParticles(
        (8, 8), (8, 8, 8), 1., np.arange(0, 9, 2), np.tile([0, 1], 4),
        np.zeros(4, np.int64), (np.stack([np.eye(3), tilted]),), read,
        noise_group=np.asarray([1, 0, 1, 0]),
    )
    config = Config(n_classes=2, q=1, iterations=12)
    bootstrap = tomo.tilt_bootstrap
    observed_rng = []

    def record_bootstrap(data, rng, *args, **kwargs):
        before = deepcopy(rng.bit_generator.state)
        result = bootstrap(data, rng, *args, **kwargs)
        observed_rng.append((before, deepcopy(rng.bit_generator.state)))
        return result

    monkeypatch.setattr(mixture_initialization, "tilt_bootstrap", record_bootstrap)
    monkeypatch.setattr(state_module, "tilt_bootstrap", record_bootstrap)
    expected = [mixture_initialization.initialize_tilts(
        particles, seed=config.seed + 104729 * k, diameter_ang=6., q=config.q,
    ) for k in range(config.n_classes)]
    expected_rng = observed_rng[:]
    observed_rng.clear()
    original_noise = mixture_initialization.initial_noise
    noise_reads = []

    def count_noise(images, *args, **kwargs):
        noise_reads.append(len(images))
        return original_noise(images, *args, **kwargs)

    monkeypatch.setattr(mixture_initialization, "initial_noise", count_noise)
    actual = state_module.initialize_state(particles, config, 6.)
    assert len(noise_reads) == 1
    assert observed_rng == expected_rng
    assert_matches(actual.noise, expected[0][1])
    for k, (theta, _, info) in enumerate(expected):
        assert_matches(actual.theta[k], theta)
        recorded = actual.initialization["components"][k]
        for key in ("seed", "bootstrap_particles", "noise_groups", "bootstrap_radius", "projection", "accumulation"):
            assert recorded[key] == info[key]
    assert actual.initialization["components"][0] == expected[0][2]
    shared = actual.initialization["components"][1]
    assert shared["noise_tilt_images"] == 0
    assert shared["noise_shared_from_component"] == 0
    assert shared["noise_component_seed"] == config.seed
