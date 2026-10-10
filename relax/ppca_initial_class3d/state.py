"""K independent model/optimizer states, one physical-particle observation model."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from relax.ppca_initial_class3d.initialization import initialize, initialize_tilts, seed_model
from relax.ppca_initial_class3d.membership import Membership
from relax.ppca_initial_model.tomo import TiltParticles, tilt_bootstrap
from relax.ppca_initial_model.update import Moments, empty_moments
from relax.ppca_refinement.residual_statistics import full_float32


@dataclass
class State:
    theta: object  # [K, frequency, q+1], complex64
    moments: tuple[Moments, ...]
    noise: object  # shared [shell], or [noise group, shell] for tomography
    class_prior: np.ndarray
    iteration: int
    order: np.ndarray
    rng_state: dict
    offset_variance: float
    radius: int
    initialization: dict
    direction_prior: object = None  # shared angular distribution, not multiplied by K
    direction_order: int = -1
    membership: Membership | None = None
    sampling: object = None  # auto_sampling.SamplingState of an auto-sampled run, else None


def initialize_state(dataset, config, diameter_ang):
    tomo = isinstance(dataset, TiltParticles)
    initializer = initialize_tilts if tomo else initialize
    models, metadata = [], []
    for k in range(config.n_classes):
        seed = config.seed + 104729 * k
        if tomo and k > 0:
            theta, info = _initialize_tilt_model(
                dataset, seed=seed, diameter_ang=diameter_ang, q=config.q, noise_component_seed=config.seed
            )
        else:
            theta, estimated_noise, info = initializer(
                dataset, seed=seed, diameter_ang=diameter_ang, q=config.q
            )
        models.append(theta)
        metadata.append(info)
        if k == 0:
            # All mixture classes see the same noise estimated by component zero.
            noise = estimated_noise
    rng = np.random.default_rng(config.seed + 1)
    order = rng.permutation(dataset.n_images)
    return State(
        jnp.stack(models), tuple(empty_moments(m) for m in models), noise,
        np.full(config.n_classes, 1 / config.n_classes, np.float32),
        0, order, rng.bit_generator.state, 100.0 / dataset.voxel_size**2, 0,
        {"components": metadata, "noise_component_seed": config.seed},
    )


@full_float32
def _initialize_tilt_model(particles, *, seed, diameter_ang, q, noise_component_seed):
    """The legacy TOMO model prefix, without re-estimating its shared noise.

    Both operations use the mixture's signed seed and the same seeded RNG sequence
    as ``initialize_tilts``. Noise-only image reads occur after these operations
    in the legacy initializer, so omitting them cannot change this component's
    bootstrap, blob fields or model. Component zero uses the full mixture
    initializer, including its one shared noise estimate.
    """
    rng = np.random.default_rng(seed)
    radius = max(1, int(np.floor(0.07 * particles.grid_size + 0.5)))
    rhs, lhs, chosen = tilt_bootstrap(particles, rng, q + 1, radius)
    theta = seed_model(rhs, lhs, particles.volume_shape, particles.voxel_size, diameter_ang, radius, rng)
    return theta, {
        "seed": seed,
        "bootstrap_particles": chosen.tolist(),
        "noise_tilt_images": 0,
        "noise_groups": particles.n_noise_groups,
        "bootstrap_radius": radius,
        "projection": "complex64",
        "accumulation": "complex64/float32",
        "noise_initializer": "shared spectrum from component 0; no additional noise-only image reads",
        "noise_shared_from_component": 0,
        "noise_component_seed": noise_component_seed,
    }


@jax.jit
def _optimizer_validity(first, second):
    """Two scalar flags; never copy optimizer volumes to the host to inspect them."""
    return jnp.stack((
        jnp.all(jnp.isfinite(first)) & jnp.all(jnp.isfinite(second)),
        jnp.all(second >= 0),
    ))


@jax.jit
def _all_finite(values):
    return jnp.all(jnp.isfinite(values))


@jax.jit
def _finite_positive(values):
    return jnp.all(jnp.isfinite(values)) & jnp.all(values > 0)


def validate_state(state, config):
    """Fail at checkpoint/update boundaries, including every optimizer component.

    Large model, noise and optimizer arrays stay on their device: only reduced
    Boolean flags cross to the host. Small controller priors keep their existing
    host checks and summation order, including the normalization tolerance.
    """
    shape = state.theta.shape
    if len(shape) != 3 or shape[0] != config.n_classes or shape[2] != config.q + 1:
        raise ValueError("Mixture checkpoint/model K or q mismatch")
    if state.theta.dtype != jnp.complex64 or state.noise.dtype != jnp.float32:
        raise TypeError("Mixture PPCA requires production complex64/float32")
    if len(state.moments) != config.n_classes:
        raise ValueError("Every mixture component needs its own optimizer state")
    for moment in state.moments:
        if (moment.first.shape != (2,) + shape[1:] or moment.second.shape != (shape[1], 2)
                or moment.initialized.shape != (2, shape[1])):
            raise ValueError("Mixture optimizer shape mismatch")
        if moment.first.dtype != jnp.complex64 or moment.second.dtype != jnp.float32 or moment.initialized.dtype != jnp.bool_:
            raise TypeError("Mixture optimizer requires complex64/float32")
        finite, nonnegative = jax.device_get(_optimizer_validity(moment.first, moment.second))
        if not finite:
            raise ValueError("Nonfinite mixture optimizer state")
        if not nonnegative:
            raise ValueError("Negative mixture second moments")
    if not bool(jax.device_get(_all_finite(state.theta))):
        raise ValueError("Nonfinite mixture PPCA model")
    if not bool(jax.device_get(_finite_positive(state.noise))):
        raise ValueError("Mixture noise must be finite and positive")
    prior = np.asarray(state.class_prior)
    if prior.shape != (config.n_classes,) or np.any(prior <= 0) or not np.all(np.isfinite(prior)):
        raise ValueError("Mixture class priors must be finite and positive")
    if not np.isclose(prior.sum(), 1, atol=1e-6, rtol=0):
        raise ValueError("Mixture class priors must sum to one")
    if not np.isfinite(state.offset_variance) or state.offset_variance <= 0:
        raise ValueError("Mixture offset variance must be finite and positive")
    if not isinstance(state.iteration, (int, np.integer)) or not 0 <= state.iteration <= config.iterations:
        raise ValueError("Mixture checkpoint iteration is outside the configured schedule")
    if state.direction_prior is not None:
        angular = np.asarray(state.direction_prior)
        if angular.ndim != 1 or not np.all(np.isfinite(angular)) or np.any(angular < 0) or angular.sum() <= 0:
            raise ValueError("Mixture angular prior is invalid")
    if state.membership is not None:
        state.membership.validate(n_particles=len(state.order), n_classes=config.n_classes,
                                  iteration=state.iteration)
    if state.sampling is not None:
        sampling = state.sampling
        if (not config.auto_sampling or sampling.healpix_order < 1 or sampling.healpix_order > config.max_order
                or not np.isfinite(sampling.offset_step_px) or sampling.offset_step_px <= 0
                or not np.isfinite(sampling.offset_range_px) or sampling.offset_range_px < 0):
            raise ValueError("Invalid auto-sampling state for this configuration")
