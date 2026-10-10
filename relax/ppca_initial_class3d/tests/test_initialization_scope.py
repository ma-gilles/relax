"""Signed mixture seeding preserves bootstrap arithmetic without changing K=1."""

from copy import deepcopy
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches
from recovar.core import fourier_transform_utils as ftu
from scipy.spatial.transform import Rotation

from relax.ppca_initial_class3d import initialization as mixture
from relax.ppca_initial_class3d.state import initialize_state
from relax.ppca_initial_model import initialization as single
from relax.ppca_initial_model import tomo

pytestmark = pytest.mark.unit


def _previous_signed_formula(rhs, lhs, shape, voxel, diameter_ang, radius, rng):
    """The pre-isolation formula applies the sign before FFT and masking."""
    channels = rhs.shape[0]
    half_shape = ftu.volume_shape_to_half_volume_shape(shape)
    reconstruction = jnp.where(lhs > 0, rhs / jnp.where(lhs > 0, lhs, 1), 0)
    real = np.asarray(ftu.get_idft3_real(reconstruction.reshape((channels,) + half_shape), shape), np.float32)
    diameter = diameter_ang / voxel
    x = np.arange(shape[0]) - shape[0] // 2
    centre = np.sqrt(sum(a**2 for a in np.meshgrid(x, x, x, indexing="ij"))) <= diameter / 4
    sign = np.float32(-1 if np.mean(real[0][centre]) < 0 else 1)
    fields = []
    for volume in real:
        field = single._blob_field(volume, diameter, rng) - np.float32(0.5) * single._blob_field(volume, diameter, rng)
        fields.append(sign * field * (np.std(volume, dtype=np.float32) / np.std(field, dtype=np.float32)))
    augmented = single.seed_maps_to_model(np.stack(fields))
    theta = ftu.get_dft3_real(augmented).reshape(channels, -1).T
    return single.bandlimit_and_mask(theta, shape, radius, single.support_mask(shape[0], diameter)), sign


@pytest.mark.parametrize("q", [1, 2, 5])
@pytest.mark.parametrize("sign", [-1, 1])
def test_signed_seed_matches_pre_isolation_formula_and_rng(q, sign):
    n, radius = 16, 3
    x = np.arange(n) - n // 2
    blob = np.exp(-sum(a**2 for a in np.meshgrid(x, x, x, indexing="ij")) / 18).astype(np.float32)
    volumes = sign * np.stack([(1 + np.float32(k / 10)) * blob for k in range(q + 1)])
    rhs = ftu.get_dft3_real(volumes).reshape(q + 1, -1)
    lhs = jnp.ones(rhs.shape, jnp.float32)
    actual_rng, expected_rng = np.random.default_rng(47), np.random.default_rng(47)
    actual = mixture.seed_model(rhs, lhs, (n,) * 3, 1., 8., radius, actual_rng)
    expected, bootstrap_sign = _previous_signed_formula(rhs, lhs, (n,) * 3, 1., 8., radius, expected_rng)
    assert bootstrap_sign == sign
    assert_matches(actual, expected)
    assert actual.dtype == jnp.complex64
    assert actual_rng.bit_generator.state == expected_rng.bit_generator.state


class _SPAData:
    grid_size, image_shape, volume_shape, voxel_size = 8, (8, 8), (8, 8, 8), 1.

    def __init__(self, images):
        self.images = images
        self.n_images = len(images)

    def iter_batches(self, batch_size, *, indices, by_image):
        assert by_image
        for start in range(0, len(indices), batch_size):
            ids = indices[start : start + batch_size]
            yield self.images[ids], None, None, np.zeros((len(ids), 1)), None, ids, ids

    def process_images_half(self, images, *, apply_image_mask):
        assert not apply_image_mask
        return ftu.get_dft2_real(jnp.asarray(images))

    def ctf_evaluator(self, params, image_shape, voxel, *, half_image):
        assert image_shape == (8, 8) and voxel == 1. and half_image
        return jnp.ones((len(params), 40), jnp.float32)


def _dataset(tomography, sign):
    x = np.arange(8) - 4
    blob = np.exp(-(x[:, None] ** 2 + x[None, :] ** 2) / 4).astype(np.float32)
    images = sign * (blob[None] + np.random.default_rng(18).normal(0, .005, (12, 8, 8))).astype(np.float32)
    if not tomography:
        return _SPAData(images)
    half = ftu.get_dft2_real(jnp.asarray(images)).reshape(12, -1)

    def read(ids):
        return images[ids], half[ids], jnp.ones((len(ids), 40), jnp.float32)

    frames = Rotation.from_euler("y", [-30, 0, 30], degrees=True).as_matrix()
    return tomo.TiltParticles(
        (8, 8), (8, 8, 8), 1., np.arange(0, 13, 3), np.tile(np.arange(3), 4),
        np.zeros(4, np.int64), (frames,), read, noise_group=np.asarray([1, 0, 1, 0]),
    )


@pytest.mark.parametrize("tomography", [False, True], ids=["spa", "et"])
@pytest.mark.parametrize("sign", [-1, 1])
def test_initializers_keep_bootstrap_noise_and_rng_scoped(monkeypatch, tomography, sign):
    data = _dataset(tomography, sign)
    upstream = tomo.initialize_tilts if tomography else single.initialize
    local = mixture.initialize_tilts if tomography else mixture.initialize
    original_single_seed, original_tomo_seed = single.seed_model, tomo.seed_model
    reference, reference_noise, reference_info = upstream(data, seed=43, diameter_ang=6., q=1)
    signed_seed = mixture.seed_model
    observed = []

    def check_previous_formula(rhs, lhs, shape, voxel, diameter_ang, radius, rng):
        reference_rng = np.random.default_rng()
        reference_rng.bit_generator.state = deepcopy(rng.bit_generator.state)
        expected, bootstrap_sign = _previous_signed_formula(rhs, lhs, shape, voxel, diameter_ang, radius, reference_rng)
        actual = signed_seed(rhs, lhs, shape, voxel, diameter_ang, radius, rng)
        assert_matches(actual, expected)
        assert rng.bit_generator.state == reference_rng.bit_generator.state
        observed.append(bootstrap_sign)
        return actual

    # Only the package-local hook is replaced; shared modules retain their identities.
    monkeypatch.setattr(mixture, "seed_model", check_previous_formula)
    actual, noise, info = local(data, seed=43, diameter_ang=6., q=1)
    assert observed == [sign]
    assert_matches(actual, np.float32(sign) * reference)
    assert_matches(noise, reference_noise)
    assert info == reference_info
    assert single.seed_model is original_single_seed and tomo.seed_model is original_tomo_seed
    assert single.seed_model is not mixture.seed_model


@pytest.mark.parametrize("tomography", [False, True], ids=["spa", "et"])
def test_each_component_uses_the_signed_local_seed(tomography):
    data = _dataset(tomography, -1)
    config = SimpleNamespace(n_classes=3, q=1, seed=43)
    result = initialize_state(data, config, 6.)
    full_initialize = mixture.initialize_tilts if tomography else mixture.initialize
    for k in range(config.n_classes):
        expected, noise, _ = full_initialize(data, seed=config.seed + 104729 * k, diameter_ang=6., q=1)
        assert_matches(result.theta[k], expected)
        if k == 0:
            assert_matches(result.noise, noise)
