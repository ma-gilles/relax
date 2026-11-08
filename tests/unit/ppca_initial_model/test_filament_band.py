"""The filament band's pose grid, attachment support, configuration and seeding."""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.ppca_initial_model import checkpoint
from relax.ppca_initial_model.config import Config
from relax.ppca_initial_model.filament_band import FilamentBand, attachment_support_mask, uniform_log_prior
from relax.ppca_initial_model.initialization import seed_model

pytestmark = pytest.mark.unit


def test_grid_is_rotations_about_the_axis_and_shifts_along_it():
    band = FilamentBand(length_ang=240.0, axial_step_ang=15.0, phi_step_deg=5.0)
    rotations, phis = band.rotations(), np.deg2rad(band.phis_deg())
    assert rotations.shape == (72, 3, 3) and band.eulers().shape == (72, 3)
    z = np.array([0.0, 0.0, 1.0])
    # Each candidate fixes the axis and turns the plane by its own angle (RELION Euler (phi, 0, 0)).
    assert_matches(rotations @ z.astype(np.float32), np.tile(z, (72, 1)).astype(np.float32))
    turned = np.einsum("rij,j->ri", rotations.astype(np.float64), [1.0, 0.0, 0.0])
    assert_matches(np.abs(np.cos(phis)).astype(np.float32), np.abs(turned[:, 0]).astype(np.float32))
    assert np.unique(np.round(np.arctan2(turned[:, 1], turned[:, 0]), 6)).size == 72
    a = band.axial_offsets_ang()
    assert_matches(a, np.arange(-120.0, 121.0, 15.0))  # symmetric, includes 0, ends within length / 2
    t = band.translations_px(voxel_size=8.0)
    assert t.dtype == np.float32 and np.all(t[:, :2] == 0)
    assert_matches(t[:, 2], (a / 8.0).astype(np.float32))
    assert_matches(np.float32(np.sum(np.exp(uniform_log_prior(17).astype(np.float64)))), np.float32(1.0))


def test_band_steps_are_validated():
    with pytest.raises(ValueError):
        FilamentBand(length_ang=240.0, phi_step_deg=7.0)  # does not divide 360
    with pytest.raises(ValueError):
        FilamentBand(length_ang=240.0, axial_step_ang=0.0)
    assert FilamentBand(length_ang=0.0).axial_offsets_ang().tolist() == [0.0]


def test_attachment_support_sits_on_plus_x_at_its_radius():
    n, voxel = 64, 8.0
    band = FilamentBand(length_ang=80.0, support_center_ang=120.0, support_radius_ang=40.0)
    mask = np.asarray(attachment_support_mask(n, voxel, band))
    assert mask.dtype == np.float32 and mask.shape == (n, n, n)
    center = (n // 2 + 15, n // 2, n // 2)  # 120 A / 8 A on axis 0 (RELION x, relax/io/map_io.py)
    assert mask[center] == 1.0 and mask[n // 2, n // 2, n // 2] == 0.0  # the filament axis is outside
    weights = mask / mask.sum()
    centroid = [float(np.sum(np.indices(mask.shape)[i] * weights)) for i in range(3)]
    assert_matches(np.asarray(centroid, np.float32), np.asarray(center, np.float32))
    assert np.all(mask[: n // 2] == 0)  # nothing on -x
    with pytest.raises(ValueError, match="leaves"):
        attachment_support_mask(n, voxel, FilamentBand(length_ang=80.0, support_center_ang=230.0))


def test_config_band_needs_its_own_full_grid_and_occupancy_needs_the_band():
    config = Config(band_length_ang=240.0, occupancy=0.1)
    assert config.band() == FilamentBand(240.0)
    assert Config().band() is None
    with pytest.raises(ValueError, match="own full grid"):
        Config(band_length_ang=240.0, oversampling=1)
    with pytest.raises(ValueError, match="filament band"):
        Config(occupancy=0.1)
    with pytest.raises(ValueError, match="Occupancy"):
        Config(band_length_ang=240.0, occupancy=1.5)


def test_checkpoints_without_band_fields_keep_their_identity():
    """A checkpoint written before the band fields existed holds the default (full-grid) model."""
    old = {k: v for k, v in dataclasses.asdict(Config()).items() if k not in checkpoint._ADDED_CONFIG_DEFAULTS}
    assert checkpoint._identity_config(old) == checkpoint._identity_config(dataclasses.asdict(Config()))
    assert checkpoint._identity_config(
        dataclasses.asdict(Config(band_length_ang=240.0))
    ) != checkpoint._identity_config(old)


def test_band_seed_lives_in_the_attachment_support():
    n, voxel, q = 48, 8.0, 2
    band = FilamentBand(length_ang=80.0, support_center_ang=80.0, support_radius_ang=32.0)
    shape = (n, n, n)
    half = n * n * (n // 2 + 1)
    rng = np.random.default_rng(0)
    rhs = jnp.asarray(rng.normal(size=(q + 1, half)) + 1j * rng.normal(size=(q + 1, half)), jnp.complex64)
    lhs = jnp.ones((q + 1, half), jnp.float32)
    theta = seed_model(rhs, lhs, shape, voxel, 200.0, n // 2 - 1, np.random.default_rng(1), band=band)
    import recovar.core.fourier_transform_utils as ftu

    real = np.asarray(
        ftu.get_idft3_real(theta.T.reshape((q + 1,) + ftu.volume_shape_to_half_volume_shape(shape)), shape)
    )
    energy = np.sum(real**2, axis=0)
    mask = np.asarray(attachment_support_mask(n, voxel, band))
    # Masked in real space then band-limited: nearly all the seed's energy lies in the support.
    assert np.sum(energy * (mask > 0)) / np.sum(energy) > 0.9
    assert np.sum(energy[: n // 2]) / np.sum(energy) < 0.05  # not about the box centre or on -x
