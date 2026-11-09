"""--initial-mean: a RELION-convention map replaces the random initial mean (initialization.with_initial_mean).

The mean must land in the representation the stream projects ``mu`` from, which is the one
``FixedCore.from_map`` builds for a core (both go through recovar's RELION -> internal conversion and the
centered half-Fourier layout), masked by the attachment support and held within the sphere of radius n/2 in
Fourier space, as the random seed is.
"""

import mrcfile
import numpy as np
import pytest
import recovar.core.fourier_transform_utils as ftu
from helpers.float_compare import assert_matches
from recovar.utils.helpers import load_relion_volume

from relax.ppca_initial_model.filament_band import FilamentBand, attachment_support_mask
from relax.ppca_initial_model.fixed_core import FixedCore
from relax.ppca_initial_model.initialization import with_initial_mean


def _write(path, volume, voxel):
    with mrcfile.new(path) as m:
        m.set_data(volume.astype(np.float32))
        m.voxel_size = voxel


def test_initial_mean_is_the_scaled_masked_map_in_the_stream_layout(tmp_path):
    n, voxel, q, scale = 32, 8.0, 2, 0.75
    band = FilamentBand(length_ang=80.0, support_center_ang=56.0, support_radius_ang=32.0)
    relion = np.random.default_rng(3).normal(size=(n, n, n)).astype(np.float32)
    path = tmp_path / "mean.mrc"
    _write(path, relion, voxel)
    rng = np.random.default_rng(4)
    n_half = n * n * (n // 2 + 1)
    theta = (rng.normal(size=(n_half, q + 1)) + 1j * rng.normal(size=(n_half, q + 1))).astype(np.complex64)

    out = np.asarray(with_initial_mean(theta, path, scale, (n, n, n), voxel, 200.0, band=band))

    internal = np.asarray(load_relion_volume(str(path)), np.float32)
    mask = np.asarray(attachment_support_mask(n, voxel, band))
    expected = FixedCore.from_volume(
        internal * np.float32(scale) * mask, volume_shape=(n, n, n), voxel_size=voxel, volume_domain="real"
    )
    inside = np.asarray(ftu.get_grid_of_radial_distances_real((n, n, n), rounded=False)).reshape(-1) <= n // 2
    assert out.dtype == np.complex64
    assert_matches(out[:, 0], np.asarray(expected.volume_half) * inside)
    np.testing.assert_array_equal(out[:, 1:], theta[:, 1:])  # loadings untouched


def test_initial_mean_refuses_a_map_of_another_box_or_voxel(tmp_path):
    n, voxel = 16, 8.0
    theta = np.zeros((n * n * (n // 2 + 1), 2), np.complex64)
    _write(tmp_path / "small.mrc", np.zeros((n - 2,) * 3), voxel)
    with pytest.raises(ValueError, match="the model is"):
        with_initial_mean(theta, tmp_path / "small.mrc", 1.0, (n, n, n), voxel, 100.0)
    _write(tmp_path / "voxel.mrc", np.zeros((n,) * 3), 2 * voxel)
    with pytest.raises(ValueError, match="the model is"):
        with_initial_mean(theta, tmp_path / "voxel.mrc", 1.0, (n, n, n), voxel, 100.0)
