"""The exported display map is the inverse of the fitted Fourier tent basis.

The independent reference integrates each continuous, piecewise-linear Fourier
basis function by Gaussian quadrature. It does not use a sinc formula or either
production gridding helper. Model/export/projection arithmetic stays float32;
only this small numerical integration is a float64 diagnostic reference.
"""

from types import SimpleNamespace

import jax.numpy as jnp
import mrcfile
import numpy as np
import pytest
from helpers.float_compare import assert_matches
from recovar.core import fourier_transform_utils as ftu
from scipy.spatial.transform import Rotation

from relax.ppca_initial_class3d.outputs import export_models
from relax.ppca_refinement.full_row_stream import _project_half

pytestmark = pytest.mark.unit


def _model(n):
    """Four real Hermitian coefficient maps with no Nyquist-boundary support."""
    rng = np.random.default_rng(617)
    spectra = np.zeros((2, 2, n, n, n), np.complex64)
    center = n // 2
    frequencies = ((1, 0, 0), (0, 1, 1), (1, -1, 1), (2, 0, -1))
    for spectrum in spectra.reshape(4, n, n, n):
        spectrum[center, center, center] = np.float32(n**3)
        for frequency in frequencies:
            index = np.asarray(frequency)
            value = np.complex64(0.1 * n**3 * (rng.normal() + 1j * rng.normal()))
            spectrum[tuple(center + index)] = value
            spectrum[tuple(center - index)] = value.conjugate()
    half = ftu.full_volume_to_half_volume(jnp.asarray(spectra), (n, n, n))
    theta = half.reshape(2, 2, -1).swapaxes(1, 2)
    state = SimpleNamespace(theta=theta, class_prior=np.asarray([0.4, 0.6], np.float32),
                            noise=np.ones(n // 2 + 1, np.float32))
    dataset = SimpleNamespace(volume_shape=(n, n, n), grid_size=n, voxel_size=2.5)
    return spectra, state, dataset


def _tent_inverse_integral(spectra):
    """Inverse-integrate sum_j theta_j prod_d max(1-|k_d-j_d|,0)."""
    n = spectra.shape[-1]
    nodes, weights = np.polynomial.legendre.leggauss(16)
    # Separate the two polynomial pieces at the tent's kink, u=0.
    u = np.r_[(nodes - 1) / 2, (nodes + 1) / 2]
    quadrature = np.tile(weights / 2, 2) * (1 - np.abs(u))
    positions = np.arange(n) - n // 2
    frequencies = np.arange(n) - n // 2
    exponent = 2j * np.pi * positions[:, None, None] * (frequencies[None, :, None] + u) / n
    basis = np.sum(np.exp(exponent) * quadrature, axis=-1) / n
    integrated = np.einsum("xa,yb,zc,...abc->...xyz", basis, basis, basis, spectra, optimize=True)
    window_1d = np.sum(np.exp(2j * np.pi * positions[:, None] * u / n) * quadrature, axis=-1).real
    window = window_1d[:, None, None] * window_1d[None, :, None] * window_1d[None, None, :]
    return integrated.real.astype(np.float32), window.astype(np.float32)


def _tent_slices(spectra, rotations, radius):
    """Directly evaluate the Fourier basis at independently built image-plane points."""
    n = spectra.shape[-1]
    axis = np.arange(n) - n // 2
    packed = np.r_[np.arange(n // 2), -n // 2]
    x, y = np.meshgrid(packed, axis, indexing="xy")
    plane = np.stack([x, y, np.zeros_like(x)], axis=-1).reshape(-1, 3)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    slices = []
    for rotation in rotations:
        points = plane @ rotation.astype(np.float64)
        tent = np.prod(np.maximum(1 - np.abs(points[:, None] - grid[None]), 0), axis=-1)
        values = spectra.reshape(2, -1) @ tent.T
        values[:, np.sum(plane**2, axis=1) > radius**2] = 0
        slices.append(values)
    return np.stack(slices, axis=1).astype(np.complex64)


@pytest.mark.parametrize("n", [8, 12])
def test_display_export_matches_inverse_fourier_tent_integral(tmp_path, n):
    spectra, state, dataset = _model(n)
    expected_display, _ = _tent_inverse_integral(spectra)
    expected_coefficients = np.fft.fftshift(
        np.fft.ifftn(np.fft.ifftshift(spectra, axes=(-3, -2, -1)), axes=(-3, -2, -1)), axes=(-3, -2, -1)
    ).real.astype(np.float32)
    export_models(tmp_path, state, dataset)
    with np.load(tmp_path / "models.npz") as saved:
        assert_matches(saved["theta"], state.theta)
    for k in range(2):
        for channel, name in enumerate(("mean", "loading000")):
            for filename, reference in (
                    (f"class{k:03d}_model_{name}.mrc", expected_coefficients[k, channel]),
                    (f"class{k:03d}_{name}_gridding_corrected.mrc", expected_display[k, channel])):
                with mrcfile.open(tmp_path / filename) as mrc:
                    assert mrc.data.dtype == np.float32
                    assert_matches(mrc.data, reference, err_msg=filename)
                    assert_matches(np.float32(mrc.voxel_size.x), np.float32(dataset.voxel_size))


def test_display_export_reconstructs_the_fitted_off_grid_slices(tmp_path):
    spectra, state, dataset = _model(8)
    _, window = _tent_inverse_integral(spectra)
    export_models(tmp_path, state, dataset)
    recovered = []
    for name in ("mean", "loading000"):
        with mrcfile.open(tmp_path / f"class000_{name}_gridding_corrected.mrc") as mrc:
            recovered.append(ftu.get_dft3_real(jnp.asarray(mrc.data.copy() / window)).reshape(-1))
    rotations = Rotation.from_euler("xyz", [[17, -23, 31], [-41, 13, 27]], degrees=True).as_matrix().astype(np.float32)
    static = SimpleNamespace(projection_max_r=2.5, image_shape=(8, 8), volume_shape=(8, 8, 8),
                             disc_type="linear_interp", relion_texture_interp=False)
    expected = _tent_slices(spectra[0], rotations, static.projection_max_r)
    original = _project_half(SimpleNamespace(augmented=state.theta[0].T), jnp.asarray(rotations), static)
    restored = _project_half(SimpleNamespace(augmented=jnp.stack(recovered)), jnp.asarray(rotations), static)
    assert original.dtype == restored.dtype == jnp.complex64
    assert_matches(original, expected, err_msg="Production projector versus independently evaluated tents")
    assert_matches(restored, expected, err_msg="Display map restored to the fitted Fourier coefficients")
