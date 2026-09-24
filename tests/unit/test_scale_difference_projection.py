"""RELION's applyScaleDifference for an optics group on another pixel size and box (CPU).

A group image of ``box_g`` pixels at ``angpix_g`` samples the reference (``ori`` at
``angpix_ref``) at ``k / s_g`` voxels, ``s_g = box_g angpix_g / (ori angpix_ref)``
(``obs_model.cpp:1332-1339``; the projector uses the inverse of the scaled matrix).
Projecting the reference with the rotation divided by ``s_g`` must match projecting
the same analytic volume sampled directly on the group's grid.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from relax.helpers.projection import project_relion_projector_half_spectrum
from relax.relion.relion_projector_setup import setup_relion_projector

REF = (32, 4.0)
GROUP = (28, 4.0 * 32 / 24)  # a coarser pixel and a different box


def _analytic_volume(n, angpix):
    """Real-space blobs at fixed physical positions (angstrom), sampled on an n^3 grid."""
    x = (np.arange(n) - n // 2) * angpix
    zz, yy, xx = np.meshgrid(x, x, x, indexing="ij")
    vol = np.exp(-((xx - 10) ** 2 + (yy + 4) ** 2 + zz**2) / (2 * 9.0**2))
    vol += 0.6 * np.exp(-((xx + 12) ** 2 + (yy - 14) ** 2 + (zz - 6) ** 2) / (2 * 7.0**2))
    return vol * angpix**3  # a sampled integral, so both grids carry one physical amplitude


def _project(volume, n, rotation, image_size, padding=2):
    projector, _ = setup_relion_projector(jnp.asarray(volume), n // 2 - 1, ori_size=n, padding_factor=padding)
    out = project_relion_projector_half_spectrum(
        projector, jnp.asarray(rotation)[None], (image_size, image_size), n // 2 - 1, padding
    )
    return np.asarray(out).reshape(image_size, image_size // 2 + 1)


@pytest.mark.unit
def test_scaled_rotation_matches_projection_on_the_group_grid():
    rotation = Rotation.from_euler("ZYZ", [37.0, 61.0, -112.0], degrees=True).as_matrix()
    (n_ref, a_ref), (n_g, a_g) = REF, GROUP
    s_g = n_g * a_g / (n_ref * a_ref)

    scaled = _project(_analytic_volume(n_ref, a_ref), n_ref, rotation / s_g, n_g)
    direct = _project(_analytic_volume(n_g, a_g), n_g, rotation, n_g)

    # Compare inside the lower of the two Nyquist radii, in physical frequency.
    ky = np.fft.fftfreq(n_g) * n_g
    kx = np.arange(n_g // 2 + 1)
    radius = np.hypot(*np.meshgrid(ky, kx, indexing="ij"))
    inside = radius < 0.8 * min(n_g // 2, n_ref // 2 * s_g)
    a, b = scaled[inside], direct[inside]
    correlation = np.abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
    assert correlation > 0.9999
    # The unscaled rotation projects the wrong frequencies.
    wrong = _project(_analytic_volume(n_ref, a_ref), n_ref, rotation, n_g)[inside]
    assert np.abs(np.vdot(wrong, b)) / (np.linalg.norm(wrong) * np.linalg.norm(b)) < 0.99
