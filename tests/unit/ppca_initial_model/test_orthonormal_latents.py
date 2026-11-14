"""orthonormal_latents: distances between z' equal distances between the particles' real-space maps."""

import numpy as np
import pytest

from relax.ppca_initial_model.latents import orthonormal_latents, pc_maps


def _theta(box, q, rng):
    # Real-space PCs of very different norms and strongly correlated directions, stored as rfft half volumes.
    base = rng.standard_normal((q, box, box, box))
    mix = np.triu(rng.standard_normal((q, q))) + 3 * np.eye(q)
    real = np.einsum("ij,j...->i...", mix, base) * np.geomspace(5.0, 0.2, q)[:, None, None, None]
    half = np.stack([np.fft.rfftn(m).ravel() for m in real], axis=1)
    mean = np.fft.rfftn(rng.standard_normal((box,) * 3)).ravel()[:, None]
    return np.concatenate([mean, half], axis=1), real


def test_distances_are_map_distances():
    rng = np.random.default_rng(0)
    box, q = 8, 4
    theta, real = _theta(box, q, rng)
    np.testing.assert_allclose(pc_maps(theta, box), real, atol=1e-10)
    z = rng.standard_normal((50, q))
    z_orth, scales, rotation = orthonormal_latents(theta, z, box)
    maps = np.einsum("pq,q...->p...", z, real).reshape(len(z), -1)
    d_maps = np.linalg.norm(maps[:, None] - maps[None], axis=-1)
    d_orth = np.linalg.norm(z_orth[:, None] - z_orth[None], axis=-1)
    np.testing.assert_allclose(d_orth, d_maps, rtol=1e-9, atol=1e-9)
    assert np.all(np.diff(scales) <= 0)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(q), atol=1e-12)


def test_orthonormal_pcs_only_rescale():
    # Orthogonal PCs: z' is z with each axis multiplied by its PC's norm (up to order and sign).
    box, q = 8, 3
    real = np.zeros((q, box, box, box))
    for j, norm in enumerate((1.0, 4.0, 2.0)):
        real[j].flat[j * 7] = norm
    theta = np.concatenate(
        [np.zeros((box * box * (box // 2 + 1), 1), complex), np.stack([np.fft.rfftn(m).ravel() for m in real], axis=1)],
        axis=1,
    )
    z = np.random.default_rng(1).standard_normal((20, q))
    z_orth, scales, _ = orthonormal_latents(theta, z, box)
    np.testing.assert_allclose(scales, [4.0, 2.0, 1.0])
    np.testing.assert_allclose(np.abs(z_orth), np.abs(z[:, [1, 2, 0]] * [4.0, 2.0, 1.0]), atol=1e-12)


def test_shape_errors():
    theta = np.zeros((8 * 8 * 5, 3), complex)
    with pytest.raises(ValueError):
        orthonormal_latents(theta, np.zeros((4, 3)), 8)
    with pytest.raises(ValueError):
        pc_maps(theta, 6)
