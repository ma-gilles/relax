"""Latent coordinates in an orthonormal PC basis.

A PPCA map is ``x = mu + U z`` with PCs ``U`` (the columns of ``theta[:, 1:]``) that are neither orthogonal nor of
equal norm, so distances between raw (or per-axis standardized) ``z`` do not measure how different two maps are.
With the PCs' Gram matrix in real space ``G = M^T M = V diag(lam) V^T`` (``M`` the real-space PC maps as columns),
``z' = z V sqrt(lam)`` satisfies ``|z'_i - z'_j| = |M (z_i - z_j)|``: distances between ``z'`` are distances between
the particles' maps. Use ``z'`` (centred, not standardized) for UMAP and clustering.
"""

import numpy as np


def box_from_theta(theta):
    """The cubic box whose rfft half volume has ``theta.shape[0]`` voxels."""
    rows = int(np.asarray(theta).shape[0])
    for box in range(2, 2049):
        if box * box * (box // 2 + 1) == rows:
            return box
    raise ValueError(f"{rows} rows is no cubic box's rfft half volume")


def pc_maps(theta, box):
    """Real-space PC maps ``(q, box, box, box)`` from ``theta`` ``(box * box * (box // 2 + 1), 1 + q)`` (rfft half)."""
    theta = np.asarray(theta)
    half = (box, box, box // 2 + 1)
    if theta.ndim != 2 or theta.shape[0] != int(np.prod(half)) or theta.shape[1] < 2:
        raise ValueError(f"theta must be ({np.prod(half)}, 1 + q) for box {box}, got {theta.shape}")
    return np.stack([np.fft.irfftn(theta[:, j].reshape(half), s=(box,) * 3) for j in range(1, theta.shape[1])])


def orthonormal_latents(theta, z, box):
    """Return ``(z_orth, scales, rotation)``: ``z_orth = z @ rotation * scales``, columns by decreasing scale.

    ``scales`` are the PCs' singular values in real space (per-voxel sum convention of ``numpy.fft.irfftn``), so
    ``scales / scales[0]`` says how much map each orthonormal direction moves relative to the first.
    """
    z = np.asarray(z, np.float64)
    maps = pc_maps(theta, box)
    maps = maps.reshape(len(maps), -1).astype(np.float64)
    if z.ndim != 2 or z.shape[1] != len(maps):
        raise ValueError(f"z must be (P, q) with q = {len(maps)} PCs, got {z.shape}")
    lam, rotation = np.linalg.eigh(maps @ maps.T)
    lam, rotation = lam[::-1], rotation[:, ::-1]
    scales = np.sqrt(np.clip(lam, 0.0, None))
    return z @ rotation * scales[None, :], scales, rotation
