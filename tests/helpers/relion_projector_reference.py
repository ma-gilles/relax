"""RELION's ``Projector::project`` in numpy, batched over rotations (a test reference).

It reads the projector storage that ``relion_bind.compute_fourier_transform_map`` builds and
follows src/projector.cpp ``Projector::project`` for a 3D reference and 2D images:

- ``Ainv = A^-1 * padding_factor``;
- the output circle ``x <= sqrt(r_out^2 - y^2)`` with ``r_out = out_size / 2``;
- the reference sphere ``r_max``;
- Hermitian reflection for negative x;
- trilinear weights in x, then y, then z.

The output is RELION's FFTW half-image layout. Rows ``0..out_size/2`` hold y ``0..out_size/2``;
the rest hold the negative rows.

It is independent of relax's production projection. Tests that need many projections of one
reference use it; ``relion_bind.project_volume`` rebuilds the projector for every call.
"""

from __future__ import annotations

import numpy as np


def make_projector(bind, volume, ori_size: int, padding_factor: int, current_size: int) -> dict:
    """RELION's projector storage for ``volume`` (``computeFourierTransformMap``, trilinear)."""
    data, _power, _ori, pad, r_max, _r_min_nn, _interp = bind.compute_fourier_transform_map(
        volume, ori_size, padding_factor, 1, current_size, True, 2
    )
    return {"data": np.asarray(data), "pad": int(pad), "r_max": int(r_max)}


def project(projector: dict, rotations, out_size: int) -> np.ndarray:
    """Project the reference for rotations [M, 3, 3] into [M, out_size, out_size // 2 + 1] images."""
    data, pad, r_max = projector["data"], projector["pad"], projector["r_max"]
    zs, ys, xs = data.shape
    start_y, start_z = -(ys // 2), -(zs // 2)
    r_out = out_size // 2
    rows = np.arange(out_size)
    y = np.where(rows <= r_out, rows, rows - out_size)
    Y, X = np.meshgrid(y, np.arange(out_size // 2 + 1), indexing="ij")
    in_circle = (r_out**2 - Y**2 >= 0) & (X <= np.floor(np.sqrt(np.maximum(r_out**2 - Y**2, 0))))
    xf, yf = X[in_circle].astype(np.float64), Y[in_circle].astype(np.float64)
    a_inv = np.linalg.inv(np.asarray(rotations, dtype=np.float64)) * pad
    xp = a_inv[:, 0, 0, None] * xf + a_inv[:, 0, 1, None] * yf
    yp = a_inv[:, 1, 0, None] * xf + a_inv[:, 1, 1, None] * yf
    zp = a_inv[:, 2, 0, None] * xf + a_inv[:, 2, 1, None] * yf
    inside = xp * xp + yp * yp + zp * zp <= (r_max * pad) ** 2
    negative = xp < 0
    xp, yp, zp = (np.where(negative, -c, c) for c in (xp, yp, zp))
    x0, y0, z0 = (np.floor(c).astype(np.int64) for c in (xp, yp, zp))
    fx, fy, fz = xp - x0, yp - y0, zp - z0
    y0 = y0 - start_y
    z0 = z0 - start_z
    inside &= (x0 >= 0) & (x0 + 1 < xs) & (y0 >= 0) & (y0 + 1 < ys) & (z0 >= 0) & (z0 + 1 < zs)
    x0, y0, z0 = (np.where(inside, c, 0) for c in (x0, y0, z0))
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
    d00 = data[z0, y0, x0] + fx * (data[z0, y0, x1] - data[z0, y0, x0])
    d01 = data[z1, y0, x0] + fx * (data[z1, y0, x1] - data[z1, y0, x0])
    d10 = data[z0, y1, x0] + fx * (data[z0, y1, x1] - data[z0, y1, x0])
    d11 = data[z1, y1, x0] + fx * (data[z1, y1, x1] - data[z1, y1, x0])
    dy0 = d00 + fy * (d10 - d00)
    dy1 = d01 + fy * (d11 - d01)
    value = dy0 + fz * (dy1 - dy0)
    value = np.where(inside, np.where(negative, np.conj(value), value), 0.0)
    out = np.zeros((len(a_inv), out_size, out_size // 2 + 1), dtype=np.complex128)
    out[:, in_circle] = value
    return out
