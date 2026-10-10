"""A strict IEEE evaluation of RELION's ``CTF::getCTF`` expression in NumPy, for rounding checks only.

RELION's own code is the value oracle (the binding, tests/unit/test_relion_exact_ctf_rows.py), but a
compiled RELION may contract products and sums into fused multiply-adds, so it agrees with any other
evaluation only to about 1e-13 of the CTF's range. This module evaluates the source expression
(ctf.h:209-253) one operation at a time with glibc's ``sin`` and ``exp``: what relax's CTF program must
reproduce up to the last unit of ``sin`` and ``exp``, and the reference for counting float32 flips.
"""

import numpy as np


def ctf_fftw_half(params, size: int, pixel_size: float, *, gamma_offset=None, mag_matrix=None) -> np.ndarray:
    """``(n, size, size // 2 + 1)`` float64; ``params`` rows are defU, defV, defAng (degrees), voltage (kV),
    Cs (mm), Q0, B-factor (A^2), scale, phase shift (degrees)."""

    params = np.asarray(params, dtype=np.float64).reshape(-1, 9)
    extent = float(size) * float(pixel_size)
    rows = np.arange(size)
    y = np.broadcast_to((np.where(rows <= size // 2, rows, rows - size) / extent)[:, None], (size, size // 2 + 1))
    x = np.broadcast_to((np.arange(size // 2 + 1) / extent)[None, :], (size, size // 2 + 1))
    if mag_matrix is not None:
        mag = np.asarray(mag_matrix, dtype=np.float64)
        x, y = mag[0, 0] * x + mag[0, 1] * y, mag[1, 0] * x + mag[1, 1] * y
    u2 = x * x + y * y
    u4 = u2 * u2
    out = np.empty((params.shape[0], size, size // 2 + 1))
    for row, (du, dv, angle, voltage, cs, q0, bfactor, scale, phase) in enumerate(params):
        volts = voltage * 1e3
        lam = 12.2643247 / np.sqrt(volts * (1.0 + volts * 0.978466e-6))
        k1 = np.pi / 2 * 2 * lam
        k2 = np.pi / 2 * (cs * 1e7) * lam * lam * lam
        k3 = np.arctan(q0 / np.sqrt(1 - q0 * q0))
        k5 = phase * np.pi / 180
        az = angle * np.pi / 180
        sin_az, cos_az = np.sin(az), np.cos(az)
        axx = cos_az * cos_az * -du + sin_az * sin_az * -dv
        axy = cos_az * sin_az * -du - sin_az * cos_az * -dv
        ayy = sin_az * sin_az * -du + cos_az * cos_az * -dv
        gamma = k1 * (axx * x * x + 2.0 * axy * x * y + ayy * y * y) + k2 * u4 - k5 - k3
        if gamma_offset is not None:
            gamma = gamma + np.asarray(gamma_offset, dtype=np.float64).reshape(size, size // 2 + 1)
        ctf = -np.sin(gamma)
        if bfactor != 0.0:
            ctf = ctf * np.exp(-bfactor / 4.0 * u2)
        ctf = ctf * scale
        out[row] = np.where(np.abs(ctf) < 1e-8, np.where(ctf >= 0, 1e-8, -1e-8), ctf)
    return out

