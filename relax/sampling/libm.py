"""The C library's ``acos`` and ``atan2`` over arrays, as RELION's host code evaluates them.

NumPy dispatches some float64 functions to SIMD kernels whose rounding is not the
C library's: on della's Intel (AVX512) nodes ``np.arccos`` differs from glibc's
``acos`` in about 9% of arguments and ``np.arctan2`` in about 0.1% (probe job
14748696), while ``np.sin``, ``np.cos`` and ``np.sqrt`` agree. A one-ulp angle
decides exact cone and radius ties against RELION, so the sampling ports take
these two functions from :mod:`math`, which calls the C library.
"""

from __future__ import annotations

import math

import numpy as np

_ACOS = np.frompyfunc(math.acos, 1, 1)
_ATAN2 = np.frompyfunc(math.atan2, 2, 1)


def acos(values) -> np.ndarray:
    """``acos`` of each element, float64, from the C library."""

    values = np.asarray(values, dtype=np.float64)
    return np.asarray(_ACOS(values), dtype=np.float64).reshape(values.shape)


def atan2(y, x) -> np.ndarray:
    """``atan2(y, x)`` of each element pair, float64, from the C library."""

    y, x = np.broadcast_arrays(np.asarray(y, dtype=np.float64), np.asarray(x, dtype=np.float64))
    return np.asarray(_ATAN2(y, x), dtype=np.float64).reshape(y.shape)


__all__ = ["acos", "atan2"]
