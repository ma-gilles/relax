"""RELION's numeric macros (src/macros.h): the one home of ``ROUND``, ``CEIL`` and ``PI`` in relax."""

import math

import numpy as np

# macros.h ``PI``; the same double as ``math.pi``.
PI = 3.14159265358979323846


def relion_round(value: float) -> int:
    """RELION ``ROUND`` (macros.h:197): nearest integer, halves away from zero."""

    return int(math.floor(value + 0.5)) if value >= 0.0 else -int(math.floor(-value + 0.5))


def relion_ceil(value: float) -> int:
    """RELION ``CEIL`` (macros.h:212)."""

    if value == int(value):
        return int(value)
    return int(value + 1) if value > 0 else int(value)


def relion_round_array(values) -> np.ndarray:
    """``ROUND`` of every element, computed in the array's own dtype; returns that dtype (integer-valued).

    Callers cast the input first when RELION rounds a double (``np.asarray(v, np.float64)``) and the
    result when they need integers (``.astype(np.int64)``).
    """

    x = np.asarray(values)
    return np.where(x >= 0, np.floor(x + 0.5), -np.floor(-x + 0.5))
