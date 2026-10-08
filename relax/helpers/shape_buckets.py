"""Static-shape bucket helpers for dense RELION refinement.

JAX recompiles when array shapes change.  RELION-style pruning changes local
support sizes frequently, so hot EM loops should round those sizes to a small
set of padded shape classes and pass masks/counts for the valid rows.
"""

from __future__ import annotations

import numpy as np


def round_up_to_multiple(value: int, multiple: int) -> int:
    """Round ``value`` up to a positive multiple."""

    value = int(value)
    multiple = int(multiple)
    if value < 0:
        raise ValueError(f"value must be non-negative, got {value}")
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}")
    return ((value + multiple - 1) // multiple) * multiple


def pow2_ceil(value: int, *, minimum: int = 1) -> int:
    """The smallest power of two at least ``max(value, minimum)``: ``1 << (n - 1).bit_length()``."""

    return 1 << (max(int(value), int(minimum)) - 1).bit_length()


def pow2_floor(value: int) -> int:
    """The largest power of two at most ``value`` (a positive integer)."""

    if value < 1:
        raise ValueError(f"pow2_floor needs a positive value, got {value}")
    return 1 << (int(value).bit_length() - 1)


def power_bucket(
    value: int,
    *,
    base: int = 2,
    minimum: int = 1,
    maximum: int | None = None,
) -> int:
    """Return a power-of-``base`` padded bucket for low-cardinality shapes."""

    value = int(value)
    minimum = int(minimum)
    base = int(base)
    if base < 2:
        raise ValueError(f"base must be at least 2, got {base}")
    if value <= 0:
        return max(1, minimum)
    target = max(value, minimum)
    bucket = 1
    while bucket < target:
        bucket *= base
    if maximum is not None:
        bucket = min(bucket, int(maximum))
    return max(bucket, value, minimum)


def power_of_two_bucket(value: int, *, minimum: int = 1, maximum: int | None = None) -> int:
    """Return a power-of-two padded bucket for low-cardinality shape classes."""

    return power_bucket(value, base=2, minimum=minimum, maximum=maximum)


def coarse_bucket(value: int, *, small_power2_max: int, large_multiple: int, minimum: int = 1) -> int:
    """Bucket small values by power-of-two and large values by coarse multiples."""

    value = int(value)
    if value <= int(small_power2_max):
        return power_of_two_bucket(value, minimum=minimum, maximum=small_power2_max)
    return round_up_to_multiple(value, large_multiple)


def pad_axis(array, axis: int, size: int, *, value=0):
    """Pad one axis to ``size`` without changing existing values."""

    arr = np.asarray(array)
    axis = int(axis)
    size = int(size)
    if axis < 0:
        axis += arr.ndim
    if axis < 0 or axis >= arr.ndim:
        raise ValueError(f"axis {axis} out of bounds for array with ndim={arr.ndim}")
    if arr.shape[axis] > size:
        raise ValueError(f"cannot pad axis {axis} from {arr.shape[axis]} down to {size}")
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (0, size - arr.shape[axis])
    return np.pad(arr, pad_width, mode="constant", constant_values=value)
