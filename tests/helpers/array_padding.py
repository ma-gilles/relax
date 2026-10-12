"""Test helper: pad one array axis (moved from relax/runtime/shape_buckets.py, PLAN e1; no relax module uses it)."""

import numpy as np


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
