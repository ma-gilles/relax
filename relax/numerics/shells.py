"""The integer Fourier shell of every voxel: the one home of the per-voxel shell rule.

Callers keep their own shell sums (``bincount``, ``np.add.at`` and the fixed-order
reductions of :mod:`relax.numerics.deterministic_reduce` sum in different orders), their
masks and their pair weights; they take the shell labels from here.

Rules (radii are non-negative):

- ``"half_up"``: ``floor(r + 0.5)``, RELION's ``ROUND`` (macros.h) for ``r >= 0``.
- ``"half_even"``: ``rint(r)``, numpy's ``round`` and CUDA's ``__float2int_rn``.
- ``"floor"``: ``floor(r)``.

The two rounding rules differ only on an exact half-integer radius, which a padded radius
``sqrt(n) / padding_factor`` can be, and a float32 ``sqrt`` of a large integer can round to;
so each caller names its rule and the precision of its ``sqrt``.
"""

from __future__ import annotations

import numpy as np

SHELL_RULES = ("half_up", "half_even", "floor")
GRID_ROWS = ("centred", "fftw")


def shell_of_radius(radius, *, rule: str, padding_factor=None, index_dtype=np.int64, xp=np):
    """Shell of each radius, computed in the radius array's own dtype, cast to ``index_dtype``.

    ``padding_factor`` divides the radius first (a padded grid binned into the original
    size's shells); ``None`` skips the division. ``xp`` is ``numpy`` or ``jax.numpy``; with
    ``jax.numpy`` the operations are traced in place (no transfer, no new jit region).
    """

    if padding_factor is not None:
        radius = radius / padding_factor
    if rule == "half_up":
        shells = xp.floor(radius + 0.5)
    elif rule == "half_even":
        shells = xp.rint(radius)
    elif rule == "floor":
        shells = xp.floor(radius)
    else:
        raise ValueError(f"rule must be one of {SHELL_RULES}, got {rule!r}")
    return shells.astype(index_dtype)


def shell_of_radius_sq(
    radius_sq, *, rule: str, real_dtype=np.float64, padding_factor=None, index_dtype=np.int64, xp=np
):
    """:func:`shell_of_radius` of ``sqrt(radius_sq)``, the ``sqrt`` taken in ``real_dtype``."""

    return shell_of_radius(
        xp.sqrt(radius_sq.astype(real_dtype)),
        rule=rule,
        padding_factor=padding_factor,
        index_dtype=index_dtype,
        xp=xp,
    )


def _axis_frequencies(size: int, rows: str) -> np.ndarray:
    index = np.arange(size, dtype=np.int64)
    if rows == "centred":
        return index - size // 2
    if rows == "fftw":
        return np.where(index <= size // 2, index, index - size)
    raise ValueError(f"rows must be one of {GRID_ROWS}, got {rows!r}")


def fourier_radius_sq(shape, *, rows: str, half: bool) -> np.ndarray:
    """``|k|^2`` (int64) of every voxel of a 2-D or 3-D Fourier grid of real-space ``shape``.

    ``rows`` orders every axis but a half last axis: ``"centred"`` (``-(n // 2)`` first) or
    ``"fftw"`` (``0`` first). The sign of a Nyquist frequency does not enter ``|k|^2``.
    ``half`` keeps the non-negative ``shape[-1] // 2 + 1`` frequencies of the last axis
    (the rfft layout).
    """

    shape = tuple(int(s) for s in shape)
    axes = [_axis_frequencies(size, rows) for size in shape[:-1]]
    axes.append(np.arange(shape[-1] // 2 + 1, dtype=np.int64) if half else _axis_frequencies(shape[-1], rows))
    radius_sq = np.zeros(tuple(axis.size for axis in axes), dtype=np.int64)
    for dim, axis in enumerate(axes):
        view = [1] * len(axes)
        view[dim] = axis.size
        radius_sq = radius_sq + axis.reshape(view) ** 2
    return radius_sq


def shell_index(
    shape, *, rows: str, half: bool, rule: str, real_dtype=np.float64, padding_factor=None, index_dtype=np.int64
) -> np.ndarray:
    """Host shell label of every voxel of the grid :func:`fourier_radius_sq` describes."""

    return shell_of_radius_sq(
        fourier_radius_sq(shape, rows=rows, half=half),
        rule=rule,
        real_dtype=real_dtype,
        padding_factor=padding_factor,
        index_dtype=index_dtype,
    )
