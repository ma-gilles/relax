"""Static-shape bucket helpers for dense RELION refinement.

JAX recompiles when array shapes change.  RELION-style pruning changes local
support sizes frequently, so hot EM loops should round those sizes to a small
set of padded shape classes and pass masks/counts for the valid rows.
"""

from __future__ import annotations


def round_up_to_multiple(value: int, multiple: int) -> int:
    """Round ``value`` up to a positive multiple."""

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
