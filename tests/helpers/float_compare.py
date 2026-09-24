"""Equality checks for test arrays: exact for discrete values, banded for floats.

No test requires bitwise or ULP-exact equality of floating-point values (user rule,
2026-09-24): GPU reductions race, and even deterministic CPU paths change their last bits
with a compiler, library or fusion change. Integers, booleans, strings and other discrete
values (shapes, counts, indices, capacities) are still compared exactly.

A floating-point pair passes when every element satisfies

    |actual - desired| <= rtol * max(max|actual|, max|desired|)

over the finite entries, with NaN and infinite entries required at the same positions with
the same values. The scale is the array's largest magnitude rather than each element's,
because reduction noise is proportional to the magnitude of the summed terms, so a small
element next to large ones carries the large ones' absolute error.

The default ``rtol`` per precision is sized to the measured same-code noise with a margin:
float32 races differ by about one ULP (6e-8 relative, e.g. the local exact noise core and the
resident ``wsum_norm_correction`` at 3.5e-8), float64 ones by a few ULP (2e-16 to 5e-16,
e.g. the local noise pixel capacity and the rigid round-trip). A test that measured more
noise passes its own ``rtol`` and says where the value comes from.
"""

from __future__ import annotations

import numpy as np

RTOL = {
    np.dtype(np.float16): 1e-3,
    np.dtype(np.float32): 1e-6,
    np.dtype(np.float64): 1e-13,
}


def _as_array(value) -> np.ndarray:
    return np.asarray(value)


def _is_float(array: np.ndarray) -> bool:
    return array.dtype.kind in "fc"


def default_rtol(*arrays) -> float:
    """The loosest default band among the floating-point operands (the least precise one)."""
    rtols = []
    for array in arrays:
        dtype = array.dtype
        if dtype.kind == "c":
            dtype = np.finfo(dtype).dtype
        if dtype.kind == "f":
            rtols.append(RTOL.get(np.dtype(dtype), RTOL[np.dtype(np.float32)]))
    return max(rtols) if rtols else 0.0


def mismatch(actual, desired, *, rtol: float | None = None) -> str | None:
    """Why ``actual`` does not match ``desired``, or None when it does."""
    a, d = _as_array(actual), _as_array(desired)
    if a.shape != d.shape:
        if a.ndim and d.ndim:
            return f"shape {a.shape} != {d.shape}"
        a, d = np.broadcast_arrays(a, d)
    if not (_is_float(a) or _is_float(d)):
        if a.dtype.kind in "OUSV" or d.dtype.kind in "OUSV":
            same = bool(np.all(a == d)) if a.size else True
        else:
            same = bool(np.array_equal(a, d))
        return None if same else f"{int(np.count_nonzero(a != d))} of {a.size} discrete values differ"
    tol = default_rtol(*(x for x in (a, d) if _is_float(x))) if rtol is None else float(rtol)
    a = a.astype(np.result_type(a.dtype, np.float64) if a.dtype.kind != "c" else np.complex128)
    d = d.astype(np.result_type(d.dtype, np.float64) if d.dtype.kind != "c" else np.complex128)
    fa, fd = np.isfinite(a), np.isfinite(d)
    if not np.array_equal(fa, fd):
        return "non-finite entries at different positions"
    nonfinite = ~fa
    if np.any(nonfinite):
        na, nd = a[nonfinite], d[nonfinite]
        if not np.array_equal(np.isnan(na), np.isnan(nd)) or not np.array_equal(na[~np.isnan(na)], nd[~np.isnan(nd)]):
            return "non-finite entries differ"
    if not np.any(fa):
        return None
    diff = np.abs(a[fa] - d[fa])
    scale = max(float(np.max(np.abs(a[fa]))), float(np.max(np.abs(d[fa]))))
    worst = float(np.max(diff))
    if worst <= tol * scale:
        return None
    return f"max |diff| {worst:.3e} > rtol {tol:g} x scale {scale:.3e} ({int(np.count_nonzero(diff > tol * scale))} entries)"


def matches(actual, desired, *, rtol: float | None = None, equal_nan: bool = True) -> bool:
    """True when ``actual`` matches ``desired``: exactly if discrete, within the band if float.

    NaN entries must sit at the same positions (``equal_nan`` is accepted for drop-in use).
    """
    return mismatch(actual, desired, rtol=rtol) is None


def assert_matches(
    actual, desired, err_msg: str = "", verbose: bool = True, *, rtol: float | None = None, strict: bool = False
) -> None:
    """Assert ``matches(actual, desired)``; the message names the largest difference.

    The signature follows ``numpy.testing.assert_array_equal``; ``strict`` also requires equal
    shapes and dtypes.
    """
    if strict:
        a, d = _as_array(actual), _as_array(desired)
        assert a.shape == d.shape and a.dtype == d.dtype, f"{err_msg}: {a.shape}/{a.dtype} != {d.shape}/{d.dtype}"
    reason = mismatch(actual, desired, rtol=rtol)
    if reason is not None:
        raise AssertionError(f"{err_msg}: {reason}" if err_msg else reason)


def assert_trees_match(actual, desired, *, rtol: float | None = None, err_msg: str = "") -> None:
    """``assert_matches`` over matching dict/list/tuple structures of arrays and scalars."""
    if isinstance(desired, dict):
        assert isinstance(actual, dict) and actual.keys() == desired.keys(), f"{err_msg}: keys differ"
        for key in desired:
            assert_trees_match(actual[key], desired[key], rtol=rtol, err_msg=f"{err_msg}[{key!r}]")
    elif isinstance(desired, (list, tuple)) and not np.isscalar(desired):
        assert len(actual) == len(desired), f"{err_msg}: lengths differ"
        for index, (x, y) in enumerate(zip(actual, desired)):
            assert_trees_match(x, y, rtol=rtol, err_msg=f"{err_msg}[{index}]")
    else:
        assert_matches(actual, desired, rtol=rtol, err_msg=err_msg)


def flip_fraction(actual, desired) -> float:
    """Fraction of discrete entries that differ (hard assignments, significance flags)."""
    a, d = _as_array(actual), _as_array(desired)
    assert a.shape == d.shape, f"shape {a.shape} != {d.shape}"
    return float(np.count_nonzero(a != d)) / max(a.size, 1)
