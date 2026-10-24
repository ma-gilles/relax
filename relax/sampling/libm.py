"""The C library's ``acos`` and ``atan2`` over arrays, as RELION's host code evaluates them.

NumPy dispatches some float64 functions to SIMD kernels whose rounding is not the
C library's: on della's Intel (AVX512) nodes ``np.arccos`` differs from glibc's
``acos`` in about 9% of arguments and ``np.arctan2`` in about 0.1% (a probe
run), while ``np.sin``, ``np.cos`` and ``np.sqrt`` agree. A one-ulp angle
decides exact cone and radius ties against RELION, so the sampling ports take
these two functions from the C library: ``librelax_host_math.so``
(``relax/host_math``), a host-only loop of the scalar ``acos``/``atan2`` that
CPython's :mod:`math` also calls, built without fast-math or vectorization. It
replaced a :func:`numpy.frompyfunc` over :mod:`math` (the same values, 0.12 us
an element: 52 s of a multi-optics VDAM run, 437M elements).

``RELAX_HOST_MATH_LIB`` pins a built library (the test tiers build it with the
other natives, ``scripts/build_test_natives.sh``); without it the library is built
on first use with the host C compiler (``make -C relax/host_math``), as the EM CUDA
library is (:class:`recovar.cuda_build.NativeLibrary`).
"""

from __future__ import annotations

import ctypes
import pathlib

import numpy as np
from recovar.cuda_build import NativeLibrary

_LIBRARY = NativeLibrary(
    name="relax host math",
    filename="librelax_host_math.so",
    make_dir=pathlib.Path(__file__).resolve().parents[1] / "host_math",
    source_names=("relax_host_math.c", "Makefile"),
    lib_env="RELAX_HOST_MATH_LIB",
    registrations=(),
    optional_registrations={},
)
_SYMBOLS = ("relax_host_atan2_f64", "relax_host_acos_f64")
_DOUBLES = ctypes.POINTER(ctypes.c_double)
_FUNCTIONS: dict = {}


def _functions():
    """The library's two loops, loaded (or built) once."""

    if not _FUNCTIONS:
        lib = _LIBRARY.get_lib()
        missing = [name for name in _SYMBOLS if not hasattr(lib, name)]
        if missing:
            raise RuntimeError(f"{_LIBRARY.loaded_path} lacks {missing}: rebuild it with make -C relax/host_math")
        atan2_loop, acos_loop = lib.relax_host_atan2_f64, lib.relax_host_acos_f64
        atan2_loop.argtypes = (_DOUBLES, _DOUBLES, _DOUBLES, ctypes.c_int64)
        acos_loop.argtypes = (_DOUBLES, _DOUBLES, ctypes.c_int64)
        atan2_loop.restype = acos_loop.restype = None
        _FUNCTIONS.update(atan2=atan2_loop, acos=acos_loop)
    return _FUNCTIONS


def _pointer(array: np.ndarray):
    return array.ctypes.data_as(_DOUBLES)


def acos(values) -> np.ndarray:
    """``acos`` of each element, float64, from the C library."""

    values = np.require(values, dtype=np.float64, requirements="C")
    if np.any(np.abs(values) > 1.0):
        # math.acos's domain error, which the frompyfunc this replaced raised; the C library would return NaN.
        raise ValueError("math domain error: acos of a value outside [-1, 1]")
    out = np.empty_like(values)
    _functions()["acos"](_pointer(values), _pointer(out), values.size)
    return out


def atan2(y, x) -> np.ndarray:
    """``atan2(y, x)`` of each element pair, float64, from the C library."""

    y, x = np.broadcast_arrays(np.asarray(y, dtype=np.float64), np.asarray(x, dtype=np.float64))
    y, x = np.require(y, requirements="C"), np.require(x, requirements="C")
    out = np.empty(y.shape, dtype=np.float64)
    _functions()["atan2"](_pointer(y), _pointer(x), _pointer(out), y.size)
    return out


__all__ = ["acos", "atan2"]
