"""Custom-call scratch inside JAX's allocator (relax#120).

relax's CUDA handlers take their device scratch from XLA's FFI scratch allocator
(relax/cuda/ffi_scratch.cuh), which carves it from JAX's preallocated pool. The CUDA default
memory pool that ``cudaMallocAsync`` draws from lies outside that pool: on a 16 GB card only
about 0.24 GiB is left there, and the resident planners cannot count it.

``clean_and_poisoned`` runs a handler call twice, the second time after every free byte of
JAX's pool was written with NaN and released, so a handler that read its scratch before
writing it, or took less than it uses, changes its result. It also returns the bytes the
driver's default memory pool handed out during both calls, which must be zero.
"""

from __future__ import annotations

import ctypes

import jax
import jax.numpy as jnp
import numpy as np

_CU_MEMPOOL_ATTR_USED_MEM_CURRENT = 7
_CU_MEMPOOL_ATTR_USED_MEM_HIGH = 8
_POISON_BLOCK_BYTES = 256 << 20


def _driver_call(driver, name, *args):
    status = getattr(driver, name)(*args)
    if status != 0:
        raise RuntimeError(f"{name} returned CUDA driver status {status}")


class _DefaultMemPool:
    """The driver's default memory pool of ``device``: its used bytes and their high-water."""

    def __init__(self, device):
        self._driver = ctypes.CDLL("libcuda.so.1")
        handle = ctypes.c_int()
        _driver_call(self._driver, "cuDeviceGet", ctypes.byref(handle), device.local_hardware_id)
        self._pool = ctypes.c_void_p()
        _driver_call(self._driver, "cuDeviceGetDefaultMemPool", ctypes.byref(self._pool), handle)

    def _get(self, attribute):
        value = ctypes.c_uint64()
        _driver_call(self._driver, "cuMemPoolGetAttribute", self._pool, attribute, ctypes.byref(value))
        return value.value

    def reset_high(self):
        """Lower the high-water to the current used bytes; returns them."""
        zero = ctypes.c_uint64(0)
        _driver_call(
            self._driver, "cuMemPoolSetAttribute", self._pool, _CU_MEMPOOL_ATTR_USED_MEM_HIGH, ctypes.byref(zero)
        )
        return self._get(_CU_MEMPOOL_ATTR_USED_MEM_CURRENT)

    def high(self):
        return self._get(_CU_MEMPOOL_ATTR_USED_MEM_HIGH)


def poison_free_pool(device) -> None:
    """Write NaN over (nearly) every free byte of JAX's pool on ``device`` and release it again."""
    stats = device.memory_stats() or {}
    free = int(stats.get("bytes_limit", 0)) - int(stats.get("bytes_in_use", 0))
    blocks = []
    with jax.default_device(device):
        for _ in range(max(0, free - _POISON_BLOCK_BYTES) // _POISON_BLOCK_BYTES):
            blocks.append(jnp.full(_POISON_BLOCK_BYTES // 4, jnp.nan, jnp.float32))
    jax.block_until_ready(blocks)
    del blocks


def _host(result):
    return jax.tree.map(np.asarray, jax.device_get(result))


def clean_and_poisoned(call, device):
    """``(clean, poisoned, default_pool_bytes)`` for ``call()`` (a pytree of device arrays).

    ``default_pool_bytes`` is the high-water of the driver's default memory pool over both calls
    above the bytes it held before them.
    """
    pool = _DefaultMemPool(device)
    jax.block_until_ready(jnp.zeros((), jnp.float32, device=device))
    held = pool.reset_high()
    clean = _host(call())
    poison_free_pool(device)
    poisoned = _host(call())
    return clean, poisoned, pool.high() - held
