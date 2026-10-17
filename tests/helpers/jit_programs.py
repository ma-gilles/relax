"""Counting the programs a ``jax.jit`` function compiles, in a process other tests share."""

import jax
from jax._src import pjit

# jaxlib keeps every jit function's compiled programs in two process-wide LRU lists of 8192 entries.
_GLOBAL_PROGRAM_LISTS = (pjit._cpp_pjit_cache_fun_only, pjit._cpp_pjit_cache_explicit_attributes)


def clear_programs(jitted) -> int:
    """Clear ``jitted``'s compiled programs and return the count jaxlib then reports for it.

    jaxlib 0.9.0.1 miscounts a function's programs once its shared LRU list is full: a new program evicts the
    least recently used program of the whole process, but the eviction decrements the count of the function
    that compiled, not the count of the function that lost the program (``EvictNode`` calls ``RemoveNode`` on
    the wrong cache in XLA bb760b04 xla/pjrt/lru_cache.h; fixed upstream). In a long run the counts drift by
    any amount, negative included, and a compile inside a full list does not change the compiling function's
    count at all.

    So this empties both lists first when either is over half full (``jax.clear_caches()``), leaving room for
    the test's programs: no eviction can happen while it counts. The drift a count already carries stays, and
    a test asserts how many programs it added, ``jitted._cache_size() - cleared``, never the count itself.
    """
    if any(programs.size() > programs.capacity() // 2 for programs in _GLOBAL_PROGRAM_LISTS):
        jax.clear_caches()
    jitted.clear_cache()
    return jitted._cache_size()
