"""Counting the programs a ``jax.jit`` function compiles, in a process other tests share."""


def clear_programs(jitted) -> int:
    """Clear ``jitted``'s compiled programs and return the count jaxlib then reports for it.

    The count is not 0 in every process: after a ``jax.clear_caches()`` (the memory-map fixture of
    tests/conftest.py calls it in a long run) jaxlib 0.9.0.1 reports an existing function's programs with a
    negative offset, which clearing returns to. A test that compiles programs asserts how many the count
    grew by, ``jitted._cache_size() - cleared``, never the count itself.
    """
    jitted.clear_cache()
    return jitted._cache_size()
