"""clear_programs counts a function's new programs exactly after other functions filled jaxlib's LRU lists."""

import jax
import numpy as np
import pytest
from helpers.jit_programs import _GLOBAL_PROGRAM_LISTS, clear_programs

pytestmark = pytest.mark.unit


@jax.jit
def _fails_to_trace(x):
    raise ValueError("fills one LRU entry per shape without compiling")


@jax.jit
def _counted(x):
    return x + 1


def test_programs_counted_exactly_after_the_shared_lists_filled():
    _counted(np.ones(1, np.float32))
    # Each failed trace keeps an entry, so other programs fill the lists cheaply.
    capacity = max(programs.capacity() for programs in _GLOBAL_PROGRAM_LISTS)
    for size in range(2, capacity + 12):
        with pytest.raises(ValueError):
            _fails_to_trace(np.ones(size, np.float32))
    assert any(programs.size() == programs.capacity() for programs in _GLOBAL_PROGRAM_LISTS)
    cleared = clear_programs(_counted)
    for size in (5, 6, 7):
        _counted(np.ones(size, np.float32))
    _counted(np.ones(5, np.float32))
    assert _counted._cache_size() - cleared == 3
    _fails_to_trace.clear_cache()
    _counted.clear_cache()
