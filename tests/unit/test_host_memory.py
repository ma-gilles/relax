"""relax.helpers.host_memory.return_freed_heap: glibc malloc_trim, a no-op elsewhere (relax#34)."""

import logging

import numpy as np
import pytest

from relax.helpers import host_memory

pytestmark = pytest.mark.unit


def test_freed_heap_returns_and_logs_the_resident_drop(caplog):
    if host_memory._glibc_malloc_trim() is None:
        pytest.skip("not glibc")
    blocks = [np.ones(1 << 17) for _ in range(512)]  # 512 x 1 MiB, below the mmap threshold: on the heap
    del blocks
    with caplog.at_level(logging.INFO, logger=host_memory.__name__):
        dropped = host_memory.return_freed_heap("a test")
    assert isinstance(dropped, int)
    assert "Returned freed host heap after a test: resident" in caplog.text


def test_freed_heap_is_a_no_op_off_glibc(monkeypatch, caplog):
    monkeypatch.setattr(host_memory, "_glibc_malloc_trim", lambda: None)
    with caplog.at_level(logging.DEBUG, logger=host_memory.__name__):
        assert host_memory.return_freed_heap("a test") == 0
    assert caplog.text == ""
