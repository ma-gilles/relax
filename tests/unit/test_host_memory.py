"""relax.runtime.host_memory.return_freed_heap: glibc malloc_trim, a no-op elsewhere (relax#34)."""

import logging

import numpy as np
import pytest

from relax.runtime import host_memory

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


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_cgroup_v2_limit_is_the_tightest_ancestor(tmp_path):
    root = tmp_path / "cgroup"
    _write(root / "slurm/job_1/memory.max", "1000\n")
    _write(root / "slurm/job_1/step/memory.max", "max\n")
    _write(root / "slurm/memory.max", "2000\n")
    _write(tmp_path / "self", "0::/slurm/job_1/step\n")
    assert host_memory._cgroup_memory_limit(tmp_path / "self", root) == 1000


def test_cgroup_v1_limit_is_read_from_the_memory_controller(tmp_path):
    """Only the v1 hierarchy: without this limit a Slurm job plans against the node's memory and is OOM-killed."""
    root = tmp_path / "cgroup"
    _write(root / "memory/slurm/job_1/step/task_0/memory.limit_in_bytes", "9223372036854771712\n")
    _write(root / "memory/slurm/job_1/memory.limit_in_bytes", "274877906944\n")
    _write(root / "memory/slurm/memory.limit_in_bytes", "9223372036854771712\n")
    _write(root / "memory/memory.limit_in_bytes", "9223372036854771712\n")
    _write(tmp_path / "self", "12:cpuset:/\n3:pids:/x\n2:memory:/slurm/job_1/step/task_0\n1:name=systemd:/x\n")
    assert host_memory._cgroup_memory_limit(tmp_path / "self", root) == 274877906944


def test_cgroup_without_memory_limit(tmp_path):
    assert host_memory._cgroup_memory_limit(tmp_path / "missing", tmp_path) is None
    _write(tmp_path / "self", "1:name=systemd:/x\n")
    assert host_memory._cgroup_memory_limit(tmp_path / "self", tmp_path) is None
