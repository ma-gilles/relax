"""Host memory a process may use (physical memory, capped by the job's cgroup v2 limit), and returning freed heap."""

import ctypes
import functools
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _cgroup_memory_limit():
    """The tightest memory limit over this process's cgroup v2 ancestors (a Slurm job's), or None."""
    try:
        relative = next(
            line.split(":", 2)[2].strip()
            for line in Path("/proc/self/cgroup").read_text().splitlines()
            if line.startswith("0::")
        )
    except (OSError, StopIteration):
        return None
    limits = []
    group = Path("/sys/fs/cgroup") / relative.lstrip("/")
    for directory in (group, *group.parents):
        try:
            value = (directory / "memory.max").read_text().strip()
        except OSError:
            continue
        if value != "max":
            limits.append(int(value))
        if directory == Path("/sys/fs/cgroup"):
            break
    return min(limits) if limits else None


def available_memory_bytes():
    """Physical memory, capped by the job's cgroup limit when there is one."""
    physical = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    limit = _cgroup_memory_limit()
    return physical if limit is None else min(physical, limit)


def resident_bytes():
    """This process's resident set size."""
    return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")


@functools.cache
def _glibc_malloc_trim():
    """glibc's ``malloc_trim``, or None off glibc."""
    try:
        libc = ctypes.CDLL(None)
        libc.gnu_get_libc_version  # only glibc exports it
        trim = libc.malloc_trim
    except (OSError, AttributeError):
        return None
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int
    return trim


def return_freed_heap(where: str, *, level: int = logging.INFO) -> int:
    """Hand the heap's free pages back to the system (glibc ``malloc_trim(0)``, every arena); a no-op off glibc.

    A pass whose host work runs in a thread pool (one thread per class) leaves freed blocks in each thread's malloc
    arena, so the resident set keeps growing over a pass's same-sized table blocks although the live data does not
    (relax#34: Class3D K4 10k, 16.3 GiB peak by default against 13.7 GiB with two arenas). Logs the resident set
    before and after at ``level`` and returns the bytes it dropped (0 off glibc).
    """

    trim = _glibc_malloc_trim()
    if trim is None:
        return 0
    before, t0 = resident_bytes(), time.perf_counter()
    trim(0)
    seconds = time.perf_counter() - t0
    dropped = before - resident_bytes()
    logger.log(
        level,
        "Returned freed host heap after %s: resident %.2f -> %.2f GiB in %.3f s",
        where,
        before / 2**30,
        (before - dropped) / 2**30,
        seconds,
    )
    return dropped
