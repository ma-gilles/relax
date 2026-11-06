"""Host memory a process may use (physical memory, capped by the job's cgroup limit), and returning freed heap."""

import ctypes
import functools
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _cgroup_limits(group, root, name):
    """The limits in file ``name`` of cgroup directory ``group`` and of its ancestors up to ``root``."""
    limits = []
    for directory in (group, *group.parents):
        try:
            value = (directory / name).read_text().strip()
        except OSError:
            value = "max"
        if value != "max":
            limits.append(int(value))
        if directory == root:
            break
    return limits


def _cgroup_memory_limit(cgroup_file=Path("/proc/self/cgroup"), root=Path("/sys/fs/cgroup")):
    """The tightest memory limit over this process's cgroup ancestors (a Slurm job's), or None.

    cgroup v2 keeps the limit in ``memory.max`` under the ``0::`` path; cgroup v1 in
    ``memory/<path>/memory.limit_in_bytes`` under the ``memory`` controller's path, a huge value when unlimited.
    """
    try:
        lines = cgroup_file.read_text().splitlines()
    except OSError:
        return None
    limits = []
    for line in lines:
        _, controllers, relative = line.split(":", 2)
        relative = relative.strip().lstrip("/")
        if controllers == "":
            limits += _cgroup_limits(root / relative, root, "memory.max")
        elif "memory" in controllers.split(","):
            limits += _cgroup_limits(root / "memory" / relative, root / "memory", "memory.limit_in_bytes")
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
