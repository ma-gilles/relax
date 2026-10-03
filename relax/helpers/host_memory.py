"""Host memory a process may use: physical memory, capped by the job's cgroup v2 limit."""

import os
from pathlib import Path


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
