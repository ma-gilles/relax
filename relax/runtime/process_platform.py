"""The process's platform switches, read where they are used (rule 16's process-level exception).

These variables choose the platform of the whole process, not a setting of one run: JAX reads
``JAX_PLATFORMS``/``JAX_PLATFORM_NAME`` when its backend starts, and recovar's CUDA loader reads
``RECOVAR_DISABLE_CUDA`` whenever it is asked for a kernel. relax asks the same question at call time, through
this module only, so its choice agrees with what JAX and the loader do in the same process (GPU tests change
them between cases). A command does not copy them into a record: a copy taken at the command could disagree
with the loader that reads the live value.
"""

from __future__ import annotations

import os


def jax_cpu_forced() -> bool:
    """Whether ``JAX_PLATFORM_NAME`` or ``JAX_PLATFORMS`` forces JAX onto the CPU."""

    platform_name = os.environ.get("JAX_PLATFORM_NAME", "").strip().lower()
    if platform_name == "cpu":
        return True
    platforms = os.environ.get("JAX_PLATFORMS", "").strip().lower()
    if not platforms:
        return False
    requested = [token.strip() for token in platforms.split(",") if token.strip()]
    return bool(requested) and all(token == "cpu" for token in requested)
