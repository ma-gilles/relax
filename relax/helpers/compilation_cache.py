"""recovar's persistent JAX compilation cache at relax entry points.

``recovar.jax_config`` chooses the cache directory (``RECOVAR_JAX_CACHE_DIR``
or ``~/.cache/recovar/jax_compile``) and its 10 ms threshold through
environment variables, which JAX reads only when it is imported. An entry
point that imports JAX before recovar therefore runs with the cache off
(directory None, JAX's 1 s threshold). :func:`activate_recovar_compilation_cache`
applies that environment to the live JAX config.

JAX never evicts without ``jax_compilation_cache_max_size``, and its LRU
eviction scans the whole directory on every write, which is unusable for
EM's thousands of small programs. Eviction policy here instead: at most once
per :data:`PRUNE_INTERVAL_S`, a background thread started by the entry point
keeps the :data:`MAX_ENTRIES` most recently written entries and deletes the
rest, oldest first (first in, first out: a pruned entry is recompiled and
written again on its next use). The shared default directory held 735k entries
(4.0 GB) on 2026-09-26.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_ENTRIES = 100_000
PRUNE_INTERVAL_S = 6 * 3600
_CACHE_SUFFIX = "-cache"
_ATIME_SUFFIX = "-atime"
_PRUNE_STAMP = ".relax_prune_stamp"


def activate_recovar_compilation_cache(*, prune_in_background: bool = True) -> str | None:
    """Apply recovar's cache environment to the live JAX config; return the directory.

    ``RECOVAR_DISABLE_JAX_CACHE`` leaves no directory in the environment, so the
    config is then left alone.
    """

    import jax

    directory = os.environ.get("JAX_COMPILATION_CACHE_DIR", "").strip()
    if not directory:
        return None
    jax.config.update("jax_compilation_cache_dir", directory)
    jax.config.update(
        "jax_persistent_cache_min_compile_time_secs",
        float(os.environ.get("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0.01")),
    )
    if prune_in_background:
        start_background_prune(directory)
    return directory


def start_background_prune(directory, *, max_entries: int = MAX_ENTRIES) -> threading.Thread | None:
    """Prune ``directory`` on a daemon thread unless it was pruned within the interval."""

    if not _claim_prune(Path(directory)):
        return None
    thread = threading.Thread(
        target=_prune_quietly, args=(Path(directory), max_entries), name="relax-jax-cache-prune", daemon=True
    )
    thread.start()
    return thread


def prune_compilation_cache(directory, *, max_entries: int = MAX_ENTRIES) -> int:
    """Delete all but the ``max_entries`` most recently written entries; return how many were deleted."""

    directory = Path(directory)
    entries = []
    with os.scandir(directory) as scan:
        for entry in scan:
            if entry.name.endswith(_CACHE_SUFFIX):
                try:
                    entries.append((entry.stat().st_mtime_ns, entry.name))
                except FileNotFoundError:
                    continue
    excess = len(entries) - int(max_entries)
    if excess <= 0:
        return 0
    entries.sort()
    for _mtime, name in entries[:excess]:
        key = name[: -len(_CACHE_SUFFIX)]
        for path in (directory / name, directory / f"{key}{_ATIME_SUFFIX}"):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    return excess


def _claim_prune(directory: Path) -> bool:
    stamp = directory / _PRUNE_STAMP
    try:
        if time.time() - stamp.stat().st_mtime < PRUNE_INTERVAL_S:
            return False
    except FileNotFoundError:
        pass
    try:
        directory.mkdir(parents=True, exist_ok=True)
        stamp.touch()
    except OSError:
        return False
    return True


def _prune_quietly(directory: Path, max_entries: int) -> None:
    try:
        deleted = prune_compilation_cache(directory, max_entries=max_entries)
    except OSError as error:
        logger.warning("JAX compilation cache prune of %s failed: %s", directory, error)
        return
    if deleted:
        logger.info("Pruned %d oldest JAX compilation cache entries from %s", deleted, directory)
