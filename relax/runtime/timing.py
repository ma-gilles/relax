"""Wall-clock timing for logs and run reports.

``Stopwatch`` is the one timer of the refinement (deep review, duplicates concept 6): it reads
``time.perf_counter`` and never synchronises a device, so a duration it reports for asynchronous JAX work is
the host's dispatch time unless the code it measures already blocks. A caller that wants device time blocks
explicitly where the algorithm does.
"""

from __future__ import annotations

import time


class Stopwatch:
    """Seconds since construction, on ``time.perf_counter``; ``seconds`` may be read any number of times."""

    __slots__ = ("_start",)

    def __init__(self) -> None:
        self._start = time.perf_counter()

    @property
    def seconds(self) -> float:
        return time.perf_counter() - self._start
