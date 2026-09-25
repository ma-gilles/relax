"""Which engine each K=1 E-step pass of the current iteration ran, for the run's results.

The resident drivers are the K=1 default, and a pass they do not cover runs on the earlier
engine with a logged reason (``resident_engine_selection``). The log alone makes a fallback on
a real dataset easy to miss, so the routing also appends one entry per pass here, and the
iteration loop moves the entries into the per-iteration ``pass2_engine_trajectory`` of the
results. Entries read ``"<pass>:<engine>"`` or ``"<pass>:<engine> (<reason>)"``.
"""

from __future__ import annotations

_entries: list[str] = []


def record_pass_engine(pass_kind: str, engine: str, reason: str | None = None) -> None:
    """Append the engine one pass ran on (``pass_kind`` is ``global`` or ``local``)."""

    _entries.append(f"{pass_kind}:{engine}" if not reason else f"{pass_kind}:{engine} ({reason})")


def take_pass_engines() -> list[str]:
    """Return the entries recorded since the last call and clear them."""

    entries = list(_entries)
    _entries.clear()
    return entries
