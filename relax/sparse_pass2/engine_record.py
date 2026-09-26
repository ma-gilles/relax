"""Which engine each K=1 E-step pass of the current iteration ran, for the run's results.

The resident drivers are the K=1 default, and a pass they do not cover runs on the earlier
engine with a logged reason (``resident_engine_selection``). The log alone makes a fallback on
a real dataset easy to miss, so the routing also appends one entry per pass here, and the
iteration loop moves the entries into the per-iteration ``pass2_engine_trajectory`` of the
results. Entries read ``"<pass>:<engine>"`` or ``"<pass>:<engine> (<reason>)"``.

Every engine other than the resident one is deprecated (docs/development/em_status.md,
"One engine: removal TODO"). :func:`warn_deprecated_engine` logs one warning per engine,
pass kind and reason the first time a run routes a pass to one of them.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_entries: list[str] = []

# Record labels of the deprecated engines, and what they are.
_DEPRECATED_ENGINES = {
    "compact": "the compact (bucketed sparse) pass 2",
    "exact_local": "the exact local engine",
    "local": "the VDAM exact-local E-step (exact local engine)",
    "dense": "the dense run_em engine",
    "per_image_reference": "the per-image reference pass 2 (dense run_em per image)",
}

_warned: set[tuple[str, str, str]] = set()


def warn_deprecated_engine(engine: str, pass_kind: str, reason: str) -> None:
    """Log once per engine, pass kind and reason that a pass ran on a deprecated engine.

    Numbers in the reason (sizes, budgets) do not make a new warning, so a refusal that
    repeats every iteration with other byte counts is logged once.
    """

    key = (engine, pass_kind, re.sub(r"\d+(\.\d+)?", "#", reason))
    if key in _warned:
        return
    _warned.add(key)
    logger.warning(
        "DEPRECATED engine: a %s pass runs on %s because %s. relax keeps one pass-2 engine, the "
        "resident one; this engine is removed once the resident engine covers this case "
        "(docs/development/em_status.md, 'One engine: removal TODO').",
        pass_kind,
        _DEPRECATED_ENGINES.get(engine, engine),
        reason,
    )


def record_pass_engine(pass_kind: str, engine: str, reason: str | None = None) -> None:
    """Append the engine one pass ran on (``pass_kind`` is ``global`` or ``local``).

    A non-resident engine with a reason also logs the deprecation warning; a route that
    records no reason warns at its call site.
    """

    if engine != "resident" and reason:
        warn_deprecated_engine(engine, pass_kind, reason)
    _entries.append(f"{pass_kind}:{engine}" if not reason else f"{pass_kind}:{engine} ({reason})")


def take_pass_engines() -> list[str]:
    """Return the entries recorded since the last call and clear them."""

    entries = list(_entries)
    _entries.clear()
    return entries
