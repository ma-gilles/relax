"""The slack every structure ceiling of the repository applies (owner ruling, 2026-10-05).

Ceilings (``docs/development/refinement_structure_metrics.json``, the VDAM line budgets in
``tests/unit/initial_model/test_refactor_invariants.py``, any later one) are review signals, not exact
limits. For a ceiling ``c`` the slack is ``max(SLACK_MINIMUM, ceil(c * SLACK_PERCENT / 100))``: 5% of the
ceiling, rounded up, and at least 1. A total above ``c`` but at most ``c + slack`` warns, naming the
metric; only a total above ``c + slack`` fails. Every ceiling test applies the slack through
``exceeded`` and ``within_slack`` here; a ceiling checked without slack is a defect.
"""

from __future__ import annotations

SLACK_PERCENT = 5
SLACK_MINIMUM = 1


def slack_for(ceiling: int) -> int:
    """How far a total may pass ``ceiling`` before the check fails."""
    return max(SLACK_MINIMUM, -(-ceiling * SLACK_PERCENT // 100))


def exceeded(totals: dict, ceilings: dict) -> list[str]:
    """One line per total above its ceiling plus slack; empty when the check passes."""
    return [
        f"{name}: {totals[name]:,} exceeds the ceiling {ceiling:,} by more than its slack {slack_for(ceiling):,}"
        for name, ceiling in ceilings.items()
        if totals[name] > ceiling + slack_for(ceiling)
    ]


def within_slack(totals: dict, ceilings: dict) -> list[str]:
    """One warning per total above its ceiling but within its slack."""
    return [
        f"warning: {name}: {totals[name]:,} is above the ceiling {ceiling:,}, within its slack {slack_for(ceiling):,}"
        for name, ceiling in ceilings.items()
        if ceiling < totals[name] <= ceiling + slack_for(ceiling)
    ]
