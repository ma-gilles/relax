"""The hooks the adaptive K-class E-step calls for its diagnostics (code rule 15): they observe, never steer.

:func:`relax.classification.k_class.run_dense_k_class_em_adaptive` takes a :class:`KClassProbe` (``probe``) and hands
it to each of its pass-1 requests, so a K-class probe is also a :class:`relax.scoring.probes.Pass1Probe`. The default
:class:`NoProbe` observes nothing. ``relax.diagnostics.k_class_diagnostics.KClassDiagnostics`` is the implementation
the commands build: the environment's score-dump labels and the ``RELAX_GLOBAL_WINNER_SUMMARY_*`` summary, on top of
pass 1's dumps and support audit.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import ContextManager, Protocol

from relax.scoring.probes import NoProbe as Pass1NoProbe
from relax.scoring.probes import Pass1Probe


class KClassProbe(Pass1Probe, Protocol):
    """What the K-class E-step asks its diagnostics, and what it hands them, besides pass 1's hooks."""

    def coarse_score_dump_label(self) -> ContextManager:
        """The scope of the first-iteration coarse probe, which labels the per-pose score dumps written inside it."""

    def wants_class_runner_up(self) -> bool:
        """Whether the first-iteration coarse probe also returns each class's runner-up (for
        :meth:`firstiter_winners_scored`); the winners do not depend on it."""

    def firstiter_winners_scored(self, **values) -> None:
        """The first-iteration coarse probe chose its winners: its dataset, pass-1 statistics, class assignments,
        grid sizes and iteration (``relax.diagnostics.global_winner_summary.maybe_dump_global_winner_summary``)."""


class NoProbe(Pass1NoProbe):
    """The :class:`KClassProbe` that observes nothing."""

    def coarse_score_dump_label(self) -> ContextManager:
        return nullcontext()

    def wants_class_runner_up(self) -> bool:
        return False

    def firstiter_winners_scored(self, **values) -> None:
        pass
