"""The K-class E-step's diagnostics probe (:class:`relax.classification.probes.KClassProbe`) that the commands build.

Every hook reads its environment switches when it is called; with none set it writes nothing and only reads the
environment.
"""

from __future__ import annotations

import os

from relax.diagnostics.coarse_gaussian_diagnostics import SignificanceDiagnostics
from relax.diagnostics.global_winner_summary import maybe_dump_global_winner_summary
from relax.diagnostics.local_debug import score_dump_label


class KClassDiagnostics(SignificanceDiagnostics):
    """Pass 1's dumps and support audit (:class:`SignificanceDiagnostics`), the ``coarse`` label of the per-pose
    score dumps (:func:`relax.diagnostics.local_debug.score_dump_label`) and the first-iteration global-winner summary
    (``RELAX_GLOBAL_WINNER_SUMMARY_PATH``, :func:`maybe_dump_global_winner_summary`)."""

    def coarse_score_dump_label(self):
        return score_dump_label("coarse")

    def wants_class_runner_up(self) -> bool:
        return bool(os.environ.get("RELAX_GLOBAL_WINNER_SUMMARY_PATH", "").strip())

    def firstiter_winners_scored(self, **values) -> None:
        maybe_dump_global_winner_summary(**values)
