"""The hooks pass 1 calls for its diagnostics (code rule 15): dumps and audits that observe the pass, never steer it.

Pass 1 receives a :class:`Pass1Probe` in its request (``Pass1Request.probe``) and never imports an implementation;
:class:`NoProbe`, the default, observes nothing. ``relax.diagnostics.coarse_gaussian_diagnostics.SignificanceDiagnostics``
is the implementation the commands build: the ``RELAX_SIGNIFICANCE_DUMP_*`` score dumps and the
``RECOVAR_COARSE_SIGNIFICANCE_SUPPORT_AUDIT`` support hash, each switched on by the environment when it is called.
"""

from __future__ import annotations

from typing import Protocol


class Pass1Probe(Protocol):
    """What pass 1 asks its diagnostics, and what it hands them."""

    def score_dump_requested(self, *, current_size, debug_iteration) -> bool:
        """Whether this pass (its ``current_size`` and ``debug_iteration``) dumps the scores of chosen images. A dump
        batch publishes at once and keeps its support mask on the host; its results are the same."""

    def score_dump_rows(self, experiment_dataset, indices):
        """The rows of the batch of dataset images ``indices`` whose scores the dump records (int64), or None."""

    def significance_batch_scored(self, **values) -> None:
        """A dump batch was published: its weights, support, scores and coarse operands
        (``coarse_gaussian_diagnostics.maybe_dump_k_class_significance_batch`` names them)."""

    def tree_rescore_scored(self, **values) -> None:
        """The tree rescore scored the two candidates of a batch's ambiguous images
        (``coarse_gaussian_diagnostics.maybe_dump_tree_rescore_batch`` names the values)."""

    def support_audit(self, significant_sample_indices, *, samples_per_class: int) -> dict | None:
        """The audit of the pass's per-class significant samples that its statistics report, or None."""


class NoProbe:
    """The :class:`Pass1Probe` that observes nothing: no dump, no audit."""

    def score_dump_requested(self, *, current_size, debug_iteration) -> bool:
        return False

    def score_dump_rows(self, experiment_dataset, indices):
        return None

    def significance_batch_scored(self, **values) -> None:
        pass

    def tree_rescore_scored(self, **values) -> None:
        pass

    def support_audit(self, significant_sample_indices, *, samples_per_class: int) -> dict | None:
        return None
