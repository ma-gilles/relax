"""Which batches of pass 1 are score-dump batches, and which of their rows are dumped (the pass's probe decides)."""

from typing import NamedTuple

import numpy as np

from relax.scoring.probes import Pass1Probe


class DumpTargets(NamedTuple):
    """One batch's dump request: ``enabled`` is a dump batch; ``rows`` the batch rows whose scores are recorded.

    ``rows`` is ``None`` when no dump target image is in the batch (the batch still dumps the per-image results).
    """

    enabled: bool
    rows: np.ndarray | None


def select_dump_targets(
    probe: Pass1Probe, experiment_dataset, indices, *, collect_significance, current_size, debug_iteration
) -> DumpTargets:
    """The dump request of the batch of dataset images ``indices``.

    A dump batch needs the significant samples (``collect_significance``) and a pass the ``probe`` dumps
    (:meth:`~relax.scoring.probes.Pass1Probe.score_dump_requested`). The probe names the target rows; the pass-1
    program records their raw (pre-prior) and with-prior scores, to be diffed against RELION's ``exp_Mweight_diff2``
    without keeping the whole ``[batch, n_classes, n_rot * n_trans]`` score cache.
    """

    enabled = collect_significance and probe.score_dump_requested(
        current_size=current_size,
        debug_iteration=debug_iteration,
    )
    rows = probe.score_dump_rows(experiment_dataset, indices) if enabled else None
    return DumpTargets(enabled=enabled, rows=rows)
