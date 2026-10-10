"""Which batches of pass 1 are score-dump batches, and which of their rows are dumped (``RELAX_SIGNIFICANCE_DUMP_*``)."""

from typing import NamedTuple

import numpy as np

from relax.diagnostics.coarse_gaussian_diagnostics import significance_debug_dump_matches
from relax.io.batch_fetch import original_image_indices
from relax.runtime.env_flags import parse_env_int_set


class DumpTargets(NamedTuple):
    """One batch's dump request: ``enabled`` is a dump batch; ``rows`` the batch rows whose scores are recorded.

    ``rows`` is ``None`` when no dump target image is in the batch (the batch still dumps the per-image results).
    """

    enabled: bool
    rows: np.ndarray | None


def select_dump_targets(experiment_dataset, indices, *, collect_significance, current_size, debug_iteration) -> DumpTargets:
    """The dump request of the batch of dataset images ``indices``.

    A dump batch needs the significant samples (``collect_significance``) and an iteration the dump asks for.
    The target images are the original image indices listed in ``RELAX_SIGNIFICANCE_DUMP_ORIGINAL_INDICES``; the
    pass-1 program records their raw (pre-prior) and with-prior scores, to be diffed against RELION's
    ``exp_Mweight_diff2`` without keeping the whole ``[batch, n_classes, n_rot * n_trans]`` score cache.
    """

    enabled = collect_significance and significance_debug_dump_matches(
        current_size=current_size,
        debug_iteration=debug_iteration,
    )
    rows = None
    if enabled:
        targets = parse_env_int_set("RELAX_SIGNIFICANCE_DUMP_ORIGINAL_INDICES")
        if targets:
            original = original_image_indices(experiment_dataset, np.asarray(indices, dtype=np.int64))
            positions = np.flatnonzero(np.isin(original, np.fromiter(targets, dtype=np.int64)))
            if positions.size:
                rows = positions.astype(np.int64)
    return DumpTargets(enabled=enabled, rows=rows)
