"""One class's device compaction of a coarse significance mask, for tests.

Moved from relax/fine_pass/resident_significance.py, where only tests called it: a one-class call of the live
``compact_batch_significance_classes``.
"""

from relax.fine_pass.resident_significance import compact_batch_significance_classes


def compact_batch_significance(
    batch_sig_mask,
    *,
    actual_batch_size: int,
    n_coarse_rot: int,
    n_coarse_trans: int,
    batch_n_sig,
):
    """One class's :func:`compact_batch_significance_classes` result."""

    return compact_batch_significance_classes(
        batch_sig_mask,
        n_classes=1,
        actual_batch_size=actual_batch_size,
        n_coarse_rot=n_coarse_rot,
        n_coarse_trans=n_coarse_trans,
        batch_n_sig=batch_n_sig,
    )[0]
