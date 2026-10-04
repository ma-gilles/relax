"""Diagnostic reporting and dump helpers for exact-local RELION refinement."""

from __future__ import annotations

import os
from contextlib import contextmanager

import numpy as np


def _local_layout_sample_counts(layout):
    """Count allowed rotation/translation pairs in each image for reporting."""
    if layout.sample_mask_bits is None:
        return np.asarray(layout.rotation_counts, dtype=np.int64) * int(layout.translation_grid.shape[0])
    return np.asarray(
        [
            int(np.count_nonzero(layout.sample_mask_rows(start, stop)))
            for start, stop in zip(layout.rotation_offsets[:-1], layout.rotation_offsets[1:])
        ],
        dtype=np.int64,
    )


def log_local_denominator_support(logger, layout, mode, env_name):
    """Report broad-denominator support without changing its layout."""
    denominator_valid_samples_per_image = _local_layout_sample_counts(layout)
    logger.info(
        "RELION local adaptive pass 2 diagnostic: denominator support mode=%s fine valid candidates median=%d max=%d via %s",
        mode,
        int(np.median(denominator_valid_samples_per_image)) if denominator_valid_samples_per_image.size else 0,
        int(np.max(denominator_valid_samples_per_image)) if denominator_valid_samples_per_image.size else 0,
        env_name,
    )


def log_local_adaptive_support(logger, parent_layout, significant_sample_indices, current_translations, pass2_layout):
    """Report retained parent support and available fine candidates."""
    parent_samples_per_image = np.asarray(
        [
            (
                int(np.count_nonzero(parent_layout.sample_mask_rows(start, stop)))
                if parent_layout.sample_mask_bits is not None
                else int(stop - start) * int(current_translations.shape[0])
            )
            if sig is None
            else int(np.asarray(sig).size)
            for sig, start, stop in zip(
                significant_sample_indices, parent_layout.rotation_offsets[:-1], parent_layout.rotation_offsets[1:]
            )
        ],
        dtype=np.int64,
    )
    valid_samples_per_image = _local_layout_sample_counts(pass2_layout)
    logger.info(
        "RELION local adaptive pass 2 mask: parent significant samples median=%d max=%d; fine valid candidates median=%d max=%d",
        int(np.median(parent_samples_per_image)) if parent_samples_per_image.size else 0,
        int(np.max(parent_samples_per_image)) if parent_samples_per_image.size else 0,
        int(np.median(valid_samples_per_image)) if valid_samples_per_image.size else 0,
        int(np.max(valid_samples_per_image)) if valid_samples_per_image.size else 0,
    )


@contextmanager
def score_dump_label(label: str, *, local: bool = False):
    """Append a phase/class label and restore the process environment on exit.

    Local scores and fused posteriors retain their separate caller prefixes.
    Call sites create a fresh scope for each engine invocation.
    """
    names = (
        ("RELAX_LOCAL_SCORE_DUMP_LABEL", "RELAX_LOCAL_FUSED_POSTERIOR_DUMP_LABEL")
        if local else ("RELAX_DEBUG_PER_POSE_DUMP_LABEL",)
    )
    previous = {}
    for name in names:
        old = os.environ.get(name)
        previous[name] = old
        os.environ[name] = f"{old}_{label}" if old else label
    try:
        yield
    finally:
        for name, old in previous.items():
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old


