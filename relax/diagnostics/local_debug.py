"""Diagnostic reporting and dump helpers for exact-local RELION refinement."""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from relax.helpers.env_flags import parse_int_set


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


@dataclass(frozen=True)
class DensePerPoseScoreDumpRequest:
    """Dense/global per-pose score dump request parsed from environment."""

    dump_dir: Path | None = None
    target: int | None = None
    dump_preprior: bool = False
    target_is_original: bool = False

    @property
    def enabled(self) -> bool:
        return self.dump_dir is not None and self.target is not None


def parse_dense_noise_component_dump_request():
    """Return optional per-particle dense noise component dump settings."""

    dump_dir = os.environ.get("RELAX_DENSE_NOISE_COMPONENT_DUMP_DIR")
    dump_indices = os.environ.get("RELAX_DENSE_NOISE_COMPONENT_DUMP_GLOBAL_INDICES")
    dump_current_size = os.environ.get("RELAX_DENSE_NOISE_COMPONENT_DUMP_CURRENT_SIZE")
    if not dump_dir or not dump_indices:
        return None, set(), None
    targets = parse_int_set(dump_indices) or set()
    if not targets:
        return None, set(), None
    requested_current_sizes = parse_int_set(dump_current_size)
    dump_path = Path(dump_dir)
    dump_path.mkdir(parents=True, exist_ok=True)
    return dump_path, targets, requested_current_sizes


def parse_dense_per_pose_score_dump_request() -> DensePerPoseScoreDumpRequest:
    """Return optional dense/global per-pose score dump settings."""

    dump_dir = os.environ.get("RELAX_DEBUG_PER_POSE_DUMP_DIR")
    dump_target = os.environ.get("RELAX_DEBUG_PER_POSE_DUMP_TARGET")
    if not dump_dir or dump_target is None:
        return DensePerPoseScoreDumpRequest()
    try:
        target = int(dump_target)
    except ValueError:
        return DensePerPoseScoreDumpRequest()
    dump_path = Path(dump_dir)
    dump_path.mkdir(parents=True, exist_ok=True)
    dump_preprior = os.environ.get("RELAX_DEBUG_PER_POSE_DUMP_PREPRIOR")
    target_is_original = os.environ.get("RELAX_DEBUG_PER_POSE_DUMP_TARGET_IS_ORIGINAL")
    return DensePerPoseScoreDumpRequest(
        dump_dir=dump_path,
        target=target,
        dump_preprior=bool(dump_preprior and dump_preprior != "0"),
        target_is_original=bool(target_is_original and target_is_original != "0"),
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


def _dump_label_suffix(label: str | None) -> str:
    if not label:
        return ""
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip())
    return f"_{label}" if label else ""


def dense_score_dump_label_suffix() -> str:
    """Return the sanitized optional label suffix shared by dense score dumps."""
    return _dump_label_suffix(os.environ.get("RELAX_DEBUG_PER_POSE_DUMP_LABEL"))

def maybe_write_dense_per_pose_score_dump(
    *,
    request: DensePerPoseScoreDumpRequest,
    indices,
    scores,
    block_index: int,
    preprior: bool = False,
    original_indices=None,
) -> None:
    """Dump one dense/global score block for a targeted input image."""

    if not request.enabled:
        return
    if preprior and not request.dump_preprior:
        return
    try:
        match_indices = original_indices if request.target_is_original else indices
        hits = np.where(np.asarray(match_indices, dtype=np.int64) == int(request.target))[0]
        if len(hits) == 0:
            return
        row = int(hits[0])
        suffix = "_preprior" if preprior else ""
        label_suffix = dense_score_dump_label_suffix()
        scores_target = np.asarray(scores[row], dtype=np.float64)
        np.save(
            request.dump_dir / f"target{int(request.target):06d}{label_suffix}_block{int(block_index):04d}{suffix}.npy",
            scores_target,
        )
    except Exception:
        return


