"""Fused CUDA sparse pass-2 posterior: wrapper contracts, reference and kernel parity."""
import os

import numpy as np
import pytest

from relax.cuda import kernels as em_cuda_kernels
from relax.sparse_pass2 import sparse_pass2_posterior as posterior

pytestmark = pytest.mark.unit

_ENV = posterior._SPARSE_PASS2_CUDA_POSTERIOR_ENV


def reference_fused_posterior(scores, log_z, *, adaptive_fraction, keep_all=False, external_sum_weight=None):
    """Independent NumPy statement of the fused semantics (sequential float32 scan).

    Order-free outputs (max, argmax, masks, counts, pass-through log-Z) are
    exact; the float32 cumulative sum is sequential, so threshold-adjacent
    quantities may differ from a CUB scan at the last float32 bit.
    """

    scores = np.asarray(scores, dtype=np.float32)
    rows = scores.shape[0]
    flat = scores.reshape(rows, -1)
    n = flat.shape[1]
    finite = np.isfinite(flat)
    masked = np.where(finite, flat, -np.inf).astype(np.float32)
    best = masked.max(axis=1)
    argmax = masked.argmax(axis=1)
    has_finite = np.isfinite(best)
    log_z = np.asarray(log_z, dtype=np.float64)
    has_finite_norm = has_finite & np.isfinite(log_z)
    safe_log_z = np.where(has_finite_norm, log_z, 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        probs = np.exp(masked.astype(np.float64) - safe_log_z[:, None])
    probs = np.where(has_finite_norm[:, None] & np.isfinite(probs), probs, 0.0)
    safe_best = np.where(has_finite, best, np.float32(0.0)).astype(np.float32)
    exponent_add = (np.float32(50.0) - safe_best).astype(np.float32)
    exponent = (masked + exponent_add[:, None]).astype(np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        raw = np.where(exponent < np.float32(-88.0), np.float32(0.0), np.exp(exponent)).astype(np.float32)
    raw = np.where(finite & np.isfinite(raw), raw, np.float32(0.0)).astype(np.float32)
    sorted_raw = np.sort(raw, axis=1)
    cumulative = np.empty_like(sorted_raw)
    for r in range(rows):
        acc = np.float32(0.0)
        for i in range(n):
            acc = np.float32(acc + sorted_raw[r, i])
            cumulative[r, i] = acc
    fine_sum = cumulative[:, -1]
    sum_weight = fine_sum if external_sum_weight is None else np.asarray(external_sum_weight, dtype=np.float32)
    has_mass = has_finite & np.isfinite(sum_weight) & (sum_weight > 0)
    if keep_all:
        threshold = np.zeros(rows, dtype=np.float32)
        mask = has_mass[:, None] & finite & (raw > 0)
    else:
        parsed = np.float32(adaptive_fraction)
        target = ((1.0 - np.float64(parsed)) * fine_sum.astype(np.float64)).astype(np.float32)
        idx = np.array([np.searchsorted(cumulative[r], target[r], side="right") for r in range(rows)])
        idx = np.minimum(idx, n - 1)
        threshold = sorted_raw[np.arange(rows), idx]
        mask = has_mass[:, None] & finite & (raw >= threshold[:, None])
    safe_sum = np.where(has_mass, sum_weight, np.float32(1.0)).astype(np.float32)
    normalized = (raw / safe_sum[:, None]).astype(np.float32)
    recon = np.where(mask, normalized, np.float32(0.0)).astype(np.float32)
    shape = scores.shape
    return dict(
        log_z=safe_log_z,
        probs=probs.reshape(shape),
        best_log_score=np.where(has_finite_norm, best, -np.inf).astype(np.float32),
        best_argmax=np.where(has_finite_norm, argmax, 0).astype(np.int64),
        max_posterior=recon.max(axis=1).astype(np.float32),
        normalized_weights=normalized.reshape(shape),
        reconstruction_probs=recon.reshape(shape),
        mask=mask.reshape(shape),
        n_significant=mask.sum(axis=1).astype(np.int32),
        sum_weight=sum_weight.astype(np.float32),
        threshold=threshold.astype(np.float32),
    )


def reference_log_z(scores):
    scores = np.asarray(scores, dtype=np.float32).reshape(np.shape(scores)[0], -1)
    masked = np.where(np.isfinite(scores), scores, -np.inf).astype(np.float32)
    best = masked.max(axis=1)
    has_finite = np.isfinite(best)
    safe_best = np.where(has_finite, best, np.float32(0.0)).astype(np.float32)
    shifted = np.where(has_finite[:, None], (masked - safe_best[:, None]).astype(np.float32), -np.inf)
    total = np.exp(shifted.astype(np.float64)).sum(axis=1)
    has_mass = has_finite & (total > 0) & np.isfinite(total)
    with np.errstate(divide="ignore"):
        return np.where(has_mass, safe_best.astype(np.float64) + np.log(np.where(has_mass, total, 1.0)), -np.inf)


def make_scores(shape, seed, *, all_inf_row=False, nan=False):
    rng = np.random.default_rng(seed)
    scores = (rng.normal(size=shape) * 30 - 200).astype(np.float32)
    pad = rng.random(shape) < 0.3
    scores[pad] = -np.inf
    if nan:
        scores[tuple(0 for _ in shape)] = np.nan
    if all_inf_row and shape[0] > 1:
        scores[1] = -np.inf
    # Guarantee a finite winner somewhere in every other row.
    for r in range(shape[0]):
        if all_inf_row and r == 1:
            continue
        flat = scores[r].reshape(-1)
        if not np.isfinite(flat).any():
            flat[0] = -150.0
    return scores


def test_row_state_bytes_match_header():
    header = os.path.join(os.path.dirname(em_cuda_kernels.__file__), "sparse_pass2_posterior.cuh")
    body = open(header).read().split("struct RowState", 1)[1].split("};", 1)[0]
    sizes = {"float": 4, "double": 8, "int": 4}
    fields = [line.split()[0] for line in body.splitlines() if line.strip() and line.strip()[0] not in "{/"]
    total = 0
    for kind in fields:
        size = sizes[kind]
        total = (total + size - 1) // size * size + size
    total = (total + 7) // 8 * 8
    assert total == em_cuda_kernels._SPARSE_PASS2_ROW_STATE_BYTES


def _gpu_library_has(symbol):
    em_cuda_kernels._ensure_ffi()
    return hasattr(em_cuda_kernels._get_lib(), symbol)


