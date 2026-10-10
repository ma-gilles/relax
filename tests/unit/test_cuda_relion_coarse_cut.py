"""RELION's coarse significance cut by radix select (relion_coarse_cut_f32) against its definition.

The definition: a row's positive weights sorted ascending, cumulative sums in float64, the cut at
the first weight whose cumulative sum exceeds RELION's float32 tail target, RELION's
maximum_significants floor on the threshold index.
"""

import numpy as np
import pytest
from helpers.float_compare import assert_matches

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

pytestmark = pytest.mark.unit


def _reference_cut(weights, fraction, max_significants):
    sums, thresholds, cutoffs = [], [], []
    for row in np.asarray(weights, dtype=np.float32):
        positive = np.sort(row[row > 0.0])
        if positive.size == 0:
            sums.append(np.float32(0.0))
            thresholds.append(np.float32(0.0))
            cutoffs.append(0)
            continue
        cumulative = np.cumsum(positive.astype(np.float64))
        total = np.float32(cumulative[-1])
        target = np.float32((1.0 - np.float64(np.float32(fraction))) * np.float64(total))
        index = min(int(np.searchsorted(cumulative, np.float64(target), side="right")), positive.size - 1)
        if max_significants > 0:
            index = max(index, positive.size - max_significants)
        sums.append(total)
        thresholds.append(positive[index])
        cutoffs.append(positive.size - index)
    return np.asarray(sums, np.float32), np.asarray(thresholds, np.float32), np.asarray(cutoffs, np.int32)


def _rows(rng, n_rows, n_cols):
    # Posterior-like rows: a few large weights, a long tail spanning many binades, zeros.
    log_weights = 50.0 - rng.gamma(1.5, 12.0, size=(n_rows, n_cols))
    weights = np.where(log_weights < -88.0, 0.0, np.exp(log_weights)).astype(np.float32)
    weights[rng.random(weights.shape) < 0.2] = 0.0
    weights[np.arange(n_rows), rng.integers(0, n_cols, n_rows)] = np.float32(np.exp(50.0))
    return weights


def _cut(weights, fraction, max_significants, gpu_device):
    from relax.cuda import kernels as em_cuda_kernels

    with jax.default_device(gpu_device):
        out = em_cuda_kernels.relion_coarse_cut_f32(
            jnp.asarray(weights), adaptive_fraction=fraction, max_significants=max_significants
        )
        return tuple(np.asarray(value) for value in jax.device_get(out))


@pytest.mark.gpu
@pytest.mark.parametrize("n_cols", [1, 7, 4096, 300_001])
@pytest.mark.parametrize("max_significants", [-1, 25])
def test_coarse_cut_matches_its_definition(monkeypatch, custom_cuda_lib, gpu_device, n_cols, max_significants):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    rng = np.random.default_rng(n_cols)
    weights = _rows(rng, 5, n_cols)
    weights[3] = 0.0  # a row without mass
    for fraction in (0.999, 0.9):
        total, threshold, cutoff = _cut(weights, fraction, max_significants, gpu_device)
        ref_total, ref_threshold, ref_cutoff = _reference_cut(weights, fraction, max_significants)
        assert_matches(total, ref_total)
        assert_matches(cutoff, ref_cutoff)
        # The cut weight is one of the row's weights: the same number of weights reach it.
        assert_matches((weights >= threshold[:, None]).sum(1), (weights >= ref_threshold[:, None]).sum(1))


@pytest.mark.gpu
def test_coarse_cut_places_the_cut_inside_a_tied_run(monkeypatch, custom_cuda_lib, gpu_device):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    # 1000 equal weights of 1 below one weight of 1000: the 0.75 target falls in the tied run.
    row = np.concatenate([np.ones(1000, np.float32), [np.float32(1000.0)], np.zeros(17, np.float32)])
    weights = np.stack([row, row[::-1].copy()])
    total, threshold, cutoff = _cut(weights, 0.75, -1, gpu_device)
    ref_total, ref_threshold, ref_cutoff = _reference_cut(weights, 0.75, -1)
    assert_matches(total, ref_total)
    assert_matches(cutoff, ref_cutoff)
    assert_matches(threshold, ref_threshold)


@pytest.mark.gpu
def test_coarse_cut_keeps_relions_cub_sort_and_scan_cut(monkeypatch, custom_cuda_lib, gpu_device):
    """RELION's CUB sort and float32 scan, the reference: the same cut rank at pass-1 row sizes.

    The float32 scan's rounding can move a cut by a sample in principle; on 17 K15-sized rows
    (11.6 M weights, bench 14827636) and two K15/K4 replays the ranks were equal.
    """

    import recovar.cuda_backproject as cuda_backproject

    from relax.cuda import kernels as em_cuda_kernels
    from relax.sampling.oversampling import _relion_cuda_f32_tail_target

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    weights = _rows(np.random.default_rng(2026), 4, 2_000_000)
    fraction = 0.999
    total, _, cutoff = _cut(weights, fraction, -1, gpu_device)
    with jax.default_device(gpu_device):
        matrix = jnp.asarray(weights)
        _, cumulative = em_cuda_kernels.relion_cub_sort_scan_batched_f32(matrix)
        target = _relion_cuda_f32_tail_target(cumulative[:, -1], fraction)
        index = jax.vmap(lambda row, t: jnp.searchsorted(row, t, side="right"))(cumulative, target)
        index = jnp.maximum(index, weights.shape[1] - jnp.sum(matrix > 0, axis=1))
        cub_total, cub_cutoff = jax.device_get((cumulative[:, -1], weights.shape[1] - index))
    assert_matches(total, np.asarray(cub_total), rtol=1e-6)
    assert np.all(np.abs(cutoff.astype(np.int64) - np.asarray(cub_cutoff, np.int64)) <= 1)

