"""Posterior normalization must not depend on computational rotation blocks."""

import jax.numpy as jnp
import numpy as np
import pytest

from relax.ppca_refinement.dense_dataset import _centered_score_partition
from relax.ppca_refinement.local_dataset import _centered_cached_pose_weights

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype,tolerance", [(np.float32, 2e-6), (np.float64, 2e-12)])
def test_score_partition_preserves_mass_with_padding_and_large_offset(dtype, tolerance):
    """Use the same finite scores and masked poses under several block cuts."""
    index = np.arange(2 * 3 * 257).reshape(2, 3, 257)
    scores = (((index * 37) % 113) - 56).astype(dtype) * dtype(0.25)
    scores[index % 17 == 0] = -np.inf
    reference = None
    for offset in (-1700, 8192):
        shifted = scores + dtype(offset)
        for width in (128, 256, 512):
            blocks = [jnp.asarray(shifted[:, :, i : i + width])
                      for i in range(0, shifted.shape[2], width)]
            center, log_partition = _centered_score_partition(blocks)
            posterior = np.concatenate([
                np.asarray(jnp.exp(block - center[:, None, None] - log_partition[:, None, None]))
                for block in blocks
            ], axis=2)
            np.testing.assert_allclose(posterior.sum(axis=(1, 2)), 1, atol=tolerance, rtol=0)
            if reference is None:
                reference = posterior
            np.testing.assert_allclose(posterior, reference, atol=tolerance, rtol=tolerance)
            assert np.array_equal(posterior[index % 17 == 0], np.zeros(np.count_nonzero(index % 17 == 0)))


def test_cached_local_exact_and_topk_weights_ignore_rounded_absolute_logz():
    # The absolute float32 logZ rounds to the center at this offset; the
    # cached exact and top-k M-step paths must retain the centered partition.
    scores = np.array([[[8192.0, 8191.5, 8189.0, -np.inf]]], dtype=np.float32)
    exact = np.asarray(_centered_cached_pose_weights(jnp.asarray(scores)))
    top_scores = jnp.asarray(scores.reshape(1, -1)[:, :2])
    top = np.asarray(_centered_cached_pose_weights(jnp.asarray(scores), top_scores))
    reference = np.exp(scores.astype(np.float64) - 8192.0)
    reference /= reference.sum()
    np.testing.assert_allclose(exact, reference, atol=2e-7, rtol=2e-7)
    np.testing.assert_allclose(top, reference.reshape(1, -1)[:, :2], atol=2e-7, rtol=2e-7)
    assert exact.sum() == pytest.approx(1.0, abs=2e-7)
    assert top.sum() < 1.0  # top-k retains mass; it does not renormalize it.
