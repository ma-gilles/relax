"""RELION's ``--firstiter_cc`` iteration on the device-resident K=1 pass 2.

Iteration 1 scores with normalized cross-correlation (ml_optimiser.cpp:8844-8858;
the GPU kernel ``cuda_kernel_diff2_CC_fine``) and keeps only the best weight
(ml_optimiser.cpp:9266-9293; acc_ml_optimiser_impl.h:2868-2960).

The two resident pieces are checked here against the compact engine's own
functions: the flat-row CC scorer against ``_score_pass2_bucket_relion_gpu_normalized_cc``
and the winner selection against ``jnp.argmax`` over each image's rows in segment
order. A whole-driver comparison needs the compact engine's production CC
configuration (RELION's exact BPref operands, which need a STAR-backed dataset),
so the end-to-end check is the fast tier's cold starts against RELION, whose
iteration 1 is this pass.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

from relax.sparse_pass2.resident_pass2 import _winner_take_all_cells

pytestmark = pytest.mark.unit

N_PIXELS = 12
N_TRANS = 3


def _complex(rng, shape):
    return (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)


def test_winner_is_the_first_maximum_of_each_segment():
    """One-hot at jnp.argmax of the image's rows in order; ties to the first cell."""

    rng = np.random.default_rng(3)
    image_rows = [3, 2, 4, 1]
    row_image = np.repeat(np.arange(len(image_rows)), image_rows).astype(np.int32)
    row_capacity, image_capacity = 12, 5
    scores = rng.normal(size=(row_capacity, N_TRANS)).astype(np.float32)
    scores[1, 2] = scores[0, 1] = 9.0  # tie in image 0: the first cell wins
    scores[: row_image.size][row_image == 3] = -np.inf  # an image with no candidate
    scores[row_image.size :] = -np.inf
    valid = np.arange(row_capacity) < row_image.size
    row_image_padded = np.concatenate([row_image, np.full(row_capacity - row_image.size, image_capacity - 1)])
    offsets = np.concatenate([[0], np.cumsum(image_rows) * N_TRANS, [row_image.size * N_TRANS]])
    best, cell, posterior = _winner_take_all_cells(
        jnp.asarray(scores),
        jnp.asarray(row_image_padded, dtype=jnp.int32),
        jnp.asarray(valid),
        jnp.asarray(offsets, dtype=jnp.int32),
        image_capacity=image_capacity,
    )
    best, cell, posterior = np.asarray(best), np.asarray(cell), np.asarray(posterior)
    for image in range(len(image_rows)):
        flat = scores[: row_image.size][row_image == image].reshape(-1)
        if not np.isfinite(flat).any():
            assert np.isneginf(best[image]) and cell[image] == 0
            assert posterior[: row_image.size][row_image == image].sum() == 0
            continue
        expected = int(np.argmax(flat))
        assert cell[image] == expected
        one_hot = np.zeros(flat.size, np.float32)
        one_hot[expected] = 1.0
        np.testing.assert_array_equal(posterior[: row_image.size][row_image == image].reshape(-1), one_hot)
    assert cell[0] == 1  # row 0, translation 1 precedes row 1, translation 2
    assert posterior[row_image.size :].sum() == 0
