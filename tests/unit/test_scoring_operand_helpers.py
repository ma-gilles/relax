"""Shared scoring helpers: projection blocks, the streaming logsumexp and the CC correction algebra.

Kept from the dense big-JIT tests when the dense engine was removed (2026-10-03): these
helpers and conventions are shared by the coarse pass and the sparse pass-2 operands.
"""

import numpy as np
import pytest
from helpers.float_compare import assert_matches

pytest.importorskip("jax")
import jax.numpy as jnp
from test_sparse_pass2_bucketed_parity import IMAGE_SHAPE, IMAGE_SIZE, MockDataset

from relax.helpers import projection as projection_helpers
from relax.scoring.scoring import _update_logsumexp

pytestmark = pytest.mark.unit

VOLUME_SHAPE = (4, 4, 4)
N_ROT = 3
N_HALF = 4 * (4 // 2 + 1)


def test_dense_projection_helpers_use_relion_texture_interpolation(monkeypatch):
    captured_kwargs = {}

    def fake_slice_volume(volume, rotations_block, image_shape, volume_shape, disc_type, **kwargs):
        captured_kwargs.update(kwargs)
        return jnp.ones((rotations_block.shape[0], N_HALF), dtype=jnp.complex64)

    monkeypatch.setattr(projection_helpers.core, "slice_volume", fake_slice_volume)

    proj_half, proj_abs2_half = projection_helpers.compute_projections_block(
        jnp.ones(int(np.prod(VOLUME_SHAPE)), dtype=jnp.complex64),
        jnp.ones((N_ROT, 3, 3), dtype=jnp.float32),
        (4, 4),
        VOLUME_SHAPE,
        "linear_interp",
    )

    assert captured_kwargs["half_image"] is True
    assert captured_kwargs["relion_texture_interp"] is True
    np.testing.assert_allclose(np.asarray(proj_abs2_half), np.ones_like(np.asarray(proj_abs2_half)))
    assert proj_half.shape == (N_ROT, N_HALF)


def test_logsumexp_initial_block_handles_underflow_without_nan():
    scores = jnp.asarray(
        [[[-1517.0, -1600.0]], [[-933.0, -940.0]]],
        dtype=jnp.float32,
    )
    init_max = jnp.full((2,), -jnp.inf)
    init_sum = jnp.zeros((2,), dtype=jnp.float64)

    max_s, sum_exp = _update_logsumexp(init_max, init_sum, scores)
    assert_matches(np.asarray(max_s), np.array([-1517.0, -933.0]))
    assert np.all(np.isfinite(np.asarray(sum_exp)))


def _unshifted_cc_operands(dataset, *, image_corrections=None, scale_corrections=None):
    from recovar.core.configs import ForwardModelConfig
    from recovar.reconstruction import noise as noise_utils

    from relax.sparse_pass2.sparse_pass2_bucket_io import prepare_unshifted_bucket_operands

    config = ForwardModelConfig.from_dataset(dataset, disc_type="linear_interp", process_fn=dataset.process_images)
    batch, _, _, ctf_params, _, _, image_indices = next(dataset.iter_batches(dataset.n_images))
    noise = jnp.linspace(0.5, 1.5, IMAGE_SIZE, dtype=jnp.float32)
    return prepare_unshifted_bucket_operands(
        dataset,
        batch,
        ctf_params,
        image_indices,
        noise_variance_half=noise_utils.to_batched_half_pixel_noise(noise, IMAGE_SHAPE).squeeze(),
        config=config,
        score_with_masked_images=False,
        image_corrections=image_corrections,
        scale_corrections=scale_corrections,
        image_pre_shifts=None,
        use_float64_scoring=True,
        score_mode="normalized_cc",
    )


def test_relion_firstiter_cc_keeps_cross_scale_correction():
    """Pin RELION GPU firstiter-CC scale algebra on the pass-2 operands.

    RELION's accelerated path divides Fimg by scale in acc_ml_optimiser_impl.h, then
    buildCorrImage multiplies the CC image by scale^2 in acc_helper_functions_impl.h.
    With relax's convention image_corrections=(avg_norm/normcorr)*scale, the net
    cross-term correction remains image_corrections; only the image-power term
    divides out the scale, and the model energy carries scale^2.
    """

    dataset = MockDataset(n_images=2, seed=7)
    batch_corr = np.asarray([1.2, 0.8], dtype=np.float64)
    batch_scale = np.asarray([1.5, 0.5], dtype=np.float64)
    plain = _unshifted_cc_operands(dataset)
    corrected = _unshifted_cc_operands(
        dataset,
        image_corrections=batch_corr.astype(np.float32),
        scale_corrections=batch_scale.astype(np.float32),
    )
    corr = np.asarray(batch_corr.astype(np.float32), dtype=np.float64)[:, None]
    scale = np.asarray(batch_scale.astype(np.float32), dtype=np.float64)[:, None]

    # The CTF-weighted cross-term operand carries the full image correction.
    np.testing.assert_allclose(
        np.asarray(corrected.score_weighted_half), np.asarray(plain.score_weighted_half) * corr, rtol=1e-12
    )
    # The image power (the CC denominator's Xi2) carries only the image-only correction.
    np.testing.assert_allclose(
        np.asarray(corrected.batch_norm), np.asarray(plain.batch_norm) * (corr / scale) ** 2, rtol=1e-12
    )
    # The model-energy weight carries the scale squared.
    np.testing.assert_allclose(
        np.asarray(corrected.ctf2_score_half), np.asarray(plain.ctf2_score_half) * scale**2, rtol=1e-12
    )
