"""Adaptive oversampling (order 1) against the dense stream over the child grid (section 14).

Pass 2 of the oversampled stream scores each image's children of its significant coarse samples, one job per
sample. The dense stream over the whole child grid, with those samples as each image's coarse support, scores the
same poses with the same priors, so every statistic must agree within float32 reduction order; with fraction 1 and
no cap the support is the whole grid.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches
from scipy.spatial.transform import Rotation
from test_full_fine_streaming import IMAGE_SHAPE, N_HALF, _half_volume, _TinyData

from relax import sampling
from relax.ppca_refinement.config import GeometryConfig, ScheduleConfig, ScoringConfig
from relax.ppca_refinement.full_row_stream import (
    _normalize,
    accumulate_full_row_tile,
    full_row_tile_embeddings,
    prepare_full_row_stream,
)
from relax.ppca_refinement.oversampled_stream import (
    OVERSAMPLED_ENGINE,
    _significant_samples,
    accumulate_oversampled_tile,
    oversampled_tile_embeddings,
    prepare_oversampled_stream,
    relion_child_grids,
)

pytestmark = pytest.mark.unit

STATISTICS = ("lhs_tri", "residual_gradient", "residual_num", "residual_den", "embeddings")


def _spa_stream(*, metric_trace_only=False, order=0, n_images=4, device=None):
    """A pass-1 stream over HEALPix order ``order`` and a 1 px shift grid, as the controller builds it."""
    rng = np.random.default_rng(7)
    images = (rng.standard_normal((n_images, N_HALF)) + 1j * rng.standard_normal((n_images, N_HALF))).astype(
        np.complex64
    )
    images[:, 0] += 30.0  # a mean image the model partly explains, so posteriors are not flat
    mu = _half_volume(rng)
    W = np.stack([_half_volume(rng, 0.3), _half_volume(rng, 0.2)], axis=1)
    rotations = sampling.get_relion_hidden_rotation_grid(order, matrices=True).astype(np.float32)
    translations = sampling.get_relion_translation_grid(max_pixel=1, pixel_offset=1).astype(np.float32)
    rotation_prior = np.log(rng.dirichlet(np.ones(len(rotations)))).astype(np.float32)
    translation_prior = (-np.sum(translations**2, axis=-1) / 2.0).astype(np.float32)
    coarse = prepare_full_row_stream(
        _TinyData(images),
        mu,
        W,
        noise_variance=np.full(IMAGE_SHAPE, 40.0, np.float32),
        rotations=rotations,
        translations=translations,
        rotation_log_prior=rotation_prior,
        translation_log_prior=translation_prior - np.log(np.sum(np.exp(translation_prior))),
        rotation_parent=np.zeros(len(rotations), np.int32),
        translation_parent=np.zeros(len(translations), np.int32),
        n_coarse_rotations=1,
        n_coarse_translations=1,
        geometry=GeometryConfig(current_size=6, q=2, volume_domain="fourier_half"),
        # 32-row blocks leave a remainder block on the 576-row child grid.
        schedule=ScheduleConfig(image_batch_size=n_images, rotation_block_size=32),
        scoring=ScoringConfig(relion_texture_interp=False, full_real_observation=True),
        metric_trace_only=metric_trace_only,
        gemm_precision="fp32",
        device=device,
    )
    return coarse, relion_child_grids(order, translations, 1.0)


# Tolerances from the measured float32 reduction-order noise of this problem. The engines sum the same terms in
# another order, and on GPU the moment scatter's atomics and the CUDA epilogue add their own: the dense stream run
# on GPU and on CPU differs by up to 1e-6 (statistics) and 1e-5 (top-pose posterior), and this test's engines by
# up to 2e-6 and 2e-5 on an A100. The scalar diagnostics are sums over every pose and the top-pose posterior divides
# by one; reordering the dense stream on CPU alone (rotation blocks of 32 versus 7) moves them by up to 1e-5.
STATISTICS_RTOL = 5e-6
DIAGNOSTIC_RTOL = 5e-5


def _assert_same_statistics(actual, expected, rtol=STATISTICS_RTOL):
    for name in STATISTICS if expected.lhs_tri is not None else ("metric_trace",) + STATISTICS[1:]:
        assert_matches(np.asarray(getattr(actual, name)), np.asarray(getattr(expected, name)), err_msg=name, rtol=rtol)
    assert actual.n_images == expected.n_images
    np.testing.assert_array_equal(actual.original_image_ids, expected.original_image_ids)
    assert_matches(np.float32(actual.log_likelihood), np.float32(expected.log_likelihood), rtol=DIAGNOSTIC_RTOL)
    a, e = actual.diagnostics, expected.diagnostics
    for key in ("offset_second_sum_px2", "latent_covariance_trace_mean", "pose_entropy_mean"):
        assert_matches(np.float32(a[key]), np.float32(e[key]), err_msg=key, rtol=DIAGNOSTIC_RTOL)
    assert_matches(np.float32(a["pmax_mean"]), np.float32(e["pmax_mean"]), rtol=DIAGNOSTIC_RTOL)
    assert_matches(a["max_posterior_per_image"], e["max_posterior_per_image"], rtol=DIAGNOSTIC_RTOL)
    for key in ("best_rotation_idx", "best_translation_idx", "n_significant_per_image"):
        np.testing.assert_array_equal(a[key], e[key], err_msg=key)


def _parent_mass(stats, rotation_children):
    mass = np.asarray(stats.diagnostics["rotation_mass"])
    return mass.reshape(-1, rotation_children).sum(axis=1)


@pytest.mark.parametrize("metric_trace_only", [False, True])
def test_fraction_one_equals_the_dense_child_grid(metric_trace_only):
    """With fraction 1 and no cap every coarse sample is significant: pass 2 is the dense child grid."""
    coarse, (fine_rotations, fine_translations) = _spa_stream(metric_trace_only=metric_trace_only)
    n_poses = (coarse.arrays.rotations.shape[0] - 1) * coarse.translations.shape[0]
    ostream = prepare_oversampled_stream(
        coarse, fine_rotations, fine_translations, adaptive_fraction=1.0, max_significant=n_poses, job_chunk=96
    )
    assert (ostream.rotation_children, ostream.translation_children) == (8, 4)
    ids = np.arange(4)
    actual = accumulate_oversampled_tile(ostream, ids)
    expected = accumulate_full_row_tile(ostream.fine, ids, [None] * 4)
    assert actual.diagnostics["engine"] == OVERSAMPLED_ENGINE
    assert np.all(actual.diagnostics["significant_samples_per_image"] == n_poses)
    _assert_same_statistics(actual, expected)
    assert_matches(actual.diagnostics["rotation_mass"], _parent_mass(expected, 8), rtol=DIAGNOSTIC_RTOL)


@pytest.mark.parametrize("fraction, cap", [(0.9, 40), (0.999, 7), (0.5, 100)])
def test_significant_children_equal_the_dense_child_grid_on_that_support(fraction, cap):
    """Each image's children of its significant samples only: the dense child grid with that coarse support."""
    coarse, (fine_rotations, fine_translations) = _spa_stream()
    ostream = prepare_oversampled_stream(
        coarse, fine_rotations, fine_translations, adaptive_fraction=fraction, max_significant=cap, job_chunk=16
    )
    ids = np.arange(4)
    actual = accumulate_oversampled_tile(ostream, ids)
    significant = actual.diagnostics["significant_samples"]
    counts = actual.diagnostics["significant_samples_per_image"]
    assert np.all((counts >= 1) & (counts <= cap)) and len(significant) == 4
    expected = accumulate_full_row_tile(ostream.fine, ids, significant)
    _assert_same_statistics(actual, expected)
    assert_matches(actual.diagnostics["rotation_mass"], _parent_mass(expected, 8), rtol=DIAGNOSTIC_RTOL)
    embedded = oversampled_tile_embeddings(ostream, ids)
    reference = full_row_tile_embeddings(ostream.fine, ids, significant)
    assert_matches(np.asarray(embedded.embeddings), np.asarray(reference.embeddings), rtol=STATISTICS_RTOL)


def _relion_significant(weights, fraction, cap):
    """RELION's loop (``ml_optimiser.cpp:9602-9716``) in float64: the kept weights' threshold, then every weight
    at least it."""
    order = np.sort(weights[weights > 0])[::-1]
    total, count, threshold = 0.0, 0, None
    for w in order:
        if count < cap:
            count += 1
            threshold = w
        total += w
        if total > fraction * weights.sum():
            break
    return weights >= threshold


@pytest.mark.parametrize("fraction, cap", [(0.9, 50), (0.5, 3), (0.999, 1000), (0.99, 20)])
def test_significance_follows_relion(fraction, cap):
    rng = np.random.default_rng(11)
    B, R, T = 3, 40, 5
    score = rng.standard_normal((R + 2, B, T)).astype(np.float32) * 3.0
    score[R:] = -np.inf  # padding rows of the row table
    rows = np.append(np.arange(R), [R, R]).astype(np.int32)
    posterior = _normalize(jnp.asarray(score), jnp.asarray(rows), n_blocks=1, block_size=R + 2)
    rotation, translation, valid = (
        np.asarray(x)
        for x in _significant_samples(
            jnp.asarray(score),
            jnp.asarray(rows),
            posterior,
            jnp.float32(fraction),
            np.int32(B),
            n_significant=min(cap, R * T),
            every=False,
        )
    )
    for b in range(B):
        weights = np.exp(score[:R, b].astype(np.float64) - np.max(score[:R, b]))
        weights /= weights.sum()
        expected = _relion_significant(weights.reshape(-1), fraction, cap)
        kept = np.zeros(R * T, bool)
        kept[rotation[b][valid[b]] * T + translation[b][valid[b]]] = True
        np.testing.assert_array_equal(kept, expected)


def test_padding_images_have_no_significant_samples():
    score = jnp.zeros((4, 3, 2), jnp.float32)
    rows = jnp.arange(4, dtype=jnp.int32)
    posterior = _normalize(score, rows, n_blocks=1, block_size=4)
    _, _, valid = _significant_samples(
        score, rows, posterior, jnp.float32(0.5), np.int32(2), n_significant=8, every=False
    )
    assert np.asarray(valid)[:2].any(axis=1).all() and not np.asarray(valid)[2:].any()


def test_subtomogram_children_equal_the_dense_child_grid():
    """Tilt-series particles (shared latent over tilts, 3D shifts with 8 children each) through the tilt reader."""
    import test_tomo_ppca as tomo

    _particles, coarse, _arrays = tomo.make_problem(seed=4)
    perturb = Rotation.from_rotvec(np.random.default_rng(1).normal(scale=0.05, size=(8, 3))).as_matrix()
    coarse_rotations = np.asarray(coarse.arrays.rotations[:-1])
    fine_rotations = np.einsum("rab,cbd->rcad", coarse_rotations, perturb).reshape(-1, 3, 3).astype(np.float32)
    fine_translations, parent = sampling.get_oversampled_translation_grid(tomo.TRANSLATIONS, 1.5, 1)
    assert fine_translations.shape == (8 * len(tomo.TRANSLATIONS), 3)
    np.testing.assert_array_equal(parent, np.repeat(np.arange(len(tomo.TRANSLATIONS)), 8))
    ostream = prepare_oversampled_stream(
        coarse, fine_rotations, fine_translations, adaptive_fraction=0.95, max_significant=9, job_chunk=8
    )
    ids = np.arange(3)
    actual = accumulate_oversampled_tile(ostream, ids)
    expected = accumulate_full_row_tile(ostream.fine, ids, actual.diagnostics["significant_samples"])
    _assert_same_statistics(actual, expected)
    assert_matches(actual.diagnostics["rotation_mass"], _parent_mass(expected, 8), rtol=DIAGNOSTIC_RTOL)
