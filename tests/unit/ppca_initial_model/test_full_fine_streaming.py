"""Streamed full rotation rows versus the independent local layout and host-mask engine."""

import jax.numpy as jnp
import numpy as np
import pytest
import recovar.core.fourier_transform_utils as ftu
from helpers.float_compare import assert_matches

from relax import sampling
from relax.local.local_layout import build_pass2_hypothesis_layout
from relax.ppca_refinement.config import GeometryConfig, ScheduleConfig, ScoringConfig, SparsePass2Config
from relax.ppca_refinement.dense_dataset import (
    _per_image_pose_prior_block,
    accumulate_dense_ppca_statistics,
    compute_dense_ppca_embeddings,
)
from relax.ppca_refinement.full_row_stream import (
    FULL_ROW_ENGINE,
    accumulate_full_row_tile,
    coarse_support_mask,
    full_row_pose_log_prior,
    full_row_tile_embeddings,
    prepare_full_row_stream,
)
from relax.ppca_refinement.residual_statistics import full_float32

pytestmark = pytest.mark.unit

TRANSLATIONS = np.asarray([[-2, 0], [0, 0], [2, 0]], np.float32)
ROTATION_PRIOR = np.asarray([-0.5, -1.0, -1.5], np.float32)
SIGNIFICANT = [np.asarray([0, 1, 3, 8], np.int32), None, np.asarray([0, 4, 6], np.int32)]
LAYOUT_KWARGS = dict(
    n_coarse_rotations=3, n_coarse_translations=3, nside_level=1,
    translations=TRANSLATIONS, oversampling_order=1, translation_step=2.0,
    rotation_log_prior=ROTATION_PRIOR, rotation_index_order="relion", allow_empty=False,
)


def _layouts():
    reference = build_pass2_hypothesis_layout(SIGNIFICANT, **LAYOUT_KWARGS)
    shared = build_pass2_hypothesis_layout([None], **LAYOUT_KWARGS)
    fine_translations, parent = sampling.get_oversampled_translation_grid(TRANSLATIONS, 2.0, oversampling_order=1)
    assert np.array_equal(np.asarray(fine_translations, np.float32), shared.translation_grid)
    return reference, shared, np.asarray(parent, np.int32)


def test_device_prior_expansion_matches_independent_local_layout():
    reference, shared, parent = _layouts()
    translation_prior = np.linspace(-0.2, -1.3, shared.translation_grid.shape[0]).astype(np.float32)
    coarse = coarse_support_mask(SIGNIFICANT, 3, 3)
    assert coarse.shape == (3, 3, 3)
    prior = np.asarray(full_row_pose_log_prior(
        jnp.asarray(coarse), jnp.asarray(shared.rotation_posterior_ids_flat), jnp.asarray(parent),
        jnp.asarray(shared.rotation_log_priors_flat), jnp.asarray(translation_prior),
    ))
    assert prior.shape == (3, 24, 12) and prior.dtype == np.float32
    for image in range(3):
        begin, end = map(int, reference.rotation_offsets[image:image + 2])
        # Every image keeps the full shared rotation row, in the same order.
        assert np.array_equal(reference.rotations_flat[begin:end], shared.rotations_flat)
        assert np.array_equal(reference.rotation_ids_flat[begin:end], shared.rotation_ids_flat)
        assert np.array_equal(reference.rotation_posterior_ids_flat[begin:end], shared.rotation_posterior_ids_flat)
        mask = reference.sample_mask_rows(begin, end)
        assert np.array_equal(np.isfinite(prior[image]), mask)
        host = _per_image_pose_prior_block(
            batch_start=0, batch_count=1, r0=0, r1=end - begin, n_trans=12,
            rotation_log_prior=reference.rotation_log_priors_flat[begin:end],
            translation_log_prior=translation_prior, rotation_translation_mask=mask, class_log_prior=0.0,
        )
        assert_matches(prior[image], np.asarray(host)[0])


IMAGE_SHAPE = (8, 8)
VOLUME_SHAPE = (8, 8, 8)
N_HALF = IMAGE_SHAPE[0] * (IMAGE_SHAPE[1] // 2 + 1)


def _identity_ctf(params, image_shape, voxel_size, *, half_image=False):
    del voxel_size
    n_pix = image_shape[0] * (image_shape[1] // 2 + 1) if half_image else image_shape[0] * image_shape[1]
    return jnp.ones((params.shape[0], n_pix), dtype=jnp.float32)


class _TinyData:
    def __init__(self, images):
        self.image_shape = IMAGE_SHAPE
        self.volume_shape = VOLUME_SHAPE
        self.grid_size = IMAGE_SHAPE[0]
        self.voxel_size = 1.0
        self.n_images = self.n_units = int(images.shape[0])
        self.dtype = jnp.complex64
        self.ctf_evaluator = _identity_ctf
        self.CTF_params = np.zeros((self.n_images, 9), dtype=np.float32)
        self._images = np.asarray(images, dtype=np.complex64)
        self.image_source = self

    def process_images(self, images, apply_image_mask=False):
        return images

    def process_images_half(self, images, apply_image_mask=False):
        return images

    @property
    def already_prefetches(self):
        return True

    def iter_batches(self, batch_size, *, indices=None, by_image=False, **kwargs):
        indices = np.arange(self.n_images) if indices is None else np.asarray(indices, dtype=np.int64)
        for start in range(0, indices.size, int(batch_size)):
            idx = indices[start:start + int(batch_size)]
            params = jnp.asarray(self.CTF_params[idx])
            yield jnp.asarray(self._images[idx]), None, None, params, None, idx, idx

    def original_image_indices_from_local(self, local):
        return np.asarray(local, dtype=np.int64) + 100


def _half_volume(rng, scale=1.0):
    real = rng.standard_normal(VOLUME_SHAPE).astype(np.float32)
    full = np.fft.fftshift(np.fft.fftn(real)).astype(np.complex64)
    return (scale * np.asarray(ftu.full_volume_to_half_volume(full, VOLUME_SHAPE)).reshape(-1)).astype(np.complex64)


@pytest.fixture(scope="module")
def tile_problem():
    reference, shared, parent = _layouts()
    rng = np.random.default_rng(3)
    images = (rng.standard_normal((3, N_HALF)) + 1j * rng.standard_normal((3, N_HALF))).astype(np.complex64)
    mu = _half_volume(rng)
    W = np.stack([_half_volume(rng, 0.3), _half_volume(rng, 0.2)], axis=1)
    translation_prior = (-np.sum(shared.translation_grid**2, axis=-1) / 8.0).astype(np.float32)
    common = dict(
        noise_variance=np.full(IMAGE_SHAPE, 40.0, np.float32),
        geometry=GeometryConfig(current_size=6, q=2, volume_domain="fourier_half"),
        # Five-row blocks leave a four-row remainder block.
        schedule=ScheduleConfig(image_batch_size=3, rotation_block_size=5),
        scoring=ScoringConfig(relion_texture_interp=False, full_real_observation=True),
    )
    stream = prepare_full_row_stream(
        _TinyData(images), mu, W,
        rotations=shared.rotations_flat, translations=shared.translation_grid,
        rotation_log_prior=shared.rotation_log_priors_flat, translation_log_prior=translation_prior,
        rotation_parent=shared.rotation_posterior_ids_flat, translation_parent=parent,
        n_coarse_rotations=3, n_coarse_translations=3, **common,
    )
    host = dict(
        rotations=shared.rotations_flat, translations=shared.translation_grid,
        rotation_translation_mask=np.stack([
            reference.sample_mask_rows(int(reference.rotation_offsets[i]), int(reference.rotation_offsets[i + 1]))
            for i in range(3)
        ]),
        rotation_log_prior=shared.rotation_log_priors_flat, translation_log_prior=translation_prior,
        image_indices=np.arange(3), **common,
    )
    return _TinyData(images), mu, W, stream, host


@pytest.mark.parametrize("factor_once", [True, False])
def test_device_resident_tile_matches_host_mask_statistics(tile_problem, factor_once):
    dataset, mu, W, stream, host = tile_problem
    expected = full_float32(accumulate_dense_ppca_statistics)(
        dataset, mu, W, sparse_pass2=SparsePass2Config(enabled=False), collect_residuals=True,
        factor_once_score=factor_once, **host,
    )
    actual = accumulate_full_row_tile(stream, np.arange(3), SIGNIFICANT, factor_once=factor_once)
    assert actual.diagnostics["engine"] == FULL_ROW_ENGINE
    assert actual.n_images == expected.n_images == 3
    assert np.array_equal(actual.original_image_ids, expected.original_image_ids)
    for name in ("rhs", "lhs_tri", "residual_gradient", "residual_num", "residual_den", "embeddings"):
        assert_matches(np.asarray(getattr(actual, name)), np.asarray(getattr(expected, name)))
    assert_matches(actual.log_likelihood, expected.log_likelihood)
    for key in ("rotation_mass", "offset_second_sum_px2", "latent_covariance_trace_mean",
                "pose_entropy_mean", "pmax_mean", "max_posterior_per_image"):
        assert_matches(np.asarray(actual.diagnostics[key]), np.asarray(expected.diagnostics[key]))
    for key in ("best_rotation_idx", "best_translation_idx", "n_significant_per_image"):
        assert np.array_equal(np.asarray(actual.diagnostics[key]), np.asarray(expected.diagnostics[key]))


def test_device_resident_tile_embeddings_match_host_mask(tile_problem):
    dataset, mu, W, stream, host = tile_problem
    expected = full_float32(compute_dense_ppca_embeddings)(dataset, mu, W, **host)
    actual = full_row_tile_embeddings(stream, np.arange(3), SIGNIFICANT)
    assert actual.n_images == 3 and np.array_equal(actual.original_image_ids, expected.original_image_ids)
    assert_matches(np.asarray(actual.embeddings), np.asarray(expected.embeddings))


def test_full_row_stream_rejects_parents_outside_coarse_grid(tile_problem):
    dataset, mu, W, stream, host = tile_problem
    kwargs = {key: host[key] for key in ("rotations", "translations", "rotation_log_prior", "translation_log_prior",
                                         "noise_variance", "geometry", "schedule", "scoring")}
    with pytest.raises(ValueError, match="rotation parent"):
        prepare_full_row_stream(
            dataset, mu, W, rotation_parent=np.full(24, 3), translation_parent=np.zeros(12),
            n_coarse_rotations=3, n_coarse_translations=3, **kwargs,
        )
