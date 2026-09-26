"""Parity tests: bucketed sparse pass-2 vs per-image reference path.

The bucketed batched implementation in
``compute_pass2_stats_sparse_bucketed`` must produce numerically
identical outputs (modulo floating-point rounding) to the legacy
per-image loop preserved as
``_compute_pass2_stats_sparse_perimage_reference``.

These tests construct synthetic single-class data with varied per-image
significant rotation counts (the trigger for the JIT-recompile bug the
batched path is designed to fix), then call both paths and compare the
M-step accumulators ``Ft_y`` / ``Ft_ctf``, hard assignments, and per-image
RELION stats.
"""


import numpy as np
import pytest
from helpers.float_compare import assert_matches

pytest.importorskip("jax")
import jax.numpy as jnp
from helpers.em_arrays import _raw_real_image_2d

import recovar.core.fourier_transform_utils as ftu
from relax.scoring.sparse_bucket_arrays import _build_bucket_arrays, _prepare_per_image_pass2_inputs
from relax.sparse_pass2.sparse_pass2_bucket_io import _reorder_to_indices
from relax.sparse_pass2.sparse_pass2_posterior import _normalize_pass2_bucket_with_log_z
from relax.sparse_pass2.sparse_pass2_scoring import _score_pass2_bucket_relion_gpu_diff2

pytestmark = pytest.mark.unit


def _z_rotation(angle):
    c = np.float32(np.cos(angle))
    s = np.float32(np.sin(angle))
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


# ---------------------------------------------------------------------------
# Mock dataset (mirrors test_adaptive_oversampling.MockDataset)
# ---------------------------------------------------------------------------

IMAGE_SHAPE = (8, 8)
IMAGE_SIZE = 64
VOLUME_SHAPE = (8, 8, 8)
VOLUME_SIZE = 512


def _identity_ctf(params, image_shape=None, voxel_size=None, *, half_image=False):
    if half_image:
        h, w = image_shape if image_shape is not None else IMAGE_SHAPE
        sz = h * (w // 2 + 1)
    else:
        sz = IMAGE_SIZE
    return jnp.ones((params.shape[0], sz), dtype=jnp.float32)


def _unit_image_mask(dtype=jnp.float32):
    return jnp.linspace(0.2, 1.0, IMAGE_SIZE, dtype=dtype).reshape(IMAGE_SHAPE)


def _raw_real_process(batch, apply_image_mask=False):
    images = jnp.asarray(batch)
    if apply_image_mask:
        images = images * _unit_image_mask(images.dtype)
    return ftu.get_dft2(images).reshape((images.shape[0], -1)).astype(jnp.complex64)


def _raw_real_process_half(batch, apply_image_mask=False):
    images = jnp.asarray(batch)
    if apply_image_mask:
        images = images * _unit_image_mask(images.dtype)
    return ftu.get_dft2_real(images).reshape((images.shape[0], -1)).astype(jnp.complex64)


class MockDataset:
    def __init__(self, n_images=10, seed=42):
        self.image_shape = IMAGE_SHAPE
        self.image_size = IMAGE_SIZE
        self.grid_size = IMAGE_SHAPE[0]
        self.volume_shape = VOLUME_SHAPE
        self.volume_size = VOLUME_SIZE
        self.n_images = n_images
        self.n_units = n_images
        self.voxel_size = 1.0
        self.dtype = jnp.complex64
        self.CTF_params = np.zeros((n_images, 9), dtype=np.float32)
        self.ctf_evaluator = staticmethod(_identity_ctf)
        self.process_images = staticmethod(_raw_real_process)
        self.process_images_half = staticmethod(_raw_real_process_half)

        self.rotation_matrices = np.tile(np.eye(3, dtype=np.float32), (n_images, 1, 1))
        self.translations = np.zeros((n_images, 2), dtype=np.float32)
        self.premultiplied_ctf = False

        rng = np.random.default_rng(seed)
        self._images = np.zeros((n_images, *IMAGE_SHAPE), dtype=np.float32)
        for i in range(n_images):
            self._images[i] = _raw_real_image_2d(IMAGE_SHAPE, seed=rng.integers(10000))

        class _ImageSource:
            process_images = staticmethod(_raw_real_process)
            process_images_half = staticmethod(_raw_real_process_half)

        self.image_source = _ImageSource()

    def iter_batches(self, batch_size, *, indices=None, by_image=False, **kwargs):
        _ = kwargs
        if indices is None:
            indices = np.arange(self.n_images)
        indices = np.asarray(indices)
        for chunk_start in range(0, len(indices), max(1, batch_size)):
            chunk_end = min(chunk_start + max(1, batch_size), len(indices))
            idx = np.asarray(indices[chunk_start:chunk_end])
            yield (
                jnp.asarray(self._images[idx]),
                jnp.asarray(self.rotation_matrices[idx]),
                jnp.asarray(self.translations[idx]),
                jnp.asarray(self.CTF_params[idx]),
                None,
                idx,
                idx,
            )

    def get_valid_frequency_indices(self, pixel_res):
        return np.ones(self.volume_size, dtype=bool)

    def update_poses(self, rotations, translations):
        self.rotation_matrices = np.asarray(rotations)
        self.translations = np.asarray(translations)


# ---------------------------------------------------------------------------
# Tolerance helpers
# ---------------------------------------------------------------------------


def _max_relative_error(a, b, eps=1e-12):
    a = np.asarray(a)
    b = np.asarray(b)
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(a), np.abs(b))
    denom = np.maximum(denom, eps)
    return float(np.max(diff / denom))


def _compare_outputs(
    out_ref,
    out_bucket,
    atol=1e-5,
    rtol=1e-5,
    accumulator_scaled_rtol=None,
):
    """Compare per-image and accumulated outputs with tight tolerance."""
    Ft_y_ref = out_ref.Ft_y
    Ft_ctf_ref = out_ref.Ft_ctf
    ha_ref = out_ref.hard_assignment
    best_rot_ref = out_ref.best_rotations
    best_tr_ref = out_ref.best_translations
    best_idx_ref = out_ref.best_rotation_indices
    stats_ref = out_ref.relion_stats
    Ft_y_b = out_bucket.Ft_y
    Ft_ctf_b = out_bucket.Ft_ctf
    ha_b = out_bucket.hard_assignment
    best_rot_b = out_bucket.best_rotations
    best_tr_b = out_bucket.best_translations
    best_idx_b = out_bucket.best_rotation_indices
    stats_b = out_bucket.relion_stats

    # M-step accumulators must be very close. Some float32 winner-take-all
    # routes sum identical selected rows in different batch/reduction orders;
    # for those, scale the maximum absolute error by the accumulator peak so
    # near-zero cancellation voxels do not dominate the comparison.
    if accumulator_scaled_rtol is None:
        np.testing.assert_allclose(np.asarray(Ft_y_ref), np.asarray(Ft_y_b), atol=atol, rtol=rtol)
        np.testing.assert_allclose(np.asarray(Ft_ctf_ref), np.asarray(Ft_ctf_b), atol=atol, rtol=rtol)
    else:
        for label, reference, bucketed in (
            ("Ft_y", Ft_y_ref, Ft_y_b),
            ("Ft_ctf", Ft_ctf_ref, Ft_ctf_b),
        ):
            reference = np.asarray(reference)
            bucketed = np.asarray(bucketed)
            peak = max(float(np.max(np.abs(reference))), np.finfo(np.float32).tiny)
            scaled_error = float(np.max(np.abs(reference - bucketed))) / peak
            assert scaled_error <= accumulator_scaled_rtol, (
                f"{label} peak-scaled error {scaled_error:.9g} exceeds {accumulator_scaled_rtol:.9g}"
            )

    # Hard assignments must match exactly (decoded from probs argmax).
    assert_matches(np.asarray(ha_ref), np.asarray(ha_b))
    assert_matches(np.asarray(best_idx_ref), np.asarray(best_idx_b))
    np.testing.assert_allclose(np.asarray(best_rot_ref), np.asarray(best_rot_b), atol=1e-6)
    np.testing.assert_allclose(np.asarray(best_tr_ref), np.asarray(best_tr_b), atol=1e-6)

    # RELION stats must match within float32 precision.
    np.testing.assert_allclose(
        np.asarray(stats_ref.log_evidence_per_image),
        np.asarray(stats_b.log_evidence_per_image),
        atol=atol,
        rtol=rtol,
    )
    np.testing.assert_allclose(
        np.asarray(stats_ref.best_log_score_per_image),
        np.asarray(stats_b.best_log_score_per_image),
        atol=atol,
        rtol=rtol,
    )
    np.testing.assert_allclose(
        np.asarray(stats_ref.max_posterior_per_image),
        np.asarray(stats_b.max_posterior_per_image),
        atol=atol,
        rtol=rtol,
    )
    np.testing.assert_allclose(
        np.asarray(stats_ref.rotation_posterior_sums),
        np.asarray(stats_b.rotation_posterior_sums),
        atol=atol,
        rtol=rtol,
    )

    # Noise stats (when present)
    assert (out_ref.noise_stats is None) == (out_bucket.noise_stats is None)
    if out_ref.noise_stats is not None:
        ns_ref = out_ref.noise_stats
        ns_b = out_bucket.noise_stats
        np.testing.assert_allclose(
            np.asarray(ns_ref.wsum_sigma2_noise),
            np.asarray(ns_b.wsum_sigma2_noise),
            atol=atol,
            rtol=rtol,
        )
        np.testing.assert_allclose(
            np.asarray(ns_ref.wsum_img_power),
            np.asarray(ns_b.wsum_img_power),
            atol=atol,
            rtol=rtol,
        )
        np.testing.assert_allclose(
            ns_ref.wsum_sigma2_offset,
            ns_b.wsum_sigma2_offset,
            atol=atol,
            rtol=rtol,
        )
        np.testing.assert_allclose(ns_ref.sumw, ns_b.sumw, atol=atol)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def test_sparse_pass2_score_matches_relion_common_min_direct_diff2_for_finite_inputs():
    """Sparse pass-2 uses RELION's common-min float32 direct-diff2 scores."""
    rng = np.random.default_rng(123)
    batch, n_rot, n_trans, n_half = 2, 3, 4, 5
    image = (
        rng.normal(size=(batch, n_trans, n_half)) + 1j * rng.normal(size=(batch, n_trans, n_half))
    ).astype(np.complex64)
    ctf = rng.uniform(0.35, 1.25, size=(batch, n_half)).astype(np.float32)
    noise = rng.uniform(0.7, 2.0, size=(batch, n_half)).astype(np.float32)
    proj = (
        rng.normal(size=(batch, n_rot, n_half)) + 1j * rng.normal(size=(batch, n_rot, n_half))
    ).astype(np.complex64)
    half_weights = np.array([1.0, 2.0, 2.0, 2.0, 1.0], dtype=np.float32)
    rot_prior = rng.normal(scale=0.1, size=(batch, n_rot)).astype(np.float32)
    trans_prior = rng.normal(scale=0.1, size=(batch, n_trans)).astype(np.float32)
    mask = np.ones((batch, n_rot, n_trans), dtype=bool)

    corr_img_score = ctf * ctf / noise
    shifted_corrected = image / ctf[:, None, :]
    scores = _score_pass2_bucket_relion_gpu_diff2(
        jnp.asarray(shifted_corrected),
        jnp.asarray(corr_img_score),
        jnp.asarray(proj),
        jnp.asarray(half_weights),
        jnp.asarray(rot_prior),
        jnp.asarray(trans_prior),
        jnp.asarray(mask),
    )

    expected = np.empty((batch, n_rot, n_trans), dtype=np.float32)
    for b in range(batch):
        weights = corr_img_score[b] * half_weights
        raw_diff2 = np.empty((n_rot, n_trans), dtype=np.float32)
        for r in range(n_rot):
            diff = proj[b, r][None, :] - shifted_corrected[b]
            diff2 = 0.5 * np.sum(np.abs(diff) ** 2 * weights[None, :], axis=-1)
            raw_diff2[r] = diff2.astype(np.float32)
        common_min = np.min(raw_diff2)
        for r in range(n_rot):
            prior = np.float32(np.float32(rot_prior[b, r]) + trans_prior[b])
            expected[b, r] = np.float32(np.float32(prior + common_min) - raw_diff2[r])

    np.testing.assert_allclose(np.asarray(scores), expected, atol=5e-5, rtol=5e-5)


def test_sparse_pass2_score_masks_nonfinite_direct_diff2_candidates():
    """Overflowed direct-diff candidates should become impossible, not NaN."""
    scores = _score_pass2_bucket_relion_gpu_diff2(
        jnp.array([[[jnp.inf + 0.0j]]], dtype=jnp.complex64),
        jnp.ones((1, 1), dtype=jnp.float32),
        jnp.zeros((1, 1, 1), dtype=jnp.complex64),
        jnp.ones((1,), dtype=jnp.float32),
        jnp.zeros((1, 1), dtype=jnp.float32),
        jnp.zeros((1, 1), dtype=jnp.float32),
        jnp.ones((1, 1, 1), dtype=bool),
    )

    assert np.isneginf(float(np.asarray(scores[0, 0, 0])))


def test_sparse_pass2_logz_normalization_masks_nonfinite_scores():
    """Precomputed-log-Z sparse normalization must also stay finite."""
    scores = jnp.array([[[jnp.nan, -jnp.inf]], [[0.0, -1.0]]], dtype=jnp.float32)
    log_z = jnp.array([jnp.nan, 0.5], dtype=jnp.float32)

    safe_log_z, probs, best_log_score, best_argmax, max_posterior = _normalize_pass2_bucket_with_log_z(scores, log_z)

    assert float(np.asarray(safe_log_z[0])) == pytest.approx(0.0)
    np.testing.assert_allclose(np.asarray(probs[0]).sum(), 0.0, atol=1e-7)
    assert np.isneginf(float(np.asarray(best_log_score[0])))
    assert int(np.asarray(best_argmax[0])) == 0
    assert float(np.asarray(max_posterior[0])) == pytest.approx(0.0)
    assert np.all(np.isfinite(np.asarray(probs)))
    assert float(np.asarray(safe_log_z[1])) == pytest.approx(0.5)


def test_sparse_pass2_mstep_rotations_follow_score_selection_padding_and_reorder():
    score_rotations = np.stack([_z_rotation(angle) for angle in (0.1, 0.2, 0.3, 0.4)])
    mstep_rotations = np.stack([_z_rotation(angle) for angle in (1.1, 1.2, 1.3, 1.4)])
    parent_map = np.asarray([0, 0, 1, 1], dtype=np.int32)
    significant_samples = [
        np.asarray([0, 1], dtype=np.int32),
        np.asarray([1], dtype=np.int32),
    ]

    per_image = _prepare_per_image_pass2_inputs(
        significant_samples,
        n_coarse_rot=12,
        n_coarse_trans=1,
        nside_level=0,
        oversampling_order=0,
        n_fine_trans=1,
        fine_translation_parent=np.asarray([0], dtype=np.int32),
        rotation_log_prior=np.zeros(12, dtype=np.float32),
        random_perturbation=0.0,
        fine_rotations_override=score_rotations,
        fine_mstep_rotations_override=mstep_rotations,
        fine_rotation_parent_override=parent_map,
    )

    assert_matches(per_image["oversampled_rots"][0], score_rotations)
    assert_matches(per_image["oversampled_mstep_rots"][0], mstep_rotations)
    assert_matches(per_image["oversampled_rots"][1], score_rotations[2:])
    assert_matches(per_image["oversampled_mstep_rots"][1], mstep_rotations[2:])

    arrays = _build_bucket_arrays(
        {"bucket_size": 6, "image_indices": np.asarray([0, 1], dtype=np.int64)},
        per_image,
        n_fine_trans=1,
    )
    assert arrays["mstep_rotations"] is not arrays["rotations"]
    reordered_score, reordered_mstep = _reorder_to_indices(
        np.asarray([1, 0], dtype=np.int64),
        np.asarray([0, 1], dtype=np.int64),
        arrays["rotations"],
        arrays["mstep_rotations"],
    )
    identity = np.eye(3, dtype=np.float32)
    expected_score = np.stack(
        [
            np.concatenate([score_rotations[2:], np.broadcast_to(identity, (4, 3, 3))]),
            np.concatenate([score_rotations, np.broadcast_to(identity, (2, 3, 3))]),
        ]
    )
    expected_mstep = np.stack(
        [
            np.concatenate([mstep_rotations[2:], np.broadcast_to(identity, (4, 3, 3))]),
            np.concatenate([mstep_rotations, np.broadcast_to(identity, (2, 3, 3))]),
        ]
    )
    assert_matches(reordered_score, expected_score)
    assert_matches(reordered_mstep, expected_mstep)

    aliased = _prepare_per_image_pass2_inputs(
        significant_samples,
        n_coarse_rot=12,
        n_coarse_trans=1,
        nside_level=0,
        oversampling_order=0,
        n_fine_trans=1,
        fine_translation_parent=np.asarray([0], dtype=np.int32),
        rotation_log_prior=np.zeros(12, dtype=np.float32),
        random_perturbation=0.0,
        fine_rotations_override=score_rotations,
        fine_rotation_parent_override=parent_map,
    )
    aliased_arrays = _build_bucket_arrays(
        {"bucket_size": 6, "image_indices": np.asarray([0, 1], dtype=np.int64)},
        aliased,
        n_fine_trans=1,
    )
    assert aliased_arrays["mstep_rotations"] is aliased_arrays["rotations"]


def test_sparse_pass2_prepare_per_image_inputs_honors_explicit_float64_dtype():
    """``fine_rotations_override``/``fine_mstep_rotations_override`` must stay at
    whatever real dtype the caller requests (RELION's own RFLOAT fine-search
    rotations never narrow to float in a double-precision build) instead of
    always collapsing to float32.
    """
    score_rotations = np.stack([_z_rotation(angle) for angle in (0.1, 0.2, 0.3, 0.4)]).astype(np.float64)
    mstep_rotations = np.stack([_z_rotation(angle) for angle in (1.1, 1.2, 1.3, 1.4)]).astype(np.float64)
    parent_map = np.asarray([0, 0, 1, 1], dtype=np.int32)
    significant_samples = [np.asarray([0, 1], dtype=np.int32)]

    kwargs = dict(
        significant_sample_indices=significant_samples,
        n_coarse_rot=12,
        n_coarse_trans=1,
        nside_level=0,
        oversampling_order=0,
        n_fine_trans=1,
        fine_translation_parent=np.asarray([0], dtype=np.int32),
        rotation_log_prior=np.full(12, 1.0 + 2.0**-40, dtype=np.float64),
        random_perturbation=0.0,
        fine_rotations_override=score_rotations,
        fine_mstep_rotations_override=mstep_rotations,
        fine_rotation_parent_override=parent_map,
    )

    default_out = _prepare_per_image_pass2_inputs(**kwargs)
    assert default_out["oversampled_rots"][0].dtype == np.float32
    assert default_out["oversampled_mstep_rots"][0].dtype == np.float32

    f64_out = _prepare_per_image_pass2_inputs(**kwargs, dtype=np.float64)
    assert f64_out["oversampled_rots"][0].dtype == np.float64
    assert f64_out["oversampled_mstep_rots"][0].dtype == np.float64
    assert f64_out["log_prior"][0].dtype == np.float64
    assert f64_out["log_prior"][0][0] == 1.0 + 2.0**-40
    np.testing.assert_allclose(f64_out["oversampled_rots"][0], score_rotations, atol=1e-12)
    np.testing.assert_allclose(f64_out["oversampled_mstep_rots"][0], mstep_rotations, atol=1e-12)
    bucket = _build_bucket_arrays(
        {"bucket_size": 6, "image_indices": np.asarray([0], dtype=np.int64)},
        f64_out,
        n_fine_trans=1,
    )
    assert bucket["rotations"].dtype == np.float64
    assert bucket["mstep_rotations"].dtype == np.float64
    assert bucket["log_prior"].dtype == np.float64
    assert_matches(bucket["rotations"][0, :4], score_rotations)
    assert_matches(bucket["mstep_rotations"][0, :4], mstep_rotations)
    assert bucket["log_prior"][0, 0] == 1.0 + 2.0**-40
    np.testing.assert_allclose(
        f64_out["oversampled_rots"][0], default_out["oversampled_rots"][0], atol=1e-6
    )


