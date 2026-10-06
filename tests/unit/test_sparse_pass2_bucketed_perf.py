"""Sanity perf test: bucketed sparse pass-2 must NOT recompile per image.

The original ``compute_pass2_stats_sparse`` has a Python for-loop over
particles that calls ``run_em(..., image_batch_size=1, ...)`` once per
image, with a different XLA shape each time.  On the 5k fixture this
caused thousands of separate JIT compiles and made iter-1 take >50 min.

This test:
  * builds a synthetic dataset with N images that have *varied* per-image
    significant-rotation counts (the trigger for the recompile bug);
  * monkey-patches ``jax.jit`` so we count the number of distinct
    compiled trace cache keys produced during one call;
  * asserts the bucketed path produces ≪ N compiled programs (i.e.,
    bounded by the number of bucket sizes), whereas the per-image
    reference path would scale with the number of distinct rotation
    counts.

We use a tiny mock dataset so the test is fast on a login node.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers.float_compare import assert_matches

pytest.importorskip("jax")
import jax
import jax.numpy as jnp
import recovar.core as core
import recovar.core.fourier_transform_utils as ftu
from helpers.em_arrays import _raw_real_image_2d
from helpers.fine_grid_significance_reference import _build_fine_grid_significance_mask
from recovar.core.configs import ForwardModelConfig

from relax.classification.k_class import _fine_support_stats
from relax.helpers.fourier_window import make_fourier_window_spec
from relax.helpers.preprocessing import apply_half_translation_phases, half_translation_phase_table
from relax.local.local_backprojection import (
    compute_local_ctf_sums,
    compute_local_ctf_sums_from_probs_sum_t,
)
from relax.scoring.significant_samples import (
    ComplementSignificantSampleIndices,
)
from relax.scoring.sparse_bucket_arrays import (
    _bucket_pass2_inputs,
    _coalesce_tail_bucket_sizes,
    _prepare_per_image_pass2_inputs,
)
from relax.sparse_pass2.sparse_pass2_adjoint import _accumulate_adjoint_block_chunked, _adjoint_block_chunk_rows
from relax.sparse_pass2.sparse_pass2_bucket_io import (
    _half_translation_phase_table_for_indices,
    _prepare_bucket_io,
    _relion_translation_angles_f32,
)
from relax.sparse_pass2.sparse_pass2_budget import (
    _max_projected_rotations_per_call_for_pass,
    _nvidia_smi_visible_device_memory_bytes,
    _projection_budget_pixels_for_pass,
    _projection_cache_budget_complex_dtype,
    _projection_cache_fits_budget,
    _projection_cache_transient_bytes,
)
from relax.sparse_pass2.sparse_pass2_policy import (
    _native_dual_weighted_sums_supported_for_operands,
    _projection_cache_enabled_for_pass,
)
from relax.sparse_pass2.sparse_pass2_projection_blocks import (
    _compute_sparse_pass2_projections_block,
    _compute_sparse_pass2_windowed_projections_block,
)
from relax.sparse_pass2.sparse_pass2_scoring import (
    _relion_cuda_corr_img_from_native_noise_variance,
    _relion_cuda_corr_img_from_rfloat_ctf,
    _relion_cuda_pixel_correction_from_rfloat_ctf,
)

pytestmark = pytest.mark.unit


def test_relion_corr_img_squares_rfloat_ctf_before_xfloat_cast():
    inverse_noise = np.asarray([0.13333298, 1.750001], dtype=np.float32)
    ctf_rfloat = np.asarray([0.994443123456, -0.7135792468], dtype=np.float64)
    expected = np.asarray(
        inverse_noise.astype(np.float64) * (ctf_rfloat * ctf_rfloat),
        dtype=np.float32,
    )
    float_ctf = ctf_rfloat.astype(np.float32)
    rejected_float_path = np.asarray(
        inverse_noise * (float_ctf * float_ctf),
        dtype=np.float32,
    )
    assert np.any(expected != rejected_float_path)

    actual = np.asarray(
        _relion_cuda_corr_img_from_rfloat_ctf(inverse_noise, ctf_rfloat)
    )
    assert_matches(actual, expected)


def test_relion_corr_img_applies_xfloat_scale_square_after_rfloat_ctf_cast():
    inverse_noise = np.asarray([0.7455480098724365], dtype=np.float32)
    ctf_rfloat = np.asarray([-0.48569034637572805], dtype=np.float64)
    scale = np.asarray([[0.30007338523864746]], dtype=np.float32)
    corr_unscaled = np.asarray(
        inverse_noise.astype(np.float64) * (ctf_rfloat * ctf_rfloat),
        dtype=np.float32,
    )
    scale_squared = np.asarray(scale * scale, dtype=np.float32)
    expected = np.asarray(corr_unscaled * scale_squared, dtype=np.float32)
    rejected_fused_rfloat_path = np.asarray(
        inverse_noise.astype(np.float64)
        * (ctf_rfloat * ctf_rfloat)
        * scale.astype(np.float64)
        * scale.astype(np.float64),
        dtype=np.float32,
    )
    assert np.any(expected != rejected_fused_rfloat_path)

    actual = np.asarray(
        _relion_cuda_corr_img_from_rfloat_ctf(
            inverse_noise,
            ctf_rfloat,
            scale,
        )
    )
    assert_matches(actual, expected)


def test_relion_corr_img_preserves_double_accelerator_precision():
    inverse_noise = np.asarray([0.133332981234, 1.7500012345], dtype=np.float64)
    ctf_rfloat = np.asarray([0.994443123456, -0.7135792468], dtype=np.float64)
    expected = inverse_noise * (ctf_rfloat * ctf_rfloat)

    actual = np.asarray(
        _relion_cuda_corr_img_from_rfloat_ctf(
            inverse_noise,
            ctf_rfloat,
            output_dtype=jnp.float64,
        )
    )
    assert actual.dtype == np.float64
    assert_matches(actual, expected)


def test_relion_corr_img_converts_noise_units_after_native_xfloat_product():
    image_shape = (384, 384)
    native_noise_variance = np.asarray(
        [1.262594e-6, 1.36236e-6],
        dtype=np.float64,
    )
    fourier_scale = np.float64(image_shape[0] ** 4)
    recovar_noise_variance = native_noise_variance * fourier_scale
    ctf_rfloat = np.asarray([0.994443123456, -0.7135792468], dtype=np.float64)
    scale = np.asarray([1.0, 0.30007338523864746], dtype=np.float32)
    native_inverse_noise = np.asarray(1.0 / native_noise_variance, dtype=np.float32)
    native_corr = np.asarray(
        native_inverse_noise.astype(np.float64) * (ctf_rfloat * ctf_rfloat),
        dtype=np.float32,
    )
    native_corr = np.asarray(native_corr * (scale * scale), dtype=np.float32)
    expected = np.asarray(
        native_corr / np.float32(image_shape[0] ** 4),
        dtype=np.float32,
    )
    rejected_early_conversion = np.asarray(
        np.asarray(1.0 / recovar_noise_variance, dtype=np.float32).astype(np.float64)
        * (ctf_rfloat * ctf_rfloat),
        dtype=np.float32,
    )
    rejected_early_conversion = np.asarray(
        rejected_early_conversion * (scale * scale),
        dtype=np.float32,
    )
    assert np.any(expected != rejected_early_conversion)

    actual = np.asarray(
        _relion_cuda_corr_img_from_native_noise_variance(
            recovar_noise_variance,
            ctf_rfloat,
            image_shape,
            scale,
        )
    )
    assert_matches(actual, expected)


def test_relion_pixel_correction_divides_by_rfloat_ctf_before_xfloat_cast():
    scale = np.asarray([[1.0]], dtype=np.float32)
    ctf_rfloat = np.asarray(
        [[0.07354116995482596, 0.1265216380265534]], dtype=np.float64
    )
    initial = np.asarray(1.0 / scale, dtype=np.float32)
    expected = np.asarray(
        initial.astype(np.float64) / ctf_rfloat,
        dtype=np.float32,
    )
    rejected_float_path = np.asarray(
        initial / ctf_rfloat.astype(np.float32),
        dtype=np.float32,
    )
    assert np.any(expected != rejected_float_path)

    actual = np.asarray(
        _relion_cuda_pixel_correction_from_rfloat_ctf(scale, ctf_rfloat)
    )
    assert_matches(actual, expected)


def test_relion_pixel_correction_preserves_double_accelerator_precision():
    scale = np.asarray([[1.0000000123]], dtype=np.float64)
    ctf_rfloat = np.asarray(
        [[0.07354116995482596, 0.1265216380265534]], dtype=np.float64
    )
    expected = (1.0 / scale) / ctf_rfloat

    actual = np.asarray(
        _relion_cuda_pixel_correction_from_rfloat_ctf(
            scale,
            ctf_rfloat,
            output_dtype=jnp.float64,
        )
    )
    assert actual.dtype == np.float64
    assert_matches(actual, expected)


# Mock dataset (mirrors test_sparse_pass2_bucketed_parity.MockDataset).
IMAGE_SHAPE = (8, 8)
IMAGE_SIZE = 64
VOLUME_SHAPE = (8, 8, 8)
VOLUME_SIZE = 512


def test_compute_local_ctf_sums_from_probs_sum_t_matches_dense_helper():
    rng = np.random.default_rng(112358)
    probs = rng.random((3, 5, 7), dtype=np.float32)
    probs[0, 2, :] = 0.0
    ctf2_over_nv = 0.25 + rng.random((3, 11), dtype=np.float32)

    dense = compute_local_ctf_sums(jnp.asarray(probs), jnp.asarray(ctf2_over_nv))
    from_probs_sum = compute_local_ctf_sums_from_probs_sum_t(
        jnp.sum(jnp.asarray(probs), axis=-1),
        jnp.asarray(ctf2_over_nv),
    )

    np.testing.assert_allclose(np.asarray(from_probs_sum), np.asarray(dense), rtol=1e-6, atol=1e-6)
    assert_matches(np.asarray(from_probs_sum)[0, 2], np.zeros_like(np.asarray(from_probs_sum)[0, 2]))


def test_pass2_projection_cache_override_supports_matched_dump_ab(monkeypatch):
    fine_rotations = np.zeros((3, 3, 3), dtype=np.float32)
    monkeypatch.delenv("RELAX_SPARSE_PASS2_PROJECTION_CACHE", raising=False)

    assert _projection_cache_enabled_for_pass(
        fine_rotations_override=fine_rotations,
        dump_pass2_operands=False,
    )
    assert not _projection_cache_enabled_for_pass(
        fine_rotations_override=fine_rotations,
        dump_pass2_operands=True,
    )

    monkeypatch.setenv("RELAX_SPARSE_PASS2_PROJECTION_CACHE", "on")
    assert _projection_cache_enabled_for_pass(
        fine_rotations_override=fine_rotations,
        dump_pass2_operands=False,
    )
    assert _projection_cache_enabled_for_pass(
        fine_rotations_override=fine_rotations,
        dump_pass2_operands=True,
    )

    monkeypatch.setenv("RELAX_SPARSE_PASS2_PROJECTION_CACHE", "off")
    assert not _projection_cache_enabled_for_pass(
        fine_rotations_override=fine_rotations,
        dump_pass2_operands=False,
    )
    assert not _projection_cache_enabled_for_pass(
        fine_rotations_override=fine_rotations,
        dump_pass2_operands=True,
    )
    assert not _projection_cache_enabled_for_pass(
        fine_rotations_override=None,
        dump_pass2_operands=False,
    )

    monkeypatch.setenv("RELAX_SPARSE_PASS2_PROJECTION_CACHE", "invalid")
    with pytest.raises(ValueError, match="must be 'auto', 'on', or 'off'"):
        _projection_cache_enabled_for_pass(
            fine_rotations_override=fine_rotations,
            dump_pass2_operands=False,
        )


def _assert_relion_stats_close(actual, expected, *, rtol=1e-5, atol=1e-5):
    np.testing.assert_allclose(
        np.asarray(actual.log_evidence_per_image),
        np.asarray(expected.log_evidence_per_image),
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        np.asarray(actual.best_log_score_per_image),
        np.asarray(expected.best_log_score_per_image),
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        np.asarray(actual.max_posterior_per_image),
        np.asarray(expected.max_posterior_per_image),
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        np.asarray(actual.rotation_posterior_sums),
        np.asarray(expected.rotation_posterior_sums),
        rtol=rtol,
        atol=atol,
    )


def _assert_noise_stats_close(actual, expected, *, rtol=1e-5, atol=1e-5):
    if actual is None or expected is None:
        assert actual is None and expected is None
        return
    assert len(actual) == len(expected)
    for actual_stats, expected_stats in zip(actual, expected, strict=True):
        np.testing.assert_allclose(
            np.asarray(actual_stats.wsum_sigma2_noise),
            np.asarray(expected_stats.wsum_sigma2_noise),
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            np.asarray(actual_stats.wsum_img_power),
            np.asarray(expected_stats.wsum_img_power),
            rtol=rtol,
            atol=atol,
        )
        actual_norm = getattr(actual_stats, "wsum_norm_correction", None)
        expected_norm = getattr(expected_stats, "wsum_norm_correction", None)
        if actual_norm is None or expected_norm is None:
            assert actual_norm is None and expected_norm is None
        else:
            np.testing.assert_allclose(
                np.asarray(actual_norm),
                np.asarray(expected_norm),
                rtol=rtol,
                atol=atol,
            )
        for field in ("wsum_scale_correction_xa", "wsum_scale_correction_aa"):
            actual_scale = getattr(actual_stats, field, None)
            expected_scale = getattr(expected_stats, field, None)
            if actual_scale is None or expected_scale is None:
                assert actual_scale is None and expected_scale is None
            else:
                np.testing.assert_allclose(
                    np.asarray(actual_scale),
                    np.asarray(expected_scale),
                    rtol=rtol,
                    atol=atol,
                )
        assert actual_stats.wsum_sigma2_offset == pytest.approx(expected_stats.wsum_sigma2_offset, abs=atol)
        assert actual_stats.sumw == pytest.approx(expected_stats.sumw, abs=atol)


def _assert_noise_residual_terms_close(actual, expected, *, rtol=1e-5, atol=1e-5):
    if actual is None or expected is None:
        assert actual is None and expected is None
        return
    assert len(actual) == len(expected)
    for actual_stats, expected_stats in zip(actual, expected, strict=True):
        np.testing.assert_allclose(
            np.asarray(actual_stats.wsum_sigma2_noise),
            np.asarray(expected_stats.wsum_sigma2_noise),
            rtol=rtol,
            atol=atol,
        )
        assert actual_stats.wsum_sigma2_offset == pytest.approx(expected_stats.wsum_sigma2_offset, abs=atol)


def _assert_k_class_noise_sumw_matches_class_mass(result, *, rtol=1e-5, atol=1e-5):
    assert result.noise_stats is not None
    class_mass_source = result.class_mstep_posterior_sums
    if class_mass_source is None:
        class_mass_source = result.class_posterior_sums
    class_mass = np.asarray(class_mass_source, dtype=np.float64)
    for stats, expected_mass in zip(result.noise_stats, class_mass, strict=True):
        assert float(stats.sumw) == pytest.approx(float(expected_mass), rel=rtol, abs=atol)
    if result.aggregate_noise_stats is not None:
        assert float(result.aggregate_noise_stats.sumw) == pytest.approx(float(np.sum(class_mass)), rel=rtol, abs=atol)
        np.testing.assert_allclose(
            np.asarray(result.aggregate_noise_stats.wsum_img_power),
            np.sum([np.asarray(stats.wsum_img_power) for stats in result.noise_stats], axis=0),
            rtol=rtol,
            atol=atol,
        )


def _assert_best_pose_outputs_close(actual, expected, *, rtol=1e-6, atol=1e-6):
    for field in (
        "per_class_best_pose_rotations",
        "per_class_best_pose_translations",
        "per_class_best_pose_rotation_ids",
        "best_pose_rotations",
        "best_pose_translations",
        "best_pose_rotation_ids",
    ):
        actual_value = getattr(actual, field)
        expected_value = getattr(expected, field)
        if actual_value is None or expected_value is None:
            assert actual_value is None and expected_value is None
            continue
        if isinstance(actual_value, tuple):
            assert len(actual_value) == len(expected_value)
            for actual_item, expected_item in zip(actual_value, expected_value, strict=True):
                np.testing.assert_allclose(
                    np.asarray(actual_item),
                    np.asarray(expected_item),
                    rtol=rtol,
                    atol=atol,
                )
        else:
            np.testing.assert_allclose(
                np.asarray(actual_value),
                np.asarray(expected_value),
                rtol=rtol,
                atol=atol,
            )


def _assert_k_class_extra_outputs_close(actual, expected, *, rtol=1e-5, atol=1e-5):
    _assert_relion_stats_close(actual.stats, expected.stats, rtol=rtol, atol=atol)
    assert len(actual.per_class_stats) == len(expected.per_class_stats)
    for actual_stats, expected_stats in zip(actual.per_class_stats, expected.per_class_stats, strict=True):
        _assert_relion_stats_close(actual_stats, expected_stats, rtol=rtol, atol=atol)
    _assert_noise_stats_close(actual.noise_stats, expected.noise_stats, rtol=rtol, atol=atol)
    _assert_noise_stats_close(
        None if actual.aggregate_noise_stats is None else (actual.aggregate_noise_stats,),
        None if expected.aggregate_noise_stats is None else (expected.aggregate_noise_stats,),
        rtol=rtol,
        atol=atol,
    )
    _assert_best_pose_outputs_close(actual, expected)


def _identity_ctf(params, image_shape=None, voxel_size=None, *, half_image=False):
    if half_image:
        h, w = image_shape if image_shape is not None else IMAGE_SHAPE
        sz = h * (w // 2 + 1)
    else:
        sz = IMAGE_SIZE
    return jnp.ones((params.shape[0], sz), dtype=jnp.float32)


def _raw_real_process(batch, apply_image_mask=False):
    _ = apply_image_mask
    images = jnp.asarray(batch)
    return ftu.get_dft2(images).reshape((images.shape[0], -1)).astype(jnp.complex64)


def _raw_real_process_half(batch, apply_image_mask=False):
    _ = apply_image_mask
    images = jnp.asarray(batch)
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


def test_bucket_count_bounded_under_varied_per_image_rotation_counts():
    """Number of buckets must be bounded by the number of unique quantized sizes,
    not by the number of distinct per-image counts (and certainly not by N_images).
    """
    rng = np.random.default_rng(7)
    n_images = 500
    n_coarse_rot = 48
    n_coarse_trans = 2
    n_fine_trans = 2 * (4**1)  # = 8

    # Random per-image significant counts in [1, 20] — many distinct counts.
    counts = rng.integers(low=1, high=21, size=n_images)
    # Build (rot * n_coarse_trans + trans) flat indices: pick coarse rot, pair with trans 0
    # so candidate_mask is non-empty (fine trans 0 is parent of trans 0).
    sig_indices = [
        (rng.choice(n_coarse_rot, size=int(c), replace=False).astype(np.int32) * n_coarse_trans).astype(np.int32)
        for c in counts
    ]

    # Build per-image inputs the way compute_pass2_stats_sparse_bucketed does.
    from relax.scoring.sparse_bucket_arrays import _prepare_per_image_pass2_inputs

    # fine_translation_parent maps fine trans -> coarse trans. With oversampling=1
    # in 2D, each coarse trans expands to 4 children, so trans 0..3 map to coarse 0,
    # trans 4..7 map to coarse 1.
    fine_trans_parent = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)

    per_image = _prepare_per_image_pass2_inputs(
        sig_indices,
        n_coarse_rot=n_coarse_rot,
        n_coarse_trans=n_coarse_trans,
        nside_level=1,
        oversampling_order=1,
        n_fine_trans=n_fine_trans,
        fine_translation_parent=fine_trans_parent,
        rotation_log_prior=None,
        random_perturbation=0.0,
    )

    buckets = _bucket_pass2_inputs(per_image, n_fine_trans=n_fine_trans)
    n_distinct_counts = len({int(rots.shape[0]) for rots in per_image["oversampled_rots"]})

    # Quantization should collapse many distinct counts into a few buckets.
    assert len(buckets) < n_distinct_counts + 1, (
        f"Expected fewer than {n_distinct_counts + 1} buckets after quantization, got {len(buckets)}."
    )
    # Must be much smaller than n_images — that's the whole point.
    assert len(buckets) < n_images / 10, f"Got {len(buckets)} buckets for {n_images} images — bucketing too granular."


def test_default_sparse_pass2_budget_keeps_broad_support_batched():
    """Broad soft K-class supports must not fall back to one image per launch."""

    n_images = 26
    n_fine_trans = 116
    n_rot = 1024
    per_image = {
        "oversampled_rots": [np.zeros((n_rot, 3, 3), dtype=np.float32) for _ in range(n_images)],
    }

    default_buckets = _bucket_pass2_inputs(
        per_image,
        n_fine_trans=n_fine_trans,
        max_images_per_microbatch=13,
    )
    old_cap_buckets = _bucket_pass2_inputs(
        per_image,
        n_fine_trans=n_fine_trans,
        max_hypotheses_per_microbatch=100_000,
        max_images_per_microbatch=13,
    )

    assert [len(bucket["image_indices"]) for bucket in default_buckets] == [8, 8, 8, 2]
    assert len(old_cap_buckets) == n_images


def test_single_class_sparse_pass2_can_coalesce_small_bucket_tail(monkeypatch):
    """Small sparse tails can opt into fewer execution shapes."""

    monkeypatch.delenv("RELAX_LOCAL_BUCKET_QUANTUM", raising=False)
    n_fine_trans = 116
    counts = [16] * 12 + [32] * 7 + [64] * 5 + [128] * 3 + [256]
    per_image = {
        "oversampled_rots": [
            np.zeros((int(count), 3, 3), dtype=np.float32)
            for count in counts
        ],
    }

    baseline = _bucket_pass2_inputs(
        per_image,
        n_fine_trans=n_fine_trans,
        max_hypotheses_per_microbatch=10**12,
        max_images_per_microbatch=1000,
    )
    coalesced = _bucket_pass2_inputs(
        per_image,
        n_fine_trans=n_fine_trans,
        max_hypotheses_per_microbatch=10**12,
        max_images_per_microbatch=1000,
        small_bucket_coalesce_size=128,
    )

    assert sorted({int(bucket["bucket_size"]) for bucket in baseline}) == [16, 32, 64, 128, 256]
    assert sorted({int(bucket["bucket_size"]) for bucket in coalesced}) == [128, 256]
    assert len(coalesced) < len(baseline)
    assert sum(len(bucket["image_indices"]) for bucket in coalesced) == len(counts)


def test_sparse_pass2_tail_bucket_coalescing_merges_only_bounded_high_tail():
    bucket_sizes = np.asarray(
        [4096] * 20 + [8192] * 2 + [12288] * 3 + [16384] * 2,
        dtype=np.int64,
    )

    coalesced = _coalesce_tail_bucket_sizes(
        bucket_sizes,
        max_images=8,
        max_inflation=2.0,
        min_bucket_size=4096,
        max_images_per_microbatch=1000,
    )

    assert sorted(np.unique(coalesced).astype(int).tolist()) == [4096, 16384]
    assert np.count_nonzero(coalesced == 4096) == 20
    assert np.count_nonzero(coalesced == 16384) == 7


def test_sparse_pass2_tail_bucket_coalescing_respects_inflation_cap():
    bucket_sizes = np.asarray([4096, 4096, 16384], dtype=np.int64)

    coalesced = _coalesce_tail_bucket_sizes(
        bucket_sizes,
        max_images=3,
        max_inflation=1.2,
        min_bucket_size=4096,
        max_images_per_microbatch=1000,
    )

    assert_matches(coalesced, bucket_sizes)


def test_relion_windowed_projection_budget_accounts_for_centered_full_half_transient(monkeypatch):
    monkeypatch.delenv("RELAX_SPARSE_PASS2_MAX_PROJECTED_ROTATIONS", raising=False)
    monkeypatch.delenv("RELAX_SPARSE_PASS2_PROJECTION_CACHE_MAX_BYTES", raising=False)

    large_gpu = 80 * 1024**3
    n_half = 128 * (128 // 2 + 1)
    default_pixels = _projection_budget_pixels_for_pass(
        n_half,
        use_window=True,
        use_relion_projector=False,
    )
    relion_pixels = _projection_budget_pixels_for_pass(
        n_half,
        use_window=True,
        use_relion_projector=True,
    )

    assert default_pixels == n_half
    assert relion_pixels == 8 * n_half

    default_cap = _max_projected_rotations_per_call_for_pass(
        device_memory_bytes=large_gpu,
        n_projection_pixels=default_pixels,
        projection_complex_dtype=np.complex64,
        include_abs2=False,
    )
    relion_dtype = _projection_cache_budget_complex_dtype(
        np.complex64,
        np.complex64,
        use_relion_projector=True,
    )
    relion_cap = _max_projected_rotations_per_call_for_pass(
        device_memory_bytes=large_gpu,
        n_projection_pixels=relion_pixels,
        projection_complex_dtype=relion_dtype,
        include_abs2=False,
    )

    assert default_cap == 51622
    assert relion_cap == 3226

    default_cap64 = _max_projected_rotations_per_call_for_pass(
        device_memory_bytes=large_gpu,
        n_projection_pixels=default_pixels,
        projection_complex_dtype=np.complex128,
        include_abs2=False,
    )
    relion_cap64 = _max_projected_rotations_per_call_for_pass(
        device_memory_bytes=large_gpu,
        n_projection_pixels=relion_pixels,
        projection_complex_dtype=np.complex128,
        include_abs2=False,
    )

    assert default_cap64 == 25811
    assert relion_cap64 == 3226

    relion_chunk_bytes = _projection_cache_transient_bytes(
        relion_cap,
        n_half,
        projection_complex_dtype=relion_dtype,
        include_abs2=False,
    )
    assert relion_chunk_bytes <= 512 * 1024**2

    monkeypatch.setenv("RELAX_SPARSE_PASS2_PROJECTION_CACHE_MAX_BYTES", "0")
    assert (
        _max_projected_rotations_per_call_for_pass(
            device_memory_bytes=large_gpu,
            n_projection_pixels=relion_pixels,
            projection_complex_dtype=np.complex64,
            include_abs2=False,
        )
        is None
    )


def test_sparse_pass2_adjoint_block_chunking_accumulates_all_rows(monkeypatch):
    from relax.sparse_pass2 import (
        sparse_pass2_adjoint,
    )

    flat_block = jnp.arange(30, dtype=jnp.float32).reshape(10, 3)
    rotations = jnp.zeros((10, 3, 3), dtype=jnp.float32)
    volume = jnp.zeros(2, dtype=jnp.float32)
    calls = []

    def fake_adjoint_slice_volume_half(
        half_block,
        rotations_block,
        volume_in,
        image_shape,
        volume_shape,
        disc_type,
        half_image,
        half_volume=False,
    ):
        del rotations_block, image_shape, volume_shape, disc_type, half_image, half_volume
        calls.append(int(half_block.shape[0]))
        return volume_in + jnp.sum(half_block)

    monkeypatch.setattr(sparse_pass2_adjoint, "_adjoint_slice_volume_half", fake_adjoint_slice_volume_half)

    row_bytes = flat_block.shape[1] * np.dtype(np.float32).itemsize
    assert _adjoint_block_chunk_rows(flat_block, max_block_bytes=4 * row_bytes + 1) == 4
    # The chunks run in one program per chunk shape, which traces the fake: start and end without
    # a program compiled around another adjoint.
    sparse_pass2_adjoint._adjoint_rows.clear_cache()
    try:
        actual = _accumulate_adjoint_block_chunked(
            flat_block,
            rotations,
            volume,
            use_windowed_adjoint=False,
            image_shape=(8, 8),
            volume_shape=(8, 8, 8),
            disc_type="linear_interp",
            half_image=True,
            half_volume=False,
            max_r=None,
            relion_x_half=False,
            max_block_bytes=4 * row_bytes + 1,
            log_label="test",
        )
    finally:
        sparse_pass2_adjoint._adjoint_rows.clear_cache()

    # Rows 0-3 and 4-7 share the four-row program; rows 8-9 trace the second.
    assert calls == [4, 2]
    np.testing.assert_allclose(np.asarray(actual), np.asarray(volume + jnp.sum(flat_block)))


def test_relion_x_half_bp_per_particle_launch_is_off_by_default(monkeypatch):
    from relax.diagnostics import bpref_diagnostics

    monkeypatch.delenv("RELAX_RELION_X_HALF_BP_PER_PARTICLE_LAUNCH", raising=False)
    assert bpref_diagnostics.relion_x_half_bp_per_particle_launch_enabled() is False
    monkeypatch.setenv("RELAX_RELION_X_HALF_BP_PER_PARTICLE_LAUNCH", "1")
    assert bpref_diagnostics.relion_x_half_bp_per_particle_launch_enabled() is True


def test_relion_x_half_bp_fused_atomics_is_off_by_default(monkeypatch):
    from relax.diagnostics import bpref_diagnostics

    monkeypatch.delenv("RELAX_RELION_X_HALF_BP_FUSED_ATOMICS", raising=False)
    assert bpref_diagnostics.relion_x_half_bp_fused_atomics_enabled() is False
    monkeypatch.setenv("RELAX_RELION_X_HALF_BP_FUSED_ATOMICS", "1")
    assert bpref_diagnostics.relion_x_half_bp_fused_atomics_enabled() is True


def test_sparse_pass2_projection_cache_estimate_accounts_for_projection_dtype():
    n_rot = 11
    n_half = 17

    assert _projection_cache_budget_complex_dtype(jnp.complex64, jnp.complex64) == np.dtype(np.complex64)
    assert _projection_cache_budget_complex_dtype(jnp.complex64, jnp.complex128) == np.dtype(np.complex128)
    assert _projection_cache_budget_complex_dtype(jnp.complex128, jnp.complex64) == np.dtype(np.complex128)
    assert _projection_cache_budget_complex_dtype(
        jnp.complex64,
        jnp.complex64,
        use_relion_projector=True,
    ) == np.dtype(np.complex128)
    assert _projection_cache_transient_bytes(
        n_rot,
        n_half,
        projection_complex_dtype=jnp.complex64,
        include_abs2=True,
    ) == n_rot * n_half * (np.dtype(np.complex64).itemsize + np.dtype(np.float32).itemsize)
    assert _projection_cache_transient_bytes(
        n_rot,
        n_half,
        projection_complex_dtype=jnp.complex128,
        include_abs2=True,
    ) == n_rot * n_half * (np.dtype(np.complex128).itemsize + np.dtype(np.float64).itemsize)
    assert _projection_cache_transient_bytes(
        n_rot,
        n_half,
        projection_complex_dtype=jnp.complex128,
        include_abs2=False,
    ) == n_rot * n_half * np.dtype(np.complex128).itemsize


def test_sparse_pass2_projection_cache_budget_accounts_for_k_classes():
    assert _projection_cache_fits_budget(100, 250, n_classes=2)
    assert not _projection_cache_fits_budget(100, 250, n_classes=3)


def test_relion_windowed_projection_cache_estimate_admits_retained_window_cache():
    n_fine_rot = 147_456
    n_half = 128 * (128 // 2 + 1)
    n_windowed = 1_624
    n_recon_windowed = 1_227
    cap_12g = 12 * 1024**3

    retained_window_cache_estimate = _projection_cache_transient_bytes(
        n_fine_rot,
        n_windowed,
        projection_complex_dtype=np.complex64,
        include_abs2=False,
    ) + _projection_cache_transient_bytes(
        n_fine_rot,
        n_recon_windowed,
        projection_complex_dtype=np.complex64,
        include_abs2=True,
    )
    relion_centered_transient_estimate = _projection_cache_transient_bytes(
        n_fine_rot,
        _projection_budget_pixels_for_pass(
            n_half,
            use_window=True,
            use_relion_projector=True,
        ),
        projection_complex_dtype=_projection_cache_budget_complex_dtype(
            np.complex64,
            np.complex64,
            use_relion_projector=True,
        ),
        include_abs2=False,
    )

    assert retained_window_cache_estimate < cap_12g
    assert relion_centered_transient_estimate > cap_12g
    assert _projection_cache_fits_budget(retained_window_cache_estimate, cap_12g)

    per_call_cap = _max_projected_rotations_per_call_for_pass(
        device_memory_bytes=80 * 1024**3,
        n_projection_pixels=_projection_budget_pixels_for_pass(
            n_half,
            use_window=True,
            use_relion_projector=True,
        ),
        projection_complex_dtype=_projection_cache_budget_complex_dtype(
            np.complex64,
            np.complex64,
            use_relion_projector=True,
        ),
        include_abs2=False,
    )
    assert per_call_cap is not None
    assert per_call_cap < n_fine_rot


def test_single_class_sparse_pass2_bucket_quantum_can_coarsen_pathological_tail(monkeypatch):
    """Document tail coarsening for outlier-heavy K=1 cases."""

    monkeypatch.delenv("RELAX_LOCAL_BUCKET_QUANTUM", raising=False)
    n_fine_trans = 116
    counts = [1812, 4461, 9728, 12801, 23041, 45057, 76800]
    per_image_inputs = {
        "oversampled_rots": [
            np.broadcast_to(np.eye(3, dtype=np.float32), (count, 3, 3)).copy()
            for count in counts
        ],
    }

    default_buckets = _bucket_pass2_inputs(
        per_image_inputs,
        n_fine_trans=n_fine_trans,
        rotation_block_size_for_quantization=5000,
        max_hypotheses_per_microbatch=10**12,
        max_images_per_microbatch=1000,
    )
    default_sizes = [int(bucket["bucket_size"]) for bucket in default_buckets]

    monkeypatch.setenv("RELAX_LOCAL_BUCKET_QUANTUM", "512")
    fine_buckets = _bucket_pass2_inputs(
        per_image_inputs,
        n_fine_trans=n_fine_trans,
        rotation_block_size_for_quantization=5000,
        max_hypotheses_per_microbatch=10**12,
        max_images_per_microbatch=1000,
    )
    fine_sizes = [int(bucket["bucket_size"]) for bucket in fine_buckets]

    assert default_sizes == [4096, 8192, 12288, 16384, 24576, 49152, 77824]
    assert fine_sizes == [2048, 4608, 9728, 13312, 23552, 45568, 76800]
    assert len(set(default_sizes)) <= len(set(fine_sizes))
    assert default_sizes[-1] >= counts[-1]


def test_sparse_pass2_auto_projection_cap_prevents_one_image_tail_oversize(monkeypatch):
    monkeypatch.delenv("RELAX_SPARSE_PASS2_MAX_PROJECTED_ROTATIONS", raising=False)
    n_half = 128 * (128 // 2 + 1)
    cap = _max_projected_rotations_per_call_for_pass(
        device_memory_bytes=80 * 1024**3,
        n_projection_pixels=n_half,
        projection_complex_dtype=np.complex64,
        include_abs2=False,
    )
    assert cap is not None
    assert cap < 90_112
    assert cap > 10_000

    cap_with_abs2 = _max_projected_rotations_per_call_for_pass(
        device_memory_bytes=80 * 1024**3,
        n_projection_pixels=n_half,
        projection_complex_dtype=np.complex64,
        include_abs2=True,
    )
    assert cap_with_abs2 is not None
    assert cap_with_abs2 < cap

    monkeypatch.setenv("RELAX_SPARSE_PASS2_MAX_PROJECTED_ROTATIONS", "1234")
    assert (
        _max_projected_rotations_per_call_for_pass(
            device_memory_bytes=80 * 1024**3,
            n_projection_pixels=n_half,
            projection_complex_dtype=np.complex64,
            include_abs2=False,
        )
        == 1234
    )


def _minimal_per_image_inputs(rotation_counts, n_fine_trans):
    oversampled_rots = [
        np.broadcast_to(np.eye(3, dtype=np.float32), (int(count), 3, 3)).copy()
        for count in rotation_counts
    ]
    return {
        "oversampled_rots": oversampled_rots,
        "oversampled_mstep_rots": oversampled_rots,
        "oversampled_rot_indices": [
            np.arange(int(count), dtype=np.int64)
            for count in rotation_counts
        ],
        "unique_rot": [
            np.arange(int(count), dtype=np.int32)
            for count in rotation_counts
        ],
        "parent_map": [
            np.arange(int(count), dtype=np.int32)
            for count in rotation_counts
        ],
        "log_prior": [
            np.zeros(int(count), dtype=np.float32)
            for count in rotation_counts
        ],
        "candidate_mask": [
            np.ones((int(count), n_fine_trans), dtype=bool)
            for count in rotation_counts
        ],
    }


def _make_compact_pair_sparse_mstep_case(dtype=np.float64):
    rng = np.random.default_rng(241)
    batch = 4
    n_rot = 5
    n_trans = 6
    n_pairs = 12
    n_pixels = 7
    local_rotation_row = np.asarray(
        [
            [1, 1, 3, 4, -1, 0, 2, 2, 5, 1, 0, 0],
            [0, 2, 2, 2, 3, 3, 1, 1, 4, -2, 0, 0],
            [4, 4, 4, 1, 1, 0, 2, 3, 3, 0, 0, 0],
            [2, 2, 0, 0, 1, 4, 4, 3, 3, 3, 0, 0],
        ],
        dtype=np.int32,
    )
    translation_idx = np.asarray(
        [
            [2, 2, 1, 4, 3, 0, 5, 5, 0, 6, 1, 0],
            [1, 3, 3, 4, 0, 0, 5, 5, 2, 2, -1, 0],
            [0, 0, 2, 2, 2, 1, 4, 4, 4, 0, 0, 0],
            [5, 5, 2, 2, 3, 1, 1, 4, 0, 6, 0, 0],
        ],
        dtype=np.int32,
    )
    pair_mask = np.asarray(
        [
            [True, True, True, True, True, False, True, True, True, True, False, False],
            [True, True, True, True, True, True, True, True, True, True, True, False],
            [True, True, True, True, True, False, True, True, True, False, False, False],
            [True, True, True, True, True, True, True, True, False, True, False, False],
        ],
        dtype=bool,
    )
    pair_probs = rng.random((batch, n_pairs)).astype(dtype)
    pair_probs[0, 5] = np.nan
    pair_probs[0, 10] = np.inf
    pair_probs[1, 9] = np.nan
    pair_probs[1, 10] = np.inf
    pair_probs[2, 9:] = np.nan
    pair_probs[3, 8] = np.inf
    pair_probs[3, 10:] = np.nan
    complex_dtype = np.complex128 if dtype == np.float64 else np.complex64
    shifted_recon = (
        rng.standard_normal((batch, n_trans, n_pixels)).astype(dtype)
        + 1j * rng.standard_normal((batch, n_trans, n_pixels)).astype(dtype)
    ).astype(complex_dtype)
    shifted_image = (
        rng.standard_normal((batch, n_trans, n_pixels)).astype(dtype)
        + 1j * rng.standard_normal((batch, n_trans, n_pixels)).astype(dtype)
    ).astype(complex_dtype)
    ctf2_over_nv = (0.25 + rng.random((batch, n_pixels))).astype(dtype)
    return {
        "pair_probs": pair_probs,
        "local_rotation_row": local_rotation_row,
        "translation_idx": translation_idx,
        "pair_mask": pair_mask,
        "shifted_recon": shifted_recon,
        "shifted_image": shifted_image,
        "ctf2_over_nv": ctf2_over_nv,
        "n_rot": n_rot,
    }


def _call_rotation_sums(fn, case):
    return fn(
        jnp.asarray(case["pair_probs"]),
        jnp.asarray(case["local_rotation_row"]),
        jnp.asarray(case["translation_idx"]),
        jnp.asarray(case["pair_mask"]),
        jnp.asarray(case["shifted_recon"]),
        jnp.asarray(case["ctf2_over_nv"]),
        n_rotation_rows=case["n_rot"],
    )


def _call_image_sums(fn, case):
    return fn(
        jnp.asarray(case["pair_probs"]),
        jnp.asarray(case["local_rotation_row"]),
        jnp.asarray(case["translation_idx"]),
        jnp.asarray(case["pair_mask"]),
        jnp.asarray(case["shifted_image"]),
        n_rotation_rows=case["n_rot"],
    )


def _call_rotation_and_image_sums(fn, case):
    return fn(
        jnp.asarray(case["pair_probs"]),
        jnp.asarray(case["local_rotation_row"]),
        jnp.asarray(case["translation_idx"]),
        jnp.asarray(case["pair_mask"]),
        jnp.asarray(case["shifted_recon"]),
        jnp.asarray(case["shifted_image"]),
        jnp.asarray(case["ctf2_over_nv"]),
        n_rotation_rows=case["n_rot"],
    )


def _assert_tree_allclose(actual, expected, *, rtol, atol):
    assert len(actual) == len(expected)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        np.testing.assert_allclose(
            np.asarray(actual_value),
            np.asarray(expected_value),
            rtol=rtol,
            atol=atol,
        )


def _make_late_iter_sparse_kclass_inputs(*, n_classes=4, n_images=8, n_rot=512, n_fine_trans=116):
    valid_pairs_per_image = 192
    per_image_inputs_by_class = []
    for class_index in range(n_classes):
        masks = []
        for image_idx in range(n_images):
            mask = np.zeros((n_rot, n_fine_trans), dtype=bool)
            flat = (np.arange(valid_pairs_per_image, dtype=np.int64) * 37 + image_idx * 11 + class_index * 17) % (
                n_rot * n_fine_trans
            )
            mask[flat // n_fine_trans, flat % n_fine_trans] = True
            masks.append(mask)
        per_image_inputs_by_class.append(
            {
                "oversampled_rots": [
                    np.broadcast_to(np.eye(3, dtype=np.float32), (n_rot, 3, 3)).copy()
                    for _ in range(n_images)
                ],
                "oversampled_rot_indices": [
                    np.arange(n_rot, dtype=np.int64) + class_index * 10_000
                    for _ in range(n_images)
                ],
                "log_prior": [
                    np.linspace(-1.0, 1.0, n_rot, dtype=np.float32)
                    for _ in range(n_images)
                ],
                "candidate_mask": masks,
            }
        )
    return per_image_inputs_by_class


@pytest.mark.parametrize(
    "requested,noise,prob_dtype,recon_dtype,noise_dtype,expected",
    [
        (True, True, jnp.float32, jnp.complex64, jnp.complex64, True),
        (True, True, jnp.float64, jnp.complex64, jnp.complex64, False),
        (True, True, jnp.float32, jnp.complex128, jnp.complex64, False),
        (True, True, jnp.float32, jnp.complex64, jnp.complex128, False),
        (False, True, jnp.float32, jnp.complex64, jnp.complex64, False),
        (True, False, jnp.float32, jnp.complex64, jnp.complex64, False),
    ],
)
def test_native_dual_dispatch_checks_actual_operand_dtypes(
    requested, noise, prob_dtype, recon_dtype, noise_dtype, expected,
):
    assert _native_dual_weighted_sums_supported_for_operands(
        requested=requested,
        accumulate_noise=noise,
        probability_dtype=prob_dtype,
        reconstruction_dtype=recon_dtype,
        noise_dtype=noise_dtype,
    ) is expected


def test_sparse_pass2_projection_cap_chunks_projection_calls(monkeypatch):
    from relax.sparse_pass2 import (
        sparse_pass2_projection_blocks,
    )

    calls = []

    def fake_project(volume_block, rotations_block, image_shape, volume_shape, disc_type, **kwargs):
        del volume_block, image_shape, volume_shape, disc_type
        calls.append(int(rotations_block.shape[0]))
        n_half = 5
        rotation_ids = jnp.asarray(rotations_block[:, 0, 0], dtype=jnp.float32)
        proj = rotation_ids[:, None] + jnp.arange(n_half, dtype=jnp.float32)[None, :]
        proj = proj.astype(jnp.complex64)
        return_abs2 = kwargs.get("return_abs2", None)
        return proj, None if return_abs2 is False else jnp.abs(proj) ** 2

    monkeypatch.setattr(sparse_pass2_projection_blocks, "_compute_projections_block", fake_project)
    rotations = np.zeros((10, 3, 3), dtype=np.float32)
    rotations[:, 0, 0] = np.arange(10, dtype=np.float32)

    proj, abs2 = _compute_sparse_pass2_projections_block(
        jnp.zeros(VOLUME_SIZE, dtype=jnp.complex64),
        jnp.asarray(rotations),
        IMAGE_SHAPE,
        VOLUME_SHAPE,
        "linear_interp",
        max_projected_rotations=4,
        return_abs2=False,
    )

    assert calls == [4, 4, 2]
    assert abs2 is None
    assert_matches(np.asarray(proj[:, 0].real), np.arange(10, dtype=np.float32))


def test_sparse_pass2_windowed_projection_cap_keeps_only_requested_pixels(monkeypatch):
    from relax.sparse_pass2 import (
        sparse_pass2_projection_blocks,
    )

    calls = []

    def fake_project(volume_block, rotations_block, image_shape, volume_shape, disc_type, **kwargs):
        del volume_block, image_shape, volume_shape, disc_type
        calls.append(int(rotations_block.shape[0]))
        n_half = 6
        rotation_ids = jnp.asarray(rotations_block[:, 0, 0], dtype=jnp.float32)
        proj = rotation_ids[:, None] * 10.0 + jnp.arange(n_half, dtype=jnp.float32)[None, :]
        return proj.astype(jnp.complex64), None

    monkeypatch.setattr(sparse_pass2_projection_blocks, "_compute_projections_block", fake_project)
    rotations = np.zeros((7, 3, 3), dtype=np.float32)
    rotations[:, 0, 0] = np.arange(7, dtype=np.float32)

    score, recon, recon_abs2 = _compute_sparse_pass2_windowed_projections_block(
        jnp.zeros(VOLUME_SIZE, dtype=jnp.complex64),
        jnp.asarray(rotations),
        IMAGE_SHAPE,
        VOLUME_SHAPE,
        "linear_interp",
        score_indices=jnp.asarray([0, 2], dtype=jnp.int32),
        recon_indices=jnp.asarray([1, 5], dtype=jnp.int32),
        max_projected_rotations=3,
    )

    assert calls == [3, 3, 1]
    assert score.shape == (7, 2)
    assert recon.shape == (7, 2)
    assert recon_abs2.shape == (7, 2)
    assert_matches(np.asarray(score[4].real), np.asarray([40.0, 42.0], dtype=np.float32))
    assert_matches(np.asarray(recon[4].real), np.asarray([41.0, 45.0], dtype=np.float32))
    # The chunks are written into the outputs in place (bench 14572645): every
    # chunk's rows, the short last one included, land at their own offsets.
    assert recon_abs2.dtype == jnp.float32
    assert_matches(np.asarray(score[:, 0].real), 10.0 * np.arange(7, dtype=np.float32))
    assert_matches(np.asarray(recon_abs2[6]), np.asarray([61.0**2, 65.0**2], dtype=np.float32))
    score_only, no_recon, no_abs2 = _compute_sparse_pass2_windowed_projections_block(
        jnp.zeros(VOLUME_SIZE, dtype=jnp.complex64),
        jnp.asarray(rotations),
        IMAGE_SHAPE,
        VOLUME_SHAPE,
        "linear_interp",
        score_indices=jnp.asarray([0, 2], dtype=jnp.int32),
        max_projected_rotations=3,
    )
    assert no_recon is None and no_abs2 is None
    assert_matches(np.asarray(score_only), np.asarray(score))
    padded, padded_recon, padded_abs2 = _compute_sparse_pass2_windowed_projections_block(
        jnp.zeros(VOLUME_SIZE, dtype=jnp.complex64),
        jnp.asarray(rotations),
        IMAGE_SHAPE,
        VOLUME_SHAPE,
        "linear_interp",
        score_indices=jnp.asarray([0, 2], dtype=jnp.int32),
        recon_indices=jnp.asarray([1, 5], dtype=jnp.int32),
        max_projected_rotations=3,
        output_rows=10,
    )
    # output_rows: the outputs are that long, zero past the rotations (bench 14576794).
    for full, rows7 in ((padded, score), (padded_recon, recon), (padded_abs2, recon_abs2)):
        assert full.shape == (10, 2)
        assert_matches(np.asarray(full)[:7], np.asarray(rows7))
        assert not np.any(np.asarray(full)[7:])


def test_class_call_length_pads_to_an_eighth_of_the_power_of_two():
    from relax.sparse_pass2.sparse_pass2_projection_blocks import class_call_length

    assert [class_call_length(n) for n in (1, 256, 257, 2000, 2049, 8192, 8193, 70000)] == [
        256, 256, 512, 2048, 2560, 8192, 10240, 81920,
    ]


def test_class_rows_project_in_place_within_the_streamed_chunk_plan(monkeypatch):
    """K>1 streamed projections fill one set of row-capacity arrays, inside the chunk plan.

    A K15 VDAM chunk (62ab6ae, it151) ran out of memory gathering its classes' padded calls:
    every class with a few rows still projected a whole stream quantum, and the calls were
    concatenated and gathered next to the output, ``K * quantum`` rows past the plan's
    ``_STREAM_PEAK_COPIES * row_capacity``. Here 15 classes of 3 rows each pad to 256-rotation
    calls (class_call_length; 3840 rotations of calls) in a 64-row chunk: the live projection
    arrays stay within the plan's two row-capacity copies, and every row holds its own class's
    projection.
    """

    from relax.sparse_pass2 import resident_pass2 as rp
    from relax.sparse_pass2 import sparse_pass2_projection_blocks
    from relax.sparse_pass2.sparse_pass2_budget import _projection_cache_transient_bytes
    from relax.sparse_pass2.sparse_pass2_projection_blocks import project_rows_by_class

    n_classes, rows_per_class, row_capacity, chunk_rotations = 15, 3, 64, 8
    # Windows of realistic width next to the 36-byte rotation matrices a call uploads.
    n_half = 64
    score_indices = jnp.arange(0, 48, 2, dtype=jnp.int32)
    recon_indices = jnp.arange(24, 64, dtype=jnp.int32)
    baseline = {}
    peak = [0]

    def fake_project(volume_block, rotations_block, image_shape, volume_shape, disc_type, **kwargs):
        del image_shape, volume_shape, disc_type
        # Row value 10 * row id + pixel, plus 1000 * the class (the volume's value).
        proj = (
            jnp.asarray(rotations_block[:, 0, 0], dtype=jnp.float32)[:, None] * 10.0
            + jnp.arange(n_half, dtype=jnp.float32)[None, :]
            + 1000.0 * jnp.real(volume_block[0])
        ).astype(jnp.complex64)
        live = sum(int(x.nbytes) for x in jax.live_arrays()) - baseline["bytes"]
        peak[0] = max(peak[0], live)
        return proj, None

    monkeypatch.setattr(sparse_pass2_projection_blocks, "_compute_projections_block", fake_project)
    row_class = np.repeat(np.arange(n_classes), rows_per_class)
    n_valid = row_class.size
    matrices = np.zeros((n_valid, 3, 3), dtype=np.float32)
    matrices[:, 0, 0] = np.arange(n_valid, dtype=np.float32)
    volumes = [jnp.full(VOLUME_SIZE, float(k), dtype=jnp.complex64) for k in range(n_classes)]
    finalized = []

    def project(matrices, *, class_index, n_rows, outputs, output_row_ids, finalize):
        finalized.append(bool(finalize))
        return _compute_sparse_pass2_windowed_projections_block(
            volumes[class_index],
            jnp.asarray(matrices),
            IMAGE_SHAPE,
            VOLUME_SHAPE,
            "linear_interp",
            score_indices=score_indices,
            recon_indices=recon_indices,
            max_projected_rotations=chunk_rotations,
            output_rows=n_rows,
            outputs=outputs,
            output_row_ids=output_row_ids,
        )

    baseline["bytes"] = sum(int(x.nbytes) for x in jax.live_arrays())
    score, recon, recon_abs2 = project_rows_by_class(
        project,
        matrices,
        row_class,
        n_rows=row_capacity,
    )

    assert finalized == [False] * (n_classes - 1) + [True]
    assert score.shape == (row_capacity, score_indices.size) and recon.shape == (row_capacity, recon_indices.size)
    expected = 10.0 * np.arange(n_valid)[:, None] + 1000.0 * row_class[:, None]
    assert_matches(np.asarray(score.real)[:n_valid], expected + np.asarray(score_indices))
    assert_matches(np.asarray(recon.real)[:n_valid], expected + np.asarray(recon_indices))
    assert_matches(np.asarray(recon_abs2)[:n_valid], (expected + np.asarray(recon_indices)) ** 2)
    for values in (score, recon, recon_abs2):
        assert not np.any(np.asarray(values)[n_valid:])
    bytes_per_rotation = _projection_cache_transient_bytes(
        1, int(score_indices.size), projection_complex_dtype=np.complex64, include_abs2=False
    ) + _projection_cache_transient_bytes(
        1, int(recon_indices.size), projection_complex_dtype=np.complex64, include_abs2=True
    )
    plan_bytes = rp._STREAM_PEAK_COPIES * row_capacity * bytes_per_rotation
    # The live arrays at a projector call: the outputs, the call's padded rotations, the
    # previous chunk being dropped. The concatenate-and-gather held every class's whole call.
    assert peak[0] <= plan_bytes, (peak[0], plan_bytes)


def test_sparse_pass2_windowed_projection_uses_relion_projector_branch(monkeypatch):
    from relax.sparse_pass2 import (
        sparse_pass2_projection_blocks,
    )

    calls = []
    relion_projector_half = jnp.ones((4, 4, 3), dtype=jnp.complex64)

    def fake_relion_projector(
        volume_relion_half,
        rotations_block,
        image_shape,
        *,
        r_max,
        padding_factor,
        return_abs2,
        centered_rows,
        dense_scale,
        relion_texture_interp,
        projector_output_size=None,
        mask_current_image_disk=True,
        relion_kernel=None,
        pixel_indices=None,
    ):
        del projector_output_size
        # Sparse pass 2 scores with RELION's fine diff2 kernel.
        assert relion_kernel == "fine"
        assert pixel_indices is None
        calls.append(
            {
                "n_rot": int(rotations_block.shape[0]),
                "image_shape": tuple(image_shape),
                "r_max": int(r_max),
                "padding_factor": int(padding_factor),
                "return_abs2": bool(return_abs2),
                "centered_rows": bool(centered_rows),
                "dense_scale": bool(dense_scale),
                "relion_texture_interp": relion_texture_interp,
                "mask_current_image_disk": bool(mask_current_image_disk),
                "projector_shape": tuple(volume_relion_half.shape),
            }
        )
        n_half = 6
        rotation_ids = jnp.asarray(rotations_block[:, 0, 0], dtype=jnp.float32)
        proj = rotation_ids[:, None] * 10.0 + jnp.arange(n_half, dtype=jnp.float32)[None, :]
        proj = proj.astype(jnp.complex64)
        return proj, None if not return_abs2 else jnp.abs(proj) ** 2

    monkeypatch.setattr(sparse_pass2_projection_blocks, "_compute_relion_projector_projections_block", fake_relion_projector)
    rotations = np.zeros((7, 3, 3), dtype=np.float32)
    rotations[:, 0, 0] = np.arange(7, dtype=np.float32)

    score, recon, recon_abs2 = _compute_sparse_pass2_windowed_projections_block(
        jnp.zeros(VOLUME_SIZE, dtype=jnp.complex64),
        jnp.asarray(rotations),
        IMAGE_SHAPE,
        VOLUME_SHAPE,
        "linear_interp",
        score_indices=jnp.asarray([0, 2], dtype=jnp.int32),
        recon_indices=jnp.asarray([1, 5], dtype=jnp.int32),
        max_projected_rotations=3,
        relion_projector_half=relion_projector_half,
        relion_projector_r_max=3,
        projection_padding_factor=2,
        relion_texture_interp=False,
        mask_current_image_disk=False,
    )

    assert [call["n_rot"] for call in calls] == [3, 3, 1]
    for call in calls:
        assert call == {
            "n_rot": call["n_rot"],
            "image_shape": IMAGE_SHAPE,
            "r_max": 3,
            "padding_factor": 2,
            "return_abs2": False,
            "centered_rows": True,
            "dense_scale": True,
            "relion_texture_interp": False,
            "mask_current_image_disk": False,
            "projector_shape": (4, 4, 3),
        }
    assert score.shape == (7, 2)
    assert recon.shape == (7, 2)
    assert recon_abs2.shape == (7, 2)
    assert_matches(np.asarray(score[4].real), np.asarray([40.0, 42.0], dtype=np.float32))
    assert_matches(np.asarray(recon[4].real), np.asarray([41.0, 45.0], dtype=np.float32))


def test_relion_score_window_keeps_particle_crop_separate_from_model_radius():
    from relax.sparse_pass2.sparse_pass2_projection_blocks import _projection_kwargs_for_relion_score_window

    kwargs = _projection_kwargs_for_relion_score_window(
        {"max_r": 28.0, "return_abs2": False},
        use_relion_projector=True,
        current_size=58,
    )

    assert kwargs == {
        "max_r": 28.0,
        "return_abs2": False,
        "projector_output_size": 58,
    }


def test_sparse_pass2_windowed_projection_cap_casts_chunks_before_concat(monkeypatch):
    from relax.sparse_pass2 import (
        sparse_pass2_projection_blocks,
    )

    def fake_project(volume_block, rotations_block, image_shape, volume_shape, disc_type, **kwargs):
        del volume_block, image_shape, volume_shape, disc_type, kwargs
        n_half = 6
        rotation_ids = jnp.asarray(rotations_block[:, 0, 0], dtype=jnp.float64)
        proj = rotation_ids[:, None] * 10.0 + jnp.arange(n_half, dtype=jnp.float64)[None, :]
        return proj.astype(jnp.complex128), None

    monkeypatch.setattr(sparse_pass2_projection_blocks, "_compute_projections_block", fake_project)
    rotations = np.zeros((7, 3, 3), dtype=np.float32)
    rotations[:, 0, 0] = np.arange(7, dtype=np.float32)

    score, recon, recon_abs2 = _compute_sparse_pass2_windowed_projections_block(
        jnp.zeros(VOLUME_SIZE, dtype=jnp.complex128),
        jnp.asarray(rotations),
        IMAGE_SHAPE,
        VOLUME_SHAPE,
        "linear_interp",
        score_indices=jnp.asarray([0, 2], dtype=jnp.int32),
        recon_indices=jnp.asarray([1, 5], dtype=jnp.int32),
        max_projected_rotations=3,
        output_complex_dtype=jnp.complex64,
        output_abs2_dtype=jnp.float32,
    )

    assert score.dtype == jnp.complex64
    assert recon.dtype == jnp.complex64
    assert recon_abs2.dtype == jnp.float32
    assert_matches(np.asarray(recon[4].real), np.asarray([41.0, 45.0], dtype=np.float32))


def test_prepare_bucket_io_routes_direct_score_translation_through_relion_cuda(
    monkeypatch,
):
    from relax.cuda import kernels as em_cuda_kernels

    ds = MockDataset(n_images=3, seed=20260727)
    batch_indices = np.asarray([0, 2], dtype=np.int64)
    batch = jnp.asarray(ds._images[batch_indices])
    config = ForwardModelConfig.from_dataset(
        ds,
        disc_type="linear_interp",
        process_fn=ds.process_images,
    )
    n_half = IMAGE_SHAPE[0] * (IMAGE_SHAPE[1] // 2 + 1)
    window_spec = make_fourier_window_spec(
        IMAGE_SHAPE,
        current_size=6,
        n_half=n_half,
        include_recon_window=True,
    )
    fine_translations = np.asarray(
        [[-0.75, 0.25], [0.25, -0.75]],
        dtype=np.float32,
    )
    translation_angles = jnp.asarray(
        _relion_translation_angles_f32(fine_translations, IMAGE_SHAPE),
        dtype=jnp.float32,
    )
    calls = []

    def fake_translate(images, angles, pixel_indices, image_shape):
        calls.append(
            {
                "images": np.asarray(images),
                "angles": np.asarray(angles),
                "pixel_indices": np.asarray(pixel_indices),
                "image_shape": image_shape,
            }
        )
        return jnp.full(
            (images.shape[0] * angles.shape[0], images.shape[1]),
            jnp.complex64(7.0 + 3.0j),
        )

    monkeypatch.setattr(
        em_cuda_kernels,
        "relion_translate_score_f32",
        fake_translate,
    )
    result = _prepare_bucket_io(
        experiment_dataset=ds,
        batch=batch,
        ctf_params=jnp.asarray(ds.CTF_params[batch_indices]),
        image_indices=batch_indices,
        noise_variance_half=jnp.ones(n_half, dtype=jnp.float32),
        fine_translations=fine_translations,
        config=config,
        n_trans=fine_translations.shape[0],
        score_with_masked_images=True,
        half_spectrum_scoring=True,
        image_corrections=np.ones(ds.n_units, dtype=np.float32),
        scale_corrections=np.ones(ds.n_units, dtype=np.float32),
        image_pre_shifts=None,
        use_float64_scoring=False,
        return_direct_scoring_io=True,
        score_mode="gaussian",
        window_indices=window_spec.score_indices,
        recon_window_indices=window_spec.recon_indices,
        relion_score_translation_angles=translation_angles,
        return_windowed_shifted=True,
    )

    direct_score_calls = [
        call
        for call in calls
        if np.array_equal(
            call["pixel_indices"],
            np.asarray(window_spec.score_indices, dtype=np.int32),
        )
    ]
    assert direct_score_calls
    assert_matches(
        direct_score_calls[0]["pixel_indices"],
        np.asarray(window_spec.score_indices, dtype=np.int32),
    )
    assert_matches(
        direct_score_calls[0]["angles"], np.asarray(translation_angles)
    )
    assert direct_score_calls[0]["image_shape"] == IMAGE_SHAPE
    assert_matches(
        np.asarray(result[7]),
        np.full(
            (
                batch_indices.size * fine_translations.shape[0],
                window_spec.score_indices.shape[0],
            ),
            np.complex64(7.0 + 3.0j),
        ),
    )


def test_prepare_bucket_io_exact_cc_keeps_relion_image_and_corr_operands_separate():
    ds = MockDataset(n_images=2, seed=20260809)
    batch_indices = np.asarray([0, 1], dtype=np.int64)
    batch = jnp.asarray(ds._images[batch_indices])
    config = ForwardModelConfig.from_dataset(
        ds,
        disc_type="linear_interp",
        process_fn=ds.process_images,
    )
    n_half = IMAGE_SHAPE[0] * (IMAGE_SHAPE[1] // 2 + 1)
    common = dict(
        experiment_dataset=ds,
        batch=batch,
        ctf_params=jnp.asarray(ds.CTF_params[batch_indices]),
        image_indices=batch_indices,
        noise_variance_half=jnp.ones(n_half, dtype=jnp.float32),
        fine_translations=jnp.zeros((1, 2), dtype=jnp.float32),
        config=config,
        n_trans=1,
        score_with_masked_images=False,
        half_spectrum_scoring=True,
        image_corrections=None,
        scale_corrections=None,
        image_pre_shifts=None,
        use_float64_scoring=False,
        return_direct_scoring_io=True,
        score_mode="normalized_cc",
    )

    folded = _prepare_bucket_io(**common)
    exact = _prepare_bucket_io(
        **common,
        relion_exact_normalized_cc_operands=True,
    )
    folded_image = np.asarray(folded[7]).reshape(batch_indices.size, 1, n_half)
    corrected_image = np.asarray(exact[7]).reshape(batch_indices.size, 1, n_half)
    corr_image = np.asarray(exact[3])[:, None, :]

    assert np.asarray(exact[2]).dtype == np.float64
    assert not np.array_equal(corrected_image, folded_image)
    np.testing.assert_allclose(
        corrected_image * corr_image,
        folded_image,
        rtol=2e-5,
        atol=2e-6,
    )


def test_prepare_bucket_io_exact_bpref_translation_keeps_recovar_fft_units_and_native_operands(monkeypatch):
    """Final-Q 5b4e1e7311 exact-BPref operands, in this tree's factoring.

    Final Q scaled the image by 1/N^2 before the (linear) translate kernel and
    by N^2 after it. This tree translates the image in RECOVAR's unnormalized
    FFT units and applies RELION's native units (fftw.cpp forward 1/N^2,
    minvsigma2 without the N^4 variance scale) only in the native operands the
    fused BPref consumes. The translated reconstruction operand, the observable
    both trees accumulate, must be identical to Q's; the native triple carries
    Q's normalized-image assertion.
    """
    from relax.cuda import kernels as em_cuda_kernels
    from relax.relion import relion_ctf

    ds = MockDataset(n_images=1, seed=1701)

    class _Backend:
        image_mask_mode = "relion_background_fill"
        relion_fourier_backend = "relion_cuda"

    ds.image_source.backend = _Backend()

    def process_half(batch, apply_image_mask=False, **kwargs):
        del apply_image_mask
        processed = _raw_real_process_half(batch)
        factors = jnp.asarray(kwargs["relion_normalization_factors"], dtype=processed.real.dtype)
        return processed * factors[:, None]

    ds.process_images_half = process_half
    batch_indices = np.asarray([0], dtype=np.int64)
    batch = jnp.asarray(ds._images[batch_indices])
    config = ForwardModelConfig.from_dataset(ds, disc_type="linear_interp", process_fn=ds.process_images)
    n_half = IMAGE_SHAPE[0] * (IMAGE_SHAPE[1] // 2 + 1)
    ctf_half = np.linspace(0.5, 1.5, n_half, dtype=np.float64)[None, :]
    monkeypatch.setattr(
        relion_ctf,
        "_relion_exact_ctf_half_from_source_star",
        lambda *args, **kwargs: ctf_half,
    )
    calls = []

    def fake_translate(images, weights, angles, pixel_indices, image_shape):
        calls.append((np.asarray(images), np.asarray(weights)))
        del pixel_indices, image_shape
        return jnp.repeat(images * weights, int(angles.shape[0]), axis=0)

    monkeypatch.setattr(em_cuda_kernels, "relion_translate_bpref_f32", fake_translate)

    def fake_translate_score(images, angles, _pixel_indices, _image_shape):
        return jnp.repeat(jnp.asarray(images), int(angles.shape[0]), axis=0)

    monkeypatch.setattr(em_cuda_kernels, "relion_translate_score_f32", fake_translate_score)
    noise = np.linspace(0.75, 1.25, n_half, dtype=np.float64)
    result = _prepare_bucket_io(
        experiment_dataset=ds,
        batch=batch,
        ctf_params=jnp.asarray(ds.CTF_params[batch_indices]),
        image_indices=batch_indices,
        noise_variance_half=jnp.asarray(noise, dtype=jnp.float64),
        fine_translations=jnp.zeros((1, 2), dtype=jnp.float32),
        config=config,
        n_trans=1,
        score_with_masked_images=False,
        half_spectrum_scoring=False,
        image_corrections=np.ones(1, dtype=np.float32),
        scale_corrections=np.ones(1, dtype=np.float32),
        image_pre_shifts=None,
        use_float64_scoring=False,
        relion_score_translation_angles=jnp.zeros((1, 2), dtype=jnp.float32),
        relion_exact_bpref_operands=True,
        return_native_bpref_operands=True,
    )

    raw = np.asarray(_raw_real_process_half(batch))
    fft_size = float(np.prod(IMAGE_SHAPE))
    inverse_noise = np.reciprocal(noise).astype(np.float32)
    expected_weight = ctf_half.astype(np.float32) * inverse_noise[None, :]
    assert len(calls) == 1
    assert_matches(calls[0][0], raw)
    assert_matches(calls[0][1], expected_weight)
    assert_matches(np.asarray(result[1]), raw * expected_weight)
    native_image, native_ctf, native_minvsigma2 = result[-3:]
    assert_matches(np.asarray(native_image), raw * np.float32(1.0 / fft_size))
    assert_matches(np.asarray(native_ctf), ctf_half.astype(np.float32))
    assert_matches(
        np.asarray(native_minvsigma2),
        np.reciprocal(noise / (fft_size * fft_size)).astype(np.float32),
    )


def test_prepare_bucket_io_routes_relion_cuda_operands_to_score_and_reconstruction():
    ds = MockDataset(n_images=2, seed=714)

    class _Backend:
        image_mask_mode = "relion_background_fill"
        relion_fourier_backend = "relion_cuda"

    ds.image_source.backend = _Backend()
    calls = []

    def capture_process(batch, apply_image_mask=False, **kwargs):
        calls.append((np.asarray(batch), apply_image_mask, kwargs))
        processed = _raw_real_process_half(batch, apply_image_mask=apply_image_mask)
        factors = jnp.asarray(kwargs["relion_normalization_factors"], dtype=processed.real.dtype)
        return processed * factors[:, None]

    ds.process_images_half = capture_process
    batch_indices = np.asarray([0, 1], dtype=np.int64)
    batch = jnp.asarray(ds._images[batch_indices])
    config = ForwardModelConfig.from_dataset(ds, disc_type="linear_interp", process_fn=ds.process_images)
    n_half = IMAGE_SHAPE[0] * (IMAGE_SHAPE[1] // 2 + 1)
    image_corrections = np.asarray([0.8, 1.5], dtype=np.float32)
    scale_corrections = np.asarray([2.0, 0.5], dtype=np.float32)
    image_pre_shifts = np.asarray([[1.0, -1.0], [-2.0, 1.0]], dtype=np.float32)

    result = _prepare_bucket_io(
        ds,
        batch,
        jnp.asarray(ds.CTF_params[batch_indices]),
        batch_indices,
        jnp.ones(n_half, dtype=jnp.float32),
        jnp.zeros((1, 2), dtype=jnp.float32),
        config,
        1,
        True,
        False,
        image_corrections,
        scale_corrections,
        image_pre_shifts,
        False,
    )

    assert [call[1] for call in calls] == [True, False]
    for raw_batch, _, kwargs in calls:
        assert_matches(raw_batch, ds._images)
        assert_matches(
            kwargs["relion_normalization_factors"],
            image_corrections / scale_corrections,
        )
        assert_matches(kwargs["relion_integer_shifts"], image_pre_shifts.astype(np.int32))

    expected_recon = _raw_real_process_half(batch) * jnp.asarray(image_corrections)[:, None]
    np.testing.assert_allclose(np.asarray(result[1]), np.asarray(expected_recon), rtol=1e-6, atol=1e-6)


def test_prepare_bucket_io_windowed_reuses_unmasked_recon_shift_for_noise(monkeypatch):
    """Unmasked windowed prepare can reuse the recon-window shifted image."""

    from relax.sparse_pass2 import (
        sparse_pass2_bucket_io,
    )

    ds = MockDataset(n_images=4, seed=613)
    batch_indices = np.asarray([0, 1, 2], dtype=np.int64)
    batch = jnp.asarray(ds._images[batch_indices])
    ctf_params = jnp.asarray(ds.CTF_params[batch_indices])
    config = ForwardModelConfig.from_dataset(
        ds,
        disc_type="linear_interp",
        process_fn=ds.process_images,
    )
    n_half = IMAGE_SHAPE[0] * (IMAGE_SHAPE[1] // 2 + 1)
    window_spec = make_fourier_window_spec(
        IMAGE_SHAPE,
        current_size=6,
        n_half=n_half,
        include_recon_window=True,
    )
    assert window_spec.n_score != window_spec.n_recon
    fine_translations = jnp.asarray(
        [
            [0.0, 0.0],
            [0.25, -0.5],
            [1.0, 0.75],
        ],
        dtype=jnp.float32,
    )
    common_kwargs = dict(
        experiment_dataset=ds,
        batch=batch,
        ctf_params=ctf_params,
        image_indices=batch_indices,
        noise_variance_half=jnp.linspace(0.8, 1.4, n_half, dtype=jnp.float32),
        fine_translations=fine_translations,
        config=config,
        n_trans=int(fine_translations.shape[0]),
        half_spectrum_scoring=True,
        image_corrections=np.asarray([1.0, 0.93, 1.11, 1.07], dtype=np.float32),
        scale_corrections=np.asarray([1.0, 1.08, 0.91, 1.03], dtype=np.float32),
        image_pre_shifts=None,
        use_float64_scoring=False,
        return_direct_scoring_io=True,
        score_only=False,
        score_mode="gaussian",
        window_indices=window_spec.score_indices,
        recon_window_indices=window_spec.recon_indices,
        return_windowed_shifted=True,
    )
    original_apply = sparse_pass2_bucket_io.apply_half_translation_phases
    call_shapes = []

    def counting_apply(images, phases):
        call_shapes.append((tuple(images.shape), tuple(phases.shape)))
        return original_apply(images, phases)

    monkeypatch.setattr(sparse_pass2_bucket_io, "apply_half_translation_phases", counting_apply)
    unmasked = _prepare_bucket_io(**common_kwargs, score_with_masked_images=False)
    assert len(call_shapes) == 3
    assert_matches(np.asarray(unmasked[5]), np.asarray(unmasked[1]))

    call_shapes.clear()
    _prepare_bucket_io(**common_kwargs, score_with_masked_images=True)
    assert len(call_shapes) == 4


def test_sparse_pass2_device_memory_probe_honors_visible_device():
    smi_output = "\n".join(
        [
            "0, GPU-a100, 40960",
            "1, GPU-h100, 81559",
        ],
    )

    assert _nvidia_smi_visible_device_memory_bytes(smi_output, "1") == 81559 * 1024**2
    assert _nvidia_smi_visible_device_memory_bytes(smi_output, "GPU-h100") == 81559 * 1024**2
    assert _nvidia_smi_visible_device_memory_bytes(smi_output, "h100") == 81559 * 1024**2
    assert _nvidia_smi_visible_device_memory_bytes(smi_output, "-1") is None
    # CUDA_VISIBLE_DEVICES='' hides every device (CPU runs); it is not "unset".
    assert _nvidia_smi_visible_device_memory_bytes(smi_output, "") is None
    assert _nvidia_smi_visible_device_memory_bytes(smi_output, None) == 40960 * 1024**2


def test_half_translation_phase_table_matches_generic_off_relion_nyquist_row_and_column():
    rng = np.random.default_rng(13)
    image_shape = (16, 16)
    n_half = image_shape[0] * (image_shape[1] // 2 + 1)
    weighted_half = jnp.asarray(
        rng.normal(size=(3, n_half)).astype(np.float32)
        + 1j * rng.normal(size=(3, n_half)).astype(np.float32),
        dtype=jnp.complex64,
    )
    translations = jnp.asarray(rng.normal(size=(5, 2)).astype(np.float32))

    tiled_images = jnp.repeat(weighted_half[:, None, :], translations.shape[0], axis=1).reshape(
        weighted_half.shape[0] * translations.shape[0],
        -1,
    )
    tiled_translations = jnp.repeat(translations[None], weighted_half.shape[0], axis=0).reshape(
        weighted_half.shape[0] * translations.shape[0],
        -1,
    )
    generic = core.translate_images(tiled_images, tiled_translations, image_shape, half_image=True)
    phase_table = apply_half_translation_phases(
        weighted_half,
        half_translation_phase_table(translations, image_shape),
    )

    core_lattice = np.asarray(
        ftu.get_k_coordinate_of_each_pixel_half(image_shape, voxel_size=1, scaled=True)
    )
    ky = np.rint(core_lattice[:, 1] * image_shape[0]).astype(np.int64)
    kx = np.rint(core_lattice[:, 0] * image_shape[0]).astype(np.int64)
    # The generic translate keeps the core helper's -N/2 labels; RELION's are +N/2 on both.
    non_nyquist = (ky != -(image_shape[0] // 2)) & (kx != -(image_shape[0] // 2))
    assert_matches(
        np.asarray(phase_table)[:, non_nyquist], np.asarray(generic)[:, non_nyquist]
    )


def test_indexed_half_translation_phase_table_matches_full_slice():
    translations = jnp.asarray(
        [
            [0.0, 0.0],
            [0.25, -0.5],
            [1.0, 0.75],
        ],
        dtype=jnp.float32,
    )
    pixel_indices = jnp.asarray([0, 2, 5, 7, 11], dtype=jnp.int32)
    full = half_translation_phase_table(translations, IMAGE_SHAPE)
    indexed = _half_translation_phase_table_for_indices(translations, IMAGE_SHAPE, pixel_indices)
    np.testing.assert_allclose(
        np.asarray(indexed),
        np.asarray(full)[:, np.asarray(pixel_indices)],
        rtol=1e-6,
        atol=1e-6,
    )


def test_em_translation_phase_dot_products_request_highest_precision():
    """Prevent A100 TF32 rounding from changing RELION translation phases."""

    translations = jnp.asarray(
        [
            [-2.0479863, 1.9520137],
            [1.9520137, -1.0479863],
        ],
        dtype=jnp.float32,
    )
    pixel_indices = jnp.asarray([0, 2, 5, 7, 11], dtype=jnp.int32)
    n_half = IMAGE_SHAPE[0] * (IMAGE_SHAPE[1] // 2 + 1)
    images = jnp.ones((translations.shape[0], n_half), dtype=jnp.complex64)

    phase_jaxprs = (
        jax.make_jaxpr(lambda value: half_translation_phase_table(value, IMAGE_SHAPE))(translations),
        jax.make_jaxpr(
            lambda value: _half_translation_phase_table_for_indices(
                value,
                IMAGE_SHAPE,
                pixel_indices,
            )
        )(translations),
        jax.make_jaxpr(
            lambda image, value: core.translate_images(
                image,
                value,
                IMAGE_SHAPE,
                half_image=True,
            )
        )(images, translations),
    )

    for phase_jaxpr in phase_jaxprs:
        jaxpr_text = str(phase_jaxpr)
        assert "dot_general[" in jaxpr_text
        assert "precision=(Precision.HIGHEST, Precision.HIGHEST)" in jaxpr_text


def test_fine_rotation_override_preserves_fine_grid_order_and_parent_map():
    fine_rotations = np.arange(6 * 9, dtype=np.float32).reshape(6, 3, 3)
    fine_parent = np.array([0, 1, 0, 2, 1, 2], dtype=np.int64)

    per_image = _prepare_per_image_pass2_inputs(
        [np.array([0, 2], dtype=np.int32)],
        n_coarse_rot=3,
        n_coarse_trans=1,
        nside_level=1,
        oversampling_order=1,
        n_fine_trans=1,
        fine_translation_parent=np.array([0], dtype=np.int32),
        rotation_log_prior=None,
        random_perturbation=0.0,
        fine_rotations_override=fine_rotations,
        fine_rotation_parent_override=fine_parent,
    )

    assert_matches(per_image["oversampled_rot_indices"][0], np.array([0, 2, 3, 5]))
    assert_matches(per_image["parent_map"][0], np.array([0, 0, 1, 1], dtype=np.int32))
    assert_matches(per_image["oversampled_rots"][0], fine_rotations[[0, 2, 3, 5]])


def test_fine_rotation_override_can_follow_relion_parent_execution_order():
    fine_rotations = np.arange(6 * 9, dtype=np.float32).reshape(6, 3, 3)
    # Order-1 RECOVAR parent ids are psi-slow/direction-fast. Parent 48 is
    # direction 0, psi 1, so RELION executes it before parent 1.
    fine_parent = np.array([0, 0, 1, 1, 48, 48], dtype=np.int64)

    per_image = _prepare_per_image_pass2_inputs(
        [np.array([0, 1, 48], dtype=np.int32)],
        n_coarse_rot=576,
        n_coarse_trans=1,
        nside_level=1,
        oversampling_order=1,
        n_fine_trans=1,
        fine_translation_parent=np.array([0], dtype=np.int32),
        rotation_log_prior=None,
        random_perturbation=0.0,
        fine_rotations_override=fine_rotations,
        fine_rotation_parent_override=fine_parent,
        relion_parent_execution_order=True,
    )

    expected_indices = np.array([0, 1, 4, 5, 2, 3], dtype=np.int64)
    assert_matches(per_image["oversampled_rot_indices"][0], expected_indices)
    assert_matches(
        per_image["parent_map"][0],
        np.array([0, 0, 2, 2, 1, 1], dtype=np.int32),
    )
    assert_matches(per_image["oversampled_rots"][0], fine_rotations[expected_indices])


def test_kclass_fine_mask_complement_matches_explicit_dense_support():
    n_rot_coarse = 3
    n_trans_coarse = 2
    n_rot_fine = 5
    n_trans_fine = 3
    rot_parent_map = np.asarray([0, 0, 1, 2, 2], dtype=np.int64)
    trans_parent_map = np.asarray([0, 1, 1], dtype=np.int64)
    excluded = np.asarray([1, 4], dtype=np.int32)
    full_mask = np.ones(n_rot_coarse * n_trans_coarse, dtype=bool)
    full_mask[excluded] = False
    explicit = np.flatnonzero(full_mask).astype(np.int32)
    complement = ComplementSignificantSampleIndices(
        excluded_indices=excluded,
        total_size=int(full_mask.size),
    )

    explicit_mask = _build_fine_grid_significance_mask(
        [explicit],
        n_rot_coarse,
        n_trans_coarse,
        n_rot_fine,
        n_trans_fine,
        rot_oversampling_factor=1,
        trans_oversampling_factor=1,
        rot_parent_map=rot_parent_map,
        trans_parent_map=trans_parent_map,
        n_images=1,
    )
    complement_mask = _build_fine_grid_significance_mask(
        [complement],
        n_rot_coarse,
        n_trans_coarse,
        n_rot_fine,
        n_trans_fine,
        rot_oversampling_factor=1,
        trans_oversampling_factor=1,
        rot_parent_map=rot_parent_map,
        trans_parent_map=trans_parent_map,
        n_images=1,
    )
    assert_matches(complement_mask, explicit_mask)

    explicit_stats = _fine_support_stats(
        [[explicit]],
        n_rot_coarse=n_rot_coarse,
        n_trans_coarse=n_trans_coarse,
        rot_parent_map=rot_parent_map,
        trans_parent_map=trans_parent_map,
        n_rot_fine=n_rot_fine,
        n_trans_fine=n_trans_fine,
    )
    complement_stats = _fine_support_stats(
        [[complement]],
        n_rot_coarse=n_rot_coarse,
        n_trans_coarse=n_trans_coarse,
        rot_parent_map=rot_parent_map,
        trans_parent_map=trans_parent_map,
        n_rot_fine=n_rot_fine,
        n_trans_fine=n_trans_fine,
    )
    assert complement_stats == explicit_stats


