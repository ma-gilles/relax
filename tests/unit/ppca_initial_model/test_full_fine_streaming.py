"""Streamed full rotation rows versus the independent local layout and host-mask engine."""

from pathlib import Path

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


def make_tile_problem(device=None):
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
        n_coarse_rotations=3, n_coarse_translations=3, device=device, **common,
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


@pytest.fixture(scope="module")
def tile_problem():
    return make_tile_problem()


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
    # Scalar summaries are float32 reductions returned as Python floats.
    assert_matches(np.float32(actual.log_likelihood), np.float32(expected.log_likelihood))
    for key in ("rotation_mass", "offset_second_sum_px2", "latent_covariance_trace_mean",
                "pose_entropy_mean", "pmax_mean", "max_posterior_per_image"):
        assert_matches(np.float32(actual.diagnostics[key]), np.float32(expected.diagnostics[key]))
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


def _moment_image_problem(dtype):
    from recovar.ppca.triangular import pack_upper_tri

    rng = np.random.default_rng(5)
    B, T, R, P, F = 3, 4, 5, 3, 7
    cdtype = np.complex128 if dtype == np.float64 else np.complex64
    score = rng.standard_normal((B, T, R)).astype(dtype)
    logZ = np.log(np.sum(np.exp(score), axis=(1, 2))).astype(dtype)
    z = rng.standard_normal((B, T, R, P - 1))
    alpha = np.concatenate([np.ones((B, T, R, 1)), z], axis=-1)
    cov = rng.standard_normal((B, T, R, P, P)) * 0.1
    G = alpha[..., :, None] * alpha[..., None, :] + cov @ np.swapaxes(cov, -1, -2)
    weights = rng.choice([1.0, 2.0], F)
    Y1_recon = (rng.standard_normal((B, T, F)) + 1j * rng.standard_normal((B, T, F))).astype(cdtype)
    ctf2_recon = rng.uniform(0.1, 1.0, (B, F)).astype(dtype)
    projections = (rng.standard_normal((R, P, F)) + 1j * rng.standard_normal((R, P, F))).astype(cdtype)
    return dict(score=score, logZ=logZ, alpha=alpha.astype(dtype), G_tri=np.asarray(pack_upper_tri(G)).astype(dtype),
                weights=weights.astype(dtype), Y1_recon=Y1_recon, ctf2_recon=ctf2_recon, projections=projections)


def _compare_moment_image_residuals(dtype, rtol=None):
    from relax.ppca_refinement.engine import pose_moment_images
    from relax.ppca_refinement.residual_statistics import (
        residual_image_statistics,
        residual_statistics_from_moment_images,
    )

    d = {k: jnp.asarray(v) for k, v in _moment_image_problem(dtype).items()}
    gamma = jnp.exp(d["score"] - d["logZ"][:, None, None])
    w = d["weights"]
    direct, correction, _ = full_float32(residual_image_statistics)(
        d["score"], d["alpha"], d["G_tri"], d["logZ"], d["Y1_recon"] * w, d["ctf2_recon"] * w, d["projections"]
    )
    rhs, lhs = full_float32(pose_moment_images)(
        gamma, d["alpha"], d["G_tri"], d["Y1_recon"], d["ctf2_recon"],
        rhs_dtype=d["Y1_recon"].dtype, lhs_dtype=d["ctf2_recon"].dtype,
    )
    residual, correction_over_w = full_float32(residual_statistics_from_moment_images)(rhs, lhs, d["projections"])
    assert np.asarray(correction_over_w).dtype == dtype
    assert_matches(np.asarray(residual), np.asarray(direct / w), rtol=rtol)
    assert_matches(np.asarray(correction_over_w), np.asarray(correction / w), rtol=rtol)


def test_moment_image_residuals_match_direct_residual_statistics():
    """Float32: the linear rewrite of the residual through the M-step images."""
    _compare_moment_image_residuals(np.float32)


def test_moment_image_residuals_match_direct_residual_statistics_f64():
    """Float64 companion of the same rewrite."""
    import jax

    with jax.enable_x64(True):
        _compare_moment_image_residuals(np.float64)


# Coarse rotation 2 is unsupported by every image; images 0 and 1 keep disjoint rotations.
PRUNED = [np.asarray([0, 1], np.int32), np.asarray([3, 5], np.int32), np.asarray([1, 4], np.int32)]


def test_device_resident_union_rows_match_per_image_host_layout(tile_problem):
    dataset, mu, W, stream, host = tile_problem
    reference = build_pass2_hypothesis_layout(PRUNED, **LAYOUT_KWARGS)
    actual = accumulate_full_row_tile(stream, np.arange(3), PRUNED, factor_once=False)
    assert actual.diagnostics["supported_image_rows"] == reference.total_local_rotations == 32
    assert actual.diagnostics["scored_image_rows"] == 3 * 20  # union of 16 rows in four 5-row blocks
    parts = []
    for image in range(3):
        begin, end = map(int, reference.rotation_offsets[image:image + 2])
        options = {key: host[key] for key in ("translations", "translation_log_prior", "noise_variance",
                                              "geometry", "schedule", "scoring")}
        parts.append(full_float32(accumulate_dense_ppca_statistics)(
            dataset, mu, W, rotations=reference.rotations_flat[begin:end],
            rotation_translation_mask=reference.sample_mask_rows(begin, end),
            rotation_log_prior=reference.rotation_log_priors_flat[begin:end], image_indices=np.asarray([image]),
            sparse_pass2=SparsePass2Config(enabled=False), collect_residuals=True, **options,
        ))
        # Fine rows of the shared grid: children stay contiguous per coarse parent.
        rows = reference.rotation_posterior_ids_flat[begin:end] * 8 + np.arange(end - begin) % 8
        parts[-1].diagnostics["global_rows"] = rows
    for name in ("rhs", "lhs_tri", "residual_gradient", "residual_num", "residual_den"):
        assert_matches(np.asarray(getattr(actual, name)), sum(np.asarray(getattr(p, name)) for p in parts))
    assert_matches(np.asarray(actual.embeddings), np.concatenate([np.asarray(p.embeddings) for p in parts]))
    assert_matches(np.float32(actual.log_likelihood), np.float32(sum(p.log_likelihood for p in parts)))
    expected_mass = np.zeros(24, np.float32)
    for p in parts:
        np.add.at(expected_mass, p.diagnostics["global_rows"], np.asarray(p.diagnostics["rotation_mass"]))
    assert_matches(actual.diagnostics["rotation_mass"], expected_mass)
    assert np.all(actual.diagnostics["rotation_mass"][16:] == 0)
    for key in ("offset_second_sum_px2",):
        assert_matches(np.float32(actual.diagnostics[key]), np.float32(sum(p.diagnostics[key] for p in parts)))
    best = [int(p.diagnostics["global_rows"][int(np.asarray(p.diagnostics["best_rotation_idx"])[0])]) for p in parts]
    assert actual.diagnostics["best_rotation_idx"].tolist() == best
    assert np.array_equal(
        actual.diagnostics["n_significant_per_image"],
        np.concatenate([np.asarray(p.diagnostics["n_significant_per_image"]) for p in parts]),
    )


def test_device_workers_keep_item_order_and_full_float32():
    import jax

    from relax.ppca_initial_model.iteration_loop import _coarse_significance, _on_devices

    device = jax.devices()[0]
    seen = list(_on_devices([device, device], lambda _d, item: (item, jax.config.jax_default_matmul_precision), range(7)))
    assert seen == [(item, "highest") for item in range(7)]

    class _Result:
        def __init__(self, ids):
            self.significant_sample_indices = [np.asarray([i]) for i in ids]

    chunks = []
    rows = _coarse_significance([device] * 3, 4, np.arange(21), lambda ids: chunks.append(ids) or _Result(ids))
    # Chunks start on image-batch boundaries, so every batch keeps its one-device images.
    assert [c.tolist()[0] for c in chunks] == [0, 8, 16] and all(len(c) % 4 == 0 for c in chunks[:-1])
    assert [int(r[0]) for r in rows] == list(range(21))


def test_two_devices_reproduce_one_device_tiles():
    """Tiles on two (forced CPU) devices match one device and merge on the first."""
    import os
    import subprocess
    import sys

    here = Path(__file__).resolve().parent
    script = f"""
import sys
sys.path[:0] = [{str(here.parent.parent)!r}, {str(here)!r}]
import jax
import numpy as np
import test_full_fine_streaming as t
from helpers.float_compare import assert_matches
from relax.ppca_initial_model.iteration_loop import _on_devices, _to_device
from relax.ppca_refinement.full_row_stream import accumulate_full_row_tile
devices = jax.devices()
assert len(devices) == 2
problems = {{d.id: t.make_tile_problem(d) for d in devices}}
tiles = [(0, 2), (2, 3)]
def run(device, tile):
    stream = problems[device.id][3]
    part = accumulate_full_row_tile(stream, np.arange(*tile), t.PRUNED[tile[0]:tile[1]], factor_once=tile[1] - tile[0] > 1)
    return _to_device(part, devices[0])
one = [run(devices[0], tile) for tile in tiles]
two = list(_on_devices(devices, run, tiles))
assert two[1].rhs.devices() == {{devices[0]}}
for a, b in zip(one, two):
    for name in ("rhs", "lhs_tri", "residual_gradient", "residual_num", "embeddings"):
        assert_matches(np.asarray(getattr(b, name)), np.asarray(getattr(a, name)))
print("ok")
"""
    env = {**os.environ, "XLA_FLAGS": "--xla_force_host_platform_device_count=2", "JAX_PLATFORMS": "cpu",
           "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0 and result.stdout.strip().endswith("ok"), result.stderr[-3000:]


def test_support_tile_order_groups_similar_support_and_keeps_full_rows():
    from relax.ppca_initial_model.iteration_loop import _support_tile_order

    assert _support_tile_order([None, None, None], 4, 3, 2).tolist() == [0, 1, 2]
    # Packed ids are rotation * 3 + translation; supports {0}, {3}, {0, 1}, {3, 2}, {0}.
    significant = [np.asarray([0, 1]), np.asarray([9]), np.asarray([2, 3]), np.asarray([10, 6]), np.asarray([1])]
    order = _support_tile_order(significant, 4, 3, 2)
    assert sorted(order.tolist()) == list(range(5))
    tiles = [set(order[i:i + 2].tolist()) for i in range(0, 5, 2)]
    assert {0, 4} in tiles  # the two single-rotation {0} images share one tile


def test_fine_score_functions_share_the_shifted_frame():
    """Blocked and factor-once scores both exclude the same pose-invariant -y_norm/2."""
    import jax
    from recovar.ppca.pose_marginal import compute_ppca_pose_scores_and_moments_no_contrast

    from relax.ppca_refinement.engine import (
        _per_pose_stats_block,
        dense_pose_ppca_score_with_moments_blocked,
        dense_pose_ppca_score_with_moments_factor_once,
    )

    with jax.enable_x64(True):
        rng = np.random.default_rng(9)
        Y1 = jnp.asarray(rng.standard_normal((2, 3, 5)) + 1j * rng.standard_normal((2, 3, 5)))
        proj = jnp.asarray(rng.standard_normal((4, 3, 5)) + 1j * rng.standard_normal((4, 3, 5)))
        ctf2 = jnp.asarray(rng.uniform(0.2, 1.0, (2, 5)))
        y_norm = jnp.asarray([1500.0, 2100.0])
        blocked = dense_pose_ppca_score_with_moments_blocked(Y1, proj, ctf2, y_norm)
        factor = dense_pose_ppca_score_with_moments_factor_once(Y1, proj, ctf2, y_norm)
        absolute, _, _ = compute_ppca_pose_scores_and_moments_no_contrast(
            *_per_pose_stats_block(Y1, proj, ctf2, y_norm), return_moments=True
        )
        assert_matches(np.asarray(blocked.score_offset), -0.5 * np.asarray(y_norm))
        assert_matches(np.asarray(factor.score_offset), np.asarray(blocked.score_offset))
        assert_matches(np.asarray(blocked.score + blocked.score_offset[:, None, None]), np.asarray(absolute))
        assert_matches(np.asarray(factor.score), np.asarray(blocked.score))
        assert_matches(np.asarray(factor.logZ), np.asarray(blocked.logZ))


def test_compensated_block_sums_keep_small_contributions_under_jit():
    """Increments below half a float32 ULP of the total are kept, as in pass-2 volume sums."""
    import jax

    from relax.ppca_refinement.full_row_stream import compensated_add

    step = jax.jit(compensated_add)
    total = jnp.full((4,), 1.0e6, jnp.float32)
    compensation = jnp.zeros_like(total)
    plain = total
    for _ in range(1000):
        total, compensation = step(total, compensation, jnp.full((4,), 0.01, jnp.float32))
        plain = plain + jnp.float32(0.01)
    assert np.all(np.asarray(plain) == np.float32(1.0e6))  # the naive float32 sum loses every term
    assert_matches(np.asarray(total, np.float64), np.full(4, 1.0e6 + 10.0))
