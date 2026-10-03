"""Streamed full rotation rows versus the independent local layout and host-mask engine."""


import jax.numpy as jnp
import numpy as np
import pytest
import recovar.core.fourier_transform_utils as ftu
from helpers.float_compare import assert_matches

from relax import sampling
from relax.helpers.half_spectrum import make_half_image_weights, make_shell_indices_half
from relax.local.local_layout import build_pass2_hypothesis_layout
from relax.ppca_refinement.config import GeometryConfig, ScheduleConfig, ScoringConfig, SparsePass2Config
from relax.ppca_refinement.dense_dataset import (
    _per_image_pose_prior_block,
    accumulate_dense_ppca_statistics,
    compute_dense_ppca_embeddings,
)
from relax.ppca_refinement.full_row_stream import (
    FULL_ROW_ENGINE,
    TILE_FRAGMENTATION_HEADROOM,
    TracePPCAStats,
    accumulate_full_row_tile,
    accumulate_full_row_tiles,
    coarse_support_mask,
    full_row_pose_log_prior,
    full_row_tile_embeddings,
    plan_tile_images,
    prepare_full_row_stream,
    tile_bytes,
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


def make_tile_problem(device=None, q=2, metric_trace_only=False, gemm_precision="fp32"):
    reference, shared, parent = _layouts()
    rng = np.random.default_rng(3)
    images = (rng.standard_normal((3, N_HALF)) + 1j * rng.standard_normal((3, N_HALF))).astype(np.complex64)
    mu = _half_volume(rng)
    W = (np.stack([_half_volume(rng, 0.3), _half_volume(rng, 0.2)], axis=1) if q == 2
         else np.stack([_half_volume(rng, 0.2) for _ in range(q)], axis=1))
    translation_prior = (-np.sum(shared.translation_grid**2, axis=-1) / 8.0).astype(np.float32)
    common = dict(
        noise_variance=np.full(IMAGE_SHAPE, 40.0, np.float32),
        geometry=GeometryConfig(current_size=6, q=q, volume_domain="fourier_half"),
        # Five-row blocks leave a four-row remainder block.
        schedule=ScheduleConfig(image_batch_size=3, rotation_block_size=5),
        scoring=ScoringConfig(relion_texture_interp=False, full_real_observation=True),
    )
    stream = prepare_full_row_stream(
        _TinyData(images), mu, W,
        rotations=shared.rotations_flat, translations=shared.translation_grid,
        rotation_log_prior=shared.rotation_log_priors_flat, translation_log_prior=translation_prior,
        rotation_parent=shared.rotation_posterior_ids_flat, translation_parent=parent,
        n_coarse_rotations=3, n_coarse_translations=3, device=device, metric_trace_only=metric_trace_only,
        gemm_precision=gemm_precision, **common,
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
    actual = accumulate_full_row_tile(stream, np.arange(3), SIGNIFICANT)
    assert actual.diagnostics["engine"] == FULL_ROW_ENGINE
    assert actual.rhs is None  # the streamed engine produces no RHS volume
    assert actual.n_images == expected.n_images == 3
    assert np.array_equal(actual.original_image_ids, expected.original_image_ids)
    for name in ("lhs_tri", "residual_gradient", "residual_num", "residual_den", "embeddings"):
        assert_matches(np.asarray(getattr(actual, name)), np.asarray(getattr(expected, name)))
    # Scalar summaries are float32 reductions returned as Python floats.
    assert_matches(np.float32(actual.log_likelihood), np.float32(expected.log_likelihood))
    for key in ("offset_second_sum_px2", "latent_covariance_trace_mean", "pose_entropy_mean"):
        assert_matches(np.float32(actual.diagnostics[key]), np.float32(expected.diagnostics[key]))
    truth = _float64_diagnostics(stream, np.arange(3), SIGNIFICANT)
    for key in ("rotation_mass", "max_posterior_per_image"):
        _assert_matches_or_more_accurate(actual.diagnostics[key], expected.diagnostics[key], truth[key])
    for key in ("best_rotation_idx", "best_translation_idx", "n_significant_per_image"):
        assert np.array_equal(np.asarray(actual.diagnostics[key]), np.asarray(expected.diagnostics[key]))
    assert_matches(np.sum(actual.diagnostics["rotation_mass"]), np.float32(actual.n_images))


def test_trace_only_stream_matches_host_mask_metric_trace(tile_problem):
    """Momentum SGD's trace-only metric is the trace of the host-mask LHS; every other statistic is unchanged."""
    dataset, mu, W, _stream, host = tile_problem
    expected = full_float32(accumulate_dense_ppca_statistics)(
        dataset, mu, W, sparse_pass2=SparsePass2Config(enabled=False), collect_residuals=True, **host,
    )
    stream = make_tile_problem(metric_trace_only=True)[3]
    actual = accumulate_full_row_tile(stream, np.arange(3), SIGNIFICANT)
    assert isinstance(actual, TracePPCAStats) and actual.lhs_tri is None and actual.rhs is None
    P = W.shape[1] + 1
    lhs = np.asarray(expected.lhs_tri, np.float64)
    trace = sum(lhs[:, k] for k, (i, j) in enumerate(zip(*np.triu_indices(P))) if i == j)
    assert np.asarray(actual.metric_trace).shape == (lhs.shape[0],)
    assert_matches(np.asarray(actual.metric_trace), trace.astype(np.float32))
    for name in ("residual_gradient", "residual_num", "residual_den", "embeddings"):
        assert_matches(np.asarray(getattr(actual, name)), np.asarray(getattr(expected, name)))
    assert_matches(np.float32(actual.log_likelihood), np.float32(expected.log_likelihood))


def test_device_resident_tile_embeddings_match_host_mask(tile_problem):
    dataset, mu, W, stream, host = tile_problem
    expected = full_float32(compute_dense_ppca_embeddings)(dataset, mu, W, **host)
    actual = full_row_tile_embeddings(stream, np.arange(3), SIGNIFICANT)
    assert actual.n_images == 3 and np.array_equal(actual.original_image_ids, expected.original_image_ids)
    assert_matches(np.asarray(actual.embeddings), np.asarray(expected.embeddings))


@pytest.mark.parametrize("factor_once", [False, True])
def test_one_artificial_parent_recompute_matches_full_coarse_reference(tile_problem, factor_once):
    dataset, mu, W, _stream, host = tile_problem
    rotations, translations = host["rotations"], host["translations"]
    stream = prepare_full_row_stream(
        dataset,
        mu,
        W,
        rotations=rotations,
        translations=translations,
        rotation_log_prior=host["rotation_log_prior"],
        translation_log_prior=host["translation_log_prior"],
        rotation_parent=np.zeros(len(rotations), np.int32),
        translation_parent=np.zeros(len(translations), np.int32),
        n_coarse_rotations=1,
        n_coarse_translations=1,
        noise_variance=host["noise_variance"],
        geometry=host["geometry"],
        schedule=host["schedule"],
        scoring=host["scoring"],
        gemm_precision="fp32",  # the float32 equivalence check; TF32 is judged by science
    )
    expected = full_float32(accumulate_dense_ppca_statistics)(
        dataset,
        mu,
        W,
        sparse_pass2=SparsePass2Config(enabled=False),
        collect_residuals=True,
        factor_once_score=factor_once,
        **{key: value for key, value in host.items() if key != "rotation_translation_mask"},
    )
    actual = accumulate_full_row_tile(stream, np.arange(3), [None] * 3)
    assert actual.diagnostics["supported_image_rows"] == 3 * len(rotations)
    for name in ("lhs_tri", "residual_gradient", "residual_num", "residual_den", "embeddings"):
        assert_matches(np.asarray(getattr(actual, name)), np.asarray(getattr(expected, name)))
    truth = _float64_diagnostics(stream, np.arange(3), [None] * 3)["rotation_mass"]
    _assert_matches_or_more_accurate(actual.diagnostics["rotation_mass"], expected.diagnostics["rotation_mass"], truth)
    assert_matches(np.sum(actual.diagnostics["rotation_mass"]), np.float32(actual.n_images))


def test_tile_planner_caps_tiles_to_device_memory(tile_problem):
    """Tiles hold at most the requested images, else the most whose counted bytes fit the budget."""
    _dataset, _mu, _W, stream, _host = tile_problem
    assert plan_tile_images(stream, 150) == 150  # the tiny problem fits any device (CPU: no cap)
    budget = tile_bytes(stream, 10) / (1 - TILE_FRAGMENTATION_HEADROOM)
    assert plan_tile_images(stream, 150, memory_bytes=budget, device_bytes=budget) == 10
    assert tile_bytes(stream, 11) > (1 - TILE_FRAGMENTATION_HEADROOM) * budget
    assert plan_tile_images(stream, 150, memory_bytes=1) == 1
    # A reader that reports more bytes per image (a subtomogram reader's tilts) gets smaller tiles.
    def heavy(stream, n):
        return 100 * n * 4096, 10 * n * 4096

    def loader(*args, **kwargs):
        raise AssertionError("not read")

    loader.operand_bytes = heavy
    assert plan_tile_images(stream._replace(tile_loader=loader), 150, memory_bytes=budget) < 10


def test_pass2_row_skip_drops_only_negligible_rows(tile_problem):
    """Skipped rows keep their pass-1 positions empty; the statistics move by at most the dropped mass."""
    dataset, mu, W, _stream, host = tile_problem
    rotations, translations = host["rotations"], host["translations"]
    kwargs = dict(
        rotations=rotations,
        translations=translations,
        rotation_log_prior=host["rotation_log_prior"],
        translation_log_prior=host["translation_log_prior"],
        rotation_parent=np.zeros(len(rotations), np.int32),
        translation_parent=np.zeros(len(translations), np.int32),
        n_coarse_rotations=1,
        n_coarse_translations=1,
        noise_variance=host["noise_variance"],
        geometry=host["geometry"],
        schedule=host["schedule"],
        scoring=host["scoring"],
        gemm_precision="fp32",
    )
    dense = accumulate_full_row_tile(prepare_full_row_stream(dataset, mu, W, **kwargs), np.arange(3), [None] * 3)
    mass = np.asarray(dense.diagnostics["rotation_mass"], np.float64)
    assert dense.diagnostics["pass2_rows"] == dense.diagnostics["scored_rows"]
    # A floor that only a row with almost no mass falls below visits every row and changes nothing.
    tiny = prepare_full_row_stream(dataset, mu, W, pass2_mass_floor=1e-30, **kwargs)
    same = accumulate_full_row_tile(tiny, np.arange(3), [None] * 3)
    for name in ("lhs_tri", "residual_gradient", "embeddings"):
        assert_matches(np.asarray(getattr(same, name)), np.asarray(getattr(dense, name)))
    # A floor that keeps most rows runs pass 2 in place: nothing is skipped.
    most = prepare_full_row_stream(dataset, mu, W, pass2_mass_floor=float(np.quantile(mass, 0.1)), **kwargs)
    kept_most = accumulate_full_row_tile(most, np.arange(3), [None] * 3)
    assert kept_most.diagnostics["pass2_rows"] == kept_most.diagnostics["scored_rows"]
    assert_matches(np.asarray(kept_most.lhs_tri), np.asarray(dense.lhs_tri))
    # Skip the lighter three quarters of the rows (by total mass, an upper bound on each image's mass).
    floor = float(np.quantile(mass[mass > 0], 0.75))
    skip = prepare_full_row_stream(dataset, mu, W, pass2_mass_floor=floor, **kwargs)
    skipped = accumulate_full_row_tile(skip, np.arange(3), [None] * 3)
    visited = np.asarray(skipped.diagnostics["rotation_mass"]) > 0
    assert 0 < skipped.diagnostics["pass2_rows"] < skipped.diagnostics["scored_rows"]
    assert_matches(np.asarray(skipped.diagnostics["rotation_mass"])[visited], mass[visited].astype(np.float32))
    assert np.all(mass[~visited] < len(rotations) * 3 * floor)
    dropped = float(np.sum(mass[~visited])) / 3  # mean dropped mass per image
    for name in ("lhs_tri", "embeddings"):
        a, b = np.asarray(getattr(skipped, name), np.float64), np.asarray(getattr(dense, name), np.float64)
        assert np.linalg.norm(a - b) <= (dropped + 1e-5) * np.linalg.norm(b) * 10


def test_pass2_floor_zero_visits_every_row(tile_problem):
    """tau = 0 is the no-skip path: no row selection, identical statistics."""
    from relax.ppca_refinement.full_row_stream import _pass2_rows

    _dataset, _mu, _W, stream, _host = tile_problem
    assert stream.pass2_mass_floor == 0.0
    assert _pass2_rows(stream, None, None, 1) is None
    explicit = stream._replace(pass2_mass_floor=0.0)
    a = accumulate_full_row_tile(stream, np.arange(3), SIGNIFICANT)
    b = accumulate_full_row_tile(explicit, np.arange(3), SIGNIFICANT)
    for name in ("lhs_tri", "residual_gradient", "residual_num", "embeddings"):
        assert_matches(np.asarray(getattr(b, name)), np.asarray(getattr(a, name)))
    assert b.diagnostics["pass2_rows"] == b.diagnostics["scored_rows"]


def test_rotation_mass_returns_to_pass1_rows():
    """Compacted rows' masses land at their pass-1 positions; padding adds nothing."""
    from relax.ppca_refinement.full_row_stream import _expand_rotation_mass

    compact = jnp.asarray([5.0, 7.0, 11.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    positions = jnp.asarray([1, 4, 6, 0], jnp.int32)  # three visited rows, one padding slot
    out = np.asarray(_expand_rotation_mass(compact, positions, np.int32(3)))
    assert out.tolist() == [0.0, 5.0, 0.0, 0.0, 7.0, 0.0, 11.0, 0.0]


def _padded_loader(n_dummies):
    """The single-particle reader, with ``n_dummies`` zero-operand images appended after the real ones."""
    from relax.ppca_refinement import full_row_stream as frs

    def load(stream, image_indices, significant_rows, *, collect_observation):
        tile, observation, layout = frs._load_tile(
            stream, image_indices, significant_rows, collect_observation=collect_observation
        )
        B, T = len(image_indices), int(stream.translations.shape[0])
        pad = -((B + n_dummies) * T) % frs._SHIFT_ALIGN if stream.static.cuda_kernels else 0
        extra = n_dummies * T
        Y1 = jnp.pad(tile.Y1[:, : B * T], ((0, 0), (0, extra + pad)))
        Y1_recon = jnp.pad(tile.Y1_recon[: B * T], ((0, extra + pad), (0, 0)))
        tile = tile._replace(
            coarse_mask=jnp.concatenate([tile.coarse_mask, jnp.repeat(tile.coarse_mask[:1], n_dummies, 0)]),
            Y1=Y1,
            ctf2=jnp.pad(tile.ctf2, ((0, 0), (0, n_dummies))),
            Y1_recon=Y1_recon,
            ctf2_recon=jnp.pad(tile.ctf2_recon, ((0, n_dummies), (0, 0))),
            y_norm=jnp.pad(tile.y_norm, (0, n_dummies)),
        )
        layout["n_real"] = B
        return tile, observation, layout

    return load


def test_padded_tile_matches_the_real_tile(tile_problem):
    """Dummy images appended to keep a compiled tile size add no statistics and report nothing."""
    _dataset, _mu, _W, stream, _host = tile_problem
    real = accumulate_full_row_tile(stream, np.arange(3), SIGNIFICANT)
    padded_stream = stream._replace(tile_loader=_padded_loader(2), image_batch_size=5)
    padded = accumulate_full_row_tile(padded_stream, np.arange(3), SIGNIFICANT)
    assert padded.n_images == 3 and np.asarray(padded.embeddings).shape == np.asarray(real.embeddings).shape
    for name in ("lhs_tri", "residual_gradient", "residual_num", "residual_den", "embeddings"):
        assert_matches(np.asarray(getattr(padded, name)), np.asarray(getattr(real, name)))
    assert_matches(np.float32(padded.log_likelihood), np.float32(real.log_likelihood))
    for key in ("rotation_mass", "max_posterior_per_image"):
        assert_matches(np.asarray(padded.diagnostics[key]), np.asarray(real.diagnostics[key]))
    for key in ("offset_second_sum_px2", "latent_covariance_trace_mean", "pose_entropy_mean"):
        assert_matches(np.float32(padded.diagnostics[key]), np.float32(real.diagnostics[key]))
    assert np.array_equal(padded.diagnostics["n_significant_per_image"], real.diagnostics["n_significant_per_image"])
    embedded = full_row_tile_embeddings(padded_stream, np.arange(3), SIGNIFICANT)
    assert embedded.n_images == 3
    assert_matches(np.asarray(embedded.embeddings), np.asarray(real.embeddings))


def test_tile_size_buckets_cap_compiled_shapes():
    """Tiles round up to at most TILE_SIZE_BUCKETS sizes per plan, each holding the tile."""
    from relax.ppca_refinement.full_row_stream import TILE_SIZE_BUCKETS, tile_size_bucket

    for planned in (1, 7, 16, 150):
        sizes = {tile_size_bucket(n, planned) for n in range(1, planned + 1)}
        assert len(sizes) <= TILE_SIZE_BUCKETS and max(sizes) == planned
        assert all(tile_size_bucket(n, planned) >= n for n in range(1, planned + 1))
    assert sorted({tile_size_bucket(n, 150) for n in range(1, 151)}) == [10, 19, 38, 75, 150]
    with pytest.raises(ValueError):
        tile_size_bucket(151, 150)


def test_pipelined_tiles_match_separate_tiles(tile_problem):
    """Pipelining (next tile loaded, previous finished, kept buffer reused) changes no tile's statistics."""
    _dataset, _mu, _W, stream, _host = tile_problem
    # Two equal-size tiles share one kept buffer; the last, smaller one gets its own.
    tiles = [(np.arange(2), PRUNED[:2]), (np.arange(1, 3), PRUNED[1:]), (np.arange(2, 3), PRUNED[2:])]
    pipelined = accumulate_full_row_tiles(stream, tiles)
    for (ids, support), actual in zip(tiles, pipelined, strict=True):
        expected = accumulate_full_row_tile(stream, ids, support)
        assert np.array_equal(actual.original_image_ids, expected.original_image_ids)
        for name in ("lhs_tri", "residual_gradient", "residual_num", "residual_den", "embeddings"):
            assert_matches(np.asarray(getattr(actual, name)), np.asarray(getattr(expected, name)))


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
    # The moment-image form takes component-major projections, (P, R, F).
    residual, correction_over_w = full_float32(residual_statistics_from_moment_images)(
        rhs, lhs, jnp.swapaxes(d["projections"], 0, 1)
    )
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
    actual = accumulate_full_row_tile(stream, np.arange(3), PRUNED)
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
    for name in ("lhs_tri", "residual_gradient", "residual_num", "residual_den"):
        assert_matches(np.asarray(getattr(actual, name)), sum(np.asarray(getattr(p, name)) for p in parts))
    assert_matches(np.asarray(actual.embeddings), np.concatenate([np.asarray(p.embeddings) for p in parts]))
    assert_matches(np.float32(actual.log_likelihood), np.float32(sum(p.log_likelihood for p in parts)))
    expected_mass = np.zeros(24, np.float32)
    for p in parts:
        np.add.at(expected_mass, p.diagnostics["global_rows"], np.asarray(p.diagnostics["rotation_mass"]))
    truth = _float64_diagnostics(stream, np.arange(3), PRUNED)["rotation_mass"]
    _assert_matches_or_more_accurate(actual.diagnostics["rotation_mass"], expected_mass, truth)
    assert np.all(actual.diagnostics["rotation_mass"][16:] == 0)
    for key in ("offset_second_sum_px2",):
        assert_matches(np.float32(actual.diagnostics[key]), np.float32(sum(p.diagnostics[key] for p in parts)))
    best = [int(p.diagnostics["global_rows"][int(np.asarray(p.diagnostics["best_rotation_idx"])[0])]) for p in parts]
    assert actual.diagnostics["best_rotation_idx"].tolist() == best
    assert np.array_equal(
        actual.diagnostics["n_significant_per_image"],
        np.concatenate([np.asarray(p.diagnostics["n_significant_per_image"]) for p in parts]),
    )


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


def _float64_tile_statistics(stream, image_indices, significant):
    """The streamed engine's tile statistics with every operand and sum in float64 (x64, XLA moments)."""
    import jax

    from relax.ppca_refinement import full_row_stream as frs
    from relax.ppca_refinement.engine import _enforce_augmented_x0

    def up(tree):
        def one(x):
            if not hasattr(x, "dtype"):
                return x
            if jnp.issubdtype(x.dtype, jnp.complexfloating):
                return x.astype(jnp.complex128)
            return x.astype(jnp.float64) if jnp.issubdtype(x.dtype, jnp.floating) else x

        return type(tree)(*[one(x) for x in tree])

    with jax.enable_x64(True), jax.default_matmul_precision("highest"):
        # The CUDA stream kernels are float32-only; the float64 truth runs the XLA formulation.
        xla = stream._replace(static=stream.static._replace(cuda_kernels=False))
        tile32, observation_power, layout = frs._load_tile(xla, image_indices, significant, collect_observation=True)
        tile, arrays = up(tile32), up(stream.arrays)
        wide = xla._replace(arrays=arrays)
        n_images = len(image_indices)
        capacity = len(stream.block_starts) * stream.rotation_block_size
        q = stream.static.basis_size - 1
        T = len(stream.translations)
        kept = frs._empty_kept(capacity, n_images, T, q, jnp.float64)
        for start in stream.block_starts[: layout["n_blocks"]]:
            kept = frs._score_block(arrays, tile, kept, start, static=wide.static,
                                    block_size=stream.rotation_block_size)
        posterior = frs._normalize(
            kept.score, tile.rows, n_blocks=layout["n_blocks"], block_size=stream.rotation_block_size
        )
        carry = up(frs._empty_carry(stream, n_images, observation_power))
        carry = frs._run_pass2(wide, tile, kept, posterior, layout["n_blocks"], carry, moments=True)
        volume_shape = stream.static.volume_shape
        weights = make_half_image_weights(stream.static.image_shape)
        shells = np.asarray(make_shell_indices_half(stream.static.image_shape))
        residual_num = np.zeros(shells.max() + 1)
        np.add.at(residual_num, shells, np.asarray(weights * carry.residual_power))
        lhs_tri, residual = frs._unpack_moments(carry.moments, stream.static)
        rotation_mass = np.zeros(stream.rotation_parent.size)
        rotation_mass[layout["rows"]] = np.asarray(carry.rotation_mass)[: layout["rows"].size]
        pmax = np.exp(np.asarray(posterior.top_score - posterior.center) - np.asarray(posterior.centered_logZ))
        return {
            "lhs_tri": np.asarray(jnp.swapaxes(
                _enforce_augmented_x0(lhs_tri.astype(jnp.complex128), volume_shape).real, 0, 1)),
            "residual_gradient": np.asarray(residual),
            "residual_num": residual_num,
            "embeddings": np.asarray(carry.embedding),
            "diagnostics": {"rotation_mass": rotation_mass, "max_posterior_per_image": pmax},
        }


def _float64_diagnostics(stream, image_indices, significant):
    return _float64_tile_statistics(stream, image_indices, significant)["diagnostics"]


def _assert_matches_or_more_accurate(actual, host, truth):
    """``actual`` matches the float32 ``host`` value (band 1e-6), or is no further than it from float64 ``truth``.

    Posterior-weight diagnostics carry the float32 rounding of the pose scores (relative to their
    magnitude) through ``exp``, so two float32 association orders can differ by a few 1e-6 while
    both stay that close to float64. A different order passes only by being at least as accurate.
    """

    def band(a, b):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        return float(np.max(np.abs(a - b)) / max(np.max(np.abs(a)), np.max(np.abs(b))))

    if band(actual, host) <= 1e-6:
        return
    assert band(actual, truth) <= band(host, truth), (band(actual, host), band(actual, truth), band(host, truth))


def _relative_l2(actual, truth):
    return float(np.linalg.norm(np.asarray(actual, np.complex128) - truth) / np.linalg.norm(truth))


@pytest.mark.parametrize("q", [4, 10])
def test_general_rank_coarse_recompute_matches_dense_reference(q):
    """Real projection, score, moments and streamed pass 2 at general rank, against float64.

    The reference is the streamed formulation evaluated in float64. The float32 engine's
    relative L2 error against it must be no larger than that of the independent float32
    host-mask path (both of its score formulations), field by field, or within 1e-6.
    """
    data, mu, W, stream, host = make_tile_problem(q=q)
    truth = _float64_tile_statistics(stream, np.arange(3), SIGNIFICANT)
    actual = accumulate_full_row_tile(stream, np.arange(3), SIGNIFICANT)
    previous = [
        full_float32(accumulate_dense_ppca_statistics)(
            data, mu, W, sparse_pass2=SparsePass2Config(enabled=False), collect_residuals=True,
            factor_once_score=factor_once, **host)
        for factor_once in (False, True)
    ]
    truth.pop("diagnostics")
    for name, value in truth.items():
        error = _relative_l2(getattr(actual, name), value)
        previous_error = max(_relative_l2(getattr(p, name), value) for p in previous)
        assert error <= max(1e-6, previous_error), (name, error, previous_error)
    assert_matches(np.asarray(actual.residual_den), np.asarray(previous[0].residual_den))
    assert actual.rhs is None
    assert actual.residual_gradient.shape[-1] == q + 1
    assert actual.lhs_tri.shape[-1] == (q + 1) * (q + 2) // 2
    assert actual.embeddings.shape == (3, q)
    assert actual.residual_gradient.dtype == jnp.complex64
    assert actual.lhs_tri.dtype == actual.embeddings.dtype == jnp.float32


def test_tf32_stream_gemms_need_ampere_and_auto_selects_them(tile_problem):
    """``"tf32"`` changes the stream GEMMs on sm_80+ GPUs and is refused elsewhere; ``"auto"`` picks it there.

    TF32 keeps 10 mantissa bits, so its statistics leave the float32 run's noise (about 1e-7);
    whether that is acceptable is decided by end-to-end science, not here.
    """
    import jax

    device = jax.local_devices()[0]
    capable = device.platform == "gpu" and float(getattr(device, "compute_capability", "0")) >= 8.0
    with pytest.raises(ValueError, match="gemm_precision"):
        make_tile_problem(gemm_precision="bf16")
    if not capable:
        with pytest.raises(ValueError, match="compute capability"):
            make_tile_problem(gemm_precision="tf32")
        assert make_tile_problem(gemm_precision="auto")[3].static.gemm_precision == "fp32"
        return
    _dataset, _mu, _W, stream, _host = tile_problem
    assert make_tile_problem(gemm_precision="auto")[3].static.gemm_precision == "tf32"
    fp32 = accumulate_full_row_tile(stream, np.arange(3), SIGNIFICANT)
    tf32 = accumulate_full_row_tile(make_tile_problem(gemm_precision="tf32")[3], np.arange(3), SIGNIFICANT)
    assert fp32.diagnostics["gemm_precision"] == "fp32" and tf32.diagnostics["gemm_precision"] == "tf32"
    for name in ("lhs_tri", "residual_gradient", "embeddings"):
        a, b = np.asarray(getattr(fp32, name)), np.asarray(getattr(tf32, name))
        assert np.all(np.isfinite(b))
        assert _relative_l2(b, a.astype(np.complex128)) > 1e-5, name  # the switch reaches the GEMMs
