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


def _spa_stream(*, metric_trace_only=False, order=0, n_images=4, device=None, current_size=6):
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
        geometry=GeometryConfig(current_size=current_size, q=2, volume_domain="fourier_half"),
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
        coarse,
        fine_rotations,
        fine_translations,
        adaptive_fraction=fraction,
        max_significant=cap,
        job_chunk=16,
        fine_fraction=1.0,  # every scored child is accumulated, as the dense stream does
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
    rotation, translation, valid, mass, capped = (
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
        # The mass the kept samples hold, and whether the cap stopped them short of the fraction.
        assert_matches(np.float32(mass[b]), np.float32(weights.reshape(-1)[expected].sum()), rtol=DIAGNOSTIC_RTOL)
        assert capped[b] == (np.sort(weights.reshape(-1))[::-1][:cap].sum() <= fraction)


@pytest.mark.parametrize("fine_fraction, tilts", [(0.9, False), (0.5, False), (0.999, False), (0.8, True)])
def test_pass2_accumulates_the_significant_fine_samples(monkeypatch, fine_fraction, tilts):
    """Pass 2 accumulates each image's significant fine samples (RELION's rule over the child weights, in float64
    here from the job scores) and visits only the (image, child rotation) rows that hold one.

    The job scores and latent means are the engine's (they equal the dense child grid's: the tests above with every
    child accumulated); the expected kept mass per coarse rotation, embeddings and rows follow from them.
    """
    from relax.ppca_refinement import oversampled_stream as oss

    if tilts:
        import test_tomo_ppca as tomo

        _particles, coarse, _arrays = tomo.make_problem(seed=4)
        perturb = Rotation.from_rotvec(np.random.default_rng(1).normal(scale=0.05, size=(8, 3))).as_matrix()
        fine_rotations = np.einsum("rab,cbd->rcad", np.asarray(coarse.arrays.rotations[:-1]), perturb)
        fine_rotations = fine_rotations.reshape(-1, 3, 3).astype(np.float32)
        fine_translations, _ = sampling.get_oversampled_translation_grid(tomo.TRANSLATIONS, 1.5, 1)
        ids = np.arange(3)
    else:
        coarse, (fine_rotations, fine_translations) = _spa_stream()
        ids = np.arange(4)
    ostream = prepare_oversampled_stream(
        coarse,
        fine_rotations,
        fine_translations,
        adaptive_fraction=0.95,
        max_significant=9,
        job_chunk=8,
        fine_fraction=fine_fraction,
    )
    scored, score_jobs = [], oss._score_jobs

    def recording(results, arrays, tile, phases, table, order, start, **kwargs):
        slots = np.asarray(order)[int(start) : int(start) + kwargs["chunk"]]
        jobs = [np.asarray(x)[slots] for x in table]
        out = score_jobs(results, arrays, tile, phases, table, order, start, **kwargs)
        scored.append([*jobs, np.asarray(out.score)[slots], np.asarray(out.mean)[slots]])
        return out

    monkeypatch.setattr(oss, "_score_jobs", recording)
    actual = accumulate_oversampled_tile(ostream, ids)
    image, rotation, _translation, valid, score, mean = (np.concatenate(parts) for parts in zip(*scored))
    n_coarse = coarse.arrays.rotations.shape[0] - 1
    mass = np.zeros(n_coarse)
    embeddings = np.zeros((len(ids), mean.shape[1]))
    rows = 0
    for b in range(len(ids)):
        jobs = np.flatnonzero(valid & (image == b))
        weights = np.exp(score[jobs].astype(np.float64) - score[jobs].max())  # (J_b, R_c, T_c)
        weights /= weights.sum()
        kept = _relion_significant(weights.reshape(-1), fine_fraction, weights.size).reshape(weights.shape)
        assert weights[kept].sum() > fine_fraction and not kept.all()
        np.add.at(mass, rotation[jobs], (weights * kept).sum(axis=(1, 2)))
        embeddings[b] = np.einsum("jrt,jqrt->q", weights * kept, mean[jobs])
        rows += int(kept.any(axis=2).sum())
    diagnostics = actual.diagnostics
    assert diagnostics["pass2_rows"] == rows < diagnostics["scored_fine_rows"] == int(valid.sum()) * 8
    assert_matches(diagnostics["rotation_mass"], mass.astype(np.float32), rtol=DIAGNOSTIC_RTOL)
    assert_matches(np.asarray(actual.embeddings), embeddings.astype(np.float32), rtol=DIAGNOSTIC_RTOL)
    # Every scored child accumulated: the rows are the scored ones.
    every = accumulate_oversampled_tile(ostream._replace(fine_fraction=1.0), ids)
    assert every.diagnostics["pass2_rows"] == every.diagnostics["scored_fine_rows"]


def test_padding_images_have_no_significant_samples():
    score = jnp.zeros((4, 3, 2), jnp.float32)
    rows = jnp.arange(4, dtype=jnp.int32)
    posterior = _normalize(score, rows, n_blocks=1, block_size=4)
    _, _, valid, mass, capped = _significant_samples(
        score, rows, posterior, jnp.float32(0.5), np.int32(2), n_significant=8, every=False
    )
    assert np.asarray(valid)[:2].any(axis=1).all() and not np.asarray(valid)[2:].any()
    assert not np.asarray(mass)[2:].any() and not np.asarray(capped)[2:].any()


@pytest.mark.parametrize("planned", [None, 8])
def test_subtomogram_children_equal_the_dense_child_grid(planned):
    """Tilt-series particles (shared latent over tilts, 3D shifts with 8 children each) through the tilt reader;
    with a planned tile size the reader pads the 3 particles to a tile of 4."""
    import test_tomo_ppca as tomo

    _particles, coarse, _arrays = tomo.make_problem(seed=4)
    coarse = coarse._replace(tile_images=planned)
    perturb = Rotation.from_rotvec(np.random.default_rng(1).normal(scale=0.05, size=(8, 3))).as_matrix()
    coarse_rotations = np.asarray(coarse.arrays.rotations[:-1])
    fine_rotations = np.einsum("rab,cbd->rcad", coarse_rotations, perturb).reshape(-1, 3, 3).astype(np.float32)
    fine_translations, parent = sampling.get_oversampled_translation_grid(tomo.TRANSLATIONS, 1.5, 1)
    assert fine_translations.shape == (8 * len(tomo.TRANSLATIONS), 3)
    np.testing.assert_array_equal(parent, np.repeat(np.arange(len(tomo.TRANSLATIONS)), 8))
    ostream = prepare_oversampled_stream(
        coarse,
        fine_rotations,
        fine_translations,
        adaptive_fraction=0.95,
        max_significant=9,
        job_chunk=8,
        fine_fraction=1.0,
    )
    ids = np.arange(3)
    actual = accumulate_oversampled_tile(ostream, ids)
    expected = accumulate_full_row_tile(ostream.fine, ids, actual.diagnostics["significant_samples"])
    _assert_same_statistics(actual, expected)
    assert_matches(actual.diagnostics["rotation_mass"], _parent_mass(expected, 8), rtol=DIAGNOSTIC_RTOL)


def test_pass1_on_the_coarse_window_equals_the_dense_child_grid_at_the_pass2_window():
    """RELION's coarse image size: pass 1 on a smaller window selects the samples, pass 2 scores at the full window."""
    coarse, (fine_rotations, fine_translations) = _spa_stream(current_size=4)
    pass2, _ = _spa_stream(current_size=6)
    ostream = prepare_oversampled_stream(
        coarse,
        fine_rotations,
        fine_translations,
        pass2=pass2,
        adaptive_fraction=0.9,
        max_significant=40,
        job_chunk=16,
        fine_fraction=1.0,
    )
    ids = np.arange(4)
    actual = accumulate_oversampled_tile(ostream, ids)
    expected = accumulate_full_row_tile(ostream.fine, ids, actual.diagnostics["significant_samples"])
    _assert_same_statistics(actual, expected)
    # The smaller window selects other samples than the full one.
    same_window = prepare_oversampled_stream(
        pass2, fine_rotations, fine_translations, adaptive_fraction=0.9, max_significant=40, job_chunk=16
    )
    full = accumulate_oversampled_tile(same_window, ids).diagnostics["significant_samples"]
    assert any(not np.array_equal(a, b) for a, b in zip(actual.diagnostics["significant_samples"], full))


@pytest.mark.parametrize("tilts", [False, True])
def test_pass1_tile_from_the_pass2_tile_is_the_readers(tilts):
    """Pass 1's operands, formed from the tile's unshifted pass-2 operands (window pixels times the coarse
    translations' phase factors), are the ones the tile reader builds for the pass-1 stream."""
    from relax.ppca_refinement import full_row_stream as frs
    from relax.ppca_refinement import oversampled_stream as oss

    if tilts:
        import test_tomo_ppca as tomo

        _particles, coarse, _arrays = tomo.make_problem(seed=4)
        # A planned tile of 8 pads the 3 particles to a tile of 4 (the reader's size buckets).
        coarse = coarse._replace(tile_images=8)
        pass2, ids = coarse, np.arange(3)
        fine_rotations = np.repeat(np.asarray(coarse.arrays.rotations[:-1]), 8, axis=0)
        fine_translations, _ = sampling.get_oversampled_translation_grid(tomo.TRANSLATIONS, 1.5, 1)
    else:
        coarse, (fine_rotations, fine_translations) = _spa_stream(current_size=4)
        pass2, ids = _spa_stream(current_size=6)[0], np.arange(4)
    ostream = prepare_oversampled_stream(coarse, fine_rotations, fine_translations, pass2=pass2, max_significant=4)
    slots = np.asarray(ostream.pass1_slots)
    if tilts:
        np.testing.assert_array_equal(slots[slots >= 0], np.flatnonzero(slots >= 0))  # one window
    else:
        assert 0 < (slots >= 0).sum() < frs._window_pixels(pass2)  # a smaller window
    support = [None] * len(ids)
    expected, _, expected_layout = frs._read_tile(coarse, ids, support, collect_observation=False)
    tile, _, layout = frs._read_tile(ostream.base, ids, support, collect_observation=False)
    actual, actual_layout = oss._pass1_tile(ostream, tile, layout, support)
    assert actual.Y1.shape == expected.Y1.shape and actual.ctf2.shape == expected.ctf2.shape
    assert actual.y_norm.shape[0] == (4 if tilts else len(ids)) and actual_layout.get("n_real") == expected_layout.get(
        "n_real"
    )
    # One complex product per pixel in another order than the reader's: float32 rounding of the product.
    assert_matches(np.asarray(actual.Y1), np.asarray(expected.Y1), rtol=STATISTICS_RTOL)
    assert_matches(np.asarray(actual.ctf2), np.asarray(expected.ctf2), rtol=STATISTICS_RTOL)
    np.testing.assert_array_equal(np.asarray(actual.rows), np.asarray(expected.rows))
    np.testing.assert_array_equal(np.asarray(actual.coarse_mask), np.asarray(expected.coarse_mask))
    assert {k: actual_layout[k] for k in ("n_blocks", "scored_rows")} == {
        k: expected_layout[k] for k in ("n_blocks", "scored_rows")
    }


def test_capped_images_report_the_mass_they_hold():
    """A cap below what the fraction needs stops every image short of it, and its mass says by how much."""
    coarse, (fine_rotations, fine_translations) = _spa_stream()
    capped = prepare_oversampled_stream(
        coarse, fine_rotations, fine_translations, adaptive_fraction=0.999, max_significant=2, job_chunk=8
    )
    diagnostics = accumulate_oversampled_tile(capped, np.arange(4)).diagnostics
    assert np.all(diagnostics["significant_capped_per_image"])
    assert np.all(diagnostics["significant_samples_per_image"] == 2)
    assert np.all(diagnostics["significant_mass_per_image"] < 0.999)
    free = prepare_oversampled_stream(
        coarse, fine_rotations, fine_translations, adaptive_fraction=0.5, max_significant=100, job_chunk=32
    )
    diagnostics = accumulate_oversampled_tile(free, np.arange(4)).diagnostics
    assert not np.any(diagnostics["significant_capped_per_image"])
    assert np.all(diagnostics["significant_mass_per_image"] > 0.5)


# These random particles have nearly flat posteriors: 0.999 of the mass needs most of the coarse poses.
@pytest.mark.parametrize("cap, fraction, warned", [(100, 0.05, False), (1, 0.999, True)])
def test_controller_runs_oversampled_updates_and_flags_the_cap(tmp_path, caplog, cap, fraction, warned):
    """Two oversampled updates of subtomogram particles through the controller: each update records the windows and
    the significant samples, and a cap that stops most images short of their mass logs a warning."""
    import json
    import logging

    import jax.numpy as jnp
    import recovar.core.fourier_transform_utils as ftu
    import test_tomo_ppca as tomo

    from relax.ppca_initial_model import iteration_loop
    from relax.ppca_initial_model.config import Config
    from relax.ppca_initial_model.tomo import TiltParticles

    iteration_loop._rotation_grid.cache_clear()
    iteration_loop._direction_ids.cache_clear()
    n = 16
    rng = np.random.default_rng(2)
    image_frames = [[0, 1, 2], [0, 2], [1, 2, 0], [0, 1, 2]] * 2
    n_images = sum(len(f) for f in image_frames)
    raw = rng.standard_normal((n_images, n, n)).astype(np.float32)
    half = np.asarray(ftu.get_dft2_real(raw)).reshape(n_images, -1).astype(np.complex64)
    offsets = np.concatenate([[0], np.cumsum([len(f) for f in image_frames])]).astype(np.int64)
    particles = TiltParticles(
        image_shape=(n, n),
        volume_shape=(n, n, n),
        voxel_size=10.0,
        image_offsets=offsets,
        image_frame=np.concatenate(image_frames).astype(np.int64),
        particle_group=np.asarray([0, 0, 0, 0, 1, 1, 1, 1]),
        group_frames=(tomo.FRAMES[:3], tomo.FRAMES[1:]),
        read=lambda ids: (raw[ids], jnp.asarray(half[ids]), jnp.full((len(ids), half.shape[1]), 0.8, jnp.float32)),
    )
    config = Config(
        q=2,
        iterations=2,
        stages=((1, 3, 0),),
        oversampling=1,
        max_significant=cap,
        target_mass=fraction,
        stream_coarse_recompute=True,
        shift_range=1,
        shift_step=1,
        image_batch_size=3,
        rotation_block_size=32,
    )
    try:
        with caplog.at_level(logging.WARNING, logger="relax.ppca_initial_model.iteration_loop"):
            state = iteration_loop.run(particles, config, tmp_path, {"test": True}, diameter_ang=120.0)
    finally:
        iteration_loop._rotation_grid.cache_clear()
        iteration_loop._direction_ids.cache_clear()
    rows = [json.loads(line) for line in (tmp_path / "iterations.jsonl").read_text().splitlines()]
    assert [row["iteration"] for row in rows] == [1, 2]
    for row in rows:
        record = row["oversampling"]
        assert record["order"] == 1 and record["max_significant"] == cap
        assert record["pass1_image_size"] <= record["pass2_image_size"] == 6
        assert 1 <= record["samples_median"] <= cap
        assert record["capped_fraction"] == (1.0 if warned else 0.0)
        assert 0 < record["accumulated_fine_row_fraction"] <= 1
    assert any("consider a larger --maxsig" in r.message for r in caplog.records) == warned
    assert np.all(np.isfinite(np.asarray(state.theta)))
    assert np.load(tmp_path / "embeddings.npz")["z"].shape == (8, 2)


def test_statistics_do_not_depend_on_the_job_chunk():
    """The job chunk is a memory plan: chunks of 2 jobs and one chunk of all give the same statistics within
    float32 reduction order (the moment scatter sums the same rows in another grouping)."""
    coarse, (fine_rotations, fine_translations) = _spa_stream()
    ids = np.arange(4)
    results = []
    for chunk in (2, 1024):
        ostream = prepare_oversampled_stream(
            coarse, fine_rotations, fine_translations, adaptive_fraction=0.9, max_significant=40, job_chunk=chunk
        )
        results.append(accumulate_oversampled_tile(ostream, ids))
    small, large = results
    _assert_same_statistics(small, large)
    for key in ("significant_samples_per_image", "pass2_rows", "scored_fine_rows"):
        np.testing.assert_array_equal(small.diagnostics[key], large.diagnostics[key])


def test_job_chunk_fits_one_kernel_launch():
    """The CUDA projector and scatter take 65535 rotations per launch: a chunk's jobs times their 8 child rotations
    times the tile's frames stay within it (41 tilts: 128 jobs, not 256, which failed at the k3conf stage)."""
    from types import SimpleNamespace

    from relax.ppca_refinement import oversampled_stream as oss

    def stream(frames, cuda_kernels=True):
        loader = SimpleNamespace(operand_bytes=lambda *a: (0, 0), max_frames=lambda _stream: frames)
        base = SimpleNamespace(tile_loader=loader if frames > 1 else None)
        return SimpleNamespace(
            fine=SimpleNamespace(static=SimpleNamespace(cuda_kernels=cuda_kernels)), base=base, rotation_children=8
        )

    assert oss.launch_job_chunk(stream(1)) == oss.MAX_JOB_CHUNK == 1024
    assert oss.launch_job_chunk(stream(41)) == 128
    assert 128 * 8 * 41 <= oss.KERNEL_LAUNCH_ROTATIONS < 256 * 8 * 41
    assert oss.launch_job_chunk(stream(41, cuda_kernels=False)) == 1024
    with pytest.raises(ValueError, match="one kernel launch"):
        oss.launch_job_chunk(stream(10000))


def test_chunks_cover_the_kept_positions_in_a_few_sizes():
    """A tile's jobs run in whole chunks of the planned size, then of 1/4 and 1/16 of it; the padding is less than
    one smallest chunk and addresses the padding slot."""
    from relax.ppca_refinement import oversampled_stream as oss

    assert (
        oss._chunk_sizes(1024) == (1024, 256, 64) and oss._chunk_sizes(8) == (8, 2, 1) and oss._chunk_sizes(1) == (1,)
    )
    rng = np.random.default_rng(0)
    for n_kept in (0, 1, 63, 64, 2000, 6500):
        mask = np.zeros(15000, bool)
        mask[rng.choice(15000, n_kept, replace=False)] = True
        order, count, plan = oss._chunks(mask.reshape(150, 100), (1024, 256, 64), padding=15000)
        order = np.asarray(order)
        assert count == n_kept and order.shape == (15000 + 1024,)
        np.testing.assert_array_equal(order[:n_kept], np.flatnonzero(mask))
        assert np.all(order[n_kept:] == 15000)
        covered = np.concatenate([np.arange(start, start + size) for start, size in plan] or [np.zeros(0, int)])
        np.testing.assert_array_equal(covered, np.arange(covered.size))  # consecutive, no overlap
        assert n_kept <= covered.size < n_kept + 64 and covered.size <= order.size
        assert [size for _, size in plan] == sorted((size for _, size in plan), reverse=True)
    assert [size for _, size in oss._chunks(np.ones(6500, bool), (1024, 256, 64), 0)[2]] == [1024] * 6 + [256] + [
        64
    ] * 2


def test_job_chunk_plan_fits_the_budget():
    """Pass 2's job programs go through the planner: the chunk halves until their compiled memory fits."""
    from relax.ppca_refinement.oversampled_stream import MAX_JOB_CHUNK, job_program_bytes, plan_job_chunk

    coarse, (fine_rotations, fine_translations) = _spa_stream()
    ostream = prepare_oversampled_stream(coarse, fine_rotations, fine_translations)
    assert plan_job_chunk(ostream, 4, memory_bytes=1e15, device_bytes=1e15) == MAX_JOB_CHUNK
    budget = job_program_bytes(ostream, 4, 16)
    planned = plan_job_chunk(ostream, 4, memory_bytes=budget, device_bytes=1)
    assert planned <= 16 and job_program_bytes(ostream, 4, planned) <= budget
    with pytest.raises(ValueError, match="pass-2 job"):
        plan_job_chunk(ostream, 4, memory_bytes=1, device_bytes=1)
