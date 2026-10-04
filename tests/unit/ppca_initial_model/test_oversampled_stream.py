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


def test_padding_images_have_no_significant_samples():
    score = jnp.zeros((4, 3, 2), jnp.float32)
    rows = jnp.arange(4, dtype=jnp.int32)
    posterior = _normalize(score, rows, n_blocks=1, block_size=4)
    _, _, valid, mass, capped = _significant_samples(
        score, rows, posterior, jnp.float32(0.5), np.int32(2), n_significant=8, every=False
    )
    assert np.asarray(valid)[:2].any(axis=1).all() and not np.asarray(valid)[2:].any()
    assert not np.asarray(mass)[2:].any() and not np.asarray(capped)[2:].any()


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


def test_pass1_on_the_coarse_window_equals_the_dense_child_grid_at_the_pass2_window():
    """RELION's coarse image size: pass 1 on a smaller window selects the samples, pass 2 scores at the full window."""
    coarse, (fine_rotations, fine_translations) = _spa_stream(current_size=4)
    pass2, _ = _spa_stream(current_size=6)
    ostream = prepare_oversampled_stream(
        coarse, fine_rotations, fine_translations, pass2=pass2, adaptive_fraction=0.9, max_significant=40, job_chunk=16
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
    assert any("consider a larger --maxsig" in r.message for r in caplog.records) == warned
    assert np.all(np.isfinite(np.asarray(state.theta)))
    assert np.load(tmp_path / "embeddings.npz")["z"].shape == (8, 2)


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
