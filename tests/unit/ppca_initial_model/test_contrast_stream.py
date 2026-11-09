"""The contrast latent in the streamed engine and the controller: second-moment diagnostics, sd update, checkpoint, config."""

import dataclasses
import json
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches
from helpers.full_row_tile import accumulate_full_row_tile
from test_full_fine_streaming import (
    IMAGE_SHAPE,
    SIGNIFICANT,
    VOLUME_SHAPE,
    _half_volume,
    _TinyData,
    make_tile_problem,
)

from relax.ppca_initial_model import checkpoint, iteration_loop
from relax.ppca_initial_model.config import Config
from relax.ppca_initial_model.state import State
from relax.ppca_refinement import contrast
from relax.ppca_refinement.config import GeometryConfig, ScheduleConfig, ScoringConfig, SparsePass2Config
from relax.ppca_refinement.dense_dataset import accumulate_dense_ppca_statistics
from relax.ppca_refinement.full_row_stream import prepare_full_row_stream
from relax.ppca_refinement.residual_statistics import full_float32

pytestmark = pytest.mark.unit

N_HALF = IMAGE_SHAPE[0] * (IMAGE_SHAPE[1] // 2 + 1)


def _single_pose_stream(q: int, sd: float | None):
    """A tile of 3 images scored at one pose (one rotation, one translation); the model is ``[mu, W]`` or, with ``sd``,
    ``[mu, W, sd mu]``. Returns the dataset and the stream."""
    rng = np.random.default_rng(3)
    images = (rng.standard_normal((3, N_HALF)) + 1j * rng.standard_normal((3, N_HALF))).astype(np.complex64)
    mu = _half_volume(rng)
    loadings = np.stack([_half_volume(rng, 0.3) for _ in range(q)], axis=1) if q else np.zeros((mu.size, 0), np.complex64)
    columns = loadings if sd is None else np.asarray(contrast.with_contrast_column(mu, loadings, sd))[:, 1:]
    dataset = _TinyData(images)
    stream = prepare_full_row_stream(
        dataset,
        mu,
        columns,
        noise_variance=np.full(IMAGE_SHAPE, 40.0, np.float32),
        rotations=np.eye(3, dtype=np.float32)[None],
        translations=np.zeros((1, 2), np.float32),
        rotation_log_prior=np.zeros(1, np.float32),
        translation_log_prior=np.zeros(1, np.float32),
        rotation_parent=np.zeros(1, np.int32),
        translation_parent=np.zeros(1, np.int32),
        n_coarse_rotations=1,
        n_coarse_translations=1,
        geometry=GeometryConfig(current_size=6, q=columns.shape[1], volume_domain="fourier_half"),
        schedule=ScheduleConfig(image_batch_size=3, rotation_block_size=1),
        scoring=ScoringConfig(relion_texture_interp=False, full_real_observation=True),
        gemm_precision="fp32",
    )
    return dataset, stream


@pytest.mark.parametrize("sd", [None, 0.3])
def test_stream_reports_the_second_moment_of_every_latent(sd):
    """At a single pose the posterior is Gaussian: the summed second moments are |mean|^2 plus the covariance trace."""
    _, stream = _single_pose_stream(q=2, sd=sd)
    stats = accumulate_full_row_tile(stream, np.arange(3), [None] * 3)
    second = np.asarray(stats.diagnostics["latent_second_moment_sum"])
    latents = stats.embeddings.shape[1]
    assert second.shape == (latents + 1,)
    assert_matches(second[0], np.float32(3.0))  # the posterior mass: one per image
    embeddings = np.asarray(stats.embeddings, np.float64)
    trace = stats.diagnostics["latent_covariance_trace_mean"] * 3
    assert_matches(np.float32(second[1:].sum()), np.float32(np.sum(embeddings**2) + trace))
    assert np.all(second[1:] >= np.sum(embeddings**2, axis=0) - 1e-5)  # E[x^2] >= E[x]^2 per latent


def test_folding_the_tied_stream_leaves_a_model_of_the_untied_shape():
    """The statistics of [mu, W, sd mu] fold to those of [mu, W]; the contrast latent stays the last embedding."""
    _, stream = _single_pose_stream(q=2, sd=0.3)
    stats = accumulate_full_row_tile(stream, np.arange(3), [None] * 3)
    folded = contrast.fold_statistics(stats, 0.3)
    assert stats.lhs_tri.shape[1] == 10 and folded.lhs_tri.shape[1] == 6
    assert stats.residual_gradient.shape[1] == 4 and folded.residual_gradient.shape[1] == 3
    u, estimate = contrast.split_embeddings(folded.embeddings, 0.3)
    assert u.shape == (3, 2) and estimate.shape == (3,)
    assert_matches(np.asarray(estimate), 1 + 0.3 * np.asarray(stats.embeddings)[:, -1])


def test_config_validates_the_contrast_model():
    assert Config().contrast == "none"
    assert Config(contrast="latent").contrast_sd == 0.2
    for bad in (
        dict(contrast="grid"),
        dict(contrast="latent", contrast_sd=0.0),
        dict(contrast="latent", contrast_sd=0.9),
        dict(contrast="latent", contrast_start=0),
        dict(contrast="latent", oversampling=1),
        dict(contrast="latent", stream_coarse_recompute=False),
    ):
        with pytest.raises(ValueError):
            Config(**bad)


def test_checkpoint_identity_ignores_default_contrast_settings_and_keeps_the_sd(tmp_path):
    theta = jnp.ones((5, 3), jnp.complex64)
    state = State(theta, None, jnp.ones(3, jnp.float32), 3, np.arange(8), {}, 2.0, 4, {}, sgd_momentum=0 * theta)
    config = Config(optimizer="momentum_sgd")
    path = tmp_path / "plain.npz"
    checkpoint.save(path, state, config, {"input": "abc"})
    with np.load(path, allow_pickle=False) as arrays:
        arrays = dict(arrays)
    metadata = json.loads(str(arrays["metadata"]))
    for key in ("contrast", "contrast_sd", "contrast_estimate_sd", "contrast_start", "contrast_sd_min", "contrast_sd_max"):
        metadata["config"].pop(key)  # a checkpoint written before the contrast settings existed
    arrays["metadata"] = json.dumps(metadata)
    old = tmp_path / "old.npz"
    np.savez(old, **arrays)
    restored = checkpoint.load(old, config, {"input": "abc"})
    assert restored.contrast_sd is None
    latent = dataclasses.replace(config, contrast="latent")
    with pytest.raises(ValueError, match="identity mismatch"):
        checkpoint.load(old, latent, {"input": "abc"})
    on = dataclasses.replace(state, contrast_sd=0.25)
    checkpoint.save(tmp_path / "on.npz", on, latent, {"input": "abc"})
    assert checkpoint.load(tmp_path / "on.npz", latent, {"input": "abc"}).contrast_sd == 0.25


def test_the_controller_updates_the_sd_from_the_second_moment_and_logs_it(tmp_path, monkeypatch):
    shape = (4, 4, 4)
    n_freq = shape[0] * shape[1] * (shape[2] // 2 + 1)
    monkeypatch.setattr(iteration_loop, "support_mask", lambda *_args: jnp.ones(shape, jnp.float32))
    monkeypatch.setattr(iteration_loop, "bandlimit_and_mask", lambda value, *_args: value)
    monkeypatch.setattr(iteration_loop.grids, "get_relion_hidden_rotation_grid", lambda *_a, **_k: np.zeros((1, 3)))
    monkeypatch.setattr(iteration_loop.grids, "rotation_direction_ids", lambda *_a, **_k: np.zeros(1, np.int64))
    second_moment = 1.5  # mean E[e^2] over the batch the stand-in engine reports
    gain = 0.5  # the contrast latent's posterior mean of every image: the global gain the sd must not count as spread
    seen = []

    def groups(_dataset, state, _config, halves, iteration, **_kwargs):
        seen.append((iteration, state.contrast_sd))
        return [
            SimpleNamespace(
                residual_gradient=jnp.zeros((n_freq, 3), jnp.complex64),
                lhs_tri=None,
                rhs=None,
                residual_num=jnp.ones(3, jnp.float32),
                residual_den=jnp.ones(3, jnp.float32),
                original_image_ids=np.asarray(ids),
                embeddings=jnp.zeros((len(ids), 3)).at[:, -1].set(gain),
                n_images=len(ids),
                log_likelihood=0.0,
                diagnostics={
                    "offset_second_sum_px2": 2.0,
                    "rotation_mass": np.array([float(len(ids))]),
                    "latent_covariance_trace_mean": 1.0,
                    "pose_entropy_mean": 0.0,
                    "pmax_mean": 1.0,
                    "latent_second_moment_sum": np.array([len(ids), 0.0, 0.0, second_moment * len(ids)]),
                },
            )
            for ids in halves
        ]

    monkeypatch.setattr(iteration_loop, "expectation_groups", groups)
    monkeypatch.setattr(iteration_loop, "_curvature", lambda stats, p: len(stats) * jnp.ones((n_freq, 2), jnp.float32))
    dataset = SimpleNamespace(n_images=40, grid_size=4, volume_shape=shape, voxel_size=1.0)
    config = Config(
        iterations=3, stages=((1, 1, 0),), optimizer="momentum_sgd", contrast="latent", contrast_sd=0.2,
        contrast_start=2, stochastic_batch_size=8, stochastic_all_iterations=True, skip_final_embeddings=True,
        gemm_precision="fp32",
    )
    theta = jnp.zeros((n_freq, 3), jnp.complex64)
    state = State(
        theta, None, jnp.ones(3, jnp.float32), 0, np.arange(40), np.random.default_rng(1).bit_generator.state, 2.0, 1,
        {"seed": 1}, sgd_momentum=jnp.zeros_like(theta), contrast_sd=0.2,
    )
    identity = {"fixture": "tiny"}
    checkpoint.save(tmp_path / "checkpoint_0000.npz", state, config, identity)
    final = iteration_loop.run(dataset, config, tmp_path, identity, diameter_ang=2.0, resume=tmp_path / "checkpoint_0000.npz")
    records = [json.loads(line) for line in (tmp_path / "iterations.jsonl").read_text().splitlines()]
    assert [r["iteration"] for r in records] == [1, 2, 3]
    assert "contrast" not in records[0]  # before contrast_start the E-step has unit contrast
    assert [s for _, s in seen][0] == 0.2
    expected = 0.2
    for record in records[1:]:
        assert record["contrast"]["sd"] == pytest.approx(expected)
        centred = second_moment - gain**2
        blended = 0.9 * expected**2 + 0.1 * expected**2 * centred  # beta = 0.9 on a stochastic batch
        expected = float(np.sqrt(blended))
        assert record["contrast"]["next_sd"] == pytest.approx(expected)
        assert record["contrast"]["latent_second_moment"] == pytest.approx(second_moment)
        assert record["contrast"]["latent_mean"] == pytest.approx(gain)
        assert record["contrast"]["estimate_sd"] == pytest.approx(0.0, abs=1e-7)  # one gain for every image
    assert final.contrast_sd == pytest.approx(expected)
    assert checkpoint.load(tmp_path / "checkpoint_0003.npz", config, identity).contrast_sd == pytest.approx(expected)


def test_dense_and_streamed_engines_agree_on_the_second_moments_and_the_folded_statistics():
    """The host-mask dense engine reports the same per-latent second moments as the stream, and folds the same way."""
    dataset, mu, W, stream, host = make_tile_problem(q=3)
    dense = full_float32(accumulate_dense_ppca_statistics)(
        dataset, mu, W, sparse_pass2=SparsePass2Config(enabled=False), collect_residuals=True, **host
    )
    streamed = accumulate_full_row_tile(stream, np.arange(3), SIGNIFICANT)
    assert_matches(
        np.asarray(streamed.diagnostics["latent_second_moment_sum"]),
        np.asarray(dense.diagnostics["latent_second_moment_sum"]),
    )
    assert_matches(np.asarray(dense.diagnostics["latent_second_moment_sum"])[0], np.float32(3.0))
    sd = 0.3
    folded_dense, folded_streamed = contrast.fold_statistics(dense, sd), contrast.fold_statistics(streamed, sd)
    for name in ("lhs_tri", "residual_gradient", "embeddings"):
        assert_matches(np.asarray(getattr(folded_streamed, name)), np.asarray(getattr(folded_dense, name)))
    assert folded_dense.lhs_tri.shape[1] == 6 and folded_dense.residual_gradient.shape[1] == 3  # [mu, W_1, W_2]


def test_the_dense_helper_runs_the_engine_on_the_tied_model_and_folds_it():
    dataset, mu, W, _stream, host = make_tile_problem(q=2)
    options = dict(sparse_pass2=SparsePass2Config(enabled=False), collect_residuals=True, **host)
    folded = full_float32(contrast.accumulate_dense_with_contrast)(dataset, mu, W, 0.3, **options)
    tied_options = {**options, "geometry": dataclasses.replace(host["geometry"], q=3)}
    tied = full_float32(accumulate_dense_ppca_statistics)(
        dataset, mu, np.asarray(contrast.with_contrast_column(mu, W, 0.3))[:, 1:], **tied_options
    )
    expected = contrast.fold_statistics(tied, 0.3)
    for name in ("lhs_tri", "residual_gradient", "embeddings"):
        assert_matches(np.asarray(getattr(folded, name)), np.asarray(getattr(expected, name)))
    assert folded.lhs_tri.shape[1] == 6 and folded.embeddings.shape[1] == 3  # [mu, W_1, W_2] and the contrast latent
