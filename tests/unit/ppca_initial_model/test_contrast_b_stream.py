"""Model B in the streamed engine and the controller: the plain engine at no spread, the dense reference at a pose, config."""

import dataclasses
import json
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.contrast_b_reference import reference_pose, unpack_gram
from helpers.float_compare import assert_matches
from helpers.full_row_tile import accumulate_full_row_tile
from test_full_fine_streaming import IMAGE_SHAPE, _half_volume, _TinyData

from relax.ppca_initial_model import checkpoint, iteration_loop
from relax.ppca_initial_model.config import Config
from relax.ppca_initial_model.state import State
from relax.ppca_refinement import contrast_b
from relax.ppca_refinement.config import GeometryConfig, ScheduleConfig, ScoringConfig
from relax.ppca_refinement.full_row_kernels import _posterior_block_contrast, _project_half, empty_carry, score_tile
from relax.ppca_refinement.full_row_stream import prepare_full_row_stream, read_tile
from relax.ppca_refinement.residual_statistics import full_float32

pytestmark = pytest.mark.unit

N_HALF = IMAGE_SHAPE[0] * (IMAGE_SHAPE[1] // 2 + 1)


def _single_pose_stream(q: int, contrast: contrast_b.ContrastPrior | None, images=None, **extra):
    """A tile of 3 images scored at one pose (one rotation, one translation) under the plain model ``[mu, W]``; ``images``
    default to noise."""
    rng = np.random.default_rng(3)
    noise_images = (rng.standard_normal((3, N_HALF)) + 1j * rng.standard_normal((3, N_HALF))).astype(np.complex64)
    mu = _half_volume(rng)
    loadings = np.stack([_half_volume(rng, 1.0) for _ in range(q)], axis=1)
    return prepare_full_row_stream(
        _TinyData(noise_images if images is None else images),
        mu,
        loadings,
        noise_variance=np.full(IMAGE_SHAPE, 40.0, np.float32),
        rotations=np.eye(3, dtype=np.float32)[None],
        translations=np.zeros((1, 2), np.float32),
        rotation_log_prior=np.zeros(1, np.float32),
        translation_log_prior=np.zeros(1, np.float32),
        rotation_parent=np.zeros(1, np.int32),
        translation_parent=np.zeros(1, np.int32),
        n_coarse_rotations=1,
        n_coarse_translations=1,
        geometry=GeometryConfig(current_size=6, q=q, volume_domain="fourier_half"),
        schedule=ScheduleConfig(image_batch_size=3, rotation_block_size=1),
        scoring=ScoringConfig(relion_texture_interp=False, full_real_observation=True),
        gemm_precision="fp32",
        contrast_b=contrast,
        **extra,
    )


def _model_images(q: int, contrasts=(0.8, 1.0, 1.3)) -> np.ndarray:
    """Images of the stream's own model at its pose, ``c (a + B z)`` plus noise of the stream's variance, one contrast per image."""
    reference = _single_pose_stream(q, None)
    rng = np.random.default_rng(7)
    projections = np.asarray(_project_half(reference.arrays, reference.arrays.rotations[:1], reference.static))[:, 0]
    latents = rng.standard_normal((len(contrasts), q))
    signal = projections[0][None] + latents @ projections[1:]
    noise = np.sqrt(40.0 / 2) * (rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape))
    return (np.asarray(contrasts)[:, None] * signal + noise).astype(np.complex64)


def test_the_integrated_contrast_without_spread_gives_the_statistics_of_the_plain_engine():
    images = _model_images(2)
    plain = accumulate_full_row_tile(_single_pose_stream(2, None, images), np.arange(3), [None] * 3)
    integrated = accumulate_full_row_tile(
        _single_pose_stream(2, contrast_b.ContrastPrior(3, 1.0, 1e-4, True), images), np.arange(3), [None] * 3
    )
    embeddings = np.asarray(integrated.embeddings)
    assert embeddings.shape == (3, 3)  # [E z (2), E c]
    assert_matches(embeddings[:, :2], np.asarray(plain.embeddings), rtol=1e-4)
    assert_matches(embeddings[:, 2], np.ones(3, np.float32), rtol=1e-4)
    assert_matches(np.asarray(integrated.lhs_tri), np.asarray(plain.lhs_tri), rtol=1e-4)
    assert_matches(np.asarray(integrated.residual_gradient), np.asarray(plain.residual_gradient), rtol=1e-4)
    assert_matches(np.float32(integrated.log_likelihood), np.float32(plain.log_likelihood), rtol=1e-4)
    assert_matches(np.asarray(integrated.diagnostics["latent_second_moment_sum"]), np.asarray(plain.diagnostics["latent_second_moment_sum"]), rtol=1e-4)


@full_float32
def _kept_and_moments(stream):
    """The pass-1 buffers of one tile of 3 images and the pass-2 moments of its only rotation block."""
    tile, observation_power, layout = read_tile(stream, np.arange(3), [None] * 3, collect_observation=True)
    kept, posterior = score_tile(stream, tile, layout["n_blocks"])
    carry = empty_carry(stream, 3, observation_power)
    start = stream.block_starts[0]
    weights, sums, carry = _posterior_block_contrast(
        carry, stream.arrays, kept, posterior, start, stream.rotation_block_size, stream.static
    )
    return kept, weights, sums, carry


def test_pass_two_recomputes_the_node_posteriors_the_dense_reference_gives():
    q, kappa, sigma = 2, 1.0, 0.3
    stream = _single_pose_stream(q, contrast_b.ContrastPrior(5, kappa, sigma, False), _model_images(q))
    kept, weights, sums, carry = _kept_and_moments(stream)
    inner = np.asarray(kept.latent_mean)[:, 0]  # (P, B, T) of the only row
    gram = np.transpose(np.asarray(kept.latent_covariance)[:1], (2, 0, 1))  # (tri(P), 1, B)
    index = {(i, j): k for k, (i, j) in enumerate(zip(*np.triu_indices(q + 1)))}
    for b in range(3):
        nu, h, H = unpack_gram(gram, q, 0, b)
        _, e_c, e_c2, e_cz, e_c2z, e_c2zz, e_z = reference_pose(inner[0, b, 0], inner[1:, b, 0].astype(np.float64), nu, h, H, kappa, sigma)
        assert_matches(np.asarray(weights)[0, 0, b, 0], np.float32(e_c), rtol=2e-3)  # gamma = 1 at the only pose
        assert_matches(np.asarray(weights)[1:, 0, b, 0], e_cz.astype(np.float32), rtol=2e-3)
        assert_matches(np.asarray(sums)[index[(0, 0)], 0, b], np.float32(e_c2), rtol=2e-3)
        assert_matches(np.asarray(sums)[index[(1, 2)], 0, b], np.float32(e_c2zz[0, 1]), rtol=3e-3)
        assert_matches(np.asarray(sums)[index[(0, 1)], 0, b], np.float32(e_c2z[0]), rtol=3e-3)
        assert_matches(np.asarray(carry.embedding)[b], np.concatenate([e_z, [e_c]]).astype(np.float32), rtol=2e-3)


def test_the_stream_refuses_two_contrast_models_and_a_degenerate_prior():
    with pytest.raises(ValueError, match="two models"):
        _single_pose_stream(1, contrast_b.ContrastPrior(3, 1.0, 0.3, True), latent_center=0.1)
    with pytest.raises(ValueError, match="positive prior mean"):
        _single_pose_stream(1, contrast_b.ContrastPrior(3, 1.0, 0.0, True))


def test_config_validates_the_quadrature_model():
    assert Config(contrast="quadrature").contrast_nodes == 3
    for bad in (
        dict(contrast="quadrature", contrast_nodes=0),
        dict(contrast="quadrature", oversampling=1),
        dict(contrast="quadrature", contrast_sd=0.0),
        dict(contrast="quadrature", stream_coarse_recompute=False),
    ):
        with pytest.raises(ValueError):
            Config(**bad)


def test_contrast_latents_of_both_models():
    embeddings = np.array([[0.4, -0.2, 0.5], [1.0, 2.0, -1.0]], np.float32)
    z, scaled, estimate, n_floored = iteration_loop.contrast_latents(Config(contrast="quadrature"), 0.3, embeddings)
    assert_matches(z, embeddings[:, :2])
    assert_matches(estimate, embeddings[:, 2])
    assert_matches(scaled, embeddings[:, :2] * embeddings[:, 2:3])
    assert n_floored == 0
    z, scaled, estimate, n_floored = iteration_loop.contrast_latents(Config(contrast="latent"), 0.3, embeddings)
    assert_matches(estimate, 1 + 0.3 * embeddings[:, 2])
    assert_matches(scaled, embeddings[:, :2])
    assert n_floored == 0


def test_the_prior_of_the_integrated_contrast_follows_the_gain_and_the_sd():
    config = Config(contrast="quadrature", contrast_nodes=4, contrast_start=2)
    state = State(jnp.ones((5, 3), jnp.complex64), None, jnp.ones(3, jnp.float32), 3, np.arange(8), {}, 2.0, 4, {}, contrast_sd=0.25, contrast_gain=0.1)
    assert iteration_loop.contrast_prior_for(config, state, 1) is None  # before contrast_start
    prior = iteration_loop.contrast_prior_for(config, state, 2)
    assert prior.nodes == 4 and prior.normalise
    assert prior.mean == pytest.approx(1.1) and prior.sd == pytest.approx(1.1 * 0.25)
    assert iteration_loop.contrast_prior_for(dataclasses.replace(config, contrast="latent"), state, 2) is None
    assert iteration_loop.contrast_prior_for(dataclasses.replace(config, contrast_center=False), state, 2).normalise is False


def test_the_controller_follows_the_mean_and_relative_spread_of_the_contrast(tmp_path, monkeypatch):
    shape = (4, 4, 4)
    n_freq = shape[0] * shape[1] * (shape[2] // 2 + 1)
    monkeypatch.setattr(iteration_loop, "support_mask", lambda *_args: jnp.ones(shape, jnp.float32))
    monkeypatch.setattr(iteration_loop, "bandlimit_and_mask", lambda value, *_args: value)
    monkeypatch.setattr(iteration_loop.grids, "get_relion_hidden_rotation_grid", lambda *_a, **_k: np.zeros((1, 3)))
    monkeypatch.setattr(iteration_loop.grids, "rotation_direction_ids", lambda *_a, **_k: np.zeros(1, np.int64))
    mean, second_moment = 1.2, 1.2**2 + 0.09  # the batch's E[c] and E[c^2]: a relative spread of 0.25
    priors = []

    def groups(_dataset, state, config, halves, iteration, **_kwargs):
        priors.append(iteration_loop.contrast_prior_for(config, state, iteration))
        return [
            SimpleNamespace(
                residual_gradient=jnp.zeros((n_freq, 3), jnp.complex64),
                lhs_tri=None,
                rhs=None,
                residual_num=jnp.ones(3, jnp.float32),
                residual_den=jnp.ones(3, jnp.float32),
                original_image_ids=np.asarray(ids),
                embeddings=jnp.zeros((len(ids), 3)).at[:, -1].set(mean),
                n_images=len(ids),
                log_likelihood=0.0,
                diagnostics={
                    "offset_second_sum_px2": 2.0,
                    "rotation_mass": np.array([float(len(ids))]),
                    "latent_covariance_trace_mean": 1.0,
                    "pose_entropy_mean": 0.0,
                    "pmax_mean": 1.0,
                    "latent_second_moment_sum": np.array([second_moment * len(ids), 0.0, 0.0]),
                },
            )
            for ids in halves
        ]

    monkeypatch.setattr(iteration_loop, "expectation_groups", groups)
    monkeypatch.setattr(iteration_loop, "_curvature", lambda stats, p: len(stats) * jnp.ones((n_freq, 2), jnp.float32))
    dataset = SimpleNamespace(n_images=40, grid_size=4, volume_shape=shape, voxel_size=1.0)
    config = Config(
        iterations=3, stages=((1, 1, 0),), optimizer="momentum_sgd", contrast="quadrature", contrast_sd=0.2,
        stochastic_batch_size=8, stochastic_all_iterations=True, skip_final_embeddings=True, gemm_precision="fp32",
    )
    theta = jnp.zeros((n_freq, 3), jnp.complex64)
    state = State(
        theta, None, jnp.ones(3, jnp.float32), 0, np.arange(40), np.random.default_rng(1).bit_generator.state, 2.0, 1,
        {"seed": 1}, sgd_momentum=jnp.zeros_like(theta), contrast_sd=0.2,
    )
    identity = {"fixture": "tiny"}
    checkpoint.save(tmp_path / "checkpoint_0000.npz", state, config, identity)
    iteration_loop.run(dataset, config, tmp_path, identity, diameter_ang=2.0, resume=tmp_path / "checkpoint_0000.npz")
    records = [json.loads(line) for line in (tmp_path / "iterations.jsonl").read_text().splitlines()]
    sd, gain = 0.2, 0.0
    for record in records:
        assert record["contrast"]["sd"] == pytest.approx(sd)
        assert record["contrast"]["gain"] == pytest.approx(gain)
        assert record["contrast"]["estimate_mean"] == pytest.approx(mean)
        relative = (second_moment - mean**2) / mean**2
        sd = float(np.sqrt(0.9 * sd**2 + 0.1 * relative))  # beta = 0.9 on a stochastic batch
        gain = 0.8 * gain + 0.2 * (mean - 1.0)
        assert record["contrast"]["next_sd"] == pytest.approx(sd)
    assert priors[0].mean == pytest.approx(1.0) and priors[1].mean == pytest.approx(1.0 + 0.2 * (mean - 1.0))
