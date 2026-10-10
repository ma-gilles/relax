"""Joint class/pose PPCA contracts, including independent multi-tilt references.

The tomography reference evaluates full observation-space Gaussian covariances
in float64, separately for each physical particle and component. Combining
their evidences with SciPy logsumexp supplies an independent class posterior;
the production engine always scores and accumulates in float32.
"""

import argparse
import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches
from scipy.special import logsumexp
from test_full_fine_streaming import _TinyData
from test_tomo_ppca import IMAGE_SHAPE, NOISE, _particles, brute_force, make_problem

from relax.commands.ppca_initial_model import add_args
from relax.ppca_initial_class3d import checkpoint
from relax.ppca_initial_class3d.config import Config
from relax.ppca_initial_class3d.state import State
from relax.ppca_initial_class3d.stream import (
    _mixture_normalization,
    accumulate_mixture_full_row_tile,
    prepare_mixture_full_row_stream,
)
from relax.ppca_initial_model.tomo import load_tilt_tile
from relax.ppca_initial_model.update import empty_moments
from relax.ppca_refinement.config import GeometryConfig, ScheduleConfig, ScoringConfig
from relax.ppca_refinement.full_row_stream import (
    accumulate_full_row_tile,
    prepare_full_row_stream,
)

pytestmark = pytest.mark.unit


def _stream(dataset, problem, *, tomography):
    """Prepare the actual public SPA or TOMO stream with a small full pose grid."""
    rotations = problem["rotations"]
    translations = problem["translations"]
    return prepare_full_row_stream(
        dataset, problem["mu"], problem["W"],
        noise_variance=np.full(IMAGE_SHAPE, NOISE, np.float32),
        rotations=rotations, translations=translations,
        rotation_log_prior=problem["rotation_log_prior"],
        translation_log_prior=problem["translation_log_prior"],
        rotation_parent=np.zeros(len(rotations), np.int32),
        translation_parent=np.zeros(len(translations), np.int32),
        n_coarse_rotations=1, n_coarse_translations=1,
        geometry=GeometryConfig(current_size=6, q=problem["W"].shape[1], volume_domain="fourier_half"),
        schedule=ScheduleConfig(image_batch_size=dataset.n_images, rotation_block_size=2),
        scoring=ScoringConfig(relion_texture_interp=False, full_real_observation=True),
        tile_loader=load_tilt_tile if tomography else None,
        gemm_precision="fp32",
    )


@pytest.fixture(scope="module")
def tomo_case():
    translations = np.asarray([[0, 0, 0], [0.7, 0, 0]], np.float32)
    image_frames = [[0, 1], [1], [1, 0]]
    particles, _, problem = make_problem(
        q=2, image_frames=image_frames, translations=translations,
    )
    problem.update(
        translations=translations,
        rotations=problem["rotations"][:3],
        rotation_log_prior=np.log(np.asarray([0.2, 0.3, 0.5], np.float32)),
    )
    first = _stream(particles, problem, tomography=True)
    other = {
        **problem,
        "mu": np.asarray(problem["mu"] * np.float32(0.98) + problem["W"][:, 0] * np.float32(0.025)),
        "W": np.asarray(problem["W"] * np.float32(1.15)),
    }
    second = _stream(particles, other, tomography=True)
    return particles, (first, second), (problem, other), image_frames


@pytest.fixture(scope="module")
def tomo_reference(tomo_case):
    particles, streams, problems, image_frames = tomo_case
    references = []
    for i in range(particles.n_images):
        image_ids, _ = particles.particle_images([i])
        row = []
        for stream, problem in zip(streams, problems, strict=True):
            one_problem = {
                **problem, "images": problem["images"][image_ids], "ctf": problem["ctf"][image_ids],
            }
            one = _particles(
                one_problem["images"], one_problem["ctf"], [image_frames[i]], particles.group_frames[0],
            )
            row.append(brute_force(one, stream, one_problem))
        references.append(row)
    prior = np.asarray([0.35, 0.65], np.float64)
    scores = np.asarray([[entry["log_likelihood"] for entry in row] for row in references]) + np.log(prior)
    evidence = logsumexp(scores, axis=1)
    responsibility = np.exp(scores - evidence[:, None])
    return references, prior, evidence, responsibility


def test_tomo_distinct_classes_match_independent_joint_gaussians(tomo_case, tomo_reference):
    particles, streams, _, _ = tomo_case
    refs, prior, evidence, responsibility = tomo_reference
    ids = np.arange(particles.n_images)
    result = accumulate_mixture_full_row_tile(prepare_mixture_full_row_stream(streams, prior), ids, [None] * len(ids))
    # Same production-float32/reference-float64 band as the inherited TOMO
    # reference suite: measured worst 4e-6 over seeds 5--7, with a 5x margin.
    tol = 2e-5
    assert_matches(result.class_probabilities, responsibility, rtol=tol)
    assert_matches(result.class_probabilities.sum(axis=1), np.ones(len(ids), np.float32))
    assert_matches(result.component_mass, responsibility.sum(axis=0), rtol=tol)
    assert_matches(result.log_likelihood, evidence.sum(), rtol=tol)
    assert np.ptp(responsibility[:, 0]) > 1e-4  # genuine data-dependent soft assignments
    assert_matches(result.embeddings, np.stack([
        np.stack([entry["embeddings"][0] for entry in row]) for row in refs
    ]), rtol=tol)
    for k, stats in enumerate(result.statistics):
        for field in ("lhs_tri", "residual_gradient"):
            expected = sum(responsibility[i, k] * refs[i][k][field] for i in range(len(ids)))
            assert_matches(getattr(stats, field), expected, rtol=tol)
        assert_matches(stats.diagnostics["rotation_mass"], sum(
            responsibility[i, k] * refs[i][k]["rotation_mass"] for i in range(len(ids))
        ), rtol=tol)
        assert_matches(stats.embeddings, np.stack([
            responsibility[i, k] * refs[i][k]["embeddings"][0] for i in range(len(ids))
        ]), rtol=tol)
        assert_matches(stats.residual_den, np.zeros_like(stats.residual_den))
    # Each reference includes observation power. Class weighting must count
    # it once, and the count is five visible images, not three particles or K*5.
    expected_num = sum(
        responsibility[i, k] * refs[i][k]["residual_num"]
        for i in range(len(ids)) for k in range(len(streams))
    )
    assert_matches(result.residual_num, expected_num, rtol=tol)
    assert_matches(result.residual_den, sum(row[0]["residual_den"] for row in refs))
    np.testing.assert_array_equal(result.original_image_ids, ids)
    assert np.asarray(result.embeddings).dtype == np.float32
    assert all(np.asarray(s.residual_gradient).dtype == np.complex64 for s in result.statistics)


@pytest.mark.gpu
def test_native_gpu_tomo_class_pose_moments_match_reference(tomo_case, tomo_reference):
    """Exercise real fused kernels (K=2, q=2), not a CPU fallback."""
    import jax

    assert jax.default_backend() == "gpu"
    assert all(stream.static.cuda_kernels for stream in tomo_case[1])
    test_tomo_distinct_classes_match_independent_joint_gaussians(tomo_case, tomo_reference)


@pytest.mark.parametrize("tomography", [False, True], ids=["spa", "tomo"])
def test_one_class_and_identical_component_split_preserve_single_ppca(tomo_case, tomography):
    particles, streams, problems, _ = tomo_case
    if tomography:
        stream = streams[0]
        ids = np.arange(particles.n_images)
    else:
        problem = {**problems[0], "translations": problems[0]["translations"][:, :2]}
        dataset = _TinyData(problem["images"][:3])
        stream = _stream(dataset, problem, tomography=False)
        ids = np.arange(dataset.n_images)
    support = [None] * len(ids)
    single = accumulate_full_row_tile(stream, ids, support)
    one = accumulate_mixture_full_row_tile(prepare_mixture_full_row_stream([stream], [1]), ids, support)
    two = accumulate_mixture_full_row_tile(prepare_mixture_full_row_stream([stream, stream], [.2, .8]), ids, support)
    assert_matches(one.class_probabilities, np.ones((len(ids), 1), np.float32))
    assert_matches(two.class_probabilities, np.tile(np.asarray([.2, .8], np.float32), (len(ids), 1)))
    for field in ("lhs_tri", "residual_gradient"):
        assert_matches(getattr(one.statistics[0], field), getattr(single, field))
        assert_matches(sum(getattr(s, field) for s in two.statistics), getattr(single, field))
    for result in (one, two):
        assert_matches(result.residual_num, single.residual_num)
        assert_matches(result.residual_den, single.residual_den)
        assert_matches(result.log_likelihood, single.log_likelihood, rtol=1e-6)
        expected_z = np.broadcast_to(np.asarray(single.embeddings)[:, None, :], result.embeddings.shape)
        assert_matches(result.embeddings, expected_z)
        np.testing.assert_array_equal(result.original_image_ids, single.original_image_ids)
    np.testing.assert_array_equal(two.best_class, np.ones(len(ids), np.int32))


def test_class_permutation_and_inference_only_preserve_particle_outputs(tomo_case):
    particles, streams, _, _ = tomo_case
    ids = np.asarray([2, 0, 1])
    prior = np.asarray([.35, .65], np.float32)
    result = accumulate_mixture_full_row_tile(prepare_mixture_full_row_stream(streams, prior), ids, [None] * len(ids))
    swapped = accumulate_mixture_full_row_tile(
        prepare_mixture_full_row_stream(streams[::-1], prior[::-1]), ids, [None] * len(ids), moments=False,
    )
    assert swapped.statistics == () and swapped.residual_num is None and swapped.residual_den is None
    assert_matches(swapped.class_probabilities[:, ::-1], result.class_probabilities)
    assert_matches(swapped.embeddings[:, ::-1], result.embeddings)
    assert_matches(swapped.max_posterior_per_image, result.max_posterior_per_image)
    assert_matches(swapped.log_likelihood, result.log_likelihood, rtol=1e-6)
    np.testing.assert_array_equal(swapped.best_rotation_idx[:, ::-1], result.best_rotation_idx)
    np.testing.assert_array_equal(swapped.best_translation_idx[:, ::-1], result.best_translation_idx)
    np.testing.assert_array_equal(1 - swapped.best_class, result.best_class)
    np.testing.assert_array_equal(swapped.original_image_ids, ids)


def test_padding_particles_have_no_mixture_mass_or_noise(tomo_case):
    _, streams, _, _ = tomo_case
    ids = np.asarray([1, 0, 2])
    padded = tuple(s._replace(tile_images=4) for s in streams)
    prior = [.35, .65]
    result = accumulate_mixture_full_row_tile(prepare_mixture_full_row_stream(streams, prior), ids, [None] * 3)
    actual = accumulate_mixture_full_row_tile(prepare_mixture_full_row_stream(padded, prior), ids, [None] * 3)
    assert actual.class_probabilities.shape == (3, 2)
    assert_matches(actual.component_mass.sum(), np.float32(3))
    for field in ("class_probabilities", "embeddings", "residual_num", "residual_den"):
        assert_matches(getattr(actual, field), getattr(result, field))
    for a, b in zip(actual.statistics, result.statistics, strict=True):
        assert_matches(a.lhs_tri, b.lhs_tri)
        assert_matches(a.residual_gradient, b.residual_gradient)


def test_rank_ten_components_have_class_conditional_ten_dimensional_embeddings():
    translations = np.zeros((1, 3), np.float32)
    particles, _, problem = make_problem(
        q=10, image_frames=[[0]], frames=np.eye(3)[None], translations=translations,
    )
    problem.update(
        translations=translations, rotations=problem["rotations"][:1],
        rotation_log_prior=np.zeros(1, np.float32),
    )
    stream = _stream(particles, problem, tomography=True)
    result = accumulate_mixture_full_row_tile(
        prepare_mixture_full_row_stream([stream, stream], [.4, .6]), [0], [None], moments=False,
    )
    assert result.embeddings.shape == (1, 2, 10)
    assert np.all(np.isfinite(np.asarray(result.embeddings)))
    assert_matches(result.class_probabilities, np.asarray([[.4, .6]], np.float32))
    assert_matches(result.embeddings[:, 0], result.embeddings[:, 1])


def test_inference_retains_conditional_embeddings_when_class_mass_underflows(tomo_case):
    _, streams, _, _ = tomo_case
    stream = streams[0]
    ids = np.arange(3)
    smallest = np.nextafter(np.float32(0), np.float32(1))
    result = accumulate_mixture_full_row_tile(
        prepare_mixture_full_row_stream([stream, stream], [smallest, 1]), ids, [None] * 3, moments=False,
    )
    single = accumulate_full_row_tile(stream, ids, [None] * 3)
    assert np.all(result.class_probabilities[:, 0] < np.finfo(np.float32).tiny)
    assert_matches(result.embeddings[:, 0], single.embeddings)
    assert_matches(result.embeddings[:, 1], single.embeddings)


def test_joint_normalization_preserves_prior_odds_with_large_common_scores():
    # Adding log(.2/.8) to 1e8 first would round both priors away in float32.
    centers = np.asarray([[1e8, 1e8], [1e8, 1e8]], np.float32)
    partitions = np.asarray([[0, 0], [np.log(100), 0]], np.float32)
    prior = np.asarray([.2, .8], np.float32)
    _, log_z, probability, pass2, best_class, pmax = _mixture_normalization(
        jnp.asarray(centers), jnp.asarray(partitions), jnp.log(jnp.asarray(prior)),
    )
    evidence = logsumexp(partitions.astype(np.float64) + np.log(prior.astype(np.float64)), axis=1)
    expected = np.exp(partitions + np.log(prior) - evidence[:, None])
    assert_matches(probability, expected)
    assert_matches(log_z, evidence)
    assert_matches(np.exp(partitions - np.asarray(pass2)), expected)
    # Marginal class MAP and the class of the joint MAP pose are different.
    assert int(np.argmax(np.asarray(probability)[1])) == 0
    np.testing.assert_array_equal(best_class, [1, 1])
    assert_matches(pmax, np.exp(np.log(prior[1]) - evidence))


@pytest.mark.parametrize("prior", [[0, 1], [-1, 2], [np.nan, 1], [np.inf, 1], [1]])
def test_mixture_refuses_invalid_component_priors(tomo_case, prior):
    with pytest.raises(ValueError, match="positive prior"):
        prepare_mixture_full_row_stream(tomo_case[1], prior)


def test_mixture_refuses_class_specific_observation_noise(tomo_case):
    _, streams, _, _ = tomo_case
    different = streams[1]._replace(noise_variance_half=streams[1].noise_variance_half * 2)
    with pytest.raises(ValueError, match="observation noise"):
        prepare_mixture_full_row_stream([streams[0], different], [.5, .5])


@pytest.mark.parametrize("change", [
    {"n_classes": 0}, {"n_classes": True}, {"n_classes": 1.5}, {"q": 17},
    {"class_pseudocount": 0}, {"class_pseudocount": np.nan},
    {"oversampling": 2}, {"oversampling": 0, "oversampling_start": 5}, {"optimizer": "momentum_sgd"},
])
def test_mixture_configuration_rejects_unsupported_or_invalid_models(change):
    with pytest.raises(ValueError):
        Config(**change)


def test_command_has_separate_class_and_pc_counts_and_tomography_input():
    parser = argparse.ArgumentParser()
    add_args(parser)
    tomo = parser.parse_args(["--ios", "optimisation_set.star", "--K", "10", "--q", "10", "-o", "out"])
    spa = parser.parse_args(["training.json", "--K", "3", "--q", "2", "-o", "out"])
    assert (tomo.n_classes, tomo.q, tomo.ios, tomo.manifest) == (10, 10, "optimisation_set.star", None)
    assert (spa.n_classes, spa.q, spa.ios, spa.manifest) == (3, 2, None, "training.json")
    default = parser.parse_args(["training.json", "-o", "out"])
    assert default.n_classes == 1 and default.q == 2


def test_mixture_checkpoint_preserves_all_components_and_rejects_changed_identity(tmp_path):
    config = Config(n_classes=2, q=2)
    theta = jnp.arange(2 * 5 * 3, dtype=jnp.float32).reshape(2, 5, 3).astype(jnp.complex64) * (1 + 1j)
    histories = tuple(dataclasses.replace(
        empty_moments(model), first=jnp.stack([model, -model]),
    ) for model in theta)
    state = State(
        theta, histories, jnp.ones((2, 3), jnp.float32), np.asarray([.3, .7], np.float32),
        7, np.asarray([3, 2, 0, 1]), np.random.default_rng(11).bit_generator.state,
        2.0, 4, {"components": [{"seed": 11}, {"seed": 104740}]},
        np.ones(12, np.float32) / 12, 0,
    )
    path = tmp_path / "mixture.npz"
    identity = {"particles": "immutable-hash", "source": {"head": "draft"}}
    checkpoint.save(path, state, config, identity)
    actual = checkpoint.load(path, config, {**identity, "source": {"head": "updated"}})
    for field in ("theta", "noise", "class_prior", "direction_prior"):
        assert_matches(getattr(actual, field), getattr(state, field))
    for a, b in zip(actual.moments, state.moments, strict=True):
        assert_matches(a.first, b.first)
        assert_matches(a.second, b.second)
        np.testing.assert_array_equal(a.initialized, b.initialized)
    np.testing.assert_array_equal(actual.order, state.order)
    assert actual.rng_state == state.rng_state
    assert actual.iteration == 7 and actual.radius == 4 and actual.direction_order == 0
    assert_matches(actual.offset_variance, state.offset_variance)
    assert actual.initialization == state.initialization
    for other_config, other_identity in (
        (dataclasses.replace(config, n_classes=3), identity),
        (dataclasses.replace(config, q=3), identity),
        (config, {**identity, "particles": "different-hash"}),
    ):
        with pytest.raises(ValueError, match="identity mismatch"):
            checkpoint.load(path, other_config, other_identity)
    assert not path.with_suffix(".tmp").exists()
