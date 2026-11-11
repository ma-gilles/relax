"""Model B, the contrast on the whole signal integrated per pose: the per-pose math against a dense reference."""

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.contrast_b_reference import N_IMAGES, N_POSES, N_TRANSLATIONS, make_statistics, reference_pose, unpack_gram
from helpers.float_compare import assert_matches

from relax.ppca_refinement import contrast_b
from relax.ppca_refinement.full_row_kernels import latent_scores, second_moment_sums

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("q", [1, 3])
def test_scores_and_moments_match_a_dense_reference(q):
    # Seven nodes: the worst relative difference from the float64 dense reference is 1.4e-6 (float32 arithmetic).
    kappa, sigma = 1.0, 0.3
    inner, gram = make_statistics(q)
    rule = contrast_b.hermite_rule(7)
    prior = jnp.zeros((N_IMAGES, N_POSES, N_TRANSLATIONS))
    score = np.asarray(contrast_b.latent_scores_b(jnp.asarray(inner), jnp.asarray(gram), prior, kappa, sigma, rule, q))
    gamma = np.full((N_POSES, N_IMAGES, N_TRANSLATIONS), 0.5, np.float32)
    moments = contrast_b.posterior_moments_b(jnp.asarray(inner), jnp.asarray(gram), jnp.asarray(gamma), kappa, sigma, rule, q)
    index = {(i, j): k for k, (i, j) in enumerate(zip(*np.triu_indices(q + 1)))}
    for r in range(N_POSES):
        for b in range(N_IMAGES):
            nu, h, H = unpack_gram(gram, q, r, b)
            expected_sums = np.zeros((len(index),))
            expected_embedding = np.zeros(q + 1)
            for t_index in range(N_TRANSLATIONS):
                log_z, e_c, e_c2, e_cz, e_c2z, e_c2zz, e_z = reference_pose(
                    inner[0, r, b, t_index], inner[1:, r, b, t_index].astype(np.float64), nu, h, H, kappa, sigma
                )
                assert_matches(score[r, b, t_index], np.float32(log_z), rtol=1e-5)
                assert_matches(np.asarray(moments.weights)[0, r, b, t_index], np.float32(0.5 * e_c), rtol=1e-5)
                assert_matches(np.asarray(moments.weights)[1:, r, b, t_index], (0.5 * e_cz).astype(np.float32), rtol=1e-5)
                expected_sums[index[(0, 0)]] += 0.5 * e_c2
                expected_embedding += 0.5 * np.concatenate([e_z, [e_c]])
                for i in range(1, q + 1):
                    expected_sums[index[(0, i)]] += 0.5 * e_c2z[i - 1]
                    for j in range(i, q + 1):
                        expected_sums[index[(i, j)]] += 0.5 * e_c2zz[i - 1, j - 1]
            assert_matches(np.asarray(moments.sums)[:, r, b], expected_sums.astype(np.float32), rtol=1e-5)
            assert_matches(np.asarray(moments.embedding)[:, r, b], expected_embedding.astype(np.float32), rtol=1e-5)


def test_more_nodes_converge_to_the_dense_reference():
    q, kappa, sigma = 3, 1.0, 0.3
    inner, gram = make_statistics(q, nu=30.0)  # a wide posterior of the contrast: the nodes matter
    prior = jnp.zeros((N_IMAGES, N_POSES, N_TRANSLATIONS))
    errors = []
    for n_nodes in (1, 3, 5):
        score = np.asarray(
            contrast_b.latent_scores_b(jnp.asarray(inner), jnp.asarray(gram), prior, kappa, sigma, contrast_b.hermite_rule(n_nodes), q)
        )
        worst = 0.0
        for r in range(N_POSES):
            for b in range(N_IMAGES):
                nu, h, H = unpack_gram(gram, q, r, b)
                for t_index in range(N_TRANSLATIONS):
                    reference = reference_pose(inner[0, r, b, t_index], inner[1:, r, b, t_index].astype(np.float64), nu, h, H, kappa, sigma)[0]
                    worst = max(worst, abs(float(score[r, b, t_index]) - reference))
        errors.append(worst)
    # Largest log-evidence error over all poses, incl. wrong ones (docs/math/ppca_contrast.md section 11 for the nodes).
    assert errors[0] > errors[1] > errors[2]
    assert errors[1] < 0.01 and errors[2] < 2e-3  # measured: 0.24, 6.4e-3 and 7.5e-4 nats for 1, 3 and 5 nodes


def test_without_contrast_spread_the_score_is_the_unit_contrast_score_and_the_moments_are_the_posterior_moments():
    q = 3
    inner, gram = make_statistics(q)
    prior = jnp.asarray(np.random.default_rng(1).normal(size=(N_IMAGES, N_POSES, N_TRANSLATIONS)).astype(np.float32))
    unit_score, mean, covariance = latent_scores(jnp.asarray(inner), jnp.asarray(gram), prior)
    rule = contrast_b.hermite_rule(3)
    score = contrast_b.latent_scores_b(jnp.asarray(inner), jnp.asarray(gram), prior, 1.0, 1e-3, rule, q)
    assert_matches(np.asarray(score), np.asarray(unit_score), rtol=1e-5)
    gamma = jnp.exp(unit_score - jnp.max(unit_score))
    moments = contrast_b.posterior_moments_b(jnp.asarray(inner), jnp.asarray(gram), gamma, 1.0, 1e-3, rule, q)
    unit_sums = second_moment_sums(gamma, mean, covariance)
    assert_matches(np.asarray(moments.sums), np.asarray(unit_sums), rtol=1e-4)
    assert_matches(np.asarray(moments.weights)[1:], np.asarray(gamma[None] * mean), rtol=1e-4)
    assert_matches(np.asarray(moments.weights)[0], np.asarray(gamma), rtol=1e-4)
    assert_matches(np.asarray(moments.embedding)[-1], np.asarray(jnp.sum(gamma, axis=-1)), rtol=1e-4)


def test_without_loadings_the_integral_is_the_conjugate_gaussian_for_any_node_count():
    kappa, sigma = 1.0, 0.3
    inner, gram = make_statistics(0)
    prior = jnp.zeros((N_IMAGES, N_POSES, N_TRANSLATIONS))
    nu = gram[0][..., None]
    t = inner[0]
    closed = -0.5 * np.log(1 + sigma**2 * nu) + 0.5 * ((t + kappa / sigma**2) ** 2 / (nu + 1 / sigma**2) - kappa**2 / sigma**2)
    for n_nodes in (1, 3):
        score = contrast_b.latent_scores_b(jnp.asarray(inner), jnp.asarray(gram), prior, kappa, sigma, contrast_b.hermite_rule(n_nodes), 0)
        assert_matches(np.asarray(score), closed.astype(np.float32), rtol=1e-4)
    moments = contrast_b.posterior_moments_b(
        jnp.asarray(inner), jnp.asarray(gram), jnp.ones((N_POSES, N_IMAGES, N_TRANSLATIONS)), kappa, sigma, contrast_b.hermite_rule(3), 0
    )
    posterior_mean = (t + kappa / sigma**2) / (nu + 1 / sigma**2)
    assert_matches(np.asarray(moments.weights)[0], posterior_mean.astype(np.float32), rtol=1e-4)
    assert moments.sums.shape[0] == 1 and moments.weights.shape[0] == 1


def test_the_prior_mean_and_sd_enter_the_posterior_of_the_contrast():
    """A prior centred at 1.2 with sd 0.2 moves the estimate toward 1.2 relative to a prior centred at 1."""
    q = 2
    inner, gram = make_statistics(q, nu=15.0, contrast_sd=0.0)
    prior = jnp.zeros((N_IMAGES, N_POSES, N_TRANSLATIONS))
    gamma = jnp.ones((N_POSES, N_IMAGES, N_TRANSLATIONS))
    rule = contrast_b.hermite_rule(5)
    low = contrast_b.posterior_moments_b(jnp.asarray(inner), jnp.asarray(gram), gamma, 1.0, 0.2, rule, q)
    high = contrast_b.posterior_moments_b(jnp.asarray(inner), jnp.asarray(gram), gamma, 1.2, 0.2, rule, q)
    assert np.all(np.asarray(high.weights)[0] > np.asarray(low.weights)[0])
    del prior
