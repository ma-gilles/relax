"""The empty component equals the brute-force two-component mixture posterior."""

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.ppca_refinement.full_row_kernels import Posterior
from relax.ppca_refinement.occupancy import check_occupancy, log_partition_correction, with_empty_component


def _posterior(score):
    """Pass-1 normalization of ``score`` ``(R, B, T)``, as ``full_row_kernels._normalize``."""
    center = jnp.max(score, axis=(0, 2))
    centered_logZ = jnp.log(jnp.sum(jnp.exp(score - center[None, :, None]), axis=(0, 2)))
    zeros = jnp.zeros(score.shape[1], jnp.int32)
    return Posterior(center, centered_logZ, center, zeros, zeros)


def _pass2_weights(score, posterior):
    return np.asarray(jnp.exp(score - posterior.center[None, :, None] - posterior.centered_logZ[None, :, None]))


@pytest.mark.parametrize("occupancy", [0.05, 0.3, 0.9])
def test_weights_and_p_occupied_match_the_mixture(occupancy):
    rng = np.random.default_rng(0)
    R, B, T = 7, 5, 3
    log_prior = np.log(np.full((R, 1, T), 1.0 / (R * T)))
    # Evidence from clearly empty (scores well below 0) to clearly occupied (well above).
    score = jnp.asarray(log_prior + rng.normal(0, 2, (R, B, T)) + np.linspace(-15, 15, B)[None, :, None], jnp.float32)
    posterior, p_occupied = with_empty_component(_posterior(score), occupancy)

    s = np.asarray(score, np.float64)
    occupied = occupancy * np.exp(s)
    z = occupied.sum(axis=(0, 2)) + (1 - occupancy)
    # float32 normalization against the float64 mixture: measured largest relative differences 2e-7 (P),
    # 6.2e-7 (weights), 4e-8 (log Z), inside the default band.
    assert_matches(np.asarray(p_occupied), occupied.sum(axis=(0, 2)) / z)
    assert_matches(_pass2_weights(score, posterior), occupied / z[None, :, None])
    # Monotone in the evidence: the empty images come first.
    assert np.all(np.diff(np.asarray(p_occupied)) > 0)
    # Absolute log-partition of the mixture.
    log_z = np.asarray(posterior.center + posterior.centered_logZ) + log_partition_correction(occupancy)
    assert_matches(log_z.astype(np.float32), np.log(z))


def test_unit_occupancy_is_the_plain_posterior():
    score = jnp.asarray(np.random.default_rng(1).normal(0, 3, (4, 6, 2)), jnp.float32)
    plain = _posterior(score)
    posterior, p_occupied = with_empty_component(plain, 1.0)
    assert posterior is plain  # occupancy 1 is the plain model, not a recomputation of it
    assert_matches(np.asarray(p_occupied), np.ones(6, np.float32))


def test_padding_images_keep_no_mass():
    score = jnp.asarray(np.random.default_rng(2).normal(0, 1, (3, 4, 2)), jnp.float32)
    plain = _posterior(score)
    padded = plain._replace(center=plain.center.at[2:].set(jnp.inf))  # as drop_padding
    posterior, p_occupied = with_empty_component(padded, 0.1)
    assert np.all(np.isinf(np.asarray(posterior.center[2:])))
    assert np.all(np.isfinite(np.asarray(posterior.centered_logZ)))
    assert_matches(np.asarray(p_occupied[2:]), np.zeros(2, np.float32))
    assert np.all(_pass2_weights(score, posterior)[:, 2:] == 0)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_occupancy_prior_is_validated(bad):
    with pytest.raises(ValueError):
        check_occupancy(bad)
    assert check_occupancy(None) is None
