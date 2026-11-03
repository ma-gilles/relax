"""Algebra of the projected heteroscedastic PPCA of body poses, on linear Gaussian cases with known truth (CPU)."""

from __future__ import annotations

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.multibody import pose_ppca as pp


def cameras(rng, n, *, cone_deg=None):
    """``[n, 2, 3]`` projections P from random rotations, optionally with viewing directions in a cone about z."""
    q = np.linalg.qr(rng.standard_normal((n, 3, 3)))[0]
    if cone_deg is not None:
        angle = np.deg2rad(cone_deg) * rng.uniform(0, 1, n)
        phi = rng.uniform(0, 2 * np.pi, n)
        axis = np.stack([np.cos(angle) * 0 + np.sin(phi), -np.cos(phi), 0 * phi], axis=1)
        K = np.zeros((n, 3, 3))
        K[:, 0, 2], K[:, 2, 0] = axis[:, 1], -axis[:, 1]
        K[:, 1, 2], K[:, 2, 1] = -axis[:, 0], axis[:, 0]
        q = np.eye(3) + np.sin(angle)[:, None, None] * K + (1 - np.cos(angle))[:, None, None] * K @ K
    return q[:, :2, :]


def noise(rng, n, *, rank=11):
    """Heteroscedastic ``R [n, 11, 11]`` of a given rank, with cross terms."""
    roots = rng.standard_normal((n, 11, rank)) * rng.uniform(0.1, 0.6, (n, 1, 1))
    return roots @ np.swapaxes(roots, 1, 2)


def linear_case(seed=1, n=4000, q=3, tau=0.15, rank=5, cone_deg=None):
    rng = np.random.default_rng(seed)
    W = pp.canonical_gauge(rng.standard_normal((12, q)) * np.linspace(3.0, 1.0, q))
    mean = rng.standard_normal(12)
    z = rng.standard_normal((n, q))
    x = mean + z @ W.T + tau * rng.standard_normal((n, 12))
    P = cameras(rng, n, cone_deg=cone_deg)
    R = noise(rng, n, rank=rank)
    A = pp.operators(P)
    e = np.einsum("nij,nj->ni", np.linalg.cholesky(R + 1e-12 * np.eye(11)), rng.standard_normal((n, 11)))
    y = np.einsum("nij,nj->ni", A, x) + e
    return dict(y=y, P=P, R=R, W=W, mean=mean, tau2=tau**2, z=z, x=x)


def truth_fit(case):
    return pp.PPCAFit(case["mean"], case["W"], case["tau2"], 1e-6, np.zeros(1), True)


def test_marginal_likelihood_matches_a_dense_gaussian():
    from scipy.stats import multivariate_normal

    case = linear_case(n=30, rank=5)
    fit = truth_fit(case)
    got = pp.per_particle_log_likelihood(case["y"], case["P"], case["R"], fit.mean, fit.covariance)
    A = pp.operators(case["P"])
    want = [
        multivariate_normal(a @ fit.mean, a @ fit.covariance @ a.T + r).logpdf(y)
        for a, r, y in zip(A, case["R"], case["y"])
    ]
    assert_matches(got, np.asarray(want), rtol=1e-11)


def test_posterior_is_the_conditional_of_the_joint_gaussian():
    case = linear_case(n=12, rank=5)
    fit = truth_fit(case)
    post = pp.latent_posterior(case["y"], case["P"], case["R"], fit)
    q = fit.rank
    for i, a in enumerate(pp.operators(case["P"])):
        # Dense joint precision of (z, x) given y, in the feature block with R through a pseudo-free route: the
        # conditional of the joint Gaussian (z, y).
        cross = fit.W.T @ a.T
        T = a @ fit.covariance @ a.T + case["R"][i]
        gain = np.linalg.solve(T, cross.T).T
        assert_matches(post.z_map[i], gain @ (case["y"][i] - a @ fit.mean), rtol=1e-11)
        assert_matches(post.covariance[i], np.eye(q) - gain @ cross.T, rtol=1e-11)


def test_sufficient_statistics_give_the_posterior():
    case = linear_case(n=200, rank=5)
    fit = truth_fit(case)
    post = pp.latent_posterior(case["y"], case["P"], case["R"], fit)
    q = fit.rank
    inverse = np.linalg.inv(np.eye(q) + post.information)
    assert_matches(post.covariance, inverse, rtol=1e-10)
    assert_matches(post.z_map, np.einsum("nij,nj->ni", inverse, post.linear), rtol=1e-10)


def test_recovar_moments_agree_once_its_omitted_terms_are_restored():
    """RECOVAR's helper drops ``-0.5 logdet Psi`` and the residual norm's own Psi terms; they restore the marginal."""
    import jax.numpy as jnp
    from recovar.ppca.pose_marginal import compute_ppca_pose_scores_and_moments_no_contrast as helper

    case = linear_case(n=40, rank=5)
    fit = truth_fit(case)
    post = pp.latent_posterior(case["y"], case["P"], case["R"], fit)
    A = pp.operators(case["P"])
    residual = case["y"] - A @ fit.mean
    solved = np.linalg.solve(post.psi, residual[:, :, None])[:, :, 0]
    norm = np.sum(residual * solved, axis=1)
    zero = np.zeros(40)
    q = fit.rank
    out = helper(
        jnp.asarray(norm)[:, None, None], jnp.asarray(zero)[:, None, None], jnp.asarray(zero)[:, None, None],
        jnp.asarray(post.linear)[:, None, None, :], jnp.zeros((40, 1, 1, q)), jnp.asarray(post.information)[:, None, None],
        return_moments=False,
    )
    score = np.asarray(out[0] if isinstance(out, tuple) else out).reshape(-1)
    logdet = np.linalg.slogdet(post.psi)[1]
    restored = score - 0.5 * logdet - 0.5 * 11 * np.log(2 * np.pi)
    want = pp.per_particle_log_likelihood(case["y"], case["P"], case["R"], fit.mean, fit.covariance)
    assert_matches(restored, want, rtol=1e-9)


@pytest.mark.parametrize("rank", [5, 11])
def test_em_is_monotone_and_recovers_the_population(rank):
    case = linear_case(n=4000, rank=rank)
    fit = pp.fit_projected_ppca(case["y"], case["P"], case["R"], 3, floor=1e-5, tolerance=1e-7)
    assert fit.converged
    assert np.all(np.diff(fit.log_likelihood) > -1e-6 * np.abs(fit.log_likelihood[1:]))
    estimate, truth = fit.W @ fit.W.T + fit.tau2 * np.eye(12), case["W"] @ case["W"].T + case["tau2"] * np.eye(12)
    assert np.linalg.norm(estimate - truth) / np.linalg.norm(truth) < 0.05
    assert abs(fit.tau2 - case["tau2"]) / case["tau2"] < 0.4
    assert pp.log_likelihood(case["y"], case["P"], case["R"], fit) >= pp.log_likelihood(case["y"], case["P"], case["R"], truth_fit(case))


def test_em_is_monotone_with_a_binding_floor():
    case = linear_case(n=1500, tau=0.02)
    fit = pp.fit_projected_ppca(case["y"], case["P"], case["R"], 3, floor=0.2, tolerance=1e-7)
    assert fit.tau2_at_floor and fit.tau2 == 0.2
    assert np.all(np.diff(fit.log_likelihood) > -1e-6 * np.abs(fit.log_likelihood[1:]))


def test_em_does_not_depend_on_its_start():
    case = linear_case(n=2000)
    a = pp.fit_projected_ppca(case["y"], case["P"], case["R"], 3, floor=1e-5, tolerance=1e-9)
    b = pp.fit_projected_ppca(case["y"], case["P"], case["R"], 3, floor=1e-5, tolerance=1e-9, seed=5)
    assert abs(a.log_likelihood[-1] - b.log_likelihood[-1]) < 1e-5 * len(case["y"])
    assert np.linalg.norm(a.covariance - b.covariance) / np.linalg.norm(a.covariance) < 1e-3


def test_fully_observed_noiseless_reduces_to_classical_ppca():
    rng = np.random.default_rng(3)
    n = 3000
    x = rng.standard_normal((n, 3)) @ rng.standard_normal((3, 12)) + 0.3 * rng.standard_normal((n, 12))
    P = np.tile(np.eye(3), (n, 1, 1))
    fit = pp.fit_projected_ppca(x, P, np.zeros((n, 12, 12)), 3, floor=1e-6, tolerance=1e-9)
    values = np.linalg.eigvalsh(np.cov(x.T, bias=True))[::-1]
    assert_matches(fit.tau2, float(values[3:].mean()), rtol=1e-5)
    assert_matches(fit.amplitudes**2 + fit.tau2, values[:3], rtol=1e-5)
    assert_matches(fit.mean, x.mean(axis=0), rtol=1e-9)


def test_mean_depth_is_invisible_to_a_single_view():
    """With one viewing direction a change of the mean along it changes no likelihood, so it is not information."""
    case = linear_case(n=300)
    fixed = cameras(np.random.default_rng(2), 1)
    P = np.tile(fixed, (300, 1, 1))
    normal = np.cross(fixed[0, 0], fixed[0, 1])
    fit = truth_fit(case)
    moved = np.concatenate([np.zeros(9), 7.0 * normal])
    a = pp.per_particle_log_likelihood(case["y"], P, case["R"], fit.mean, fit.covariance)
    b = pp.per_particle_log_likelihood(case["y"], P, case["R"], fit.mean + moved, fit.covariance)
    assert_matches(a, b, rtol=1e-11)


def test_unseen_depth_keeps_its_prior_uncertainty():
    """A latent that only moves the depth of one view: its posterior keeps variance one, a full view cone resolves it."""
    case = linear_case(n=400)
    normal = np.array([0.0, 0.0, 1.0])
    W = np.zeros((12, 1))
    W[9:, 0] = 4.0 * normal
    fit = pp.PPCAFit(np.zeros(12), W, 0.01, 1e-6, np.zeros(1), True)
    view = np.tile(np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]), (400, 1, 1))
    R = np.tile(0.04 * np.eye(11), (400, 1, 1))
    blind = pp.latent_posterior(case["y"], view, R, fit)
    assert_matches(blind.covariance[:, 0, 0], np.ones(400), rtol=1e-10)
    seen = pp.latent_posterior(case["y"], cameras(np.random.default_rng(4), 400), R, fit)
    assert np.median(seen.covariance[:, 0, 0]) < 0.05


def test_posterior_is_calibrated_at_true_parameters():
    case = linear_case(n=6000)
    post = pp.latent_posterior(case["y"], case["P"], case["R"], truth_fit(case))
    error = post.z_map - case["z"]
    chi2 = np.einsum("ni,ni->n", error, np.linalg.solve(post.covariance, error[:, :, None])[:, :, 0])
    assert abs(chi2.mean() / 3 - 1.0) < 0.05
    assert abs(np.mean(chi2 < 6.2514) - 0.9) < 0.015  # the 90% point of chi-square with 3 degrees of freedom


def test_withheld_coordinate_is_calibrated_and_matches_a_dense_conditional():
    case = linear_case(n=6000)
    fit = truth_fit(case)
    held = case["y"][:, 10]
    mean, variance = pp.withheld_prediction(case["y"], case["P"], case["R"], fit, 10)
    a = pp.operators(case["P"][:1])[0]
    T = a @ fit.covariance @ a.T + case["R"][0]
    keep = np.arange(10)
    gain = np.linalg.solve(T[np.ix_(keep, keep)], T[keep, 10])
    assert_matches(mean[0], (a @ fit.mean)[10] + gain @ (case["y"][0, :10] - (a @ fit.mean)[:10]), rtol=1e-11)
    assert_matches(variance[0], T[10, 10] - gain @ T[keep, 10], rtol=1e-11)
    standard = (held - mean) / np.sqrt(variance)
    assert abs(np.mean(standard**2) - 1.0) < 0.05


def test_gauge_keeps_the_decoded_mean_and_sets_principal_covariance():
    case = linear_case(n=50)
    fit = truth_fit(case)
    assert_matches(pp.canonical_gauge(fit.W), fit.W, rtol=1e-11)
    post = pp.latent_posterior(case["y"], case["P"], case["R"], fit)
    coordinates, covariance = post.principal(fit)
    a, b = post.structured_mean[0] - post.structured_mean[1], coordinates[0] - coordinates[1]
    assert_matches(a @ a, b @ b, rtol=1e-10)
    assert_matches(covariance[0], np.diag(fit.amplitudes) @ post.covariance[0] @ np.diag(fit.amplitudes), rtol=1e-11)


def test_shrinkage_records_keep_gain_distinct_from_posterior_covariance():
    case = linear_case(n=30)
    post = pp.latent_posterior(case["y"], case["P"], case["R"], truth_fit(case))
    gain = post.covariance @ post.information
    # With an informative prior the gain is below one: the posterior covariance is not the noise of z_map.
    noise_of_map = gain @ post.covariance
    assert np.all(np.trace(noise_of_map, axis1=1, axis2=2) < np.trace(post.covariance, axis1=1, axis2=2))
