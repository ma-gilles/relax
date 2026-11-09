#!/usr/bin/env python3
"""Model A (contrast on the mean, closed form) against the exact physical model B (contrast on everything), per image.

Images are drawn from the physical model ``y = c (a + B z) + noise`` (``a`` the projected mean, ``B`` the projected
loadings, ``c ~ N(1, sd^2)``, ``z ~ N(0, I)``, white unit noise) at a single known pose. The exact posterior of model B
is computed by quadrature over ``c`` on a fine grid with ``z`` integrated in closed form (docs/math/ppca_contrast.md,
section 3); model A's posterior is its Gaussian closed form; the unit-contrast model is the ``c = 1`` case. The table
gives, per loading-to-mean energy ratio and contrast sd, how well each model recovers the contrast and the latent
coordinates and how far model A's contrast posterior is from B's.

The pose table asks how well each model tells the true pose from wrong ones: wrong poses carry the mean and loadings
of a pose whose images correlate with the true ones by ``POSE_CORRELATION`` (neighbouring poses of the grid), and each
model's evidence ``log p(y | pose)`` ranks them. ``margin`` is the true pose's evidence over the best wrong one.

Run (CPU, seconds):
    python scripts/ppca_contrast/model_a_vs_exact.py
"""

import numpy as np
from scipy.special import logsumexp

N_PIXELS, N_LATENT, N_IMAGES = 600, 3, 1500
MEAN_SNR = 200.0  # nu = <a, a> / noise variance, of the order of a particle's integrated signal-to-noise
POSE_CORRELATION, N_WRONG_POSES = 0.95, 12


def simulate(rng, ratio: float, sd: float, n_images: int = N_IMAGES):
    """One toy particle set: ``a`` (n,), ``B`` (n, q), the images and the true contrast and latents."""
    a = rng.normal(size=N_PIXELS)
    a *= np.sqrt(MEAN_SNR) / np.linalg.norm(a)
    loadings = rng.normal(size=(N_PIXELS, N_LATENT))
    loadings *= np.sqrt(ratio * MEAN_SNR / N_LATENT) / np.linalg.norm(loadings, axis=0)
    contrast = 1.0 + sd * rng.normal(size=n_images)
    z = rng.normal(size=(n_images, N_LATENT))
    y = contrast[:, None] * (a + z @ loadings.T) + rng.normal(size=(n_images, N_PIXELS))
    return a, loadings, y, contrast, z


def statistics(a, loadings, y):
    """The six per-pose statistics of every image (white unit noise)."""
    return dict(s=np.sum(y**2, axis=1), t=y @ a, nu=a @ a, g=y @ loadings, h=loadings.T @ a, H=loadings.T @ loadings)


def exact_posterior(st, sd: float, grid: np.ndarray):
    """Model B: posterior mean of c and E[c z]-free summaries by quadrature over ``grid``; returns E[c] and E[z] per image."""
    prior = np.exp(-0.5 * ((grid - 1) / sd) ** 2)
    n, q = st["g"].shape
    log_weight = np.empty((n, len(grid)))
    z_mean = np.empty((n, len(grid), q))
    eye = np.eye(q)
    for k, c in enumerate(grid):
        precision = eye + c**2 * st["H"]
        b = c * (st["g"] - c * st["h"])
        solved = np.linalg.solve(precision, b.T).T
        r = st["s"] - 2 * c * st["t"] + c**2 * st["nu"]
        log_weight[:, k] = -0.5 * (r - np.sum(b * solved, axis=1) + np.linalg.slogdet(precision)[1]) + np.log(prior[k])
        z_mean[:, k] = solved
    weight = np.exp(log_weight - log_weight.max(axis=1, keepdims=True))
    weight /= weight.sum(axis=1, keepdims=True)
    mean_c = weight @ grid
    mean_z = np.einsum("nk,nkq->nq", weight, z_mean)
    sd_c = np.sqrt(weight @ grid**2 - mean_c**2)
    return mean_c, mean_z, sd_c


def model_a_posterior(st, sd: float):
    """Model A: closed-form posterior of ``(e, z)``; returns the contrast estimate, ``u`` and the contrast's posterior sd."""
    q = st["g"].shape[1]
    h_a = np.zeros((q + 1, q + 1))
    h_a[0, 0], h_a[0, 1:], h_a[1:, 0], h_a[1:, 1:] = sd**2 * st["nu"], sd * st["h"], sd * st["h"], st["H"]
    precision = np.eye(q + 1) + h_a
    b = np.concatenate([sd * (st["t"] - st["nu"])[:, None], st["g"] - st["h"]], axis=1)
    mean = np.linalg.solve(precision, b.T).T
    covariance = np.linalg.inv(precision)
    return 1 + sd * mean[:, 0], mean[:, 1:], sd * np.sqrt(covariance[0, 0])


def unit_posterior(st):
    """The unit-contrast model: ``E[z | y]`` at ``c = 1``."""
    q = st["g"].shape[1]
    return np.linalg.solve(np.eye(q) + st["H"], (st["g"] - st["h"]).T).T


def log_evidence_unit(st):
    """``log p(y | pose)`` of the unit-contrast model, up to the pose-independent constant."""
    q = st["g"].shape[1]
    precision = np.eye(q) + st["H"]
    b = st["g"] - st["h"]
    solved = np.linalg.solve(precision, b.T).T
    return -0.5 * (st["s"] - 2 * st["t"] + st["nu"] - np.sum(b * solved, axis=1) + np.linalg.slogdet(precision)[1])


def log_evidence_a(st, sd: float):
    """Model A's evidence: the Gaussian marginal with the contrast latent."""
    q = st["g"].shape[1]
    h_a = np.zeros((q + 1, q + 1))
    h_a[0, 0], h_a[0, 1:], h_a[1:, 0], h_a[1:, 1:] = sd**2 * st["nu"], sd * st["h"], sd * st["h"], st["H"]
    precision = np.eye(q + 1) + h_a
    b = np.concatenate([sd * (st["t"] - st["nu"])[:, None], st["g"] - st["h"]], axis=1)
    solved = np.linalg.solve(precision, b.T).T
    return -0.5 * (st["s"] - 2 * st["t"] + st["nu"] - np.sum(b * solved, axis=1) + np.linalg.slogdet(precision)[1])


def log_evidence_exact(st, sd: float, grid: np.ndarray):
    """Model B's evidence: the quadrature of the contrast prior against the closed-form ``z`` marginal."""
    log_prior = -0.5 * ((grid - 1) / sd) ** 2
    log_prior -= logsumexp(log_prior)
    q = st["g"].shape[1]
    terms = np.empty((len(st["s"]), len(grid)))
    for k, c in enumerate(grid):
        precision = np.eye(q) + c**2 * st["H"]
        b = c * (st["g"] - c * st["h"])
        solved = np.linalg.solve(precision, b.T).T
        r = st["s"] - 2 * c * st["t"] + c**2 * st["nu"]
        terms[:, k] = -0.5 * (r - np.sum(b * solved, axis=1) + np.linalg.slogdet(precision)[1]) + log_prior[k]
    return logsumexp(terms, axis=1)


def wrong_pose(rng, a, loadings):
    """The mean and loadings of a pose whose images correlate with the true pose's by ``POSE_CORRELATION``."""
    rho = POSE_CORRELATION

    def mix(x):
        other = rng.normal(size=x.shape)
        other *= np.linalg.norm(x, axis=0) / np.linalg.norm(other, axis=0)
        return rho * x + np.sqrt(1 - rho**2) * other

    return mix(a), mix(loadings)


def pose_margins(rng, a, loadings, y, sd: float, grid: np.ndarray) -> dict:
    """Per model, the fraction of images whose true pose has the largest evidence and the mean margin over the best wrong pose."""
    models = {
        "unit": lambda st: log_evidence_unit(st),
        "A": lambda st: log_evidence_a(st, sd),
        "B": lambda st: log_evidence_exact(st, sd, grid),
    }
    poses = [(a, loadings)] + [wrong_pose(rng, a, loadings) for _ in range(N_WRONG_POSES)]
    out = {}
    for name, evidence in models.items():
        scores = np.stack([evidence(statistics(pa, pl, y)) for pa, pl in poses], axis=1)
        margin = scores[:, 0] - scores[:, 1:].max(axis=1)
        out[f"{name}_pose_correct"] = float(np.mean(margin > 0))
        out[f"{name}_margin"] = float(np.mean(margin))
    return out


def r2(estimate, truth):
    """Fraction of the variance of ``truth`` explained by ``estimate`` after the best scalar per coordinate."""
    explained = 0.0
    for k in range(truth.shape[1]):
        slope = np.dot(estimate[:, k], truth[:, k]) / np.dot(estimate[:, k], estimate[:, k])
        explained += 1 - np.sum((truth[:, k] - slope * estimate[:, k]) ** 2) / np.sum(truth[:, k] ** 2)
    return explained / truth.shape[1]


def compare(
    ratio: float, sd: float, seed: int = 0, model_sd: float | None = None, n_images: int = N_IMAGES, n_grid: int = 801
) -> dict:
    """The metrics of one (loading ratio, true contrast sd) setting; the models use ``model_sd`` (default: the true sd)."""
    msd = sd if model_sd is None else model_sd
    rng = np.random.default_rng(seed)
    a, loadings, y, contrast, z = simulate(rng, ratio, sd, n_images)
    st = statistics(a, loadings, y)
    grid = np.linspace(max(1 - 6 * msd, 0.05), 1 + 6 * msd, n_grid)
    c_b, z_b, sd_b = exact_posterior(st, msd, grid)
    c_a, u_a, sd_a = model_a_posterior(st, msd)
    z_unit = unit_posterior(st)
    poses = pose_margins(np.random.default_rng(seed + 1), a, loadings, y, msd, grid)
    return {
        **poses,
        "corr_c_B": np.corrcoef(c_b, contrast)[0, 1] if sd else float("nan"),
        "corr_c_A": np.corrcoef(c_a, contrast)[0, 1] if sd else float("nan"),
        "corr_c_A_vs_B": np.corrcoef(c_a, c_b)[0, 1],
        "mean_sd_c_B": float(np.mean(sd_b)),
        "mean_sd_c_A": float(np.mean(sd_a)),
        "r2_z_B": r2(z_b, z),
        "r2_z_A_over_c": r2(u_a / c_a[:, None], z),
        "r2_z_A_scaled": r2(u_a, z),
        "r2_z_unit": r2(z_unit, z),
    }


def main() -> None:
    """Print the posterior table and the pose table."""
    tables = (
        ["corr_c_B", "corr_c_A", "corr_c_A_vs_B", "mean_sd_c_B", "mean_sd_c_A", "r2_z_B", "r2_z_A_over_c", "r2_z_A_scaled", "r2_z_unit"],
        [f"{m}_{k}" for k in ("pose_correct", "margin") for m in ("unit", "A", "B")],
    )
    # At true sd 0 the models still allow a contrast spread of 0.15: what modelling a contrast that is absent costs.
    rows = {
        (ratio, sd): compare(ratio, sd, model_sd=0.15 if sd == 0 else None)
        for ratio in (0.1, 0.3, 1.0, 3.0)
        for sd in (0.0, 0.15, 0.3)
    }
    for keys in tables:
        print(f"{'ratio':>6} {'sd':>5} " + " ".join(f"{k:>14}" for k in keys))
        for (ratio, sd), row in rows.items():
            print(f"{ratio:6.1f} {sd:5.2f} " + " ".join(f"{row[k]:14.3f}" for k in keys))
        print()


if __name__ == "__main__":
    main()
