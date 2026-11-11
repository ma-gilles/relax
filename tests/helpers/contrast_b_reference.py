"""Reference computations for the model B tests: simulated pass-1 operands and a dense-grid posterior of the contrast."""

import numpy as np
from scipy.special import logsumexp

N_PIXELS, N_POSES, N_IMAGES, N_TRANSLATIONS = 80, 3, 3, 2


def make_statistics(q: int, nu: float = 120.0, ratio: float = 1.0, contrast_sd: float = 0.3, seed: int = 0):
    """Pass-1 operands of ``N_POSES`` poses (the first is the true one) for ``N_IMAGES`` images at ``N_TRANSLATIONS``
    translations: ``inner`` ``(q + 1, R, B, T)`` and the packed Gram ``(tri(q + 1), R, B)``, from white-noise images of
    ``y = c (a + B z) + noise``."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=N_PIXELS) * np.sqrt(nu / N_PIXELS)
    loadings = rng.normal(size=(N_PIXELS, q)) * np.sqrt(ratio * nu / (N_PIXELS * max(q, 1)))
    n = N_IMAGES * N_TRANSLATIONS
    c = 1.0 + contrast_sd * rng.normal(size=n)
    y = c[:, None] * (a + rng.normal(size=(n, q)) @ loadings.T) + rng.normal(size=(n, N_PIXELS))
    poses = [(a, loadings)] + [
        (0.9 * a + 0.4 * rng.normal(size=N_PIXELS) * np.sqrt(nu / N_PIXELS), loadings + 0.3 * rng.normal(size=loadings.shape) * np.sqrt(nu / N_PIXELS))
        for _ in range(N_POSES - 1)
    ]
    inner = np.stack([np.stack([y @ pa] + [y @ pb for pb in pl.T]) for pa, pl in poses], axis=1)  # (P, R, n)
    inner = inner.reshape(q + 1, N_POSES, N_IMAGES, N_TRANSLATIONS)
    index = {(i, j): k for k, (i, j) in enumerate(zip(*np.triu_indices(q + 1)))}
    gram = np.zeros((len(index), N_POSES, N_IMAGES))
    for r, (pa, pl) in enumerate(poses):
        columns = np.concatenate([pa[:, None], pl], axis=1)
        full = columns.T @ columns
        for (i, j), k in index.items():
            gram[k, r] = full[i, j]
    return inner.astype(np.float32), gram.astype(np.float32)


def reference_pose(t, g, nu, h, H, kappa, sigma, n_grid: int = 4001):
    """``(log Z, E[c], E[c^2], E[c z], E[c^2 z], E[c^2 z z^T], E[z])`` of one pose by a dense grid over ``c`` and direct solves."""
    q = len(g)
    grid = np.linspace(kappa - 7 * sigma, kappa + 7 * sigma, n_grid)
    log_f, means, covs = np.empty(n_grid), np.empty((n_grid, q)), np.empty((n_grid, q, q))
    for k, c in enumerate(grid):
        precision = np.eye(q) + c**2 * H
        b = c * (g - c * h)
        means[k] = np.linalg.solve(precision, b)
        covs[k] = np.linalg.inv(precision)
        log_f[k] = (
            -0.5 * (c**2 * nu - 2 * c * t - b @ means[k] + np.linalg.slogdet(precision)[1])
            - 0.5 * ((c - kappa) / sigma) ** 2
            - np.log(sigma * np.sqrt(2 * np.pi))
        )
    rho = np.exp(log_f - logsumexp(log_f))
    second_zz = np.einsum("k,kij->ij", rho * grid**2, covs + means[:, :, None] * means[:, None, :])
    return (
        logsumexp(log_f) + np.log(grid[1] - grid[0]), rho @ grid, rho @ grid**2, (rho * grid) @ means,
        (rho * grid**2) @ means, second_zz, rho @ means,
    )


def unpack_gram(gram, q: int, r: int, b: int):
    """``(nu, h, H)`` of pose ``r`` and image ``b`` from the packed Gram."""
    index = {(i, j): k for k, (i, j) in enumerate(zip(*np.triu_indices(q + 1)))}
    nu = gram[index[(0, 0)], r, b]
    h = np.array([gram[index[(0, j)], r, b] for j in range(1, q + 1)])
    H = np.array([[gram[index[(min(i, j), max(i, j))], r, b] for j in range(1, q + 1)] for i in range(1, q + 1)])
    return nu, h, H
