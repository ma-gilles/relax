"""Projected heteroscedastic PPCA of one body's pose table: an analysis of finished poses, never an input to refinement.

The complete feature of a particle is ``x = [vec(Q L), t + Q m]`` (12 numbers, angstroms; ``relax.multibody.pose_uq``).
Its translation along the viewing direction is not observed, so a particle gives ``y_i = A_i x_i + e_i`` with
``A_i = blockdiag(I_9, P_i)``, ``P_i`` the 2x3 camera projection, and ``e_i ~ N(0, R_i)`` from the pose uncertainty::

    z_i ~ N(0, I_q)     x_i = mu + W z_i + xi_i     xi_i ~ N(0, tau^2 I_12)     y_i = A_i x_i + e_i

``R_i`` is supplied and held fixed; ``tau^2`` is model residual variance, not a calibration of ``R_i``. A single fit
takes ``q`` and one positive ``floor`` on ``tau^2``. The marginal covariance of ``y_i`` is
``T_i = A_i Sigma A_i^T + R_i`` with ``Sigma = W W^T + tau^2 I``, and the fit is exact EM with the complete feature as
the missing data: the E-step's moments ``b_i`` and ``V_i`` give ``mu`` and the expected scatter ``S``, and the M-step
is the PPCA-constrained maximiser over ``S``, which also holds when the floor binds (the likelihood is unimodal in
``tau^2`` around the unconstrained mean of the bottom eigenvalues).

Everything is host NumPy float64, the fitter's only path: ``R_i`` is rank deficient at first order, so ``T_i`` is
conditioned by ``tau^2`` and single precision would not resolve it. No volume machinery is involved. Arrays are
``y [N, D]``, ``P [N, D - 9, 3]``, ``R [N, D, D]``; ``D = 11`` in use and ``D = 12`` (``P`` the identity) is the fully
observed reduction.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

FEATURES = 12
ROTATION_FEATURES = 9
BATCH = 20000


@dataclass(frozen=True)
class PPCAFit:
    """A fitted model: ``W [12, q]`` with orthogonal columns in descending norm and a fixed sign, the first column of
    the gauge (:func:`canonical_gauge`)."""

    mean: np.ndarray
    W: np.ndarray
    tau2: float
    floor: float
    log_likelihood: np.ndarray  # observed log likelihood before each M-step; the last entry is at the returned model
    converged: bool

    @property
    def rank(self) -> int:
        return int(self.W.shape[1])

    @property
    def amplitudes(self) -> np.ndarray:
        """The singular values of ``W``, angstroms."""
        return np.linalg.norm(self.W, axis=0)

    @property
    def effective_rank(self) -> int:
        """Columns of ``W`` the fit kept above zero."""
        return int(np.sum(self.amplitudes > 0.0))

    @property
    def tau2_at_floor(self) -> bool:
        return bool(self.tau2 <= self.floor)

    @property
    def covariance(self) -> np.ndarray:
        return self.W @ self.W.T + self.tau2 * np.eye(FEATURES)


@dataclass(frozen=True)
class LatentPosterior:
    """Per-particle Gaussian posteriors of ``z``, and what a later density estimate needs (never additive noise).

    Conditional on ``z``, ``z_map`` is a shrunken observation with gain ``covariance @ information`` and noise covariance
    ``covariance @ information @ covariance``. ``covariance`` itself is not a measurement noise."""

    z_map: np.ndarray  # [N, q], standardized; also the posterior mean
    covariance: np.ndarray  # [N, q, q]
    information: np.ndarray  # [N, q, q] I_i = W^T A_i^T Psi_i^-1 A_i W
    linear: np.ndarray  # [N, q] h_i = W^T A_i^T Psi_i^-1 (y_i - A_i mu)
    psi: np.ndarray  # [N, D, D] Psi_i = tau^2 A_i A_i^T + R_i
    structured_mean: np.ndarray  # [N, 12] mu + W z_map
    feature_mean: np.ndarray  # [N, 12] b_i, the complete-feature posterior mean; differs from structured_mean for tau > 0

    def principal(self, fit: PPCAFit) -> tuple[np.ndarray, np.ndarray]:
        """Physical principal coordinates ``amplitudes * z_map`` (angstroms) and their posterior covariance. Euclidean
        distance in them is the decoded structured-mean distance."""
        amplitudes = fit.amplitudes
        return self.z_map * amplitudes, self.covariance * amplitudes[None, :, None] * amplitudes[None, None, :]


def operators(P: np.ndarray) -> np.ndarray:
    """``A_i = blockdiag(I_9, P_i)`` of a batch, ``[n, D, 12]``. Build per batch: never keep them for all particles."""
    P = np.asarray(P, dtype=np.float64)
    n, rows = P.shape[:2]
    A = np.zeros((n, ROTATION_FEATURES + rows, FEATURES))
    A[:, np.arange(ROTATION_FEATURES), np.arange(ROTATION_FEATURES)] = 1.0
    A[:, ROTATION_FEATURES:, ROTATION_FEATURES:] = P
    return A


def _batches(n: int):
    for start in range(0, n, BATCH):
        yield slice(start, min(start + BATCH, n))


def _spd_solver(matrix: np.ndarray):
    """``solve(b)`` for SPD ``matrix [n, D, D]`` and ``b [n, D, k]`` through its Cholesky factor, and ``logdet``."""
    root = np.linalg.cholesky(matrix)
    transpose = np.swapaxes(root, 1, 2)

    def solve(b):
        return np.linalg.solve(transpose, np.linalg.solve(root, b))

    return solve, 2.0 * np.sum(np.log(np.diagonal(root, axis1=1, axis2=2)), axis=1)


def _batch_log_likelihood(residual: np.ndarray, solve, logdet: np.ndarray) -> np.ndarray:
    quadratic = np.sum(residual * solve(residual[:, :, None])[:, :, 0], axis=1)
    return -0.5 * (logdet + quadratic + residual.shape[1] * np.log(2.0 * np.pi))


def per_particle_log_likelihood(y, P, R, mean, covariance) -> np.ndarray:
    """``log N(y_i; A_i mean, A_i covariance A_i^T + R_i)`` of every particle."""
    y, R = np.asarray(y, dtype=np.float64), np.asarray(R, dtype=np.float64)
    out = np.empty(len(y))
    for b in _batches(len(y)):
        A = operators(P[b])
        T = A @ covariance @ np.swapaxes(A, 1, 2) + R[b]
        solve, logdet = _spd_solver(T)
        out[b] = _batch_log_likelihood(y[b] - A @ mean, solve, logdet)
    return out


def _moments(y, P, R, mean, covariance):
    """One E-step: the new mean, the expected scatter about it, and the observed log likelihood."""
    n = len(y)
    shift_sum, scatter, log_likelihood = np.zeros(FEATURES), np.zeros((FEATURES, FEATURES)), 0.0
    for b in _batches(n):
        A = operators(P[b])
        covariance_At = covariance @ np.swapaxes(A, 1, 2)
        solve, logdet = _spd_solver(A @ covariance_At + R[b])
        residual = y[b] - A @ mean
        log_likelihood += float(np.sum(_batch_log_likelihood(residual, solve, logdet)))
        shift = (covariance_At @ solve(residual[:, :, None]))[:, :, 0]
        gain = covariance_At @ solve(np.swapaxes(covariance_At, 1, 2))
        shift_sum += shift.sum(axis=0)
        scatter += shift.T @ shift + (covariance[None] - gain).sum(axis=0)
    shift = shift_sum / n
    return mean + shift, scatter / n - np.outer(shift, shift), log_likelihood


def maximize_covariance(scatter: np.ndarray, rank: int, floor: float) -> tuple[np.ndarray, float]:
    """The PPCA-constrained M-step of a scatter matrix: ``(W, tau2)`` with ``tau2 >= floor``."""
    values, vectors = np.linalg.eigh(scatter)
    values, vectors = values[::-1], vectors[:, ::-1]
    tau2 = max(float(np.mean(values[rank:])), float(floor))
    return vectors[:, :rank] * np.sqrt(np.maximum(values[:rank] - tau2, 0.0)), tau2


def canonical_gauge(W: np.ndarray) -> np.ndarray:
    """``W`` with orthogonal columns in descending norm, each with its largest-magnitude entry positive."""
    left, values, _ = np.linalg.svd(W, full_matrices=False)
    out = left * values
    signs = np.sign(out[np.argmax(np.abs(left), axis=0), np.arange(W.shape[1])])
    return out * np.where(signs == 0.0, 1.0, signs)


def _initial(y, P, rank: int, floor: float, seed: int | None) -> tuple[np.ndarray, np.ndarray, float]:
    """The zero-depth lift's mean and scatter, an initialisation only: its depth is not a measurement. ``seed`` rotates
    the start to test for local optima."""
    lifted = np.concatenate([y[:, :ROTATION_FEATURES], np.einsum("nji,nj->ni", P, y[:, ROTATION_FEATURES:])], axis=1)
    mean = lifted.mean(axis=0)
    W, tau2 = maximize_covariance(np.cov(lifted.T), rank, floor)
    if seed is not None:
        rotation = np.linalg.qr(np.random.default_rng(seed).standard_normal((rank, rank)))[0]
        W = W @ rotation + 0.1 * np.sqrt(tau2) * np.random.default_rng(seed + 1).standard_normal(W.shape)
    return mean, W, tau2


def fit_projected_ppca(y, P, R, rank: int, *, floor: float, max_iterations: int = 5000, tolerance: float = 1e-8,
                       seed: int | None = None) -> PPCAFit:
    """EM from the zero-depth lift, until the observed log likelihood per particle changes by less than ``tolerance``.

    ``floor`` is the positive lower bound on ``tau^2`` (square angstroms). The likelihood trace is checked by the
    caller; a likelihood decrease beyond rounding is an error here, because EM cannot decrease it."""
    if floor <= 0.0:
        raise ValueError("tau^2 needs a positive floor")
    y, P, R = (np.asarray(v, dtype=np.float64) for v in (y, P, R))
    mean, W, tau2 = _initial(y, P, rank, floor, seed)
    trace, converged = [], False
    for _ in range(max_iterations):
        new_mean, scatter, log_likelihood = _moments(y, P, R, mean, W @ W.T + tau2 * np.eye(FEATURES))
        if trace and log_likelihood < trace[-1] - 1e-7 * abs(trace[-1]):
            raise ArithmeticError(f"EM decreased the likelihood from {trace[-1]} to {log_likelihood}")
        if trace and abs(log_likelihood - trace[-1]) < tolerance * len(y):
            trace.append(log_likelihood)
            converged = True
            break
        trace.append(log_likelihood)
        mean = new_mean
        W, tau2 = maximize_covariance(scatter, rank, floor)
    return PPCAFit(mean, canonical_gauge(W), tau2, float(floor), np.asarray(trace), converged)


def log_likelihood(y, P, R, fit: PPCAFit) -> float:
    return float(np.sum(per_particle_log_likelihood(y, P, R, fit.mean, fit.covariance)))


def latent_posterior(y, P, R, fit: PPCAFit) -> LatentPosterior:
    """The latent posterior of every particle through ``T_i`` (module docstring)."""
    y, P, R = (np.asarray(v, dtype=np.float64) for v in (y, P, R))
    n, q, dim = len(y), fit.rank, y.shape[1]
    out = dict(
        z_map=np.empty((n, q)), covariance=np.empty((n, q, q)), information=np.empty((n, q, q)), linear=np.empty((n, q)),
        psi=np.empty((n, dim, dim)), structured_mean=np.empty((n, FEATURES)), feature_mean=np.empty((n, FEATURES)),
    )
    covariance = fit.covariance
    for b in _batches(n):
        A = operators(P[b])
        At = np.swapaxes(A, 1, 2)
        AW = A @ fit.W
        residual = (y[b] - A @ fit.mean)[:, :, None]
        solve, _ = _spd_solver(A @ covariance @ At + R[b])
        solved = solve(residual)
        cross = np.swapaxes(AW, 1, 2)
        out["z_map"][b] = (cross @ solved)[:, :, 0]
        out["covariance"][b] = np.eye(q) - cross @ solve(AW)
        out["feature_mean"][b] = fit.mean + (covariance @ At @ solved)[:, :, 0]
        psi = fit.tau2 * A @ At + R[b]
        psi_solve, _ = _spd_solver(psi)
        out["psi"][b] = psi
        out["information"][b] = cross @ psi_solve(AW)
        out["linear"][b] = (cross @ psi_solve(residual))[:, :, 0]
    out["structured_mean"] = fit.mean + out["z_map"] @ fit.W.T
    return LatentPosterior(**out)


def withheld_prediction(y, P, R, fit: PPCAFit, coordinate: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and variance of observed coordinate ``coordinate`` of each particle given its others, from ``T_i``: the
    coordinate is hidden before inference, and the Schur complement carries every noise cross-covariance."""
    y, P, R = (np.asarray(v, dtype=np.float64) for v in (y, P, R))
    keep = np.delete(np.arange(y.shape[1]), coordinate)
    mean, variance = np.empty(len(y)), np.empty(len(y))
    covariance = fit.covariance
    for b in _batches(len(y)):
        A = operators(P[b])
        T = A @ covariance @ np.swapaxes(A, 1, 2) + R[b]
        residual = y[b] - A @ fit.mean
        solve, _ = _spd_solver(T[:, keep[:, None], keep])
        cross = T[:, coordinate, keep][:, :, None]
        solved = solve(np.concatenate([residual[:, keep][:, :, None], cross], axis=2))
        mean[b] = (A @ fit.mean)[:, coordinate] + np.sum(cross[:, :, 0] * solved[:, :, 0], axis=1)
        variance[b] = T[:, coordinate, coordinate] - np.sum(cross[:, :, 0] * solved[:, :, 1], axis=1)
    return mean, variance


def with_floor(fit: PPCAFit, floor: float) -> PPCAFit:
    """``fit`` read with another declared floor, for reporting."""
    return replace(fit, floor=float(floor))
