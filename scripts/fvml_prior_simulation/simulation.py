"""Building blocks of the FvML-mixture prior simulation: grids, targets, sampling, the fits and their distances.

A target is a viewing-direction distribution on the sphere: either a known FvML mixture (the ground truth of a
recovery test) or a probability vector on a HEALPix grid taken from a refinement run (density uniform inside each
cell). A trial draws ``n`` directions from it, bins them on the fit grid (the masses an E-step with perfect poses
would give), and estimates the distribution three ways: the histogram (RELION's ``pdf_direction`` update), the
point-atom FvML fit and the cell-mass FvML fit (``relax/sampling/fvml_mixture.py``). Directions follow RELION's
HEALPix convention (NEST ordering, a pixel's vector is the viewing direction of its (rot, tilt)).
"""

from __future__ import annotations

import json
import math
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import healpy as hp
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from relax.sampling.fvml_mixture import (  # noqa: E402
    FitResult,
    FvmlMixture,
    fit_fvml_mixture,
    fit_fvml_mixture_cells,
    init_fvml_mixture,
    kappa_max_for_grid,
    log_density,
)
from scripts.pose_prior_benchmark.evaluate_run import distribution_metrics  # noqa: E402

TRUTH_QUADRATURE_LEVELS = 3  # at least this many finer HEALPix orders integrate a mixture over one cell
QUADRATURE_ORDER = 8  # and at least this order (0.23 deg spacing)
WITHIN_CELL_LEVELS = 12  # finer orders whose pixels place a draw uniformly inside a cell of a probability vector
METHODS = ("histogram", "point", "cell")


def sphere_grid(order: int) -> np.ndarray:
    """HEALPix pixel centres (NEST) at ``order`` as unit vectors, shape ``(12 * 4**order, 3)``."""
    nside = 1 << order
    return np.stack(hp.pix2vec(nside, np.arange(hp.nside2npix(nside)), nest=True), axis=1)


def cell_children(order: int, levels: int) -> np.ndarray:
    """Centres of the ``4**levels`` NEST children of every pixel at ``order``, shape ``(12 * 4**order, 4**levels, 3)``."""
    return sphere_grid(order + levels).reshape(12 * 4**order, 4**levels, 3)


def coarsen(pmf: np.ndarray, levels: int) -> np.ndarray:
    """Sum a NEST probability vector over groups of ``4**levels`` children: the vector ``levels`` orders coarser."""
    return pmf.reshape(-1, 4**levels).sum(axis=1)


def spread(pmf: np.ndarray, levels: int) -> np.ndarray:
    """Share each cell's probability equally among its ``4**levels`` NEST children (density constant in a cell)."""
    return np.repeat(pmf / 4**levels, 4**levels)


def bin_directions(points: np.ndarray, order: int) -> np.ndarray:
    """Count ``(n, 3)`` unit vectors per HEALPix (NEST) pixel at ``order``."""
    nside = 1 << order
    pixels = hp.vec2pix(nside, points[:, 0], points[:, 1], points[:, 2], nest=True)
    return np.bincount(pixels, minlength=hp.nside2npix(nside)).astype(np.float64)


def mixture_cell_probabilities(mixture: FvmlMixture, order: int) -> np.ndarray:
    """Probability of each HEALPix cell at ``order`` under the mixture, normalized to sum to 1.

    The mean density over the cell's children at order ``max(order + 3, QUADRATURE_ORDER)`` (midpoint rule; its
    error falls 4x per order).
    """
    levels = max(TRUTH_QUADRATURE_LEVELS, QUADRATURE_ORDER - order)
    log_values = log_density(mixture, sphere_grid(order + levels)).reshape(-1, 4**levels)
    log_cell = logsumexp(log_values, axis=1)
    return np.exp(log_cell - logsumexp(log_cell))


def _orthonormal_frame(mean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors completing ``mean`` to an orthonormal frame."""
    helper = np.array([1.0, 0.0, 0.0]) if abs(mean[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    first = np.cross(mean, helper)
    first /= np.linalg.norm(first)
    return first, np.cross(mean, first)


def sample_fvml(rng: np.random.Generator, n: int, mean: np.ndarray, kappa: float) -> np.ndarray:
    """``n`` FvML draws about the unit ``mean`` (inverse CDF of the cosine, uniform azimuth), shape ``(n, 3)``."""
    u = rng.random(n)
    if kappa > 0.0:
        cosine = 1.0 + np.log(u + (1.0 - u) * np.exp(-2.0 * kappa)) / kappa
    else:
        cosine = 2.0 * u - 1.0
    azimuth = 2.0 * math.pi * rng.random(n)
    sine = np.sqrt(np.clip(1.0 - cosine**2, 0.0, None))
    first, second = _orthonormal_frame(mean)
    return (
        (sine * np.cos(azimuth))[:, None] * first + (sine * np.sin(azimuth))[:, None] * second + cosine[:, None] * mean
    )


def sample_mixture(mixture: FvmlMixture, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` draws from the mixture (a lobe's draw takes one of its symmetry images at random), shape ``(n, 3)``."""
    counts = rng.multinomial(n, mixture.weights)
    parts = [sample_fvml(rng, int(counts[0]), np.array([0.0, 0.0, 1.0]), 0.0)]
    n_g = mixture.symmetry.shape[0]
    for j in range(mixture.kappas.size):
        images = rng.multinomial(int(counts[j + 1]), np.full(n_g, 1.0 / n_g))
        for g, count in zip(mixture.symmetry, images):
            parts.append(sample_fvml(rng, int(count), g @ mixture.means[j], float(mixture.kappas[j])))
    points = np.concatenate(parts)
    return points[rng.permutation(n)]


def sample_cells(pmf: np.ndarray, order: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` draws from a probability vector at ``order`` whose density is uniform inside each cell, shape ``(n, 3)``.

    A draw picks a cell by the vector, then one of its ``4**WITHIN_CELL_LEVELS`` equal-area NEST children.
    """
    cells = np.repeat(np.arange(pmf.size), rng.multinomial(n, pmf / pmf.sum()))
    children = 4**WITHIN_CELL_LEVELS
    pixels = cells.astype(np.int64) * children + rng.integers(0, children, size=n)
    points = np.stack(hp.pix2vec(1 << (order + WITHIN_CELL_LEVELS), pixels, nest=True), axis=1)
    return points[rng.permutation(n)]


@dataclass(frozen=True)
class Target:
    """A viewing-direction distribution to sample: a mixture, or a probability vector at ``order`` (``pmf``).

    ``symmetry`` is the ``(n_g, 3, 3)`` group the fits use; ``n_components`` the fits' component count (the uniform
    one included); ``eval_order`` the HEALPix order of the fine comparison; ``source`` says where it came from.
    """

    name: str
    symmetry: np.ndarray
    n_components: int
    eval_order: int
    source: str
    mixture: FvmlMixture | None = None
    pmf: np.ndarray | None = None
    order: int | None = None

    def truth(self, order: int) -> np.ndarray:
        """The target's probability of each HEALPix cell at ``order``."""
        if self.mixture is not None:
            return mixture_cell_probabilities(self.mixture, order)
        if order <= self.order:
            return coarsen(self.pmf, self.order - order)
        return spread(self.pmf, order - self.order)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """``n`` directions drawn from the target, shape ``(n, 3)``."""
        if self.mixture is not None:
            return sample_mixture(self.mixture, n, rng)
        return sample_cells(self.pmf, self.order, n, rng)


def save_target(target: Target, path: Path) -> None:
    """Write a target to an ``.npz`` file (read back by :func:`load_target`)."""
    arrays: dict[str, Any] = {
        "name": target.name,
        "symmetry": target.symmetry,
        "n_components": target.n_components,
        "eval_order": target.eval_order,
        "source": target.source,
    }
    if target.mixture is not None:
        arrays.update(weights=target.mixture.weights, means=target.mixture.means, kappas=target.mixture.kappas)
    else:
        arrays.update(pmf=target.pmf, order=target.order)
    np.savez(path, **arrays)


def load_target(path: Path) -> Target:
    """Read a target written by :func:`save_target`."""
    data = np.load(path)
    common = dict(
        name=str(data["name"]),
        symmetry=data["symmetry"],
        n_components=int(data["n_components"]),
        eval_order=int(data["eval_order"]),
        source=str(data["source"]),
    )
    if "weights" in data:
        mixture = FvmlMixture(data["weights"], data["means"], data["kappas"], data["symmetry"])
        return Target(**common, mixture=mixture)
    return Target(**common, pmf=data["pmf"], order=int(data["order"]))


def fit_best_of_starts(
    order: int,
    masses: np.ndarray,
    starts: list[FvmlMixture],
    *,
    kappa_max: float,
    cell_levels: int | None,
    max_iter: int,
    tol: float,
) -> tuple[FitResult, int]:
    """Fit from every start and keep the largest final objective; returns the fit and its start's index.

    ``cell_levels`` None is the point-atom fit on the pixel centres, an integer the cell-mass fit whose cells are
    represented by ``4**cell_levels`` children.
    """
    if cell_levels is None:
        directions = sphere_grid(order)
        fits = [
            fit_fvml_mixture(directions, masses, s, kappa_max=kappa_max, max_iter=max_iter, tol=tol) for s in starts
        ]
    else:
        cells = cell_children(order, cell_levels)
        fits = [
            fit_fvml_mixture_cells(cells, masses, s, kappa_max=kappa_max, max_iter=max_iter, tol=tol) for s in starts
        ]
    best = int(np.argmax([fit.objective[-1] for fit in fits]))
    return fits[best], best


def lobe_errors(fitted: FvmlMixture, truth: FvmlMixture) -> dict[str, Any]:
    """Match fitted lobes to true lobes (least total angle, best over the symmetry images) and report their errors.

    Only for equal lobe counts. Angles in degrees; ``kappa_ratio`` is fitted over true.
    """
    if fitted.kappas.size != truth.kappas.size:
        return {}
    images = np.einsum("gab,jb->jga", truth.symmetry, truth.means)
    cosines = np.einsum("kb,jgb->kjg", fitted.means, images).max(axis=2)
    angles = np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))
    rows, cols = linear_sum_assignment(angles)
    order = np.argsort(cols)
    matched = rows[order]
    return {
        "uniform_weight_error": float(fitted.weights[0] - truth.weights[0]),
        "lobe_weight_error": (fitted.weights[1:][matched] - truth.weights[1:]).tolist(),
        "lobe_angle_error_deg": angles[matched, np.arange(truth.kappas.size)].tolist(),
        "kappa_ratio": (fitted.kappas[matched] / truth.kappas).tolist(),
    }


def mixture_record(mixture: FvmlMixture) -> dict[str, list]:
    """The fitted parameters as JSON-ready lists."""
    return {
        "weights": mixture.weights.tolist(),
        "means": mixture.means.tolist(),
        "kappas": mixture.kappas.tolist(),
    }


def run_trial(
    target: Target,
    n_particles: int | None,
    replicate: int,
    orders: list[int],
    *,
    n_components: int,
    n_starts: int,
    sigma_in_spacings: float,
    cell_levels: int,
    max_iter: int,
    tol: float,
    reference_mass: float,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """One draw of ``n_particles`` directions (None: no draw, the target's own cell probabilities), fitted per order.

    The fits have ``n_components`` components (the uniform one included) and the cap ``kappa_max_for_grid`` with
    ``sigma_in_spacings``. Returns one row per (order, method) with the distances to the truth at the fit order and at
    the target's ``eval_order`` and the held-out score (:func:`heldout_log_likelihood` of a second, independent draw of
    the same size), and the masses per order (for figures). The same k-means++ starts seed both FvML fits.
    """
    seed = np.random.SeedSequence([replicate, n_particles or 0, zlib.crc32(target.name.encode())])
    sample_rng, start_rng, heldout_rng = (np.random.default_rng(s) for s in seed.spawn(3))
    points = None if n_particles is None else target.sample(n_particles, sample_rng)
    heldout = None if n_particles is None else target.sample(n_particles, heldout_rng)
    truth_eval = target.truth(target.eval_order)
    rows, masses_by_order = [], {}
    for order in orders:
        truth_fit = target.truth(order)
        masses = truth_fit * reference_mass if points is None else bin_directions(points, order)
        masses_by_order[f"order{order}"] = masses
        heldout_masses = None if heldout is None else bin_directions(heldout, order)
        n_cells = masses.size
        kappa_max = kappa_max_for_grid(n_cells, sigma_in_spacings)
        directions = sphere_grid(order)
        starts = [
            init_fvml_mixture(
                directions, masses, n_components, kappa_max=kappa_max, rng=start_rng, symmetry=target.symmetry
            )
            for _ in range(n_starts)
        ]
        common = {
            "target": target.name,
            "n_particles": n_particles if n_particles is not None else "inf",
            "replicate": replicate,
            "order": order,
            "n_cells": n_cells,
            "c": sigma_in_spacings,
            "kappa_max": kappa_max,
            "n_components": n_components,
            "heldout_log_likelihood_truth": heldout_log_likelihood(heldout_masses, truth_fit),
        }
        histogram = masses / masses.sum()
        rows.append(
            {
                **common,
                "method": "histogram",
                **_distances(truth_fit, histogram, "fit"),
                **_distances(truth_eval, _to_order(histogram, order, target.eval_order), "eval"),
                "heldout_log_likelihood": heldout_log_likelihood(heldout_masses, histogram),
            }
        )
        for method, levels in (("point", None), ("cell", cell_levels)):
            began = time.perf_counter()
            fit, best_start = fit_best_of_starts(
                order, masses, starts, kappa_max=kappa_max, cell_levels=levels, max_iter=max_iter, tol=tol
            )
            seconds = time.perf_counter() - began
            fitted = mixture_cell_probabilities(fit.mixture, order)
            row = {
                **common,
                "method": method,
                **_distances(truth_fit, fitted, "fit"),
                "heldout_log_likelihood": heldout_log_likelihood(heldout_masses, fitted),
                **_distances(truth_eval, mixture_cell_probabilities(fit.mixture, target.eval_order), "eval"),
                "objective": float(fit.objective[-1]),
                "n_iter": fit.n_iter,
                "converged": bool(fit.converged),
                "best_start": best_start,
                "seconds_all_starts": seconds,
                "lobes_at_cap": int(np.sum(fit.mixture.kappas >= kappa_max * (1.0 - 1e-9))),
                "fit": mixture_record(fit.mixture),
            }
            if target.mixture is not None:
                row.update(lobe_errors(fit.mixture, target.mixture))
            rows.append(row)
    return rows, masses_by_order


def heldout_log_likelihood(heldout_masses: np.ndarray | None, pmf: np.ndarray) -> float:
    """Mean log probability per held-out particle of its cell under ``pmf``; -inf when a held-out cell has none.

    NaN without a held-out draw.
    """
    if heldout_masses is None:
        return math.nan
    occupied = heldout_masses > 0
    if np.any(pmf[occupied] <= 0):
        return -math.inf
    return float(heldout_masses[occupied] @ np.log(pmf[occupied]) / heldout_masses.sum())


def _to_order(pmf: np.ndarray, order: int, to_order: int) -> np.ndarray:
    """A cell probability vector moved from ``order`` to ``to_order`` (summed when coarser, spread when finer)."""
    return coarsen(pmf, order - to_order) if to_order <= order else spread(pmf, to_order - order)


def _distances(truth: np.ndarray, estimate: np.ndarray, suffix: str) -> dict[str, float]:
    """The benchmark's distances from ``estimate`` to ``truth`` (``distribution_metrics``), keys suffixed."""
    metrics = distribution_metrics(truth, estimate)
    return {f"{key}_{suffix}": value for key, value in metrics.items()}


def write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    """Write result rows as JSON lines (non-finite values as the strings "inf", "-inf" and "nan")."""
    with path.open("w") as handle:
        for row in rows:
            clean = {k: (str(v) if isinstance(v, float) and not math.isfinite(v) else v) for k, v in row.items()}
            handle.write(json.dumps(clean) + "\n")
