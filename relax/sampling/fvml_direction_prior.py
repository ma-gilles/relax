"""The FvML-mixture direction prior of a refinement: one half's fit to its posterior mass over the directions.

RELION's learned direction prior (``pdf_direction``) is the normalized sum of a half's posterior weights over the
HEALPix directions of its sampling order. ``relax refine --fvml-direction-prior N``
(``relax.parity.fvml_prior_source``) replaces it, after every numbered iteration, with a uniform component plus ``N``
FvML lobes fitted to that sum (``docs/math/fvml_mixture_direction_prior.md``, sections 10 and 11). The fit:

* the cell-mass likelihood (``fit_fvml_mixture_cells``) on the sum, added up to order ``min(order, FIT_MAX_ORDER)``
  (the NEST children of a direction are its finer pixels, so the coarser sum is exact), each cell represented by its
  ``4**CELL_LEVELS`` children;
* concentrations capped at ``kappa_max_for_grid`` of the fit grid with ``c = CAP_SPACINGS``;
* a half's first fit is the best of ``N_COLD_STARTS`` k-means++ starts (seeded by the half), every later fit starts
  from the half's previous one (with its concentrations clipped to the cap).

The prior then holds every direction's cell probability under the mixture, on whatever HEALPix order a search
samples: there is no reset at an order change. C1 only (the directions are the whole sphere's NEST pixels).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from relax.sampling.fvml_mixture import (
    FvmlMixture,
    cell_log_probabilities,
    fit_fvml_mixture_cells,
    healpix_cell_points,
    init_fvml_mixture,
    kappa_max_for_grid,
)
from relax.sampling.grids import rotation_grid_size

logger = logging.getLogger(__name__)

FIT_MAX_ORDER = 5  # finer sums are fitted here: lobes narrower than a quarter of 1.8 deg do not matter to a prior
CELL_LEVELS = 2  # a cell is the mean of its 16 children
CAP_SPACINGS = 0.25  # kappa_max = 1 / (c delta)^2 with delta the fit grid's spacing
N_COLD_STARTS = 5
MAX_EM_ITERATIONS = 1000
TOL = 1e-7  # objective gain per particle (nats) at which a fit stops
STARTS = ("cold", "warm")


@dataclass(frozen=True, eq=False)
class FvmlPriorFit:
    """One half's fitted mixture and how it was fitted.

    ``fit_order`` is the HEALPix order of the cell masses it was fitted to and ``kappa_max`` the cap there; ``start``
    is ``"cold"`` or ``"warm"``; ``em_iterations``, ``converged`` and ``objective`` (log-likelihood per unit mass,
    nats per particle) describe the EM run (all of the cold starts' iterations), ``seconds`` its wall time.
    """

    mixture: FvmlMixture
    fit_order: int
    kappa_max: float
    start: str
    em_iterations: int
    converged: bool
    objective: float
    seconds: float

    @property
    def n_lobes(self) -> int:
        """Lobes besides the uniform component."""
        return int(self.mixture.weights.size) - 1


def fit_direction_prior(
    masses: np.ndarray, order: int, *, n_lobes: int, previous: FvmlPriorFit | None, seed: int
) -> FvmlPriorFit:
    """Fit the mixture to ``masses``, a half's posterior mass on each C1 direction of HEALPix ``order`` (NEST).

    From ``previous`` when given (warm), else the best of ``N_COLD_STARTS`` k-means++ starts drawn with ``seed``.
    The masses are normalized to sum to one, so the stopping rule is per particle.
    """
    masses = np.asarray(masses, dtype=np.float64).reshape(-1)
    if masses.size != 12 * 4**order:
        raise ValueError(f"an FvML direction prior is fitted to the C1 directions of order {order}, got {masses.size}")
    if not np.all(np.isfinite(masses)) or np.any(masses < 0.0) or masses.sum() <= 0.0:
        raise ValueError("the posterior masses of a direction prior must be finite, non-negative and not all zero")
    fit_order = min(order, FIT_MAX_ORDER)
    masses = masses.reshape(-1, 4 ** (order - fit_order)).sum(axis=1)
    masses = masses / masses.sum()
    kappa_max = kappa_max_for_grid(masses.size, CAP_SPACINGS)
    cells = healpix_cell_points(fit_order, CELL_LEVELS)
    options = {"kappa_max": kappa_max, "max_iter": MAX_EM_ITERATIONS, "tol": TOL}
    began = time.perf_counter()
    if previous is not None:
        if previous.n_lobes != n_lobes:
            raise ValueError(f"a warm start has {previous.n_lobes} lobes, the prior {n_lobes}")
        start = previous.mixture
        start = FvmlMixture(start.weights, start.means, np.minimum(start.kappas, kappa_max), start.symmetry)
        fit = fit_fvml_mixture_cells(cells, masses, start, **options)
        em_iterations = fit.n_iter
    else:
        rng = np.random.default_rng(seed)
        directions = healpix_cell_points(fit_order, 0)[:, 0]
        fits = [
            fit_fvml_mixture_cells(
                cells, masses, init_fvml_mixture(directions, masses, n_lobes + 1, kappa_max=kappa_max, rng=rng),
                **options,
            )
            for _ in range(N_COLD_STARTS)
        ]
        fit = max(fits, key=lambda f: f.objective[-1])
        em_iterations = sum(f.n_iter for f in fits)
    return FvmlPriorFit(
        mixture=fit.mixture,
        fit_order=fit_order,
        kappa_max=kappa_max,
        start="warm" if previous is not None else "cold",
        em_iterations=em_iterations,
        converged=bool(fit.converged),
        objective=float(fit.objective[-1]),
        seconds=time.perf_counter() - began,
    )


def direction_log_probabilities(fit: FvmlPriorFit, order: int) -> np.ndarray:
    """Log probability of each C1 direction of HEALPix ``order`` (NEST) under the fitted mixture."""
    return cell_log_probabilities(fit.mixture, order, CELL_LEVELS)


def rotation_log_prior(fit: FvmlPriorFit, order: int, *, dtype: np.dtype) -> np.ndarray:
    """The prior's log on every rotation of the C1 grid of HEALPix ``order``: rotation ``r`` lies on direction
    ``r % n_directions`` (``make_relion_direction_log_prior``'s layout), in ``dtype``."""
    log_direction = direction_log_probabilities(fit, order).astype(dtype)
    return np.tile(log_direction, rotation_grid_size(order, "C1") // log_direction.size)


def log_fit(half_index: int, order: int, fit: FvmlPriorFit) -> None:
    """One log line per half: the fit's start, EM run and components."""
    lobes = ", ".join(
        f"({w:.3f}, rot {r:.1f}, tilt {t:.1f}, kappa {k:.1f})"
        for w, (r, t), k in zip(fit.mixture.weights[1:], _rot_tilt_deg(fit.mixture.means), fit.mixture.kappas)
    )
    logger.info(
        "FvML direction prior half-%d: %s fit at order %d (masses at order %d), %d EM iterations%s, %.1f s, "
        "objective %.6f per particle; uniform %.3f, lobes (weight, rot, tilt, kappa) %s",
        half_index + 1, fit.start, fit.fit_order, order, fit.em_iterations, "" if fit.converged else " (not converged)",
        fit.seconds, fit.objective, fit.mixture.weights[0], lobes,
    )


def _rot_tilt_deg(means: np.ndarray) -> np.ndarray:
    """RELION's (rot, tilt) in degrees of unit viewing directions ``(n, 3)``: ``v = (cos rot sin tilt, sin rot sin
    tilt, cos tilt)``."""
    rot = np.degrees(np.arctan2(means[:, 1], means[:, 0]))
    tilt = np.degrees(np.arccos(np.clip(means[:, 2], -1.0, 1.0)))
    return np.stack([rot, tilt], axis=1)


# The model STAR blocks: the fit's facts as a list, its components as a loop (component 0 is the uniform one).
STAR_FIT_BLOCK = "relax_direction_prior_fvml"
STAR_COMPONENT_BLOCK = "relax_direction_prior_fvml_components"


def star_fit_items(fit: FvmlPriorFit) -> list[tuple[str, object]]:
    """The labels and values of the fit's list block."""
    return [
        ("relax_fvml_lobes", fit.n_lobes),
        ("relax_fvml_fit_order", fit.fit_order),
        ("relax_fvml_kappa_max", float(fit.kappa_max)),
        ("relax_fvml_start", fit.start),
        ("relax_fvml_em_iterations", fit.em_iterations),
        ("relax_fvml_converged", fit.converged),
        ("relax_fvml_objective", fit.objective),
        ("relax_fvml_seconds", float(fit.seconds)),
    ]


def star_component_columns(fit: FvmlPriorFit) -> dict[str, object]:
    """The columns of the components' loop block: weight, mean direction as RELION's (rot, tilt) and as a unit
    vector, and concentration (0 for the uniform component, whose direction columns are 0)."""
    mixture = fit.mixture
    means = np.vstack([np.zeros(3), mixture.means])
    angles = np.vstack([np.zeros(2), _rot_tilt_deg(mixture.means)])
    return {
        "relax_fvml_component": np.arange(mixture.weights.size),
        "relax_fvml_weight": mixture.weights,
        "relax_fvml_rot_deg": angles[:, 0],
        "relax_fvml_tilt_deg": angles[:, 1],
        "relax_fvml_kappa": np.concatenate([[0.0], mixture.kappas]),
        "relax_fvml_mean_x": means[:, 0],
        "relax_fvml_mean_y": means[:, 1],
        "relax_fvml_mean_z": means[:, 2],
    }


def fit_from_star(items: dict, columns: dict) -> FvmlPriorFit:
    """The fit a model STAR's two blocks hold (values as strings, as ``read_star_blocks`` returns them)."""
    weights = np.asarray(columns["relax_fvml_weight"], dtype=np.float64)
    means = np.stack([np.asarray(columns[f"relax_fvml_mean_{a}"], dtype=np.float64) for a in "xyz"], axis=1)[1:]
    kappas = np.asarray(columns["relax_fvml_kappa"], dtype=np.float64)[1:]
    start = items["relax_fvml_start"]
    if start not in STARTS:
        raise ValueError(f"{STAR_FIT_BLOCK}: unknown start {start!r}")
    return FvmlPriorFit(
        mixture=FvmlMixture(weights / weights.sum(), means / np.linalg.norm(means, axis=1, keepdims=True), kappas),
        fit_order=int(items["relax_fvml_fit_order"]),
        kappa_max=float(items["relax_fvml_kappa_max"]),
        start=start,
        em_iterations=int(items["relax_fvml_em_iterations"]),
        converged=bool(int(float(items["relax_fvml_converged"]))),
        objective=float(items["relax_fvml_objective"]),
        seconds=float(items["relax_fvml_seconds"]),
    )
