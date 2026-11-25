"""The FvML-mixture direction prior of a refinement: a fit to a posterior mass over the directions.

RELION's learned direction prior (``pdf_direction``) is the normalized sum of the posterior weights over the HEALPix
directions of its sampling order: one per half in Refine3D, one per class in Class3D. ``relax refine
--fvml-direction-prior N`` (``relax.parity.fvml_prior_source``) replaces each, after every numbered iteration, with
a uniform component plus ``N`` FvML lobes fitted to that sum (``docs/math/fvml_mixture_direction_prior.md``,
sections 10 and 11). The fit:

* the cell-mass likelihood (``fit_fvml_mixture_weighted_cells``) on the sum, folded onto order ``min(order,
  FIT_MAX_ORDER)`` (``fold_masses``), each cell represented by its HEALPix children ``CELL_LEVELS`` orders finer
  (``orbit_cells``);
* concentrations capped at ``kappa_max_for_grid`` of the fit order's whole-sphere grid with ``c = CAP_SPACINGS``;
* a first fit is the best of ``N_COLD_STARTS`` k-means++ starts (seeded by the caller), every later fit starts
  from the previous one (with its concentrations clipped to the cap).

Under a point group RELION samples only the directions of one asymmetric unit (``relax.sampling.healpix``): each
holds the mass of its orbit. The mixture is then invariant under the group (each lobe copied by its rotations,
``FvmlMixture.symmetry``) and ``N`` counts the lobes of one unit. A direction's cell is the set of orbits assigned
to it, which differs from its own pixel along the unit's ragged edge; the cells tile the sphere. The prior holds every
sampled direction's cell probability under the mixture, normalized over the unit, on whatever HEALPix order a
search samples: there is no reset at an order change.
"""

from __future__ import annotations

import functools
import logging
import time
from dataclasses import dataclass

import healpy as hp
import numpy as np

from relax.sampling.fvml_mixture import (
    FvmlMixture,
    fit_fvml_mixture_weighted_cells,
    init_fvml_mixture,
    kappa_max_for_grid,
    weighted_cell_log_probabilities,
)
from relax.sampling.grids import rotation_grid_size
from relax.sampling.healpix import healpix_sampling
from relax.sampling.symmetry import rotational_operators

logger = logging.getLogger(__name__)

FIT_MAX_ORDER = 5  # finer sums are fitted here: lobes narrower than a quarter of 1.8 deg do not matter to a prior
CELL_LEVELS = 2  # a cell is the mean of its 16 children
CAP_SPACINGS = 0.25  # kappa_max = 1 / (c delta)^2 with delta the fit grid's spacing
N_COLD_STARTS = 5
MAX_EM_ITERATIONS = 1000
TOL = 1e-7  # objective gain per particle (nats) at which a fit stops
STARTS = ("cold", "warm")
REGION_RINGS = 2  # neighbour rings around the asymmetric unit's pixels that hold a point of every orbit
AREA_TOL = 0.02  # the cells' quadrature of the sphere's area must be this close to it (C7 at order 2: 1.1%)


@dataclass(frozen=True, eq=False)
class FvmlPriorFit:
    """One fitted mixture and how it was fitted.

    ``symmetry`` is the run's point group (the mixture's lobes are copied by its rotations, and its directions are
    those of one asymmetric unit); ``fit_order`` is the HEALPix order of the cell masses it was fitted to and
    ``kappa_max`` the cap there; ``start`` is ``"cold"`` or ``"warm"``; ``em_iterations``, ``converged`` and
    ``objective`` (log-likelihood per unit mass, nats per particle) describe the EM run (all of the cold starts'
    iterations), ``seconds`` its wall time.
    """

    mixture: FvmlMixture
    symmetry: str
    fit_order: int
    kappa_max: float
    start: str
    em_iterations: int
    converged: bool
    objective: float
    seconds: float

    @property
    def n_lobes(self) -> int:
        """Lobes besides the uniform component (in one asymmetric unit)."""
        return int(self.mixture.weights.size) - 1


def direction_pixels(order: int, symmetry: str) -> np.ndarray:
    """The NEST pixel of each of RELION's directions at HEALPix ``order`` under ``symmetry``, in its order: every
    pixel in C1, one asymmetric unit's otherwise (``relax.sampling.healpix.healpix_sampling``)."""
    return np.asarray(healpix_sampling(order, symmetry)["directions_ipix"], dtype=np.int64)


def symmetry_maps(symmetry: str) -> np.ndarray:
    """The point group's rotations ``(n_g, 3, 3)``, identity first: the maps that copy each lobe. RELION's images of
    a direction ``v`` are ``R^T v`` over the same group (``removeSymmetryEquivalentPoints``), the same set. RELION
    builds some groups from rounded constants (T is orthogonal to 1e-7 only), so each is replaced by its nearest
    rotation."""
    operators = rotational_operators(symmetry)
    u, _, vt = np.linalg.svd(operators)
    maps = u @ vt
    if not np.allclose(maps[0], np.eye(3), rtol=0.0, atol=1e-12):
        raise RuntimeError(f"the {symmetry} operators do not start with the identity")
    return maps


@dataclass(frozen=True, eq=False)
class OrbitCells:
    """The cells of RELION's directions at one order under a point group, in the layout of
    ``fit_fvml_mixture_weighted_cells``.

    ``children`` ``(n, 3)`` are HEALPix pixel centres ``CELL_LEVELS`` orders finer, grouped by direction (direction
    ``m``'s from ``starts[m]``), and ``weights`` ``(n,)`` the number of such pixels each stands for. Direction
    ``m``'s cell is its share of the orbits (``_orbit_shares``); together the cells tile the sphere, and
    ``weights.sum()`` is the finer order's pixel count up to the quadrature (exactly in C1, where a cell is the
    pixel's ``4**CELL_LEVELS`` children with weight 1).
    """

    children: np.ndarray
    weights: np.ndarray
    starts: np.ndarray


def _image_parents(vectors: np.ndarray, parents: np.ndarray, order: int, maps: np.ndarray) -> np.ndarray:
    """``(n_g, n)``: the NEST pixel at ``order`` of each image ``g v`` of ``vectors`` (``(n, 3)``); row 0, the
    identity's, is ``parents``, each point's own pixel."""
    out = np.empty((maps.shape[0], parents.size), dtype=np.int64)
    out[0] = parents
    if maps.shape[0] > 1:
        images = np.einsum("gab,nb->gna", maps[1:], vectors)
        out[1:] = hp.vec2pix(1 << order, images[..., 0], images[..., 1], images[..., 2], nest=True)
    return out


def _orbit_shares(
    vectors: np.ndarray, image_parents: np.ndarray, order: int, symmetry: str, maps: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """How the orbit of each point of ``vectors`` is shared among RELION's directions at ``order``: parallel arrays
    of point index, direction row and share (each point's shares sum to 1).

    Equal shares to the kept pixels its images fall in, counted per image: one pixel inside the asymmetric unit, two
    where RELION keeps a direction and its image (it scores both alike). An orbit that falls in no kept pixel (the
    unit's ragged edge) goes whole to the kept direction nearest to any of its images. Every point of an orbit gets
    the same shares.
    """
    pixels = direction_pixels(order, symmetry)
    row_of_pixel = np.full(12 * 4**order, -1, dtype=np.int64)
    row_of_pixel[pixels] = np.arange(pixels.size)
    rows = row_of_pixel[image_parents]
    kept = rows >= 0
    n_kept = kept.sum(axis=0)
    image_index, point_index = np.nonzero(kept)
    shares = 1.0 / n_kept[point_index]
    point_rows = rows[image_index, point_index]
    stray = np.flatnonzero(n_kept == 0)
    if stray.size:
        centres = np.stack(hp.pix2vec(1 << order, pixels, nest=True), axis=1)
        images = np.einsum("gab,nb->gna", maps, vectors[stray])
        nearest = np.einsum("gna,ka->nkg", images, centres).max(axis=2).argmax(axis=1)
        point_index = np.concatenate([point_index, stray])
        point_rows = np.concatenate([point_rows, nearest])
        shares = np.concatenate([shares, np.ones(stray.size)])
    return point_index, point_rows, shares


def cell_region(order: int, symmetry: str) -> np.ndarray:
    """The pixels at ``order`` whose children sample the cells: the kept directions' pixels and ``REGION_RINGS``
    rings of their neighbours, so that every orbit, those of the gaps along the asymmetric unit's ragged edge
    included, has points in it (every pixel in C1)."""
    region = direction_pixels(order, symmetry)
    for _ in range(REGION_RINGS):
        neighbours = hp.get_all_neighbours(1 << order, region, nest=True).reshape(-1)
        region = np.union1d(region, neighbours[neighbours >= 0])
    return region


def orbit_cells(order: int, symmetry: str) -> OrbitCells:
    """The cells of RELION's directions at HEALPix ``order`` under ``symmetry`` (``OrbitCells``).

    In C1 each pixel's ``4**CELL_LEVELS`` NEST children with weight 1 (what the construction below gives for the
    trivial group, built directly: at order 8 there are 12.6 million). Otherwise ``_point_group_cells``, cached.
    """
    if symmetry_maps(symmetry).shape[0] == 1:
        n_pixels = 12 * 4**order
        children = np.stack(hp.pix2vec(1 << (order + CELL_LEVELS), np.arange(n_pixels * 4**CELL_LEVELS), nest=True), 1)
        return OrbitCells(
            children=children, weights=np.ones(children.shape[0]), starts=np.arange(n_pixels) * 4**CELL_LEVELS
        )
    return _point_group_cells(order, symmetry)


@functools.lru_cache(maxsize=8)
def _point_group_cells(order: int, symmetry: str) -> OrbitCells:
    """The cells of a non-trivial point group (read-only arrays).

    The children of ``cell_region``'s pixels; a child with ``k`` of its ``n_g`` images in the region stands for
    ``n_g / k`` children of the sphere, split among the directions by ``_orbit_shares``. For an invariant density
    ``f`` the weighted sum over the region's children is then the midpoint rule for the integral of ``f`` over the
    sphere (each orbit counted once), and over one cell's children the integral over its share of the orbits.
    """
    maps = symmetry_maps(symmetry)
    n_sphere_children = 12 * 4 ** (order + CELL_LEVELS)
    region = cell_region(order, symmetry)
    in_region = np.zeros(12 * 4**order, dtype=bool)
    in_region[region] = True
    children = ((region[:, None] << (2 * CELL_LEVELS)) + np.arange(4**CELL_LEVELS)[None, :]).reshape(-1)
    vectors = np.stack(hp.pix2vec(1 << (order + CELL_LEVELS), children, nest=True), axis=1)
    image_parents = _image_parents(vectors, children >> (2 * CELL_LEVELS), order, maps)
    weights = maps.shape[0] / in_region[image_parents].sum(axis=0)
    if abs(weights.sum() / n_sphere_children - 1.0) > AREA_TOL:
        raise RuntimeError(
            f"the {symmetry} cells at order {order} cover {weights.sum() / n_sphere_children:.4f} of the sphere"
        )
    point_index, rows, shares = _orbit_shares(vectors, image_parents, order, symmetry, maps)
    n_directions = direction_pixels(order, symmetry).size
    by_row = np.argsort(rows, kind="stable")
    point_index, rows, shares = point_index[by_row], rows[by_row], shares[by_row]
    counts = np.bincount(rows, minlength=n_directions)
    if np.any(counts == 0):
        raise RuntimeError(f"{int(np.sum(counts == 0))} {symmetry} directions at order {order} have an empty cell")
    cells = OrbitCells(
        children=vectors[point_index], weights=weights[point_index] * shares, starts=np.cumsum(counts) - counts
    )
    for array in (cells.children, cells.weights, cells.starts):
        array.setflags(write=False)
    return cells


@functools.lru_cache(maxsize=8)
def _fold_fractions(order: int, fit_order: int, symmetry: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """How each of RELION's directions at ``order`` splits among those of the coarser ``fit_order``: parallel
    read-only arrays of fine row, coarse row and fraction (each fine row's fractions sum to 1).

    A fine cell's (``orbit_cells``) children go to the coarse cells by ``_orbit_shares`` at ``fit_order``, each with
    its weight: the fine cell's mass is taken as spread over it like its area.
    """
    cells = orbit_cells(order, symmetry)
    maps = symmetry_maps(symmetry)
    child_pixels = hp.vec2pix(
        1 << fit_order, cells.children[:, 0], cells.children[:, 1], cells.children[:, 2], nest=True
    )
    image_parents = _image_parents(cells.children, child_pixels, fit_order, maps)
    child_index, coarse_rows, shares = _orbit_shares(cells.children, image_parents, fit_order, symmetry, maps)
    fine_of_child = np.repeat(np.arange(cells.starts.size), np.diff(np.append(cells.starts, cells.weights.size)))
    fine_rows = fine_of_child[child_index]
    n_coarse = direction_pixels(fit_order, symmetry).size
    pairs, inverse = np.unique(fine_rows * n_coarse + coarse_rows, return_inverse=True)
    amounts = np.bincount(inverse, weights=cells.weights[child_index] * shares)
    fine, coarse = pairs // n_coarse, pairs % n_coarse
    fractions = amounts / np.bincount(fine, weights=amounts, minlength=cells.starts.size)[fine]
    for array in (fine, coarse, fractions):
        array.setflags(write=False)
    return fine, coarse, fractions


def fold_masses(masses: np.ndarray, order: int, fit_order: int, symmetry: str) -> np.ndarray:
    """The posterior ``masses`` on RELION's directions at ``order`` summed onto those of the coarser ``fit_order``,
    each split over the coarse cells its cell overlaps (``_fold_fractions``); in C1, onto its NEST parent."""
    if fit_order > order:
        raise ValueError(f"directions of order {order} cannot fold onto the finer order {fit_order}")
    if fit_order == order:
        return np.asarray(masses, dtype=np.float64).copy()
    fine, coarse, fractions = _fold_fractions(order, fit_order, symmetry)
    n_coarse = direction_pixels(fit_order, symmetry).size
    return np.bincount(coarse, weights=np.asarray(masses, dtype=np.float64)[fine] * fractions, minlength=n_coarse)


def fit_direction_prior(
    masses: np.ndarray, order: int, *, symmetry: str, n_lobes: int, previous: FvmlPriorFit | None, seed: int
) -> FvmlPriorFit:
    """Fit the mixture to ``masses``, a posterior mass on each of RELION's directions of HEALPix ``order`` under
    ``symmetry`` (in the order of ``direction_pixels``).

    From ``previous`` when given (warm), else the best of ``N_COLD_STARTS`` k-means++ starts drawn with ``seed``.
    The masses are normalized to sum to one, so the stopping rule is per particle.
    """
    masses = np.asarray(masses, dtype=np.float64).reshape(-1)
    n_directions = direction_pixels(order, symmetry).size
    if masses.size != n_directions:
        raise ValueError(
            f"an FvML direction prior is fitted to the {n_directions} {symmetry} directions of order {order}, "
            f"got {masses.size}"
        )
    if not np.all(np.isfinite(masses)) or np.any(masses < 0.0) or masses.sum() <= 0.0:
        raise ValueError("the posterior masses of a direction prior must be finite, non-negative and not all zero")
    fit_order = min(order, FIT_MAX_ORDER)
    pixels = direction_pixels(fit_order, symmetry)
    masses = fold_masses(masses, order, fit_order, symmetry)
    masses = masses / masses.sum()
    kappa_max = kappa_max_for_grid(12 * 4**fit_order, CAP_SPACINGS)
    cells = orbit_cells(fit_order, symmetry)
    layout = (cells.children, cells.weights, cells.starts)
    options = {"kappa_max": kappa_max, "max_iter": MAX_EM_ITERATIONS, "tol": TOL}
    began = time.perf_counter()
    if previous is not None:
        if previous.n_lobes != n_lobes:
            raise ValueError(f"a warm start has {previous.n_lobes} lobes, the prior {n_lobes}")
        if previous.symmetry != symmetry:
            raise ValueError(f"a warm start has symmetry {previous.symmetry}, the prior {symmetry}")
        start = previous.mixture
        start = FvmlMixture(start.weights, start.means, np.minimum(start.kappas, kappa_max), start.symmetry)
        fit = fit_fvml_mixture_weighted_cells(*layout, masses, start, **options)
        em_iterations = fit.n_iter
    else:
        rng = np.random.default_rng(seed)
        directions = np.stack(hp.pix2vec(1 << fit_order, pixels, nest=True), axis=1)
        maps = symmetry_maps(symmetry)
        fits = [
            fit_fvml_mixture_weighted_cells(
                *layout,
                masses,
                init_fvml_mixture(directions, masses, n_lobes + 1, kappa_max=kappa_max, rng=rng, symmetry=maps),
                **options,
            )
            for _ in range(N_COLD_STARTS)
        ]
        fit = max(fits, key=lambda f: f.objective[-1])
        em_iterations = sum(f.n_iter for f in fits)
    return FvmlPriorFit(
        mixture=fit.mixture,
        symmetry=symmetry,
        fit_order=fit_order,
        kappa_max=kappa_max,
        start="warm" if previous is not None else "cold",
        em_iterations=em_iterations,
        converged=bool(fit.converged),
        objective=float(fit.objective[-1]),
        seconds=time.perf_counter() - began,
    )


def direction_log_probabilities(fit: FvmlPriorFit, order: int) -> np.ndarray:
    """Log probability of each of RELION's directions of HEALPix ``order`` under the fit's symmetry (in the order of
    ``direction_pixels``): the mixture's mass on the direction's cell (``orbit_cells``), normalized over them."""
    cells = orbit_cells(order, fit.symmetry)
    return weighted_cell_log_probabilities(fit.mixture, cells.children, cells.weights, cells.starts)


def rotation_log_prior(fit: FvmlPriorFit, order: int, *, dtype: np.dtype) -> np.ndarray:
    """The prior's log on every rotation of the grid of HEALPix ``order`` under the fit's symmetry: rotation ``r``
    lies on direction ``r % n_directions`` (``make_relion_direction_log_prior``'s layout), in ``dtype``."""
    log_direction = direction_log_probabilities(fit, order).astype(dtype)
    return np.tile(log_direction, rotation_grid_size(order, fit.symmetry) // log_direction.size)


def log_fit(label: str, order: int, fit: FvmlPriorFit) -> None:
    """One log line per fit (``label``: ``"half-1"``, ``"class 2"``): its start, EM run and components."""
    lobes = ", ".join(
        f"({w:.3f}, rot {r:.1f}, tilt {t:.1f}, kappa {k:.1f})"
        for w, (r, t), k in zip(fit.mixture.weights[1:], _rot_tilt_deg(fit.mixture.means), fit.mixture.kappas)
    )
    logger.info(
        "FvML direction prior %s: %s %s fit at order %d (masses at order %d), %d EM iterations%s, %.1f s, "
        "objective %.6f per particle; uniform %.3f, lobes (weight, rot, tilt, kappa) %s",
        label, fit.start, fit.symmetry, fit.fit_order, order, fit.em_iterations,
        "" if fit.converged else " (not converged)", fit.seconds, fit.objective, fit.mixture.weights[0], lobes,
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


def star_block_names(class_index: int | None) -> tuple[str, str]:
    """The two blocks of a fit: ``STAR_FIT_BLOCK`` and ``STAR_COMPONENT_BLOCK`` for a K=1 half's (``None``), with
    ``_class_<class_index + 1>`` appended for a Class3D model's class ``class_index`` (from 0), as RELION numbers
    ``model_pdf_orient_class``."""
    if class_index is None:
        return STAR_FIT_BLOCK, STAR_COMPONENT_BLOCK
    return f"{STAR_FIT_BLOCK}_class_{class_index + 1}", f"{STAR_COMPONENT_BLOCK}_class_{class_index + 1}"


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


def fit_from_star(items: dict, columns: dict, *, symmetry: str) -> FvmlPriorFit:
    """The fit a model STAR's two blocks hold (values as strings, as ``read_star_blocks`` returns them), in a run
    with point group ``symmetry`` (the sampling STAR's ``rlnSymmetryGroup``)."""
    weights = np.asarray(columns["relax_fvml_weight"], dtype=np.float64)
    means = np.stack([np.asarray(columns[f"relax_fvml_mean_{a}"], dtype=np.float64) for a in "xyz"], axis=1)[1:]
    kappas = np.asarray(columns["relax_fvml_kappa"], dtype=np.float64)[1:]
    start = items["relax_fvml_start"]
    if start not in STARTS:
        raise ValueError(f"{STAR_FIT_BLOCK}: unknown start {start!r}")
    return FvmlPriorFit(
        mixture=FvmlMixture(
            weights / weights.sum(), means / np.linalg.norm(means, axis=1, keepdims=True), kappas,
            symmetry_maps(symmetry),
        ),
        symmetry=symmetry,
        fit_order=int(items["relax_fvml_fit_order"]),
        kappa_max=float(items["relax_fvml_kappa_max"]),
        start=start,
        em_iterations=int(items["relax_fvml_em_iterations"]),
        converged=bool(int(float(items["relax_fvml_converged"]))),
        objective=float(items["relax_fvml_objective"]),
        seconds=float(items["relax_fvml_seconds"]),
    )
