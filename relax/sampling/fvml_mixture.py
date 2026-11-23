"""Weighted EM for a mixture of FvML (von Mises-Fisher on S^2) distributions fitted to direction atoms.

Fits the rotation prior of the pose-prior M-step, derived in docs/math/fvml_mixture_direction_prior.md: the atoms
are viewing directions with non-negative masses (the posterior mass the E-step puts on each direction), component 0
is the uniform density with a free weight, and the other components are FvML lobes, each optionally copied by a
finite set of orthogonal maps (antipodal symmetry, a point group). ``fit_fvml_mixture`` treats each mass as sitting
at its direction; ``fit_fvml_mixture_cells`` treats it as the mass of a grid cell (the grouped-data likelihood).
The recommended setting is the cell fit with the default cap of ``kappa_max_for_grid`` (lobes at least a quarter
of a grid spacing wide): with a looser cap than one spacing the point fit's quantization bias shows (section 10 of
the document). ``healpix_cell_points`` and ``cell_log_probabilities`` put a mixture on RELION's C1 direction grid
(HEALPix NEST pixels); ``relax.sampling.fvml_direction_prior`` fits it as a refinement's direction prior. Host
float64 NumPy.

Use::

    import numpy as np
    from relax.sampling.fvml_mixture import fit_fvml_mixture_cells, init_fvml_mixture, kappa_max_for_grid

    kappa_max = kappa_max_for_grid(directions.shape[0])  # directions: (D, 3) pixel centres of a whole-sphere grid
    start = init_fvml_mixture(directions, masses, 4, kappa_max=kappa_max, rng=np.random.default_rng(0))
    fit = fit_fvml_mixture_cells(cell_points, masses, start, kappa_max=kappa_max)  # cell_points: (D, C, 3) children
    fit.mixture.weights, fit.mixture.means, fit.mixture.kappas, fit.objective
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import healpy as hp
import numpy as np
from scipy.special import logsumexp

LOG_4PI = math.log(4.0 * math.pi)
UNIT_NORM_TOL = 1e-6  # directions may come from a float32 grid
DECREASE_TOL = 1e-9  # relative: EM cannot lower the objective by more than rounding
SERIES_BELOW = 0.1  # Langevin function and its slope use their Taylor series under this kappa
NEWTON_MAX_ITER = 100
EMPTY_CLUSTER_SHARE = 1e-3  # lobe-weight share given to an empty k-means cluster at initialization


def _identity_symmetry() -> np.ndarray:
    """The trivial symmetry: a single identity map, shape (1, 3, 3)."""
    return np.eye(3)[None]


@dataclass(frozen=True, eq=False)
class FvmlMixture:
    """Uniform component plus FvML lobes: ``lambda(v) = w_0 / 4 pi + sum_j w_j mean_g f(v; g mu_j, kappa_j)``.

    ``weights`` is ``(n_r,)`` and sums to 1 (index 0 is the uniform component), ``means`` is ``(n_r - 1, 3)`` unit
    vectors, ``kappas`` is ``(n_r - 1,)``, and ``symmetry`` is ``(n_g, 3, 3)`` orthogonal maps (identity only by
    default). With the symmetry a lobe's weight is shared by its ``n_g`` images ``g mu_j``.
    """

    weights: np.ndarray
    means: np.ndarray
    kappas: np.ndarray
    symmetry: np.ndarray = field(default_factory=_identity_symmetry)

    def __post_init__(self) -> None:
        """Store float64 copies and reject shapes and values outside the model."""
        weights = np.array(self.weights, dtype=np.float64)
        means = np.array(self.means, dtype=np.float64)
        kappas = np.array(self.kappas, dtype=np.float64)
        symmetry = np.array(self.symmetry, dtype=np.float64)
        n_lobes = weights.size - 1
        if weights.ndim != 1 or n_lobes < 1:
            raise ValueError(f"weights must be (n_r,) with n_r >= 2, got {weights.shape}")
        if means.shape != (n_lobes, 3) or kappas.shape != (n_lobes,):
            raise ValueError(f"means {means.shape} and kappas {kappas.shape} must be ({n_lobes}, 3) and ({n_lobes},)")
        if symmetry.ndim != 3 or symmetry.shape[1:] != (3, 3):
            raise ValueError(f"symmetry must be (n_g, 3, 3), got {symmetry.shape}")
        if np.any(weights < 0.0) or abs(weights.sum() - 1.0) > 1e-9:
            raise ValueError("weights must be non-negative and sum to 1")
        if np.any(np.abs(np.linalg.norm(means, axis=1) - 1.0) > UNIT_NORM_TOL):
            raise ValueError("means must be unit vectors")
        if not np.all(np.isfinite(kappas)) or np.any(kappas < 0.0):
            raise ValueError("kappas must be finite and non-negative")
        if np.any(np.abs(symmetry @ symmetry.transpose(0, 2, 1) - np.eye(3)) > 1e-9):
            raise ValueError("symmetry maps must be orthogonal")
        for name, value in (("weights", weights), ("means", means), ("kappas", kappas), ("symmetry", symmetry)):
            object.__setattr__(self, name, value)


@dataclass(frozen=True, eq=False)
class FitResult:
    """Outcome of a fit: the mixture, the objective at every iterate and whether it converged.

    ``objective[0]`` is the start and ``objective[-1]`` belongs to ``mixture``; the objective is the fit's
    log-likelihood plus the Dirichlet log-prior of the weights (zero for ``dirichlet=1``).
    """

    mixture: FvmlMixture
    objective: np.ndarray
    converged: bool

    @property
    def n_iter(self) -> int:
        """Number of EM iterations taken."""
        return int(self.objective.size) - 1


def log_fvml(cosines: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """Log FvML density at ``cosines = mean . v`` for concentration ``kappa``; stable for any ``kappa >= 0``.

    Equals ``log(kappa / (4 pi sinh kappa)) + kappa * cosines``, computed without overflow, and ``-log(4 pi)`` at
    ``kappa = 0``. ``cosines`` and ``kappa`` broadcast.
    """
    kappa = np.asarray(kappa, dtype=np.float64)
    positive = kappa > 0.0
    safe = np.where(positive, kappa, 1.0)
    log_kappa_over_decay = np.where(positive, np.log(safe) - np.log(-np.expm1(-2.0 * safe)), -math.log(2.0))
    return log_kappa_over_decay - math.log(2.0 * math.pi) + kappa * (np.asarray(cosines, dtype=np.float64) - 1.0)


def langevin(kappa: float) -> float:
    """``A(kappa) = coth(kappa) - 1/kappa``, the mean cosine of an FvML sample; ``A(0) = 0``."""
    if kappa < SERIES_BELOW:
        k2 = kappa * kappa
        return kappa * (1.0 / 3.0 - k2 * (1.0 / 45.0 - k2 * (2.0 / 945.0 - k2 / 4725.0)))
    decay = math.exp(-2.0 * kappa)
    return (1.0 + decay) / (1.0 - decay) - 1.0 / kappa


def _langevin_slope(kappa: float) -> float:
    """``A'(kappa) = 1/kappa^2 - 1/sinh(kappa)^2``, the variance of the cosine."""
    if kappa < SERIES_BELOW:
        k2 = kappa * kappa
        return 1.0 / 3.0 - k2 * (1.0 / 15.0 - k2 * (2.0 / 189.0 - k2 / 675.0))
    decay = math.exp(-2.0 * kappa)
    return 1.0 / (kappa * kappa) - 4.0 * decay / (1.0 - decay) ** 2


def inverse_langevin(r_bar: float, kappa_max: float) -> float:
    """Solve ``A(kappa) = r_bar`` and clip to ``[0, kappa_max]``; ``r_bar >= 1`` gives ``kappa_max``.

    Safeguarded Newton from the start ``r (3 - r^2) / (1 - r^2)`` of Banerjee et al. (2005), with a bisection bracket.
    """
    if r_bar <= 0.0:
        return 0.0
    if r_bar >= 1.0:
        return kappa_max
    kappa = r_bar * (3.0 - r_bar * r_bar) / (1.0 - r_bar * r_bar)
    low, high = 0.0, math.inf
    for _ in range(NEWTON_MAX_ITER):
        value = langevin(kappa)
        low, high = (kappa, high) if value < r_bar else (low, kappa)
        step = (value - r_bar) / _langevin_slope(kappa)
        candidate = kappa - step
        if not low < candidate < high:
            candidate = 0.5 * (low + high) if math.isfinite(high) else 2.0 * low
        converged = abs(candidate - kappa) <= 1e-15 * candidate
        kappa = candidate
        if converged:
            return min(kappa, kappa_max)
    raise RuntimeError(f"inverse_langevin did not converge for r_bar={r_bar!r}")


def kappa_max_for_grid(n_sphere_directions: int, min_sigma_in_spacings: float = 0.25) -> float:
    """Concentration cap that keeps a lobe at least ``min_sigma_in_spacings`` node spacings wide.

    ``n_sphere_directions`` is the size of the equal-area grid over the whole sphere (``12 * 4**order`` for HEALPix),
    so the spacing is ``delta = sqrt(4 pi / n)``; the cap is ``1 / (min_sigma_in_spacings * delta)**2``, the
    concentration whose lobe has angular standard deviation ``kappa**-0.5`` equal to that width.
    """
    if n_sphere_directions < 1 or min_sigma_in_spacings <= 0.0:
        raise ValueError("n_sphere_directions and min_sigma_in_spacings must be positive")
    return n_sphere_directions / (4.0 * math.pi * min_sigma_in_spacings**2)


def _transform(directions: np.ndarray, symmetry: np.ndarray) -> np.ndarray:
    """``(D, n_g, 3)`` array of ``g^T v`` for every direction and symmetry map (the data in each lobe's frame)."""
    return np.einsum("gab,ma->mgb", symmetry, directions)


def _log_joint(mixture: FvmlMixture, transformed: np.ndarray) -> np.ndarray:
    """``(D, 1 + (n_r - 1) n_g)`` log joint ``log a_k(v_m)``: the uniform column, then lobes (component-major)."""
    n_atoms, n_g, _ = transformed.shape
    with np.errstate(divide="ignore"):  # a zero weight is a -inf log joint
        log_weights = np.log(mixture.weights)
    cosines = np.einsum("mgb,jb->mjg", transformed, mixture.means)
    log_lobes = log_weights[None, 1:, None] - math.log(n_g) + log_fvml(cosines, mixture.kappas[None, :, None])
    uniform = np.full((n_atoms, 1), log_weights[0] - LOG_4PI)
    return np.concatenate([uniform, log_lobes.reshape(n_atoms, -1)], axis=1)


def log_density(mixture: FvmlMixture, directions: np.ndarray) -> np.ndarray:
    """``log lambda(v)`` of the mixture at ``(D, 3)`` unit vectors, shape ``(D,)``."""
    directions = np.asarray(directions, dtype=np.float64)
    return logsumexp(_log_joint(mixture, _transform(directions, mixture.symmetry)), axis=1)


def _check_atoms(directions: np.ndarray, masses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Float64 copies of the atoms; rejects non-unit directions and negative, non-finite or all-zero masses."""
    directions = np.array(directions, dtype=np.float64)
    masses = np.array(masses, dtype=np.float64)
    if directions.ndim != 2 or directions.shape[1] != 3 or masses.shape != (directions.shape[0],):
        raise ValueError(f"directions must be (D, 3) and masses (D,), got {directions.shape} and {masses.shape}")
    if np.any(np.abs(np.linalg.norm(directions, axis=1) - 1.0) > UNIT_NORM_TOL):
        raise ValueError("directions must be unit vectors")
    if not np.all(np.isfinite(masses)) or np.any(masses < 0.0) or masses.sum() <= 0.0:
        raise ValueError("masses must be finite, non-negative and not all zero")
    return directions, masses


def _check_kappa_max(kappa_max: float) -> None:
    """Reject a missing, non-positive or infinite cap: the objective is unbounded without one."""
    if not (math.isfinite(kappa_max) and kappa_max > 0.0):
        raise ValueError(f"kappa_max must be finite and positive, got {kappa_max!r}")


def _m_step(
    mixture: FvmlMixture,
    responsibilities: np.ndarray,
    masses: np.ndarray,
    transformed: np.ndarray,
    kappa_max: float,
    dirichlet: float,
) -> FvmlMixture:
    """Closed-form maximizer of the EM surrogate: Dirichlet-MAP weights, resultant directions, ``A^-1`` concentrations."""
    n_atoms, n_g, _ = transformed.shape
    resp_lobes = responsibilities[:, 1:].reshape(n_atoms, -1, n_g)
    weighted = masses[:, None, None] * resp_lobes
    lobe_mass = weighted.sum(axis=(0, 2))
    resultant = np.einsum("mjg,mgb->jb", weighted, transformed)
    counts = np.concatenate([[masses @ responsibilities[:, 0]], lobe_mass]) + (dirichlet - 1.0)
    means, kappas = mixture.means.copy(), mixture.kappas.copy()
    for j in range(means.shape[0]):
        norm = float(np.linalg.norm(resultant[j]))
        if lobe_mass[j] <= 0.0:
            continue  # no mass: direction and concentration are undetermined, keep them
        kappas[j] = inverse_langevin(min(norm / lobe_mass[j], 1.0), kappa_max)
        if norm > 0.0:
            means[j] = resultant[j] / norm
    return FvmlMixture(counts / counts.sum(), means, kappas, mixture.symmetry)


def _objective(log_likelihood: float, weights: np.ndarray, dirichlet: float) -> float:
    """The log-likelihood plus the Dirichlet log-prior ``(dirichlet - 1) sum_j log w_j``."""
    value = log_likelihood
    if dirichlet > 1.0:
        value += (dirichlet - 1.0) * float(np.sum(np.log(weights)))
    return value


def _check_fit_options(start: FvmlMixture, kappa_max: float, dirichlet: float, max_iter: int, tol: float) -> None:
    """Reject a bad cap, Dirichlet concentration, iteration budget or tolerance, and a start above the cap."""
    _check_kappa_max(kappa_max)
    if dirichlet < 1.0 or max_iter < 0 or tol < 0.0:
        raise ValueError("dirichlet must be >= 1, max_iter >= 0 and tol >= 0")
    if np.any(start.kappas > kappa_max):
        raise ValueError("start has a concentration above kappa_max")


def _run_em(
    start: FvmlMixture,
    e_step: Callable[[FvmlMixture], tuple[float, np.ndarray, np.ndarray]],
    transformed: np.ndarray,
    *,
    kappa_max: float,
    dirichlet: float,
    max_iter: int,
    tol: float,
    total_mass: float,
) -> FitResult:
    """EM loop shared by the fits: ``e_step(mixture)`` gives the log-likelihood, per-atom masses and responsibilities.

    The M-step is :func:`_m_step` on the atoms ``transformed`` with those masses; stops when the gain of the objective
    (log-likelihood plus Dirichlet term) is at most ``tol * total_mass`` or after ``max_iter`` M-steps; raises if the
    objective drops.
    """
    mixture, history, converged = start, [], False
    for iteration in range(max_iter + 1):
        log_likelihood, masses, responsibilities = e_step(mixture)
        history.append(_objective(log_likelihood, mixture.weights, dirichlet))
        if iteration > 0:
            gain = history[-1] - history[-2]
            if gain < -DECREASE_TOL * abs(history[-1]):
                raise RuntimeError(f"EM objective decreased at iteration {iteration}: {history[-2]} -> {history[-1]}")
            if gain <= tol * total_mass:
                converged = True
                break
        if iteration == max_iter:
            break
        mixture = _m_step(mixture, responsibilities, masses, transformed, kappa_max, dirichlet)
    return FitResult(mixture, np.array(history), converged)


def fit_fvml_mixture(
    directions: np.ndarray,
    masses: np.ndarray,
    start: FvmlMixture,
    *,
    kappa_max: float,
    dirichlet: float = 1.0,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> FitResult:
    """EM for the mixture on weighted direction atoms, from ``start`` (the warm start or an initialization).

    ``directions`` is ``(D, 3)`` unit vectors, ``masses`` is ``(D,)`` non-negative posterior masses (their sum is
    the particle count, which sets the strength of the Dirichlet prior). Every ``kappa`` stays in ``[0, kappa_max]``;
    the cap is required because the unregularized objective is unbounded (``kappa_max_for_grid`` gives a grid rule).
    ``dirichlet >= 1`` is the symmetric Dirichlet concentration on the weights (1 is flat). Stops when an iteration
    gains at most ``tol`` nats of objective per unit of mass (per particle) or after ``max_iter`` M-steps; raises if
    the objective ever drops. The objective's likelihood is ``sum_m mass_m log lambda(v_m)``.
    """
    directions, masses = _check_atoms(directions, masses)
    _check_fit_options(start, kappa_max, dirichlet, max_iter, tol)
    transformed = _transform(directions, start.symmetry)

    def e_step(mixture: FvmlMixture) -> tuple[float, np.ndarray, np.ndarray]:
        """Log-likelihood of the atoms and their component responsibilities."""
        log_joint = _log_joint(mixture, transformed)
        log_lambda = logsumexp(log_joint, axis=1)
        return float(masses @ log_lambda), masses, np.exp(log_joint - log_lambda[:, None])

    return _run_em(
        start,
        e_step,
        transformed,
        kappa_max=kappa_max,
        dirichlet=dirichlet,
        max_iter=max_iter,
        tol=tol,
        total_mass=float(masses.sum()),
    )


def fit_fvml_mixture_cells(
    cell_points: np.ndarray,
    masses: np.ndarray,
    start: FvmlMixture,
    *,
    kappa_max: float,
    dirichlet: float = 1.0,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> FitResult:
    """EM for the grouped-data likelihood ``sum_m mass_m log P(cell m)``: each mass belongs to a cell, not a point.

    ``cell_points`` is ``(D, C, 3)``: the centres of ``C`` equal-area children of each of ``D`` equal-area cells that
    tile the sphere (the HEALPix pixels of a finer order). ``P(cell m)`` is the quadrature
    ``(4 pi / D) mean_c lambda(v_mc)``. The E-step spreads each cell's mass over its children in proportion to the
    current density and over the components; the M-step is the closed form of :func:`fit_fvml_mixture` on the
    children. Other arguments and the return value are those of :func:`fit_fvml_mixture`; the objective's
    likelihood is the log probability of the cell masses.
    """
    cell_points = np.asarray(cell_points, dtype=np.float64)
    if cell_points.ndim != 3 or cell_points.shape[2] != 3:
        raise ValueError(f"cell_points must be (D, C, 3), got {cell_points.shape}")
    n_cells, n_children, _ = cell_points.shape
    masses = np.asarray(masses, dtype=np.float64)
    if masses.shape != (n_cells,):
        raise ValueError(f"masses must be ({n_cells},), got {masses.shape}")
    children, _ = _check_atoms(cell_points.reshape(-1, 3), np.repeat(masses, n_children))
    _check_fit_options(start, kappa_max, dirichlet, max_iter, tol)
    transformed = _transform(children, start.symmetry)
    log_child_area = math.log(4.0 * math.pi / n_cells) - math.log(n_children)

    def e_step(mixture: FvmlMixture) -> tuple[float, np.ndarray, np.ndarray]:
        """Log probability of the cell masses, each child's share of its cell's mass, and component responsibilities."""
        log_joint = _log_joint(mixture, transformed)
        log_child = logsumexp(log_joint, axis=1)
        log_child_by_cell = log_child.reshape(n_cells, n_children)
        log_cell = logsumexp(log_child_by_cell, axis=1)
        child_share = np.exp(log_child_by_cell - log_cell[:, None])
        log_likelihood = float(masses @ (log_cell + log_child_area))
        child_masses = (masses[:, None] * child_share).reshape(-1)
        return log_likelihood, child_masses, np.exp(log_joint - log_child[:, None])

    return _run_em(
        start,
        e_step,
        transformed,
        kappa_max=kappa_max,
        dirichlet=dirichlet,
        max_iter=max_iter,
        tol=tol,
        total_mass=float(masses.sum()),
    )


def _best_images(transformed: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per atom: nearest center, its cosine, and the atom's image ``g^T v`` for the best symmetry map."""
    similarity = np.einsum("mgb,kb->mkg", transformed, centers)
    best_map = similarity.argmax(axis=2)
    best_similarity = np.take_along_axis(similarity, best_map[:, :, None], axis=2)[:, :, 0]
    labels = best_similarity.argmax(axis=1)
    rows = np.arange(labels.size)
    return labels, best_similarity[rows, labels], transformed[rows, best_map[rows, labels]]


def init_fvml_mixture(
    directions: np.ndarray,
    masses: np.ndarray,
    n_components: int,
    *,
    kappa_max: float,
    rng: np.random.Generator,
    symmetry: np.ndarray | None = None,
    uniform_weight: float = 0.2,
    lloyd_iterations: int = 10,
) -> FvmlMixture:
    """Cold start: mass-weighted spherical k-means++ with ``n_components - 1`` lobes plus the uniform component.

    Seeds are drawn with probability proportional to ``mass * (1 - cosine to the nearest chosen seed)^2`` (the cosine
    is the best over the symmetry maps), refined by ``lloyd_iterations`` hard-assignment steps; each lobe takes the
    cluster's resultant direction, ``A^-1`` of its mean resultant length and its share of the mass, scaled to
    ``1 - uniform_weight``. An empty cluster keeps its seed with a share of ``EMPTY_CLUSTER_SHARE``. The EM only
    finds a local maximum, so run several seeds and keep the largest final objective.
    """
    directions, masses = _check_atoms(directions, masses)
    _check_kappa_max(kappa_max)
    if n_components < 2 or not 0.0 < uniform_weight < 1.0:
        raise ValueError("n_components must be >= 2 and uniform_weight in (0, 1)")
    symmetry = _identity_symmetry() if symmetry is None else np.asarray(symmetry, dtype=np.float64)
    transformed = _transform(directions, symmetry)
    n_lobes = n_components - 1
    centers = directions[[rng.choice(masses.size, p=masses / masses.sum())]]
    for _ in range(n_lobes - 1):
        nearest_cosine = _best_images(transformed, centers)[1]
        score = masses * (1.0 - nearest_cosine) ** 2
        if score.sum() <= 0.0:
            raise ValueError(f"fewer than {n_lobes} distinct directions carry mass")
        centers = np.vstack([centers, directions[rng.choice(masses.size, p=score / score.sum())]])
    for _ in range(lloyd_iterations):
        labels, _, images = _best_images(transformed, centers)
        for j in range(n_lobes):
            total = (masses[labels == j, None] * images[labels == j]).sum(axis=0)
            if np.linalg.norm(total) > 0.0:
                centers[j] = total / np.linalg.norm(total)
    labels, _, images = _best_images(transformed, centers)
    cluster_mass = np.array([masses[labels == j].sum() for j in range(n_lobes)])
    resultants = np.array(
        [np.linalg.norm((masses[labels == j, None] * images[labels == j]).sum(axis=0)) for j in range(n_lobes)]
    )
    r_bar = np.divide(resultants, cluster_mass, out=np.zeros(n_lobes), where=cluster_mass > 0.0)
    kappas = np.array([inverse_langevin(min(float(r), 1.0), kappa_max) for r in r_bar])
    shares = np.maximum(cluster_mass / masses.sum(), EMPTY_CLUSTER_SHARE)
    weights = np.concatenate([[uniform_weight], (1.0 - uniform_weight) * shares / shares.sum()])
    return FvmlMixture(weights, centers, kappas, symmetry)


def healpix_cell_points(order: int, levels: int) -> np.ndarray:
    """Centres of the ``4**levels`` NEST children of every HEALPix pixel at ``order``: ``(12 * 4**order, 4**levels, 3)``.

    The pixels are RELION's C1 directions at ``order``, numbered as its ``rlnOrientationDistribution`` (NEST).
    """
    nside = 1 << (order + levels)
    points = np.stack(hp.pix2vec(nside, np.arange(hp.nside2npix(nside)), nest=True), axis=1)
    return points.reshape(12 * 4**order, 4**levels, 3)


def cell_log_probabilities(mixture: FvmlMixture, order: int, levels: int) -> np.ndarray:
    """Log probability of each HEALPix pixel at ``order`` (NEST) under ``mixture``, normalized over the sphere.

    The midpoint rule over the pixel's ``4**levels`` children, the quadrature of :func:`fit_fvml_mixture_cells`.
    Kept in logs: far from every lobe of a mixture with a small uniform weight a probability can underflow.
    """
    log_values = log_density(mixture, healpix_cell_points(order, levels).reshape(-1, 3)).reshape(12 * 4**order, -1)
    log_cell = logsumexp(log_values, axis=1)
    return log_cell - logsumexp(log_cell)
