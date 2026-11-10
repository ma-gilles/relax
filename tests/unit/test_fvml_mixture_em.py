"""FvML-mixture EM (scripts/lib/fvml_mixture_em.py): the density, the closed-form M-step against a numerical
maximizer, monotone ascent, recovery of a generating mixture, symmetry, and the concentration cap."""

import math

import healpy as hp
import numpy as np
import pytest
from helpers.float_compare import assert_matches
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from scripts.lib.fvml_mixture_em import (
    FvmlMixture,
    fit_fvml_mixture,
    init_fvml_mixture,
    inverse_langevin,
    kappa_max_for_grid,
    langevin,
    log_density,
    log_fvml,
)

NSIDE = 16
ANTIPODAL = np.stack([np.eye(3), -np.eye(3)])
Z_AXIS, X_AXIS = np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])


def _rot_z(angle: float) -> np.ndarray:
    """Rotation about z by ``angle`` radians."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _c3() -> np.ndarray:
    """The cyclic group of order 3 about z, shape (3, 3, 3)."""
    return np.stack([_rot_z(2.0 * math.pi * i / 3.0) for i in range(3)])


def _two_maps_not_a_group() -> np.ndarray:
    """Identity and a 0.7 rad rotation about z: not closed under products, so ``g`` and ``g^-1`` give different lobes."""
    return np.stack([np.eye(3), _rot_z(0.7)])


def _sphere_grid(nside: int) -> np.ndarray:
    """HEALPix pixel centres (NEST), shape (12 nside^2, 3)."""
    return np.stack(hp.pix2vec(nside, np.arange(hp.nside2npix(nside)), nest=True), axis=1)


def _random_unit(rng: np.random.Generator, n: int) -> np.ndarray:
    """``n`` uniform random unit vectors, shape (n, 3)."""
    x = rng.normal(size=(n, 3))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _sample_fvml(rng: np.random.Generator, n: int, mean: np.ndarray, kappa: float) -> np.ndarray:
    """``n`` FvML draws about unit ``mean`` by the closed-form inverse CDF of the cosine."""
    u = rng.random(n)
    cosine = 1.0 + np.log(u + (1.0 - u) * np.exp(-2.0 * kappa)) / kappa
    phi = 2.0 * math.pi * rng.random(n)
    sine = np.sqrt(np.clip(1.0 - cosine**2, 0.0, None))
    helper = X_AXIS if abs(mean[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(mean, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(mean, e1)
    return (sine * np.cos(phi))[:, None] * e1 + (sine * np.sin(phi))[:, None] * e2 + cosine[:, None] * mean


def _bin_masses(points: np.ndarray, nside: int) -> np.ndarray:
    """Per-pixel counts of ``points`` on the HEALPix grid: the aggregate posterior mass of an E-step."""
    pixels = hp.vec2pix(nside, points[:, 0], points[:, 1], points[:, 2], nest=True)
    return np.bincount(pixels, minlength=hp.nside2npix(nside)).astype(np.float64)


def _textbook_log_fvml(cosines: np.ndarray, kappa: float) -> np.ndarray:
    """``log(kappa / (4 pi sinh kappa)) + kappa cos``, valid while ``sinh`` does not overflow."""
    return np.log(kappa / (4.0 * math.pi * np.sinh(kappa))) + kappa * cosines


def _mixture_data(seed: int, n: int = 100_000) -> tuple[np.ndarray, np.ndarray, FvmlMixture]:
    """Grid, binned masses and the generating mixture: 0.25 uniform + 0.40 FvML(z, 25) + 0.35 FvML(x, 120)."""
    rng = np.random.default_rng(seed)
    truth = FvmlMixture([0.25, 0.40, 0.35], [Z_AXIS, X_AXIS], [25.0, 120.0])
    counts = rng.multinomial(n, truth.weights)
    points = np.concatenate(
        [_random_unit(rng, counts[0])]
        + [_sample_fvml(rng, counts[j + 1], truth.means[j], truth.kappas[j]) for j in range(2)]
    )
    return _sphere_grid(NSIDE), _bin_masses(points, NSIDE), truth


def _best_of_starts(
    directions: np.ndarray, masses: np.ndarray, n_components: int, kappa_max: float, seed: int, n_starts: int = 5
):
    """Fit from ``n_starts`` k-means++ starts and return the result with the largest objective."""
    rng = np.random.default_rng(seed)
    fits = [
        fit_fvml_mixture(
            directions,
            masses,
            init_fvml_mixture(directions, masses, n_components, kappa_max=kappa_max, rng=rng),
            kappa_max=kappa_max,
        )
        for _ in range(n_starts)
    ]
    return max(fits, key=lambda fit: fit.objective[-1])


def test_log_fvml_matches_textbook_formula_and_stays_finite() -> None:
    """The stable density equals the textbook one where that is representable, and is finite at kappa 1e8."""
    cosines = np.linspace(-1.0, 1.0, 11)
    for kappa in (1e-6, 0.5, 5.0, 50.0, 300.0):
        assert_matches(log_fvml(cosines, kappa), _textbook_log_fvml(cosines, kappa), err_msg=f"kappa={kappa}")
    assert np.all(np.isfinite(log_fvml(cosines, 1e8)))
    assert_matches(log_fvml(cosines, 0.0), np.full_like(cosines, -math.log(4.0 * math.pi)))


def test_log_fvml_integrates_to_one_over_the_sphere() -> None:
    """The cosine integral ``2 pi int exp(log f) dc`` is 1 for each kappa (Gauss-Legendre in the cosine)."""
    nodes, weights = np.polynomial.legendre.leggauss(800)
    for kappa in (0.0, 0.5, 5.0, 50.0, 300.0):
        total = 2.0 * math.pi * float(weights @ np.exp(log_fvml(nodes, kappa)))
        assert_matches(total, 1.0, rtol=1e-10, err_msg=f"kappa={kappa}")


def test_langevin_series_joins_the_closed_form() -> None:
    """The Langevin function agrees with ``coth - 1/kappa`` in long double across the series switch at 0.1."""
    for kappa in (1e-3, 0.0999, 0.1001, 1.0):
        k = np.longdouble(kappa)
        reference = float(1.0 / np.tanh(k) - 1.0 / k)
        # the long-double reference cancels to about 1e-19 / (kappa / 3) relative, 3e-13 at kappa 1e-3
        assert_matches(langevin(kappa), reference, rtol=1e-11, err_msg=f"kappa={kappa}")
    tiny = 1e-6  # the closed form is unusable here; the first Taylor terms are exact to 1e-30
    assert_matches(langevin(tiny), tiny / 3.0 - tiny**3 / 45.0, rtol=1e-13)


def test_inverse_langevin_round_trips_and_clips() -> None:
    """``A(A^-1(r)) = r`` over the whole range, ``r >= 1`` and large ``A^-1`` clip to the cap, ``r = 0`` is 0."""
    r_values = np.array([1e-8, 1e-4, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.9999])
    kappas = np.array([inverse_langevin(r, 1e12) for r in r_values])
    assert_matches(np.array([langevin(k) for k in kappas]), r_values, rtol=1e-11)
    assert np.all(np.diff(kappas) > 0.0)
    assert inverse_langevin(0.0, 100.0) == 0.0
    assert_matches(inverse_langevin(1.0, 100.0), 100.0)
    assert_matches(inverse_langevin(0.99, 10.0), 10.0)


def test_kappa_max_for_grid_is_the_spacing_rule() -> None:
    """The cap is ``1/delta^2`` with ``delta^2 = 4 pi / n`` and scales with the minimum width in spacings."""
    n = hp.nside2npix(NSIDE)
    assert_matches(kappa_max_for_grid(n), n / (4.0 * math.pi))
    assert_matches(kappa_max_for_grid(n, 0.5), 4.0 * n / (4.0 * math.pi))
    with pytest.raises(ValueError):
        kappa_max_for_grid(0)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(weights=[1.0], means=np.zeros((0, 3)), kappas=[]),
        dict(weights=[0.5, 0.6], means=[Z_AXIS], kappas=[1.0]),
        dict(weights=[0.5, 0.5], means=[[0.0, 0.0, 2.0]], kappas=[1.0]),
        dict(weights=[0.5, 0.5], means=[Z_AXIS], kappas=[-1.0]),
        dict(weights=[0.5, 0.5], means=[Z_AXIS], kappas=[1.0], symmetry=2.0 * np.eye(3)[None]),
    ],
)
def test_mixture_rejects_values_outside_the_model(kwargs: dict) -> None:
    """A mixture needs one lobe, weights summing to 1, unit means, kappa >= 0 and orthogonal symmetry maps."""
    with pytest.raises(ValueError):
        FvmlMixture(**kwargs)


@pytest.mark.parametrize(
    "symmetry", [None, ANTIPODAL, _c3(), _two_maps_not_a_group()], ids=["plain", "antipodal", "c3", "two-maps"]
)
def test_log_density_matches_the_explicit_mixture(symmetry: np.ndarray | None) -> None:
    """``log_density`` equals the written-out ``w0/4pi + sum_j w_j mean_g f(v; g mu_j, kappa_j)``."""
    rng = np.random.default_rng(1)
    ops = np.eye(3)[None] if symmetry is None else symmetry
    mixture = FvmlMixture([0.3, 0.5, 0.2], _random_unit(rng, 2), [4.0, 60.0], ops)
    v = _random_unit(rng, 50)
    density = mixture.weights[0] / (4.0 * math.pi)
    for j in range(2):
        for g in ops:
            density = density + mixture.weights[j + 1] / len(ops) * np.exp(
                _textbook_log_fvml(v @ (g @ mixture.means[j]), mixture.kappas[j])
            )
    assert_matches(log_density(mixture, v), np.log(density))


def _surrogate(
    weights: np.ndarray,
    means: np.ndarray,
    kappas: np.ndarray,
    v: np.ndarray,
    rho: np.ndarray,
    resp: np.ndarray,
    ops: np.ndarray,
) -> float:
    """EM surrogate ``sum_m rho_m sum_k r_mk log a_k(v_m)`` with fixed responsibilities ``resp`` (D, 1 + lobes * maps)."""
    n_g = len(ops)
    total = float(rho @ resp[:, 0]) * (math.log(weights[0]) - math.log(4.0 * math.pi))
    for j in range(len(kappas)):
        for gi, g in enumerate(ops):
            lf = _textbook_log_fvml(v @ (g @ means[j]), kappas[j])
            total += float(rho @ (resp[:, 1 + j * n_g + gi] * (math.log(weights[j + 1]) - math.log(n_g) + lf)))
    return total


@pytest.mark.parametrize("symmetry", [None, ANTIPODAL, _two_maps_not_a_group()], ids=["plain", "antipodal", "two-maps"])
def test_one_m_step_is_the_numerical_maximizer_of_the_surrogate(symmetry: np.ndarray | None) -> None:
    """The closed-form M-step reaches at least the best value of a numerical search over all parameters."""
    rng = np.random.default_rng(2)
    ops = np.eye(3)[None] if symmetry is None else symmetry
    v, rho = _random_unit(rng, 40), rng.random(40) * 3.0 + 0.1
    start = FvmlMixture(softmax(rng.normal(size=3)), _random_unit(rng, 2), rng.uniform(2.0, 30.0, 2), ops)
    # responsibilities at the start, from the textbook density
    cols = [np.full(40, math.log(start.weights[0]) - math.log(4.0 * math.pi))]
    for j in range(2):
        for g in ops:
            cols.append(
                math.log(start.weights[j + 1])
                - math.log(len(ops))
                + _textbook_log_fvml(v @ (g @ start.means[j]), start.kappas[j])
            )
    log_joint = np.stack(cols, axis=1)
    resp = np.exp(log_joint - logsumexp(log_joint, axis=1, keepdims=True))

    stepped = fit_fvml_mixture(v, rho, start, kappa_max=1e4, max_iter=1).mixture
    closed = _surrogate(stepped.weights, stepped.means, stepped.kappas, v, rho, resp, ops)

    def negative(x: np.ndarray) -> float:
        """Negative surrogate in unconstrained coordinates (softmax weights, free means, log kappas up to 6)."""
        means = x[3:9].reshape(2, 3)
        unit = means / np.linalg.norm(means, axis=1, keepdims=True)
        return -_surrogate(softmax(x[:3]), unit, np.exp(np.minimum(x[9:], 6.0)), v, rho, resp, ops)

    finals = [
        -minimize(
            negative,
            np.concatenate([rng.normal(size=3), _random_unit(rng, 2).ravel(), np.log(rng.uniform(1.0, 40.0, 2))]),
            method="BFGS",
            options={"gtol": 1e-9},
        ).fun
        for _ in range(8)
    ]
    finals = [value for value in finals if math.isfinite(value)]
    assert len(finals) >= 4 and stepped.kappas.max() < math.exp(6.0)  # the search covers the closed-form optimum
    assert closed >= max(finals) - 1e-7 * abs(max(finals))


@pytest.mark.parametrize("dirichlet", [1.0, 5.0])
def test_objective_never_decreases_and_dirichlet_keeps_weights_positive(dirichlet: float) -> None:
    """Every EM iterate raises the (MAP) objective; a Dirichlet concentration above 1 floors the weights."""
    directions, masses, _ = _mixture_data(seed=3, n=20_000)
    kappa_max = kappa_max_for_grid(directions.shape[0])
    rng = np.random.default_rng(3)
    start = init_fvml_mixture(directions, masses, 4, kappa_max=kappa_max, rng=rng)
    fit = fit_fvml_mixture(directions, masses, start, kappa_max=kappa_max, dirichlet=dirichlet, max_iter=60)
    assert fit.n_iter > 1
    assert np.all(np.diff(fit.objective) >= -1e-9 * abs(fit.objective[-1]))
    assert fit.objective[-1] > fit.objective[0]
    assert np.all(fit.mixture.kappas <= kappa_max)
    if dirichlet > 1.0:
        assert np.all(fit.mixture.weights >= (dirichlet - 1.0) / (masses.sum() + 4 * (dirichlet - 1.0)) * (1 - 1e-9))


def test_fit_recovers_the_generating_mixture_from_binned_draws() -> None:
    """Best of 5 k-means++ starts on 1e5 binned draws returns the weights, directions and widths of the truth."""
    directions, masses, truth = _mixture_data(seed=4)
    kappa_max = kappa_max_for_grid(directions.shape[0])
    fit = _best_of_starts(directions, masses, 3, kappa_max, seed=4)
    order = np.argsort(fit.mixture.kappas)  # truth has kappa 25 (z) before 120 (x)
    assert fit.converged
    # measured on this seed: weights within 0.002 of the truth, kappa 2.6 % (25) and 3.7 % (120) low (node
    # quantization alone predicts 0.5 % and 3.9 %, docs/math/fvml_mixture_direction_prior.md section 6), means 0.12 deg off
    assert_matches(fit.mixture.weights[[0, *(order + 1)]], truth.weights, rtol=0.02)
    for fitted, true in zip(fit.mixture.kappas[order], truth.kappas):
        assert_matches(fitted, true, rtol=0.06)
    cos_to_truth = np.abs(np.einsum("jb,jb->j", fit.mixture.means[order], truth.means))
    assert np.all(cos_to_truth > math.cos(math.radians(1.0)))


def test_symmetry_on_one_sector_equals_all_copies_at_a_third_of_the_mass() -> None:
    """With a symmetry group, fitting the atoms of one sector is the same fit as all copies at mass / n_g."""
    rng = np.random.default_rng(5)
    ops = _c3()
    v, rho = _random_unit(rng, 80), rng.random(80) + 0.1
    copies = np.concatenate([v @ g.T for g in ops])
    start = FvmlMixture(softmax(rng.normal(size=3)), _random_unit(rng, 2), rng.uniform(2.0, 30.0, 2), ops)
    kwargs = dict(kappa_max=500.0, max_iter=25, tol=0.0)
    sector = fit_fvml_mixture(v, rho, start, **kwargs)
    full = fit_fvml_mixture(copies, np.tile(rho / 3.0, 3), start, **kwargs)
    assert_matches(sector.objective, full.objective, rtol=1e-10)
    assert_matches(sector.mixture.weights, full.mixture.weights, rtol=1e-9)
    assert_matches(sector.mixture.kappas, full.mixture.kappas, rtol=1e-9)
    probe = _random_unit(rng, 30)
    assert_matches(log_density(sector.mixture, probe), log_density(full.mixture, probe), rtol=1e-9)


def test_antipodal_symmetry_recovers_a_two_lobed_truth_that_a_plain_fit_cannot() -> None:
    """Draws from 0.3 uniform + 0.7 FvML(mu, 40) with a random sign: the antipodal fit finds mu, kappa and the weight."""
    rng = np.random.default_rng(6)
    mu = np.array([0.3, 0.2, 0.93])
    mu /= np.linalg.norm(mu)
    n = 40_000
    in_lobe = rng.random(n) < 0.7
    points = _random_unit(rng, n)
    points[in_lobe] = _sample_fvml(rng, int(in_lobe.sum()), mu, 40.0) * rng.choice(
        [-1.0, 1.0], size=(int(in_lobe.sum()), 1)
    )
    directions, masses = _sphere_grid(NSIDE), _bin_masses(points, NSIDE)
    kappa_max = kappa_max_for_grid(directions.shape[0])
    results = {}
    for name, ops in (("antipodal", ANTIPODAL), ("plain", None)):
        starts = [
            init_fvml_mixture(directions, masses, 2, kappa_max=kappa_max, rng=np.random.default_rng(s), symmetry=ops)
            for s in range(3)
        ]
        fits = [fit_fvml_mixture(directions, masses, s, kappa_max=kappa_max) for s in starts]
        results[name] = max(fits, key=lambda f: f.objective[-1])
    sym = results["antipodal"].mixture
    # measured on this seed: weights (0.2984, 0.7016), kappa 39.69 (node quantization predicts 39.5), mean 0.02 deg off,
    # objective 56 120 above the one-lobe fit, which cannot place two opposite lobes with one mean
    assert_matches(sym.weights, np.array([0.3, 0.7]), rtol=0.01)
    assert_matches(sym.kappas, np.array([40.0]), rtol=0.03)
    assert abs(float(sym.means[0] @ mu)) > math.cos(math.radians(1.0))
    assert results["antipodal"].objective[-1] > results["plain"].objective[-1] + 1e4


def test_cap_binds_on_a_dominant_atom_and_the_objective_grows_with_it() -> None:
    """One atom carrying most of the mass sends kappa to the cap; a higher cap gives a higher objective (unbounded)."""
    directions = _sphere_grid(8)
    masses = np.ones(directions.shape[0])
    masses[5] = 50.0
    start = FvmlMixture([0.5, 0.5], [directions[5]], [200.0])
    objectives = []
    for cap in (1e3, 1e6):
        fit = fit_fvml_mixture(directions, masses, start, kappa_max=cap, max_iter=200)
        assert fit.converged
        assert_matches(fit.mixture.kappas, np.array([cap]))
        objectives.append(fit.objective[-1])
    assert objectives[1] - objectives[0] > 0.9 * 50.0 * math.log(1e3)


def test_init_is_valid_deterministic_and_symmetry_aware() -> None:
    """k-means++ gives a valid mixture, the same one for the same seed, and merges antipodal copies into one lobe."""
    directions, masses, _ = _mixture_data(seed=7, n=20_000)
    kappa_max = kappa_max_for_grid(directions.shape[0])
    first = init_fvml_mixture(directions, masses, 3, kappa_max=kappa_max, rng=np.random.default_rng(0))
    again = init_fvml_mixture(directions, masses, 3, kappa_max=kappa_max, rng=np.random.default_rng(0))
    assert_matches(first.weights[0], 0.2)
    assert_matches(first.weights.sum(), 1.0)
    assert np.all(first.kappas <= kappa_max)
    assert_matches(first.means, again.means)
    assert_matches(first.kappas, again.kappas)
    # two antipodal blobs and one lobe: the symmetry-aware seed puts the single lobe on the blob axis with a high kappa
    rng = np.random.default_rng(1)
    blobs = np.concatenate([_sample_fvml(rng, 5000, Z_AXIS, 100.0), _sample_fvml(rng, 5000, -Z_AXIS, 100.0)])
    grid = _sphere_grid(NSIDE)
    two_blob_masses = _bin_masses(blobs, NSIDE)
    sym = init_fvml_mixture(grid, two_blob_masses, 2, kappa_max=kappa_max, rng=rng, symmetry=ANTIPODAL)
    assert abs(float(sym.means[0] @ Z_AXIS)) > 0.99 and sym.kappas[0] > 50.0


def test_init_needs_enough_distinct_directions() -> None:
    """More lobes than directions carrying mass is an error, not a silent duplicate."""
    directions = _sphere_grid(2)
    masses = np.zeros(directions.shape[0])
    masses[0] = 1.0
    with pytest.raises(ValueError):
        init_fvml_mixture(directions, masses, 3, kappa_max=100.0, rng=np.random.default_rng(0))


@pytest.mark.parametrize(
    "change",
    [
        "non_unit_direction",
        "negative_mass",
        "zero_mass",
        "no_cap",
        "infinite_cap",
        "start_above_cap",
        "dirichlet_below_one",
    ],
)
def test_fit_rejects_invalid_inputs(change: str) -> None:
    """The fit refuses bad atoms, a missing or infinite cap, a start above the cap and a Dirichlet below 1."""
    directions = _sphere_grid(2)
    masses = np.ones(directions.shape[0])
    start = FvmlMixture([0.5, 0.5], [Z_AXIS], [10.0])
    kwargs: dict = dict(kappa_max=100.0)
    if change == "non_unit_direction":
        directions = directions * 1.1
    elif change == "negative_mass":
        masses[0] = -1.0
    elif change == "zero_mass":
        masses = np.zeros_like(masses)
    elif change == "no_cap":
        kwargs["kappa_max"] = 0.0
    elif change == "infinite_cap":
        kwargs["kappa_max"] = math.inf
    elif change == "start_above_cap":
        kwargs["kappa_max"] = 5.0
    elif change == "dirichlet_below_one":
        kwargs["dirichlet"] = 0.5
    with pytest.raises(ValueError):
        fit_fvml_mixture(directions, masses, start, **kwargs)
