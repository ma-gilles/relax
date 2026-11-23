"""FvML-mixture prior simulation (scripts/fvml_prior_simulation/simulation.py): the samplers, grid moves, cell
probabilities, lobe matching and one small end-to-end trial."""

import math

import numpy as np
from helpers.float_compare import assert_matches

from scripts.fvml_prior_simulation.simulation import (
    METHODS,
    Target,
    bin_directions,
    coarsen,
    heldout_log_likelihood,
    lobe_errors,
    mixture_cell_probabilities,
    run_trial,
    sample_cells,
    sample_mixture,
    sphere_grid,
    spread,
)
from relax.sampling.fvml_mixture import FvmlMixture, langevin

Z_AXIS = np.array([0.0, 0.0, 1.0])
C2_Z = np.stack([np.eye(3), np.diag([-1.0, -1.0, 1.0])])


def _unit(vector: list[float]) -> np.ndarray:
    """``vector`` scaled to unit length."""
    v = np.asarray(vector, dtype=np.float64)
    return v / np.linalg.norm(v)


def test_coarsen_undoes_spread_and_keeps_the_total() -> None:
    """Spreading a vector over children and summing them back is the identity; both keep the total."""
    rng = np.random.default_rng(0)
    pmf = rng.random(12 * 4**2)
    assert_matches(coarsen(spread(pmf, 2), 2), pmf)
    assert_matches(coarsen(pmf, 1).sum(), pmf.sum())


def test_sample_cells_keeps_every_draw_inside_its_cell() -> None:
    """Draws from a vector with three non-empty cells land only in them, in proportion to their probabilities."""
    order = 3
    pmf = np.zeros(12 * 4**order)
    pmf[[5, 300, 700]] = [0.2, 0.3, 0.5]
    counts = bin_directions(sample_cells(pmf, order, 200_000, np.random.default_rng(1)), order)
    assert set(np.flatnonzero(counts)) == {5, 300, 700}
    # binomial sd at 2e5 draws is at most 0.0011 of the total
    assert_matches(counts[[5, 300, 700]] / counts.sum(), np.array([0.2, 0.3, 0.5]), rtol=0.01)


def test_sample_cells_spreads_draws_over_the_whole_cell() -> None:
    """Inside one cell the draws fill its children evenly (density constant in a cell)."""
    order = 2
    pmf = np.zeros(12 * 4**order)
    pmf[17] = 1.0
    children = bin_directions(sample_cells(pmf, order, 160_000, np.random.default_rng(2)), order + 2)
    inside = children[17 * 16 : 18 * 16]
    assert inside.sum() == children.sum()
    # 1e4 per child expected, Poisson sd 100
    assert_matches(inside, np.full(16, 1e4), rtol=0.05)


def test_sample_mixture_draws_each_lobe_with_its_weight_spread_and_symmetry_images() -> None:
    """A C2 lobe off the axis: half of its draws about each image, mean cosine A(kappa) about the nearer one."""
    mean, kappa = _unit([1.0, 0.0, 1.0]), 100.0
    images = np.stack([g @ mean for g in C2_Z])  # 90 deg apart, so caps of cosine 0.9 (26 deg) do not overlap
    rng = np.random.default_rng(3)
    cosines = sample_mixture(FvmlMixture([0.25, 0.75], [mean], [kappa], C2_Z), 200_000, rng) @ images.T
    in_cap = cosines.max(axis=1) > 0.9
    # lobe draws fall outside the caps with probability exp(-10); uniform draws inside with 2 * (1 - 0.9) / 2 = 0.1;
    # binomial sd of the share is 1e-3
    assert_matches(in_cap.mean(), 0.75 + 0.25 * 0.1, rtol=0.01)
    assert_matches(np.bincount(cosines[in_cap].argmax(axis=1)) / in_cap.sum(), np.array([0.5, 0.5]), rtol=0.02)
    lobe_only = sample_mixture(FvmlMixture([0.0, 1.0], [mean], [kappa], C2_Z), 100_000, rng) @ images.T
    # sd of the mean cosine over 1e5 draws is 3e-5 (cosine sd 1 / kappa)
    assert_matches(lobe_only.max(axis=1).mean(), langevin(kappa), rtol=2e-4)


def test_mixture_cell_probabilities_integrate_the_density() -> None:
    """Cell probabilities sum to 1 and agree with a finer quadrature of the same mixture."""
    mixture = FvmlMixture([0.3, 0.7], [_unit([0.2, -0.4, 0.9])], [80.0])
    cells = mixture_cell_probabilities(mixture, 3)
    reference = coarsen(mixture_cell_probabilities(mixture, 6), 3)  # quadrature at order 9 instead of 8
    assert_matches(cells.sum(), 1.0)
    # midpoint rule, error 4x smaller per order: measured 6.6e-5 of the largest cell (deterministic)
    assert_matches(cells, reference, rtol=1e-4)


def test_lobe_errors_match_permuted_lobes_through_their_symmetry_images() -> None:
    """Fitted lobes listed in another order, one at its symmetry image, are matched to the right true lobes."""
    truth = FvmlMixture([0.2, 0.5, 0.3], [_unit([1.0, 0.0, 1.0]), _unit([0.0, 1.0, -1.0])], [10.0, 100.0], C2_Z)
    fitted = FvmlMixture([0.25, 0.3, 0.45], [truth.means[1], C2_Z[1] @ truth.means[0]], [110.0, 9.0], C2_Z)
    errors = lobe_errors(fitted, truth)
    assert_matches(np.array(errors["kappa_ratio"]), np.array([0.9, 1.1]))
    assert_matches(np.array(errors["lobe_weight_error"]), np.array([-0.05, 0.0]), rtol=1e-12)
    assert max(errors["lobe_angle_error_deg"]) < 1e-5
    assert_matches(errors["uniform_weight_error"], 0.05)
    assert lobe_errors(FvmlMixture([0.5, 0.5], [Z_AXIS], [1.0]), truth) == {}


def test_trial_scores_every_method_on_both_grids() -> None:
    """A small trial on a one-lobe truth: three methods per order, finite distances, the fits beat the histogram."""
    truth = FvmlMixture([0.4, 0.6], [_unit([0.3, 0.3, 0.9])], [15.0])
    target = Target("tiny", np.eye(3)[None], 2, eval_order=3, source="test", mixture=truth)
    rows, masses = run_trial(
        target,
        3000,
        0,
        [2],
        n_components=2,
        n_starts=2,
        sigma_in_spacings=1.0,
        cell_levels=1,
        max_iter=500,
        tol=1e-10,
        reference_mass=1e6,
    )
    assert [row["method"] for row in rows] == list(METHODS)
    assert masses["order2"].sum() == 3000
    for row in rows:
        assert row["order"] == 2 and row["n_particles"] == 3000
        assert math.isfinite(row["js_bits_fit"]) and math.isfinite(row["js_bits_eval"])
        assert row["c"] == 1.0 and row["n_components"] == 2
    by_method = {row["method"]: row for row in rows}
    # a fit cannot score the held-out draw better than the truth beyond sampling noise
    truth_score = rows[0]["heldout_log_likelihood_truth"]
    for method in ("point", "cell"):
        assert by_method[method]["js_bits_eval"] < by_method["histogram"]["js_bits_eval"]
        assert by_method[method]["heldout_log_likelihood"] <= truth_score + 0.01
        assert by_method[method]["converged"]
        assert len(by_method[method]["kappa_ratio"]) == 1
    assert sphere_grid(2).shape == (192, 3)


def test_heldout_log_likelihood_is_the_mean_log_cell_probability() -> None:
    """Mean log probability of the held-out particles' cells; -inf when one lands on a zero cell; NaN without a draw."""
    pmf = np.array([0.5, 0.25, 0.25, 0.0])
    assert_matches(
        heldout_log_likelihood(np.array([2.0, 1.0, 1.0, 0.0]), pmf), (2 * math.log(0.5) + 2 * math.log(0.25)) / 4
    )
    assert heldout_log_likelihood(np.array([1.0, 0.0, 0.0, 1.0]), pmf) == -math.inf
    assert math.isnan(heldout_log_likelihood(None, pmf))


def test_replay_population_sums_a_mixture_record_by_weight() -> None:
    """A mixture record is the weighted sum of its components; a zero-concentration Kent component is uniform."""
    from scripts.fvml_prior_simulation.replay_refinement import population

    kent = {"poses": "kent", "dataset_params_option": [4.0, 1.0, [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]}
    flat = {"poses": "kent", "dataset_params_option": [0.0, 0.0, [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]}
    record = {"poses": "mixture", "components": [{"weight": 0.66, "poses": kent}, {"weight": 0.34, "poses": flat}]}
    mixed = population(record, 3)
    assert mixed.shape == (768,)
    assert_matches(mixed.sum(), 1.0)
    assert_matches(mixed, 0.66 * population(kent, 3) + 0.34 / 768)


def test_replay_heldout_keys_are_a_fixed_share_independent_of_order() -> None:
    """The held-out set has the requested share and depends on the particles, not on their order in the file."""
    from scripts.fvml_prior_simulation.replay_refinement import heldout_keys

    keys = [(i, f"stack{i % 3}.mrcs") for i in range(1000)]
    chosen = heldout_keys(keys, 0.2, seed=7)
    assert len(chosen) == 200
    assert chosen <= set(keys)
    assert heldout_keys(list(reversed(keys)), 0.2, seed=7) == chosen
    assert heldout_keys(keys, 0.2, seed=8) != chosen


def test_replay_warm_start_is_clipped_to_the_new_cap() -> None:
    """A warm start keeps weights and means and clips every concentration to the cap."""
    from scripts.fvml_prior_simulation.replay_refinement import cap_start

    mixture = FvmlMixture([0.2, 0.5, 0.3], np.stack([Z_AXIS, _unit([1.0, 0.0, 0.0])]), [50.0, 900.0])
    capped = cap_start(mixture, 244.0)
    assert_matches(capped.kappas, np.array([50.0, 244.0]))
    assert_matches(capped.weights, mixture.weights)
    assert_matches(capped.means, mixture.means)
