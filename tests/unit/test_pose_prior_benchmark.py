"""The pose-prior benchmark's evaluator metrics and its simulator's Kent guard."""

import logging
import math

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.sampling.healpix import n_pixels, pixel_rot_tilt
from scripts.pose_prior_benchmark import evaluate_run as ev
from scripts.pose_prior_benchmark import simulate_dataset as sim


def _write_model_star(path, prior):
    """A half-set model STAR holding only the direction-prior block."""
    lines = ["data_model_pdf_orient_class_1\n", "\n", "loop_\n", "_rlnOrientationDistribution #1\n"]
    lines += [f"{value:.12g}\n" for value in prior]
    path.write_text("".join(lines))


@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_direction_histogram_puts_each_pixel_centre_in_its_own_pixel(order):
    pixels = np.arange(n_pixels(order))
    rot, tilt = pixel_rot_tilt(order, pixels)
    counts = ev.direction_histogram(rot, tilt, order)
    assert_matches(counts, np.ones(n_pixels(order)))


def test_distribution_metrics_of_identical_distributions_are_zero():
    p = np.array([4.0, 1.0, 0.0, 3.0])
    metrics = ev.distribution_metrics(p, p / p.sum())
    for name in ("tv", "js_bits", "hellinger", "kl_bits", "truth_mass_on_zero_prior"):
        assert metrics[name] == 0.0


def test_distribution_metrics_of_disjoint_distributions_are_maximal():
    metrics = ev.distribution_metrics(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    assert_matches(metrics["tv"], 1.0)
    assert_matches(metrics["js_bits"], 1.0)
    assert_matches(metrics["hellinger"], 1.0)
    assert math.isinf(metrics["kl_bits"])
    assert_matches(metrics["truth_mass_on_zero_prior"], 1.0)


def test_distribution_metrics_match_a_two_point_example():
    # truth (1, 0) against estimate (1/2, 1/2): KL = 1 bit, TV = 1/2, JS from the midpoint (3/4, 1/4).
    metrics = ev.distribution_metrics(np.array([2.0, 0.0]), np.array([0.5, 0.5]))
    midpoint_js = 0.5 * math.log2(1 / 0.75) + 0.5 * (0.5 * math.log2(0.5 / 0.75) + 0.5 * math.log2(0.5 / 0.25))
    assert_matches(metrics["kl_bits"], 1.0)
    assert_matches(metrics["tv"], 0.5)
    assert_matches(metrics["js_bits"], midpoint_js)
    assert_matches(metrics["hellinger"], math.sqrt(0.5 * ((1 - math.sqrt(0.5)) ** 2 + 0.5)))


def test_crossing_summary_reports_first_and_sustained_crossings():
    curve = np.array([1.0, 0.9, 0.6, 0.4, 0.6, 0.3, 0.2, 0.1, 0.1])
    summary = ev.crossing_summary(curve, 0.5, voxel=2.0, box=40)
    assert summary["first_shell"] == 3
    assert summary["sustained_shell"] == 5
    assert_matches(summary["first_A"], 40 * 2.0 / 3)
    assert_matches(summary["sustained_A"], 40 * 2.0 / 5)
    never = ev.crossing_summary(np.ones(6), 0.5, voxel=2.0, box=40)
    assert never["first_A"] is None and never["sustained_A"] is None


def test_read_direction_prior_accepts_a_probability_vector_and_rejects_others(tmp_path):
    good = tmp_path / "good_model.star"
    _write_model_star(good, [0.25, 0.25, 0.5])
    assert_matches(ev.read_direction_prior(good), [0.25, 0.25, 0.5])
    bad = tmp_path / "bad_model.star"
    _write_model_star(bad, [0.25, 0.25, 0.25])
    with pytest.raises(ValueError, match="not a probability vector"):
        ev.read_direction_prior(bad)
    empty = tmp_path / "empty_model.star"
    empty.write_text("data_model_general\n\n_rlnNrClasses 1\n")
    assert ev.read_direction_prior(empty) is None


CAP = {"poses": "kent", "dataset_params_option": [8.0, 2.0, [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]}


def test_uniform_population_prior_is_flat():
    assert_matches(ev.population_prior({"poses": "uniform"}, 2), np.full(n_pixels(2), 1.0 / n_pixels(2)))


@pytest.mark.parametrize("mu0", [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
def test_kent_population_prior_is_as_close_to_the_simulators_viewing_directions_as_an_ideal_sample(mu0):
    from recovar.simulation import simulator

    order, n_images = 2, 100000
    option = [8.0, 2.0, [1.0, 1.0, 0.0], mu0]
    _, rotations, _ = simulator.get_pose_ctf_generator(option)(n_images, 128)
    viewing = np.asarray(rotations)[:, 2, :]  # RELION's viewing direction is the third row of the rotation
    tilt = np.degrees(np.arccos(np.clip(viewing[:, 2], -1.0, 1.0)))
    rot = np.degrees(np.arctan2(viewing[:, 1], viewing[:, 0]))
    sample = ev.direction_histogram(rot, tilt, order) / n_images

    population = ev.population_prior({"poses": "kent", "dataset_params_option": option}, order)
    ideal = np.bincount(
        np.random.default_rng(0).choice(population.size, n_images, p=population), minlength=population.size
    )
    assert_matches(population.sum(), 1.0)
    tv_simulator = 0.5 * np.abs(population - sample).sum()
    tv_ideal = 0.5 * np.abs(population - ideal / n_images).sum()
    assert tv_simulator < 1.5 * tv_ideal  # measured ratio 0.87 (cap) and 0.87 (belt)


def test_prior_rows_score_the_population_as_perfect_and_a_uniform_prior_as_the_baseline(tmp_path):
    order = 3
    population = ev.population_prior(CAP, order)
    pixels = np.random.default_rng(0).choice(population.size, 3000, p=population)
    rot, tilt = pixel_rot_tilt(order, pixels)
    halves = {half: {"true_rot": rot, "true_tilt": tilt} for half in ev.HALVES}
    _write_model_star(tmp_path / "run_it003_half1_model.star", population)
    _write_model_star(tmp_path / "run_it003_half2_model.star", np.full(population.size, 1.0 / population.size))

    rows = {row["half"]: row for row in ev.prior_rows(tmp_path, [3], halves, CAP)}

    assert rows[1]["healpix_order"] == order and rows[1]["n_particles"] == 3000
    assert rows[1]["js_bits"] < 1e-12  # float64 round-off of a divergence that is exactly zero
    assert_matches(rows[1]["js_skill"], 1.0)
    assert_matches(rows[2]["js_bits"], rows[2]["js_bits_uniform"])
    assert abs(rows[2]["js_skill"]) < 1e-12  # a uniform prior is the zero of the skill score
    assert 0 < rows[1]["js_bits_oracle"] < rows[1]["js_bits_uniform"]  # a sampled histogram is far better than flat
    assert rows[1]["kl_bits"] < 1e-12 and rows[1]["hellinger"] < 1e-6 and rows[1]["truth_mass_on_zero_prior"] == 0
    assert_matches(rows[2]["hellinger"], rows[2]["hellinger_uniform"])
    assert_matches(rows[2]["kl_bits"], rows[2]["kl_bits_uniform"])
    assert 0 < rows[1]["js_skill_oracle"] < 1  # the true-pose histogram sits between the uniform prior and the truth
    assert 0 <= rows[1]["zero_prior_fraction_oracle"] < 1 and 0 <= rows[1]["truth_mass_on_zero_prior_oracle"] < 1
    assert rows[1]["js_bits_uniform"] > 0.5  # the cap holds most of the mass in a small part of the sphere


def test_a_run_seeded_from_the_input_poses_is_refused(tmp_path):
    np.savez(tmp_path / "refinement_results.npz", initial_pose_source_resolved=np.asarray("none"))
    assert ev.require_unseeded_run(tmp_path) == "none"
    np.savez(tmp_path / "refinement_results.npz", initial_pose_source_resolved=np.asarray("input_star"))
    with pytest.raises(ValueError, match="--initial-pose-source none"):
        ev.require_unseeded_run(tmp_path)


def test_summary_figure_is_written_for_an_infinite_kl_and_an_undefined_skill(tmp_path):
    order = 2
    n = n_pixels(order)
    prior = np.zeros(n)
    prior[: n // 2] = 2.0 / n  # zero on half of the directions, where the uniform truth has mass
    for half in ev.HALVES:
        _write_model_star(tmp_path / f"run_it001_half{half}_model.star", prior)
    rot, tilt = pixel_rot_tilt(order, np.arange(n))
    halves = {half: {"true_rot": rot, "true_tilt": tilt} for half in ev.HALVES}
    rows = ev.prior_rows(tmp_path, [1], halves, {"poses": "uniform"})
    assert math.isinf(rows[0]["kl_bits"]) and math.isnan(rows[0]["js_skill"])

    curve = np.linspace(1.0, 0.0, 20)
    result = {"rows": rows, "curves": {"half_maps": curve, "gt_merged": curve}, "box": 40, "voxel": 2.0}
    figure = tmp_path / "summary.png"
    ev.plot_summary(result, "test", figure)
    assert figure.stat().st_size > 0


def test_direction_error_is_the_angle_between_viewing_directions():
    error = ev.direction_error_deg(
        np.array([0.0, 90.0, 10.0]),
        np.array([90.0, 90.0, 0.0]),
        np.array([90.0, 90.0, 170.0]),
        np.array([90.0, 90.0, 0.0]),
    )
    assert_matches(error, [90.0, 0.0, 0.0])


def test_kent_guard_stops_the_simulation_when_the_sampler_falls_back():
    logger = logging.getLogger("test_pose_prior_benchmark.simulator")
    guard = sim.KentFallbackGuard()
    logger.addHandler(guard)
    try:
        logger.warning("sphstat is not installed; using internal Kent/Fisher-Bingham fallback sampler")
        with pytest.raises(RuntimeError, match="would not follow"):
            logger.warning("Kent rejection fallback accepted %d/%d samples after %d draws", 5, 10, 100)
    finally:
        logger.removeHandler(guard)


def test_pose_options_and_flag_checks():
    kent = sim.parse_args(
        ["--output-dir", "x", "--n-images", "10", "--grid-size", "64", "--poses", "kent", "--kappa", "8", "--beta", "2"]
        + ["--mu", "1", "1", "0", "--mu0", "0", "1", "0"]
    )
    assert sim.pose_option(kent) == [8.0, 2.0, [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    uniform = sim.parse_args(["--output-dir", "x", "--n-images", "10", "--grid-size", "64", "--poses", "uniform"])
    assert sim.pose_option(uniform) == "uniform"
    with pytest.raises(SystemExit):
        sim.parse_args(["--output-dir", "x", "--n-images", "10", "--grid-size", "64", "--poses", "kent"])
    with pytest.raises(SystemExit):
        sim.parse_args(
            ["--output-dir", "x", "--n-images", "10", "--grid-size", "64", "--poses", "uniform", "--kappa", "8"]
        )
