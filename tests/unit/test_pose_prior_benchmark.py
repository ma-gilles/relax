"""The pose-prior benchmark: evaluator metrics, the simulator's Kent guard, fixed priors and the variant comparison."""

import csv
import json
import logging
import math

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.parity.fixed_direction_prior import PROVENANCE_FILE, FixedDirectionPrior
from relax.sampling.healpix import n_pixels, pixel_rot_tilt
from scripts.pose_prior_benchmark import compare_variants as cmp
from scripts.pose_prior_benchmark import evaluate_run as ev
from scripts.pose_prior_benchmark import simulate_dataset as sim
from scripts.pose_prior_benchmark import true_prior
from scripts.pose_prior_benchmark import write_fixed_prior as wfp


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


# ---- fixed priors and the variant comparison ----


def _prior_args(*flags):
    return wfp.parse_args([*flags, "--order", "2", "--out", "unused.npy"])


def _poses_json(tmp_path, poses):
    path = tmp_path / "pose_distribution.json"
    path.write_text(json.dumps(poses))
    return str(path)


def test_uniform_and_truth_priors_of_a_uniform_dataset_are_flat(tmp_path):
    poses = _poses_json(tmp_path, {"poses": "uniform"})
    for flags in (["--kind", "uniform"], ["--kind", "truth", "--poses-json", poses]):
        assert_matches(wfp.fixed_prior(_prior_args(*flags)), np.full(n_pixels(2), 1.0 / n_pixels(2)))


def test_truth_prior_is_the_datasets_population_and_a_kent_prior_the_explicit_one(tmp_path):
    truth = wfp.fixed_prior(_prior_args("--kind", "truth", "--poses-json", _poses_json(tmp_path, CAP)))
    assert_matches(truth, true_prior.population_prior(CAP, 2))
    belt_flags = ["--kind", "kent", "--kappa", "8", "--beta", "2", "--mu", "1", "1", "0", "--mu0", "0", "1", "0"]
    belt = wfp.fixed_prior(_prior_args(*belt_flags))
    assert_matches(belt, true_prior.kent_population_prior(8.0, 2.0, [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], 2))
    assert 0.5 * np.abs(belt - truth).sum() > 0.9  # measured 0.92: an equatorial belt against a polar cap


def test_fixed_prior_needs_the_flags_of_its_kind():
    with pytest.raises(SystemExit, match="--poses-json"):
        wfp.fixed_prior(_prior_args("--kind", "truth"))
    with pytest.raises(SystemExit, match="--kappa, --beta, --mu, --mu0"):
        wfp.fixed_prior(_prior_args("--kind", "kent"))


def test_written_prior_is_one_relax_reads_at_the_requested_order(tmp_path):
    out = tmp_path / "prior.npy"
    wfp.main(["--kind", "truth", "--poses-json", _poses_json(tmp_path, CAP), "--order", "3", "--out", str(out)])
    prior = FixedDirectionPrior.from_file(out)
    assert prior.order == 3
    assert_matches(prior.values, true_prior.population_prior(CAP, 3))


def test_direction_prior_used_is_read_from_the_sidecar_relax_writes(tmp_path):
    assert ev.read_direction_prior_used(tmp_path) == {"source": "learned"}
    record = {"path": "/prior.npy", "sha256": "abc", "healpix_order": 7}
    (tmp_path / PROVENANCE_FILE).write_text(json.dumps(record))
    assert ev.read_direction_prior_used(tmp_path) == {"source": "fixed", **record}


def _evaluation(tmp_path, name, *, source, auc, median, js_skill):
    """A minimal evaluate_run output directory: the metrics, FSC curves and pose errors the comparison reads."""
    directory = tmp_path / name
    directory.mkdir()
    pose = {"p90": 3 * median, "fraction_over_90": 0.01}
    metrics = {
        "run": {"n_iterations": 12, "final_healpix_order": 5, "direction_prior_used": {"source": source}},
        "resolution": {
            "box": 8,
            "voxel_size_A": 2.0,
            "reported": {"final_all_data_A": 9.5},
            "gt_merged": {"auc": auc, "at_0.5": {"first_A": None}, "at_0.143": {"first_A": 6.0}},
            "half_maps": {"auc": 0.6, "at_0.143": {"first_A": 7.0}},
        },
        "pose_direction_error_deg": {
            "half1": {"median": median, **pose},
            "half2": {"median": median + 1.0, **pose},
        },
        "prior_final": {"half1": {"js_bits": 0.1, "js_skill": js_skill}, "half2": {"js_bits": 0.3, "js_skill": None}},
    }
    (directory / "metrics.json").write_text(json.dumps(metrics))
    with open(directory / "fsc_curves.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["shell", "resolution_A", "half_maps", "gt_merged"])
        for shell, fsc in enumerate([1.0, 0.9, 0.7, 0.4, 0.1]):
            writer.writerow([shell, 8 * 2.0 / shell if shell else "inf", 1.0, fsc])
    with open(directory / "fsc_by_iteration.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration", "healpix_order", "gt_auc"])
        for iteration, order in enumerate([2, 2, 3, 3, 4], start=1):
            writer.writerow([iteration, order, auc * iteration / 5])
    errors = np.random.default_rng(0).lognormal(np.log(median), 1.0, size=(2, 200)).clip(max=180.0)
    np.savez(directory / "pose_errors.npz", half1=errors[0].astype(np.float32), half2=errors[1].astype(np.float32))
    return directory


def test_variants_are_compared_in_a_table_and_a_figure(tmp_path):
    learned = _evaluation(tmp_path, "learned", source="learned", auc=0.8, median=2.0, js_skill=0.9)
    fixed = _evaluation(tmp_path, "fixed_true", source="fixed", auc=0.85, median=1.5, js_skill=0.9)
    out = tmp_path / "out"
    cmp.main(["--runs", f"learned={learned}", f"fixed_true={fixed}", "--out-dir", str(out), "--label", "test"])
    with open(out / "variants.csv", newline="") as handle:
        rows = {row["variant"]: row for row in csv.DictReader(handle)}
    assert list(rows) == ["learned", "fixed_true"]
    assert [rows[v]["direction_prior"] for v in rows] == ["learned", "fixed"]
    assert_matches(float(rows["learned"]["pose_error_median_deg"]), 2.5)  # the mean of the two halves
    assert_matches(float(rows["fixed_true"]["gt_fsc_auc"]), 0.85)
    assert_matches(float(rows["learned"]["learned_prior_js_bits"]), 0.2)
    assert rows["learned"]["gt_resolution_0.5_A"] == ""  # the curve never crossed
    assert rows["learned"]["learned_prior_js_skill"] == ""  # one half has no skill (a uniform truth)
    assert (out / "variants.png").stat().st_size > 0
    iterations, orders, area = cmp.read_iteration_curve(learned)
    assert_matches(iterations, np.arange(1, 6))
    assert_matches(orders, np.array([2, 2, 3, 3, 4]))
    assert_matches(area, 0.8 * np.arange(1, 6) / 5)
    frequency, fsc = cmp.read_gt_curve(learned, json.loads((learned / "metrics.json").read_text()))
    assert_matches(frequency, np.arange(5) / (8 * 2.0))
    assert_matches(fsc, [1.0, 0.9, 0.7, 0.4, 0.1])


def test_pose_errors_of_both_halves_are_read_in_order(tmp_path):
    np.savez(tmp_path / "pose_errors.npz", half1=np.array([1.0, 2.0], np.float32), half2=np.array([3.0], np.float32))
    assert_matches(cmp.read_pose_errors(tmp_path), np.array([1.0, 2.0, 3.0]))


def test_error_density_integrates_to_one_and_peaks_at_the_typical_error():
    errors = 5.0 * 10 ** np.random.default_rng(0).normal(0.0, 0.1, size=20000)
    grid, density = cmp.error_density(errors)
    assert np.all(np.diff(np.log10(grid)) > 0)
    assert_matches(density.sum() * np.log10(grid[1] / grid[0]), 1.0)
    assert abs(np.log10(grid[np.argmax(density)]) - np.log10(5.0)) < 0.05


def test_error_density_keeps_the_mass_beyond_the_ends_of_the_axis():
    low, high = cmp.POSE_ERROR_RANGE_DEG
    grid, density = cmp.error_density(np.array([low / 10, high, high]))
    step = np.log10(grid[1] / grid[0])
    assert_matches(density.sum() * step, 1.0)
    assert density[0] > 0 and density[-1] > density[0]


def test_runs_are_label_equals_directory():
    assert cmp.parse_runs(["a=/x", "b=/y"]) == {"a": cmp.Path("/x"), "b": cmp.Path("/y")}
    with pytest.raises(SystemExit):
        cmp.parse_runs(["no-separator"])


# ---- the per-iteration trace ----


def _noisy(gt, level, seed):
    return gt + level * np.random.default_rng(seed).standard_normal(gt.shape)


def test_fsc_by_iteration_scores_each_half_map_against_the_truth_and_the_other_half():
    gt = np.random.default_rng(0).standard_normal((16, 16, 16))
    maps = {1: (_noisy(gt, 3.0, 1), _noisy(gt, 3.0, 2)), 2: (_noisy(gt, 0.3, 3), _noisy(gt, 0.3, 4))}
    rows = ev.fsc_by_iteration(gt, maps, {1: 2, 2: 3}, voxel=2.0, box=16)
    assert [(r["iteration"], r["healpix_order"]) for r in rows] == [(1, 2), (2, 3)]
    assert 0.0 < rows[0]["gt_auc"] < rows[1]["gt_auc"] <= 1.0  # a less noisy half map correlates better
    assert_matches(rows[1]["gt_auc"], 0.5 * (rows[1]["gt_auc_half1"] + rows[1]["gt_auc_half2"]))
    assert rows[1]["gt_resolution_0.5_half1_A"] is None  # the low-noise half map never drops below 0.5
    assert rows[0]["gt_resolution_0.5_half1_A"] is not None
    perfect = ev.fsc_by_iteration(gt, {1: (gt, gt)}, {1: 2}, voxel=2.0, box=16)
    assert_matches(perfect[0]["gt_auc"], 1.0)
    assert perfect[0]["half_map_resolution_0.143_A"] is None


def test_iteration_table_is_written_with_empty_cells_for_a_crossing_that_never_happened(tmp_path):
    rows = [
        {"iteration": 1, "healpix_order": 2, "gt_auc": 0.4, "half_map_resolution_0.143_A": 12.0},
        {"iteration": 2, "healpix_order": 3, "gt_auc": 0.5, "half_map_resolution_0.143_A": None},
    ]
    result = {
        "metrics": {},
        "rows": [{"iteration": 1, "half": 1}],
        "iteration_rows": rows,
        "curves": {"half_maps": np.ones(3)},
        "pose_errors": {1: np.array([0.5, 3.0]), 2: np.array([170.0])},
        "box": 4,
        "voxel": 2.0,
    }
    ev.write_tables(result, tmp_path)
    with np.load(tmp_path / "pose_errors.npz") as archive:
        assert_matches(archive["half1"], np.array([0.5, 3.0]))
        assert_matches(archive["half2"], np.array([170.0]))
    with open(tmp_path / "fsc_by_iteration.csv", newline="") as handle:
        written = list(csv.DictReader(handle))
    assert [r["half_map_resolution_0.143_A"] for r in written] == ["12.0", ""]
    assert [int(r["healpix_order"]) for r in written] == [2, 3]
