"""The pose-prior benchmark: evaluator metrics, the simulator's Kent guard, fixed priors, the variant comparison and the
ab-initio start map."""

import csv
import json
import logging
import math

import mrcfile
import numpy as np
import pandas as pd
import pytest
import starfile
from helpers.float_compare import assert_matches
from scipy import ndimage
from scipy.spatial.transform import Rotation

from relax.diagnostics.gt_registration import apply_rigid_volume_transform
from relax.io.map_io import require_relion_convention_reference
from relax.parity.fixed_direction_prior import PROVENANCE_FILE, FixedDirectionPrior
from relax.sampling.healpix import n_pixels, pixel_rot_tilt
from scripts.conical_fsc import ConicalFsc, cone_axes, conical_fsc, conical_fsc_summary
from scripts.pose_prior_benchmark import align_to_truth as al
from scripts.pose_prior_benchmark import compare_variants as cmp
from scripts.pose_prior_benchmark import conical_fsc_plot as cfp
from scripts.pose_prior_benchmark import evaluate_run as ev
from scripts.pose_prior_benchmark import initial_model as im
from scripts.pose_prior_benchmark import reconstruct_with_poses as rwp
from scripts.pose_prior_benchmark import simulate_dataset as sim
from scripts.pose_prior_benchmark import star_nonuniformity as snu
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
    curves = {"half_maps": curve, "gt_merged": curve, "gt_merged_aligned": curve}
    result = {"rows": rows, "curves": curves, "box": 40, "voxel": 2.0}
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


# Three cones of a box-8 map (shells 0 to 2, 500 voxels each): the half maps' and the truth pair's cFSC curves.
_HALF_MAP_CONE_CURVES = np.array([[np.nan, 0.9, 0.9], [np.nan, 0.9, 0.7], [np.nan, 0.9, 0.5]])
_GT_CONE_CURVES = np.array([[np.nan, 0.9, 0.8], [np.nan, 0.9, 0.6], [np.nan, 0.8, 0.3]])


def _write_conical_fsc(directory):
    """The cFSC summaries ``metrics.json`` holds and the ``conical_fsc.npz`` evaluate_run writes, for the curves
    above; returns the summaries."""
    axes = cone_axes(3)
    cfscs = {
        name: ConicalFsc(axes, 20.0, curves, np.full((3, 3), 500))
        for name, curves in (("half_maps", _HALF_MAP_CONE_CURVES), ("gt_merged_aligned", _GT_CONE_CURVES))
    }
    np.savez(
        directory / "conical_fsc.npz",
        axes=axes,
        **{f"{name}_curves": cfsc.curves for name, cfsc in cfscs.items()},
        **{f"{name}_voxels": cfsc.voxels for name, cfsc in cfscs.items()},
    )
    return {name: conical_fsc_summary(cfsc, voxel=2.0, box=8) for name, cfsc in cfscs.items()}


def _evaluation(tmp_path, name, *, source, auc, median, js_skill, with_conical=True):
    """A minimal evaluate_run output directory: the metrics, FSC curves, pose errors and cFSC the comparison reads.
    The aligned FSC area is ``auc`` and the aligned pose error ``median``; as refined they are 0.05 lower and 0.5
    higher."""
    directory = tmp_path / name
    directory.mkdir()
    pose = {"p90": 3 * median, "fraction_over_90": 0.01}
    refined_pose = {"p90": 3 * median + 1.0, "fraction_over_90": 0.02}
    metrics = {
        "run": {"n_iterations": 12, "final_healpix_order": 5, "direction_prior_used": {"source": source}},
        "resolution": {
            "box": 8,
            "voxel_size_A": 2.0,
            "reported": {"final_all_data_A": 9.5},
            "gt_merged": {"auc": auc - 0.05, "at_0.5": {"first_A": 9.0}, "at_0.143": {"first_A": 6.5}},
            "gt_merged_aligned": {"auc": auc, "at_0.5": {"first_A": None}, "at_0.143": {"first_A": 6.0}},
            "half_maps": {"auc": 0.6, "at_0.143": {"first_A": 7.0}},
            "alignment": {"angle_deg": 0.35, "shift_A": 0.1},
        },
        "pose_direction_error_aligned_deg": {
            "half1": {"median": median, **pose},
            "half2": {"median": median + 1.0, **pose},
        },
        "pose_direction_error_deg": {
            "half1": {"median": median + 0.5, **refined_pose},
            "half2": {"median": median + 1.5, **refined_pose},
        },
        "pose_frame_offset": {"angle_deg": 0.45},
        "prior_final": {"half1": {"js_bits": 0.1, "js_skill": js_skill}, "half2": {"js_bits": 0.3, "js_skill": None}},
    }
    if with_conical:
        metrics["conical_fsc"] = _write_conical_fsc(directory)
    (directory / "metrics.json").write_text(json.dumps(metrics))
    with open(directory / "fsc_curves.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["shell", "resolution_A", "half_maps", "gt_merged", "gt_merged_aligned"])
        for shell, fsc in enumerate([1.0, 0.9, 0.7, 0.4, 0.1]):
            writer.writerow([shell, 8 * 2.0 / shell if shell else "inf", 1.0, 0.9 * fsc, fsc])
    with open(directory / "fsc_by_iteration.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration", "healpix_order", "fourier_radius", "gt_auc", "half_map_auc"])
        for iteration, order in enumerate([2, 2, 3, 3, 4], start=1):
            writer.writerow([iteration, order, 10 * iteration, auc * iteration / 5, 0.5 + 0.1 * iteration])
    errors = np.random.default_rng(0).lognormal(np.log(median), 1.0, size=(2, 200)).clip(max=180.0)
    np.savez(
        directory / "pose_errors.npz",
        half1=(errors[0] + 0.5).astype(np.float32),
        half2=(errors[1] + 0.5).astype(np.float32),
        aligned_half1=errors[0].astype(np.float32),
        aligned_half2=errors[1].astype(np.float32),
    )
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
    assert_matches(float(rows["learned"]["pose_error_median_aligned_deg"]), 2.5)  # the mean of the two halves
    assert_matches(float(rows["learned"]["pose_error_median_as_refined_deg"]), 3.0)
    assert_matches(float(rows["fixed_true"]["gt_fsc_auc_aligned"]), 0.85)
    assert_matches(float(rows["fixed_true"]["gt_fsc_auc_as_refined"]), 0.80)
    assert_matches(
        [float(rows["learned"][k]) for k in ("pose_offset_deg", "map_offset_deg", "map_shift_A")], [0.45, 0.35, 0.1]
    )
    assert_matches(float(rows["learned"]["learned_prior_js_bits"]), 0.2)
    # cFAR: the cone that crosses 0.143 never has the largest area; the smallest cone's curve ends at 0.3
    weights = 4.0 * math.pi * np.arange(3) ** 2
    gt_area = [weights[1:] @ np.nan_to_num(curve[1:]) for curve in _GT_CONE_CURVES]
    half_area = [weights[1:] @ np.nan_to_num(curve[1:]) for curve in _HALF_MAP_CONE_CURVES]
    assert_matches(float(rows["learned"]["gt_cfar_aligned"]), min(gt_area) / max(gt_area))
    assert_matches(float(rows["learned"]["half_map_cfar"]), min(half_area) / max(half_area))
    assert rows["learned"]["gt_worst_cone_resolution_0.143_A_aligned"] == ""  # no cone crosses 0.143
    assert rows["learned"]["gt_resolution_0.5_A_aligned"] == ""  # the aligned curve never crossed
    assert_matches(float(rows["learned"]["gt_resolution_0.143_A_aligned"]), 6.0)
    assert rows["learned"]["learned_prior_js_skill"] == ""  # one half has no skill (a uniform truth)
    assert (out / "variants.png").stat().st_size > 0
    curve = cmp.read_iteration_curve(learned)
    assert_matches(curve.iteration, np.arange(1, 6))
    assert_matches(curve.order, np.array([2, 2, 3, 3, 4]))
    assert_matches(curve.radius, 10 * np.arange(1, 6))
    assert_matches(curve.gt_area, 0.8 * np.arange(1, 6) / 5)
    assert_matches(curve.half_area, 0.5 + 0.1 * np.arange(1, 6))
    frequency, fsc = cmp.read_gt_curve(learned, json.loads((learned / "metrics.json").read_text()))
    assert_matches(frequency, np.arange(5) / (8 * 2.0))
    assert_matches(fsc, [1.0, 0.9, 0.7, 0.4, 0.1])  # the aligned curve; the as-refined one is 0.9 times it


def test_the_cone_curve_of_a_variant_is_the_mean_and_sd_over_its_cones(tmp_path):
    directory = _evaluation(tmp_path, "learned", source="learned", auc=0.8, median=2.0, js_skill=0.9)
    curve = cmp.read_conical_curve(directory, json.loads((directory / "metrics.json").read_text()))
    assert_matches(curve.frequency, np.arange(3) / (8 * 2.0))
    assert_matches(curve.mean[1:], _GT_CONE_CURVES[:, 1:].mean(axis=0))
    assert_matches(curve.std[1:], _GT_CONE_CURVES[:, 1:].std(axis=0))
    assert np.isnan(curve.mean[0]) and np.isnan(curve.std[0])  # shell 0 is in no cone


def test_a_variant_evaluated_before_the_conical_fsc_is_refused_not_compared_without_it(tmp_path):
    old = _evaluation(tmp_path, "old", source="learned", auc=0.8, median=2.0, js_skill=0.9, with_conical=False)
    with pytest.raises(ValueError, match="old: metrics.json has no conical_fsc; rerun evaluate_run.py"):
        cmp.main(["--runs", f"old={old}", "--out-dir", str(tmp_path / "out"), "--label", "test"])


def test_aligned_pose_errors_of_both_halves_are_read_in_order(tmp_path):
    np.savez(
        tmp_path / "pose_errors.npz",
        half1=np.array([9.0, 9.0], np.float32),
        half2=np.array([9.0], np.float32),
        aligned_half1=np.array([1.0, 2.0], np.float32),
        aligned_half2=np.array([3.0], np.float32),
    )
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
    rows = ev.fsc_by_iteration(gt, maps, {1: 2, 2: 3}, {1: 28, 2: 56}, voxel=2.0, box=16)
    assert [(r["iteration"], r["healpix_order"], r["fourier_radius"]) for r in rows] == [(1, 2, 28), (2, 3, 56)]
    assert 0.0 < rows[0]["gt_auc"] < rows[1]["gt_auc"] <= 1.0  # a less noisy half map correlates better
    assert_matches(rows[1]["gt_auc"], 0.5 * (rows[1]["gt_auc_half1"] + rows[1]["gt_auc_half2"]))
    assert 0.0 < rows[0]["half_map_auc"] < 0.5  # two independent noisy copies agree little
    assert rows[1]["gt_resolution_0.5_half1_A"] is None  # the low-noise half map never drops below 0.5
    assert rows[0]["gt_resolution_0.5_half1_A"] is not None
    perfect = ev.fsc_by_iteration(gt, {1: (gt, gt)}, {1: 2}, {1: 8}, voxel=2.0, box=16)
    assert_matches(perfect[0]["gt_auc"], 1.0)
    assert_matches(perfect[0]["half_map_auc"], 1.0)
    assert perfect[0]["half_map_resolution_0.143_A"] is None


def test_fourier_radius_is_half_the_image_size_both_half_sets_ran_with(tmp_path):
    for half, size in ((1, 118), (2, 118)):
        (tmp_path / f"run_it003_half{half}_model.star").write_text(
            f"data_model_general\n\n_rlnCurrentResolution 19.4\n_rlnCurrentImageSize {size}\n"
        )
    assert ev.read_fourier_radius(tmp_path, 3) == 59
    (tmp_path / "run_it003_half2_model.star").write_text("data_model_general\n\n_rlnCurrentImageSize 120\n")
    with pytest.raises(ValueError, match="different image sizes"):
        ev.read_fourier_radius(tmp_path, 3)


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
        "conical_fsc": {
            "half_maps": ConicalFsc(cone_axes(2), 20.0, np.full((2, 3), 0.5), np.full((2, 3), 7)),
            "gt_merged_aligned": ConicalFsc(cone_axes(2), 20.0, np.full((2, 3), 0.25), np.full((2, 3), 9)),
        },
        "pose_errors": {1: np.array([0.5, 3.0]), 2: np.array([170.0])},
        "aligned_pose_errors": {1: np.array([0.4, 2.5]), 2: np.array([169.0])},
        "box": 4,
        "voxel": 2.0,
    }
    ev.write_tables(result, tmp_path)
    with np.load(tmp_path / "pose_errors.npz") as archive:
        assert_matches(archive["half1"], np.array([0.5, 3.0]))
        assert_matches(archive["half2"], np.array([170.0]))
        assert_matches(archive["aligned_half1"], np.array([0.4, 2.5]))
        assert_matches(archive["aligned_half2"], np.array([169.0]))
    with np.load(tmp_path / "conical_fsc.npz") as archive:
        assert sorted(archive.files) == [
            "axes",
            "gt_merged_aligned_curves",
            "gt_merged_aligned_voxels",
            "half_maps_curves",
            "half_maps_voxels",
        ]
        assert_matches(archive["axes"], cone_axes(2))
        assert_matches(archive["half_maps_curves"], np.full((2, 3), 0.5))
        assert_matches(archive["gt_merged_aligned_curves"], np.full((2, 3), 0.25))
        assert archive["gt_merged_aligned_voxels"].tolist() == [[9, 9, 9]] * 2
    with open(tmp_path / "fsc_by_iteration.csv", newline="") as handle:
        written = list(csv.DictReader(handle))
    assert [r["half_map_resolution_0.143_A"] for r in written] == ["12.0", ""]
    assert [int(r["healpix_order"]) for r in written] == [2, 3]


# ---- relion_reconstruct from given poses ----


def _reconstruction(tmp_path, name, *, auc):
    """A minimal reconstruct_with_poses output directory: the FSC summary and curve the comparison reads."""
    directory = tmp_path / name
    directory.mkdir()
    summary = {"auc": auc, "at_0.5": {"first_A": 12.0}, "at_0.143": {"first_A": None}}
    metrics = {"resolution": {"box": 8, "voxel_size_A": 2.0, "gt_reconstruction": summary}}
    (directory / "metrics.json").write_text(json.dumps(metrics))
    with open(directory / "fsc_curve.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["shell", "resolution_A", "gt_reconstruction"])
        for shell, fsc in enumerate([1.0, 0.8, 0.5, 0.3, 0.1]):
            writer.writerow([shell, 8 * 2.0 / shell if shell else "inf", fsc * auc])
    return directory


def test_reconstructions_are_compared_in_a_table_and_a_figure(tmp_path):
    learned = _evaluation(tmp_path, "learned", source="learned", auc=0.8, median=2.0, js_skill=0.9)
    fixed = _evaluation(tmp_path, "fixed_true", source="fixed", auc=0.85, median=1.5, js_skill=0.9)
    estimated = [_reconstruction(tmp_path, f"est_{v}", auc=a) for v, a in (("learned", 0.5), ("fixed_true", 0.4))]
    angles = [_reconstruction(tmp_path, f"ang_{v}", auc=a) for v, a in (("learned", 0.55), ("fixed_true", 0.45))]
    true = _reconstruction(tmp_path, "true", auc=0.9)
    out = tmp_path / "out"
    cmp.main(
        ["--runs", f"learned={learned}", f"fixed_true={fixed}"]
        + ["--reconstructions", f"learned={estimated[0]}", f"fixed_true={estimated[1]}"]
        + ["--angle-reconstructions", f"learned={angles[0]}", f"fixed_true={angles[1]}"]
        + ["--true-reconstruction", str(true), "--out-dir", str(out), "--label", "test"]
    )
    with open(out / "reconstructions.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [(r["poses"], r["variant"]) for r in rows] == [
        ("estimated", "learned"),
        ("estimated", "fixed_true"),
        ("estimated_angles", "learned"),
        ("estimated_angles", "fixed_true"),
        ("true", ""),
    ]
    assert_matches([float(r["gt_fsc_auc"]) for r in rows], [0.5, 0.4, 0.55, 0.45, 0.9])
    assert rows[0]["gt_resolution_0.143_A"] == ""  # the curve never crossed
    assert (out / "variants.png").stat().st_size > 0
    reconstruction = cmp.read_reconstruction(estimated[0])
    assert_matches(reconstruction.frequency, np.arange(5) / (8 * 2.0))
    assert_matches(reconstruction.fsc, 0.5 * np.array([1.0, 0.8, 0.5, 0.3, 0.1]))


def test_reconstructions_must_be_given_together_and_cover_every_run(tmp_path):
    learned = _evaluation(tmp_path, "learned", source="learned", auc=0.8, median=2.0, js_skill=0.9)
    only = _reconstruction(tmp_path, "est", auc=0.5)
    base = ["--runs", f"learned={learned}", "--out-dir", str(tmp_path / "out")]
    with pytest.raises(SystemExit):
        cmp.main(base + ["--reconstructions", f"learned={only}"])
    with pytest.raises(SystemExit):
        cmp.main(
            base
            + ["--reconstructions", f"other={only}", "--angle-reconstructions", f"other={only}"]
            + ["--true-reconstruction", str(only)]
        )


def test_zero_origin_star_keeps_every_other_column(tmp_path):
    import pandas as pd
    import starfile

    optics = pd.DataFrame({"rlnOpticsGroup": [1], "rlnImagePixelSize": [2.0]})
    particles = pd.DataFrame(
        {
            "rlnImageName": ["1@a.mrcs", "2@a.mrcs"],
            "rlnAngleRot": [10.0, 20.0],
            "rlnOriginXAngst": [1.5, -2.0],
            "rlnOriginYAngst": [0.5, 3.0],
        }
    )
    star = tmp_path / "in.star"
    starfile.write({"optics": optics, "particles": particles}, star)
    rwp.write_zero_origin_star(star, tmp_path / "out.star")
    written = starfile.read(tmp_path / "out.star", always_dict=True)["particles"]
    assert_matches(written["rlnOriginXAngst"].to_numpy(), np.zeros(2))
    assert_matches(written["rlnOriginYAngst"].to_numpy(), np.zeros(2))
    assert_matches(written["rlnAngleRot"].to_numpy(), np.array([10.0, 20.0]))
    assert list(written["rlnImageName"]) == ["1@a.mrcs", "2@a.mrcs"]


def test_pose_star_is_the_last_numbered_iteration_or_the_datasets_true_poses(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    for iteration in (1, 2, 10):
        for half in (1, 2):
            (run / f"run_it{iteration:03d}_half{half}_model.star").write_text("")
    np.savez(run / "refinement_results.npz", initial_pose_source_resolved="none")
    assert rwp.pose_star(tmp_path, run) == (run / "run_it010_data.star", 10)
    assert rwp.pose_star(tmp_path, None) == (tmp_path / "particles.star", None)
    np.savez(run / "refinement_results.npz", initial_pose_source_resolved="input")
    with pytest.raises(ValueError, match="needs --initial-pose-source none"):
        rwp.pose_star(tmp_path, run)


def test_relion_reconstruct_is_called_with_the_ctf_and_the_module_environment(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(rwp.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    rwp.run_relion_reconstruct(
        tmp_path / "poses.star", tmp_path, tmp_path / "map.mrc", "relion/5.0.1/cpu", tmp_path / "log.txt"
    )
    (command,), kwargs = calls[0]
    assert command[:2] == ["bash", "-lc"]
    assert command[2].startswith("module load relion/5.0.1/cpu && relion_reconstruct --i ")
    assert "--ctf --pad 2" in command[2]
    assert kwargs["cwd"] == tmp_path and kwargs["check"] is True


def test_reconstruction_is_scored_by_its_fsc_to_the_ground_truth():
    gt = np.random.default_rng(0).standard_normal((16, 16, 16))
    summary, curve = rwp.score_reconstruction(gt, gt, voxel=2.0)
    assert_matches(summary["gt_reconstruction"]["auc"], 1.0)
    assert summary["box"] == 16
    assert_matches(curve, np.ones_like(curve))
    with pytest.raises(ValueError, match="correlate"):
        rwp.score_reconstruction(-gt, gt, voxel=2.0)
    with pytest.raises(ValueError, match="shape"):
        rwp.score_reconstruction(gt[:8, :8, :8], gt, voxel=2.0)


def test_frame_offset_is_the_rotation_between_true_and_estimated_directions_despite_noise_and_outliers():
    rng = np.random.default_rng(0)
    n = 4000
    tilt = np.where(rng.random(n) < 0.9, rng.uniform(0, 50, n), rng.uniform(50, 90, n))
    true = ev.viewing_directions(rng.uniform(-180, 180, n), tilt)
    offset = Rotation.from_rotvec(np.radians([2.0, -1.0, 2.5]))
    noisy = offset.apply(true) + 0.07 * rng.standard_normal((n, 3))  # about 4 degrees of pose error
    est = noisy / np.linalg.norm(noisy, axis=1, keepdims=True)
    wrong = rng.random(n) < 0.1
    est[wrong] = ev.viewing_directions(rng.uniform(-180, 180, wrong.sum()), rng.uniform(0, 180, wrong.sum()))
    result = ev.frame_offset(true, est)
    # 6 degrees of noise over 3600 particles, mostly near the pole (sin tilt about 0.4), fits the rotation about the
    # pole to about 0.25 degrees
    assert_matches(result["angle_deg"], np.degrees(np.linalg.norm(offset.as_rotvec())), rtol=0.12)
    assert 0.88 < result["inlier_fraction"] < 0.93  # the 10% of random directions, bar the few that land near the fit


def test_frame_offset_of_identical_directions_is_zero():
    true = ev.viewing_directions(np.linspace(-180, 170, 50), np.linspace(5, 120, 50))
    result = ev.frame_offset(true, true)
    assert result["angle_deg"] == pytest.approx(0.0, abs=1e-3)
    assert result["inlier_fraction"] == 1.0


def test_turning_the_estimates_back_by_the_frame_offset_removes_it():
    rng = np.random.default_rng(0)
    true = ev.viewing_directions(rng.uniform(-180, 180, 300), rng.uniform(5, 100, 300))
    offset = Rotation.from_rotvec(np.radians([0.3, -0.2, 0.4]))
    est = offset.apply(true)
    est_rot, est_tilt = np.degrees(np.arctan2(est[:, 1], est[:, 0])), np.degrees(np.arccos(est[:, 2]))
    true_rot, true_tilt = np.degrees(np.arctan2(true[:, 1], true[:, 0])), np.degrees(np.arccos(true[:, 2]))
    result = ev.frame_offset(true, est)
    assert_matches(np.asarray(result["rotation_matrix"]), offset.as_matrix())
    aligned = ev.aligned_direction_error_deg(
        true_rot, true_tilt, est_rot, est_tilt, np.asarray(result["rotation_matrix"])
    )
    refined = ev.direction_error_deg(true_rot, true_tilt, est_rot, est_tilt)
    assert aligned.max() < 1e-4  # arccos resolves a cosine within 1e-16 of 1 to about 1e-6 degrees
    assert refined.max() > 0.1  # the offset is 0.55 degrees; a direction on its axis does not move


def _blob_volume(n: int, seed: int) -> np.ndarray:
    """A smooth asymmetric map: eight Gaussian blobs of 2.5 to 4.5 voxels in a cube of ``n`` voxels."""
    rng = np.random.default_rng(seed)
    axis = np.arange(n) - (n - 1) / 2
    x = np.meshgrid(axis, axis, axis, indexing="ij")
    volume = np.zeros((n,) * 3)
    for _ in range(8):
        centre, sigma = rng.uniform(-0.25 * n, 0.25 * n, 3), rng.uniform(2.5, 4.5, 3)
        volume += rng.uniform(0.5, 1.5) * np.exp(-sum(((x[i] - centre[i]) / sigma[i]) ** 2 for i in range(3)) / 2)
    return volume


def _offset_map(volume: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``volume`` turned 1.5 degrees about a random axis and shifted (0.3, -0.2, 0.1) voxel: the map and the
    transform (rotation matrix, shift)."""
    axis = np.random.default_rng(100 + seed).normal(size=3)
    rotation = Rotation.from_rotvec(np.radians(1.5) * axis / np.linalg.norm(axis)).as_matrix()
    shift = np.array([0.3, -0.2, 0.1])
    return apply_rigid_volume_transform(volume, rotation, shift, order=3), rotation, shift


@pytest.mark.parametrize("seed", range(4))
def test_a_fit_recovers_a_known_rotation_and_shift(seed):
    volume = _blob_volume(64, seed)
    moved, rotation, shift = _offset_map(volume, seed)
    fit = al.fit_rigid_to_reference(moved, volume, shell=16)
    # over these four maps the rotation matrix is off by at most 1.6e-5 (0.001 degrees), the shift by at most 5.2e-4
    # of its 0.3 voxel
    assert_matches(fit.rotation @ rotation, np.eye(3), rtol=1e-4)
    assert_matches(-fit.translation_voxels, fit.rotation @ shift, rtol=3e-3)
    assert fit.iterations <= 4


@pytest.mark.parametrize("seed", range(4))
def test_a_fit_is_undone_by_its_inverse(seed):
    volume = _blob_volume(64, seed)
    fit = al.fit_rigid_to_reference(_offset_map(volume, seed)[0], volume, shell=16)
    # the two cubic-spline interpolations leave at most 5.1e-4 of the map's maximum
    assert_matches(al.apply_fit(al.apply_fit(volume, fit), al.inverse_fit(fit)), volume, rtol=2e-3)


def test_a_fit_refuses_a_map_that_correlates_negatively_with_the_reference():
    volume = _blob_volume(32, 0)
    with pytest.raises(ValueError, match="correlates with the reference at scale"):
        al.fit_rigid_to_reference(-volume, volume, shell=8)


def test_a_fit_that_has_not_converged_is_refused(monkeypatch):
    volume = _blob_volume(32, 0)
    monkeypatch.setattr(al, "MAX_ITERATIONS", 1)
    with pytest.raises(ValueError, match="did not converge in 1 iterations"):
        al.fit_rigid_to_reference(_offset_map(volume, 0)[0], volume, shell=8)


def test_the_run_maps_are_aligned_to_the_truth_by_the_merged_maps_transform():
    truth = _blob_volume(64, 0)
    moved, _, _ = _offset_map(truth, 0)
    maps = {"gt": truth, "merged": 0.8 * moved, "half1": 0.8 * moved, "half2": 0.8 * moved}
    aligned, record, round_trip = ev.align_run_maps(maps, voxel=2.0, box=64)
    assert record["band_shell"] == 13  # 10 A at a 128 A box
    assert_matches(record["angle_deg"], 1.5, rtol=3e-3)  # fitted to 1.5005 degrees on this map
    assert_matches(record["scale"], 1.25, rtol=1e-3)  # the maps are 0.8 of the truth
    report, curves = ev.resolution_metrics(maps, aligned, 2.0, 64)
    inside_band = slice(1, record["band_shell"] + 1)
    assert round_trip.shape == curves["gt_merged_aligned"].shape
    assert round_trip[inside_band].min() > 0.9999
    assert_matches(record["interpolation_round_trip_fsc_auc"], ev.normalized_fsc_auc(round_trip))
    assert curves["gt_merged_aligned"][inside_band].min() > 0.9999
    assert curves["gt_merged"][inside_band].min() < 0.999
    for name in ev.RUN_MAP_NAMES:
        assert f"gt_{name}_aligned" in report


def _maps_missing_a_cone_about_z(n: int, seed: int) -> dict[str, np.ndarray]:
    """Run maps that share their signal with the truth everywhere but in the double cone of half-angle 30 degrees
    about z (a polar cap of views), with independent noise per half and a little in the truth."""
    rng = np.random.default_rng(seed)
    freqs = np.fft.fftfreq(n) * n
    kx, ky, kz = np.meshgrid(freqs, freqs, freqs, indexing="ij")
    keep = np.abs(kz) < np.sqrt(kx**2 + ky**2 + kz**2) * math.cos(math.radians(30.0))
    signal = np.fft.ifftn(np.fft.fftn(rng.standard_normal((n, n, n))) * keep).real
    half1 = signal + 0.5 * rng.standard_normal((n, n, n))
    half2 = signal + 0.5 * rng.standard_normal((n, n, n))
    return {
        "gt": signal + 0.2 * rng.standard_normal((n, n, n)),
        "half1": half1,
        "half2": half2,
        "merged": (half1 + half2) / 2,
    }


def test_conical_fsc_metrics_find_the_missing_cone_of_both_pairs():
    maps = _maps_missing_a_cone_about_z(48, seed=1)
    aligned = {name: maps[name] for name in ev.RUN_MAP_NAMES}
    metrics, cfscs = ev.conical_fsc_metrics(maps, aligned, voxel=2.0, box=48, half_angle_deg=20.0, n_axes=60)
    assert set(metrics) == set(cfscs) == {"half_maps", "gt_merged_aligned"}
    for name, summary in metrics.items():
        assert summary["n_axes"] == 60 and summary["half_angle_deg"] == 20.0
        assert summary["cfar"] < 0.1, name  # measured: 0.008 (half maps) and 0 (truth pair)
        assert summary["min_axis"]["tilt_deg"] < 30.0, name
        assert summary["max_axis"]["tilt_deg"] > 50.0, name
        assert cfscs[name].curves.shape == (60, 48 // 2 - 1)
    axes = cone_axes(60)
    for name, (lhs, rhs) in {
        "half_maps": (maps["half1"], maps["half2"]),
        "gt_merged_aligned": (aligned["merged"], maps["gt"]),
    }.items():
        assert_matches(cfscs[name].curves, conical_fsc(lhs, rhs, axes, 20.0).curves)


def test_conical_fsc_metrics_score_the_aligned_merged_map_not_the_refined_one():
    maps = _maps_missing_a_cone_about_z(48, seed=2)
    aligned = {name: maps[name] for name in ev.RUN_MAP_NAMES}
    aligned["merged"] = np.random.default_rng(9).standard_normal(maps["gt"].shape)  # unrelated to the truth
    metrics, cfscs = ev.conical_fsc_metrics(maps, aligned, voxel=2.0, box=48, half_angle_deg=20.0, n_axes=20)
    refined = ev.conical_fsc_metrics(maps, {name: maps[name] for name in ev.RUN_MAP_NAMES}, 2.0, 48, 20.0, 20)[1]
    # the half-map pair does not move with the alignment; the truth pair follows the aligned map
    assert_matches(cfscs["half_maps"].curves, refined["half_maps"].curves)
    assert np.nanmax(np.abs(cfscs["gt_merged_aligned"].curves - refined["gt_merged_aligned"].curves)) > 0.3


def test_the_heatmap_cell_takes_its_nearest_axis_up_to_the_opposite_direction():
    cfsc = ConicalFsc(
        axes=np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
        half_angle_deg=20.0,
        curves=np.zeros((2, 4)),
        voxels=np.zeros((2, 4), dtype=int),
    )
    rot_edges, tilt_edges, values = cfp.relative_wauc_map(cfsc, np.array([0.5, 1.0]), step_deg=1.0)
    assert values.shape == (90, 360)

    def rot_cell(rot: float) -> int:
        """The column of the cell holding ``rot``."""
        return int(np.floor(rot - rot_edges[0]))

    assert_matches(values[0, rot_cell(30.0)], 0.5)  # tilt 0.5 deg: the z axis
    assert_matches(values[89, rot_cell(0.5)], 1.0)  # tilt 89.5 deg, rot 0.5: the x axis
    assert_matches(values[89, rot_cell(-179.5)], 1.0)  # rot -179.5: the opposite of x, the same cone


def test_curve_statistics_average_only_shells_every_cone_measures():
    cfsc = ConicalFsc(
        axes=np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
        half_angle_deg=20.0,
        curves=np.array([[np.nan, 0.9, 0.8, 0.7], [np.nan, 0.1, 0.4, 0.5]]),
        voxels=np.array([[0, 300, 300, 300], [0, 50, 300, 300]]),
    )
    stats = cfp.curve_statistics(cfsc, min_voxels=100, first_shell=2)
    assert np.all(np.isnan([stats[key][0:2] for key in stats]))  # shell 1 is below first_shell
    assert_matches(stats["mean"][2:], np.array([0.6, 0.6]))
    assert_matches(stats["std"][2:], np.array([0.2, 0.1]))
    assert_matches(stats["min"][2:], np.array([0.4, 0.5]))
    assert_matches(stats["max"][2:], np.array([0.8, 0.7]))


def test_the_conical_figure_is_written_for_both_pairs(tmp_path):
    maps = _maps_missing_a_cone_about_z(48, seed=3)
    aligned = {name: maps[name] for name in ev.RUN_MAP_NAMES}
    metrics, cfscs = ev.conical_fsc_metrics(maps, aligned, voxel=2.0, box=48, half_angle_deg=20.0, n_axes=30)
    curves = {
        "half_maps": ev.shell_fsc(maps["half1"], maps["half2"]),
        "gt_merged_aligned": ev.shell_fsc(maps["merged"], maps["gt"]),
    }
    result = {"conical_fsc": cfscs, "curves": curves, "metrics": {"conical_fsc": metrics}}
    pairs = ev.conical_figure_pairs(result)
    assert list(pairs) == ["half maps", "final map vs ground truth, aligned"]
    path = tmp_path / "conical_fsc.png"
    cfp.plot_conical_fsc(pairs, 2.0, 48, "demo", ev.PALETTE, path)
    assert path.stat().st_size > 10_000


def test_crop_images_keeps_pixel_values_and_scales_white_noise_variance_by_the_retained_energy():
    assert_matches(im.crop_images(np.full((2, 32, 32), 3.0), 16), np.full((2, 16, 16), 3.0))
    x = np.arange(32)
    wave = np.tile(np.cos(2 * np.pi * 2 * x / 32), (32, 1))[None]  # two periods across the image
    assert_matches(im.crop_images(wave, 16)[0], np.tile(np.cos(2 * np.pi * 2 * np.arange(16) / 16), (16, 1)))
    noise = np.random.default_rng(0).standard_normal((64, 32, 32))
    # (box - 1)^2 modes keep all their energy, the 2 box - 1 Nyquist-row and -column modes half of it;
    # the retained share varies by about 1% between draws of this size
    retained = ((16 - 1) ** 2 + (2 * 16 - 1) / 2) / 32**2
    assert im.crop_images(noise, 16).var() == pytest.approx(noise.var() * retained, rel=0.05)


def _write_particles(directory, images, names):
    """A particle STAR with pose columns and the stack ``a.mrcs`` of ``images`` in ``directory``."""
    with mrcfile.new(directory / "a.mrcs", overwrite=True) as handle:
        handle.set_data(images.astype(np.float32))
    optics = {"rlnOpticsGroup": [1], "rlnImagePixelSize": [2.0], "rlnImageSize": [images.shape[-1]]}
    particles = {
        "rlnImageName": names,
        "rlnDefocusU": [10000.0 + 1000 * i for i in range(len(names))],
        **{c: [float(i) for i in range(len(names))] for c in im.POSE_COLUMNS},
    }
    starfile.write({"optics": pd.DataFrame(optics), "particles": pd.DataFrame(particles)}, directory / "particles.star")
    return directory / "particles.star"


def test_prepare_particles_crops_the_listed_slices_drops_the_poses_and_rescales_the_pixel(tmp_path):
    images = np.random.default_rng(0).standard_normal((3, 16, 16)).astype(np.float32)
    source = _write_particles(tmp_path, images, ["3@a.mrcs", "1@a.mrcs", "2@a.mrcs"])  # out of stack order
    assert im.prepare_particles(source, tmp_path / "out", box=8) == 3
    blocks = starfile.read(tmp_path / "out" / "particles.star", always_dict=True)
    assert list(blocks["particles"].columns) == ["rlnImageName", "rlnDefocusU"]
    assert blocks["particles"]["rlnImageName"].tolist() == [f"{i}@particles.8.mrcs" for i in (1, 2, 3)]
    assert blocks["particles"]["rlnDefocusU"].tolist() == [10000.0, 11000.0, 12000.0]
    assert blocks["optics"]["rlnImagePixelSize"].tolist() == [4.0]
    assert blocks["optics"]["rlnImageSize"].tolist() == [8]
    with mrcfile.open(tmp_path / "out" / "particles.8.mrcs") as handle:
        assert handle.voxel_size.x == 4.0
        assert_matches(handle.data, im.crop_images(images[[2, 0, 1]], 8))


def test_prepare_particles_at_the_stacks_own_box_only_drops_the_poses(tmp_path):
    images = np.random.default_rng(1).standard_normal((2, 8, 8)).astype(np.float32)
    source = _write_particles(tmp_path, images, ["1@a.mrcs", "2@a.mrcs"])
    im.prepare_particles(source, tmp_path / "out", box=8)
    with mrcfile.open(tmp_path / "out" / "particles.8.mrcs") as handle:
        assert_matches(handle.data, images)
    optics = starfile.read(tmp_path / "out" / "particles.star", always_dict=True)["optics"]
    assert optics["rlnImagePixelSize"].tolist() == [2.0]


@pytest.mark.parametrize("box", [7, 12])
def test_prepare_particles_refuses_an_odd_or_larger_box(tmp_path, box):
    source = _write_particles(tmp_path, np.zeros((1, 8, 8)), ["1@a.mrcs"])
    with pytest.raises(ValueError, match="even"):
        im.prepare_particles(source, tmp_path / "out", box=box)


def test_prepare_particles_refuses_particles_in_several_stacks(tmp_path):
    source = _write_particles(tmp_path, np.zeros((2, 8, 8)), ["1@a.mrcs", "1@b.mrcs"])
    with pytest.raises(ValueError, match="2 stacks"):
        im.prepare_particles(source, tmp_path / "out", box=4)


def test_resample_volume_pads_in_fourier_space_and_keeps_the_density_values():
    volume = ndimage.gaussian_filter(np.random.default_rng(0).standard_normal((16, 16, 16)), 2.0)
    padded = im.resample_volume(volume, 32)
    assert padded.shape == (32, 32, 32)
    assert_matches(padded[::2, ::2, ::2], volume)  # the original voxels are every second one


def test_upsample_writes_the_map_on_the_reference_grid_and_refuses_a_different_physical_box(tmp_path):
    volume = ndimage.gaussian_filter(np.random.default_rng(2).standard_normal((8, 8, 8)), 1.0)
    like = np.zeros((16, 16, 16))
    im.write_map(tmp_path / "small.mrc", volume, voxel=4.0)
    im.write_map(tmp_path / "like.mrc", like, voxel=2.0)
    im.main(
        [
            "upsample",
            "--map",
            str(tmp_path / "small.mrc"),
            "--like",
            str(tmp_path / "like.mrc"),
            "--out",
            str(tmp_path / "up.mrc"),
        ]
    )
    upsampled, voxel = im.read_map(tmp_path / "up.mrc")
    assert upsampled.shape == (16, 16, 16) and voxel == 2.0
    assert_matches(upsampled[::2, ::2, ::2], volume, rtol=1e-6)  # both maps went through float32 files (5e-8 measured)
    with pytest.raises(ValueError, match="rescale"):
        im.upsample(volume, 3.0, like, 2.0)


def test_a_written_map_carries_the_label_relax_refine_requires_of_a_start_map(tmp_path):
    im.write_map(tmp_path / "start.mrc", np.zeros((4, 4, 4)), voxel=2.0)
    # the name has no `_relion` marker, so only the label lets it pass
    require_relion_convention_reference(tmp_path / "start.mrc")


def _hand_fit(mirror_x):
    """A registration fit that keeps the map in place (identity rotation), mirrored or not."""
    from relax.diagnostics.gt_registration import RigidHandFit

    return RigidHandFit(
        rotation_matrix=np.eye(3),
        translation_voxels=np.zeros(3),
        mirror_x=mirror_x,
        lowpass_score=0.9,
        start_lowpass_scores=(0.9,),
        failed_starts=(),
        fine_start_score=0.9,
        fine_score=0.9,
        fine_evaluations=1,
        lowpass_optimizer_success=True,
        lowpass_evaluations=1,
        fine_optimizer_success=True,
        fine_optimizer_message="",
    )


def _registration_with_fits(monkeypatch, fits):
    """Replace the rigid fit by ``fits`` (the fit itself is tested with ``relax.diagnostics.gt_registration``)."""
    monkeypatch.setattr("relax.diagnostics.gt_metrics.relion_alignment_rotations", lambda order: np.eye(3)[None])
    monkeypatch.setattr("relax.diagnostics.gt_registration.fit_rigid_both_hands", lambda *args, **kwargs: fits)


def test_registration_keeps_the_hand_with_the_larger_fsc_area_and_applies_its_transform(monkeypatch):
    reference = ndimage.gaussian_filter(np.random.default_rng(0).standard_normal((16, 16, 16)), 1.5)
    moving = reference[::-1].copy()  # the mirror image
    _registration_with_fits(monkeypatch, {"proper": _hand_fit(False), "mirror": _hand_fit(True)})
    registered, summary = im.register(moving, reference, voxel=4.0)
    assert summary["hand"] == "mirror"
    assert summary["fsc_auc_margin_over_other_hand"] > 0.5
    assert summary["hands"]["proper"]["gt_fsc"]["auc"] < summary["hands"]["mirror"]["gt_fsc"]["auc"]
    assert_matches(registered, reference)


def test_registration_refuses_a_map_whose_hand_failed_to_fit(monkeypatch):
    reference = np.random.default_rng(0).standard_normal((8, 8, 8))
    _registration_with_fits(monkeypatch, {"proper": _hand_fit(False)})
    with pytest.raises(ValueError, match="mirror"):
        im.register(reference, reference, voxel=4.0)


def test_registered_map_and_its_record_are_written_beside_each_other(tmp_path, monkeypatch):
    reference = ndimage.gaussian_filter(np.random.default_rng(1).standard_normal((16, 16, 16)), 1.5)
    im.write_map(tmp_path / "gt.mrc", reference, voxel=4.0)
    im.write_map(tmp_path / "ab_initio.mrc", reference[::-1], voxel=4.0)
    _registration_with_fits(monkeypatch, {"proper": _hand_fit(False), "mirror": _hand_fit(True)})
    im.main(
        [
            "register",
            "--map",
            str(tmp_path / "ab_initio.mrc"),
            "--reference",
            str(tmp_path / "gt.mrc"),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )
    registered, voxel = im.read_map(tmp_path / "out" / im.REGISTERED_NAME)
    assert voxel == 4.0
    assert_matches(registered, reference, rtol=1e-6)
    assert json.loads((tmp_path / "out" / "registration.json").read_text())["hand"] == "mirror"


def _azimuth_and_tilt_deg(vectors):
    """Azimuth in [0, 360) and tilt of unit vectors, degrees."""
    return np.degrees(np.arctan2(vectors[..., 1], vectors[..., 0])) % 360.0, np.degrees(np.arccos(vectors[..., 2]))


def test_expand_directions_gives_a_c3_particle_the_rotations_about_z_by_120_degrees():
    rot, tilt = np.array([10.0, -50.0]), np.array([40.0, 120.0])
    orbits = snu.expand_directions(ev.viewing_directions(rot, tilt), "C3").reshape(2, 3, 3)
    azimuth, orbit_tilt = _azimuth_and_tilt_deg(orbits)
    assert_matches(np.sort(azimuth, axis=1), np.sort((rot[:, None] + np.array([0.0, 120.0, 240.0])) % 360.0, axis=1))
    assert_matches(orbit_tilt, np.repeat(tilt[:, None], 3, axis=1))


def test_expand_directions_of_c1_is_the_input_and_of_d3_gives_six_distinct_directions():
    vectors = ev.viewing_directions(np.array([10.0, -50.0]), np.array([40.0, 120.0]))
    assert_matches(snu.expand_directions(vectors, "C1"), vectors)
    orbits = snu.expand_directions(vectors, "D3").reshape(2, 6, 3)
    assert_matches(np.linalg.norm(orbits, axis=-1), np.ones((2, 6)))
    assert [len({tuple(np.round(v, 9)) for v in orbit}) for orbit in orbits] == [6, 6]


def test_direction_counts_agree_with_the_evaluators_histogram_of_the_same_angles():
    rng = np.random.default_rng(0)
    rot, tilt = rng.uniform(-180.0, 180.0, 5000), np.degrees(np.arccos(rng.uniform(-1.0, 1.0, 5000)))
    for order in (1, 3, 4):
        assert_matches(
            snu.direction_counts(ev.viewing_directions(rot, tilt), order), ev.direction_histogram(rot, tilt, order)
        )


def test_concentration_of_all_particles_in_one_direction_is_log2_of_the_direction_count():
    n = n_pixels(1)
    counts = np.zeros(n)
    counts[7] = 100.0
    measures = snu.concentration(counts)
    assert_matches(measures["kl_from_uniform_bits"], math.log2(n))
    assert_matches(measures["sphere_fraction_with_half_of_mass"], 1.0 / n)
    assert_matches(measures["directions_without_a_particle"], (n - 1.0) / n)


def test_poses_uniform_in_a_c3_wedge_are_uniform_after_expansion_and_far_from_it_before():
    # Measured on seeds 0-2 (30000 particles, order 2): expanded KL 0.0010-0.0016 bits (the floor is 0.0015), raw 1.46.
    rng = np.random.default_rng(0)
    rot, tilt = rng.uniform(-60.0, 60.0, 30000), np.degrees(np.arccos(rng.uniform(-1.0, 1.0, 30000)))
    row = snu.nonuniformity_by_order(ev.viewing_directions(rot, tilt), "C3", (2,), 4, 0)[0]
    assert row["expanded_kl_from_uniform_bits"] < 0.01
    assert abs(row["expanded_kl_above_floor_bits"]) < 0.01
    assert row["unexpanded_kl_from_uniform_bits"] > 1.4


def test_nonuniformity_without_symmetry_scores_the_raw_directions_and_the_floor_is_that_of_uniform_particles():
    rng = np.random.default_rng(1)
    vectors = ev.viewing_directions(rng.uniform(-180.0, 180.0, 4000), 30.0 * rng.uniform(0.0, 1.0, 4000))
    row = snu.nonuniformity_by_order(vectors, "C1", (2,), 3, 0)[0]
    assert_matches(row["expanded_kl_from_uniform_bits"], row["unexpanded_kl_from_uniform_bits"])
    assert row["expanded_kl_from_uniform_bits"] > 2.0  # a 30 degree cap holds the whole sample
    assert 0.0 <= row["uniform_mean_kl_from_uniform_bits"] < 0.2


def test_kent_scenario_kl_lists_the_kent_scenarios_of_a_config_with_their_population_kl(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "scenarios:\n"
        "  cap:\n"
        "    poses: {kind: kent, kappa: 8.0, beta: 2.0, mu: [1.0, 1.0, 0.0], mu0: [0.0, 0.0, 1.0]}\n"
        "  flat:\n"
        "    poses: {kind: uniform}\n"
        "  cap_again:\n"
        "    dataset_from: cap\n"
    )
    result = snu.kent_scenario_kl(config, (2, 3))
    assert list(result) == ["cap"]
    for order in (2, 3):
        p = true_prior.kent_population_prior(8.0, 2.0, [1.0, 1.0, 0.0], [0.0, 0.0, 1.0], order)
        assert_matches(result["cap"][order], ev.kl_bits(p, np.full(p.size, 1.0 / p.size)))


def test_expanded_star_reads_back_as_the_expanded_directions(tmp_path):
    vectors = snu.expand_directions(ev.viewing_directions(np.array([10.0, -50.0]), np.array([40.0, 120.0])), "C3")
    path = tmp_path / "expanded.star"
    snu.write_expanded_star(path, vectors)
    assert_matches(snu.read_viewing_directions(path), vectors, rtol=1e-6)
    assert set(starfile.read(path)) == {"optics", "particles"}  # the blocks angdist looks up by name
