"""The pose-prior benchmark: evaluator metrics, the simulator's Kent guard, fixed priors, the variant comparison, the
ab-initio start map and the particles' translational offsets."""

import csv
import json
import logging
import math
import sys
from pathlib import Path

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
from relax.sampling.healpix import euler_angles_to_matrix, euler_matrix_to_angles, n_pixels, pixel_rot_tilt
from scripts.conical_fsc import ConicalFsc, cone_axes, conical_fsc, conical_fsc_summary
from scripts.pose_prior_benchmark import add_offsets as ao
from scripts.pose_prior_benchmark import align_to_truth as al
from scripts.pose_prior_benchmark import compare_variants as cmp
from scripts.pose_prior_benchmark import conical_fsc_plot as cfp
from scripts.pose_prior_benchmark import evaluate_run as ev
from scripts.pose_prior_benchmark import initial_model as im
from scripts.pose_prior_benchmark import mix_pools as mp
from scripts.pose_prior_benchmark import reconstruct_with_poses as rwp
from scripts.pose_prior_benchmark import render_volume as rv
from scripts.pose_prior_benchmark import simulate_dataset as sim
from scripts.pose_prior_benchmark import star_nonuniformity as snu
from scripts.pose_prior_benchmark import summarize_sweep as ss
from scripts.pose_prior_benchmark import sweep
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


def test_the_viewing_direction_is_the_third_row_of_the_pose_matrix():
    rng = np.random.default_rng(0)
    rot, tilt, psi = rng.uniform(-180, 180, 50), rng.uniform(0, 180, 50), rng.uniform(-180, 180, 50)
    matrices = euler_angles_to_matrix(np.stack([rot, tilt, psi], axis=-1))
    assert_matches(matrices[:, 2, :], ev.viewing_directions(rot, tilt))


def _within(actual, desired, tolerance_deg):
    """The angles agree within ``tolerance_deg`` (arccos resolves a cosine within 1e-16 of 1 to about 1e-6 degrees)."""
    assert np.abs(np.asarray(actual, float) - np.asarray(desired, float)).max() < tolerance_deg


def _inplane(true, est, rotation=None):
    return ev.inplane_error_deg(np.asarray(true, float).reshape(-1, 3), np.asarray(est, float).reshape(-1, 3), rotation)


def test_inplane_error_of_poses_that_differ_only_in_psi_is_the_psi_difference_wrapped():
    true = [[20.0, 60.0, 10.0], [20.0, 60.0, 170.0], [-130.0, 100.0, 0.0]]
    est = [[20.0, 60.0, 40.0], [20.0, 60.0, -170.0], [-130.0, 100.0, 0.0]]
    _within(_inplane(true, est), [30.0, 20.0, 0.0], 1e-6)
    _within(ev.direction_error_deg(*np.asarray(true).T[:2], *np.asarray(est).T[:2]), [0.0, 0.0, 0.0], 1e-4)


def test_inplane_error_of_poses_that_differ_only_in_tilt_is_zero():
    # a tilt change at fixed psi is a rotation about an axis in the image plane: all of it is out of plane
    true = [[30.0, 50.0, 25.0], [-100.0, 120.0, -60.0]]
    est = [[30.0, 58.0, 25.0], [-100.0, 112.0, -60.0]]
    _within(_inplane(true, est), [0.0, 0.0], 1e-6)
    _within(ev.direction_error_deg(*np.asarray(true).T[:2], *np.asarray(est).T[:2]), [8.0, 8.0], 1e-4)


def test_a_pose_at_the_pole_that_differs_in_rot_differs_in_plane_only():
    # at tilt 0 the viewing direction does not depend on rot: rot and psi both turn the image
    _within(_inplane([0.0, 0.0, 0.0], [30.0, 0.0, 0.0]), [30.0], 1e-6)
    _within(ev.direction_error_deg(0.0, 0.0, 30.0, 0.0), 0.0, 1e-4)


def test_inplane_error_is_the_image_turn_whatever_the_out_of_plane_swing():
    rng = np.random.default_rng(1)
    n = 200
    true = rng.uniform([-180, 10, -180], [180, 170, 180], (n, 3))
    true_matrix = euler_angles_to_matrix(true)
    gamma = rng.uniform(-40, 40, n)
    image_turn = Rotation.from_euler("z", gamma, degrees=True).as_matrix()
    # a volume rotation about an axis perpendicular to the true viewing direction moves the direction and adds no twist
    axis = rng.normal(size=(n, 3))
    direction = true_matrix[:, 2, :]
    axis -= np.sum(axis * direction, axis=-1, keepdims=True) * direction
    axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
    swing_deg = rng.uniform(1.0, 60.0, n)
    swing = Rotation.from_rotvec(axis * np.radians(swing_deg)[:, None]).as_matrix()
    est = euler_matrix_to_angles(image_turn @ true_matrix @ swing)
    _within(_inplane(true, est), np.abs(gamma), 1e-6)
    _within(ev.direction_error_deg(true[:, 0], true[:, 1], est[:, 0], est[:, 1]), swing_deg, 1e-4)


def test_inplane_error_is_defined_for_opposite_directions():
    error = _inplane([[0.0, 90.0, 0.0], [10.0, 0.0, 5.0]], [[180.0, 90.0, 0.0], [10.0, 180.0, 5.0]])
    assert np.all(np.isfinite(error))


def test_turning_the_estimates_back_by_the_frame_offset_removes_its_inplane_part_too():
    rng = np.random.default_rng(2)
    true = rng.uniform([-180, 10, -180], [180, 170, 180], (300, 3))
    offset = Rotation.from_rotvec(np.radians([3.0, -2.0, 4.0])).as_matrix()
    gamma = 7.0
    image_turn = Rotation.from_euler("z", gamma, degrees=True).as_matrix()
    est = euler_matrix_to_angles(image_turn @ euler_angles_to_matrix(true) @ offset.T)  # est = offset true, then gamma
    _within(_inplane(true, est, offset), np.full(300, gamma), 1e-6)
    assert np.abs(_inplane(true, est) - gamma).max() > 0.5  # without the alignment the offset adds to the error
    _within(ev.aligned_direction_error_deg(*true.T[:2], *est.T[:2], offset), np.zeros(300), 1e-4)


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


def _evaluation(tmp_path, name, *, source, auc, median, js_skill, with_conical=True, origin_median=2.0):
    """A minimal evaluate_run output directory: the metrics, FSC curves, pose errors and cFSC the comparison reads.
    The aligned FSC area is ``auc`` and the aligned pose error ``median``; as refined they are 0.05 lower and 0.5
    higher. ``origin_median`` (A) is the origin error of the first half set, 1 A more in the second; the p90 is three
    times it and the share over a pixel 0.5 (first half) and 0.25 (second); the particles' origin errors are
    ``origin_median`` and ``origin_median + 1`` A."""
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
        "pose_inplane_error_aligned_deg": {
            "half1": {"median": 0.5 * median, **pose},
            "half2": {"median": 0.5 * median + 1.0, **pose},
        },
        "pose_inplane_error_deg": {
            "half1": {"median": 0.5 * median + 0.5, **refined_pose},
            "half2": {"median": 0.5 * median + 1.5, **refined_pose},
        },
        "pose_frame_offset": {"angle_deg": 0.45},
        "prior_final": {"half1": {"js_bits": 0.1, "js_skill": js_skill}, "half2": {"js_bits": 0.3, "js_skill": None}},
    }
    if with_conical:
        metrics["conical_fsc"] = _write_conical_fsc(directory)
    metrics["origin_error_angstrom"] = {
        "half1": {"median": origin_median, "p90": 3 * origin_median, "fraction_over_pixel": 0.5},
        "half2": {"median": origin_median + 1.0, "p90": 3 * origin_median + 1.0, "fraction_over_pixel": 0.25},
    }
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
        inplane_half1=(0.5 * errors[0] + 0.5).astype(np.float32),
        inplane_half2=(0.5 * errors[1] + 0.5).astype(np.float32),
        aligned_inplane_half1=(0.5 * errors[0]).astype(np.float32),
        aligned_inplane_half2=(0.5 * errors[1]).astype(np.float32),
        origin_half1=np.full(200, origin_median, np.float32),
        origin_half2=np.full(200, origin_median + 1.0, np.float32),
    )
    return directory


def test_variants_are_compared_in_a_table_and_a_figure(tmp_path):
    learned = _evaluation(tmp_path, "learned", source="learned", auc=0.8, median=2.0, js_skill=0.9)
    fixed = _evaluation(tmp_path, "fixed_true", source="fixed", auc=0.85, median=1.5, js_skill=0.9, origin_median=1.0)
    out = tmp_path / "out"
    cmp.main(["--runs", f"learned={learned}", f"fixed_true={fixed}", "--out-dir", str(out), "--label", "test"])
    with open(out / "variants.csv", newline="") as handle:
        rows = {row["variant"]: row for row in csv.DictReader(handle)}
    assert list(rows) == ["learned", "fixed_true"]
    assert [rows[v]["direction_prior"] for v in rows] == ["learned", "fixed"]
    assert_matches(float(rows["learned"]["pose_error_median_aligned_deg"]), 2.5)  # the mean of the two halves
    assert_matches(float(rows["learned"]["pose_error_median_as_refined_deg"]), 3.0)
    assert_matches(float(rows["learned"]["inplane_error_median_aligned_deg"]), 1.5)
    assert_matches(float(rows["fixed_true"]["gt_fsc_auc_aligned"]), 0.85)
    assert_matches(float(rows["fixed_true"]["gt_fsc_auc_as_refined"]), 0.80)
    assert_matches(
        [float(rows["learned"][k]) for k in ("pose_offset_deg", "map_offset_deg", "map_shift_A")], [0.45, 0.35, 0.1]
    )
    assert_matches(float(rows["learned"]["learned_prior_js_bits"]), 0.2)
    origin_keys = ("origin_error_median_A", "origin_error_p90_A", "origin_error_fraction_over_pixel")
    assert_matches([float(rows["learned"][k]) for k in origin_keys], [2.5, 6.5, 0.375])
    assert_matches([float(rows["fixed_true"][k]) for k in origin_keys], [1.5, 3.5, 0.375])
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
    # the best and the worst cone are those of the largest and the smallest wAuC: curves 0 and 2 of the fixture
    assert_matches(curve.best[1:], _GT_CONE_CURVES[0, 1:])
    assert_matches(curve.worst[1:], _GT_CONE_CURVES[2, 1:])


def test_the_best_and_worst_cone_are_refused_when_the_metrics_name_other_axes(tmp_path):
    directory = _evaluation(tmp_path, "learned", source="learned", auc=0.8, median=2.0, js_skill=0.9)
    metrics = json.loads((directory / "metrics.json").read_text())
    metrics["conical_fsc"]["gt_merged_aligned"]["max_axis"]["tilt_deg"] += 10.0
    with pytest.raises(ValueError, match="disagree on the cone of max_axis"):
        cmp.read_conical_curve(directory, metrics)


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


def test_aligned_inplane_errors_of_both_halves_are_read_in_order(tmp_path):
    np.savez(
        tmp_path / "pose_errors.npz",
        aligned_half1=np.array([9.0, 9.0], np.float32),
        aligned_half2=np.array([9.0], np.float32),
        inplane_half1=np.array([8.0, 8.0], np.float32),
        inplane_half2=np.array([8.0], np.float32),
        aligned_inplane_half1=np.array([1.5, 2.5], np.float32),
        aligned_inplane_half2=np.array([3.5], np.float32),
    )
    assert_matches(cmp.read_inplane_errors(tmp_path), np.array([1.5, 2.5, 3.5]))


def test_a_variant_evaluated_before_the_inplane_error_is_refused_not_compared_without_it(tmp_path):
    old = _evaluation(tmp_path, "old", source="learned", auc=0.5, median=2.0, js_skill=0.7)
    with np.load(old / "pose_errors.npz") as archive:
        kept = {k: archive[k] for k in archive.files if "inplane" not in k}
    np.savez(old / "pose_errors.npz", **kept)
    with pytest.raises(ValueError, match="pose_errors.npz has no in-plane errors; rerun evaluate_run.py"):
        cmp.read_inplane_errors(old)
    metrics = json.loads((old / "metrics.json").read_text())
    del metrics["pose_inplane_error_aligned_deg"]
    with pytest.raises(ValueError, match="old: metrics.json has no in-plane pose error; rerun evaluate_run.py"):
        cmp.summary_row("old", metrics)


def test_origin_errors_of_both_halves_are_read_in_order_in_pixels(tmp_path):
    np.savez(
        tmp_path / "pose_errors.npz",
        origin_half1=np.array([1.5, 0.25], np.float32),
        origin_half2=np.array([3.0], np.float32),
    )
    assert_matches(cmp.read_origin_errors(tmp_path, 0.5), np.array([3.0, 0.5, 6.0]))


def test_a_variant_evaluated_before_the_origin_error_is_refused_not_compared_without_it(tmp_path):
    old = _evaluation(tmp_path, "old", source="learned", auc=0.5, median=2.0, js_skill=0.7)
    with np.load(old / "pose_errors.npz") as archive:
        kept = {k: archive[k] for k in archive.files if "origin" not in k}
    np.savez(old / "pose_errors.npz", **kept)
    with pytest.raises(ValueError, match="pose_errors.npz has no origin errors; rerun evaluate_run.py"):
        cmp.read_origin_errors(old, 2.0)
    metrics = json.loads((old / "metrics.json").read_text())
    del metrics["origin_error_angstrom"]
    with pytest.raises(ValueError, match="old: metrics.json has no origin error; rerun evaluate_run.py"):
        cmp.summary_row("old", metrics)


def test_runs_of_different_pixel_sizes_are_not_compared(tmp_path):
    first = _evaluation(tmp_path, "first", source="learned", auc=0.5, median=2.0, js_skill=0.7)
    second = _evaluation(tmp_path, "second", source="learned", auc=0.5, median=2.0, js_skill=0.7)
    metrics = json.loads((second / "metrics.json").read_text())
    metrics["resolution"]["voxel_size_A"] = 3.0
    (second / "metrics.json").write_text(json.dumps(metrics))
    with pytest.raises(SystemExit, match="different pixel sizes"):
        cmp.main(["--runs", f"first={first}", f"second={second}", "--out-dir", str(tmp_path / "out")])


def test_error_density_follows_the_given_range():
    grid, density = cmp.error_density(np.array([1.0, 1.0, 1.0]), cmp.ORIGIN_ERROR_RANGE_PX)
    low, high = cmp.ORIGIN_ERROR_RANGE_PX
    assert low < grid[0] < 1.1 * low and high / 1.1 < grid[-1] < high  # the grid spans the range, not the degrees one
    assert abs(np.log10(grid[np.argmax(density)])) < 0.05


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
        "inplane_errors": {1: np.array([0.6, 1.5]), 2: np.array([20.0])},
        "aligned_inplane_errors": {1: np.array([0.5, 1.0]), 2: np.array([19.0])},
        "origin_errors": {1: np.array([1.5, 0.25]), 2: np.array([2.0])},
        "box": 4,
        "voxel": 2.0,
    }
    ev.write_tables(result, tmp_path)
    with np.load(tmp_path / "pose_errors.npz") as archive:
        assert_matches(archive["half1"], np.array([0.5, 3.0]))
        assert_matches(archive["half2"], np.array([170.0]))
        assert_matches(archive["aligned_half1"], np.array([0.4, 2.5]))
        assert_matches(archive["aligned_half2"], np.array([169.0]))
        assert_matches(archive["inplane_half1"], np.array([0.6, 1.5]))
        assert_matches(archive["inplane_half2"], np.array([20.0]))
        assert_matches(archive["aligned_inplane_half1"], np.array([0.5, 1.0]))
        assert_matches(archive["aligned_inplane_half2"], np.array([19.0]))
        assert_matches(archive["origin_half1"], np.array([1.5, 0.25]))
        assert_matches(archive["origin_half2"], np.array([2.0]))
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


def _true_pose_reconstruction(tmp_path, auc):
    """A minimal reconstruct_with_poses output directory of the true poses: the FSC summary the comparison reads."""
    directory = tmp_path / "true_poses"
    directory.mkdir()
    summary = {"auc": auc, "at_0.5": {"first_A": 12.0}, "at_0.143": {"first_A": None}}
    metrics = {"resolution": {"box": 8, "voxel_size_A": 2.0, "gt_reconstruction": summary}}
    (directory / "metrics.json").write_text(json.dumps(metrics))
    return directory


def test_the_true_pose_ceiling_is_a_column_of_the_table_and_a_line_in_the_figure(tmp_path):
    learned = _evaluation(tmp_path, "learned", source="learned", auc=0.8, median=2.0, js_skill=0.9)
    fixed = _evaluation(tmp_path, "fixed_true", source="fixed", auc=0.85, median=1.5, js_skill=0.9)
    true = _true_pose_reconstruction(tmp_path, auc=0.9)
    out = tmp_path / "out"
    cmp.main(
        ["--runs", f"learned={learned}", f"fixed_true={fixed}", "--true-reconstruction", str(true)]
        + ["--out-dir", str(out), "--label", "test"]
    )
    with open(out / "variants.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert_matches([float(r["gt_fsc_auc_true_poses"]) for r in rows], [0.9, 0.9])
    assert (out / "variants.png").stat().st_size > 0
    assert not (out / "reconstructions.csv").exists()
    assert cmp.read_true_pose_auc(true) == 0.9


def test_without_a_true_pose_reconstruction_the_ceiling_column_is_empty(tmp_path):
    learned = _evaluation(tmp_path, "learned", source="learned", auc=0.8, median=2.0, js_skill=0.9)
    out = tmp_path / "out"
    cmp.main(["--runs", f"learned={learned}", "--out-dir", str(out), "--label", "test"])
    with open(out / "variants.csv", newline="") as handle:
        assert [r["gt_fsc_auc_true_poses"] for r in csv.DictReader(handle)] == [""]


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


def _band_limited_images(n, box, seed):
    """Random images (n, box, box) without Nyquist-row or -column content, which a fractional shift cannot keep."""
    spectrum = np.fft.fft2(np.random.default_rng(seed).standard_normal((n, box, box)))
    spectrum[:, box // 2, :] = 0.0
    spectrum[:, :, box // 2] = 0.0
    return np.fft.ifft2(spectrum).real.astype(np.float32)


def test_shift_images_moves_whole_pixels_like_np_roll_with_x_along_the_columns():
    images = np.random.default_rng(0).standard_normal((3, 16, 16)).astype(np.float32)
    shifts = np.array([[2.0, 0.0], [0.0, -3.0], [5.0, 4.0]])
    expected = np.stack(
        [np.roll(image, (int(sy), int(sx)), axis=(0, 1)) for image, (sx, sy) in zip(images, shifts)]
    )
    assert_matches(ao.shift_images(images, shifts), expected)


def test_a_fractional_shift_is_undone_by_the_opposite_shift():
    images = _band_limited_images(4, 32, seed=1)
    shifts = np.array([[1.37, -2.61], [0.5, 0.5], [-7.25, 3.9], [0.0, 0.01]])
    assert_matches(ao.shift_images(ao.shift_images(images, shifts), -shifts), images)


def test_a_fractional_shift_keeps_white_noise_variance_up_to_the_nyquist_modes():
    noise = np.random.default_rng(2).standard_normal((64, 64, 64)).astype(np.float32)
    shifted = ao.shift_images(noise, np.random.default_rng(3).uniform(-5, 5, size=(64, 2)))
    # the 2 box - 1 Nyquist-row and -column modes keep at most half their energy: at most 1.6% at box 64
    assert 0.97 < shifted.var() / noise.var() <= 1.0


def test_offsets_are_gaussian_with_the_requested_sigma_and_repeat_for_a_seed():
    offsets = ao.draw_offsets(200000, 7.0, seed=3)
    assert offsets.shape == (200000, 2)
    assert offsets.std(axis=0) == pytest.approx([7.0, 7.0], rel=0.01)
    assert np.abs(offsets.mean(axis=0)).max() < 0.05
    assert_matches(ao.draw_offsets(200000, 7.0, seed=3), offsets)
    assert not np.allclose(ao.draw_offsets(200000, 7.0, seed=4), offsets)


def _centred_delta_dataset(directory, n, box=16, pixel=2.0):
    """A dataset of ``n`` images that are a delta at the box centre, with zero origins and a pose-distribution record."""
    images = np.zeros((n, box, box), dtype=np.float32)
    images[:, box // 2, box // 2] = 1.0
    with mrcfile.new(directory / "particles.16.mrcs", overwrite=True) as handle:
        handle.set_data(images)
        handle.voxel_size = pixel
    optics = {"rlnOpticsGroup": [1], "rlnImagePixelSize": [pixel], "rlnImageSize": [box]}
    particles = {
        "rlnImageName": [f"{i + 1}@particles.16.mrcs" for i in range(n)],
        "rlnDefocusU": [10000.0 + 1000 * i for i in range(n)],
        "rlnAngleRot": [float(i) for i in range(n)],
        "rlnOriginXAngst": [0.0] * n,
        "rlnOriginYAngst": [0.0] * n,
    }
    starfile.write({"optics": pd.DataFrame(optics), "particles": pd.DataFrame(particles)}, directory / "particles.star")
    for name in ao.COPIED_FILES:
        (directory / name).write_bytes(name.encode())
    (directory / "pose_distribution.json").write_text(json.dumps({"poses": "uniform", "n_images": n}))
    return directory


def test_add_offsets_displaces_each_particle_by_minus_its_origin_and_records_the_offsets(tmp_path, monkeypatch):
    source = _centred_delta_dataset(tmp_path, n=4)
    whole_pixels = np.array([[2.0, 0.0], [0.0, -3.0], [-1.0, 4.0], [0.0, 0.0]])
    monkeypatch.setattr(ao, "draw_offsets", lambda n, sigma, seed: whole_pixels * 2.0)  # the pixel is 2 A
    out = tmp_path / "shifted"
    record = ao.add_offsets(source, out, sigma_angstrom=7.0, seed=5)
    blocks = starfile.read(out / "particles.star", always_dict=True)
    particles = blocks["particles"]
    assert_matches(particles[list(ao.ORIGIN_COLUMNS)].to_numpy(), whole_pixels * 2.0)
    original = starfile.read(source / "particles.star", always_dict=True)["particles"]
    assert particles.drop(columns=list(ao.ORIGIN_COLUMNS)).equals(original.drop(columns=list(ao.ORIGIN_COLUMNS)))
    with mrcfile.open(out / "particles.16.mrcs") as handle:
        stack = np.asarray(handle.data)
        assert handle.voxel_size.x == 2.0
    for image, (ox, oy) in zip(stack, whole_pixels):
        row, column = np.unravel_index(np.argmax(image), image.shape)
        assert (column, row) == (8 - ox, 8 - oy)  # the delta sits at the centre minus the origin, in pixels
        assert image.max() == pytest.approx(1.0, abs=1e-5)
    # RELION moves each particle back by its origin: the centred images return
    with mrcfile.open(source / "particles.16.mrcs") as handle:
        assert_matches(ao.shift_images(stack, whole_pixels), np.asarray(handle.data))
    assert [(out / name).read_bytes() for name in ao.COPIED_FILES] == [name.encode() for name in ao.COPIED_FILES]
    stored = json.loads((out / "pose_distribution.json").read_text())
    assert stored == {"poses": "uniform", "n_images": 4, "offsets": record}
    assert record["sigma_angstrom"] == 7.0 and record["seed"] == 5
    assert record["max_radius_angstrom"] == pytest.approx(np.hypot(*(whole_pixels * 2.0).T).max())


def test_add_offsets_refuses_a_source_with_origins_a_bad_sigma_and_particles_out_of_stack_order(tmp_path):
    source = _centred_delta_dataset(tmp_path, n=3)
    with pytest.raises(ValueError, match="positive"):
        ao.add_offsets(source, tmp_path / "a", sigma_angstrom=0.0, seed=1)
    blocks = starfile.read(source / "particles.star", always_dict=True)
    blocks["particles"]["rlnImageName"] = ["2@particles.16.mrcs", "1@particles.16.mrcs", "3@particles.16.mrcs"]
    starfile.write(blocks, source / "particles.star", overwrite=True)
    with pytest.raises(ValueError, match="row order"):
        ao.add_offsets(source, tmp_path / "b", sigma_angstrom=7.0, seed=1)
    blocks["particles"]["rlnImageName"] = [f"{i + 1}@particles.16.mrcs" for i in range(3)]
    blocks["particles"]["rlnOriginXAngst"] = [0.0, 1.5, 0.0]
    starfile.write(blocks, source / "particles.star", overwrite=True)
    with pytest.raises(ValueError, match="non-zero origins"):
        ao.add_offsets(source, tmp_path / "c", sigma_angstrom=7.0, seed=1)


def test_origin_error_is_the_distance_between_true_and_estimated_origins():
    true = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 1.0], [-6.0, 8.0]])
    estimated = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [-6.0, 0.0]])
    errors = ev.origin_error_angstrom(true, estimated)
    assert_matches(errors, np.array([0.0, 5.0, 0.0, 8.0]))
    summary = ev.origin_error_summary(errors, true, voxel=2.0)
    assert_matches(summary["median"], 2.5)
    assert_matches(summary["fraction_over_pixel"], 0.5)  # 5 A and 8 A exceed one 2 A pixel
    assert_matches(summary["true_rms_radius"], math.sqrt((0 + 25 + 2 + 100) / 4))
    assert summary["p99"] == pytest.approx(np.percentile(errors, 99))


def _origin_star(path, names, rot, origins, subset=None):
    """A particle STAR with angles and origins (and half-set labels when ``subset`` is given)."""
    optics = pd.DataFrame({"rlnOpticsGroup": [1], "rlnImagePixelSize": [2.0], "rlnImageSize": [8]})
    particles = {
        "rlnImageName": names,
        "rlnAngleRot": rot,
        "rlnAngleTilt": [10.0] * len(names),
        "rlnAnglePsi": [0.0] * len(names),
        "rlnOriginXAngst": [o[0] for o in origins],
        "rlnOriginYAngst": [o[1] for o in origins],
    }
    if subset is not None:
        particles["rlnRandomSubset"] = subset
    starfile.write({"optics": optics, "particles": pd.DataFrame(particles)}, path, overwrite=True)


def test_true_and_estimated_origins_are_matched_by_image_name_and_split_by_half(tmp_path):
    dataset, run = tmp_path / "dataset", tmp_path / "run"
    dataset.mkdir()
    run.mkdir()
    names = [f"{i}@a.mrcs" for i in (1, 2, 3, 4)]
    true_origins = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)]
    _origin_star(dataset / "particles.star", names, [0.0, 1.0, 2.0, 3.0], true_origins)
    order = [3, 0, 2, 1]  # the run lists the particles in another order
    estimated = [(0.5, 0.5), (1.5, 1.5), (2.5, 2.5), (3.5, 3.5)]
    _origin_star(
        run / "run_it003_data.star",
        [names[i] for i in order],
        [float(i) for i in order],
        [estimated[i] for i in order],
        subset=[1, 2, 1, 2],
    )
    halves = ev.true_directions_per_half(run, 3, dataset)
    assert_matches(halves[1]["true_origin"], np.array([true_origins[3], true_origins[2]]))
    assert_matches(halves[1]["est_origin"], np.array([estimated[3], estimated[2]]))
    assert_matches(halves[2]["true_origin"], np.array([true_origins[0], true_origins[1]]))
    assert_matches(halves[2]["est_origin"], np.array([estimated[0], estimated[1]]))


def test_a_particle_star_without_origins_is_refused(tmp_path):
    path = tmp_path / "particles.star"
    starfile.write(
        {
            "optics": pd.DataFrame({"rlnOpticsGroup": [1]}),
            "particles": pd.DataFrame(
                {"rlnImageName": ["1@a.mrcs"], "rlnAngleRot": [0.0], "rlnAngleTilt": [0.0], "rlnOriginXAngst": [0.0]}
            ),
        },
        path,
    )
    with pytest.raises(ValueError, match="rlnOriginYAngst"):
        ev.read_particles(path)


# ---- the sweep: mixtures of pools ----

KENT_RECORD = {"poses": "kent", "dataset_params_option": [4.0, 1.0, [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]}


def test_a_mixture_prior_is_the_weighted_sum_of_its_components_priors():
    record = {
        "poses": "mixture",
        "components": [{"weight": 0.25, "poses": KENT_RECORD}, {"weight": 0.75, "poses": {"poses": "uniform"}}],
    }
    expected = 0.25 * true_prior.population_prior(KENT_RECORD, 2) + 0.75 * np.full(n_pixels(2), 1.0 / n_pixels(2))
    prior = true_prior.population_prior(record, 2)
    assert_matches(prior, expected)
    assert_matches(prior.sum(), 1.0)


def test_a_mixture_of_one_component_is_that_component_and_bad_mixtures_are_refused():
    single = {"poses": "mixture", "components": [{"weight": 1.0, "poses": KENT_RECORD}]}
    assert_matches(true_prior.population_prior(single, 2), true_prior.population_prior(KENT_RECORD, 2))
    uniform = {"poses": "uniform"}
    for weights in ([0.5, 0.4], [1.2, -0.2], [0.0, 1.0]):
        record = {"poses": "mixture", "components": [{"weight": w, "poses": uniform} for w in weights]}
        with pytest.raises(ValueError, match="weights must be positive and sum to 1"):
            true_prior.population_prior(record, 2)
    with pytest.raises(ValueError, match="unknown pose distribution"):
        true_prior.population_prior({"poses": "belt"}, 2)


def test_the_evaluator_reads_a_mixture_record(tmp_path):
    record = {"poses": "mixture", "components": [{"weight": 1.0, "poses": {"poses": "uniform"}}]}
    (tmp_path / "pose_distribution.json").write_text(json.dumps(record))
    assert ev.read_pose_distribution(tmp_path) == record
    (tmp_path / "pose_distribution.json").write_text(json.dumps({"poses": "belt"}))
    with pytest.raises(ValueError, match="unknown pose distribution"):
        ev.read_pose_distribution(tmp_path)


def _reference_map(path, data):
    with mrcfile.new(path, overwrite=True) as handle:
        handle.set_data(np.asarray(data, dtype=np.float32))


def _pool(tmp_path, name, n, record, defocus0=10000.0):
    """A pool: the centred delta dataset of ``n`` images with its own pose record, micrograph names 1..n and the
    reference maps every pool shares."""
    directory = tmp_path / name
    directory.mkdir()
    _centred_delta_dataset(directory, n)
    blocks = starfile.read(directory / "particles.star", always_dict=True)
    blocks["particles"]["rlnMicrographName"] = [str(i + 1) for i in range(n)]
    blocks["particles"]["rlnDefocusU"] = defocus0 + np.arange(n)
    starfile.write(blocks, directory / "particles.star", overwrite=True)
    (directory / "pose_distribution.json").write_text(json.dumps({**record, "n_images": n, "pose_seed": 0}))
    for i, file in enumerate(mp.REFERENCE_FILES):
        _reference_map(directory / file, np.random.default_rng(i).standard_normal((4, 4, 4)))
    return directory


def test_a_mixture_takes_the_first_particles_of_each_pool_and_links_their_stacks(tmp_path):
    kent = _pool(tmp_path, "kent_pool", 5, KENT_RECORD)
    uniform = _pool(tmp_path, "uniform_pool", 6, {"poses": "uniform", "dataset_params_option": "uniform"}, 20000.0)
    out = tmp_path / "mix"
    record = mp.mix_pools([mp.Pool("kent", kent, 2), mp.Pool("uniform", uniform, 3)], out)
    particles = starfile.read(out / "particles.star", always_dict=True)["particles"]
    assert particles["rlnImageName"].tolist() == [
        "1@kent.particles.16.mrcs",
        "2@kent.particles.16.mrcs",
        "1@uniform.particles.16.mrcs",
        "2@uniform.particles.16.mrcs",
        "3@uniform.particles.16.mrcs",
    ]
    assert particles["rlnMicrographName"].tolist() == ["kent_1", "kent_2", "uniform_1", "uniform_2", "uniform_3"]
    assert_matches(particles["rlnDefocusU"].to_numpy(), [10000.0, 10001.0, 20000.0, 20001.0, 20002.0])
    for label, pool in (("kent", kent), ("uniform", uniform)):
        link = out / f"{label}.particles.16.mrcs"
        assert link.is_symlink() and link.resolve() == (pool / "particles.16.mrcs").resolve()
    assert (out / mp.REFERENCE_FILES[0]).read_bytes() == (kent / mp.REFERENCE_FILES[0]).read_bytes()
    assert record["poses"] == "mixture" and record["n_images"] == 5
    assert [(c["weight"], c["poses"]["poses"]) for c in record["components"]] == [(0.4, "kent"), (0.6, "uniform")]
    assert "n_images" not in record["components"][0]["poses"]  # only what the true prior needs
    assert json.loads((out / "pose_distribution.json").read_text()) == record
    expected = 0.4 * true_prior.population_prior(KENT_RECORD, 1) + 0.6 * np.full(n_pixels(1), 1.0 / n_pixels(1))
    assert_matches(true_prior.population_prior(record, 1), expected)


def test_a_mixture_of_a_prefix_is_a_subset_of_a_larger_mixture(tmp_path):
    kent = _pool(tmp_path, "kent_pool", 6, KENT_RECORD)
    uniform = _pool(tmp_path, "uniform_pool", 6, {"poses": "uniform"})
    small = mp.mix_pools([mp.Pool("kent", kent, 2), mp.Pool("uniform", uniform, 2)], tmp_path / "small")
    large = mp.mix_pools([mp.Pool("kent", kent, 4), mp.Pool("uniform", uniform, 4)], tmp_path / "large")
    names = {n: starfile.read(tmp_path / n / "particles.star", always_dict=True)["particles"] for n in ("small", "large")}
    assert set(names["small"]["rlnImageName"]) <= set(names["large"]["rlnImageName"])
    assert small["n_images"] == 4 and large["n_images"] == 8


def test_mixing_pools_refuses_a_count_beyond_the_pool_other_optics_other_references_and_repeated_labels(tmp_path):
    kent = _pool(tmp_path, "kent_pool", 4, KENT_RECORD)
    uniform = _pool(tmp_path, "uniform_pool", 4, {"poses": "uniform"})
    out = tmp_path / "mix"
    with pytest.raises(ValueError, match="has 4 particles, 5 asked"):
        mp.mix_pools([mp.Pool("kent", kent, 5)], out)
    with pytest.raises(ValueError, match="labels must be distinct"):
        mp.mix_pools([mp.Pool("a", kent, 1), mp.Pool("a", uniform, 1)], out)
    with pytest.raises(ValueError, match="no pools"):
        mp.mix_pools([], out)
    other = _pool(tmp_path, "other_optics", 4, {"poses": "uniform"})
    blocks = starfile.read(other / "particles.star", always_dict=True)
    blocks["optics"]["rlnImagePixelSize"] = 3.0
    starfile.write(blocks, other / "particles.star", overwrite=True)
    with pytest.raises(ValueError, match="optics of pool other differ"):
        mp.mix_pools([mp.Pool("kent", kent, 1), mp.Pool("other", other, 1)], out)
    for directory, scale in ((kent, 1.0), (uniform, 1.01)):
        for i, name in enumerate(mp.REFERENCE_FILES):
            _reference_map(directory / name, scale * np.random.default_rng(i).standard_normal((4, 4, 4)))
    with pytest.raises(ValueError, match="differs from that of pool kent"):
        mp.mix_pools([mp.Pool("kent", kent, 1), mp.Pool("uniform", uniform, 1)], out)


SWEEP = {
    "kent": {"kappa": 4.0, "beta": 1.0, "mu": [1.0, 1.0, 0.0], "mu0": [0.0, 0.0, 1.0]},
    "kent_weights": [0.0, 0.33, 1.0],
    "noise_levels": [0.5, 2.0],
    "n_images": [50000, 100000],
    "pool_seeds": {"kent": 17, "uniform": 18},
    "variants": ["learned", "fixed_true", "fixed_uniform"],
    "offsets": {
        "sigma_angstrom": 7.0, "noise_levels": [2.0], "kent_weights": [0.0, 1.0], "n_images": [50000],
        "pool_seeds": {"kent": 29, "uniform": 30},
    },
}


def test_the_sweep_names_its_scenarios_by_share_count_and_noise():
    assert sweep.mixture_name(0.33, 50000, 0.5) == "mix_w033_n50k_nz0p5"
    assert sweep.mixture_name(1.0, 100000, 2.0) == "mix_w100_n100k_nz2"
    assert sweep.pool_name("kent", 1.0) == "pool_kent_nz1"
    assert sweep.offsets_pool_name("uniform", 0.5) == "pool_uniform_off_nz0p5"
    assert sweep.offsets_mixture_name(0.66, 100000, 1.0) == "off_w066_n100k_nz1"
    with pytest.raises(ValueError, match="multiples of 1000"):
        sweep.count_tag(1500)


def test_the_sweep_has_two_pools_per_noise_level_and_a_mixture_per_cell():
    scenarios = sweep.sweep_scenarios(SWEEP)
    pools = {name: entry for name, entry in scenarios.items() if name.startswith("pool_") and "_off_" not in name}
    assert sorted(pools) == ["pool_kent_nz0p5", "pool_kent_nz2", "pool_uniform_nz0p5", "pool_uniform_nz2"]
    assert pools["pool_kent_nz2"]["dataset"] == {"noise_level": 2.0, "n_images": 100000, "seed": 17}
    assert pools["pool_uniform_nz2"]["dataset"]["seed"] == 18
    assert pools["pool_uniform_nz2"]["poses"] == {**pools["pool_kent_nz2"]["poses"], "kappa": 0.0, "beta": 0.0}
    assert pools["pool_kent_nz0p5"]["poses"] == {"kind": "kent", **SWEEP["kent"]}
    assert all(entry["variants"] == [] for entry in pools.values())
    mixtures = {name: entry for name, entry in scenarios.items() if name.startswith("mix_")}
    assert len(mixtures) == 3 * 2 * 2
    middle = mixtures["mix_w033_n50k_nz2"]["mixture_of"]
    assert middle == [
        {"label": "kent", "scenario": "pool_kent_nz2", "count": 16500},
        {"label": "uniform", "scenario": "pool_uniform_nz2", "count": 33500},
    ]


def test_a_kent_pool_without_concentration_or_ovalness_is_uniform():
    flat = {"poses": "kent", "dataset_params_option": [0.0, 0.0, [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]}
    assert_matches(true_prior.population_prior(flat, 3), np.full(n_pixels(3), 1.0 / n_pixels(3)))


def test_a_pure_mixture_has_one_pool_and_a_uniform_one_has_no_fixed_true_variant():
    scenarios = sweep.sweep_scenarios(SWEEP)
    assert scenarios["mix_w100_n100k_nz0p5"]["mixture_of"] == [
        {"label": "kent", "scenario": "pool_kent_nz0p5", "count": 100000}
    ]
    assert scenarios["mix_w000_n50k_nz0p5"]["mixture_of"] == [
        {"label": "uniform", "scenario": "pool_uniform_nz0p5", "count": 50000}
    ]
    assert scenarios["mix_w000_n50k_nz0p5"]["variants"] == ["learned", "fixed_uniform"]
    assert scenarios["mix_w033_n50k_nz0p5"]["variants"] == ["learned", "fixed_true", "fixed_uniform"]


def test_the_sweep_displaces_the_pools_of_the_offsets_cells_and_mixes_them_like_the_centred_ones():
    scenarios = sweep.sweep_scenarios(SWEEP)
    assert scenarios["pool_kent_off_nz2"] == {
        "shifted_from": "pool_kent_nz2",
        "offsets": {"sigma_angstrom": 7.0, "seed": 29},
        "variants": [],
    }
    assert scenarios["pool_uniform_off_nz2"]["offsets"]["seed"] == 30
    assert not [name for name in scenarios if "_off_nz0p5" in name]  # only the noise levels of the offsets entry
    mixtures = {name: entry for name, entry in scenarios.items() if name.startswith("off_")}
    assert sorted(mixtures) == ["off_w000_n50k_nz2", "off_w100_n50k_nz2"]
    assert mixtures["off_w100_n50k_nz2"]["mixture_of"] == [
        {"label": "kent", "scenario": "pool_kent_off_nz2", "count": 50000}
    ]
    assert mixtures["off_w000_n50k_nz2"]["variants"] == ["learned", "fixed_uniform"]
    for share_tag in ("w000", "w100"):  # the same particle counts as the centred cell
        assert [p["count"] for p in mixtures[f"off_{share_tag}_n50k_nz2"]["mixture_of"]] == [
            p["count"] for p in scenarios[f"mix_{share_tag}_n50k_nz2"]["mixture_of"]
        ]
    without = sweep.sweep_scenarios({**SWEEP, "offsets": None})
    assert not [name for name in without if name.startswith("off_") or "_off_" in name]


def test_the_offsets_cells_must_be_cells_of_the_sweep():
    for key, value in (("noise_levels", [3.0]), ("kent_weights", [0.5]), ("n_images", [70000])):
        with pytest.raises(ValueError, match=f"offsets {key}"):
            sweep.sweep_scenarios({**SWEEP, "offsets": {**SWEEP["offsets"], key: value}})


def test_the_sweep_runs_every_mixture_then_the_mixtures_with_offsets():
    names = sweep.sweep_run_scenarios(SWEEP)
    assert len(names) == 3 * 2 * 2 + 2 and names[-2:] == ["off_w000_n50k_nz2", "off_w100_n50k_nz2"]
    assert set(names[:-2]) == {n for n in sweep.sweep_scenarios(SWEEP) if n.startswith("mix_")}
    assert sweep.sweep_run_scenarios({**SWEEP, "offsets": None}) == names[:-2]


def test_the_sweep_refuses_a_kent_share_outside_zero_and_one():
    with pytest.raises(ValueError, match="kent_weights must lie in"):
        sweep.sweep_scenarios({**SWEEP, "kent_weights": [0.5, 1.2]})


def _fake_variants_csv(root, scenario, auc, ceiling=0.9, variants=("learned", "fixed_true", "fixed_uniform")):
    """A comparison/variants.csv of ``scenario`` with the columns the summary reads, one row per variant."""
    directory = root / scenario / "comparison"
    directory.mkdir(parents=True)
    columns = (
        "variant", "iterations", "final_healpix_order", "gt_fsc_auc_aligned", "gt_fsc_auc_true_poses",
        "gt_resolution_0.143_A_aligned", "pose_error_median_aligned_deg", "pose_error_fraction_over_90_aligned",
        "inplane_error_median_aligned_deg", "gt_cfar_aligned", "origin_error_median_A",
    )
    with open(directory / "variants.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for i, variant in enumerate(variants):
            writer.writerow({
                "variant": variant, "iterations": 15, "final_healpix_order": 5, "gt_fsc_auc_aligned": auc + 0.01 * i,
                "gt_fsc_auc_true_poses": ceiling, "gt_resolution_0.143_A_aligned": "" if i == 0 else 7.5,
                "pose_error_median_aligned_deg": 2.0 + i, "pose_error_fraction_over_90_aligned": 0.01 * (i + 1),
                "inplane_error_median_aligned_deg": 0.5, "gt_cfar_aligned": 0.01, "origin_error_median_A": 1.3,
            })


SMALL_SWEEP = {
    "kent_weights": [0.0, 1.0],
    "noise_levels": [0.5, 1.0],
    "n_images": [50000, 100000],
    "variants": ["learned", "fixed_true", "fixed_uniform"],
    "offsets": {
        "sigma_angstrom": 7.0, "noise_levels": [1.0], "kent_weights": [0.0, 1.0], "n_images": [100000],
        "pool_seeds": {"kent": 29, "uniform": 30},
    },
}


def test_the_summary_collects_every_scenario_with_its_coordinates_and_the_ceiling_fraction(tmp_path):
    for cell in ss.sweep_cells(SMALL_SWEEP):
        _fake_variants_csv(tmp_path, cell["scenario"], auc=0.5)
    rows, missing = ss.collect_rows([tmp_path], SMALL_SWEEP)
    assert missing == [] and len(rows) == (2 * 2 * 2 + 2) * 3
    first = rows[0]
    assert (first["scenario"], first["kent_weight"], first["n_images"], first["noise_level"]) == ("mix_w000_n50k_nz0p5", 0.0, 50000, 0.5)
    assert_matches(first["gt_fsc_auc_fraction_of_true_poses"], 0.5 / 0.9)
    assert first["gt_resolution_0.143_A_aligned"] is None  # an empty cell stays empty
    shifted = [row for row in rows if "offsets_sigma_A" in row]
    assert {row["scenario"] for row in shifted} == {"off_w000_n100k_nz1", "off_w100_n100k_nz1"}
    assert {row["n_images"] for row in shifted} == {100000}


def test_the_summary_lists_the_scenarios_that_have_no_comparison_and_draws_the_rest(tmp_path):
    cells = ss.sweep_cells(SMALL_SWEEP)
    for cell in cells[:-2]:
        _fake_variants_csv(tmp_path, cell["scenario"], auc=0.5)
    rows, missing = ss.collect_rows([tmp_path], SMALL_SWEEP)
    assert missing == [cell["scenario"] for cell in cells[-2:]]
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="allow-missing"):
        ss.main(["--output-root", str(tmp_path), "--out-dir", str(out), "--config", str(_sweep_yaml(tmp_path))])
    ss.main(["--output-root", str(tmp_path), "--out-dir", str(out), "--config", str(_sweep_yaml(tmp_path)), "--allow-missing"])
    assert (out / "summary.csv").stat().st_size > 0
    assert all((out / f"summary_{column}.png").stat().st_size > 0 for column, *_ in ss.METRICS)
    assert not (out / "summary_offsets.png").exists()  # the cells with offsets did not finish


def _sweep_yaml(tmp_path):
    """An override file that gives the summary ``SMALL_SWEEP``."""
    import yaml

    path = tmp_path / "small_sweep.yaml"
    path.write_text(yaml.safe_dump({"sweep": {k: v for k, v in SMALL_SWEEP.items()}}))
    return path


def test_the_summary_draws_the_cells_with_offsets_against_their_centred_counterparts(tmp_path):
    for cell in ss.sweep_cells(SMALL_SWEEP):
        _fake_variants_csv(tmp_path, cell["scenario"], auc=0.5)
    out = tmp_path / "out"
    ss.main(["--output-root", str(tmp_path), "--out-dir", str(out), "--config", str(_sweep_yaml(tmp_path))])
    assert (out / "summary_offsets.png").stat().st_size > 0
    with open(out / "summary.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 30 and "offsets_sigma_A" in rows[0]


def test_override_files_are_merged_over_the_default_sweep(tmp_path):
    path = tmp_path / "override.yaml"
    path.write_text("sweep:\n  noise_levels: [3.0]\n  offsets: null\n")
    merged = ss.load_sweep([path])
    assert merged["noise_levels"] == [3.0] and merged["offsets"] is None
    assert merged["kent"]["kappa"] == 4.0 and merged["n_images"] == [50000, 100000]  # untouched entries stay


def test_the_summary_takes_each_scenario_from_the_first_root_that_has_it(tmp_path):
    cells = ss.sweep_cells(SMALL_SWEEP)
    first, second = tmp_path / "first", tmp_path / "second"
    for cell in cells:
        _fake_variants_csv(second if cell["scenario"].startswith("off_") else first, cell["scenario"], auc=0.5)
    rows, missing = ss.collect_rows([second, first], SMALL_SWEEP)
    assert missing == [] and len(rows) == 30
    rows, missing = ss.collect_rows([first], SMALL_SWEEP)
    assert missing == [cell["scenario"] for cell in cells if cell["scenario"].startswith("off_")]


def test_a_mixture_of_displaced_pools_records_the_offsets_and_refuses_a_mix_with_centred_pools(tmp_path):
    offsets = {"kind": "gaussian", "sigma_angstrom": 7.0, "seed": 29}
    kent = _pool(tmp_path, "kent_off", 4, {**KENT_RECORD, "offsets": offsets})
    uniform = _pool(tmp_path, "uniform_off", 4, {"poses": "uniform", "offsets": {**offsets, "seed": 30}})
    record = mp.mix_pools([mp.Pool("kent", kent, 2), mp.Pool("uniform", uniform, 2)], tmp_path / "mix")
    assert record["offsets"]["sigma_angstrom"] == 7.0
    assert [record["offsets"]["pools"][label]["seed"] for label in ("kent", "uniform")] == [29, 30]
    assert json.loads((tmp_path / "mix" / "pose_distribution.json").read_text()) == record
    centred = _pool(tmp_path, "centred", 4, {"poses": "uniform"})
    assert "offsets" not in mp.mix_pools([mp.Pool("centred", centred, 2)], tmp_path / "plain")
    with pytest.raises(ValueError, match="some pools are displaced"):
        mp.mix_pools([mp.Pool("kent", kent, 1), mp.Pool("centred", centred, 1)], tmp_path / "bad")
    wide = _pool(tmp_path, "wide", 4, {"poses": "uniform", "offsets": {**offsets, "sigma_angstrom": 9.0}})
    with pytest.raises(ValueError, match="different sigmas"):
        mp.mix_pools([mp.Pool("kent", kent, 1), mp.Pool("wide", wide, 1)], tmp_path / "bad")


# ---- renders of the final maps ----


def test_the_preferred_axis_is_the_cap_centre_and_a_distribution_without_one_gives_z():
    axis = rv.preferred_axis(KENT_RECORD)
    assert abs(axis[2]) > 0.999 and axis[2] > 0
    flat = {"poses": "kent", "dataset_params_option": [0.0, 0.0, [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]}
    assert_matches(rv.preferred_axis(flat), np.array([0.0, 0.0, 1.0]))


def test_a_view_turns_the_map_about_its_centre_so_the_direction_faces_the_viewer_and_up_is_up():
    rng = np.random.default_rng(0)
    center = np.array([100.0, 120.0, 90.0])
    for _ in range(5):
        direction, up = rng.standard_normal(3), rng.standard_normal(3)
        matrix = np.array(rv.view_matrix(direction, up, center)).reshape(3, 4)
        rotation, translation = matrix[:, :3], matrix[:, 3]
        assert_matches(rotation @ rotation.T, np.eye(3))
        assert_matches(np.linalg.det(rotation), 1.0)
        unit = direction / np.linalg.norm(direction)
        perpendicular_up = up - unit * (unit @ up)
        assert_matches(rotation @ unit, [0.0, 0.0, 1.0])
        assert_matches(rotation @ (perpendicular_up / np.linalg.norm(perpendicular_up)), [0.0, 1.0, 0.0])
        assert_matches(rotation @ center + translation, center)


def test_the_front_view_looks_down_the_axis_and_the_side_view_across_it_with_the_axis_up():
    for axis in (np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]), np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)):
        views = rv.view_directions(axis)
        assert_matches(views["front"][0], axis)
        assert abs(views["front"][0] @ views["front"][1]) < 1e-12
        assert abs(views["side"][0] @ axis) < 1e-12
        assert_matches(views["side"][1], axis)
        for direction, up in views.values():
            assert_matches(np.linalg.norm(direction), 1.0)
            assert_matches(np.linalg.norm(up), 1.0)


def test_a_map_is_drawn_at_the_level_that_encloses_the_reference_volume():
    rng = np.random.default_rng(1)
    reference = rng.random((12, 12, 12))
    n_voxels = rv.enclosed_voxels(reference, 0.7)
    other = 5.0 * rng.random((12, 12, 12)) - 1.0  # another scale and offset
    level = rv.matched_level(other, n_voxels)
    assert np.count_nonzero(other > level) == n_voxels
    assert rv.enclosed_voxels(reference) == np.count_nonzero(reference > rv.GT_LEVEL_FRACTION * reference.max())


def _fake_xvfb_run(tmp_path):
    """An ``xvfb-run`` stand-in that runs the driver script with a ChimeraX ``run`` that makes the PNG of every save
    command and fails on a command that contains FAILME."""
    path = tmp_path / "fake_xvfb_run"
    path.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, sys, types\n"
        "def run(session, command):\n"
        "    if 'FAILME' in command:\n"
        "        raise RuntimeError('boom')\n"
        "    if command.startswith('save '):\n"
        "        pathlib.Path(command.split()[1]).write_bytes(b'png')\n"
        "modules = {name: types.ModuleType(name) for name in ('chimerax', 'chimerax.core', 'chimerax.core.commands')}\n"
        "modules['chimerax.core.commands'].run = run\n"
        "sys.modules.update(modules)\n"
        "exec(pathlib.Path(sys.argv[-1]).read_text(), {'session': None})\n"
    )
    path.chmod(0o755)
    return path


def _render_dataset(tmp_path, box=8, voxel=2.0):
    """A dataset directory with a pose record and a ground-truth map, and a second map of the same grid."""
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "pose_distribution.json").write_text(json.dumps(KENT_RECORD))
    volume = np.random.default_rng(2).random((box, box, box)).astype(np.float32)
    for name in ("reference_gt_relion.mrc", "refined.mrc"):
        with mrcfile.new(dataset / name) as handle:
            handle.set_data(volume)
            handle.voxel_size = voxel
    return dataset


def test_a_render_job_writes_a_front_and_a_side_image_through_chimerax(tmp_path):
    dataset = _render_dataset(tmp_path)
    prefix = tmp_path / "out" / "learned"
    written = rv.render_maps(
        dataset, [(dataset / "refined.mrc", prefix)], "chimerax-not-run", str(_fake_xvfb_run(tmp_path)), image_pixels=64
    )
    assert written == [tmp_path / "out" / "learned_front.png", tmp_path / "out" / "learned_side.png"]
    assert all(path.read_bytes() == b"png" for path in written)


def test_a_failed_chimerax_command_a_missing_image_and_another_grid_are_refused(tmp_path):
    dataset = _render_dataset(tmp_path)
    fake = str(_fake_xvfb_run(tmp_path))
    with pytest.raises(RuntimeError, match="failed commands"):
        rv.render_maps(dataset, [(dataset / "refined.mrc", tmp_path / "FAILME")], "chimerax", fake)
    silent = tmp_path / "silent_xvfb_run"
    silent.write_text("#!/bin/sh\nexit 0\n")
    silent.chmod(0o755)
    with pytest.raises(RuntimeError, match="missing images"):
        rv.render_maps(dataset, [(dataset / "refined.mrc", tmp_path / "out")], "chimerax", str(silent))
    with mrcfile.new(dataset / "other.mrc") as handle:
        handle.set_data(np.zeros((8, 8, 8), np.float32))
        handle.voxel_size = 3.0
    with pytest.raises(ValueError, match="voxel size"):
        rv.render_maps(dataset, [(dataset / "other.mrc", tmp_path / "out")], "chimerax", fake)


def test_the_chimerax_commands_draw_every_map_in_every_view_with_one_camera():
    geometry = rv.MapGeometry(256, 2.125)
    matrices = {view: [1.0] * 12 for view in rv.VIEWS}
    commands = rv.chimerax_commands(
        [(Path("/a.mrc"), Path("/out/a"), 0.5), (Path("/b.mrc"), Path("/out/b"), 0.25)], geometry, matrices, 800
    )
    assert sum(command.startswith("save ") for command in commands) == 4
    assert sum(command.startswith("view matrix models #1,") for command in commands) == 4
    assert sum(command.startswith("camera mono") for command in commands) == 1
    assert any("level 0.25 " in command for command in commands)
    assert "save /out/b_side.png width 800 height 800 supersample 3" in commands
    compile(rv.driver_script(commands, Path("/log")), "driver", "exec")


def test_the_crop_of_a_view_holds_every_image_and_blank_images_are_refused():
    first, second = np.ones((100, 100, 3)), np.ones((100, 100, 3))
    first[20:30, 40:50] = 0.2
    second[60:70, 10:20] = 0.2
    top, bottom, left, right = cmp.render_crop([first, second])
    assert (top, bottom, left, right) == (20 - 4, 70 + 4, 10 - 4, 50 + 4)
    with pytest.raises(ValueError, match="blank"):
        cmp.render_crop([np.ones((10, 10, 3))])


def test_the_comparison_figure_has_a_render_block_when_the_images_exist(tmp_path):
    import matplotlib.image

    learned = _evaluation(tmp_path, "learned", source="learned", auc=0.8, median=2.0, js_skill=0.9)
    fixed = _evaluation(tmp_path, "fixed_true", source="fixed", auc=0.85, median=1.5, js_skill=0.9)
    image = np.ones((60, 60, 3))
    image[15:45, 10:50] = 0.3
    truth = tmp_path / "render_truth"
    for directory, stem in ((tmp_path, "render_truth"), (learned, "render"), (fixed, "render")):
        for view in rv.VIEWS:
            matplotlib.image.imsave(directory / f"{stem}_{view}.png", image)
    plain, drawn = tmp_path / "plain", tmp_path / "drawn"
    arguments = ["--runs", f"learned={learned}", f"fixed_true={fixed}", "--label", "test"]
    cmp.main([*arguments, "--out-dir", str(plain)])
    cmp.main([*arguments, "--out-dir", str(drawn), "--renders", str(truth)])
    assert matplotlib.image.imread(drawn / "variants.png").shape[0] > matplotlib.image.imread(plain / "variants.png").shape[0]
    (learned / "render_side.png").unlink()
    with pytest.raises(SystemExit, match="missing render images"):
        cmp.main([*arguments, "--out-dir", str(tmp_path / "again"), "--renders", str(truth)])


def test_chimerax_gets_the_library_directory_beside_its_executable(tmp_path, monkeypatch):
    library = tmp_path / "usr" / "libexec" / "UCSF-ChimeraX" / "lib"
    library.mkdir(parents=True)
    (tmp_path / "usr" / "bin").mkdir()
    executable = tmp_path / "usr" / "bin" / "chimerax"
    executable.write_text("")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/already/there")
    assert rv.chimerax_environment(str(executable))["LD_LIBRARY_PATH"] == f"{library}:/already/there"
    monkeypatch.delenv("LD_LIBRARY_PATH")
    assert rv.chimerax_environment(str(executable))["LD_LIBRARY_PATH"] == str(library)
    assert "LD_LIBRARY_PATH" not in rv.chimerax_environment(str(tmp_path / "elsewhere" / "chimerax"))
