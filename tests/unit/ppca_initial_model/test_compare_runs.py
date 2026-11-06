"""The run comparison of scripts/ppca_schedule/compare_runs.py: noise from replicates, flags and the table."""

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.ppca_schedule import compare_runs as cmp

pytestmark = pytest.mark.unit


def test_the_noise_is_the_range_of_the_replicates_but_at_least_the_floor():
    rows = [{"pose": 4.0}, {"pose": 4.6}, {"pose": 4.2}]
    assert cmp.noise(rows, "pose", 0.2) == pytest.approx(0.6)
    assert cmp.noise(rows[:1], "pose", 0.2) == 0.2
    assert cmp.noise([{"pose": 4.0}, {"pose": 4.05}], "pose", 0.2) == 0.2


def test_runs_of_the_same_arm_and_seed_in_two_rounds_get_distinct_labels():
    runs = [Path("r15/C/sgd/seed11"), Path("r16/C/sgd/seed11"), Path("r16/C/other/seed11")]
    labels = cmp.run_labels(runs)
    assert labels[runs[2]] == "other/seed11"
    assert len({labels[run] for run in runs}) == 3
    assert labels[runs[0]].endswith("sgd/seed11") and labels[runs[0]] != labels[runs[1]]


@pytest.mark.parametrize(
    "value, direction, expected",
    [(4.0, -1, "+"), (6.0, -1, "-"), (4.9, -1, " "), (0.95, 1, "+"), (0.5, 1, "-"), (0.5, 0, " ")],
)
def test_a_value_is_flagged_only_when_it_differs_from_the_baseline_by_more_than_twice_the_noise(value, direction, expected):
    baseline = 5.0 if direction == -1 else 0.8
    noise_level = 0.2 if direction == -1 else 0.05
    assert cmp.flag(value, baseline, noise_level, direction) == expected


def test_the_band_ratio_is_the_model_power_over_the_scaled_truth_power():
    model = np.arange(32, dtype=float)
    truth = np.arange(32, dtype=float)
    ratios = cmp.band_ratios(model, truth, 1.0, "pc")
    assert ratios == {"pc_pow_1-4": 1.0, "pc_pow_5-15": 1.0, "pc_pow_16-31": 1.0}
    assert cmp.band_ratios(model, truth, 4.0, "mean")["mean_pow_5-15"] == pytest.approx(0.25)
    assert cmp.model_scale(np.full(32, 8.0), np.full(32, 2.0)) == 4.0


def test_the_wall_time_spans_the_first_and_last_timestamped_lines(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("2026-11-05 10:00:00,100 a INFO start\nno stamp\n2026-11-05 10:12:30,900 a INFO end\n")
    assert cmp.wall_minutes(log) == pytest.approx(12.5)
    log.write_text("2026-11-05 10:00:00,100 a INFO only\n")
    assert cmp.wall_minutes(log) is None


def test_a_run_table_flags_the_runs_against_the_baseline_and_leaves_references_unflagged(tmp_path):
    for arm, pose in (("base", 5.0), ("better", 3.9), ("ref", 3.0)):
        run = tmp_path / arm / "seed11"
        run.mkdir(parents=True)
        (run / "suite_scores.json").write_text(
            json.dumps({"checkpoints": {"3000": {"pose_direct_median_deg": pose, "state_fsc_direct": 0.9}}})
        )
        (run / "latent_scores.json").write_text(json.dumps({"between_state_r2": 0.7}))
    rows = {f"{arm}/seed11": cmp.load_run(tmp_path / arm / "seed11", 3000) for arm in ("base", "better", "ref")}
    table = cmp.format_table(rows, "base/seed11", [], cmp.METRICS, frozenset({"ref/seed11"}))
    cells = {line.split("|")[1].strip(): line.split("|")[2] for line in table.splitlines()[2:]}
    assert cells["base/seed11"].strip() == "5.00" and cells["better/seed11"].strip() == "3.90+"
    assert cells["ref/seed11"].strip() == "3.00"
