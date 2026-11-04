"""The estimates scripts/plot_ppca_pose_accuracy.py reads from a run's iterations.jsonl and writes."""

import json
import sys

import pytest

from relax.ppca_initial_model.order_rule import angular_step_deg
from scripts import plot_ppca_pose_accuracy as pose_accuracy

pytestmark = pytest.mark.unit


def _plain(iteration: int) -> dict:
    """An iteration record that made no estimate."""
    return {"iteration": iteration, "healpix_order": 1, "radius": 15, "direction_prior": [0.1] * 50}


def _estimate(iteration: int, *, order: int = 1, acc_rot: float = 20.0) -> dict:
    """An iteration record that carries a ``pose_accuracy`` estimate (the mean-only value is one degree above)."""
    terms = {"mean": 0.1, "log_det": -0.01, "trace": 0.2}
    accuracy = {"acc_rot": acc_rot, "acc_rot_mean_only": acc_rot + 1, "terms_at_error": terms, "n_trials": 100}
    return {**_plain(iteration), "healpix_order": order, "next_order": order, "pose_accuracy": accuracy}


def _write(path, records) -> None:
    """One JSON record per line."""
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def _run(tmp_path, order_rule: str = "accuracy", signal: str = "divergence"):
    """A run directory with a run.json and an iterations.jsonl of estimates at iterations 10, 20 and 30."""
    (tmp_path / "run.json").write_text(json.dumps({"config": {"order_rule": order_rule, "accuracy_signal": signal}}))
    records = [_estimate(k, order=1 + k // 20, acc_rot=40.0 - k) if k % 10 == 0 else _plain(k) for k in range(1, 31)]
    _write(tmp_path / "iterations.jsonl", records)
    return tmp_path


def test_only_the_estimates_are_read_and_a_resume_supersedes_the_lines_it_repeats(tmp_path):
    """A run resumed from iteration 20 appends iteration 20 again: the earlier 20 and 30 give way to the new ones."""
    records = [_estimate(k, acc_rot=30.0) if k % 10 == 0 else _plain(k) for k in range(1, 31)]
    records += [_estimate(20, acc_rot=11.0), _plain(21), _estimate(30, acc_rot=9.0)]
    _write(tmp_path / "iterations.jsonl", records)
    read = pose_accuracy.read_estimates(tmp_path / "iterations.jsonl")
    assert [row["iteration"] for row in read] == [10, 20, 30]
    assert [row["pose_accuracy"]["acc_rot"] for row in read] == [30.0, 11.0, 9.0]
    assert set(read[0]) == set(pose_accuracy.KEPT)


def test_a_record_whose_iteration_follows_long_arrays_is_read(tmp_path):
    """A VDAM record opens with its shell gates, so its iteration is far into the line."""
    gates = {"gates": [[0.5, 0.5]] * 2000, "signal": [0.1] * 2000}
    plain = {**gates, **_plain(1)}
    estimate = {**gates, **_estimate(2, acc_rot=8.0)}
    _write(tmp_path / "iterations.jsonl", [plain, estimate])
    (read,) = pose_accuracy.read_estimates(tmp_path / "iterations.jsonl")
    assert read["iteration"] == 2 and read["pose_accuracy"]["acc_rot"] == 8.0


def test_a_file_without_an_estimate_is_refused(tmp_path):
    """A missing measurement is an error, not an empty figure."""
    _write(tmp_path / "iterations.jsonl", [_plain(k) for k in range(1, 5)])
    with pytest.raises(ValueError, match="no pose_accuracy estimate"):
        pose_accuracy.read_estimates(tmp_path / "iterations.jsonl")


def test_a_line_without_an_iteration_is_refused(tmp_path):
    """A line that is not an iteration record stops the read instead of being skipped."""
    (tmp_path / "iterations.jsonl").write_text(json.dumps(_estimate(10)) + "\n" + json.dumps({"other": 1}) + "\n")
    with pytest.raises(ValueError, match="line 2: no iteration"):
        pose_accuracy.read_estimates(tmp_path / "iterations.jsonl")


@pytest.mark.parametrize("signal", ["divergence", "mean_only"])
def test_the_signal_of_an_accuracy_run_is_read_and_a_stages_run_is_refused(tmp_path, signal):
    """Only a run with --order-rule accuracy estimates the accuracy; the table's runs have nothing to plot."""
    assert pose_accuracy.read_signal(_run(tmp_path, signal=signal)) == signal
    with pytest.raises(ValueError, match="only --order-rule accuracy"):
        pose_accuracy.read_signal(_run(tmp_path, order_rule="stages"))


def test_a_table_row_flattens_the_estimate_and_adds_the_step_of_the_order(tmp_path):
    """Order, plan, radius, both accuracies, the divergence terms and the angular step of the order in use."""
    (row,) = pose_accuracy.tabulate([_estimate(10, order=2, acc_rot=12.5)])
    assert row == {
        "iteration": 10,
        "order": 2,
        "next_order": 2,
        "radius": 15,
        "step_deg": angular_step_deg(2),
        "acc_rot": 12.5,
        "acc_rot_mean_only": 13.5,
        "mean": 0.1,
        "log_det": -0.01,
        "trace": 0.2,
        "n_trials": 100,
    }


@pytest.mark.parametrize("signal", ["divergence", "mean_only"])
def test_the_command_writes_the_table_and_the_figure_for_either_signal(tmp_path, monkeypatch, signal):
    """One record per estimate in the JSON, and a non-empty PNG."""
    run, out = tmp_path / "run", tmp_path / "out"
    run.mkdir()
    _run(run, signal=signal)
    monkeypatch.setattr(sys, "argv", ["plot_ppca_pose_accuracy.py", str(run), "-o", str(out)])
    pose_accuracy.main()
    table = json.loads((out / "pose_accuracy.json").read_text())
    assert [row["iteration"] for row in table] == [10, 20, 30]
    assert [row["acc_rot"] for row in table] == [30.0, 20.0, 10.0]
    assert (out / "pose_accuracy.png").stat().st_size > 0
