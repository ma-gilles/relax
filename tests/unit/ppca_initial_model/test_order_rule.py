"""The accuracy-driven HEALPix order (Config.order_rule): RELION's updateAngularSampling rule for the PPCA mean."""

import dataclasses
import json

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.ppca_initial_model import checkpoint
from relax.ppca_initial_model.config import Config
from relax.ppca_initial_model.order_rule import angular_step_deg, final_order, next_order, order_for
from relax.ppca_initial_model.state import State
from relax.ppca_initial_model.update import empty_moments

pytestmark = pytest.mark.unit


def _state(next_order=None):
    theta = jnp.zeros((4, 2), jnp.complex64)
    return State(
        theta, empty_moments(theta), jnp.ones(3, jnp.float32), 0, np.arange(4), {}, 2.0, 2, {}, next_order=next_order
    )


@pytest.mark.parametrize(
    "options, message",
    [
        ({"order_rule": "fixed"}, "stages or accuracy"),
        ({"order_rule": "accuracy", "accuracy_interval": 0}, "positive integer"),
        ({"order_rule": "accuracy", "order_max": True}, "positive integer"),
        ({"order_rule": "accuracy", "stages": ((1, 4, 3),), "order_max": 2}, "below the first stage"),
        ({"order_rule": "accuracy", "final_order_max": 3}, "at least order_max"),
        ({"order_rule": "accuracy", "oversampling": 1}, "without oversampling"),
        ({"order_rule": "accuracy", "accuracy_signal": "kl"}, "divergence or mean_only"),
        ({"order_rule": "accuracy", "accuracy_kl_threshold": 0.0}, "finite and positive"),
        ({"order_rule": "accuracy", "accuracy_kl_threshold": float("inf")}, "finite and positive"),
    ],
)
def test_the_order_rule_options_are_validated(options, message):
    with pytest.raises(ValueError, match=message):
        Config(**options)


def test_the_angular_step_halves_with_each_order():
    assert [angular_step_deg(k) for k in (1, 2, 3, 4, 5)] == [30.0, 15.0, 7.5, 3.75, 1.875]


@pytest.mark.parametrize(
    "order, acc_rot, expected",
    [
        (3, 20.0, 3),  # 7.5 degrees is below 0.75 * 20: fine enough
        (3, 10.01, 3),  # 7.5 is below 0.75 * 10.01
        (3, 10.0, 4),  # a step equal to the share is not below it, as in RELION
        (3, 9.9, 4),  # 7.5 is at least 0.75 * 9.9: one finer
        (4, 4.0, 4),  # the cap
        (1, 1.0, 2),  # one order per estimate, however inaccurate the grid
    ],
)
def test_the_order_rises_by_one_while_the_step_is_at_least_three_quarters_of_the_accuracy(order, acc_rot, expected):
    assert next_order(order, acc_rot, order_max=4) == expected


def test_the_final_pass_takes_the_first_fine_enough_order_up_to_its_cap():
    assert final_order(3, 2.0, final_order_max=5) == 5  # steps 7.5, 3.75 and 1.875 against 0.75 * 2 = 1.5
    assert final_order(3, 2.0, final_order_max=4) == 4
    assert final_order(3, 20.0, final_order_max=5) == 3


def test_the_order_comes_from_the_table_or_the_state():
    table = Config(stages=((1, 4, 1), (5, 8, 3)))
    assert [order_for(table, _state(), i) for i in (1, 4, 5)] == [1, 1, 3]
    accuracy = Config(stages=((1, 4, 2), (5, 8, 3)), order_rule="accuracy")
    assert order_for(accuracy, _state(), 7) == 2
    assert order_for(accuracy, _state(next_order=4), 7) == 4


def test_the_planned_order_is_saved_and_an_older_checkpoint_resumes_without_one(tmp_path):
    config = Config(q=1, order_rule="accuracy")
    identity = {"input": "x"}
    checkpoint.save(tmp_path / "a.npz", _state(next_order=3), config, identity)
    assert checkpoint.load(tmp_path / "a.npz", config, identity).next_order == 3

    with np.load(tmp_path / "a.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    metadata = json.loads(str(arrays["metadata"]))
    del metadata["next_order"]
    np.savez(tmp_path / "old.npz", **{**arrays, "metadata": json.dumps(metadata)})
    assert checkpoint.load(tmp_path / "old.npz", config, identity).next_order is None


def test_an_older_checkpoint_of_the_stages_rule_keeps_its_identity():
    identity = checkpoint._identity_config(dataclasses.asdict(Config(q=1)))
    settings = {"order_rule", "order_max", "final_order_max", "accuracy_interval", "accuracy_signal", "accuracy_kl_threshold"}
    assert not settings & identity.keys()
    changed = checkpoint._identity_config(dataclasses.asdict(Config(q=1, order_rule="accuracy")))
    assert changed["order_rule"] == "accuracy"


def _run_command(star, output, *extra):
    """``relax ppca_initial_model`` on ``star`` for 8 iterations of a fixed batch of 8 particles."""
    from relax.commands import ppca_initial_model

    argv = [
        "--i", str(star), "--particle-diameter", "12", "-o", str(output), "--q", "1", "--seed", "11",
        "--iterations", "8", "--stages", "[[1,2,1]]", "--stochastic-batch-size", "8", "--stochastic-all-iterations",
        "--image-batch-size", "4", "--rotation-block-size", "64", "--skip-final-embeddings", *extra,
    ]  # fmt: skip
    ppca_initial_model.main(argv)


def _log(directory):
    return [json.loads(line) for line in open(directory / "iterations.jsonl")]


def test_an_accuracy_run_logs_its_accuracy_and_orders_and_resumes_to_the_same_orders_and_model(monkeypatch, tmp_path):
    """The first iteration runs at the table's order, each estimate (every second iteration and the last) plans the
    next one, and a run stopped at 4 and resumed ends at the orders and the model of the uninterrupted run."""
    from test_radius_rule import _write_star

    monkeypatch.setenv("SLURM_JOB_ID", "test")
    star = _write_star(tmp_path / "data", 16)
    options = ("--order-rule", "accuracy", "--order-max", "2", "--accuracy-interval", "2")
    full = tmp_path / "full"
    _run_command(star, full, *options)
    log = _log(full)
    estimated = [row["iteration"] for row in log if "pose_accuracy" in row]
    assert estimated == [2, 4, 6, 8]
    assert [row["healpix_order"] for row in log][0] == 1
    for row, following in zip(log, log[1:]):
        assert following["healpix_order"] == row["next_order"]
    assert all(1 <= row["healpix_order"] <= 2 for row in log) and log[-1]["next_order"] <= 2
    for row in log:
        if "pose_accuracy" in row:
            accuracy = row["pose_accuracy"]
            assert 0 < accuracy["acc_rot"] <= 35 and 0 < accuracy["acc_rot_mean_only"] <= 35
            assert accuracy["n_trials"] == row["half_counts"][0] and set(accuracy["terms_at_error"]) == {
                "mean",
                "log_det",
                "trace",
            }

    split = tmp_path / "split"
    _run_command(star, split, *options, "--stop-after", "4")
    _run_command(star, split, *options, "--resume", str(split / "checkpoint_0004.npz"))
    resumed = _log(split)
    assert [row["healpix_order"] for row in resumed] == [row["healpix_order"] for row in log]
    with np.load(full / "checkpoint_0008.npz") as a, np.load(split / "checkpoint_0008.npz") as b:
        assert_matches(b["theta"], a["theta"])


def test_the_stages_rule_logs_no_order_plan(monkeypatch, tmp_path):
    from test_radius_rule import _write_star

    monkeypatch.setenv("SLURM_JOB_ID", "test")
    star = _write_star(tmp_path / "data", 16)
    _run_command(star, tmp_path / "run")
    log = _log(tmp_path / "run")
    assert all("next_order" not in row and "pose_accuracy" not in row for row in log)
    assert [row["healpix_order"] for row in log] == [1] * 8


def test_the_accuracy_threshold_is_a_command_line_option_whose_default_is_relions():
    import argparse

    from relax.commands import ppca_initial_model

    parser = argparse.ArgumentParser()
    ppca_initial_model.add_args(parser)
    required = ["--i", "x", "--particle-diameter", "1", "-o", "y"]
    assert parser.parse_args(required).accuracy_kl_threshold is None
    assert parser.parse_args([*required, "--accuracy-kl-threshold", "18.4"]).accuracy_kl_threshold == 18.4
