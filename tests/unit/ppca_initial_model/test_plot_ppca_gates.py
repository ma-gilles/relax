"""How scripts/plot_ppca_gates.py reads an iterations.jsonl: partial last lines, repeated runs and gate images."""

import json

import numpy as np
import pytest

from scripts import plot_ppca_gates as gates

pytestmark = pytest.mark.unit


def record(iteration: int, radius: int = 2, order: int = 1, gate: float = 0.5) -> dict:
    """One iterations.jsonl row: per shell (four), the mean's gate and the loadings' gate (half of it)."""
    return {
        "iteration": iteration,
        "radius": radius,
        "healpix_order": order,
        "gates": [[gate, gate / 2]] * 4,
        "fudge": 1.0,
        "step": 0.9,
        "ignored_field": 1,
    }


def write_jsonl(path, rows: list[dict], tail: str = "") -> None:
    """Write ``rows`` one per line, then ``tail`` (a possibly incomplete last line)."""
    path.write_text("".join(json.dumps(row) + "\n" for row in rows) + tail)


def test_only_the_plotted_fields_are_kept(tmp_path):
    write_jsonl(tmp_path / "iterations.jsonl", [record(1), record(2)])

    records = gates.read_records(tmp_path / "iterations.jsonl")

    assert [r["iteration"] for r in records] == [1, 2]
    assert set(records[0]) == set(gates.KEYS)


def test_an_incomplete_last_line_is_dropped_only_when_the_run_is_in_progress(tmp_path):
    path = tmp_path / "iterations.jsonl"
    write_jsonl(path, [record(1), record(2)], tail='{"iteration": 3, "rad')

    assert [r["iteration"] for r in gates.read_records(path, allow_partial_last_line=True)] == [1, 2]
    with pytest.raises(json.JSONDecodeError):
        gates.read_records(path)


def test_an_unparsable_line_before_the_last_is_an_error_even_in_progress(tmp_path):
    path = tmp_path / "iterations.jsonl"
    path.write_text(json.dumps(record(1)) + "\nnot json\n" + json.dumps(record(3)) + "\n")

    with pytest.raises(json.JSONDecodeError):
        gates.read_records(path, allow_partial_last_line=True)


def test_a_run_without_gates_is_rejected_with_the_reason(tmp_path):
    row = record(1)
    del row["gates"]
    write_jsonl(tmp_path / "iterations.jsonl", [row])

    with pytest.raises(ValueError, match="only --optimizer vdam"):
        gates.read_records(tmp_path / "iterations.jsonl")


def test_iteration_one_starts_a_run_and_a_repeated_iteration_supersedes_what_followed_it():
    records = [record(i) for i in (1, 2, 3, 4)] + [record(3, gate=0.9), record(4, gate=0.9)] + [record(1), record(2)]

    runs = gates.split_runs(records)

    assert [[r["iteration"] for r in run] for run in runs] == [[1, 2, 3, 4], [1, 2]]
    assert [r["gates"][0][0] for r in runs[0]] == [0.5, 0.5, 0.9, 0.9]  # the resumed 3 and 4 replaced the first ones


def test_gate_images_are_nan_beyond_each_updates_radius():
    run = [record(1, radius=1, gate=0.8), record(2, radius=3, gate=0.6)]

    images = gates.gate_images(run)  # (channel, shell, iteration)

    assert images.shape == (2, 4, 2)
    np.testing.assert_array_equal(np.isnan(images[0, :, 0]), [False, False, True, True])
    np.testing.assert_array_equal(np.isnan(images[0, :, 1]), [False, False, False, False])
    assert images[0, 0, 0] == pytest.approx(0.8) and images[1, 0, 1] == pytest.approx(0.3)


def test_stage_starts_mark_a_change_of_radius_or_order_only():
    run = [record(1), record(2), record(3, radius=3), record(4, radius=3), record(5, radius=3, order=2)]

    assert gates.stage_starts(run) == [(3, "r=3, HP1"), (5, "r=3, HP2")]
