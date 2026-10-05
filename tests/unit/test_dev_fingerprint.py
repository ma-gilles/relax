"""Pure logic of the controller fingerprint tool (``scripts/dev/fingerprint.py``); no refinement runs here."""

import json
from dataclasses import dataclass

import numpy as np
import pytest

from scripts.dev import fingerprint


@dataclass
class _Pair:
    values: object
    order: int | None


def test_flatten_separates_value_dtype_shape_container_and_placement():
    base = fingerprint.flatten({"a": np.arange(4, dtype=np.float32), "b": [1.5, None], "c": _Pair(np.zeros(2), 3)})

    assert base["/b#"] == "list[2]"
    assert base["/b[0]"] == "float:" + (1.5).hex()
    assert base["/c<_Pair>/order"] == "3"
    assert base["/a"].startswith("numpy float32 (4,) sha256:")
    changed_value = fingerprint.flatten({"a": np.array([0, 1, 2, 4], dtype=np.float32)})
    changed_dtype = fingerprint.flatten({"a": np.arange(4, dtype=np.float64)})
    changed_shape = fingerprint.flatten({"a": np.arange(4, dtype=np.float32).reshape(2, 2)})
    assert len({base["/a"], changed_value["/a"], changed_dtype["/a"], changed_shape["/a"]}) == 4
    assert fingerprint.flatten({"b": (1.5, None)})["/b#"] == "tuple[2]"
    assert fingerprint.flatten([])["#"] == "list[0]"


def test_flatten_drops_wall_times_and_scrubs_strings():
    flat = fingerprint.flatten(
        {"wall_times": [1.0], "path": "/scratch/run/tmpabc/out.npz"},
        scrub=lambda text: text.replace("/scratch/run/tmpabc", fingerprint.TMP_TOKEN),
    )

    assert not any("wall_times" in key for key in flat)
    assert flat["/path"] == repr("<TMP>/out.npz")


def test_flatten_reads_object_arrays_element_by_element():
    flat = fingerprint.flatten(np.array([np.zeros(2), None], dtype=object))

    assert flat["#"] == "list[2]"
    assert flat["[0]"].startswith("numpy float64 (2,)")
    assert flat["[1]"] == "None"


def test_operand_digest_changes_with_any_operand():
    operands = dict(means=np.ones(3, dtype=np.float32), prior=_Pair(np.arange(3.0), 2), step=1.5, label="x")
    seed = fingerprint.digest_operands(operands)

    assert seed == fingerprint.digest_operands(dict(reversed(operands.items())))
    assert seed != fingerprint.digest_operands(dict(operands, means=np.ones(3, dtype=np.float64)))
    assert seed != fingerprint.digest_operands(dict(operands, prior=_Pair(np.arange(3.0), None)))
    assert seed != fingerprint.digest_operands(dict(operands, step=1.25))
    assert seed != fingerprint.digest_operands(dict(operands, extra=None))


def test_log_row_keeps_the_template_and_drops_wall_times():
    plain = fingerprint.log_row("relax.a", "INFO", "order %d", "order 2")
    timed = fingerprint.log_row("relax.a", "INFO", "done in %.2fs", "done in 0.31s")
    node = fingerprint.log_row("relax.b", "INFO", "batch planner memory inputs: %s", "free=3 GB")

    assert plain == ["log", "relax.a", "INFO", "order %d", "order 2"]
    assert timed == ["log", "relax.a", "INFO", "done in %.2fs", ""]
    assert node[-1] == ""


def test_trace_order_does_not_depend_on_which_worker_logged_first():
    def rows(worker_order):
        return (
            [("MainThread", ["log", "before"])]
            + [(name, ["log", f"{name} step {step}"]) for step, name in worker_order]
            + [("MainThread", ["log", "after"])]
        )

    one = fingerprint.ordered_trace(rows([(0, "em-half-0"), (0, "em-half-1"), (1, "em-half-0"), (1, "em-half-1")]))
    other = fingerprint.ordered_trace(rows([(0, "em-half-1"), (0, "em-half-0"), (1, "em-half-1"), (1, "em-half-0")]))

    assert one == other
    assert [row[1] for row in one] == [
        "before", "em-half-0 step 0", "em-half-0 step 1", "em-half-1 step 0", "em-half-1 step 1", "after",
    ]


def _fingerprint(**cases):
    return {"source": "tree", "cases": cases}


def _case(result, trace=()):
    return {"status": {"error": "None"}, "result": dict(result), "files": {}, "checkpoints": {}, "trace": list(trace)}


def test_equal_fingerprints_have_no_difference():
    case = _case({"/mean": "numpy float32 (2,) sha256:aa"}, [["log", "relax.a", "INFO", "x", "x"]])

    counts, lines = fingerprint.diff_fingerprints(_fingerprint(k1=case), _fingerprint(k1=case))

    assert counts == {"outputs": 0, "trace": 0, "log": 0}
    assert fingerprint.accepted(counts)
    assert lines[-1].startswith("1 cases compared; 0 differences")
    assert fingerprint.differing_cases(_fingerprint(k1=case), _fingerprint(k1=case)) == []


def test_every_kind_of_difference_is_counted_and_named():
    a = _fingerprint(
        k1=_case({"/mean": "aa", "/gone": "1"}, [["log", "relax.a", "INFO", "x", "x"], ["call", "engine", "seed=1"]]),
        k2=_case({"/mean": "bb"}),
        only_a=_case({}),
    )
    b = _fingerprint(
        k1=_case({"/mean": "ab", "/new": "2"}, [["log", "relax.moved", "INFO", "x", "x"], ["call", "engine", "seed=1"]]),
        k2=_case({"/mean": "bb"}),
    )

    counts, lines = fingerprint.diff_fingerprints(a, b)
    report = "\n".join(lines)

    # one missing case, three result keys, and the removed and added trace row of the renamed logger
    assert counts == {"outputs": 1 + 3, "trace": 0, "log": 2}
    assert not fingerprint.accepted(counts)
    assert "only log rows differ" not in report
    assert "CASE only_a: only in A" in report
    assert "DIFF k1 result /mean: aa != ab" in report
    assert "DIFF k1 result /gone: 1 != <absent>" in report
    assert "DIFF k1 result /new: <absent> != 2" in report
    assert "TRACE [1 cases, e.g. k1] -log | relax.a | INFO | x | x" in report
    assert not any(line.startswith("DIFF k2") for line in lines)
    assert fingerprint.differing_cases(a, b) == ["k1", "only_a"]


def test_a_mutation_matches_through_reindentation_and_replaces_line_for_line():
    source = "def f():\n    if ready:\n        value = build(\n            a,\n        )\n    return [tau2, tau2]\n"

    mutated, replaced = fingerprint.apply_mutation(source, "value = build(\na,", "value = build(\na[::-1],")

    assert replaced == 1
    assert mutated == source.replace("            a,\n", "            a[::-1],\n")
    assert fingerprint.apply_mutation(source, "return [tau2, tau2]", "return [tau2, tau2 * 2]")[0].endswith(
        "    return [tau2, tau2 * 2]\n"
    )
    assert fingerprint.apply_mutation(source, "absent()", "pass") == (source, 0)
    with pytest.raises(ValueError, match="line for line"):
        fingerprint.apply_mutation(source, "a,\n)", "b")


def test_the_case_and_mutation_tables_are_well_formed():
    for name, (description, keywords) in fingerprint.CASES.items():
        assert description
        assert name.startswith(f"k{keywords['n_classes']}_")
    assert {keywords["n_classes"] for _, keywords in fingerprint.CASES.values()} == {1, 2, 4}
    names = [mutation[0] for mutation in fingerprint.MUTATIONS]
    assert len(names) == len(set(names))
    for _, old, new, description, detected in fingerprint.MUTATIONS:
        assert len(old.splitlines()) == len(new.splitlines())
        assert old != new and description and isinstance(detected, bool)


def test_scratch_inside_the_checkout_is_refused():
    inside = fingerprint.REPO_ROOT / "scripts" / "dev" / "fingerprint_scratch"

    with pytest.raises(SystemExit, match="inside the checkout"):
        fingerprint._work_dir(type("Args", (), {"work_dir": str(inside)})())
    assert not inside.exists()


_LOG = ["log", "relax.a", "INFO", "x", "x"]
_MOVED_LOG = ["log", "relax.a", "INFO", "y", "y"]
_CALL = ["call", "engine", "seed=1"]


def _write_pair(tmp_path, a, b):
    paths = tmp_path / "a.json", tmp_path / "b.json"
    for path, value in zip(paths, (a, b), strict=True):
        path.write_text(json.dumps(value))
    return [str(path) for path in paths]


@pytest.mark.parametrize(
    ("case_b", "exit_code", "accepted_line"),
    [
        # a log record moved past a call: only log rows differ, accepted under rule 2
        (_case({"/mean": "aa"}, [_CALL, _LOG, _MOVED_LOG]), 0, "only log rows differ (2); accepted under rule 2"),
        (_case({"/mean": "ab"}, [_LOG, _CALL, _MOVED_LOG]), 1, None),
        (_case({"/mean": "aa"}, [_LOG, ["call", "engine", "seed=2"], _MOVED_LOG]), 1, None),
    ],
    ids=["log-only", "one-result", "one-call-row"],
)
def test_diff_accepts_only_a_log_only_difference(tmp_path, capsys, case_b, exit_code, accepted_line):
    a = _fingerprint(k1=_case({"/mean": "aa"}, [_LOG, _CALL, _MOVED_LOG]))
    paths = _write_pair(tmp_path, a, _fingerprint(k1=case_b))

    assert fingerprint.main(["diff", *paths]) == exit_code
    out = capsys.readouterr().out
    if accepted_line:
        assert accepted_line in out
    else:
        assert "only log rows differ" not in out
