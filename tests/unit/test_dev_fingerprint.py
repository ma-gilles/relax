"""Pure logic of the controller fingerprint tool (``scripts/dev/fingerprint.py``); no refinement runs here."""

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from scripts.dev import fingerprint, fingerprint_cli


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


def test_written_files_drop_only_the_named_timing_and_host_keys():
    """G10: archives and JSON files drop keys by exact name, the ones measured to vary between two runs of one tree
    (wall times, git state, host memory); a key that only contains "time", "git" or "memory" is compared."""
    from scripts.dev import ppca_fingerprint, vdam_fingerprint

    assert fingerprint.DROPPED_ARCHIVE_KEYS == {"wall_times", "total_time", "setup_phase_cumulative_s"}
    assert fingerprint.DROPPED_LEDGER_KEYS == {
        "git_provenance", "git_commit", "setup_phase_seconds", "total_time_s", "wall_times_trajectory",
    }
    assert vdam_fingerprint.TIMING_KEYS == {
        f"{stage}_time_s" for stage in
        ("expectation", "mstep", "pre_artifact", "projector_refresh", "schedule", "state_update", "subset")
    }
    assert ppca_fingerprint.HOST_KEYS == {"elapsed_seconds", "memory_gb", "memory_fraction"}
    compared = ("timing_dir", "n_wall_times", "timing_rows", "timing_summary", "jax_persistent_cache_min_compile_time_secs",
                "runtime_s", "digit")
    for table in (fingerprint.DROPPED_ARCHIVE_KEYS, fingerprint.DROPPED_LEDGER_KEYS, vdam_fingerprint.TIMING_KEYS,
                  ppca_fingerprint.HOST_KEYS):
        assert not table.intersection(compared)


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


def test_engine_tuning_is_digested_by_its_fields_that_differ_from_the_default():
    """G9c: an absent EngineTuning and a default one digest the same; a field set off its default is an operand."""
    import dataclasses

    from relax.runtime.engine_tuning import EngineTuning

    default = EngineTuning()
    changed = dataclasses.replace(default, prepare_image_batch=default.prepare_image_batch + 1)
    operands = {"current_size": 8}
    assert fingerprint.without_probe({**operands, "engine_tuning": default}) == operands
    assert fingerprint.without_probe({**operands, "engine_tuning": changed}) == {
        **operands, "engine_tuning": {"prepare_image_batch": default.prepare_image_batch + 1}
    }
    seed = fingerprint.digest_operands(operands)
    assert fingerprint.digest_operands({**operands, "engine_tuning": default}) == seed
    assert fingerprint.digest_operands(fingerprint.without_probe({**operands, "engine_tuning": default})) == seed
    assert fingerprint.digest_operands({**operands, "engine_tuning": changed}) != seed
    assert fingerprint.digest_operands(fingerprint.without_probe({**operands, "engine_tuning": changed})) != seed
    with pytest.raises(AssertionError, match="not an EngineTuning"):
        fingerprint.without_probe({"engine_tuning": {"prepare_image_batch": 1}})


def test_an_engine_keyword_at_its_default_is_the_same_call_as_an_omitted_one():
    projector = np.ones(3)
    omitted = {"current_size": 8, "relion_projector_half": projector}
    passed = {**omitted, "symmetry_label": "C1", "class_rotation_log_prior": None, "mstep_relion_x_half": False,
              "relion_translation_angle_scale": 1.0, "coarse_current_size": None}

    assert fingerprint.engine_kwargs_without_defaults(passed) == omitted
    # A value off the default, or of another type than the default, stays an operand.
    for key, value in (("symmetry_label", "D2"), ("mstep_relion_x_half", True), ("mstep_relion_x_half", np.bool_(False)),
                       ("relion_translation_angle_scale", 0.5), ("optics_group_ids", np.zeros(2))):
        assert key in fingerprint.engine_kwargs_without_defaults({key: value})
    # A keyword the engine reads without a shared default is never dropped.
    assert fingerprint.engine_kwargs_without_defaults({"pass2_use_float64_scoring": None}) == {
        "pass2_use_float64_scoring": None
    }


def _engine_modules() -> list[Path]:
    """The relax modules ``relax.classification.k_class`` imports, directly or through each other (function-level
    imports included): the engine ``run_dense_k_class_em_adaptive`` and every module its callees live in."""

    root = Path(fingerprint_cli.REPO_ROOT)

    def module_path(name: str) -> Path | None:
        path = root / Path(*name.split("."))
        for candidate in (path.with_suffix(".py"), path / "__init__.py"):
            if candidate.is_file():
                return candidate
        return None

    seen, todo = set(), ["relax.classification.k_class"]
    while todo:
        name = todo.pop()
        path = module_path(name)
        if name in seen or path is None:
            continue
        seen.add(name)
        package = name if path.name == "__init__.py" else name.rpartition(".")[0]
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                todo += [alias.name for alias in node.names if alias.name.startswith("relax")]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    base = ".".join(package.split(".")[: len(package.split(".")) - node.level + 1] + ([base] if base else []))
                if base.startswith("relax"):
                    todo += [base, *(f"{base}.{alias.name}" for alias in node.names)]
    return sorted(module_path(name) for name in seen)


def _engine_keyword_defaults(paths: list[Path], keys) -> dict[str, set[str]]:
    """Per key, the defaults the engine modules read it with: a named parameter's default, or the default of a
    ``<...kwargs>.get(key[, default])`` (None when omitted), as source text."""

    found = {key: set() for key in keys}
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = node.args
                positional = args.posonlyargs + args.args
                pairs = [*zip(positional[len(positional) - len(args.defaults):], args.defaults),
                         *zip(args.kwonlyargs, args.kw_defaults)]
                for arg, default in pairs:
                    if default is not None and arg.arg in found:
                        found[arg.arg].add(ast.unparse(default))
            elif (
                isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get"
                and ast.unparse(node.func.value).endswith("kwargs")
                and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value in found
            ):
                found[node.args[0].value].add(ast.unparse(node.args[1]) if len(node.args) > 1 else "None")
    return found


def test_engine_keyword_defaults_match_the_engine():
    """Every key of ENGINE_KWARG_DEFAULTS is read by the engine or a callee, and only at the table's default: a
    default changed in relax while callers pass the old one would otherwise leave the digest on both sides (G9b)."""

    paths = _engine_modules()
    assert Path(fingerprint_cli.REPO_ROOT, "relax", "classification", "k_class.py") in paths
    found = _engine_keyword_defaults(paths, fingerprint.ENGINE_KWARG_DEFAULTS)
    for key, default in fingerprint.ENGINE_KWARG_DEFAULTS.items():
        assert found[key], f"{key}: no engine module reads it with a default; delete it from ENGINE_KWARG_DEFAULTS"
        assert {ast.literal_eval(text) for text in found[key]} == {default}, (
            f"{key}: the table says {default!r}, the engine modules read {sorted(found[key])}"
        )


def test_log_row_keeps_the_template_and_drops_wall_times():
    plain = fingerprint.log_row("relax.a", "INFO", "order %d", "order 2")
    timed = fingerprint.log_row("relax.a", "INFO", "done in %.2fs", "done in 0.31s")
    node = fingerprint.log_row("relax.b", "INFO", "batch planner memory inputs: %s", "free=3 GB")

    assert plain == ["log", "relax.a", "INFO", "order %d", "order 2"]
    assert timed == ["log", "relax.a", "INFO", "done in %.2fs", ""]
    assert node[-1] == ""


def test_trace_order_does_not_depend_on_when_a_background_thread_logged():
    def rows(worker_order):
        return (
            [("MainThread", ["log", "before"])]
            + [(name, ["log", f"{name} step {step}"]) for step, name in worker_order]
            + [("MainThread", ["log", "after"])]
        )

    one = fingerprint.ordered_trace(rows([(0, "writer-b"), (0, "writer-a"), (1, "writer-b"), (1, "writer-a")]))
    other = fingerprint.ordered_trace(rows([(0, "writer-a"), (1, "writer-a"), (0, "writer-b"), (1, "writer-b")]))

    assert one == other
    assert [row[1] for row in one] == [
        "before", "after", "writer-a step 0", "writer-a step 1", "writer-b step 0", "writer-b step 1",
    ]


def _fingerprint(**cases):
    return {"source": "tree", "cases": cases}


def _counts(**nonzero):
    """A diff's counts: every class of DIFFERENCE_CLASSES and ``unaccepted``, zero unless given."""
    return {**dict.fromkeys((*fingerprint.DIFFERENCE_CLASSES, "unaccepted"), 0), **nonzero}


def _accept(pattern, change):
    """An ACCEPTED_CHANGES table of one entry."""
    return ((pattern, change, "the change under test"),)


def _case(result, trace=()):
    return {"status": {"error": "None"}, "result": dict(result), "files": {}, "checkpoints": {}, "trace": list(trace)}


def test_equal_fingerprints_have_no_difference():
    case = _case({"/mean": "numpy float32 (2,) sha256:aa"}, [["log", "relax.a", "INFO", "x", "x"]])

    counts, lines = fingerprint.diff_fingerprints(_fingerprint(k1=case), _fingerprint(k1=case))

    assert counts == _counts()
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
    assert counts == _counts(outputs=1 + 3, log=2)
    assert not fingerprint.accepted(counts)
    assert "only log rows differ" not in report
    assert "CASE only_a: only in A" in report
    assert "DIFF k1 result /mean: aa != ab" in report
    assert "DIFF k1 result /gone: 1 != <absent>" in report
    assert "DIFF k1 result /new: <absent> != 2" in report
    assert "TRACE [1 cases, e.g. k1] -log | relax.a | INFO | x | x" in report
    assert not any(line.startswith("DIFF k2") for line in lines)
    assert fingerprint.differing_cases(a, b) == ["k1", "only_a"]


def test_the_change_classes_pass_only_through_an_accepted_changes_entry():
    """G9a: an added, retired, moved or redefaulted leaf fails unless an ACCEPTED_CHANGES entry of its class names it,
    and the report names each such leaf; the tables start empty."""
    assert fingerprint.ACCEPTED_CHANGES == ()
    from scripts.dev import pass1_fingerprint, ppca_fingerprint, vdam_fingerprint

    for harness in (vdam_fingerprint, ppca_fingerprint, pass1_fingerprint):
        assert harness.ACCEPTED_CHANGES == ()
    for table in (fingerprint.ACCEPTED_CHANGES, vdam_fingerprint.ACCEPTED_CHANGES, ppca_fingerprint.ACCEPTED_CHANGES,
                  pass1_fingerprint.ACCEPTED_CHANGES):
        for pattern, change, reason in table:
            assert re.compile(pattern) and change in fingerprint.CHANGE_CLASSES and reason.strip()
    key = "refine0/inputs[1]/options/debug/new"
    a = _fingerprint(k1=_case({"/mean": "aa"}), k2=_case({"/mean": "aa"}))
    b = _fingerprint(k1=_case({key: "None", "/mean": "aa"}), k2=_case({key: "None", "/mean": "aa"}))
    counts, lines = fingerprint.diff_fingerprints(a, b)
    assert counts == _counts(added=2, unaccepted=2) and not fingerprint.accepted(counts)
    assert f"UNACCEPTED added result {key} [2 cases, e.g. k1]: no ACCEPTED_CHANGES entry" in lines
    assert lines[-1] == "controller inputs added (2); NOT accepted: 2 match no ACCEPTED_CHANGES entry: new option fields"
    # An entry of another class, or for another path, does not accept it.
    for table in (_accept("debug/new$", "retired"), _accept("debug/other$", "added")):
        assert not fingerprint.accepted(fingerprint.diff_fingerprints(a, b, accepted_changes=table)[0])


def test_a_new_controller_input_is_listed_and_accepted_but_a_new_result_is_not():
    a = _fingerprint(k1=_case({"refine0/inputs[1]/options/debug/x": "1", "/mean": "aa"}))
    b = _fingerprint(k1=_case({"refine0/inputs[1]/options/debug/x": "1", "refine0/inputs[1]/options/debug/new": "None",
                               "/mean": "aa"}))
    counts, lines = fingerprint.diff_fingerprints(a, b, accepted_changes=_accept(r"/debug/new$", "added"))
    assert counts == _counts(added=1) and fingerprint.accepted(counts)
    assert "ADDED k1 result refine0/inputs[1]/options/debug/new: None" in lines
    assert lines[-1] == "controller inputs added (1); accepted by ACCEPTED_CHANGES: new option fields"
    # A removed or changed input, or a new leaf anywhere else, is an output difference.
    for flat in ({"/mean": "aa"}, {"refine0/inputs[1]/options/debug/x": "2", "/mean": "aa"},
                 {"refine0/inputs[1]/options/debug/x": "1", "/mean": "aa", "/new": "1"}):
        counts, _ = fingerprint.diff_fingerprints(a, _fingerprint(k1=_case(flat)))
        assert counts["outputs"] == 1 and not fingerprint.accepted(counts)


def test_a_controller_input_retired_at_its_declared_default_is_listed_and_accepted_but_one_that_was_on_is_not():
    a = _fingerprint(k1=_case({"refine0/inputs[1]/options/debug/x": "1",
                               "refine0/inputs[1]/options/debug/off": "None =default",
                               "refine0/inputs[1]/options/debug/flag": "False =default", "/mean": "aa"}))
    b = _fingerprint(k1=_case({"refine0/inputs[1]/options/debug/x": "1", "/mean": "aa"}))
    retiring = _accept(r"/debug/(off|flag)$", "retired")
    counts, lines = fingerprint.diff_fingerprints(a, b, accepted_changes=retiring)
    assert counts == _counts(retired=2) and fingerprint.accepted(counts)
    assert "RETIRED k1 result refine0/inputs[1]/options/debug/off: None =default" in lines
    assert lines[-1] == (
        "controller inputs retired (2); accepted by ACCEPTED_CHANGES: option fields removed at their declared default"
    )
    # A plain None or False (no =default mark) is no longer an off value: its removal is an output difference.
    plain = _fingerprint(k1=_case({"refine0/inputs[1]/options/debug/x": "1", "refine0/inputs[1]/options/debug/off": "None",
                                   "refine0/inputs[1]/options/debug/flag": "False", "/mean": "aa"}))
    counts, _ = fingerprint.diff_fingerprints(plain, b, accepted_changes=retiring)
    assert counts == _counts(outputs=2) and not fingerprint.accepted(counts)
    # The keyword mapping's length changes with its members, and is accepted with them.
    with_length = dict(a["cases"]["k1"]["result"], **{"refine0/inputs[1]#": "dict[3]"})
    counts, _ = fingerprint.diff_fingerprints(
        _fingerprint(k1=_case(with_length)),
        _fingerprint(k1=_case({"refine0/inputs[1]#": "dict[2]", "refine0/inputs[1]/options/debug/x": "1",
                               "refine0/inputs[1]/observer": "None", "/mean": "aa"})),
        accepted_changes=(*retiring, *_accept(r"/observer$", "added")),
    )
    assert counts == _counts(added=1, retired=2)
    # A removed input that was on, or a removed result leaf, is an output difference.
    for flat in ({"/mean": "aa"}, {"refine0/inputs[1]/options/debug/x": "1"}):
        counts, _ = fingerprint.diff_fingerprints(
            _fingerprint(k1=_case({"refine0/inputs[1]/options/debug/x": "1", "/mean": "aa"})), _fingerprint(k1=_case(flat))
        )
        assert counts["outputs"] == 1 and not fingerprint.accepted(counts)


def test_a_controller_input_moved_to_another_group_with_its_value_is_listed_and_accepted():
    old, new = fingerprint.MOVED_INPUTS[0]
    moved_a = f"refine0/inputs[1]/options<RefinementOptions>{old}X>/order"
    moved_b = moved_a.replace(old, new)
    a = _fingerprint(k1=_case({moved_a: "numpy int64 (6,) sha256:aa", "/mean": "aa"}))
    b = _fingerprint(k1=_case({moved_b: "numpy int64 (6,) sha256:aa", "/mean": "aa"}))
    counts, lines = fingerprint.diff_fingerprints(a, b)
    assert counts == _counts(moved=1, unaccepted=1) and not fingerprint.accepted(counts)
    assert f"UNACCEPTED moved result {moved_a} -> {moved_b} [1 cases, e.g. k1]: no ACCEPTED_CHANGES entry" in lines
    counts, lines = fingerprint.diff_fingerprints(a, b, accepted_changes=_accept(re.escape(new), "moved"))
    assert counts == _counts(moved=1)
    assert fingerprint.accepted(counts)
    assert f"MOVED k1 result {moved_a} -> {moved_b}" in lines
    assert lines[-1] == (
        "controller inputs moved (1); accepted by ACCEPTED_CHANGES: option fields moved or record fields renamed "
        "(MOVED_INPUTS, RENAMED_FIELDS)"
    )
    # The same value whose field's default changed in the move (only the =default mark differs) is moved too.
    b = _fingerprint(k1=_case({moved_b: "numpy int64 (6,) sha256:aa =default", "/mean": "aa"}))
    assert fingerprint.diff_fingerprints(a, b)[0]["moved"] == 1
    # A moved input whose value changed is an output difference.
    b = _fingerprint(k1=_case({moved_b: "numpy int64 (6,) sha256:bb", "/mean": "aa"}))
    counts, _ = fingerprint.diff_fingerprints(a, b)
    assert counts["moved"] == 0 and counts["outputs"] == 1 and not fingerprint.accepted(counts)


def test_a_renamed_record_field_with_its_value_is_accepted_in_any_section():
    old, new = fingerprint.RENAMED_FIELDS[0]
    a_key, b_key = f"[0]/{old}", f"[0]/{new}"
    a = _fingerprint(k1=dict(_case({"/mean": "aa"}), checkpoints={a_key: "'8'"}))
    b = _fingerprint(k1=dict(_case({"/mean": "aa"}), checkpoints={b_key: "'8'"}))
    counts, _ = fingerprint.diff_fingerprints(a, b, accepted_changes=_accept("box_size$", "moved"))
    assert counts["moved"] == 1 and counts["outputs"] == 0 and fingerprint.accepted(counts)
    b = _fingerprint(k1=dict(_case({"/mean": "aa"}), checkpoints={b_key: "'9'"}))
    counts, _ = fingerprint.diff_fingerprints(a, b)
    assert counts["moved"] == 0 and counts["outputs"] == 2 and not fingerprint.accepted(counts)


def test_a_controller_input_whose_default_changed_is_listed_and_accepted_but_a_changed_value_is_not():
    key = "refine0/inputs[1]/options<RefinementOptions>/parity<P>/perturb_factor"
    a = _fingerprint(k1=_case({key: "float:0x1.0000000000000p-1", "/mean": "aa"}))
    b = _fingerprint(k1=_case({key: "float:0x1.0000000000000p-1 =default", "/mean": "aa"}))
    counts, lines = fingerprint.diff_fingerprints(a, b, accepted_changes=_accept("perturb_factor$", "redefaulted"))
    assert counts == _counts(redefaulted=1)
    assert fingerprint.accepted(counts)
    assert f"REDEFAULTED k1 result {key}: float:0x1.0000000000000p-1 -> float:0x1.0000000000000p-1 =default" in lines
    assert lines[-1].startswith("controller inputs redefaulted (1); accepted")
    # A changed value, or the same mark change on a result leaf, is an output difference.
    for flat_a, flat_b in (({key: "1"}, {key: "2 =default"}), ({"/mean": "1"}, {"/mean": "1 =default"})):
        counts, _ = fingerprint.diff_fingerprints(_fingerprint(k1=_case(flat_a)), _fingerprint(k1=_case(flat_b)))
        assert counts["redefaulted"] == 0 and counts["outputs"] == 1 and not fingerprint.accepted(counts)


def test_a_checkpoint_field_whose_default_was_removed_is_listed_and_accepted_but_a_changed_value_is_not():
    key = "[0]/<IterationSnapshot>/class_weights"
    a = _fingerprint(k1=dict(_case({"/mean": "aa"}), checkpoints={key: "None =default"}))
    b = _fingerprint(k1=dict(_case({"/mean": "aa"}), checkpoints={key: "None"}))
    accepting = _accept("class_weights$", "redefaulted")
    counts, lines = fingerprint.diff_fingerprints(a, b, accepted_changes=accepting)
    assert counts == _counts(redefaulted=1)
    assert fingerprint.accepted(counts)
    assert f"REDEFAULTED k1 checkpoints {key}: None =default -> None" in lines
    # The same for a checkpoint leaf recorded as the repr of its flattened leaf.
    a = _fingerprint(k1=dict(_case({"/mean": "aa"}), checkpoints={key: "'None =default'"}))
    b = _fingerprint(k1=dict(_case({"/mean": "aa"}), checkpoints={key: "'None'"}))
    counts, _ = fingerprint.diff_fingerprints(a, b, accepted_changes=accepting)
    assert counts["redefaulted"] == 1 and counts["outputs"] == 0 and fingerprint.accepted(counts)
    # A changed checkpoint value is still an output difference, with or without the mark.
    b = _fingerprint(k1=dict(_case({"/mean": "aa"}), checkpoints={key: "numpy float64 (2,) sha256:aa"}))
    counts, _ = fingerprint.diff_fingerprints(a, b)
    assert counts["redefaulted"] == 0 and counts["outputs"] == 1 and not fingerprint.accepted(counts)


def test_a_dataclass_field_at_its_declared_default_is_marked_and_may_be_retired():
    import dataclasses

    @dataclasses.dataclass
    class Options:
        prefix: str = "run"
        dims: tuple = ()
        count: int = 3

    leaves = fingerprint.flatten({"o": Options(dims=(1,))}, "refine0/inputs[1]")
    assert leaves["refine0/inputs[1]/o<Options>/prefix"].endswith(fingerprint.DEFAULT_MARK)
    assert not leaves["refine0/inputs[1]/o<Options>/dims#"].endswith(fingerprint.DEFAULT_MARK)
    a = _fingerprint(k1=_case({**leaves, "/mean": "aa"}))
    b = _fingerprint(k1=_case({key: value for key, value in {**leaves, "/mean": "aa"}.items() if "/prefix" not in key}))
    assert fingerprint.diff_fingerprints(a, b)[0]["retired"] == 1
    b = _fingerprint(k1=_case({key: value for key, value in {**leaves, "/mean": "aa"}.items() if "/dims" not in key}))
    assert fingerprint.diff_fingerprints(a, b)[0]["outputs"] >= 1


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
    controller_cases = {name: case for name, case in fingerprint.CASES.items() if "main" not in case[1]}
    for name, (description, keywords) in controller_cases.items():
        assert description
        assert name.startswith(f"k{keywords['n_classes']}_")
    assert {keywords["n_classes"] for _, keywords in controller_cases.values()} == {1, 2, 4}
    main_cases = {name: case[1]["main"] for name, case in fingerprint.CASES.items() if "main" in case[1]}
    assert main_cases and all(name.startswith("main_") for name in main_cases)
    assert {case["command"] for case in main_cases.values()} == {"refine", "class3d"}
    names = [mutation[0] for mutation in fingerprint.MUTATIONS]
    assert len(names) == len(set(names))
    for _, old, new, description in fingerprint.MUTATIONS:
        assert len(old.splitlines()) == len(new.splitlines())
        assert old != new and description


def test_only_the_three_known_blind_spots_are_expected_blind():
    """G13: the refinement harness lists exactly its three undetected mutations and pass1 its one, each with a
    reason; vdam and ppca list none (k1_mstep_join_radius is detected and is not listed)."""
    from scripts.dev import pass1_fingerprint, ppca_fingerprint, vdam_fingerprint

    assert set(fingerprint.EXPECTED_BLIND) == {
        "next_sampling_star_skipped", "previous_data_vs_prior_doubled", "star_prior_translations_reversed",
    }
    assert all(reason.strip() for reason in fingerprint.EXPECTED_BLIND.values())
    assert set(fingerprint.EXPECTED_BLIND) <= {mutation[0] for mutation in fingerprint.MUTATIONS}
    assert set(pass1_fingerprint.EXPECTED_BLIND) == {"tree_margin_bound"}
    for harness in (vdam_fingerprint, ppca_fingerprint, pass1_fingerprint):
        assert all(reason.strip() for reason in harness.EXPECTED_BLIND.values())
        assert set(harness.EXPECTED_BLIND) <= {mutation[0] for mutation in harness.MUTATIONS}
        assert harness.HARNESS.expected_blind is harness.EXPECTED_BLIND
    assert vdam_fingerprint.EXPECTED_BLIND == {} and ppca_fingerprint.EXPECTED_BLIND == {}
    assert fingerprint.HARNESS.expected_blind is fingerprint.EXPECTED_BLIND


def test_selftest_fails_an_undetected_unlisted_mutation_and_a_detected_listed_one(tmp_path, monkeypatch, capsys):
    """An unlisted mutation must be detected; a listed one prints BLIND when it is not, and fails as a stale entry
    when it is."""
    import dataclasses

    detected = {"caught", "stale"}

    def run_tree(_harness, _source, out, _work_dir, _names, **_options):
        out.write_text(json.dumps({"mutant": out.stem}))
        return 0

    def differing_cases(_clean, mutant):
        return ["k1"] if any(mutant["mutant"].endswith(name) for name in detected) else []

    monkeypatch.setattr(fingerprint_cli, "run_tree", run_tree)
    mutations = tuple((name, "old", "new", f"{name} breaks") for name in ("caught", "missed", "blind", "stale"))
    harness = dataclasses.replace(
        fingerprint.HARNESS, mutations=mutations, differing_cases=differing_cases,
        mutated_tree=lambda *_args: {"relax/x.py": 1},
        expected_blind={"blind": "no case reaches it", "stale": "no case reaches it"},
    )
    args = fingerprint_cli.argparse.Namespace(work_dir=str(tmp_path), source=str(tmp_path), rev=None, mutations=[], jobs=1)

    assert fingerprint_cli.command_selftest(harness, args) == 1
    out = capsys.readouterr().out
    assert "OK    caught:" in out
    assert "FAIL  missed: missed breaks [relax/x.py] -> not detected (not in EXPECTED_BLIND)" in out
    assert "BLIND blind: blind breaks [relax/x.py] -> not detected, as EXPECTED_BLIND says: no case reaches it" in out
    assert "FAIL  stale:" in out and "listed in EXPECTED_BLIND (a stale entry; delete it)" in out
    assert "4 mutations, 2 failed, 1 blind (EXPECTED_BLIND)" in out
    # A failed mutation's tree and log stay for diagnosis; a passed one's tree is removed.
    assert f"(kept: {tmp_path / (harness.file_prefix + 'mutant_missed')}," in out
    assert not (tmp_path / (harness.file_prefix + "mutant_caught")).exists()
    harness = dataclasses.replace(harness, expected_blind={"gone": "no such mutation"})
    with pytest.raises(SystemExit, match="EXPECTED_BLIND names no mutation: gone"):
        fingerprint_cli.command_selftest(harness, args)


def test_scratch_inside_the_checkout_is_refused():
    inside = fingerprint_cli.REPO_ROOT / "scripts" / "dev" / "fingerprint_scratch"

    with pytest.raises(SystemExit, match="inside the checkout"):
        fingerprint_cli._work_dir(fingerprint.HARNESS, type("Args", (), {"work_dir": str(inside)})())
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


def test_vdam_result_and_checkpoint_fields_whose_default_was_removed_are_redefaulted_but_a_changed_value_is_not():
    from scripts.dev import vdam_fingerprint

    diff = vdam_fingerprint.HARNESS.diff_fingerprints
    key = "<NativeInitialModelResult>/state<InitialModelState>/K"
    checkpoint = "run_it000_model.star<InitialModelState>/K"
    a = _fingerprint(k1=dict(_case({key: "1 =default"}), checkpoints={checkpoint: "1 =default"}))
    b = _fingerprint(k1=dict(_case({key: "1"}), checkpoints={checkpoint: "1"}))
    counts, _ = diff(a, b)
    assert counts == _counts(redefaulted=2, unaccepted=2) and not fingerprint.accepted(counts)
    counts, lines = diff(a, b, accepted_changes=_accept("/K$", "redefaulted"))
    assert counts == _counts(redefaulted=2)
    assert fingerprint.accepted(counts)
    assert f"REDEFAULTED k1 result {key}: 1 =default -> 1" in lines
    assert lines[-1].startswith("controller inputs redefaulted (2); accepted")
    # A changed value is an output difference, with or without the mark; the refinement harness keeps result
    # leaves strict.
    b = _fingerprint(k1=dict(_case({key: "2"}), checkpoints={checkpoint: "2 =default"}))
    counts, _ = diff(a, b)
    assert counts["redefaulted"] == 0 and counts["outputs"] == 2 and not fingerprint.accepted(counts)
    b = _fingerprint(k1=_case({key: "1"}))
    counts, _ = fingerprint.diff_fingerprints(_fingerprint(k1=_case({key: "1 =default"})), b)
    assert counts["redefaulted"] == 0 and counts["outputs"] == 1


def test_a_rev_export_has_every_top_level_package_and_imports_the_command_entry(tmp_path):
    """relax#123: ``export_rev`` exports every top-level package of the rev (``relax_cli`` since da6742f0), so the
    exported tree imports the refinement command entry on its own; an export of ``relax`` and ``tests`` alone did not."""
    import subprocess
    import sys

    export = fingerprint_cli.export_rev("HEAD", tmp_path)

    packages = sorted(path.name for path in fingerprint_cli.REPO_ROOT.iterdir() if (path / "__init__.py").is_file())
    assert {"relax", "relax_cli"} <= set(packages)
    assert all((export / name / "__init__.py").is_file() for name in packages)
    imported = subprocess.run(
        [sys.executable, "-c", "import relax.commands.refinement_run as entry; print(entry.__file__)"],
        cwd=tmp_path, capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "PYTHONPATH": str(export), "PYTHONNOUSERSITE": "1",
             "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""},
    )
    assert imported.returncode == 0, imported.stderr[-2000:]
    assert imported.stdout.strip().startswith(str(export))


def test_a_mutant_tree_links_every_top_level_package_and_imports_the_command_entry(tmp_path):
    """relax#123: the selftest's mutant tree copies ``relax`` (mutated) and links ``tests`` and every other top-level
    package of its source, so a mutant imports ``relax_cli`` and the refinement command entry as its source does."""
    import subprocess
    import sys

    (tmp_path / "work").mkdir()
    source = fingerprint_cli.export_rev("HEAD", tmp_path / "work")
    target = tmp_path / "mutant"
    target.mkdir()
    changed = fingerprint.mutated_tree(source, target, *fingerprint.MUTATIONS[0][1:3])

    assert sum(changed.values()) == 1
    assert not (target / "relax").is_symlink() and (target / "relax_cli").is_symlink() and (target / "tests").is_symlink()
    imported = subprocess.run(
        [sys.executable, "-c", "import relax_cli, relax.commands.refinement_run as entry; print(entry.__file__)"],
        cwd=tmp_path, capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "PYTHONPATH": str(target), "PYTHONNOUSERSITE": "1",
             "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""},
    )
    assert imported.returncode == 0, imported.stderr[-2000:]
    assert imported.stdout.strip().startswith(str(target))
