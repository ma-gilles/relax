#!/usr/bin/env python
"""Classify the lines of the numbered-iteration loop: per mode, and for the shared lines by kind.

    python scripts/dev/loop_kinds.py [FILE] [--calls]    # FILE defaults to relax/refinement/iteration_loop.py

It reads the ``while`` loop of ``refine_single_volume`` and prints how many lines only K=1 or only Class3D
executes (an explicit test of the mode, or a condition only that mode admits), and what the shared lines are:
calls with wide argument lists, installs, history writes, logs, diagnostics, checkpoint, control flow.
``--calls`` also lists each shared wide call with its line and length. Use it to measure what a refactor
slice would remove before writing it (``docs/development/refactor_procedure.md``). The admission markers
below name conditions of this loop; a controller in another module needs its own.
"""

from __future__ import annotations

import argparse
import ast
import collections
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTROLLER = "refine_single_volume"
K1_ADMISSION = (
    "uses_native_auto_refine",
    "state_swap_target_this_iteration",
    "frozen_initial_scoring_state is not None",
    "replay_mean_variance is not None",
    "stop_after_local_search",
)
CLASS_ADMISSION = (
    "follower_setup.follower_scale_state is not None",
    "single_class_iteration",
    "_kclass_dump_dir",
    "k_class_enabled and replay_result.class_weights",
)
CLASS_NAMES = {"seeded_start", "seed_after_cc", "single_class_iteration", "seed_iteration"}
K1_NAMES = {"recovar_state_swap_snapshot", "state_swap_target_this_iteration", "replay_mean_variance"}
DIAGNOSTIC_TESTS = ("save_intermediates_dir", "_parity_dump", "_dump_dir", "checkpoint_writer")
WIDE_CALL = "calls with wide argument lists"


def _lines(node) -> set[int]:
    return set(range(node.lineno, node.end_lineno + 1))


def _span(nodes) -> set[int]:
    return set(range(nodes[0].lineno, nodes[-1].end_lineno + 1)) if nodes else set()


def mode_lines(loop: ast.While) -> tuple[set[int], set[int]]:
    """The loop lines only K=1 executes and the lines only Class3D executes."""
    k1, cl = set(), set()
    for node in ast.walk(loop):
        if isinstance(node, ast.If):
            test = ast.unparse(node.test)
            if test == "k_class_enabled":
                cl |= _span(node.body)
                k1 |= _span(node.orelse)
            elif test == "not k_class_enabled":
                k1 |= _span(node.body) | {node.lineno}
                cl |= _span(node.orelse)
            elif test == "use_local":
                k1 |= _span(node.body) | {node.lineno}
            elif test == "follower_setup.follower_scale_state is None":
                cl |= _span(node.orelse)
            elif any(marker in test for marker in K1_ADMISSION):
                k1 |= _lines(node)
            elif any(marker in test for marker in CLASS_ADMISSION) or test == "use_local and k_class_enabled":
                cl |= _lines(node)
        elif isinstance(node, ast.Assign):
            names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if names & CLASS_NAMES:
                cl |= _lines(node)
            is_probe = isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "_apply_state_swap_probe"
            if names & K1_NAMES or is_probe:
                k1 |= _lines(node)
    body = set(range(loop.lineno + 1, loop.end_lineno + 1))
    k1 &= body
    cl &= body
    return k1, cl - k1


def _statement_kind(stmt, diag) -> str:
    is_call = isinstance(stmt, (ast.Assign, ast.Expr)) and isinstance(stmt.value, ast.Call)
    func = ast.unparse(stmt.value.func) if is_call else ""
    n_lines = stmt.end_lineno - stmt.lineno + 1
    if diag == "checkpoint":
        return "checkpoint"
    if diag:
        return "diagnostics"
    if func.startswith("logger."):
        return "logs"
    if func.startswith("history.") or (isinstance(stmt, ast.Assign) and "history." in ast.unparse(stmt.targets[0])):
        return "history writes"
    if func.startswith("_parity_dump.") or "os.environ" in ast.unparse(stmt):
        return "diagnostics"
    if isinstance(stmt, (ast.Raise, ast.Break, ast.Delete, ast.Pass, ast.Import, ast.ImportFrom)):
        return "control flow"
    if is_call:
        return WIDE_CALL if n_lines >= 4 else "short calls"
    if isinstance(stmt, (ast.Assign, ast.AugAssign)) and n_lines >= 4:
        return "multi-line installs and expressions"
    return "installs (assignments)"


def _classify(stmt, diag, kind: dict[int, str]) -> None:
    here = _lines(stmt)
    if isinstance(stmt, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
        test = ast.unparse(stmt.test) if isinstance(stmt, ast.If) else ""
        inner = diag or any(marker in test for marker in DIAGNOSTIC_TESTS)
        label = "checkpoint" if "checkpoint_writer" in test else None
        for line in here:
            kind.setdefault(line, "control flow")
        for child in ast.iter_child_nodes(stmt):
            if isinstance(child, ast.stmt):
                _classify(child, label or inner, kind)
        for handler in getattr(stmt, "handlers", []):
            for child in handler.body:
                _classify(child, label or inner, kind)
        if label or inner:
            for line in here:
                kind[line] = label or ("diagnostics" if kind[line] == "control flow" or diag is None else kind[line])
        return
    name = "half closure (scoring call and recording)" if isinstance(stmt, ast.FunctionDef) else _statement_kind(stmt, diag)
    for line in here:
        kind[line] = name


def classify_loop(source: str, controller: str = CONTROLLER) -> dict:
    """Line counts of the controller's ``while`` loop: body, per mode, shared by kind, and the shared wide calls."""
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == controller)
    loop = next(n for n in ast.walk(function) if isinstance(n, ast.While))
    k1, cl = mode_lines(loop)
    body = set(range(loop.lineno + 1, loop.end_lineno + 1))
    shared = body - k1 - cl
    kind: dict[int, str] = {}
    for stmt in loop.body:
        _classify(stmt, None, kind)
    text = source.splitlines()
    counts = collections.Counter()
    for line in sorted(shared):
        stripped = text[line - 1].strip()
        if not stripped:
            counts["blank"] += 1
        elif stripped.startswith("#"):
            counts["comments"] += 1
        else:
            counts[kind.get(line, "control flow")] += 1
    calls, seen = [], set()
    for node in ast.walk(loop):
        if not (isinstance(node, (ast.Assign, ast.Expr)) and isinstance(node.value, ast.Call)):
            continue
        if node.lineno in shared and kind.get(node.lineno) == WIDE_CALL and node.lineno not in seen:
            seen.add(node.lineno)
            calls.append((node.lineno, node.end_lineno - node.lineno + 1, ast.unparse(node.value.func)))
    return {"body": len(body), "k1": len(k1), "class3d": len(cl), "shared": len(shared), "kinds": counts, "calls": calls}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file", nargs="?", default=str(REPO_ROOT / "relax/refinement/iteration_loop.py"))
    parser.add_argument("--calls", action="store_true", help="list each shared call with a wide argument list")
    args = parser.parse_args(argv)
    result = classify_loop(Path(args.file).read_text())
    print(
        f"loop body {result['body']} lines: K1-only {result['k1']}, "
        f"Class3D-only {result['class3d']}, shared {result['shared']}"
    )
    for name, value in result["kinds"].most_common():
        print(f"  {value:5d}  {name}")
    if args.calls:
        for line, length, func in result["calls"]:
            print(f"  L{line:5d} {length:3d} {func}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
