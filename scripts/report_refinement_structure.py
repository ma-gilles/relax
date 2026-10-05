"""Report structural metrics for production refinement Python modules and hold them under ceilings.

    python scripts/report_refinement_structure.py --format markdown            # the current metrics
    python scripts/report_refinement_structure.py --check <ceilings.json>       # warn above a ceiling, exit 1 above its slack
    python scripts/report_refinement_structure.py --lower-ceilings <ceilings.json>

The checked file (``docs/development/refinement_structure_metrics.json``) records upper bounds, not
the current values, so a commit regenerates it only to tighten it. ``--lower-ceilings`` rewrites each
bound to the current value where that is lower (the line totals keep ``LINE_TOTAL_HEADROOM``) and
never raises one. Raising a ceiling is a hand edit of the file, with the reason in the commit message.

The ceilings are review signals with slack (owner ruling, 2026-10-05). For a ceiling ``c`` the slack is
``max(SLACK_MINIMUM, ceil(c * SLACK_PERCENT / 100))``: 5% of the ceiling, rounded up, and at least 1.
A total above ``c`` but at most ``c + slack`` prints a warning naming the metric; only a total above
``c + slack`` fails ``--check``.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

LARGE_ARGUMENT_THRESHOLD = 10
VERY_LARGE_ARGUMENT_THRESHOLD = 20
EXCLUDED_FILENAMES = {"__init__.py"}
CEILING_SCHEMA_VERSION = 2
# Totals that may not grow. File and function counts are not bounded: splitting a module raises them.
STRUCTURE_CEILINGS = (
    "maximum_function_line_span",
    "maximum_parameter_count",
    "large_argument_function_count",
    "very_large_argument_function_count",
)
# The line totals grow with ordinary feature work, so their ceilings sit this fraction above the value
# they were recorded from, rounded up to LINE_TOTAL_ROUNDING lines.
LINE_TOTAL_CEILINGS = ("physical_lines", "nonblank_noncomment_lines")
LINE_TOTAL_HEADROOM = 0.05
LINE_TOTAL_ROUNDING = 100
# How far a total may pass its ceiling with a warning before the check fails; see the module docstring.
SLACK_PERCENT = 5
SLACK_MINIMUM = 1


def _parameter_count(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    positional = [*node.args.posonlyargs, *node.args.args]
    if positional and positional[0].arg in {"self", "cls"}:
        positional = positional[1:]
    return (
        len(positional)
        + len(node.args.kwonlyargs)
        + int(node.args.vararg is not None)
        + int(node.args.kwarg is not None)
    )


def collect_metrics(repo_root: Path) -> dict:
    """Return deterministic source metrics for ``relax/refinement``."""

    source_root = repo_root / "relax" / "refinement"
    files = sorted(path for path in source_root.glob("*.py") if path.name not in EXCLUDED_FILENAMES)
    file_rows = []
    function_rows = []
    for path in files:
        source = path.read_text()
        lines = source.splitlines()
        tree = ast.parse(source, filename=str(path))
        relative_path = path.relative_to(repo_root).as_posix()
        functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        file_rows.append(
            {
                "path": relative_path,
                "physical_lines": len(lines),
                "nonblank_noncomment_lines": sum(
                    bool(line.strip()) and not line.lstrip().startswith("#") for line in lines
                ),
                "function_count": len(functions),
            }
        )
        for node in functions:
            function_rows.append(
                {
                    "path": relative_path,
                    "line": node.lineno,
                    "name": node.name,
                    "parameter_count": _parameter_count(node),
                    "line_span": node.end_lineno - node.lineno + 1,
                }
            )

    large_functions = sorted(
        (row for row in function_rows if row["parameter_count"] >= LARGE_ARGUMENT_THRESHOLD),
        key=lambda row: (-row["parameter_count"], row["path"], row["line"]),
    )
    largest_functions = sorted(
        function_rows,
        key=lambda row: (-row["line_span"], row["path"], row["line"]),
    )
    return {
        "schema_version": 1,
        "scope": {
            "root": "relax/refinement",
            "glob": "*.py",
            "excluded_filenames": sorted(EXCLUDED_FILENAMES),
            "large_argument_threshold": LARGE_ARGUMENT_THRESHOLD,
            "very_large_argument_threshold": VERY_LARGE_ARGUMENT_THRESHOLD,
        },
        "totals": {
            "file_count": len(file_rows),
            "physical_lines": sum(row["physical_lines"] for row in file_rows),
            "nonblank_noncomment_lines": sum(row["nonblank_noncomment_lines"] for row in file_rows),
            "function_count": len(function_rows),
            "large_argument_function_count": len(large_functions),
            "very_large_argument_function_count": sum(
                row["parameter_count"] >= VERY_LARGE_ARGUMENT_THRESHOLD for row in function_rows
            ),
            "maximum_parameter_count": max((row["parameter_count"] for row in function_rows), default=0),
            "maximum_function_line_span": max((row["line_span"] for row in function_rows), default=0),
        },
        "files": file_rows,
        "large_argument_functions": large_functions,
        "largest_functions": largest_functions[:10],
    }


def _format_markdown(metrics: dict) -> str:
    totals = metrics["totals"]
    rows = [
        ("Production files", totals["file_count"]),
        ("Physical production lines", totals["physical_lines"]),
        ("Nonblank, non-comment production lines", totals["nonblank_noncomment_lines"]),
        ("Production functions", totals["function_count"]),
        ("Functions with at least 10 parameters", totals["large_argument_function_count"]),
        ("Functions with at least 20 parameters", totals["very_large_argument_function_count"]),
        ("Largest parameter count", totals["maximum_parameter_count"]),
        ("Largest function span", totals["maximum_function_line_span"]),
    ]
    output = ["| Metric | Value |", "| --- | ---: |"]
    output.extend(f"| {label} | {value:,} |" for label, value in rows)
    output.extend(["", "Large-argument functions:", ""])
    output.extend(
        f"- `{row['path']}:{row['line']} {row['name']}`: {row['parameter_count']} parameters, {row['line_span']} lines"
        for row in metrics["large_argument_functions"]
    )
    return "\n".join(output)


def ceilings_for(totals: dict) -> dict:
    """The tightest ceilings the current ``totals`` allow."""
    ceilings = {name: totals[name] for name in STRUCTURE_CEILINGS}
    for name in LINE_TOTAL_CEILINGS:
        with_headroom = totals[name] * (1 + LINE_TOTAL_HEADROOM)
        ceilings[name] = -int(-with_headroom // LINE_TOTAL_ROUNDING) * LINE_TOTAL_ROUNDING
    return ceilings


def slack_for(ceiling: int) -> int:
    """How far a total may pass ``ceiling`` before the check fails."""
    return max(SLACK_MINIMUM, -(-ceiling * SLACK_PERCENT // 100))


def exceeded(totals: dict, ceilings: dict) -> list[str]:
    """One line per total above its ceiling plus slack; empty when the check passes."""
    return [
        f"{name}: {totals[name]:,} exceeds the ceiling {ceiling:,} by more than its slack {slack_for(ceiling):,}"
        for name, ceiling in ceilings.items()
        if totals[name] > ceiling + slack_for(ceiling)
    ]


def within_slack(totals: dict, ceilings: dict) -> list[str]:
    """One warning per total above its ceiling but within its slack."""
    return [
        f"warning: {name}: {totals[name]:,} is above the ceiling {ceiling:,}, within its slack {slack_for(ceiling):,}"
        for name, ceiling in ceilings.items()
        if ceiling < totals[name] <= ceiling + slack_for(ceiling)
    ]


def lowered(ceilings: dict, totals: dict) -> dict:
    """``ceilings`` tightened to what ``totals`` allow; no bound is raised and none is added."""
    tightest = ceilings_for(totals)
    return {name: min(ceiling, tightest[name]) for name, ceiling in ceilings.items()}


def read_ceilings(path: Path) -> dict:
    payload = json.loads(path.read_text())
    ceilings = payload.get("ceilings", {}).get("totals")
    if payload.get("schema_version") != CEILING_SCHEMA_VERSION or not isinstance(ceilings, dict):
        raise ValueError(f"unsupported refinement structure ceilings: {path}")
    unknown = sorted(set(ceilings) - {*STRUCTURE_CEILINGS, *LINE_TOTAL_CEILINGS})
    if unknown:
        raise ValueError(f"unknown ceilings in {path}: {', '.join(unknown)}")
    return ceilings


def _source_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def lower_ceilings(path: Path, totals: dict, repo_root: Path) -> list[str]:
    """Rewrite ``path`` with every ceiling the code is below tightened; return what changed."""
    payload = json.loads(path.read_text())
    before = read_ceilings(path)
    after = lowered(before, totals)
    changes = [f"{name}: {before[name]:,} -> {after[name]:,}" for name in before if after[name] != before[name]]
    if changes:
        payload["ceilings"] = {
            "recorded_date": dt.date.today().isoformat(),
            "source_commit": _source_commit(repo_root),
            "totals": after,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n")
    return changes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument(
        "--check",
        type=Path,
        help="Warn when a current total exceeds its ceiling in this file; fail when it exceeds the ceiling plus slack",
    )
    parser.add_argument(
        "--lower-ceilings",
        type=Path,
        help="After a clean-up: lower the ceilings in this file to what the code now allows; never raises one",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    metrics = collect_metrics(repo_root)
    if args.lower_ceilings is not None:
        path = args.lower_ceilings if args.lower_ceilings.is_absolute() else repo_root / args.lower_ceilings
        ceilings = read_ceilings(path)
        over = exceeded(metrics["totals"], ceilings)
        warnings = within_slack(metrics["totals"], ceilings)
        changes = lower_ceilings(path, metrics["totals"], repo_root)
        print("\n".join(changes) if changes else "no ceiling can be lowered", file=sys.stderr)
        if warnings or over:
            print("left unchanged, the code is above it:\n" + "\n".join(warnings + over), file=sys.stderr)
        return 1 if over else 0
    if args.check is not None:
        path = args.check if args.check.is_absolute() else repo_root / args.check
        ceilings = read_ceilings(path)
        over = exceeded(metrics["totals"], ceilings)
        warnings = within_slack(metrics["totals"], ceilings)
        if warnings:
            print("\n".join(warnings), file=sys.stderr)
        if over:
            print("\n".join(over), file=sys.stderr)
            return 1

    if args.format == "markdown":
        print(_format_markdown(metrics))
    else:
        print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
