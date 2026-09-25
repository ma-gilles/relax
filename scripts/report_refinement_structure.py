"""Report structural metrics for production refinement Python modules."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

LARGE_ARGUMENT_THRESHOLD = 10
VERY_LARGE_ARGUMENT_THRESHOLD = 20
EXCLUDED_FILENAMES = {"__init__.py"}


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


def _checked_snapshot(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1 or "current" not in payload:
        raise ValueError(f"unsupported refinement structure snapshot: {path}")
    return payload["current"]["metrics"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument(
        "--check",
        type=Path,
        help="Fail unless current metrics equal the checked snapshot's current metrics",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    metrics = collect_metrics(repo_root)
    if args.check is not None:
        snapshot_path = args.check
        if not snapshot_path.is_absolute():
            snapshot_path = repo_root / snapshot_path
        expected = _checked_snapshot(snapshot_path)
        if metrics != expected:
            print(json.dumps({"expected": expected, "actual": metrics}, indent=2, sort_keys=True))
            return 1

    if args.format == "markdown":
        print(_format_markdown(metrics))
    else:
        print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
