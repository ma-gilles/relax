"""Forward a tier's coarse-engine choice through the public CLI and check execution."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


CHOICES = ("auto", "gemm_hybrid", "gemm_dense")
ENV_NAME = "RELAX_TIER_COARSE_ENGINE"


def selected_coarse_engine() -> str:
    choice = os.environ.get(ENV_NAME, "auto")
    if choice not in CHOICES:
        raise ValueError(f"{ENV_NAME} must be one of {CHOICES}, got {choice!r}")
    return choice


def selected_cli_args() -> list[str]:
    """Only test harnesses read the environment; production receives a typed CLI choice."""
    choice = selected_coarse_engine()
    return [] if choice == "auto" else ["--coarse-engine", choice]


def _nonnegative_count(call: dict, key: str) -> int:
    value = call.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AssertionError(f"missing, noninteger or negative {key} in coarse execution: {call}")
    return value


def _execution_calls(output_dir: Path, *, output_kind: str) -> list[dict]:
    if output_kind == "vdam":
        paths = sorted(output_dir.glob("run_it*_recovar_meta.json"))
        if not paths:
            raise AssertionError(f"no VDAM iteration metadata in {output_dir}")
        return [call for path in paths for call in json.loads(path.read_text()).get("coarse_engine_calls", [])]
    if output_kind == "kclass":
        path = output_dir / "summary.json"
        if not path.is_file():
            raise AssertionError(f"no K-class replay summary at {path}")
        return json.loads(path.read_text()).get("coarse_engine_calls", [])
    path = output_dir / "refinement_results.npz"
    if not path.is_file():
        raise AssertionError(f"no refinement results at {path}")
    with np.load(path, allow_pickle=False) as data:
        if "coarse_engine_trajectory" not in data:
            raise AssertionError(f"no coarse engine execution trajectory in {path}")
        trajectory = json.loads(str(data["coarse_engine_trajectory"].item()))
        final = json.loads(str(data["final_all_data_coarse_engines"].item())) if "final_all_data_coarse_engines" in data else []
    return [call for iteration in trajectory for call in iteration] + list(final)


def assert_selected_global_execution(output_dir: Path, *, output_kind: str = "standard") -> None:
    """Fail if a selected global path never executed the requested GEMM strategy."""
    choice = selected_coarse_engine()
    if choice == "auto":
        return
    calls = _execution_calls(Path(output_dir), output_kind=output_kind)
    if not calls:
        raise AssertionError(f"{choice} selected but no global coarse calls were recorded in {output_dir}")
    expected_strategy = "hybrid" if choice == "gemm_hybrid" else "dense"
    for call in calls:
        if call.get("requested") != choice or call.get("resolved") != choice:
            raise AssertionError(f"coarse engine fell back or was not selected: {call}")
        if call.get("strategy") != expected_strategy:
            raise AssertionError(f"wrong coarse strategy: {call}")
        images = _nonnegative_count(call, "images")
        classes = _nonnegative_count(call, "classes")
        rotations = _nonnegative_count(call, "rotations")
        translations = _nonnegative_count(call, "translations")
        coarse = _nonnegative_count(call, "coarse_candidates_per_image")
        fine = _nonnegative_count(call, "fine_candidates_per_image")
        evaluated = _nonnegative_count(call, "evaluated_fine_candidates_total")
        known = call.get("selected_fine_candidates_known")
        if not isinstance(known, bool):
            raise AssertionError(f"missing or nonboolean selected_fine_candidates_known: {call}")
        if known:
            selected = _nonnegative_count(call, "selected_fine_candidates_total")
        else:
            if "selected_fine_candidates_total" not in call or call["selected_fine_candidates_total"] is not None:
                raise AssertionError(f"unknown selected fine count must be null: {call}")
            selected = None
        current_size = _nonnegative_count(call, "current_size")
        reconstruction_size = _nonnegative_count(call, "reconstruction_current_size")
        hypotheses = _nonnegative_count(call, "hypotheses_per_image")
        if min(images, classes, rotations, translations, coarse, fine, current_size, reconstruction_size) <= 0:
            raise AssertionError(f"nonpositive executed grid: {call}")
        if hypotheses != classes * rotations * translations:
            raise AssertionError(f"scored hypothesis count disagrees with grid: {call}")
        if evaluated <= 0 or evaluated > images * fine or (selected is not None and selected > evaluated):
            raise AssertionError(f"fine candidate counts are inconsistent: {call}")
        policy = call["posterior_policy"]
        if policy not in {"gaussian", "cc_winner"} or call["score_mode"] not in {"gaussian", "normalized_cc"}:
            raise AssertionError(f"unknown coarse score/posterior policy: {call}")
        if (policy == "gaussian") != (call["score_mode"] == "gaussian"):
            raise AssertionError(f"coarse score and posterior policy disagree: {call}")
        precision = call.get("precision")
        if not isinstance(precision, dict) or precision != {
            "score": "float32", "projection": "float32", "mstep": "float32"
        }:
            raise AssertionError(f"selected engine used unexpected precision: {call}")
        if expected_strategy == "hybrid":
            if call.get("fine_engine") != "resident" or classes * rotations * translations != coarse:
                raise AssertionError(f"hybrid did not use the complete coarse grid and resident fine pass: {call}")
        elif (not known or call.get("fine_engine") is not None or call.get("pruned") is not False
              or classes * rotations * translations != fine or evaluated != images * fine
              or selected != (images if policy == "cc_winner" else images * fine)):
            raise AssertionError(f"dense GEMM did not score the complete expanded grid: {call}")


def run_selected_command(
    cmd: list[str], *, require_global: bool = True, output_kind: str = "standard", **kwargs
) -> subprocess.CompletedProcess:
    """Launch a parity subprocess through its public selector and verify real work."""
    argv = [*cmd, *selected_cli_args()]
    proc = subprocess.run(argv, **kwargs)
    if proc.returncode or not require_global or selected_coarse_engine() == "auto":
        return proc
    for flag in ("--output_dir", "--output-dir", "--output", "--o"):
        if flag in argv:
            path = Path(argv[argv.index(flag) + 1])
            # InitialModel's --o is an output prefix (for example out/run),
            # while refinement's --output is an output directory.
            if output_kind == "vdam":
                path = path.parent
            assert_selected_global_execution(path, output_kind=output_kind)
            return proc
    raise AssertionError(f"selected coarse-engine command has no output directory: {argv}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check a completed selected coarse-engine run")
    parser.add_argument("--assert-standard", type=Path, required=True, metavar="OUTPUT_DIR")
    parser.add_argument("--coarse-engine", choices=CHOICES[1:], required=True)
    args = parser.parse_args(argv)
    os.environ[ENV_NAME] = args.coarse_engine
    try:
        assert_selected_global_execution(args.assert_standard)
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        print(f"coarse-engine execution audit failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
