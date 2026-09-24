#!/usr/bin/env python
"""Same-code GPU noise envelope of the fast parity tier's metrics.

GPU runs of the same code are not bit-reproducible: racing reductions (atomics, scatter
sums) move printed metrics by about 1e-13 to 1e-6 between two runs of the same commit on the
same GPU model. This script measures that spread from pairs of fast-tier pytest basetemps
that ran the same code, and checks a control-vs-candidate difference against it.

Metrics per case: the approved gate metrics (``scripts/em_tier_fsc.py``: min and mean FSC-AUC
against the RELION oracle, minimum in-band shell FSC, mean per-particle |dPmax|) and every
numeric value in the case's ``em_parity_quality_fast_ledger_*.json`` (walltime excluded).

    python scripts/em_tier_noise_envelope.py build --pair LABEL:GPU_MODEL:ROOT_A:ROOT_B ... \\
        --output tests/tiers/gpu_noise_envelope.json
    python scripts/em_tier_noise_envelope.py check --control ROOT_A --candidate ROOT_B

A difference inside the envelope cannot be told apart from GPU noise, so it needs no rerun.
A larger difference is a real change of the numbers, not by itself a failure: merges are
judged by the approved gates in ``tests/tiers/fsc_thresholds.json``, never by bitwise equality.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.em_tier_fsc import find_case_dirs, score_relax_case  # noqa: E402

ENVELOPE = REPO_ROOT / "tests" / "tiers" / "gpu_noise_envelope.json"
SKIPPED_LEDGER_KEYS = ("walltime", "timestamp")
NOISE_FACTOR = 10.0
NOISE_FLOOR = 1e-9


def case_metrics(name: str, case_dir: Path) -> dict[str, float]:
    scored = score_relax_case(name, case_dir)
    out = {f"gate.{k}": v for k, v in scored["summary"].items()}
    pmax = scored.get("pmax", {}).get("relax_minus_relion", {}).get("mean_abs")
    if pmax is not None:
        out["gate.pmax_mean_abs"] = pmax
    for ledger in sorted(case_dir.glob("em_parity_quality_fast_ledger_*.json")):
        for key, value in json.loads(ledger.read_text()).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not any(s in key for s in SKIPPED_LEDGER_KEYS):
                    out[f"ledger.{key.removeprefix(name + '_')}"] = float(value)
    return out


@functools.cache
def run_metrics(root: Path) -> dict[str, dict[str, float]]:
    return {name: case_metrics(name, path) for name, path in sorted(find_case_dirs(root).items())}


def diff_runs(a: Path, b: Path) -> dict[str, dict[str, float]]:
    ma, mb = run_metrics(a), run_metrics(b)
    return {
        case: {k: abs(ma[case][k] - mb[case][k]) for k in sorted(set(ma[case]) & set(mb[case]))}
        for case in sorted(set(ma) & set(mb))
    }


def accumulate(pairs: list[str]) -> tuple[list[dict], dict]:
    sources, envelope = [], {}
    for spec in pairs:
        label, model, root_a, root_b = spec.split(":", 3)
        diffs = diff_runs(Path(root_a), Path(root_b))
        sources.append({"label": label, "gpu_model": model, "a": root_a, "b": root_b, "cases": sorted(diffs)})
        for case, metrics in diffs.items():
            for key, d in metrics.items():
                row = envelope.setdefault(case, {}).setdefault(
                    key, {"max_abs_diff": 0.0, "n_pairs": 0, "by_source": {}}
                )
                row["max_abs_diff"] = max(row["max_abs_diff"], d)
                row["n_pairs"] += 1
                row["by_source"][label] = d
        print(f"{label}: {len(diffs)} cases", flush=True)
    for metrics in envelope.values():
        for row in metrics.values():
            row["noise_limit"] = max(NOISE_FACTOR * row["max_abs_diff"], NOISE_FLOOR)
    return sources, envelope


def cmd_build(args: argparse.Namespace) -> int:
    sources, envelope = accumulate(args.pair)
    payload = {
        "description": args.description,
        "noise_limit": f"max({NOISE_FACTOR:g} x max_abs_diff, {NOISE_FLOOR:g}): few same-code pairs per case",
        "sources": sources,
        "cases": envelope,
    }
    if args.cross_model_pair:
        cross_sources, cross = accumulate(args.cross_model_pair)
        payload["cross_model"] = {
            "note": "reported only; `check` uses the same-model envelope",
            "sources": cross_sources,
            "cases": cross,
        }
    args.output.write_text(json.dumps(payload, indent=1) + "\n")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    envelope = json.loads(args.envelope.read_text())["cases"]
    outside = 0
    for case, metrics in diff_runs(args.control, args.candidate).items():
        for key, d in metrics.items():
            limit = envelope.get(case, {}).get(key, {}).get("noise_limit")
            if limit is None:
                verdict = "no envelope"
            elif d <= limit:
                verdict = "inside"
            else:
                verdict, outside = "OUTSIDE", outside + 1
            print(
                f"{case:36s} {key:48s} |diff| {d:.3e}  envelope {limit if limit is None else f'{limit:.3e}'}  {verdict}"
            )
    print(f"{outside} metric(s) outside the same-code envelope (judge them by the approved gates)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    build = sub.add_parser("build", help="measure the envelope from same-code run pairs")
    build.add_argument("--pair", action="append", required=True, help="LABEL:GPU_MODEL:ROOT_A:ROOT_B")
    build.add_argument("--cross-model-pair", action="append", default=[], help="same code on two GPU models")
    build.add_argument("--description", default="")
    build.add_argument("--output", type=Path, default=ENVELOPE)
    check = sub.add_parser("check", help="compare a control and a candidate run against the envelope")
    check.add_argument("--control", type=Path, required=True)
    check.add_argument("--candidate", type=Path, required=True)
    check.add_argument("--envelope", type=Path, default=ENVELOPE)
    args = parser.parse_args(argv)
    return cmd_build(args) if args.mode == "build" else cmd_check(args)


if __name__ == "__main__":
    sys.exit(main())
