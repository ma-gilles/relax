#!/usr/bin/env python3
"""Regenerate K=1 scorecard final merged-map FSC metrics with RELION gridding correction.

Scorecard v1 was recorded while relax's final all-data pass skipped RELION's
gridding correction (``griddingCorrect`` in RELION's ``backprojector.cpp``) for
the final merged map. The final pass now always applies it. The correction is
the last real-space step of the reconstruction (after the spherical soft mask)
and ``final_merged.mrc`` is ``real(idft3(reconstruction))``, so the corrected
map equals the saved uncorrected map divided by RELION's radial ``sinc^2`` to
float32 rounding (``tests/unit/test_relion_functions.py::
test_final_gridding_correction_equals_post_hoc_division_of_uncorrected_map``).

For each case this script reads the case's cited FSC trajectory report
(``scripts/audit_k1_fsc_trajectory.py`` output), reproduces its final merged
cross-engine FSC-AUC and GT FSC-AUC delta from the saved maps with the same
loaders, shell definition and sign policy, then recomputes both with the
gridding-corrected RECOVAR merged map. Numbered-iteration metrics, the final
split halves (``final_half*_unfil.mrc`` were always corrected) and the RELION
maps are unchanged. A case whose ``refinement_results.npz`` records
``final_all_data_grid_correct = True`` is reported unchanged.

Correlation is not computed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from scripts.audit_k1_fsc_trajectory import _map_metric
from scripts.summarize_em_completion_bench import _load_recovar_volume, _load_relax_volume, _load_relion_volume

SCHEMA = "em_k1_scorecard_final_gridding_regeneration_v1"
GRIDDING_PADDING_FACTOR = 2
FINAL_THRESHOLD_FAILURE_PREFIX = "final merged "


class RegenerationError(RuntimeError):
    """Raised when a case cannot be regenerated or does not reproduce its record."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def gridding_corrected_saved_map(volume: np.ndarray, padding_factor: int = GRIDDING_PADDING_FACTOR) -> np.ndarray:
    """Return RELION's gridding correction of a saved uncorrected map, rounded like a saved map.

    ``relax.reconstruction.relion_functions_relion._gridding_correct_trilinear_np``
    divides by ``sinc^2(r / (N * padding_factor))`` with ``r`` measured from voxel
    index ``N/2``. The result is rounded to float32 because final maps are
    written as float32 MRC files, then returned as float64 for the FSC.
    """

    from relax.reconstruction.relion_functions_relion import _gridding_correct_trilinear_np

    vol = np.asarray(volume, dtype=np.float64)
    if vol.ndim != 3 or len(set(vol.shape)) != 1:
        raise RegenerationError(f"expected a cubic volume, got shape {vol.shape}")
    corrected = _gridding_correct_trilinear_np(vol, vol.shape[0], padding_factor)
    return corrected.astype(np.float32).astype(np.float64)


def final_grid_correct_recorded(refinement_results: Path) -> bool:
    """Return the ``final_all_data_grid_correct`` flag the run recorded."""

    with np.load(refinement_results, allow_pickle=False) as payload:
        if "final_all_data_grid_correct" not in payload.files:
            raise RegenerationError(f"{refinement_results} does not record final_all_data_grid_correct")
        return bool(np.asarray(payload["final_all_data_grid_correct"]).item())


def saved_map_matches_final_mean(refinement_results: Path, saved_merged: np.ndarray) -> dict[str, Any]:
    """Compare ``final_merged.mrc`` with ``real(idft3(final_mean_ft))`` from the run's NPZ.

    This checks that the saved map is the plain inverse transform of the final
    reconstruction, i.e. no real-space step followed the (skipped) correction.
    """

    from recovar.core import fourier_transform_utils as ftu

    with np.load(refinement_results, allow_pickle=False) as payload:
        if "final_mean_ft" not in payload.files:
            return {"available": False}
        final_mean_ft = np.asarray(payload["final_mean_ft"])
    shape = saved_merged.shape
    real = np.real(np.asarray(ftu.get_idft3(final_mean_ft.reshape(shape)))).astype(np.float32)
    residual = np.abs(real.astype(np.float64) - saved_merged)
    scale = float(np.max(np.abs(saved_merged)))
    return {
        "available": True,
        "max_abs": float(np.max(residual)),
        "max_abs_relative_to_map_max": float(np.max(residual) / scale) if scale > 0 else None,
        "bitwise_equal": bool(np.array_equal(real.astype(np.float64), saved_merged)),
    }


def _final_metrics(
    rec_merged: np.ndarray,
    rel_merged: np.ndarray,
    gt: np.ndarray,
    *,
    gt_sign_invariant: bool,
) -> dict[str, Any]:
    scratch: dict[str, np.ndarray] = {}
    cross = _map_metric(
        rec_merged, rel_merged, sign_invariant=False, shellwise_key="final_cross_merged", shellwise=scratch
    )
    rec_gt = _map_metric(
        rec_merged, gt, sign_invariant=gt_sign_invariant, shellwise_key="final_recovar_gt_merged", shellwise=scratch
    )
    rel_gt = _map_metric(
        rel_merged, gt, sign_invariant=gt_sign_invariant, shellwise_key="final_relion_gt_merged", shellwise=scratch
    )
    delta = None
    if rec_gt["fsc_auc"] is not None and rel_gt["fsc_auc"] is not None:
        delta = float(rec_gt["fsc_auc"] - rel_gt["fsc_auc"])
    return {
        "final_cross_engine_fsc_auc": cross["fsc_auc"],
        "final_recovar_gt_fsc_auc": rec_gt["fsc_auc"],
        "final_relion_gt_fsc_auc": rel_gt["fsc_auc"],
        "final_gt_fsc_auc_delta": delta,
    }


def final_gate_failures(
    cross: float | None,
    delta: float | None,
    *,
    min_cross: float,
    min_delta: float,
) -> list[str]:
    """Return the audit's final-map gate failures (same wording as the audit)."""

    failures: list[str] = []
    if cross is None or not math.isfinite(float(cross)):
        failures.append("final merged cross-engine FSC-AUC is missing or non-finite")
    elif float(cross) < min_cross:
        failures.append(f"final merged cross-engine FSC-AUC {float(cross):.9f} < {min_cross:.9f}")
    if delta is None or not math.isfinite(float(delta)):
        failures.append("final merged GT FSC-AUC delta is missing or non-finite")
    elif float(delta) < min_delta:
        failures.append(f"final merged GT FSC-AUC delta {float(delta):+.9f} < {min_delta:+.9f}")
    return failures


def regenerate_case(
    case_id: str,
    fsc_report: Path,
    *,
    min_cross: float,
    min_delta: float,
    reproduction_tolerance: float,
    check_final_mean: bool = True,
    hash_inputs: bool = True,
) -> dict[str, Any]:
    """Reproduce and regenerate one case's final merged-map metrics."""

    fsc_report = Path(fsc_report)
    report = json.loads(fsc_report.read_text())
    paths = report["paths"]
    recovar_dir = Path(paths["recovar_dir"])
    relion_dir = Path(paths["relion_dir"])
    inputs = {
        "fsc_report": fsc_report,
        "recovar_final_merged": recovar_dir / "final_merged.mrc",
        "recovar_refinement_results": recovar_dir / "refinement_results.npz",
        "relion_final_merged": relion_dir / "run_class001.mrc",
        "gt_volume": Path(paths["gt_volume"]),
    }
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise RegenerationError(f"{case_id}: missing inputs {missing}")
    gt_sign_invariant = report["gt_sign_policy"]["used"] == "sign_invariant"
    recorded_final = report["final"]
    recorded = {
        "final_cross_engine_fsc_auc": recorded_final["cross_engine"]["merged"]["fsc_auc"],
        "final_gt_fsc_auc_delta": recorded_final["merged_gt_fsc_auc_delta"],
    }

    rec_merged = _load_relax_volume(inputs["recovar_final_merged"])
    rel_merged = _load_relion_volume(inputs["relion_final_merged"])
    gt = _load_recovar_volume(inputs["gt_volume"])

    reproduced = _final_metrics(rec_merged, rel_merged, gt, gt_sign_invariant=gt_sign_invariant)
    discrepancy = max(
        abs(float(reproduced[key]) - float(recorded[key]))
        for key in ("final_cross_engine_fsc_auc", "final_gt_fsc_auc_delta")
    )
    if not discrepancy <= reproduction_tolerance:
        raise RegenerationError(
            f"{case_id}: saved maps do not reproduce the recorded final metrics "
            f"(max discrepancy {discrepancy:.3e} > {reproduction_tolerance:.1e})"
        )

    already_corrected = final_grid_correct_recorded(inputs["recovar_refinement_results"])
    if already_corrected:
        regenerated = dict(reproduced)
        route = "unchanged: run recorded final_all_data_grid_correct=True"
    else:
        corrected = gridding_corrected_saved_map(rec_merged)
        regenerated = _final_metrics(corrected, rel_merged, gt, gt_sign_invariant=gt_sign_invariant)
        route = "post-hoc RELION griddingCorrect of the saved uncorrected final_merged.mrc"

    non_final_failures = [
        failure
        for failure in list(report.get("failures", []))
        if not str(failure).startswith(FINAL_THRESHOLD_FAILURE_PREFIX)
    ]
    new_final_failures = final_gate_failures(
        regenerated["final_cross_engine_fsc_auc"],
        regenerated["final_gt_fsc_auc_delta"],
        min_cross=min_cross,
        min_delta=min_delta,
    )
    row: dict[str, Any] = {
        "case_id": case_id,
        "route": route,
        "recorded_final_all_data_grid_correct": already_corrected,
        "gt_sign_policy": report["gt_sign_policy"]["used"],
        "recorded": recorded,
        "reproduced_uncorrected": reproduced,
        "reproduction_max_abs_discrepancy": discrepancy,
        "regenerated": regenerated,
        "recorded_result": report["status"],
        "non_final_failures": non_final_failures,
        "final_failures": new_final_failures,
        "result": "pass" if not (non_final_failures or new_final_failures) else "fail",
        "inputs": {name: str(path) for name, path in inputs.items()},
    }
    if hash_inputs:
        row["inputs_sha256"] = {name: sha256_file(path) for name, path in inputs.items()}
    if check_final_mean:
        row["saved_map_vs_final_mean_ft"] = saved_map_matches_final_mean(
            inputs["recovar_refinement_results"], rec_merged
        )
    return row


REGENERATION_METHOD = "regenerated post hoc with RELION griddingCorrect on the saved final maps"
SCRIPT_PATH = "scripts/regenerate_em_k1_scorecard_final_gridding.py"
V1_SCHEMA = "recovar.em_relion_parity_scorecard.v1"
V2_SCHEMA = "recovar.em_relion_parity_scorecard.v2"
V2_GRID_CORRECTION = "always-on (RELION griddingCorrect)"
LEDGER_SCHEMA_RE = re.compile(r"em_k1_gui_grid0_local_highshell_full34_superseding_ledger_v(\d+)")


def gridding_metadata() -> dict[str, Any]:
    return {
        "reference": "RELION backprojector.cpp BackProjector::reconstruct -> griddingCorrect",
        "formula": "map / sinc^2(r / (N * padding_factor)), r from voxel index N/2",
        "padding_factor": GRIDDING_PADDING_FACTOR,
        "implementation": "relax.reconstruction.relion_functions_relion._gridding_correct_trilinear_np",
        "applies_to": "final merged map only; numbered iterations and final_half*_unfil.mrc were already corrected",
    }


def merge_regenerations(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge per-case regeneration payloads into one case-ordered row list."""

    rows: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        if payload.get("schema") != SCHEMA:
            raise RegenerationError(f"unsupported regeneration schema {payload.get('schema')!r}")
        for row in payload["cases"]:
            if row["case_id"] in rows:
                raise RegenerationError(f"duplicate regeneration row for {row['case_id']}")
            rows[row["case_id"]] = row
    return [rows[key] for key in sorted(rows)]


def build_superseding_ledger(
    v1_scorecard: dict[str, Any],
    previous_ledger_sha256: str,
    rows: list[dict[str, Any]],
    *,
    ledger_schema: str,
    generated_utc: str,
    fixture_manifest_sha256: str,
) -> dict[str, Any]:
    """Build the ledger that re-scores every case's final merged map."""

    source = v1_scorecard["current_snapshot"]["source_ledger"]
    if previous_ledger_sha256 != source["sha256"]:
        raise RegenerationError("previous ledger SHA-256 differs from the v1 scorecard current snapshot")
    previous = LEDGER_SCHEMA_RE.fullmatch(source["schema"])
    current = LEDGER_SCHEMA_RE.fullmatch(ledger_schema)
    if previous is None or current is None or int(current.group(1)) != int(previous.group(1)) + 1:
        raise RegenerationError(f"ledger schema {ledger_schema!r} must advance {source['schema']!r} by one")
    case_ids = [case["id"] for case in v1_scorecard["cases"]]
    if [row["case_id"] for row in rows] != case_ids:
        raise RegenerationError("regeneration rows must cover every frozen case exactly once, in order")
    topology = {case["id"]: case["intermediate_result"] for case in v1_scorecard["cases"]}
    strict = {status: 0 for status in ("pass", "fail", "not_run")}
    topo = dict(strict)
    for row in rows:
        strict[row["result"]] += 1
        topo[topology[row["case_id"]]] += 1
    contract = dict(v1_scorecard["acceptance_contract"])
    contract["grid_correction"] = V2_GRID_CORRECTION
    return {
        "schema": ledger_schema,
        "generated_utc": generated_utc,
        "suite_id": v1_scorecard["suite_id"],
        "suite_version": 2,
        "frozen_denominator": v1_scorecard["frozen_denominator"],
        "frozen_case_definitions_sha256": v1_scorecard["frozen_case_definitions_sha256"],
        "fixture_manifest_sha256": fixture_manifest_sha256,
        "supersedes": {"schema": source["schema"], "sha256": source["sha256"]},
        "acceptance_contract": contract,
        "counts": {"strict": strict, "topology": topo},
        "regeneration": {"method": REGENERATION_METHOD, "script": SCRIPT_PATH, "gridding": gridding_metadata()},
        "updates": rows,
        "status_note": (
            "Every case's final merged-map metrics re-scored with RELION's final gridding correction; "
            "numbered-iteration evidence is unchanged."
        ),
    }


def build_v2_scorecard(
    v1_scorecard: dict[str, Any],
    v1_scorecard_path: str,
    v1_scorecard_sha256: str,
    ledger: dict[str, Any],
    ledger_path: str,
    ledger_sha256: str,
    *,
    snapshot_id: str,
    source_heads: list[str],
) -> dict[str, Any]:
    """Build scorecard v2 from frozen v1 and the regeneration ledger."""

    if v1_scorecard.get("schema") != V1_SCHEMA:
        raise RegenerationError("expected a v1 scorecard")
    rows = {row["case_id"]: row for row in ledger["updates"]}
    cases = []
    for case in v1_scorecard["cases"]:
        row = rows[case["id"]]
        regenerated = row["regenerated"]
        cases.append(
            {
                "id": case["id"],
                "name": case["name"],
                "definition": case["definition"],
                "result": row["result"],
                "intermediate_result": case["intermediate_result"],
                "final_cross_engine_fsc_auc": regenerated["final_cross_engine_fsc_auc"],
                "final_gt_fsc_auc_delta": regenerated["final_gt_fsc_auc_delta"],
                "source_head": case["source_head"],
                "jobs": case["jobs"],
                "suite_v1": {
                    "result": case["result"],
                    "final_cross_engine_fsc_auc": case["final_cross_engine_fsc_auc"],
                    "final_gt_fsc_auc_delta": case["final_gt_fsc_auc_delta"],
                },
                "final_metric_provenance": {
                    "method": REGENERATION_METHOD,
                    "script": SCRIPT_PATH,
                    "route": row["route"],
                    "recorded_final_all_data_grid_correct": row["recorded_final_all_data_grid_correct"],
                    "reproduction_max_abs_discrepancy": row["reproduction_max_abs_discrepancy"],
                    "inputs": row["inputs"],
                    "inputs_sha256": row["inputs_sha256"],
                },
            }
        )
    counts = ledger["counts"]["strict"]
    history = [dict(snapshot, suite_version=1) for snapshot in v1_scorecard["history"]]
    history.append(
        {
            "id": snapshot_id,
            "recorded_utc": ledger["generated_utc"],
            "source_heads": list(source_heads),
            "counts": counts,
            "evidence_schema": ledger["schema"],
            "evidence_sha256": ledger_sha256,
            "status_note": (
                "Suite version 2: final merged maps gridding-corrected as RELION does; "
                "final metrics regenerated post hoc from the saved maps."
            ),
            "suite_version": 2,
        }
    )
    return {
        "schema": V2_SCHEMA,
        "suite_id": v1_scorecard["suite_id"],
        "suite_version": 2,
        "frozen_denominator": v1_scorecard["frozen_denominator"],
        "frozen_case_definitions_sha256": v1_scorecard["frozen_case_definitions_sha256"],
        "metric_policy": v1_scorecard["metric_policy"],
        "fixture_policy": (
            "Version 2 keeps the version 1 case parameters and artifact-pinned fixtures and changes only the "
            "final-map contract: final all-data maps are gridding-corrected as in RELION. Newly generated "
            "datasets remain non-scoring replicate diagnostics."
        ),
        "acceptance_contract": ledger["acceptance_contract"],
        "previous_version": {
            "schema": V1_SCHEMA,
            "path": v1_scorecard_path,
            "sha256": v1_scorecard_sha256,
            "current_snapshot_id": v1_scorecard["current_snapshot"]["id"],
        },
        "regeneration": {
            "method": REGENERATION_METHOD,
            "script": SCRIPT_PATH,
            "gridding": gridding_metadata(),
        },
        "current_snapshot": {
            "id": snapshot_id,
            "counts": counts,
            "source_ledger": {
                "schema": ledger["schema"],
                "generated_utc": ledger["generated_utc"],
                "sha256": ledger_sha256,
                "path": ledger_path,
            },
        },
        "history": history,
        "cases": cases,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def build_main(args: argparse.Namespace) -> int:
    repo = Path(__file__).resolve().parents[1]
    v1_path = repo / args.v1_scorecard
    v1 = json.loads(v1_path.read_text())
    rows = merge_regenerations([json.loads(path.read_text()) for path in args.regeneration])
    ledger = build_superseding_ledger(
        v1,
        sha256_file(args.previous_ledger),
        rows,
        ledger_schema=args.ledger_schema,
        generated_utc=args.generated_utc,
        fixture_manifest_sha256=sha256_file(repo / args.fixture_manifest),
    )
    ledger_path = repo / args.ledger_output
    _write_json(ledger_path, ledger)
    scorecard = build_v2_scorecard(
        v1,
        str(args.v1_scorecard),
        sha256_file(v1_path),
        ledger,
        str(args.ledger_output),
        sha256_file(ledger_path),
        snapshot_id=args.snapshot_id,
        source_heads=args.source_head,
    )
    _write_json(repo / args.scorecard_output, scorecard)
    print(f"strict counts {ledger['counts']['strict']}")
    return 0


def regenerate_main(args: argparse.Namespace) -> int:
    cases = json.loads(args.cases.read_text())
    rows = []
    for case in cases:
        if args.only and case["id"] not in args.only:
            continue
        row = regenerate_case(
            case["id"],
            Path(case["fsc_report"]),
            min_cross=args.min_cross_merged_fsc_auc,
            min_delta=args.min_merged_gt_delta,
            reproduction_tolerance=args.reproduction_tolerance,
            check_final_mean=not args.skip_final_mean_check,
        )
        rows.append(row)
        print(
            f"{row['case_id']} reproduce={row['reproduction_max_abs_discrepancy']:.2e} "
            f"cross {row['recorded']['final_cross_engine_fsc_auc']:.9f}->"
            f"{row['regenerated']['final_cross_engine_fsc_auc']:.9f} "
            f"delta {row['recorded']['final_gt_fsc_auc_delta']:+.9f}->"
            f"{row['regenerated']['final_gt_fsc_auc_delta']:+.9f} "
            f"{row['recorded_result']}->{row['result']}",
            flush=True,
        )
    payload = {
        "schema": SCHEMA,
        "script": SCRIPT_PATH,
        "gridding_correction": gridding_metadata(),
        "thresholds": {
            "merged_cross_engine_fsc_auc_min": args.min_cross_merged_fsc_auc,
            "recovar_minus_relion_merged_gt_fsc_auc_min": args.min_merged_gt_delta,
        },
        "reproduction_tolerance": args.reproduction_tolerance,
        "metric_policy": "Shellwise FSC and normalized FSC-AUC only; correlation is not computed.",
        "cases": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    regen = commands.add_parser("regenerate", help="re-score cases from their saved maps")
    regen.add_argument("--cases", type=Path, required=True, help="JSON list of {id, fsc_report} objects")
    regen.add_argument("--output", type=Path, required=True)
    regen.add_argument("--only", nargs="*", default=None, help="Restrict to these case ids")
    regen.add_argument("--min-cross-merged-fsc-auc", type=float, default=0.995)
    regen.add_argument("--min-merged-gt-delta", type=float, default=-0.002)
    regen.add_argument("--reproduction-tolerance", type=float, default=1e-9)
    regen.add_argument("--skip-final-mean-check", action="store_true")
    regen.set_defaults(func=regenerate_main)

    build = commands.add_parser("build", help="write the superseding ledger and scorecard v2")
    build.add_argument("--regeneration", type=Path, nargs="+", required=True)
    build.add_argument("--previous-ledger", type=Path, required=True)
    build.add_argument("--ledger-schema", required=True)
    build.add_argument("--generated-utc", required=True)
    build.add_argument("--snapshot-id", required=True)
    build.add_argument("--source-head", nargs="+", required=True, help="commit boundary of the snapshot")
    build.add_argument("--v1-scorecard", type=Path, default=Path("docs/math/em_relion_parity_scorecard_v1.json"))
    build.add_argument(
        "--fixture-manifest", type=Path, default=Path("docs/math/em_relion_parity_fixture_manifest_v2.json")
    )
    build.add_argument("--ledger-output", type=Path, required=True, help="repository-relative path")
    build.add_argument("--scorecard-output", type=Path, default=Path("docs/math/em_relion_parity_scorecard_v2.json"))
    build.set_defaults(func=build_main)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
