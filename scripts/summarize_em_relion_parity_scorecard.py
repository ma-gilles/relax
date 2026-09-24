#!/usr/bin/env python
"""Validate and render the frozen RECOVAR/RELION parity scorecard.

The scorecard is deliberately checked into the repository.  A suite version
has a fixed denominator and fixed case definitions; adding or changing a case
requires a new suite version rather than silently moving the current score.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import NamedTuple, cast

try:
    from scripts.file_hash import sha256_file
except ModuleNotFoundError:
    from file_hash import sha256_file  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[1]
V1_SCORECARD = REPO_ROOT / "docs" / "math" / "em_relion_parity_scorecard_v1.json"
V2_SCORECARD = REPO_ROOT / "docs" / "math" / "em_relion_parity_scorecard_v2.json"
DEFAULT_SCORECARD = V2_SCORECARD
DEFAULT_FIXTURE_MANIFEST = REPO_ROOT / "docs" / "math" / "em_relion_parity_fixture_manifest_v2.json"
DEFAULT_K4_SNAPSHOT = REPO_ROOT / "docs" / "math" / "em_k4_backend_trajectory_snapshot_v2.json"
V1_SUITE_ID = "k1-gui-grid0-local-highshell-full34"
SCHEMA_V1 = "recovar.em_relion_parity_scorecard.v1"
SCHEMA_V2 = "recovar.em_relion_parity_scorecard.v2"
SUITE_VERSION_BY_SCHEMA = {SCHEMA_V1: 1, SCHEMA_V2: 2}
# Suite version 1 was scored with the final all-data gridding correction off;
# version 2 scores RELION-equivalent maps, which always apply griddingCorrect.
GRID_CORRECTION_BY_SUITE_VERSION = {1: "unset/default-off", 2: "always-on (RELION griddingCorrect)"}
FROZEN_THRESHOLDS = {
    "merged_cross_engine_fsc_auc_min": 0.995,
    "recovar_minus_relion_merged_gt_fsc_auc_min": -0.002,
}
V2_REGENERATION_METHOD = "regenerated post hoc with RELION griddingCorrect on the saved final maps"
V2_REGENERATION_SCRIPT = "scripts/regenerate_em_k1_scorecard_final_gridding.py"
V2_PROVENANCE_INPUTS = {
    "fsc_report",
    "recovar_final_merged",
    "recovar_refinement_results",
    "relion_final_merged",
    "gt_volume",
}
V2_FIXTURE_SUITE_ID = f"{V1_SUITE_ID}-artifact-pinned-v2"
V1_FROZEN_DENOMINATOR = 34
V1_FROZEN_CASE_DEFINITIONS_SHA256 = "9e3f2cb7192eb2cbf8a50181cf47de8562adfb98734bab05a736fb7d4d404fc1"
V2_FIXTURE_MANIFEST_SHA256 = "422a79a0a7703d92f9777266e8c34ccd3a7cf5963b354e57a7d9a18f227babee"
K4_SNAPSHOT_V2_SHA256 = "bc10d0555488b22f0bc8d54afe5afc5288064ddb4708bd1c75f3b55dd4c0060a"
VALID_RESULTS = {"pass", "fail", "not_run"}
REQUIRED_DEFINITION_FIELDS = {
    "contrast_std",
    "dataset_params_option",
    "grid",
    "image_offset_n_std",
    "n_images",
    "noise_level",
    "noise_model",
    "noise_scale_std",
    "pdb_bfactor",
    "percent_outliers",
    "put_extra_particles",
    "seed",
    "volume_radius",
}
REQUIRED_FIXTURE_FILES = {
    "ctf.pkl",
    "generation_config.json",
    "particles.star",
    "poses.pkl",
    "reference_gt.mrc",
    "reference_gt_relion.mrc",
    "reference_init.mrc",
    "reference_init_relion.mrc",
    "simulation_info.pkl",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")
GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")


class ProposalEvidence(NamedTuple):
    case_id: str
    case_root: Path
    science_job: str
    audit_job: str


def frozen_case_definitions_sha256(cases: list[dict]) -> str:
    """Hash only the immutable suite identity and fixture definitions."""

    manifest = [
        {
            "id": case.get("id"),
            "name": case.get("name"),
            "definition": case.get("definition"),
        }
        for case in cases
    ]
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def requires_final_grid_correction(scorecard: dict) -> bool:
    """Return whether new fixed-suite evidence must record final gridding correction on."""

    return scorecard["suite_version"] >= 2


def load_and_validate(path: Path) -> dict:
    scorecard = json.loads(path.read_text())
    schema = scorecard.get("schema")
    if schema not in SUITE_VERSION_BY_SCHEMA:
        raise ValueError("unsupported scorecard schema")
    suite_version = SUITE_VERSION_BY_SCHEMA[schema]
    if scorecard.get("suite_version") != suite_version:
        raise ValueError(f"{schema} scorecard must have suite_version={suite_version}")
    if scorecard.get("suite_id") != V1_SUITE_ID:
        raise ValueError(f"suite_id must remain {V1_SUITE_ID!r}")
    contract = scorecard.get("acceptance_contract")
    if not isinstance(contract, dict):
        raise ValueError("acceptance_contract must be an object")
    for name, value in FROZEN_THRESHOLDS.items():
        if contract.get(name) != value:
            raise ValueError(f"acceptance threshold {name} changed without an explicit decision")
    if contract.get("grid_correction") != GRID_CORRECTION_BY_SUITE_VERSION[suite_version]:
        raise ValueError(
            f"suite version {suite_version} grid_correction must be "
            f"{GRID_CORRECTION_BY_SUITE_VERSION[suite_version]!r}"
        )

    cases = scorecard.get("cases")
    if not isinstance(cases, list):
        raise ValueError("cases must be a list")
    denominator = scorecard.get("frozen_denominator")
    if denominator != V1_FROZEN_DENOMINATOR:
        raise ValueError(f"v1 frozen_denominator must remain {V1_FROZEN_DENOMINATOR}")
    if denominator != len(cases):
        raise ValueError(f"frozen_denominator={denominator} but found {len(cases)} cases")

    expected_ids = [f"k1-{index:02d}" for index in range(1, denominator + 1)]
    actual_ids = [case.get("id") for case in cases]
    if actual_ids != expected_ids:
        raise ValueError("case IDs must be ordered, contiguous, and frozen")
    names = [case.get("name") for case in cases]
    if len(set(names)) != len(names) or not all(isinstance(name, str) and name for name in names):
        raise ValueError("case names must be non-empty and unique")

    calculated_definition_sha256 = frozen_case_definitions_sha256(cases)
    recorded_definition_sha256 = scorecard.get("frozen_case_definitions_sha256")
    if recorded_definition_sha256 != V1_FROZEN_CASE_DEFINITIONS_SHA256:
        raise ValueError(
            "frozen case-definition digest changed without a new case set: "
            f"expected={V1_FROZEN_CASE_DEFINITIONS_SHA256} recorded={recorded_definition_sha256!r}"
        )
    if calculated_definition_sha256 != V1_FROZEN_CASE_DEFINITIONS_SHA256:
        raise ValueError(
            "frozen case definitions changed without a suite-version change: "
            f"expected={V1_FROZEN_CASE_DEFINITIONS_SHA256} calculated={calculated_definition_sha256}"
        )

    for case in cases:
        result = case.get("result")
        if result not in VALID_RESULTS:
            raise ValueError(f"{case['id']}: invalid result {result!r}")
        if case.get("intermediate_result") not in VALID_RESULTS:
            raise ValueError(f"{case['id']}: invalid intermediate_result")
        definition = case.get("definition")
        if not isinstance(definition, dict) or set(definition) != REQUIRED_DEFINITION_FIELDS:
            raise ValueError(f"{case['id']}: incomplete or expanded frozen definition")
        if not case.get("source_head") or not case.get("jobs"):
            raise ValueError(f"{case['id']}: missing immutable source/job evidence")

    calculated = Counter(case["result"] for case in cases)
    recorded = scorecard.get("current_snapshot", {}).get("counts", {})
    expected_counts = {status: calculated.get(status, 0) for status in ("pass", "fail", "not_run")}
    if recorded != expected_counts:
        raise ValueError(f"recorded counts {recorded} do not match cases {expected_counts}")

    history = scorecard.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("history must contain at least the current snapshot")
    history_ids = [snapshot.get("id") for snapshot in history]
    if len(set(history_ids)) != len(history_ids):
        raise ValueError("history snapshot IDs must be unique")
    for snapshot in history:
        counts = snapshot.get("counts")
        if not isinstance(counts, dict) or set(counts) != VALID_RESULTS:
            raise ValueError(f"{snapshot.get('id')}: invalid history counts")
        if any(not isinstance(value, int) or value < 0 for value in counts.values()):
            raise ValueError(f"{snapshot.get('id')}: history counts must be non-negative integers")
        if sum(counts.values()) != denominator:
            raise ValueError(f"{snapshot.get('id')}: history counts do not preserve frozen denominator")
        if not snapshot.get("source_heads") or not snapshot.get("evidence_sha256"):
            raise ValueError(f"{snapshot.get('id')}: missing immutable history evidence")
    if history[-1]["id"] != scorecard["current_snapshot"]["id"]:
        raise ValueError("last history row must be the current snapshot")
    if history[-1]["counts"] != recorded:
        raise ValueError("last history counts must match current snapshot")
    if suite_version >= 2:
        _validate_v2_regeneration(scorecard, path)

    case_ids = set(actual_ids)
    replicates = scorecard.get("replicate_diagnostics", [])
    if not isinstance(replicates, list):
        raise ValueError("replicate_diagnostics must be a list")
    for replicate in replicates:
        case_id = replicate.get("case_id")
        if case_id not in case_ids:
            raise ValueError(f"unknown replicate case ID: {case_id!r}")
        if replicate.get("scoring") is not False:
            raise ValueError(f"{case_id}: regenerated replicate must be explicitly non-scoring")
        if replicate.get("trajectory_result") not in {"pass", "fail"}:
            raise ValueError(f"{case_id}: invalid replicate trajectory result")
        if replicate.get("intermediate_result") not in {"pass", "fail"}:
            raise ValueError(f"{case_id}: invalid replicate intermediate result")
        for field in ("particle_stack_sha256", "fixed_fixture_particle_stack_sha256"):
            value = replicate.get(field)
            if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
                raise ValueError(f"{case_id}: invalid replicate {field}")
        if replicate["particle_stack_sha256"] == replicate["fixed_fixture_particle_stack_sha256"]:
            raise ValueError(f"{case_id}: replicate unexpectedly matches the fixed fixture bytes")
        jobs = replicate.get("jobs")
        if not isinstance(jobs, dict) or set(jobs) != {"science", "audit"}:
            raise ValueError(f"{case_id}: replicate jobs must identify science and audit")
    return scorecard


def _validate_v2_regeneration(scorecard: dict, path: Path) -> None:
    """Pin v2 to the untouched v1 history file and require per-case regeneration provenance."""

    previous = scorecard.get("previous_version")
    if not isinstance(previous, dict) or previous.get("schema") != SCHEMA_V1:
        raise ValueError("v2 scorecard must identify its v1 predecessor")
    previous_path = REPO_ROOT / str(previous.get("path"))
    if not previous_path.is_file() or sha256_file(previous_path) != previous.get("sha256"):
        raise ValueError("v1 history scorecard is missing or its bytes changed")
    v1 = json.loads(previous_path.read_text())
    if v1.get("current_snapshot", {}).get("id") != previous.get("current_snapshot_id"):
        raise ValueError("v2 predecessor snapshot differs from the v1 current snapshot")
    history_versions = [snapshot.get("suite_version") for snapshot in scorecard["history"]]
    if history_versions[-1] != 2 or any(version not in {1, 2} for version in history_versions):
        raise ValueError("v2 history rows must record suite_version 1 or 2, ending with 2")
    source = scorecard["current_snapshot"].get("source_ledger", {})
    ledger_path = source.get("path")
    if not isinstance(ledger_path, str) or not ledger_path:
        raise ValueError("v2 current snapshot must name its checked-in regeneration ledger")
    ledger_file = REPO_ROOT / ledger_path
    if not ledger_file.is_file() or sha256_file(ledger_file) != source.get("sha256"):
        raise ValueError("v2 regeneration ledger is missing or its bytes changed")
    v1_cases = {case["id"]: case for case in v1["cases"]}
    for case in scorecard["cases"]:
        v1_case = v1_cases[case["id"]]
        for field in ("name", "definition", "source_head", "jobs", "intermediate_result"):
            if case.get(field) != v1_case.get(field):
                raise ValueError(f"{case['id']}: v2 changed immutable v1 field {field}")
        suite_v1 = case.get("suite_v1")
        expected_v1 = {
            "result": v1_case["result"],
            "final_cross_engine_fsc_auc": v1_case["final_cross_engine_fsc_auc"],
            "final_gt_fsc_auc_delta": v1_case["final_gt_fsc_auc_delta"],
        }
        if suite_v1 != expected_v1:
            raise ValueError(f"{case['id']}: suite_v1 values differ from the v1 scorecard")
        provenance = case.get("final_metric_provenance")
        if not isinstance(provenance, dict):
            raise ValueError(f"{case['id']}: missing final-metric provenance")
        if provenance.get("method") != V2_REGENERATION_METHOD or provenance.get("script") != V2_REGENERATION_SCRIPT:
            raise ValueError(f"{case['id']}: final-metric provenance does not name the regeneration method")
        digests = provenance.get("inputs_sha256")
        if not isinstance(digests, dict) or set(digests) != V2_PROVENANCE_INPUTS:
            raise ValueError(f"{case['id']}: final-metric provenance lacks input digests")
        if any(not isinstance(value, str) or SHA256_RE.fullmatch(value) is None for value in digests.values()):
            raise ValueError(f"{case['id']}: invalid final-metric input digest")
        cross = case.get("final_cross_engine_fsc_auc")
        delta = case.get("final_gt_fsc_auc_delta")
        final_pass = (
            cross is not None
            and delta is not None
            and cross >= FROZEN_THRESHOLDS["merged_cross_engine_fsc_auc_min"]
            and delta >= FROZEN_THRESHOLDS["recovar_minus_relion_merged_gt_fsc_auc_min"]
        )
        if case["result"] == "pass" and not final_pass:
            raise ValueError(f"{case['id']}: v2 pass does not satisfy the frozen final-map thresholds")


def load_and_validate_fixture_manifest(path: Path, scorecard: dict) -> dict:
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "recovar.em_k1_fixture_manifest.v1":
        raise ValueError("unsupported fixture manifest schema")
    if manifest.get("suite_id") != V2_FIXTURE_SUITE_ID or manifest.get("suite_version") != 2:
        raise ValueError("fixture manifest must identify the artifact-pinned v2 suite")
    if manifest.get("frozen_denominator") != scorecard["frozen_denominator"]:
        raise ValueError("fixture manifest denominator differs from the scorecard")
    if manifest.get("frozen_case_definitions_sha256") != scorecard["frozen_case_definitions_sha256"]:
        raise ValueError("fixture manifest case-definition digest differs from the scorecard")

    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != scorecard["frozen_denominator"]:
        raise ValueError("fixture manifest must contain the frozen number of cases")
    expected_identity = [(case["id"], case["name"]) for case in scorecard["cases"]]
    actual_identity = [(case.get("id"), case.get("name")) for case in cases]
    if actual_identity != expected_identity:
        raise ValueError("fixture manifest case IDs/names differ from the scorecard")

    source_dirs = []
    for case in cases:
        case_id = case["id"]
        source_data_dir = case.get("source_data_dir")
        if not isinstance(source_data_dir, str) or not source_data_dir:
            raise ValueError(f"{case_id}: missing source_data_dir")
        source_path = Path(source_data_dir)
        if source_path.is_absolute() or ".." in source_path.parts:
            raise ValueError(f"{case_id}: source_data_dir must be root-relative and contained")
        source_dirs.append(source_data_dir)

        files = case.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError(f"{case_id}: fixture file list is empty")
        names = [row.get("name") for row in files]
        if len(names) != len(set(names)):
            raise ValueError(f"{case_id}: duplicate fixture filenames")
        if not REQUIRED_FIXTURE_FILES.issubset(names):
            raise ValueError(f"{case_id}: required fixture files are missing")
        stacks = [
            name for name in names if isinstance(name, str) and name.startswith("particles.") and name.endswith(".mrcs")
        ]
        if len(stacks) != 1:
            raise ValueError(f"{case_id}: expected exactly one particle stack")
        for row in files:
            name = row.get("name")
            if not isinstance(name, str) or not name or Path(name).name != name:
                raise ValueError(f"{case_id}: fixture filename must be a basename")
            if not isinstance(row.get("size"), int) or row["size"] <= 0:
                raise ValueError(f"{case_id}/{name}: invalid fixture size")
            if not isinstance(row.get("sha256"), str) or SHA256_RE.fullmatch(row["sha256"]) is None:
                raise ValueError(f"{case_id}/{name}: invalid SHA-256")
    if len(source_dirs) != len(set(source_dirs)):
        raise ValueError("fixture manifest source_data_dir values must be unique")
    actual_manifest_sha256 = sha256_file(path)
    if actual_manifest_sha256 != V2_FIXTURE_MANIFEST_SHA256:
        raise ValueError(
            "fixture manifest bytes changed without a suite-version change: "
            f"expected={V2_FIXTURE_MANIFEST_SHA256} actual={actual_manifest_sha256}"
        )
    return manifest


def load_and_validate_k4_snapshot(path: Path) -> dict:
    snapshot = json.loads(path.read_text())
    if snapshot.get("schema") != "em_k4_backend_trajectory_snapshot_v2":
        raise ValueError("unsupported K=4 snapshot schema")
    if sha256_file(path) != K4_SNAPSHOT_V2_SHA256:
        raise ValueError("K=4 snapshot bytes changed without a new pinned snapshot version")
    if snapshot.get("quality_metric_policy") != "shellwise FSC/FSC-AUC; correlation is not used":
        raise ValueError("K=4 snapshot must retain the FSC/FSC-AUC metric policy")
    iterations = snapshot.get("numbered_iterations")
    classes = snapshot.get("classes")
    pass_counts = snapshot.get("direct_fsc_auc_passes_by_iteration")
    if iterations != 15 or classes != 4:
        raise ValueError("K=4 v2 snapshot must retain 15 iterations and 4 classes")
    if (
        not isinstance(pass_counts, list)
        or len(pass_counts) != iterations
        or any(not isinstance(value, int) or not 0 <= value <= classes for value in pass_counts)
    ):
        raise ValueError("K=4 snapshot has invalid per-iteration pass counts")
    total = iterations * classes
    if snapshot.get("direct_fsc_auc_checks_total") != total:
        raise ValueError("K=4 snapshot direct-check denominator changed")
    if snapshot.get("direct_fsc_auc_checks_passed") != sum(pass_counts):
        raise ValueError("K=4 snapshot direct-check count does not match its iteration vector")
    if snapshot.get("iterations_all_classes_passed") != sum(value == classes for value in pass_counts):
        raise ValueError("K=4 snapshot all-class count does not match its iteration vector")
    if snapshot.get("direct_fsc_auc_gate") != 0.995:
        raise ValueError("K=4 snapshot direct FSC-AUC gate changed")
    if snapshot.get("exact_control_topology") is not True:
        raise ValueError("K=4 snapshot lacks exact control topology")
    if snapshot.get("same_physical_gpu_comparison") is not True:
        raise ValueError("K=4 snapshot is not a same-physical-GPU comparison")
    if snapshot.get("grid_correction") != "unset":
        raise ValueError("K=4 snapshot must keep grid correction unset")
    if snapshot.get("forced_final_all_data_after_nonconvergence") is not False:
        raise ValueError("K=4 snapshot forced invalid final all-data")
    return snapshot


def _frozen_mask_section(cases: list[dict]) -> list[str]:
    """Masked FSC of the fixed-suite fixtures with their frozen GT masks (reporting only)."""
    repo = str(Path(__file__).resolve().parents[1])
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from scripts.masked_fsc import DEFAULT_SCORES

    rows = {}
    for row in json.loads(DEFAULT_SCORES.read_text())["rows"]:
        match = re.fullmatch(r"fixed34_k1(\d\d)_ac5177d2_job(\d+)", row["label"])
        if match:
            rows[f"k1-{match.group(1)}"] = (row, match.group(2))
    lines = [
        "",
        "## Frozen-mask masked FSC (reporting only)",
        "",
        "Each fixture has one frozen mask made from its ground-truth map (registry `docs/benchmarks/frozen_masks.json`,",
        "method `docs/benchmarks/masked_fsc_method.md`). Maps are those of the autonomous run",
        "`em_k1_guigrid_localhighshell_full34_autonomous_ac5177d2_20260719T174000Z`; a case whose scorecard science job",
        "differs was superseded by a later run and its masked values describe the ac5177d2 maps. No gate reads these values.",
        "These values predate suite version 2 and were not regenerated with the final gridding correction.",
        "",
        "| Case | Maps job | Scorecard job? | RELION masked (Å) | relax masked (Å) | Masked AUC RELION / relax "
        "| Cross-engine masked AUC | GT masked AUC RELION / relax |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]

    def fmt(value, digits):
        return "—" if value is None else f"{value:.{digits}f}"

    for case in cases:
        if case["id"] not in rows:
            lines.append(f"| `{case['id']}` | — | — | — | — | — | — | — |")
            continue
        row, job = rows[case["id"]]
        engines = {e: row[e] or {} for e in ("relion", "relax")}
        cross = row["cross_engine_masked_band_auc"]
        gt = row["gt_masked_band_auc"] or {"relion": None, "relax": None}
        lines.append(
            f"| `{case['id']}` | {job} | {'yes' if job == case['jobs']['science'] else 'no'} | "
            f"{fmt(engines['relion'].get('masked_resolution_A'), 2)} | {fmt(engines['relax'].get('masked_resolution_A'), 2)} | "
            f"{fmt(engines['relion'].get('masked_corrected_band_auc'), 4)} / {fmt(engines['relax'].get('masked_corrected_band_auc'), 4)} | "
            f"{fmt(None if cross is None else cross['merged'], 4)} | {fmt(gt['relion'], 4)} / {fmt(gt['relax'], 4)} |"
        )
    return lines


def _grid_correction_phrase(scorecard: dict) -> str:
    if requires_final_grid_correction(scorecard):
        return "final all-data gridding correction always on (RELION `griddingCorrect`)"
    return "grid correction unset/off"


def _suite_v2_regeneration_section(scorecard: dict) -> list[str]:
    """Show how suite version 2 re-scored version 1's final merged maps."""

    if not requires_final_grid_correction(scorecard):
        return []
    previous = scorecard["previous_version"]
    regeneration = scorecard["regeneration"]
    changed = [case for case in scorecard["cases"] if case["result"] != case["suite_v1"]["result"]]
    changed_text = (
        "Pass/fail changed for: "
        + ", ".join(f"`{case['id']}` ({case['suite_v1']['result']} -> {case['result']})" for case in changed)
        + "."
        if changed
        else "No case changed pass/fail under the unchanged thresholds."
    )
    lines = [
        "",
        "## Suite version 2 regeneration",
        "",
        "Version 1 scored final merged maps written without RELION's final gridding correction "
        "(`backprojector.cpp` `griddingCorrect`); relax now always applies it. The correction is the last "
        "real-space step of the reconstruction, so each case's final merged-map metrics were "
        f"{regeneration['method']} (`{regeneration['script']}`): the saved uncorrected "
        "`final_merged.mrc` is divided by the radial sinc^2 (padding factor "
        f"{regeneration['gridding']['padding_factor']}) and re-scored with the audit's FSC shells and sign policy. "
        "Each case first reproduced its recorded version 1 values from the same files. Numbered-iteration "
        "metrics, split halves and RELION maps are unchanged. Version 1 remains as history in "
        f"`{previous['path']}` (SHA-256 `{previous['sha256']}`).",
        "",
        changed_text,
        "",
        "| Case | v1 result | v1 cross-engine FSC-AUC | v1 GT delta | v2 result | v2 cross-engine FSC-AUC | v2 GT delta |",
        "|---|---|---:|---:|---|---:|---:|",
    ]
    for case in scorecard["cases"]:
        v1 = case["suite_v1"]
        lines.append(
            f"| `{case['id']}` | {v1['result']} | {v1['final_cross_engine_fsc_auc']:.9f} | "
            f"{v1['final_gt_fsc_auc_delta']:+.9f} | {case['result']} | "
            f"{case['final_cross_engine_fsc_auc']:.9f} | {case['final_gt_fsc_auc_delta']:+.9f} |"
        )
    return lines


def render_markdown(
    scorecard: dict,
    fixture_manifest: dict,
    fixture_manifest_sha256: str,
    k4_snapshot: dict,
    k4_snapshot_sha256: str,
) -> str:
    cases = scorecard["cases"]
    counts = scorecard["current_snapshot"]["counts"]
    passed = counts["pass"]
    total = scorecard["frozen_denominator"]
    evaluated = passed + counts["fail"]
    intermediate_passed = sum(case["intermediate_result"] == "pass" for case in cases)
    source = scorecard["current_snapshot"]["source_ledger"]
    source_schema_match = re.fullmatch(
        r"em_k1_gui_grid0_local_highshell_full34_superseding_ledger_v(\d+)",
        source["schema"],
    )
    if source_schema_match is None:
        raise ValueError("current snapshot is not a superseding ledger")
    next_ledger_schema = (
        "em_k1_gui_grid0_local_highshell_full34_superseding_ledger_v"
        f"{int(source_schema_match.group(1)) + 1}"
    )
    fixture_bytes = sum(row["size"] for case in fixture_manifest["cases"] for row in case["files"])
    history = scorecard["history"]
    first_passed = history[0]["counts"]["pass"]
    previous_passed = history[-2]["counts"]["pass"] if len(history) > 1 else passed

    lines = [
        "# RECOVAR / RELION EM Parity Scorecard",
        "",
        f"**K=1 fixed-suite score: {passed} / {total} passing "
        f"({evaluated} / {total} evaluated; {intermediate_passed} / {total} intermediate-topology passes).**",
        "",
        f"**K=4 fixed-trajectory score: {k4_snapshot['direct_fsc_auc_checks_passed']} / "
        f"{k4_snapshot['direct_fsc_auc_checks_total']} direct class checks passing "
        f"({k4_snapshot['iterations_all_classes_passed']} / {k4_snapshot['numbered_iterations']} "
        "iterations pass all classes).**",
        "",
        f"Suite: `{scorecard['suite_id']}` (version {scorecard['suite_version']}; denominator frozen at {total}).",
        f"Frozen case-definition SHA-256: `{scorecard['frozen_case_definitions_sha256']}`.",
        "",
        "A checked box means the complete autonomous FSC/FSC-AUC trajectory contract passed. "
        "Unchecked cases remain in the denominator. New diagnostics do not enter this suite; changing "
        "the case set or scientific definitions requires a new suite version.",
        "",
        "The artifact-pinned fixture manifest is checked into the repository and binds all "
        f"{len(fixture_manifest['cases'])} cases ({fixture_bytes:,} bytes) to exact file sizes and SHA-256 "
        f"digests. Manifest SHA-256: `{fixture_manifest_sha256}`. Regenerated inputs are non-scoring "
        "replicates.",
        "",
        "Acceptance uses shellwise FSC and normalized FSC-AUC, exact schedule/topology, convergence/finalization "
        "semantics, same-physical-GPU RELION/RECOVAR pairs, "
        f"{_grid_correction_phrase(scorecard)}, and no forced K-class-like "
        "finalization. Correlation is not computed or gated.",
        "",
        f"Evidence snapshot: `{source['schema']}`, generated `{source['generated_utc']}`, JSON SHA-256 "
        f"`{source['sha256']}`" + (f" (`{source['path']}`)." if source.get("path") else "."),
        f"K=4 evidence snapshot: `{k4_snapshot['snapshot_id']}`, JSON SHA-256 "
        f"`{k4_snapshot_sha256}`.",
        f"Progress: {passed - first_passed:+d} passing cases since the first frozen snapshot; "
        f"{passed - previous_passed:+d} since the previous snapshot.",
        "",
        "## K=1 fixed cases",
        "",
        "| Done | Case | Fixture | Trajectory | Topology | Final cross-engine FSC-AUC | Final GT delta | Jobs |",
        "|---|---|---|---|---|---:|---:|---|",
    ]
    for case in cases:
        checked = "[x]" if case["result"] == "pass" else "[ ]"
        cross = case.get("final_cross_engine_fsc_auc")
        delta = case.get("final_gt_fsc_auc_delta")
        cross_text = "—" if cross is None else f"{cross:.9f}"
        delta_text = "—" if delta is None else f"{delta:+.9f}"
        jobs = case["jobs"]
        job_text = f"science {jobs['science']}; trajectory {jobs['trajectory']}; intermediate {jobs['intermediate']}"
        lines.append(
            f"| {checked} | `{case['id']}` | `{case['name']}` | {case['result']} | "
            f"{case['intermediate_result']} | {cross_text} | {delta_text} | {job_text} |"
        )

    lines += [
        "",
        "## K=4 fixed trajectory",
        "",
        f"Each row contains four class-level FSC-AUC checks at the frozen "
        f"`{k4_snapshot['direct_fsc_auc_gate']:.3f}` gate. A checked row passes all four classes; "
        "unchecked rows and failed class checks remain in their frozen denominators.",
        "",
        "| Done | Iteration | Class checks passed |",
        "|---|---:|---:|",
    ]
    for iteration, iteration_passes in enumerate(
        k4_snapshot["direct_fsc_auc_passes_by_iteration"],
        start=1,
    ):
        checked = "[x]" if iteration_passes == k4_snapshot["classes"] else "[ ]"
        lines.append(
            f"| {checked} | {iteration} | {iteration_passes} / {k4_snapshot['classes']} |"
        )

    lines += [
        "",
        "## Progress history",
        "",
        "| Snapshot | Suite | Date (UTC) | Commit boundary | Passed | Δ passed | Failed | Not evaluated/error |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    prior_passed = None
    for snapshot in history:
        snapshot_counts = snapshot["counts"]
        heads = ", ".join(f"`{head[:12]}`" for head in snapshot["source_heads"])
        delta_text = "—" if prior_passed is None else f"{snapshot_counts['pass'] - prior_passed:+d}"
        lines.append(
            f"| `{snapshot['id']}` | {snapshot.get('suite_version', scorecard['suite_version'])} | "
            f"{snapshot['recorded_utc']} | {heads} | "
            f"{snapshot_counts['pass']} | {delta_text} | {snapshot_counts['fail']} | "
            f"{snapshot_counts['not_run']} |"
        )
        prior_passed = snapshot_counts["pass"]
    lines += _suite_v2_regeneration_section(scorecard)
    lines += _frozen_mask_section(cases)
    lines += [
        "",
        "## Archived experiment history",
        "",
        "The [original post-snapshot diagnostics](https://github.com/ma-gilles/recovar-experiments/"
        "blob/6ced5f78857ad7cc75d5eee257ccbaa8dc17ffe7/docs/math/"
        "em_relion_parity_scorecard.md#current-k1-case-7-bounded-metric)",
        "retain the historical interventions, failed runs and evidence hashes. Their source-specific",
        "results do not qualify the current implementation or change the frozen suite above.",
    ]
    replicates = scorecard.get("replicate_diagnostics", [])
    if replicates:
        lines += [
            "",
            "## Non-scoring regenerated-data diagnostics",
            "",
            "These runs exercise the same parameter definitions with newly generated particle bytes. "
            "They are useful robustness evidence but never change the fixed-suite score.",
            "",
            "| Case | Trajectory | Topology | Final cross-engine FSC-AUC | Final GT delta | Jobs |",
            "|---|---|---|---:|---:|---|",
        ]
        for replicate in replicates:
            jobs = replicate["jobs"]
            lines.append(
                f"| `{replicate['case_id']}` | {replicate['trajectory_result']} | "
                f"{replicate['intermediate_result']} | {replicate['final_cross_engine_fsc_auc']:.9f} | "
                f"{replicate['final_gt_fsc_auc_delta']:+.9f} | "
                f"science {jobs['science']}; audit {jobs['audit']} |"
            )
    lines += [
        "",
        "Generate this PR-ready table with:",
        "",
        "```bash",
        "pixi run python scripts/summarize_em_relion_parity_scorecard.py",
        "```",
        "",
        "Verify that the checked scorecard and frozen snapshots are current with:",
        "",
        "```bash",
        "pixi run python scripts/summarize_em_relion_parity_scorecard.py \\",
        "  --check docs/math/em_relion_parity_scorecard.md",
        "```",
        "",
        "After a terminal strict auditor passes, build a fail-closed candidate",
        "superseding ledger with `--proposal-output`. The command validates the",
        "pinned fixture-manifest bytes and re-hashes every materialized byte, clean source and",
        "submitted job/case-table identity, same physical GPU, autonomous",
        "FSC/topology audits, convergence/finalization contract, and evidence",
        "hashes. It never mutates the checked scorecard. For example:",
        "",
        "```bash",
        "pixi run python scripts/summarize_em_relion_parity_scorecard.py \\",
        "  --proposal-previous-ledger /absolute/path/to/current-ledger.json \\",
        f"  --proposal-ledger-schema {next_ledger_schema} \\",
        "  --proposal-generated-utc 2026-07-26T21:00:00+00:00 \\",
        '  --proposal-status-note "Case k1-NN passed immutable strict evidence." \\',
        "  --proposal-evidence 'k1-NN|/absolute/path/to/case-root|SCIENCE_JOB|AUDIT_JOB' \\",
        "  --proposal-output /absolute/path/to/proposed-ledger.json",
        "```",
        "",
        "Launch a scoring rerun with `--scorecard`. This fail-closed mode requires the",
        "checked-in fixture manifest/root pair and forces autonomous RELION pairing,",
        "per-iteration RECOVAR maps and valid convergence-only finalization. Final",
        "all-data maps are always gridding-corrected; the launcher refuses the retired",
        "`RELAX_FINAL_ALL_DATA_GRID_CORRECT` selector. For example:",
        "",
        "```bash",
        'EM_K1_MATRIX_FIXTURE_MANIFEST="$PWD/docs/math/em_relion_parity_fixture_manifest_v2.json" \\',
        "EM_K1_MATRIX_FIXTURE_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex \\",
        "EM_K1_MATRIX_CASES=2,3 \\",
        "./scripts/run_em_k1_robustness_matrix_slurm.sh --scorecard",
        "```",
        "",
    ]
    return "\n".join(lines)


def parse_proposal_evidence(value: str) -> ProposalEvidence:
    """Parse CASE_ID|CASE_ROOT|SCIENCE_JOB|AUDIT_JOB proposal evidence."""

    fields = value.split("|")
    if len(fields) != 4 or any(not field for field in fields):
        raise argparse.ArgumentTypeError("proposal evidence must be CASE_ID|CASE_ROOT|SCIENCE_JOB|AUDIT_JOB")
    case_id, case_root, science_job, audit_job = fields
    if re.fullmatch(r"k1-\d{2}", case_id) is None:
        raise argparse.ArgumentTypeError(f"invalid fixed-suite case ID: {case_id!r}")
    for label, job_id in (("science", science_job), ("audit", audit_job)):
        if not job_id.isdigit():
            raise argparse.ArgumentTypeError(f"{label} job ID must contain only digits")
    root = Path(case_root)
    if not root.is_absolute():
        raise argparse.ArgumentTypeError("proposal case root must be absolute")
    return ProposalEvidence(case_id, root, science_job, audit_job)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json_file(path: Path) -> dict:
    _require(path.is_file(), f"missing JSON evidence: {path}")
    value = json.loads(path.read_text())
    _require(isinstance(value, dict), f"expected a JSON object: {path}")
    return value


def _same_definition_value(expected: str, actual: object) -> bool:
    if isinstance(actual, str) and expected == actual:
        return True
    try:
        return Decimal(expected) == Decimal(str(actual))
    except InvalidOperation:
        return False


def _validate_case_definition(case: dict, case_config: dict) -> None:
    config_names = {
        "grid": "grid_size",
        "contrast_std": "contrast_std",
        "dataset_params_option": "dataset_params_option",
        "image_offset_n_std": "image_offset_n_std",
        "n_images": "n_images",
        "noise_level": "noise_level",
        "noise_model": "noise_model",
        "noise_scale_std": "noise_scale_std",
        "pdb_bfactor": "pdb_bfactor",
        "percent_outliers": "percent_outliers",
        "put_extra_particles": "put_extra_particles",
        "seed": "seed",
        "volume_radius": "volume_radius",
    }
    for definition_name, config_name in config_names.items():
        _require(config_name in case_config, f"{case['id']}: case config lacks {config_name}")
        expected = case["definition"][definition_name]
        actual = case_config[config_name]
        _require(
            _same_definition_value(expected, actual),
            f"{case['id']}: frozen {definition_name}={expected!r} but case config has {actual!r}",
        )
    _require(case_config.get("trajectory_mode") == "autonomous", f"{case['id']}: trajectory is not autonomous")
    _require(case_config.get("run_relion") == 1, f"{case['id']}: RELION pairing was disabled")


def _validate_materialized_fixture(
    case: dict,
    case_root: Path,
    fixture_manifest: dict,
    fixture_manifest_sha256: str,
) -> None:
    materialization = _json_file(case_root / "data" / "fixture_materialization.json")
    _require(
        materialization.get("schema") == "recovar.em_k1_fixture_materialization.v1",
        f"{case['id']}: unsupported fixture materialization schema",
    )
    _require(materialization.get("case_id") == case["id"], f"{case['id']}: materialized case ID differs")
    _require(materialization.get("case_name") == case["name"], f"{case['id']}: materialized case name differs")
    _require(
        materialization.get("manifest_sha256") == fixture_manifest_sha256,
        f"{case['id']}: materialized fixture manifest SHA-256 differs",
    )
    manifest_cases = {row["id"]: row for row in fixture_manifest["cases"]}
    expected_case = manifest_cases[case["id"]]
    expected_files = {row["name"]: (row["size"], row["sha256"]) for row in expected_case["files"]}
    actual_rows = materialization.get("files")
    if not isinstance(actual_rows, list):
        raise ValueError(f"{case['id']}: materialization files are missing")
    _require(
        len(actual_rows) == len(expected_files) and all(isinstance(row, dict) for row in actual_rows),
        f"{case['id']}: materialization file rows are malformed or duplicated",
    )
    actual_files = {
        row.get("name"): (row.get("size"), row.get("sha256")) for row in actual_rows if isinstance(row, dict)
    }
    _require(len(actual_files) == len(actual_rows), f"{case['id']}: materialization contains duplicate filenames")
    _require(actual_files == expected_files, f"{case['id']}: materialized fixture identities differ")
    for name, (expected_size, expected_sha256) in expected_files.items():
        path = case_root / "data" / name
        _require(path.is_file(), f"{case['id']}: materialized fixture file is missing: {path}")
        _require(path.stat().st_size == expected_size, f"{case['id']}: fixture size changed: {path}")
        _require(
            sha256_file(path) == expected_sha256,
            f"{case['id']}: fixture SHA-256 changed: {path}",
        )


def _read_clean_source_head(run_root: Path, science_job: str, case_id: str) -> str:
    matches = sorted((run_root / "job_provenance").glob(f"*_{science_job}"))
    _require(len(matches) == 1, f"{case_id}: expected exactly one provenance directory for job {science_job}")
    provenance = matches[0]
    head = (provenance / "git_head.txt").read_text().strip()
    _require(GIT_SHA_RE.fullmatch(head) is not None, f"{case_id}: invalid source HEAD")
    _require(
        (provenance / "git_status_porcelain.txt").read_text() == "",
        f"{case_id}: science source worktree was dirty",
    )
    _require((provenance / "git_diff.patch").read_text() == "", f"{case_id}: science source had a patch")
    return head


def _validate_gpu_pair(case_root: Path, case_id: str) -> str:
    pair = _json_file(case_root / "paired_gpu_uuid.json")
    values = [
        pair.get("physical_gpu_uuid"),
        pair.get("relion_gpu_uuid"),
        pair.get("recovar_gpu_uuid"),
    ]
    _require(
        all(isinstance(value, str) and value.startswith("GPU-") for value in values),
        f"{case_id}: invalid physical GPU UUID evidence",
    )
    _require(len(set(values)) == 1, f"{case_id}: RELION and RECOVAR physical GPU UUIDs differ")
    paired_gpu_uuid = cast(str, values[0])
    for path in (
        case_root / "relion_ref" / "physical_gpu_uuid.txt",
        case_root / "relion_ref" / "post_physical_gpu_uuid.txt",
        case_root / "recovar" / "physical_gpu_uuid.txt",
        case_root / "recovar" / "runtime_physical_gpu_uuid.txt",
    ):
        _require(path.is_file(), f"{case_id}: missing physical GPU evidence: {path}")
        _require(path.read_text().strip() == paired_gpu_uuid, f"{case_id}: inconsistent physical GPU evidence: {path}")
    return paired_gpu_uuid


def _read_submission_env(run_root: Path) -> dict[str, str]:
    submission = {}
    for line in (run_root / "submission.env").read_text().splitlines():
        if "=" in line:
            name, value = line.split("=", 1)
            submission[name] = value
    return submission


def _validate_job_identity(
    run_root: Path,
    case_root: Path,
    case: dict,
    science_job: str,
    source_head: str,
) -> None:
    case_id = case["id"]
    submission = _read_submission_env(run_root)
    _require(submission.get("HEAD") == source_head, f"{case_id}: submission/source HEADs differ")
    submitted_jobs = submission.get("CASE_JOB_IDS", "").strip("'\"").split()
    _require(science_job in submitted_jobs, f"{case_id}: science job is absent from submission.env")

    with (run_root / "selected_cases.tsv").open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="|"))
    selected = [
        row
        for row in rows
        if row.get("case_root") and Path(cast(str, row["case_root"])).resolve() == case_root.resolve()
    ]
    _require(len(selected) == 1, f"{case_id}: expected one exact selected-cases row")
    row = selected[0]
    expected_index = str(int(case_id.removeprefix("k1-")))
    _require(row.get("index") == expected_index, f"{case_id}: selected-cases index differs")
    _require(row.get("name") == case["name"], f"{case_id}: selected-cases name differs")
    _require(row.get("case_job_id") == science_job, f"{case_id}: selected-cases science job differs")


def _validate_runtime_contract(
    run_root: Path,
    case_root: Path,
    case_id: str,
    *,
    grid_correction_required: bool,
) -> None:
    """Check the submitted runtime contract.

    Suite version 1 required the retired ``RELAX_FINAL_ALL_DATA_GRID_CORRECT``
    selector unset or off. From suite version 2 the final all-data pass always
    gridding-corrects and the selector is retired, so any recorded value marks a
    stale launcher; the run's ``refinement_results.npz`` must record the
    correction on (checked in ``build_proposal_update``).
    """

    submission = _read_submission_env(run_root)
    _require(
        submission.get("EM_K1_MATRIX_TRAJECTORY_MODE") == "autonomous",
        f"{case_id}: submission trajectory mode was not autonomous",
    )
    _require(submission.get("EM_K1_MATRIX_RUN_RELION") == "1", f"{case_id}: RELION was not enabled")
    grid_correction = submission.get(
        "RELAX_FINAL_ALL_DATA_GRID_CORRECT",
        "",
    ).strip().lower()
    if grid_correction_required:
        _require(
            grid_correction == "",
            f"{case_id}: submission records the retired RELAX_FINAL_ALL_DATA_GRID_CORRECT selector",
        )
    else:
        _require(
            grid_correction in {"", "0", "false", "no", "off"},
            f"{case_id}: final all-data grid correction was enabled",
        )
    job_scripts = sorted((run_root / "jobs").glob(f"em_k1_matrix_*{case_root.name}*.sh"))
    _require(len(job_scripts) == 1, f"{case_id}: could not identify the science job script")
    _require(
        "unset RELAX_FINAL_ALL_DATA_AFTER_MAX_ITER" in job_scripts[0].read_text(),
        f"{case_id}: science job did not fail closed on forced after-max finalization",
    )


def _validate_recorded_final_grid_correction(recorded: bool, case_id: str, *, required: bool) -> None:
    """Check ``final_all_data_grid_correct`` from the run's ``refinement_results.npz``."""

    if required:
        _require(recorded, f"{case_id}: final all-data maps were not gridding-corrected (suite version 2 requires it)")
    else:
        _require(not recorded, f"{case_id}: final all-data grid correction was enabled")


def _validate_autonomous_fsc_audit_schema(fsc: dict, case_id: str) -> None:
    """Require a current autonomous FSC audit for fixed-suite evidence."""

    _require(
        fsc.get("schema") == "em_k1_fsc_trajectory_audit_v3",
        f"{case_id}: wrong FSC audit schema",
    )
    start_iteration = fsc.get("relion_start_iteration")
    _require(
        type(start_iteration) is int and start_iteration == 1,
        f"{case_id}: fixed-suite FSC audit did not start at physical RELION iteration 1",
    )


def build_proposal_update(
    scorecard: dict,
    fixture_manifest: dict,
    fixture_manifest_sha256: str,
    evidence: ProposalEvidence,
) -> dict:
    """Build one fail-closed pass update from immutable fixed-suite evidence."""

    cases = {case["id"]: case for case in scorecard["cases"]}
    _require(evidence.case_id in cases, f"unknown fixed-suite case: {evidence.case_id}")
    case = cases[evidence.case_id]
    _require(case["result"] != "pass", f"{evidence.case_id}: case already passes the fixed suite")

    case_root = evidence.case_root.resolve()
    _require(case_root.is_dir(), f"{evidence.case_id}: missing case root: {case_root}")
    _require(case_root.parent.name == "cases", f"{evidence.case_id}: case root must be under a cases directory")
    run_root = case_root.parent.parent
    _require((run_root / "SAFE_TO_DELETE").is_file(), f"{evidence.case_id}: run root lacks SAFE_TO_DELETE")

    case_config = _json_file(case_root / "case_config.json")
    _require(case_config.get("name") == case["name"], f"{evidence.case_id}: case config name differs")
    _validate_case_definition(case, case_config)
    _validate_materialized_fixture(case, case_root, fixture_manifest, fixture_manifest_sha256)
    source_head = _read_clean_source_head(run_root, evidence.science_job, evidence.case_id)
    _validate_job_identity(run_root, case_root, case, evidence.science_job, source_head)
    paired_gpu_uuid = _validate_gpu_pair(case_root, evidence.case_id)
    grid_correction_required = requires_final_grid_correction(scorecard)
    _validate_runtime_contract(
        run_root,
        case_root,
        evidence.case_id,
        grid_correction_required=grid_correction_required,
    )

    audit_logs = sorted((run_root / "audits").glob(f"*{evidence.case_id}*_{evidence.audit_job}.out"))
    _require(len(audit_logs) == 1, f"{evidence.case_id}: expected one audit log for job {evidence.audit_job}")
    audit_status = (case_root / "trajectory_analysis" / "audit_status.txt").read_text().splitlines()
    _require(
        audit_status == ["fsc_status=0", "intermediate_status=0"],
        f"{evidence.case_id}: strict audit status is not fully passing",
    )

    analysis = case_root / "trajectory_analysis"
    fsc_path = analysis / "k1_fsc_trajectory.json"
    topology_path = analysis / "k1_intermediate_trajectory.json"
    shellwise_path = analysis / "k1_fsc_trajectory_shellwise.npz"
    fsc = _json_file(fsc_path)
    topology = _json_file(topology_path)
    _validate_autonomous_fsc_audit_schema(fsc, evidence.case_id)
    _require(fsc.get("status") == "pass", f"{evidence.case_id}: FSC trajectory did not pass")
    _require(fsc.get("failures") == [], f"{evidence.case_id}: FSC audit reports failures")
    _require(fsc.get("topology_failures") == [], f"{evidence.case_id}: FSC audit reports topology failures")
    _require(
        fsc.get("thresholds")
        == {
            "merged_cross_engine_fsc_auc_min": scorecard["acceptance_contract"]["merged_cross_engine_fsc_auc_min"],
            "recovar_minus_relion_merged_gt_fsc_auc_min": scorecard["acceptance_contract"][
                "recovar_minus_relion_merged_gt_fsc_auc_min"
            ],
        },
        f"{evidence.case_id}: FSC audit thresholds differ from the frozen contract",
    )
    _require(
        fsc.get("gt_sign_policy", {}).get("used") == "signed",
        f"{evidence.case_id}: FSC audit did not use signed curves",
    )
    _require(
        topology.get("schema") == "em_k1_intermediate_trajectory_audit_v1",
        f"{evidence.case_id}: wrong topology audit schema",
    )
    _require(topology.get("status") == "pass", f"{evidence.case_id}: intermediate topology did not pass")
    _require(topology.get("topology_failures") == [], f"{evidence.case_id}: topology audit reports failures")
    _require(
        topology.get("numeric_artifact_failures") == [],
        f"{evidence.case_id}: intermediate audit reports invalid numeric artifacts",
    )

    numbered = fsc.get("numbered_iterations")
    topology_numbered = topology.get("numbered_iterations")
    if not isinstance(numbered, list) or not numbered:
        raise ValueError(f"{evidence.case_id}: missing numbered FSC trajectory")
    if not isinstance(topology_numbered, list) or len(topology_numbered) != len(numbered):
        raise ValueError(f"{evidence.case_id}: FSC/topology iteration counts differ")
    numbered_fsc = [float(row["cross_engine"]["merged"]["fsc_auc"]) for row in numbered]
    worst_index = min(range(len(numbered_fsc)), key=numbered_fsc.__getitem__)
    worst_iteration = int(numbered[worst_index]["relion_iteration"])
    current_sizes = [int(row["topology"]["current_size"]["recovar"]) for row in topology_numbered]
    for fsc_row, row in zip(numbered, topology_numbered, strict=True):
        _require(
            row["relion_iteration"] == fsc_row["relion_iteration"],
            f"{evidence.case_id}: FSC/topology iteration identities differ",
        )
        _require(
            row["topology"]["current_size"]["exact_equal"] is True
            and row["topology"]["healpix_order"]["exact_equal"] is True,
            f"{evidence.case_id}: selected topology is not exact",
        )

    final_cross = float(fsc["final"]["cross_engine"]["merged"]["fsc_auc"])
    final_delta = float(fsc["final"]["merged_gt_fsc_auc_delta"])
    contract = scorecard["acceptance_contract"]
    _require(
        final_cross >= contract["merged_cross_engine_fsc_auc_min"],
        f"{evidence.case_id}: final cross-engine FSC-AUC is below the frozen threshold",
    )
    _require(
        final_delta >= contract["recovar_minus_relion_merged_gt_fsc_auc_min"],
        f"{evidence.case_id}: final GT FSC-AUC delta is below the frozen threshold",
    )

    import numpy as np

    results_path = case_root / "recovar" / "refinement_results.npz"
    with np.load(results_path, allow_pickle=False) as results:
        required = {
            "convergence_has_converged",
            "convergence_iteration",
            "final_all_data_ran",
            "final_all_data_grid_correct",
            "current_sizes",
        }
        _require(required.issubset(results.files), f"{evidence.case_id}: incomplete refinement finalization evidence")
        converged = bool(np.asarray(results["convergence_has_converged"]).reshape(()))
        convergence_iteration = int(np.asarray(results["convergence_iteration"]).reshape(()))
        final_all_data = bool(np.asarray(results["final_all_data_ran"]).reshape(()))
        grid_correction = bool(np.asarray(results["final_all_data_grid_correct"]).reshape(()))
        saved_current_sizes = [int(value) for value in np.asarray(results["current_sizes"]).reshape(-1)]
    _require(converged, f"{evidence.case_id}: RECOVAR did not converge")
    _require(
        convergence_iteration == len(numbered),
        f"{evidence.case_id}: convergence iteration differs from the audited trajectory",
    )
    _require(final_all_data, f"{evidence.case_id}: converged run lacks final all-data")
    _validate_recorded_final_grid_correction(grid_correction, evidence.case_id, required=grid_correction_required)
    _require(saved_current_sizes == current_sizes, f"{evidence.case_id}: saved/audited current-size schedules differ")
    _require(shellwise_path.is_file(), f"{evidence.case_id}: missing shellwise FSC evidence")

    return {
        "case_id": case["id"],
        "case_name": case["name"],
        "result": "pass",
        "intermediate_result": "pass",
        "final_cross_engine_fsc_auc": final_cross,
        "final_gt_fsc_auc_delta": final_delta,
        "worst_numbered_cross_engine_fsc_auc": numbered_fsc[worst_index],
        "worst_numbered_iteration": worst_iteration,
        "numbered_iterations": len(numbered),
        "current_sizes": current_sizes,
        "converged": converged,
        "convergence_iteration": convergence_iteration,
        "final_all_data": final_all_data,
        "grid_correction": grid_correction,
        "source_head": source_head,
        "source_status": "clean",
        "fixture_manifest_sha256": fixture_manifest_sha256,
        "paired_gpu_uuid": paired_gpu_uuid,
        "jobs": {
            "science": evidence.science_job,
            "trajectory": evidence.audit_job,
            "intermediate": evidence.audit_job,
        },
        "run_root": str(case_root),
        "evidence_sha256": {
            "fsc_json": sha256_file(fsc_path),
            "topology_json": sha256_file(topology_path),
            "shellwise_npz": sha256_file(shellwise_path),
        },
    }


def build_superseding_ledger(
    scorecard: dict,
    fixture_manifest: dict,
    fixture_manifest_sha256: str,
    previous_ledger_path: Path,
    ledger_schema: str,
    generated_utc: str,
    evidence_rows: list[ProposalEvidence],
    status_note: str,
) -> dict:
    """Validate pass evidence and construct a deterministic superseding ledger."""

    schema_match = re.fullmatch(
        r"em_k1_gui_grid0_local_highshell_full34_superseding_ledger_v(\d+)",
        ledger_schema,
    )
    _require(schema_match is not None, "invalid superseding ledger schema")
    try:
        generated_at = datetime.fromisoformat(generated_utc)
    except ValueError as error:
        raise ValueError("generated UTC must be a valid explicit +00:00 timestamp") from error
    _require(
        generated_utc.endswith("+00:00") and generated_at.utcoffset() == timedelta(0),
        "generated UTC must be an explicit +00:00 timestamp",
    )
    _require(bool(status_note.strip()), "proposal status note must be non-empty")
    current_source = scorecard["current_snapshot"]["source_ledger"]
    current_schema_match = re.fullmatch(
        r"em_k1_gui_grid0_local_highshell_full34_superseding_ledger_v(\d+)",
        current_source["schema"],
    )
    _require(current_schema_match is not None, "current snapshot is not a superseding ledger")
    _require(
        int(cast(re.Match[str], schema_match).group(1)) > int(cast(re.Match[str], current_schema_match).group(1)),
        "proposal ledger version must advance the current snapshot",
    )
    _require(previous_ledger_path.is_file(), f"missing previous ledger: {previous_ledger_path}")
    _require(
        sha256_file(previous_ledger_path) == current_source["sha256"],
        "previous ledger SHA-256 differs from the scorecard current snapshot",
    )
    previous = _json_file(previous_ledger_path)
    _require(previous.get("schema") == current_source["schema"], "previous ledger schema differs")
    if not evidence_rows:
        raise ValueError("at least one pass evidence row is required")
    case_ids = [row.case_id for row in evidence_rows]
    _require(len(case_ids) == len(set(case_ids)), "proposal contains duplicate case IDs")
    updates = [
        build_proposal_update(scorecard, fixture_manifest, fixture_manifest_sha256, row) for row in evidence_rows
    ]

    strict_counts = Counter(case["result"] for case in scorecard["cases"])
    topology_counts = Counter(case["intermediate_result"] for case in scorecard["cases"])
    cases = {case["id"]: case for case in scorecard["cases"]}
    for update in updates:
        prior = cases[update["case_id"]]
        strict_counts[prior["result"]] -= 1
        strict_counts["pass"] += 1
        topology_counts[prior["intermediate_result"]] -= 1
        topology_counts["pass"] += 1
    return {
        "schema": ledger_schema,
        "generated_utc": generated_utc,
        "suite_id": scorecard["suite_id"],
        "frozen_denominator": scorecard["frozen_denominator"],
        "frozen_case_definitions_sha256": scorecard["frozen_case_definitions_sha256"],
        "fixture_manifest_sha256": fixture_manifest_sha256,
        "supersedes": {
            "schema": current_source["schema"],
            "sha256": current_source["sha256"],
        },
        "acceptance_contract": scorecard["acceptance_contract"],
        "counts": {
            "strict": {status: strict_counts.get(status, 0) for status in ("pass", "fail", "not_run")},
            "topology": {status: topology_counts.get(status, 0) for status in ("pass", "fail", "not_run")},
        },
        "updates": updates,
        "status_note": status_note,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorecard", type=Path, default=DEFAULT_SCORECARD)
    parser.add_argument("--fixture-manifest", type=Path, default=DEFAULT_FIXTURE_MANIFEST)
    parser.add_argument("--k4-snapshot", type=Path, default=DEFAULT_K4_SNAPSHOT)
    parser.add_argument("--check", type=Path, help="fail if this generated Markdown file is stale")
    parser.add_argument(
        "--proposal-evidence",
        action="append",
        type=parse_proposal_evidence,
        default=[],
        metavar="CASE_ID|CASE_ROOT|SCIENCE_JOB|AUDIT_JOB",
        help="validated fixed-suite pass evidence; repeat for multiple cases",
    )
    parser.add_argument("--proposal-previous-ledger", type=Path)
    parser.add_argument("--proposal-ledger-schema")
    parser.add_argument("--proposal-generated-utc")
    parser.add_argument("--proposal-status-note")
    parser.add_argument("--proposal-output", type=Path)
    args = parser.parse_args()

    scorecard = load_and_validate(args.scorecard)
    fixture_manifest = load_and_validate_fixture_manifest(args.fixture_manifest, scorecard)
    k4_snapshot = load_and_validate_k4_snapshot(args.k4_snapshot)
    if args.proposal_output is not None:
        required = {
            "--proposal-previous-ledger": args.proposal_previous_ledger,
            "--proposal-ledger-schema": args.proposal_ledger_schema,
            "--proposal-generated-utc": args.proposal_generated_utc,
            "--proposal-status-note": args.proposal_status_note,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing or not args.proposal_evidence:
            missing += [] if args.proposal_evidence else ["--proposal-evidence"]
            parser.error("proposal mode requires " + ", ".join(missing))
        if args.proposal_output.exists():
            raise SystemExit(f"refusing to overwrite existing proposal ledger: {args.proposal_output}")
        ledger = build_superseding_ledger(
            scorecard,
            fixture_manifest,
            sha256_file(args.fixture_manifest),
            args.proposal_previous_ledger,
            args.proposal_ledger_schema,
            args.proposal_generated_utc,
            args.proposal_evidence,
            args.proposal_status_note,
        )
        args.proposal_output.parent.mkdir(parents=True, exist_ok=True)
        args.proposal_output.write_text(json.dumps(ledger, indent=2) + "\n")
        print(f"wrote validated superseding-ledger proposal: {args.proposal_output}")
        print(f"proposal SHA-256: {sha256_file(args.proposal_output)}")
        return 0
    rendered = render_markdown(
        scorecard,
        fixture_manifest,
        sha256_file(args.fixture_manifest),
        k4_snapshot,
        sha256_file(args.k4_snapshot),
    )
    if args.check is not None:
        checked_text = args.check.read_text()
        if checked_text != rendered:
            raise SystemExit(f"stale generated scorecard: {args.check}")
        print(f"scorecard valid and current: {args.check}")
        return 0
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
