from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from scripts import regenerate_em_k1_scorecard_final_gridding as regen

REPO_ROOT = Path(__file__).resolve().parents[2]
V1_SCORECARD = REPO_ROOT / "docs" / "math" / "em_relion_parity_scorecard_v1.json"


def _sinc2(n: int, padding_factor: int = 2) -> np.ndarray:
    coords = np.arange(n, dtype=np.float64) - n / 2.0
    r = np.sqrt(coords[:, None, None] ** 2 + coords[None, :, None] ** 2 + coords[None, None, :] ** 2)
    return np.sinc(r / (n * padding_factor)) ** 2


@pytest.mark.unit
def test_gridding_correction_divides_by_relion_radial_sinc_squared():
    rng = np.random.default_rng(3)
    volume = rng.normal(size=(8, 8, 8))

    corrected = regen.gridding_corrected_saved_map(volume)

    expected = (volume / _sinc2(8)).astype(np.float32).astype(np.float64)
    assert_matches(corrected, expected)
    assert corrected[4, 4, 4] == np.float32(volume[4, 4, 4])
    # Radial about voxel N/2, so the saved-file axis transpose commutes with the correction.
    assert_matches(
        regen.gridding_corrected_saved_map(volume.transpose(2, 1, 0)), corrected.transpose(2, 1, 0)
    )


def _write_case(tmp_path: Path, monkeypatch, *, grid_correct_recorded: bool, failures=()) -> tuple[Path, dict]:
    rng = np.random.default_rng(11)
    n = 12
    gt = rng.normal(size=(n, n, n))
    relion = gt + 0.3 * rng.normal(size=(n, n, n))
    uncorrected = relion * _sinc2(n)
    recovar_dir = tmp_path / "recovar"
    relion_dir = tmp_path / "relion_ref"
    recovar_dir.mkdir()
    relion_dir.mkdir()
    arrays = {
        (recovar_dir / "final_merged.mrc").resolve(): uncorrected,
        (relion_dir / "run_class001.mrc").resolve(): relion,
        (tmp_path / "reference_gt.mrc").resolve(): gt,
    }
    for path in arrays:
        path.write_bytes(b"map")
    np.savez(
        recovar_dir / "refinement_results.npz",
        final_all_data_grid_correct=np.bool_(grid_correct_recorded),
    )

    def load(path):
        return arrays[Path(path).resolve()]

    monkeypatch.setattr(regen, "_load_recovar_volume", load)
    monkeypatch.setattr(regen, "_load_relion_volume", load)
    recorded = regen._final_metrics(uncorrected, relion, gt, gt_sign_invariant=False)
    report = {
        "status": "fail" if failures else "pass",
        "failures": list(failures),
        "paths": {
            "recovar_dir": str(recovar_dir),
            "relion_dir": str(relion_dir),
            "gt_volume": str(tmp_path / "reference_gt.mrc"),
        },
        "gt_sign_policy": {"used": "signed"},
        "final": {
            "cross_engine": {"merged": {"fsc_auc": recorded["final_cross_engine_fsc_auc"]}},
            "merged_gt_fsc_auc_delta": recorded["final_gt_fsc_auc_delta"],
        },
    }
    report_path = tmp_path / "k1_fsc_trajectory.json"
    report_path.write_text(json.dumps(report))
    return report_path, recorded


def _regenerate(report_path: Path, **overrides):
    kwargs = dict(min_cross=0.995, min_delta=-0.002, reproduction_tolerance=1e-12, check_final_mean=False)
    kwargs.update(overrides)
    return regen.regenerate_case("k1-99", report_path, **kwargs)


@pytest.mark.unit
def test_regeneration_reproduces_then_rescores_the_corrected_merged_map(tmp_path, monkeypatch):
    report_path, recorded = _write_case(tmp_path, monkeypatch, grid_correct_recorded=False)

    row = _regenerate(report_path)

    assert row["reproduction_max_abs_discrepancy"] == 0.0
    assert row["route"].startswith("post-hoc RELION griddingCorrect")
    # The corrected map is the RELION map up to float32 rounding.
    assert row["regenerated"]["final_cross_engine_fsc_auc"] == pytest.approx(1.0, abs=1e-6)
    assert row["regenerated"]["final_gt_fsc_auc_delta"] == pytest.approx(0.0, abs=1e-6)
    assert recorded["final_cross_engine_fsc_auc"] < row["regenerated"]["final_cross_engine_fsc_auc"]
    assert row["result"] == "pass"
    assert set(row["inputs_sha256"]) == {
        "fsc_report",
        "recovar_final_merged",
        "recovar_refinement_results",
        "relion_final_merged",
        "gt_volume",
    }


@pytest.mark.unit
def test_regeneration_leaves_already_corrected_runs_unchanged(tmp_path, monkeypatch):
    report_path, recorded = _write_case(tmp_path, monkeypatch, grid_correct_recorded=True)

    row = _regenerate(report_path)

    assert row["route"].startswith("unchanged")
    assert row["regenerated"]["final_cross_engine_fsc_auc"] == recorded["final_cross_engine_fsc_auc"]
    assert row["regenerated"]["final_gt_fsc_auc_delta"] == recorded["final_gt_fsc_auc_delta"]


@pytest.mark.unit
def test_regeneration_refuses_maps_that_do_not_reproduce_the_record(tmp_path, monkeypatch):
    report_path, _ = _write_case(tmp_path, monkeypatch, grid_correct_recorded=False)
    report = json.loads(report_path.read_text())
    report["final"]["merged_gt_fsc_auc_delta"] += 1e-6
    report_path.write_text(json.dumps(report))

    with pytest.raises(regen.RegenerationError, match="do not reproduce"):
        _regenerate(report_path)


@pytest.mark.unit
def test_regeneration_keeps_numbered_failures_and_replaces_final_ones(tmp_path, monkeypatch):
    report_path, _ = _write_case(
        tmp_path,
        monkeypatch,
        grid_correct_recorded=False,
        failures=(
            "final merged cross-engine FSC-AUC 0.900000000 < 0.995000000",
            "it003 merged cross-engine FSC-AUC 0.990000000 < 0.995000000",
        ),
    )

    row = _regenerate(report_path)

    assert row["final_failures"] == []
    assert row["non_final_failures"] == ["it003 merged cross-engine FSC-AUC 0.990000000 < 0.995000000"]
    assert row["result"] == "fail"


@pytest.mark.unit
def test_final_gates_use_the_unchanged_thresholds():
    assert regen.final_gate_failures(0.995, -0.002, min_cross=0.995, min_delta=-0.002) == []
    assert regen.final_gate_failures(0.9949, 0.0, min_cross=0.995, min_delta=-0.002) == [
        "final merged cross-engine FSC-AUC 0.994900000 < 0.995000000"
    ]
    assert regen.final_gate_failures(1.0, -0.0021, min_cross=0.995, min_delta=-0.002) == [
        "final merged GT FSC-AUC delta -0.002100000 < -0.002000000"
    ]


def _rows(v1: dict, result_for=lambda case_id: "pass") -> list[dict]:
    return [
        {
            "case_id": case["id"],
            "route": "post-hoc",
            "recorded_final_all_data_grid_correct": False,
            "reproduction_max_abs_discrepancy": 0.0,
            "regenerated": {"final_cross_engine_fsc_auc": 0.999, "final_gt_fsc_auc_delta": 0.0},
            "result": result_for(case["id"]),
            "inputs": {"fsc_report": "/x"},
            "inputs_sha256": {"fsc_report": "0" * 64},
        }
        for case in v1["cases"]
    ]


@pytest.mark.unit
def test_v2_build_supersedes_v1_and_records_provenance():
    v1 = json.loads(V1_SCORECARD.read_text())
    rows = _rows(v1, result_for=lambda case_id: "fail" if case_id == "k1-04" else "pass")
    ledger = regen.build_superseding_ledger(
        v1,
        v1["current_snapshot"]["source_ledger"]["sha256"],
        rows,
        ledger_schema="em_k1_gui_grid0_local_highshell_full34_superseding_ledger_v13",
        generated_utc="2026-09-24T00:00:00+00:00",
        fixture_manifest_sha256="f" * 64,
    )
    scorecard = regen.build_v2_scorecard(
        v1,
        "docs/math/v1.json",
        "a" * 64,
        ledger,
        "docs/math/ledger.json",
        "b" * 64,
        snapshot_id="strict-k1-v2-test",
        source_heads=["c" * 40],
    )

    assert ledger["counts"]["strict"] == {"pass": 33, "fail": 1, "not_run": 0}
    assert ledger["acceptance_contract"]["grid_correction"] == regen.V2_GRID_CORRECTION
    assert ledger["acceptance_contract"]["merged_cross_engine_fsc_auc_min"] == 0.995
    assert ledger["acceptance_contract"]["recovar_minus_relion_merged_gt_fsc_auc_min"] == -0.002
    assert scorecard["suite_version"] == 2
    assert scorecard["history"][-1]["id"] == scorecard["current_snapshot"]["id"] == "strict-k1-v2-test"
    assert [row["suite_version"] for row in scorecard["history"]] == [1] * len(v1["history"]) + [2]
    case = scorecard["cases"][3]
    assert case["suite_v1"]["result"] == "fail"
    assert case["final_metric_provenance"]["method"] == regen.REGENERATION_METHOD
    assert case["definition"] == v1["cases"][3]["definition"]


@pytest.mark.unit
def test_v2_build_refuses_a_ledger_that_does_not_advance_the_chain():
    v1 = json.loads(V1_SCORECARD.read_text())
    with pytest.raises(regen.RegenerationError, match="must advance"):
        regen.build_superseding_ledger(
            v1,
            v1["current_snapshot"]["source_ledger"]["sha256"],
            _rows(v1),
            ledger_schema="em_k1_gui_grid0_local_highshell_full34_superseding_ledger_v14",
            generated_utc="2026-09-24T00:00:00+00:00",
            fixture_manifest_sha256="f" * 64,
        )
    with pytest.raises(regen.RegenerationError, match="previous ledger"):
        regen.build_superseding_ledger(
            v1,
            "0" * 64,
            _rows(v1),
            ledger_schema="em_k1_gui_grid0_local_highshell_full34_superseding_ledger_v13",
            generated_utc="2026-09-24T00:00:00+00:00",
            fixture_manifest_sha256="f" * 64,
        )
