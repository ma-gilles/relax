import json
from pathlib import Path

import pytest

from scripts import report_refinement_structure

REPO_ROOT = Path(__file__).resolve().parents[2]
CEILINGS = REPO_ROOT / "docs" / "development" / "refinement_structure_metrics.json"
TOTALS = {
    "maximum_function_line_span": 900,
    "maximum_parameter_count": 30,
    "large_argument_function_count": 12,
    "very_large_argument_function_count": 3,
    "physical_lines": 10_000,
    "nonblank_noncomment_lines": 8_000,
}


def test_refinement_structure_stays_within_recorded_ceilings():
    totals = report_refinement_structure.collect_metrics(REPO_ROOT)["totals"]
    ceilings = report_refinement_structure.read_ceilings(CEILINGS)

    assert set(ceilings) == set(TOTALS)
    assert report_refinement_structure.exceeded(totals, ceilings) == [], (
        "relax/refinement grew past a recorded ceiling: shrink it, or raise the ceiling by hand in "
        f"{CEILINGS.relative_to(REPO_ROOT)} with the reason in the commit message"
    )


def test_refinement_structure_scope_counts_every_refinement_python_module():
    metrics = report_refinement_structure.collect_metrics(REPO_ROOT)
    expected_paths = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "relax" / "refinement").glob("*.py")
        if path.name not in report_refinement_structure.EXCLUDED_FILENAMES
    }

    assert {row["path"] for row in metrics["files"]} == expected_paths


def test_only_a_total_above_its_ceiling_is_reported():
    ceilings = dict(TOTALS)
    assert report_refinement_structure.exceeded(TOTALS, ceilings) == []
    smaller = {name: value - 1 for name, value in TOTALS.items()}
    assert report_refinement_structure.exceeded(smaller, ceilings) == []
    grown = dict(TOTALS, maximum_function_line_span=901, file_count=99, function_count=999)
    assert report_refinement_structure.exceeded(grown, ceilings) == [
        "maximum_function_line_span: 901 exceeds the ceiling 900"
    ]


def test_line_totals_get_headroom_and_structure_totals_none():
    assert report_refinement_structure.ceilings_for(TOTALS) == dict(
        TOTALS, physical_lines=10_500, nonblank_noncomment_lines=8_400
    )
    assert report_refinement_structure.ceilings_for(dict(TOTALS, physical_lines=10_001))["physical_lines"] == 10_600


def test_lowering_tightens_and_never_raises():
    ceilings = report_refinement_structure.ceilings_for(TOTALS)
    cleaned = dict(TOTALS, maximum_function_line_span=400, physical_lines=9_000)
    grown = dict(TOTALS, maximum_parameter_count=40, nonblank_noncomment_lines=20_000)

    assert report_refinement_structure.lowered(ceilings, cleaned) == dict(
        ceilings, maximum_function_line_span=400, physical_lines=9_500
    )
    assert report_refinement_structure.lowered(ceilings, grown) == ceilings


def _write(path, ceilings):
    path.write_text(json.dumps({"schema_version": 2, "baseline": {"kept": True}, "ceilings": {"totals": ceilings}}))


def test_lower_ceilings_rewrites_the_file_only_when_a_bound_drops(tmp_path):
    path = tmp_path / "ceilings.json"
    ceilings = report_refinement_structure.ceilings_for(TOTALS)
    _write(path, ceilings)
    before = path.read_text()

    assert report_refinement_structure.lower_ceilings(path, TOTALS, tmp_path) == []
    assert report_refinement_structure.lower_ceilings(path, dict(TOTALS, maximum_parameter_count=99), tmp_path) == []
    assert path.read_text() == before

    changes = report_refinement_structure.lower_ceilings(path, dict(TOTALS, maximum_parameter_count=20), tmp_path)
    assert changes == ["maximum_parameter_count: 30 -> 20"]
    payload = json.loads(path.read_text())
    assert payload["baseline"] == {"kept": True}
    assert payload["ceilings"]["totals"] == dict(ceilings, maximum_parameter_count=20)


def test_check_and_lower_modes_exit_nonzero_above_a_ceiling(tmp_path, capsys):
    totals = report_refinement_structure.collect_metrics(REPO_ROOT)["totals"]
    path = tmp_path / "ceilings.json"
    _write(path, dict(report_refinement_structure.ceilings_for(totals), maximum_parameter_count=1))
    before = path.read_text()

    assert report_refinement_structure.main(["--check", str(path)]) == 1
    assert "maximum_parameter_count" in capsys.readouterr().err
    assert report_refinement_structure.main(["--lower-ceilings", str(path)]) == 1
    assert path.read_text() == before
    assert report_refinement_structure.main(["--check", str(CEILINGS), "--format", "markdown"]) == 0
    assert "| Largest function span |" in capsys.readouterr().out


def test_a_snapshot_of_current_values_is_refused(tmp_path):
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps({"schema_version": 1, "current": {"metrics": {}}}))
    with pytest.raises(ValueError, match="unsupported refinement structure ceilings"):
        report_refinement_structure.read_ceilings(path)
