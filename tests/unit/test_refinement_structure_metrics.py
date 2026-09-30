from pathlib import Path

from scripts import report_refinement_structure

REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = REPO_ROOT / "docs" / "development" / "refinement_structure_metrics.json"


def test_refinement_structure_snapshot_matches_production_sources():
    actual = report_refinement_structure.collect_metrics(REPO_ROOT)
    expected = report_refinement_structure._checked_snapshot(SNAPSHOT)

    assert actual == expected


def test_refinement_structure_scope_counts_every_refinement_python_module():
    metrics = report_refinement_structure.collect_metrics(REPO_ROOT)
    expected_paths = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "relax" / "refinement").glob("*.py")
        if path.name not in report_refinement_structure.EXCLUDED_FILENAMES
    }

    assert {row["path"] for row in metrics["files"]} == expected_paths
