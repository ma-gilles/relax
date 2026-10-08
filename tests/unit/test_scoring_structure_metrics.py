from pathlib import Path

from scripts import report_refinement_structure
from scripts.dev import ceilings as slack

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "relax/scoring"
CEILINGS = REPO_ROOT / "docs" / "development" / "scoring_structure_metrics.json"


def test_scoring_structure_stays_within_recorded_ceilings():
    totals = report_refinement_structure.collect_metrics(REPO_ROOT, PACKAGE)["totals"]
    ceilings = report_refinement_structure.read_ceilings(CEILINGS)

    assert slack.exceeded(totals, ceilings) == [], (
        "relax/scoring grew past a recorded ceiling and its slack: shrink it, or raise the ceiling by hand in "
        f"{CEILINGS.relative_to(REPO_ROOT)} with the reason in the commit message"
    )


def test_scoring_scope_counts_every_scoring_python_module():
    metrics = report_refinement_structure.collect_metrics(REPO_ROOT, PACKAGE)

    assert metrics["scope"]["root"] == PACKAGE
    assert {row["path"] for row in metrics["files"]} == {
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "relax" / "scoring").glob("*.py")
        if path.name not in report_refinement_structure.EXCLUDED_FILENAMES
    }


def test_the_metrics_file_records_the_slack_the_script_applies():
    import json

    recorded = json.loads(CEILINGS.read_text())["slack"]
    assert (recorded["percent"], recorded["minimum"]) == (slack.SLACK_PERCENT, slack.SLACK_MINIMUM)
