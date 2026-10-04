"""A test session cannot write into a real receipt ledger or run root named by its environment."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import RUN_CONTROL_ENV, repo_subprocess_env

REPO_ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.unit

# Every test that calls scripts/write_test_receipt.py, in process or as a child.
RECEIPT_WRITING_TESTS = (
    "tests/unit/test_run_test_tier.py::test_receipt_is_written_and_appended",
    "tests/unit/test_run_test_tier.py::test_selected_mode_reaches_summary_and_older_receipt_defaults_auto",
)


def test_session_starts_without_run_control_variables():
    assert "RELAX_TEST_RECEIPTS" in RUN_CONTROL_ENV
    assert [name for name in RUN_CONTROL_ENV if name in os.environ] == []


def test_receipt_writing_tests_are_all_listed():
    callers = sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "tests").rglob("test_*.py")
        if "write_test_receipt" in path.read_text() and path != Path(__file__).resolve()
    )
    assert callers == sorted({name.split("::")[0] for name in RECEIPT_WRITING_TESTS})


def test_unit_tests_cannot_write_to_an_exported_ledger(tmp_path):
    """The tier job exports RELAX_TEST_RECEIPTS; a unit test once appended a passing long receipt there."""
    ledger = tmp_path / "real_receipts.jsonl"
    env = repo_subprocess_env()
    env.update({name: str(tmp_path / name) for name in RUN_CONTROL_ENV})
    env.update(RELAX_TEST_RECEIPTS=str(ledger), CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu")
    child = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", f"--basetemp={tmp_path / 'child'}", "-q"]
        + list(RECEIPT_WRITING_TESTS),
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert child.returncode == 0, child.stdout[-4000:] + child.stderr[-2000:]
    assert f"{len(RECEIPT_WRITING_TESTS)} passed" in child.stdout
    assert not ledger.exists(), ledger.read_text()
    assert [p.name for p in tmp_path.iterdir() if p.name in RUN_CONTROL_ENV] == []
