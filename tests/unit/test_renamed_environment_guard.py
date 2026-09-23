"""importing relax refuses renamed relax-only RECOVAR_* environment variables (relax split P5)."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

import relax

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
TABLE = json.loads((REPO / "relax" / "renamed_environment.json").read_text())


def test_a_renamed_name_is_an_error_naming_the_new_name():
    old, new = next(iter(TABLE["renamed"].items()))
    with pytest.raises(RuntimeError, match=re.escape(f"{old} -> {new}")):
        relax._reject_renamed_environment(environ={old: "1"})


def test_a_renamed_prefix_is_an_error():
    prefix = next(old for old in TABLE["renamed"] if old.endswith("_"))
    with pytest.raises(RuntimeError, match=re.escape(prefix + "X")):
        relax._reject_renamed_environment(environ={prefix + "X": "1"})


@pytest.mark.parametrize("group", ["read_by_recovar", "recorded_in_json"])
def test_exempt_names_and_new_names_are_accepted(group):
    environ = {name: "1" for name in TABLE["exempt"][group]}
    environ.update({new: "1" for new in TABLE["renamed"].values()})
    relax._reject_renamed_environment(environ=environ)


def test_exempt_and_renamed_are_disjoint_and_names_are_relax_only():
    exempt = {name for names in TABLE["exempt"].values() for name in names}
    assert not exempt & set(TABLE["renamed"])
    assert {"RECOVAR_DISABLE_CUDA", "RECOVAR_CUDA_LIB", "RECOVAR_EM_XLA_DEFAULTS"} <= exempt


def test_relax_source_reads_no_renamed_name():
    names = set(TABLE["renamed"])
    tracked = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "relax", "tests", "scripts"], capture_output=True, text=True, check=True
    ).stdout.split()
    offenders = []
    for rel in tracked:
        if rel.endswith(".json"):
            continue
        try:
            text = (REPO / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        hits = set(re.findall(r"\bRECOVAR_[A-Z0-9_]+\b", text)) & names
        offenders += [f"{rel}: {name}" for name in sorted(hits)]
    assert offenders == [], offenders[:20]


def test_import_fails_with_a_stale_name(tmp_path):
    old = next(iter(TABLE["renamed"]))
    result = subprocess.run(
        [sys.executable, "-c", "import relax"],
        cwd=REPO,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO), "JAX_PLATFORMS": "cpu", old: "1"},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert old in result.stderr
