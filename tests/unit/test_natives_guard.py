"""Tests that need the RELION binding skip with one reason without it and stop the run under --require-natives."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from helpers import natives

from scripts import run_test_tier

REPO_ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.unit

# Runs pytest with the repository's conftest as a plugin and the binding made unimportable, so the
# session behaves as on a checkout where it was never built, whether or not this one has it.
CHILD = """\
import sys
sys.modules["relax.relion_bind._relion_bind_core"] = None
import pytest
raise SystemExit(pytest.main(sys.argv[1:]))
"""
MODULE_LEVEL = """\
from relax.relion_bind._relion_bind_core import get_ctf_image

def test_never_collected():
    raise AssertionError
"""
MARKED = """\
import pytest
from helpers import natives

@pytest.fixture
def bind():
    return natives.relion_bind_core()

@pytest.mark.requires_relion_bind
def test_marked():
    from relax.relion_bind import _relion_bind_core

def test_through_fixture(bind):
    raise AssertionError

def test_needs_nothing():
    pass
"""


def _child(tmp_path, *args):
    (tmp_path / "test_module_level.py").write_text(MODULE_LEVEL)
    (tmp_path / "test_marked.py").write_text(MARKED)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", PYTHONNOUSERSITE="1")
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT), str(REPO_ROOT / "tests")])
    command = [
        sys.executable,
        "-c",
        CHILD,
        "-p",
        "conftest",
        "-p",
        "no:cacheprovider",
        "-rs",
        "--rootdir",
        str(tmp_path),
    ]
    return subprocess.run([*command, *args, str(tmp_path)], cwd=tmp_path, env=env, capture_output=True, text=True)


def test_without_the_binding_its_tests_skip_with_one_reason(tmp_path):
    child = _child(tmp_path)
    assert child.returncode == 0, child.stdout[-3000:] + child.stderr[-3000:]
    assert "1 passed, 3 skipped" in child.stdout
    skips = [line for line in child.stdout.splitlines() if line.startswith("SKIPPED")]
    assert len(skips) == 3 and all(natives.SKIP_REASON in " ".join(line.split()) for line in skips), skips


def test_collection_alone_succeeds_without_the_binding(tmp_path):
    child = _child(tmp_path, "--collect-only", "-q")
    assert child.returncode == 0, child.stdout[-3000:] + child.stderr[-3000:]
    assert "3 tests collected" in child.stdout and "error" not in child.stdout.splitlines()[-1]


def test_require_natives_stops_the_run_instead_of_skipping(tmp_path):
    child = _child(tmp_path, "--require-natives")
    assert child.returncode == pytest.ExitCode.USAGE_ERROR
    assert f"--require-natives: {natives.BINDING} does not import" in child.stderr
    assert "passed" not in child.stdout and "skipped" not in child.stdout


def test_module_level_binding_imports_are_recognised():
    assert natives.imports_binding_at_module_level("from relax.relion_bind._relion_bind_core import a, b\n")
    assert natives.imports_binding_at_module_level("from relax.relion_bind import _relion_bind_core as bind\n")
    assert natives.imports_binding_at_module_level("import relax.relion_bind._relion_bind_core\n")
    assert not natives.imports_binding_at_module_level(
        "import pytest\nbind = pytest.importorskip('relax.relion_bind._relion_bind_core')\n"
    )
    assert not natives.imports_binding_at_module_level(
        "def test_x():\n    from relax.relion_bind import _relion_bind_core\n"
    )
    assert not natives.imports_binding_at_module_level("from relax.relion_bind import conversions\n")


def test_no_test_imports_the_binding_unguarded():
    """A test body or fixture that imports the binding carries the marker, uses the helper or importorskip."""
    import ast

    unguarded = []
    for path in sorted((REPO_ROOT / "tests").rglob("test_*.py")):
        source = path.read_text()
        if "_relion_bind_core" not in source or natives.imports_binding_at_module_level(source):
            continue
        if "pytestmark" in source and natives.MARKER in source.split("pytestmark", 1)[1].split("\n", 1)[0]:
            continue
        for function in (
            n for n in ast.walk(ast.parse(source)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ):
            text = ast.unparse(function)
            imports = [
                n
                for n in ast.walk(function)
                if isinstance(n, (ast.Import, ast.ImportFrom)) and "_relion_bind_core" in ast.unparse(n)
            ]
            guarded = natives.MARKER in text or "importorskip" in text or "_skip_if_unavailable" in text
            in_try = any(isinstance(n, ast.Try) and any(i in ast.walk(n) for i in imports) for n in ast.walk(function))
            if imports and not guarded and not in_try:
                unguarded.append(f"{path.relative_to(REPO_ROOT)}::{function.name}")
    assert unguarded == []


def test_every_tier_pytest_item_requires_natives():
    argv = run_test_tier._pytest("python", "tests/unit/test_x.py", flags=["--run-gpu"])
    assert argv == [
        *("python", "-m", "pytest", "-p", "no:cacheprovider", "-v", "-s"),
        *("--require-natives", "--run-gpu", "tests/unit/test_x.py"),
    ]
    for tier in ("smoke", "long"):
        items = (
            run_test_tier.plan(tier, REPO_ROOT, "HEAD")
            if tier == "smoke"
            else run_test_tier.long_plan(REPO_ROOT, sys.executable, None)
        )
        tests = [item for item in items if "pytest" in item.argv or "scripts/run_em_fast_guard.sh" in item.argv]
        assert tests and all(item.argv.count("--require-natives") == 1 for item in tests)
    # The fast guard hands its arguments to pytest.
    guard = (REPO_ROOT / "scripts" / "run_em_fast_guard.sh").read_text()
    assert 'exec "$PYTHON_BIN" -m pytest "${tests[@]}" -q "$@"' in guard
