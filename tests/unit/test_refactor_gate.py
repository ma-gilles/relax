"""The static steps of ``scripts/dev/refactor_gate.sh`` fail on a finding the base lacks, on a small git tree."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_SCRIPTS = ("refactor_gate.sh", "refactor_module.sh", "call_arity_check.py")

BASE_FILES = {
    "relax/__init__.py": "",
    "relax/engine.py": "def engine(a, *, tuning):\n    return a\n",
    "relax/caller.py": "from relax.engine import engine\n\n\ndef run():\n    return engine(1, tuning=2)\n",
    "tests/__init__.py": "",
}
# Each edit adds one finding of one step: a call that omits a required keyword (arity), an unused import
# (ruff) and a trailing blank (diffcheck).
HEAD_FILES = {
    "relax/caller.py": "import os\n\nfrom relax.engine import engine\n\n\ndef run():\n    return engine(1) \n",
}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=gate", "-c", "user.email=gate@example.invalid", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit(repo: Path, files: dict[str, str]) -> str:
    for name, text in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "tree")
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    repo = tmp_path_factory.mktemp("gate_repo")
    _git(repo, "init", "-q")
    (repo / "scripts" / "dev").mkdir(parents=True)
    for name in GATE_SCRIPTS:
        shutil.copy(REPO_ROOT / "scripts" / "dev" / name, repo / "scripts" / "dev" / name)
    base = _commit(repo, BASE_FILES)
    _commit(repo, HEAD_FILES)
    return repo, base, tmp_path_factory.mktemp("gate_scratch")


def _gate(tree, step: str, base: str):
    repo, _, scratch = tree
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(scratch),
        "REFACTOR_SCRATCH": str(scratch),
        "REFACTOR_WORKTREE": str(repo),
        "REFACTOR_PYTHON": sys.executable,
        "REFACTOR_TAG": f"{step}_{base[:7]}",
    }
    return subprocess.run(
        ["bash", "scripts/dev/refactor_gate.sh", step, base], cwd=repo, env=env, capture_output=True, text=True
    )


@pytest.mark.parametrize(
    ("step", "summary", "finding"),
    [
        ("arity", "call arity findings", "relax/caller.py"),
        ("ruff", "ruff findings", "F401"),
        ("diffcheck", "git diff --check", "trailing whitespace"),
    ],
)
def test_a_finding_the_base_lacks_fails_the_step(tree, step, summary, finding):
    result = _gate(tree, step, tree[1])
    assert result.returncode == 1, result.stdout + result.stderr
    assert summary in result.stdout
    assert finding in result.stdout


@pytest.mark.parametrize("step", ["arity", "ruff", "diffcheck"])
def test_the_head_against_itself_passes_the_step(tree, step):
    result = _gate(tree, step, "HEAD")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("\n") == 1
