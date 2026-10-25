"""The static steps of ``scripts/dev/refactor_gate.sh`` fail on a finding the base lacks, on a small git tree."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_SCRIPTS = ("refactor_gate.sh", "refactor_module.sh", "call_arity_check.py", "harness_patterns.txt")

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


def _env(tree, step: str, base: str) -> dict[str, str]:
    repo, _, scratch = tree
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": str(scratch),
        "REFACTOR_SCRATCH": str(scratch),
        "REFACTOR_WORKTREE": str(repo),
        "REFACTOR_PYTHON": sys.executable,
        "REFACTOR_TAG": f"{step}_{base[:7]}",
    }


def _gate(tree, step: str, base: str):
    return subprocess.run(
        ["bash", "scripts/dev/refactor_gate.sh", step, base],
        cwd=tree[0],
        env=_env(tree, step, base),
        capture_output=True,
        text=True,
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


def test_a_change_to_the_gate_inputs_fails_until_the_owner_accepts_it(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "scripts" / "dev").mkdir(parents=True)
    for name in GATE_SCRIPTS:
        shutil.copy(REPO_ROOT / "scripts" / "dev" / name, repo / "scripts" / "dev" / name)
    base = _commit(repo, BASE_FILES)
    head = _commit(repo, {"scripts/dev/fingerprint_cases.py": "CASES = {}\n", "tests/tiers/pins.json": "{}\n"})
    tree = (repo, base, tmp_path / "scratch")
    result = _gate(tree, "harness", base)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "scripts/dev/fingerprint_cases.py tests/tiers/pins.json" in result.stdout
    accepted = subprocess.run(
        ["bash", "scripts/dev/refactor_gate.sh", "harness", base],
        cwd=repo,
        env={**_env(tree, "harness", base), "REFACTOR_ACCEPT_HARNESS": head},
        capture_output=True,
        text=True,
    )
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr
    assert _gate(tree, "harness", "HEAD").returncode == 0


def test_a_ruff_or_arity_checker_that_does_not_run_fails_its_step(tree, tmp_path):
    # A Python without ruff exits 1 like a run with findings, and a crashing checker prints nothing: both must fail,
    # not read as "0 new".
    broken = tmp_path / "python"
    broken.write_text(f'#!/bin/sh\ncase "$1$2" in -mruff) echo "No module named ruff" >&2; exit 1;; esac\n'
                      f'case "$1" in *call_arity_check.py) exit 1;; esac\nexec {sys.executable} "$@"\n')
    broken.chmod(0o755)
    for step in ("ruff", "arity"):
        result = subprocess.run(
            ["bash", "scripts/dev/refactor_gate.sh", step, "HEAD"],
            cwd=tree[0],
            env={**_env(tree, step, "HEAD"), "REFACTOR_PYTHON": str(broken)},
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0, (step, result.stdout + result.stderr)


def test_harness_patterns_cover_the_tier_runner_and_pytest_configuration_and_see_renames(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "scripts" / "dev").mkdir(parents=True)
    for name in GATE_SCRIPTS:
        shutil.copy(REPO_ROOT / "scripts" / "dev" / name, repo / "scripts" / "dev" / name)
    base = _commit(repo, {**BASE_FILES, "scripts/dev/fingerprint_cases.py": "CASES = {}\n"})
    _commit(repo, {"tests/helpers/specs.py": "X = 1\n", "tests/unit/conftest.py": "", "pyproject.toml": "[tool.pytest]\n",
                   "scripts/run_test_tier.py": "", "pytest.ini": "[pytest]\n", "tox.ini": "", "setup.cfg": "",
                   "relax/sitecustomize.py": "", "usercustomize.py": ""})
    _git(repo, "mv", "scripts/dev/fingerprint_cases.py", "scripts/cases.py")
    _git(repo, "commit", "-q", "-m", "rename a harness file out of the patterns")
    result = _gate((repo, base, tmp_path / "scratch"), "harness", base)
    assert result.returncode == 1, result.stdout + result.stderr
    for path in ("pyproject.toml", "scripts/dev/fingerprint_cases.py", "scripts/run_test_tier.py",
                 "tests/helpers/specs.py", "tests/unit/conftest.py", "pytest.ini", "tox.ini", "setup.cfg",
                 "relax/sitecustomize.py", "usercustomize.py"):
        assert path in result.stdout, path


def _test_tree(tmp_path, head_files, *, delete=()):
    """A tree with the CPU list and the module test directories: the base passes two tests."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "scripts" / "dev").mkdir(parents=True)
    for name in GATE_SCRIPTS:
        shutil.copy(REPO_ROOT / "scripts" / "dev" / name, repo / "scripts" / "dev" / name)
    files = {
        **BASE_FILES,
        "scripts/dev/refactor_cpu_unit_list.txt": "tests/unit/test_one.py\n",
        "tests/unit/test_one.py": "def test_a():\n    pass\n",
        "tests/unit/initial_model/test_vdam.py": "def test_b():\n    pass\n",
        "tests/unit/ppca_initial_model/test_ppca.py": "def test_c():\n    pass\n",
    }
    base = _commit(repo, files)
    for name in delete:
        _git(repo, "rm", "-q", "-r", name)
    if head_files:
        head = _commit(repo, head_files)
    else:
        _git(repo, "commit", "-q", "-m", "delete")
        head = _git(repo, "rev-parse", "HEAD")
    return (repo, base, tmp_path / "scratch"), head


def _gate_steps(tree, *steps, env=None):
    script = "; ".join(f"bash scripts/dev/refactor_gate.sh {step}" for step in steps)
    return subprocess.run(["bash", "-c", script], cwd=tree[0], env={**_env(tree, "tests", tree[1]), **(env or {})},
                          capture_output=True, text=True)


def test_a_head_that_passes_fewer_tests_than_its_base_fails_until_accepted(tmp_path):
    tree, head = _test_tree(tmp_path, {"tests/unit/test_one.py": "import pytest\n\n\ndef test_a():\n    pytest.skip('gone')\n"})
    result = _gate_steps(tree, "tests head HEAD", f"counts {tree[1]}")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "1 tests the base passes are missing or not passed on the head" in result.stdout
    assert "tests.unit.test_one::test_a" in result.stdout
    accepted = _gate_steps(tree, "tests head HEAD", f"counts {tree[1]}", env={"REFACTOR_ACCEPT_COUNTS": head})
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr


def test_a_head_that_swaps_a_passing_test_for_a_new_one_fails_the_counts(tmp_path):
    """The counts compare test ids, not totals: deleting one passing test and adding another keeps the number."""
    tree, _ = _test_tree(tmp_path, {"tests/unit/test_one.py": "def test_renamed():\n    pass\n"})
    result = _gate_steps(tree, "tests head HEAD", f"counts {tree[1]}")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "tests.unit.test_one::test_a" in result.stdout


def test_a_module_test_directory_the_head_deleted_fails_the_tests_step(tmp_path):
    tree, _ = _test_tree(tmp_path, None, delete=("tests/unit/initial_model",))
    result = _gate_steps(tree, "tests head HEAD")
    assert result.returncode != 0, result.stdout + result.stderr
    assert "tests/unit/initial_model (a module's test directory)" in result.stdout
