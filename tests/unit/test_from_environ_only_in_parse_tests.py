"""Tests build option records with explicit values; only the records' parse tests call ``from_environ`` (rule 22).

A repository-hygiene scan of the test tree, a structure check in the sense of rule 13: it parses tests/ with
``ast`` and reports where a call is made, and pins no production code. A test that needs a record's value passes
it (``Record(...)``, ``dataclasses.replace(Record(), field=value)`` or a builder in ``tests/helpers/``) instead
of setting a variable and calling ``Record.from_environ()``; the reading itself is tested in the files below.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

TESTS = Path(__file__).resolve().parents[1]

# The records' parse tests and their guards against the unset environment, relative to tests/.
PARSE_TEST_FILES = frozenset({
    "unit/test_engine_tuning_environment.py",  # EngineTuning (and VdamEnvironment's engine quantum)
    "unit/test_scoring_overrides_environment.py",  # ScoringOverrides
    "unit/test_scoring_variants_environment.py",  # ScoringVariants
    "unit/initial_model/test_vdam_environment.py",  # VdamEnvironment
    # RefinementEnvironment, ReconstructionPrograms, FinalPassOptions, ExecutionOptions, DensePrecisionPolicy,
    # and every record's guard against the unset environment.
    "unit/test_refinement_environment.py",
})


def _from_environ_calls(path: Path) -> list[int]:
    """Line numbers of the ``<anything>.from_environ(...)`` calls in ``path``."""
    tree = ast.parse(path.read_text(), filename=str(path))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "from_environ"
    ]


def test_from_environ_is_called_only_in_the_parse_test_files():
    offenders = [
        f"tests/{path.relative_to(TESTS).as_posix()}:{line}"
        for path in sorted(TESTS.rglob("*.py"))
        if path.relative_to(TESTS).as_posix() not in PARSE_TEST_FILES
        for line in _from_environ_calls(path)
    ]
    assert not offenders, (
        "pass the record's values instead of setting a variable and calling from_environ() (rule 22); "
        f"parse tests belong in one of {sorted(PARSE_TEST_FILES)}: {offenders}"
    )


def test_every_parse_test_file_exists_and_calls_from_environ():
    """The allowlist names live files only, so a renamed or emptied parse-test file is noticed."""
    for rel in sorted(PARSE_TEST_FILES):
        assert _from_environ_calls(TESTS / rel), rel


def test_the_scan_finds_a_call_through_a_module_and_with_arguments(tmp_path):
    sample = tmp_path / "test_sample.py"
    sample.write_text(
        "from relax.vdam import native_options\n"
        "def test_x():\n"
        "    native_options.VdamEnvironment.from_environ()\n"
        "    EngineTuning.from_environ(stable_window_quantum=24)\n"
        "    from_environ = 'a name, not a call'\n"
    )
    assert _from_environ_calls(sample) == [3, 4]
