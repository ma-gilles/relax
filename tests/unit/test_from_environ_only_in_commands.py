"""In relax, only the commands call a record's ``from_environ()`` (rule 16).

A structure check in the sense of rule 13: it parses relax/ with ``ast`` and reports where a call is made, and
pins no production code. Each ``Record.from_environ()`` stays next to its record, but the command calls it and
passes the record down; a record that holds another record (``RefinementEnvironment``'s engine and scoring,
``ScoringVariants``' reconstruction programs, ``VdamEnvironment``'s engine and scoring) takes it as an argument
of its own ``from_environ`` instead of reading it there. The tests' side (only the parse tests call it) is
``test_from_environ_only_in_parse_tests.py``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

RELAX = Path(__file__).resolve().parents[2] / "relax"
COMMANDS = "commands/"


def _from_environ_calls(path: Path) -> list[int]:
    """Line numbers of the ``<anything>.from_environ(...)`` calls in ``path``."""
    tree = ast.parse(path.read_text(), filename=str(path))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "from_environ"
    ]


def test_from_environ_is_called_only_in_relax_commands():
    offenders = [
        f"relax/{path.relative_to(RELAX).as_posix()}:{line}"
        for path in sorted(RELAX.rglob("*.py"))
        if not path.relative_to(RELAX).as_posix().startswith(COMMANDS)
        for line in _from_environ_calls(path)
    ]
    assert not offenders, (
        "only relax/commands calls from_environ(); build the record there and pass it in (rule 16): "
        f"{offenders}"
    )


def test_the_commands_call_from_environ():
    """The scan sees the commands' calls, so an empty result above is not a scan that finds nothing."""
    assert any(_from_environ_calls(path) for path in sorted((RELAX / COMMANDS).rglob("*.py")))
