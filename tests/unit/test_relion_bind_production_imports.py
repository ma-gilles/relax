"""Production code does not import RELION's compiled binding.

The user's rule (2026-09-29): relax's production paths do not rely on RELION
code. ``relax.relion_bind`` is the oracle of tests, diagnostics and parity
tools only. ``ALLOWED_PREFIXES`` and ``ALLOWED_FILES`` list the places that may
import it; ``PENDING`` (empty since the last port) would name a production module
whose port is still open (docs/development/em_status.md, "RELION binding removal").
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "relax"

ALLOWED_PREFIXES = ("relax/relion_bind/", "relax/diagnostics/", "relax/reference/")
# Builds the oracle; it is a tool, not a refinement path.
ALLOWED_FILES = {"relax/commands/build_relion_bind.py"}
PENDING: set[str] = set()


def _imports_binding(path: Path) -> bool:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any("relion_bind" in alias.name for alias in node.names):
            return True
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if "relion_bind" in module or any(alias.name == "relion_bind" for alias in node.names):
                return True
        if isinstance(node, ast.Call):
            name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            if name in {"import_module", "__import__"} and any(
                isinstance(arg, ast.Constant) and isinstance(arg.value, str) and "relion_bind" in arg.value
                for arg in node.args
            ):
                return True
    return False


def _binding_importers() -> set[str]:
    found = set()
    for path in sorted(PACKAGE.rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith(ALLOWED_PREFIXES) or relative in ALLOWED_FILES:
            continue
        if _imports_binding(path):
            found.add(relative)
    return found


def test_production_modules_do_not_import_the_relion_binding():
    unexpected = _binding_importers() - PENDING
    assert not unexpected, f"production modules import relax.relion_bind: {sorted(unexpected)}"


def test_pending_list_only_names_modules_that_still_import_the_binding():
    finished = PENDING - _binding_importers()
    assert not finished, f"ported modules still listed as pending; remove them: {sorted(finished)}"
