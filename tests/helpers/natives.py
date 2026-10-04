"""Tests that need the built RELION binding skip without it, or fail when natives are required.

``relax.relion_bind._relion_bind_core`` is a compiled extension (``pixi run build-relion-bind``, or
an existing build named by ``RECOVAR_RELION_BIND_BUILD_DIR``); a fresh checkout does not have it.
Without it:

- a test marked ``requires_relion_bind`` is skipped with ``SKIP_REASON``;
- a test module that imports the binding at module level is skipped as a whole with the same
  reason, instead of stopping collection with an import error;
- with ``--require-natives`` (the tier runner passes it, because it builds the natives) the session
  stops before collection: a missing binding is then a setup failure, not a skip.

Mark a test ``requires_relion_bind`` when it imports the binding inside its body or calls relax code
that does. A fixture that hands the binding to its tests takes it from ``relion_bind_core()``, which
skips those tests in the same way.
"""

from __future__ import annotations

import ast
import importlib
from functools import lru_cache
from pathlib import Path

import pytest

BINDING = "relax.relion_bind._relion_bind_core"
MARKER = "requires_relion_bind"
SKIP_REASON = (
    f"the RELION binding {BINDING} is not built: run `pixi run build-relion-bind` or set "
    "RECOVAR_RELION_BIND_BUILD_DIR to a directory holding it (--require-natives makes this a failure)"
)


@lru_cache(maxsize=1)
def relion_bind_error() -> str | None:
    """None when the binding imports, else the import error as text."""
    try:
        importlib.import_module(BINDING)
    except ImportError as error:
        return f"{type(error).__name__}: {error}"
    return None


def relion_bind_core():
    """The binding module, for a fixture; skips the requesting test when the binding is not built."""
    if relion_bind_error() is not None:
        pytest.skip(SKIP_REASON)
    return importlib.import_module(BINDING)


def imports_binding_at_module_level(source: str) -> bool:
    """Whether ``source`` imports the binding in a top-level statement, which fails at collection without it."""
    package, _, core = BINDING.rpartition(".")
    if core not in source:
        return False
    for node in ast.parse(source).body:
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            if node.module == BINDING or (node.module == package and any(a.name == core for a in node.names)):
                return True
        elif isinstance(node, ast.Import) and any(a.name == BINDING for a in node.names):
            return True
    return False


def module_needs_binding(path: Path) -> bool:
    return imports_binding_at_module_level(Path(path).read_text())
