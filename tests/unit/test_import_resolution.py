"""Every import of a relax name resolves, in relax/, tests/ and scripts/, in any scope (LAYOUT_PROPOSAL.md section 7).

A module move rewrites imports inside functions too, many of them in GPU-only, subtomogram-only or multi-shape
branches that a CPU run never executes. This test reads the source instead of running it: each
``from relax.x import name`` must name a module of the repository or something ``relax.x`` defines at module level,
and each ``import relax.x.y`` a module that exists. An import inside a ``try`` that catches ImportError is a stated
fallback (the fingerprint harnesses import old paths to run on earlier revisions) and is not checked.
"""

import ast
from functools import cache
from pathlib import Path

import pytest

import relax

pytestmark = pytest.mark.unit

REPO = Path(relax.__file__).parent.parent
SCANNED = ("relax", "tests", "scripts")
# Compiled extensions: no Python source to read.
NATIVE_MODULES = {"relax.relion_bind._relion_bind_core"}


def _path_of(module):
    path = REPO.joinpath(*module.split("."))
    if path.with_suffix(".py").is_file():
        return path.with_suffix(".py")
    if (path / "__init__.py").is_file():
        return path / "__init__.py"
    return None


def _bound_names(statements):
    """Names a block binds at its own level, looking through if/try/with/for blocks."""
    names = set()
    for node in statements:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                names.update(n.id for n in ast.walk(target) if isinstance(n, ast.Name))
        elif isinstance(node, ast.Import):
            names.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names)
        for field in ("body", "orelse", "finalbody", "handlers"):
            block = getattr(node, field, None)
            if isinstance(block, list) and not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names |= _bound_names(block)
        if isinstance(node, ast.ExceptHandler):
            names |= _bound_names(node.body)
    return names


@cache
def _module_names(module):
    """What ``module`` defines at module level; ``None`` when it resolves names dynamically (``__getattr__``, ``*``)."""
    tree = ast.parse(_path_of(module).read_text())
    if any(isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names) for n in tree.body):
        return None
    names = _bound_names(tree.body)
    return None if "__getattr__" in names else names


def _guarded(tree):
    """Nodes inside a ``try`` whose handlers catch ImportError (or ModuleNotFoundError), handlers included."""
    guarded = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and any(
            isinstance(h.type, ast.Name) and h.type.id in ("ImportError", "ModuleNotFoundError") for h in node.handlers
        ):
            for block in (node.body, *(h.body for h in node.handlers)):
                for statement in block:
                    guarded.update(id(n) for n in ast.walk(statement))
    return guarded


def _unresolved(source, where, package):
    """The relax imports of ``source`` (shown as ``where``) that name nothing; relative ones resolve in ``package``."""
    tree = ast.parse(source, where)
    guarded = _guarded(tree)
    problems = []
    for node in ast.walk(tree):
        if id(node) in guarded:
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                if (
                    alias.name.startswith("relax.")
                    and alias.name not in NATIVE_MODULES
                    and _path_of(alias.name) is None
                ):
                    problems.append(f"{where}:{node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = package.split(".")[: len(package.split(".")) - node.level + 1]
                base = ".".join([*anchor, *filter(None, [base])])
            if not (base == "relax" or base.startswith("relax.")) or base in NATIVE_MODULES:
                continue
            if _path_of(base) is None:
                problems.append(f"{where}:{node.lineno}: from {base} (no such module)")
                continue
            defined = _module_names(base)
            for alias in node.names:
                if alias.name == "*" or defined is None or alias.name in defined:
                    continue
                if f"{base}.{alias.name}" in NATIVE_MODULES or _path_of(f"{base}.{alias.name}") is not None:
                    continue
                problems.append(f"{where}:{node.lineno}: from {base} import {alias.name}")
    return problems


def test_every_relax_import_names_something_that_exists():
    problems = []
    for top in SCANNED:
        for path in sorted((REPO / top).rglob("*.py")):
            if ".pixi" in path.parts:
                continue
            relative = path.relative_to(REPO)
            problems.extend(_unresolved(path.read_text(), str(relative), ".".join(relative.parts[:-1])))
    assert problems == []


def test_a_stale_lazy_import_is_reported():
    source = (
        "def f():\n    from relax.sampling import no_such_function_anywhere\n"
        "    from relax.numerics.no_such_module import x\n    import relax.no_such_package.y\n"
        "    from .healpix import angular_sampling_deg\n    from relax import sampling\n"
        "    try:\n        from relax.helpers import gone\n    except ImportError:\n        pass\n"  # a stated fallback
    )
    problems = _unresolved(source, "probe.py", "relax.sampling")
    assert sorted(p.split(": ", 1)[1] for p in problems) == [
        "from relax.numerics.no_such_module (no such module)",
        "from relax.sampling import no_such_function_anywhere",
        "import relax.no_such_package.y",
    ]
