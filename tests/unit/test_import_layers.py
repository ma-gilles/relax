"""relax's packages form layers, and no import points up (the package layout, LAYOUT_PROPOSAL.md section 2).

Every import of a relax module counts, at the top level, inside a function (lazy) or under TYPE_CHECKING. An import
is upward when its package sits later in ``LAYERS`` than the importing package, or when one controller imports
another. ``ALLOWED_UPWARD`` lists the upward edges main still has, each with why; it can only shrink: the test also
fails when a listed edge has gone, so the change that removes an edge deletes its line. Without the listed edges the
package graph has no cycle.
"""

import ast
from collections import defaultdict
from pathlib import Path

import pytest

import relax

pytestmark = pytest.mark.unit

ROOT = Path(relax.__file__).parent

# Bottom to top; inside a band the packages are ordered too. "(top)" is relax/__init__.py and relax/command_line.py.
LAYERS = (
    ("foundation", ("io", "runtime", "cuda", "numerics", "relion_bind", "types")),
    ("primitives", ("fourier", "relion", "projection", "sampling")),
    ("engines", ("reconstruction", "fine_pass", "scoring", "dense", "local_search", "ppca_refinement")),
    ("operations", ("classification", "star", "steps")),
    ("controllers", ("refinement", "vdam", "ppca_initial_model")),
    ("adapters", ("diagnostics", "parity")),
    ("command", ("commands", "(top)")),
)
CONTROLLERS = {"refinement", "vdam", "ppca_initial_model"}
# The command band is one layer: its members may import each other.
SAME_RANK = {"(top)": "commands"}

_A = "A: a step or record that other packages use lives in a controller (layout S1, S7, S8, S12)"
_B = "B: a diagnostics or parity hook reached from the algorithm (layout S9-S11; k_class and scoring S15)"
_C = "C: relion mixes image formation, STAR readers and one engine's files (layout S3-S6)"
_D = "D: engines import each other upward (layout S7, S13, S14)"
_E = "E: a module in the wrong layer (layout S2, S6, S8d, S9)"

# (importing module, imported module) -> why the upward edge is still there.
ALLOWED_UPWARD = {
    ("relax.classification.k_class", "relax.refinement.engine_record"): _A,
    ("relax.fine_pass.bucket_arrays", "relax.local_search.layout"): _D,
    ("relax.fine_pass.dispatch", "relax.refinement.engine_record"): _A,
    ("relax.refinement.dense_half", "relax.diagnostics.parity_dump"): _B,
    ("relax.refinement.expectation", "relax.diagnostics.bpref_diagnostics"): _B,
    ("relax.refinement.expectation", "relax.diagnostics.parity_dump"): _B,
    ("relax.refinement.finalization", "relax.diagnostics.bpref_diagnostics"): _B,
    ("relax.refinement.firstiter_bpref", "relax.diagnostics.bpref_diagnostics"): _B,
    ("relax.refinement.iteration_loop", "relax.diagnostics.bpref_diagnostics"): _B,
    ("relax.refinement.local_half", "relax.diagnostics.local_debug"): _B,
    ("relax.vdam.estep_setup", "relax.refinement.engine_record"): _A,
}


def _rank(package):
    package = SAME_RANK.get(package, package)
    for layer, (_, members) in enumerate(LAYERS):
        if package in members:
            return layer, members.index(package)
    raise AssertionError(f"package {package!r} is in no layer of LAYERS")


def _module_name(path):
    parts = path.relative_to(ROOT.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _package_of(module):
    parts = module.split(".")
    if len(parts) == 1 or parts[1] in ("command_line",):
        return "(top)"
    return parts[1]


def _is_module(name):
    path = ROOT.parent.joinpath(*name.split("."))
    return path.with_suffix(".py").is_file() or (path / "__init__.py").is_file()


def _relax_imports(source, package):
    """The relax modules ``source`` imports, in any scope: ``from pkg import module`` names the module.

    ``package`` is the importing module's package, against which relative imports resolve.
    """
    targets = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = package.split(".")[: len(package.split(".")) - node.level + 1]
                base = ".".join([*anchor, *filter(None, [base])])
            for alias in node.names:
                name = f"{base}.{alias.name}"
                targets.add(name if _is_module(name) else base)
    return {t for t in targets if t == "relax" or t.startswith("relax.")}


def _edges():
    """(importing module, imported module) for every import between two relax packages."""
    edges = set()
    for path in sorted(ROOT.rglob("*.py")):
        source = _module_name(path)
        package = source if path.name == "__init__.py" else source.rpartition(".")[0]
        for target in _relax_imports(path.read_text(), package):
            while not _is_module(target) and "." in target:
                target = target.rpartition(".")[0]
            if _package_of(source) != _package_of(target):
                edges.add((source, target))
    return edges


def _upward(source, target):
    a, b = _package_of(source), _package_of(target)
    return _rank(a) < _rank(b) or (a in CONTROLLERS and b in CONTROLLERS)


def _package_cycles(edges):
    """Strongly connected sets of two or more packages (Tarjan)."""
    graph = defaultdict(set)
    for source, target in edges:
        graph[_package_of(source)].add(_package_of(target))
    index, low, stack, on_stack, cycles = {}, {}, [], set(), []

    def visit(node):
        index[node] = low[node] = len(index)
        stack.append(node)
        on_stack.add(node)
        for nxt in graph[node]:
            if nxt not in index:
                visit(nxt)
                low[node] = min(low[node], low[nxt])
            elif nxt in on_stack:
                low[node] = min(low[node], index[nxt])
        if low[node] == index[node]:
            component = set()
            while True:
                member = stack.pop()
                on_stack.discard(member)
                component.add(member)
                if member == node:
                    break
            if len(component) > 1:
                cycles.append(sorted(component))

    for node in list(graph):
        if node not in index:
            visit(node)
    return cycles


def test_every_package_has_a_layer():
    packages = {_package_of(_module_name(path)) for path in ROOT.rglob("*.py")}
    for package in packages:
        _rank(package)


def test_no_import_points_up_beyond_the_allowlist():
    upward = {edge for edge in _edges() if _upward(*edge)}
    assert sorted(upward - set(ALLOWED_UPWARD)) == [], "a new upward or controller-to-controller import"
    assert sorted(set(ALLOWED_UPWARD) - upward) == [], "an allowed upward import is gone: delete its entry"


def test_the_package_graph_has_no_cycle_outside_the_allowlist():
    edges = _edges() - set(ALLOWED_UPWARD)
    assert _package_cycles(edges) == []


def test_the_upward_ratchet_sees_lazy_and_type_checking_imports():
    source = (
        "from typing import TYPE_CHECKING\n"
        "import numpy\n"
        "if TYPE_CHECKING:\n    from relax.vdam import driver\n"
        "def f():\n    from relax.refinement.ports import RunObserver\n"
        "    from ..numerics import shells\n"
    )
    found = _relax_imports(source, "relax.sampling")
    assert found == {"relax.vdam.driver", "relax.refinement.ports", "relax.numerics.shells"}
    assert _upward("relax.sampling", "relax.vdam.driver") and _upward("relax.sampling", "relax.refinement.ports")
    assert not _upward("relax.sampling", "relax.numerics.shells")
    assert _upward("relax.vdam.driver", "relax.refinement.ports")  # controller to controller
    assert not _upward("relax.commands.refine", "relax.command_line")  # one command band
