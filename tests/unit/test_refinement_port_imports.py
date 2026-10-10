"""Production modules never import a port implementation or a diagnostic (code rules 15 and 51).

Observers (``relax.diagnostics.observers``) and comparison input sources (``relax.parity``, which alone holds the
RELION replay, the frozen boundary and the state-swap probe) are chosen by the command and handed to the
controller through ``relax.refinement.ports``; the algorithm imports only the ports. Outside the command boundary
no module of any ``relax`` package imports ``relax.diagnostics`` or ``relax.parity``, except the edges listed in
``ALLOWED_IMPORTS``: each names why it is still there, and the list can only shrink.
"""

import ast
import json
import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest

import relax
import relax.refinement

pytestmark = pytest.mark.unit

# The command boundary chooses the implementations; every other module of the package is the algorithm.
# particle_loading is the command's particle reader (only full_refinement imports it; it reads the flags).
COMMAND_MODULES = {"full_refinement", "command_options", "particle_loading"}
IMPLEMENTATIONS = ("relax.diagnostics.observers", "relax.parity")
# Modules that may import relax.diagnostics and relax.parity: the two packages themselves and the command layer.
BOUNDARY = (
    "relax.diagnostics",
    "relax.parity",
    "relax.command_line",
    "relax.commands",
    *(f"relax.refinement.{name}" for name in sorted(COMMAND_MODULES)),
)
DIAGNOSTIC_PACKAGES = ("relax.diagnostics", "relax.parity")

_INLINE_DUMP = "an inline dump/capture called from the algorithm; it needs a RunObserver hook (deep2 PLAN c2)"
_INLINE_CHECK = "an inline finite/accumulator check that raises inside the algorithm; it needs a port (deep2 PLAN c2)"
_REPLAY = "a replay policy called by the controller; it needs an InputSource method (deep2 PLAN c3 follow-up)"

# (importing module, imported module) -> why the edge is still there.
ALLOWED_IMPORTS = {
    ("relax.classification.k_class", "relax.diagnostics.coarse_gaussian_diagnostics"): _INLINE_DUMP,
    ("relax.classification.k_class", "relax.diagnostics.coarse_score_diagnostics"): _INLINE_DUMP,
    ("relax.classification.k_class", "relax.diagnostics.global_winner_summary"): _INLINE_DUMP,
    ("relax.classification.k_class", "relax.diagnostics.local_debug"): _INLINE_DUMP,
    ("relax.refinement.expectation", "relax.diagnostics.bpref_diagnostics"): _INLINE_DUMP,
    ("relax.refinement.expectation", "relax.diagnostics.parity_dump"): _INLINE_DUMP,
    ("relax.refinement.finalization", "relax.diagnostics.bpref_diagnostics"): _INLINE_DUMP,
    ("relax.refinement.half_scoring", "relax.diagnostics.parity_dump"): _INLINE_DUMP,
    ("relax.refinement.iteration_loop", "relax.diagnostics.bpref_diagnostics"): _INLINE_DUMP,
    ("relax.refinement.iteration_loop", "relax.diagnostics.iteration"): _INLINE_DUMP,
    ("relax.refinement.iteration_loop", "relax.diagnostics.reconstruction"): _INLINE_CHECK,
    ("relax.refinement.iteration_loop", "relax.parity.relion_replay"): _REPLAY,
    ("relax.refinement.local_half", "relax.diagnostics.local_debug"): _INLINE_DUMP,
    ("relax.refinement.result_files", "relax.diagnostics.parity_dump"): "the run report's timing rows (" + _INLINE_DUMP + ")",
    ("relax.relion.relion_normalization", "relax.diagnostics.finite_check"): _INLINE_CHECK,
    ("relax.scoring.pass1_assembly", "relax.diagnostics.coarse_score_diagnostics"): _INLINE_DUMP,
    ("relax.scoring.pass1_dump", "relax.diagnostics.coarse_gaussian_diagnostics"): _INLINE_DUMP,
    ("relax.scoring.pass1_publish", "relax.diagnostics.coarse_gaussian_diagnostics"): _INLINE_DUMP,
    ("relax.scoring.tree_rescore", "relax.diagnostics.coarse_gaussian_diagnostics"): _INLINE_DUMP,
    ("relax.sparse_pass2.firstiter_bpref", "relax.diagnostics.bpref_diagnostics"): _INLINE_DUMP,
    ("relax.sparse_pass2.firstiter_bpref", "relax.diagnostics.finite_check"): _INLINE_CHECK,
    ("relax.sparse_pass2.sparse_pass2_bucket_io", "relax.diagnostics.finite_check"): _INLINE_CHECK,
    ("relax.sparse_pass2.sparse_pass2_bucket_io", "relax.diagnostics.sparse_pass2_dump"): _INLINE_DUMP,
}


def _in(name, prefixes):
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def _module_or_package(root, name, package):
    """``name`` when it is a module of the repository, else ``package``."""
    path = root.parent.joinpath(*name.split("."))
    return name if path.with_suffix(".py").is_file() or path.is_dir() else package


def _diagnostic_imports():
    """Every (module, imported diagnostics/parity module) edge outside the boundary, lazy imports included."""
    root = Path(relax.__file__).parent
    edges = set()
    for path in sorted(root.rglob("*.py")):
        parts = path.relative_to(root.parent).with_suffix("").parts
        module = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        if _in(module, BOUNDARY):
            continue
        package = module if parts[-1] == "__init__" else module.rpartition(".")[0]
        for node in ast.walk(ast.parse(path.read_text(), str(path))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    base = ".".join([*package.split(".")[: len(package.split(".")) - node.level + 1], *filter(None, [base])])
                # ``from package import module`` imports the module; ``from module import name`` the module.
                names = [_module_or_package(root, f"{base}.{alias.name}", base) for alias in node.names]
            else:
                continue
            edges.update((module, name) for name in names if _in(name, DIAGNOSTIC_PACKAGES))
    return edges


def test_no_production_module_imports_diagnostics_or_parity_beyond_the_allowlist():
    edges = _diagnostic_imports()
    assert sorted(edges - set(ALLOWED_IMPORTS)) == [], "a new import of relax.diagnostics/relax.parity"
    assert sorted(set(ALLOWED_IMPORTS) - edges) == [], "an allowlisted import is gone: delete its entry"


def test_the_algorithm_modules_import_no_port_implementation():
    names = sorted(
        f"relax.refinement.{info.name}"
        for info in pkgutil.iter_modules(relax.refinement.__path__)
        if info.name not in COMMAND_MODULES
    )
    assert "relax.refinement.iteration_loop" in names and "relax.refinement.ports" in names
    script = (
        "import importlib, json, sys\n"
        f"for name in {names!r}: importlib.import_module(name)\n"
        "print(json.dumps(sorted(name for name in sys.modules)))\n"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    loaded = json.loads(out.stdout.strip().splitlines()[-1])
    leaked = {name for name in loaded if _in(name, IMPLEMENTATIONS)}
    # The allowlisted edges of the refinement load their targets and the package that holds them.
    allowed = {target for source, target in ALLOWED_IMPORTS if source.startswith("relax.refinement.")}
    allowed |= {target.rpartition(".")[0] for target in allowed}
    assert sorted(leaked - allowed) == []


def test_the_command_does_import_the_observers_and_the_input_sources():
    """The checks above would pass vacuously if the implementations lived somewhere else."""
    implementations = ("relax.diagnostics.observers", "relax.parity.relion_replay_source", "relax.parity.oracle_admission")
    out = subprocess.run(
        [sys.executable, "-c",
         f"import sys, relax.refinement.full_refinement; print(all(n in sys.modules for n in {implementations!r}))"],
        capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip().splitlines()[-1] == "True"
    edges = {target for _, target in _diagnostic_imports()}
    assert "relax.diagnostics.finite_check" in edges and "relax.parity.relion_replay" in edges
