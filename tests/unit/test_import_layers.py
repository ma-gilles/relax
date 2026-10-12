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
    ("relax.classification.k_class", "relax.diagnostics.coarse_gaussian_diagnostics"): _B,
    ("relax.classification.k_class", "relax.diagnostics.coarse_score_diagnostics"): _B,
    ("relax.classification.k_class", "relax.diagnostics.global_winner_summary"): _B,
    ("relax.classification.k_class", "relax.diagnostics.local_debug"): _B,
    ("relax.classification.k_class", "relax.refinement.engine_record"): _A,
    ("relax.diagnostics.vdam_native_mstep", "relax.commands.initial_model"): _E,
    ("relax.diagnostics.vdam_native_mstep", "relax.parity.vdam_mstep_replay"): _E,
    ("relax.fine_pass.bucket_arrays", "relax.local_search.layout"): _D,
    ("relax.fine_pass.bucket_arrays", "relax.scoring.compact_candidates"): _D,
    ("relax.fine_pass.bucket_arrays", "relax.scoring.significant_samples"): _D,
    ("relax.fine_pass.bucket_io", "relax.diagnostics.finite_check"): _B,
    ("relax.fine_pass.bucket_io", "relax.diagnostics.sparse_pass2_dump"): _B,
    ("relax.fine_pass.bucket_io", "relax.refinement.precision"): _A,
    ("relax.fine_pass.dispatch", "relax.refinement.engine_record"): _A,
    ("relax.fine_pass.resident_candidates", "relax.scoring.compact_candidates"): _D,
    ("relax.fine_pass.resident_operands", "relax.refinement.precision"): _A,
    ("relax.fine_pass.resident_pass2", "relax.dense.gemm_coarse_engine"): _D,
    ("relax.fine_pass.resident_scoring", "relax.local_search.resident_layout"): _D,
    ("relax.fine_pass.resident_significance", "relax.scoring.significant_samples"): _D,
    ("relax.fine_pass.resident_tilts", "relax.refinement.tomo_particles"): _A,
    ("relax.fine_pass.window", "relax.refinement.precision"): _A,
    ("relax.ppca_initial_model.config", "relax.vdam.schedules"): _E,
    ("relax.refinement.command_options", "relax.parity.frozen_boundary_cli"): _B,
    ("relax.refinement.command_options", "relax.parity.state_swap_probe"): _B,
    ("relax.refinement.dense_half", "relax.diagnostics.parity_dump"): _B,
    ("relax.refinement.expectation", "relax.diagnostics.bpref_diagnostics"): _B,
    ("relax.refinement.expectation", "relax.diagnostics.parity_dump"): _B,
    ("relax.refinement.finalization", "relax.diagnostics.bpref_diagnostics"): _B,
    ("relax.refinement.firstiter_bpref", "relax.diagnostics.bpref_diagnostics"): _B,
    ("relax.refinement.firstiter_bpref", "relax.diagnostics.finite_check"): _B,
    ("relax.refinement.full_refinement", "relax"): _E,
    ("relax.refinement.full_refinement", "relax.diagnostics.observers"): _B,
    ("relax.refinement.full_refinement", "relax.parity.archive_provenance"): _B,
    ("relax.refinement.full_refinement", "relax.parity.frozen_boundary_cli"): _B,
    ("relax.refinement.full_refinement", "relax.parity.initial_model_replay"): _B,
    ("relax.refinement.full_refinement", "relax.parity.oracle_admission"): _B,
    ("relax.refinement.full_refinement", "relax.parity.relion_replay_source"): _B,
    ("relax.refinement.full_refinement", "relax.parity.replay_inputs"): _B,
    ("relax.refinement.full_refinement", "relax.parity.startup_noise_inputs"): _B,
    ("relax.refinement.full_refinement", "relax.parity.state_swap_probe"): _B,
    ("relax.refinement.iteration_loop", "relax.diagnostics.bpref_diagnostics"): _B,
    ("relax.refinement.iteration_loop", "relax.diagnostics.iteration"): _B,
    ("relax.refinement.iteration_loop", "relax.diagnostics.reconstruction"): _B,
    ("relax.refinement.iteration_loop", "relax.parity.relion_replay"): _B,
    ("relax.refinement.local_half", "relax.diagnostics.local_debug"): _B,
    ("relax.refinement.result_files", "relax.diagnostics.parity_dump"): _B,
    ("relax.relion.initial_model_io", "relax.refinement.run_files"): _A,
    ("relax.relion.initial_model_io", "relax.sampling"): _C,
    ("relax.relion.initial_model_io", "relax.vdam.state"): _C,
    ("relax.relion.input_particle_table", "relax.sampling.expected_accuracy"): _C,
    ("relax.relion.input_poses", "relax.refinement.iteration_history"): _A,
    ("relax.relion.metadata", "relax.sampling.orientation_priors"): _C,
    ("relax.relion.metadata", "relax.sampling.symmetry"): _C,
    ("relax.relion.normalization", "relax.diagnostics.finite_check"): _B,
    ("relax.relion.tomo_input", "relax.sampling.healpix"): _C,
    ("relax.sampling.expected_accuracy", "relax.refinement.optics_shapes"): _A,
    ("relax.sampling.expected_accuracy", "relax.refinement.projector_preparation"): _A,
    ("relax.sampling.expected_accuracy", "relax.refinement.tomo_half"): _A,
    ("relax.scoring.pass1_assembly", "relax.diagnostics.coarse_score_diagnostics"): _B,
    ("relax.scoring.pass1_dump", "relax.diagnostics.coarse_gaussian_diagnostics"): _B,
    ("relax.scoring.pass1_publish", "relax.diagnostics.coarse_gaussian_diagnostics"): _B,
    ("relax.scoring.tomo_coarse", "relax.refinement.tomo_particles"): _A,
    ("relax.scoring.tree_rescore", "relax.diagnostics.coarse_gaussian_diagnostics"): _B,
    ("relax.vdam.adaptive_estep", "relax.refinement.optics_shapes"): _A,
    ("relax.vdam.bootstrap_iref", "relax.refinement.optics_shapes"): _A,
    ("relax.vdam.bootstrap_iref", "relax.refinement.tomo_half"): _A,
    ("relax.vdam.driver", "relax.refinement.optics_shapes"): _A,
    ("relax.vdam.driver", "relax.refinement.tomo_half"): _A,
    ("relax.vdam.estep_setup", "relax.refinement.engine_record"): _A,
    ("relax.vdam.estep_setup", "relax.refinement.optics_shapes"): _A,
    ("relax.vdam.iteration_loop", "relax.refinement.refinement_state"): _A,
    ("relax.vdam.native_sampling", "relax.refinement.optics_shapes"): _A,
    ("relax.vdam.native_sampling", "relax.refinement.refinement_state"): _A,
    ("relax.vdam.native_sampling", "relax.refinement.tomo_half"): _A,
    ("relax.vdam.shape_class_estep", "relax.refinement.shape_class_scoring"): _A,
    ("relax.vdam.tomo_estep", "relax.refinement.scoring_policy"): _A,
    ("relax.vdam.tomo_estep", "relax.refinement.tomo_half"): _A,
    ("relax.vdam.tomo_estep", "relax.refinement.tomo_particles"): _A,
    ("relax.vdam.tomo_estep", "relax.refinement.tomo_scoring"): _A,
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
