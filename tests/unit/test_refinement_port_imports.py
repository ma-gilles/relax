"""The algorithm's modules never import a port implementation (code rule 15).

Observers (``relax.diagnostics.observers``) and comparison input sources (``relax.parity``) are chosen by the
command and handed to the controller through ``relax.refinement.ports``; the algorithm imports only the ports.
"""

import json
import pkgutil
import subprocess
import sys

import pytest

import relax.refinement

pytestmark = pytest.mark.unit

# The command boundary chooses the implementations; every other module of the package is the algorithm.
COMMAND_MODULES = {"full_refinement", "command_options"}
IMPLEMENTATIONS = ("relax.diagnostics.observers", "relax.parity")


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
    leaked = [name for name in loaded if name.startswith(IMPLEMENTATIONS)]
    assert leaked == []


def test_the_command_does_import_the_observers():
    """The check above would pass vacuously if the implementation lived somewhere else."""
    out = subprocess.run(
        [sys.executable, "-c", "import sys, relax.refinement.full_refinement; print('relax.diagnostics.observers' in sys.modules)"],
        capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip().splitlines()[-1] == "True"
