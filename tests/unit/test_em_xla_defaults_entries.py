"""The EM entry points opt in to the EM-scoped XLA default before importing jax.

Split from recovar tests/unit/test_em_xla_defaults.py (relax split): the policy
(``recovar.jax_config.em_xla_flag_additions``) stays tested in recovar; the entry
points that set ``RECOVAR_EM_XLA_DEFAULTS`` live here. Each test imports an entry in a
fresh interpreter and reads the XLA flags jax was configured with.
"""

from __future__ import annotations

import json
import os
import subprocess

import pytest

pytestmark = pytest.mark.unit

# Every EM entry point must opt in.
ENTRIES = ("relax.refinement.full_refinement", "relax.commands.initial_model")
MARKER = "RECOVAR_EM_XLA_DEFAULTS"
# The flag recovar's EM policy adds to XLA_FLAGS when the marker is on.
EM_FLAG = "--xla_gpu_autotune_level=0"


def _flags_after_import(entry, marker):
    from conftest import repo_python_command, repo_subprocess_env

    env = repo_subprocess_env({k: v for k, v in os.environ.items() if k not in (MARKER, "XLA_FLAGS")})
    if marker is not None:
        env[MARKER] = marker
    code = f"import os, json, {entry}; print(json.dumps([os.environ.get('{MARKER}'), os.environ.get('XLA_FLAGS', '')]))"
    proc = subprocess.run(repo_python_command("-c", code), env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("entry", ENTRIES)
def test_the_em_entry_sets_the_marker_before_importing_jax(entry):
    """`XLA_FLAGS` is read when jax is imported, so the order is load-bearing: importing the entry turns the
    EM flags on."""
    marker, flags = _flags_after_import(entry, None)
    assert marker == "1"
    assert EM_FLAG in flags.split()


@pytest.mark.parametrize("entry", ENTRIES)
def test_the_entry_uses_setdefault_so_an_explicit_zero_wins(entry):
    marker, flags = _flags_after_import(entry, "0")
    assert marker == "0"
    assert EM_FLAG not in flags.split()
