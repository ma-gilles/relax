"""InitialModel's CLI invocations reach XLA with the EM defaults.

``relax/commands/initial_model.py`` sets ``RECOVAR_EM_XLA_DEFAULTS`` before its
own imports, but ``relax initial_model`` (console script) and
``python -m relax.commands.initial_model`` import ``recovar``, and with it
``recovar.jax_config``, before that module runs. The EM default
``--xla_gpu_autotune_level=0`` therefore never reached XLA for InitialModel,
and autotuning stayed on (with it XLA's fusion autotuner, whose Triton
reductions over-read their operands on Hopper; D2 10345 diagnosis). The ``relax``
package now sets the marker itself, before importing ``recovar``, whenever it is
imported.

The end-to-end cases run the real invocations in child interpreters and read
``XLA_FLAGS`` at exit through a ``sitecustomize`` hook.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import relax

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
MARKER = "RECOVAR_EM_XLA_DEFAULTS"
FLAG = "--xla_gpu_autotune_level=0"


def test_importing_relax_sets_the_marker():
    environ: dict[str, str] = {}
    assert relax._configure_em_xla_defaults(environ=environ) == "1"
    assert environ == {MARKER: "1"}


@pytest.mark.parametrize("explicit", ["0", ""])
def test_an_explicit_marker_wins(explicit):
    environ = {MARKER: explicit}
    assert relax._configure_em_xla_defaults(environ=environ) == explicit


def _xla_flags_at_exit(cmd: list[str], tmp_path: Path, extra_env: dict[str, str] | None = None) -> str:
    hook = tmp_path / "sitecustomize.py"
    hook.write_text(
        "import atexit, os\n"
        "atexit.register(lambda: print('XLA_FLAGS_AT_EXIT=' + os.environ.get('XLA_FLAGS', ''), flush=True))\n"
    )
    env = {
        "PATH": "/usr/bin:/bin",
        # The pinned recovar may come from PYTHONPATH (co-development) rather than site-packages.
        "PYTHONPATH": os.pathsep.join(p for p in (str(tmp_path), str(REPO), os.environ.get("PYTHONPATH", "")) if p),
        "JAX_PLATFORMS": "cpu",
        "RECOVAR_DISABLE_CUDA": "1",
        "PYTHONNOUSERSITE": "1",
        "HOME": str(tmp_path),
        **(extra_env or {}),
    }
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=REPO, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("XLA_FLAGS_AT_EXIT=")]
    assert lines, proc.stdout[-2000:]
    return lines[-1].split("=", 1)[1]


def test_end_to_end_console_script_reaches_xla_with_the_em_default(tmp_path):
    code = (
        "import sys; sys.argv = ['relax', 'initial_model', '--help'];"
        " from relax.command_line import main_commands; main_commands()"
    )
    assert FLAG in _xla_flags_at_exit([sys.executable, "-c", code], tmp_path).split()


def test_end_to_end_python_m_reaches_xla_with_the_em_default(tmp_path):
    cmd = [sys.executable, "-m", "relax.commands.initial_model", "--help"]
    assert FLAG in _xla_flags_at_exit(cmd, tmp_path).split()


def test_end_to_end_any_relax_import_reaches_xla_with_the_em_default(tmp_path):
    code = "import sys; sys.argv = ['relax', 'build_cuda', '--help']; import relax, recovar"
    assert FLAG in _xla_flags_at_exit([sys.executable, "-c", code], tmp_path).split()


def test_end_to_end_recovar_alone_keeps_autotuning(tmp_path):
    assert FLAG not in _xla_flags_at_exit([sys.executable, "-c", "import recovar"], tmp_path).split()
