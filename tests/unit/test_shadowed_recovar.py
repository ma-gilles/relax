"""relax commands refuse a recovar that shadows the installed one (relax/__init__.py _reject_shadowed_recovar)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import relax

pytestmark = pytest.mark.unit
REPO_ROOT = Path(__file__).resolve().parents[2]


def test_a_shadowing_recovar_is_refused_for_relax_commands_only(tmp_path):
    import recovar

    fake = tmp_path / "recovar" / "__init__.py"
    fake.parent.mkdir()
    fake.write_text("")
    command = {"argv": ["/env/bin/relax", "refine"], "orig_argv": ["python", "/env/bin/relax", "refine"], "environ": {}}
    with pytest.raises(RuntimeError, match="relax refuses to start: recovar resolves to") as err:
        relax._reject_shadowed_recovar(origin=str(fake), **command)
    assert str(fake.resolve()) in str(err.value) and str(Path(recovar.__file__).resolve().parent) in str(err.value)
    # The installed recovar, a non-command interpreter and the explicit override all pass.
    relax._reject_shadowed_recovar(origin=recovar.__file__, **command)
    relax._reject_shadowed_recovar(origin=str(fake), argv=["pytest"], orig_argv=["python", "-m", "pytest"], environ={})
    relax._reject_shadowed_recovar(origin=str(fake), **{**command, "environ": {"RELAX_ALLOW_SHADOWED_RECOVAR": "1"}})


def test_python_m_from_a_recovar_checkout_is_refused(tmp_path):
    """``python -m relax.commands.<cmd>`` started inside a recovar checkout: the working directory comes first on
    sys.path, so ``recovar`` would be the checkout's."""

    (tmp_path / "recovar").mkdir()
    (tmp_path / "recovar" / "__init__.py").write_text("")
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu")
    env.pop("RELAX_ALLOW_SHADOWED_RECOVAR", None)
    proc = subprocess.run(
        [sys.executable, "-m", "relax.commands.build_cuda", "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert (
        "relax refuses to start: recovar resolves to " + str((tmp_path / "recovar" / "__init__.py").resolve())
        in proc.stderr
    )
