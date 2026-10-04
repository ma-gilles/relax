"""relax refuses a recovar that shadows the installed one (relax/__init__.py _reject_shadowed_recovar)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import relax

pytestmark = pytest.mark.unit
REPO_ROOT = Path(__file__).resolve().parents[2]


def test_a_shadowing_recovar_is_refused(tmp_path):
    import recovar

    fake = tmp_path / "recovar" / "__init__.py"
    fake.parent.mkdir()
    fake.write_text("")
    with pytest.raises(RuntimeError, match="relax refuses to import: recovar resolves to") as err:
        relax._reject_shadowed_recovar(origin=str(fake), environ={})
    assert str(fake.resolve()) in str(err.value) and str(Path(recovar.__file__).resolve().parent) in str(err.value)
    # The installed recovar and the explicit override pass.
    relax._reject_shadowed_recovar(origin=recovar.__file__, environ={})
    relax._reject_shadowed_recovar(origin=str(fake), environ={"RELAX_ALLOW_SHADOWED_RECOVAR": "1"})


@pytest.mark.parametrize(
    "argv",
    [["-m", "relax.commands.build_cuda", "--help"], ["-c", "import relax"]],
    ids=["python_m_command", "python_c_import"],
)
def test_relax_started_inside_a_recovar_checkout_is_refused(tmp_path, argv):
    """Started inside a recovar checkout, the working directory comes first on sys.path, so ``recovar`` would be
    the checkout's; every entry point (a command or a plain import) is refused, and the override admits it."""

    (tmp_path / "recovar").mkdir()
    (tmp_path / "recovar" / "__init__.py").write_text("")
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu")
    env.pop("RELAX_ALLOW_SHADOWED_RECOVAR", None)
    proc = subprocess.run([sys.executable, *argv], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert proc.returncode != 0
    expected = "relax refuses to import: recovar resolves to " + str((tmp_path / "recovar" / "__init__.py").resolve())
    assert expected in proc.stderr
    allowed = subprocess.run(
        [sys.executable, "-c", "import relax"],
        cwd=tmp_path,
        env=dict(env, RELAX_ALLOW_SHADOWED_RECOVAR="1"),
        capture_output=True,
        text=True,
    )
    assert "relax refuses to import" not in allowed.stderr
