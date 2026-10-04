"""The pure logic of scripts/dev/doctor.py: pin and lock parsing, GPU choice, launcher roots, exit status."""

from pathlib import Path

import pytest

from scripts.dev import doctor

REPO_ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.unit

LOCK = """\
version: 6
environments:
  default:
    channels:
    - url: https://conda.anaconda.org/conda-forge/
    packages:
      linux-64:
      - conda: https://conda.anaconda.org/conda-forge/linux-64/aom-3.9.1-hac33072_0.conda
      - conda: https://conda.anaconda.org/conda-forge/linux-64/blas-2.116-mkl.tar.bz2
      - pypi: https://files.pythonhosted.org/packages/aa/bb/jax_cuda12_plugin-0.9.0.1-cp311-cp311-manylinux_2_27_x86_64.whl
      - pypi: https://files.pythonhosted.org/packages/cc/dd/Some.Sdist-1.2.3.tar.gz
      - pypi: git+https://github.com/ma-gilles/recovar?rev=c60e3c26f893#c60e3c26f893
      - pypi: ./
  dev:
    packages:
      linux-64:
      - conda: https://conda.anaconda.org/conda-forge/linux-64/only-dev-1.0-0.conda
      - pypi: ../recovar
packages:
- conda: https://conda.anaconda.org/conda-forge/linux-64/not-an-environment-entry-1.0-0.conda
"""


def test_recovar_pin_is_read_from_this_manifest():
    text = '[feature.dev.pypi-dependencies]\nrecovar = { path = "../recovar", editable = true }\n'
    assert doctor.recovar_pin(text) is None
    pinned = text + '[feature.pinned.pypi-dependencies]\nrecovar = { git = "https://x/recovar", rev = "c60e3c26f8" }\n'
    assert doctor.recovar_pin(pinned) == "c60e3c26f8"
    pin = doctor.recovar_pin((REPO_ROOT / "pixi.toml").read_text())
    assert pin is not None and len(pin) == 40 and pin in (REPO_ROOT / "pixi.lock").read_text()


def test_locked_packages_of_one_environment():
    locked = doctor.locked_packages(LOCK)
    assert locked == {
        "conda": {"aom-3.9.1-hac33072_0", "blas-2.116-mkl"},
        "pypi": {"jax-cuda12-plugin": "0.9.0.1", "some-sdist": "1.2.3"},
        "git": {"recovar": "c60e3c26f893"},
    }
    assert doctor.locked_packages(LOCK, "dev")["conda"] == {"only-dev-1.0-0"}
    real = doctor.locked_packages((REPO_ROOT / "pixi.lock").read_text())
    assert len(real["conda"]) > 100 and real["pypi"]["jax"] == "0.9.0.1" and "recovar" in real["git"]


def test_lock_differences_name_what_the_environment_lacks():
    locked = doctor.locked_packages(LOCK)
    conda = {"aom-3.9.1-hac33072_0", "blas-2.116-mkl", "extra-1-0"}
    assert doctor.lock_differences(locked, conda, {"jax-cuda12-plugin": "0.9.0.1", "some-sdist": "1.2.3"}) == []
    assert doctor.lock_differences(locked, {"aom-3.9.1-hac33072_0"}, {"jax-cuda12-plugin": "0.9.0"}) == [
        "conda blas-2.116-mkl is not installed",
        "pypi jax-cuda12-plugin is 0.9.0, the lock has 0.9.0.1",
        "pypi some-sdist 1.2.3 is not installed",
    ]


def test_gpu_zero_is_never_offered_and_busy_gpus_are_not_idle():
    gpus = "0, GPU-aaaa, 3\n1, GPU-bbbb, 40123\n2, GPU-cccc, 12\n3, GPU-dddd, 5\n"
    apps = "GPU-dddd, 4242\n"
    assert doctor.idle_gpus(gpus, apps) == ([(2, "GPU-cccc")], [1, 3])
    assert doctor.idle_gpus("0, GPU-aaaa, 0\n") == ([], [])
    assert doctor.idle_gpus("1, GPU-bbbb, [N/A]\n") == ([], [1])


def test_launcher_roots_are_the_fixed_part_of_output_defaults():
    script = (
        'SCRATCH_DIR="${EM_K1_MATRIX_SCRATCH_DIR:-/scratch/x/em_work/codex/${RUN_ID}}"\n'
        'RUNTIME_ROOT="${EM_K1_MATRIX_RUNTIME_ROOT:-/scratch/x/em_work/codex/runtime/${RUN_ID}}"\n'
        'RELION_SRC_DIR="${RELION_SRC_DIR:-/scratch/x/relion/src}"\n'
    )
    assert doctor.launcher_roots(script) == [
        ("EM_K1_MATRIX_SCRATCH_DIR", "/scratch/x/em_work/codex"),
        ("EM_K1_MATRIX_RUNTIME_ROOT", "/scratch/x/em_work/codex/runtime"),
    ]


def test_only_a_fail_makes_the_exit_status_nonzero_and_fixes_follow_their_line():
    results = [
        doctor.Result(doctor.OK, "jax", "0.9.0.1", "not shown"),
        doctor.Result(doctor.WARN, "natives", "no pack", "bash scripts/build_test_natives.sh <dir>"),
    ]
    assert doctor.exit_status(results) == 0
    assert doctor.render(results).splitlines() == [
        "OK   jax: 0.9.0.1",
        "WARN natives: no pack",
        "     fix: bash scripts/build_test_natives.sh <dir>",
        "1 OK, 1 WARN, 0 FAIL",
    ]
    assert doctor.exit_status([*results, doctor.Result(doctor.FAIL, "pixi.lock", "differs", "pixi install")]) == 1


def test_checks_that_need_no_environment_report_on_this_checkout(tmp_path, monkeypatch):
    monkeypatch.delenv("RECOVAR_RELION_BIND_BUILD_DIR", raising=False)  # the tiers export a built binding
    assert doctor.check_fixtures(REPO_ROOT).name == "fixtures"
    assert [result.name for result in doctor.check_write_roots(REPO_ROOT)] == ["tier run root", "launcher roots"]
    assert doctor.check_git(tmp_path).status == doctor.WARN
    missing = doctor.check_pixi_directory(tmp_path, environment_ok=False)
    assert (missing.status, "pixi install --frozen" in missing.fix) == (doctor.WARN, True)
    binding = doctor.check_relion_binding(tmp_path, None)
    assert (binding.status, "build-relion-bind" in binding.fix) == (doctor.WARN, True)
