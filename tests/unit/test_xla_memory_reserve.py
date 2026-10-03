"""The XLA pool reserve for RELION's projector texture (relax.helpers.xla_memory_reserve)."""

from __future__ import annotations

import os
import struct
import subprocess
import sys
from pathlib import Path

import mrcfile
import numpy as np
import pytest
from conftest import repo_subprocess_env

from relax.helpers import xla_memory_reserve as reserve

H100_TOTAL_BYTES = 81559 * 1024**2  # nvidia-smi memory.total of the della H100s


def test_texture_bytes_follow_relions_projector_slab():
    # 10202 at current size 626: the 7.9 GB texture that no longer fitted beside a 0.90 pool.
    assert reserve.relion_projector_texture_bytes(626, 2) == 1255 * 1255 * 628 * 8
    # Full box 800, the largest slab of that run.
    assert reserve.relion_projector_texture_bytes(800, 2) == 1603 * 1603 * 802 * 8
    assert reserve.relion_projector_texture_bytes(800, 2) == pytest.approx(16.49e9, rel=1e-3)
    # Odd boxes use r = box // 2, as RELION's integer division does.
    assert reserve.relion_projector_texture_bytes(257, 2) == reserve.relion_projector_texture_bytes(256, 2)


def test_small_boxes_keep_the_configured_fraction():
    assert reserve.xla_memory_fraction(256, 2, H100_TOTAL_BYTES) == 0.90
    assert reserve.xla_memory_fraction(400, 2, H100_TOTAL_BYTES) == 0.90


def test_large_boxes_leave_the_texture_and_context_outside_the_pool():
    for box in (560, 626, 700, 800):
        fraction = reserve.xla_memory_fraction(box, 2, H100_TOTAL_BYTES, context_bytes=3 * 1024**3)
        outside = (1.0 - fraction) * H100_TOTAL_BYTES
        texture = reserve.relion_projector_texture_bytes(box, 2)
        assert fraction < 0.90
        assert outside == pytest.approx(texture + 3 * 1024**3, rel=1e-9)
    assert reserve.xla_memory_fraction(800, 2, H100_TOTAL_BYTES) == pytest.approx(0.7696, abs=1e-4)


def test_a_lower_configured_fraction_and_the_floor_are_respected():
    assert reserve.xla_memory_fraction(800, 2, H100_TOTAL_BYTES, current_fraction=0.5) == 0.5
    # A texture larger than half the card cannot be accommodated; XLA keeps its floor.
    assert reserve.xla_memory_fraction(1200, 2, 40 * 1024**3) == 0.5


def test_model_box_is_read_from_the_first_existing_reference_header(tmp_path):
    path = tmp_path / "ref.mrc"
    with mrcfile.new(str(path)) as handle:
        handle.set_data(np.zeros((24, 24, 24), dtype=np.float32))
    assert reserve.model_box_from_map_headers([None, str(tmp_path / "missing.mrc"), str(path)]) == 24
    assert reserve.model_box_from_map_headers([None, str(tmp_path / "missing.mrc")]) is None


def test_reserve_lowers_the_environment_before_backend_start(monkeypatch):
    monkeypatch.setenv(reserve.MEM_FRACTION_ENV, ".90")
    monkeypatch.setattr(reserve, "_jax_backend_initialized", lambda: False)
    monkeypatch.setattr(reserve, "_visible_device_total_bytes", lambda: H100_TOTAL_BYTES)
    record = reserve.reserve_projector_texture_memory(800, padding_factor=2)
    assert record["fraction"] == pytest.approx(0.7696, abs=1e-4)
    assert record["texture_bytes"] == 1603 * 1603 * 802 * 8
    assert float(reserve.os.environ[reserve.MEM_FRACTION_ENV]) == pytest.approx(record["fraction"], abs=1e-4)
    assert "15.35 GiB RELION projector texture of a 800-pixel model" in reserve.format_reserve_record(record)

    monkeypatch.setenv(reserve.MEM_FRACTION_ENV, ".90")
    assert reserve.reserve_projector_texture_memory(256, padding_factor=2)["fraction"] == 0.90
    assert reserve.os.environ[reserve.MEM_FRACTION_ENV] == ".90"


def test_reserve_does_nothing_after_backend_start_or_without_a_device(monkeypatch):
    monkeypatch.setenv(reserve.MEM_FRACTION_ENV, ".90")
    monkeypatch.setattr(reserve, "_visible_device_total_bytes", lambda: H100_TOTAL_BYTES)
    monkeypatch.setattr(reserve, "_jax_backend_initialized", lambda: True)
    record = reserve.reserve_projector_texture_memory(800, padding_factor=2)
    assert "skipped" in record and "skipped" in reserve.format_reserve_record(record)
    monkeypatch.setattr(reserve, "_jax_backend_initialized", lambda: False)
    monkeypatch.setattr(reserve, "_visible_device_total_bytes", lambda: None)
    assert reserve.reserve_projector_texture_memory(800, padding_factor=2) is None
    assert reserve.reserve_projector_texture_memory(None, padding_factor=2) is None
    assert reserve.os.environ[reserve.MEM_FRACTION_ENV] == ".90"


def test_reference_maps_follow_the_drivers_flags(tmp_path, monkeypatch):
    data_dir = tmp_path / "inputs"
    data_dir.mkdir()
    with mrcfile.new(str(data_dir / "reference_init_relion.mrc")) as handle:
        handle.set_data(np.zeros((32, 32, 32), dtype=np.float32))
    assert reserve.reference_maps_from_argv(["-m", "pytest", "tests/unit"]) == []
    paths = reserve.reference_maps_from_argv(["--data_dir", str(data_dir), "--max_iter", "3"])
    assert reserve.model_box_from_map_headers(paths) == 32
    explicit = reserve.reference_maps_from_argv(["--data_dir", str(data_dir), "--init_volume", "/x/ref.mrc"])
    assert explicit[0] == "/x/ref.mrc"
    classes = reserve.reference_maps_from_argv(["--data_dir", str(data_dir), "--init_class_volumes", "/a.mrc, /b.mrc"])
    assert classes[1] == "/a.mrc"

    monkeypatch.setenv(reserve.MEM_FRACTION_ENV, ".90")
    monkeypatch.setattr(reserve, "_jax_backend_initialized", lambda: False)
    monkeypatch.setattr(reserve, "_visible_device_total_bytes", lambda: H100_TOTAL_BYTES)
    assert reserve.reserve_for_reference_maps(paths, padding_factor=2)["model_box"] == 32
    assert reserve.reserve_for_reference_maps([], padding_factor=2) is None


def test_module_imports_only_the_standard_library():
    """Entry points load this file by path before jax, recovar or relax is imported;
    a relax or recovar import here would start the JAX backend before the reserve."""

    import ast
    import sys

    tree = ast.parse(Path(reserve.__file__).read_text())
    top_level = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
            top_level.add(node.module.split(".")[0])
    assert top_level <= set(sys.stdlib_module_names), top_level


def _raw_mrc_header(path, box):
    """A header-only MRC file of a ``box``-cubed float32 map (the reserve reads only the header)."""

    header = bytearray(1024)
    struct.pack_into("<4i", header, 0, box, box, box, 2)
    header[208:212] = b"MAP "
    header[212:216] = bytes([0x44, 0x44, 0, 0])
    path.write_bytes(bytes(header))


@pytest.mark.parametrize("launch", ["console", "module"])
def test_refine_command_reserves_before_the_jax_backend_starts(tmp_path, launch):
    """``relax refine`` (console script or ``python -m relax.commands.refine``) applies the reserve
    in ``relax/__init__.py``, before recovar starts the backend: a 640-pixel reference lowers the
    fraction below 0.90."""

    data_dir = tmp_path / "inputs"
    data_dir.mkdir()
    _raw_mrc_header(data_dir / "reference_init_relion.mrc", 640)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    smi = fake_bin / "nvidia-smi"
    smi.write_text("#!/bin/sh\necho '0, GPU-fake, 81559'\n")
    smi.chmod(0o755)
    report = "import os, relax; print(os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']); print(relax._XLA_RESERVE_LOG_LINE)"
    if launch == "console":
        command = [sys.executable, "-c", f"import sys; sys.argv = ['relax', 'refine', '--data_dir', {str(data_dir)!r}]; {report}"]
    else:
        # ``python -m relax.commands.refine`` names the command in sys.orig_argv; this child
        # imports the package the way that launch does, with the command's arguments in argv.
        launcher = tmp_path / "launch_module.py"
        launcher.write_text(f"import sys\nsys.orig_argv = [sys.executable, '-m', 'relax.commands.refine']\n{report}\n")
        command = [sys.executable, str(launcher), "--data_dir", str(data_dir)]
    # The launcher script's directory, not the repo root, heads the child's sys.path, so pin the import root:
    # a shared environment's editable install can otherwise resolve ``relax`` to another checkout.
    env = repo_subprocess_env({k: v for k, v in os.environ.items() if k != reserve.MEM_FRACTION_ENV})
    env.update(PATH=f"{fake_bin}:{env.get('PATH', '')}", CUDA_VISIBLE_DEVICES="0", JAX_PLATFORMS="cpu")
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-2000:]
    fraction_line, log_line = result.stdout.strip().splitlines()[-2:]
    expected = reserve.xla_memory_fraction(640, 2, H100_TOTAL_BYTES)
    assert expected < 0.90
    assert float(fraction_line) == pytest.approx(expected, abs=1e-4)
    assert "640-pixel model" in log_line
