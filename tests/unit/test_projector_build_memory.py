"""The host projector build fits the single working set the device allows (relax#46)."""

import subprocess

import jax
import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.helpers import xla_memory_reserve
from relax.relion import relion_projector_setup as setup

pytestmark = pytest.mark.unit

GIB = 2**30


def test_single_working_set_is_a_quarter_of_the_pool_limit():
    assert xla_memory_reserve.single_working_set_bytes(int(11.18 * GIB)) == int(0.25 * 11.18 * GIB)


@pytest.mark.parametrize(
    ("working_set_gib", "chunk_gib"),
    [(None, 2.0), (18.0, 2.0), (2.55, 2.55 / 3.5), (0.35, 0.1)],  # off GPU; 80 GB; 16 GB (box 448); tiny
)
def test_chunks_are_lowered_to_the_working_set(working_set_gib, chunk_gib):
    working_set = None if working_set_gib is None else int(working_set_gib * GIB)
    assert setup._host_build_chunk_bytes(working_set) == pytest.approx(chunk_gib * GIB, rel=1e-6)


def test_xy_stage_on_the_host_gives_the_same_window(monkeypatch):
    reference = np.random.default_rng(65).normal(size=(16,) * 3).astype(np.float64)
    monkeypatch.setattr(setup, "single_working_set_bytes", lambda: None)
    whole = setup.setup_relion_projector_on_host(reference, 6, box_size=16, padding_factor=2)
    # A one-byte working set sends the xy stage to the host and makes every chunk one plane.
    monkeypatch.setattr(setup, "single_working_set_bytes", lambda: 1)
    calls = []
    real = setup._window_in_chunks
    monkeypatch.setattr(setup, "_window_in_chunks", lambda *a, **k: calls.append(k["xy_on_host"]) or real(*a, **k))
    on_host = setup.setup_relion_projector_on_host(reference, 6, box_size=16, padding_factor=2)
    assert calls == [True]
    assert_matches(on_host[0], whole[0])
    assert_matches(on_host[1], whole[1])


_PEAK_CHILD = """
import jax, numpy as np
from relax.relion import relion_projector_setup as setup
setup.single_working_set_bytes = lambda: int(2.55 * 2**30)  # a 16 GB card's pool at box 448 (limit 10.22 GiB)
reference = np.random.default_rng(0).standard_normal((448,) * 3)
device = jax.devices()[0]
before = device.memory_stats()["bytes_in_use"]
setup.setup_relion_projector_on_host(reference, 224, box_size=448, padding_factor=2)
stats = device.memory_stats()
print("PEAK_GIB", (stats["peak_bytes_in_use"] - before) / 2**30, "LARGEST_GIB", stats["largest_alloc_size"] / 2**30)
"""


@pytest.mark.gpu
def test_box_448_build_fits_a_16_gb_working_set():
    """Measured on an A100: high-water 3.22 GiB, largest allocation 0.73 GiB (fixed 2 GiB chunks and the xy stage on
    the device: 10.37 GiB and 2.70 GiB, which a real V100 and P100 could not place; relax#46)."""
    if jax.default_backend() != "gpu":
        pytest.skip("needs a GPU")
    from conftest import repo_python_command, repo_subprocess_env

    env = repo_subprocess_env()
    # A fresh process with the default BFC allocator, so the peak and the largest allocation are this build's.
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env.pop("TF_GPU_ALLOCATOR", None)
    env.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)
    proc = subprocess.run(repo_python_command("-c", _PEAK_CHILD), env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr[-2000:]
    (line,) = [line for line in proc.stdout.splitlines() if line.startswith("PEAK_GIB")]
    fields = line.split()
    assert float(fields[1]) <= 4.0 and float(fields[3]) <= 1.0, line
