"""The physical free-memory reading comes from NVML, selected as nvidia-smi's rows are."""

from __future__ import annotations

import ctypes

import pytest

from relax.runtime import memory_budget as budget
from relax.runtime.engine_tuning import EngineTuning

pytestmark = pytest.mark.unit


class _FakeNvml:
    """``nvmlDeviceGetMemoryInfo`` for two devices, handles 10 and 11."""

    def __init__(self, free_by_handle):
        self.free_by_handle = free_by_handle

    def nvmlDeviceGetMemoryInfo(self, handle, memory_ref):
        memory = ctypes.cast(memory_ref, ctypes.POINTER(budget._NvmlMemory)).contents
        memory.free = self.free_by_handle[handle]
        return 0


@pytest.fixture
def two_devices(monkeypatch):
    fake = _FakeNvml({10: 111, 11: 222})
    handles = {"0": 10, "GPU-aaa": 10, "1": 11, "GPU-bbb": 11}
    monkeypatch.setattr(budget, "_nvml_devices", lambda: (fake, handles))


@pytest.mark.parametrize(
    ("visible", "free"),
    [(None, 111), ("1", 222), ("GPU-bbb", 222), ("GPU-zzz,0", 111), ("", None), ("-1", None), ("GPU-zzz", None)],
)
def test_the_first_visible_device_is_read(two_devices, visible, free):
    assert budget._nvml_free_memory_bytes(visible) == free


def test_the_subprocess_is_the_fallback_without_nvml(monkeypatch):
    monkeypatch.setattr(budget, "_nvml_devices", lambda: False)
    calls = []

    class _Done:
        returncode = 0
        stdout = "0, GPU-aaa, 7\n"

    def run(*args, **kwargs):
        calls.append(args)
        return _Done()

    monkeypatch.setattr(budget.subprocess, "run", run)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert budget.device_free_memory_bytes() == 7 * 1024**2
    assert calls


def test_the_device_total_is_nvidia_smis_whole_mebibytes_without_a_subprocess(monkeypatch):
    """The memory limit reads NVML's total, floored to the MiB figure nvidia-smi prints, and spawns nothing."""

    class _Total:
        def nvmlDeviceGetMemoryInfo(self, handle, memory_ref):
            memory = ctypes.cast(memory_ref, ctypes.POINTER(budget._NvmlMemory)).contents
            memory.total = 81559 * 1024**2 + 12345
            return 0

    def no_subprocess(*args, **kwargs):
        raise AssertionError("nvidia-smi was spawned although NVML answered")

    monkeypatch.setattr(budget, "_nvml_devices", lambda: (_Total(), {"0": 10, "GPU-aaa": 10}))
    monkeypatch.setattr(budget.subprocess, "run", no_subprocess)
    monkeypatch.setattr(budget, "_jax_allocator_limit_bytes", lambda: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-aaa")
    assert budget.device_memory_limit_bytes(None) == 81559 * 1024**2
    monkeypatch.setattr(budget, "_jax_allocator_limit_bytes", lambda: 40 * 1024**3)
    assert budget.device_memory_limit_bytes(None) == 40 * 1024**3


@pytest.mark.parametrize("value", ["eighty", "80GB", "0", "-4", "nan", "inf", " "])
def test_a_malformed_device_memory_override_is_refused(monkeypatch, value):
    """deep2 V2-4: RELAX_SPARSE_PASS2_DEVICE_MEMORY_GB is a positive number or an error, never ignored."""
    monkeypatch.setenv("RELAX_SPARSE_PASS2_DEVICE_MEMORY_GB", value)
    with pytest.raises(ValueError, match="RELAX_SPARSE_PASS2_DEVICE_MEMORY_GB must be a positive number"):
        EngineTuning.from_environ()


def test_a_device_memory_override_is_taken_and_an_empty_one_is_unset(monkeypatch):
    """The command reads the override once (EngineTuning.device_memory_bytes); the probe takes it instead."""
    monkeypatch.setenv("RELAX_SPARSE_PASS2_DEVICE_MEMORY_GB", "1.5")
    override = EngineTuning.from_environ().device_memory_bytes
    assert override == int(1.5 * 1024**3)
    assert budget.device_memory_limit_bytes(override) == int(1.5 * 1024**3)
    monkeypatch.setenv("RELAX_SPARSE_PASS2_DEVICE_MEMORY_GB", "")
    assert EngineTuning.from_environ().device_memory_bytes is None is EngineTuning().device_memory_bytes
    monkeypatch.setattr(budget, "_nvml_memory_bytes", lambda *args: 7 * 1024**2)
    monkeypatch.setattr(budget, "_jax_allocator_limit_bytes", lambda: None)
    assert budget.device_memory_limit_bytes(None) == 7 * 1024**2
