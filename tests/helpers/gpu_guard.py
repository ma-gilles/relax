"""Keep tests off GPU 0 of the shared development machine.

Physical GPU 0 of the shared login/development node belongs to other users. Outside a Slurm
allocation a test run may use a GPU only when ``CUDA_VISIBLE_DEVICES`` names GPUs other than
GPU 0 (by index or UUID, as ``nvidia-smi -L`` lists them); otherwise the test session runs on
the CPU. Inside Slurm the scheduler's assignment is respected. ``RELAX_ALLOW_GPU0=1`` lifts the
guard on a machine whose GPU 0 is not shared.
"""

from __future__ import annotations

import re
import subprocess

ALLOW_ENV = "RELAX_ALLOW_GPU0"


def gpu_uuids_by_index() -> dict[int, str]:
    """``{index: UUID}`` from ``nvidia-smi -L``; empty when no driver or GPU is present."""
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    found = {}
    for line in out.splitlines():
        match = re.match(r"\s*GPU (\d+):.*\(UUID: (GPU-[0-9a-fA-F-]+)\)", line)
        if match:
            found[int(match.group(1))] = match.group(2)
    return found


def cpu_only_reason(environ, uuids: dict[int, str]) -> str | None:
    """Why this session must stay on the CPU, or None when its GPU selection is allowed."""
    if environ.get("SLURM_JOB_ID") or environ.get(ALLOW_ENV, "") == "1":
        return None
    visible = environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return "outside Slurm with CUDA_VISIBLE_DEVICES unset, GPU 0 of the shared node would be used"
    gpu0 = uuids.get(0)
    for entry in (e.strip() for e in visible.split(",")):
        if entry == "0" or (gpu0 is not None and entry.startswith("GPU-") and gpu0.startswith(entry)):
            return f"outside Slurm, CUDA_VISIBLE_DEVICES={visible} includes GPU 0 of the shared node"
    return None
