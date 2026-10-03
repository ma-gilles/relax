"""The GPU matrix runner's live-memory summary (scripts/gpu_matrix/run_cell.py)."""

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "gpu_matrix"))
import run_cell  # noqa: E402

pytestmark = pytest.mark.unit


def test_live_memory_is_summarized_per_stage(tmp_path):
    mib = 1 << 20
    t0 = time.mktime(time.strptime("2026-10-03 13:52:30", "%Y-%m-%d %H:%M:%S"))
    log = "\n".join(
        [
            "2026-10-03 13:52:30,574 relax.ppca_refinement.full_row_stream INFO PPCA tile plan: 150 of 150 images",
            "unrelated line",
            "2026-10-03 13:52:40,100 relax.ppca_refinement.full_row_stream INFO PPCA tile plan: 33 of 150 images",
        ]
    )
    samples = [(1, 100, 100), (5, 300, 300), (11, 200, 300), (15, 900, 900), (20, 400, 900)]
    path = tmp_path / "live_memory.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"t": t0 + dt, "bytes_in_use": a * mib, "peak_bytes_in_use": p * mib, "bytes_limit": 1000 * mib})
            for dt, a, p in samples
        )
    )
    summary = run_cell.live_memory(path, log)
    assert summary["live_peak_mib"] == 900 and summary["pool_limit_mib"] == 1000
    assert [s["max_in_use_mib"] for s in summary["live_stages"]] == [300, 900]
    assert [s["peak_in_use_at_end_mib"] for s in summary["live_stages"]] == [300, 900]
    assert run_cell.live_memory(tmp_path / "missing.jsonl", log) == {"live_peak_mib": None}


def test_vdam_stages_end_at_its_written_iterations(tmp_path):
    """VDAM logs no iteration line; its written ``run_itNNN_model.star`` files end the stages."""

    import os

    mib = 1 << 20
    t0 = 1_700_000_000.0
    for iteration, dt in ((1, 10), (10, 30)):
        model = tmp_path / f"run_it{iteration:03d}_model.star"
        model.write_text("")
        os.utime(model, (t0 + dt, t0 + dt))
    path = tmp_path / "live_memory.jsonl"
    samples = [(5, 100, 100), (20, 700, 700), (25, 300, 700), (40, 200, 700)]
    path.write_text(
        "\n".join(
            json.dumps({"t": t0 + dt, "bytes_in_use": a * mib, "peak_bytes_in_use": p * mib, "bytes_limit": 1000 * mib})
            for dt, a, p in samples
        )
    )
    stages = run_cell.live_memory(path, "no stage lines", tmp_path)["live_stages"]
    assert [s["stage"] for s in stages] == ["VDAM iterations up to 1", "VDAM iterations up to 10"]
    assert [s["max_in_use_mib"] for s in stages] == [100, 700]


def test_the_live_memory_hook_does_not_start_jax(tmp_path):
    """Loading the hook must not initialise the JAX backend (relax sizes its XLA pool before that)."""

    import os
    import subprocess

    hook = Path(run_cell.__file__).resolve().parent / "live_memory"
    code = "import sys; print('jax' in sys.modules)"
    env = dict(os.environ, PYTHONPATH=str(hook), RELAX_MATRIX_LIVE_MEMORY=str(tmp_path / "live.jsonl"))
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
