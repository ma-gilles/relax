#!/usr/bin/env python3
"""Run one cross-GPU matrix cell and record whether it completed, its wall and its peak GPU memory.

    run_cell.py CELL --out DIR [--fixture NAME=PATH ...] [--emulate-gb X]

The relax command runs as a user runs it (``python -m relax.commands.<cmd>``) in a child process.
Peak GPU memory is the largest sum of ``nvidia-smi --query-compute-apps`` used memory over that
process tree, sampled every second; it counts the XLA pool, the CUDA context and every allocation
outside the pool. ``cell.json`` in ``--out`` holds the result.

``--emulate-gb X`` makes a larger card behave as an X GB card: a ballast process holds the rest of
the device, so the physical free memory relax probes matches the small card, and
``XLA_PYTHON_CLIENT_MEM_FRACTION`` is set so the XLA pool limit is what the small card would get
(recovar's 0.90, or relax's projector-texture reserve for ``refine``/``class3d``, scaled to the
device total). nvidia-smi still reports the real total; relax caps it by the pool limit.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cells import CELLS, FIXTURES  # noqa: E402

OOM_PATTERNS = (
    "RESOURCE_EXHAUSTED",
    "out of memory",
    "Out of memory",
    "CUDA_ERROR_OUT_OF_MEMORY",
    "cudaErrorMemoryAllocation",
    "Failed to allocate",
)


def gpu_identity() -> dict:
    out = (
        subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,compute_cap,memory.total,driver_version,compute_mode",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        .stdout.strip()
        .splitlines()
    )
    rows = [
        dict(
            zip(
                ("index", "uuid", "name", "compute_cap", "memory_total_mib", "driver", "compute_mode"),
                (f.strip() for f in line.split(",")),
            )
        )
        for line in out
    ]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    first = visible.split(",")[0].strip() if visible else ""
    for row in rows:
        if first and first in (row["index"], row["uuid"]):
            return row
    return rows[0]


def _children(pid: int) -> set[int]:
    """``pid`` and all its descendants, from /proc."""

    parents: dict[int, int] = {}
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            try:
                with open(f"/proc/{entry}/stat") as handle:
                    fields = handle.read().rsplit(")", 1)[1].split()
                parents[int(entry)] = int(fields[1])
            except (OSError, IndexError, ValueError):
                continue
    tree = {pid}
    changed = True
    while changed:
        changed = False
        for child, parent in parents.items():
            if parent in tree and child not in tree:
                tree.add(child)
                changed = True
    return tree


class PeakSampler(threading.Thread):
    """Samples the GPU memory used by a process tree every ``period`` seconds."""

    def __init__(self, pid: int, period: float = 1.0):
        super().__init__(daemon=True)
        self.pid, self.period = pid, period
        self.peak_mib = 0
        self.samples = 0
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                ).stdout
                tree = _children(self.pid)
                used = 0
                for line in out.strip().splitlines():
                    pid_text, mib = (f.strip() for f in line.split(",")[:2])
                    if pid_text.isdigit() and int(pid_text) in tree and mib.isdigit():
                        used += int(mib)
                self.peak_mib = max(self.peak_mib, used)
                self.samples += 1
            except Exception:
                pass
            self._stop.wait(self.period)

    def stop(self) -> None:
        self._stop.set()


BALLAST = r"""
import ctypes, glob, os, sys, time
target_free = int(float(sys.argv[1]) * 1024**3)
libs = glob.glob(os.path.join(sys.prefix, "lib/python3*/site-packages/nvidia/cuda_runtime/lib/libcudart.so*"))
rt = ctypes.CDLL(libs[0])
free, total = ctypes.c_size_t(), ctypes.c_size_t()
def info():
    assert rt.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)) == 0
    return free.value, total.value
held = []
for chunk in (1 << 30, 1 << 26):
    while True:
        f, _ = info()
        if f - chunk < target_free:
            break
        p = ctypes.c_void_p()
        if rt.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(chunk)) != 0:
            break
        held.append(p)
f, t = info()
print(f"ballast: holding {len(held)} blocks, free {f / 1024**3:.2f} GiB of {t / 1024**3:.2f} GiB", flush=True)
open(sys.argv[2], "w").write(f"{f} {t}\n")
while True:
    time.sleep(3600)
"""


def _touch_when(marker: Path, stop_file: Path, proc: subprocess.Popen) -> None:
    """Create ``stop_file`` once ``marker`` exists (a run-length cap that keeps the run's schedule)."""

    while proc.poll() is None:
        if marker.exists():
            stop_file.touch()
            return
        time.sleep(0.5)


def _relax_module(relative: str):
    """A dependency-free relax module loaded by path, without importing relax (and JAX)."""

    path = HERE.parents[1] / "relax" / relative
    spec = importlib.util.spec_from_file_location("_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Log lines that start a stage: a refinement iteration, a PPCA tile plan (one per PPCA stage).
STAGE_PATTERN = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ .*(=== RELION Iteration \d+|PPCA tile plan: .*)$")


def _artifact_stages(out: Path) -> list[tuple[float, str]]:
    """VDAM InitialModel writes no per-iteration log line; its written iterations (``run_itNNN_model.star``)
    end its stages: each stage starts when the previous written iteration's model was saved."""

    written = sorted(
        (path.stat().st_mtime, int(match.group(1)))
        for path in out.glob("run_it*_model.star")
        if (match := re.match(r"run_it(\d+)_model\.star$", path.name))
    )
    starts, previous = [], 0.0
    for stamp, iteration in written:
        starts.append((previous, f"VDAM iterations up to {iteration}"))
        previous = stamp
    return starts


def live_memory(path: Path, log_text: str, out: Path | None = None) -> dict:
    """The run's live device memory: its peak, and per stage the largest sampled ``bytes_in_use`` and the
    allocator's ``peak_bytes_in_use`` at the stage's end (cumulative). Stages start at ``STAGE_PATTERN`` lines,
    or, for a run without them (VDAM), at its written iterations (:func:`_artifact_stages`)."""

    if not path.exists():
        return {"live_peak_mib": None}
    samples = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    samples = [s for s in samples if s.get("bytes_in_use") is not None]
    if not samples:
        return {"live_peak_mib": None}
    mib = 1 << 20
    starts = []
    for line in log_text.splitlines():
        match = STAGE_PATTERN.match(line)
        if match:
            stamp = time.mktime(time.strptime(match.group(1), "%Y-%m-%d %H:%M:%S"))
            starts.append((stamp, match.group(2)[:120]))
    if not starts and out is not None:
        starts = _artifact_stages(out)
    stages = []
    for index, (begin, label) in enumerate(starts):
        end = starts[index + 1][0] if index + 1 < len(starts) else float("inf")
        inside = [s for s in samples if begin <= s["t"] < end]
        if inside:
            stages.append(
                {
                    "stage": label,
                    "max_in_use_mib": round(max(s["bytes_in_use"] for s in inside) / mib),
                    "peak_in_use_at_end_mib": round((inside[-1].get("peak_bytes_in_use") or 0) / mib),
                }
            )
    return {
        "live_peak_mib": round(max(s.get("peak_bytes_in_use") or s["bytes_in_use"] for s in samples) / mib),
        "pool_limit_mib": round((samples[-1].get("bytes_limit") or 0) / mib),
        "live_stages": stages,
    }


def emulated_fraction(module: str, box: int, emulate_gb: float, total_bytes: int) -> float:
    """The XLA pool fraction of the real device that gives the pool an ``emulate_gb`` card would get."""

    small = emulate_gb * 1024**3
    fraction = 0.90
    if module.rsplit(".", 1)[-1] in ("refine", "class3d"):
        padding_factor = _relax_module("relion/geometry.py").PROJECTION_PADDING_FACTOR
        fraction = _relax_module("helpers/xla_memory_reserve.py").xla_memory_fraction(box, padding_factor, small)
    return fraction * small / total_bytes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cell", choices=sorted(CELLS))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fixture", action="append", default=[], help="NAME=PATH overriding cells.FIXTURES")
    parser.add_argument("--emulate-gb", type=float, default=None)
    parser.add_argument("--timeout", type=float, default=None, help="seconds")
    parser.add_argument(
        "--live-memory",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("MATRIX_LIVE_MEMORY", "1") != "0",
        help="record live memory from inside the run (default; MATRIX_LIVE_MEMORY=0 turns it off). "
        "Keep it off in any run whose wall is quoted: the sampling thread shares the run's process",
    )
    args = parser.parse_args()

    spec = CELLS[args.cell]
    fixtures = dict(FIXTURES)
    for item in args.fixture:
        name, path = item.split("=", 1)
        fixtures[name] = path
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cpus = os.environ.get("SLURM_CPUS_PER_TASK", "8")

    def fill(text: str) -> str:
        text = re.sub(r"\{fx:([A-Za-z0-9_]+)\}", lambda m: fixtures[m.group(1)], text)
        return text.replace("{out}", str(out)).replace("{cpus}", cpus)

    argv = [fill(a) for a in spec["args"]]
    env = dict(os.environ)
    gpu = gpu_identity()
    record = {
        "cell": args.cell,
        "module": spec["module"],
        "argv": argv,
        "gpu": gpu,
        "node": os.uname().nodename,
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "emulate_gb": args.emulate_gb,
    }
    ballast = None
    if args.emulate_gb is not None:
        total = int(gpu["memory_total_mib"]) * 1024**2
        stamp = out / "ballast.free"
        ballast = subprocess.Popen(
            [sys.executable, "-c", BALLAST, str(args.emulate_gb), str(stamp)],
            env={**env, "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", "0")},
            stdout=open(out / "ballast.log", "w"),
            stderr=subprocess.STDOUT,
        )
        for _ in range(120):
            if stamp.exists() or ballast.poll() is not None:
                break
            time.sleep(1)
        if not stamp.exists():
            raise SystemExit(f"ballast failed: {(out / 'ballast.log').read_text()[-2000:]}")
        free_b, _ = (int(v) for v in stamp.read_text().split())
        env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = (
            f"{emulated_fraction(spec['module'], spec['box'], args.emulate_gb, total):.4f}"
        )
        record["emulation"] = {
            "physical_free_after_ballast_gib": free_b / 1024**3,
            "xla_mem_fraction": env["XLA_PYTHON_CLIENT_MEM_FRACTION"],
        }

    # Live memory (memory_stats) from inside the relax process: live_memory/sitecustomize.py.
    live = out / "live_memory.jsonl"
    live.unlink(missing_ok=True)
    record["live_memory_hook"] = bool(args.live_memory)
    if args.live_memory:
        env["RELAX_MATRIX_LIVE_MEMORY"] = str(live)
        env["PYTHONPATH"] = os.pathsep.join(p for p in (str(HERE / "live_memory"), env.get("PYTHONPATH")) if p)
    log = out / "run.log"
    start = time.time()
    with open(log, "w") as handle:
        proc = subprocess.Popen(
            [sys.executable, "-m", spec["module"], *argv],
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        sampler = PeakSampler(proc.pid)
        sampler.start()
        if spec.get("stop_when"):
            marker, stop_file = (Path(fill(p)) for p in spec["stop_when"])
            threading.Thread(target=_touch_when, args=(marker, stop_file, proc), daemon=True).start()
        try:
            rc = proc.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            rc = proc.wait()
            record["timeout"] = True
        sampler.stop()
    wall = time.time() - start
    if ballast is not None:
        ballast.kill()
    text = log.read_text(errors="replace")
    ok_pattern = spec.get("ok_pattern")
    completed = rc == 0 or bool(ok_pattern and ok_pattern in text)
    record.update(
        exit_code=rc,
        completed=completed,
        wall_s=round(wall, 1),
        peak_gpu_mib=sampler.peak_mib,
        memory_samples=sampler.samples,
        oom=[p for p in OOM_PATTERNS if p in text] if not completed else [],
        log_tail=text[-6000:] if not completed else "",
    )
    record.update(live_memory(live, text, out))
    (out / "cell.json").write_text(json.dumps(record, indent=1) + "\n")
    print(
        json.dumps(
            {
                k: record.get(k)
                for k in ("cell", "completed", "exit_code", "wall_s", "peak_gpu_mib", "live_peak_mib", "oom")
            }
        )
    )
    print("GPU:", gpu["name"], gpu["compute_cap"], gpu["memory_total_mib"], "MiB")
    return 0 if completed else 1


if __name__ == "__main__":
    sys.exit(main())
