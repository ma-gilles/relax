#!/usr/bin/env python3
"""Run several matrix cells in one allocation, one worker per GPU, each cell on one GPU.

    run_cells.py --out-root DIR SPEC [SPEC ...]

A SPEC is ``CELL`` or ``CELL@GB`` (``--emulate-gb GB``). Specs are taken in order by whichever GPU
is free next, so list the longest first. Extra ``run_cell.py`` options (``--fixture``) pass through
``--cell-arg``. Each cell writes ``<out-root>/<CELL>[_emuGB]/cell.json``; ``summary.json`` collects them.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("specs", nargs="+")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--cell-arg", action="append", default=[])
    args = parser.parse_args()

    visible = [d.strip() for d in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if d.strip()]
    work: queue.Queue = queue.Queue()
    for spec in args.specs:
        work.put(spec)
    results: dict[str, dict] = {}
    lock = threading.Lock()

    def worker(device: str) -> None:
        while True:
            try:
                spec = work.get_nowait()
            except queue.Empty:
                return
            cell, _, gb = spec.partition("@")
            name = cell + (f"_emu{gb}" if gb else "")
            out = args.out_root / name
            cmd = [sys.executable, str(HERE / "run_cell.py"), cell, "--out", str(out), *args.cell_arg]
            if gb:
                cmd += ["--emulate-gb", gb]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": device}
            out.mkdir(parents=True, exist_ok=True)
            with open(out / "runner.log", "w") as log:
                rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            record_path = out / "cell.json"
            record = json.loads(record_path.read_text()) if record_path.exists() else {"cell": cell, "runner_rc": rc}
            with lock:
                results[name] = {
                    k: record.get(k)
                    for k in ("cell", "completed", "wall_s", "peak_gpu_mib", "oom", "exit_code", "emulate_gb")
                } | {"runner_rc": rc}
                print(json.dumps({name: results[name]}), flush=True)

    threads = [threading.Thread(target=worker, args=(d,)) for d in visible]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    (args.out_root / "summary.json").write_text(json.dumps(results, indent=1) + "\n")
    return 0 if all(r.get("completed") for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
