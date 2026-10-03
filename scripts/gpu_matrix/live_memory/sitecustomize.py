"""Record the JAX GPU allocator's live memory once a second for the GPU matrix (scripts/gpu_matrix/run_cell.py).

run_cell.py puts this directory first on the relax process's PYTHONPATH and names the output JSONL in
``RELAX_MATRIX_LIVE_MEMORY``; Python then imports this file as ``sitecustomize`` at start-up. A daemon thread
reads ``memory_stats()`` (``bytes_in_use``, ``peak_bytes_in_use``, ``bytes_reserved``, ``bytes_limit``) only
after the program has initialised the JAX backend itself, so the run's own XLA memory settings stay in force.
"""

import os

_PATH = os.environ.get("RELAX_MATRIX_LIVE_MEMORY")
if _PATH:
    import json
    import sys
    import threading
    import time

    def _poll(path: str) -> None:
        with open(path, "a", buffering=1) as out:
            while True:
                time.sleep(1.0)
                bridge = sys.modules.get("jax._src.xla_bridge")
                if bridge is None or not getattr(bridge, "_backends", None):
                    continue
                try:
                    import jax

                    devices = [d for d in jax.devices() if d.platform in ("gpu", "cuda")]
                    if not devices:
                        return
                    stats = devices[0].memory_stats() or {}
                except Exception:
                    continue
                record = {
                    key: stats.get(key)
                    for key in ("bytes_in_use", "peak_bytes_in_use", "bytes_reserved", "bytes_limit")
                }
                out.write(json.dumps({"t": time.time(), **record}) + "\n")

    threading.Thread(target=_poll, args=(_PATH,), daemon=True, name="gpu-matrix-live-memory").start()
