#!/usr/bin/env python
"""Record a test-tier receipt: source SHA, tier, pass/fail, job id and GPU model.

    python scripts/write_test_receipt.py --run-root <dir> --tier medium --status pass --job 14363910 \
        --gpu-model "NVIDIA H100 80GB HBM3"

Writes ``<run-root>/RECEIPT.json`` and appends the same record as one JSON line to the
file named by ``RELAX_TEST_RECEIPTS`` (an agent's handoff log) when that is set. The tier
commands call this themselves; call it directly for a tier run by hand.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
from pathlib import Path


def _source(run_root: Path) -> dict:
    plan = run_root / "PLAN.json"
    if plan.exists():
        return json.loads(plan.read_text()).get("source", {})
    src = run_root / "src" if (run_root / "src").exists() else Path.cwd()
    head = subprocess.check_output(["git", "-C", str(src), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "-C", str(src), "status", "--porcelain"], text=True).strip())
    return {"head": head, "dirty": dirty, "src": str(src)}


def _json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--tier", required=True, choices=["cpu", "smoke", "medium", "long", "baseline"])
    parser.add_argument("--status", required=True, choices=["pass", "fail"])
    parser.add_argument("--job", required=True, help="Slurm job id(s), or local:<GPU UUID>")
    parser.add_argument("--gpu-model", required=True, help="GPU model(s) the tier ran on, comma-separated")
    parser.add_argument("--wall-s", type=float)
    args = parser.parse_args(argv)
    source = _source(args.run_root)
    receipt = {
        "sha": source.get("head"),
        "dirty": source.get("dirty"),
        "diff_sha256": source.get("diff_sha256"),
        "tier": args.tier,
        "status": args.status,
        "job": args.job,
        "gpu_model": args.gpu_model,
        "wall_s": args.wall_s,
        "run_root": str(args.run_root),
        "summary": str(args.run_root / "SUMMARY.json"),
        # Provenance: the files relax and recovar were imported from, and the native sources.
        "imports": _json(args.run_root / "SUMMARY.json").get("imports"),
        "native_sources": _json(args.run_root / "natives" / "NATIVE.json").get("native_sources"),
        "written_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (args.run_root / "RECEIPT.json").write_text(json.dumps(receipt, indent=1) + "\n")
    log = os.environ.get("RELAX_TEST_RECEIPTS")
    if log:
        with open(log, "a") as f:
            f.write(json.dumps(receipt) + "\n")
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
