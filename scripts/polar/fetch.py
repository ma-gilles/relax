#!/usr/bin/env python3
"""Copy a completed Polar run to an explicit Della directory."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from submit import DEFAULT_DATA_ROOT, remote, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="12-digit hex run ID from submission.json")
    parser.add_argument("destination", type=Path, help="Della directory for the run")
    parser.add_argument("--host", default="polar")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{12}", args.run_id):
        parser.error("run ID must contain 12 lowercase hex digits")
    source = f"{args.data_root.rstrip('/')}/runs/{args.run_id}"
    remote(args.host, "test", "-f", f"{source}/submission.json")
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    run("rsync", "-a", "--partial", f"{args.host}:{source}/", f"{destination}/")
    print(destination)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError) as exc:
        print(f"polar fetch: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
