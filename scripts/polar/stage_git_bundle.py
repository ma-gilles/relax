#!/usr/bin/env python3
"""Stage committed checkout history for Polar jobs with Git ancestry gates."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

from submit import DEFAULT_ROOT, digest, remote, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--host", default="polar")
    parser.add_argument("--root", default=f"{DEFAULT_ROOT}/git")
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    head = run("git", "-C", str(checkout), "rev-parse", "HEAD")
    with tempfile.TemporaryDirectory() as temp:
        bundle = Path(temp) / "history.bundle"
        run("git", "-C", str(checkout), "bundle", "create", str(bundle), "HEAD")
        run("git", "-C", str(checkout), "bundle", "verify", str(bundle))
        sha = digest(bundle)
        target = f"{args.root.rstrip('/')}/{sha}.bundle"
        remote(args.host, "mkdir", "-p", args.root.rstrip("/"))
        if subprocess.run(["ssh", args.host, shlex.join(["test", "-f", target])], check=False).returncode:
            run("rsync", "-a", "--partial", str(bundle), f"{args.host}:{target}")
        if remote(args.host, "sha256sum", target).split()[0] != sha:
            raise RuntimeError("Git bundle changed during transfer")
    print(json.dumps({"head": head, "bundle_sha256": sha, "polar_path": target}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError) as exc:
        print(f"polar stage_git_bundle: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
