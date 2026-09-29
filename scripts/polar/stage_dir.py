#!/usr/bin/env python3
"""Copy a directory from Della to a checksum-verified Polar snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from submit import DEFAULT_ROOT, digest, remote, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--name", required=True, help="stable directory label, such as relion")
    parser.add_argument("--host", default="polar")
    parser.add_argument("--root", default=f"{DEFAULT_ROOT}/deps")
    args = parser.parse_args()
    source = args.source.resolve()
    if not source.is_dir() or not args.name.replace("-", "").replace("_", "").isalnum():
        parser.error("source must be a directory and --name must be alphanumeric")
    paths = sorted(path for path in source.rglob("*") if path.is_file() or path.is_symlink())
    if not paths or any(path.is_symlink() or "\n" in str(path) for path in paths):
        parser.error("empty directory, symlink, or newline in a path")
    manifest = "".join(f"{digest(path)}  {path.relative_to(source).as_posix()}\n" for path in paths)
    identity = hashlib.sha256(manifest.encode()).hexdigest()
    target = f"{args.root.rstrip('/')}/{args.name}/{identity}/{source.name}"
    present = subprocess.run(["ssh", args.host, shlex.join(["test", "-f", f"{target}/.polar-sha256"])], check=False)
    if present.returncode == 0:
        remote(args.host, "bash", "-lc", f"cd {shlex.quote(target)} && sha256sum -c .polar-sha256 >/dev/null")
    else:
        staging = f"{args.root.rstrip('/')}/{args.name}/{identity}.uploading-{uuid.uuid4().hex[:10]}"
        remote(args.host, "mkdir", "-p", f"{staging}/{source.name}")
        run("rsync", "-a", "--partial", f"{source}/", f"{args.host}:{staging}/{source.name}/")
        with tempfile.TemporaryDirectory() as temp:
            manifest_file = Path(temp) / ".polar-sha256"
            manifest_file.write_text(manifest)
            run("rsync", "-a", str(manifest_file), f"{args.host}:{staging}/{source.name}/.polar-sha256")
        remote(
            args.host,
            "bash",
            "-lc",
            f"cd {shlex.quote(staging)}/{shlex.quote(source.name)} && sha256sum -c .polar-sha256 >/dev/null",
        )
        remote(args.host, "touch", f"{staging}/.polar-ready")
        remote(args.host, "chmod", "-R", "a-w", staging)
        remote(args.host, "mv", "-T", staging, f"{args.root.rstrip('/')}/{args.name}/{identity}")
    print(json.dumps({"source": str(source), "files": len(paths), "sha256": identity, "polar_path": target}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError) as exc:
        print(f"polar stage_dir: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
