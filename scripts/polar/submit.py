#!/usr/bin/env python3
"""Freeze a Della checkout and explicit input files, then submit on Polar.

This command runs on Della. Only ``sbatch`` runs the workload on Polar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

DEFAULT_ROOT = "/scratch/universal/mg6942/relax-polar"
DEFAULT_DATA_ROOT = "/scratch/network/mg6942/relax-polar"


def run(*argv: str, input_bytes: bytes | None = None) -> str:
    result = subprocess.run(argv, input=input_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode:
        raise RuntimeError(f"{shlex.join(argv)} failed ({result.returncode}): {result.stderr.decode(errors='replace')}")
    return result.stdout.decode().strip()


def remote(host: str, *argv: str) -> str:
    return run("ssh", host, shlex.join(argv))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_snapshot(checkout: Path, host: str, root: str) -> tuple[str, str, str]:
    raw = subprocess.check_output(["git", "-C", str(checkout), "ls-files", "-c", "-o", "--exclude-standard", "-z"])
    paths = sorted({Path(os.fsdecode(item)) for item in raw.split(b"\0") if item})
    if any((checkout / path).is_symlink() for path in paths):
        raise RuntimeError("source symlinks are unsupported; copy their target into the checkout")
    paths = [path for path in paths if (checkout / path).is_file()]
    if not paths or any("\n" in str(path) or "\r" in str(path) for path in paths):
        raise RuntimeError("empty checkout or source path containing a newline")
    manifest = "".join(f"{digest(checkout / path)}  {path.as_posix()}\n" for path in paths)
    source_id = hashlib.sha256(manifest.encode()).hexdigest()
    source = f"{root}/code/{source_id}"
    head = run("git", "-C", str(checkout), "rev-parse", "HEAD")
    exists = subprocess.run(["ssh", host, shlex.join(["test", "-f", f"{source}/.polar-ready"])], check=False)
    if exists.returncode == 0:
        remote(host, "bash", "-lc", f"cd {shlex.quote(source)} && sha256sum -c .polar-sha256 >/dev/null")
        return source_id, source, head
    staging = f"{source}.uploading-{uuid.uuid4().hex[:10]}"
    remote(host, "mkdir", "-p", staging)
    with tempfile.TemporaryDirectory() as temp:
        listing = Path(temp) / "files"
        listing.write_bytes(b"\0".join(os.fsencode(str(path)) for path in paths) + b"\0")
        run(
            "rsync",
            "-a",
            "--from0",
            f"--files-from={listing}",
            f"{checkout}/",
            f"{host}:{staging}/",
        )
        manifest_file = Path(temp) / ".polar-sha256"
        manifest_file.write_text(manifest)
        run("rsync", "-a", str(manifest_file), f"{host}:{staging}/.polar-sha256")
        metadata = Path(temp) / ".polar-source.json"
        metadata.write_text(json.dumps({"head": head, "source_id": source_id, "files": len(paths)}, indent=2) + "\n")
        run("rsync", "-a", str(metadata), f"{host}:{staging}/.polar-source.json")
    remote(host, "bash", "-lc", f"cd {shlex.quote(staging)} && sha256sum -c .polar-sha256 >/dev/null")
    remote(host, "touch", f"{staging}/.polar-ready")
    remote(host, "chmod", "-R", "a-w", staging)
    remote(host, "mv", "-T", staging, source)
    return source_id, source, head


def stage_inputs(paths: list[Path], host: str, root: str) -> tuple[str, dict[str, str]]:
    names = [path.name for path in paths]
    if len(names) != len(set(names)):
        raise RuntimeError("input basenames must be unique")
    hashes = {path.name: digest(path) for path in paths}
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    target = f"{root}/data/{identity}"
    remote(host, "mkdir", "-p", target)
    for path in paths:
        remote_path = f"{target}/{path.name}"
        run("rsync", "-a", "--partial", str(path), f"{host}:{remote_path}")
        actual = remote(host, "sha256sum", remote_path).split()[0]
        if actual != hashes[path.name]:
            raise RuntimeError(f"input changed during transfer: {path}")
    return target, hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", help="Slurm script relative to the checkout")
    parser.add_argument("--checkout", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--input", type=Path, action="append", default=[], help="explicit Della data file; repeatable")
    parser.add_argument("--host", default="polar")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="Polar input/output root")
    parser.add_argument(
        "--set",
        dest="settings",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="additional Slurm environment variable; repeatable",
    )
    parser.add_argument(
        "--stage-only", action="store_true", help="sync code without submitting (for environment setup)"
    )
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    job = (checkout / args.job).resolve()
    if not job.is_relative_to(checkout) or not job.is_file():
        parser.error("job must be a file inside --checkout")
    job_relative = job.relative_to(checkout).as_posix()
    inputs = [path.resolve() for path in args.input]
    if any(not path.is_file() for path in inputs):
        parser.error("every --input must be a regular file")
    settings = {}
    for item in args.settings:
        if "=" not in item:
            parser.error("--set requires NAME=VALUE")
        name, value = item.split("=", 1)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name) or "," in value or "\n" in value:
            parser.error("invalid --set name or value (commas and newlines are unsupported)")
        if name in {"POLAR_SOURCE_ROOT", "POLAR_INPUT_ROOT", "POLAR_ENV_ROOT", "POLAR_RUN_ROOT"}:
            parser.error(f"{name} is managed by the submitter")
        settings[name] = value
    lock = checkout / "pixi.lock"
    if not lock.is_file():
        parser.error("checkout needs a pixi.lock")
    env_key = digest(lock)
    root = args.root.rstrip("/")
    data_base = args.data_root.rstrip("/")
    env_root = f"{root}/envs/{env_key}"
    source_id, source, head = stage_snapshot(checkout, args.host, root)
    if args.stage_only:
        print(
            json.dumps(
                {
                    "head": head,
                    "source_id": source_id,
                    "source": source,
                    "env_lock_sha256": env_key,
                    "env_root": env_root,
                },
                indent=2,
            )
        )
        return
    remote(args.host, "test", "-f", f"{env_root}/.polar-ready")
    data_root, input_hashes = stage_inputs(inputs, args.host, data_base)
    run_id = uuid.uuid4().hex[:12]
    run_root = f"{data_base}/runs/{run_id}"
    remote(args.host, "mkdir", "-p", run_root)
    exports = ",".join(
        [
            "ALL",
            f"POLAR_SOURCE_ROOT={source}",
            f"POLAR_INPUT_ROOT={data_root}",
            f"POLAR_ENV_ROOT={env_root}",
            f"POLAR_RUN_ROOT={run_root}",
            *(f"{name}={value}" for name, value in settings.items()),
        ]
    )
    job_id = remote(
        args.host,
        "sbatch",
        "--parsable",
        f"--chdir={source}",
        f"--output={run_root}/slurm-%j.out",
        f"--error={run_root}/slurm-%j.err",
        f"--export={exports}",
        f"{source}/{job_relative}",
    )
    receipt = {
        "job_id": job_id,
        "head": head,
        "source_id": source_id,
        "source": source,
        "env_lock_sha256": env_key,
        "inputs": input_hashes,
        "settings": settings,
        "run_root": run_root,
        "stdout": f"{run_root}/slurm-{job_id}.out",
        "stderr": f"{run_root}/slurm-{job_id}.err",
    }
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "submission.json"
        path.write_text(json.dumps(receipt, indent=2) + "\n")
        run("rsync", "-a", str(path), f"{args.host}:{run_root}/submission.json")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"polar submit: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
