#!/usr/bin/env python3
"""Transfer a lock-matched packed pixi environment and install it with Polar Slurm."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from submit import DEFAULT_ROOT, digest, remote, run, stage_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--env", type=Path, help="existing Della pixi environment to pack")
    parser.add_argument("--archive", type=Path, help="already packed environment archive")
    parser.add_argument("--host", default="polar")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    args = parser.parse_args()
    if bool(args.env) == bool(args.archive):
        parser.error("supply exactly one of --env and --archive")
    checkout = args.checkout.resolve()
    lock = checkout / "pixi.lock"
    if not lock.is_file():
        parser.error("checkout needs a pixi.lock")
    key = digest(lock)
    root = args.root.rstrip("/")
    env_root = f"{root}/envs/{key}"
    exists = subprocess.run(["ssh", args.host, shlex.join(["test", "-f", f"{env_root}/.polar-ready"])], check=False)
    if exists.returncode == 0:
        print(f"Environment already ready: {env_root}")
        return
    if args.env:
        env = args.env.resolve()
        env_checkout = env.parent.parent.parent
        if digest(env_checkout / "pixi.lock") != key:
            parser.error("environment checkout and target checkout have different pixi.lock hashes")
        archive = (env_checkout.parent / f"polar-env-{key[:12]}.tar.gz").resolve()
        run(
            "conda-pack",
            "-p",
            str(env),
            "-o",
            str(archive),
            "--ignore-editable-packages",
            "--ignore-missing-files",
            "--compress-level",
            "1",
            "--n-threads",
            "8",
            "--quiet",
        )
    else:
        archive = args.archive.resolve()
        if not archive.is_file():
            parser.error("--archive does not exist")
    _, source, _ = stage_snapshot(checkout, args.host, root)
    remote_archive = f"{root}/bootstrap/env-{key}.tar.gz"
    remote(args.host, "mkdir", "-p", f"{root}/bootstrap")
    run("rsync", "-a", "--partial", str(archive), f"{args.host}:{remote_archive}")
    local_hash = digest(archive)
    remote_hash = remote(args.host, "sha256sum", remote_archive).split()[0]
    if local_hash != remote_hash:
        raise RuntimeError("environment archive changed during transfer")
    installer = Path(__file__).with_name("install_packed_env.sbatch")
    job = f"{root}/bootstrap/install-{digest(installer)[:12]}.sbatch"
    run("rsync", "-a", str(installer), f"{args.host}:{job}")
    exports = ",".join(
        ["ALL", f"POLAR_ENV_ARCHIVE={remote_archive}", f"POLAR_ENV_ROOT={env_root}", f"POLAR_SOURCE_ROOT={source}"]
    )
    job_id = remote(
        args.host,
        "sbatch",
        "--parsable",
        f"--output={root}/bootstrap/install-%j.out",
        f"--error={root}/bootstrap/install-%j.err",
        f"--export={exports}",
        job,
    )
    print(f"install_job_id={job_id}")
    print(f"archive_sha256={local_hash}")
    print(f"stdout={root}/bootstrap/install-{job_id}.out")
    print(f"stderr={root}/bootstrap/install-{job_id}.err")
    print(f"environment={env_root}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"polar bootstrap: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
