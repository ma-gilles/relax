#!/usr/bin/env python3
"""Stage selected checksum-pinned EM fixture sets on Polar network scratch.

Run on Della. The recorded source bytes remain immutable; a Polar job may
relocate absolute STAR paths into its private run directory later.
"""

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

from submit import DEFAULT_DATA_ROOT, digest, remote, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sets", nargs="+", help="names in tests/fixtures/em_fixture_manifest.json")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[2] / "tests/fixtures/em_fixture_manifest.json")
    parser.add_argument("--host", default="polar")
    parser.add_argument("--root", default=f"{DEFAULT_DATA_ROOT}/data/fixture_sets")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    all_sets = json.loads(manifest_path.read_text())["sets"]
    names = sorted(set(args.sets))
    if any(name not in all_sets or not name.replace("_", "").isalnum() for name in names):
        parser.error("unknown or unsafe fixture set name")
    selected = {name: all_sets[name] for name in names}
    for name, entry in selected.items():
        root = Path(entry["root"])
        for rel, (size, expected) in entry["files"].items():
            path = root / rel
            if Path(rel).is_absolute() or ".." in Path(rel).parts or "\n" in rel:
                parser.error(f"unsafe fixture path: {name}/{rel}")
            if not path.is_file() or path.stat().st_size != size or digest(path) != expected:
                raise RuntimeError(f"fixture differs from its recorded size or SHA256: {path}")

    record = {"source_manifest_sha256": digest(manifest_path), "sets": selected}
    encoded = (json.dumps(record, sort_keys=True, indent=2) + "\n").encode()
    identity = hashlib.sha256(encoded).hexdigest()
    target = f"{args.root.rstrip('/')}/{identity}"
    if subprocess.run(["ssh", args.host, shlex.join(["test", "-f", f"{target}/.polar-ready"])], check=False).returncode == 0:
        remote(args.host, "bash", "-lc", f"cd {shlex.quote(target)} && sha256sum -c .polar-sha256 >/dev/null")
    else:
        staging = f"{target}.uploading-{uuid.uuid4().hex[:10]}"
        remote(args.host, "mkdir", "-p", staging)
        checksums = []
        with tempfile.TemporaryDirectory() as temp:
            for name, entry in selected.items():
                listing = Path(temp) / f"{name}.files"
                listing.write_bytes(b"\0".join(rel.encode() for rel in entry["files"]) + b"\0")
                remote(args.host, "mkdir", "-p", f"{staging}/raw/{name}")
                run("rsync", "-a", "--partial", "--from0", f"--files-from={listing}", f"{entry['root']}/", f"{args.host}:{staging}/raw/{name}/")
                checksums.extend(f"{sha}  raw/{name}/{rel}\n" for rel, (_, sha) in entry["files"].items())
            file = Path(temp) / "fixture-provenance.json"
            file.write_bytes(encoded)
            run("rsync", "-a", str(file), f"{args.host}:{staging}/fixture-provenance.json")
            checksums.append(f"{hashlib.sha256(encoded).hexdigest()}  fixture-provenance.json\n")
            file = Path(temp) / ".polar-sha256"
            file.write_text("".join(checksums))
            run("rsync", "-a", str(file), f"{args.host}:{staging}/.polar-sha256")
        remote(args.host, "bash", "-lc", f"cd {shlex.quote(staging)} && sha256sum -c .polar-sha256 >/dev/null")
        remote(args.host, "touch", f"{staging}/.polar-ready")
        remote(args.host, "chmod", "-R", "a-w", staging)
        remote(args.host, "mv", "-T", staging, target)
    print(json.dumps({"fixture_path": target, "identity": identity, "sets": names, "files": sum(len(s["files"]) for s in selected.values())}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError) as exc:
        print(f"polar stage_fixture_sets: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
