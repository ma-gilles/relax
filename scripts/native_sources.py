#!/usr/bin/env python
"""Tie built native libraries to the source tree they were built from.

The native libraries a run loads (relax's EM CUDA library, the RELION binding and recovar's
custom CUDA library) are built from ``relax/cuda``, ``relax/relion_bind``, the build script and
the installed recovar commit. A run that loads libraries built from other sources is not
native-exact for its commit, so every tool that freezes a candidate records this digest at
build time and checks it before running:

    python scripts/native_sources.py record <natives_dir> [--root <checkout>]
    python scripts/native_sources.py check <natives_dir> [--root <checkout>]

``record`` adds ``native_sources`` to ``<natives_dir>/NATIVE.json``; ``check`` exits 1 with a
message naming both digests when the libraries were built from different sources, or when the
record is missing.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_SOURCE_DIRS = ("relax/cuda", "relax/relion_bind")
NATIVE_SOURCE_FILES = ("scripts/build_test_natives.sh",)
SKIP_PARTS = {"__pycache__", "build"}
SKIP_SUFFIXES = {".pyc", ".so", ".o", ".log"}


def _files(root: Path) -> list[Path]:
    files = [root / f for f in NATIVE_SOURCE_FILES]
    for d in NATIVE_SOURCE_DIRS:
        files += [
            p
            for p in (root / d).rglob("*")
            if p.is_file() and not SKIP_PARTS & set(p.relative_to(root).parts) and p.suffix not in SKIP_SUFFIXES
        ]
    return sorted(files)


def _recovar_commit() -> str:
    try:
        url = json.loads(importlib.metadata.distribution("recovar").read_text("direct_url.json") or "{}")
    except importlib.metadata.PackageNotFoundError:
        return "not installed"
    return url.get("vcs_info", {}).get("commit_id", "unknown")


def native_sources(root: Path = REPO_ROOT) -> dict:
    """Digest of the native source tree under ``root`` plus the installed recovar commit."""
    digest = hashlib.sha256()
    files = _files(root)
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    recovar = _recovar_commit()
    digest.update(b"recovar\0" + recovar.encode())
    return {"sha256": digest.hexdigest(), "files": len(files), "recovar_commit": recovar}


def check(natives: Path, root: Path = REPO_ROOT) -> str | None:
    """None when ``natives`` was built from ``root``'s native sources, else the reason."""
    record_path = natives / "NATIVE.json"
    if not record_path.is_file():
        return f"{record_path} is missing: the natives carry no record of their sources"
    built = json.loads(record_path.read_text()).get("native_sources")
    if not built:
        return f"{record_path} has no native_sources record (built before the guard); rebuild the natives"
    now = native_sources(root)
    if built["sha256"] != now["sha256"]:
        return (
            f"stale natives: {natives} were built from native sources {built['sha256'][:12]} "
            f"(recovar {built['recovar_commit'][:9]}), but {root} has {now['sha256'][:12]} "
            f"(recovar {now['recovar_commit'][:9]}); rebuild them with scripts/build_test_natives.sh"
        )
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["record", "check"])
    parser.add_argument("natives", type=Path)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    if args.mode == "record":
        path = args.natives / "NATIVE.json"
        record = json.loads(path.read_text()) if path.is_file() else {}
        record["native_sources"] = native_sources(args.root)
        path.write_text(json.dumps(record, indent=1) + "\n")
        print(f"native sources {record['native_sources']['sha256'][:12]} recorded in {path}")
        return 0
    reason = check(args.natives, args.root)
    if reason:
        print(reason, file=sys.stderr)
        return 1
    print(f"natives {args.natives} match the native sources of {args.root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
