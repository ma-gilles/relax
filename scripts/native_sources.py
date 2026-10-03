#!/usr/bin/env python
"""Tie built native libraries to the source tree they were built from.

The native libraries a run loads (relax's EM CUDA library, the RELION binding and recovar's
custom CUDA library) are built from ``relax/cuda``, ``relax/relion_bind``, the build script and
the installed recovar commit. A run that loads libraries built from other sources is not
native-exact for its commit, so every tool that freezes a candidate records this digest at
build time and checks it before running:

    python scripts/native_sources.py record <natives_dir> [--root <checkout>]
    python scripts/native_sources.py check <natives_dir> [--root <checkout>]
    python scripts/native_sources.py imports <snapshot>

``imports`` checks the other half of a run's provenance: a snapshot that shares a pixi
environment with an editable relax install must still import relax from the snapshot (set
PYTHONPATH to it), not from the live worktree the install points to, or a rebase during a
queued job changes the code under test. It prints relax.__file__ and recovar.__file__ as the
run's processes resolve them and exits 1 when relax comes from outside the snapshot.

``record`` adds ``native_sources`` to ``<natives_dir>/NATIVE.json``; ``check`` exits 1 with a
message naming both digests when the libraries were built from different sources, or when the
record is missing.

``record`` also writes ``recovar_kernels``: the path and digest of the recovar CUDA sources the
build actually compiled, resolved as the build resolved them (a ``python`` started in the
current working directory, with the current environment), so run ``record`` where the build ran.
``python -m`` puts the working directory first on sys.path: a build started inside a recovar
checkout compiled that checkout's kernels while the installed recovar commit said otherwise
(natives_bab1082, 2026-10-03; the 2026-09-25 shadowing trap). ``check`` refuses natives whose
compiled kernel sources differ from those of the recovar a run imports, and natives without the
record.
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
# recovar's CUDA kernel sources (recovar/cuda: the pipeline library and the public headers relax includes).
RECOVAR_KERNEL_SUFFIXES = {".cu", ".cuh", ".h", ".inc"}


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


def _resolved_recovar_dir(cwd: Path, env: dict | None = None) -> Path:
    """The recovar package directory a ``python`` started in ``cwd`` imports (as ``python -m`` resolves it)."""
    import os
    import subprocess

    code = "import os, recovar; print(os.path.dirname(os.path.realpath(recovar.__file__)))"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=dict(os.environ if env is None else env, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu"),
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(proc.stdout.strip().splitlines()[-1])


def recovar_kernels(recovar_dir: Path) -> dict:
    """Path and digest of the CUDA kernel sources under ``recovar_dir/cuda``."""
    cuda = Path(recovar_dir).resolve() / "cuda"
    files = sorted(
        p for p in cuda.rglob("*") if p.is_file() and (p.suffix in RECOVAR_KERNEL_SUFFIXES or p.name == "Makefile")
    )
    if not files:
        raise RuntimeError(f"no recovar CUDA sources under {cuda}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(cuda).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return {"path": str(cuda), "sha256": digest.hexdigest(), "files": len(files)}


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
    compiled = json.loads(record_path.read_text()).get("recovar_kernels")
    if not compiled:
        return (
            f"{record_path} does not record which recovar kernel sources were compiled (built before the "
            "guard); rebuild the natives with scripts/build_test_natives.sh"
        )
    pinned = recovar_kernels(_resolved_recovar_dir(Path("/")))
    if compiled["sha256"] != pinned["sha256"]:
        return (
            f"natives {natives} compiled the recovar kernels at {compiled['path']} ({compiled['sha256'][:12]}), "
            f"but this run imports recovar with kernels {pinned['path']} ({pinned['sha256'][:12]}); a build "
            "started inside a recovar checkout compiles that checkout. Rebuild with scripts/build_test_natives.sh"
        )
    return None


def import_provenance(snapshot: Path, env: dict | None = None) -> dict:
    """relax.__file__ and recovar.__file__ as a fresh process with ``env`` resolves them.

    The process starts outside the snapshot so that only PYTHONPATH (not the working
    directory) can put the snapshot first, as for any tool that runs from elsewhere.
    """
    import os
    import subprocess

    code = "import json, relax, recovar; print(json.dumps({'relax': relax.__file__, 'recovar': recovar.__file__}))"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=dict(os.environ if env is None else env, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu"),
        cwd="/",
        capture_output=True,
        text=True,
    )
    if proc.returncode:
        return {"error": proc.stderr[-2000:]}
    files = json.loads(proc.stdout.strip().splitlines()[-1])
    files["relax_from_snapshot"] = Path(files["relax"]).resolve().is_relative_to(Path(snapshot).resolve())
    return files


def check_imports(snapshot: Path, env: dict | None = None) -> tuple[dict, str | None]:
    files = import_provenance(snapshot, env)
    if "error" in files:
        return files, f"cannot import relax and recovar: {files['error']}"
    if not files["relax_from_snapshot"]:
        return files, (
            f"relax is imported from {files['relax']}, not from the snapshot {snapshot}: an editable install "
            f"points at a live worktree; run with PYTHONPATH={snapshot}"
        )
    return files, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["record", "check", "imports"])
    parser.add_argument("natives", type=Path, help="the natives directory (the snapshot for imports)")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    if args.mode == "record":
        path = args.natives / "NATIVE.json"
        record = json.loads(path.read_text()) if path.is_file() else {}
        record["native_sources"] = native_sources(args.root)
        record["recovar_kernels"] = recovar_kernels(_resolved_recovar_dir(Path.cwd()))
        path.write_text(json.dumps(record, indent=1) + "\n")
        print(f"native sources {record['native_sources']['sha256'][:12]} recorded in {path}")
        return 0
    if args.mode == "imports":
        files, reason = check_imports(args.natives)
        print(json.dumps(files))
        if reason:
            print(reason, file=sys.stderr)
        return 1 if reason else 0
    reason = check(args.natives, args.root)
    if reason:
        print(reason, file=sys.stderr)
        return 1
    print(f"natives {args.natives} match the native sources of {args.root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
