#!/usr/bin/env python3
"""Relocate staged EM STAR references for one Polar replay run.

The raw fixture snapshot is unchanged. Only text files containing a Della
absolute path get new bytes; their derived hashes are written to a run-local
manifest and relocation ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("staged", type=Path)
    parser.add_argument("source_copy", type=Path)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()

    staged = args.staged.resolve(strict=True)
    source_copy = args.source_copy.resolve(strict=True)
    run_root = args.run_root.resolve(strict=True)
    record = json.loads((staged / "fixture-provenance.json").read_text())
    manifest_file = source_copy / "tests/fixtures/em_fixture_manifest.json"
    if sha256(manifest_file) != record["source_manifest_sha256"]:
        raise RuntimeError("source and staged fixture manifests differ")
    manifest = json.loads(manifest_file.read_text())
    roots = {name: run_root / "fixtures" / name for name in record["sets"]}
    replacements = sorted(
        ((entry["root"], str(roots[name])) for name, entry in record["sets"].items()),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )
    changed = {}
    for name, entry in record["sets"].items():
        destination = roots[name]
        destination.mkdir(parents=True, exist_ok=False)
        derived_files = {}
        for rel, (size, original_sha) in entry["files"].items():
            raw = staged / "raw" / name / rel
            if not raw.is_file() or raw.stat().st_size != size or sha256(raw) != original_sha:
                raise RuntimeError(f"staged fixture no longer matches manifest: {raw}")
            path = destination / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            content = None
            if path.suffix.lower() in {".star", ".txt", ".json"}:
                original = raw.read_text()
                relocated = original
                for old, new in replacements:
                    relocated = relocated.replace(old, new)
                if relocated != original:
                    content = relocated.encode()
            if content is None:
                path.symlink_to(raw)
                derived_files[rel] = [size, original_sha]
            else:
                path.write_bytes(content)
                derived_sha = hashlib.sha256(content).hexdigest()
                derived_files[rel] = [len(content), derived_sha]
                changed[f"{name}/{rel}"] = {"raw_sha256": original_sha, "relocated_sha256": derived_sha}
        manifest["sets"][name]["root"] = str(destination)
        manifest["sets"][name]["files"] = derived_files

    manifest_file.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    ledger = {
        "source_manifest_sha256": record["source_manifest_sha256"],
        "staged_fixture_root": str(staged),
        "relocated_roots": {name: str(path) for name, path in roots.items()},
        "changed_text_files": changed,
        "run_manifest_sha256": sha256(manifest_file),
    }
    (run_root / "fixture-relocation.json").write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(f"Relocated {len(changed)} text files; manifest {ledger['run_manifest_sha256']}", flush=True)


if __name__ == "__main__":
    main()
