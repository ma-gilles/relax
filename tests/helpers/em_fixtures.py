"""External EM fixtures the test tiers read, resolved through tests/fixtures/em_fixture_manifest.json.

Module-level path constants use ``fixture_root(name)``, which reads the manifest only. Each
test then calls ``require_fixture_sets(*names)`` (or ``fixture_dir``/``fixture_file``) for the
sets it reads, or ``fixture_file(name, rel)`` for single files. The first call in a process
verifies the files (existence, size and sha256) and raises ``FixtureError`` on any difference,
so a missing or changed fixture fails the test instead of skipping it. ``scripts/build_em_fixture_manifest.py`` writes the manifest.

Launchers verify the sets of a whole tier before submitting work:

    python tests/helpers/em_fixtures.py k1_5k128_data k1_5k128_relion_os0
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[1] / "fixtures" / "em_fixture_manifest.json"


class FixtureError(RuntimeError):
    """A manifest fixture is missing, incomplete or differs from its recorded hash."""


@lru_cache(maxsize=1)
def _manifest() -> dict:
    return json.loads(MANIFEST.read_text())["sets"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fixture_root(name: str) -> Path:
    """Return the manifest root of fixture set ``name`` without reading the data."""
    if name not in _manifest():
        raise FixtureError(f"fixture set {name!r} is not in {MANIFEST}")
    return Path(_manifest()[name]["root"])


def require_fixture_sets(*names: str) -> None:
    """Verify each named set; raises ``FixtureError`` on the first missing or changed one."""
    for name in names:
        fixture_dir(name)


@lru_cache(maxsize=None)
def _verified(name: str, rel: str) -> Path:
    """Check one manifest file of set ``name``: present, same size, same sha256."""
    entry = _manifest().get(name)
    if entry is None:
        raise FixtureError(f"fixture set {name!r} is not in {MANIFEST}")
    if rel not in entry["files"]:
        raise FixtureError(f"{rel!r} is not a manifest file of fixture set {name!r}")
    size, digest = entry["files"][rel]
    path = Path(entry["root"]) / rel
    if not path.is_file():
        raise FixtureError(f"fixture set {name!r}: missing {path}")
    if path.stat().st_size != size:
        raise FixtureError(f"fixture set {name!r}: {path} has size {path.stat().st_size}, manifest {size}")
    if _sha256(path) != digest:
        raise FixtureError(f"fixture set {name!r}: sha256 of {path} differs from the manifest")
    return path


@lru_cache(maxsize=None)
def fixture_dir(name: str) -> Path:
    """Return the root of fixture set ``name`` after verifying every file in it."""
    root = fixture_root(name)
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda rel: _verified(name, rel), _manifest()[name]["files"]))
    return root


def fixture_file(name: str, rel: str) -> Path:
    """Return one verified file of fixture set ``name`` (only that file is hashed)."""
    return _verified(name, rel)


def main(argv: list[str]) -> int:
    names = argv or sorted(_manifest())
    failed = 0
    for name in names:
        try:
            root = fixture_dir(name)
        except FixtureError as exc:
            failed += 1
            print(f"FAIL {exc}", flush=True)
        else:
            print(f"ok   {name} ({len(_manifest()[name]['files'])} files) {root}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
