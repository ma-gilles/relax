#!/usr/bin/env python
"""Check that every fingerprint selftest mutation (refinement, VDAM and pass 1) still finds its anchor text in the source.

    python scripts/dev/check_mutation_anchors.py         # exit 1 when a mutation is stale

A mutation of ``scripts/dev/fingerprint.py`` replaces a piece of controller text. A refactor that rewrites
that text leaves the mutation with nothing to replace, and the selftest would then fail only after minutes
of runs. This applies each mutation to the tracked ``relax/*.py`` files, without running anything, and
reports the ones whose number of replacements is not the expected one. Run it after every edit of a
fingerprinted module (``docs/development/refactor_procedure.md``).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.dev import fingerprint, pass1_fingerprint, vdam_fingerprint  # noqa: E402

MUTATIONS = (*fingerprint.MUTATIONS, *vdam_fingerprint.MUTATIONS, *pass1_fingerprint.MUTATIONS)


def stale_mutations(root: Path = REPO_ROOT) -> list[tuple[str, int, int]]:
    """``(name, replacements found, replacements expected)`` for each mutation that no longer applies."""
    listed = subprocess.run(
        ["git", "ls-files", "relax/*.py"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.split()
    sources = [(root / name).read_text() for name in listed]
    stale = []
    for mutation in MUTATIONS:
        name, old, new = mutation[:3]
        expected = mutation[5] if len(mutation) > 5 else 1
        first_line = old.strip().splitlines()[0].strip()
        found = sum(fingerprint.apply_mutation(text, old, new)[1] for text in sources if first_line in text)
        if found != expected:
            stale.append((name, found, expected))
    return stale


def main() -> int:
    stale = stale_mutations()
    for name, found, expected in stale:
        print(f"STALE {name}: {found} replacements, expected {expected}")
    print(f"mutation anchors checked: {len(MUTATIONS)} stale: {len(stale)}")
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
