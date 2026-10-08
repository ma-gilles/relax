"""Worktree provenance gate for RELION-parity tests and benchmarks.

A directory name is not proof of which branch is checked out. A worktree
named ``recovar_wt_parity_*`` may have been switched to an unrelated branch
by a previous session. Mixing branches has cost days of debugging on this
project; this module provides a hard gate that any parity-sensitive entry
point can call to fail fast.

The gate prints HEAD/branch/dirty state and asserts that every commit listed
in :data:`REQUIRED_PARITY_ANCESTORS` is in HEAD's ancestry. When any required
commit is missing, ``assert_parity_ancestors_or_exit`` exits with status 2
so the caller can distinguish "missing kernel-parity fix" from "real
regression"; ``assert_parity_ancestors`` raises ``ParityAncestryError`` for
test contexts that need an exception.
"""

from __future__ import annotations

import subprocess
import sys

from relax.helpers.git_provenance import git_head_or_none, git_worktree_provenance

# Load-bearing parity fix commits. If any of these is NOT an ancestor of
# HEAD, the worktree is missing a known-required parity fix and replay
# results will not be machine-precision against RELION. Update only when
# adding new fixes.
#
# relax's history starts at its import commit, which imported RECOVAR
# 0e77fe5b0. That RECOVAR commit contains the five fixes this table named
# while the EM code lived in RECOVAR: 7834dc0b (current_size off-by-one +
# circular Fourier window), 5f21574a (float64 scoring + current_size replay
# from model.star), 0650b550 (image pre-centering, normcorr, prior sign,
# float64 logsumexp), b125883f (shell-mapping at pf=2) and 0903a64c (pf^3
# tau2 correction + join radius scaling). Those SHAs are not in relax's history.
REQUIRED_PARITY_ANCESTORS: tuple[tuple[str, str], ...] = (
    ("49b1773d", "relax import of RECOVAR 0e77fe5b0 (carries the RECOVAR parity fixes listed above)"),
)


class ParityAncestryError(RuntimeError):
    """Raised when a required parity-fix commit is not in HEAD's ancestry."""


def _git_ancestor_of_head(sha: str) -> bool:
    try:
        subprocess.check_call(
            ["git", "merge-base", "--is-ancestor", sha, "HEAD"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def _missing_parity_ancestors() -> list[tuple[str, str]]:
    """Return the (sha, description) pairs for required commits not in HEAD's ancestry."""
    return [(sha, desc) for sha, desc in REQUIRED_PARITY_ANCESTORS if not _git_ancestor_of_head(sha)]


def print_provenance_banner(stream=sys.stdout) -> dict[str, str | int]:
    """Print HEAD/branch/dirty state. Returns the same fields as a dict."""
    info = git_worktree_provenance()
    cwd = info["cwd"]
    branch = info["branch"]
    head = info["head"]
    print("=" * 72, flush=True, file=stream)
    print("Parity replay provenance:", flush=True, file=stream)
    print(f"  cwd:    {cwd}", flush=True, file=stream)
    print(f"  branch: {branch}", flush=True, file=stream)
    print(f"  HEAD:   {head}", flush=True, file=stream)
    print(f"  dirty:  {info['dirty_count']} uncommitted file(s)", flush=True, file=stream)
    print(f"  diff:   {info['diff_sha256']}", flush=True, file=stream)
    print(f"  tree:   {info['worktree_fingerprint_sha256']}", flush=True, file=stream)
    return info


def assert_parity_ancestors() -> None:
    """Raise ParityAncestryError if any required parity-fix commit is missing."""
    missing = _missing_parity_ancestors()
    if missing:
        head = git_head_or_none() or "<unknown>"
        details = "\n".join(f"  - {sha} ({desc}) NOT in ancestry of {head[:8]}" for sha, desc in missing)
        raise ParityAncestryError(
            "Worktree is missing required parity-fix commits in HEAD's ancestry:\n"
            + details
            + "\nReplay parity results from this branch are not trustworthy. Switch to a "
            "branch that contains these commits before running parity tests."
        )


def assert_parity_ancestors_or_exit(exit_status: int = 2) -> None:
    """Print banner, then exit with ``exit_status`` if any required commit is missing.

    Used by CLI entry points (scripts/run_multi_iter_parity.py and similar)
    where a non-zero exit code lets the caller distinguish "wrong branch"
    from "real regression".
    """
    print_provenance_banner()
    missing = _missing_parity_ancestors()
    if missing:
        head = git_head_or_none() or "<unknown>"
        print("=" * 72, flush=True)
        print(
            "ERROR: this worktree is missing required parity-fix commits in HEAD's ancestry:",
            flush=True,
        )
        for sha, desc in missing:
            print(f"  - {sha} ({desc}) NOT in ancestry of {head[:8]}", flush=True)
        print(
            "Replay parity results from this branch are not trustworthy. Switch to a "
            "branch that contains these commits before running parity tests.",
            flush=True,
        )
        sys.exit(exit_status)
    print(
        f"  parity-fix ancestors: all {len(REQUIRED_PARITY_ANCESTORS)} present ✓",
        flush=True,
    )
    print("=" * 72, flush=True)
