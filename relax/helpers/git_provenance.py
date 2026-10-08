"""Git provenance of the running code: HEAD and a fingerprint of the exact worktree content.

The refinement archive and the benchmark ledger record it; ``relax.parity.parity_provenance`` builds the parity
ancestry gate on it.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def git_head_or_none() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip() or None
        )
    except Exception:
        return None


def _safe_git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "symbolic-ref", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "<detached>"


def _safe_git_status_porcelain() -> str:
    try:
        return subprocess.check_output(
            ["git", "status", "--porcelain=v1"], text=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        return ""


def _safe_git_diff_binary() -> bytes:
    try:
        return subprocess.check_output(
            ["git", "diff", "--binary", "--no-ext-diff", "HEAD", "--"], stderr=subprocess.DEVNULL
        )
    except Exception:
        return b""


def _safe_git_untracked_files() -> list[str]:
    try:
        raw = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    return [part.decode("utf-8", errors="surrogateescape") for part in raw.split(b"\0") if part]


def git_worktree_provenance(*, max_untracked_file_bytes: int = 64 * 1024 * 1024) -> dict[str, str | int | list[str]]:
    """Return a compact fingerprint for the exact dirty worktree content.

    HEAD plus ``git status`` is not enough for long-running parity matrices:
    two queued Slurm jobs can report the same commit and dirty file list while
    running different uncommitted patches. The fingerprint includes the binary
    tracked diff, porcelain status, and hashes of reasonably sized untracked
    files. Very large untracked files are represented by metadata so scratch
    artifacts do not dominate provenance collection.
    """

    status = _safe_git_status_porcelain()
    diff = _safe_git_diff_binary()
    untracked = _safe_git_untracked_files()
    hasher = hashlib.sha256()
    hasher.update(b"recovar-git-worktree-provenance-v1\0")
    hasher.update(status.encode("utf-8", errors="surrogateescape"))
    hasher.update(b"\0tracked-diff\0")
    hasher.update(diff)

    untracked_hashes: list[str] = []
    for relpath in sorted(untracked):
        path = Path(relpath)
        hasher.update(b"\0untracked-path\0")
        hasher.update(relpath.encode("utf-8", errors="surrogateescape"))
        try:
            stat = path.stat()
        except OSError:
            hasher.update(b"\0missing\0")
            untracked_hashes.append(f"{relpath}\tmissing")
            continue
        if not path.is_file() or stat.st_size > max_untracked_file_bytes:
            metadata = f"type={'file' if path.is_file() else 'nonfile'} size={stat.st_size} mtime_ns={stat.st_mtime_ns}"
            hasher.update(metadata.encode("utf-8"))
            untracked_hashes.append(f"{relpath}\t{metadata}")
            continue
        file_hash = hashlib.sha256()
        try:
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    file_hash.update(chunk)
        except OSError:
            hasher.update(b"\0unreadable\0")
            untracked_hashes.append(f"{relpath}\tunreadable")
            continue
        digest = file_hash.hexdigest()
        hasher.update(digest.encode("ascii"))
        untracked_hashes.append(f"{relpath}\t{digest}")

    diff_sha256 = hashlib.sha256(diff).hexdigest()
    return {
        "cwd": str(Path.cwd().resolve()),
        "branch": _safe_git_branch(),
        "head": git_head_or_none() or "<unknown>",
        "dirty_count": len(status.splitlines()),
        "status_porcelain": status,
        "diff_sha256": diff_sha256,
        "worktree_fingerprint_sha256": hasher.hexdigest(),
        "untracked_count": len(untracked),
        "untracked_file_hashes": untracked_hashes,
        "fingerprint_version": "recovar-git-worktree-provenance-v1",
    }
