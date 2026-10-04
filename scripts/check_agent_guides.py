"""Check mirrored agent contracts and the local file links of every tracked Markdown file.

Findings come in four classes, each printed with its file and line:

- ``missing link target``: a relative link whose target does not exist in the checkout.
- ``absolute-path link``: a link to an absolute filesystem path. It names one site's (often one
  account's) directory, so it cannot be checked from a checkout; cite the path as plain text.
- ``unreadable link target``: the current account may not inspect the target (the path leaves the
  checkout through a symlink or ``..`` into a directory it cannot enter).
- mirror and guide findings: a mirrored pair differs, or a required guide is missing or empty.

The exit status is 1 when there is any finding.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

MIRRORS = (
    ("AGENTS.md", "CLAUDE.md"),
    ("relax/AGENTS.md", "relax/CLAUDE.md"),
    ("relax/ppca_refinement/AGENTS.md", "relax/ppca_refinement/CLAUDE.md"),
    ("tests/AGENTS.md", "tests/CLAUDE.md"),
)
REQUIRED_GUIDES = ("CONTRIBUTING.md", "tests/CLAUDE.md")
# Directories that hold no tracked documents; used only when the checkout has no git metadata.
UNTRACKED_PARTS = {".git", ".pixi", ".tmp", "site", "build", "dist", "node_modules", "__pycache__"}

# A code span, a math span, or an inline link whose target is captured. A link label may wrap
# over lines but not over a blank line.
_SPAN_OR_LINK = re.compile(
    r"`+[^`\n]*`+|\$\$.*?\$\$|\$[^$\s](?:[^$\n]*[^$\s])?\$"
    r"|\[(?:[^\]\n]|\n(?![ \t]*\n))*\]\((?P<target>[^\s)]+)\)",
    re.DOTALL,
)
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")


def tracked_markdown(root: Path) -> list[Path]:
    """Every Markdown file git tracks under ``root``; every Markdown file there when it is not a checkout."""
    try:
        listed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--", "*.md"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        listed = ""
    if listed:
        return sorted(root / name for name in listed.split("\0") if name)
    return sorted(p for p in root.rglob("*.md") if not UNTRACKED_PARTS & set(p.relative_to(root).parts))


def without_fenced_blocks(content: str) -> str:
    """``content`` with the lines of fenced code blocks blanked, so every offset stays where it was."""
    lines = []
    fence = None
    for line in content.split("\n"):
        marker = _FENCE.match(line)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence) and not line[marker.end() :].strip():
                fence = None
            lines.append(" " * len(line))
        else:
            lines.append(" " * len(line) if fence else line)
    return "\n".join(lines)


def link_matches(content: str):
    """Yield the regex match of each inline link outside fenced blocks, code spans and math spans."""
    for match in _SPAN_OR_LINK.finditer(without_fenced_blocks(content)):
        if match.group("target") is not None:
            yield match


def link_targets(content: str):
    """Yield ``(line_number, target)`` for each inline link; the line is where the target is written."""
    for match in link_matches(content):
        yield content.count("\n", 0, match.start("target")) + 1, match.group("target")


def classify_link(document: Path, target: str) -> str | None:
    """The finding class of one link target in ``document``, or None when it is valid or not a local file link."""
    link = urlsplit(target)
    if link.scheme or link.netloc or not link.path:
        return None
    path = unquote(link.path)
    if path.startswith("/"):
        return "absolute-path link"
    try:
        (document.parent / path).stat()
    except (FileNotFoundError, NotADirectoryError):
        return "missing link target"
    except OSError as error:
        return f"unreadable link target ({error.strerror or type(error).__name__})"
    return None


def check_links(root: Path, documents: list[Path]) -> list[str]:
    errors = []
    for document in documents:
        name = document.relative_to(root)
        try:
            content = document.read_text()
        except OSError as error:
            errors.append(f"{name}: unreadable document ({error.strerror or type(error).__name__})")
            continue
        for line_number, target in link_targets(content):
            finding = classify_link(document, target)
            if finding:
                errors.append(f"{name}:{line_number}: {finding} {target}")
    return errors


def check_guides(root: Path) -> list[str]:
    errors = []
    for first, second in MIRRORS:
        paths = root / first, root / second
        if not all(path.is_file() for path in paths):
            errors.append(f"missing required mirror: {first} or {second}")
        elif paths[0].read_bytes() != paths[1].read_bytes():
            errors.append(f"agent contracts differ: {first} and {second}")

    guides = {root / name for pair in MIRRORS for name in pair}
    guides.update(root / name for name in REQUIRED_GUIDES)
    guides.update((root / "docs/development").glob("*.md"))
    for guide in sorted(guides):
        if not guide.is_file():
            errors.append(f"missing guide: {guide.relative_to(root)}")
        elif not guide.read_text().strip():
            errors.append(f"empty guide: {guide.relative_to(root)}")

    documents = sorted({*tracked_markdown(root), *(guide for guide in guides if guide.is_file())})
    return errors + check_links(root, documents)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    errors = check_guides(root)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        classes = ("missing link target", "absolute-path link", "unreadable link target")
        counts = {name: sum(f": {name}" in error for error in errors) for name in classes}
        counts["mirror or guide"] = len(errors) - sum(counts.values())
        print("; ".join(f"{count} {name}" for name, count in counts.items() if count), file=sys.stderr)
        return 1
    print(f"Agent mirrors and the file links of {len(tracked_markdown(root))} Markdown files are valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
