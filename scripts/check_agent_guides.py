"""Check the agent guides and the local file links of every tracked Markdown file.

``AGENTS.md`` is the one guide of a directory; ``CLAUDE.md`` beside it is a symbolic link to it, so both agent
tools read the same file. Findings come in four classes, each printed with its file and line:

- ``missing link target``: a relative link whose target does not exist in the checkout.
- ``absolute-path link``: a link to an absolute filesystem path. It names one site's (often one
  account's) directory, so it cannot be checked from a checkout; cite the path as plain text.
- ``unreadable link target``: the current account may not inspect the target (the path leaves the
  checkout through a symlink or ``..`` into a directory it cannot enter).
- guide findings: a directory's ``CLAUDE.md`` is not the relative symbolic link ``AGENTS.md``, an
  ``AGENTS.md`` has no such link beside it, or a required guide is missing or empty.

The exit status is 1 when there is any finding.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

GUIDE, GUIDE_LINK = "AGENTS.md", "CLAUDE.md"
REQUIRED_GUIDES = ("AGENTS.md", "CONTRIBUTING.md", "relax/AGENTS.md", "tests/AGENTS.md")
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


def is_guide_link(path: Path) -> bool:
    """Whether ``path`` is a ``CLAUDE.md`` that is the relative symbolic link ``AGENTS.md``."""
    return path.name == GUIDE_LINK and path.is_symlink() and os.readlink(path) == GUIDE


def check_guide_links(root: Path, documents: list[Path]) -> list[str]:
    """One finding per directory whose ``CLAUDE.md`` is not a symbolic link to the ``AGENTS.md`` beside it."""
    errors = []
    for directory in sorted({document.parent for document in documents if document.name in (GUIDE, GUIDE_LINK)}):
        guide, link = directory / GUIDE, directory / GUIDE_LINK
        name = link.relative_to(root)
        if guide.is_symlink() or not guide.is_file():
            errors.append(f"guide link without its guide: {name} needs a regular file {GUIDE} beside it")
        elif not link.is_symlink() and not link.exists():
            errors.append(f"missing guide link: {name} (ln -s {GUIDE} {GUIDE_LINK})")
        elif not is_guide_link(link):
            errors.append(f"guide link is not the symbolic link {GUIDE}: {name}")
    return errors


def check_guides(root: Path) -> list[str]:
    tracked = tracked_markdown(root)
    errors = check_guide_links(root, tracked)

    guides = {root / name for name in REQUIRED_GUIDES}
    guides.update((root / "docs/development").glob("*.md"))
    for guide in sorted(guides):
        if not guide.is_file():
            errors.append(f"missing guide: {guide.relative_to(root)}")
        elif not guide.read_text().strip():
            errors.append(f"empty guide: {guide.relative_to(root)}")

    # A guide link has the links of its guide, which is checked under its own name.
    documents = sorted({*tracked, *(guide for guide in guides if guide.is_file())})
    return errors + check_links(root, [document for document in documents if not is_guide_link(document)])


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    errors = check_guides(root)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        classes = ("missing link target", "absolute-path link", "unreadable link target")
        counts = {name: sum(f": {name}" in error for error in errors) for name in classes}
        counts["guide"] = len(errors) - sum(counts.values())
        print("; ".join(f"{count} {name}" for name, count in counts.items() if count), file=sys.stderr)
        return 1
    documents = [document for document in tracked_markdown(root) if not is_guide_link(document)]
    print(f"The agent guides and the file links of {len(documents)} Markdown files are valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
