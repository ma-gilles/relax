"""The document checker: link classes, what it scans, and that this checkout passes it."""

from pathlib import Path

import pytest

from scripts import check_agent_guides

REPO_ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.unit


def _findings(tmp_path, text):
    document = tmp_path / "docs" / "note.md"
    document.parent.mkdir(exist_ok=True)
    document.write_text(text)
    return check_agent_guides.check_links(tmp_path, [document])


def test_each_link_class_is_reported_with_file_and_line(tmp_path):
    (tmp_path / "present.py").write_text("")
    findings = _findings(
        tmp_path,
        "[ok](../present.py#L3) [web](https://example.org/x) [anchor](#top)\n"
        "[gone](../absent.py)\n"
        "[site](/scratch/some/account/HANDOFF.json)\n",
    )
    assert findings == [
        "docs/note.md:2: missing link target ../absent.py",
        "docs/note.md:3: absolute-path link /scratch/some/account/HANDOFF.json",
    ]


def test_absolute_link_is_never_statted(tmp_path, monkeypatch):
    def refuse(self, **kwargs):
        raise AssertionError(f"stat({self})")

    monkeypatch.setattr(Path, "stat", refuse)
    document = tmp_path / "note.md"
    assert check_agent_guides.classify_link(document, "/home/other/private.md") == "absolute-path link"


def test_target_the_account_may_not_inspect_is_a_finding_not_a_crash(tmp_path, monkeypatch):
    original = Path.stat

    def stat(self, **kwargs):
        if self.name == "private.md":
            raise PermissionError(13, "Permission denied")
        return original(self, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    assert _findings(tmp_path, "[handoff](../linked/private.md)\n") == [
        "docs/note.md:1: unreadable link target (Permission denied) ../linked/private.md"
    ]


def test_code_math_and_wrapped_labels(tmp_path):
    findings = _findings(
        tmp_path,
        "```\n[fenced](absent_a.md)\n```\n"
        "`[code](absent_b.md)` and $$\\big[w\\big](t)$$\n"
        "see the [wrapped\nlabel](absent_c.md)\n",
    )
    assert findings == ["docs/note.md:6: missing link target absent_c.md"]


def test_every_markdown_file_is_scanned_without_git_metadata(tmp_path):
    for name in ("README.md", "docs/math/a.md", ".pixi/envs/default/x.md", "site/y.md"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("")
    names = [p.relative_to(tmp_path).as_posix() for p in check_agent_guides.tracked_markdown(tmp_path)]
    assert names == ["README.md", "docs/math/a.md"]


def test_this_checkout_has_no_finding():
    documents = check_agent_guides.tracked_markdown(REPO_ROOT)
    assert any(p.relative_to(REPO_ROOT).parts[:2] == ("docs", "math") for p in documents)
    assert check_agent_guides.check_guides(REPO_ROOT) == []
