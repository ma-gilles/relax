"""The Polar submitter's source snapshot (scripts/polar/submit.py snapshot_paths)."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]


def _submit():
    spec = importlib.util.spec_from_file_location("polar_submit", REPO_ROOT / "scripts/polar/submit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checkout(tmp_path):
    checkout = tmp_path / "checkout"
    (checkout / "docs").mkdir(parents=True)
    (checkout / "AGENTS.md").write_text("guide\n")
    (checkout / "docs" / "AGENTS.md").write_text("docs guide\n")
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    return checkout


def test_a_link_to_a_file_inside_the_checkout_is_shipped_as_that_file(tmp_path):
    submit = _submit()
    checkout = _checkout(tmp_path)
    (checkout / "CLAUDE.md").symlink_to("AGENTS.md")
    (checkout / "docs" / "CLAUDE.md").symlink_to("AGENTS.md")
    paths = submit.snapshot_paths(checkout)
    assert [path.as_posix() for path in paths] == ["AGENTS.md", "CLAUDE.md", "docs/AGENTS.md", "docs/CLAUDE.md"]
    # The manifest hashes what the link resolves to, which is what rsync --copy-links ships.
    assert submit.digest(checkout / "CLAUDE.md") == submit.digest(checkout / "AGENTS.md")
    assert submit.digest(checkout / "docs" / "CLAUDE.md") == submit.digest(checkout / "docs" / "AGENTS.md")


@pytest.mark.parametrize("target", ["outside", "missing"])
def test_a_link_that_leaves_the_checkout_or_has_no_target_is_refused(tmp_path, target):
    submit = _submit()
    checkout = _checkout(tmp_path)
    (tmp_path / "outside.md").write_text("elsewhere\n")
    (checkout / "LINK.md").symlink_to("../outside.md" if target == "outside" else "nothing.md")
    with pytest.raises(RuntimeError, match="source symlink LINK.md points outside the checkout or at no file"):
        submit.snapshot_paths(checkout)
