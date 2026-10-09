"""The code-smell checker ``scripts/dev/smell_check.py`` on small synthetic repositories: each check fires on a
positive example and stays quiet on a negative one, and the baseline makes an existing finding not new."""

from __future__ import annotations

import json
import textwrap

from scripts.dev import smell_check


def make_repo(root, files):
    (root / "pyproject.toml").write_text('[project.scripts]\nrelax = "relax.command_line:main"\n')
    base = {"relax/__init__.py": "", "relax/command_line.py": "import relax.algo\n", "relax/algo.py": ""}
    for rel, text in {**base, **files}.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text))
    return smell_check.Repo(root)


def keys(findings, check=None):
    return {f.key for f in findings if check is None or f.check == check}


def test_self_cast(tmp_path):
    repo = make_repo(
        tmp_path,
        {
            "relax/algo.py": """
        def f(x, options, y):
            x = int(x)
            size = float(options.size)
            z = int(y)
            other = int(options.size)
            return x, size, z, other
    """
        },
    )
    assert keys(smell_check.check_self_cast(repo)) == {"self-cast:relax.algo:f:x", "self-cast:relax.algo:f:size"}


def test_record_fields_unread_and_test_only(tmp_path):
    repo = make_repo(
        tmp_path,
        {
            "relax/algo.py": """
            from dataclasses import dataclass
            from typing import NamedTuple

            @dataclass(frozen=True)
            class Rec:
                used: int
                tested: int
                unread: int

            class Pair(NamedTuple):
                left: int
                right: int

            def g(r, p):
                return r.used + p.left + getattr(p, "right")
        """,
            "tests/test_algo.py": "def test(r):\n    assert r.tested\n",
        },
    )
    found = keys(smell_check.check_record_fields(repo))
    assert found == {"field-unread:relax.algo:Rec.unread", "field-test-only:relax.algo:Rec.tested"}


def test_unreachable_module_and_dead_defs(tmp_path):
    repo = make_repo(
        tmp_path,
        {
            "relax/algo.py": """
            from relax import helper

            def used():
                return helper.helped()

            def only_tests():
                return 1

            def dead():
                return 2

            def main():
                return used()
        """,
            "relax/helper.py": "def helped():\n    return 0\n",
            "relax/orphan.py": "def lonely():\n    return 0\n",
            "relax/command_line.py": "from relax.algo import main\n",
            "tests/test_algo.py": "from relax.algo import only_tests\n",
        },
    )
    graph = smell_check.Graph(repo)
    assert keys(smell_check.check_unreachable_modules(repo, graph)) == {"unreachable-module:relax.orphan"}
    dead = smell_check.check_dead_defs(repo, graph)
    assert "dead-def:relax.algo:dead" in keys(dead, "dead-def")
    assert "def-outside-only:relax.algo:only_tests" in keys(dead, "def-outside-only")
    assert "dead-def:relax.orphan:lonely" in keys(dead, "dead-def")
    assert not {k for k in keys(dead) if k.endswith((":used", ":main", ":helped"))}


def test_near_duplicates(tmp_path):
    body = "    s = 0\n    for v in xs:\n        s = s + v * 2\n    t = s - 1\n    u = t * t\n    return u + 1\n"
    repo = make_repo(
        tmp_path,
        {
            "relax/algo.py": "def a(xs):\n" + body + "\n\ndef b(ys):\n" + body.replace("xs", "ys"),
            "relax/helper.py": "def c(xs):\n" + body.replace("u + 1", "u + 3").replace("t * t", "t * t * t"),
            "relax/other.py": "def d(xs):\n    return sorted(set(xs))\n",
        },
    )
    found = keys(smell_check.check_near_duplicates(repo))
    assert "near-duplicate:exact:relax.algo:a~relax.algo:b" in found
    assert any(k.startswith("near-duplicate:near:") and "relax.helper:c" in k for k in found)
    assert not any("relax.other:d" in k for k in found)


def test_env_reads_outside_allowlist(tmp_path):
    reads = (
        'import os\n\ndef f():\n    return os.environ["A"], os.environ.get("B"), os.getenv("C"), "D" in os.environ\n'
    )
    repo = make_repo(tmp_path, {"relax/algo.py": reads, "relax/helpers/env_flags.py": reads})
    assert keys(smell_check.check_env_reads(repo)) == {f"env-read:relax.algo:f:{v}" for v in "ABCD"}


def test_diagnostics_imports(tmp_path):
    repo = make_repo(
        tmp_path,
        {
            "relax/algo.py": "from relax.diagnostics import observers\n\ndef f():\n    from .parity import dump\n",
            "relax/command_line.py": "import relax.diagnostics.observers\n",
            "relax/diagnostics/__init__.py": "",
            "relax/diagnostics/observers.py": "from relax.parity import dump\n",
            "relax/parity/__init__.py": "",
            "relax/parity/dump.py": "",
        },
    )
    assert keys(smell_check.check_diagnostics_imports(repo)) == {
        "diagnostics-import:relax.algo:relax.diagnostics",
        "diagnostics-import:relax.algo:relax.parity",
    }


def test_strip_literal(tmp_path):
    repo = make_repo(tmp_path, {"relax/algo.py": "def f(s):\n    return s.rstrip('.mrc'), s.strip('/'), s.strip()\n"})
    assert keys(smell_check.check_strip_literal(repo)) == {"strip-literal:relax.algo:f:rstrip('.mrc')"}


def test_alias_locals(tmp_path):
    repo = make_repo(
        tmp_path,
        {
            "relax/algo.py": """
        def f(options, state, self_like):
            size = options.size
            rate = options.rate
            rate = rate * 2
            state = state.next
            n = self_like.n
            self_like.n = 3
            fn = lambda: options.size
            k = options.k
            return size + rate + n + k + fn() + sum(k for _ in range(2))

        class C:
            def m(self):
                x = self.x
                return x
    """
        },
    )
    assert keys(smell_check.check_alias_locals(repo)) == {"alias-local:relax.algo:f:size=options.size"}


def test_private_imports(tmp_path):
    repo = make_repo(
        tmp_path,
        {
            "relax/helper.py": "def _hidden():\n    return 1\n\n\ndef _seen():\n    return 2\n\n\ndef public():\n    return _hidden()\n",
            "relax/pkg/__init__.py": "",
            "relax/pkg/inner.py": "def _deep():\n    return 3\n",
            "relax/algo.py": """
            import relax.helper as h
            import relax.pkg.inner
            from relax import helper
            from relax.helper import _hidden, public, __doc__
            from relax.pkg import inner as alias
            import numpy as np

            def f():
                return (
                    _hidden() + public() + h._seen() + helper._hidden() + relax.pkg.inner._deep()
                    + alias._deep() + np._private + h.__name__
                )
        """,
        },
    )
    assert keys(smell_check.check_private_imports(repo)) == {
        "private-import:relax.algo:relax.helper:_hidden",
        "private-import:relax.algo:relax.helper:_seen",
        "private-import:relax.algo:relax.pkg.inner:_deep",
    }


def test_single_use_imports_over_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(smell_check, "SINGLE_USE_IMPORT_THRESHOLD", 2)
    repo = make_repo(
        tmp_path,
        {
            "relax/helper.py": "a = b = c = d = 1\n",
            "relax/algo.py": """
            from __future__ import annotations
            from os.path import join
            from numpy import asarray
            from relax.helper import a, b, c as see, d

            __all__ = ["d"]

            def f():
                return join(asarray(a), b, b, see)
        """,
            "relax/few.py": "from relax.helper import a, b\n\n\ndef g():\n    return a + b\n",
        },
    )
    assert keys(smell_check.check_single_use_imports(repo)) == {
        "single-use-import:relax.algo:asarray",
        "single-use-import:relax.algo:a",
        "single-use-import:relax.algo:see",
    }


def test_baseline_makes_finding_not_new(tmp_path, capsys):
    make_repo(tmp_path, {"relax/algo.py": "def f(x):\n    x = int(x)\n    return x\n"})
    (tmp_path / "scripts/dev").mkdir(parents=True)
    assert smell_check.main(["--root", str(tmp_path), "--strict"]) == 1
    assert "self-cast: 1 findings, 1 new vs baseline" in capsys.readouterr().out
    assert smell_check.main(["--root", str(tmp_path), "--write-baseline"]) == 0
    baseline = json.loads((tmp_path / smell_check.BASELINE).read_text())["findings"]
    assert baseline == ["dead-def:relax.algo:f", "self-cast:relax.algo:f:x"]
    capsys.readouterr()
    (tmp_path / "relax/algo.py").write_text("\n\ndef f(x):\n    x = int(x)\n    return x\n")
    assert smell_check.main(["--root", str(tmp_path), "--strict"]) == 0
    assert "self-cast: 1 findings, 0 new vs baseline" in capsys.readouterr().out
