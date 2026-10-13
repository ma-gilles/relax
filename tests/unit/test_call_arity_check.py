"""The static call-arity check ``scripts/dev/call_arity_check.py`` on small source trees."""

import textwrap

from scripts.dev import call_arity_check

ENGINE = """
from dataclasses import dataclass, field
from typing import NamedTuple


def engine(a, b=1, *, tuning, overrides=None):
    return a


def open_engine(a, **kwargs):
    return a


def options():
    return dict(tuning=1, b=2)


@dataclass
class Config:
    size: int
    scale: float = 1.0
    cache: dict = field(default_factory=dict, init=False)


class Pair(NamedTuple):
    left: int
    right: int = 0


def decorated_by_something(a):
    return a


decorated_by_something = staticmethod(decorated_by_something)
"""


def findings(tmp_path, caller: str, engine: str = ENGINE) -> list[str]:
    (tmp_path / "relax").mkdir()
    (tmp_path / "relax" / "__init__.py").write_text("")
    (tmp_path / "relax" / "engine.py").write_text(textwrap.dedent(engine))
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_caller.py").write_text(textwrap.dedent(caller))
    return call_arity_check.Index(tmp_path).findings()


def test_a_missing_required_keyword_is_found(tmp_path):
    assert findings(
        tmp_path,
        """
        from relax.engine import engine
        engine(1)
    """,
    ) == ["tests/test_caller.py: relax.engine.engine: missing required keyword-only argument 'tuning'"]


def test_an_unknown_keyword_and_too_many_positionals_are_found(tmp_path):
    assert findings(
        tmp_path,
        """
        import relax.engine as eng
        eng.engine(1, 2, 3, tuning=0, colour=1)
    """,
    ) == [
        "tests/test_caller.py: relax.engine.engine: too many positional arguments (3 > 2)",
        "tests/test_caller.py: relax.engine.engine: unexpected keyword 'colour'",
    ]


def test_dotted_imports_and_dataclass_and_namedtuple_constructors_are_checked(tmp_path):
    assert findings(
        tmp_path,
        """
        import relax.engine
        from relax import engine as module


        def run():
            relax.engine.Config(scale=2.0)
            module.Config(1, cache={})
            module.Pair(1, 2, 3)
    """,
    ) == [
        "tests/test_caller.py: relax.engine.Config: missing required argument 'size'",
        "tests/test_caller.py: relax.engine.Config: unexpected keyword 'cache'",
        "tests/test_caller.py: relax.engine.Pair: too many positional arguments (3 > 2)",
    ]


def test_a_splat_counts_with_the_keys_of_its_mapping(tmp_path):
    assert findings(
        tmp_path,
        """
        from relax.engine import engine, options


        def run(flag):
            common = dict(b=2)
            if flag:
                common["overrides"] = 1
            engine(1, **common)
            engine(1, **options())
            full = dict(common)
            full.update(tuning=3)
            engine(1, **full)
            engine(1, **{"tuning": 1, "mode": 2})
    """,
    ) == [
        "tests/test_caller.py: relax.engine.engine: missing required keyword-only argument 'tuning'",
        "tests/test_caller.py: relax.engine.engine: unexpected keyword 'mode' (from a ** mapping)",
    ]


def test_a_store_after_the_call_does_not_count_for_it(tmp_path):
    (tmp_path / "late_key").mkdir()
    assert (
        findings(
            tmp_path / "late_key",
            """
        from relax.engine import engine


        def run():
            args = {"tuning": 1}
            engine(1, **args)
            args["colour"] = 2
    """,
        )
        == []
    )
    (tmp_path / "late_tuning").mkdir()
    assert findings(
        tmp_path / "late_tuning",
        """
        from relax.engine import engine


        def run():
            args = {}
            engine(1, **args)
            args["tuning"] = 2
    """,
    ) == ["tests/test_caller.py: relax.engine.engine: missing required keyword-only argument 'tuning'"]


def test_no_finding_without_a_resolved_callee_or_known_splat(tmp_path):
    assert (
        findings(
            tmp_path,
            """
        import pytest

        from relax.engine import engine, open_engine
        from relax import engine as module


        def run(kwargs, *positional, engine_like):
            engine(*positional, **kwargs)          # unknown splat: nothing to say about tuning
            open_engine(1, colour=2, size=3)       # the callee takes **kwargs
            engine_like(1)                          # a parameter, not the import
            module.missing(1)                       # not defined in the module
            unknown_module.engine(1)                # not imported
            module.decorated_by_something()         # rebound at module level: ambiguous
            with pytest.raises(TypeError):
                engine(1)                           # the test expects the TypeError
            local = dict(tuning=1)
            local.pop("tuning")
            engine(1, **local)                      # a removal keeps what the mapping may carry


        def shadowed():
            engine = print
            engine(1)
    """,
        )
        == []
    )


def test_main_prints_the_sorted_findings_and_exits_zero(tmp_path, capsys):
    findings(
        tmp_path,
        """
        from relax.engine import engine
        engine(1, 2, 3)
    """,
    )
    assert call_arity_check.main([str(tmp_path)]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "tests/test_caller.py: relax.engine.engine: missing required keyword-only argument 'tuning'",
        "tests/test_caller.py: relax.engine.engine: too many positional arguments (3 > 2)",
    ]
