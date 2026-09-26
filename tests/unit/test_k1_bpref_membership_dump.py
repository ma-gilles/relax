
import pytest

from relax.diagnostics import bpref_diagnostics


@pytest.mark.unit
@pytest.mark.parametrize(
    ("directory", "iteration", "half", "context", "expected"),
    [
        ("", None, None, (2, 1), False),
        ("  ", "invalid", "3", (2, 1), False),
        ("enabled", None, None, (-1, -1), True),
        ("enabled", "2", "1", (2, 1), True),
        ("enabled", "3", "1", (2, 1), False),
        ("enabled", "3", "invalid", (2, 1), False),
        ("enabled", "2", "2", (2, 1), False),
        ("enabled", "2", "", (2, 1), True),
        ("enabled", "2", "3", (2, 1), ValueError),
        ("enabled", "invalid", None, (2, 1), ValueError),
    ],
)
def test_membership_selector_preserves_filter_order(monkeypatch, directory, iteration, half, context, expected):
    monkeypatch.setenv("RELAX_BPREF_MEMBERSHIP_DUMP_DIR", directory)
    for name, value in [("ITERATION", iteration), ("HALF", half)]:
        name = "RELAX_BPREF_MEMBERSHIP_DUMP_" + name
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    monkeypatch.setattr(bpref_diagnostics, "_bpref_contribution_context", {"iteration": context[0], "half": context[1]})
    if expected is ValueError:
        with pytest.raises(ValueError):
            bpref_diagnostics._bpref_membership_dump_requested()
    else:
        assert bpref_diagnostics._bpref_membership_dump_requested() is expected


