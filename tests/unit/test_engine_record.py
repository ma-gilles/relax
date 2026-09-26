"""The resident default records which engine each pass ran on, and why it fell back."""

import pytest

from relax.sparse_pass2.engine_record import record_pass_engine, take_pass_engines
from relax.sparse_pass2.sparse_pass2_policy import (
    resident_engine_selection,
)


def test_entries_are_taken_once():
    take_pass_engines()
    record_pass_engine("global", "resident")
    record_pass_engine("local", "exact_local", "parent probe")
    assert take_pass_engines() == ["global:resident", "local:exact_local (parent probe)"]
    assert take_pass_engines() == []


@pytest.mark.parametrize("raw, mode", [(None, "default"), ("", "default"), ("1", "explicit"), ("0", "off")])
def test_selection_modes(monkeypatch, raw, mode):
    name = "RELAX_LOCAL_SEARCH_RESIDENT"
    if raw is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, raw)
    assert resident_engine_selection(name) == mode


@pytest.fixture
def _fresh_deprecation_warnings(monkeypatch):
    from relax.sparse_pass2 import engine_record

    monkeypatch.setattr(engine_record, "_warned", set())
    take_pass_engines()
    yield
    take_pass_engines()


def _deprecation_messages(caplog):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("DEPRECATED engine")]


def test_a_deprecated_engine_warns_once_per_reason(caplog, _fresh_deprecation_warnings):
    caplog.set_level("WARNING", logger="relax.sparse_pass2.engine_record")
    record_pass_engine("global", "resident")
    assert _deprecation_messages(caplog) == []

    record_pass_engine("local", "exact_local", "even the smallest row capacity 64 needs 1.42 GiB")
    record_pass_engine("local", "exact_local", "even the smallest row capacity 32 needs 0.71 GiB")
    record_pass_engine("local", "exact_local", "full-box final pass")
    record_pass_engine("local", "exact_local", "full-box final pass")
    messages = _deprecation_messages(caplog)
    assert len(messages) == 2
    assert "the exact local engine because even the smallest row capacity 64" in messages[0]
    assert "a local pass runs on the exact local engine because full-box final pass" in messages[1]
    assert all("'One engine: removal TODO'" in message for message in messages)
    # The record entries themselves are unchanged.
    assert take_pass_engines()[1:3] == [
        "local:exact_local (even the smallest row capacity 64 needs 1.42 GiB)",
        "local:exact_local (even the smallest row capacity 32 needs 0.71 GiB)",
    ]


def test_the_vdam_exact_local_route_warns_once(caplog, _fresh_deprecation_warnings):
    """``run_local_k_class_em`` is the VDAM exact-local E-step; it warns before any work."""

    from relax.classification.k_class import run_local_k_class_em

    caplog.set_level("WARNING", logger="relax.sparse_pass2.engine_record")
    for _ in range(2):
        # A rejected keyword stops the call right after the warning, before any device work.
        with pytest.raises(ValueError, match="controls these arguments directly"):
            run_local_k_class_em(None, None, None, None, "linear_interp", normalization_log_z=None)
    (message,) = _deprecation_messages(caplog)
    assert "VDAM exact-local E-step" in message and "--pass2_engine auto at K=1" in message
