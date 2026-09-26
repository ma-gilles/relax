"""The resident default records which engine each pass ran on, and why it fell back."""

import pytest

from relax.sparse_pass2 import dispatch
from relax.sparse_pass2.engine_record import record_pass_engine, take_pass_engines
from relax.sparse_pass2.sparse_pass2_policy import (
    ResidentConfigurationUnsupported,
    resident_engine_selection,
    resident_refusal_reason,
)


def test_entries_are_taken_once():
    take_pass_engines()
    record_pass_engine("global", "resident")
    record_pass_engine("local", "exact_local", "parent probe")
    assert take_pass_engines() == ["global:resident", "local:exact_local (parent probe)"]
    assert take_pass_engines() == []


@pytest.mark.parametrize("raw, mode", [(None, "default"), ("", "default"), ("1", "explicit"), ("0", "off")])
def test_selection_modes(monkeypatch, raw, mode):
    name = "RELAX_SPARSE_PASS2_RESIDENT"
    if raw is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, raw)
    assert resident_engine_selection(name) == mode


def test_default_falls_back_to_compact_and_records_the_reason():
    take_pass_engines()
    refusal = ResidentConfigurationUnsupported(
        "The device-resident K=1 sparse pass 2 (RELAX_SPARSE_PASS2_RESIDENT=1) does not implement "
        "this configuration: the RELION x-half M-step is required. Clear the flag to use the compact "
        "engine; this path never falls back silently."
    )
    assert resident_refusal_reason(refusal) == "the RELION x-half M-step is required"

    def resident(*args, **kwargs):
        raise refusal

    def compact(*args, **kwargs):
        return ("compact", args, kwargs)

    out = dispatch._resident_with_compact_default(resident, compact, lambda: None, 1, relion_projector_half="p")
    assert out == ("compact", (1,), {"relion_projector_half": "p"})
    assert take_pass_engines() == ["global:compact (the RELION x-half M-step is required)"]

    out = dispatch._resident_with_compact_default(lambda *a, **k: "resident", compact, lambda: None)
    assert out == "resident"
    assert take_pass_engines() == ["global:resident"]


def test_default_propagates_errors_that_are_not_configuration_refusals():
    def resident(*args, **kwargs):
        raise NotImplementedError("a mid-run limit")

    with pytest.raises(NotImplementedError, match="mid-run"):
        dispatch._resident_with_compact_default(resident, lambda *a, **k: None, lambda: None)


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

    record_pass_engine("global", "compact", "even the smallest row capacity 64 needs 1.42 GiB")
    record_pass_engine("global", "compact", "even the smallest row capacity 32 needs 0.71 GiB")
    record_pass_engine("local", "exact_local", "full-box final pass")
    record_pass_engine("local", "exact_local", "full-box final pass")
    messages = _deprecation_messages(caplog)
    assert len(messages) == 2
    assert "the compact (bucketed sparse) pass 2 because even the smallest row capacity 64" in messages[0]
    assert "a local pass runs on the exact local engine because full-box final pass" in messages[1]
    assert all("'One engine: removal TODO'" in message for message in messages)
    # The record entries themselves are unchanged.
    assert take_pass_engines()[1:3] == [
        "global:compact (even the smallest row capacity 64 needs 1.42 GiB)",
        "global:compact (even the smallest row capacity 32 needs 0.71 GiB)",
    ]


def test_the_compact_fallback_warns_with_the_refusal(caplog, _fresh_deprecation_warnings):
    caplog.set_level("WARNING", logger="relax.sparse_pass2.engine_record")

    def resident(*args, **kwargs):
        raise ResidentConfigurationUnsupported(
            "The device-resident K=1 sparse pass 2 (RELAX_SPARSE_PASS2_RESIDENT=1) does not implement "
            "this configuration: float64 scoring is a diagnostic mode. Clear the flag to use the compact "
            "engine; this path never falls back silently."
        )

    dispatch._resident_with_compact_default(resident, lambda *a, **k: "compact", lambda: None)
    (message,) = _deprecation_messages(caplog)
    assert "compact (bucketed sparse) pass 2 because float64 scoring is a diagnostic mode" in message


def test_the_vdam_k1_default_warns_that_it_runs_exact_local(caplog, _fresh_deprecation_warnings):
    from types import SimpleNamespace

    from relax.vdam.dense_adapter import _run_vdam_pass2_route

    caplog.set_level("WARNING", logger="relax.sparse_pass2.engine_record")
    for _ in range(2):
        result = _run_vdam_pass2_route("auto", 1, None, lambda: SimpleNamespace(meta={}))
        assert result.meta["pass2_engines"] == ["global:local"]
    (message,) = _deprecation_messages(caplog)
    assert "VDAM exact-local E-step" in message and "--pass2_engine auto selects it for K=1" in message
