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
