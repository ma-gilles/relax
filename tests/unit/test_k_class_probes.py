"""The K-class E-step's probes (layout S15): the default observes nothing, the commands' one reads the switches."""

import os

import numpy as np
import pytest

from relax.classification.probes import NoProbe
from relax.diagnostics.k_class_diagnostics import KClassDiagnostics
from relax.diagnostics.observers import command_k_class_probe
from relax.diagnostics.vdam_observers import KClassEstepDiagnostics, vdam_command_observer
from relax.vdam.ports import VdamObserver

pytestmark = pytest.mark.unit
LABEL = "RELAX_DEBUG_PER_POSE_DUMP_LABEL"
SWITCHES = (
    "RELAX_SIGNIFICANCE_DUMP_DIR",
    "RELAX_SIGNIFICANCE_DUMP_ORIGINAL_INDICES",
    "RELAX_GLOBAL_WINNER_SUMMARY_PATH",
    "RECOVAR_COARSE_SIGNIFICANCE_SUPPORT_AUDIT",
    LABEL,
)


@pytest.fixture(autouse=True)
def _no_switches(monkeypatch):
    for name in SWITCHES:
        monkeypatch.delenv(name, raising=False)


def test_the_default_probe_observes_nothing_whatever_the_switches(monkeypatch, tmp_path):
    monkeypatch.setenv("RELAX_SIGNIFICANCE_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("RELAX_SIGNIFICANCE_DUMP_ORIGINAL_INDICES", "0")
    monkeypatch.setenv("RELAX_GLOBAL_WINNER_SUMMARY_PATH", str(tmp_path / "summary.npz"))
    monkeypatch.setenv("RECOVAR_COARSE_SIGNIFICANCE_SUPPORT_AUDIT", "1")
    probe = NoProbe()
    assert not probe.score_dump_requested(current_size=8, debug_iteration=1)
    assert probe.score_dump_rows(None, np.arange(3)) is None
    assert not probe.wants_class_runner_up()
    assert probe.support_audit([[None]], samples_per_class=4) is None
    with probe.coarse_score_dump_label():
        assert LABEL not in os.environ


def test_the_commands_probe_reads_each_switch_when_it_is_called(monkeypatch, tmp_path):
    probe = command_k_class_probe()
    assert isinstance(probe, KClassDiagnostics)
    assert not probe.score_dump_requested(current_size=8, debug_iteration=1)
    assert not probe.wants_class_runner_up()
    assert probe.support_audit([[None]], samples_per_class=4) is None
    monkeypatch.setenv("RELAX_SIGNIFICANCE_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("RELAX_SIGNIFICANCE_DUMP_ORIGINAL_INDICES", "0")
    monkeypatch.setenv("RELAX_GLOBAL_WINNER_SUMMARY_PATH", str(tmp_path / "summary.npz"))
    monkeypatch.setenv("RECOVAR_COARSE_SIGNIFICANCE_SUPPORT_AUDIT", "1")
    assert probe.score_dump_requested(current_size=8, debug_iteration=1)
    assert probe.wants_class_runner_up()
    assert probe.support_audit([[None]], samples_per_class=4)["selected_count_sum"] == 4
    with probe.coarse_score_dump_label():
        assert os.environ[LABEL] == "coarse"
    assert LABEL not in os.environ


def test_the_initial_model_command_hands_its_e_step_the_k_class_diagnostics():
    """With no variable set the command's observer is still the diagnostics one, whose E-step probe carries the
    K-class engine's switches; the port's own observer keeps the no-op probe."""
    assert isinstance(vdam_command_observer({}).expectation_probe(), KClassEstepDiagnostics)
    assert isinstance(VdamObserver().expectation_probe(), NoProbe)


@pytest.mark.parametrize(("raw", "on"), [("", False), ("0", False), (" False ", False), ("off", False), ("1", True)])
def test_vdam_profile_switch_takes_false_tokens_as_off(raw, on):
    from relax.diagnostics.vdam_observers import PROFILE_ENV, StageProfile

    observer = vdam_command_observer({PROFILE_ENV: raw})
    assert isinstance(observer.stage_profile(), StageProfile) is on
