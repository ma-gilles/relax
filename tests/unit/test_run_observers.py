"""The run observers (code rule 15): what the controller tells them, and which ones a command builds."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from helpers.tiny_refinement import run_tiny_refinement

from relax.commands import refinement_parser
from relax.diagnostics import observers
from relax.refinement.ports import FinishedIteration, RunObserver

pytestmark = pytest.mark.unit


class Recorder(RunObserver):
    """Records the order of the hooks a run calls, and nothing else."""

    def __init__(self):
        self.events = []

    def iteration_started(self, iteration):
        self.events.append(("start", iteration))

    def stage_finished(self, iteration, stage):
        self.events.append((stage, iteration))

    def maps_reconstructed(self, maps):
        self.events.append(("maps", maps.iteration))

    def poses_updated(self, iteration, **values):
        self.events.append(("poses", iteration))

    def iteration_finished(self, finished):
        self.events.append(("finished", finished.iteration))

    def final_half_scored(self, scored):
        self.events.append(("final_half", scored.half.index))


def test_the_controller_tells_the_observer_each_moment_of_each_iteration_in_order(monkeypatch):
    recorder = Recorder()
    result = run_tiny_refinement(monkeypatch, max_iter=2, observer=recorder)
    one = ["start", "e_step", "recon", "fsc", "maps", "poses", "noise_update", "convergence", "finished"]
    assert recorder.events == [(name, i) for i in range(2) for name in one] + [("final_half", 0), ("final_half", 1)]
    # Nothing the observer asked for (its defaults) changes what the run keeps.
    assert result.history.rotation_posterior_trajectory_per_half == []


def _args(*arguments):
    return refinement_parser.parse_refinement_args(["--data_dir", "d", "--output", "o", *arguments])


def test_the_command_builds_the_observers_its_flags_and_environment_ask_for(monkeypatch, tmp_path):
    for name in ("RELAX_PARITY_DUMP_DIR", "RELAX_PARITY_TIMING_DIR"):
        monkeypatch.delenv(name, raising=False)
    assert type(observers.command_observer(_args(), timing_dir=None)) is RunObserver
    # --timing_dir reaches the parity observer as its directory, without the environment.
    timed = observers.command_observer(_args(), timing_dir=tmp_path / "flag")
    assert isinstance(timed, observers.ParityDumpObserver) and timed.timing_dir == tmp_path / "flag"
    monkeypatch.setenv("RELAX_PARITY_TIMING_DIR", str(tmp_path / "timing"))
    observer = observers.command_observer(_args(), timing_dir=None)
    assert isinstance(observer, observers.ParityDumpObserver) and observer.timing_dir == tmp_path / "timing"
    assert observers.command_observer(_args(), timing_dir=tmp_path / "flag").timing_dir == tmp_path / "flag"
    group = observers.command_observer(_args("--save_intermediates_dir", str(tmp_path / "dump")), timing_dir=None)
    assert isinstance(group, observers.ObserverGroup)
    assert [type(o) for o in group.observers] == [observers.IntermediatesObserver, observers.ParityDumpObserver]
    assert group.keeps_rotation_posteriors and group.collects_local_search_profiles
    # Timing alone does not need the unregularized maps; the full capture and the intermediates do.
    assert not observers.ParityDumpObserver(tmp_path / "timing").wants_unfiltered_maps(1)
    monkeypatch.setenv("RELAX_PARITY_DUMP_DIR", str(tmp_path / "parity"))
    assert observers.ParityDumpObserver(None).wants_unfiltered_maps(1)


def test_the_parity_observer_writes_the_capture_or_else_the_timing(monkeypatch):
    calls = []
    monkeypatch.setattr(observers, "dump_numbered_iteration", lambda iteration, **kw: calls.append(("capture", iteration)))
    monkeypatch.setattr(observers.parity_dump, "dump_timing_iteration", lambda **kw: calls.append(("timing", kw)))
    finished = FinishedIteration(*([None] * len(FinishedIteration._fields)))._replace(
        iteration=3, init_relion_iteration=2,
    )
    for active, timing in ((True, Path("t")), (False, Path("t")), (False, None)):
        monkeypatch.setattr(observers.parity_dump, "is_active", lambda active=active: active)
        observers.ParityDumpObserver(timing).iteration_finished(finished)
    assert calls == [("capture", 3), ("timing", dict(out=Path("t"), iteration=3, init_relion_iteration=2))]


def test_a_group_hands_each_hook_to_every_observer_in_order():
    first, second = Recorder(), Recorder()
    group = observers.ObserverGroup([first, second])
    group.iteration_started(0)
    group.stage_finished(0, "e_step")
    assert first.events == second.events == [("start", 0), ("e_step", 0)]
    assert not group.wants_unfiltered_maps(1) and not group.keeps_rotation_posteriors


def test_the_environment_dump_observers_come_from_their_variables(monkeypatch, tmp_path):
    for name in ("RELAX_PARITY_DUMP_DIR", "RELAX_PARITY_TIMING_DIR", "RELAX_BPREF_PREJOIN_DUMP_DIR",
                 "RELAX_BPREF_ACCUM_DUMP_DIR", "RELAX_NOISE_DEBUG_DUMP_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RELAX_KCLASS_DUMP_DIR", str(tmp_path / "kclass"))
    monkeypatch.setenv("RELAX_PREMASK_DUMP_DIR", str(tmp_path / "premask"))
    found = observers.observers_from_environment(timing_dir=None)
    assert [type(o) for o in found] == [observers.ClassDumpObserver, observers.PremaskObserver]
    assert [o.directory for o in found] == [str(tmp_path / "kclass"), str(tmp_path / "premask")]


_TARGET_HALF_ENV = ("RELAX_SIGNIFICANCE_DUMP_TARGET_HALF", "RELAX_PASS2_DUMP_TARGET_HALF")


def test_the_controller_asks_the_observer_which_halves_each_numbered_iteration_scores(monkeypatch):
    class Halves(RunObserver):
        def __init__(self):
            self.asked = []

        def scored_halves(self, numbered_relion_iteration, *, n_classes, experiment_datasets):
            self.asked.append((numbered_relion_iteration, n_classes, len(experiment_datasets)))
            return (0, 1)

    observer = Halves()
    run_tiny_refinement(monkeypatch, max_iter=2, observer=observer)
    assert observer.asked == [(1, 1, 2), (2, 1, 2)]
    assert RunObserver().scored_halves(1, n_classes=1, experiment_datasets=[None, None]) == (0, 1)


def test_a_one_half_dump_target_is_the_environment_target_half_observer(monkeypatch, tmp_path):
    for name in _TARGET_HALF_ENV:
        monkeypatch.delenv(name, raising=False)
    assert not any(isinstance(o, observers.TargetHalfObserver) for o in observers.observers_from_environment(timing_dir=None))
    for name, value in {
        "RELAX_SIGNIFICANCE_DUMP_TARGET_HALF": "2", "RELAX_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET": "1",
        "RELAX_SIGNIFICANCE_DUMP_DIR": str(tmp_path), "RELAX_SIGNIFICANCE_DUMP_ITERATION": "2",
        "RELAX_SIGNIFICANCE_DUMP_ORIGINAL_INDICES": "1,3",
    }.items():
        monkeypatch.setenv(name, value)
    target = [o for o in observers.observers_from_environment(timing_dir=None) if isinstance(o, observers.TargetHalfObserver)]
    assert len(target) == 1
    datasets = [SimpleNamespace(dataset_indices=np.asarray(rows)) for rows in ([2, 4], [1, 3])]
    # Only the target iteration scores the target half alone; a group defers to the observer that targets one.
    assert target[0].scored_halves(1, n_classes=1, experiment_datasets=datasets) == (0, 1)
    assert target[0].scored_halves(2, n_classes=1, experiment_datasets=datasets) == (1,)
    group = observers.ObserverGroup([RunObserver(), target[0]])
    assert group.scored_halves(2, n_classes=1, experiment_datasets=datasets) == (1,)


def test_the_controller_tells_the_observer_of_each_recorded_numbered_half(monkeypatch):
    class Halves(RunObserver):
        def __init__(self):
            self.recorded = []

        def numbered_half_recorded(self, recorded):
            assert recorded.particles.index == recorded.half_index
            assert recorded.per_half.coarse_ha[recorded.half_index] is recorded.score_result.coarse_ha
            self.recorded.append((recorded.iteration, recorded.half_index))

    observer = Halves()
    run_tiny_refinement(monkeypatch, max_iter=2, observer=observer)
    assert observer.recorded == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_the_parity_capture_asks_for_the_pose_eulers_only_while_it_dumps(monkeypatch, tmp_path):
    monkeypatch.delenv("RELAX_PARITY_DUMP_DIR", raising=False)
    assert not RunObserver().keeps_pose_rotation_eulers
    assert not observers.ParityDumpObserver(None).keeps_pose_rotation_eulers
    monkeypatch.setenv("RELAX_PARITY_DUMP_DIR", str(tmp_path / "parity"))
    assert observers.ParityDumpObserver(None).keeps_pose_rotation_eulers
    assert observers.ObserverGroup([RunObserver(), observers.ParityDumpObserver(None)]).keeps_pose_rotation_eulers


def test_a_bpref_device_panel_variable_builds_its_observer_ahead_of_the_parity_capture(monkeypatch, tmp_path):
    from relax.diagnostics import bpref_diagnostics

    for name in (*bpref_diagnostics.BPREF_DIAGNOSTIC_ENVS, "RELAX_PARITY_DUMP_DIR", "RELAX_PARITY_TIMING_DIR"):
        monkeypatch.delenv(name, raising=False)
    assert not any(isinstance(o, observers.BpRefDiagnosticsObserver) for o in observers.observers_from_environment(timing_dir=None))
    monkeypatch.setenv("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR", str(tmp_path / "panels"))
    monkeypatch.setenv("RELAX_PARITY_TIMING_DIR", str(tmp_path / "timing"))
    assert [type(o) for o in observers.observers_from_environment(timing_dir=None)][:2] == [
        observers.BpRefDiagnosticsObserver, observers.ParityDumpObserver,
    ]
    flushed = []
    monkeypatch.setattr(observers.bpref_diagnostics, "flush_selected_bpref_device_panel", lambda **kw: flushed.append(kw))
    # A scored half's panel is flushed; an empty half has none.
    for n_units in (2, 0):
        observers.BpRefDiagnosticsObserver().numbered_half_recorded(SimpleNamespace(
            iteration=3, half_index=1, particles=SimpleNamespace(dataset=SimpleNamespace(n_units=n_units)),
        ))
    assert flushed == [dict(iteration_index=3, half_index=1)]
