"""The run observers (code rule 15): what the controller tells them, and which ones a command builds."""

import pytest
from helpers.tiny_refinement import run_tiny_refinement

from relax.diagnostics import observers
from relax.refinement import command_options
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
    return command_options.parse_refinement_args(["--data_dir", "d", "--output", "o", *arguments])


def test_the_command_builds_the_observers_its_flags_and_environment_ask_for(monkeypatch, tmp_path):
    for name in ("RELAX_PARITY_DUMP_DIR", "RELAX_PARITY_TIMING_DIR"):
        monkeypatch.delenv(name, raising=False)
    assert observers.command_observer(_args()) is None
    monkeypatch.setenv("RELAX_PARITY_TIMING_DIR", str(tmp_path / "timing"))
    assert isinstance(observers.command_observer(_args()), observers.ParityDumpObserver)
    group = observers.command_observer(_args("--save_intermediates_dir", str(tmp_path / "dump")))
    assert isinstance(group, observers.ObserverGroup)
    assert [type(o) for o in group.observers] == [observers.IntermediatesObserver, observers.ParityDumpObserver]
    assert group.keeps_rotation_posteriors and group.collects_local_search_profiles
    # Timing alone does not need the unregularized maps; the full capture and the intermediates do.
    assert not observers.ParityDumpObserver().wants_unfiltered_maps(1)
    monkeypatch.setenv("RELAX_PARITY_DUMP_DIR", str(tmp_path / "parity"))
    assert observers.ParityDumpObserver().wants_unfiltered_maps(1)


def test_the_parity_observer_writes_the_capture_or_else_the_timing(monkeypatch):
    calls = []
    monkeypatch.setattr(observers, "dump_numbered_iteration", lambda iteration, **kw: calls.append(("capture", iteration)))
    monkeypatch.setattr(observers.parity_dump, "dump_timing_iteration", lambda **kw: calls.append(("timing", kw)))
    finished = FinishedIteration(*([None] * len(FinishedIteration._fields)))._replace(
        iteration=3, init_relion_iteration=2,
    )
    for active, timing in ((True, True), (False, True), (False, False)):
        monkeypatch.setattr(observers.parity_dump, "is_active", lambda active=active: active)
        monkeypatch.setattr(observers.parity_dump, "timing_is_active", lambda timing=timing: timing)
        observers.ParityDumpObserver().iteration_finished(finished)
    assert calls == [("capture", 3), ("timing", dict(iteration=3, init_relion_iteration=2))]


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
    found = observers.observers_from_environment()
    assert [type(o) for o in found] == [observers.ClassDumpObserver, observers.PremaskObserver]
    assert [o.directory for o in found] == [str(tmp_path / "kclass"), str(tmp_path / "premask")]
