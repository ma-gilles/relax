"""The accumulator captures and the finite guard see the native halves, in order, before the join."""

from unittest.mock import Mock

import numpy as np
import pytest
from helpers.reconstruction_settings import reconstruction_settings
from helpers.tiny_refinement import CallTrace, run_tiny_refinement

from relax.diagnostics import observers
from relax.diagnostics import reconstruction as diagnostics
from relax.refinement import iteration_loop, maximization
from relax.refinement.refinement_options import ReconstructionPrograms

pytestmark = pytest.mark.unit


@pytest.fixture
def settings():
    return reconstruction_settings(
        box_size=8, voxel_size=1.5, volume_shape=(8, 8, 8),
        padding_factor=2, projection_padding_factor=1, minres_map=5,
        width_mask_edge=5, fmask_edge=2, tau2_fudge=1.0,
        particle_diameter_angstrom=None, first_iteration_lowpass_angstrom=None, programs=ReconstructionPrograms.from_environ(),
    )


@pytest.mark.parametrize(
    ('iteration', 'target', 'k_class_enabled', 'captured'),
    [(0, None, False, True), (2, 3, False, True), (2, 1, False, False), (2, 3, True, False), (0, 1, True, False)],
)
def test_prejoin_capture_routing(iteration, target, k_class_enabled, captured, settings, monkeypatch):
    """A K=1 iteration's raw half accumulators are captured, at the target iteration when one is set."""
    events = []
    numerators = [np.array([1 + 2j]), np.array([3 + 4j])]
    denominators = [np.array([5.0]), np.array([6.0])]
    raw_pixel_size = np.float32(1.5)

    def save(directory, **kwargs):
        events.append('capture')
        assert directory == '/capture'
        assert kwargs['stage'] == 'prejoin'
        assert kwargs['iteration'] == iteration
        assert kwargs['current_size'] == 6
        assert kwargs['box_size'] == settings.box_size
        assert kwargs['volume_shape'] == settings.volume_shape
        assert kwargs['padding_factor'] == settings.padding_factor
        assert kwargs['accumulator_shape'] == (16, 16, 16)
        # Raw source-pixel metadata must remain an explicit operand.
        assert kwargs['voxel_size'] is raw_pixel_size
        for key, value in zip(('Ft_y_0', 'Ft_y_1', 'Ft_ctf_0', 'Ft_ctf_1'), numerators + denominators):
            assert kwargs[key] is value

    monkeypatch.setattr(observers, 'write_bpref_accumulators', save)
    observer = observers.BpRefAccumulatorObserver(prejoin_dir='/capture', target_iteration=target)
    observer.half_accumulators_ready(
        iteration, numerators=numerators, denominators=denominators, settings=settings, current_size=6,
        accumulator_shape=(16, 16, 16), k_class_enabled=k_class_enabled, pixel_size_angstrom=raw_pixel_size,
    )
    assert events == (['capture'] if captured else [])


@pytest.mark.parametrize(('iteration', 'mode', 'expected_events'),
                         [(0, 'off', ['mode']), (2, 'warn', ['mode', 'guard']), (2, 'raise', ['mode', 'guard'])])
def test_half_accumulator_guard_routing(iteration, mode, expected_events, monkeypatch):
    events = []
    numerators = [np.array([1 + 2j]), np.array([3 + 4j])]
    denominators = [np.array([5.0]), np.array([6.0])]
    log = Mock()

    def guard_mode():
        events.append('mode')
        return mode

    def check(arrays, *, context):
        events.append('guard')
        assert tuple(arrays) == ('Ft_y_0', 'Ft_y_1', 'Ft_ctf_0', 'Ft_ctf_1')
        for actual, expected in zip(arrays.values(), numerators + denominators):
            assert actual is expected
        assert context == f'iteration={iteration} relion_iteration={12 + iteration + 1}'

    monkeypatch.setattr(diagnostics.finite_check, 'half_accumulator_guard_mode', guard_mode)
    monkeypatch.setattr(diagnostics.finite_check, 'check_half_accumulators', check)
    diagnostics.check_half_accumulators_before_join(
        numerators, denominators, iteration=iteration, init_relion_iteration=12, log=log,
    )
    assert events == expected_events
    assert log.info.call_count == int(iteration == 0)


def test_malformed_capture_target_refuses_when_the_observers_are_built(monkeypatch):
    monkeypatch.delenv('RELAX_BPREF_PREJOIN_DUMP_DIR', raising=False)
    monkeypatch.setenv('RELAX_BPREF_BOUNDARY_DUMP_ITERATION', 'bad')
    with pytest.raises(ValueError):
        observers.observers_from_environment()


def test_capture_precedes_guard_failure(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(observers, 'write_bpref_accumulators', lambda *args, **kwargs: events.append('capture'))
    monkeypatch.setattr(diagnostics.finite_check, 'half_accumulator_guard_mode', lambda: 'raise')

    def check(*args, **kwargs):
        events.append('guard')
        raise diagnostics.finite_check.FiniteCheckError('half 2 poisoned')

    monkeypatch.setattr(diagnostics.finite_check, 'check_half_accumulators', check)
    with pytest.raises(diagnostics.finite_check.FiniteCheckError, match='half 2 poisoned'):
        run_tiny_refinement(monkeypatch, final_after_max_iter=False,
                            observer=observers.BpRefAccumulatorObserver(prejoin_dir=str(tmp_path)))
    assert events == ['capture', 'guard']


@pytest.mark.parametrize("n_classes", [1, 2])
def test_actual_controller_audits_before_join_and_snapshot(n_classes, monkeypatch, tmp_path):
    """Each iteration audits the raw half accumulators before its M-step; Class3D combines the halves in the
    M-step; K=1 joins them, releases the previous maps, dumps the joined accumulators, then updates the prior."""
    trace = CallTrace(monkeypatch)
    trace.wrap(iteration_loop, "check_half_accumulators_before_join", "audit")
    trace.wrap(iteration_loop, "class_maximization", "class")
    trace.wrap(maximization, "_combine_optional_half_accumulators", "combine")
    trace.wrap(iteration_loop, "k1_maximization", "k1")
    trace.wrap(maximization, "join_half_accumulators_at_low_resolution", "join")
    trace.wrap(maximization, "_snapshot_and_release_previous_k1_means", "release")
    monkeypatch.setattr(observers, "write_bpref_accumulators", lambda *args, **kwargs: None)
    trace.wrap(observers, "write_bpref_accumulators", "dump")
    trace.wrap(maximization, "estimate_split_half_prior", "prior")
    run_tiny_refinement(monkeypatch, n_classes=n_classes, final_after_max_iter=False,
                        observer=observers.BpRefAccumulatorObserver(accum_dir=str(tmp_path)))

    if n_classes > 1:
        assert trace.labels() == 2 * ["audit", "class", "combine", "combine"]
        assert all(call.inside == ("class",) for call in trace.calls("combine"))
        return
    assert trace.labels() == 2 * ["audit", "k1", "join", "release", "dump", "prior"]
    for join, dump, prior in zip(trace.calls("join"), trace.calls("dump"), trace.calls("prior"), strict=True):
        joined_y0, joined_y1, joined_ctf0, joined_ctf1, _ = join.result
        assert dump.kwargs["stage"] == "accum"
        assert [dump.kwargs[name] for name in ("Ft_y_0", "Ft_y_1", "Ft_ctf_0", "Ft_ctf_1")] == [
            joined_y0, joined_y1, joined_ctf0, joined_ctf1,
        ]
        assert prior.args[0][0] is joined_y0 and prior.args[1][1] is joined_ctf1
