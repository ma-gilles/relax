"""Prejoin capture and finite auditing preserve native half identity and order."""

import inspect
from unittest.mock import Mock

import numpy as np
import pytest

from relax.diagnostics import reconstruction as diagnostics
from relax.refinement import iteration_loop
from relax.refinement.mean_helpers import ReconstructionSettings

pytestmark = pytest.mark.unit


@pytest.fixture
def settings():
    return ReconstructionSettings(
        grid_size=8, voxel_size=1.5, volume_shape=(8, 8, 8),
        padding_factor=2, projection_padding_factor=1, minres_map=5,
        width_mask_edge=5, fmask_edge=2, tau2_fudge=1.0,
        particle_diameter_angstrom=None, first_iteration_lowpass_angstrom=None,
    )


@pytest.mark.parametrize(
    ('iteration', 'target', 'k_class_enabled', 'mode', 'matches', 'expected_events'),
    [
        (0, None, False, 'off', True, ['capture', 'mode']),
        (2, '3', False, 'warn', True, ['capture', 'mode', 'guard']),
        (2, '1', False, 'raise', False, ['mode', 'guard']),
        (2, '3', True, 'raise', True, ['mode', 'guard']),
        (0, '1', True, 'off', True, ['mode']),
        (2, '', False, 'warn', True, ['capture', 'mode', 'guard']),
    ],
)
def test_prejoin_capture_guard_routing(
    iteration, target, k_class_enabled, mode, matches, expected_events, settings, monkeypatch,
):
    events = []
    numerators = [np.array([1 + 2j]), np.array([3 + 4j])]
    denominators = [np.array([5.0]), np.array([6.0])]
    log = Mock()
    monkeypatch.setenv('RELAX_BPREF_PREJOIN_DUMP_DIR', '/capture')
    if target is None:
        monkeypatch.delenv('RELAX_BPREF_BOUNDARY_DUMP_ITERATION', raising=False)
    else:
        monkeypatch.setenv('RELAX_BPREF_BOUNDARY_DUMP_ITERATION', target)
    def save(directory, **kwargs):
        events.append('capture')
        assert directory == '/capture'
        assert kwargs['stage'] == 'prejoin'
        assert kwargs['iteration'] == iteration
        assert kwargs['current_size'] == 6
        assert kwargs['grid_size'] == settings.grid_size
        assert kwargs['volume_shape'] == settings.volume_shape
        assert kwargs['padding_factor'] == settings.padding_factor
        assert kwargs['accumulator_shape'] == (16, 16, 16)
        # Raw source-pixel metadata must remain an explicit operand.
        assert kwargs['voxel_size'] is raw_pixel_size
        for key, value in zip(('Ft_y_0', 'Ft_y_1', 'Ft_ctf_0', 'Ft_ctf_1'), numerators + denominators):
            assert kwargs[key] is value
    def guard_mode():
        events.append('mode')
        return mode
    def check(arrays, *, context):
        events.append('guard')
        assert tuple(arrays) == ('Ft_y_0', 'Ft_y_1', 'Ft_ctf_0', 'Ft_ctf_1')
        for actual, expected in zip(arrays.values(), numerators + denominators):
            assert actual is expected
        assert context == f'iteration={iteration} relion_iteration={12 + iteration + 1}'
    raw_pixel_size = np.float32(1.5)
    monkeypatch.setattr(diagnostics, 'write_bpref_accumulators', save)
    monkeypatch.setattr(diagnostics.finite_check, 'half_accumulator_guard_mode', guard_mode)
    monkeypatch.setattr(diagnostics.finite_check, 'check_half_accumulators', check)
    result = diagnostics.audit_prejoin_accumulators(
        numerators, denominators, settings, iteration=iteration, current_size=6,
        accumulator_shape=(16, 16, 16), k_class_enabled=k_class_enabled,
        init_relion_iteration=12,
        pixel_size_angstrom=raw_pixel_size, log=log,
    )
    assert result is matches
    assert events == expected_events
    assert log.info.call_count == int(iteration == 0)


def test_malformed_capture_target_refuses_before_guard_even_without_capture(settings, monkeypatch):
    monkeypatch.delenv('RELAX_BPREF_PREJOIN_DUMP_DIR', raising=False)
    monkeypatch.setenv('RELAX_BPREF_BOUNDARY_DUMP_ITERATION', 'bad')
    guard = Mock()
    monkeypatch.setattr(diagnostics.finite_check, 'half_accumulator_guard_mode', guard)
    with pytest.raises(ValueError):
        diagnostics.audit_prejoin_accumulators(
            [None, None], [None, None], settings, iteration=0, current_size=6,
            accumulator_shape=(16, 16, 16), k_class_enabled=True,
            init_relion_iteration=12,
            pixel_size_angstrom=1.5, log=Mock(),
        )
    guard.assert_not_called()


def test_capture_precedes_guard_failure(settings, monkeypatch):
    events = []
    monkeypatch.setenv('RELAX_BPREF_PREJOIN_DUMP_DIR', '/capture')
    monkeypatch.delenv('RELAX_BPREF_BOUNDARY_DUMP_ITERATION', raising=False)
    monkeypatch.setattr(diagnostics, 'write_bpref_accumulators', lambda *args, **kwargs: events.append('capture'))
    monkeypatch.setattr(diagnostics.finite_check, 'half_accumulator_guard_mode', lambda: 'raise')
    def check(*args, **kwargs):
        events.append('guard')
        raise diagnostics.finite_check.FiniteCheckError('half 2 poisoned')
    monkeypatch.setattr(diagnostics.finite_check, 'check_half_accumulators', check)
    with pytest.raises(diagnostics.finite_check.FiniteCheckError, match='half 2 poisoned'):
        diagnostics.audit_prejoin_accumulators(
            [None, None], [None, None], settings, iteration=0, current_size=6,
            accumulator_shape=(16, 16, 16), k_class_enabled=False,
            init_relion_iteration=12,
            pixel_size_angstrom=1.5, log=Mock(),
        )
    assert events == ['capture', 'guard']


def test_actual_controller_audits_before_join_and_snapshot():
    source = inspect.getsource(iteration_loop.refine_single_volume)
    audit = source.index('reconstruction_diagnostics.audit_prejoin_accumulators(')
    combine = source.index('Ft_y_combined = _combine_optional_half_accumulators(', audit)
    join = source.index('join_half_accumulators_at_low_resolution(', combine)
    snapshot = source.index('_snapshot_and_release_previous_k1_means(reference_model.maps)', join)
    postjoin_dump = source.index('reconstruction_diagnostics.write_bpref_accumulators(', snapshot)
    split_prior = source.index('estimate_split_half_prior(', postjoin_dump)
    assert audit < combine < join < snapshot < postjoin_dump < split_prior
