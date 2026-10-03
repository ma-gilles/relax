"""Numbered phase preparation preserves borrowed grids and local policy admission."""

from __future__ import annotations

import ast
import gc
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from helpers.float_compare import assert_matches
from test_numbered_expectation import numbered_inputs

from relax.refinement import expectation
from relax.refinement.iteration_planning import ExpectationWindows
from relax.sampling import TrialGrid

pytestmark = pytest.mark.unit


def preparation_inputs(*, local=False, adaptive=False):
    half, phase, scoring = numbered_inputs(local=local, adaptive=adaptive)
    inputs = dict(
        local_sampling=phase.sampling if local else None,
        variant=phase.variant, use_adaptive=adaptive,
        base_translations=phase.sampling.base_translations,
        current_healpix_order=2, oversampling_order=int(adaptive),
        translation_step=np.float64(.75), random_perturbation=np.float64(.125),
        adaptive_pass1_rotations=None, coarse_rotation_ids=np.arange(4, dtype=np.int64),
        coarse_angular_step_deg=np.float64(15.0), options=scoring['options'],
        iteration=5, numbered_relion_iteration=17,
        collect_local_search_profile=True, local_profile_history=[],
    )
    return phase.grid, ExpectationWindows(model_size=4, image_size=2, image_box_size=4), inputs


@pytest.mark.parametrize('adaptive', [False, True])
@pytest.mark.parametrize('order', [0, 2])
@pytest.mark.parametrize('device_grid', [False, True])
def test_dense_phase_keeps_canonical_pose_rows_separate_from_device_coarse_rotations(adaptive, order, device_grid):
    grid, windows, inputs = preparation_inputs(adaptive=adaptive)
    device = np.full_like(grid.rotations, .125) if device_grid else None
    inputs.update(oversampling_order=order, adaptive_pass1_rotations=device)
    # Dense preparation does not consume local debug controls/history.
    inputs['options'] = SimpleNamespace(adaptive=SimpleNamespace(coarse_engine='gemm_dense'),
                                        symmetry=SimpleNamespace(point_group='D2'))
    phase = expectation.prepare_numbered_expectation(grid, windows, **inputs)
    assert phase.grid is grid
    assert phase.grid.rotation_eulers is grid.rotation_eulers
    assert grid.rotation_eulers.dtype == np.float64
    assert phase.sampling.effective_rotations is (device if adaptive and device is not None else grid.rotations)
    assert phase.sampling.coarse_scoring_rotations is (device if order == 0 else None)
    assert phase.sampling.current_translations is grid.translations
    assert phase.sampling.base_translations is inputs['base_translations']
    assert phase.sampling.coarse_rotation_ids is inputs['coarse_rotation_ids']
    assert phase.sampling.coarse_angular_step_deg is inputs['coarse_angular_step_deg']
    assert phase.sampling.translation_step is inputs['translation_step']
    assert phase.sampling.random_perturbation is inputs['random_perturbation']
    assert phase.sampling.current_healpix_order == 2
    assert phase.sampling.cs_for_engine == 2
    # A full model box stays explicit when the image window is narrower.
    assert phase.sampling.model_current_size_for_engine == 4
    assert phase.sampling.symmetry == 'D2'
    assert phase.sampling.coarse_engine == 'gemm_dense'
    assert phase.variant is inputs['variant']
    assert phase.local_diagnostics is None
    assert phase.use_adaptive is adaptive


@pytest.mark.parametrize('model_size, image_size, box_size, model_window, image_window', [
    (8, 8, 8, None, None), (8, 4, 8, 8, 4), (4, 4, 8, 4, 4),
])
def test_dense_phase_preserves_full_grid_sentinels_and_distinct_support(
    model_size, image_size, box_size, model_window, image_window,
):
    grid, _, inputs = preparation_inputs()
    phase = expectation.prepare_numbered_expectation(
        grid, ExpectationWindows(model_size=model_size, image_size=image_size, image_box_size=box_size),
        **inputs,
    )
    assert phase.sampling.model_current_size_for_engine == model_window
    assert phase.sampling.cs_for_engine == image_window


@pytest.mark.parametrize('order', [0, 1])
def test_local_phase_borrows_sampling_and_only_reads_adaptive_policy_for_parent_expansion(monkeypatch, order):
    grid, windows, inputs = preparation_inputs(local=True, adaptive=order > 0)
    events = []
    monkeypatch.setattr(expectation, '_local_adaptive_pass2_full_parent_enabled', lambda: events.append('full') or True)
    monkeypatch.setattr(expectation, '_local_adaptive_pass2_rotation_only_enabled', lambda: events.append('rotation') or True)
    monkeypatch.setattr(expectation, '_local_adaptive_pass2_denominator_support_mode', lambda: events.append('denominator') or 'rotation')
    # Dense-only options and geometry may be absent on the local path.
    inputs['options'] = SimpleNamespace(debug=SimpleNamespace(save_intermediates_dir='capture',
                                                             stop_after_local_search_score_only=True))
    phase = expectation.prepare_numbered_expectation(grid, object(), **inputs)
    assert events == (['full', 'rotation', 'denominator'] if order else [])
    assert phase.sampling is inputs['local_sampling']
    assert phase.grid is grid
    diagnostics = phase.local_diagnostics
    assert diagnostics.local_profile_history is inputs['local_profile_history']
    assert diagnostics.iteration == 5
    assert diagnostics.debug_iteration == 17
    assert diagnostics.save_intermediates_dir == 'capture'
    assert diagnostics.collect_local_search_profile is True
    assert diagnostics.diagnostic_score_only is True
    assert diagnostics.adaptive_pass2_full_parent is bool(order)
    assert diagnostics.adaptive_pass2_rotation_only is bool(order)
    assert diagnostics.adaptive_pass2_denominator_mode == ('rotation' if order else None)


@pytest.mark.parametrize('local, adaptive', [(False, False), (False, True), (True, False)])
def test_actual_controller_binds_current_grid_windows_and_guarded_coarse_metadata(local, adaptive):
    grid, windows, inputs = preparation_inputs(local=local, adaptive=adaptive)
    controller = Path(expectation.__file__).with_name('iteration_loop.py')
    loop = next(n for n in ast.walk(ast.parse(controller.read_text())) if isinstance(n, ast.While))
    assignment = next(n for n in loop.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'numbered_expectation' for t in n.targets))

    class Coarse:
        @property
        def angular_step_deg(self):
            if not adaptive:
                raise AssertionError('inactive coarse plan must not be read')
            return inputs['coarse_angular_step_deg']

    scope = dict(
        prepare_numbered_expectation=expectation.prepare_numbered_expectation,
        sampling=SimpleNamespace(TrialGrid=TrialGrid),
        effective_rotations=grid.rotations, effective_rotation_eulers=grid.rotation_eulers,
        effective_mstep_rotations=grid.mstep_rotations, current_translations=grid.translations,
        expectation_windows=windows, local_sampling=inputs['local_sampling'],
        numbered_variant=inputs['variant'], use_adaptive=adaptive,
        base_translations=inputs['base_translations'], current_rotation_grid=SimpleNamespace(healpix_order=2),
        state=SimpleNamespace(adaptive_oversampling=inputs['oversampling_order'], translation_step=inputs['translation_step']),
        random_perturbation=inputs['random_perturbation'], adaptive_pass1_rotations=inputs['adaptive_pass1_rotations'],
        coarse_rotation_ids_for_scoring=inputs['coarse_rotation_ids'], coarse_image_plan=Coarse(),
        options=inputs['options'], iteration=5, numbered_relion_iteration=17,
        collect_local_search_profile=True, history=SimpleNamespace(local_profile_history=inputs['local_profile_history']),
    )
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(controller), 'exec'), scope)
    phase = scope['numbered_expectation']
    assert phase.grid.rotations is grid.rotations
    assert phase.grid.rotation_eulers is grid.rotation_eulers
    assert phase.grid.mstep_rotations is grid.mstep_rotations
    assert phase.grid.translations is grid.translations
    if local:
        assert phase.sampling is inputs['local_sampling']
    else:
        assert phase.sampling.coarse_angular_step_deg is (inputs['coarse_angular_step_deg'] if adaptive else None)
    assert phase.variant is inputs['variant']


def test_actual_end_boundary_drops_phase_before_cache_policy():
    grid, windows, inputs = preparation_inputs()
    phase = expectation.prepare_numbered_expectation(grid, windows, **inputs)
    held = weakref.ref(phase)
    scope = dict(numbered_expectation=phase, numbered_tomo_sampling=object(), numbered_variant=inputs['variant'])
    del phase
    controller = Path(expectation.__file__).with_name('iteration_loop.py')
    loop = next(n for n in ast.walk(ast.parse(controller.read_text())) if isinstance(n, ast.While))
    release = next(n for n in loop.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
                   and n.value.value is None and any(isinstance(t, ast.Name) and t.id == 'numbered_expectation' for t in n.targets))
    exec(compile(ast.Module(body=[release], type_ignores=[]), str(controller), 'exec'), scope)
    gc.collect()
    assert held() is None
    assert scope['numbered_expectation'] is None
    assert scope['numbered_tomo_sampling'] is None
    assert scope['numbered_variant'] is None
    # Borrowed operands still live through their existing scientific owners.
    assert grid.rotations.shape == (4, 3, 3)
    assert_matches(grid.rotation_eulers, np.arange(12, dtype=np.float64).reshape(4, 3))
