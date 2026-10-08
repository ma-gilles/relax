"""Numbered phase preparation preserves borrowed grids and local policy admission."""

from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
from test_numbered_expectation import numbered_inputs

from relax.refinement import expectation
from relax.refinement.iteration_planning import ExpectationWindows
from relax.refinement.refinement_options import LocalAdaptivePass2Support

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
        collect_local_search_profile=True, local_profile_history=[], observer=scoring['observer'],
    )
    return phase.grid, ExpectationWindows(model_size=4, image_current_size=2, image_box_size=4), inputs


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
        grid, ExpectationWindows(model_size=model_size, image_current_size=image_size, image_box_size=box_size),
        **inputs,
    )
    assert phase.sampling.model_current_size_for_engine == model_window
    assert phase.sampling.cs_for_engine == image_window


@pytest.mark.parametrize('order', [0, 1])
def test_local_phase_borrows_sampling_and_carries_only_observation_policy(order):
    grid, windows, inputs = preparation_inputs(local=True, adaptive=order > 0)
    # Dense-only options and geometry may be absent on the local path.
    inputs['options'] = SimpleNamespace(local_search=SimpleNamespace(stop_after_local_search_score_only=True))
    phase = expectation.prepare_numbered_expectation(grid, object(), **inputs)
    assert phase.sampling is inputs['local_sampling']
    assert phase.grid is grid
    diagnostics = phase.local_diagnostics
    assert diagnostics.local_profile_history is inputs['local_profile_history']
    assert diagnostics.iteration == 5
    assert diagnostics.debug_iteration == 17
    assert diagnostics.observer is inputs['observer']
    assert diagnostics.collect_local_search_profile is True
    assert diagnostics.diagnostic_score_only is True


def test_local_adaptive_pass2_support_applies_only_with_parent_oversampling():
    """The run's adaptive pass-2 support (ScoringVariants) reaches scoring only where pass 2 oversamples."""
    support = LocalAdaptivePass2Support(full_parent=True, rotation_only=True, denominator_mode='rotation_only')
    assert support.at(1) is support
    assert support.at(0) == LocalAdaptivePass2Support(full_parent=False, rotation_only=False, denominator_mode=None)


@pytest.mark.parametrize('adaptive', [False, True])
def test_actual_controller_binds_current_grid_windows_and_guarded_coarse_metadata(monkeypatch, adaptive):
    """Each iteration's phase scores this iteration's trial grid with this iteration's variant; only an adaptive
    pass reads the coarse plan's angular step."""
    from helpers.tiny_refinement import CallTrace, run_tiny_refinement

    from relax.refinement import iteration_loop
    from relax.refinement.refinement_options import AdaptiveOptions

    trace = CallTrace(monkeypatch)
    trace.wrap(iteration_loop, 'iteration_trial_grid', 'grid')
    trace.wrap(iteration_loop, 'plan_adaptive_image_size', 'coarse_plan')
    trace.wrap(iteration_loop, 'DenseVariantPolicy', 'variant')
    trace.wrap(iteration_loop, 'prepare_numbered_expectation', 'phase')
    run_tiny_refinement(
        monkeypatch, final_after_max_iter=False, adaptive=AdaptiveOptions(adaptive_oversampling=int(adaptive)),
    )
    grids, variants, phases = trace.calls('grid'), trace.calls('variant'), trace.calls('phase')
    assert len(grids) == len(variants) == len(phases) == 2
    for grid, variant, phase in zip(grids, variants, phases, strict=True):
        assert phase.args[0] is grid.result
        assert phase.result.grid.rotations is grid.result.rotations
        assert phase.result.grid.rotation_eulers is grid.result.rotation_eulers
        assert phase.result.grid.mstep_rotations is grid.result.mstep_rotations
        assert phase.result.grid.translations is grid.result.translations
        assert phase.result.variant is variant.result
    if adaptive:
        plans = trace.calls('coarse_plan')
        assert [phase.result.sampling.coarse_angular_step_deg for phase in phases] == [
            plan.result.angular_step_deg for plan in plans
        ]
    else:
        assert all(phase.kwargs['coarse_angular_step_deg'] is None for phase in phases)
        assert all(phase.result.sampling.coarse_angular_step_deg is None for phase in phases)


def test_actual_end_boundary_drops_phase_before_cache_policy(monkeypatch):
    """The iteration's phase and variant are released before the between-iterations cache policy runs."""
    from helpers.tiny_refinement import CallTrace, run_tiny_refinement

    from relax.refinement import iteration_loop

    held = []
    checked = []

    def clear_caches():
        gc.collect()
        assert all(reference() is None for reference in held)
        checked.append(len(held))

    trace = CallTrace(monkeypatch)
    for name in ('DenseVariantPolicy', 'prepare_numbered_expectation'):
        trace.wrap(iteration_loop, name, after=lambda call: held.append(weakref.ref(call.result)), keep_operands=False)
    # The cache policy clears JAX's caches between iterations (RELAX_RELION_CLEAR_JAX_CACHES_BETWEEN_ITERS).
    monkeypatch.setenv('RELAX_RELION_CLEAR_JAX_CACHES_BETWEEN_ITERS', '1')
    monkeypatch.setattr(iteration_loop.jax, 'clear_caches', clear_caches)
    run_tiny_refinement(monkeypatch, final_after_max_iter=False)
    assert checked == [2, 4]
