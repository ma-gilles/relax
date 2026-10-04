"""Completed-iteration correction reporting and parity capture boundaries."""

import ast
import inspect
import logging
from types import SimpleNamespace

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.diagnostics import iteration as captures
from relax.diagnostics import parity_dump
from relax.refinement import iteration_loop
from relax.relion.relion_normalization import NormScaleCorrectionReport, NormScaleCorrectionUpdateResult

pytestmark = pytest.mark.unit


def _controller():
    return next(n for n in ast.parse(inspect.getsource(iteration_loop)).body
                if isinstance(n, ast.FunctionDef) and n.name == 'refine_single_volume')


def _correction_block():
    loop = next(n for n in ast.walk(_controller()) if isinstance(n, (ast.For, ast.While))
                and any(isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Name)
                        and s.targets[0].id == 'correction_report' for s in n.body))
    start = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign)
                 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'correction_report')
    stop = next(i for i in range(start + 1, len(loop.body))
                if isinstance(loop.body[i], ast.If)
                and ast.unparse(loop.body[i].test) == 'follower_setup.follower_scale_state is not None')
    return compile(ast.Module(body=loop.body[start:stop], type_ignores=[]), '<actual-corrections>', 'exec')


def test_missing_report_fields_are_independent_and_remain_absent():
    report = NormScaleCorrectionReport()
    another = NormScaleCorrectionReport()
    parts = list(vars(report).values())
    assert len({id(part) for part in parts}) == 4
    assert all(part == [None, None] for part in parts)
    report.norm_corrections_per_half[0] = np.ones(2, dtype=np.float32)
    assert report.group_scale_corrections_per_half == [None, None]
    assert another.norm_corrections_per_half == [None, None]


@pytest.mark.parametrize('mode', ['missing', 'native', 'follower', 'follower_missing'])
def test_actual_correction_caller_installs_runtime_arrays_and_keeps_reporting_separate(mode):
    follower = mode.startswith('follower')
    missing = mode.endswith('missing')
    halves = [SimpleNamespace(dataset=SimpleNamespace(n_units=n), group_ids=None,
                              image_corrections=None, scale_corrections=None) for n in (2, 0)]
    result = NormScaleCorrectionUpdateResult(
        norm_corrections_per_half=[np.array([2., 3.], dtype=np.float32), np.zeros(0, dtype=np.float32)],
        avg_norm_correction_per_half=[2.5, 1.],
        group_scale_corrections_per_half=[np.array([1.], dtype=np.float32), np.zeros(1, dtype=np.float32)],
        image_corrections_per_half=[np.array([4., 5.], dtype=np.float32), np.zeros(0, dtype=np.float32)],
        scale_corrections_per_half=[np.array([6., 7.], dtype=np.float32), np.zeros(0, dtype=np.float32)],
        zero_norm_residual_counts=[0, 0],
    )
    follower_images = [np.array([8., 9.], dtype=np.float32), np.zeros(0, dtype=np.float32)]
    follower_scales = [np.array([10., 11.], dtype=np.float32), np.zeros(0, dtype=np.float32)]
    serialized_scales = [np.array([12., 13.], dtype=np.float64), None]
    setup = SimpleNamespace(follower_scale_state=object() if follower else None)
    events = []

    def prepare(stats, particles, **inputs):
        assert particles is halves
        assert inputs['do_scale_correction'] is (not follower)
        assert inputs['do_norm_correction'] is True
        assert inputs['dtype'] is np.float32
        assert inputs['group_ids_per_half'][0].dtype == np.int64
        np.testing.assert_array_equal(inputs['group_ids_per_half'][0], [0, 0])
        assert inputs['group_ids_per_half'][1].size == 0
        events.append('prepare')
        return result

    def update_followers(follower_setup, **inputs):
        assert follower_setup is setup and inputs['norm_scale_update'] is result
        for half, images, scales in zip(halves, follower_images, follower_scales, strict=True):
            half.image_corrections, half.scale_corrections = images, scales
        events.append('followers')
        return serialized_scales

    def log_update(update, **inputs):
        assert update is result
        events.append('log')

    namespace = dict(
        NormScaleCorrectionReport=NormScaleCorrectionReport, np=np,
        per_half=SimpleNamespace(noise_stats=None if missing else [SimpleNamespace(wsum_norm_correction=object()),
                                                  SimpleNamespace(wsum_norm_correction=None)]),
        experiment_datasets=[half.dataset for half in halves], halves=halves,
        follower_setup=setup, first_iteration=SimpleNamespace(relion_firstiter_cc=False), tomo_halves=False,
        scoring_dtype=np.float32, iteration=2, current_size=12, logger=logging.getLogger(__name__),
        prepare_norm_scale_update=prepare, _update_relion_follower_corrections=update_followers,
        log_norm_scale_update=log_update,
    )
    if mode == 'follower_missing':
        with pytest.raises(RuntimeError, match='requires per-half norm/scale statistics'):
            exec(_correction_block(), namespace)
        assert events == []
        return
    exec(_correction_block(), namespace)
    report = namespace['correction_report']
    if missing:
        assert events == []
        assert all(part == [None, None] for part in vars(report).values())
        return
    assert events == (['prepare', 'followers', 'log'] if follower else ['prepare', 'log'])
    assert report.group_scale_corrections_per_half is (serialized_scales if follower else result.group_scale_corrections_per_half)
    assert report.norm_corrections_per_half is result.norm_corrections_per_half
    assert report.avg_norm_correction_per_half is result.avg_norm_correction_per_half
    assert report.zero_norm_residual_counts is result.zero_norm_residual_counts
    for k, half in enumerate(halves):
        assert half.image_corrections is (follower_images[k] if follower else result.image_corrections_per_half[k])
        assert half.scale_corrections is (follower_scales[k] if follower else result.scale_corrections_per_half[k])


def _capture_inputs():
    poses = [SimpleNamespace(eulers_deg=np.arange(6, dtype=np.float64).reshape(2, 3),
                             translations_pixels=np.array([[1., 2.], [3., 4.]], dtype=np.float32)),
             SimpleNamespace(eulers_deg=np.zeros((0, 3), dtype=np.float64),
                             translations_pixels=np.zeros((0, 2), dtype=np.float32))]
    halves = [SimpleNamespace(image_corrections=np.array([1., 2.], dtype=np.float32),
                              scale_corrections=np.array([3., 4.], dtype=np.float32),
                              group_ids=np.array([0, 1], dtype=np.int32), group_count=2),
              SimpleNamespace(image_corrections=np.zeros(0, dtype=np.float32),
                              scale_corrections=np.zeros(0, dtype=np.float32),
                              group_ids=np.zeros(0, dtype=np.int32), group_count=2)]
    report = NormScaleCorrectionReport(
        group_scale_corrections_per_half=[np.array([.75, 1.25]), None],
        norm_corrections_per_half=[np.array([2., 3.], dtype=np.float32), None],
        avg_norm_correction_per_half=[2.5, None], zero_norm_residual_counts=[1, None],
    )
    return dict(
        init_relion_iteration=7, state=SimpleNamespace(translation_step=2., translation_range=9.),
        current_size=12, sigma_offset_angstrom=np.float32(3.5), random_perturbation=.125,
        settings=SimpleNamespace(tau2_fudge=np.float64(4.), grid_size=16, volume_shape=(8, 8, 8)),
        pixel_size_angstrom=np.float32(1.25), ave_pmax=np.float32(.75),
        fsc=np.array([.9, .4, 0.], dtype=np.float32), noise_variance=np.array([2., 3., 4.], dtype=np.float32),
        means=[None, None], unfiltered_means=[None, None], poses=poses, half_inputs=halves,
        corrections=report, scale_correction_data_vs_prior=np.array([4., 5., 6.], dtype=np.float32),
        log=logging.getLogger(__name__),
    )


@pytest.mark.parametrize('perturbation', [None, np.float32(.25)])
@pytest.mark.parametrize('has_instance', [False, True])
def test_capture_adapts_metadata_and_borrows_original_arrays(monkeypatch, perturbation, has_instance):
    inputs = _capture_inputs()
    inputs['random_perturbation'] = perturbation
    if has_instance:
        inputs['state'].perturbation_instance = 5
    captured = {}
    monkeypatch.setattr(parity_dump, 'dump_iteration', lambda **values: captured.update(values))
    captures.dump_numbered_iteration(2, **inputs)
    assert captured['iteration'] == 2 and captured['init_relion_iteration'] == 7
    assert captured['current_size'] == 12 and captured['grid_size'] == 16
    assert captured['volume_shape'] == (8, 8, 8)
    assert captured['random_perturbation_instance'] == (5 if has_instance else 0)
    assert_matches(captured['random_perturbation'], 0. if perturbation is None else perturbation)
    assert captured['voxel_size'] is inputs['pixel_size_angstrom']
    for field in ['fsc', 'sigma2_noise']:
        assert captured[field].dtype == np.float64
        assert_matches(captured[field], inputs['fsc' if field == 'fsc' else 'noise_variance'])
    assert captured['means'] is inputs['means']
    assert captured['unreg_means'] is inputs['unfiltered_means']
    for k in range(2):
        assert captured['new_iter_best_rotation_eulers'][k] is inputs['poses'][k].eulers_deg
        assert captured['new_iter_best_translations'][k] is inputs['poses'][k].translations_pixels
        assert captured['image_corrections'][k] is inputs['half_inputs'][k].image_corrections
        assert captured['group_ids'][k] is inputs['half_inputs'][k].group_ids
    assert captured['group_scale_corrections'] is inputs['corrections'].group_scale_corrections_per_half
    assert captured['norm_corrections'] is inputs['corrections'].norm_corrections_per_half


def test_completed_iteration_capture_keeps_serialized_schema_and_absent_follower_half(tmp_path, monkeypatch):
    monkeypatch.setenv('RELAX_PARITY_DUMP_DIR', str(tmp_path))
    parity_dump._E_STEP.clear()
    parity_dump._ITER_TIMERS.clear()
    inputs = _capture_inputs()
    for k, pose in enumerate(inputs['poses']):
        n = len(pose.eulers_deg)
        parity_dump.collect_e_step(
            half=k, em_stats=SimpleNamespace(log_evidence_per_image=np.zeros(n),
                                             best_log_score_per_image=np.zeros(n),
                                             max_posterior_per_image=np.ones(n, dtype=np.float32),
                                             rotation_posterior_sums=np.zeros(4, dtype=np.float32)),
            hard_assignment=np.zeros(n, dtype=np.int64), coarse_hard_assignment=None,
            noise_stats=None, Ft_y=None, Ft_ctf=None, pose_rotation_eulers=None,
            best_pose_rotation_eulers=pose.eulers_deg,
            best_pose_translations=pose.translations_pixels, translation_search_base=None,
        )
    captures.dump_numbered_iteration(2, **inputs)
    with np.load(tmp_path / 'iter_010.npz', allow_pickle=False) as payload:
        assert payload['relion_iteration'] == 10
        assert payload['half1_zero_norm_residual_count'] == 1
        assert 'half2_group_scale_corrections' not in payload.files
        assert 'half2_avg_norm_correction' not in payload.files
        assert payload['half1_group_ids'].dtype == np.int64
        assert payload['half1_best_eulers_total'].dtype == np.float32
        assert payload['sigma2_noise'].dtype == np.float64
        assert_matches(payload['half1_group_scale_corrections'], [.75, 1.25])
        assert_matches(payload['half1_avg_norm_correction'], 2.5)
        assert_matches(payload['scale_correction_data_vs_prior'], inputs['scale_correction_data_vs_prior'])
        np.testing.assert_array_equal(payload['half1_group_particle_counts'], [1, 1])


def test_capture_conversion_failure_retains_warning_and_iteration_context(caplog):
    inputs = _capture_inputs()

    class Unavailable:
        def __array__(self, *args, **kwargs):
            raise ValueError('capture rows unavailable')

    inputs['fsc'] = Unavailable()
    with caplog.at_level(logging.WARNING):
        captures.dump_numbered_iteration(2, **inputs)
    assert 'parity_dump.dump_iteration failed at iter 2: capture rows unavailable' in caplog.text


@pytest.mark.parametrize('timing', [False, True])
def test_actual_inactive_capture_gate_does_not_read_full_payload(timing):
    block = next(n for n in ast.walk(_controller()) if isinstance(n, ast.If)
                 and ast.unparse(n.test) == '_parity_dump.is_active()'
                 and any(isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                         and call.func.id == 'dump_numbered_iteration' for call in ast.walk(n)))
    events = []

    def reject_full(*args, **kwargs):
        raise AssertionError('full capture must remain gated')

    namespace = dict(
        _parity_dump=SimpleNamespace(is_active=lambda: False, timing_is_active=lambda: timing,
                                    dump_timing_iteration=lambda **values: events.append(values)),
        dump_numbered_iteration=reject_full, iteration=2, init_relion_iteration=7,
        t0=1., logger=logging.getLogger(__name__),
    )
    exec(compile(ast.Module(body=[block], type_ignores=[]), '<actual-capture-gate>', 'exec'), namespace)
    assert len(events) == int(timing)
    if timing:
        assert events[0]['iteration'] == 2 and events[0]['init_relion_iteration'] == 7
