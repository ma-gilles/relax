"""Completed-iteration correction reporting and parity capture boundaries."""

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


def test_missing_report_fields_are_independent_and_remain_absent():
    report = NormScaleCorrectionReport()
    another = NormScaleCorrectionReport()
    parts = list(vars(report).values())
    assert len({id(part) for part in parts}) == 4
    assert all(part == [None, None] for part in parts)
    report.norm_corrections_per_half[0] = np.ones(2, dtype=np.float32)
    assert report.group_scale_corrections_per_half == [None, None]
    assert another.norm_corrections_per_half == [None, None]


class _Writer:
    def __init__(self):
        self.snapshots = []

    def due(self, relion_iteration):
        return True

    def wants_unfiltered_maps(self, relion_iteration, *, n_classes):
        return False

    def __call__(self, snapshot):
        self.snapshots.append(snapshot)


@pytest.mark.parametrize('mode', ['missing', 'native', 'follower', 'follower_missing'])
def test_actual_correction_caller_installs_runtime_arrays_and_keeps_reporting_separate(monkeypatch, mode):
    """The controller installs each numbered update's corrections on the halves (the follower state installs
    them under the follower emulation) and reports them separately; a half without statistics changes nothing,
    and the follower emulation refuses to go on without them."""
    from helpers.tiny_refinement import CallTrace, follower_scale_replay, run_tiny_refinement

    from relax.refinement.refinement_options import CheckpointOptions

    follower = mode.startswith('follower')
    missing = mode.endswith('missing')
    updates, follower_arrays = [], []

    def numbered(per_half, halves, **inputs):
        assert inputs['do_scale_correction'] is (not follower)
        assert inputs['do_norm_correction'] is True
        assert inputs['dtype'] == np.float32
        if missing:
            return None
        sizes = [int(half.dataset.n_units) for half in halves]
        value = float(len(updates) + 2)
        update = NormScaleCorrectionUpdateResult(
            norm_corrections_per_half=[np.full(n, value, dtype=np.float32) for n in sizes],
            avg_norm_correction_per_half=[value, value + 0.5],
            group_scale_corrections_per_half=[np.full(2, value, dtype=np.float32)] * 2,
            image_corrections_per_half=[np.full(n, value + 1.0, dtype=np.float32) for n in sizes],
            scale_corrections_per_half=[np.full(n, 1.0 / value, dtype=np.float32) for n in sizes],
            zero_norm_residual_counts=[0, 0],
        )
        updates.append((update, halves))
        return update

    def update_followers(follower_setup, **inputs):
        assert follower_setup.follower_scale_state is not None
        assert inputs['norm_scale_update'] is updates[-1][0]
        images = [np.full(half.dataset.n_units, 8.0, dtype=np.float32) for half in inputs['relion_half_inputs']]
        scales = [np.full(half.dataset.n_units, 0.5, dtype=np.float32) for half in inputs['relion_half_inputs']]
        for half, image, scale in zip(inputs['relion_half_inputs'], images, scales, strict=True):
            half.image_corrections, half.scale_corrections = image, scale
        serialized = [np.array([12., 13.], dtype=np.float64), None]
        follower_arrays.append((images, scales, serialized))
        return serialized

    def installed(call):
        update, halves = updates[-1]
        images, scales = (follower_arrays[-1][:2] if follower else
                          (update.image_corrections_per_half, update.scale_corrections_per_half))
        for k, half in enumerate(halves):
            assert half.image_corrections is images[k] and half.scale_corrections is scales[k]

    monkeypatch.setattr(iteration_loop, 'numbered_norm_scale_update', numbered)
    monkeypatch.setattr(iteration_loop, '_update_relion_follower_corrections', update_followers)
    trace = CallTrace(monkeypatch)
    trace.wrap(iteration_loop, 'norm_scale_report', 'report')
    trace.wrap(iteration_loop, 'log_norm_scale_update', 'log', before=installed)
    writer = _Writer()
    run = dict(final_after_max_iter=False, checkpoint=CheckpointOptions(writer=writer))
    if follower:
        run.update(n_classes=2, replay=follower_scale_replay(2))
    if mode == 'follower_missing':
        with pytest.raises(RuntimeError, match='requires per-half norm/scale statistics'):
            run_tiny_refinement(monkeypatch, **run)
        assert trace.labels() == []
        return
    run_tiny_refinement(monkeypatch, **run)
    if missing:
        assert trace.labels() == []
        # The run files then hold RELION's default average norm correction, the squared box size.
        assert [snapshot.avg_norm_correction for snapshot in writer.snapshots] == [(64.0, 64.0)] * 2
        return
    assert trace.labels() == ['report', 'log'] * 2
    for at, (report, (update, _)) in enumerate(zip(trace.calls('report'), updates, strict=True)):
        assert report.args[0] is update
        expected_group_scales = follower_arrays[at][2] if follower else update.group_scale_corrections_per_half
        assert report.args[1] is expected_group_scales
        assert report.result.norm_corrections_per_half is update.norm_corrections_per_half
        assert report.result.avg_norm_correction_per_half is update.avg_norm_correction_per_half
        assert report.result.zero_norm_residual_counts is update.zero_norm_residual_counts
        assert writer.snapshots[at].avg_norm_correction == tuple(update.avg_norm_correction_per_half)


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
def test_actual_inactive_capture_gate_does_not_read_full_payload(monkeypatch, timing):
    """Without a parity dump the parity observer never builds the full capture; a timing dump gets one row per
    iteration."""
    from helpers.tiny_refinement import run_tiny_refinement

    from relax.diagnostics import observers

    events = []

    def reject_full(*args, **kwargs):
        raise AssertionError('full capture must remain gated')

    monkeypatch.setattr(parity_dump, 'is_active', lambda: False)
    monkeypatch.setattr(parity_dump, 'timing_is_active', lambda: timing)
    monkeypatch.setattr(parity_dump, 'dump_timing_iteration', lambda **values: events.append(values))
    monkeypatch.setattr(observers, 'dump_numbered_iteration', reject_full)
    run_tiny_refinement(monkeypatch, final_after_max_iter=False, observer=observers.ParityDumpObserver())
    assert [(event['iteration'], event['init_relion_iteration']) for event in events] == (
        [(0, 0), (1, 0)] if timing else []
    )
