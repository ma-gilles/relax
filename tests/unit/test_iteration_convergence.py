"""Accuracy admission, physical change metrics and numbered convergence timing."""

import logging

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.helpers.convergence import ExpectationStatistics, RefinementState, hard_class_change_fraction
from relax.helpers.resolution import ImageGeometry
from relax.parity.relion_replay_source import RelionReplay, RelionReplaySource
from relax.refinement.convergence import reset_follower_counter_once, update_iteration_convergence
from relax.refinement.half_inputs import PoseComparison
from relax.refinement.refinement_options import (
    CheckpointOptions,
    KClassOptions,
    RefinementOptions,
    RefinementSchedule,
    SymmetryOptions,
)

pytestmark = pytest.mark.unit
LOG = logging.getLogger(__name__)


def _operands(options, state, dtype, pixel_size, overrides):
    """The operands both operations take; ``overrides`` adds or replaces keywords."""
    rotations = np.tile(np.eye(3, dtype=dtype), (3, 1, 1))
    current_shifts = np.asarray([[1, 0], [0, 2], [0, 0]], dtype=dtype)
    kwargs = dict(
        image_geometry=ImageGeometry(image_shape=(16, 16), pixel_size_angstrom=pixel_size),
        iteration=0,
        source=RelionReplaySource.for_run(overrides.pop('relion_replay', None), options),
        scheduling_resolution_shell=4,
        translations=np.asarray([[0, 0], [1, 0]], dtype=dtype),
        current_assignments=np.asarray([0, 2, 4], dtype=np.int32),
        previous_assignments=np.asarray([0, 2, 4], dtype=np.int32),
        max_posterior=np.asarray([0.1, 0.2, 0.3], dtype=dtype),
        ave_pmax=0.6,
        significant_counts=np.asarray([1, 4, 9], dtype=np.int32),
        exact_acc_rot=0.25,
        exact_acc_trans=0.5,
        log=LOG,
    )
    poses = PoseComparison(rotations, rotations.copy(), current_shifts, np.zeros_like(current_shifts))
    poses = overrides.pop('pose_comparison', poses)
    kwargs.update(overrides)
    if state is None:
        state = RefinementState(healpix_order=2, acc_rot=1.0, acc_trans=2.0, current_resolution=8.0)
    return state, poses, kwargs


def _with_statistics(kwargs):
    """Gather the four particle statistics into the record the operations take."""
    kwargs = dict(kwargs)
    kwargs['statistics'] = ExpectationStatistics(
        assignments=kwargs.pop('current_assignments'), previous_assignments=kwargs.pop('previous_assignments'),
        max_posterior=kwargs.pop('max_posterior'), ave_pmax=kwargs.pop('ave_pmax'), ave_pmax_mass=1.0,
    )
    return kwargs


def _update_k1(*, options=None, state=None, dtype=np.float32, pixel_size=1.5, **overrides):
    options = RefinementOptions() if options is None else options
    native_sampling_boundary = overrides.pop('native_sampling_boundary', True)
    state, poses, kwargs = _operands(options, state, dtype, pixel_size, overrides)
    # As the controller calls it for K=1.
    result = update_iteration_convergence(
        state, poses, options, sampling_decision_now=not native_sampling_boundary, class_change_fraction=0.0,
        **_with_statistics(kwargs),
    )
    return result._replace(state=reset_follower_counter_once(result.state, options, iteration=kwargs['iteration']))


def _update_class(*, options=None, state=None, dtype=np.float32, pixel_size=1.5, **overrides):
    options = RefinementOptions(k_class=KClassOptions(n_classes=4)) if options is None else options
    overrides.setdefault('current_classes', np.zeros(3, dtype=np.int32))
    overrides.setdefault('previous_classes', np.zeros(3, dtype=np.int32))
    state, poses, kwargs = _operands(options, state, dtype, pixel_size, overrides)
    # As the controller calls it for Class3D.
    fraction = hard_class_change_fraction(kwargs.pop('current_classes'), kwargs.pop('previous_classes'))
    return update_iteration_convergence(
        state, poses, options, sampling_decision_now=False, class_change_fraction=fraction, **_with_statistics(kwargs),
    )


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('pixel_type', [float, np.float32, np.float64])
def test_change_metrics_use_physical_pixels_and_optimizer_pmax(dtype, pixel_type):
    result = _update_k1(dtype=dtype, pixel_size=pixel_type(1.5))
    assert result.state.iteration == 1
    assert_matches(result.state.current_resolution, 6.0)
    assert_matches(result.state.previous_resolution, 8.0)
    # The metric is returned as a Python float after reduction in the pose dtype.
    assert_matches(
        np.asarray(result.state.current_changes_optimal_offsets_angstrom, dtype=dtype),
        np.asarray(np.sqrt(5 * 1.5**2 / 6), dtype=dtype),
    )
    assert_matches(result.state.ave_Pmax, 0.6)
    assert_matches(result.accuracy.acc_rot, 0.25)
    assert_matches(result.accuracy.acc_trans, 0.5)
    assert_matches(result.state.acc_rot, 1.0)
    assert_matches(result.state.acc_trans, 2.0)


def _stalled_state(*, auto_sampling):
    return RefinementState(
        healpix_order=2, auto_sampling=auto_sampling,
        has_fine_enough_angular_sampling=True, nr_iter_wo_resol_gain=3,
        nr_iter_wo_large_hidden_variable_changes=3, current_resolution=6.0,
    )


def test_native_k1_does_not_make_an_end_of_iteration_sampling_decision():
    result = _update_k1(state=_stalled_state(auto_sampling=True))
    assert result.state.healpix_order == 2
    assert result.state.has_converged is False
    assert result.state.has_fine_enough_angular_sampling is True


def test_class3d_does_not_make_an_end_of_iteration_sampling_decision():
    result = _update_class(state=_stalled_state(auto_sampling=False))
    assert result.state.healpix_order == 2
    assert result.state.has_converged is False
    assert result.state.has_fine_enough_angular_sampling is True


@pytest.mark.parametrize('iteration,initial_iteration,continued,reset', [
    (0, 0, False, True),
    (1, 0, False, False),
    (0, 4, False, False),
    (0, 0, True, False),
])
def test_follower_counter_reset_is_only_for_fresh_auto_refine(iteration, initial_iteration, continued, reset):
    options = RefinementOptions(
        schedule=RefinementSchedule(init_relion_iteration=initial_iteration),
        checkpoint=CheckpointOptions(resume=object() if continued else None),
    )
    result = _update_k1(options=options, iteration=iteration)
    assert result.state.suppress_hidden_variable_increment_once is reset


def test_fresh_class3d_does_not_reset_the_follower_counter():
    result = _update_class(iteration=0)
    assert result.state.suppress_hidden_variable_increment_once is False


@pytest.mark.parametrize('exact_accuracy', [None, 0.25])
def test_support_width_only_drives_convergence_when_opted_in_and_exact_is_absent(monkeypatch, exact_accuracy):
    monkeypatch.setenv('RELAX_EM_USE_APPROX_ACC_ROT_FOR_CONVERGENCE', '1')
    result = _update_k1(exact_acc_rot=exact_accuracy)
    if exact_accuracy is None:
        assert np.isfinite(result.state.acc_rot)
        assert result.state.acc_rot != 1.0
    else:
        assert_matches(result.state.acc_rot, 1.0)
    assert result.accuracy.acc_rot == exact_accuracy


@pytest.mark.parametrize('counts', [None, np.zeros(0, dtype=np.int32)])
def test_missing_support_counts_do_not_create_an_accuracy_estimate(monkeypatch, counts):
    monkeypatch.setenv('RELAX_EM_USE_APPROX_ACC_ROT_FOR_CONVERGENCE', '1')
    result = _update_k1(significant_counts=counts, exact_acc_rot=None)
    assert_matches(result.state.acc_rot, 1.0)
    assert result.accuracy.acc_rot is None


def test_class_identity_changes_and_symmetry_are_measured_in_the_same_particle_frame():
    quarter_turn = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    rotations = np.tile(quarter_turn, (3, 1, 1))
    previous = np.tile(np.eye(3), (3, 1, 1))
    shifts = np.zeros((3, 2))
    poses = PoseComparison(rotations, previous, shifts, shifts)
    options = RefinementOptions(k_class=KClassOptions(n_classes=4), symmetry=SymmetryOptions(point_group='C4'))
    result = _update_class(
        options=options, pose_comparison=poses,
        current_classes=np.asarray([0, 1, 2]), previous_classes=np.asarray([0, 1, 3]),
    )
    assert_matches(result.state.current_changes_optimal_orientations, 0.0)
    assert_matches(result.state.current_changes_optimal_classes, 1 / 3)


@pytest.mark.parametrize('sealed', [False, True])
def test_numbered_replay_accuracy_precedes_state_update_and_controls_follow_it(tmp_path, sealed):
    (tmp_path/'run_it004_optimiser.star').write_text(
        'data_optimiser_general\n\n'
        '_rlnOverallAccuracyRotations 0.125\n'
        '_rlnOverallAccuracyTranslationsAngst 0.75\n'
        '_rlnNumberOfIterWithoutResolutionGain 7\n'
        '_rlnNumberOfIterWithoutChangingAssignments 8\n'
        '_rlnChangesOptimalOffsets 0.625\n'
        '_rlnHasConverged 1\n'
    )
    options = RefinementOptions(schedule=RefinementSchedule(init_relion_iteration=3))
    result = _update_k1(options=options, relion_replay=RelionReplay(
        perturb_replay_relion_dir=str(tmp_path), sealed_sampling_state={} if sealed else None,
    ))
    if sealed:
        assert result.accuracy.metadata is None
        assert_matches(result.state.acc_rot, 1.0)
        assert result.state.has_converged is False
    else:
        assert_matches(result.state.acc_rot, 0.125)
        assert_matches(result.state.acc_trans, 0.75)
        assert_matches(result.state.current_changes_optimal_offsets_angstrom, 0.625)
        assert result.state.nr_iter_wo_resol_gain == 7
        assert result.state.nr_iter_wo_large_hidden_variable_changes == 8
        assert result.state.has_converged is True


def test_one_shot_counter_reset_is_consumed_in_the_following_numbered_update():
    first = _update_k1()
    second = _update_k1(state=first.state, iteration=1)
    assert first.state.suppress_hidden_variable_increment_once is True
    assert second.state.suppress_hidden_variable_increment_once is False
    assert second.state.nr_iter_wo_large_hidden_variable_changes == 0


def test_expired_replay_does_not_read_the_configured_startup_directory(tmp_path):
    (tmp_path/'run_it001_optimiser.star').write_text(
        'data_optimiser_general\n\n_rlnOverallAccuracyRotations 0.125\n_rlnHasConverged 1\n'
    )
    # The replay ended before this iteration (--replay-override-max-iter 0).
    result = _update_k1(relion_replay=RelionReplay(perturb_replay_relion_dir=str(tmp_path), perturb_replay_max_iter=0))
    assert result.accuracy.metadata is None
    assert_matches(result.state.acc_rot, 1.0)
    assert result.state.has_converged is False


def test_class_frame_mismatch_still_refuses_the_update():
    with pytest.raises(ValueError, match='current_classes and previous_classes must have matching shapes'):
        _update_class(current_classes=np.zeros(3, dtype=np.int32), previous_classes=np.zeros(2, dtype=np.int32))
