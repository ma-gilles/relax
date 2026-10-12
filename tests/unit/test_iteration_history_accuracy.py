"""The history records an iteration's accuracies, with explicit placeholders for what was not estimated."""

from types import SimpleNamespace

import numpy as np
import pytest

from relax.refinement.iteration_history import RefinementHistory

pytestmark = pytest.mark.unit

STATE = SimpleNamespace(current_changes_optimal_orientations=1.5, current_changes_optimal_offsets_angstrom=0.25)


def test_missing_estimates_are_recorded_as_nan_and_minus_one_per_class():
    history = RefinementHistory()
    history.record_pose_accuracy_diagnostics(
        SimpleNamespace(acc_rot=None, acc_trans=None),
        SimpleNamespace(acc_rot_per_class=None, acc_trans_per_class_angstrom=None, class_counts=None,
                        status="skipped_firstiter_cc"),
        STATE, n_classes=3,
    )
    assert np.isnan(history.acc_rot_trajectory[-1]) and np.isnan(history.acc_trans_trajectory[-1])
    assert history.acc_rot_per_class_trajectory[-1].shape == (3,) and np.all(np.isnan(history.acc_rot_per_class_trajectory[-1]))
    assert np.all(np.isnan(history.acc_trans_per_class_trajectory[-1]))
    np.testing.assert_array_equal(history.expected_accuracy_class_counts_trajectory[-1], [-1, -1, -1])
    assert history.expected_accuracy_status_trajectory[-1] == "skipped_firstiter_cc"
    assert history.smallest_change_angles_trajectory[-1] == 1.5
    assert history.smallest_change_offsets_trajectory[-1] == 0.25


def test_estimates_are_recorded_as_given():
    history = RefinementHistory()
    per_class_rot, per_class_trans, counts = np.array([1.0, 2.0]), np.array([0.5, 0.75]), np.array([3, 1])
    history.record_pose_accuracy_diagnostics(
        SimpleNamespace(acc_rot=np.float32(1.25), acc_trans=0.5),
        SimpleNamespace(acc_rot_per_class=per_class_rot, acc_trans_per_class_angstrom=per_class_trans,
                        class_counts=counts, status="ok"),
        STATE, n_classes=2,
    )
    assert history.acc_rot_trajectory[-1] == 1.25 and isinstance(history.acc_rot_trajectory[-1], float)
    assert history.acc_trans_trajectory[-1] == 0.5
    assert history.acc_rot_per_class_trajectory[-1] is per_class_rot
    assert history.acc_trans_per_class_trajectory[-1] is per_class_trans
    assert history.expected_accuracy_class_counts_trajectory[-1] is counts
    assert history.expected_accuracy_status_trajectory[-1] == "ok"
