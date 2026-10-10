"""Checkpoint ownership, layout selection and controller capture lifetime."""

import weakref

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.helpers.convergence import RefinementState
from relax.refinement import iteration_snapshot
from relax.refinement.half_inputs import initialize_halfsets
from relax.refinement.iteration_snapshot import IterationSnapshot, SnapshotCapture
from relax.sampling.orientation_priors import DirectionPrior

pytestmark = pytest.mark.unit


def _shared_inputs(dtype, offset_dimension, empty_second_half):
    """The operands both checkpoint forms take, in the layout both modes share."""
    rows = (2, 0 if empty_second_half else 2)
    halves = initialize_halfsets(
        (None, None),
        previous_best_rotation_eulers=[np.zeros((n, 3), dtype=dtype) for n in rows],
        previous_best_translations=[np.zeros((n, offset_dimension), dtype=dtype) for n in rows],
        image_corrections=[np.ones(n, dtype=dtype) for n in rows],
        scale_corrections=[np.ones(n, dtype=dtype) for n in rows],
        group_ids=[None, np.zeros(rows[1], dtype=np.int64)], group_count=None,
        optics_group_ids=(None, None),
    )
    return dict(
        unfiltered_means=None,
        data_vs_prior=np.full(5, 2, dtype=dtype), noise_shells=[np.ones((2, 5), dtype=dtype)] * 2,
        half_inputs=halves,
        max_posterior=[np.full(n, 0.75, dtype=dtype) for n in rows],
        significant_counts=[np.ones(n, dtype=np.int32) for n in rows], avg_norm_correction=(1.0, None),
    )


def _k1_inputs(dtype, offset_dimension, empty_second_half):
    """``finish_k1`` operands: one map and one tau2 curve per half, the FSC curves, one prior vector per half."""
    return dict(
        **_shared_inputs(dtype, offset_dimension, empty_second_half),
        means=[np.full((1, 8), h + 1, dtype=np.complex64) for h in range(2)],
        tau2_shells_per_half=np.ones((2, 5), dtype=dtype),
        fsc=np.linspace(1, 0, 5), fsc_for_growth=np.linspace(1, 0.5, 5),
        direction_priors=[DirectionPrior(np.full(12, 0.125, dtype=dtype), 0) for _ in range(2)],
    )


def _class_inputs(n_classes, dtype, offset_dimension, empty_second_half):
    """``finish_class`` operands: a class stack, one tau2 curve and one prior row per class, the class operands."""
    rows = (2, 0 if empty_second_half else 2)
    return dict(
        **_shared_inputs(dtype, offset_dimension, empty_second_half),
        means=[np.full((n_classes, 8), h + 1, dtype=np.complex64) for h in range(2)],
        tau2_shells=np.ones((n_classes, 5), dtype=dtype),
        class_weights=np.full(n_classes, 1 / n_classes, dtype=dtype),
        class_assignments=[np.zeros(n, dtype=np.int32) for n in rows],
        direction_priors=[DirectionPrior(np.full((n_classes, 48), 0.0625, dtype=dtype), 1) for _ in range(2)],
    )


def _begin(capture):
    return capture.begin(
        2, RefinementState(iteration=1), sigma_offset_angstrom_per_half=(2.0, 3.0),
        current_size=12, incr_size=8, has_high_fsc_at_limit=True,
        random_perturbation=0.125, acc_rot_per_class=np.ones(capture.n_classes),
        acc_trans_per_class_angstrom=np.ones(capture.n_classes),
    )


def _assert_shared_capture(result, dtype, offset_dimension, empty_second_half):
    assert isinstance(result, IterationSnapshot)
    assert result.tau2_shells.dtype == np.float64
    assert result.noise_shells[0].shape == (2, 5)
    assert result.translations[1].shape == ((0 if empty_second_half else 2), offset_dimension)
    assert result.extra['translation_dtype'] == np.dtype(dtype).name
    assert result.group_ids[0].tolist() == [0, 0]
    # A half without norm correction records RELION's 1.0 in relax's frame (grid_size ** 2).
    assert_matches(result.avg_norm_correction, (1.0, 16.0**2))


def _assert_capture_owns_its_copies(result, inputs):
    before = result.means[0].copy()
    inputs['means'][0][:] = 99
    inputs['half_inputs'][0].translations[:] = 99
    assert_matches(result.means[0], before)
    assert not np.shares_memory(result.translations[0], inputs['half_inputs'][0].translations)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('offset_dimension', [2, 3])
@pytest.mark.parametrize('empty_second_half', [False, True])
def test_k1_capture_copies_the_half_maps_and_particle_frame(dtype, offset_dimension, empty_second_half):
    inputs = _k1_inputs(dtype, offset_dimension, empty_second_half)
    capture = SnapshotCapture(n_classes=1, box_size=16, voxel_size=1.5, tau2_fudge=1.0, consistency={})
    result = capture.finish_k1(_begin(capture), **inputs)
    _assert_shared_capture(result, dtype, offset_dimension, empty_second_half)
    assert result.means[0] is not result.means[1]
    assert result.fsc.dtype == np.float64 and result.fsc_for_growth.dtype == np.float64
    assert not np.shares_memory(result.fsc, inputs['fsc'])
    assert result.direction_prior[0].shape == (12,)
    assert result.extra['direction_prior_order_half1'] == 0
    assert result.class_weights is None and result.class_assignments is None
    _assert_capture_owns_its_copies(result, inputs)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('offset_dimension', [2, 3])
@pytest.mark.parametrize('empty_second_half', [False, True])
def test_class_capture_copies_the_class_stack_and_particle_frame(dtype, offset_dimension, empty_second_half):
    inputs = _class_inputs(4, dtype, offset_dimension, empty_second_half)
    capture = SnapshotCapture(n_classes=4, box_size=16, voxel_size=1.5, tau2_fudge=1.0, consistency={})
    result = capture.finish_class(_begin(capture), **inputs)
    _assert_shared_capture(result, dtype, offset_dimension, empty_second_half)
    assert result.means[0] is result.means[1]
    assert result.fsc is None and result.fsc_for_growth is None
    assert result.direction_prior[0].shape == (4, 48)
    assert result.extra['direction_prior_order_half1'] == 1
    assert result.class_weights.dtype == np.float64
    assert result.class_assignments[1].shape == (0 if empty_second_half else 2,)
    _assert_capture_owns_its_copies(result, inputs)


class _Writer:
    """A checkpoint writer that keeps only weak references to what it is handed."""

    def __init__(self, due=True):
        self._due = due
        self.published = []

    def due(self, relion_iteration):
        return self._due

    def wants_unfiltered_maps(self, relion_iteration, *, n_classes):
        return False

    def __call__(self, snapshot):
        assert isinstance(snapshot, IterationSnapshot)
        self.published.append(weakref.ref(snapshot.means[0]))


@pytest.mark.parametrize('n_classes', [1, 4])
def test_actual_controller_releases_previous_captured_maps_before_copying_new_maps(monkeypatch, n_classes):
    """The previous checkpoint's host maps are gone before the next checkpoint copies its maps."""
    from helpers.tiny_refinement import run_tiny_refinement

    from relax.refinement.refinement_options import CheckpointOptions

    writer = _Writer()
    original = iteration_snapshot.host_array
    observed = []

    def copy_array(value, dtype=None):
        assert all(reference() is None for reference in writer.published)
        observed.append(len(writer.published))
        return original(value, dtype=dtype)

    monkeypatch.setattr(iteration_snapshot, 'host_array', copy_array)
    run_tiny_refinement(
        monkeypatch, n_classes=n_classes, final_after_max_iter=False, checkpoint=CheckpointOptions(writer=writer),
    )
    assert len(writer.published) == 2
    # Both checkpoints copied arrays, the second one after the first was published and released.
    assert 0 in observed and 1 in observed


@pytest.mark.parametrize('writer', [None, _Writer(due=False)])
def test_actual_controller_does_not_read_capture_operands_when_not_due(monkeypatch, writer):
    from helpers.tiny_refinement import CallTrace, run_tiny_refinement

    from relax.refinement.refinement_options import CheckpointOptions

    trace = CallTrace(monkeypatch).wrap(SnapshotCapture, 'begin').wrap(SnapshotCapture, 'finish')
    run_tiny_refinement(monkeypatch, final_after_max_iter=False, checkpoint=CheckpointOptions(writer=writer))
    assert trace.labels() == []
    assert writer is None or writer.published == []
