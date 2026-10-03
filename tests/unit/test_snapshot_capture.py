"""Checkpoint ownership, layout selection and controller capture lifetime."""

import ast
import inspect
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.helpers.convergence import RefinementState
from relax.helpers.orientation_priors import DirectionPrior, HalfDirectionPriors
from relax.refinement import iteration_loop, iteration_snapshot
from relax.refinement.half_inputs import initialize_halfsets
from relax.refinement.iteration_snapshot import IterationSnapshot, SnapshotCapture

pytestmark = pytest.mark.unit


def _inputs(n_classes, dtype, offset_dimension, empty_second_half):
    rows = (2, 0 if empty_second_half else 2)
    halves = initialize_halfsets(
        (None, None),
        previous_best_rotation_eulers=[np.zeros((n, 3), dtype=dtype) for n in rows],
        previous_best_translations=[np.zeros((n, offset_dimension), dtype=dtype) for n in rows],
        image_corrections=[np.ones(n, dtype=dtype) for n in rows],
        scale_corrections=[np.ones(n, dtype=dtype) for n in rows],
        group_ids=[None, np.zeros(rows[1], dtype=np.int64)], group_count=None,
    )
    maps = [np.full((n_classes, 8), h + 1, dtype=np.complex64) for h in range(2)]
    priors = [
        HalfDirectionPriors(
            shared=DirectionPrior(np.full(12, 0.125, dtype=dtype), 0),
            classes=DirectionPrior(np.full((n_classes, 48), 0.0625, dtype=dtype), 1),
        )
        for _ in range(2)
    ]
    # The controller selects these four by mode: Class3D has no split-half FSC, K=1 no class operands.
    if n_classes > 1:
        by_mode = dict(
            fsc=None, fsc_for_growth=None,
            class_weights=np.full(n_classes, 1 / n_classes, dtype=dtype),
            class_assignments=[np.zeros(n, dtype=np.int32) for n in rows],
        )
    else:
        by_mode = dict(
            fsc=np.linspace(1, 0, 5), fsc_for_growth=np.linspace(1, 0.5, 5),
            class_weights=None, class_assignments=None,
        )
    return dict(
        **by_mode, means=maps, unfiltered_means=None,
        tau2_shells=np.ones((n_classes if n_classes > 1 else 2, 5), dtype=dtype),
        data_vs_prior=np.full(5, 2, dtype=dtype), noise_shells=[np.ones((2, 5), dtype=dtype)] * 2,
        direction_priors=priors, half_inputs=halves,
        max_posterior=[np.full(n, 0.75, dtype=dtype) for n in rows],
        significant_counts=[np.ones(n, dtype=np.int32) for n in rows], avg_norm_correction=(1.0, None),
    )


def _begin(capture):
    return capture.begin(
        2, RefinementState(iteration=1), sigma_offset_angstrom_per_half=(2.0, 3.0),
        current_size=12, incr_size=8, has_high_fsc_at_limit=True,
        random_perturbation=0.125, acc_rot_per_class=np.ones(capture.n_classes),
        acc_trans_per_class_angstrom=np.ones(capture.n_classes),
    )


@pytest.mark.parametrize('n_classes', [1, 4])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('offset_dimension', [2, 3])
@pytest.mark.parametrize('empty_second_half', [False, True])
def test_complete_capture_copies_the_selected_model_and_particle_frame(n_classes, dtype, offset_dimension, empty_second_half):
    inputs = _inputs(n_classes, dtype, offset_dimension, empty_second_half)
    capture = SnapshotCapture(n_classes=n_classes, grid_size=16, voxel_size=1.5, tau2_fudge=1.0)
    result = capture.finish(_begin(capture), **inputs)
    assert isinstance(result, IterationSnapshot)
    assert result.tau2_shells.dtype == np.float64
    assert result.noise_shells[0].shape == (2, 5)
    assert result.translations[1].shape == ((0 if empty_second_half else 2), offset_dimension)
    assert result.extra['translation_dtype'] == np.dtype(dtype).name
    assert result.group_ids[0].tolist() == [0, 0]
    # A half without norm correction records RELION's 1.0 in relax's frame (grid_size ** 2).
    assert_matches(result.avg_norm_correction, (1.0, 16.0**2))
    if n_classes == 4:
        assert result.means[0] is result.means[1]
        assert result.fsc is None and result.fsc_for_growth is None
        assert result.direction_prior[0].shape == (4, 48)
        assert result.extra['direction_prior_order_half1'] == 1
        assert result.class_weights.dtype == np.float64
        assert result.class_assignments[1].shape == (0 if empty_second_half else 2,)
    else:
        assert result.means[0] is not result.means[1]
        assert result.fsc.dtype == np.float64 and result.fsc_for_growth.dtype == np.float64
        assert not np.shares_memory(result.fsc, inputs['fsc'])
        assert result.direction_prior[0].shape == (12,)
        assert result.extra['direction_prior_order_half1'] == 0
        assert result.class_weights is None and result.class_assignments is None
    before = result.means[0].copy()
    inputs['means'][0][:] = 99
    inputs['half_inputs'][0].translations[:] = 99
    assert_matches(result.means[0], before)
    assert not np.shares_memory(result.translations[0], inputs['half_inputs'][0].translations)


def _controller_capture_block():
    tree = ast.parse(inspect.getsource(iteration_loop))
    controller = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'refine_single_volume')
    block = next(n for n in ast.walk(controller) if isinstance(n, ast.If) and ast.unparse(n.test).startswith('checkpoint_writer is not None'))
    return compile(ast.Module(body=[block], type_ignores=[]), '<actual-controller-checkpoint>', 'exec')


def test_actual_controller_releases_previous_captured_maps_before_copying_new_maps(monkeypatch):
    inputs = _inputs(4, np.float32, 2, True)
    capture = SnapshotCapture(n_classes=4, grid_size=16, voxel_size=1.5, tau2_fudge=1.0)
    old = _begin(capture)
    old.values['means'] = [np.ones(8, dtype=np.complex64)]
    old_map = weakref.ref(old.values['means'][0])
    namespace = dict(
        checkpoint_writer=lambda snapshot: None, numbered_relion_iteration=2,
        k_class_enabled=True, relion_incr_size=8, relion_has_high_fsc_at_limit=True,
        snapshot_capture=capture, state=RefinementState(iteration=1),
        current_sigma_offset_angstrom_per_half=(2.0, 3.0), current_size=12,
        random_perturbation=0.125, model_acc_rot_per_class=np.ones(4), model_acc_trans_per_class=np.ones(4),
        reference_model=SimpleNamespace(maps=inputs['means']), unreg_means=None,
        mean_signal_variance_shells=inputs['tau2_shells'], previous_data_vs_prior_for_scheduling=inputs['data_vs_prior'],
        noise_model=SimpleNamespace(radial_per_half=inputs['noise_shells']), fsc=inputs['fsc'],
        class_weights=inputs['class_weights'], direction_priors=inputs['direction_priors'], halves=inputs['half_inputs'],
        class_assignments=inputs['class_assignments'], max_posterior_per_half=inputs['max_posterior'],
        significance=SimpleNamespace(per_half=inputs['significant_counts']),
        correction_report=SimpleNamespace(avg_norm_correction_per_half=inputs['avg_norm_correction']),
        snapshot=old,
    )
    del old
    def writer(snapshot):
        assert isinstance(snapshot, IterationSnapshot)
    writer.due = lambda iteration: True
    namespace['checkpoint_writer'] = writer
    original = iteration_snapshot.host_array
    observed = []
    def copy_array(value, dtype=None):
        assert old_map() is None
        observed.append(value)
        return original(value, dtype=dtype)
    monkeypatch.setattr(iteration_snapshot, 'host_array', copy_array)
    exec(_controller_capture_block(), namespace)
    assert observed
    assert isinstance(namespace['snapshot'], IterationSnapshot)


@pytest.mark.parametrize('writer', [None, SimpleNamespace(due=lambda iteration: False)])
def test_actual_controller_does_not_read_capture_operands_when_not_due(writer):
    exec(_controller_capture_block(), dict(checkpoint_writer=writer, numbered_relion_iteration=2))
