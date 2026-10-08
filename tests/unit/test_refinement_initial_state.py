"""Startup source precedence and continuation admission for refinement state."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from helpers.float_compare import assert_matches
from helpers.run_options import stand_in

from relax.helpers.convergence import RefinementState
from relax.helpers.resolution import ImageGeometry
from relax.parity.relion_replay_source import RelionReplay
from relax.refinement.iteration_planning import initialize_refinement_state
from relax.refinement.iteration_snapshot import IterationSnapshot, refinement_state_fields
from relax.refinement.refinement_options import (
    CheckpointOptions,
    KClassOptions,
)

pytestmark = pytest.mark.unit


def _geometry(pixel_size=2.25):
    return ImageGeometry(image_shape=(64, 64), pixel_size_angstrom=pixel_size)


def _initialize(options, *, subtomogram=False, pixel_size=2.25, dtype=np.float32, relion_replay=None):
    """The start-up state with the input source the command builds for ``relion_replay`` (None: native)."""
    from relax.parity.relion_replay_source import RelionReplaySource

    return initialize_refinement_state(
        options, _geometry(pixel_size), subtomogram=subtomogram, dtype=dtype,
        source=RelionReplaySource.for_run(relion_replay, options),
    )


@pytest.mark.parametrize('n_classes', [1, 4])
@pytest.mark.parametrize('subtomogram', [False, True])
@pytest.mark.parametrize('pixel_type', [float, np.float32, np.float64])
def test_startup_keeps_class_mode_grid_units_and_unknown_accuracy(n_classes, subtomogram, pixel_type):
    options = stand_in.options(
        schedule=stand_in.schedule(init_healpix_order=2, init_translation_range=4.0,
                                    init_translation_step=1.25, particle_diameter_ang=90.0),
        k_class=KClassOptions(n_classes=n_classes),
    )
    state = _initialize(options, subtomogram=subtomogram, pixel_size=pixel_type(1.416667))
    assert state.iteration == 0
    assert state.healpix_order == 2
    assert state.auto_sampling is (n_classes == 1)
    assert state.subtomogram is subtomogram
    assert_matches(state.voxel_size_angstrom, float(pixel_type(1.416667)))
    assert_matches(state.translation_range, 4.0)
    assert_matches(state.translation_step, 1.25)
    assert_matches(state.particle_diameter_angstrom, 90.0)
    assert np.isposinf(state.current_resolution)
    assert np.isposinf(state.acc_rot)
    assert np.isposinf(state.acc_trans)
    assert not state.has_converged


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('current_size,resolution_shell', [(64, 6), (8, 5)])
def test_initial_fsc_wins_over_both_lowpass_sources_and_respects_available_shells(dtype, current_size, resolution_shell):
    fsc = np.full(33, 0.2, dtype=dtype)
    fsc[:7] = 0.9
    original = fsc.copy()
    options = stand_in.options(
        schedule=stand_in.schedule(init_fsc=fsc, init_current_size=current_size, ini_high_angstrom=80.0),
        parity=stand_in.parity(relion_firstiter_ini_high_angstrom=20.0),
    )
    state = _initialize(options, dtype=dtype)
    # Truncation crosses at shell 4, then the existing RELION minres_map=5 applies.
    assert_matches(state.current_resolution, 64 * 2.25 / resolution_shell)
    assert_matches(state.previous_resolution, state.current_resolution)
    assert_matches(fsc, original)


@pytest.mark.parametrize('restart_iteration,firstiter_high,ordinary_high,expected', [
    (0, 30.0, 80.0, 28.8),
    (0, None, 80.0, 72.0),
    (0, None, None, np.inf),
    (2, 30.0, 80.0, np.inf),
])
def test_lowpass_priority_and_fresh_run_admission(restart_iteration, firstiter_high, ordinary_high, expected):
    options = stand_in.options(
        schedule=stand_in.schedule(init_relion_iteration=restart_iteration, ini_high_angstrom=ordinary_high),
        parity=stand_in.parity(relion_firstiter_ini_high_angstrom=firstiter_high),
    )
    state = _initialize(options)
    assert_matches(state.current_resolution, expected)
    assert_matches(state.previous_resolution, expected)


def test_replay_restart_uses_recorded_resolution_accuracy_and_stalls_before_fresh_sources(tmp_path):
    (tmp_path / 'run_it002_half1_model.star').write_text(
        'data_model_general\n_rlnCurrentImageSize 24\n_rlnCurrentResolution 8.0\n'
    )
    (tmp_path / 'run_it002_optimiser.star').write_text(
        'data_optimiser_general\n_rlnNumberOfIterWithoutResolutionGain 3\n'
        '_rlnNumberOfIterWithoutChangingAssignments 4\n'
        '_rlnOverallAccuracyRotations 0.75\n_rlnOverallAccuracyTranslationsAngst 0.25\n'
        '_rlnSmallestChangesOrientations 0.5\n_rlnSmallestChangesOffsets 0.125\n'
        '_rlnHasConverged 0\n'
    )
    options = stand_in.options(
        schedule=stand_in.schedule(init_relion_iteration=2, init_fsc=np.full(33, 0.9)),
        parity=stand_in.parity(relion_firstiter_ini_high_angstrom=30.0),
    )
    state = _initialize(options, relion_replay=RelionReplay(perturb_replay_relion_dir=str(tmp_path)))
    assert_matches(state.current_resolution, 8.0)
    assert_matches(state.previous_resolution, 8.0)
    assert_matches(state.acc_rot, 0.75)
    assert_matches(state.acc_trans, 0.25)
    assert state.nr_iter_wo_resol_gain == 3
    assert state.nr_iter_wo_assignment_changes == 4
    assert state.nr_iter_wo_large_hidden_variable_changes == 4


def test_sealed_state_suppresses_restart_read_and_frozen_fields_override_resolution(tmp_path):
    # If restart were read, the deliberately malformed model would fail.
    (tmp_path / 'run_it002_half1_model.star').write_text('data_model_general\n')
    options = stand_in.options(
        schedule=stand_in.schedule(init_relion_iteration=2, init_fsc=np.full(33, 0.9)),
    )
    state = _initialize(options, relion_replay=RelionReplay(
        perturb_replay_relion_dir=str(tmp_path), sealed_sampling_state={},
        frozen_refinement_state_fields=dict(current_resolution=12.0, previous_resolution=12.0, nr_iter_wo_resol_gain=2),
    ))
    assert state.healpix_order == options.schedule.init_healpix_order
    assert state.nr_iter_wo_resol_gain == 2
    assert_matches(state.current_resolution, 12.0)
    assert_matches(state.previous_resolution, 12.0)


def _snapshot():
    state = RefinementState(iteration=7, healpix_order=3, current_resolution=9.0,
                            previous_resolution=10.0, nr_iter_wo_resol_gain=2, max_healpix_order=7)
    return IterationSnapshot(
        relion_iteration=7, n_classes=1, box_size=64, pixel_size=2.25, tau2_fudge=1.0,
        means=[], tau2_shells=np.empty(0), data_vs_prior=np.empty(0), noise_shells=[],
        sigma_offset_angstrom=(2.0, 2.0), current_size=32, incr_size=10,
        has_high_fsc_at_limit=False, random_perturbation=0.0, state_fields=refinement_state_fields(state),
        rotation_eulers=[], translations=[], image_corrections=[], scale_corrections=[], group_ids=[],
    )


def test_valid_continuation_restores_recorded_state_and_keeps_configured_run_limit():
    options = stand_in.options(
        schedule=stand_in.schedule(init_relion_iteration=7, max_healpix_order=12),
        checkpoint=CheckpointOptions(resume=_snapshot()),
    )
    state = _initialize(options)
    assert state.iteration == 7
    assert state.healpix_order == 3
    assert state.max_healpix_order == 12
    assert state.nr_iter_wo_resol_gain == 2
    assert_matches(state.current_resolution, 9.0)


@pytest.mark.parametrize('field,value', [('relion_iteration', 6), ('n_classes', 4), ('box_size', 128)])
def test_continuation_refuses_mismatched_iteration_class_or_box(field, value):
    snapshot = dataclasses.replace(_snapshot(), **{field: value})
    options = stand_in.options(schedule=stand_in.schedule(init_relion_iteration=7),
                                checkpoint=CheckpointOptions(resume=snapshot))
    with pytest.raises(ValueError, match='cannot continue from the run files'):
        _initialize(options)


def test_frozen_numbered_restart_refuses_already_converged_state():
    with pytest.raises(ValueError, match='cannot already be converged'):
        _initialize(stand_in.options(), relion_replay=RelionReplay(frozen_refinement_state_fields={'has_converged': True}))


def test_k1_options_refuse_class3d_seed_classes():
    """Seed classes belong to a Class3D one-reference start; K=1 has no classes to seed."""
    seeds = np.zeros(4, dtype=np.int64)
    assert KClassOptions(n_classes=4, first_iteration_seed_classes=seeds).first_iteration_seed_classes is seeds
    assert KClassOptions(n_classes=1).first_iteration_seed_classes is None
    with pytest.raises(ValueError, match="K=1 has no classes to seed"):
        KClassOptions(n_classes=1, first_iteration_seed_classes=seeds)
