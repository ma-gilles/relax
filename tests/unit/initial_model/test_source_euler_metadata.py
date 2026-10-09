"""Exact source pose metadata survives partial InitialModel updates and publication."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from helpers.float_compare import assert_matches
from recovar.utils.helpers import R_from_relion, R_to_relion

from relax.relion import initial_model_io
from relax.vdam import adaptive_estep, native_sampling, particle_update
from relax.vdam.estep_common import SPA_META_PARTICLE_FIELDS
from relax.vdam.state import NativeParticleState

pytestmark = pytest.mark.unit


def state(n=3):
    return NativeParticleState(np.zeros((n, 2)), np.zeros(n, np.int32), np.zeros(n, np.float32))


def test_subset_source_validity_and_mixed_legacy_rows():
    value = state()
    value.best_pose_rotations = np.zeros((3, 3, 3), np.float32)
    eulers = np.array([[159.3271497477632, 126.91279408422895, 85.75518260708287], [-2.0, 30.0, 7.0]])
    particle_update.update_particle_state_from_estep_meta(
        value,
        dict(
            selected_particle_ids=np.array([2, 0]),
            best_pose_eulers_deg=eulers,
            best_pose_eulers_valid=np.array([True, False]),
        ),
        np.zeros((1, 2)),
    )
    assert_matches(value.best_pose_eulers_valid, [False, False, True])
    value.best_pose_rotations[0] = R_from_relion(eulers[1:], degrees=True).astype(np.float32)[0]
    got = native_sampling._best_eulers_from_particle_state(value, np.array([2, 0]), rotation_grid_order=0)
    assert_matches(got[0], eulers[0])
    assert_matches(
        got[1], R_to_relion(value.best_pose_rotations[[0]].astype(np.float64), degrees=True)[0]
    )
    # A matrix-only replacement invalidates that row, never a different particle.
    particle_update.update_particle_state_from_estep_meta(
        value,
        dict(selected_particle_ids=np.array([0]), best_pose_rotations=np.eye(3, dtype=np.float32)[None]),
        np.zeros((1, 2)),
    )
    assert_matches(value.best_pose_eulers_valid, [False, False, True])
    assert_matches(value.best_pose_eulers_deg[2], eulers[0])
    particle_update.update_particle_state_from_estep_meta(
        value,
        dict(selected_particle_ids=np.array([2]), best_pose_rotations=np.eye(3, dtype=np.float32)[None]),
        np.zeros((1, 2)),
    )
    assert not np.any(value.best_pose_eulers_valid)


def test_input_star_source_is_valid_before_first_visit():
    eulers = np.array([[159.3271497477632, 126.91279408422895, 85.75518260708287]])
    frame = pd.DataFrame(
        dict(
            _rlnImageName=["1@stack.mrcs"],
            _rlnAngleRot=eulers[:, 0],
            _rlnAngleTilt=eulers[:, 1],
            _rlnAnglePsi=eulers[:, 2],
        )
    )
    value = initial_model_io._particle_state_from_star(frame, SimpleNamespace(voxel_size=1.0, n_images=1))
    assert not value.visited[0] and value.best_pose_eulers_valid[0]
    assert_matches(native_sampling._best_eulers_from_particle_state(value, [0], rotation_grid_order=0), eulers)
    assert value.best_pose_rotations.dtype == np.float32


def test_subtomogram_input_star_source_is_valid_before_first_visit():
    # RELION's expected accuracy uses an unvisited subtomogram's input angles (ml_optimiser.cpp:9505);
    # without them relax skipped the estimate until every trial particle had been visited.
    eulers = np.array([[-112.22566303733562, 128.19451794974998, 133.69290541708725], [48.46, 98.49, 31.66]])
    frame = pd.DataFrame(
        dict(
            _rlnTomoParticleName=["TS_01/3", "TS_03/3"],
            _rlnOriginXAngst=[2.27, 0.98],
            _rlnOriginYAngst=[0.68, 2.47],
            _rlnOriginZAngst=[0.90, 0.48],
            _rlnAngleRot=eulers[:, 0],
            _rlnAngleTilt=eulers[:, 1],
            _rlnAnglePsi=eulers[:, 2],
        )
    )
    value = initial_model_io._tomo_particle_state_from_star(frame, pixel_size=4.25)
    assert value.best_pose_eulers_valid.all() and value.best_pose_rotations.dtype == np.float32
    assert_matches(native_sampling._best_eulers_from_particle_state(value, [1, 0], rotation_grid_order=0), eulers[[1, 0]])
    assert_matches(value.translation_offsets[0], np.array([2.27, 0.68, 0.90]) / 4.25)


def test_subtomogram_checkpoint_state_reads_class_and_pmax():
    # A diagnostic continuation's data STAR: class 0 marks a particle not yet visited.
    frame = pd.DataFrame(
        dict(
            _rlnTomoParticleName=["TS_01/1", "TS_01/2"],
            _rlnOriginXAngst=[0.0, 4.25], _rlnOriginYAngst=[0.0, 0.0], _rlnOriginZAngst=[0.0, 0.0],
            _rlnAngleRot=[1.0, 2.0], _rlnAngleTilt=[3.0, 4.0], _rlnAnglePsi=[5.0, 6.0],
            _rlnClassNumber=[0, 1], _rlnMaxValueProbDistribution=[0.0, 0.75],
        )
    )
    value = initial_model_io.tomo_checkpoint_particle_state(frame, pixel_size=4.25)
    assert value.visited.tolist() == [False, True] and value.class_assignments.tolist() == [0, 0]
    assert_matches(value.max_posterior, np.array([0.0, 0.75], np.float32))
    assert initial_model_io._tomo_particle_state_from_star(frame, pixel_size=4.25).visited is None


@pytest.mark.parametrize("fault", ["narrow", "shape", "nan", "valid_shape", "valid_dtype"])
def test_invalid_restored_source_metadata_rejected(fault):
    value = state(1)
    value.best_pose_eulers_deg = np.ones((1, 3), np.float64)
    value.best_pose_eulers_valid = np.ones(1, bool)
    if fault == "narrow":
        value.best_pose_eulers_deg = value.best_pose_eulers_deg.astype(np.float32)
    elif fault == "shape":
        value.best_pose_eulers_deg = np.ones((2, 3))
    elif fault == "nan":
        value.best_pose_eulers_deg[0, 0] = np.nan
    elif fault == "valid_shape":
        value.best_pose_eulers_valid = np.ones(2, bool)
    else:
        value.best_pose_eulers_valid = np.ones(1, np.int32)
    with pytest.raises(ValueError, match="Euler"):
        native_sampling._best_eulers_from_particle_state(value, [0], rotation_grid_order=0)


def _engine_result(**fields):
    """An engine result with only ``fields`` set (the E-step meta reads these attributes)."""
    names = ("best_pose_eulers_deg", "uncast_log_evidence_per_image", "stats", "pose_assignments", "class_assignments",
             "best_pose_rotations", "best_pose_translations", "best_pose_rotation_ids", "significant_counts")
    sums = dict(
        class_posterior_sums=np.ones(1), class_mstep_posterior_sums=np.ones(1), noise_stats=None,
        aggregate_noise_stats=None, profile_summary=None, class_assignments=np.zeros(2, dtype=np.int32),
        per_class_stats=(SimpleNamespace(rotation_posterior_sums=np.ones(1)),),
    )
    return SimpleNamespace(**{**dict.fromkeys(names), **sums, **fields})


def test_engine_rows_keep_identity_and_validity():
    source = np.array([[2.0 + 2**-40, 30.0, 4.0], [1.0, 2.0, 3.0]])
    meta = adaptive_estep.sparse_pass2_estep_meta(
        _engine_result(best_pose_eulers_deg=source), np.array([2, 0]), particle_fields=SPA_META_PARTICLE_FIELDS
    )
    assert_matches(meta["selected_particle_ids"], [2, 0])
    assert_matches(meta["best_pose_eulers_valid"], [True, True])
    assert_matches(meta["best_pose_eulers_deg"], source)
    value = state()
    particle_update.update_particle_state_from_estep_meta(value, meta, np.zeros((1, 2)))
    assert_matches(value.best_pose_eulers_valid, [True, False, True])
    assert_matches(value.best_pose_eulers_deg[2], source[0])
    # Without source rows the meta carries none, and the particles keep no valid source pose.
    meta = adaptive_estep.sparse_pass2_estep_meta(
        _engine_result(), np.array([1, 0]), particle_fields=SPA_META_PARTICLE_FIELDS
    )
    assert "best_pose_eulers_deg" not in meta and "best_pose_eulers_valid" not in meta
    assert_matches(meta["selected_particle_ids"], [1, 0])


