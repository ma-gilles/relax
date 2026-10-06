"""RELION --sigma_ang: local angular searches from iteration 1, their width and their centres."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import starfile
from helpers.float_compare import assert_matches

from relax.helpers.convergence import RefinementState, healpix_angular_step, refine_angular_sampling
from relax.helpers.resolution import ImageGeometry
from relax.refinement.command_options import validate_sigma_ang
from relax.refinement.half_inputs import HalfSet
from relax.refinement.iteration_planning import initialize_refinement_state
from relax.refinement.local_sampling import local_search_centre_half
from relax.refinement.refinement_options import (
    KClassOptions,
    LocalSearchOptions,
    RefinementOptions,
    RefinementSchedule,
)
from relax.relion.input_poses import _load_input_star_previous_best_poses, prepare_initial_poses

pytestmark = pytest.mark.unit


def test_sigma_ang_must_be_positive_for_refine3d_and_class3d():
    for n_classes in (1, 2):
        validate_sigma_ang(SimpleNamespace(n_classes=n_classes, sigma_ang=None))
        validate_sigma_ang(SimpleNamespace(n_classes=n_classes, sigma_ang=2.0))
        with pytest.raises(SystemExit, match="must be positive"):
            validate_sigma_ang(SimpleNamespace(n_classes=n_classes, sigma_ang=0.0))


@pytest.mark.parametrize("healpix_order", [2, 4])
def test_sigma_ang_turns_the_prior_on_at_iteration_one_with_its_width(healpix_order):
    options = RefinementOptions(
        schedule=RefinementSchedule(init_healpix_order=healpix_order),
        local_search=LocalSearchOptions(sigma_ang_deg=2.0),
    )
    state = initialize_refinement_state(
        options, ImageGeometry(image_shape=(64, 64), pixel_size_angstrom=2.0), subtomogram=False, dtype=np.float32
    )
    assert state.do_local_search
    assert_matches(state.sigma_rot, np.deg2rad(2.0))
    assert_matches(state.sigma_psi, np.deg2rad(2.0))


def test_without_sigma_ang_only_the_local_order_starts_local():
    for order, local in ((3, False), (4, True)):
        options = RefinementOptions(schedule=RefinementSchedule(init_healpix_order=order), k_class=KClassOptions())
        state = initialize_refinement_state(
            options, ImageGeometry(image_shape=(64, 64), pixel_size_angstrom=2.0), subtomogram=False, dtype=np.float32
        )
        assert state.do_local_search is local
        assert state.sigma_rot == 0.0


def test_sampling_refinement_keeps_the_prior_below_the_local_order_and_resets_it_there():
    sigma = float(np.deg2rad(2.0))
    state = RefinementState(
        healpix_order=2, adaptive_oversampling=1, do_local_search=True, sigma_rot=sigma, sigma_psi=sigma
    )
    below = refine_angular_sampling(state)
    assert below.healpix_order == 3 and below.do_local_search
    assert_matches(below.sigma_rot, sigma)
    at = refine_angular_sampling(below)
    assert at.healpix_order == 4 and at.do_local_search
    assert_matches(at.sigma_rot, 2.0 * np.deg2rad(healpix_angular_step(4) / 2))
    assert_matches(at.sigma_psi, at.sigma_rot)


def test_global_sampling_refinement_below_the_local_order_is_unchanged():
    new_state = refine_angular_sampling(RefinementState(healpix_order=2, adaptive_oversampling=1))
    assert not new_state.do_local_search
    assert new_state.sigma_rot == 0.0


def _half(eulers):
    return HalfSet(index=0, dataset=None, rotation_eulers=np.asarray(eulers, dtype=np.float32), translations=None)


def test_star_priors_centre_the_search_angle_by_angle_below_the_local_order():
    half = _half([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    priors = np.array([[1.0, np.nan, 3.0], [np.nan, np.nan, np.nan]])
    below = RefinementState(healpix_order=2, do_local_search=True, sigma_rot=0.03)
    centred = local_search_centre_half(half, priors, below)
    assert_matches(centred.rotation_eulers, np.array([[1.0, 20.0, 3.0], [40.0, 50.0, 60.0]], dtype=np.float32))
    assert centred.rotation_eulers.dtype == np.float32
    assert local_search_centre_half(half, priors, RefinementState(healpix_order=4, do_local_search=True)) is half
    assert local_search_centre_half(half, priors, RefinementState(healpix_order=2)) is half
    assert local_search_centre_half(half, None, below) is half


def test_input_star_priors_follow_half_order_and_999_means_absent():
    particles = pd.DataFrame(
        {
            "rlnImageName": ["1@s.mrcs", "2@s.mrcs", "3@s.mrcs"],
            "rlnAngleRot": [1.0, 2.0, 3.0],
            "rlnAngleTilt": [4.0, 5.0, 6.0],
            "rlnAnglePsi": [7.0, 8.0, 9.0],
            "rlnAngleTiltPrior": [90.0, 999.0, 80.0],
        }
    )
    halfset = pd.DataFrame({"rlnImageName": ["1@s.mrcs", "2@s.mrcs", "3@s.mrcs"], "rlnRandomSubset": [2, 1, 1]})
    poses = _load_input_star_previous_best_poses(particles, halfset, np.array([2, 1]), np.array([0]), voxel_size=1.0)
    half1, half2 = poses["angle_priors"]
    assert np.isnan(half1[:, [0, 2]]).all() and np.isnan(half2[:, [0, 2]]).all()
    assert_matches(half1[0, 1], 80.0)
    assert np.isnan(half1[1, 1])
    assert_matches(half2[0, 1], 90.0)
    no_priors = _load_input_star_previous_best_poses(
        particles.drop(columns="rlnAngleTiltPrior"), halfset, np.array([2, 1]), np.array([0]), voxel_size=1.0
    )
    assert no_priors["angle_priors"] is None


def _prepare(tmp_path, particles, *, requested_source, caplog):
    starfile.write(particles, tmp_path / "particles.star")
    halfset = pd.DataFrame({"rlnImageName": list(particles["rlnImageName"]), "rlnRandomSubset": [1, 2, 1]})
    with caplog.at_level(logging.WARNING):
        return prepare_initial_poses(
            particles,
            particle_layout=SimpleNamespace(half1_rows=np.array([0, 2]), half2_rows=np.array([1])),
            relion_halfset_particles=halfset,
            pixel_size_angstrom=1.0,
            data_dir=tmp_path,
            requested_source=requested_source,
            n_classes=1,
            init_relion_iteration=0,
            has_relion_half_sets=True,
            diagnostic_single_half=False,
            frozen_boundary=None,
            poses_npz_path=None,
            pose_iteration="last",
            has_replay_pose_source=False,
            class3d_translations=None,
            class3d_translation_path=None,
            local_search_at_start=True,
            log=logging.getLogger("test.sigma_ang"),
        )


def test_local_start_without_a_pose_source_is_centred_at_zero_with_a_warning(tmp_path, caplog):
    particles = pd.DataFrame({"rlnImageName": ["1@s.mrcs", "2@s.mrcs", "3@s.mrcs"], "rlnAngleRot": [5.0, 6.0, 7.0]})
    prepared = _prepare(tmp_path, particles, requested_source="none", caplog=caplog)
    assert prepared.provenance.resolved_source == "zero_angles"
    eulers = prepared.poses["previous_best_rotation_eulers"]
    origins = prepared.poses["previous_best_translations"]
    assert [e.shape for e in eulers] == [(2, 3), (1, 3)] and not any(np.any(e) for e in eulers)
    assert [t.shape for t in origins] == [(2, 2), (1, 2)] and not any(np.any(t) for t in origins)
    assert "centred at Euler angles (0, 0, 0)" in caplog.text


def test_local_start_from_a_star_without_angles_warns_and_uses_zero_angles(tmp_path, caplog):
    particles = pd.DataFrame({"rlnImageName": ["1@s.mrcs", "2@s.mrcs", "3@s.mrcs"], "rlnAngleRot": [5.0, 6.0, 7.0]})
    prepared = _prepare(tmp_path, particles, requested_source="input-star", caplog=caplog)
    assert prepared.provenance.resolved_source == "input_star"
    assert "lacks rlnAngleTilt, rlnAnglePsi" in caplog.text
    assert_matches(prepared.poses["previous_best_rotation_eulers"][0][:, 0], np.array([5.0, 7.0], dtype=np.float32))
    assert not np.any(prepared.poses["previous_best_rotation_eulers"][0][:, 1:])
