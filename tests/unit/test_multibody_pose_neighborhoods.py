"""The per-body pose comparison against explicitly built point residuals, and its invariances."""


from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from relax.multibody.bodies import RESIDUAL_IDENTITY_EULER
from relax.multibody.pose_neighborhoods import (
    fit_low_rank_motion,
    low_rank_distance,
    motion_summary,
    rotation_distance,
    BodyMoments,
    body_pose_lines,
    compatibility_distance,
    compatibility_distance_and_bound_hits,
    mask_moments,
    pose_features,
    zero_depth_distance,
)
from relax.multibody.subsets import body_pose_features, nested_thresholds, target_row
from relax.multibody.subtraction import BodyPoses
from scripts.lib.pose_reference import direct_free_squared_score_high_precision, direct_squared_score, true_squared_displacement

pytestmark = pytest.mark.unit
# Implementation tolerance on squared scores normalised by the body's mean squared radius: an algebra check, not a
# pose accuracy.
GATE = 1e-10


def moments_of(points, weights):
    weights = weights / weights.sum()
    centroid = weights @ points
    centred = points - centroid
    return BodyMoments(centroid, (centred * weights[:, None]).T @ centred)


def random_body(rng, n=40, offset=1.0):
    points = rng.normal(size=(n, 3)) * rng.uniform(0.2, 3.0, size=3) + offset * rng.normal(size=3)
    return points, rng.uniform(0.0, 1.0, size=n) + 1e-3


def random_pair(rng, angle=1.5):
    rotations = Rotation.from_rotvec(rng.normal(size=(2, 3)) * angle / 2).as_matrix()
    normals = Rotation.random(2, random_state=rng).as_matrix()[:, :, 2]
    return rotations, rng.normal(size=(2, 3)) * 2.0, normals


def score(points, weights, rotations, lifts, normals, bound, *, centroid=True):
    moments = moments_of(points, weights)
    if not centroid:
        moments = BodyMoments(np.zeros(3), moments.covariance)
    return compatibility_distance(pose_features(rotations, lifts, normals, moments), 1, bound)[0] ** 2


def scale_of(points, weights):
    return float((weights / weights.sum()) @ np.sum(points * points, axis=1))


def test_score_matches_explicit_point_residuals_in_random_cases():
    rng = np.random.default_rng(0)
    errors, active = [], {"interior": 0, "edge": 0, "corner": 0}
    for case in range(1200):
        points, weights = random_body(rng, n=int(rng.integers(4, 40)), offset=rng.choice([0.0, 1.0, 5.0]))
        rotations, lifts, normals = random_pair(rng)
        bound = float(rng.choice([0.1, 0.5, 1.0, 3.0, 10.0]))
        reference = direct_squared_score(points, weights, *rotations, *lifts, *normals, bound)
        fast = score(points, weights, rotations, lifts, normals, bound)
        errors.append(abs(fast - reference) / scale_of(points, weights) / max(1.0, reference / scale_of(points, weights)))
        delta = lifts[0] - lifts[1] + (rotations[0] - rotations[1]) @ moments_of(points, weights).centroid
        cross = np.cross(*normals)
        free = -np.array([np.cross(delta, normals[1]) @ cross, np.cross(delta, normals[0]) @ cross]) / (cross @ cross)
        active[["interior", "edge", "corner"][int(np.sum(np.abs(free) > bound))]] += 1
    assert max(errors) <= GATE
    assert min(active.values()) >= 100, active


@pytest.mark.parametrize("angle", [0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_nearly_parallel_views(angle, sign):
    rng = np.random.default_rng(1)
    for _ in range(30):
        points, weights = random_body(rng)
        rotations, lifts, _ = random_pair(rng)
        first = rng.normal(size=3)
        first /= np.linalg.norm(first)
        other = Rotation.from_rotvec(angle * np.cross(first, rng.normal(size=3)) / 1.0).apply(first) if angle else first
        other = sign * other / np.linalg.norm(other)
        normals = np.stack([first, other])
        for bound in (0.3, 2.0, 20.0):
            reference = direct_squared_score(points, weights, *rotations, *lifts, *normals, bound)
            fast = score(points, weights, rotations, lifts, normals, bound)
            scale = scale_of(points, weights)
            assert abs(fast - reference) / scale <= GATE * max(1.0, reference / scale)


def test_collinear_body_rotated_about_its_own_line_does_not_move():
    line = np.array([1.0, 2.0, -0.5]) / np.linalg.norm([1.0, 2.0, -0.5])
    points = np.outer(np.linspace(-3.0, 4.0, 9), line)
    weights = np.ones(9)
    rotations = Rotation.from_rotvec(np.outer([0.0, 1.1], line)).as_matrix()
    normals = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    # The rank-1 covariance's zero eigenvalues are round-off, so C^(1/2) leaves about eps times the squared radius.
    assert score(points, weights, rotations, np.zeros((2, 3)), normals, 1.0) <= 1e-15 * scale_of(points, weights)


@pytest.mark.parametrize("kind", ["planar", "sphere"])
def test_flat_or_round_bodies_still_pay_for_rotation(kind):
    rng = np.random.default_rng(2)
    points = rng.normal(size=(500, 3))
    if kind == "planar":
        points[:, 2] = 0.0
    else:
        points /= np.linalg.norm(points, axis=1, keepdims=True)
    weights = np.ones(len(points))
    rotations = Rotation.from_rotvec([[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]]).as_matrix()
    normals = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    value = score(points, weights, rotations, np.zeros((2, 3)), normals, 1.0)
    assert value > 0.1
    assert abs(value - direct_squared_score(points, weights, *rotations, np.zeros(3), np.zeros(3), *normals, 1.0)) <= GATE * value


def test_invariances():
    rng = np.random.default_rng(3)
    for _ in range(100):
        points, weights = random_body(rng)
        rotations, lifts, normals = random_pair(rng)
        bound = 1.5
        value = score(points, weights, rotations, lifts, normals, bound)
        scale = scale_of(points, weights)
        # Symmetry and self-distance.
        flipped = score(points, weights, rotations[::-1], lifts[::-1], normals[::-1], bound)
        assert abs(value - flipped) <= GATE * scale
        features = pose_features(rotations, lifts, normals, moments_of(points, weights))
        # Round-off of the depth candidates, about eps times the bound.
        assert compatibility_distance(features, 0, bound)[0] <= 1e-14 * bound
        # A common rotation of the molecular frame.
        common = Rotation.random(random_state=rng).as_matrix()
        rotated = score(points, weights, common @ rotations, lifts @ common.T, normals @ common.T, bound)
        assert abs(value - rotated) <= GATE * scale
        # Recentring: x' = x - c with translations t + Q c, as one physical pose.
        pivot = rng.normal(size=3) * 3.0
        moved = lifts + rotations @ pivot
        assert abs(value - score(points - pivot, weights, rotations, moved, normals, bound)) <= GATE * max(scale, value)
        # Units.
        assert abs(100.0 * value - score(10.0 * points, weights, rotations, 10.0 * lifts, normals, 10.0 * bound)) <= GATE * 100.0 * max(scale, value)


def test_moving_a_line_origin_without_its_interval_changes_the_score():
    rng = np.random.default_rng(4)
    points, weights = random_body(rng)
    rotations, lifts, normals = random_pair(rng)
    moved = lifts + np.array([[3.0], [0.0]]) * normals
    free = [score(points, weights, rotations, x, normals, None) for x in (lifts, moved)]
    bounded = [score(points, weights, rotations, x, normals, 0.1) for x in (lifts, moved)]
    assert abs(free[0] - free[1]) <= GATE * scale_of(points, weights)
    assert abs(bounded[0] - bounded[1]) > 1e-3


def views_of(translation, n_views, rng):
    """Zero-depth lifts and depth directions of one physical translation seen from random cameras."""
    cameras = Rotation.random(n_views, random_state=rng).as_matrix()
    normals = cameras[:, 2, :]  # A^T e_z, with A the camera
    return translation - (normals @ translation)[:, None] * normals, normals


def test_one_physical_pose_seen_from_many_views_scores_zero():
    rng = np.random.default_rng(5)
    points, weights = random_body(rng)
    translation = rng.normal(size=3)
    rotation = Rotation.random(random_state=rng).as_matrix()
    lifts, normals = views_of(translation, 200, rng)
    features = pose_features(np.tile(rotation, (200, 1, 1)), lifts, normals, moments_of(points, weights))
    bound = 1.01 * np.linalg.norm(translation)
    scale = scale_of(points, weights)
    assert np.max(compatibility_distance(features, 0, bound) ** 2) <= 1e-24 * scale
    # Negative control: a zero-filled Euclidean comparison reports motion that never happened.
    assert np.max(zero_depth_distance(features, 0) ** 2) > 0.1 * translation @ translation


def test_centroid_term_is_required_off_centre():
    rng = np.random.default_rng(6)
    points, weights = random_body(rng, offset=5.0)
    rotations, lifts, normals = random_pair(rng)
    reference = direct_squared_score(points, weights, *rotations, *lifts, *normals, 2.0)
    assert abs(score(points, weights, rotations, lifts, normals, 2.0) - reference) <= GATE * max(reference, 1.0)
    # Negative control: dropping (Q_i - Q_j) m no longer reproduces the point residuals.
    assert abs(score(points, weights, rotations, lifts, normals, 2.0, centroid=False) - reference) > 1e-3 * reference


def test_noiseless_scores_are_ordered_and_bounds_are_monotone():
    rng = np.random.default_rng(7)
    for _ in range(300):
        points, weights = random_body(rng)
        rotations = Rotation.random(2, random_state=rng).as_matrix()
        translations = rng.normal(size=(2, 3))
        normals = Rotation.random(2, random_state=rng).as_matrix()[:, 2, :]
        depths = np.sum(translations * normals, axis=1)
        lifts = translations - depths[:, None] * normals
        true = true_squared_displacement(points, weights, *rotations, *translations)
        feasible = 1.0001 * np.max(np.abs(depths))
        free = score(points, weights, rotations, lifts, normals, None)
        bounded = [score(points, weights, rotations, lifts, normals, feasible * k) for k in (1.0, 1.5, 3.0)]
        slack = GATE * scale_of(points, weights)
        assert free <= bounded[2] + slack and bounded[2] <= bounded[1] + slack and bounded[1] <= bounded[0] + slack
        assert bounded[0] <= true + slack


def test_bound_hits_report_an_active_depth_bound():
    normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    lifts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
    features = pose_features(np.tile(np.eye(3), (2, 1, 1)), lifts, normals, BodyMoments(np.zeros(3), np.eye(3)))
    assert compatibility_distance_and_bound_hits(features, 1, 1.0)[1].tolist() == [True, False]
    assert compatibility_distance_and_bound_hits(features, 1, 3.0)[1].tolist() == [False, False]
    assert compatibility_distance(features, 1, 1.0)[0] == pytest.approx(3.0)


def test_mask_moments_use_voxel_axes_and_angstroms():
    mask = np.zeros((8, 8, 8))
    mask[4, 5, 7] = 3.0  # z = 0, y = 1, x = 3
    mask[4, 5, 3] = 1.0  # x = -1
    moments = mask_moments(mask, [1.0, 0.0, 0.0], 2.0)
    np.testing.assert_allclose(moments.centroid, [2.0, 2.0, 0.0])
    x_variance = (0.75 * 2.0**2 + 0.25 * 6.0**2)
    np.testing.assert_allclose(moments.covariance, np.diag([x_variance, 0.0, 0.0]))


def test_pose_lines_follow_the_stated_lift():
    camera = Rotation.from_euler("zyz", [0.3, 0.7, -0.2]).as_matrix()[None]
    body = Rotation.from_rotvec([0.1, 0.0, 0.2]).as_matrix()[None]
    rotations, lifts, normals = body_pose_lines(camera, camera @ body, [0.0, 0.0, 0.0], [[2.25, -1.5]], [[0.5, 0.25]], 2.0)
    np.testing.assert_allclose(rotations[0], body[0])
    # e = (0.25, 0.5) after RELION's rounding of (2.25, -1.5) to (2, -2); s = 2 (e - f) = (-0.5, 0.5).
    np.testing.assert_allclose(lifts[0], camera[0].T @ [-0.5, 0.5, 0.0])
    np.testing.assert_allclose(normals[0], camera[0].T @ [0.0, 0.0, 1.0])


@pytest.mark.parametrize("angle", [1e-12, 1e-9, 1e-6, 1e-3])
def test_free_depth_score_of_nearly_parallel_views(angle):
    rng = np.random.default_rng(8)
    points, weights = random_body(rng, n=12)
    rotations, lifts, _ = random_pair(rng)
    normals = np.array([[0.0, 0.0, 1.0], [np.sin(angle), 0.0, np.cos(angle)]])
    reference = direct_free_squared_score_high_precision(points, weights, *rotations, *lifts, *normals)
    assert abs(score(points, weights, rotations, lifts, normals, None) - reference) <= GATE * max(reference, scale_of(points, weights))


def two_body_run(rng, n=200):
    mask = np.zeros((16, 16, 16))
    mask[4:9, 5:12, 6:10] = rng.uniform(0.2, 1.0, size=(5, 7, 4))
    other = np.zeros((16, 16, 16))
    other[10:14, 3:8, 2:6] = 1.0
    bodies = [
        SimpleNamespace(mask=b, com=np.array(c, dtype=float), orientation=Rotation.random(random_state=rng).as_matrix())
        for b, c in ((mask, (0.0, 0.0, -2.0)), (other, (-3.0, -3.0, 4.0)))
    ]
    particles = pd.DataFrame(rng.uniform(0.0, 180.0, size=(n, 3)), columns=["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"])
    particles[["rlnOriginXAngst", "rlnOriginYAngst"]] = rng.uniform(-6.0, 6.0, size=(n, 2))
    run = SimpleNamespace(bodies=bodies, pixel_size=1.5, checkpoint=SimpleNamespace(particles=particles))
    poses = [BodyPoses(RESIDUAL_IDENTITY_EULER + rng.normal(size=(n, 3)) * 4.0, rng.normal(size=(n, 2))) for _ in bodies]
    return run, SimpleNamespace(poses=poses, alignment_offsets_px=None)


def test_changing_another_body_changes_nothing_of_this_bodys_subsets():
    rng = np.random.default_rng(9)
    run, result = two_body_run(rng)
    changed = SimpleNamespace(alignment_offsets_px=None, poses=[result.poses[0], BodyPoses(
        result.poses[1].eulers_deg + rng.normal(size=(200, 3)) * 20.0, result.poses[1].offsets_px + rng.normal(size=(200, 2)) * 5.0
    )])
    halves = [np.arange(0, 200, 2), np.arange(1, 200, 2)]
    distances = [compatibility_distance(body_pose_features(run, r, 0), 17, 6.0) for r in (result, changed)]
    thresholds = [nested_thresholds(d, n_candidates=10, n_min_per_half=5, halves=halves) for d in distances]
    np.testing.assert_array_equal(distances[0], distances[1])
    np.testing.assert_array_equal(thresholds[0], thresholds[1])
    np.testing.assert_array_equal(distances[0] <= thresholds[0][3], distances[1] <= thresholds[1][3])
    # The other body's own vector does move.
    other = [compatibility_distance(body_pose_features(run, r, 1), 17, 6.0) for r in (result, changed)]
    assert np.max(np.abs(other[0] - other[1])) > 1.0


def test_target_row_refuses_an_ambiguous_name():
    assert target_row(np.array(["a", "b"]), "b") == 1
    with pytest.raises(ValueError):
        target_row(np.array(["a", "a"]), "a")


def _line_motion(rng, n=2000, rotation_deg=10.0, translation=8.0):
    """A body moving along one known line of rotation and translation, each particle seen from a random view."""
    level = rng.uniform(-1.0, 1.0, n)
    axis = np.array([0.3, 0.9, 0.3]) / np.linalg.norm([0.3, 0.9, 0.3])
    rotations = Rotation.from_rotvec(np.outer(level * np.deg2rad(rotation_deg), axis)).as_matrix()
    displacement = np.outer(level * translation, [1.0, 0.0, 0.0])
    normals = Rotation.random(n, random_state=rng).as_matrix()[:, :, 2]
    lifts = displacement - np.sum(displacement * normals, axis=1)[:, None] * normals
    return level, rotations, lifts, normals


MOMENTS = BodyMoments(np.array([3.0, -2.0, 1.0]), np.diag([900.0, 400.0, 100.0]))


def test_low_rank_motion_recovers_a_pure_translation_from_depth_lines():
    rng = np.random.default_rng(0)
    level, rotations, lifts, normals = _line_motion(rng, rotation_deg=0.0)
    motion = fit_low_rank_motion(pose_features(rotations, lifts, normals, MOMENTS), 1)
    assert abs(np.corrcoef(motion.scores[:, 0], level)[0, 1]) > 0.999
    assert motion.objective[-1] < 1e-3 * motion.objective[0] + 1e-8


def test_low_rank_motion_fit_never_gets_worse():
    rng = np.random.default_rng(1)
    level, rotations, lifts, normals = _line_motion(rng)
    lifts = lifts + rng.normal(size=lifts.shape) * 2.0
    lifts = lifts - np.sum(lifts * normals, axis=1)[:, None] * normals
    motion = fit_low_rank_motion(pose_features(rotations, lifts, normals, MOMENTS), 2)
    assert np.all(np.diff(motion.objective) <= 1e-9 * motion.objective[0])
    np.testing.assert_allclose(motion.components.T @ motion.components, np.eye(2), atol=1e-10)
    np.testing.assert_allclose(motion.scores.mean(axis=0), 0.0, atol=1e-8)


def test_low_rank_score_of_one_physical_pose_does_not_depend_on_the_view():
    rng = np.random.default_rng(2)
    level, rotations, lifts, normals = _line_motion(rng)
    level[:50] = 0.4
    tied = Rotation.from_rotvec(np.outer(level[:50] * np.deg2rad(10.0), np.array([0.3, 0.9, 0.3]) / np.linalg.norm([0.3, 0.9, 0.3])))
    rotations[:50] = tied.as_matrix()
    displacement = np.outer(level[:50] * 8.0, [1.0, 0.0, 0.0])
    lifts[:50] = displacement - np.sum(displacement * normals[:50], axis=1)[:, None] * normals[:50]
    motion = fit_low_rank_motion(pose_features(rotations, lifts, normals, MOMENTS), 1)
    assert np.ptp(motion.scores[:50, 0]) < 1e-3 * np.ptp(motion.scores[:, 0])


def test_low_rank_motion_follows_the_particles_when_they_are_reordered():
    rng = np.random.default_rng(3)
    level, rotations, lifts, normals = _line_motion(rng, n=500)
    order = rng.permutation(500)
    first = fit_low_rank_motion(pose_features(rotations, lifts, normals, MOMENTS), 1)
    second = fit_low_rank_motion(pose_features(rotations[order], lifts[order], normals[order], MOMENTS), 1)
    np.testing.assert_allclose(low_rank_distance(first, 7)[order], low_rank_distance(second, int(np.argsort(order)[7])), atol=1e-6)


def test_motion_summary_splits_rotation_from_translation_and_reads_the_noise_back():
    rng = np.random.default_rng(4)
    level, rotations, lifts, normals = _line_motion(rng, n=4000, rotation_deg=0.0)
    noise = rng.normal(size=lifts.shape) * 1.5
    lifts = lifts + noise - np.sum(noise * normals, axis=1)[:, None] * normals
    features = pose_features(rotations, lifts, normals, MOMENTS)
    summary = motion_summary(features, fit_low_rank_motion(features, 1), MOMENTS.covariance)
    assert summary["rotation_share_of_motion"] < 1e-6
    assert summary["rotation_noise_deg_per_axis"] < 1e-6
    np.testing.assert_allclose(summary["translation_noise_angstrom_per_direction"], 1.5, rtol=0.05)


def test_motion_summary_reads_rotation_noise_back_per_axis():
    rng = np.random.default_rng(5)
    level, rotations, lifts, normals = _line_motion(rng, n=4000, translation=0.0)
    rotations = Rotation.from_rotvec(np.deg2rad(2.0) * rng.normal(size=(4000, 3))).as_matrix() @ rotations
    features = pose_features(rotations, lifts, normals, BodyMoments(np.zeros(3), 900.0 * np.eye(3)))
    summary = motion_summary(features, fit_low_rank_motion(features, 1), 900.0 * np.eye(3))
    np.testing.assert_allclose(summary["rotation_noise_deg_per_axis"], 2.0, rtol=0.1)


def test_rotation_distance_is_the_score_with_translation_removed():
    rng = np.random.default_rng(6)
    level, rotations, lifts, normals = _line_motion(rng, n=200)
    moved = pose_features(rotations, lifts, normals, MOMENTS)
    still = pose_features(rotations, np.zeros_like(lifts) - rotations @ MOMENTS.centroid, normals, MOMENTS)
    np.testing.assert_allclose(rotation_distance(moved, 3), compatibility_distance(still, 3, 0.0), atol=1e-10)


def test_rotation_distance_cannot_see_a_pure_translation():
    rng = np.random.default_rng(7)
    level, rotations, lifts, normals = _line_motion(rng, n=200, rotation_deg=0.0)
    np.testing.assert_allclose(rotation_distance(pose_features(rotations, lifts, normals, MOMENTS), 3), 0.0, atol=1e-12)
