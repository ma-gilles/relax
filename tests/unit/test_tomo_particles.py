"""Subtomogram particle layer (S4.1): per-image matrices and shifts, per-particle sums (CPU).

These check the layer's mechanics against RELION's formulas written out per image; the
RELION-pinned one-E-step test on the S1 dataset checks the conventions end to end.
"""

import numpy as np
import pytest
from helpers.float_compare import assert_matches
from scipy.spatial.transform import Rotation

from relax.refinement import tomo_particles

pytestmark = pytest.mark.unit


def _setup(seed=0):
    rng = np.random.default_rng(seed)
    image_particle = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    projections = Rotation.random(image_particle.size, random_state=rng).as_matrix()
    poses = Rotation.random(4, random_state=rng).as_matrix()
    return rng, image_particle, projections, poses


def test_image_matrix_is_projection_times_pose():
    _rng, image_particle, projections, poses = _setup()
    out = tomo_particles.tilt_image_rotations(poses, projections)
    assert out.shape == (image_particle.size, 4, 3, 3)
    for i in range(image_particle.size):
        for h in range(4):
            np.testing.assert_allclose(out[i, h], projections[i] @ poses[h], atol=1e-14)


def test_image_shift_projects_trial_plus_old_offset():
    rng, image_particle, projections, _ = _setup(1)
    shifts = rng.normal(size=(5, 3))
    old = rng.normal(size=(3, 3))
    out = tomo_particles.tilt_image_shifts(shifts, old, projections, image_particle)
    for i, p in enumerate(image_particle):
        for t in range(5):
            s = shifts[t] + old[p]
            A = projections[i]
            expected = [
                A[0, 0] * s[0] + A[0, 1] * s[1] + A[0, 2] * s[2],
                A[1, 0] * s[0] + A[1, 1] * s[1] + A[1, 2] * s[2],
            ]
            np.testing.assert_allclose(out[i, t], expected, atol=1e-14)
    # An untilted image (Aproj = I) sees the x, y components of the 3D shift.
    flat = tomo_particles.tilt_image_shifts(shifts, np.zeros((1, 3)), np.eye(3)[None], np.array([0]))
    np.testing.assert_allclose(flat[0], shifts[:, :2])


def test_particle_scores_sum_images_and_weights_come_back_per_image():
    rng, image_particle, _projections, _poses = _setup(2)
    scores = rng.normal(size=(image_particle.size, 6))
    summed = tomo_particles.particle_scores(scores, image_particle, 3)
    for p in range(3):
        np.testing.assert_allclose(summed[p], scores[image_particle == p].sum(axis=0))
    weights = rng.random((3, 6))
    np.testing.assert_array_equal(tomo_particles.image_weights(weights, image_particle), weights[image_particle])


def test_noise_sums_are_averaged_over_a_particles_images():
    _rng, image_particle, _projections, _poses = _setup()
    scale = tomo_particles.image_noise_scale(image_particle, 3)
    np.testing.assert_allclose(scale, [1 / 3] * 3 + [1 / 2] * 2 + [1 / 4] * 4)
    # Each particle's images carry total weight one, so per-particle sums count it once.
    np.testing.assert_allclose(np.bincount(image_particle, weights=scale), np.ones(3))


def test_projection_matrices_recovered_from_the_flattened_image_matrices():
    _rng, image_particle, projections, poses = _setup(3)
    image_matrices = np.einsum("iab,ibc->iac", projections, poses[image_particle])
    recovered = tomo_particles.tilt_projection_matrices(image_matrices, poses, image_particle)
    np.testing.assert_allclose(recovered, projections, atol=1e-12)


def test_translation_angles_follow_relions_per_image_phase_operand():
    """acc_ml_optimiser_impl.h:1761-1787 and exp_model.cpp:106-114, written out per image and shift."""
    rng, image_particle, projections, _ = _setup(3)
    shifts = rng.normal(scale=2.0, size=(6, 3))
    old = rng.normal(scale=1.5, size=(3, 3))
    size = np.array([128, 128, 128, 100, 100, 128, 128, 128, 128])
    out = tomo_particles.tilt_translation_angles(shifts, old, projections, image_particle, size)
    assert out.shape == (image_particle.size, 6, 2) and out.dtype == np.float32
    expected = np.zeros(out.shape, dtype=np.float32)
    for i, p in enumerate(image_particle):
        a = projections[i]
        for t in range(6):
            x, y, z = shifts[t][0] + old[p][0], shifts[t][1] + old[p][1], shifts[t][2] + old[p][2]
            sx = a[0, 0] * x + a[0, 1] * y + a[0, 2] * z
            sy = a[1, 0] * x + a[1, 1] * y + a[1, 2] * z
            expected[i, t] = (-2 * np.pi * sx / float(size[i]), -2 * np.pi * sy / float(size[i]))
    assert_matches(out, expected)


def test_image_slots_visit_each_particles_images_in_order():
    offsets = np.array([0, 3, 5, 9])  # particles with 3, 2 and 4 images
    row_unit = np.array([0, 0, 1, 2, 2, 1])
    slots = [tomo_particles.image_slot_ids(row_unit, offsets, k) for k in range(4)]
    np.testing.assert_array_equal(slots[0], [0, 0, 3, 5, 5, 3])
    np.testing.assert_array_equal(slots[2], [2, 2, -1, 7, 7, -1])
    np.testing.assert_array_equal(slots[3], [-1, -1, -1, 8, 8, -1])
    for r, u in enumerate(row_unit):
        visited = [s[r] for s in slots if s[r] >= 0]
        assert visited == list(range(offsets[u], offsets[u + 1]))
