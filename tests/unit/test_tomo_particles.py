"""Subtomogram particle layer (S4.1): per-image matrices and shifts, per-particle sums (CPU).

These check the layer's mechanics against RELION's formulas written out per image; the
RELION-pinned one-E-step test on the S1 dataset checks the conventions end to end.
"""

import numpy as np
import pytest
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
