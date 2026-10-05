"""RELION --skip_align: the particles' stored poses as pass-2 grids, and the option's refusals."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from relax.classification.given_poses import given_pose_grids
from relax.refinement import command_options
from relax.relion import input_poses
from relax.sampling import rotation_grid_size

pytestmark = pytest.mark.unit


def _rotations(n, seed=0):
    from scipy.spatial.transform import Rotation

    return Rotation.random(n, random_state=seed).as_matrix().astype(np.float32)


@pytest.mark.parametrize("n", [1, 7, 576, 577])
def test_each_image_gets_its_own_rotation_and_translation(n):
    rotations = _rotations(n)
    translations = np.random.default_rng(1).uniform(-0.5, 0.5, size=(n, 2)).astype(np.float32)
    grids = given_pose_grids(rotations, translations)
    grid_size = rotation_grid_size(grids.healpix_order)
    # The smallest HEALPix grid that holds the list (576 rotations at order 0, 4608 at order 1).
    assert grid_size >= n and (grids.healpix_order == 0 or rotation_grid_size(grids.healpix_order - 1) < n)
    assert grids.rotations.shape == (grid_size, 3, 3)
    assert np.array_equal(grids.rotations[:n], rotations)
    assert np.array_equal(grids.rotations[n:], np.broadcast_to(np.eye(3, dtype=np.float32), (grid_size - n, 3, 3)))
    # One zero translation; each particle's own goes to pass 2 as its image translation.
    assert np.array_equal(grids.translations, np.zeros((1, 2), np.float32))
    assert np.array_equal(grids.image_translations, translations)
    assert np.array_equal(grids.rotation_parent_map, np.arange(grid_size))
    assert np.array_equal(grids.translation_parent_map, [0])
    samples = np.concatenate(grids.supports)
    assert len(grids.supports) == n and samples.dtype == np.int32
    # sample id = rotation * n_translations + translation, as pass 1 writes it, with one translation.
    assert np.array_equal(samples, np.arange(n))


def test_malformed_poses_are_refused():
    with pytest.raises(ValueError, match="given poses are"):
        given_pose_grids(np.zeros((3, 3)), np.zeros((1, 2)))


def _args(**overrides):
    values = dict(
        skip_align=True, n_classes=2, firstiter_cc=False, continue_optimiser_star=None,
        initial_pose_source="auto", relion_init_dir=None, init_previous_best_poses_npz=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_skip_align_refusals_name_the_reason():
    command_options.validate_skip_align_args(_args())
    command_options.validate_skip_align_args(_args(skip_align=False, n_classes=1, firstiter_cc=True))
    for overrides, reason in (
        (dict(n_classes=1), "needs --n_classes > 1"),
        (dict(firstiter_cc=True), "pass --no-firstiter_cc"),
        (dict(continue_optimiser_star="run_it003_optimiser.star"), "--continue is not implemented"),
        (dict(initial_pose_source="none"), "input STAR's"),
        (dict(relion_init_dir="/x"), "input STAR's"),
    ):
        with pytest.raises(SystemExit, match=reason):
            command_options.validate_skip_align_args(_args(**overrides))


def test_class3d_pose_seed_keeps_the_input_orientations_under_skip_align():
    rng = np.random.default_rng(3)
    n = 6
    particles = pd.DataFrame(
        {
            "rlnAngleRot": rng.uniform(-180, 180, n),
            "rlnAngleTilt": rng.uniform(0, 180, n),
            "rlnOriginXAngst": rng.uniform(-9, 9, n),
            "rlnOriginYAngst": rng.uniform(-9, 9, n),
        }
    )
    rows = rng.permutation(n)
    searched = input_poses._load_input_star_class3d_translations(particles, rows, voxel_size=2.0)
    assert searched["previous_best_rotation_eulers"] == [None, None]
    given = input_poses._load_input_star_class3d_translations(particles, rows, voxel_size=2.0, with_orientations=True)
    eulers, empty = given["previous_best_rotation_eulers"]
    assert empty.shape == (0, 3) and eulers.dtype == np.float32
    # An absent angle label reads as zero, as relion_refine reads it.
    expected = np.stack([particles["rlnAngleRot"], particles["rlnAngleTilt"], np.zeros(n)], axis=1)[rows]
    assert np.array_equal(eulers, expected.astype(np.float32))
    assert np.array_equal(given["previous_best_translations"][0], searched["previous_best_translations"][0])


def test_given_supports_carry_the_csr_that_limits_the_projection_cache():
    from relax.classification.k_class import _given_support_csr
    from relax.sparse_pass2.resident_significance import significant_coarse_parents

    n = 7
    grids = given_pose_grids(_rotations(n), np.zeros((n, 2)))
    n_rot = grids.rotations.shape[0]
    # A seed iteration leaves the images of the other class with no sample.
    supports = [grids.supports[i] if i % 2 else np.zeros(0, np.int32) for i in range(n)]
    support = _given_support_csr(supports, n_coarse_rot=n_rot, n_coarse_trans=1)
    assert [np.asarray(row).tolist() for row in support] == [np.asarray(s).tolist() for s in supports]
    parents = significant_coarse_parents(support, n_images=n, n_coarse_rot=n_rot, n_coarse_trans=1)
    # Only the images' own rotations (and parent 0, which an empty support takes), not the whole grid.
    assert parents.tolist() == [0, 1, 3, 5]
