"""RELION --skip_align: the particles' stored poses as pass-2 grids, and the option's refusals."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from helpers.float_compare import assert_matches

from relax.commands import refinement_arguments
from relax.refinement.given_poses import given_pose_grids
from relax.sampling.grids import rotation_grid_size
from relax.star import input_poses

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
    assert_matches(grids.rotations[:n], rotations)
    assert_matches(grids.rotations[n:], np.broadcast_to(np.eye(3, dtype=np.float32), (grid_size - n, 3, 3)))
    # One zero translation; each particle's own goes to pass 2 as its image translation.
    assert_matches(grids.translations, np.zeros((1, 2), np.float32))
    assert_matches(grids.image_translations, translations)
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
    refinement_arguments.validate_skip_align_args(_args())
    # The cross-correlation first iteration from one reference is RELION's GUI default with --skip_align.
    refinement_arguments.validate_skip_align_args(_args(firstiter_cc=True))
    refinement_arguments.validate_skip_align_args(_args(skip_align=False, n_classes=1, firstiter_cc=True))
    # From one reference per class the cross-correlation iteration scores every class (relax#78).
    refinement_arguments.validate_skip_align_args(_args(firstiter_cc=True, ref_star="references.star"))
    refinement_arguments.validate_skip_align_args(_args(firstiter_cc=True, init_class_volumes="a.mrc,b.mrc"))
    for overrides, reason in (
        (dict(n_classes=1), "needs --n_classes > 1"),
        (dict(continue_optimiser_star="run_it003_optimiser.star"), "--continue is not implemented"),
        (dict(initial_pose_source="none"), "input STAR's"),
        (dict(relion_init_dir="/x"), "input STAR's"),
    ):
        with pytest.raises(SystemExit, match=reason):
            refinement_arguments.validate_skip_align_args(_args(**overrides))


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
    assert_matches(eulers, expected.astype(np.float32))
    assert_matches(given["previous_best_translations"][0], searched["previous_best_translations"][0])


def test_given_supports_carry_the_csr_that_limits_the_projection_cache():
    from relax.classification.given_pose_classes import _given_support_csr
    from relax.fine_pass.resident_significance import significant_coarse_parents

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


def test_cross_correlation_class_at_given_poses_is_the_best_over_the_references():
    from relax.classification import given_pose_classes

    # Float64 values that differ below float32 resolution are RELION's float32 ties: the first class wins.
    cc = np.array([[0.25, 0.5, 0.3, 0.1 + 1e-12], [0.5, 0.5, 0.3 + 1e-10, 0.1], [0.1, 0.2, 0.9, 0.1]])
    assert given_pose_classes.classes_of_best_given_pose_cc(cc.astype(np.float32)).tolist() == [1, 0, 2, 0]

    scored = []

    def score_class(assignments, evidence):
        (class_index,) = np.unique(assignments)
        scored.append(int(class_index))
        # The pass's evidence: the CC less the image's own offset.
        evidence[class_index] = cc[class_index].astype(np.float32).astype(np.float64) - 2.0**21

    winners = given_pose_classes.given_pose_cc_classes(score_class, 3, 4)
    assert scored == [0, 1, 2] and winners.dtype == np.int32 and winners.tolist() == [1, 0, 2, 0]
    with pytest.raises(RuntimeError, match="not finite"):
        given_pose_classes.given_pose_cc_classes(lambda assignments, evidence: None, 3, 4)
