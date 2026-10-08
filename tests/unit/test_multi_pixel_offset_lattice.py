"""An optics group whose pixel differs from the model's stays on RELION's Angstrom offset lattice (relax#52).

RELION converts --offset_step to Angstrom with the model pixel once (ml_optimiser.cpp:597, 2593) and divides the
Angstrom grid by each image's own pixel (HealpixSampling::getTranslationsInPixel, healpix_sampling.cpp:1739-1758).
The old offset is read as ``xoff_A / my_pixel_size`` (ml_optimiser.cpp:12357) and rounded to whole pixels in the
image's own pixels (``my_old_offset.selfROUND()``, :7259); the new offset is that rounded offset plus the trial.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.refinement import optics_shapes
from relax.refinement.half_inputs import HalfSet, prepare_particle_pose_update

pytestmark = pytest.mark.unit

MODEL_PIXEL = 1.40
OWN_PIXEL = 1.36  # EMPIAR-10299's Diamond group


def _half(groups):
    """A half of images on the model pixel (group 0) and on OWN_PIXEL (group 1), by image."""

    groups = np.asarray(groups)
    pairs = [
        (SimpleNamespace(image_shape=(128, 128), voxel_size=pixel), np.flatnonzero(groups == g))
        for g, pixel in enumerate((MODEL_PIXEL, OWN_PIXEL))
    ]
    classes = optics_shapes.make_shape_classes(pairs, ref_box=128, ref_pixel=MODEL_PIXEL)
    return optics_shapes.MultiShapeHalf(
        classes, image_shape=(128, 128), volume_shape=(128,) * 3, voxel_size=MODEL_PIXEL
    )


def test_trial_translations_are_one_angstrom_grid_for_every_pixel():
    half = _half([0, 1])
    grid_ref = np.array([[0.0, 0.0], [0.5, -1.0], [-1.5, 2.0]])  # model pixels, --offset_step 1 at oversampling 1
    for shape_class in half.classes:
        out = optics_shapes.class_kwargs(
            {"current_translations": grid_ref, "translation_step": 1.0}, shape_class, half.n_units
        )
        assert_matches(out["current_translations"] * shape_class.pixel_size, grid_ref * MODEL_PIXEL)
        assert_matches(out["translation_step"] * shape_class.pixel_size, MODEL_PIXEL)


def test_old_offset_is_rounded_in_each_images_own_pixels():
    # 2.07 A is 1.479 model pixels (rounds to 1) but 1.522 own pixels (rounds to 2); -0.95 A rounds to -1 in both.
    old_angstrom = np.array([[2.07, -0.95], [2.07, -0.95]])
    groups = [0, 1]
    half = _half(groups)
    trial_angstrom = np.array([[0.70, -0.70], [0.70, -0.70]])  # one point of the shared Angstrom grid
    halves = [
        HalfSet(index=0, dataset=half, translations=old_angstrom / MODEL_PIXEL),
        HalfSet(index=1, dataset=half, translations=old_angstrom / MODEL_PIXEL),
    ]
    scores = SimpleNamespace(
        hard_assignments=[np.zeros(2, dtype=np.int32)] * 2,
        best_pose_rotations=[np.tile(np.eye(3), (2, 1, 1))] * 2,
        best_pose_rotation_eulers=[np.zeros((2, 3))] * 2,
        best_pose_translations=[trial_angstrom / MODEL_PIXEL] * 2,  # merged results are in model pixels
        pose_rotations=[None, None],
    )

    update = prepare_particle_pose_update(
        scores, halves, np.zeros((1, 2)), previous_rotations=[None, None], local_sampling=None, dtype=np.float64
    )

    own_pixels = np.array([MODEL_PIXEL, OWN_PIXEL])[:, None]
    rounded_old = np.where(old_angstrom / own_pixels >= 0, 1, -1) * np.floor(np.abs(old_angstrom / own_pixels) + 0.5)
    expected_angstrom = rounded_old * own_pixels + trial_angstrom
    assert_matches(rounded_old, [[1, -1], [2, -1]])
    for poses in update.current:
        assert_matches(poses.translations_pixels * MODEL_PIXEL, expected_angstrom)


def test_image_translation_factors():
    half = _half([1, 0, 1])
    assert_matches(
        optics_shapes.image_translation_factors(half), [MODEL_PIXEL / OWN_PIXEL, 1.0, MODEL_PIXEL / OWN_PIXEL]
    )
    assert optics_shapes.image_translation_factors(SimpleNamespace(n_units=3)) is None


def test_subtomogram_offsets_keep_their_rounding():
    # A subtomogram half is one grid (a TomoHalf, never a MultiShapeHalf): no factors, and its 3D offsets keep the
    # whole-pixel rounding of RELION's GPU path (tomo_particles.relion_gpu_old_offsets), unchanged by relax#52.
    from relax.refinement.tomo_half import TomoHalf
    from relax.refinement.tomo_particles import relion_gpu_old_offsets
    from relax.relion.relion_metadata import _relion_metadata_translations

    assert not issubclass(TomoHalf, optics_shapes.MultiShapeHalf)
    old = np.array([[1.5, -2.49, 0.51], [-0.5, 3.2, -1.7]])
    shift = np.array([[0.25, -0.5, 1.0], [0.0, 0.75, -0.25]])
    published = _relion_metadata_translations(old, shift, own_pixel_factors=None, dtype=np.float64)
    assert_matches(published, relion_gpu_old_offsets(old) + shift)
