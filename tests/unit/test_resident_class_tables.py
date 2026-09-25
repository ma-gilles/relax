"""K-class resident candidate tables: one image segment spans all classes.

``merge_class_tables`` joins K single-class tables (each class's own significant
support, rotation prior and bitsets) into RELION's class-major hidden space
(ml_optimiser.cpp:8406, :8450): rows image-major, then class-major. Each image's
expanded mask and priors must be exactly the concatenation over classes of the
single-class ones, whatever mask regime (full, dense complement, sparse, empty)
each class had for that image.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_resident_significance import _encoded_supports, _supports

from relax.scoring.sparse_bucket_arrays import _prepare_per_image_pass2_inputs
from relax.sparse_pass2.resident_candidates import (
    build_resident_candidate_tables,
    expand_mask_rows,
    materialize_chunk,
    merge_class_tables,
    plan_capacity_chunks,
)

pytestmark = pytest.mark.unit

N_COARSE_ROT = 18
N_COARSE_TRANS = 5
N_FINE_TRANS = 10
FINE_TRANS_PARENT = np.repeat(np.arange(N_COARSE_TRANS, dtype=np.int32), 2)
N_IMAGES = 9


def _class_tables(seed, n_classes):
    n_samples = N_COARSE_ROT * N_COARSE_TRANS
    fine_parent = np.repeat(np.arange(N_COARSE_ROT, dtype=np.int64), 2)
    fine_rotations = np.stack([np.eye(3, dtype=np.float32)] * fine_parent.size)
    tables = []
    for class_index in range(n_classes):
        # Different seeds give each class its own regime per image (the helper
        # puts empty, dense, full and threshold supports on fixed images; the
        # permutation moves them between images from class to class).
        supports = _supports(N_IMAGES, n_samples, seed=seed + class_index)
        order = np.random.default_rng(seed + 100 * class_index).permutation(N_IMAGES)
        supports = [supports[i] for i in order]
        inputs = _prepare_per_image_pass2_inputs(
            _encoded_supports(supports, n_samples),
            n_coarse_rot=N_COARSE_ROT,
            n_coarse_trans=N_COARSE_TRANS,
            nside_level=0,
            oversampling_order=1,
            n_fine_trans=N_FINE_TRANS,
            fine_translation_parent=FINE_TRANS_PARENT,
            rotation_log_prior=np.linspace(-1.0, 1.0, N_COARSE_ROT, dtype=np.float32) + class_index,
            random_perturbation=0.0,
            fine_rotations_override=fine_rotations,
            fine_rotation_parent_override=fine_parent,
            dtype=np.float32,
        )
        tables.append(
            build_resident_candidate_tables(
                inputs,
                n_coarse_trans=N_COARSE_TRANS,
                n_fine_trans=N_FINE_TRANS,
                fine_translation_parent=FINE_TRANS_PARENT,
            )
        )
    return tables


@pytest.mark.parametrize("n_classes", [2, 4])
def test_merged_image_segments_are_the_class_concatenation(n_classes):
    class_tables = _class_tables(seed=21, n_classes=n_classes)
    merged = merge_class_tables(class_tables)
    assert merged.n_classes == n_classes and merged.n_rows == sum(t.n_rows for t in class_tables)
    for image in range(N_IMAGES):
        rows = slice(int(merged.row_offsets[image]), int(merged.row_offsets[image + 1]))
        expected_mask = np.concatenate([expand_mask_rows(t, image, FINE_TRANS_PARENT) for t in class_tables])
        np.testing.assert_array_equal(expand_mask_rows(merged, image, FINE_TRANS_PARENT), expected_mask)
        for name in ("row_fine_rot", "row_log_prior"):
            expected = np.concatenate(
                [getattr(t, name)[t.row_offsets[image] : t.row_offsets[image + 1]] for t in class_tables]
            )
            np.testing.assert_array_equal(getattr(merged, name)[rows], expected, err_msg=name)
        expected_class = np.concatenate(
            [np.full(int(t.row_offsets[image + 1] - t.row_offsets[image]), k) for k, t in enumerate(class_tables)]
        )
        np.testing.assert_array_equal(merged.row_class[rows], expected_class)


def test_one_class_merge_keeps_the_single_class_masks():
    (single,) = _class_tables(seed=5, n_classes=1)
    merged = merge_class_tables([single])
    for image in range(N_IMAGES):
        np.testing.assert_array_equal(
            expand_mask_rows(merged, image, FINE_TRANS_PARENT),
            expand_mask_rows(single, image, FINE_TRANS_PARENT),
        )


def test_chunks_carry_the_row_class():
    merged = merge_class_tables(_class_tables(seed=8, n_classes=3))
    for chunk in plan_capacity_chunks(merged, row_capacity_ladder=(64, 256, 1024), image_capacity_ladder=(2, 4)):
        materialized = materialize_chunk(merged, chunk)
        np.testing.assert_array_equal(
            materialized["row_class"][: chunk.n_valid_rows], merged.row_class[chunk.row_start : chunk.row_stop]
        )
        assert not materialized["row_class"][chunk.n_valid_rows :].any()
