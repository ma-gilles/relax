"""Tilt-image layout of resident chunks (cryo-ET S4.2, docs/development/resident_segments.md)."""

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.sparse_pass2 import resident_tilts


@pytest.mark.unit
def test_chunk_tilt_layout_enumerates_each_rows_images_in_slot_order():
    # Four units owning 2, 3, 1 and 2 images; the chunk holds units 1..2 (images 2..5).
    offsets = resident_tilts.validate_unit_image_offsets([0, 2, 5, 6, 8], 4)
    row_unit_local = np.array([0, 0, 1, 1, 1, 0, 0])  # 5 valid rows, 2 padding rows
    layout = resident_tilts.chunk_tilt_layout(
        offsets,
        unit_start=1,
        n_valid_units=2,
        row_unit_local=row_unit_local,
        n_valid_rows=5,
        image_capacity=6,
        slot_capacity=4,
    )
    assert layout.n_valid_images == 4
    assert layout.image_ids.tolist() == [2, 3, 4, 5, -1, -1]
    assert layout.image_unit_local.tolist() == [0, 0, 0, 1, 2, 2]
    assert layout.slot_image_ids.tolist() == [
        [0, 0, 3, 3, 3, -1, -1],
        [1, 1, -1, -1, -1, -1, -1],
        [2, 2, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1],
    ]
    assert resident_tilts.max_images_per_unit(offsets) == 3


@pytest.mark.unit
def test_single_particle_units_are_the_one_image_case():
    offsets = resident_tilts.validate_unit_image_offsets(np.arange(6), 5)
    layout = resident_tilts.chunk_tilt_layout(
        offsets,
        unit_start=2,
        n_valid_units=3,
        row_unit_local=np.array([0, 1, 1, 2, 0]),
        n_valid_rows=4,
        image_capacity=4,
        slot_capacity=1,
    )
    assert layout.image_ids.tolist() == [2, 3, 4, -1]
    assert layout.slot_image_ids.tolist() == [[0, 1, 1, 2, -1]]


@pytest.mark.unit
def test_chunk_tilt_layout_refuses_overflow_and_bad_offsets():
    offsets = resident_tilts.validate_unit_image_offsets([0, 3, 4], 2)
    kwargs = dict(unit_start=0, n_valid_units=2, row_unit_local=np.array([0, 1]), n_valid_rows=2)
    with pytest.raises(ValueError, match="image capacity"):
        resident_tilts.chunk_tilt_layout(offsets, image_capacity=3, slot_capacity=3, **kwargs)
    with pytest.raises(ValueError, match="image slots"):
        resident_tilts.chunk_tilt_layout(offsets, image_capacity=4, slot_capacity=2, **kwargs)
    with pytest.raises(ValueError, match="at least one image"):
        resident_tilts.validate_unit_image_offsets([0, 2, 2], 2)


@pytest.mark.unit
def test_tilt_slot_rotations_are_each_images_inverse_of_l_a():
    from relax.sampling import _relion_euler_angles_to_matrix

    rng = np.random.default_rng(3)
    offsets = resident_tilts.validate_unit_image_offsets([0, 2, 3], 2)
    layout = resident_tilts.chunk_tilt_layout(
        offsets,
        unit_start=0,
        n_valid_units=2,
        row_unit_local=np.array([0, 1, 1, 0]),
        n_valid_rows=3,
        image_capacity=4,
        slot_capacity=2,
    )
    eulers = rng.uniform(-180, 180, size=(4, 3))
    left = np.stack([np.linalg.qr(rng.normal(size=(3, 3)))[0] * s for s in (1.0, 1.1, 0.9)])
    got = resident_tilts.tilt_slot_rotations(layout, eulers, left, dtype=np.float64)
    a = _relion_euler_angles_to_matrix(eulers)
    for slot in range(2):
        for row in range(4):
            image = layout.slot_image_ids[slot, row]
            expected = np.eye(3) if image < 0 else np.linalg.inv(left[layout.image_ids[image]] @ a[row]).T
            assert_matches(got[slot * 4 + row], expected)
