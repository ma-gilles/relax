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


@pytest.mark.unit
def test_tilt_capacity_ladders_divide_the_rows_by_the_slot_count():
    # Rows project once per slot; the Wavg tile is built per slot for the units' images, so units keep the
    # image ladder.
    rows, units = resident_tilts.tilt_capacity_ladders((8192, 32768, 131072), (32, 128, 512), slot_capacity=41)
    assert rows == (256, 512, 2048)
    assert units == (32, 128, 512)


@pytest.mark.unit
def test_slot_views_visit_every_chunk_image_once_and_their_partials_land_on_it():
    """The M-step's slot views: unit u's s-th image is chunk image offsets[u] + s, each chunk image in one slot."""

    import jax.numpy as jnp

    from relax.sparse_pass2 import resident_pass2 as rp

    offsets = resident_tilts.validate_unit_image_offsets([0, 3, 4, 6], 3)
    layout = resident_tilts.chunk_tilt_layout(
        offsets,
        unit_start=0,
        n_valid_units=3,
        row_unit_local=np.array([0, 1, 2, 2, 0]),
        n_valid_rows=5,
        image_capacity=12,
        slot_capacity=3,
    )
    views = resident_tilts._unit_slot_images(layout, unit_capacity=4, slot_capacity=3)
    np.testing.assert_array_equal(views, [[0, 3, 4, -1], [1, -1, 5, -1], [2, -1, -1, -1]])
    visited = views[views >= 0]
    np.testing.assert_array_equal(np.sort(visited), np.arange(layout.n_valid_images))

    full = rp._ChunkMstepCarry(
        Ft_y=jnp.zeros(2),
        Ft_ctf=jnp.zeros(2),
        wavg_triplet_pixels=jnp.zeros((12, 1, 3)),
        noise_shells=jnp.zeros(4),
        a2_per_image=jnp.zeros(12),
        xa_per_image=jnp.zeros(12),
    )
    for slot in range(3):
        partial = rp._ChunkMstepCarry(
            Ft_y=jnp.full(2, slot + 1.0),
            Ft_ctf=jnp.full(2, slot + 1.0),
            wavg_triplet_pixels=jnp.arange(4, dtype=jnp.float32)[:, None, None] * jnp.ones((4, 1, 3)) + 10 * slot,
            noise_shells=jnp.ones(4),
            a2_per_image=jnp.arange(4, dtype=jnp.float32) + 10 * slot,
            xa_per_image=-(jnp.arange(4, dtype=jnp.float32) + 10 * slot),
        )
        full = resident_tilts._fold_slot_carry(full, partial, views[slot], image_capacity=12)
    expected = np.zeros(12)
    for slot in range(3):
        for unit in range(4):
            if views[slot, unit] >= 0:
                expected[views[slot, unit]] = unit + 10 * slot
    assert_matches(np.asarray(full.a2_per_image), expected)
    assert_matches(np.asarray(full.xa_per_image), -expected)
    assert_matches(np.asarray(full.wavg_triplet_pixels)[:, 0, 0], expected)
    assert_matches(np.asarray(full.noise_shells), np.full(4, 3.0))
    assert_matches(np.asarray(full.Ft_y), np.full(2, 3.0))


@pytest.mark.unit
def test_mstep_translations_keep_every_translation_with_mass_and_pad_to_a_power_of_two():
    posterior = np.zeros((3, 500), dtype=np.float32)
    posterior[0, [7, 300]] = 0.5
    posterior[2, 41] = 1.0
    kept = resident_tilts.mstep_translations(posterior, 500)
    assert kept.index.size == 32
    np.testing.assert_array_equal(kept.index[kept.valid], [7, 41, 300])
    assert not np.any(kept.valid[3:])
    # Mass on every translation keeps the whole grid.
    full = resident_tilts.mstep_translations(np.ones((2, 40), dtype=np.float32), 40)
    np.testing.assert_array_equal(full.index, np.arange(40))
    assert full.valid.all()
