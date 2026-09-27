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


@pytest.mark.unit
def test_subtomogram_translation_schedule_is_relions_s1_schedule():
    """RELION's subtomogram rule: each new offset step is at least half the previous one.

    The S1 depth-fix RELION run (relion_ref, run_it00{4,5,7,8,9,10}_sampling.star) goes 4.25 -> 2.125
    -> 1.0625 -> 0.6375 A with ranges 5x its offset changes (1.344198, 0.813644, 0.358909 A) and
    acc_trans 0.425 A at oversampling 1. The SPA rule would jump straight to 0.6375 A.
    """
    from relax.helpers.convergence import RefinementState, _relion_next_translation_sampling_pixels

    pixel = 4.25
    step_px, range_px = 1.0, 5.0
    expected = [(2.125, 6.720990), (1.0625, 4.068220), (0.6375, 1.794545)]
    for changes, (want_step, want_range) in zip((1.344198, 0.813644, 0.358909), expected):
        state = RefinementState(
            adaptive_oversampling=1,
            translation_range=range_px,
            translation_step=step_px,
            voxel_size_angstrom=pixel,
            subtomogram=True,
        )
        state.acc_trans = 0.425
        state.current_changes_optimal_offsets_angstrom = changes
        range_px, step_px = _relion_next_translation_sampling_pixels(state)
        # RELION's STAR values carry 6 decimals.
        assert_matches(np.array([step_px * pixel, range_px * pixel]), np.array([want_step, want_range]), rtol=1e-6)


@pytest.mark.unit
def test_mstep_source_eulers_never_build_a_grid_of_another_size(monkeypatch):
    """A final local search at order 9 (subtomograms) keeps its own angles without building the order-9 grid."""
    from relax import sampling

    eulers = np.zeros((10, 3))
    monkeypatch.setattr(
        sampling, "_get_relion_rotation_grid_eulers_float64", lambda *a, **k: pytest.fail("built the full grid")
    )
    out = sampling._relion_mstep_source_eulers(eulers, 9)
    assert out.shape == (10, 3)


@pytest.mark.unit
def test_mstep_translation_blocks_fit_the_tile_budget_and_cover_every_kept_translation():
    kept = resident_tilts.MstepTranslations(index=np.arange(0, 600, 2, dtype=np.int64), valid=np.ones(300, dtype=bool))
    per_translation = 1000
    blocks = resident_tilts.mstep_translation_blocks(
        kept, bytes_per_translation=per_translation, tile_budget_bytes=70_000
    )
    assert {int(b.index.size) for b in blocks} == {64}
    assert all(int(b.index.size) * per_translation <= 70_000 for b in blocks)
    covered = np.concatenate([b.index[b.valid] for b in blocks])
    np.testing.assert_array_equal(covered, kept.index)
    # A chunk whose kept translations fit keeps one block.
    assert resident_tilts.mstep_translation_blocks(kept, bytes_per_translation=1, tile_budget_bytes=10**9) == (kept,)


@pytest.mark.unit
def test_tilt_unit_capacity_at_the_full_box_is_bounded_by_its_gather():
    """At the S1 full box (T = 4120, 41 tilts) a tilt unit costs its S gathered images, not T tiles.

    The SPA model (images x T x P x 3 tiles) keeps two particles per chunk here, which made the S1
    standalone's full-box iterations 7x slower; the tilt model keeps the default classes that fit.
    """
    from relax.sparse_pass2.resident_pass2 import _cap_image_capacity_ladder
    from relax.sparse_pass2.sparse_pass2_budget import _max_translation_tile_bytes_for_pass

    n_recon_pixels, slots, budget = 128 * 65, 41, _max_translation_tile_bytes_for_pass(80 * 1024**3)
    ladder = (32, 64, 128, 256)
    spa = _cap_image_capacity_ladder(ladder, n_fine_trans=4120, n_recon_pixels=n_recon_pixels, max_tile_bytes=budget)
    tilt = _cap_image_capacity_ladder(ladder, n_fine_trans=slots, n_recon_pixels=n_recon_pixels, max_tile_bytes=budget)
    assert max(spa) <= 4
    assert 32 in tilt
    assert all(units * slots * n_recon_pixels * 8 * 3 <= budget for units in tilt)


@pytest.mark.unit
def test_mstep_translation_blocks_stay_within_the_kernel_table_limit():
    kept = resident_tilts.MstepTranslations(index=np.arange(4120, dtype=np.int64), valid=np.ones(4120, dtype=bool))
    blocks = resident_tilts.mstep_translation_blocks(kept, bytes_per_translation=1, tile_budget_bytes=10**12)
    assert all(int(b.index.size) == resident_tilts.TILT_MSTEP_MAX_TRANSLATIONS for b in blocks)
    np.testing.assert_array_equal(np.concatenate([b.index[b.valid] for b in blocks]), kept.index)


@pytest.mark.unit
@pytest.mark.parametrize("pass1_size", [None, 64])
def test_final_pass_local_tomo_sampling_without_a_parent_pass_size(pass1_size):
    """The final all-data pass at order 9 has no parent-pass size (14506700 crashed on it): coarse = current."""
    from relax.refinement.tomo_half import local_tomo_sampling

    sampling = local_tomo_sampling(
        fine_order=10,
        oversampling_order=1,
        translation_range_px=0.4,
        translation_step_px=0.15,
        voxel_size=4.25,
        random_perturbation=-0.2,
        pass1_size=pass1_size,
        current_size=128,
    )
    assert sampling.healpix_order == 9 and sampling.fine_size == 128
    assert sampling.coarse_size == (128 if pass1_size is None else pass1_size)
    assert_matches(np.array([sampling.offset_range_angst, sampling.offset_step_angst]), np.array([1.7, 0.6375]))
