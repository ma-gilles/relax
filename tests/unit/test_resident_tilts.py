"""Tilt-image layout of resident chunks (cryo-ET S4.2, docs/development/resident_segments.md)."""

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.sparse_pass2 import resident_tilts


# Moved from relax/sparse_pass2/resident_tilts.py (PLAN e1): no relax module uses them, only this test file.
def validate_unit_image_offsets(unit_image_offsets, n_units: int) -> np.ndarray:
    """The CSR of images over units: ``[n_units + 1]``, starting at 0, every unit owning an image."""

    offsets = np.asarray(unit_image_offsets, dtype=np.int64).reshape(-1)
    if offsets.shape != (int(n_units) + 1,) or offsets[0] != 0:
        raise ValueError(f"unit_image_offsets must have shape ({int(n_units) + 1},) and start at 0")
    if np.any(np.diff(offsets) < 1):
        raise ValueError("every unit must own at least one image")
    return offsets


def max_images_per_unit(unit_image_offsets) -> int:
    return int(np.max(np.diff(np.asarray(unit_image_offsets, dtype=np.int64))))


@pytest.mark.unit
def test_chunk_tilt_layout_enumerates_each_rows_images_in_slot_order():
    # Four units owning 2, 3, 1 and 2 images; the chunk holds units 1..2 (images 2..5).
    offsets = validate_unit_image_offsets([0, 2, 5, 6, 8], 4)
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
    assert max_images_per_unit(offsets) == 3


@pytest.mark.unit
def test_single_particle_units_are_the_one_image_case():
    offsets = validate_unit_image_offsets(np.arange(6), 5)
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
    offsets = validate_unit_image_offsets([0, 3, 4], 2)
    kwargs = dict(unit_start=0, n_valid_units=2, row_unit_local=np.array([0, 1]), n_valid_rows=2)
    with pytest.raises(ValueError, match="image capacity"):
        resident_tilts.chunk_tilt_layout(offsets, image_capacity=3, slot_capacity=3, **kwargs)
    with pytest.raises(ValueError, match="image slots"):
        resident_tilts.chunk_tilt_layout(offsets, image_capacity=4, slot_capacity=2, **kwargs)
    with pytest.raises(ValueError, match="at least one image"):
        validate_unit_image_offsets([0, 2, 2], 2)


@pytest.mark.unit
def test_tilt_slot_rotations_are_each_images_inverse_of_l_a():
    from relax.healpix_sampling import euler_angles_to_matrix

    rng = np.random.default_rng(3)
    offsets = validate_unit_image_offsets([0, 2, 3], 2)
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
    a = euler_angles_to_matrix(eulers)
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

    offsets = validate_unit_image_offsets([0, 3, 4, 6], 3)
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
        placed = resident_tilts._place_slot_partials(
            {name: getattr(full, name) for name in resident_tilts._SLOT_PARTIAL_FIELDS},
            {name: getattr(partial, name) for name in resident_tilts._SLOT_PARTIAL_FIELDS},
            full.noise_shells,
            partial.noise_shells,
            jnp.asarray(np.where(views[slot] >= 0, views[slot], 12), dtype=jnp.int32),
        )
        full = full._replace(Ft_y=partial.Ft_y, Ft_ctf=partial.Ft_ctf, **placed)
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
def test_unit_mstep_translations_keep_each_particles_translations_with_mass():
    """Each unit keeps the translations with mass in any of its rows, ascending, padded with its last (invalid)."""

    posterior = np.zeros((4, 500), dtype=np.float32)
    posterior[0, [300, 7]] = 0.5
    posterior[1, 41] = 1.0  # unit 0's second row
    posterior[3, [2, 9, 11, 13, 17]] = 0.2  # unit 2
    row_unit = np.array([0, 0, 1, 2])  # unit 1's row has no mass; unit 3 has no rows
    kept = resident_tilts.unit_mstep_translations(posterior, row_unit, unit_capacity=4, n_fine_trans=500)
    assert kept.index.shape == (4, 8)  # five translations at most -> the next power of two
    np.testing.assert_array_equal(kept.index[0, :3], [7, 41, 300])
    np.testing.assert_array_equal(kept.index[0, 3:], 300)
    np.testing.assert_array_equal(kept.index[2, :5], [2, 9, 11, 13, 17])
    np.testing.assert_array_equal(kept.valid.sum(axis=1), [3, 0, 5, 0])
    np.testing.assert_array_equal(kept.index[[1, 3]], 0)
    # Every translation of a row with mass is kept, so the dropped ones carry exactly zero posterior.
    for row, unit in enumerate(row_unit):
        assert set(np.flatnonzero(posterior[row])) <= set(kept.index[unit][kept.valid[unit]])
    # Rows in any order give the same tables.
    shuffled = resident_tilts.unit_mstep_translations(posterior[::-1], row_unit[::-1], unit_capacity=4, n_fine_trans=500)
    np.testing.assert_array_equal(shuffled.index, kept.index)
    # A unit with mass on the whole grid keeps the grid.
    full = resident_tilts.unit_mstep_translations(np.ones((2, 40), np.float32), np.array([0, 1]), unit_capacity=2, n_fine_trans=40)
    np.testing.assert_array_equal(full.index, np.tile(np.arange(40), (2, 1)))
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
    out = sampling.relion_mstep_source_eulers(eulers, 9)
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


@pytest.mark.unit
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("coarse_size, fine_size", [(None, None), (32, 48)])
def test_numbered_iteration_tomo_sampling_reads_the_state_grid_or_local_search(local, coarse_size, fine_size):
    """A global search samples the exhaustive grid's order with the run's perturbation; a local search one
    oversampling order below its fine order with its own. A window of None is the full image."""
    from types import SimpleNamespace

    from relax.helpers.resolution import ImageGeometry
    from relax.refinement.tomo_half import TomoSampling, numbered_iteration_tomo_sampling

    sampling = numbered_iteration_tomo_sampling(
        SimpleNamespace(adaptive_oversampling=1, translation_range=0.4, translation_step=0.15),
        ImageGeometry(image_shape=(64, 64), pixel_size_angstrom=4.25),
        local_sampling=(
            SimpleNamespace(search=SimpleNamespace(healpix_order=6), perturbation=-0.2) if local else None
        ),
        grid_healpix_order=3,
        random_perturbation=0.125,
        coarse_size=coarse_size,
        fine_size=fine_size,
    )

    assert isinstance(sampling, TomoSampling)
    assert (sampling.healpix_order, sampling.oversampling_order) == (5 if local else 3, 1)
    assert sampling.random_perturbation == (-0.2 if local else 0.125)
    assert (sampling.coarse_size, sampling.fine_size) == (coarse_size or 64, fine_size or 64)
    assert_matches(np.array([sampling.offset_range_angst, sampling.offset_step_angst]), np.array([1.7, 0.6375]))


@pytest.mark.unit
def test_tilt_image_power_above_the_cutoff_carries_one_over_n_images():
    """Above the norm cutoff a tilt image adds its power with 1 / n_images, not 1 (14547997 kept it at 1:
    the noise above the current size came out n_images times RELION's)."""
    from relax.sparse_pass2.sparse_pass2_wavg import weighted_image_power_from_shells

    rng = np.random.default_rng(3)
    power = rng.uniform(1.0, 2.0, size=(5, 8))
    scale = np.array([0.5, 0.5, 1 / 3, 1 / 3, 1 / 3])  # two particles with 2 and 3 tilt images
    mass = 0.9 * scale
    replacement = rng.uniform(1.0, 2.0, size=5)
    valid = np.ones(5, dtype=bool)
    shells, per_image = weighted_image_power_from_shells(
        power,
        mass,
        replacement,
        valid,
        norm_unweighted_shell_cutoff=4,
        include_unweighted_high_shell=True,
        deterministic_norm_reduction=True,
        high_shell_mass=scale,
    )
    assert_matches(np.asarray(shells)[:5], np.sum(power[:, :5] * mass[:, None], axis=0))
    assert_matches(np.asarray(shells)[5:], np.sum(power[:, 5:] * scale[:, None], axis=0))
    assert_matches(np.asarray(per_image), np.sum(power[:, :5] * mass[:, None], axis=1) + scale * replacement)


@pytest.mark.unit
def test_slot_tables_visit_the_rows_with_mass_and_an_image_first():
    """Per slot: active rows (mass and an image in the slot) first and ascending, blocks cover them, padding drops."""

    views = np.array([[0, 3, 4, -1], [1, -1, 5, -1], [2, -1, -1, -1]])
    row_unit = np.array([0, 1, 2, 2, 0])
    row_has_mass = np.array([True, True, False, True, True])
    angles = np.arange(12 * 6 * 2, dtype=np.float32).reshape(12, 6, 2)
    # Every unit's translations [1, 4, 0, 2, 3, 5]; the first block takes its first two.
    unit_translations = np.tile([1, 4, 0, 2, 3, 5], (4, 1))
    kept = resident_tilts.MstepTranslations(index=np.array([0, 1]), valid=np.array([True, True]))
    tables = resident_tilts._slot_mstep_tables(
        views,
        row_unit=row_unit,
        row_has_mass=row_has_mass,
        n_valid_rows=5,
        row_capacity=8,
        block_rows=2,
        image_angles=angles,
        layout_image_ids=np.arange(12),
        unit_translations=unit_translations,
        translation_blocks=(kept,),
        image_capacity=12,
    )
    order, active = np.asarray(tables.order), np.asarray(tables.active)
    # slot 0: every unit has an image; rows 0, 1, 3, 4 have mass. slot 1: units 0 and 2 (rows 0, 3, 4).
    # slot 2: unit 0 only (rows 0, 4).
    for slot, rows in enumerate(([0, 1, 3, 4], [0, 3, 4], [0, 4])):
        np.testing.assert_array_equal(order[slot, : len(rows)], rows)
        assert active[slot].sum() == len(rows) and np.all(active[slot, : len(rows)])
        assert sorted(order[slot]) == list(range(8))
    np.testing.assert_array_equal(np.asarray(tables.n_blocks), [2, 2, 1])
    np.testing.assert_array_equal(np.asarray(tables.targets), np.where(views >= 0, views, 12))
    assert_matches(np.asarray(tables.angles)[1, 0, 2], angles[5][[1, 4]])
    # Several translation blocks: block b of every slot and unit holds that image's phases at the block's indices.
    blocks = (kept, resident_tilts.MstepTranslations(index=np.array([2, 5]), valid=np.array([True, False])))
    several = resident_tilts._slot_mstep_tables(
        views,
        row_unit=row_unit,
        row_has_mass=row_has_mass,
        n_valid_rows=5,
        row_capacity=8,
        block_rows=2,
        image_angles=angles,
        layout_image_ids=np.arange(12),
        unit_translations=unit_translations,
        translation_blocks=blocks,
        image_capacity=12,
    )
    assert np.asarray(several.angles).shape == (3, 2, 4, 2, 2)
    for block, translations in enumerate(blocks):
        for slot, unit in ((0, 1), (1, 2), (2, 0)):
            assert_matches(
                np.asarray(several.angles)[slot, block, unit],
                angles[views[slot, unit]][unit_translations[unit][translations.index]],
            )


@pytest.mark.unit
def test_slot_views_take_every_per_image_operand():
    """A tilt slot's M-step reads its units' images: every per-image chunk operand is in the slot view.

    An operand left out keeps the chunk's image order, so a slot's unit u read the chunk's u-th image:
    CTF-premultiplied tilt images backprojected another image's CTF weight (2026-10-03).
    """

    from relax.sparse_pass2.resident_pass2 import _ChunkStageOperands

    view = set(resident_tilts._SLOT_VIEW_FIELDS)
    chunk_only = set(resident_tilts._SLOT_CHUNK_ONLY_FIELDS)
    assert not view & chunk_only
    assert set(_ChunkStageOperands._fields) == view | chunk_only
    assert "bpref_ctf2_over_nv_recon" in view


@pytest.mark.unit
def test_tilt_mstep_holds_one_translation_blocks_tiles(monkeypatch):
    """The slot M-step translates one block at a time: its compiled temporaries do not grow with the block count.

    Unrolled over the blocks, XLA scheduled every block's translated Wavg tiles before consuming any and
    allocated them all at once (31.94 GiB on etok2s4_premult Class3D K2, against a 12.84 GiB plan that
    counts one block, as mstep_translation_blocks sizes them). The kernels are stand-ins that keep each
    block's [C_U, T, W] tile live into the carry.
    """

    import dataclasses

    import jax.numpy as jnp

    from relax.sparse_pass2 import resident_pass2 as rp

    units, rows, translations, window = 8, 16, 32, 512

    def translate(window_rows, valid, angles, rect_indices, exact_positions, *, image_shape):
        phases = jnp.sum(angles, axis=-1)[:, :, None]
        tile = (window_rows[:, None, :] * jnp.exp(1j * phases)).astype(jnp.complex64)
        return tile, tile[:, :, exact_positions]

    def block_at(start, blocks, operands, tables, carry, *, spec, cuda_backproject):
        power = jnp.abs(operands.raw_translated_wavg_rectangle) ** 2
        return carry._replace(Ft_y=carry.Ft_y + jnp.sum(power) * blocks.row_posterior.sum())

    def image_power(carry, operands, tables, row_posterior, row_ids, *, spec):
        return carry._replace(Ft_ctf=carry.Ft_ctf + jnp.sum(jnp.abs(operands.raw_translated_wavg_for_atomic)))

    def initial_carry(Ft_y, Ft_ctf, operands, tables, *, spec):
        return rp._ChunkMstepCarry(
            Ft_y=Ft_y,
            Ft_ctf=Ft_ctf,
            wavg_triplet_pixels=jnp.zeros((units, 1, 3), jnp.float32),
            noise_shells=jnp.zeros(4, jnp.float64),
            a2_per_image=jnp.zeros(units, jnp.float32),
            xa_per_image=jnp.zeros(units, jnp.float32),
        )

    monkeypatch.setattr(resident_tilts, "_slot_view_translate", translate)
    monkeypatch.setattr(rp, "_resident_mstep_block_at", block_at)
    monkeypatch.setattr(rp, "_add_chunk_wavg_image_power", image_power)
    monkeypatch.setattr(rp, "_initial_mstep_carry", initial_carry)
    fields = rp._ChunkStageOperands._fields
    operands = rp._ChunkStageOperands(
        **{name: (jnp.ones((units, 4), jnp.float32) if name in resident_tilts._SLOT_VIEW_FIELDS else None) for name in fields}
    )
    tables = rp._ChunkStageTables(**{name: None for name in rp._ChunkStageTables._fields})

    @dataclasses.dataclass(frozen=True)
    class Spec:
        mstep_block_rows: int = 8
        presum_adjoint: bool = False  # the row-sum merge (relax#27) adds one [C_R, recon] pair, not per block

    def temp_bytes(n_blocks):
        slots = resident_tilts.SlotMstepTables(
            slot=jnp.arange(2, dtype=jnp.int32),
            safe_images=jnp.zeros((2, units), jnp.int32),
            valid_images=jnp.ones((2, units), bool),
            targets=jnp.zeros((2, units), jnp.int32),
            order=jnp.tile(jnp.arange(rows, dtype=jnp.int32), (2, 1)),
            active=jnp.ones((2, rows), bool),
            units=jnp.zeros((2, rows), jnp.int32),
            n_blocks=jnp.full(2, 2, jnp.int32),
            angles=jnp.ones((2, n_blocks, units, translations, 2), jnp.float32),
        )
        mstep = initial_carry(jnp.zeros(4), jnp.zeros(4), None, None, spec=None)
        lowered = resident_tilts._tilt_mstep_program.lower(
            mstep,
            slots,
            operands,
            tables,
            jnp.ones((units, window), jnp.complex64),
            jnp.ones((rows, n_blocks * translations), jnp.float32),
            jnp.arange(n_blocks * translations, dtype=jnp.int32).reshape(n_blocks, translations),
            jnp.ones((n_blocks, translations), bool),
            jnp.arange(window, dtype=jnp.int32),
            jnp.arange(window // 2, dtype=jnp.int32),
            slot_spec=Spec(),
            image_shape=(8, 8),
        )
        return lowered.compile().memory_analysis().temp_size_in_bytes

    tile_bytes = units * translations * window * 8
    one, many = temp_bytes(1), temp_bytes(8)
    assert many < one + 2 * tile_bytes, (one, many, tile_bytes)


@pytest.mark.unit
def test_tilt_mstep_census_logs_one_line_per_pass(caplog):
    """The census line names the largest k, the smallest block and how many chunks needed several blocks."""

    import logging

    census = resident_tilts.TiltMstepCensus()
    census.log()  # a pass without tilt chunks logs nothing
    census.add(16, 16, 1)
    census.add(64, 8, 8)
    census.add(8, 16, 1)
    with caplog.at_level(logging.INFO, logger="relax.sparse_pass2.resident_tilts"):
        census.log()
    assert [r.getMessage() for r in caplog.records] == [
        "Tilt M-step translation blocks: largest k 64, smallest block 8 translations, multi-block chunks 1/3"
    ]
