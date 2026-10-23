"""Subtomogram coarse pass (S4.2): the per-image GEMM scorer, the per-particle sum and the cut, with stand-ins on CPU.

The per-image operands and the kernel are the SPA ones; relion_dump_it1_forced (em_work/cryoet_s42_20260925,
check_p3_p4.py) compares the whole pass with RELION's dumped coarse diff2 on the GPU.
"""

import numpy as np
import pytest
from helpers.float_compare import assert_matches
from oracles.tomo_coarse import tilt_image_coarse_diff2

pytest.importorskip("jax")
import jax.numpy as jnp

from relax.cuda import coarse as cuda_coarse
from relax.cuda import translate as cuda_translate
from relax.scoring import tomo_coarse, tomo_coarse_gemm


# Moved from relax/scoring/tomo_coarse.py (PLAN e1): no relax module uses them, only this test file.
def particle_coarse_diff2(image_diff2_in_slot_order):
    """A particle's coarse diff2: its images' diff2 added in slot (``img_id``) order, float32."""

    total = None
    for image_diff2 in image_diff2_in_slot_order:
        image_diff2 = jnp.asarray(image_diff2, dtype=jnp.float32)
        total = image_diff2 if total is None else total + image_diff2
    return total


def _coarse_gemm_projections(
    projector_half, rotations, layout: tomo_coarse.CoarseScoreLayout, *, model_max_r: int, padding_factor: int, texture=None
):
    """The score-window rows of a RELION ``PPref`` half at ``rotations`` ``[N, 3, 3]`` (see
    :func:`_score_window_projections` for the two layouts).

    The projection the SPA coarse GEMM scorer reads (significance.py ``_project_relion_compact_score_rows``):
    RELION's texture interpolation with the coarse diff2 kernel's row rule, in ``layout``'s score pixels.
    ``texture`` is the half's :class:`relax.cuda.projector.RelionCapacityHalfTextureF32`, staged once per pass.
    """

    return tomo_coarse_gemm._score_window_projections(
        projector_half,
        rotations,
        jnp.asarray(layout.score_indices_np, dtype=jnp.int32),
        image_shape=tuple(layout.image_shape),
        current_size=int(layout.current_size),
        model_max_r=int(model_max_r),
        padding_factor=int(padding_factor),
        texture=texture,
    )


pytestmark = pytest.mark.unit


def _fake_projector(projector_full, rotations, images, angles, weight, initial, full_to_compact, **kwargs):
    # [1, R, Tc]: depends on the rotation, the translation angle and the image's initial diff2.
    per_rot = jnp.sum(rotations, axis=(1, 2))
    return (initial[:, None, None] + per_rot[None, :, None] + angles[None, None, :, 0])[None][0]


def test_translations_are_scored_in_chunks_and_images_add_in_slot_order(monkeypatch):
    monkeypatch.setattr(cuda_coarse, "relion_coarse_diff2_projector_f32", _fake_projector)
    rng = np.random.default_rng(3)
    layout = tomo_coarse.CoarseScoreLayout(
        (8, 8), 6, np.arange(4), jnp.ones(4, bool), jnp.zeros(1, jnp.int32), jnp.ones(40)
    )
    n_rot, n_trans = 5, 300  # three chunks of at most 128 translations
    rotations = rng.normal(size=(3, n_rot, 3, 3)).astype(np.float32)
    angles = rng.normal(size=(3, n_trans, 2)).astype(np.float32)
    initial = np.array([1.5, 2.5, 3.5], np.float32)
    per_image = [
        tilt_image_coarse_diff2(
            None,
            rotations[i],
            jnp.zeros(4, jnp.complex64),
            jnp.ones(4),
            initial[i],
            angles[i],
            layout,
            model_max_r=3,
            padding_factor=2,
        )
        for i in range(3)
    ]
    for i in range(3):
        expected = initial[i] + rotations[i].sum(axis=(1, 2))[:, None] + angles[i][None, :, 0]
        assert_matches(np.asarray(per_image[i]), expected.astype(np.float32))
    total = np.asarray(particle_coarse_diff2(per_image))
    expected_total = np.float32(
        np.float32(np.asarray(per_image[0]) + np.asarray(per_image[1])) + np.asarray(per_image[2])
    )
    assert_matches(total, expected_total)


def _relion_coarse_cut(log_weight, adaptive_fraction):
    """RELION's coarse significance from float32 log weights (independent NumPy reading of acc :2245-2345)."""
    w = np.exp((log_weight + (np.float32(50.0) - log_weight.max())).astype(np.float32)).astype(np.float32)
    order = np.sort(w)
    cumulative = np.cumsum(order, dtype=np.float32)
    idx = int(np.searchsorted(cumulative, np.float32((1.0 - adaptive_fraction) * cumulative[-1]), side="right"))
    return w >= order[min(idx, w.size - 1)]


def _relion_coarse_count(log_weight, adaptive_fraction):
    """RELION's rlnNrOfSignificantSamples of the same cut: ``filteredSize - thresholdIdx`` (acc :3236), the
    samples from the cut's rank upwards, without the samples tied with the cut weight below that rank."""
    w = np.exp((log_weight + (np.float32(50.0) - log_weight.max())).astype(np.float32)).astype(np.float32)
    cumulative = np.cumsum(np.sort(w), dtype=np.float32)
    return w.size - int(np.searchsorted(cumulative, np.float32((1.0 - adaptive_fraction) * cumulative[-1]), side="right"))


def test_particle_significance_cuts_the_summed_diff2_with_its_3d_offset_prior():
    rng = np.random.default_rng(5)
    n_particles, n_rot, n_trans = 3, 40, 27
    image_diff2 = rng.uniform(1000.0, 1010.0, size=(4, n_particles, n_rot, n_trans)).astype(np.float32)
    particle = particle_coarse_diff2(list(image_diff2))
    rotation_prior = np.log(rng.uniform(0.5, 1.0, size=n_rot)).astype(np.float32)
    offset_prior = np.log(rng.uniform(0.1, 1.0, size=(n_particles, n_trans))).astype(np.float32)
    got = tomo_coarse.particle_coarse_significance(
        particle, rotation_prior, offset_prior, adaptive_fraction=0.999, max_significants=None
    )
    for p in range(n_particles):
        diff2 = np.asarray(particle[p], np.float32)
        log_weight = (rotation_prior[:, None] + offset_prior[p][None, :]) + (diff2.min() - diff2)
        expected = _relion_coarse_cut(log_weight.astype(np.float32).reshape(-1), 0.999)
        assert np.array_equal(np.asarray(got["mask"][p]), expected)
        assert int(got["n_significant"][p]) == int(expected.sum())


def test_particle_significance_counts_a_tied_cut_as_relion_records_it():
    """relax#89: with a sample tied with the cut weight below the cut's rank, the support keeps it (RELION's
    arrayOverThreshold keeps every weight >= the cut) but RELION's count does not: the subtomogram pass records
    ``cutoff_count``, not the support's size."""
    rng = np.random.default_rng(89)
    n_particles, n_rot, n_trans = 8, 20, 9
    # Every diff2 value appears twice (the second half of the rotations repeats the first), so every cut weight
    # is tied, and a cut whose rank falls on the second of a pair leaves its twin below the rank.
    half = rng.uniform(1000.0, 1012.0, size=(n_particles, n_rot // 2, n_trans)).astype(np.float32)
    particle = np.concatenate([half, half], axis=1)
    got = tomo_coarse.particle_coarse_significance(
        jnp.asarray(particle), None, np.zeros((n_particles, n_trans), np.float32), adaptive_fraction=0.999,
        max_significants=None,
    )
    tied = 0
    for p in range(n_particles):
        log_weight = (particle[p].min() - particle[p]).astype(np.float32).reshape(-1)
        expected = _relion_coarse_cut(log_weight, 0.999)
        assert np.array_equal(np.asarray(got["mask"][p]), expected)
        assert int(got["n_significant"][p]) == int(expected.sum())
        assert int(got["cutoff_count"][p]) == _relion_coarse_count(log_weight, 0.999)
        tied += int(got["n_significant"][p]) > int(got["cutoff_count"][p])
    assert tied > 0


def test_images_add_in_slot_order():
    rng = np.random.default_rng(7)
    image_diff2 = rng.uniform(1000.0, 1010.0, size=(2, 3, 5, 9)).astype(np.float32)
    total = np.asarray(tomo_coarse_gemm._add_image_diff2_in_slot_order(jnp.zeros((2, 5, 9), jnp.float32), jnp.asarray(image_diff2)))
    expected = np.zeros((2, 5, 9), np.float32)
    for s in range(3):
        expected = (expected + image_diff2[:, s]).astype(np.float32)
    assert_matches(total, expected)


def test_the_gemm_scorer_is_relions_direct_square_with_each_images_own_rows(monkeypatch):
    """Per image: initial + 0.5 sum w |p_r - y_t|^2 over its own projections and shifted pixels."""

    rng = np.random.default_rng(13)
    n_images, n_rot, n_trans, n_px = 3, 6, 5, 11
    projected = (rng.normal(size=(n_images, n_rot, n_px)) + 1j * rng.normal(size=(n_images, n_rot, n_px))).astype(np.complex64)
    images = (rng.normal(size=(n_images, n_px)) + 1j * rng.normal(size=(n_images, n_px))).astype(np.complex64)
    weight = rng.uniform(0.1, 1.0, size=(n_images, n_px)).astype(np.float32)
    initial = rng.uniform(10.0, 20.0, size=n_images).astype(np.float32)
    angles = rng.uniform(-0.2, 0.2, size=(n_images, n_trans, 2)).astype(np.float32)

    def shift(images, translation_angles, pixel_indices, image_shape):
        # A stand-in phase per image and translation: y_bt = image_b * exp(i * (tx + ty)), [B * T, P].
        phase = jnp.exp(1j * (translation_angles[..., 0] + translation_angles[..., 1])).astype(jnp.complex64)
        return (images[:, None, :] * phase[:, :, None]).reshape(-1, images.shape[1])

    monkeypatch.setattr(cuda_translate, "relion_translate_score_f32", shift)
    got, _bound = tomo_coarse_gemm._images_coarse_gemm_diff2.__wrapped__(
        jnp.asarray(projected), jnp.asarray(images), jnp.asarray(weight), jnp.asarray(initial), jnp.asarray(angles),
        jnp.arange(n_px, dtype=jnp.int32), image_shape=(8, 8),
    )
    got = np.asarray(got)
    for b in range(n_images):
        shifted = images[b][None, :] * np.exp(1j * angles[b].sum(axis=1))[:, None]
        direct = initial[b] + 0.5 * np.einsum(
            "p,rtp->rt", weight[b].astype(np.float64), np.abs(projected[b][:, None, :] - shifted[None]) ** 2
        )
        assert_matches(got[b], direct.T.astype(np.float32), rtol=1e-5)  # translation-major


@pytest.mark.parametrize("available_gb", [70, 30, 10, None])
def test_the_coarse_pass_budget_is_one_reading_shared_by_block_batch_and_flushes(monkeypatch, available_gb):
    """Operand block, GEMM batch and flushes share three quarters of one reading, less the full projectors not
    yet built; an 80 GB card keeps the block's and the batch's fixed caps, an unknown device all the caps."""

    gib = 1 << 30
    reading = None if available_gb is None else available_gb * gib
    monkeypatch.setattr(tomo_coarse.memory_budget, "current_device_available_bytes", lambda: reading)
    resident = 650 << 20  # two classes' full projectors at box 192
    budget = tomo_coarse._coarse_pass_budget(resident)
    if reading is None:
        assert budget == (tomo_coarse._COARSE_OPERAND_BLOCK_BYTES, tomo_coarse._COARSE_BATCH_BYTES, 1 << 62)
        return
    assert budget.operand_block + budget.batch + budget.flush <= 0.75 * (reading - resident) + 1
    if available_gb == 70:
        assert (budget.operand_block, budget.batch) == (tomo_coarse._COARSE_OPERAND_BLOCK_BYTES, 2 * gib)


# Per-particle bytes of a flush: et09_box64 at iteration 100 (36864 rotations x 123 translations, 41 tilts of 924
# score pixels) and a w2_09_box192-like pass (4608 x 27, 41 tilts of 14000 pixels), whose operands dominate.
_ET09 = dict(diff2_bytes=36864 * 123 * 4, operand_bytes=41 * (924 * 12 + 4 + 123 * 8))
_W2_09 = dict(diff2_bytes=4608 * 27 * 4, operand_bytes=41 * (14000 * 12 + 4 + 27 * 8))


def _flush_held_bytes(particles, diff2_bytes, operand_bytes):
    return particles * (
        2 * (diff2_bytes + diff2_bytes // 4 + operand_bytes)
        + operand_bytes
        + (tomo_coarse._CUT_DIFF2_COPIES + 1) * diff2_bytes
    )


@pytest.mark.parametrize("card_gb, available_gb", [(80, 70), (40, 30), (16, 10)])
def test_flush_plan_keeps_two_flushes_and_their_cut_within_the_flush_budget(monkeypatch, card_gb, available_gb):
    """Two held flushes (diff2, mask, operands), the dispatch's operand copy and the cut's diff2 copies fit the
    flush budget less its rescore share; on an 80 GB card et09's flush is the 512 MB diff2 flush as before and
    w2_09's 1000 particles are one flush."""

    monkeypatch.setattr(tomo_coarse.memory_budget, "current_device_available_bytes", lambda: available_gb * 2**30)
    flush_bytes = tomo_coarse._coarse_pass_budget(0).flush
    for sizes in (_ET09, _W2_09):
        particles, rescore, by_diff2, by_device = tomo_coarse._flush_plan(
            **sizes, batch_particles=1, flush_bytes=flush_bytes, describe="test"
        )
        assert 1 <= particles <= min(by_diff2, by_device)
        assert _flush_held_bytes(particles, **sizes) + rescore <= flush_bytes
        assert particles * sizes["diff2_bytes"] <= tomo_coarse._SIGNIFICANCE_BATCH_BYTES
    if card_gb == 80:
        plan = tomo_coarse._flush_plan(**_ET09, batch_particles=1, flush_bytes=flush_bytes, describe="test")
        assert plan[0] == (512 << 20) // _ET09["diff2_bytes"]
        plan = tomo_coarse._flush_plan(**_W2_09, batch_particles=1, flush_bytes=flush_bytes, describe="test")
        assert plan[0] >= 1000
    if card_gb == 16:
        # The diff2-only plan held more than the whole flush budget at box 192 (relax#28).
        old = tomo_coarse._SIGNIFICANCE_BATCH_BYTES // _W2_09["diff2_bytes"]
        assert _flush_held_bytes(old, **_W2_09) > flush_bytes


def test_flush_plan_keeps_whole_batches_and_names_an_impossible_particle():
    flush_bytes = 1 << 30
    rescore = int(tomo_coarse._RESCORE_SHARE * flush_bytes)
    by_device = (flush_bytes - rescore) // _flush_held_bytes(1, **_W2_09)
    power = 1 << (by_device.bit_length() - 1)  # rounded down to a power of two, then to whole batches
    assert 4 < power <= by_device < 2 * power
    for batch in (4, 3):
        plan = tomo_coarse._flush_plan(**_W2_09, batch_particles=batch, flush_bytes=flush_bytes, describe="test")
        assert plan == (power // batch * batch, rescore, (512 << 20) // _W2_09["diff2_bytes"], by_device)
    with pytest.raises(MemoryError, match="box 192, current size 96, 41 tilts"):
        tomo_coarse._flush_plan(
            **_W2_09, batch_particles=1, flush_bytes=1 << 20, describe="box 192, current size 96, 41 tilts"
        )


def test_rescore_groups_fit_their_bytes():
    """The undecided rotations scored again per call: a power of two within the rescore bytes, at most the
    capacity, at least one."""

    per_row = 30 * (41 * (36 + 2 * 27 * 4) + 2 * 27 * 4)
    assert tomo_coarse._rescore_rows_per_call(30, 41, 27, 100 * per_row, 4096) == 64
    assert tomo_coarse._rescore_rows_per_call(30, 41, 27, 100 * per_row, 32) == 32
    assert tomo_coarse._rescore_rows_per_call(30, 41, 27, 1, 32) == 1


@pytest.mark.parametrize("capacity", [1, 4, 16, 64, 256])
def test_significant_cells_are_the_first_true_cells_in_order(capacity):
    """The two-step compaction (rows holding a true cell, then their cells) gives ``nonzero``'s first
    ``capacity`` flat ids, -1 past the true cells, including rows with several true cells and a capacity past
    the 15 rows (16, 64) and past all 105 cells (256)."""

    rng = np.random.default_rng(capacity)
    mask = rng.random((3, 5 * 7)) < 0.04
    mask[1, 7:14] = True  # one row of seven true cells
    want = np.flatnonzero(mask)[:capacity]
    want = np.concatenate([want, np.full(capacity - want.size, -1)])
    got = np.asarray(tomo_coarse._significant_cells(jnp.asarray(mask), capacity=capacity, row_length=7))
    np.testing.assert_array_equal(got, want)


def test_cells_capacity_is_the_power_of_four_class_that_holds_the_cells():
    """A flush's compaction size is the smallest power of four that holds its cells (at least 4): the 14 power-of-two
    sizes one global iteration of the real tilt-series case compiled fall into 9 classes."""

    for n in [0, 1, 4, 5, 16, 17, 1000, 4096, 4097, 3_000_000]:
        c = tomo_coarse._cells_capacity(n)
        assert c >= max(n, 4) and c & (c - 1) == 0 and (c.bit_length() - 1) % 2 == 0
        assert c // 4 < max(n, 4)
    pow2_sizes = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 262144, 524288, 2097152, 4194304, 8388608]
    assert len({tomo_coarse._cells_capacity(n) for n in pow2_sizes}) == 9


@pytest.mark.gpu
@pytest.mark.parametrize("rows_per_call", [256, 16], ids=["windows_of_64", "windows_of_16"])
def test_group_rescore_in_chunks_and_windows_is_the_one_call_rescore(gpu_device, rows_per_call):
    """The group rescore (``_RESCORE_GROUP_PARTICLES`` particles and at most ``_RESCORE_WINDOW_ROWS`` rotations per
    call) with RELION's direct-square kernel: the diff2 and counts are bitwise those of scoring every group
    particle's undecided rotations in one call, for 19 of 21 flush particles (two chunks, the second padded) with
    up to 230 undecided rotations each (four windows of 64)."""

    import jax
    from recovar.cuda_backproject import custom_cuda_requested

    from relax.scoring.exact_cut import first_rows

    if not custom_cuda_requested():
        pytest.skip("custom CUDA is disabled")
    with jax.default_device(gpu_device):
        n_particles, n_slots, n_rot, n_trans, capacity = 21, 2, 256, 24, 256
        rng = np.random.default_rng(41)
        f = _gemm_fixture(rng, n_images=n_particles * n_slots, n_rot=n_rot, n_trans=n_trans)
        layout = f["layout"]
        rotation_table = jnp.asarray(f["rotations"].reshape(n_particles, n_slots, n_rot, 3, 3))
        operands = (
            jnp.asarray(f["images"]).reshape(n_particles, n_slots, -1),
            jnp.asarray(f["weight"]).reshape(n_particles, n_slots, -1),
            jnp.asarray(f["initial"]).reshape(n_particles, n_slots),
            jnp.asarray(f["angles"]).reshape(n_particles, n_slots, n_trans, 2),
        )
        counts = rng.integers(9, 231, size=n_particles)
        rows = np.zeros((n_particles, n_rot), bool)
        for p_, c in enumerate(counts):
            rows[p_, rng.choice(n_rot, size=int(c), replace=False)] = True
        diff2 = jnp.asarray(rng.uniform(100.0, 200.0, size=(n_particles, n_rot, n_trans)).astype(np.float32))
        scored = jnp.zeros(n_particles, jnp.int32)
        positions = np.setdiff1d(np.arange(n_particles), [3, 17])

        def rescore(units, values, ids, use, group_operands):
            row_rotations = jnp.take_along_axis(rotation_table[np.asarray(units)], ids[:, None, :, None, None], axis=2)
            return tomo_coarse._exact_rows_diff2(
                values, ids, use, row_rotations, (f["full"],), *group_operands, layout.full_to_compact,
                r_pad=n_rot, current_size=layout.current_size, physical_image_size=f["box"],
                model_max_r=f["max_r"], padding_factor=f["pad"],
            )

        got, got_scored = tomo_coarse._rescore_group(
            rescore, lambda n_pad: rows_per_call, np.arange(n_particles), diff2, scored, jnp.asarray(rows),
            operands, positions, capacity,
        )
        ids, use = first_rows(jnp.asarray(rows[positions]), capacity=capacity)
        one_call = rescore(positions, diff2[positions], ids, use, tuple(v[positions] for v in operands))
        want = np.asarray(diff2).copy()
        want[positions] = np.asarray(one_call)
        want_scored = np.zeros(n_particles, np.int32)
        want_scored[positions] = counts[positions]
        assert not np.array_equal(want, np.asarray(diff2))
        np.testing.assert_array_equal(np.asarray(got).view(np.uint32), want.view(np.uint32))
        np.testing.assert_array_equal(np.asarray(got_scored), want_scored)


def test_shared_frames_group_a_tomograms_particles_and_rank_frames_in_slot_order():
    """Images with one left matrix are one frame; the frames are ranked in an order every particle's slots follow,
    and the particles sharing frames go together. None when frames hold fewer than two images on average, when a
    particle has two images in one frame, or when two particles order two frames both ways."""

    rng = np.random.default_rng(3)
    tomo_a, tomo_b = rng.normal(size=(4, 3, 3)), rng.normal(size=(3, 3, 3))
    # Particles 0, 2, 4 in tomogram A (frames 0-3, 1-3, 0-2), 1, 3, 5 in tomogram B (frames 0-2, 0-1, 1-2).
    images = [tomo_a[[0, 1, 2, 3]], tomo_b[[0, 1, 2]], tomo_a[[1, 2, 3]], tomo_b[[0, 1]], tomo_a[[0, 1, 2]], tomo_b[[1, 2]]]
    offsets = np.concatenate([[0], np.cumsum([len(i) for i in images])])
    frames = tomo_coarse.shared_frames(np.concatenate(images), offsets)
    assert frames is not None
    key = frames.image_key
    assert np.unique(key).size == 7
    for unit in range(6):
        assert np.all(np.diff(frames.key_rank[key[offsets[unit] : offsets[unit + 1]]]) > 0)
    np.testing.assert_array_equal(frames.unit_order, [0, 2, 4, 1, 3, 5])
    np.testing.assert_array_equal(np.concatenate(images)[frames.key_image[key]], np.concatenate(images))
    # One image per frame: nothing to share.
    assert tomo_coarse.shared_frames(rng.normal(size=(6, 3, 3)), [0, 3, 6]) is None
    # Two images of one particle in one frame.
    assert tomo_coarse.shared_frames(tomo_a[[0, 0, 1, 0, 1, 1]], [0, 3, 6]) is None
    # Particle 0 sees frame 0 before frame 1, particle 1 the other way.
    assert tomo_coarse.shared_frames(tomo_a[[0, 1, 1, 0]], [0, 2, 4]) is None


def test_coarse_batches_share_one_shape_within_the_budget():
    batches = tomo_coarse._coarse_batches(
        [130, 90, 200, 50, 70], n_slots=39, n_trans=81, budget_bytes=40 * 256 * 81 * 4 * 2
    )
    assert [b[1] for b in batches] == [256] * 3 and all(b[2] == 2 and b[3] == 39 for b in batches)
    np.testing.assert_array_equal(np.concatenate([b[0] for b in batches]), np.arange(5))
    # One particle over the budget is scored a block of slots at a time.
    ((units, r_pad, p_pad, slot_block),) = tomo_coarse._coarse_batches(
        [4608], n_slots=39, n_trans=515, budget_bytes=4608 * 515 * 4 * 6
    )
    assert (r_pad, p_pad, slot_block) == (4608, 1, 6)


def test_coarse_batch_holds_the_fallback_projection_bytes_within_the_device_share(monkeypatch):
    """The cryoet_s1 iteration-1 half-2 coarse pass on a 16 GB P100 (Polar 413469): 600 particles of 41 tilt images,
    4608 rotations, 81 translations, 278 score pixels, complex128 projector without a texture. The fixed 2 GiB batch
    at 8 B per pixel asked for one 6.42 GiB program buffer with 6.6 GiB physically free and ran out of memory."""

    from relax.projection import projection
    from relax.runtime import memory_budget as budget

    gib = 1 << 30
    rotations, slots, trans, pixels = 4608, 41, 81, 278
    # The JAX fallback holds the rows in the projector's complex dtype and one temporary: 32 B a complex128 pixel.
    assert tomo_coarse._coarse_projection_bytes_per_pixel(
        [None], np.complex128, box_size=128, current_size=26, model_max_r=24
    ) == 32
    assert tomo_coarse._coarse_projection_bytes_per_pixel(
        [None], np.complex64, box_size=128, current_size=26, model_max_r=24
    ) == 16
    monkeypatch.setattr(projection, "relion_coarse_packed_rows_serve", lambda *a: True)
    assert tomo_coarse._coarse_projection_bytes_per_pixel(
        [object()], np.complex64, box_size=128, current_size=26, model_max_r=24
    ) == 8

    # The device share: a quarter of what the allocator can still hand out (6.6 GiB free beside half 1's state).
    monkeypatch.setattr(budget, "device_free_memory_bytes", lambda: int(6.6 * gib))
    monkeypatch.setattr(budget, "jax_allocator_free_memory_bytes", lambda: int(9.8 * gib))
    monkeypatch.setattr(budget, "jax_allocator_pool_free_bytes", lambda: 0)
    share = tomo_coarse._coarse_pass_budget(0).batch
    assert share == int(0.25 * 0.75 * 6.6 * gib)
    batches = tomo_coarse._coarse_batches(
        [rotations] * 600, n_slots=slots, n_trans=trans, n_pixels=pixels, projection_bytes_per_pixel=32,
        budget_bytes=share,
    )
    _, r_pad, p_pad, slot_block = batches[0]
    actual_bytes_per_image = r_pad * (trans * 4 + pixels * 32)  # the program buffer of Polar 413469, per image
    assert p_pad * slot_block * actual_bytes_per_image <= share
    # The old accounting (8 B per pixel, a fixed 2 GiB) let 4 particles x 41 slots through: the 6.42 GiB buffer,
    # which the 32 B model overstates by 1.1% (31.6 B a pixel measured).
    ((_, _, old_p_pad, old_slot_block), *_) = tomo_coarse._coarse_batches(
        [rotations] * 600, n_slots=slots, n_trans=trans, n_pixels=pixels, budget_bytes=2 * gib
    )
    assert old_p_pad * old_slot_block * actual_bytes_per_image == pytest.approx(6892094720, rel=0.02)
    # Unknown device readings keep the fixed cap.
    monkeypatch.setattr(budget, "device_free_memory_bytes", lambda: None)
    monkeypatch.setattr(budget, "jax_allocator_free_memory_bytes", lambda: None)
    assert tomo_coarse._coarse_pass_budget(0).batch == tomo_coarse._COARSE_BATCH_BYTES


def test_coarse_batch_plan_on_an_80gb_card_is_the_fixed_2gib_plan(monkeypatch):
    """On an 80 GB card the device-aware budget changes nothing: the texture path's rows take 8 B a pixel and a
    quarter of the free memory exceeds the fixed 2 GiB, so the batches are those of the fixed budget (the policy in
    docs/development/gpu_compatibility.md: small-card sizing is a no-op on large cards)."""

    from relax.projection import projection
    from relax.runtime import memory_budget as budget

    gib = 1 << 30
    monkeypatch.setattr(budget, "device_free_memory_bytes", lambda: 66 * gib)
    monkeypatch.setattr(budget, "jax_allocator_free_memory_bytes", lambda: 70 * gib)
    monkeypatch.setattr(budget, "jax_allocator_pool_free_bytes", lambda: 0)
    monkeypatch.setattr(projection, "relion_coarse_packed_rows_serve", lambda *a: True)
    assert tomo_coarse._coarse_pass_budget(0).batch == tomo_coarse._COARSE_BATCH_BYTES
    bytes_per_pixel = tomo_coarse._coarse_projection_bytes_per_pixel(
        [object()], np.complex64, box_size=128, current_size=48, model_max_r=24
    )
    assert bytes_per_pixel == 8
    for rotations, pixels in ((4608, 278), (36864, 921), (448, 2520)):
        planned = tomo_coarse._coarse_batches(
            [rotations] * 600, n_slots=41, n_trans=81, n_pixels=pixels,
            projection_bytes_per_pixel=bytes_per_pixel, budget_bytes=tomo_coarse._coarse_pass_budget(0).batch,
        )
        fixed = tomo_coarse._coarse_batches([rotations] * 600, n_slots=41, n_trans=81, n_pixels=pixels)
        assert [(list(u), r, p, s) for u, r, p, s in planned] == [(list(u), r, p, s) for u, r, p, s in fixed]


@pytest.mark.gpu
def test_the_per_image_kernel_matches_the_one_image_calls(gpu_device):
    """The batched per-image launch and one call per image score every (image, rotation, translation) alike."""

    import jax
    from recovar.cuda_backproject import custom_cuda_requested

    if not custom_cuda_requested():
        pytest.skip("custom CUDA is disabled")
    rng = np.random.default_rng(11)
    box, size, pad, max_r = 32, 32, 2, 16
    layout = tomo_coarse.coarse_score_layout((box, box), size, half_spectrum_scoring=True, square_window=False)
    n_px = int(layout.score_indices_np.size)
    side = 2 * max_r * pad + 3
    with jax.default_device(gpu_device):
        projector = jnp.asarray((rng.normal(size=(side,) * 3) + 1j * rng.normal(size=(side,) * 3)).astype(np.complex64))
        n_images, n_rot, n_trans = 5, 144, 81  # 128 main-segment rotations and a 16-rotation tail
        q = rng.normal(size=(n_images * n_rot, 4))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        w, x, y, z = q.T
        rotations = (
            np.stack(
                [
                    np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                    np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                    np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
                ],
                -2,
            )
            .astype(np.float32)
            .reshape(n_images, n_rot, 3, 3)
        )
        images = (rng.normal(size=(n_images, n_px)) + 1j * rng.normal(size=(n_images, n_px))).astype(np.complex64)
        weight = rng.uniform(0.1, 1.0, size=(n_images, n_px)).astype(np.float32)
        initial = rng.uniform(10.0, 20.0, size=n_images).astype(np.float32)
        angles = rng.uniform(-0.2, 0.2, size=(n_images, n_trans, 2)).astype(np.float32)
        batched = cuda_coarse.relion_coarse_diff2_projector_per_image_f32(
            projector,
            jnp.asarray(rotations),
            jnp.asarray(images),
            jnp.asarray(angles),
            jnp.asarray(weight),
            jnp.asarray(initial),
            layout.full_to_compact,
            current_size=size,
            physical_image_size=box,
            model_max_r=max_r,
            padding_factor=pad,
        )
        for b in range(n_images):
            single = tilt_image_coarse_diff2(
                projector,
                rotations[b],
                images[b],
                weight[b],
                initial[b],
                angles[b],
                layout,
                model_max_r=max_r,
                padding_factor=pad,
            )
            assert_matches(np.asarray(batched[b]), np.asarray(single), err_msg=f"image {b}")


def _gemm_fixture(rng, *, n_images=3, n_rot=256, n_trans=200, box=32):
    """A projector half for ``box`` (32 by default), random rotations, images, weights, d0 and translation phases
    on the GPU."""

    from relax.projection.projection import relion_projector_half_to_texture_full

    size, pad, max_r = box, 2, box // 2
    layout = tomo_coarse.coarse_score_layout((box, box), size, half_spectrum_scoring=True, square_window=False)
    n_px = int(layout.score_indices_np.size)
    side = 2 * max_r * pad + 3
    half = jnp.asarray(
        (rng.normal(size=(side, side, side // 2 + 1)) + 1j * rng.normal(size=(side, side, side // 2 + 1))).astype(np.complex64)
    )
    q = rng.normal(size=(n_images * n_rot, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    rotations = np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    ).astype(np.float32)
    return dict(
        box=box, pad=pad, max_r=max_r, layout=layout, half=half,
        full=relion_projector_half_to_texture_full(half).astype(jnp.complex64),
        rotations=rotations, n_images=n_images, n_rot=n_rot,
        images=(rng.normal(size=(n_images, n_px)) + 1j * rng.normal(size=(n_images, n_px))).astype(np.complex64),
        weight=rng.uniform(0.1, 1.0, size=(n_images, n_px)).astype(np.float32),
        initial=rng.uniform(10.0, 20.0, size=n_images).astype(np.float32),
        angles=rng.uniform(-0.2, 0.2, size=(n_images, n_trans, 2)).astype(np.float32),
    )


@pytest.mark.gpu
@pytest.mark.parametrize("box", [32, 72], ids=["2_pixel_blocks", "11_pixel_blocks"])
def test_the_projection_kernels_model_energy_is_the_weighted_power_of_its_rows(gpu_device, box):
    """The packed projection with its model energy: the rows are the packed projection's, and each row's energy is
    sum_j w_j (Re_j^2 + Im_j^2) of its stored values with its own image's weights (float64 reference), for
    images whose rotations do not fill whole blocks of four (rows of two images in one block, a last short block).
    """

    import jax
    from recovar.cuda_backproject import custom_cuda_requested

    from relax.projection.projection import project_relion_coarse_packed_rows, project_relion_coarse_packed_rows_energy

    if not custom_cuda_requested():
        pytest.skip("custom CUDA is disabled")
    n_images, n_rot = 3, 130
    with jax.default_device(gpu_device):
        f = _gemm_fixture(np.random.default_rng(41), n_images=n_images, n_rot=n_rot, n_trans=3, box=box)
        layout = f["layout"]
        texture = tomo_coarse._coarse_capacity_texture(
            f["half"], layout, model_max_r=f["max_r"], padding_factor=f["pad"], class_index=0
        )
        assert texture is not None
        args = (texture, jnp.asarray(f["rotations"]), jnp.asarray(layout.score_indices_np, jnp.int32))
        static = dict(image_shape=tuple(layout.image_shape), projector_output_size=int(layout.current_size))
        packed = np.asarray(project_relion_coarse_packed_rows(*args, **static))
        fused, energy = project_relion_coarse_packed_rows_energy(*args, jnp.asarray(f["weight"]), **static)
        fused, energy = np.asarray(fused), np.asarray(energy)
        texture.close_after(jnp.asarray(energy))
    assert_matches(fused, packed, strict=True)
    n_px = packed.shape[1] // 2
    rows = packed.astype(np.float64).reshape(n_images, n_rot, 2 * n_px)
    power = rows[..., :n_px] ** 2 + rows[..., n_px:] ** 2
    expected = np.einsum("ip,irp->ir", f["weight"].astype(np.float64), power).reshape(-1)
    assert energy.shape == (n_images * n_rot,) and energy.dtype == np.float32
    assert_matches(energy, expected)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "persistent_texture, kernel_energy",
    [(False, False), (True, False), (True, True)],
    ids=["complex_rows", "packed_rows", "packed_rows_kernel_energy"],
)
def test_the_gemm_scorer_follows_relions_direct_square_kernel(gpu_device, persistent_texture, kernel_energy):
    """The coarse pass's GEMM scorer against the fused direct-square kernel, image by image, on the
    complex projections and on the packed rows a persistent texture projects, with the model energy from the
    scorer's product or from the projection kernel (the slot-block programs' route).

    The two round differently (the expansion d0 + 0.5 A + 0.5 C - X against RELION's sum of
    squares), so they agree to a few float32 ULP of the terms' scale, not bitwise: 1.8e-6 of
    the largest diff2 measured on this fixture (job 14856295), checked at 1e-5.
    """

    import jax
    from recovar.cuda_backproject import custom_cuda_requested

    if not custom_cuda_requested():
        pytest.skip("custom CUDA is disabled")
    with jax.default_device(gpu_device):
        f = _gemm_fixture(np.random.default_rng(17))
        layout = f["layout"]
        texture = (
            tomo_coarse._coarse_capacity_texture(
                f["half"], layout, model_max_r=f["max_r"], padding_factor=f["pad"], class_index=0
            )
            if persistent_texture else None
        )
        assert (texture is not None) == persistent_texture
        score_indices = jnp.asarray(layout.score_indices_np, jnp.int32)
        projected, model_energy = tomo_coarse_gemm._score_window_projections(
            f["half"], jnp.asarray(f["rotations"]), score_indices, image_shape=tuple(layout.image_shape),
            current_size=int(layout.current_size), model_max_r=f["max_r"], padding_factor=f["pad"], texture=texture,
            weight=jnp.asarray(f["weight"]),
        )
        projected = projected.reshape(f["n_images"], f["n_rot"], -1)
        assert projected.dtype == (jnp.float32 if persistent_texture else jnp.complex64)
        assert (model_energy is not None) == persistent_texture
        gemm, bounds = tomo_coarse_gemm._images_coarse_gemm_diff2(
            projected, jnp.asarray(f["images"]), jnp.asarray(f["weight"]), jnp.asarray(f["initial"]),
            jnp.asarray(f["angles"]), score_indices, model_energy if kernel_energy else None,
            image_shape=(f["box"], f["box"]),
        )
        gemm, bounds = np.asarray(gemm), np.asarray(bounds)
        if texture is not None:
            texture.close()
        rotations = f["rotations"].reshape(f["n_images"], f["n_rot"], 3, 3)
        for b in range(f["n_images"]):
            direct = np.asarray(
                tilt_image_coarse_diff2(
                    f["full"], rotations[b], f["images"][b], f["weight"][b], f["initial"][b], f["angles"][b], layout,
                    model_max_r=f["max_r"], padding_factor=f["pad"],
                )
            )
            assert_matches(gemm[b].T, direct, rtol=1e-5, err_msg=f"image {b}")
            # The re-score at the cut relies on this bound of the GEMM's distance from the direct square.
            bound = float(bounds[b])
            worst = float(np.max(np.abs(gemm[b].T - direct)))
            assert 0.0 < worst <= bound, f"image {b}"
            # The worst case over the pixels: measured 18 float32 units of the diff2 here, the bound about 2000.
            assert bound <= 1e-3 * float(np.max(direct))


@pytest.mark.gpu
@pytest.mark.parametrize("persistent_texture", [False, True], ids=["complex_rows", "packed_rows"])
def test_frame_programs_score_as_the_slot_block_programs(gpu_device, persistent_texture):
    """Particles whose images share frames (one left matrix per tilt frame): projecting each frame once and scoring
    its images in one call (coarse_gemm_frame, frames in slot order, padded calls) gives each particle the totals
    and GEMM error bounds of the slot-block programs that project every image (coarse_gemm_slot_block), including a
    particle with a padded slot."""

    import jax
    from recovar.cuda_backproject import custom_cuda_requested

    if not custom_cuda_requested():
        pytest.skip("custom CUDA is disabled")
    n_group, n_slots, n_frames, n_rot, n_trans, slot_block = 3, 4, 5, 256, 37, 2
    with jax.default_device(gpu_device):
        f = _gemm_fixture(np.random.default_rng(37), n_images=n_group * n_slots, n_rot=n_rot, n_trans=n_trans)
        layout = f["layout"]
        texture = (
            tomo_coarse._coarse_capacity_texture(
                f["half"], layout, model_max_r=f["max_r"], padding_factor=f["pad"], class_index=0
            )
            if persistent_texture else None
        )
        frame_rotations = jnp.asarray(f["rotations"][: n_frames * n_rot].reshape(n_frames, n_rot, 3, 3))
        # Particle p's slot s is frame s + p (particle 1 starts one frame later); particle 2 has three images.
        frame_of = np.add.outer(np.arange(n_group), np.arange(n_slots)) % n_frames
        valid = np.ones((n_group, n_slots), bool)
        valid[2, 3] = False
        shape = (n_group, n_slots)
        unshifted = jnp.asarray(np.where(valid[..., None], f["images"].reshape(*shape, -1), 0))
        weight = jnp.asarray(np.where(valid[..., None], f["weight"].reshape(*shape, -1), 0))
        initial = jnp.asarray(np.where(valid, f["initial"].reshape(shape), 0))
        angles = jnp.asarray(f["angles"].reshape(*shape, n_trans, 2))
        score_indices = jnp.asarray(layout.score_indices_np, jnp.int32)
        static = dict(
            image_shape=(f["box"], f["box"]), current_size=int(layout.current_size), model_max_r=f["max_r"],
            padding_factor=f["pad"], texture=texture,
        )
        rotations = jnp.where(
            jnp.asarray(valid)[..., None, None, None], frame_rotations[jnp.asarray(frame_of)], 0.0
        )  # [G, S, R, 3, 3], as the batch path lays them out
        total = jnp.zeros((n_group, n_trans, n_rot), jnp.float32)
        error = jnp.zeros(n_group, jnp.float32)
        for first in range(0, n_slots, slot_block):
            total, error = tomo_coarse_gemm.coarse_gemm_slot_block(
                total, error, f["half"], rotations, unshifted, weight, initial, angles, score_indices,
                jnp.int32(first), count=slot_block, **static,
            )
        frame_total = jnp.zeros((n_group, n_trans, n_rot), jnp.float32)
        bounds = jnp.zeros((n_group, n_slots), jnp.float32)
        for frame in range(n_frames):  # every particle's slots follow the frame order
            units, slots = np.nonzero((frame_of == frame) & valid)
            n_pad = 4  # padded entries (unit index = the group size) change nothing
            unit_index = np.full(n_pad, n_group, np.int32)
            slot_index = np.zeros(n_pad, np.int32)
            unit_index[: units.size], slot_index[: units.size] = units, slots
            frame_total, bounds = tomo_coarse_gemm.coarse_gemm_frame(
                frame_total, bounds, f["half"], frame_rotations, jnp.int32(frame), unshifted, weight, initial, angles,
                score_indices, jnp.asarray(unit_index), jnp.asarray(slot_index), **static,
            )
        frame_error = tomo_coarse_gemm.slot_block_error(bounds, slot_block=slot_block)
        if texture is not None:
            texture.close_after(frame_total)  # the programs used it inside a trace: close after their result
    assert_matches(np.asarray(frame_total), np.asarray(total))
    assert_matches(np.asarray(frame_error), np.asarray(error))
    assert np.count_nonzero(np.asarray(total)) == total.size


@pytest.mark.gpu
@pytest.mark.parametrize("box", [12, 32, 72], ids=["72_pixels", "544_pixels", "2664_pixels"])
def test_the_gemm_error_bound_holds_over_a_million_samples(gpu_device, box):
    """The exact cut's bound on the GEMM scorer's distance from the direct square, over more than 1e6 samples
    at three score-window sizes: the largest observed difference stays under it."""

    import jax
    from recovar.cuda_backproject import custom_cuda_requested

    if not custom_cuda_requested():
        pytest.skip("custom CUDA is disabled")
    with jax.default_device(gpu_device):
        n_images, n_rot, n_trans = 3, 2816, 120
        f = _gemm_fixture(np.random.default_rng(29 + box), n_images=n_images, n_rot=n_rot, n_trans=n_trans, box=box)
        layout = f["layout"]
        projected = _coarse_gemm_projections(
            f["half"], jnp.asarray(f["rotations"]), layout, model_max_r=f["max_r"], padding_factor=f["pad"], texture=None
        ).reshape(n_images, n_rot, -1)
        gemm, bound = tomo_coarse_gemm._images_coarse_gemm_diff2(
            projected, jnp.asarray(f["images"]), jnp.asarray(f["weight"]), jnp.asarray(f["initial"]),
            jnp.asarray(f["angles"]), jnp.asarray(layout.score_indices_np, jnp.int32), image_shape=(box, box),
        )
        gemm, bound = np.asarray(gemm), np.asarray(bound)
        rotations = f["rotations"].reshape(n_images, n_rot, 3, 3)
        assert gemm.size >= 1_000_000
        for b in range(n_images):
            direct = np.asarray(
                tilt_image_coarse_diff2(
                    f["full"], rotations[b], f["images"][b], f["weight"][b], f["initial"][b], f["angles"][b], layout,
                    model_max_r=f["max_r"], padding_factor=f["pad"],
                )
            )
            worst = float(np.max(np.abs(gemm[b].T - direct)))
            units = worst / float(np.spacing(np.float32(np.max(direct))))
            print(f"box {box} pixels {layout.score_indices_np.size} image {b}: worst {worst:.4g} = {units:.1f} float32 units of the diff2; bound {bound[b]:.4g}")
            assert worst <= bound[b], f"image {b}"


@pytest.mark.gpu
@pytest.mark.parametrize("persistent_texture", [False, True], ids=["complex_rows", "packed_rows"])
def test_the_bound_from_the_scorers_model_energy_leaves_the_diff2_bitwise(gpu_device, persistent_texture):
    """The per-image diff2 is bitwise the scorer called without its model energy, and the bound takes the largest
    model energy of the scorer's GEMM: the bound shares the scores' program and changes no score."""

    import jax
    from recovar.cuda_backproject import custom_cuda_requested

    from relax.cuda import translate as cuda_translate
    from relax.scoring.coarse_kernels import relion_coarse_gaussian_gemm_scores_jit
    from relax.scoring.exact_cut import gemm_error_bound

    if not custom_cuda_requested():
        pytest.skip("custom CUDA is disabled")
    with jax.default_device(gpu_device):
        f = _gemm_fixture(np.random.default_rng(31), n_images=4, n_rot=640, n_trans=57)
        layout = f["layout"]
        texture = (
            tomo_coarse._coarse_capacity_texture(
                f["half"], layout, model_max_r=f["max_r"], padding_factor=f["pad"], class_index=0
            )
            if persistent_texture else None
        )
        projected = _coarse_gemm_projections(
            f["half"], jnp.asarray(f["rotations"]), layout, model_max_r=f["max_r"], padding_factor=f["pad"],
            texture=texture,
        ).reshape(f["n_images"], f["n_rot"], -1)
        images, weight = jnp.asarray(f["images"]), jnp.asarray(f["weight"])
        initial, angles = jnp.asarray(f["initial"]), jnp.asarray(f["angles"])
        score_indices = jnp.asarray(layout.score_indices_np, jnp.int32)
        image_shape = (f["box"], f["box"])
        diff2, bound = tomo_coarse_gemm._images_coarse_gemm_diff2(
            projected, images, weight, initial, angles, score_indices, image_shape=image_shape
        )

        @jax.jit
        def scorer_alone(projected, images, weight, initial, angles):
            n_trans = angles.shape[1]
            shifted = cuda_translate.relion_translate_score_f32(images, angles, score_indices, image_shape).reshape(
                images.shape[0], n_trans, -1
            )

            def one(reference, shifted, weight, initial):
                scores = relion_coarse_gaussian_gemm_scores_jit(
                    reference, None, shifted[None], weight[None], initial.reshape(1), jnp.int32(1),
                    n_images=1, n_trans=n_trans, image_shape=image_shape, volume_shape=(1, 1, 1),
                    translation_major=True,
                )
                return -scores[0]

            return jax.vmap(one)(projected, shifted, weight, initial)

        np.testing.assert_array_equal(np.asarray(diff2), np.asarray(scorer_alone(projected, images, weight, initial, angles)))
        packed = projected if not jnp.iscomplexobj(projected) else jnp.concatenate([projected.real, projected.imag], -1)
        n_px = packed.shape[-1] // 2
        model_energy = jnp.einsum(
            "bp,brp->br", weight, packed[..., :n_px] ** 2 + packed[..., n_px:] ** 2, precision=jax.lax.Precision.HIGHEST
        )
        image_energy = jnp.sum(jnp.abs(images) ** 2 * weight, axis=-1)
        expected = gemm_error_bound(jnp.max(model_energy, axis=1), image_energy, initial, n_px)
        assert_matches(np.asarray(bound), np.asarray(expected), rtol=1e-5)
        if texture is not None:
            texture.close()


@pytest.mark.gpu
@pytest.mark.parametrize("block_rows", [None, 2], ids=["one_block", "blocks_of_two_rows"])
def test_direct_rows_are_the_one_image_kernels_rows_in_slot_order(gpu_device, monkeypatch, block_rows):
    """A particle's re-scored rotations: any subset of rows, at any position in the call, gives the full
    direct-square pass's values for those rows, the images added in slot order as RELION adds them
    (``total = (total + Xi2_k / 2) + s_k``); scored in blocks of rows (the kernel's int32 output count) they are
    the one-block values."""

    import jax
    from recovar.cuda_backproject import custom_cuda_requested

    from relax.scoring import exact_cut

    if not custom_cuda_requested():
        pytest.skip("custom CUDA is disabled")
    with jax.default_device(gpu_device):
        n_particles, n_slots, n_rot = 2, 3, 256
        f = _gemm_fixture(np.random.default_rng(23), n_images=n_particles * n_slots, n_rot=n_rot, n_trans=200)
        layout = f["layout"]
        rotations = f["rotations"].reshape(n_particles, n_slots, n_rot, 3, 3)
        full = np.zeros((n_particles, n_rot, 200), np.float32)
        for p_ in range(n_particles):
            total = np.zeros((n_rot, 200), np.float32)
            for s_ in range(n_slots):
                b = p_ * n_slots + s_
                image = np.asarray(
                    tilt_image_coarse_diff2(
                        f["full"], rotations[p_, s_], f["images"][b], f["weight"][b], np.float32(0.0), f["angles"][b],
                        layout, model_max_r=f["max_r"], padding_factor=f["pad"],
                    )
                )
                total = (total + np.float32(f["initial"][b])) + image
            full[p_] = total
        ids = np.array([[250, 3, 129, 128, 0], [7, 255, 64, 200, 127]])

        def direct_rows():
            return np.asarray(
                tomo_coarse.direct_rows_diff2(
                    f["full"],
                    jnp.asarray(np.take_along_axis(rotations, ids[:, None, :, None, None], axis=2)),
                    jnp.asarray(f["images"]).reshape(n_particles, n_slots, -1),
                    jnp.asarray(f["weight"]).reshape(n_particles, n_slots, -1),
                    jnp.asarray(f["initial"]).reshape(n_particles, n_slots),
                    jnp.asarray(f["angles"]).reshape(n_particles, n_slots, 200, 2),
                    layout.full_to_compact, current_size=layout.current_size, physical_image_size=f["box"],
                    model_max_r=f["max_r"], padding_factor=f["pad"],
                )
            )

        one_block = direct_rows()
        if block_rows is not None:
            monkeypatch.setattr(exact_cut, "DIRECT_ROWS_BLOCK_ELEMENTS", n_particles * n_slots * 200 * block_rows)
            rows = direct_rows()
            assert_matches(rows, one_block)
        assert_matches(one_block, np.take_along_axis(full, ids[:, :, None], axis=1))


@pytest.mark.gpu
@pytest.mark.parametrize("n_rot", [300, 301, 303], ids=["whole_blocks", "one_in_the_last_block", "three_in_the_last_block"])
def test_packed_rows_are_the_complex_projections_split(gpu_device, n_rot):
    """The persistent texture's packed ``[Re | Im]`` rows are the complex coarse projections (the same texture
    reads and dense scale, in one kernel), so the GEMM's reference operand does not change, also for a last
    block holding fewer than the kernel's rotations per block."""

    import jax
    from recovar.cuda_backproject import custom_cuda_requested

    if not custom_cuda_requested():
        pytest.skip("custom CUDA is disabled")
    with jax.default_device(gpu_device):
        f = _gemm_fixture(np.random.default_rng(23), n_images=1, n_rot=n_rot)
        layout, rotations = f["layout"], jnp.asarray(f["rotations"])
        texture = tomo_coarse._coarse_capacity_texture(
            f["half"], layout, model_max_r=f["max_r"], padding_factor=f["pad"], class_index=0
        )
        assert texture is not None
        packed = np.asarray(
            _coarse_gemm_projections(
                f["half"], rotations, layout, model_max_r=f["max_r"], padding_factor=f["pad"], texture=texture
            )
        )
        texture.close()
        complex_rows = np.asarray(
            _coarse_gemm_projections(f["half"], rotations, layout, model_max_r=f["max_r"], padding_factor=f["pad"])
        )
    assert packed.shape == (complex_rows.shape[0], 2 * complex_rows.shape[1])
    assert np.count_nonzero(complex_rows) > complex_rows.size // 2
    assert_matches(packed, np.concatenate([complex_rows.real, complex_rows.imag], axis=-1))


@pytest.mark.gpu
def test_the_class_texture_is_refilled_for_the_next_pass(gpu_device):
    """A pass's class texture is the previous pass's owner refilled with the new projector (the compiled coarse
    programs capture its handle), and it projects what a texture staged fresh from that projector projects."""

    import jax
    from recovar.cuda_backproject import custom_cuda_requested

    if not custom_cuda_requested():
        pytest.skip("custom CUDA is disabled")
    with jax.default_device(gpu_device):
        f = _gemm_fixture(np.random.default_rng(29), n_images=1, n_rot=200)
        layout, rotations = f["layout"], jnp.asarray(f["rotations"])
        kwargs = dict(model_max_r=f["max_r"], padding_factor=f["pad"])
        first = tomo_coarse._coarse_capacity_texture(f["half"], layout, class_index=0, **kwargs)
        used = _coarse_gemm_projections(f["half"], rotations, layout, texture=first, **kwargs)
        tomo_coarse._CLASS_TEXTURES[0] = (first, used)  # the pass's end (particle_coarse_supports)
        changed = jnp.asarray(f["half"] * (0.5 + 0.25j))
        second = tomo_coarse._coarse_capacity_texture(changed, layout, class_index=0, **kwargs)
        assert second is first
        refilled = np.asarray(_coarse_gemm_projections(changed, rotations, layout, texture=second, **kwargs))
        tomo_coarse._CLASS_TEXTURES.pop(0)[0].close_after(jnp.asarray(refilled))
        fresh = tomo_coarse._coarse_capacity_texture(changed, layout, class_index=1, **kwargs)
        expected = np.asarray(_coarse_gemm_projections(changed, rotations, layout, texture=fresh, **kwargs))
        tomo_coarse._CLASS_TEXTURES.pop(1)[0].close_after(jnp.asarray(expected))
    assert fresh is not first
    # Both textures hold the same projector and interpolate it with the same kernel: the default float32 band
    # (no bitwise float asserts, user rule 2026-09-24).
    assert_matches(refilled, expected)


def test_padded_particles_with_their_own_rotation_priors_cut_as_one_by_one():
    """A local search's batch: per-particle rotation priors, padded rotations (a -inf prior) never significant."""

    rng = np.random.default_rng(9)
    n_trans, counts = 6, (10, 7, 4)
    r_pad = max(counts)
    diff2 = rng.uniform(1000.0, 1004.0, size=(3, r_pad, n_trans)).astype(np.float32)
    priors = [np.log(rng.uniform(0.5, 1.0, size=n)).astype(np.float32) for n in counts]
    offset_prior = np.log(rng.uniform(0.1, 1.0, size=(3, n_trans))).astype(np.float32)
    padded_prior = np.full((3, r_pad), -np.inf, dtype=np.float32)
    for p, n in enumerate(counts):
        diff2[p, n:] = diff2[p, n - 1]  # padded rotations repeat the particle's last one
        padded_prior[p, :n] = priors[p]
    batched = tomo_coarse.particle_coarse_significance(
        diff2, padded_prior, offset_prior, adaptive_fraction=0.999, max_significants=None
    )
    for p, n in enumerate(counts):
        alone = tomo_coarse.particle_coarse_significance(
            diff2[p : p + 1, :n], priors[p], offset_prior[p : p + 1], adaptive_fraction=0.999, max_significants=None
        )
        mask = np.asarray(batched["mask"][p]).reshape(r_pad, n_trans)
        assert not mask[n:].any()
        np.testing.assert_array_equal(mask[:n].reshape(-1), np.asarray(alone["mask"][0]))
        assert_matches(np.asarray(batched["pmax"][p]), np.asarray(alone["pmax"][0]))


def _fake_coarse_kernels(monkeypatch, rng, flushes):
    """Stand-ins for the coarse pass's device kernels (diff2 a function of the scorer matrices and the class's
    "projector" value alone, so the classes compete and a re-scored rotation keeps its value) and a capture of
    every significance call. Returns ``(captured, layout)``; ``flush_per_particle`` cuts one particle at a time."""

    def fake_block(total, error, class_value, rotations, unshifted, weight, initial, angles, score_indices, first, *, count, **kwargs):
        block = slice(int(first), int(first) + count)
        rotations, weight, initial, angles = rotations[:, block], weight[:, block], initial[:, block], angles[:, block]
        # Each image adds its initial diff2, its rotation's sum, its angle and its weight; each class shifts its
        # diff2 by its own "projector" value times the rotation index, so the classes compete.
        image = image_diff2(class_value, rotations, weight, initial, angles)
        for slot in range(image.shape[1]):
            total = total + image[:, slot].swapaxes(1, 2)  # the pass accumulates translation-major
        return total, error

    def image_diff2(class_value, rotations, weight, initial, angles):
        # [P, S, R, T], a function of the matrices alone (a re-scored rotation is not at its grid position).
        trace = jnp.sum(rotations, axis=(3, 4))
        per_rot = trace + class_value * jnp.sin(jnp.float32(40.0) * trace)
        return initial[:, :, None, None] + per_rot[:, :, :, None] + angles[:, :, None, :, 0] + jnp.sum(weight, axis=2)[:, :, None, None]

    def direct(class_value, rotations, unshifted, weight, initial, angles, full_to_compact, **kwargs):
        # The rotations the cut leaves undecided are scored again: here by the same values.
        image = image_diff2(jnp.real(class_value), rotations, weight, initial, angles)
        total = jnp.zeros(image.shape[:1] + image.shape[2:], jnp.float32)
        for slot in range(image.shape[1]):
            total = total + image[:, slot]
        return total

    def operands(experiment_dataset, image_start, image_stop, layout, **kwargs):
        n = int(image_stop) - int(image_start)
        return (
            jnp.zeros((n, 4), jnp.complex64),
            jnp.asarray(rng.uniform(size=(n, 4)), jnp.float32),
            jnp.asarray(rng.uniform(1.0, 2.0, size=n), jnp.float32),
        )

    captured = []
    significance = tomo_coarse.particle_coarse_significance

    def capture(diff2, rotation_prior, translation_prior, **kwargs):
        captured.append((np.asarray(diff2), np.asarray(rotation_prior), np.asarray(translation_prior)))
        return significance(diff2, rotation_prior, translation_prior, **kwargs)

    def pass1_rotations(eulers, random_perturbation, angular_sampling_deg, *, left_matrices):
        # [S, R, 3, 3]: RELION's device matrices need the RELION binding; any per-image stand-in will do.
        # Traceable: the exact cut builds its rotations inside a program.
        base = jnp.asarray(eulers, dtype=jnp.float32)[:, :, None] * jnp.eye(3, dtype=jnp.float32)[None, :, :] / 180.0
        # Each image's matrices depend on its own left matrix (not on its place in the call).
        left = jnp.asarray(left_matrices, dtype=jnp.float32)
        return base[None] + left[:, 0, 0][:, None, None, None]

    from relax.sampling import device_rotations

    monkeypatch.setattr(device_rotations, "relion_adaptive_pass1_rotations", pass1_rotations)
    monkeypatch.setattr(
        device_rotations,
        "pass1_rotations_left_f32",
        lambda eulers, right_f32, left_matrices: pass1_rotations(eulers, 0.0, 0.0, left_matrices=left_matrices),
    )
    from relax.projection import projection

    monkeypatch.setattr(projection, "relion_projector_half_to_texture_full", lambda value: jnp.asarray(value))
    monkeypatch.setattr(tomo_coarse, "direct_rows_diff2", direct)
    monkeypatch.setattr(tomo_coarse, "coarse_gemm_slot_block", fake_block)
    monkeypatch.setattr(tomo_coarse, "_all_image_coarse_operands", operands)
    monkeypatch.setattr(tomo_coarse, "particle_coarse_significance", capture)
    if flushes == "flush_per_particle":
        batches = tomo_coarse._coarse_batches
        monkeypatch.setattr(tomo_coarse, "_coarse_batches", lambda *a, **k: batches(*a, **{**k, "budget_bytes": 1}))
        monkeypatch.setattr(tomo_coarse, "_SIGNIFICANCE_BATCH_BYTES", 1)
    layout = tomo_coarse.CoarseScoreLayout(
        (8, 8), 6, np.arange(4), jnp.ones(4, bool), jnp.zeros(1, jnp.int32), jnp.ones(40)
    )
    return captured, layout


@pytest.mark.parametrize("flushes", ["one_flush", "flush_per_particle"])
def test_k_class_particles_cut_their_weights_over_every_class_jointly(monkeypatch, flushes):
    """Class3D: every image is scored against each class; the particle's (class, rotation, translation) weights
    are cut once (convertAllSquaredDifferencesToWeights sorts all classes' weights together), and each cell
    goes back to its class's support. One particle per batch and flush exercises the deferred readback and
    the compaction-size guess (exact first, then guessed, recompacted on overflow)."""

    rng = np.random.default_rng(11)
    offsets = np.array([0, 2, 5, 6])
    n_images, n_rot, n_trans = int(offsets[-1]), 5, 7

    captured, layout = _fake_coarse_kernels(monkeypatch, rng, flushes)
    class_prior = np.log(rng.uniform(0.2, 1.0, size=(2, n_rot))).astype(np.float32)
    supports, pmax, counts = tomo_coarse.particle_coarse_supports(
        None,
        unit_image_offsets=offsets,
        image_projections=np.tile(np.eye(3), (n_images, 1, 1)),
        image_left=np.eye(3)[None] * (1.0 + np.arange(n_images))[:, None, None],
        unit_old_offsets_px=np.zeros((3, 3)),
        coarse_eulers_deg=rng.uniform(0.0, 180.0, size=(n_rot, 3)),
        random_perturbation=0.0,
        angular_sampling_deg=15.0,
        coarse_translations_px=rng.uniform(-1.0, 1.0, size=(n_trans, 3)),
        projector_half=(jnp.float32(0.3), jnp.float32(-0.2)),
        layout=layout,
        noise_variance_half=None,
        rotation_log_prior=class_prior,
        unit_translation_log_prior=np.log(rng.uniform(0.1, 1.0, size=(3, n_trans))).astype(np.float32),
        adaptive_fraction=0.999,
        max_significants=None,
        model_max_r=3,
        padding_factor=2,
        box_size=8,
    )
    assert len(supports) == 2 and all(len(class_supports) == 3 for class_supports in supports)
    # A batch with an undecided rotation is cut again (same translation prior): its last cut is published.
    last_cut = {}
    for call in captured:
        last_cut[call[2].tobytes()] = call
    captured = list(last_cut.values())
    assert len(captured) == (1 if flushes == "one_flush" else 3)
    diff2 = np.concatenate([c[0] for c in captured])
    translation_prior = np.concatenate([c[2] for c in captured])
    rotation_prior = captured[0][1]
    r_pad = diff2.shape[1] // 2
    # The classes' rotations side by side, each padded at -inf.
    assert_matches(rotation_prior.reshape(2, r_pad)[:, :n_rot], class_prior)
    assert np.all(np.isneginf(rotation_prior.reshape(2, r_pad)[:, n_rot:]))
    n_significant = 0
    for p in range(3):
        log_weight = (rotation_prior[:, None] + translation_prior[p][None, :]) + (diff2[p].min() - diff2[p])
        expected = _relion_coarse_cut(log_weight.astype(np.float32).reshape(-1), 0.999).reshape(2, r_pad, n_trans)
        for k in range(2):
            np.testing.assert_array_equal(supports[k][p], np.flatnonzero(expected[k]))
            n_significant += supports[k][p].size
        # One count per particle over both classes, as RELION records it (before ties).
        assert counts[p] == _relion_coarse_count(log_weight.astype(np.float32).reshape(-1), 0.999)
    assert n_significant > 3  # the cut keeps more than each particle's winner
    assert pmax.shape == (3,) and counts.shape == (3,) and counts.dtype == np.int32


@pytest.mark.parametrize("flushes", ["one_flush", "flush_per_particle"])
def test_k_class_particles_with_their_own_rotations_are_cut_over_the_scored_classes(monkeypatch, flushes):
    """Class3D under an orientational prior (--sigma_ang, --sigma_tilt): every class is scored on the particle's
    own rotations, the particle's prior over them is the whole orientation weight of each scored class (no
    pdf_class, acc_ml_optimiser_impl.h:1960-1972), the weights are cut once over (class, rotation, translation),
    and a class RELION skips (pdf_class 0) keeps no sample. The supports index the grid's rotations."""

    rng = np.random.default_rng(13)
    offsets = np.array([0, 2, 5, 6])
    n_images, n_rot, n_trans, n_classes, empty = int(offsets[-1]), 6, 7, 3, 1
    captured, layout = _fake_coarse_kernels(monkeypatch, rng, flushes)
    unit_rotations = [np.array([0, 2, 3]), np.array([1, 5]), np.arange(n_rot)]
    unit_priors = [np.log(rng.uniform(0.2, 1.0, size=rows.size)).astype(np.float32) for rows in unit_rotations]
    supports, pmax, counts = tomo_coarse.particle_coarse_supports(
        None,
        unit_image_offsets=offsets,
        image_projections=np.tile(np.eye(3), (n_images, 1, 1)),
        image_left=np.eye(3)[None] * (1.0 + np.arange(n_images))[:, None, None],
        unit_old_offsets_px=np.zeros((3, 3)),
        coarse_eulers_deg=rng.uniform(0.0, 180.0, size=(n_rot, 3)),
        random_perturbation=0.0,
        angular_sampling_deg=15.0,
        coarse_translations_px=rng.uniform(-1.0, 1.0, size=(n_trans, 3)),
        projector_half=(jnp.float32(0.3), jnp.float32(5.0), jnp.float32(-0.2)),
        layout=layout,
        noise_variance_half=None,
        rotation_log_prior=None,
        unit_translation_log_prior=np.log(rng.uniform(0.1, 1.0, size=(3, n_trans))).astype(np.float32),
        adaptive_fraction=0.999,
        max_significants=None,
        model_max_r=3,
        padding_factor=2,
        box_size=8,
        unit_rotation_ids=unit_rotations,
        unit_rotation_log_priors=unit_priors,
        empty_classes=(empty,),
    )
    assert len(supports) == n_classes and all(len(class_supports) == 3 for class_supports in supports)
    # A batch with an undecided rotation is cut again (same translation prior): its last cut is published.
    last_cut = {}
    for call in captured:
        last_cut[call[2].tobytes()] = call
    captured = list(last_cut.values())
    assert len(captured) == (1 if flushes == "one_flush" else 3)
    diff2 = np.concatenate([c[0] for c in captured])
    rotation_prior = np.concatenate([c[1] for c in captured])
    translation_prior = np.concatenate([c[2] for c in captured])
    r_pad = diff2.shape[1] // n_classes
    n_significant = 0
    for p, rows in enumerate(unit_rotations):
        # The particle's own prior under every scored class; the skipped class and the padding at -inf.
        prior = rotation_prior[p].reshape(n_classes, r_pad)
        for k in range(n_classes):
            if k == empty:
                assert np.all(np.isneginf(prior[k]))
            else:
                assert_matches(prior[k, : rows.size], unit_priors[p])
                assert np.all(np.isneginf(prior[k, rows.size :]))
        scored = np.isfinite(rotation_prior[p])
        log_weight = (rotation_prior[p][:, None] + translation_prior[p][None, :]) + (
            diff2[p][scored].min() - diff2[p]
        )
        expected = _relion_coarse_cut(log_weight.astype(np.float32).reshape(-1), 0.999).reshape(
            n_classes, r_pad, n_trans
        )
        assert not expected[empty].any() and not expected[:, rows.size :].any()
        for k in range(n_classes):
            local_rotation, translation = np.nonzero(expected[k])
            np.testing.assert_array_equal(supports[k][p], rows[local_rotation] * n_trans + translation)
            n_significant += supports[k][p].size
        assert counts[p] == _relion_coarse_count(log_weight.astype(np.float32).reshape(-1), 0.999)
    assert all(supports[empty][p].size == 0 for p in range(3))
    assert n_significant > 3  # the cut keeps more than each particle's winner
    assert pmax.shape == (3,)


def test_near_cut_rows_take_a_tenth_of_the_gemm_bound(monkeypatch):
    """The cut's margin takes _GEMM_ERROR_MARGIN_SHARE of the GEMM bound: a rotation 0.05 bounds from the cut is
    scored again, one 0.5 bounds away keeps its GEMM score (the full bound would score it again), relax#43."""

    bound = np.float32(100.0)
    diff2 = np.asarray([1000.0, 1000.0 + 0.05 * bound, 1000.0 + 0.5 * bound, 1000.0 + 5 * bound], np.float32)
    diff2 = diff2.reshape(1, 4, 1)
    values = (diff2.min() - diff2).reshape(1, -1).astype(np.float32)
    mask = np.zeros((1, 4), bool)
    mask[0, 0] = True  # max_significants 1 keeps the best sample
    args = (jnp.asarray(diff2), jnp.asarray(values), jnp.asarray(mask), jnp.asarray([1], np.int32),
            jnp.asarray([bound]), jnp.asarray([1]))
    rows, _, _ = tomo_coarse._near_cut_rows(*args, max_significants=1, adaptive_fraction=0.999)
    np.testing.assert_array_equal(np.asarray(rows)[0], [True, True, False, False])
    monkeypatch.setattr(tomo_coarse, "_GEMM_ERROR_MARGIN_SHARE", 1.0)
    tomo_coarse._near_cut_rows.clear_cache()
    rows, _, _ = tomo_coarse._near_cut_rows(*args, max_significants=1, adaptive_fraction=0.999)
    np.testing.assert_array_equal(np.asarray(rows)[0], [True, True, True, False])
    tomo_coarse._near_cut_rows.clear_cache()


def test_near_cut_rows_are_the_cut_and_minimum_neighbourhoods_of_capped_particles():
    """Undecided samples: within twice the bound of the cut's log weight, or of the smallest diff2, and only for
    a particle that max_significants cuts."""

    n_rot, n_trans, cap = 6, 4, 6
    diff2 = np.full((3, n_rot, n_trans), 1100.0, np.float32)
    # Particle 0 and 1 share their diff2: a top five (two a hair apart at the minimum), two samples a hair apart
    # at rank six, the rest far.
    for p_ in (0, 1):
        diff2[p_, 0, :] = [1000.0, 1001.0, 1002.0, 1003.0]
        diff2[p_, 2, 1] = 1010.0
        diff2[p_, 4, 3] = np.nextafter(np.float32(1010.0), np.float32(2000.0))
        diff2[p_, 5, 0] = np.nextafter(np.float32(1000.0), np.float32(2000.0))  # a hair above the smallest
    values = (diff2.min(axis=(1, 2), keepdims=True) - diff2).reshape(3, -1).astype(np.float32)
    order = np.argsort(-values, axis=1, kind="stable")
    mask = np.zeros(values.shape, bool)
    np.put_along_axis(mask, order[:, :cap], True, axis=1)
    cutoff_count = np.array([cap, cap - 1, cap], np.int32)  # particle 1 is cut by the adaptive fraction
    error = np.full(3, 4 * np.spacing(np.float32(1100.0)), np.float32)
    rows, most, _ = tomo_coarse._near_cut_rows(
        jnp.asarray(diff2), jnp.asarray(values), jnp.asarray(mask), jnp.asarray(cutoff_count), jnp.asarray(error),
        jnp.asarray([3, 3, 3]), max_significants=cap, adaptive_fraction=0.999,
    )
    rows = np.asarray(rows)
    np.testing.assert_array_equal(rows[0], [True, False, True, False, True, True])  # minimum, cut pair, near-minimum
    # Particle 1 is cut by the adaptive fraction, between samples far apart: its GEMM scores decide.
    assert not rows[1].any()
    assert rows[2].all()  # every sample of particle 2 ties at the cut
    assert int(most) == n_rot

    ids, use = tomo_coarse.first_rows(jnp.asarray(rows), capacity=4)
    np.testing.assert_array_equal(np.asarray(ids)[0], [0, 2, 4, 5])
    np.testing.assert_array_equal(np.asarray(use), [[True] * 4, [False] * 4, [True] * 4])
    exact = jnp.asarray(np.arange(3 * 4 * n_trans, dtype=np.float32).reshape(3, 4, n_trans))
    put = np.asarray(tomo_coarse.put_rows(jnp.asarray(diff2), ids, use, exact))
    assert_matches(put[0][[0, 2, 4, 5]], np.asarray(exact)[0])
    assert_matches(put[0][[1, 3]], diff2[0][[1, 3]])
    assert_matches(put[1], diff2[1])


@pytest.mark.parametrize("call_bytes", [64 << 20, 2000], ids=["one_call", "groups_of_two"])
@pytest.mark.parametrize("local", [False, True], ids=["global", "local"])
def test_undecided_scorer_rotations_are_the_per_particle_builds(monkeypatch, local, call_bytes):
    """The exact cut's scorer matrices built on the device from its undecided rotations are those of
    one rotation-kernel call per particle with its own images (the earlier host loop)."""

    def pass1_rotations(eulers_deg, right_f32, left_matrices):
        # Traceable and elementwise, so a program and the eager reference round alike.
        e = jnp.asarray(eulers_deg, dtype=jnp.float32)
        base = jnp.sin(e[:, :, None] * jnp.float32(0.01) + jnp.arange(3, dtype=jnp.float32)[None, None, :])
        left = jnp.asarray(left_matrices, dtype=jnp.float32)
        return left[:, None, :, :] * base[None, :, :, :]

    def reference(eulers_deg, random_perturbation, angular_sampling_deg, *, left_matrices):
        return pass1_rotations(eulers_deg, None, left_matrices)

    monkeypatch.setattr(tomo_coarse, "_UNDECIDED_ROTATION_CALL_BYTES", call_bytes)
    _check_undecided_scorer_rotations(pass1_rotations, local, 0.0, reference=reference)


@pytest.mark.gpu
@pytest.mark.parametrize("local", [False, True], ids=["global", "local"])
def test_undecided_scorer_rotations_are_the_rotation_kernels_per_particle(gpu_device, monkeypatch, local):
    """The same with RELION's rotation kernel (perturbed, left matrices): the per-particle calls' matrices.

    The program takes the perturbation as its float32 matrix (``pass1_rotations_left_f32``); the reference builds
    each particle's rows from the perturbation itself (``relion_adaptive_pass1_rotations``)."""

    import jax
    from recovar.cuda_backproject import custom_cuda_requested

    from relax.sampling.device_rotations import pass1_rotations_left_f32, relion_adaptive_pass1_rotations

    if jax.default_backend() != "gpu" or not custom_cuda_requested():
        pytest.skip("needs the custom CUDA rotation kernel")

    def reference(*args, **kwargs):
        return np.asarray(relion_adaptive_pass1_rotations(*args, **kwargs))

    monkeypatch.setattr(tomo_coarse, "_UNDECIDED_ROTATION_CALL_BYTES", 2000)
    for random_perturbation in (0.37, -0.21, 0.0):
        _check_undecided_scorer_rotations(pass1_rotations_left_f32, local, random_perturbation, reference=reference)


def test_exact_cut_eulers_pad_a_local_grid_to_its_capacity_class_and_keep_the_layout_bitwise():
    """A local search's compact grid is padded with zero rows to its capacity class (the two grids of iteration 24
    of the real tilt-series case, and of iteration 11, share one class each), a global grid keeps its length, and the
    exact cut's scorer matrices from the padded table are bitwise those from the grid itself."""

    from relax.fine_pass.resident_statistics import resident_image_capacity

    for sizes in ((928_738, 950_302), (131_222, 135_021)):
        assert len({resident_image_capacity(n) for n in sizes}) == 1
    rng = np.random.default_rng(11)
    eulers = rng.uniform(-180.0, 180.0, size=(1000, 3))
    assert tomo_coarse._exact_cut_eulers(eulers, local=False).shape == (1000, 3)
    padded = np.asarray(tomo_coarse._exact_cut_eulers(eulers, local=True))
    assert padded.shape == (resident_image_capacity(1000), 3) and padded.shape[0] > 1000
    assert np.array_equal(padded[:1000].view(np.uint32), eulers.astype(np.float32).view(np.uint32))
    assert not padded[1000:].any()

    def pass1_rotations(eulers_deg, right_f32, left_matrices):
        e = jnp.asarray(eulers_deg, dtype=jnp.float32)
        base = jnp.sin(e[:, :, None] * jnp.float32(0.01) + jnp.arange(3, dtype=jnp.float32)[None, None, :])
        return jnp.asarray(left_matrices, dtype=jnp.float32)[:, None, :, :] * base[None, :, :, :]

    n_particles, slots, capacity, r_pad = 6, 3, 8, 128
    counts = np.array([97, 120, 1, 64, 128, 33])
    unit_rows = [np.sort(rng.choice(1000, size=int(c), replace=False)) for c in counts]
    particle_left = [rng.normal(size=(int(n), 3, 3)) for n in (3, 1, 2, 3, 2, 3)]
    ids = rng.integers(0, r_pad, size=(n_particles, capacity)).astype(np.int32)
    use = rng.random((n_particles, capacity)) < 0.8
    layouts = [
        np.asarray(
            tomo_coarse._undecided_scorer_rotations(
                jnp.asarray(ids), jnp.asarray(use), counts, unit_rows, table, particle_left, 0.0, 15.0,
                r_pad=r_pad, slots=slots, pass1_rotations=pass1_rotations,
            )
        )
        for table in (jnp.asarray(eulers.astype(np.float32)), jnp.asarray(padded))
    ]
    assert layouts[0].any()
    assert np.array_equal(layouts[0].view(np.uint32), layouts[1].view(np.uint32))


def _check_undecided_scorer_rotations(pass1_rotations, local, random_perturbation, reference):
    rng = np.random.default_rng(7)
    n_particles, slots, capacity, n_rot, r_pad = 5, 4, 3, 11, 16
    n_images = np.array([4, 2, 3, 1, 4])
    counts = np.array([11, 9, 11, 6, 10]) if local else np.full(n_particles, n_rot)
    unit_rows = [np.sort(rng.choice(n_rot, size=int(c), replace=False)) for c in counts] if local else None
    eulers = rng.uniform(-180.0, 180.0, size=(n_rot, 3))
    particle_left = [rng.normal(size=(int(n), 3, 3)) for n in n_images]
    # Class-major ids of two classes, some past a particle's count (padded rotations); particle 2 has none.
    ids = rng.integers(0, 2 * r_pad, size=(n_particles, capacity)).astype(np.int32)
    use = rng.random((n_particles, capacity)) < 0.7
    use[2] = False
    use[0, 0] = True

    want = np.zeros((n_particles, slots, capacity, 3, 3), dtype=np.float32)
    for p in np.flatnonzero(use.any(axis=1)):
        grid_rows = np.minimum(np.where(use[p], ids[p] % r_pad, 0), int(counts[p]) - 1)
        if local:
            grid_rows = unit_rows[p][grid_rows]
        want[p, : n_images[p]] = np.asarray(
            reference(eulers[grid_rows], random_perturbation, 15.0, left_matrices=particle_left[p])
        )

    got = tomo_coarse._undecided_scorer_rotations(
        jnp.asarray(ids),
        jnp.asarray(use),
        counts,
        unit_rows,
        jnp.asarray(eulers.astype(np.float32)),
        particle_left,
        random_perturbation,
        15.0,
        r_pad=r_pad,
        slots=slots,
        pass1_rotations=pass1_rotations,
    )
    assert_matches(np.asarray(got), want)


@pytest.mark.parametrize("local", [False, True], ids=["global", "local"])
def test_batch_scoring_rotations_are_the_per_particle_builds(local):
    """A coarse batch's scorer matrices are one rotation-kernel call per particle with its own images and rows:
    the global layout (images whose frames are not shared, one call for the batch) as the local one. Padded
    rotations repeat the particle's last one; padded slots and particles are zero."""

    from relax.sampling import tomo_particles

    def pass1_rotations(eulers_deg, random_perturbation, angular_sampling_deg, *, left_matrices):
        e = jnp.asarray(eulers_deg, dtype=jnp.float32)
        base = jnp.sin(e[:, :, None] * jnp.float32(0.01) + jnp.arange(3, dtype=jnp.float32)[None, None, :])
        return jnp.asarray(left_matrices, dtype=jnp.float32)[:, None, :, :] * base[None, :, :, :]

    rng = np.random.default_rng(19)
    n_rot, slots, r_pad, p_pad = 7, 4, 8, 4
    offsets = np.array([0, 3, 5, 9, 10])
    image_left = rng.normal(size=(int(offsets[-1]), 3, 3))
    eulers = rng.uniform(-180.0, 180.0, size=(n_rot, 3))
    units = np.array([0, 2, 3])
    rows = [np.sort(rng.choice(n_rot, size=size, replace=False)) for size in (5, 7, 3)] if local else None

    want = np.zeros((p_pad, slots, r_pad, 3, 3), dtype=np.float32)
    for p, unit in enumerate(units):
        unit_rows = np.arange(n_rot) if rows is None else rows[p]
        left, _applies = tomo_particles.relion_left_matrices(image_left[offsets[unit] : offsets[unit + 1]])
        built = np.asarray(pass1_rotations(eulers[unit_rows], 0.0, 15.0, left_matrices=left))
        want[p, : built.shape[0]] = built[:, np.minimum(np.arange(r_pad), unit_rows.size - 1)]

    got = tomo_coarse._batch_scoring_rotations(
        units, offsets, image_left, eulers, rows, 0.0, 15.0,
        p_pad=p_pad, slots=slots, r_pad=r_pad, pass1_rotations=pass1_rotations,
    )
    assert_matches(np.asarray(got), want)


@pytest.mark.parametrize("variant", ["one_flush", "one_row", "flushes_of_three", "rescored_one_by_one"])
@pytest.mark.parametrize("n_classes", [1, 2])
def test_capped_particles_are_cut_on_the_direct_squares_of_their_undecided_rotations(monkeypatch, n_classes, variant):
    """The pass with a scorer that is off by less than the share of its stated bound the cut's margin takes
    (_GEMM_ERROR_MARGIN_SHARE): where max_significants cuts, the
    supports are the cut of the exact diff2, and only the undecided rotations were scored exactly.

    ``one_row`` leaves room for one undecided rotation per particle, so the flush learns from the device that
    its cut must be taken again, at its size. ``flushes_of_three``: one particle per batch and three per flush, so
    the last flush holds one particle and is padded to three; every flush's programs see three particles and the
    supports are the one-flush pass's."""

    rng = np.random.default_rng(31)
    offsets = np.array([0, 2, 5, 6, 9])
    n_images, n_units, n_rot, n_trans, cap = int(offsets[-1]), 4, 200, 9, 20
    bound = np.float32(8 * np.spacing(np.float32(2000.0)))

    def exact_image_diff2(class_value, rotations, initial, angles):
        # [P, S, R, T] float32; spread over a few units so that many samples sit near any cut.
        # A function of the matrices alone: a re-scored rotation is not at its grid position.
        trace = jnp.sum(rotations, axis=(3, 4))
        per_rot = trace * jnp.float32(0.37) + class_value * jnp.sin(jnp.float32(40.0) * trace)
        return (initial[:, :, None, None] + per_rot[:, :, :, None] + angles[:, :, None, :, 0]).astype(jnp.float32)

    def fake_block(total, error, class_value, rotations, unshifted, weight, initial, angles, score_indices, first, *, count, **kwargs):
        block = slice(int(first), int(first) + count)
        image = exact_image_diff2(class_value, rotations[:, block], initial[:, block], angles[:, block])
        valid = (initial[:, block] > 0)[:, :, None, None]
        # The "GEMM": the exact values off by up to half the margin's share of the stated bound per image, in a
        # fixed pattern.
        wobble = jnp.float32(0.5 * tomo_coarse._GEMM_ERROR_MARGIN_SHARE) * bound * jnp.cos(jnp.arange(image.size, dtype=jnp.float32) * 1.7).reshape(image.shape)
        image = jnp.where(valid, image + wobble, 0.0)
        for slot in range(image.shape[1]):
            total = total + image[:, slot].swapaxes(1, 2)
        return total, error + bound * jnp.sum(valid[:, :, 0, 0], axis=1)

    direct_calls = []

    def fake_direct(class_value, rotations, unshifted, weight, initial, angles, full_to_compact, **kwargs):
        direct_calls.append(int(rotations.shape[2]))
        image = jnp.where(
            (initial > 0)[:, :, None, None], exact_image_diff2(jnp.real(class_value), rotations, initial, angles), 0.0
        )
        total = jnp.zeros(image.shape[:1] + image.shape[2:], jnp.float32)
        for slot in range(image.shape[1]):
            total = total + image[:, slot]
        return total

    def operands(experiment_dataset, image_start, image_stop, layout, **kwargs):
        images = np.arange(int(image_start), int(image_stop))
        return (
            jnp.zeros((images.size, 4), jnp.complex64),
            jnp.ones((images.size, 4), jnp.float32),
            jnp.asarray(600.0 + 100.0 * np.modf(images * 0.6180339887)[0], jnp.float32),
        )

    def pass1_rotations(eulers, random_perturbation, angular_sampling_deg, *, left_matrices):
        # Traceable: the exact cut builds its rotations inside a program.
        base = jnp.asarray(eulers, dtype=jnp.float32)[:, :, None] * jnp.eye(3, dtype=jnp.float32)[None, :, :] / 180.0
        # Each image's matrices depend on its own left matrix (not on its place in the call).
        left = jnp.asarray(left_matrices, dtype=jnp.float32)
        return base[None] + left[:, 0, 0][:, None, None, None]

    captured = []
    significance = tomo_coarse.particle_coarse_significance

    def capture(diff2, rotation_prior, translation_prior, **kwargs):
        captured.append(np.asarray(diff2))
        return significance(diff2, rotation_prior, translation_prior, **kwargs)

    from relax.projection import projection
    from relax.sampling import device_rotations

    monkeypatch.setattr(device_rotations, "relion_adaptive_pass1_rotations", pass1_rotations)
    monkeypatch.setattr(
        device_rotations,
        "pass1_rotations_left_f32",
        lambda eulers, right_f32, left_matrices: pass1_rotations(eulers, 0.0, 0.0, left_matrices=left_matrices),
    )
    monkeypatch.setattr(projection, "relion_projector_half_to_texture_full", lambda value: jnp.asarray(value))
    monkeypatch.setattr(tomo_coarse, "coarse_gemm_slot_block", fake_block)
    monkeypatch.setattr(tomo_coarse, "direct_rows_diff2", fake_direct)
    monkeypatch.setattr(tomo_coarse, "_all_image_coarse_operands", operands)
    monkeypatch.setattr(tomo_coarse, "particle_coarse_significance", capture)
    layout = tomo_coarse.CoarseScoreLayout(
        (8, 8), 6, np.arange(4), jnp.ones(4, bool), jnp.zeros(1, jnp.int32), jnp.ones(40)
    )
    eulers = rng.uniform(0.0, 180.0, size=(n_rot, 3))
    translations = rng.uniform(-1.0, 1.0, size=(n_trans, 3))
    class_values = tuple(jnp.float32(v) for v in (0.3, -0.2)[:n_classes])
    kwargs = dict(
        unit_image_offsets=offsets,
        image_projections=np.tile(np.eye(3), (n_images, 1, 1)),
        image_left=np.eye(3)[None] * (1.0 + np.arange(n_images))[:, None, None],
        unit_old_offsets_px=np.zeros((n_units, 3)),
        coarse_eulers_deg=eulers,
        random_perturbation=0.0,
        angular_sampling_deg=15.0,
        coarse_translations_px=translations,
        projector_half=class_values if n_classes > 1 else class_values[0],
        layout=layout,
        noise_variance_half=None,
        rotation_log_prior=np.zeros((n_classes, n_rot), np.float32) if n_classes > 1 else None,
        unit_translation_log_prior=np.zeros((n_units, n_trans), np.float32),
        adaptive_fraction=0.999,
        max_significants=cap,
        model_max_r=3,
        padding_factor=2,
        box_size=8,
    )
    if variant == "one_row":
        monkeypatch.setattr(tomo_coarse, "_UNDECIDED_ROTATIONS", 1)
    if variant == "rescored_one_by_one":
        # The undecided rotations scored again one call per rotation: the same exact cut.
        monkeypatch.setattr(tomo_coarse, "_rescore_rows_per_call", lambda *args: 1)
        tomo_coarse._exact_rows_diff2.clear_cache()  # a trace cached by "one_row" would bypass direct_calls
    supports, _pmax, counts = tomo_coarse.particle_coarse_supports(None, **kwargs)
    if variant == "flushes_of_three":
        # One particle per batch (the stand-in GEMM's rounding pattern follows the batch layout, so both passes
        # below share it): flushes of two particles need no padding; flushes of three end with one padded to three.
        batches = tomo_coarse._coarse_batches
        monkeypatch.setattr(tomo_coarse, "_coarse_batches", lambda *a, **k: batches(*a, **{**k, "budget_bytes": 1}))
        monkeypatch.setattr(tomo_coarse, "_SIGNIFICANCE_BATCH_BYTES", 2 * n_classes * 256 * n_trans * 4)
        captured.clear()
        unpadded, unpadded_pmax, unpadded_counts = tomo_coarse.particle_coarse_supports(None, **kwargs)
        unpadded_gemm = np.concatenate(captured[0::2])  # each flush's GEMM diff2, then its exact one
        monkeypatch.setattr(tomo_coarse, "_SIGNIFICANCE_BATCH_BYTES", 3 * n_classes * 256 * n_trans * 4)
        captured.clear()
        padded, padded_pmax, padded_counts = tomo_coarse.particle_coarse_supports(None, **kwargs)
        # Two flushes (3 particles, then 1 padded to 3), each cut on GEMM scores and then on exact ones.
        assert [c.shape[0] for c in captured] == [3, 3, 3, 3]
        for k, (got, want) in enumerate(zip(*(([x] if n_classes == 1 else x) for x in (padded, unpadded)))):
            for unit in range(n_units):
                np.testing.assert_array_equal(got[unit], want[unit], err_msg=f"class {k} particle {unit}")
        assert_matches(padded_pmax, unpadded_pmax)
        np.testing.assert_array_equal(padded_counts, unpadded_counts)
        # The device share, not the diff2 bytes, bounds the flush to one particle: same supports, diff2 and Pmax.
        monkeypatch.setattr(tomo_coarse, "_SIGNIFICANCE_BATCH_BYTES", 512 << 20)
        plan = tomo_coarse._flush_plan

        def one_particle_budget(diff2_bytes, operand_bytes, **plan_kwargs):
            held = _flush_held_bytes(1, diff2_bytes, operand_bytes)
            plan_kwargs["flush_bytes"] = int(held / (1 - tomo_coarse._RESCORE_SHARE)) + 8
            return plan(diff2_bytes, operand_bytes, **plan_kwargs)

        monkeypatch.setattr(tomo_coarse, "_flush_plan", one_particle_budget)
        captured.clear()
        bounded, bounded_pmax, bounded_counts = tomo_coarse.particle_coarse_supports(None, **kwargs)
        assert [c.shape[0] for c in captured] == [1] * (2 * n_units)
        assert_matches(np.concatenate(captured[0::2]), unpadded_gemm)
        for k, (got, want) in enumerate(zip(*(([x] if n_classes == 1 else x) for x in (bounded, unpadded)))):
            for unit in range(n_units):
                np.testing.assert_array_equal(got[unit], want[unit], err_msg=f"class {k} particle {unit}")
        assert_matches(bounded_pmax, unpadded_pmax)
        np.testing.assert_array_equal(bounded_counts, unpadded_counts)
        return
    supports = [supports] if n_classes == 1 else supports
    wobbly = captured[0]
    # One batch: its GEMM cut, then the exact cut; a squeezed flush takes both again at the size it needs.
    assert len(captured) == (5 if variant == "one_row" else 2) and captured[-1].shape == wobbly.shape
    r_pad = wobbly.shape[1] // n_classes
    assert direct_calls and max(direct_calls) < n_rot  # a few rotations per particle, not the grid

    # The exact diff2 of every sample, from the same stand-ins with every rotation "undecided".
    monkeypatch.setattr(
        tomo_coarse,
        "_near_cut_rows",
        lambda diff2, *a, **k: (jnp.ones(diff2.shape[:2], bool), jnp.int32(diff2.shape[1]), jnp.int32(0)),
    )
    captured.clear()
    all_exact, _, all_exact_counts = tomo_coarse.particle_coarse_supports(None, **kwargs)
    all_exact = [all_exact] if n_classes == 1 else all_exact
    exact = captured[-1]
    assert 0.0 < np.max(np.abs(exact - wobbly)) <= 3 * bound
    n_capped = 0
    for unit in range(n_units):
        log_weight = (exact[unit].min() - exact[unit]).astype(np.float32).reshape(n_classes, r_pad, n_trans)[:, :n_rot]
        kept = np.sort(np.argsort(-log_weight.reshape(-1), kind="stable")[:cap])
        got = np.sort(np.concatenate([k * n_rot * n_trans + supports[k][unit] for k in range(n_classes)]))
        want = np.sort(np.concatenate([k * n_rot * n_trans + all_exact[k][unit] for k in range(n_classes)]))
        np.testing.assert_array_equal(got, want, err_msg=f"particle {unit}")
        if got.size == cap:
            n_capped += 1
            np.testing.assert_array_equal(got, kept, err_msg=f"particle {unit}")
    assert n_capped >= 2
    # RELION's count is the capped rank (maximum_significants), the same from the GEMM-and-exact cut as from exact scores.
    np.testing.assert_array_equal(counts, all_exact_counts)
    assert np.all(counts <= cap) and np.sum(counts == cap) >= 2


def test_coarse_diff2_constant_is_the_power_beyond_the_iterations_current_size():
    """relax#114: the coarse diff2 constant is half of RELION's powerClass Xi2 beyond ``image_current_size`` (the
    iteration's size, taken once per particle for both passes), not beyond the coarse size, and 0 at the full box."""

    from relax.fine_pass.scoring import relion_cuda_powerclass_highres_xi2_half

    box, coarse = 32, 12
    rng = np.random.default_rng(114)
    n_half = box * (box // 2 + 1)
    processed = jnp.asarray((rng.normal(size=(3, n_half)) + 1j * rng.normal(size=(3, n_half))).astype(np.complex64))

    def layout(image_current_size):
        return tomo_coarse.CoarseScoreLayout(
            (box, box), coarse, np.arange(4), jnp.ones(4, bool), jnp.zeros(1, jnp.int32), jnp.ones(n_half),
            image_current_size=image_current_size,
        )

    def beyond(size):
        return np.asarray(relion_cuda_powerclass_highres_xi2_half(processed, image_shape=(box, box), current_size=size))

    at_iteration = np.asarray(tomo_coarse.highres_xi2_half(processed, layout(24)))
    assert_matches(at_iteration, beyond(24))
    assert np.all(at_iteration < beyond(coarse))
    assert np.all(np.asarray(tomo_coarse.highres_xi2_half(processed, layout(box))) == 0)
    assert_matches(np.asarray(tomo_coarse.highres_xi2_half(processed, layout(None))), beyond(coarse))
    assert tomo_coarse.coarse_score_layout(
        (box, box), coarse, half_spectrum_scoring=True, square_window=False, image_current_size=24
    ).image_current_size == 24
