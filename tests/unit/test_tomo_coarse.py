"""Subtomogram coarse pass (S4.2): the per-image GEMM scorer, the per-particle sum and the cut, with stand-ins on CPU.

The per-image operands and the kernel are the SPA ones; relion_dump_it1_forced (em_work/cryoet_s42_20260925,
check_p3_p4.py) compares the whole pass with RELION's dumped coarse diff2 on the GPU.
"""

import numpy as np
import pytest
from helpers.float_compare import assert_matches

pytest.importorskip("jax")
import jax.numpy as jnp

from relax.cuda import kernels as em_cuda_kernels
from relax.scoring import tomo_coarse

pytestmark = pytest.mark.unit


def _fake_projector(projector_full, rotations, images, angles, weight, initial, full_to_compact, **kwargs):
    # [1, R, Tc]: depends on the rotation, the translation angle and the image's initial diff2.
    per_rot = jnp.sum(rotations, axis=(1, 2))
    return (initial[:, None, None] + per_rot[None, :, None] + angles[None, None, :, 0])[None][0]


def test_translations_are_scored_in_chunks_and_images_add_in_slot_order(monkeypatch):
    monkeypatch.setattr(em_cuda_kernels, "relion_coarse_diff2_projector_f32", _fake_projector)
    rng = np.random.default_rng(3)
    layout = tomo_coarse.CoarseScoreLayout(
        (8, 8), 6, np.arange(4), jnp.ones(4, bool), jnp.zeros(1, jnp.int32), jnp.ones(40)
    )
    n_rot, n_trans = 5, 300  # three chunks of at most 128 translations
    rotations = rng.normal(size=(3, n_rot, 3, 3)).astype(np.float32)
    angles = rng.normal(size=(3, n_trans, 2)).astype(np.float32)
    initial = np.array([1.5, 2.5, 3.5], np.float32)
    per_image = [
        tomo_coarse.tilt_image_coarse_diff2(
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
    total = np.asarray(tomo_coarse.particle_coarse_diff2(per_image))
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


def test_particle_significance_cuts_the_summed_diff2_with_its_3d_offset_prior():
    rng = np.random.default_rng(5)
    n_particles, n_rot, n_trans = 3, 40, 27
    image_diff2 = rng.uniform(1000.0, 1010.0, size=(4, n_particles, n_rot, n_trans)).astype(np.float32)
    particle = tomo_coarse.particle_coarse_diff2(list(image_diff2))
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


def test_images_add_in_slot_order():
    rng = np.random.default_rng(7)
    image_diff2 = rng.uniform(1000.0, 1010.0, size=(2, 3, 5, 9)).astype(np.float32)
    total = np.asarray(tomo_coarse._add_image_diff2_in_slot_order(jnp.zeros((2, 5, 9), jnp.float32), jnp.asarray(image_diff2)))
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

    monkeypatch.setattr(em_cuda_kernels, "relion_translate_score_f32", shift)
    got = np.asarray(
        tomo_coarse._images_coarse_gemm_diff2.__wrapped__(
            jnp.asarray(projected), jnp.asarray(images), jnp.asarray(weight), jnp.asarray(initial), jnp.asarray(angles),
            jnp.arange(n_px, dtype=jnp.int32), image_shape=(8, 8),
        )
    )
    for b in range(n_images):
        shifted = images[b][None, :] * np.exp(1j * angles[b].sum(axis=1))[:, None]
        direct = initial[b] + 0.5 * np.einsum(
            "p,rtp->rt", weight[b].astype(np.float64), np.abs(projected[b][:, None, :] - shifted[None]) ** 2
        )
        assert_matches(got[b], direct.T.astype(np.float32), rtol=1e-5)  # translation-major


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

    from relax.helpers import projection
    from relax.sparse_pass2 import sparse_pass2_budget as budget

    gib = 1 << 30
    rotations, slots, trans, pixels = 4608, 41, 81, 278
    # The JAX fallback holds the rows in the projector's complex dtype and one temporary: 32 B a complex128 pixel.
    assert tomo_coarse._coarse_projection_bytes_per_pixel(
        [None], np.complex128, image_size=128, current_size=26, model_max_r=24
    ) == 32
    assert tomo_coarse._coarse_projection_bytes_per_pixel(
        [None], np.complex64, image_size=128, current_size=26, model_max_r=24
    ) == 16
    monkeypatch.setattr(projection, "relion_coarse_packed_rows_serve", lambda *a: True)
    assert tomo_coarse._coarse_projection_bytes_per_pixel(
        [object()], np.complex64, image_size=128, current_size=26, model_max_r=24
    ) == 8

    # The device share: a quarter of what the allocator can still hand out (6.6 GiB free beside half 1's state).
    monkeypatch.setattr(budget, "_device_free_memory_bytes", lambda: int(6.6 * gib))
    monkeypatch.setattr(budget, "_jax_allocator_free_memory_bytes", lambda: int(9.8 * gib))
    monkeypatch.setattr(budget, "_jax_allocator_pool_free_bytes", lambda: 0)
    share = tomo_coarse._coarse_batch_bytes()
    assert share == int(0.25 * 6.6 * gib)
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
    monkeypatch.setattr(budget, "_device_free_memory_bytes", lambda: None)
    monkeypatch.setattr(budget, "_jax_allocator_free_memory_bytes", lambda: None)
    assert tomo_coarse._coarse_batch_bytes() == tomo_coarse._COARSE_BATCH_BYTES


def test_coarse_batch_plan_on_an_80gb_card_is_the_fixed_2gib_plan(monkeypatch):
    """On an 80 GB card the device-aware budget changes nothing: the texture path's rows take 8 B a pixel and a
    quarter of the free memory exceeds the fixed 2 GiB, so the batches are those of the fixed budget (the policy in
    docs/development/gpu_compatibility.md: small-card sizing is a no-op on large cards)."""

    from relax.helpers import projection
    from relax.sparse_pass2 import sparse_pass2_budget as budget

    gib = 1 << 30
    monkeypatch.setattr(budget, "_device_free_memory_bytes", lambda: 66 * gib)
    monkeypatch.setattr(budget, "_jax_allocator_free_memory_bytes", lambda: 70 * gib)
    monkeypatch.setattr(budget, "_jax_allocator_pool_free_bytes", lambda: 0)
    monkeypatch.setattr(projection, "relion_coarse_packed_rows_serve", lambda *a: True)
    assert tomo_coarse._coarse_batch_bytes() == tomo_coarse._COARSE_BATCH_BYTES
    bytes_per_pixel = tomo_coarse._coarse_projection_bytes_per_pixel(
        [object()], np.complex64, image_size=128, current_size=48, model_max_r=24
    )
    assert bytes_per_pixel == 8
    for rotations, pixels in ((4608, 278), (36864, 921), (448, 2520)):
        planned = tomo_coarse._coarse_batches(
            [rotations] * 600, n_slots=41, n_trans=81, n_pixels=pixels,
            projection_bytes_per_pixel=bytes_per_pixel, budget_bytes=tomo_coarse._coarse_batch_bytes(),
        )
        fixed = tomo_coarse._coarse_batches([rotations] * 600, n_slots=41, n_trans=81, n_pixels=pixels)
        assert [(list(u), r, p, s) for u, r, p, s in planned] == [(list(u), r, p, s) for u, r, p, s in fixed]


@pytest.mark.gpu
def test_the_per_image_kernel_matches_the_one_image_calls(gpu_device):
    """The batched per-image launch and one call per image score every (image, rotation, translation) alike."""

    import jax

    from relax.cuda.kernels import custom_cuda_requested

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
        batched = em_cuda_kernels.relion_coarse_diff2_projector_per_image_f32(
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
            single = tomo_coarse.tilt_image_coarse_diff2(
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

    from relax.helpers.projection import relion_projector_half_to_texture_full

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
@pytest.mark.parametrize("persistent_texture", [False, True], ids=["complex_rows", "packed_rows"])
def test_the_gemm_scorer_follows_relions_direct_square_kernel(gpu_device, persistent_texture):
    """The coarse pass's GEMM scorer against the fused direct-square kernel, image by image, on the
    complex projections and on the packed rows a persistent texture projects.

    The two round differently (the expansion d0 + 0.5 A + 0.5 C - X against RELION's sum of
    squares), so they agree to a few float32 ULP of the terms' scale, not bitwise: 1.8e-6 of
    the largest diff2 measured on this fixture (job 14856295), checked at 1e-5.
    """

    import jax

    from relax.cuda.kernels import custom_cuda_requested

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
        projected = tomo_coarse._coarse_gemm_projections(
            f["half"], jnp.asarray(f["rotations"]), layout, model_max_r=f["max_r"], padding_factor=f["pad"],
            texture=texture,
        ).reshape(f["n_images"], f["n_rot"], -1)
        assert projected.dtype == (jnp.float32 if persistent_texture else jnp.complex64)
        gemm = np.asarray(
            tomo_coarse._images_coarse_gemm_diff2(
                projected, jnp.asarray(f["images"]), jnp.asarray(f["weight"]), jnp.asarray(f["initial"]),
                jnp.asarray(f["angles"]), jnp.asarray(layout.score_indices_np, jnp.int32), image_shape=(f["box"], f["box"]),
            )
        )
        if texture is not None:
            texture.close()
        rotations = f["rotations"].reshape(f["n_images"], f["n_rot"], 3, 3)
        for b in range(f["n_images"]):
            direct = np.asarray(
                tomo_coarse.tilt_image_coarse_diff2(
                    f["full"], rotations[b], f["images"][b], f["weight"][b], f["initial"][b], f["angles"][b], layout,
                    model_max_r=f["max_r"], padding_factor=f["pad"],
                )
            )
            assert_matches(gemm[b].T, direct, rtol=1e-5, err_msg=f"image {b}")
            # The re-score at the cut relies on this bound of the GEMM's distance from the direct square.
            bound = float(
                tomo_coarse._images_gemm_error_bound(
                    projected[b : b + 1], jnp.asarray(f["images"][b : b + 1]), jnp.asarray(f["weight"][b : b + 1]),
                    jnp.asarray(f["initial"][b : b + 1]),
                )[0]
            )
            worst = float(np.max(np.abs(gemm[b].T - direct)))
            assert 0.0 < worst <= bound, f"image {b}"
            # The worst case over the pixels: measured 18 float32 units of the diff2 here, the bound about 2000.
            assert bound <= 1e-3 * float(np.max(direct))


@pytest.mark.gpu
@pytest.mark.parametrize("box", [12, 32, 72], ids=["72_pixels", "544_pixels", "2664_pixels"])
def test_the_gemm_error_bound_holds_over_a_million_samples(gpu_device, box):
    """The exact cut's bound on the GEMM scorer's distance from the direct square, over more than 1e6 samples
    at three score-window sizes: the largest observed difference stays under it."""

    import jax

    from relax.cuda.kernels import custom_cuda_requested

    if not custom_cuda_requested():
        pytest.skip("custom CUDA is disabled")
    with jax.default_device(gpu_device):
        n_images, n_rot, n_trans = 3, 2816, 120
        f = _gemm_fixture(np.random.default_rng(29 + box), n_images=n_images, n_rot=n_rot, n_trans=n_trans, box=box)
        layout = f["layout"]
        projected = tomo_coarse._coarse_gemm_projections(
            f["half"], jnp.asarray(f["rotations"]), layout, model_max_r=f["max_r"], padding_factor=f["pad"], texture=None
        ).reshape(n_images, n_rot, -1)
        gemm = np.asarray(
            tomo_coarse._images_coarse_gemm_diff2(
                projected, jnp.asarray(f["images"]), jnp.asarray(f["weight"]), jnp.asarray(f["initial"]),
                jnp.asarray(f["angles"]), jnp.asarray(layout.score_indices_np, jnp.int32), image_shape=(box, box),
            )
        )
        bound = np.asarray(
            tomo_coarse._images_gemm_error_bound(
                projected, jnp.asarray(f["images"]), jnp.asarray(f["weight"]), jnp.asarray(f["initial"])
            )
        )
        rotations = f["rotations"].reshape(n_images, n_rot, 3, 3)
        assert gemm.size >= 1_000_000
        for b in range(n_images):
            direct = np.asarray(
                tomo_coarse.tilt_image_coarse_diff2(
                    f["full"], rotations[b], f["images"][b], f["weight"][b], f["initial"][b], f["angles"][b], layout,
                    model_max_r=f["max_r"], padding_factor=f["pad"],
                )
            )
            worst = float(np.max(np.abs(gemm[b].T - direct)))
            units = worst / float(np.spacing(np.float32(np.max(direct))))
            print(f"box {box} pixels {layout.score_indices_np.size} image {b}: worst {worst:.4g} = {units:.1f} float32 units of the diff2; bound {bound[b]:.4g}")
            assert worst <= bound[b], f"image {b}"


@pytest.mark.gpu
@pytest.mark.parametrize("block_rows", [None, 2], ids=["one_block", "blocks_of_two_rows"])
def test_direct_rows_are_the_one_image_kernels_rows_in_slot_order(gpu_device, monkeypatch, block_rows):
    """A particle's re-scored rotations: any subset of rows, at any position in the call, gives the full
    direct-square pass's values for those rows, the images added in slot order; scored in blocks of rows
    (the kernel's int32 output count) they are bitwise the one-block values."""

    import jax

    from relax.cuda.kernels import custom_cuda_requested
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
            total = None
            for s_ in range(n_slots):
                b = p_ * n_slots + s_
                image = np.asarray(
                    tomo_coarse.tilt_image_coarse_diff2(
                        f["full"], rotations[p_, s_], f["images"][b], f["weight"][b], f["initial"][b], f["angles"][b],
                        layout, model_max_r=f["max_r"], padding_factor=f["pad"],
                    )
                )
                total = image if total is None else total + image
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
            np.testing.assert_array_equal(rows, one_block)
        assert_matches(one_block, np.take_along_axis(full, ids[:, :, None], axis=1))


@pytest.mark.gpu
def test_packed_rows_are_the_complex_projections_split(gpu_device):
    """The persistent texture's packed ``[Re | Im]`` rows are the complex coarse projections (the same texture
    reads and dense scale, in one kernel), so the GEMM's reference operand does not change."""

    import jax

    from relax.cuda.kernels import custom_cuda_requested

    if not custom_cuda_requested():
        pytest.skip("custom CUDA is disabled")
    with jax.default_device(gpu_device):
        f = _gemm_fixture(np.random.default_rng(23), n_images=1, n_rot=300)
        layout, rotations = f["layout"], jnp.asarray(f["rotations"])
        texture = tomo_coarse._coarse_capacity_texture(
            f["half"], layout, model_max_r=f["max_r"], padding_factor=f["pad"], class_index=0
        )
        assert texture is not None
        packed = np.asarray(
            tomo_coarse._coarse_gemm_projections(
                f["half"], rotations, layout, model_max_r=f["max_r"], padding_factor=f["pad"], texture=texture
            )
        )
        texture.close()
        complex_rows = np.asarray(
            tomo_coarse._coarse_gemm_projections(f["half"], rotations, layout, model_max_r=f["max_r"], padding_factor=f["pad"])
        )
    assert packed.shape == (complex_rows.shape[0], 2 * complex_rows.shape[1])
    assert np.count_nonzero(complex_rows) > complex_rows.size // 2
    assert_matches(packed, np.concatenate([complex_rows.real, complex_rows.imag], axis=-1))


@pytest.mark.gpu
def test_the_class_texture_is_refilled_for_the_next_pass(gpu_device):
    """A pass's class texture is the previous pass's owner refilled with the new projector (the compiled coarse
    programs capture its handle), and it projects what a texture staged fresh from that projector projects."""

    import jax

    from relax.cuda.kernels import custom_cuda_requested

    if not custom_cuda_requested():
        pytest.skip("custom CUDA is disabled")
    with jax.default_device(gpu_device):
        f = _gemm_fixture(np.random.default_rng(29), n_images=1, n_rot=200)
        layout, rotations = f["layout"], jnp.asarray(f["rotations"])
        kwargs = dict(model_max_r=f["max_r"], padding_factor=f["pad"])
        first = tomo_coarse._coarse_capacity_texture(f["half"], layout, class_index=0, **kwargs)
        used = tomo_coarse._coarse_gemm_projections(f["half"], rotations, layout, texture=first, **kwargs)
        tomo_coarse._CLASS_TEXTURES[0] = (first, used)  # the pass's end (particle_coarse_supports)
        changed = jnp.asarray(f["half"] * (0.5 + 0.25j))
        second = tomo_coarse._coarse_capacity_texture(changed, layout, class_index=0, **kwargs)
        assert second is first
        refilled = np.asarray(tomo_coarse._coarse_gemm_projections(changed, rotations, layout, texture=second, **kwargs))
        tomo_coarse._CLASS_TEXTURES.pop(0)[0].close_after(jnp.asarray(refilled))
        fresh = tomo_coarse._coarse_capacity_texture(changed, layout, class_index=1, **kwargs)
        expected = np.asarray(tomo_coarse._coarse_gemm_projections(changed, rotations, layout, texture=fresh, **kwargs))
        tomo_coarse._CLASS_TEXTURES.pop(1)[0].close_after(jnp.asarray(expected))
    assert fresh is not first
    np.testing.assert_array_equal(refilled, expected)


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


@pytest.mark.parametrize("flushes", ["one_flush", "flush_per_particle"])
def test_k_class_particles_cut_their_weights_over_every_class_jointly(monkeypatch, flushes):
    """Class3D: every image is scored against each class; the particle's (class, rotation, translation) weights
    are cut once (convertAllSquaredDifferencesToWeights sorts all classes' weights together), and each cell
    goes back to its class's support. One particle per batch and flush exercises the deferred readback and
    the compaction-size guess (exact first, then guessed, recompacted on overflow)."""

    rng = np.random.default_rng(11)
    offsets = np.array([0, 2, 5, 6])
    n_images, n_rot, n_trans = int(offsets[-1]), 5, 7

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
        base = np.asarray(eulers, dtype=np.float32)[:, :, None] * np.eye(3, dtype=np.float32)[None, :, :] / 180.0
        # Each image's matrices depend on its own left matrix (not on its place in the call).
        return np.stack([base + np.float32(left[0, 0]) for left in np.asarray(left_matrices)])

    from relax import sampling

    monkeypatch.setattr(sampling, "_relion_adaptive_pass1_rotations", pass1_rotations)
    from relax.helpers import projection

    monkeypatch.setattr(projection, "relion_projector_half_to_texture_full", lambda value: jnp.asarray(value))
    monkeypatch.setattr(tomo_coarse, "direct_rows_diff2", direct)
    monkeypatch.setattr(tomo_coarse, "_coarse_gemm_slot_block", fake_block)
    monkeypatch.setattr(tomo_coarse, "_all_image_coarse_operands", operands)
    monkeypatch.setattr(tomo_coarse, "particle_coarse_significance", capture)
    if flushes == "flush_per_particle":
        batches = tomo_coarse._coarse_batches
        monkeypatch.setattr(tomo_coarse, "_coarse_batches", lambda *a, **k: batches(*a, **{**k, "budget_bytes": 1}))
        monkeypatch.setattr(tomo_coarse, "_SIGNIFICANCE_BATCH_BYTES", 1)
    layout = tomo_coarse.CoarseScoreLayout(
        (8, 8), 6, np.arange(4), jnp.ones(4, bool), jnp.zeros(1, jnp.int32), jnp.ones(40)
    )
    class_prior = np.log(rng.uniform(0.2, 1.0, size=(2, n_rot))).astype(np.float32)
    supports, pmax = tomo_coarse.particle_coarse_supports(
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
        image_size=8,
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
    assert n_significant > 3  # the cut keeps more than each particle's winner
    assert pmax.shape == (3,)


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


@pytest.mark.parametrize("n_classes", [1, 2])
def test_capped_particles_are_cut_on_the_direct_squares_of_their_undecided_rotations(monkeypatch, n_classes):
    """The pass with a scorer that is off by less than its stated bound: where max_significants cuts, the
    supports are the cut of the exact diff2, and only the undecided rotations were scored exactly."""

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
        # The "GEMM": the exact values off by up to half the stated bound per image, in a fixed pattern.
        wobble = jnp.float32(0.5) * bound * jnp.cos(jnp.arange(image.size, dtype=jnp.float32) * 1.7).reshape(image.shape)
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
        base = np.asarray(eulers, dtype=np.float32)[:, :, None] * np.eye(3, dtype=np.float32)[None, :, :] / 180.0
        # Each image's matrices depend on its own left matrix (not on its place in the call).
        return np.stack([base + np.float32(left[0, 0]) for left in np.asarray(left_matrices)])

    captured = []
    significance = tomo_coarse.particle_coarse_significance

    def capture(diff2, rotation_prior, translation_prior, **kwargs):
        captured.append(np.asarray(diff2))
        return significance(diff2, rotation_prior, translation_prior, **kwargs)

    from relax import sampling
    from relax.helpers import projection

    monkeypatch.setattr(sampling, "_relion_adaptive_pass1_rotations", pass1_rotations)
    monkeypatch.setattr(projection, "relion_projector_half_to_texture_full", lambda value: jnp.asarray(value))
    monkeypatch.setattr(tomo_coarse, "_coarse_gemm_slot_block", fake_block)
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
        image_size=8,
    )
    supports, _pmax = tomo_coarse.particle_coarse_supports(None, **kwargs)
    supports = [supports] if n_classes == 1 else supports
    wobbly = captured[0]
    assert len(captured) == 2 and captured[1].shape == wobbly.shape  # one batch: its GEMM cut, then the exact cut
    r_pad = wobbly.shape[1] // n_classes
    assert direct_calls and max(direct_calls) < n_rot  # a few rotations per particle, not the grid

    # The exact diff2 of every sample, from the same stand-ins with every rotation "undecided".
    monkeypatch.setattr(
        tomo_coarse,
        "_near_cut_rows",
        lambda diff2, *a, **k: (jnp.ones(diff2.shape[:2], bool), jnp.int32(diff2.shape[1]), jnp.int32(0)),
    )
    captured.clear()
    all_exact, _ = tomo_coarse.particle_coarse_supports(None, **kwargs)
    all_exact = [all_exact] if n_classes == 1 else all_exact
    exact = captured[1]
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
