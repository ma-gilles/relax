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


def _gemm_fixture(rng, *, n_images=3, n_rot=256, n_trans=200):
    """A box-32 projector half, random rotations, images, weights, d0 and translation phases on the GPU."""

    from relax.helpers.projection import relion_projector_half_to_texture_full

    box, size, pad, max_r = 32, 32, 2, 16
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
            tomo_coarse._coarse_capacity_texture(f["half"], layout, model_max_r=f["max_r"], padding_factor=f["pad"])
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
        texture = tomo_coarse._coarse_capacity_texture(f["half"], layout, model_max_r=f["max_r"], padding_factor=f["pad"])
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

    def fake_block(total, class_value, rotations, unshifted, weight, initial, angles, score_indices, *, first, count, **kwargs):
        block = slice(first, first + count)
        rotations, weight, initial, angles = rotations[:, block], weight[:, block], initial[:, block], angles[:, block]
        # Each image adds its initial diff2, its rotation's sum, its angle and its weight; each class shifts its
        # diff2 by its own "projector" value times the rotation index, so the classes compete.
        per_rot = jnp.sum(rotations, axis=(3, 4)) + class_value * jnp.arange(rotations.shape[2], dtype=jnp.float32)
        image = initial[:, :, None, None] + per_rot[:, :, :, None] + angles[:, :, None, :, 0] + jnp.sum(weight, axis=2)[:, :, None, None]
        for slot in range(image.shape[1]):
            total = total + image[:, slot].swapaxes(1, 2)  # the pass accumulates translation-major
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
        return np.stack([base + np.float32(i) for i in range(left_matrices.shape[0])])

    from relax import sampling

    monkeypatch.setattr(sampling, "_relion_adaptive_pass1_rotations", pass1_rotations)
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
