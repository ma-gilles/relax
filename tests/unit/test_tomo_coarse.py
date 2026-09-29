"""Subtomogram coarse pass (S4.2, CPU): translation chunks and the per-particle sum, with a stand-in kernel.

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


def _fake_per_image_projector(projector_full, rotations, images, angles, weight, initial, full_to_compact, **kwargs):
    # [B, R, Tc]: each image's own rotations and phases, plus its initial diff2 and a weight term.
    per_rot = jnp.sum(rotations, axis=(2, 3))
    return initial[:, None, None] + per_rot[:, :, None] + angles[:, None, :, 0] + jnp.sum(weight, axis=1)[:, None, None]


def test_particles_are_scored_with_each_images_poses_and_summed_in_slot_order(monkeypatch):
    monkeypatch.setattr(em_cuda_kernels, "relion_coarse_diff2_projector_per_image_f32", _fake_per_image_projector)
    rng = np.random.default_rng(7)
    n_particles, n_slots, n_rot, n_trans = 2, 3, 5, 300  # three translation chunks
    rotations = rng.normal(size=(n_particles, n_slots, n_rot, 3, 3)).astype(np.float32)
    angles = rng.normal(size=(n_particles, n_slots, n_trans, 2)).astype(np.float32)
    weight = rng.uniform(size=(n_particles, n_slots, 4)).astype(np.float32)
    initial = rng.uniform(1.0, 2.0, size=(n_particles, n_slots)).astype(np.float32)
    total = tomo_coarse._particles_coarse_diff2.__wrapped__(
        jnp.zeros((n_particles, n_rot, n_trans), jnp.float32),
        None,
        jnp.asarray(rotations),
        jnp.zeros((n_particles, n_slots, 4), jnp.complex64),
        jnp.asarray(weight),
        jnp.asarray(initial),
        jnp.asarray(angles),
        jnp.zeros(1, jnp.int32),
        current_size=6,
        physical_image_size=8,
        model_max_r=3,
        padding_factor=2,
        n_chunks=3,
    )
    for p in range(n_particles):
        expected = np.zeros((n_rot, n_trans), np.float32)
        for s in range(n_slots):
            image = (
                initial[p, s]
                + rotations[p, s].sum(axis=(1, 2))[:, None]
                + angles[p, s][None, :, 0]
                + weight[p, s].sum()
            ).astype(np.float32)
            expected = (expected + image).astype(np.float32)
        assert_matches(np.asarray(total[p]), expected)


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


def test_k_class_particles_cut_their_weights_over_every_class_jointly(monkeypatch):
    """Class3D: every image is scored against each class; the particle's (class, rotation, translation) weights
    are cut once (convertAllSquaredDifferencesToWeights sorts all classes' weights together), and each cell
    goes back to its class's support."""

    rng = np.random.default_rng(11)
    offsets = np.array([0, 2, 5, 6])
    n_images, n_rot, n_trans = int(offsets[-1]), 5, 7

    def fake(projector_full, rotations, images, angles, weight, initial, full_to_compact, **kwargs):
        # Each class shifts its diff2 by its own "projector" value, so the classes compete.
        return _fake_per_image_projector(None, rotations, images, angles, weight, initial, full_to_compact) + (
            projector_full * jnp.arange(rotations.shape[1], dtype=jnp.float32)[None, :, None]
        )

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
    monkeypatch.setattr(em_cuda_kernels, "relion_coarse_diff2_projector_per_image_f32", fake)
    monkeypatch.setattr(tomo_coarse, "_all_image_coarse_operands", operands)
    monkeypatch.setattr(tomo_coarse, "particle_coarse_significance", capture)
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
        projector_full=(jnp.float32(0.3), jnp.float32(-0.2)),
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
    (diff2, rotation_prior, translation_prior), = captured
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
