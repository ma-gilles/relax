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
