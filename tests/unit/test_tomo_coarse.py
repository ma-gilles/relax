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
    layout = tomo_coarse.CoarseScoreLayout((8, 8), 6, np.arange(4), jnp.ones(4, bool), jnp.zeros(1, jnp.int32), jnp.ones(40))
    n_rot, n_trans = 5, 300  # three chunks of at most 128 translations
    rotations = rng.normal(size=(3, n_rot, 3, 3)).astype(np.float32)
    angles = rng.normal(size=(3, n_trans, 2)).astype(np.float32)
    initial = np.array([1.5, 2.5, 3.5], np.float32)
    per_image = [
        tomo_coarse.tilt_image_coarse_diff2(
            None, rotations[i], jnp.zeros(4, jnp.complex64), jnp.ones(4), initial[i], angles[i], layout,
            model_max_r=3, padding_factor=2,
        )
        for i in range(3)
    ]
    for i in range(3):
        expected = initial[i] + rotations[i].sum(axis=(1, 2))[:, None] + angles[i][None, :, 0]
        assert_matches(np.asarray(per_image[i]), expected.astype(np.float32))
    total = np.asarray(tomo_coarse.particle_coarse_diff2(per_image))
    expected_total = np.float32(np.float32(np.asarray(per_image[0]) + np.asarray(per_image[1])) + np.asarray(per_image[2]))
    assert_matches(total, expected_total)
