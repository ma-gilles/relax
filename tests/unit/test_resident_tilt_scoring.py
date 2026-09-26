"""Flat-row scoring of subtomogram particles: diff2 summed over each particle's tilt images (S4.2, CPU).

The CUDA kernel is replaced by a JAX stand-in that depends on the row's projection, the image's
own phase table and the image's initial diff2, so the test checks the orchestration
(image slots, per-image projections and phases, the float32 running sum in image order, the
per-particle minimum and weights) rather than the kernel arithmetic, which the GPU tests cover.
"""

import numpy as np
import pytest
from helpers.float_compare import assert_matches

pytest.importorskip("jax")
import jax.numpy as jnp

from relax.cuda import kernels as em_cuda_kernels
from relax.refinement import tomo_particles
from relax.sparse_pass2 import resident_scoring

pytestmark = pytest.mark.unit


def _fake_kernel(reference, row_image_ids, image, angles, weight, full_to_compact, current_size, initial, translation_chunk_live=None):
    # Additions only, so compiled and eager evaluation round alike (no fused multiply-add).
    angles, initial = jnp.asarray(angles), jnp.asarray(initial)
    safe = jnp.where(row_image_ids >= 0, row_image_ids, 0)
    out = (jnp.real(reference[:, :1]) + angles[safe, :, 0]) + initial[safe][:, None]
    return jnp.where((row_image_ids >= 0)[:, None], out, jnp.inf).astype(jnp.float32)


def test_rows_sum_their_particles_images_in_order(monkeypatch):
    monkeypatch.setattr(em_cuda_kernels, "relion_fine_diff2_fused_translate_runtime_flat_rows_f32", _fake_kernel)
    rng = np.random.default_rng(7)
    unit_offsets = np.array([0, 3, 5, 9])  # three particles with 3, 2 and 4 tilt images
    n_images, n_pix, n_trans = 9, 12, 5
    row_unit = np.array([0, 0, 1, 1, 1, 2, 2, 2])
    n_rows = row_unit.size
    row_is_valid = np.arange(n_rows) < 7  # the last row is padding
    base = (rng.normal(size=(n_rows, n_pix)) + 1j * rng.normal(size=(n_rows, n_pix))).astype(np.complex64)
    image = (rng.normal(size=(n_images, n_pix)) + 1j * rng.normal(size=(n_images, n_pix))).astype(np.complex64)
    corr = rng.uniform(0.5, 2.0, size=(n_images, n_pix)).astype(np.float32)
    initial = rng.uniform(10, 20, size=n_images).astype(np.float32)
    angles = rng.normal(size=(n_images, n_trans, 2)).astype(np.float32)
    half_weights = np.ones(n_pix, dtype=np.float32)
    row_log_prior = rng.normal(size=n_rows).astype(np.float32)
    unit_prior = rng.normal(size=(3, n_trans)).astype(np.float32)
    mask = rng.uniform(size=(n_rows, n_trans)) < 0.8
    slots = np.stack([tomo_particles.image_slot_ids(row_unit, unit_offsets, k) for k in range(4)])

    def project_slot(slot, image_ids):  # each image sees the row's reference through its own matrix
        return jnp.asarray(base) + image_ids.astype(jnp.complex64)[:, None]

    out = resident_scoring.score_tilt_image_rows(
        project_slot,
        slots,
        row_unit,
        row_log_prior,
        image,
        corr,
        initial,
        unit_prior,
        jnp.asarray(mask),
        jnp.asarray(row_is_valid),
        half_weights=half_weights,
        image_translation_angles=angles,
        full_to_compact=np.zeros(1, np.int32),
        logical_current_size=2,
        unit_capacity=3,
    )

    weights = np.asarray(resident_scoring._relion_cuda_fine_pixel_weights(corr, half_weights[None, :]), np.float32)
    expected = np.full((n_rows, n_trans), np.inf, dtype=np.float32)
    for r in range(n_rows):
        if not row_is_valid[r]:
            continue
        running = np.zeros(n_trans, dtype=np.float32)
        for i in range(unit_offsets[row_unit[r]], unit_offsets[row_unit[r] + 1]):
            ids = jnp.full(n_rows, i, jnp.int32)  # every row against image i; keep row r
            one = np.asarray(_fake_kernel(project_slot(0, ids), ids, image, angles, weights, None, None, initial))[r]
            running = np.float32(running + one)
        expected[r] = np.where(mask[r], running, np.inf)
    assert_matches(np.asarray(out.raw_diff2), expected)

    for u in range(3):
        rows = (row_unit == u) & row_is_valid
        assert_matches(np.asarray(out.min_diff2)[u], expected[rows].min())
    finite = np.isfinite(np.asarray(out.scores))
    assert not finite[~row_is_valid].any()
    assert (np.asarray(out.scores)[finite] <= unit_prior.max() + row_log_prior.max() + 1e-6).all()


@pytest.mark.gpu
def test_flat_row_kernel_reads_each_images_own_phase_table():
    """Rank-3 angles [B, T, 2]: each row uses its image's table, as separate rank-2 calls per image do."""
    import jax

    if jax.default_backend() != "gpu":
        pytest.skip("needs the CUDA flat-row kernel")
    rng = np.random.default_rng(11)
    size = 16
    n_full = size * (size // 2 + 1)
    n_images, n_rows, n_trans = 3, 7, 5
    full_to_compact = np.arange(n_full, dtype=np.int32)
    reference = (rng.normal(size=(n_rows, n_full)) + 1j * rng.normal(size=(n_rows, n_full))).astype(np.complex64)
    image = (rng.normal(size=(n_images, n_full)) + 1j * rng.normal(size=(n_images, n_full))).astype(np.complex64)
    weight = rng.uniform(0.1, 1.0, size=(n_images, n_full)).astype(np.float32)
    initial = rng.uniform(1, 2, size=n_images).astype(np.float32)
    angles = rng.uniform(-0.5, 0.5, size=(n_images, n_trans, 2)).astype(np.float32)
    row_image = np.array([0, 2, 1, -1, 2, 0, 1], dtype=np.int32)
    kernel = em_cuda_kernels.relion_fine_diff2_fused_translate_runtime_flat_rows_f32
    got = np.asarray(kernel(reference, row_image, image, angles, weight, full_to_compact, np.int32(size), initial))
    for b in range(n_images):
        rows = row_image == b
        one = np.asarray(
            kernel(
                reference,
                np.where(rows, b, -1).astype(np.int32),
                image,
                angles[b],
                weight,
                full_to_compact,
                np.int32(size),
                initial,
            )
        )
        assert_matches(got[rows], one[rows])
    assert np.isinf(got[row_image < 0]).all()
