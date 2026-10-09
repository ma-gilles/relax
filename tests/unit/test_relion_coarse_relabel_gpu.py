"""The coarse GEMM operands reproduce RELION's coarse row relabelling in the wrap band (GPU).

For an image on a coarser grid than the reference (scale s), a coarse window between 2 r_max and
2 s r_max makes RELION's coarse diff2 kernel project and shift the rows beyond maxR at
``i - window`` inside the model sphere (acc/cuda/cuda_kernels/diff2.cuh:86-90, 163-164). The
fused coarse projector (relion_coarse_diff2_projector_body.inc) walks RELION's layout and applies
that rule itself; the GEMM route builds its projections (``compute_relion_projector_projections_block``)
and translated images (``relion_coarse_translate``) separately. Both must give the same scores.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def _fused_and_gemm_scores(box, window, r_max, scale, *, relabel):
    from relax.cuda import kernels as em_cuda_kernels
    from relax.helpers.projection import (
        compute_relion_projector_projections_block,
        relion_projector_half_to_texture_full,
    )
    from relax.relion.relion_coarse_operands import relion_coarse_translate
    from relax.scoring.tomo_coarse import coarse_score_layout
    from relax.sparse_pass2.sparse_pass2_bucket_io import relion_translation_angles_f32

    rng = np.random.default_rng(7)
    padding_factor = 2
    size = 2 * r_max * padding_factor + 3
    projector_half = (
        rng.normal(0.0, 1.0, (size, size, r_max * padding_factor + 2))
        + 1j * rng.normal(0.0, 1.0, (size, size, r_max * padding_factor + 2))
    ).astype(np.complex64)
    rotations = (Rotation.random(20, random_state=3).as_matrix() / scale).astype(np.float32)
    image_shape = (box, box)
    # The coarse pass's square score window and the fused projector's lookup (significance.py).
    layout = coarse_score_layout(image_shape, window, half_spectrum_scoring=True, square_window=True)
    score_indices = np.asarray(layout.score_indices_np, dtype=np.int32)
    lookup = layout.full_to_compact
    n_pixels = score_indices.size
    images = (rng.normal(0.0, 1.0, (2, n_pixels)) + 1j * rng.normal(0.0, 1.0, (2, n_pixels))).astype(np.complex64)
    weight = (rng.uniform(0.1, 2.0, images.shape) * np.asarray(layout.score_active_mask)[None]).astype(np.float32)
    initial = np.zeros(2, dtype=np.float32)
    angles = jnp.asarray(relion_translation_angles_f32(rng.uniform(-3.0, 3.0, (9, 2)), image_shape), dtype=jnp.float32)
    fused = em_cuda_kernels.relion_coarse_diff2_projector_f32(
        relion_projector_half_to_texture_full(jnp.asarray(projector_half)).astype(jnp.complex64),
        jnp.asarray(rotations),
        jnp.asarray(images),
        angles,
        jnp.asarray(weight),
        jnp.asarray(initial),
        jnp.asarray(lookup),
        current_size=window,
        physical_image_size=box,
        model_max_r=r_max,
        padding_factor=padding_factor,
    )
    projected, _ = compute_relion_projector_projections_block(
        jnp.asarray(projector_half),
        jnp.asarray(rotations),
        image_shape,
        r_max=r_max,
        padding_factor=padding_factor,
        return_abs2=False,
        centered_rows=True,
        dense_scale=True,
        projector_output_size=window,
        pixel_indices=score_indices,
        relion_texture_interp=True,
        relion_kernel="coarse",
        image_r_max=jnp.asarray(window // 2, dtype=jnp.int32),
    )
    shifted = relion_coarse_translate(
        em_cuda_kernels.relion_translate_score_f32,
        jnp.asarray(images),
        angles,
        jnp.asarray(score_indices),
        score_indices,
        image_shape,
        window=window if relabel else None,
        r_max=r_max,
    )
    projected = np.asarray(projected, dtype=np.complex128)
    shifted = np.asarray(shifted, dtype=np.complex128)
    difference = projected[None, :, None, :] - shifted[:, None, :, :]
    gemm = 0.5 * np.einsum("bp,brtp->brt", weight.astype(np.float64), np.abs(difference) ** 2)
    return np.asarray(fused, dtype=np.float64), gemm


@pytest.mark.gpu
@pytest.mark.parametrize("box, window, r_max, scale", [(112, 56, 25, 1.12), (40, 40, 17, 1.2)])
def test_coarse_gemm_operands_match_the_fused_relion_kernel_in_the_band(
    monkeypatch, custom_cuda_lib, gpu_device, box, window, r_max, scale
):
    import recovar.cuda_backproject as cuda_backproject

    from relax.helpers.optics_scale import coarse_rows_wrap_inside

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    assert coarse_rows_wrap_inside(window, r_max, scale)
    with jax.default_device(gpu_device):
        fused, gemm = _fused_and_gemm_scores(box, window, r_max, scale, relabel=True)
        _, unrelabelled = _fused_and_gemm_scores(box, window, r_max, scale, relabel=False)
    # float32 kernels on both sides, different accumulation orders.
    np.testing.assert_allclose(gemm, fused, rtol=2e-5)
    # The relabelled rows matter: shifting them at their centred row moves the scores further from
    # the fused kernel's (measured 2e-7 relabelled against 9e-7 and 2.4e-5 without, 2026-09-30).
    relabelled_error = np.max(np.abs(gemm - fused) / np.abs(fused))
    assert np.max(np.abs(unrelabelled - fused) / np.abs(fused)) > 2.0 * relabelled_error
