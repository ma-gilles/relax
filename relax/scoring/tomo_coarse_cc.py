"""``--firstiter_cc`` coarse pass of subtomogram particles: each particle's coarse normalized-CC winner.

``particle_coarse_cc_winners`` scores each tilt image's coarse CC operands (``tilt_image_cc_coarse_operands``:
the single-particle exact CC operands, without pre-shift or norm), sums them per particle and keeps the first
maximum in RELION's orientation order: a winner by CC, with no significance and no memory budget, unlike the
significance pass of ``tomo_coarse``, with which it shares no helper.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def tilt_image_cc_coarse_operands(experiment_dataset, image_indices, window_indices, half_weights, *, scale_corrections=None):
    """Tilt images' ``--firstiter_cc`` coarse operands: the SPA exact CC operands, without pre-shift or norm.

    The unshifted corrected score image and the CC pixel weight ``corr_img * half_weights`` in the
    score window (significance.py's exact normalized-CC coarse pass: ``assemble_relion_cc_coarse_operands``
    of the per-image RELION FFT, RFLOAT CTF and ``1 / sum |X|^2``). RELION neither translates nor
    normalises a tomo image (acc_ml_optimiser_impl.h:429-476).
    """

    from relax.io.batch_fetch import fetch_indexed_batch
    from relax.relion.ctf import relion_exact_ctf_half_from_source_star
    from relax.relion.image_preprocessing import prepare_batch_preprocess_operands
    from relax.scoring.coarse_operands import (
        assemble_relion_cc_coarse_operands,
        process_relion_exact_coarse_half_image,
        relion_cc_inverse_power_from_processed,
    )

    image_indices = np.asarray(image_indices, dtype=np.int64)
    batch_data, _ctf_params, fetched = fetch_indexed_batch(experiment_dataset, image_indices)
    if not np.array_equal(np.asarray(fetched), image_indices):
        raise RuntimeError("the dataset returned the tilt images in another order")
    batch_data = np.asarray(batch_data)
    relion_cuda, _shifts, _corr, batch_scale, preprocess_kwargs = prepare_batch_preprocess_operands(
        experiment_dataset, batch_data, image_indices, scale_corrections=scale_corrections
    )
    if not relion_cuda:
        raise RuntimeError("the subtomogram coarse pass needs the RELION CUDA image preprocessing")
    processed = process_relion_exact_coarse_half_image(
        experiment_dataset, batch_data, True, relion_preprocess_kwargs=preprocess_kwargs, image_indices=image_indices
    )
    window = jnp.asarray(window_indices, dtype=jnp.int32)
    operands = assemble_relion_cc_coarse_operands(
        processed,
        relion_exact_ctf_half_from_source_star(experiment_dataset, image_indices, experiment_dataset.image_shape),
        relion_cc_inverse_power_from_processed(processed, window),
        jnp.asarray(batch_scale, dtype=jnp.float32),
        phase_factors=None,
        window_indices=window,
        scale_corrections_enabled=scale_corrections is not None,
    )
    return (
        jnp.asarray(operands.windowed_unshifted, dtype=jnp.complex64),
        jnp.asarray(operands.windowed_corr_img * jnp.asarray(half_weights)[window], dtype=jnp.float32),
    )


@jax.jit
def _add_tilt_image_cc_diff2(running_diff2, projections, shifted, pixel_weight):
    """``running_diff2`` ``[R, T]`` plus one tilt image's coarse CC term, RELION's way.

    ``cuda_kernel_diff2_CC_coarse`` has each of its 128 threads atomically add
    ``-X / (128 sqrt(A))`` to the particle's diff2, which already holds the earlier images' terms;
    ``X`` and ``A`` are the SPA coarse CC GEMM terms (scoring.relion_coarse_gemm_terms).
    """

    from relax.scoring.coarse_kernels import relion_coarse_gemm_terms

    n_trans = int(shifted.shape[0])
    cross, model_energy, _, _ = relion_coarse_gemm_terms(
        projections, shifted[None], pixel_weight[None], 1, n_images=1, n_trans=n_trans, wide=jnp.float32
    )
    contribution = jnp.asarray(cross[0].T, dtype=jnp.float32) / (
        jnp.float32(128.0) * jnp.sqrt(jnp.maximum(model_energy[0][:, None], jnp.float32(1e-30)))
    )
    return jax.lax.fori_loop(0, 128, lambda _, acc: acc - contribution, running_diff2, unroll=True)


def particle_coarse_cc_winners(
    experiment_dataset,
    *,
    unit_image_offsets,
    image_projections,
    image_left,
    unit_old_offsets_px,
    coarse_eulers_deg,
    relion_order,
    random_perturbation,
    angular_sampling_deg,
    coarse_translations_px,
    relion_projector_half,
    relion_projector_r_max: int,
    padding_factor: int,
    coarse_size: int,
    box_size: int,
    scale_corrections=None,
):
    """Each particle's coarse ``--firstiter_cc`` winner: the cell ``rotation * T + translation`` and its CC.

    RELION's first CC iteration sums every tilt image's normalized CC into the particle's diff2
    (acc_ml_optimiser_impl.h:1290-1347, no Xi2 offset and no priors) and keeps the minimum alone
    (convertAllSquaredDifferencesToWeights' CC branch); pass 2 then scores that sample's children.
    Each image is scored with its own coarse matrices ``L R`` (``image_left``) and phases, as the Gaussian pass
    (:func:`particle_coarse_supports`). ``relion_order`` ``[R]`` is each coarse rotation's position in
    RELION's orientation order: the first maximum in that order wins, as RELION's ordered minimum.
    ``scale_corrections`` are per dataset image.
    """

    from relax.cuda import translate as cuda_translate
    from relax.fourier.fourier_window import make_fourier_window_spec
    from relax.fourier.half_spectrum import make_scoring_half_image_weights
    from relax.projection.projection import compute_relion_projector_projections_block
    from relax.sampling import tomo_particles
    from relax.sampling.device_rotations import relion_adaptive_pass1_rotations

    image_shape = tuple(int(n) for n in experiment_dataset.image_shape)
    n_half = image_shape[0] * (image_shape[1] // 2 + 1)
    # RELION's CC kernels sum every element of the windowed FFTW image, the DC and the redundant x=0
    # column included, and Xi2 is its full power (ml_optimiser.cpp:6846-6855): the square current-size
    # crop of the CC pass (sparse_pass2_window's normalized_cc window, pass2_half_weights).
    window = np.asarray(
        make_fourier_window_spec(
            image_shape,
            int(coarse_size),
            n_half,
            square=False,
            score_square=True,
            score_include_dc=True,
            include_recon_window=False,
        ).score_indices_np,
        dtype=np.int32,
    )
    half_weights = make_scoring_half_image_weights(image_shape, relion_half_sum=True, exclude_relion_redundant_x0=False)
    offsets = np.asarray(unit_image_offsets, dtype=np.int64)
    image_projections = np.asarray(image_projections, dtype=np.float64)
    image_left = np.asarray(image_left, dtype=np.float64)
    old = tomo_particles.relion_gpu_old_offsets(np.asarray(unit_old_offsets_px, dtype=np.float64))
    coarse_eulers_deg = np.asarray(coarse_eulers_deg)
    n_rot = int(coarse_eulers_deg.shape[0])
    n_trans = int(np.asarray(coarse_translations_px).shape[0])
    relion_rank = np.asarray(relion_order, dtype=np.int64).reshape(-1)
    if relion_rank.shape != (n_rot,):
        raise ValueError("relion_order needs one position per coarse rotation")
    # Cells in RELION's orientation-major order, so the first maximum is RELION's.
    relion_cells = (np.argsort(relion_rank, kind="stable")[:, None] * n_trans + np.arange(n_trans)[None, :]).reshape(-1)
    window_device = jnp.asarray(window)
    n_units = int(offsets.size - 1)
    winners = np.zeros(n_units, dtype=np.int64)
    best = np.zeros(n_units, dtype=np.float64)
    for unit in range(n_units):
        images = np.arange(offsets[unit], offsets[unit + 1])
        unshifted, pixel_weight = tilt_image_cc_coarse_operands(
            experiment_dataset,
            images,
            window,
            half_weights,
            scale_corrections=None if scale_corrections is None else np.asarray(scale_corrections)[images],
        )
        left, _applies = tomo_particles.relion_left_matrices(image_left[images])
        rotations = np.asarray(
            relion_adaptive_pass1_rotations(
                coarse_eulers_deg, random_perturbation, angular_sampling_deg, left_matrices=left
            ),
            dtype=np.float32,
        )
        angles = tomo_particles.tilt_translation_angles(
            coarse_translations_px, old[unit : unit + 1], image_projections[images], np.zeros(images.size, int), box_size
        )
        running = jnp.zeros((n_rot, n_trans), dtype=jnp.float32)
        for slot in range(images.size):
            projections, _ = compute_relion_projector_projections_block(
                relion_projector_half,
                jnp.asarray(rotations[slot]),
                image_shape,
                r_max=int(relion_projector_r_max),
                padding_factor=int(padding_factor),
                centered_rows=True,
                dense_scale=True,
                relion_texture_interp=True,
                relion_kernel="coarse",
                projector_output_size=int(coarse_size),
            )
            shifted = cuda_translate.relion_translate_score_f32(
                unshifted[slot : slot + 1], jnp.asarray(angles[slot], dtype=jnp.float32), window_device, image_shape
            )
            running = _add_tilt_image_cc_diff2(
                running, jnp.asarray(projections, dtype=jnp.complex64)[:, window_device], shifted, pixel_weight[slot]
            )
        scores = -np.asarray(running, dtype=np.float32).reshape(-1)
        position = int(np.argmax(scores[relion_cells]))
        winners[unit] = int(relion_cells[position])
        best[unit] = float(scores[winners[unit]])
    return winners, best
