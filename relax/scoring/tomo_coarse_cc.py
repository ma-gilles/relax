"""``--firstiter_cc`` coarse pass of subtomogram particles: each particle's coarse normalized-CC winner.

``particle_coarse_cc_winners`` scores each tilt image's coarse CC operands (``tilt_image_cc_coarse_operands``:
the single-particle exact CC operands, without pre-shift or norm), sums them per particle and keeps the first
maximum in RELION's orientation order: a winner by CC, with no significance, unlike the significance pass of
``tomo_coarse``, with which it shares no helper. Particles are scored in flushes sized from the device's free
memory (``_cc_flush_particles``), each image slot of a flush in one projection and one translation launch.
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)


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
    ``X`` and ``A`` are the SPA coarse CC GEMM terms (``relax.scoring.coarse_kernels.relion_coarse_gemm_terms``).
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


@jax.jit
def _add_flush_slot_cc_diff2(running_diff2, projections, shifted, pixel_weight, live):
    """A flush's ``running_diff2`` ``[F, R, T]`` plus one image slot of each particle, where it has one (``live``):
    :func:`_add_tilt_image_cc_diff2` per particle, in turn. ``projections`` ``[F R, N]`` and ``shifted``
    ``[F T, N]`` are the slot images' projections and translated score windows; ``pixel_weight`` ``[F, N]``."""

    n_flush, n_rot, n_trans = (int(n) for n in running_diff2.shape)
    projections = jnp.asarray(projections, dtype=jnp.complex64).reshape(n_flush, n_rot, -1)
    shifted = shifted.reshape(n_flush, n_trans, -1)

    def add(particle):
        running, projection, translated, weight, is_live = particle
        return jnp.where(is_live, _add_tilt_image_cc_diff2(running, projection, translated, weight), running)

    return jax.lax.map(add, (running_diff2, projections, shifted, pixel_weight, live))


@jax.jit
def _flush_cc_winners(running_diff2, relion_cells):
    """Each flush particle's winning cell (the first maximum of ``-running_diff2`` in RELION's cell order
    ``relion_cells``) and its score."""

    scores = -running_diff2.reshape(running_diff2.shape[0], -1)
    winners = relion_cells[jnp.argmax(scores[:, relion_cells], axis=1)]
    return winners, jnp.take_along_axis(scores, winners[:, None], axis=1)[:, 0]


# A flush's images are padded to a multiple of this count (with copies of its first image), so the operand and
# rotation programs take a few image counts per pass, not one per flush.
_CC_FLUSH_IMAGE_QUANTUM = 256

# The share of the device's free memory a flush may take, and its cap. The share leaves room for the pool's
# fragmentation: without preallocation each array must fit in one of the pool's regions (a 16 GB emulated 10491 CC
# it1 ran out at 2.3 GiB with half its pool free).
_CC_FLUSH_SHARE = 0.25
_CC_FLUSH_MAX_BYTES = 8 << 30

# The largest single array of a flush (one slot's projections, ``[F R, N]`` complex64), at most.
_CC_FLUSH_MAX_ARRAY_BYTES = 512 << 20


def _cc_flush_particles(
    n_units: int, *, max_images: int, n_rot: int, n_trans: int, n_window: int, image_pixels: int, available_bytes
) -> int:
    """Particles per coarse CC flush: what ``_CC_FLUSH_SHARE`` of ``available_bytes`` (at most
    ``_CC_FLUSH_MAX_BYTES``) holds, with one slot's projections within ``_CC_FLUSH_MAX_ARRAY_BYTES``; at least one.

    Per particle a flush holds one slot's score-window projections ``[R, N]`` complex64 (and the texture's crop
    they are gathered from), its running and new diff2 ``[R, T]``, and per image its rotation matrices, phases,
    operands and the preparation's full images (``image_pixels`` each, a few float32 copies); one particle's GEMM
    terms at a time (the window packed twice, ``|p|^2``, the cross and the contribution) are fixed.
    """

    per_particle = n_rot * (16 * n_window + 8 * n_trans) + max_images * (
        36 * n_rot + 8 * n_trans + 12 * n_window + 16 * image_pixels
    )
    fixed = n_rot * (20 * n_window + 12 * n_trans) + (12 * n_window + 16 * image_pixels) * _CC_FLUSH_IMAGE_QUANTUM
    budget = _CC_FLUSH_MAX_BYTES if available_bytes is None else min(
        _CC_FLUSH_MAX_BYTES, _CC_FLUSH_SHARE * float(available_bytes)
    )
    largest = _CC_FLUSH_MAX_ARRAY_BYTES // (8 * n_rot * n_window)
    return int(max(1, min(int(n_units), largest, (int(budget) - fixed) // per_particle)))


def _cc_capacity_texture(relion_projector_half, *, r_max: int, padding_factor: int, coarse_size: int):
    """The projector's texture staged once for the pass, where the half-storage kernel serves its projections
    (the per-call route stages, synchronizes and frees one on every call), else ``None``."""

    from relax.projection.projection import relion_capacity_texture_serves

    projector_half = jnp.asarray(relion_projector_half)
    if not relion_capacity_texture_serves(
        projector_half,
        r_max=int(r_max),
        padding_factor=int(padding_factor),
        projector_output_size=int(coarse_size),
        relion_texture_interp=True,
    ):
        return None
    from relax.cuda.projector import RelionCapacityHalfTextureF32

    return RelionCapacityHalfTextureF32(projector_half, int(r_max), padding_factor=int(padding_factor))


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
    counts = np.diff(offsets)
    n_units = int(counts.size)
    winners = np.zeros(n_units, dtype=np.int64)
    best = np.zeros(n_units, dtype=np.float64)
    if n_units == 0:
        return winners, best
    from relax.runtime import memory_budget

    flush = _cc_flush_particles(
        n_units,
        max_images=int(counts.max()),
        n_rot=n_rot,
        n_trans=n_trans,
        n_window=int(window.size),
        image_pixels=int(np.prod(image_shape)),
        available_bytes=memory_budget.current_device_available_bytes(),
    )
    relion_cells_device = jnp.asarray(relion_cells, dtype=jnp.int32)
    angles_shape = (n_trans, 2)
    texture = _cc_capacity_texture(
        relion_projector_half, r_max=relion_projector_r_max, padding_factor=padding_factor, coarse_size=coarse_size
    )
    logger.info(
        "Coarse CC flushes of %d particles for %d (at most %d images each; projector texture %s)",
        flush,
        n_units,
        int(counts.max()),
        "staged once" if texture is not None else "per call",
    )
    try:
        for first in range(0, n_units, flush):
            units = np.arange(first, min(first + flush, n_units))
            unit_counts = np.zeros(flush, dtype=np.int64)
            unit_counts[: units.size] = counts[units]
            first_local = np.concatenate([[0], np.cumsum(unit_counts)[:-1]])
            n_images = int(unit_counts.sum())
            padded = -(-n_images // _CC_FLUSH_IMAGE_QUANTUM) * _CC_FLUSH_IMAGE_QUANTUM
            images = np.full(padded, offsets[units[0]], dtype=np.int64)
            images[:n_images] = np.concatenate([np.arange(offsets[unit], offsets[unit + 1]) for unit in units])
            image_unit = np.zeros(padded, dtype=np.int64)
            image_unit[:n_images] = np.repeat(np.arange(units.size), counts[units])
            unshifted, pixel_weight = tilt_image_cc_coarse_operands(
                experiment_dataset,
                images,
                window,
                half_weights,
                scale_corrections=None if scale_corrections is None else np.asarray(scale_corrections)[images],
            )
            left, _applies = tomo_particles.relion_left_matrices(image_left[images])
            rotations = jnp.asarray(
                relion_adaptive_pass1_rotations(
                    coarse_eulers_deg, random_perturbation, angular_sampling_deg, left_matrices=left
                ),
                dtype=jnp.float32,
            )
            angles = jnp.asarray(
                tomo_particles.tilt_translation_angles(
                    coarse_translations_px, old[units], image_projections[images], image_unit, box_size
                ),
                dtype=jnp.float32,
            )
            if angles.shape[1:] != angles_shape:
                raise ValueError(f"the tilt phases are {angles.shape[1:]}, not {angles_shape}")
            running = jnp.zeros((flush, n_rot, n_trans), dtype=jnp.float32)
            for slot in range(int(unit_counts.max())):
                live = unit_counts > slot
                # Each particle's slot image; a particle without one scores its first image, discarded.
                local = jnp.asarray(np.where(live, first_local + slot, 0), dtype=jnp.int32)
                projections, _ = compute_relion_projector_projections_block(
                    relion_projector_half,
                        rotations[local].reshape(-1, 3, 3),
                    image_shape,
                    r_max=int(relion_projector_r_max),
                    padding_factor=int(padding_factor),
                    centered_rows=True,
                    dense_scale=True,
                    relion_texture_interp=True,
                    relion_kernel="coarse",
                    projector_output_size=int(coarse_size),
                    capacity_texture=texture,
                    return_abs2=False,
                    # The square score window is the texture's coarse-size crop, gathered from it (checked against
                    # the crop) instead of from the zero-filled full image: the same values.
                    pixel_indices=window,
                )
                shifted = cuda_translate.relion_translate_score_f32(
                    unshifted[local], angles[local], window_device, image_shape
                )
                running = _add_flush_slot_cc_diff2(running, projections, shifted, pixel_weight[local], jnp.asarray(live))
            flush_winners, flush_best = _flush_cc_winners(running, relion_cells_device)
            winners[units] = np.asarray(flush_winners, dtype=np.int64)[: units.size]
            best[units] = np.asarray(flush_best, dtype=np.float32)[: units.size].astype(np.float64)
    finally:
        if texture is not None:
            texture.close()
    return winners, best
