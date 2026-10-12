"""Drivers that run the live resident scoring programs of ``relax.fine_pass.resident_scoring`` in tests.

Moved from relax/fine_pass/resident_scoring.py, where only tests called them: the per-half image operands built
with the production ``_prepare_bucket_io``, the device upload of one chunk, the loop of ``score_resident_chunk``
over a half's chunks, and the tilt-series composition of ``tilt_image_rows_raw_diff2`` and
``tilt_rows_scores_from_raw``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from relax.fine_pass.bucket_io import _prepare_bucket_io
from relax.fine_pass.resident_candidates import CapacityChunk, ResidentCandidateTables, materialize_chunk
from relax.fine_pass.resident_scoring import (
    score_resident_chunk,
    tilt_image_rows_raw_diff2,
    tilt_rows_scores_from_raw,
)
from relax.fine_pass.scoring import _relion_powerclass_noise_terms
from relax.io.batch_fetch import fetch_indexed_batch


@dataclass(frozen=True)
class ResidentImageOperands:
    """Per-half, per-image scoring operands kept on the device.

    Every per-image array is indexed by the local image position used by
    :class:`~recovar.em.sparse_pass2.resident_candidates.ResidentCandidateTables`.

    Fields
    ------
    score_input
        complex64 ``[n_images, n_score_pixels]``: ``direct_score_input``, the
        untranslated image divided by its score weight factors, already gathered
        to the score window. The fused kernel applies the translation per pixel.
    corr_img_score
        float32 ``[n_images, n_score_pixels]``: ``ctf2_over_nv_score``
        (``Minvsigma2 * CTF^2 * scale^2``), window-gathered.
    highres_xi2_half
        float32 ``[n_images]``: RELION's ``powerClass`` tail already divided by
        two, added by the kernel as ``initial_diff2``. ``None`` when the pass
        does not use it (only possible outside exact-Gaussian scoring).
    batch_norm
        ``[n_images]``: the per-image normalization ``_prepare_bucket_io``
        returns; carried for the later M-step/noise stages, unused here.
    translation_prior
        float32 ``[n_images, n_fine_trans]``: the per-image fine translation
        log prior, zero when the pass has no translation prior.
    half_weights
        float32 ``[n_score_pixels]``: the scoring half-image weights of the
        score window (``direct_half_weights`` in the loop).
    translation_angles
        float32 ``[n_fine_trans, 2]``: RELION fine translation angles.
    full_to_compact
        int32 ``[current_size * (current_size // 2 + 1)]``: the RELION
        full-rectangle to compact-window lookup.
    current_size
        The logical RELION current size; passed to the kernel as a runtime
        scalar, so it does not key the program.
    """

    n_images: int
    n_score_pixels: int
    n_fine_trans: int
    current_size: int
    score_input: jax.Array
    corr_img_score: jax.Array
    highres_xi2_half: jax.Array | None
    batch_norm: jax.Array
    translation_prior: jax.Array
    half_weights: jax.Array
    translation_angles: jax.Array
    full_to_compact: jax.Array

    def __post_init__(self):
        expected = (self.n_images, self.n_score_pixels)
        if self.score_input.shape != expected:
            raise ValueError(f"score_input must have shape {expected}, got {self.score_input.shape}")
        if self.corr_img_score.shape != expected:
            raise ValueError(f"corr_img_score must have shape {expected}, got {self.corr_img_score.shape}")
        if self.translation_prior.shape != (self.n_images, self.n_fine_trans):
            raise ValueError(
                "translation_prior must have shape (n_images, n_fine_trans), got "
                f"{self.translation_prior.shape}",
            )
        if self.half_weights.shape != (self.n_score_pixels,):
            raise ValueError(f"half_weights must have shape ({self.n_score_pixels},)")
        if self.translation_angles.shape != (self.n_fine_trans, 2):
            raise ValueError(f"translation_angles must have shape ({self.n_fine_trans}, 2)")
        if self.highres_xi2_half is not None and self.highres_xi2_half.shape != (self.n_images,):
            raise ValueError(f"highres_xi2_half must have shape ({self.n_images},)")


def prepare_resident_image_operands(
    experiment_dataset,
    image_indices,
    *,
    bucket_io_kwargs: dict,
    half_weights,
    full_to_compact,
    translation_angles,
    window_indices,
    windowed_prepare: bool,
    image_shape,
    current_size,
    n_fine_trans: int,
    use_exact_relion_gaussian: bool,
    accumulate_noise: bool,
    source_faithful_spectrum_norm: bool,
    fine_translation_prior_2d=None,
    score_real_dtype=jnp.float32,
    image_batch_size: int = 256,
) -> ResidentImageOperands:
    """Run ``_prepare_bucket_io`` once per half and keep its operands resident.

    ``image_indices`` are the dataset image indices of this half, in the order
    the pass processes them (RELION particle order); resident row ``i``
    corresponds to ``image_indices[i]``. ``bucket_io_kwargs`` must be exactly
    the keyword arguments the K=1 loop passes to
    :func:`_prepare_bucket_io`, minus the four per-batch positional operands
    (``experiment_dataset``, ``batch``, ``ctf_params``, ``image_indices``) and
    minus ``return_direct_scoring_io``, which is forced on here because the
    fused translate route needs ``direct_score_input``.

    ``image_batch_size`` only controls how many images one ``_prepare_bucket_io``
    call covers; the results are per image, so it does not change the output.
    The last batch is short rather than padded.

    The window gather of ``ctf2_over_nv_half`` and the ``powerClass`` terms
    reproduce the loop's post-prepare block for the score window.
    """

    image_indices = np.asarray(image_indices)
    if image_indices.ndim != 1:
        raise ValueError(f"image_indices must be 1-D, got {image_indices.shape}")
    n_images = int(image_indices.shape[0])
    if n_images == 0:
        raise ValueError("prepare_resident_image_operands needs at least one image")
    if image_batch_size <= 0:
        raise ValueError(f"image_batch_size must be positive, got {image_batch_size}")
    if "return_direct_scoring_io" in bucket_io_kwargs:
        raise ValueError("return_direct_scoring_io is forced on; remove it from bucket_io_kwargs")

    use_window = window_indices is not None
    window_indices_device = (
        None if not use_window else jnp.asarray(window_indices, dtype=jnp.int32)
    )
    # Position of each dataset image index within the requested order, so a
    # fetch that returns its own order is scattered back to that order
    # exactly (a pure gather, no arithmetic).
    position_of = {int(index): position for position, index in enumerate(image_indices.tolist())}
    if len(position_of) != n_images:
        raise ValueError("image_indices must not repeat an image")

    score_input_rows: list[np.ndarray] = [None] * n_images
    corr_img_rows: list[np.ndarray] = [None] * n_images
    xi2_rows: list[np.ndarray] = [None] * n_images
    norm_rows: list[np.ndarray] = [None] * n_images
    n_score_pixels = None
    highres_available = None

    for start in range(0, n_images, image_batch_size):
        batch_image_indices = image_indices[start : start + image_batch_size]
        batch_data, ctf_params, fetched_indices = fetch_indexed_batch(
            experiment_dataset, batch_image_indices
        )
        fetched_indices = np.asarray(fetched_indices)
        prepared = _prepare_bucket_io(
            experiment_dataset,
            jnp.asarray(batch_data),
            ctf_params,
            fetched_indices,
            return_direct_scoring_io=True,
            **bucket_io_kwargs,
        )
        (
            _shifted_score_half,
            _shifted_recon_half,
            batch_norm,
            ctf2_over_nv_half,
            _ctf2_over_nv_half_with_dc,
            _shifted_score_half_with_dc,
            processed_score_half_for_noise,
            _shifted_corrected_score_half,
            direct_score_input,
            *_rest,
        ) = prepared

        if use_window and not windowed_prepare:
            score_input = direct_score_input[:, window_indices_device]
            corr_img_score = ctf2_over_nv_half[:, window_indices_device]
        else:
            score_input = direct_score_input
            corr_img_score = ctf2_over_nv_half
        highres_xi2_half, _norm_high_shell = _relion_powerclass_noise_terms(
            processed_score_half_for_noise,
            image_shape=image_shape,
            current_size=current_size,
            use_exact_relion_gaussian=use_exact_relion_gaussian,
            accumulate_noise=accumulate_noise,
            source_faithful_spectrum_norm=source_faithful_spectrum_norm,
        )
        if highres_available is None:
            highres_available = highres_xi2_half is not None
        elif highres_available != (highres_xi2_half is not None):
            raise ValueError("powerClass tail availability changed between image batches")

        score_input = np.asarray(score_input)
        corr_img_score = np.asarray(corr_img_score)
        batch_norm = np.asarray(batch_norm).reshape(-1)
        highres_host = None if highres_xi2_half is None else np.asarray(highres_xi2_half).reshape(-1)
        if n_score_pixels is None:
            n_score_pixels = int(score_input.shape[1])
        elif int(score_input.shape[1]) != n_score_pixels:
            raise ValueError("score pixel count changed between image batches")

        for local, dataset_index in enumerate(fetched_indices.tolist()):
            position = position_of.get(int(dataset_index))
            if position is None:
                raise ValueError(
                    f"the dataset returned image {dataset_index}, which is not in image_indices",
                )
            score_input_rows[position] = score_input[local]
            corr_img_rows[position] = corr_img_score[local]
            norm_rows[position] = batch_norm[local]
            if highres_host is not None:
                xi2_rows[position] = highres_host[local]

    if any(row is None for row in score_input_rows):
        raise ValueError("the dataset did not return every requested image")

    score_input_device = jnp.asarray(np.stack(score_input_rows, axis=0))
    corr_img_device = jnp.asarray(np.stack(corr_img_rows, axis=0), dtype=score_real_dtype)
    batch_norm_device = jnp.asarray(np.stack(norm_rows, axis=0))
    highres_device = (
        None
        if not highres_available
        else jnp.asarray(np.stack(xi2_rows, axis=0), dtype=jnp.float32)
    )

    if fine_translation_prior_2d is None:
        translation_prior = jnp.zeros((n_images, int(n_fine_trans)), dtype=score_real_dtype)
    else:
        translation_prior = jnp.asarray(
            np.asarray(fine_translation_prior_2d)[image_indices], dtype=score_real_dtype
        )

    return ResidentImageOperands(
        n_images=n_images,
        n_score_pixels=int(n_score_pixels),
        n_fine_trans=int(n_fine_trans),
        current_size=int(current_size),
        score_input=score_input_device,
        corr_img_score=corr_img_device,
        highres_xi2_half=highres_device,
        batch_norm=batch_norm_device,
        translation_prior=translation_prior,
        half_weights=jnp.asarray(half_weights),
        translation_angles=jnp.asarray(translation_angles, dtype=jnp.float32),
        full_to_compact=jnp.asarray(full_to_compact, dtype=jnp.int32),
    )


def score_tilt_image_rows(
    slot_projections,  # complex64 [S * C_R, N] each (slot, row)'s projection at slot * C_R + row
    slot_image_ids,  # int32 [S, C_R] chunk-local image of each row per image slot, -1 past its particle's images
    row_unit_local,  # int32 [C_R] chunk-local particle of each row, sorted
    row_log_prior,  # real [C_R]
    chunk_image,  # complex64 [C_B, N] per tilt image
    chunk_corr,  # real [C_B, N]
    chunk_initial_diff2,  # float32 [C_B]
    unit_translation_prior,  # real [C_U, T] per particle
    candidate_mask,  # bool [C_R, T] or None
    row_is_valid,  # bool [C_R]
    *,
    half_weights,
    image_translation_angles,  # float32 [C_B, T, 2] each tilt image's phases (tomo_particles.tilt_translation_angles)
    full_to_compact,
    logical_current_size,
    unit_capacity: int,
):
    """Flat-row scores of subtomogram particles: each row's diff2 summed over its particle's tilt images.

    The posterior segment is the particle (docs/development/resident_segments.md). Every
    hypothesis row is scored against each of its particle's images, visited in slot order,
    with that image's projection ``Aproj_i R`` (``slot_projections``) and its own phases, and the
    image's diff2 is added to the row's float32 running sum: RELION initialises the weights
    to zero and every image's kernel adds its ``diff2 + initial`` (acc_ml_optimiser_impl.h:1437-1438
    and the ``img_id`` loop from :1490; diff2.cuh:323-328). The minimum, weights and
    significance then see only the per-particle sums. One call over every slot is
    :func:`tilt_image_rows_raw_diff2` and :func:`tilt_rows_scores_from_raw`; a chunk whose projections
    do not fit at once runs the first over blocks of slots, carrying the running sum.
    """

    raw_sum = tilt_image_rows_raw_diff2(
        slot_projections,
        slot_image_ids,
        chunk_image,
        chunk_corr,
        chunk_initial_diff2,
        candidate_mask,
        row_is_valid,
        half_weights=half_weights,
        image_translation_angles=image_translation_angles,
        full_to_compact=full_to_compact,
        logical_current_size=logical_current_size,
    )
    return tilt_rows_scores_from_raw(
        raw_sum,
        row_unit_local,
        row_log_prior,
        unit_translation_prior,
        candidate_mask,
        row_is_valid,
        unit_capacity=unit_capacity,
    )


def materialize_chunk_device(tables: ResidentCandidateTables, chunk: CapacityChunk) -> dict:
    """Host-gather one chunk (T5) and upload its arrays once.

    Returned dict keys match :func:`resident_candidates.materialize_chunk`; the
    array values are device arrays and ``n_valid_rows`` / ``n_valid_images``
    are device int32 scalars, so the scoring program receives them as runtime
    operands rather than as compile keys.
    """

    host = materialize_chunk(tables, chunk)
    return {
        "row_image_local": jnp.asarray(host["row_image_local"], dtype=jnp.int32),
        "row_fine_rot": jnp.asarray(host["row_fine_rot"], dtype=jnp.int32),
        "row_parent_local": jnp.asarray(host["row_parent_local"], dtype=jnp.int32),
        "row_log_prior": jnp.asarray(host["row_log_prior"], dtype=jnp.float32),
        "row_mask_bits": jnp.asarray(host["row_mask_bits"], dtype=jnp.uint32),
        "row_mask_mode": jnp.asarray(host["row_mask_mode"], dtype=jnp.int8),
        "n_valid_rows": jnp.asarray(host["n_valid_rows"], dtype=jnp.int32),
        "n_valid_images": jnp.asarray(host["n_valid_images"], dtype=jnp.int32),
        "image_ids": jnp.asarray(host["image_ids"], dtype=jnp.int32),
    }


def score_all_chunks(
    tables: ResidentCandidateTables,
    chunks: list[CapacityChunk],
    operands: ResidentImageOperands,
    projection_score_cache,
    fine_translation_parent,
    *,
    chunk_arrays: list | None = None,
):
    """Score every chunk of one half, one device program per chunk.

    This Python loop is the only host loop of the stage: every chunk's arrays
    are uploaded before the loop starts (or supplied by the caller through
    ``chunk_arrays``), so the loop itself only dispatches. It returns one
    :class:`ResidentChunkScores` per chunk, in chunk order.
    """

    if chunk_arrays is None:
        chunk_arrays = [materialize_chunk_device(tables, chunk) for chunk in chunks]
    if len(chunk_arrays) != len(chunks):
        raise ValueError("chunk_arrays must have one entry per chunk")

    projection_score_cache = jnp.asarray(projection_score_cache)
    fine_translation_parent = jnp.asarray(fine_translation_parent, dtype=jnp.int32)
    logical_current_size = jnp.asarray(operands.current_size, dtype=jnp.int32)

    results = []
    for chunk, arrays in zip(chunks, chunk_arrays, strict=True):
        results.append(
            score_resident_chunk(
                arrays["row_image_local"],
                arrays["row_fine_rot"],
                arrays["row_log_prior"],
                arrays["row_mask_bits"],
                arrays["row_mask_mode"],
                arrays["n_valid_rows"],
                arrays["image_ids"],
                projection_score_cache,
                operands.score_input,
                operands.corr_img_score,
                operands.highres_xi2_half,
                operands.translation_prior,
                half_weights=operands.half_weights,
                translation_angles=operands.translation_angles,
                full_to_compact=operands.full_to_compact,
                fine_translation_parent=fine_translation_parent,
                logical_current_size=logical_current_size,
                row_capacity=int(chunk.row_capacity),
                image_capacity=int(chunk.image_capacity),
                n_fine_trans=int(operands.n_fine_trans),
                n_score_pixels=int(operands.n_score_pixels),
            )
        )
    return results
