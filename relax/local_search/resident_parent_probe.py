"""The pass-1 parent probe of the resident local search: each image's significant samples, under its own budget.

``run_resident_parent_probe`` runs RELION's local pass 1 on the resident stages: score-only operands, rows
projected into the score window alone (``resident_local_chunk.project_class_rows``), the segmented float32
posterior with the adaptive fraction (or the first-iteration CC winner) and per-image significant samples; no
M-step and no accumulators (admission refuses a ``maximum_significants`` cap). Its chunks
carry no reconstruction operands, so it plans them with its own image ladder and cell bytes. Read by the driver
(``resident_pass2``); imports nothing from it.
"""

from __future__ import annotations

import logging
import time

import jax
import jax.numpy as jnp
import numpy as np

from relax.fine_pass import resident_chunk_inputs, resident_chunk_memory
from relax.fine_pass.bucket_io import prepare_bucket_io
from relax.fine_pass.policy import ResidentConfigurationUnsupported
from relax.fine_pass.resident_candidates import chunk_segment_offsets
from relax.fine_pass.resident_scoring import (
    _flat_rows_scores,
    expand_local_chunk_mask_jnp,
    project_resident_live_rows,
    score_resident_projected_chunk,
    score_resident_projected_chunk_normalized_cc,
)
from relax.fine_pass.scoring import (
    relion_native_score_corr_img,
    relion_powerclass_noise_terms,
    to_relion_native_fine_units,
)
from relax.local_search import resident_local_chunk
from relax.local_search.resident_layout import (
    DEFAULT_ROW_CAPACITY_LADDER,
    materialize_local_chunk,
    plan_local_capacity_chunks,
)
from relax.relion.optics_groups import noise_rows, pixel_rows
from relax.types import LocalEMResult, make_relion_stats

logger = logging.getLogger(__name__)


# Probe chunks carry no reconstruction operands: a row keeps its score-window
# projection and its per-cell score/posterior outputs, an image its score operands.
_PROBE_IMAGE_CAPACITY_LADDER = (32, 128, 512)


_PROBE_CELL_BYTES = 6 * 4  # scores, probs, weights, reconstruction probs, mask, scratch


def _prepare_chunk_score_operands(
    *,
    chunk,
    image_indices,
    experiment_dataset,
    bucket_io_kwargs,
    windowed_prepare,
    score_window_indices,
    fine_translation_prior_2d,
    score_real_dtype,
    n_fine_trans,
    image_shape,
    current_size,
    source_faithful_spectrum_norm,
    normalized_cc=False,
    native_units=False,
):
    """One probe chunk's score operands, from the same preparation as pass 2.

    :func:`~relax.fine_pass.resident_chunk_inputs.prepare_chunk_reconstruction_operands`
    minus every reconstruction, noise and Wavg tile: ``prepare_bucket_io`` runs
    with ``score_only`` and the rows are permuted and padded exactly as there.
    ``normalized_cc`` (the ``--firstiter_cc`` iteration) adds that function's
    normalized-CC operands, the translated corrected score tile and half the
    image power; ``corr_img_score`` is then the CC pixel weight.
    """

    image_indices = np.asarray(image_indices)
    batch_images, padded_ctf_params, fetched_indices, padded_fetched_indices = resident_chunk_inputs.fetch_capacity_batch(
        experiment_dataset, image_indices, chunk.image_capacity
    )
    order = resident_chunk_inputs.reorder_permutation(fetched_indices, image_indices, chunk.image_capacity)
    prepared = prepare_bucket_io(
        experiment_dataset,
        batch_images,
        padded_ctf_params,
        padded_fetched_indices,
        return_direct_scoring_io=True,
        **bucket_io_kwargs,
    )
    batch_norm = prepared[2]
    ctf2_over_nv_half = prepared[3]
    processed_score_half_for_noise = prepared[17]  # the image the power spectrum reads
    shifted_corrected_score_half = prepared[7]
    direct_score_input = prepared[8]
    if windowed_prepare:
        score_input = direct_score_input
        corr_img_score = ctf2_over_nv_half
    else:
        gather_score = jnp.asarray(score_window_indices, dtype=jnp.int32)
        score_input = direct_score_input[:, gather_score]
        corr_img_score = ctf2_over_nv_half[:, gather_score]
    if native_units:
        # RELION's units for its coarse kernel, as resident_chunk_inputs prepares them for the fine kernel.
        native_corr_img = relion_native_score_corr_img(
            pixel_rows(
                noise_rows(
                    bucket_io_kwargs["noise_variance_half"],
                    bucket_io_kwargs.get("noise_optics_groups"),
                    padded_fetched_indices,
                )
            ),
            prepared[16],
            image_shape,
            (
                jnp.asarray(prepared[14], dtype=jnp.float32)[:, None]
                if bucket_io_kwargs["scale_corrections"] is not None
                else None
            ),
            zero_dc=bool(bucket_io_kwargs["half_spectrum_scoring"]),
        )
        if score_window_indices is not None:
            native_corr_img = native_corr_img[:, jnp.asarray(score_window_indices, dtype=jnp.int32)]
        corr_img_score = native_corr_img.astype(corr_img_score.dtype)
        score_input = to_relion_native_fine_units(score_input, int(np.prod(image_shape)))
    body_prediction = bucket_io_kwargs.get("body_prediction")
    highres_xi2_half, _ = relion_powerclass_noise_terms(
        processed_score_half_for_noise,
        image_shape=image_shape,
        current_size=current_size if body_prediction is None else body_prediction.current_size,
        use_exact_relion_gaussian=True,
        accumulate_noise=False,
        source_faithful_spectrum_norm=source_faithful_spectrum_norm,
    )
    permutation = jnp.asarray(order, dtype=jnp.int32)
    valid_images = jnp.asarray(np.arange(chunk.image_capacity) < chunk.n_valid_images, dtype=bool)

    def take(values):
        return resident_chunk_inputs.zero_padded_images(values[permutation], valid_images)

    translation_prior = jnp.asarray(
        np.zeros((chunk.image_capacity, int(n_fine_trans)), dtype=np.float32)
        if fine_translation_prior_2d is None
        else resident_chunk_inputs.pad_batch_to_capacity(np.asarray(fine_translation_prior_2d)[image_indices], chunk.image_capacity),
        dtype=score_real_dtype,
    )
    operands = {
        "score_input": take(score_input),
        "corr_img_score": take(corr_img_score),
        "highres_xi2_half": take(highres_xi2_half),
        "translation_prior": resident_chunk_inputs.zero_padded_images(translation_prior, valid_images),
    }
    if normalized_cc:
        tile = shifted_corrected_score_half.reshape(chunk.image_capacity, int(n_fine_trans), -1)
        if not windowed_prepare:
            tile = tile[:, :, jnp.asarray(score_window_indices, dtype=jnp.int32)]
        operands["score_shifted_cc"] = take(tile)
        operands["cc_half_batch_norm"] = take(0.5 * jnp.reshape(batch_norm, (chunk.image_capacity,)).real)
    return operands


def _coarse_kernel_scores(
    host_chunk, ops, chunk, *, projector, half_weights, full_to_compact, translation_angles, n_fine_trans, current_size,
    image_size, model_max_r, padding_factor,
):
    """The chunk's scores with RELION's coarse kernel instead of the fine one (``relion_coarse=True``).

    Both sum the same terms, in a different order, so their float32 diff2 differ by a few units. RELION's pass 1 of a
    local search is its coarse kernel, and at diff2 about 1000 those units decide cells on the adaptive cut. Each
    image is scored at its own rows (:func:`relax.scoring.exact_cut.direct_rows_diff2`); a row's diff2 does not depend
    on the other rows of the call.
    """

    from relax.fine_pass.scoring import relion_cuda_fine_pixel_weights
    from relax.scoring.exact_cut import direct_rows_diff2

    row_image = np.asarray(host_chunk["row_image_local"], dtype=np.int64)
    n_valid_rows, n_images = int(host_chunk["n_valid_rows"]), int(host_chunk["n_valid_images"])
    rotations = np.asarray(host_chunk["rotations"], dtype=np.float32)
    counts = np.bincount(row_image[:n_valid_rows], minlength=n_images)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    width = int(counts.max())
    # A row's position within its image; an image with fewer rows repeats its first one.
    position = np.minimum(np.arange(width)[None, :], np.maximum(counts[:, None] - 1, 0))
    image_rows = rotations[starts[:, None] + position]
    weights = relion_cuda_fine_pixel_weights(ops["corr_img_score"], jnp.asarray(half_weights)[None, :]).astype(jnp.float32)
    diff2 = direct_rows_diff2(
        projector,
        jnp.asarray(image_rows)[:, None],
        jnp.asarray(ops["score_input"][:n_images], dtype=jnp.complex64)[:, None],
        weights[:n_images][:, None],
        jnp.asarray(ops["highres_xi2_half"][:n_images], dtype=jnp.float32)[:, None],
        jnp.broadcast_to(jnp.asarray(translation_angles, jnp.float32), (n_images, 1, n_fine_trans, 2)),
        jnp.asarray(full_to_compact[: current_size * (current_size // 2 + 1)], dtype=jnp.int32),
        current_size=current_size,
        physical_image_size=image_size,
        model_max_r=model_max_r,
        padding_factor=padding_factor,
    )
    rows_raw = jnp.full((int(chunk.row_capacity), n_fine_trans), jnp.inf, dtype=jnp.float32)
    row_in_image = np.arange(n_valid_rows) - starts[row_image[:n_valid_rows]]
    rows_raw = rows_raw.at[:n_valid_rows].set(diff2[row_image[:n_valid_rows], row_in_image])
    return _flat_rows_scores(
        rows_raw,
        jnp.asarray(row_image, dtype=jnp.int32),
        jnp.asarray(host_chunk["row_log_prior"], dtype=jnp.float32),
        ops["translation_prior"],
        None if host_chunk["row_mask_bits"] is None else expand_local_chunk_mask_jnp(
            jnp.asarray(host_chunk["row_mask_bits"], dtype=jnp.uint8), n_trans=n_fine_trans
        ),
        jnp.arange(int(chunk.row_capacity), dtype=jnp.int32) < n_valid_rows,
        segment_capacity=int(chunk.image_capacity),
    )


def _probe_chunk_bytes(*, row_capacity: int, image_capacity: int, n_score_pixels: int, n_fine_trans: int,
                       complex_bytes: int, projection_transient_bytes: int) -> int:
    """Device bytes one probe chunk holds at its peak, with the previous chunk's outputs.

    A row keeps its score-window projection and its ``T`` posterior cells
    (scores, probabilities, weights, reconstruction probabilities, mask and
    scratch); an image keeps its score input, CTF operand and power terms. One
    projector call is live beside the rows while they are projected. The loop
    reads a chunk's support back after enqueuing the next chunk, so the previous
    chunk's cells are live too.
    """

    t = int(n_fine_trans)
    rows = int(row_capacity) * (int(n_score_pixels) * int(complex_bytes) + t * _PROBE_CELL_BYTES)
    images = int(image_capacity) * (int(n_score_pixels) * (2 * int(complex_bytes) + 8) + t * 4)
    previous = int(row_capacity) * t * _PROBE_CELL_BYTES
    return rows + images + int(projection_transient_bytes) + previous


def run_resident_parent_probe(
    *,
    tables,
    experiment_dataset,
    bucket_io_kwargs,
    fine_translation_prior_2d,
    half_weights,
    full_to_compact,
    translation_angles,
    n_score_pixels,
    mean,
    volume_shape,
    disc_type,
    projection_kwargs,
    projection_padding_factor,
    relion_projector_half,
    relion_projector_r_max,
    relion_projector_capacity_texture,
    precision_policy,
    n_fine_trans,
    adaptive_fraction,
    windowed_prepare,
    window_indices,
    image_shape,
    current_size,
    source_faithful_spectrum_norm,
    overall_t0,
    firstiter_cc=False,
    window_union=None,
    class_projectors=None,
    relion_coarse=False,
    engine_tuning,
) -> LocalEMResult:
    """RELION's local pass 1 (the adaptive parent probe) on the resident stages.

    Scores every parent candidate with the pass-2 scorer, forms the segmented
    float32 posterior with RELION's adaptive-fraction significance, and returns, per image, the significant
    samples as ``posterior_id * n_trans + t`` in the layout's row order: the
    ``reconstruction_sample_indices_by_image`` contract from which the local
    search builds pass 2's support.
    There is no M-step, so no accumulator, statistic or reconstruction operand
    is formed. A chunk's valid rows are projected into the score window alone
    (``window_union``, through the pass's staged texture), and its support is
    read back after the next chunk is enqueued, so the device scores one chunk
    while the host reads the previous one. RELION GPU sorts and sums the pass-1
    weights in float as well (acc_ml_optimiser_impl.h findSignificantPoints);
    a cutoff tie may resolve differently from the exact engine's float64 sort.

    A Class3D layout (``class_projectors``) projects each class's rows with its
    own reference; the significance is joint over the image's classes and poses,
    and a sample's posterior id carries its class (``class * n_bins + bin``).

    ``firstiter_cc`` scores the normalized CC and keeps only each image's best
    sample (RELION's ``--firstiter_cc`` pass 1: its binarized weights leave one
    significant coarse sample, ml_optimiser.cpp:9266-9292).
    """

    from relax.cuda import posterior as cuda_posterior

    t = int(n_fine_trans)
    complex_bytes = np.dtype(precision_policy.score_complex_dtype).itemsize
    projector_output_size = projection_kwargs.get("projector_output_size")
    n_half = int(image_shape[0]) * (int(image_shape[1]) // 2 + 1)
    projection_row_bytes = resident_chunk_memory.projection_call_row_bytes(
        crop_pixels=(
            n_half
            if projector_output_size is None
            else int(projector_output_size) * (int(projector_output_size) // 2 + 1)
        ),
        output_pixels=n_half if window_union is None else int(window_union.indices.shape[0]),
        n_score_pixels=int(n_score_pixels),
        n_recon_pixels=0,
        complex_bytes=int(np.dtype(relion_projector_half.dtype).itemsize),
    )
    budget = resident_chunk_memory.resident_chunk_budget_bytes()
    projection_block_rows = resident_local_chunk.projection_block_rows(
        projection_row_bytes, budget, engine_tuning.local_projection_call_max_bytes,
    )
    row_ladder = tuple(int(v) for v in engine_tuning.local_row_capacities or DEFAULT_ROW_CAPACITY_LADDER)
    image_ladder = tuple(int(v) for v in engine_tuning.local_image_capacities or _PROBE_IMAGE_CAPACITY_LADDER)

    def peak(rows, images):
        return _probe_chunk_bytes(
            row_capacity=max(max(rows), resident_chunk_memory.overflow_row_capacity(resident_chunk_memory.max_image_rows(tables.row_offsets), rows)),
            image_capacity=max(images),
            n_score_pixels=int(n_score_pixels),
            n_fine_trans=t,
            complex_bytes=complex_bytes,
            projection_transient_bytes=min(projection_block_rows, max(rows)) * projection_row_bytes,
        )

    if budget is not None:
        while len(row_ladder) > 1 and peak(row_ladder, image_ladder) > budget:
            row_ladder = row_ladder[:-1]
        while len(image_ladder) > 1 and peak(row_ladder, image_ladder) > budget:
            image_ladder = image_ladder[:-1]
        if peak(row_ladder, image_ladder) > budget:
            raise ResidentConfigurationUnsupported(
                "The device-resident local pass-1 parent probe does not implement this configuration: "
                f"its smallest chunk needs {peak(row_ladder, image_ladder) / 1024**3:.2f} "
                f"GiB against a {budget / 1024**3:.2f} GiB budget"
            )
    chunks = plan_local_capacity_chunks(tables, row_capacity_ladder=row_ladder, image_capacity_ladder=image_ladder)
    logger.info(
        "Resident local pass-1 probe plan: %d images, %d candidate rows, %d translations -> %d chunks "
        "(row capacities %s, image capacities %s, projection block rows %d, projection window %s px); "
        "chunk peak %.2f GiB of a %s budget",
        tables.n_images,
        tables.n_rows,
        t,
        len(chunks),
        ",".join(str(v) for v in row_ladder),
        ",".join(str(v) for v in image_ladder),
        projection_block_rows,
        "full" if window_union is None else int(window_union.indices.shape[0]),
        peak(row_ladder, image_ladder) / float(1024**3),
        resident_chunk_memory.format_budget_gib(budget),
    )

    sample_ids_by_image: list[np.ndarray] = [np.zeros(0, dtype=np.int64)] * tables.n_images
    log_evidence = np.zeros(tables.n_images, dtype=np.float64)
    best_log_score = np.zeros(tables.n_images, dtype=np.float64)
    max_posterior = np.zeros(tables.n_images, dtype=np.float64)
    hard_assignments = np.full(tables.n_images, -1, dtype=np.int64)

    def start(chunk):
        """Enqueue one chunk up to its posterior; returns what :func:`finish` reads back."""

        image_indices = np.arange(chunk.image_start, chunk.image_stop, dtype=np.int64)
        host_chunk = materialize_local_chunk(tables, chunk)
        ops = _prepare_chunk_score_operands(
            chunk=chunk,
            image_indices=image_indices,
            experiment_dataset=experiment_dataset,
            bucket_io_kwargs=bucket_io_kwargs,
            windowed_prepare=windowed_prepare,
            score_window_indices=window_indices,
            fine_translation_prior_2d=fine_translation_prior_2d,
            score_real_dtype=precision_policy.score_real_dtype,
            n_fine_trans=t,
            image_shape=image_shape,
            current_size=current_size,
            source_faithful_spectrum_norm=source_faithful_spectrum_norm,
            normalized_cc=bool(firstiter_cc),
            native_units=bool(relion_coarse),
        )
        # Host rotations: the coarse kernel's wrapped-row check reads each
        # block's first rotation, which on a device array waits for the queue.
        projection_options = dict(
            score_indices=window_indices,
            recon_indices=None,
            max_projected_rotations=int(projection_block_rows),
            output_complex_dtype=precision_policy.score_complex_dtype,
            output_abs2_dtype=precision_policy.score_real_dtype,
            relion_projector_r_max=relion_projector_r_max,
            projection_padding_factor=projection_padding_factor,
            window_union=window_union,
            **projection_kwargs,
        )
        if class_projectors is None:
            score_proj, _, _, _ = project_resident_live_rows(
                mean,
                np.asarray(host_chunk["rotations"], dtype=np.dtype(precision_policy.score_real_dtype)),
                image_shape,
                volume_shape,
                disc_type,
                n_valid_rows=int(chunk.n_valid_rows),
                relion_projector_half=relion_projector_half,
                relion_projector_capacity_texture=relion_projector_capacity_texture,
                **projection_options,
            )
        else:
            (score_proj, _, _), _ = resident_local_chunk.project_class_rows(
                host_chunk,
                class_projectors,
                image_shape=image_shape,
                volume_shape=volume_shape,
                disc_type=disc_type,
                n_valid_rows=int(chunk.n_valid_rows),
                rotation_dtype=precision_policy.score_real_dtype,
                projection_options=projection_options,
                host_rotations=True,
            )
        n_valid_images_device = jnp.asarray(host_chunk["n_valid_images"], dtype=jnp.int32)
        chunk_image_ids = jnp.where(
            jnp.arange(chunk.image_capacity, dtype=jnp.int32) < n_valid_images_device,
            jnp.arange(chunk.image_capacity, dtype=jnp.int32),
            jnp.int32(-1),
        )
        segment_offsets_np = chunk_segment_offsets(tables, chunk, n_fine_trans=t)
        segment_offsets = jnp.asarray(segment_offsets_np, dtype=jnp.int32)
        if firstiter_cc:
            row_image_local = jnp.asarray(host_chunk["row_image_local"], dtype=jnp.int32)
            scored = score_resident_projected_chunk_normalized_cc(
                score_proj,
                row_image_local,
                None if host_chunk["row_mask_bits"] is None else jnp.asarray(host_chunk["row_mask_bits"], dtype=jnp.uint8),
                jnp.asarray(host_chunk["n_valid_rows"], dtype=jnp.int32),
                ops["score_shifted_cc"],
                ops["corr_img_score"],
                ops["cc_half_batch_norm"],
                half_weights=half_weights,
                full_to_compact=full_to_compact,
                row_capacity=int(chunk.row_capacity),
                n_fine_trans=t,
                block_rows=resident_local_chunk.cc_block_rows(int(chunk.row_capacity)),
            )
            del score_proj, ops
            log_z_out, best_log, best_cell, max_post, winner, n_significant = resident_local_chunk.local_firstiter_cc_posterior(
                scored.scores,
                row_image_local,
                jnp.arange(int(chunk.row_capacity), dtype=jnp.int32) < int(chunk.n_valid_rows),
                segment_offsets,
                n_valid_images_device,
                image_capacity=chunk.image_capacity,
            )
            weights = winner.reshape(-1)
            mask = weights > 0
        else:
            if relion_coarse:
                scored = _coarse_kernel_scores(
                    host_chunk, ops, chunk,
                    projector=relion_projector_capacity_texture,
                    half_weights=half_weights,
                    full_to_compact=full_to_compact,
                    translation_angles=translation_angles,
                    n_fine_trans=t,
                    current_size=int(current_size),
                    image_size=1,  # the kernel multiplies the projector by -image_size^2: native units
                    model_max_r=int(relion_projector_capacity_texture.logical_r_max),
                    padding_factor=int(projection_padding_factor),
                )
            else:
                scored = score_resident_projected_chunk(
                    score_proj,
                    jnp.asarray(host_chunk["row_image_local"], dtype=jnp.int32),
                    jnp.asarray(host_chunk["row_log_prior"], dtype=jnp.float32),
                    None if host_chunk["row_mask_bits"] is None else jnp.asarray(host_chunk["row_mask_bits"], dtype=jnp.uint8),
                    jnp.asarray(host_chunk["n_valid_rows"], dtype=jnp.int32),
                    chunk_image_ids,
                    ops["score_input"],
                    ops["corr_img_score"],
                    ops["highres_xi2_half"],
                    ops["translation_prior"],
                    half_weights=half_weights,
                    translation_angles=translation_angles,
                    full_to_compact=full_to_compact,
                    logical_current_size=jnp.asarray(current_size, dtype=jnp.int32),
                    row_capacity=int(chunk.row_capacity),
                    image_capacity=chunk.image_capacity,
                    n_fine_trans=t,
                    n_score_pixels=int(n_score_pixels),
                )
            del score_proj, ops
            scores_flat = jnp.asarray(scored.scores, dtype=jnp.float32).reshape(-1)
            log_z = cuda_posterior.sparse_pass2_segmented_log_z_f64(
                scores_flat, segment_offsets, n_valid_images_device
            )
            (
                log_z_out,
                best_log,
                best_cell,
                max_post,
                _probs,
                weights,
                _reconstruction_probs,
                mask,
                n_significant,
                _sum_weight,
                _threshold,
            ) = cuda_posterior.sparse_pass2_segmented_posterior_f32(
                scores_flat,
                segment_offsets,
                n_valid_images_device,
                log_z,
                jnp.ones((chunk.image_capacity,), dtype=jnp.float32),
                adaptive_fraction=float(adaptive_fraction),
                keep_all=False,
                use_external_sum_weight=False,
            )
        device = (mask, best_cell, log_z_out, best_log, max_post, scored.min_diff2)
        return chunk, host_chunk, segment_offsets_np, device

    def finish(chunk, host_chunk, segment_offsets_np, device):
        """Read one chunk's support back and record it per image."""

        mask, best_cell, log_z_out, best_log, max_post, min_diff2 = device
        mask_np, best_cell_np, log_z_np, best_log_np, max_post_np, min_diff2_np = jax.device_get(
            (mask, best_cell, log_z_out, best_log, max_post, min_diff2)
        )
        mask_np = np.asarray(mask_np, dtype=bool).reshape(chunk.row_capacity, t)[:chunk.n_valid_rows]
        image_row_start = segment_offsets_np.astype(np.int64)[:chunk.n_valid_images] // t
        rows, cols = np.nonzero(mask_np)
        posterior_ids = host_chunk["row_posterior_id"][:chunk.n_valid_rows].astype(np.int64)
        sample_ids = posterior_ids[rows] * np.int64(t) + cols.astype(np.int64)
        row_image = host_chunk["row_image_local"][:chunk.n_valid_rows][rows]
        # np.nonzero is row-major, and an image's rows are contiguous, so each
        # image's samples come out in the exact engine's (row, translation) order.
        bounds = np.searchsorted(row_image, np.arange(chunk.n_valid_images + 1))
        for local in range(chunk.n_valid_images):
            image = int(chunk.image_start) + local
            sample_ids_by_image[image] = sample_ids[bounds[local] : bounds[local + 1]]
            best_row = image_row_start[local] + int(best_cell_np[local]) // t
            hard_assignments[image] = (
                int(host_chunk["row_rotation_id"][best_row]) * t + int(best_cell_np[local]) % t
            )
        sl = slice(int(chunk.image_start), int(chunk.image_stop))
        # The scores are centred on RELION's common minimum; the evidences are reported absolute,
        # as pass 2 reports them (its log_score_offset is -min_diff2).
        min_diff2_np = np.asarray(min_diff2_np, dtype=np.float64)[:chunk.n_valid_images]
        log_evidence[sl] = np.asarray(log_z_np, dtype=np.float64)[:chunk.n_valid_images] - min_diff2_np
        best_log_score[sl] = np.asarray(best_log_np, dtype=np.float64)[:chunk.n_valid_images] - min_diff2_np
        max_posterior[sl] = np.asarray(max_post_np)[:chunk.n_valid_images]

    loop_t0 = time.time()
    pending = None
    for chunk in chunks:
        started = start(chunk)
        if pending is not None:
            finish(*pending)
        pending = started
    if pending is not None:
        finish(*pending)
    loop_s = time.time() - loop_t0

    if any(ids.size == 0 for ids in sample_ids_by_image):
        raise RuntimeError("Resident local pass-1 probe: an image kept no significant sample")
    stats = make_relion_stats(
        log_evidence_per_image=log_evidence,
        best_log_score_per_image=best_log_score,
        max_posterior_per_image=max_posterior,
        rotation_posterior_sums=np.zeros(0, dtype=np.float64),
        host_arrays=True,
    )
    profile = {
        "reconstruction_sample_indices_by_image": tuple(sample_ids_by_image),
        "resident_probe_chunks": np.int32(len(chunks)),
        "resident_probe_loop_time_s": np.float64(loop_s),
        "resident_probe_total_time_s": np.float64(time.time() - overall_t0),
    }
    logger.info(
        "Resident local pass-1 probe done: %d images, %d chunks, %d significant samples, "
        "%.2fs chunk loop, %.2fs total",
        tables.n_images,
        len(chunks),
        int(sum(ids.size for ids in sample_ids_by_image)),
        loop_s,
        time.time() - overall_t0,
    )
    return LocalEMResult(
        Ft_y=None,
        Ft_ctf=None,
        hard_assignments=hard_assignments,
        stats=stats,
        profile=profile,
    )
