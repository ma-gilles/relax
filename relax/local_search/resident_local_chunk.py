"""One capacity chunk of the resident local-search pass 2.

``start_resident_local_chunk`` enqueues one chunk: its operands (pre-shifted tiles or the unshifted operands),
each class's row projections (``project_class_rows``, a projector call at most ``projection_block_rows`` rows),
the row-blocked scoring or the first-iteration CC posterior, the M-step rows live first, and each class's M-step
terms. The parent probe (``resident_parent_probe``) shares the projection, the CC block rows and the CC
posterior. Imports nothing from the driver (``resident_pass2``) or the probe.
"""

from __future__ import annotations

import logging
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from relax.fine_pass import (
    resident_chunk_inputs,
    resident_chunk_posterior,
    resident_chunk_programs,
    resident_chunk_records,
    resident_image_terms,
    resident_mstep_blocks,
)
from relax.fine_pass.resident_candidates import chunk_segment_offsets, expand_local_chunk_mask_jnp
from relax.fine_pass.resident_operands import ResidentOperandsUnsupported
from relax.fine_pass.resident_scoring import (
    project_resident_live_rows,
    score_resident_chunk_in_row_blocks,
    score_resident_chunk_normalized_cc_in_row_blocks,
    score_resident_projected_chunk,
    score_resident_projected_chunk_normalized_cc,
)
from relax.local_search.resident_layout import materialize_local_chunk

logger = logging.getLogger(__name__)


# The share of the chunk budget one projector call may take; the rest is the chunk's rows, tiles and M-step block.
_PROJECTION_CALL_BUDGET_SHARE = 0.5


def projection_block_rows(projection_row_bytes: int, chunk_budget_bytes: int | None, call_max_bytes: int) -> int:
    """Rows per projector call: the per-call cap (``call_max_bytes``, ``EngineTuning.local_projection_call_max_bytes``),
    lowered to ``_PROJECTION_CALL_BUDGET_SHARE`` of the chunk budget.

    The fixed 4 GiB call was most of the smallest chunk at a 16 GB card's full-box final pass (box 448: 4.28 GiB
    against a 3.63 GiB budget, relax#49). Fewer rows per call change only how the calls group the rows; each row's
    projection is the same. Large cards keep the cap.
    """

    max_bytes = call_max_bytes
    if chunk_budget_bytes is not None:
        max_bytes = min(max_bytes, int(_PROJECTION_CALL_BUDGET_SHARE * int(chunk_budget_bytes)))
    return max(1, int(max_bytes) // int(projection_row_bytes))


def cc_block_rows(row_capacity: int) -> int:
    """Rows per normalized-CC scoring block: the largest power of two up to 128 dividing the capacity."""

    block = 128
    while int(row_capacity) % block:
        block //= 2
    return block


def local_firstiter_cc_posterior(
    scores, row_image_local, row_is_valid, segment_offsets, n_valid_images, *, image_capacity: int
):
    """RELION's ``--firstiter_cc`` posterior of a local chunk's normalized-CC scores.

    RELION zeroes every weight but the best one (ml_optimiser.cpp:9266-9292), so
    the posterior is one-hot at each image's first maximum in segment order and
    Pmax is 1, as in the global pass (``resident_chunk_posterior.winner_take_all_posterior``).
    log-Z is the log-sum-exp of the CC scores, the evidence the global pass reports.
    Returns ``(log_z, best_log_score, best_cell_index, max_posterior, row_posterior,
    n_significant)``.
    """

    from relax.cuda import posterior as cuda_posterior

    scores = jnp.asarray(scores, dtype=jnp.float32)
    log_z = cuda_posterior.sparse_pass2_segmented_log_z_f64(scores.reshape(-1), segment_offsets, n_valid_images)
    best_log_score, best_cell_index, row_posterior = resident_chunk_posterior.winner_take_all_cells(
        scores, row_image_local, row_is_valid, segment_offsets, image_capacity=image_capacity
    )
    has_winner = jnp.isfinite(best_log_score)
    return (
        jnp.asarray(log_z, dtype=jnp.float64),
        best_log_score,
        best_cell_index,
        has_winner.astype(jnp.float32),
        row_posterior,
        has_winner.astype(jnp.int32),
    )


def _unshifted_chunk_operands(experiment_dataset, image_indices, **kwargs):
    """:func:`resident_chunk_inputs.unshifted_chunk_operands`, or ``None`` where they are refused."""

    try:
        return resident_chunk_inputs.unshifted_chunk_operands(experiment_dataset, image_indices, **kwargs)
    except ResidentOperandsUnsupported as reason:
        logger.info(
            "Resident local pass-2 keeps the pre-shifted translation tiles: %s", reason
        )
        return None


class _LiveRowsFirst(NamedTuple):
    """A chunk's M-step rows, live rows first (see :func:`_live_rows_first`)."""

    row_ids: jax.Array  # int32 [C_R]: the chunk row at each M-step position
    row_image_local: jax.Array  # int32 [C_R]
    kernel_row_image_ids: jax.Array  # int32 [C_R]
    row_posterior: jax.Array  # float32 [C_R, T]


@jax.jit
def _live_rows_first(row_posterior, row_is_valid, row_image_local, kernel_row_image_ids):
    """Order a chunk's rows live first (stable) and count the live ones.

    A row is live when it is valid and has a positive reconstruction weight.
    The M-step sums are taken in a different row grouping than chunk order;
    each row's contribution is unchanged.
    """

    live = row_is_valid & jnp.any(row_posterior > 0, axis=1)
    order = jnp.argsort(jnp.where(live, 0, 1).astype(jnp.int32), stable=True).astype(jnp.int32)
    rows = _LiveRowsFirst(
        row_ids=order,
        row_image_local=row_image_local[order],
        kernel_row_image_ids=kernel_row_image_ids[order],
        row_posterior=row_posterior[order],
    )
    return rows, jnp.sum(live, dtype=jnp.int32)


class _ClassRows(NamedTuple):
    """The two fields of a chunk's rows :func:`resident_chunk_posterior.class_sub_segment_posterior` reads."""

    classes: object  # resident_chunk_programs.ChunkClassLayout
    n_valid_images: object  # int32 device scalar


def _sum_class_mstep_terms(
    class_terms, class_scale_masks_rect, *, class_layout, class_posterior, row_start, row_capacity, n_fine_trans
):
    """A Class3D chunk's M-step terms summed over classes, and the class fields of its statistics.

    ``class_terms`` are each class's ``(wavg triplet, noise shells, A2, XA)`` from its own M-step. The
    noise residual and the norm terms add over classes; each class's Wavg XA/AA enter the scale sums
    only under its own ``data_vs_prior_class > 3`` mask (acc_ml_optimiser_impl.h:4893-4912), so they are
    folded here and cleared from the triplet, as ``resident_chunk_stages.fold_class_scale_sums`` does for
    the global pass. A class's winner is a row of the layout's flat order, ``row * T + t``.
    """

    zero = jnp.float32(0.0)
    triplet = noise_shells = a2 = xa = scale_xa = scale_aa = None
    for k, (triplet_k, noise_k, a2_k, xa_k) in enumerate(class_terms):
        mask = jnp.asarray(class_scale_masks_rect[k], dtype=bool).reshape(1, -1)
        xa_scale = jnp.sum(jnp.where(mask, triplet_k[:, :, 0], zero).astype(jnp.float64), axis=1)
        aa_scale = jnp.sum(jnp.where(mask, triplet_k[:, :, 1], zero).astype(jnp.float64), axis=1)
        diff2 = triplet_k.at[:, :, :2].set(zero)
        if triplet is None:
            triplet, noise_shells, a2, xa, scale_xa, scale_aa = diff2, noise_k, a2_k, xa_k, xa_scale, aa_scale
        else:
            triplet, noise_shells = triplet + diff2, noise_shells + noise_k
            a2, xa = a2 + a2_k, xa + xa_k
            scale_xa, scale_aa = scale_xa + xa_scale, scale_aa + aa_scale
    t = jnp.int64(int(n_fine_trans))
    class_best_row = jnp.clip(
        class_layout.segment_row_start + class_posterior.best_cell_index // t, 0, jnp.int64(max(row_capacity - 1, 0))
    )
    class_fields = dict(
        row_class=class_layout.row_class,
        per_class_log_z=class_posterior.log_z,
        per_class_best_log_score=class_posterior.best_log_score,
        per_class_best_cell=jnp.where(
            jnp.isfinite(class_posterior.best_log_score),
            (jnp.int64(int(row_start)) + class_best_row) * t + class_posterior.best_cell_index % t,
            jnp.int64(-1),
        ),
        scale_xa_per_image=scale_xa,
        scale_aa_per_image=scale_aa,
    )
    return triplet, noise_shells, a2, xa, class_fields


def project_class_rows(
    host_chunk,
    class_projectors,
    *,
    image_shape,
    volume_shape,
    disc_type,
    n_valid_rows: int,
    rotation_dtype,
    projection_options: dict,
    host_rotations: bool = False,
):
    """Project a Class3D chunk's rows, each class's with its own reference, into the chunk's row order.

    Rows stay image-major then class-major, as the joint posterior needs, so a class's rows are not
    contiguous: each class's rotations are gathered to the front of a capacity-sized array and its
    projections written back at their rows (``place_rows`` of :func:`project_resident_live_rows`).
    ``host_rotations`` hands the projector host rotations, as the pass-1 probe does.
    Returns ``((score_proj, recon_proj, recon_abs2), n_projected_rows)``.
    """

    row_capacity = int(np.shape(host_chunk["rotations"])[0])
    row_class = np.asarray(host_chunk["row_class"])[: int(n_valid_rows)]
    outputs = None
    n_projected = 0
    for k, projector in enumerate(class_projectors):
        rows = np.flatnonzero(row_class == k).astype(np.int32)
        if rows.size == 0:
            continue
        rotations = np.broadcast_to(np.eye(3, dtype=np.dtype(rotation_dtype)), (row_capacity, 3, 3)).copy()
        rotations[: rows.size] = np.asarray(host_chunk["rotations"])[rows]
        place = np.arange(row_capacity, 2 * row_capacity, dtype=np.int32)
        place[: rows.size] = rows
        score_proj, recon_proj, recon_abs2, n_class_projected = project_resident_live_rows(
            projector.mean,
            rotations if host_rotations else jnp.asarray(rotations, dtype=rotation_dtype),
            image_shape,
            volume_shape,
            disc_type,
            n_valid_rows=int(rows.size),
            relion_projector_half=projector.slab,
            relion_projector_capacity_texture=projector.texture,
            place_rows=place,
            outputs=outputs,
            **projection_options,
        )
        outputs = (score_proj, recon_proj, recon_abs2)
        n_projected += int(n_class_projected)
    if outputs is None:
        raise RuntimeError("a Class3D local chunk has no valid row")
    return outputs, n_projected


def start_resident_local_chunk(
    chunk,
    *,
    tables,
    posterior_bins,
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
    projection_block_rows,
    projection_padding_factor,
    relion_projector_half,
    relion_projector_r_max,
    precision_policy,
    n_fine_trans,
    n_recon_windowed,
    n_rect,
    mstep_block_rows,
    adaptive_fraction,
    keep_all_weights,
    windowed_prepare,
    window_indices,
    recon_window_indices,
    window_union,
    relion_x_half_recon_indices,
    exact_positions_device,
    rect_indices_device,
    image_shape,
    current_size,
    mstep_current_size,
    recon_volume_shape,
    wsum_current_size=None,
    mstep_max_r=None,
    max_adjoint_block_bytes,
    noise_variance_for_noise,
    shell_indices_noise,
    group_ids_np,
    scale_corrections_np,
    translation_prior_centers_np,
    fine_translations,
    voxel_size,
    accumulate_noise,
    source_faithful_spectrum_norm,
    optics_groups_np=None,
    stats_config,
    image_tables,
    operand_route,
    firstiter_cc=False,
    relion_projector_capacity_texture=None,
    class_projectors=None,
    class_scale_masks_rect=None,
    lone_block_rows=None,
    row_partial=None,
    engine_tuning,
):
    """Enqueue one local capacity chunk's front stages; return its ``finish``.

    The front is the chunk's operands, projections, scores, segmented
    posterior and live-row order, none of which reads the half's running
    accumulators. ``finish(Ft_y_total, Ft_ctf_total, stats)`` reads the live row
    count back, runs the M-step and the image statistics and returns the
    updated accumulators, so the caller can enqueue the next chunk's front
    before it finishes this one.

    ``operand_route`` is the half's mutable choice of reconstruction operands:
    ``{"unshifted": True}`` prepares the chunk's unshifted per-image operands
    and lets the M-step translate them inside T15's translate-and-sum kernel,
    as the global pass does (T16); ``False`` keeps the pre-shifted
    ``[images, translations, pixels]`` tiles and the XLA reduction. A
    configuration the unshifted preparation refuses switches the half to the
    tiles at its first chunk.

    ``firstiter_cc`` scores the chunk's rows with RELION's normalized CC and puts each image's
    whole posterior on its best cell (:func:`local_firstiter_cc_posterior`).

    The only host work inside is the chunk's operand upload, the T7 offsets
    readback the segmented posterior performs internally, the optional
    significance-count pull and ``finish``'s live row count; no per-chunk
    result is otherwise brought back.

    ``lone_block_rows`` is set for one image's overflow chunk: its rows are projected and scored that many at a
    time, keeping only their ``[C_R, T]`` scores (:func:`~relax.fine_pass.resident_scoring.score_resident_chunk_in_row_blocks`,
    shared with the global pass), the posterior is formed over all of them at once, and each M-step block projects
    its own rows. Each row's projection and score are those of the one-call chunk; only the grouping of the
    projector calls changes. In a Class3D pass each block's rows project with their own class's reference (the
    lone image's rows are class-major, so a block spans few classes) and each class's M-step blocks with its own.
    The ``--firstiter_cc`` iteration scores the blocks with the normalized-CC block core
    (:func:`~relax.fine_pass.resident_scoring.score_resident_chunk_normalized_cc_in_row_blocks`).
    """

    from relax.cuda import posterior as cuda_posterior

    if lone_block_rows is not None:
        if int(lone_block_rows) % int(mstep_block_rows) or int(chunk.row_capacity) % int(lone_block_rows):
            raise ValueError(
                f"lone row blocks of {lone_block_rows} must divide the chunk's {chunk.row_capacity} rows and be a "
                f"multiple of the M-step's {mstep_block_rows}-row blocks"
            )
    image_indices = np.arange(chunk.image_start, chunk.image_stop, dtype=np.int64)
    # The operands' powerClass terms sum above the weighted sums' size (--strict_highres_exp).
    operand_current_size = current_size if wsum_current_size is None else wsum_current_size

    profile = engine_tuning.local_chunk_profile
    marks: dict[str, float] = {}

    def mark(name, *values):
        if not profile:
            return
        if values:
            jax.block_until_ready(values)
        marks[name] = time.time()

    mark("t0")
    host_chunk = materialize_local_chunk(tables, chunk)
    row_image_local = jnp.asarray(host_chunk["row_image_local"], dtype=jnp.int32)
    row_log_prior = jnp.asarray(host_chunk["row_log_prior"], dtype=jnp.float32)
    row_mask_bits = (
        None
        if host_chunk["row_mask_bits"] is None
        else jnp.asarray(host_chunk["row_mask_bits"], dtype=jnp.uint8)
    )
    image_ids = jnp.asarray(host_chunk["image_ids"], dtype=jnp.int32)
    n_valid_rows_device = jnp.asarray(host_chunk["n_valid_rows"], dtype=jnp.int32)
    n_valid_images_device = jnp.asarray(host_chunk["n_valid_images"], dtype=jnp.int32)
    row_is_valid = jnp.arange(chunk.row_capacity, dtype=jnp.int32) < n_valid_rows_device
    kernel_row_image_ids = jnp.where(row_is_valid, row_image_local, jnp.int32(-1))

    # --- stage 0: this chunk's operands, both families, at image capacity ---
    # Upstream merged the per-half score preparation into this call, so
    # ``prepare_bucket_io`` runs once per image instead of twice and every
    # per-chunk program is keyed on the image-capacity class rather than on
    # the chunk's occupancy.
    recon = None
    if operand_route["unshifted"]:
        recon = _unshifted_chunk_operands(
            experiment_dataset,
            image_indices,
            image_capacity=chunk.image_capacity,
            bucket_io_kwargs=bucket_io_kwargs,
            window_indices=window_indices,
            recon_window_indices=recon_window_indices,
            rect_indices_device=rect_indices_device,
            exact_positions_device=exact_positions_device,
            translation_angles=translation_angles,
            noise_shell_indices_half=image_tables.shell_indices_half,
            n_noise_shells=int(stats_config.n_shells),
            image_shape=image_shape,
            current_size=operand_current_size,
            n_fine_trans=int(n_fine_trans),
            accumulate_noise=accumulate_noise,
            source_faithful_spectrum_norm=bool(source_faithful_spectrum_norm),
            fine_translation_prior_2d=fine_translation_prior_2d,
            scale_corrections_np=scale_corrections_np,
            group_ids_np=group_ids_np,
            optics_groups_np=optics_groups_np,
            precision_policy=precision_policy,
        )
        if recon is None:
            operand_route["unshifted"] = False
    if recon is None:
        recon = resident_chunk_inputs.prepare_chunk_reconstruction_operands(
            chunk=chunk,
            image_indices=image_indices,
            experiment_dataset=experiment_dataset,
            bucket_io_kwargs=bucket_io_kwargs,
            windowed_prepare=windowed_prepare,
            recon_window_indices=recon_window_indices,
            n_fine_trans=int(n_fine_trans),
            n_recon_windowed=int(n_recon_windowed),
            image_shape=image_shape,
            current_size=operand_current_size,
            use_exact_relion_gaussian=True,
            accumulate_noise=accumulate_noise,
            source_faithful_spectrum_norm=bool(source_faithful_spectrum_norm),
            score_window_indices=window_indices,
            fine_translation_prior_2d=fine_translation_prior_2d,
            score_real_dtype=precision_policy.score_real_dtype,
            relion_score_translation_angles=translation_angles,
            rect_indices_device=rect_indices_device,
            exact_positions_device=exact_positions_device,
            scale_corrections_np=scale_corrections_np,
            group_ids_np=group_ids_np,
            optics_groups_np=optics_groups_np,
            noise_shell_indices_half=image_tables.shell_indices_half,
            n_noise_shells=int(stats_config.n_shells),
            normalized_cc=bool(firstiter_cc),
        )
    recon_operand = recon["recon_image"] if recon.get("recon_image") is not None else recon["shifted_recon"]

    mark("operands", recon_operand, recon["score_input"])

    # --- stages 1-2: project this chunk's own rows -------------------------
    # Only the valid rows: the padding past them (about 70% of a chunk at the
    # 10097 local iterations) is never scored or reconstructed.
    projection_options = dict(
        score_indices=window_indices,
        recon_indices=recon_window_indices,
        max_projected_rotations=int(projection_block_rows),
        output_complex_dtype=precision_policy.score_complex_dtype,
        output_abs2_dtype=precision_policy.score_real_dtype,
        relion_projector_r_max=relion_projector_r_max,
        projection_padding_factor=projection_padding_factor,
        window_union=window_union,
        **projection_kwargs,
    )
    chunk_rotations = jnp.asarray(host_chunk["rotations"], dtype=precision_policy.score_real_dtype)

    def project_rows(rotations, n_valid_rows, projector=None):
        """Rows projected with the pass's reference, or with one Class3D class's ``projector``."""

        if projector is not None:
            return project_resident_live_rows(
                projector.mean,
                rotations,
                image_shape,
                volume_shape,
                disc_type,
                n_valid_rows=n_valid_rows,
                relion_projector_half=projector.slab,
                relion_projector_capacity_texture=projector.texture,
                **projection_options,
            )
        return project_resident_live_rows(
            mean,
            rotations,
            image_shape,
            volume_shape,
            disc_type,
            n_valid_rows=n_valid_rows,
            relion_projector_half=relion_projector_half,
            relion_projector_capacity_texture=relion_projector_capacity_texture,
            **projection_options,
        )

    def project_score_block(start, stop):
        """Score-window projections of chunk rows ``start:stop``, each row with its own class's reference."""

        n_valid = max(0, min(int(chunk.n_valid_rows), stop) - start)
        if class_projectors is None:
            return project_rows(chunk_rotations[start:stop], n_valid)[0]
        if n_valid == 0:
            return jnp.zeros((stop - start, int(n_score_pixels)), dtype=precision_policy.score_complex_dtype)
        (score_block, _, _), _ = project_class_rows(
            {"rotations": np.asarray(host_chunk["rotations"])[start:stop],
             "row_class": np.asarray(host_chunk["row_class"])[start:stop]},
            class_projectors,
            image_shape=image_shape,
            volume_shape=volume_shape,
            disc_type=disc_type,
            n_valid_rows=n_valid,
            rotation_dtype=precision_policy.score_real_dtype,
            projection_options=projection_options,
        )
        return score_block

    if lone_block_rows is not None:
        score_proj = recon_proj = recon_abs2 = None
        n_projected_rows = int(chunk.n_valid_rows)
    elif class_projectors is None:
        score_proj, recon_proj, recon_abs2, n_projected_rows = project_rows(chunk_rotations, chunk.n_valid_rows)
    else:
        (score_proj, recon_proj, recon_abs2), n_projected_rows = project_class_rows(
            host_chunk,
            class_projectors,
            image_shape=image_shape,
            volume_shape=volume_shape,
            disc_type=disc_type,
            n_valid_rows=chunk.n_valid_rows,
            rotation_dtype=precision_policy.score_real_dtype,
            projection_options=projection_options,
        )

    mark("project", score_proj, recon_proj, recon_abs2)

    segment_offsets_np = chunk_segment_offsets(tables, chunk, n_fine_trans=n_fine_trans)
    segment_offsets = jnp.asarray(segment_offsets_np, dtype=jnp.int32)
    if firstiter_cc:
        # --- stages 3-4, --firstiter_cc: normalized CC, winner takes all ---
        if lone_block_rows is not None:

            def cc_block_reference(start):
                return project_score_block(start, min(start + int(lone_block_rows), int(chunk.row_capacity)))

            scored = score_resident_chunk_normalized_cc_in_row_blocks(
                cc_block_reference,
                row_image_local,
                row_mask_bits,
                int(chunk.n_valid_rows),
                recon["score_shifted_cc"],
                recon["corr_img_score"],
                recon["cc_half_batch_norm"],
                block_rows=int(lone_block_rows),
                half_weights=half_weights,
                full_to_compact=full_to_compact,
                row_capacity=chunk.row_capacity,
                n_fine_trans=int(n_fine_trans),
            )
        else:
            scored = score_resident_projected_chunk_normalized_cc(
                score_proj,
                row_image_local,
                row_mask_bits,
                n_valid_rows_device,
                recon["score_shifted_cc"],
                recon["corr_img_score"],
                recon["cc_half_batch_norm"],
                half_weights=half_weights,
                full_to_compact=full_to_compact,
                row_capacity=chunk.row_capacity,
                n_fine_trans=int(n_fine_trans),
                block_rows=cc_block_rows(chunk.row_capacity),
            )
        del score_proj
        scores_flat = jnp.asarray(scored.scores, dtype=jnp.float32).reshape(-1)
        mark("score", scores_flat)
        (
            log_z_out,
            best_log_score,
            best_cell_index,
            max_posterior,
            row_posterior,
            n_significant,
        ) = local_firstiter_cc_posterior(
            scored.scores,
            row_image_local,
            row_is_valid,
            segment_offsets,
            n_valid_images_device,
            image_capacity=chunk.image_capacity,
        )
    else:
        # --- stage 3: score ----------------------------------------------------
        # Chunk-local image slots address the chunk's own operands.
        chunk_image_ids = jnp.where(
            jnp.arange(chunk.image_capacity, dtype=jnp.int32) < n_valid_images_device,
            jnp.arange(chunk.image_capacity, dtype=jnp.int32),
            jnp.int32(-1),
        )
        if lone_block_rows is not None:

            def block_reference(start):
                # The projection window union covers both windows, so the block projects both; the planned
                # largest-class chunk counts both.
                return project_score_block(start, min(start + int(lone_block_rows), int(chunk.row_capacity)))

            scored = score_resident_chunk_in_row_blocks(
                block_reference,
                row_image_local,
                row_log_prior,
                expand_local_chunk_mask_jnp(row_mask_bits, n_trans=int(n_fine_trans)),
                int(chunk.n_valid_rows),
                chunk_image_ids,
                recon["score_input"],
                recon["corr_img_score"],
                recon["highres_xi2_half"],
                recon["translation_prior"],
                block_rows=int(lone_block_rows),
                half_weights=half_weights,
                translation_angles=translation_angles,
                full_to_compact=full_to_compact,
                logical_current_size=jnp.asarray(current_size, dtype=jnp.int32),
                row_capacity=chunk.row_capacity,
                image_capacity=chunk.image_capacity,
                n_fine_trans=int(n_fine_trans),
            )
        else:
            scored = score_resident_projected_chunk(
                score_proj,
                row_image_local,
                row_log_prior,
                row_mask_bits,
                n_valid_rows_device,
                chunk_image_ids,
                recon["score_input"],
                recon["corr_img_score"],
                recon["highres_xi2_half"],
                recon["translation_prior"],
                half_weights=half_weights,
                translation_angles=translation_angles,
                full_to_compact=full_to_compact,
                logical_current_size=jnp.asarray(current_size, dtype=jnp.int32),
                row_capacity=chunk.row_capacity,
                image_capacity=chunk.image_capacity,
                n_fine_trans=int(n_fine_trans),
                n_score_pixels=int(n_score_pixels),
            )
        del score_proj
        scores_flat = jnp.asarray(scored.scores, dtype=jnp.float32).reshape(-1)
        mark("score", scores_flat)

        # --- stage 4: segmented RELION float32 fine posterior -------------------
        log_z = cuda_posterior.sparse_pass2_segmented_log_z_f64(
            scores_flat, segment_offsets, n_valid_images_device
        )
        posterior = cuda_posterior.sparse_pass2_segmented_posterior_f32(
            scores_flat,
            segment_offsets,
            n_valid_images_device,
            log_z,
            jnp.ones((chunk.image_capacity,), dtype=jnp.float32),
            adaptive_fraction=float(adaptive_fraction),
            keep_all=bool(keep_all_weights),
            use_external_sum_weight=False,
        )
        (
            log_z_out,
            best_log_score,
            best_cell_index,
            max_posterior,
            _probs,
            _normalized_weights,
            reconstruction_probs,
            _mask,
            n_significant,
            _sum_weight,
            _threshold,
        ) = posterior
        row_posterior = jnp.asarray(reconstruction_probs, dtype=jnp.float32).reshape(
            chunk.row_capacity, int(n_fine_trans)
        )
    mark("posterior", row_posterior, log_z_out, best_cell_index)

    mstep_rotations = jnp.asarray(
        host_chunk["mstep_rotations"], dtype=precision_policy.score_real_dtype
    )

    # Only rows the pruned posterior keeps enter the M-step, live rows first in
    # chunk order, as in the global pass (make_mstep_block_inputs): a row with
    # no positive cell adds exact zeros to every M-step accumulator, and the
    # block walk stops after the live rows. At the 10097 full-box final pass
    # about a third of the scored rows carry no weight.
    class_layout = class_posterior = None
    if class_projectors is None:
        mstep_rows, n_live_rows = _live_rows_first(
            row_posterior, row_is_valid, row_image_local, kernel_row_image_ids
        )
    else:
        # Class3D: the image's joint posterior above, each (image, class) sub-segment's
        # evidence and winner here, and each class's live rows for its own BPref.
        n_classes = len(class_projectors)
        class_layout = resident_chunk_programs.chunk_class_layout(
            host_chunk, chunk, n_classes=n_classes, n_fine_trans=int(n_fine_trans), place=resident_chunk_records.PLACE_ON_DEVICE
        )
        class_posterior = resident_chunk_posterior.class_sub_segment_posterior(
            scores_flat.reshape(chunk.row_capacity, int(n_fine_trans)),
            _ClassRows(classes=class_layout, n_valid_images=n_valid_images_device),
            row_is_valid,
            n_segments=chunk.image_capacity * n_classes,
            n_classes=n_classes,
        )
        # A class's M-step reads only its own rows' weights: the block walk's last block runs past the
        # live rows, so the other classes' rows there must carry zero posterior, not theirs. Each class's
        # order is formed when its M-step runs, so one class's copy of the posterior lives at a time.
        def class_mstep_rows(k):
            in_class = class_layout.row_class == k
            return _live_rows_first(
                jnp.where(in_class[:, None], row_posterior, jnp.float32(0.0)),
                row_is_valid & in_class,
                row_image_local,
                kernel_row_image_ids,
            )

        mstep_rows = n_live_rows = None

    def finish(Ft_y_total, Ft_ctf_total, stats):
        """The chunk's M-step and statistics, added into the running accumulators."""

        n_live_rows_host = int(n_live_rows) if class_projectors is None else 0  # Class3D: summed over classes below

        # P3-G (default on): the three chunk-wide arrays go to the M-step entry
        # point whole and the block program gathers its rows inside the jit.
        # RELAX_LOCAL_SEARCH_RESIDENT_BLOCK_ROW_PROGRAM=0 has the Python callback
        # gather them per block, three eager dispatches each time; on the full
        # EMPIAR-10097 run that path took 2964 s against 2868 s (job 14550046).
        block_row_program = engine_tuning.local_block_row_program

        def run_mstep(rows, n_live, Ft_y, Ft_ctf, projector=None):
            if lone_block_rows is not None:
                # Each M-step block projects its own (live-first) rows: the lone image's projections never all
                # live at once. A Class3D class's blocks hold its own rows (the others carry zero posterior), so
                # they project with that class's reference.
                chunk_projections = None

                def block_projections(start, stop):
                    block_rows = rows.row_ids[start:stop]
                    _, recon_block, abs2_block, _ = project_rows(
                        chunk_rotations[block_rows], int(stop - start), projector
                    )
                    return recon_block, abs2_block, mstep_rotations[block_rows]

            elif block_row_program:
                block_projections = None
                chunk_projections = (recon_proj, recon_abs2, mstep_rotations)
            else:
                chunk_projections = None

                def block_projections(start, stop):
                    block_rows = rows.row_ids[start:stop]
                    return recon_proj[block_rows], recon_abs2[block_rows], mstep_rotations[block_rows]

            return resident_mstep_blocks.run_resident_mstep_blocks(
                block_projections,
                chunk_projections=chunk_projections,
                row_capacity=chunk.row_capacity,
                n_valid_rows=n_live,
                mstep_block_rows=int(mstep_block_rows),
                image_capacity=chunk.image_capacity,
                row_image_local=rows.row_image_local,
                kernel_row_image_ids=rows.kernel_row_image_ids,
                row_posterior=rows.row_posterior,
                row_ids=rows.row_ids,
                recon=recon,
                recon_pixel_indices=recon_window_indices,
                translation_angles=translation_angles,
                n_rect=int(n_rect),
                n_shells=int(stats_config.n_shells),
                n_recon_windowed=int(n_recon_windowed),
                glue_jit=engine_tuning.resident_glue_jit,
                noise_variance_for_noise=noise_variance_for_noise,
                shell_indices_noise=shell_indices_noise,
                exact_positions_device=exact_positions_device,
                Ft_y_total=Ft_y,
                Ft_ctf_total=Ft_ctf,
                image_shape=image_shape,
                recon_volume_shape=recon_volume_shape,
                mstep_current_size=mstep_current_size,
                mstep_max_r=mstep_max_r,
                relion_x_half_recon_indices=relion_x_half_recon_indices,
                max_adjoint_block_bytes=max_adjoint_block_bytes,
                n_optics_groups=int(stats_config.n_optics_groups),
                row_partial=row_partial,
            )

        class_fields = {}
        if class_projectors is None:
            (
                Ft_y_total,
                Ft_ctf_total,
                wavg_triplet_pixels,
                block_noise_shells,
                a2_per_image,
                xa_per_image,
            ) = run_mstep(mstep_rows, n_live_rows_host, Ft_y_total, Ft_ctf_total)
        else:
            Ft_y_total, Ft_ctf_total = list(Ft_y_total), list(Ft_ctf_total)
            class_terms = []
            for k, projector in enumerate(class_projectors):
                rows_k, n_live_device = class_mstep_rows(k)
                n_live_k = int(n_live_device)
                n_live_rows_host += n_live_k
                Ft_y_total[k], Ft_ctf_total[k], *terms = run_mstep(
                    rows_k, n_live_k, Ft_y_total[k], Ft_ctf_total[k], projector
                )
                del rows_k
                class_terms.append(terms)
            wavg_triplet_pixels, block_noise_shells, a2_per_image, xa_per_image, class_fields = _sum_class_mstep_terms(
                class_terms,
                class_scale_masks_rect,
                class_layout=class_layout,
                class_posterior=class_posterior,
                row_start=int(chunk.row_start),
                row_capacity=chunk.row_capacity,
                n_fine_trans=int(n_fine_trans),
            )

        mark("mstep", Ft_y_total, Ft_ctf_total, wavg_triplet_pixels, block_noise_shells)

        # --- stage 7: image-level statistics ------------------------------------
        translation_sqdist_ang = image_tables.translation_sqdist_ang
        if translation_prior_centers_np is not None:
            from relax.sampling.translation_prior import (
                translation_prior_centers_for_images,
                translation_sqdist_angstrom,
            )

            # Build the centres at capacity on the host so the squared-distance
            # program is keyed on the capacity class, not the occupancy; padded
            # rows multiply a zero posterior, so their value is never observable.
            padded_image_indices = resident_chunk_inputs.pad_batch_to_capacity(
                np.asarray(image_indices).reshape(-1, 1), chunk.image_capacity
            ).reshape(-1)
            centers = translation_prior_centers_for_images(
                translation_prior_centers_np,
                padded_image_indices,
                batch_size=chunk.image_capacity,
            )
            translation_sqdist_ang = resident_chunk_inputs.zero_padded_images(
                jnp.asarray(translation_sqdist_angstrom(fine_translations, centers, voxel_size)),
                jnp.asarray(np.arange(chunk.image_capacity) < chunk.n_valid_images, dtype=bool),
            )
        chunk_tables = image_tables._replace(translation_sqdist_ang=translation_sqdist_ang)

        image_row_start_np = segment_offsets_np.astype(np.int64)[:chunk.image_capacity] // int(n_fine_trans)
        image_row_count_np = (
            segment_offsets_np.astype(np.int64)[1:] - segment_offsets_np.astype(np.int64)[:-1]
        ) // int(n_fine_trans)
        image_row_start = jnp.asarray(image_row_start_np, dtype=jnp.int64)
        image_row_count = jnp.asarray(image_row_count_np, dtype=jnp.int64)

        best_row_local = jnp.asarray(best_cell_index, dtype=jnp.int64) // jnp.int64(n_fine_trans)
        slot_is_valid = jnp.arange(chunk.image_capacity, dtype=jnp.int32) < n_valid_images_device
        invalid_best = slot_is_valid & ((best_row_local < 0) | (best_row_local >= image_row_count))
        best_chunk_row = jnp.clip(
            image_row_start + best_row_local, 0, jnp.int64(max(chunk.row_capacity - 1, 0))
        ).astype(jnp.int32)
        # The winner's global row in the layout's flat order; the pose decode reads
        # rotations, M-step rotations, fine ids and source Eulers at that row.
        best_global_row = jnp.int64(int(chunk.row_start)) + best_chunk_row.astype(jnp.int64)

        chunk_posterior_bin = np.full(chunk.row_capacity, int(stats_config.n_coarse_rot), dtype=np.int32)
        chunk_posterior_bin[:chunk.n_valid_rows] = posterior_bins[int(chunk.row_start) : int(chunk.row_stop)]
        chunk_posterior_bin = jnp.asarray(chunk_posterior_bin)
        chunk_operands = resident_image_terms.ChunkImageOperands(
            row_posterior=row_posterior,
            row_image_local=row_image_local,
            row_coarse_rot=jnp.where(
                row_is_valid, chunk_posterior_bin, jnp.int32(int(stats_config.n_coarse_rot))
            ),
            image_ids=image_ids,
            group_ids=recon["group_ids"],
            image_power_shells=recon["image_power_shells"],
            relion_norm_high_shell=recon["relion_norm_high_shell"],
            wavg_triplet_pixels=wavg_triplet_pixels,
            block_noise_shells=block_noise_shells,
            a2_per_image=a2_per_image,
            xa_per_image=xa_per_image,
            class_log_z=jnp.asarray(log_z_out, dtype=jnp.float64),
            min_diff2=scored.min_diff2,
            best_log_score=best_log_score,
            max_posterior=max_posterior,
            best_cell_index=jnp.asarray(best_cell_index, dtype=jnp.int64),
            best_fine_rot=best_global_row,
            optics_groups=recon.get("optics_groups"),
            **class_fields,
        )
        stats = resident_image_terms.accumulate_chunk_image_terms(
            stats, chunk_operands, chunk_tables, config=stats_config
        )
        stats = stats._replace(
            invalid_best_rows=stats.invalid_best_rows + jnp.sum(invalid_best.astype(jnp.int64))
        )
        mark("stats", stats)
        if profile:
            order = ("t0", "operands", "project", "score", "posterior", "mstep", "stats")
            spans = {
                name: marks[name] - marks[prev]
                for prev, name in zip(order, order[1:])
                if name in marks and prev in marks
            }
            n_blocks = len(range(0, min(chunk.row_capacity, max(n_live_rows_host, 1)), int(mstep_block_rows))) or 1
            logger.info(
                "Resident local chunk profile: images=%d/%d rows=%d/%d live_rows=%d row_pad=%.1f%% "
                "blocks=%d proj_rows=%d recon_tile=%s wavg_tile=%s | %s | chunk=%.3fs",
                chunk.n_valid_images, chunk.image_capacity, chunk.n_valid_rows, chunk.row_capacity, n_live_rows_host,
                100.0 * (chunk.row_capacity - chunk.n_valid_rows) / max(chunk.row_capacity, 1),
                n_blocks, n_projected_rows,
                f"{recon_operand.dtype}{tuple(recon_operand.shape)}",
                f"{recon['raw_translated_wavg_rectangle'].dtype}"
                f"{tuple(recon['raw_translated_wavg_rectangle'].shape)}",
                " ".join(f"{k}={v:.3f}s" for k, v in spans.items()),
                marks["stats"] - marks["t0"],
            )
        return Ft_y_total, Ft_ctf_total, stats

    return finish
