"""One resident capacity chunk's run: its operands, projections and stage sequence.

``run_resident_chunk`` gathers one chunk's operands, its projections (cached or streamed) and runs the stage
sequence (``resident_chunk_stages``); an image whose rows exceed the largest row class runs alone in row blocks
(``run_lone_resident_chunk``). Read by the resident driver (``resident_pass2``); imports nothing from it.
"""

from __future__ import annotations

import logging
import time

import jax
import jax.numpy as jnp
import numpy as np

from relax.fine_pass import (
    resident_chunk_inputs,
    resident_chunk_posterior,
    resident_chunk_programs,
    resident_chunk_records,
    resident_chunk_stages,
    resident_mstep_blocks,
    resident_projections,
)
from relax.fine_pass.policy import ResidentConfigurationUnsupported
from relax.fine_pass.resident_operands import gather_resident_chunk_operands
from relax.reconstruction.half_volume_mstep import RowPartial

logger = logging.getLogger(__name__)


def run_resident_chunk(
    chunk,
    *,
    chunk_timing: bool,
    glue_jit: bool,
    tables,
    window_logical=None,
    experiment_dataset,
    bucket_io_kwargs,
    half_weights,
    translation_angles,
    full_to_compact,
    n_score_pixels,
    fine_translation_prior_2d,
    score_real_dtype,
    projection_score_cache,
    projection_recon_cache,
    projection_recon_abs2_cache,
    fine_translation_parent_device,
    mstep_grid,
    coarse_parent_grid,
    n_fine_trans,
    n_recon_windowed,
    n_rect,
    mstep_block_rows,
    adaptive_fraction,
    windowed_prepare,
    stream_projection_fn=None,
    n_fine_rot=None,
    cache_slot_fine_rot=None,
    slot_of_projection=None,
    window_indices,
    recon_window_indices,
    relion_x_half_recon_indices,
    exact_positions_device,
    rect_indices_device,
    recon_pixel_indices,
    resident_operands,
    verify_operands,
    image_shape,
    current_size,
    mstep_current_size,
    wsum_current_size=None,
    chunk_unshifted_operands=False,
    precision_policy=None,
    mstep_max_r,
    recon_volume_shape,
    max_adjoint_block_bytes,
    noise_variance_for_noise,
    program_current_size=None,
    shell_indices_noise,
    group_ids_np,
    scale_corrections_np,
    translation_prior_centers_np,
    fine_translations,
    voxel_size,
    use_exact_relion_gaussian,
    accumulate_noise,
    source_faithful_spectrum_norm,
    stats,
    stats_config,
    image_tables,
    Ft_y_total,
    Ft_ctf_total,
    submitted_keys=None,
    optics_groups_np=None,
    relion_native_fine_units=False,
    coarse_reuse=None,
    firstiter_cc=False,
    mstep_subtract_ctf_projection=False,
    union_score_take=None,
    union_recon_take=None,
    union_native_fft_size=0,
    residual_sgd_cache=None,
    residual_sgd_take=None,
    deferred=False,
    lone_block_rows=None,
    row_partial: RowPartial | None = None,
):
    """Run every resident stage for one capacity chunk.

    ``row_partial`` is the pass's partial accumulator pair, when its rows are backprojected one by one
    and the pair fits the size rule (``half_volume_mstep.allocate_row_partial``).

    ``lone_block_rows`` is set for one image's overflow chunk (a chunk past the
    plan's largest row class, :func:`chunk_runs_alone`), which runs alone and is
    scored and reconstructed in blocks of that many rows
    (:func:`run_lone_resident_chunk`, :func:`lone_block_rows`).

    ``union_score_take`` / ``union_recon_take`` / ``union_native_fft_size``
    describe a union projection cache (see ``ChunkStageTables``); they are
    None / 0 for the three-cache layout and for streamed chunks.
    ``residual_sgd_cache`` / ``residual_sgd_take`` are VDAM's subtracted-reference
    rows of that cache (``_sgd_residual_rows``), None when it has none.

    ``submitted_keys``, when given, collects the ``(program name, spec)`` keys
    this chunk submits, so the driver can say how many of them the compile-ahead
    warm-up had already compiled. Recording costs a set insert per chunk.

    Returns the updated ``(Ft_y_total, Ft_ctf_total, stats)``. Host work inside
    is the chunk's materialize/pad, its operand preparation and the T7 offsets
    readback the segmented posterior performs internally; no per-chunk result
    is pulled.

    ``deferred`` (per-stage path, timing off) enqueues the chunk up to its
    posterior and M-step row order and returns ``finish(Ft_y_total,
    Ft_ctf_total, stats)`` for the rest, whose live row ranges are read back
    on the host, so the caller can enqueue the next chunk first; the
    accumulator arguments are then unused.
    """

    # The operands' powerClass terms sum above the weighted sums' size (--strict_highres_exp).
    operand_current_size = current_size if wsum_current_size is None else wsum_current_size
    image_indices = np.arange(chunk.image_start, chunk.image_stop, dtype=np.int64)
    timing = chunk_timing
    chunk_t0 = time.time()
    stage_t = {}
    if timing:
        jax.block_until_ready(Ft_y_total)
        chunk_t0 = time.time()

    rows = resident_chunk_programs.make_chunk_row_arrays(
        tables, chunk, n_fine_trans, place=resident_chunk_records.PLACE_ON_DEVICE, n_fine_rot=n_fine_rot,
        slot_of_projection=slot_of_projection,
    )
    lone = lone_block_rows is not None
    if stream_projection_fn is not None and not lone:
        (
            rows,
            cache_slot_fine_rot,
            (projection_score_cache, projection_recon_cache, projection_recon_abs2_cache),
            mstep_grid,
            coarse_parent_grid,
        ) = resident_projections.stream_chunk_projections(
            rows,
            resident_chunk_programs.row_projection_ids(resident_chunk_programs.chunk_host_rows(tables, chunk)[1], n_fine_rot if tables.n_classes > 1 else None),
            n_valid_rows=chunk.n_valid_rows,
            row_capacity=chunk.row_capacity,
            project=stream_projection_fn,
            n_fine_rot=n_fine_rot,
            mstep_grid=mstep_grid,
            coarse_parent_grid=coarse_parent_grid,
        )
        if timing:
            jax.block_until_ready(projection_score_cache)
            stage_t["projections"] = time.time() - chunk_t0

    if resident_operands is None and chunk_unshifted_operands:
        recon = resident_chunk_inputs.unshifted_chunk_operands(
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
            use_exact_relion_gaussian=use_exact_relion_gaussian,
            accumulate_noise=accumulate_noise,
            source_faithful_spectrum_norm=bool(source_faithful_spectrum_norm),
            fine_translation_prior_2d=fine_translation_prior_2d,
            scale_corrections_np=scale_corrections_np,
            group_ids_np=group_ids_np,
            optics_groups_np=optics_groups_np,
            precision_policy=precision_policy,
            relion_native_fine_units=relion_native_fine_units,
        )
    elif resident_operands is None:
        recon = resident_chunk_inputs.prepare_chunk_reconstruction_operands(
            chunk=chunk,
            image_indices=image_indices,
            experiment_dataset=experiment_dataset,
            bucket_io_kwargs=bucket_io_kwargs,
            windowed_prepare=windowed_prepare,
            recon_window_indices=recon_window_indices,
            score_window_indices=window_indices,
            fine_translation_prior_2d=fine_translation_prior_2d,
            score_real_dtype=score_real_dtype,
            n_fine_trans=int(n_fine_trans),
            n_recon_windowed=int(n_recon_windowed),
            image_shape=image_shape,
            current_size=operand_current_size,
            use_exact_relion_gaussian=use_exact_relion_gaussian,
            accumulate_noise=accumulate_noise,
            source_faithful_spectrum_norm=source_faithful_spectrum_norm,
            relion_score_translation_angles=translation_angles,
            rect_indices_device=rect_indices_device,
            exact_positions_device=exact_positions_device,
            scale_corrections_np=scale_corrections_np,
            group_ids_np=group_ids_np,
            optics_groups_np=optics_groups_np,
            relion_native_fine_units=relion_native_fine_units,
            normalized_cc=firstiter_cc,
            noise_shell_indices_half=image_tables.shell_indices_half,
            n_noise_shells=int(stats_config.n_shells),
        )
    else:
        # The chunk's image slots are the half's images ``image_start`` to
        # ``image_stop``; the padded slots carry -1 and the gather zeroes them,
        # which is what the per-chunk preparation's capacity mask did.
        image_slots = np.full(chunk.image_capacity, -1, dtype=np.int32)
        image_slots[:chunk.n_valid_images] = image_indices[:chunk.n_valid_images]
        recon = gather_resident_chunk_operands(
            resident_operands,
            image_slots,
            translation_angles=translation_angles,
            rect_indices=rect_indices_device,
            exact_positions=exact_positions_device,
            image_shape=image_shape,
        )
        if verify_operands:
            resident_chunk_inputs.verify_resident_chunk_operands(
                recon,
                resident_chunk_inputs.prepare_chunk_reconstruction_operands(
                    chunk=chunk,
                    image_indices=image_indices,
                    experiment_dataset=experiment_dataset,
                    bucket_io_kwargs=bucket_io_kwargs,
                    windowed_prepare=windowed_prepare,
                    recon_window_indices=recon_window_indices,
                    score_window_indices=window_indices,
                    fine_translation_prior_2d=fine_translation_prior_2d,
                    score_real_dtype=score_real_dtype,
                    n_fine_trans=int(n_fine_trans),
                    n_recon_windowed=int(n_recon_windowed),
                    image_shape=image_shape,
                    current_size=operand_current_size,
                    use_exact_relion_gaussian=use_exact_relion_gaussian,
                    accumulate_noise=accumulate_noise,
                    source_faithful_spectrum_norm=source_faithful_spectrum_norm,
                    relion_score_translation_angles=translation_angles,
                    rect_indices_device=rect_indices_device,
                    exact_positions_device=exact_positions_device,
                    scale_corrections_np=scale_corrections_np,
                    group_ids_np=group_ids_np,
                    optics_groups_np=optics_groups_np,
                    relion_native_fine_units=relion_native_fine_units,
                    noise_shell_indices_half=image_tables.shell_indices_half,
                    n_noise_shells=int(stats_config.n_shells),
                ),
                translation_angles=translation_angles,
                recon_pixel_indices=recon_pixel_indices,
                image_shape=image_shape,
                n_recon_pixels=int(n_recon_windowed),
                bpref_recon_operand=resident_operands.recon_weight is not None,
                label=f"chunk images {chunk.image_start}-{chunk.image_stop}",
            )
    # The unshifted operands, the half's or this chunk's, carry recon_image; the
    # M-step translates them inside the translate-and-sum kernel.
    unshifted_operands = recon.get("recon_image") is not None
    if timing:
        jax.block_until_ready(recon["recon_image"] if unshifted_operands else recon["shifted_recon"])
        stage_t["operands"] = time.time() - chunk_t0

    # The prior squared distances are a per-image host table; building them at
    # capacity here keeps the program keyed on the capacity class and takes the
    # eager device ops out of the chunk loop. Padded slots multiply a zero
    # posterior, so their value is never observable; they are zeroed anyway.
    translation_sqdist_ang = resident_chunk_programs.make_chunk_translation_sqdist(
        image_tables.translation_sqdist_ang,
        translation_prior_centers_np=translation_prior_centers_np,
        image_indices=image_indices,
        image_capacity=chunk.image_capacity,
        n_valid_images=chunk.n_valid_images,
        fine_translations=fine_translations,
        voxel_size=voxel_size,
    )

    operands = resident_chunk_programs.make_chunk_stage_operands(recon, translation_sqdist_ang)
    stage_tables = resident_chunk_programs.make_chunk_stage_tables(
        projection_score_cache=projection_score_cache,
        projection_recon_cache=projection_recon_cache,
        projection_recon_abs2_cache=projection_recon_abs2_cache,
        mstep_grid=mstep_grid,
        coarse_parent_grid=coarse_parent_grid,
        fine_translation_parent_device=fine_translation_parent_device,
        half_weights=half_weights,
        translation_angles=translation_angles,
        full_to_compact=full_to_compact,
        noise_variance_for_noise=noise_variance_for_noise,
        shell_indices_noise=shell_indices_noise,
        exact_positions_device=exact_positions_device,
        recon_pixel_indices=recon_pixel_indices,
        relion_x_half_recon_indices=relion_x_half_recon_indices,
        image_tables=image_tables,
        cache_slot_fine_rot=cache_slot_fine_rot,
        coarse_reuse=coarse_reuse,
        window_logical=window_logical,
        union_score_take=union_score_take,
        union_recon_take=union_recon_take,
        residual_sgd_cache=residual_sgd_cache,
        residual_sgd_take=residual_sgd_take,
    )
    spec = resident_chunk_programs.make_chunk_program_spec(
        row_capacity=chunk.row_capacity,
        image_capacity=chunk.image_capacity,
        n_fine_trans=n_fine_trans,
        n_score_pixels=n_score_pixels,
        n_recon_pixels=n_recon_windowed,
        n_rect=n_rect,
        mstep_block_rows=mstep_block_rows,
        adaptive_fraction=adaptive_fraction,
        current_size=current_size if program_current_size is None else program_current_size,
        mstep_current_size=mstep_current_size,
        mstep_max_r=mstep_max_r,
        image_shape=image_shape,
        recon_volume_shape=recon_volume_shape,
        max_adjoint_block_bytes=max_adjoint_block_bytes,
        stats_config=stats_config,
        use_rfloat_ctf_wavg=recon["direct_ctf_rfloat_recon"] is not None,
        use_translate_sum_kernel=unshifted_operands,
        bpref_recon_operand=unshifted_operands and recon.get("recon_weight") is not None,
        glue_jit=glue_jit,
        reuse_coarse_normalization=coarse_reuse is not None,
        firstiter_cc=firstiter_cc,
        n_slots=int(tables.n_slots),
        mstep_subtract_ctf_projection=bool(mstep_subtract_ctf_projection),
        n_classes=tables.n_classes,
        stable_window=program_current_size is not None,
        union_native_fft_size=union_native_fft_size,
        presum_adjoint=resident_projections.carries_projection_sums(Ft_y_total),
    )

    if lone:
        if stream_projection_fn is not None and tables.n_classes > 1:
            # A streamed lone chunk keeps its rows' projection ids, class * n_fine_rot +
            # rotation; the statistics map a winner back to its rotation within the class
            # through this table, as through a streamed chunk's slot table (bench 14694988:
            # Class3D K4 100k crashed on a missing table).
            stage_tables = stage_tables._replace(
                cache_slot_fine_rot=(
                    jnp.arange(tables.n_classes * int(n_fine_rot), dtype=jnp.int32) % jnp.int32(n_fine_rot)
                )
            )
        return run_lone_resident_chunk(
            rows,
            operands,
            stage_tables,
            spec=spec,
            host_projection_ids=resident_chunk_programs.row_projection_ids(
                resident_chunk_programs.chunk_host_rows(tables, chunk)[1], n_fine_rot if tables.n_classes > 1 else None, slot_of_projection
            ),
            n_valid_rows=chunk.n_valid_rows,
            block_rows=int(lone_block_rows),
            stream_projection_fn=stream_projection_fn,
            deferred=deferred,
            carry=(Ft_y_total, Ft_ctf_total, stats),
            row_partial=row_partial,
        )

    if submitted_keys is not None:
        submitted_keys.update(resident_chunk_programs.chunk_program_keys(resident_chunk_programs.chunk_program_path(spec.glue_jit), spec))

    if deferred and not timing:
        front = resident_chunk_stages.resident_chunk_stages_front(rows, operands, stage_tables, spec=spec)

        def finish(Ft_y_total, Ft_ctf_total, stats):
            return resident_chunk_stages.resident_chunk_stages_finish(
                front,
                rows,
                operands,
                stage_tables,
                (Ft_y_total, Ft_ctf_total, stats),
                spec=spec,
                row_partial=row_partial,
            )

        return finish

    def timing_hook(name, value):
        jax.block_until_ready(value)
        stage_t[name] = time.time() - chunk_t0

    Ft_y_total, Ft_ctf_total, stats = resident_chunk_stages.run_resident_chunk_stages(
        rows,
        operands,
        stage_tables,
        (Ft_y_total, Ft_ctf_total, stats),
        spec=spec,
        timing_hook=timing_hook if timing else None,
        row_partial=row_partial,
    )

    if timing:
        jax.block_until_ready((Ft_y_total, Ft_ctf_total, stats.wsum_sigma2_noise))
        total = time.time() - chunk_t0
        operands_s = stage_t.get("operands", 0.0)
        posterior_end = stage_t.get("posterior", operands_s)
        mstep_end = stage_t.get("mstep", total)
        split = "score+posterior=%.3fs mstep=%.3fs statistics=%.3fs" % (
            posterior_end - operands_s,
            mstep_end - posterior_end,
            total - mstep_end,
        )
        blocks = sum(1 for s in range(0, chunk.row_capacity, int(mstep_block_rows)) if s < chunk.n_valid_rows)
        logger.info(
            "Resident pass-2 chunk timing: images=%d/%d rows=%d/%d occupancy=%.3f "
            "mstep_blocks=%d/%d projections=%.3fs operands=%.3fs %s total=%.3fs",
            chunk.n_valid_images, chunk.image_capacity, chunk.n_valid_rows, chunk.row_capacity,
            chunk.n_valid_rows / max(chunk.row_capacity, 1),
            blocks, chunk.row_capacity // int(mstep_block_rows),
            stage_t.get("projections", 0.0),
            operands_s,
            split,
            total,
        )
    return Ft_y_total, Ft_ctf_total, stats


def run_lone_resident_chunk(
    rows: resident_chunk_records.ChunkRowArrays,
    operands: resident_chunk_records.ChunkStageOperands,
    tables: resident_chunk_records.ChunkStageTables,
    *,
    spec: resident_chunk_records.ChunkProgramSpec,
    host_projection_ids,
    n_valid_rows: int,
    block_rows: int,
    stream_projection_fn,
    deferred: bool,
    carry: tuple,
    row_partial: RowPartial | None = None,
):
    """One image's overflow chunk, scored and reconstructed ``block_rows`` rows at a time.

    An image whose candidates exceed the largest row class (a flat coarse
    posterior keeps every coarse sample, so every fine rotation: 294,912 rows at
    HEALPix 3 in bench 14641043) is one chunk that runs alone. Holding all its
    rows' projections at once does not fit the device, so the rows are scored in
    blocks (:func:`~relax.fine_pass.resident_scoring.score_resident_chunk_in_row_blocks`),
    keeping only their ``[C_R, T]`` diff2; the image's posterior is then formed
    over all of its rows at once, and each live M-step block gathers (cached
    pass) or projects (streamed pass) its own rows' projections. RELION scores
    such an image the same way, in orientation batches against one normalizer.
    The numbers are those of the one-call chunk: only the grouping of the
    projector calls changes.
    """

    from relax.fine_pass.resident_candidates import expand_chunk_mask_jnp
    from relax.fine_pass.resident_scoring import cached_score_reference, score_resident_chunk_in_row_blocks

    if spec.firstiter_cc or (spec.presum_adjoint and stream_projection_fn is not None):
        raise ResidentConfigurationUnsupported(
            "an image past the largest row class runs in row blocks, which the first-iteration "
            "cross-correlation pass and the streamed per-rotation projection sums do not implement"
        )
    host_ids = np.asarray(host_projection_ids, dtype=np.int64)

    def block_ids(ids):
        padded = np.full(block_rows, ids[0] if ids.size else 0, dtype=np.int64)
        padded[: ids.size] = ids
        return padded

    def block_reference(start):
        ids = block_ids(host_ids[start : start + block_rows])
        if stream_projection_fn is not None:
            return stream_projection_fn(ids, block_rows)[0]
        return cached_score_reference(
            tables.projection_score_cache,
            jnp.asarray(ids, dtype=jnp.int32),
            score_take=tables.union_score_take,
            native_fft_size=int(spec.union_native_fft_size),
        )

    row_index = jnp.arange(int(spec.row_capacity), dtype=jnp.int32)
    row_is_valid = row_index < rows.n_valid_rows
    kernel_row_image_ids = jnp.where(row_is_valid, rows.row_image_local, jnp.int32(-1))
    image_index = jnp.arange(int(spec.image_capacity), dtype=jnp.int32)
    scored = score_resident_chunk_in_row_blocks(
        block_reference,
        rows.row_image_local,
        rows.row_log_prior,
        expand_chunk_mask_jnp(rows.row_mask_bits, rows.row_mask_mode, tables.fine_translation_parent),
        int(n_valid_rows),
        jnp.where(image_index < rows.n_valid_images, image_index, jnp.int32(-1)),
        operands.score_input,
        operands.corr_img_score,
        operands.highres_xi2_half,
        operands.translation_prior,
        block_rows=block_rows,
        half_weights=tables.half_weights,
        translation_angles=tables.translation_angles,
        full_to_compact=tables.full_to_compact,
        logical_current_size=resident_chunk_posterior.logical_current_size(tables, spec),
        row_capacity=int(spec.row_capacity),
        image_capacity=int(spec.image_capacity),
        n_fine_trans=int(spec.n_fine_trans),
    )
    posterior = resident_chunk_posterior.chunk_posterior_from_scores(
        scored,
        rows,
        tables,
        spec=spec,
        row_is_valid=row_is_valid,
        kernel_row_image_ids=kernel_row_image_ids,
    )
    blocks = resident_mstep_blocks.make_mstep_block_inputs(rows, posterior, n_slots=int(spec.n_slots))

    block_projections = projection_dtypes = None
    if stream_projection_fn is not None:

        def block_projections(start, n_rows):
            ids = np.asarray(jax.device_get(blocks.row_fine_rot[start : start + n_rows]), dtype=np.int64)
            _, recon, recon_abs2 = stream_projection_fn(block_ids(ids)[:n_rows], n_rows)
            return recon, recon_abs2, tables.mstep_grid[jnp.asarray(ids, dtype=jnp.int32)]

        # A streamed pass has no caches to read the projection dtypes from; one
        # row's projection gives them (the projector call converts on the host,
        # so it cannot be shape-traced).
        _, recon_one, abs2_one = stream_projection_fn(host_ids[:1], 1)
        projection_dtypes = (recon_one.dtype, abs2_one.dtype)

    def finish(Ft_y_total, Ft_ctf_total, stats):
        return resident_chunk_stages.resident_chunk_stages_finish(
            (posterior, blocks),
            rows,
            operands,
            tables,
            (Ft_y_total, Ft_ctf_total, stats),
            spec=spec,
            block_projections=block_projections,
            projection_dtypes=projection_dtypes,
            row_partial=row_partial,
        )

    return finish if deferred else finish(*carry)
