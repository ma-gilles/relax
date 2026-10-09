"""Publishing one image batch of pass 1: the host read-back of its outputs into the per-image results."""

import jax
import jax.numpy as jnp
import numpy as np

from relax.diagnostics.coarse_gaussian_diagnostics import _maybe_dump_k_class_significance_batch
from relax.scoring.pass1_results import BatchOutputs, OutputPlan, Pass1Outputs, ScoreDumpContext
from relax.scoring.significant_samples import compact_significant_sample_indices_from_mask
from relax.sparse_pass2.resident_significance import compact_batch_significance_classes


@jax.jit
def _any_over_leading_rows(mask, n_rows):
    """``jnp.any(mask[:n_rows], axis=0)`` with a runtime row count.

    The count is an operand, so one program serves every batch tail.
    """
    active = jnp.arange(mask.shape[0]) < n_rows
    return jnp.any(mask & active.reshape((-1,) + (1,) * (mask.ndim - 1)), axis=0)


def _capture_offset_free_and_absolute_float32_scores(scores, log_score_offset):
    """Capture native score margins before adding a large common offset."""

    offset_free = np.asarray(scores, dtype=np.float32)
    absolute = (
        np.asarray(scores, dtype=np.float64) + np.asarray(log_score_offset, dtype=np.float64)
    ).astype(np.float32)
    return offset_free, absolute


def publish_batch(batch: BatchOutputs, outputs: Pass1Outputs, plan: OutputPlan, dump: ScoreDumpContext) -> None:
    """Read one batch's pass-1 outputs back and store them in rows ``batch.start_idx:batch.end_idx`` of ``outputs``.

    Writes ``outputs`` in place (the only writer of those rows) and, when ``batch.debug_dump_enabled``, the score dump
    of ``dump``; returns nothing. The batch loop calls this for a batch once the next batch's operands are on the
    device, before that batch's score program: the device scores this batch while the host prepares the next (a dump
    batch publishes at once). ``plan`` says which results the pass returns.
    """

    if plan.collect_significance:
        if batch.normalization_sum_weight is not None:
            outputs.relion_f32_sum_weight[batch.start_idx:batch.end_idx] = batch.normalization_sum_weight
            outputs.relion_f32_max_posterior[batch.start_idx:batch.end_idx] = batch.normalization_max_posterior
        if plan.relion_f32_coarse_support_enabled:
            outputs.relion_f32_sum_weight[batch.start_idx:batch.end_idx] = np.asarray(
                batch.sum_weight,
                dtype=np.float32,
            )[:batch.actual_batch_size]
            batch_pmax_host = np.asarray(batch.pmax, dtype=np.float32)[:batch.actual_batch_size]
            if plan.return_relion_f32_normalization:
                outputs.relion_f32_max_posterior[batch.start_idx:batch.end_idx] = batch_pmax_host
        device_significance_batch = not batch.debug_dump_enabled
        if device_significance_batch:
            # The mask stays on the device; only the per-image ids cross
            # the bus.  ``sig_rot_any`` below is already a device
            # reduction, so it is unaffected.
            if not np.array_equal(
                np.asarray(batch.indices, dtype=np.int64),
                np.arange(batch.start_idx, batch.end_idx, dtype=np.int64),
            ):
                raise RuntimeError(
                    "the device significance compaction needs image batches in "
                    "dataset order",
                )
            batch_sig_mask_np = None
            class_results = compact_batch_significance_classes(
                batch.sig_mask,
                n_classes=plan.n_classes,
                actual_batch_size=batch.actual_batch_size,
                n_coarse_rot=plan.n_rot,
                n_coarse_trans=plan.n_trans,
                batch_n_sig=batch.n_sig if plan.n_classes == 1 else None,
            )
            for class_index, (
                batch_device_counts,
                batch_device_polarity,
                batch_device_ids,
                _batch_device_rot_any,
            ) in enumerate(class_results):
                outputs.device_significance_counts[class_index].append(batch_device_counts)
                outputs.device_significance_polarity[class_index].append(batch_device_polarity)
                outputs.device_significance_ids[class_index].append(batch_device_ids)
                outputs.device_significance_starts[class_index].append(int(batch.start_idx))
        else:
            batch_sig_mask_np = np.array(batch.sig_mask, dtype=bool, copy=True)
        outputs.sig_rot_any |= np.asarray(
            _any_over_leading_rows(batch.sig_rot_mask, batch.actual_batch_size),
            dtype=bool,
        ).reshape(plan.n_classes, plan.n_rot)
        outputs.n_sig_all[batch.start_idx:batch.end_idx] = np.asarray(batch.n_sig, dtype=np.int32)[:batch.actual_batch_size]
        outputs.cutoff_count_all[batch.start_idx:batch.end_idx] = np.asarray(
            batch.cutoff_count,
            dtype=np.int32,
        )[:batch.actual_batch_size]
    else:
        batch_sig_mask_np = None
        outputs.n_sig_all[batch.start_idx:batch.end_idx] = 0
        outputs.cutoff_count_all[batch.start_idx:batch.end_idx] = 0

    outputs.hard_assignment[batch.start_idx:batch.end_idx] = np.asarray(
        batch.best_argmax,
        dtype=np.int32,
    )[:batch.actual_batch_size]
    outputs.class_assignment[batch.start_idx:batch.end_idx] = np.asarray(
        batch.best_class,
        dtype=np.int32,
    )[:batch.actual_batch_size]

    # The exact scorers score without an image-energy offset.
    log_score_offset = np.zeros(batch.batch_size, dtype=np.float64)
    global_log_z_np = np.asarray(batch.global_log_z, dtype=np.float64)
    best_score_np = np.asarray(batch.best_score, dtype=np.float64)
    output_slice = slice(0, batch.actual_batch_size)
    outputs.normalization_log_z[batch.start_idx:batch.end_idx] = global_log_z_np[output_slice]
    outputs.normalization_log_evidence[batch.start_idx:batch.end_idx] = (
        global_log_z_np[output_slice] + log_score_offset[output_slice]
    )
    outputs.log_evidence[batch.start_idx:batch.end_idx] = outputs.normalization_log_evidence[batch.start_idx:batch.end_idx].astype(plan.score_real_dtype)
    outputs.best_log_score[batch.start_idx:batch.end_idx] = (
        best_score_np[output_slice] + log_score_offset[output_slice]
    ).astype(plan.score_real_dtype)
    if plan.relion_f32_coarse_support_enabled and plan.collect_significance:
        outputs.max_posterior[batch.start_idx:batch.end_idx] = batch_pmax_host
    else:
        outputs.max_posterior[batch.start_idx:batch.end_idx] = np.exp(
            best_score_np[output_slice] - global_log_z_np[output_slice]
        ).astype(plan.score_real_dtype)
    for class_index, class_log_z in enumerate(batch.class_log_z_values):
        outputs.class_log_evidence[class_index, batch.start_idx:batch.end_idx] = (
            np.asarray(class_log_z, dtype=np.float64)[output_slice]
            + log_score_offset[output_slice]
        )
    if plan.return_class_best:
        for class_index in range(plan.n_classes):
            offset_free, absolute = _capture_offset_free_and_absolute_float32_scores(
                batch.class_best_scores[class_index],
                log_score_offset,
            )
            outputs.class_best_offset_free_log_score[class_index, batch.start_idx:batch.end_idx] = offset_free[output_slice]
            outputs.class_best_log_score[class_index, batch.start_idx:batch.end_idx] = absolute[output_slice]
            outputs.class_hard_assignment[class_index, batch.start_idx:batch.end_idx] = np.asarray(
                batch.class_best_argmaxes[class_index][output_slice],
                dtype=np.int32,
            )
    if plan.return_class_second:
        for class_index in range(plan.n_classes):
            offset_free, absolute = _capture_offset_free_and_absolute_float32_scores(
                batch.class_second_best_scores[class_index],
                log_score_offset,
            )
            outputs.class_second_best_offset_free_log_score[class_index, batch.start_idx:batch.end_idx] = offset_free[output_slice]
            outputs.class_second_best_log_score[class_index, batch.start_idx:batch.end_idx] = absolute[output_slice]
            outputs.class_second_hard_assignment[class_index, batch.start_idx:batch.end_idx] = np.asarray(
                batch.class_second_best_argmaxes[class_index][output_slice],
                dtype=np.int32,
            )

    if batch.debug_dump_enabled:
        # Concatenate per-class per-block raw scores for the dump targets
        # into per-class arrays of shape (n_targets, n_rot, n_trans).
        target_scores_pre_prior_per_class = None
        target_scores_with_prior_per_class = None
        target_local_positions_for_dump = None
        # The pass-1 program returns the target rows' scores; the loop pulls
        # them from each block.
        score_capture_mode = "pass1_program_target_rows"
        if batch.dump_target_pre_prior_blocks_per_class is not None:
            target_scores_pre_prior_per_class = [
                np.concatenate(blocks, axis=1) if blocks else None
                for blocks in batch.dump_target_pre_prior_blocks_per_class
            ]
            target_scores_with_prior_per_class = [
                np.concatenate(blocks, axis=1) if blocks else None
                for blocks in batch.dump_target_with_prior_blocks_per_class
            ]
            target_local_positions_for_dump = batch.dump_target_local_positions
        _maybe_dump_k_class_significance_batch(
            experiment_dataset=dump.experiment_dataset,
            indices=batch.indices,
            n_classes=plan.n_classes,
            rotations=dump.rotations,
            translations=dump.translations,
            # ``batch_weights`` is the class-major concatenation of each class's
            # weights on every route.
            class_weight_mats=[
                np.asarray(
                    batch.weights.reshape(
                        batch.batch_size,
                        plan.n_classes,
                        plan.n_rot * plan.n_trans,
                    )[:, class_index, :],
                    dtype=np.float64,
                )
                for class_index in range(plan.n_classes)
            ],
            batch_sig_mask=batch_sig_mask_np,
            batch_n_sig=np.asarray(batch.n_sig, dtype=np.int64),
            hard_assignment_batch=np.asarray(batch.best_argmax, dtype=np.int64),
            class_assignment_batch=np.asarray(batch.best_class, dtype=np.int64),
            global_log_z=global_log_z_np,
            class_log_z_values=batch.class_log_z_values,
            best_score=best_score_np,
            max_posterior=outputs.max_posterior[batch.start_idx:batch.end_idx],
            rotation_log_prior_padded=dump.rotation_log_prior_padded,
            batch_translation_log_prior=batch.translation_log_prior,
            class_log_priors=dump.class_log_priors,
            current_size=dump.current_size,
            adaptive_fraction=dump.adaptive_fraction,
            max_significants=dump.max_significants,
            target_local_positions=target_local_positions_for_dump,
            target_scores_pre_prior_per_class=target_scores_pre_prior_per_class,
            target_scores_with_prior_per_class=target_scores_with_prior_per_class,
            # RELION's exact coarse operands: the Gaussian GEMM's, or the CC pass's.
            coarse_gaussian_shifted_corrected=batch.operands.shifted,
            coarse_gaussian_unshifted_corrected=batch.operands.unshifted,
            coarse_gaussian_pixel_weight=batch.operands.pixel_weight,
            coarse_gaussian_initial_diff2=batch.operands.initial_diff2,
            coarse_gaussian_score_indices=dump.score_indices,
            translation_phase_source=dump.translations_source,
            relion_projector_half=dump.relion_projector_half,
            relion_projector_r_max=dump.relion_projector_r_max,
            projection_padding_factor=dump.projection_padding_factor,
            relion_f32_sum_weight=(
                batch.sum_weight if plan.relion_f32_coarse_support_enabled else None
            ),
            relion_f32_significant_weight=(
                batch.significant_weight
                if plan.relion_f32_coarse_support_enabled
                else None
            ),
            relion_f32_cutoff_count=(
                batch.cutoff_count if plan.relion_f32_coarse_support_enabled else None
            ),
            score_capture_mode=score_capture_mode,
            debug_iteration=dump.debug_iteration,
        )

    if plan.collect_significance and not device_significance_batch:
        samples_per_class = plan.n_rot * plan.n_trans
        for local_idx, global_idx in enumerate(batch.indices):
            for class_index in range(plan.n_classes):
                c0 = class_index * samples_per_class
                c1 = c0 + samples_per_class
                mask = batch_sig_mask_np[local_idx, c0:c1]
                outputs.significant_sample_indices[class_index][global_idx] = compact_significant_sample_indices_from_mask(
                    mask,
                )
