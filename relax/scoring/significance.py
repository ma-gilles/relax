"""Coarse significance pruning for adaptive class and pose searches (pass 1 of the adaptive E-step).

The shared scorer serves K=1, multiclass refinement and initial-model searches.
It selects per-image significant samples from one class/rotation/translation
posterior, with optional diagnostic capture of the coarse scoring boundary.

A pass is a request (:class:`~relax.scoring.pass1_request.Pass1Request`), a plan
(:func:`~relax.scoring.pass1_plan.plan_pass1`: the refusals, the route and each stage's plan) and a run
(:func:`run_pass1`: the image batches and the result).
"""

import time

import jax.numpy as jnp
import numpy as np

from relax.scoring.pass1_assembly import build_full_stats, log_batch_timing, significant_samples_after_loop
from relax.scoring.pass1_batch import prepare_batch_inputs
from relax.scoring.pass1_dump import select_dump_targets
from relax.scoring.pass1_plan import Pass1Plan, plan_pass1
from relax.scoring.pass1_program import run_score_program
from relax.scoring.pass1_publish import publish_batch
from relax.scoring.pass1_request import Pass1Request
from relax.scoring.pass1_results import BatchOutputs, Pass1Outputs, Pass1Result
from relax.scoring.pass1_support import NO_SUPPORT, float32_support, generic_support
from relax.scoring.tree_rescore import (
    TreeRescoreState,
    TreeRescoreTotals,
    log_tree_rescore_totals,
    rescore_ambiguous_images,
    tree_rescore_report,
)


def run_pass1(plan: Pass1Plan) -> Pass1Result:
    """Score the image batches of a planned pass 1 and return its results."""

    outputs = Pass1Outputs.allocate(plan.output_plan)

    tree_rescore_totals = TreeRescoreTotals()

    start_idx = 0
    image_indices = np.arange(plan.n_images)
    from relax.helpers.batch_fetch import iter_indexed_batches, prefetched_batches

    _coarse_batch_starts = []
    _coarse_loop_t0 = time.time()
    pending_batch = None
    from relax.cuda.kernels import deferred_relion_preprocess_checks

    # The image preprocess kernel's finite check is read at the end of the loop: read at each call, it
    # waits for the previous batch's score program inside the next batch's preparation.
    with deferred_relion_preprocess_checks("pass 1") as preprocess_checks, prefetched_batches(
        iter_indexed_batches(plan.experiment_dataset, image_indices, plan.image_batch_size)
    ) as batches:
        for batch_data, _, _, _, _, _, indices in batches:
            _coarse_batch_starts.append(time.time())
            actual_batch_size = len(indices)
            end_idx = start_idx + actual_batch_size
            preprocess_checks.at(
                f"batch {len(_coarse_batch_starts) - 1} (images {start_idx}-{end_idx - 1} of the pass, "
                f"dataset images {int(indices[0])}-{int(indices[-1])})"
            )
            batch_inputs = prepare_batch_inputs(
                plan.batch_input_plan,
                batch_data,
                indices,
                start_idx=start_idx,
                end_idx=end_idx,
            )
            operands = plan.operand_plan.prepare(batch_inputs, indices)

            dump_targets = select_dump_targets(
                plan.experiment_dataset,
                indices,
                collect_significance=plan.collect_significance,
                current_size=plan.current_size,
                debug_iteration=plan.debug_iteration,
            )
            program_inputs = operands.program_inputs()
            dump_rows = None if dump_targets.rows is None else jnp.asarray(dump_targets.rows, dtype=jnp.int32)
            if pending_batch is not None:
                publish_batch(pending_batch, outputs, plan.output_plan, plan.dump_context)
                pending_batch = None
            scores = run_score_program(
                plan.score_program_plan,
                program_inputs,
                batch_inputs.translation_log_prior,
                actual_batch_size,
                dump_rows,
            )

            if plan.tree_rescore_enabled:
                # The bounded top-two rescore uses the batch's exact CUDA CC operands (the per-image FFT/CTF
                # assembly), not a second copy.
                rescored = rescore_ambiguous_images(
                    plan.tree_rescore_plan,
                    TreeRescoreState(
                        best_argmax=scores.best_argmax,
                        best_score=scores.best_score,
                        class_best_argmax=scores.class_best_argmaxes[0],
                        class_best_score=scores.class_best_scores[0],
                        class_second_argmax=scores.class_second_best_argmaxes[0],
                        class_second_score=scores.class_second_best_scores[0],
                    ),
                    operands.unshifted,
                    operands.corr_img,
                    experiment_dataset=plan.experiment_dataset,
                    indices=indices,
                    debug_iteration=plan.debug_iteration,
                )
                tree_rescore_totals = tree_rescore_totals.after_batch(batch_inputs.batch_size, rescored)
                scores = scores._replace(
                    best_argmax=rescored.state.best_argmax,
                    best_score=rescored.state.best_score,
                    class_best_argmaxes=[rescored.state.class_best_argmax],
                    class_best_scores=[rescored.state.class_best_score],
                    class_second_best_argmaxes=[rescored.state.class_second_argmax],
                    class_second_best_scores=[rescored.state.class_second_score],
                )

            global_log_z = scores.global_max + jnp.log(scores.global_sum)
            class_log_z_values = [
                class_max + jnp.log(class_sum) for class_max, class_sum in zip(scores.class_max, scores.class_sum)
            ]

            support = NO_SUPPORT
            if plan.collect_significance:
                if plan.relion_f32_coarse_support_enabled:
                    support = float32_support(plan.support_plan, scores, batch_inputs.translation_log_prior)
                else:
                    support = generic_support(plan.support_plan, scores, global_log_z, actual_batch_size)
                if support.winner is not None:
                    # RELION publishes the coarse winner from these weights (class 0: one class).
                    scores = scores._replace(best_argmax=support.winner)

            # A batch on the float32 support route publishes while the next batch's operands are on the
            # device and before that batch's score program (the call above the program): the device
            # scores this batch while the host prepares the next. Only a dump reads the large score and
            # operand arrays, and a dump batch publishes at once, so a waiting batch does not hold them.
            defer_publish = plan.collect_significance and plan.relion_f32_coarse_support_enabled and not dump_targets.enabled
            batch_outputs = BatchOutputs(
                start_idx=start_idx,
                end_idx=end_idx,
                actual_batch_size=actual_batch_size,
                batch_size=batch_inputs.batch_size,
                indices=indices,
                pmax=support.pmax,
                weights=None if defer_publish else support.weights,
                sig_mask=support.mask,
                sig_rot_mask=support.rotation_mask,
                n_sig=support.n_significant,
                cutoff_count=support.cutoff_count,
                sum_weight=support.sum_weight,
                significant_weight=None if defer_publish else support.significant_weight,
                normalization_sum_weight=support.normalization_sum_weight,
                normalization_max_posterior=support.normalization_max_posterior,
                best_argmax=scores.best_argmax,
                best_class=scores.best_class,
                best_score=scores.best_score,
                global_log_z=global_log_z,
                class_log_z_values=class_log_z_values,
                class_best_scores=scores.class_best_scores,
                class_best_argmaxes=scores.class_best_argmaxes,
                class_second_best_scores=scores.class_second_best_scores,
                class_second_best_argmaxes=scores.class_second_best_argmaxes,
                debug_dump_enabled=dump_targets.enabled,
                dump_target_local_positions=dump_targets.rows,
                dump_target_pre_prior_blocks_per_class=None if defer_publish else scores.dump_pre_prior_blocks,
                dump_target_with_prior_blocks_per_class=None if defer_publish else scores.dump_with_prior_blocks,
                translation_log_prior=None if defer_publish else batch_inputs.translation_log_prior,
                operands=None if defer_publish else operands,
            )
            if defer_publish:
                pending_batch = batch_outputs
            else:
                publish_batch(batch_outputs, outputs, plan.output_plan, plan.dump_context)
            start_idx = end_idx
        if pending_batch is not None:
            publish_batch(pending_batch, outputs, plan.output_plan, plan.dump_context)

    log_batch_timing(_coarse_batch_starts, _coarse_loop_t0, plan.n_images)

    significant_sample_indices = significant_samples_after_loop(outputs, plan.output_plan)

    full_stats = build_full_stats(
        outputs,
        significant_sample_indices,
        plan.output_plan,
        executed_backend=plan.executed_backend,
        gaussian_report=plan.gaussian_report,
        tree_report=(
            {"firstiter_cc_tree_top2_rescore": tree_rescore_report(tree_rescore_totals, plan.tree_rescore_max_margin)}
            if plan.tree_rescore_enabled
            else {}
        ),
    )
    if plan.tree_rescore_enabled:
        log_tree_rescore_totals(tree_rescore_totals)
    return Pass1Result(
        sig_rot_any=outputs.sig_rot_any,
        n_sig_all=outputs.n_sig_all,
        hard_assignment=outputs.hard_assignment,
        class_assignment=outputs.class_assignment,
        significant_sample_indices=significant_sample_indices,
        full_stats=full_stats,
    )


def _compute_k_class_significance_batched(experiment_dataset, noise_variance, rotations, translations, **options):
    """Find significant samples from one posterior over ``class x rotation x translation``.

    ``options`` are the keyword fields of :class:`~relax.scoring.pass1_request.Pass1Request`, which documents them.
    Returns a :class:`~relax.scoring.pass1_results.Pass1Result`.
    """

    request = Pass1Request(experiment_dataset, noise_variance, rotations, translations, **options)
    return run_pass1(plan_pass1(request))
