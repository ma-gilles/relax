"""Coarse significance pruning for adaptive class and pose searches (pass 1 of the adaptive E-step).

The shared scorer serves K=1, multiclass refinement and initial-model searches.
It selects per-image significant samples from one class/rotation/translation
posterior, with optional diagnostic capture of the coarse scoring boundary.

A pass is a request (:class:`~relax.scoring.pass1_request.Pass1Request`), a plan
(:func:`~relax.scoring.pass1_plan.plan_pass1`: the refusals, the route and each stage's plan) and a run
(:func:`run_pass1`: the image batches and the result).
"""

import time

import numpy as np

from relax.scoring.pass1_assembly import build_stats, log_batch_timing, significant_samples_after_loop
from relax.scoring.pass1_plan import Pass1Plan, plan_pass1
from relax.scoring.pass1_publish import publish_batch
from relax.scoring.pass1_request import Pass1Request
from relax.scoring.pass1_results import Pass1Outputs, Pass1Result
from relax.scoring.pass1_step import prepare_batch, score_batch
from relax.scoring.tree_rescore import TreeRescoreTotals, log_tree_rescore_totals


def run_pass1(plan: Pass1Plan) -> Pass1Result:
    """Score the image batches of a planned pass 1 and return its results."""

    outputs = Pass1Outputs.allocate(plan.output_plan)

    tree_rescore_totals = TreeRescoreTotals()

    start_idx = 0
    image_indices = np.arange(plan.n_images)
    from relax.helpers.batch_fetch import iter_indexed_batches, prefetched_batches

    batch_starts = []
    loop_start = time.time()
    pending_batch = None
    from relax.cuda.kernels import deferred_relion_preprocess_checks

    # The image preprocess kernel's finite check is read at the end of the loop: read at each call, it
    # waits for the previous batch's score program inside the next batch's preparation.
    with deferred_relion_preprocess_checks("pass 1") as preprocess_checks, prefetched_batches(
        iter_indexed_batches(plan.experiment_dataset, image_indices, plan.image_batch_size)
    ) as batches:
        for batch_data, _, _, _, _, _, indices in batches:
            batch_starts.append(time.time())
            end_idx = start_idx + len(indices)
            preprocess_checks.at(
                f"batch {len(batch_starts) - 1} (images {start_idx}-{end_idx - 1} of the pass, "
                f"dataset images {int(indices[0])}-{int(indices[-1])})"
            )
            batch = prepare_batch(plan, batch_data, indices, start_idx, end_idx)
            if pending_batch is not None:
                publish_batch(pending_batch, outputs, plan.output_plan, plan.dump_context)
                pending_batch = None
            # A batch on the float32 support route publishes while the next batch's operands are on the
            # device and before that batch's score program (the publish above): the device
            # scores this batch while the host prepares the next. Only a dump reads the large score and
            # operand arrays, and a dump batch publishes at once, so a waiting batch does not hold them.
            defer_publish = (
                plan.collect_significance and plan.route.float32_support and not batch.dump_targets.enabled
            )
            scored = score_batch(plan, batch, defer_publish=defer_publish)
            if scored.rescored is not None:
                tree_rescore_totals = tree_rescore_totals.after_batch(batch.inputs.batch_size, scored.rescored)
            if defer_publish:
                pending_batch = scored.outputs
            else:
                publish_batch(scored.outputs, outputs, plan.output_plan, plan.dump_context)
            start_idx = end_idx
        if pending_batch is not None:
            publish_batch(pending_batch, outputs, plan.output_plan, plan.dump_context)

    log_batch_timing(batch_starts, loop_start, plan.n_images)

    significant_sample_indices = significant_samples_after_loop(outputs, plan.output_plan)

    stats = build_stats(
        outputs,
        significant_sample_indices,
        plan.output_plan,
        executed_backend=plan.route.executed_backend,
    )
    if plan.tree_rescore_enabled:
        log_tree_rescore_totals(tree_rescore_totals)
    return Pass1Result(
        sig_rot_any=outputs.sig_rot_any,
        n_sig_all=outputs.n_sig_all,
        hard_assignment=outputs.hard_assignment,
        class_assignment=outputs.class_assignment,
        significant_sample_indices=significant_sample_indices,
        stats=stats,
    )


def _compute_k_class_significance_batched(experiment_dataset, noise_variance, rotations, translations, **options):
    """Find significant samples from one posterior over ``class x rotation x translation``.

    ``options`` are the keyword fields of :class:`~relax.scoring.pass1_request.Pass1Request`, which documents them.
    Returns a :class:`~relax.scoring.pass1_results.Pass1Result`.
    """

    request = Pass1Request(experiment_dataset, noise_variance, rotations, translations, **options)
    return run_pass1(plan_pass1(request))
