"""One image batch of pass 1 in two steps: :func:`prepare_batch` and :func:`score_batch`.

The controller (:func:`relax.scoring.significance.run_pass1`) prepares a batch, publishes the batch that waits for its host
read-back, and then scores the prepared one. That order puts this batch's operands on the device before the host reads
the previous batch back, and the device scores this batch while the host prepares the next.
"""

from typing import Any, NamedTuple

import jax.numpy as jnp

from relax.scoring.pass1_batch import BatchInputs, prepare_batch_inputs
from relax.scoring.pass1_dump import DumpTargets, select_dump_targets
from relax.scoring.pass1_operands import ScoreOperands
from relax.scoring.pass1_plan import Pass1Plan
from relax.scoring.pass1_program import run_score_program
from relax.scoring.pass1_results import BatchOutputs
from relax.scoring.pass1_scores import BatchScores
from relax.scoring.pass1_support import NO_SUPPORT, float32_support, generic_support
from relax.scoring.tree_rescore import TreeRescoreState, rescore_ambiguous_images


class PreparedBatch(NamedTuple):
    """One batch of the pass on the device, before its score program is launched.

    ``indices`` are its dataset images, images ``start_idx:end_idx`` of the pass. ``inputs`` are the preprocessed images,
    ``operands`` the route's score operands, ``dump_targets`` the dump request, ``program_inputs`` the three arrays the
    score program takes and ``dump_rows`` the rows it dumps, on the device (``None``: none).
    """

    indices: Any
    start_idx: int
    end_idx: int
    inputs: BatchInputs
    operands: ScoreOperands
    dump_targets: DumpTargets
    program_inputs: tuple
    dump_rows: Any


class ScoredBatch(NamedTuple):
    """One batch's outputs to publish, and what the tree rescore counted on it (``None`` when the pass has none)."""

    outputs: BatchOutputs
    rescored: Any


def prepare_batch(plan: Pass1Plan, batch_data, indices, start_idx: int, end_idx: int) -> PreparedBatch:
    """Preprocess the batch ``batch_data`` of dataset images ``indices`` and prepare its score operands."""

    inputs = prepare_batch_inputs(
        plan.batch_input_plan,
        batch_data,
        indices,
        start_idx=start_idx,
        end_idx=end_idx,
    )
    operands = plan.route.operand_plan.prepare(inputs, indices)

    dump_targets = select_dump_targets(
        plan.experiment_dataset,
        indices,
        collect_significance=plan.collect_significance,
        current_size=plan.current_size,
        debug_iteration=plan.debug_iteration,
    )
    program_inputs = operands.program_inputs()
    dump_rows = None if dump_targets.rows is None else jnp.asarray(dump_targets.rows, dtype=jnp.int32)
    return PreparedBatch(
        indices=indices,
        start_idx=start_idx,
        end_idx=end_idx,
        inputs=inputs,
        operands=operands,
        dump_targets=dump_targets,
        program_inputs=program_inputs,
        dump_rows=dump_rows,
    )


def _with_tree_rescore(plan: Pass1Plan, batch: PreparedBatch, scores: BatchScores) -> tuple:
    """``scores`` with the winners of the ambiguous images rescored, and what the rescore counted."""

    # The bounded top-two rescore uses the batch's exact CUDA CC operands (the per-image FFT/CTF
    # assembly), not a second copy.
    rescored = rescore_ambiguous_images(
        plan.route.tree_rescore_plan,
        TreeRescoreState(
            best_argmax=scores.best_argmax,
            best_score=scores.best_score,
            class_best_argmax=scores.class_best_argmaxes[0],
            class_best_score=scores.class_best_scores[0],
            class_second_argmax=scores.class_second_best_argmaxes[0],
            class_second_score=scores.class_second_best_scores[0],
        ),
        batch.operands.unshifted,
        batch.operands.corr_img,
        experiment_dataset=plan.experiment_dataset,
        indices=batch.indices,
        debug_iteration=plan.debug_iteration,
    )
    return (
        scores._replace(
            best_argmax=rescored.state.best_argmax,
            best_score=rescored.state.best_score,
            class_best_argmaxes=[rescored.state.class_best_argmax],
            class_best_scores=[rescored.state.class_best_score],
            class_second_best_argmaxes=[rescored.state.class_second_argmax],
            class_second_best_scores=[rescored.state.class_second_score],
        ),
        rescored,
    )


def score_batch(plan: Pass1Plan, batch: PreparedBatch, *, defer_publish: bool) -> ScoredBatch:
    """Launch the score program of ``batch``, rescore its ambiguous winners and form its posterior and support.

    With ``defer_publish`` the outputs leave out the large arrays (the weights, the dump blocks, the operands): the
    batch is published while the next one is on the device, and only a batch that publishes at once keeps them.
    """

    actual_batch_size = len(batch.indices)
    scores = run_score_program(
        plan.score_program_plan,
        batch.program_inputs,
        batch.inputs.translation_log_prior,
        actual_batch_size,
        batch.dump_rows,
    )

    rescored = None
    if plan.tree_rescore_enabled:
        scores, rescored = _with_tree_rescore(plan, batch, scores)

    global_log_z = scores.global_max + jnp.log(scores.global_sum)
    class_log_z_values = [class_max + jnp.log(class_sum) for class_max, class_sum in zip(scores.class_max, scores.class_sum)]

    support = NO_SUPPORT
    if plan.collect_significance:
        if plan.route.float32_support:
            support = float32_support(plan.support_plan, scores, batch.inputs.translation_log_prior)
        else:
            support = generic_support(plan.support_plan, scores, global_log_z, actual_batch_size)
        if support.winner is not None:
            # RELION publishes the coarse winner from these weights (class 0: one class).
            scores = scores._replace(best_argmax=support.winner)

    outputs = BatchOutputs(
        start_idx=batch.start_idx,
        end_idx=batch.end_idx,
        actual_batch_size=actual_batch_size,
        batch_size=batch.inputs.batch_size,
        indices=batch.indices,
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
        debug_dump_enabled=batch.dump_targets.enabled,
        dump_target_local_positions=batch.dump_targets.rows,
        dump_target_pre_prior_blocks_per_class=None if defer_publish else scores.dump_pre_prior_blocks,
        dump_target_with_prior_blocks_per_class=None if defer_publish else scores.dump_with_prior_blocks,
        translation_log_prior=None if defer_publish else batch.inputs.translation_log_prior,
        operands=None if defer_publish else batch.operands,
    )
    return ScoredBatch(outputs=outputs, rescored=rescored)
