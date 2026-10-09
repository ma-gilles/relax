"""The pass-1 score program of one image batch: the block scores, the priors and the running reductions.

One program per batch over the cached projections (``coarse_pass1_blocks``), or one call per class and rotation block
(``coarse_pass1_block``) when the cache does not fit and for ``--firstiter_cc``. ``_pass1_block_update`` is the
shared body. These are the jitted, state-passing kernels; :func:`run_score_program` is the one public entry that drives
them from a :class:`~relax.scoring.pass1_scores.ScoreProgramPlan`.
"""

from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np

from relax.scoring.pass1_scores import BatchScores, ProgramStatics, ScoreProgramPlan
from relax.scoring.scoring import (
    relion_coarse_gaussian_gemm_scores_jit,
    relion_coarse_normalized_cc_gemm_scores_jit,
    update_logsumexp,
)


@lru_cache(maxsize=8)
def pass1_batch_constants(batch_size: int, enable_x64: bool):
    """The coarse pass's per-batch initial values, built once per batch size.

    ``-inf`` at the default float dtype, float64 zeros and int32 zeros: the
    running maxima, sums and argmaxes start from these every batch and every
    class, which was 2 + 2K eager dispatches per image batch (41 s of the main
    thread on K4 100k/256, job 14514641). JAX arrays are immutable and nothing
    donates them, so one copy serves every batch. ``enable_x64`` keys the
    default float dtype.
    """

    del enable_x64
    return (
        jnp.full(batch_size, -jnp.inf),
        jnp.zeros(batch_size, dtype=jnp.float64),
        jnp.zeros(batch_size, dtype=jnp.int32),
    )


@jax.jit
def _add_coarse_prior_terms(scores, class_log_prior, rotation_log_prior_block, translation_log_prior):
    """``scores`` plus the class, rotation and translation log priors, in that order.

    The rotation block runs along the rotation axis; a 1-D translation prior is
    shared by every image, a 2-D one is per image. ``None`` skips a term.
    """

    scores = scores + class_log_prior
    if rotation_log_prior_block is not None:
        scores = scores + rotation_log_prior_block[None, :, None]
    if translation_log_prior is not None:
        if translation_log_prior.ndim == 1:
            scores = scores + translation_log_prior[None, None, :]
        else:
            scores = scores + translation_log_prior[:, None, :]
    return scores


def pass1_initial_state(batch_constants, n_classes: int):
    """The running pass-1 reductions of one image batch before its first block (``coarse_pass1_blocks``).

    ``batch_constants`` are the batch's ``(-inf, 0.0, 0)`` rows. The raw score
    maximum starts at float32 ``-inf``.
    """

    neg_inf_f, zeros_f64, zeros_i32 = batch_constants
    per_class = int(n_classes)
    return (
        (neg_inf_f, zeros_f64),
        ((neg_inf_f,) * per_class, (zeros_f64,) * per_class),
        (neg_inf_f, zeros_i32, zeros_i32),
        ((neg_inf_f,) * per_class, (zeros_i32,) * per_class, (neg_inf_f,) * per_class, (zeros_i32,) * per_class),
        jnp.full(neg_inf_f.shape, -jnp.inf, dtype=jnp.float32),
    )


def _pass1_block_update(
    block_state,
    reference,
    shifted_corrected,
    pixel_weight,
    initial_diff2,
    actual_image_count,
    prior_terms,
    translation_log_prior,
    class_index,
    rotation_start,
    dump_rows=None,
    *,
    rows: int,
    block_rows: int,
    statics: ProgramStatics,
):
    """One class's rotation block of pass 1: its scores and the running reductions it updates.

    ``block_state`` is ``((global max, sum), (class max, sum), (best score, pose, class),
    (class best score, pose, runner-up score, pose), raw score maximum)`` with the class
    entries of ``class_index`` only; the runner-up (``statics.track_class_second``, with
    ``statics.return_class_best``) is the best pose of the class other than its best one; ``class_index`` and
    ``rotation_start`` may be traced.
    ``statics.score_kind`` is ``"gaussian"`` (the GEMM scores of the projected or cached rows,
    :func:`relion_coarse_gaussian_gemm_scores_jit`, with ``initial_diff2``) or
    ``"normalized_cc"`` (RELION's coarse CC, :func:`relion_coarse_normalized_cc_gemm_scores_jit`;
    no priors, as RELION's ``--firstiter_cc`` scores have none). The padded tail rows are
    ``-inf``; the Gaussian scores get the class, rotation and translation priors
    (:func:`_add_coarse_prior_terms`). Returns the new state, the block's ``[B, rows * T]``
    support values (pre-prior with ``statics.exact_weight_order``; ``None`` without ``statics.return_values``,
    for a score-only pass) and, for the ``dump_rows`` batch
    rows of a ``RELAX_SIGNIFICANCE_DUMP_*`` target (``None`` otherwise), their pre-prior
    and with-prior ``[n_targets, rows, T]`` scores.
    """

    (
        (global_max, global_sum),
        (class_max, class_sum),
        best,
        (class_best, class_best_argmax, class_second, class_second_argmax),
        raw_score_max,
    ) = block_state
    best_score, best_argmax, best_class = best
    batch_size = int(shifted_corrected.shape[0])
    if statics.score_kind == "gaussian":
        scores = relion_coarse_gaussian_gemm_scores_jit(
            reference,
            None,
            shifted_corrected,
            pixel_weight,
            initial_diff2,
            actual_image_count,
            n_images=batch_size,
            n_trans=int(statics.n_trans),
            image_shape=statics.image_shape,
            volume_shape=statics.volume_shape,
            float64=statics.float64,
        )
    elif statics.score_kind == "normalized_cc":
        scores = relion_coarse_normalized_cc_gemm_scores_jit(
            jnp.asarray(reference, dtype=jnp.complex64),
            shifted_corrected,
            pixel_weight,
            actual_image_count,
            n_images=batch_size,
            n_trans=int(statics.n_trans),
        )
    else:
        raise ValueError(f"unknown coarse score kind {statics.score_kind!r}")
    if rows < block_rows:
        scores = jnp.where(jnp.arange(block_rows)[None, :, None] < rows, scores, -jnp.inf)
    raw_score_max = jnp.maximum(raw_score_max, jnp.max(scores.reshape(batch_size, -1), axis=1))
    pre_prior_scores = scores
    if statics.score_kind == "gaussian":
        class_log_prior, rotation_log_prior_block = prior_terms
        scores = _add_coarse_prior_terms(scores, class_log_prior, rotation_log_prior_block, translation_log_prior)
    values = None
    if statics.return_values:
        values = (pre_prior_scores if statics.exact_weight_order else scores)[:, :rows, :].reshape(batch_size, -1)
    dump = None
    if dump_rows is not None:
        dump = (pre_prior_scores[dump_rows, :rows, :], scores[dump_rows, :rows, :])
    class_max, class_sum = update_logsumexp(class_max, class_sum, scores)
    global_max, global_sum = update_logsumexp(global_max, global_sum, scores)
    flat_scores = scores.reshape(batch_size, -1)
    block_best = jnp.max(flat_scores, axis=1)
    block_argmax = jnp.argmax(flat_scores, axis=1)
    improved = block_best > best_score
    best_score = jnp.where(improved, block_best, best_score)
    best_argmax = jnp.where(improved, block_argmax + rotation_start * int(statics.n_trans), best_argmax)
    best_class = jnp.where(improved, class_index, best_class)
    if statics.return_class_best:
        class_improved = block_best > class_best
        if statics.track_class_second:
            if flat_scores.shape[1] < 2:
                raise RuntimeError("class runner-up diagnostic requires at least two poses per block")
            block_without_best = flat_scores.at[jnp.arange(batch_size), block_argmax].set(-jnp.inf)
            block_second = jnp.max(block_without_best, axis=1)
            block_second_argmax = jnp.argmax(block_without_best, axis=1)
            improved_second_from_previous = class_best >= block_second
            improved_second = jnp.where(improved_second_from_previous, class_best, block_second)
            improved_second_argmax = jnp.where(
                improved_second_from_previous,
                class_best_argmax,
                block_second_argmax + rotation_start * int(statics.n_trans),
            )
            retained_second_from_previous = class_second >= block_best
            retained_second = jnp.where(retained_second_from_previous, class_second, block_best)
            retained_second_argmax = jnp.where(
                retained_second_from_previous,
                class_second_argmax,
                block_argmax + rotation_start * int(statics.n_trans),
            )
            class_second = jnp.where(class_improved, improved_second, retained_second)
            class_second_argmax = jnp.where(class_improved, improved_second_argmax, retained_second_argmax)
        class_best = jnp.where(class_improved, block_best, class_best)
        class_best_argmax = jnp.where(
            class_improved, block_argmax + rotation_start * int(statics.n_trans), class_best_argmax
        )
    state = (
        (global_max, global_sum),
        (class_max, class_sum),
        (best_score, best_argmax, best_class),
        (class_best, class_best_argmax, class_second, class_second_argmax),
        raw_score_max,
    )
    return state, values, dump


def class_block_state(state, class_index: int):
    """``state`` (:func:`pass1_initial_state`) narrowed to class ``class_index``'s entries."""

    global_terms, (class_max, class_sum), best, class_poses, raw_score_max = state
    return (
        global_terms,
        (class_max[class_index], class_sum[class_index]),
        best,
        tuple(values[class_index] for values in class_poses),
        raw_score_max,
    )


def merge_class_block_state(state, block_state, class_index: int):
    """``state`` with class ``class_index``'s entries and the shared entries from ``block_state``."""

    _, (class_max, class_sum), _, class_poses, _ = state
    global_terms, (block_max, block_sum), best, block_poses, raw_score_max = block_state

    def put(values, value):
        return values[:class_index] + (value,) + values[class_index + 1 :]

    return (
        global_terms,
        (put(class_max, block_max), put(class_sum, block_sum)),
        best,
        tuple(put(values, value) for values, value in zip(class_poses, block_poses, strict=True)),
        raw_score_max,
    )


@partial(jax.jit, static_argnames=("blocks", "statics"))
def coarse_pass1_blocks(
    state,
    projection_cache,
    shifted_corrected,
    pixel_weight,
    initial_diff2,
    actual_image_count,
    prior_terms,
    translation_log_prior,
    dump_rows=None,
    *,
    blocks: tuple,
    statics: ProgramStatics,
):
    """Pass 1 of one image batch over every cached class and rotation block, as one program.

    One program instead of the per-class, per-block eager loop of
    :func:`_compute_k_class_significance_batched`: the same statements
    (:func:`_pass1_block_update`) in the same class-then-block order, so the same
    values. ``blocks`` holds static ``(class_index, rotation_start, rows, block_rows)``
    entries whose rows are read from the ``[K, R, P]`` cached projection table;
    ``prior_terms[i]`` is block ``i``'s class prior and rotation-prior block (``None``
    without a rotation prior). ``state`` is :func:`pass1_initial_state`: ``((global
    max, sum), (class maxima, sums), (best score, pose, class), (class best scores,
    poses, runner-up scores, poses), raw score maximum)``. Returns the new state, one ``[B, rows * T]``
    support-value block per entry and, with ``dump_rows``, each entry's dump scores
    (``None`` without). At K15, 400+ eager programs per batch kept the
    host behind the device queue while the device idled between batches (K15 50k,
    nsys 14868231).
    """

    values = []
    dumps = []
    for block_index, (class_index, r0, rows, block_rows) in enumerate(blocks):
        reference = jax.lax.dynamic_slice_in_dim(projection_cache[class_index], r0, rows, axis=0)
        if rows < block_rows:
            reference = jnp.pad(reference, ((0, block_rows - rows), (0, 0)))
        block_state, block_values, block_dump = _pass1_block_update(
            class_block_state(state, class_index),
            reference,
            shifted_corrected,
            pixel_weight,
            initial_diff2,
            actual_image_count,
            prior_terms[block_index],
            translation_log_prior,
            class_index,
            r0,
            dump_rows,
            rows=rows,
            block_rows=block_rows,
            statics=statics,
        )
        state = merge_class_block_state(state, block_state, class_index)
        values.append(block_values)
        dumps.append(block_dump)
    return state, tuple(values), None if dump_rows is None else tuple(dumps)


@partial(jax.jit, static_argnames=("rows", "block_rows", "statics"))
def coarse_pass1_block(
    block_state,
    reference,
    shifted_corrected,
    pixel_weight,
    initial_diff2,
    actual_image_count,
    prior_terms,
    translation_log_prior,
    class_index,
    rotation_start,
    dump_rows=None,
    *,
    rows: int,
    block_rows: int,
    statics: ProgramStatics,
):
    """One class's rotation block of pass 1 on its projection (:func:`_pass1_block_update`).

    A pass whose projection cache does not fit, and every ``--firstiter_cc`` pass, folds
    its blocks through this program one at a time; the class index and rotation start
    are runtime values, so one program serves every class and full block.
    """

    return _pass1_block_update(
        block_state,
        reference,
        shifted_corrected,
        pixel_weight,
        initial_diff2,
        actual_image_count,
        prior_terms,
        translation_log_prior,
        class_index,
        rotation_start,
        dump_rows,
        rows=rows,
        block_rows=block_rows,
        statics=statics,
    )


def run_score_program(
    plan: ScoreProgramPlan,
    program_inputs,
    translation_log_prior,
    actual_batch_size: int,
    dump_rows,
) -> BatchScores:
    """Score one batch: ``program_inputs`` are its ``(shifted, pixel_weight, initial_diff2)``.

    Pass 1 of the coarse GEMM scorers is one program per batch (``coarse_pass1_blocks``): over the cached
    projections in one call, or one call per class and rotation block on its projection when the cache does not fit
    and for --firstiter_cc. The program also returns the ``RELAX_SIGNIFICANCE_DUMP_*`` targets' scores (``dump_rows``,
    batch rows or ``None``) and the class runner-up. ``translation_log_prior`` is the batch's prior on the device
    (``None``, ``[T]`` or ``[B, T]``) and ``actual_batch_size`` the number of real rows.
    """

    batch_size = int(program_inputs[0].shape[0])
    neg_inf_f, zeros_f64, zeros_i32 = pass1_batch_constants(batch_size, bool(jax.config.jax_enable_x64))
    state = pass1_initial_state((neg_inf_f, zeros_f64, zeros_i32), plan.n_classes)
    if plan.projection_cache is not None and plan.statics.score_kind != "normalized_cc":
        state, values, dumps = coarse_pass1_blocks(
            state,
            plan.projection_cache,
            *program_inputs,
            actual_batch_size,
            plan.prior_terms,
            translation_log_prior,
            dump_rows,
            blocks=plan.blocks,
            statics=plan.statics,
        )
    else:
        values = []
        dumps = []
        for block_index, (class_index, r0, rows, block_rows) in enumerate(plan.blocks):
            rots_b = plan.rotations_padded[r0 : r0 + block_rows]
            if plan.statics.score_kind == "normalized_cc":
                reference, _ = plan.projector.block_once(class_index, rots_b, rotation_start=r0)
            else:
                reference, _ = plan.projector.compact_rows(class_index, rots_b, return_abs2=True)
            block_state, block_values, block_dump = coarse_pass1_block(
                class_block_state(state, class_index),
                reference,
                *program_inputs,
                actual_batch_size,
                plan.prior_terms[block_index],
                translation_log_prior,
                jnp.int32(class_index),
                jnp.int32(r0),
                dump_rows,
                rows=rows,
                block_rows=block_rows,
                statics=plan.statics,
            )
            state = merge_class_block_state(state, block_state, class_index)
            values.append(block_values)
            dumps.append(block_dump)
    support_values = jnp.concatenate(values, axis=1) if plan.statics.return_values else None
    dump_pre_prior_blocks = dump_with_prior_blocks = None
    if dump_rows is not None:
        # Per-class collectors for raw (pre-prior) score blocks at target rows.
        # Shape after concat per class: (n_targets, n_rot, n_trans)
        dump_pre_prior_blocks = [[] for _ in range(plan.n_classes)]
        dump_with_prior_blocks = [[] for _ in range(plan.n_classes)]
        for (class_index, _, _, _), (pre_prior, with_prior) in zip(plan.blocks, dumps, strict=True):
            dump_pre_prior_blocks[class_index].append(np.asarray(pre_prior, dtype=np.float64))
            dump_with_prior_blocks[class_index].append(np.asarray(with_prior, dtype=np.float64))
    (
        (global_max, global_sum),
        (class_max, class_sum),
        (best_score, best_argmax, best_class),
        (class_best, class_best_argmax, class_second, class_second_argmax),
        raw_score_max,
    ) = state
    return BatchScores(
        global_max=global_max,
        global_sum=global_sum,
        class_max=list(class_max),
        class_sum=list(class_sum),
        best_score=best_score,
        best_argmax=best_argmax,
        best_class=best_class,
        class_best_scores=list(class_best) if plan.statics.return_class_best else None,
        class_best_argmaxes=list(class_best_argmax) if plan.statics.return_class_best else None,
        class_second_best_scores=list(class_second) if plan.statics.track_class_second else None,
        class_second_best_argmaxes=list(class_second_argmax) if plan.statics.track_class_second else None,
        raw_score_max=raw_score_max,
        support_values=support_values,
        dump_pre_prior_blocks=dump_pre_prior_blocks,
        dump_with_prior_blocks=dump_with_prior_blocks,
    )
