"""Running pass 1's score program for one image batch: the blocks, the cached or projected references, the results."""

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from relax.scoring.coarse_projector import CoarseProjector
from relax.scoring.pass1_program import (
    _class_block_state,
    _coarse_pass1_block,
    _coarse_pass1_blocks,
    _merge_class_block_state,
    _pass1_batch_constants,
    _pass1_initial_state,
)


def score_blocks(n_classes: int, n_rot: int, rotation_block_size: int) -> tuple:
    """The static ``(class_index, rotation_start, rows, block_rows)`` blocks of a pass, class by class.

    ``rows`` is the block's number of real rotations (the last block of a class may be short) and ``block_rows`` the
    padded block size every program is traced at.
    """

    return tuple(
        (class_index, r0, min(rotation_block_size, n_rot - r0), int(rotation_block_size))
        for class_index in range(n_classes)
        for r0 in range(0, n_rot, rotation_block_size)
    )


def block_prior_terms(blocks: tuple, class_log_priors, rotation_log_prior_padded, rotation_block_size: int) -> tuple:
    """For each of ``blocks``: its class log prior (float32) and its rotation log prior block (``None``: no prior).

    ``class_log_priors`` is ``[K]``; ``rotation_log_prior_padded`` is ``[K, R_padded]`` or ``None``.
    """

    return tuple(
        (
            jnp.asarray(class_log_priors[class_index], dtype=jnp.float32),
            None
            if rotation_log_prior_padded is None
            else jnp.asarray(rotation_log_prior_padded[class_index, r0 : r0 + rotation_block_size]),
        )
        for class_index, r0, _, _ in blocks
    )


@dataclass(frozen=True)
class ScoreProgramPlan:
    """What every batch's score program runs over, fixed for the pass.

    ``blocks`` and ``prior_terms`` are aligned (:func:`score_blocks`, :func:`block_prior_terms`). ``rotations_padded``
    are the rotations padded with identities to whole blocks. ``projector`` yields a block's reference rows; with a
    ``projection_cache`` (the cached C64 table of the Gaussian GEMM) one program reads every block from it, otherwise
    each block is projected and scored in its own call. ``score_kind`` is ``"gaussian"`` or ``"normalized_cc"``;
    ``exact_weight_order`` asks for the pre-prior support values (RELION's log-weight order); ``return_values`` asks
    for the support values at all. The remaining fields are the kernels' static arguments
    (:func:`relax.scoring.pass1_program._pass1_block_update`).
    """

    blocks: tuple
    prior_terms: tuple
    rotations_padded: Any
    projector: CoarseProjector
    projection_cache: Any
    n_classes: int
    n_trans: int
    image_shape: tuple
    volume_shape: tuple
    float64: bool
    score_kind: str
    exact_weight_order: bool
    return_class_best: bool
    track_class_second: bool
    return_values: bool


class BatchScores(NamedTuple):
    """One batch's results of the score program, on the device (``B`` padded rows).

    ``global_max`` and ``global_sum`` are the running log-sum-exp over every pose; ``class_max`` and ``class_sum``
    hold one array per class. ``best_score``, ``best_argmax`` and ``best_class`` are the joint winner (the pose index
    is ``rotation * T + translation`` within its class). ``class_best_*`` and ``class_second_best_*`` hold one array per
    class, ``None`` unless the call asked for the best (``return_class_best``) or the runner-up
    (``track_class_second``). ``raw_score_max`` is the maximum of the pre-prior scores. ``support_values`` are the
    class-major ``[B, K * R * T]`` values the posterior is formed from (``None`` without ``return_values``).
    ``dump_pre_prior_blocks`` and ``dump_with_prior_blocks`` hold, per class, the host score blocks of the dump rows
    (``None`` without dump rows).
    """

    global_max: Any
    global_sum: Any
    class_max: list
    class_sum: list
    best_score: Any
    best_argmax: Any
    best_class: Any
    class_best_scores: list | None
    class_best_argmaxes: list | None
    class_second_best_scores: list | None
    class_second_best_argmaxes: list | None
    raw_score_max: Any
    support_values: Any
    dump_pre_prior_blocks: list | None
    dump_with_prior_blocks: list | None


def run_score_program(
    plan: ScoreProgramPlan,
    program_inputs,
    translation_log_prior,
    actual_batch_size: int,
    dump_rows,
) -> BatchScores:
    """Score one batch: ``program_inputs`` are its ``(shifted, pixel_weight, initial_diff2)``.

    Pass 1 of the coarse GEMM scorers is one program per batch (``_coarse_pass1_blocks``): over the cached
    projections in one call, or one call per class and rotation block on its projection when the cache does not fit
    and for --firstiter_cc. The program also returns the ``RELAX_SIGNIFICANCE_DUMP_*`` targets' scores (``dump_rows``,
    batch rows or ``None``) and the class runner-up. ``translation_log_prior`` is the batch's prior on the device
    (``None``, ``[T]`` or ``[B, T]``) and ``actual_batch_size`` the number of real rows.
    """

    batch_size = int(program_inputs[0].shape[0])
    neg_inf_f, zeros_f64, zeros_i32 = _pass1_batch_constants(batch_size, bool(jax.config.jax_enable_x64))
    static = dict(
        n_trans=plan.n_trans,
        image_shape=plan.image_shape,
        volume_shape=plan.volume_shape,
        float64=plan.float64,
        score_kind=plan.score_kind,
        exact_weight_order=plan.exact_weight_order,
        return_class_best=plan.return_class_best,
        track_class_second=plan.track_class_second,
        return_values=plan.return_values,
    )
    state = _pass1_initial_state((neg_inf_f, zeros_f64, zeros_i32), plan.n_classes)
    if plan.projection_cache is not None and plan.score_kind != "normalized_cc":
        state, values, dumps = _coarse_pass1_blocks(
            state,
            plan.projection_cache,
            *program_inputs,
            actual_batch_size,
            plan.prior_terms,
            translation_log_prior,
            dump_rows,
            blocks=plan.blocks,
            **static,
        )
    else:
        values = []
        dumps = []
        for block_index, (class_index, r0, rows, block_rows) in enumerate(plan.blocks):
            rots_b = plan.rotations_padded[r0 : r0 + block_rows]
            if plan.score_kind == "normalized_cc":
                reference, _ = plan.projector.block_once(class_index, rots_b, rotation_start=r0)
            else:
                reference, _ = plan.projector.compact_rows(class_index, rots_b, return_abs2=True)
            block_state, block_values, block_dump = _coarse_pass1_block(
                _class_block_state(state, class_index),
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
                **static,
            )
            state = _merge_class_block_state(state, block_state, class_index)
            values.append(block_values)
            dumps.append(block_dump)
    support_values = jnp.concatenate(values, axis=1) if plan.return_values else None
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
        class_best_scores=list(class_best) if plan.return_class_best else None,
        class_best_argmaxes=list(class_best_argmax) if plan.return_class_best else None,
        class_second_best_scores=list(class_second) if plan.track_class_second else None,
        class_second_best_argmaxes=list(class_second_argmax) if plan.track_class_second else None,
        raw_score_max=raw_score_max,
        support_values=support_values,
        dump_pre_prior_blocks=dump_pre_prior_blocks,
        dump_with_prior_blocks=dump_with_prior_blocks,
    )
