"""The pass-1 score program of one image batch: the block scores, the priors and the running reductions.

One program per batch over the cached projections (``coarse_pass1_blocks``), or one call per class and rotation block
(``coarse_pass1_block``) when the cache does not fit and for ``--firstiter_cc``. ``_pass1_block_update`` is the
shared body. These are the jitted, state-passing kernels; :mod:`relax.scoring.significance` drives them.
"""

from functools import lru_cache, partial

import jax
import jax.numpy as jnp

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
    n_trans: int,
    image_shape: tuple,
    volume_shape: tuple,
    float64: bool,
    score_kind: str,
    exact_weight_order: bool,
    return_class_best: bool,
    track_class_second: bool,
    return_values: bool = True,
):
    """One class's rotation block of pass 1: its scores and the running reductions it updates.

    ``block_state`` is ``((global max, sum), (class max, sum), (best score, pose, class),
    (class best score, pose, runner-up score, pose), raw score maximum)`` with the class
    entries of ``class_index`` only; the runner-up (``track_class_second``, with
    ``return_class_best``) is the best pose of the class other than its best one; ``class_index`` and ``rotation_start`` may be traced.
    ``score_kind`` is ``"gaussian"`` (the GEMM scores of the projected or cached rows,
    :func:`relion_coarse_gaussian_gemm_scores_jit`, with ``initial_diff2``) or
    ``"normalized_cc"`` (RELION's coarse CC, :func:`relion_coarse_normalized_cc_gemm_scores_jit`;
    no priors, as RELION's ``--firstiter_cc`` scores have none). The padded tail rows are
    ``-inf``; the Gaussian scores get the class, rotation and translation priors
    (:func:`_add_coarse_prior_terms`). Returns the new state, the block's ``[B, rows * T]``
    support values (pre-prior with ``exact_weight_order``; ``None`` without ``return_values``, for a
    score-only pass) and, for the ``dump_rows`` batch
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
    if score_kind == "gaussian":
        scores = relion_coarse_gaussian_gemm_scores_jit(
            reference,
            None,
            shifted_corrected,
            pixel_weight,
            initial_diff2,
            actual_image_count,
            n_images=batch_size,
            n_trans=int(n_trans),
            image_shape=image_shape,
            volume_shape=volume_shape,
            float64=float64,
        )
    elif score_kind == "normalized_cc":
        scores = relion_coarse_normalized_cc_gemm_scores_jit(
            jnp.asarray(reference, dtype=jnp.complex64),
            shifted_corrected,
            pixel_weight,
            actual_image_count,
            n_images=batch_size,
            n_trans=int(n_trans),
        )
    else:
        raise ValueError(f"unknown coarse score kind {score_kind!r}")
    if rows < block_rows:
        scores = jnp.where(jnp.arange(block_rows)[None, :, None] < rows, scores, -jnp.inf)
    raw_score_max = jnp.maximum(raw_score_max, jnp.max(scores.reshape(batch_size, -1), axis=1))
    pre_prior_scores = scores
    if score_kind == "gaussian":
        class_log_prior, rotation_log_prior_block = prior_terms
        scores = _add_coarse_prior_terms(scores, class_log_prior, rotation_log_prior_block, translation_log_prior)
    values = None
    if return_values:
        values = (pre_prior_scores if exact_weight_order else scores)[:, :rows, :].reshape(batch_size, -1)
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
    best_argmax = jnp.where(improved, block_argmax + rotation_start * int(n_trans), best_argmax)
    best_class = jnp.where(improved, class_index, best_class)
    if return_class_best:
        class_improved = block_best > class_best
        if track_class_second:
            if flat_scores.shape[1] < 2:
                raise RuntimeError("class runner-up diagnostic requires at least two poses per block")
            block_without_best = flat_scores.at[jnp.arange(batch_size), block_argmax].set(-jnp.inf)
            block_second = jnp.max(block_without_best, axis=1)
            block_second_argmax = jnp.argmax(block_without_best, axis=1)
            improved_second_from_previous = class_best >= block_second
            improved_second = jnp.where(improved_second_from_previous, class_best, block_second)
            improved_second_argmax = jnp.where(
                improved_second_from_previous, class_best_argmax, block_second_argmax + rotation_start * int(n_trans)
            )
            retained_second_from_previous = class_second >= block_best
            retained_second = jnp.where(retained_second_from_previous, class_second, block_best)
            retained_second_argmax = jnp.where(
                retained_second_from_previous, class_second_argmax, block_argmax + rotation_start * int(n_trans)
            )
            class_second = jnp.where(class_improved, improved_second, retained_second)
            class_second_argmax = jnp.where(class_improved, improved_second_argmax, retained_second_argmax)
        class_best = jnp.where(class_improved, block_best, class_best)
        class_best_argmax = jnp.where(class_improved, block_argmax + rotation_start * int(n_trans), class_best_argmax)
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


_PASS1_STATIC = (
    "n_trans",
    "image_shape",
    "volume_shape",
    "float64",
    "score_kind",
    "exact_weight_order",
    "return_class_best",
    "track_class_second",
    "return_values",
)


@partial(jax.jit, static_argnames=("blocks",) + _PASS1_STATIC)
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
    n_trans: int,
    image_shape: tuple,
    volume_shape: tuple,
    float64: bool,
    score_kind: str,
    exact_weight_order: bool,
    return_class_best: bool,
    track_class_second: bool = False,
    return_values: bool = True,
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
            n_trans=n_trans,
            image_shape=image_shape,
            volume_shape=volume_shape,
            float64=float64,
            score_kind=score_kind,
            exact_weight_order=exact_weight_order,
            return_class_best=return_class_best,
            track_class_second=track_class_second,
            return_values=return_values,
        )
        state = merge_class_block_state(state, block_state, class_index)
        values.append(block_values)
        dumps.append(block_dump)
    return state, tuple(values), None if dump_rows is None else tuple(dumps)


@partial(jax.jit, static_argnames=("rows", "block_rows") + _PASS1_STATIC)
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
    n_trans: int,
    image_shape: tuple,
    volume_shape: tuple,
    float64: bool,
    score_kind: str,
    exact_weight_order: bool,
    return_class_best: bool,
    track_class_second: bool = False,
    return_values: bool = True,
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
        n_trans=n_trans,
        image_shape=image_shape,
        volume_shape=volume_shape,
        float64=float64,
        score_kind=score_kind,
        exact_weight_order=exact_weight_order,
        return_class_best=return_class_best,
        track_class_second=track_class_second,
        return_values=return_values,
    )
