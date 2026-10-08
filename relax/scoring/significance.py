"""Coarse significance pruning for adaptive class and pose searches.

The shared scorer serves K=1, multiclass refinement and initial-model searches.
It selects per-image significant samples from one class/rotation/translation
posterior, with optional diagnostic capture of the coarse scoring boundary.
"""

import logging
import operator
import os
import time
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np
from recovar.utils.nvtx_shim import nvtx

from relax.diagnostics.coarse_gaussian_diagnostics import (
    _maybe_dump_k_class_significance_batch,
    _maybe_dump_tree_rescore_batch,
    _significance_debug_dump_matches,
)
from relax.diagnostics.coarse_score_diagnostics import (
    _build_coarse_significance_support_audit,
)
from relax.helpers.batch_fetch import original_image_indices
from relax.helpers.env_flags import (
    parse_env_int_set,
    parse_env_strict_flag,
)
from relax.helpers.projection_cache import build_projection_cache
from relax.relion.relion_coarse_operands import (
    _assemble_relion_exact_coarse_gaussian_operands,
    _infer_relion_coarse_healpix_order,
    _k1_relion_f32_coarse_support_enabled,
    _process_relion_exact_coarse_half_image,
    _relion_cc_inverse_power_from_processed,
    _repeat_pad_batch_axis,
    _select_relion_coarse_rescore_winner_slots,
    assemble_relion_cc_coarse_operands,
)
from relax.scoring.coarse_gaussian_gemm import (
    _COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_ENV,
    _coarse_gaussian_gemm_cached_block_rows,
    _coarse_gaussian_gemm_fit_rotation_block_size,
    _coarse_gaussian_gemm_projected_transient_budget_bytes,
    _coarse_gaussian_gemm_projection_cache_budget_bytes,
    _coarse_gaussian_gemm_projection_cache_enabled,
    _coarse_gaussian_gemm_projection_cache_stats,
    _coarse_gaussian_gemm_projection_row_bytes,
    _coarse_gaussian_gemm_resources,
    _plan_coarse_gaussian_gemm_projection_cache,
    _validate_coarse_gaussian_gemm_projection_cache_request,
)
from relax.scoring.coarse_layout import compact_projection_window_positions, plan_coarse_gaussian_square_layout
from relax.scoring.coarse_publication import coarse_square_layout_metadata, coarse_support_posterior
from relax.scoring.pass1_results import BatchOutputs
from relax.scoring.scoring import (
    _coarse_gemm_float64_requested,
    _relion_coarse_gaussian_gemm_scores_jit,
    _relion_coarse_normalized_cc_gemm_scores_jit,
    _update_logsumexp,
)
from relax.scoring.significant_samples import compact_significant_sample_indices_from_mask

_GLOBAL_PASS1_RELION_PROJECTOR_TEXTURE_ENV = "RELAX_RELION_GLOBAL_PASS1_PROJECTOR_TEXTURE_INTERP"
_COARSE_SIGNIFICANCE_SUPPORT_AUDIT_ENV = (
    "RECOVAR_COARSE_SIGNIFICANCE_SUPPORT_AUDIT"
)
_COARSE_SIGNIFICANCE_SUPPORT_AUDIT_IDS_ENV = (
    "RECOVAR_COARSE_SIGNIFICANCE_SUPPORT_AUDIT_IDS"
)
_COARSE_PAD_FINAL_IMAGE_BATCH_ENV = (
    "RELAX_COARSE_PAD_FINAL_IMAGE_BATCH"
)
NVTX_DOMAIN_EM = "recovar_em"
logger = logging.getLogger(__name__)

# Byte cap on the coarse projections one pass-1 call keeps for reuse across image
# batches (see ``_project_block_once``).
_PASS1_PROJECTION_MEMO_MAX_BYTES = 2 * 1024**3


def _pad_significance_preprocess_inputs(
    batch_data,
    ctf_params,
    integer_pre_shifts,
    batch_corr,
    batch_scale,
    relion_preprocess_kwargs,
    *,
    target_size: int,
):
    """Give a tail significance batch the same compiled image shape as full batches."""

    actual_size = int(batch_data.shape[0])
    target_size = max(actual_size, int(target_size))
    if actual_size == target_size:
        return (
            batch_data,
            ctf_params,
            integer_pre_shifts,
            batch_corr,
            batch_scale,
            relion_preprocess_kwargs,
        )
    padded_kwargs = None
    if relion_preprocess_kwargs is not None:
        padded_kwargs = {
            key: jnp.asarray(_repeat_pad_batch_axis(value, target_size))
            for key, value in relion_preprocess_kwargs.items()
        }
    return (
        _repeat_pad_batch_axis(batch_data, target_size),
        _repeat_pad_batch_axis(ctf_params, target_size),
        (
            None
            if integer_pre_shifts is None
            else _repeat_pad_batch_axis(integer_pre_shifts, target_size)
        ),
        None if batch_corr is None else _repeat_pad_batch_axis(batch_corr, target_size),
        _repeat_pad_batch_axis(batch_scale, target_size),
        padded_kwargs,
    )


def _coarse_rotated_radius_enabled(*, default: bool = False) -> bool:
    """Select canonical clipping when the active projector supports it."""
    token = os.environ.get("RELAX_K1_COARSE_ROTATED_RADIUS", "1" if default else "0")
    if token not in {"0", "1"}:
        raise ValueError("RELAX_K1_COARSE_ROTATED_RADIUS must be 0 or 1")
    return token == "1"


def _custom_cuda_ready() -> bool:
    """Whether JAX runs on a GPU with RECOVAR's custom CUDA library loaded (pass 1's FFI kernels)."""

    from recovar import cuda_backproject

    return jax.default_backend() == "gpu" and cuda_backproject.cuda_available()


def _require_exact_pass1_operands(
    *,
    use_relion_projector: bool,
    coarse_texture_interp: bool,
    half_spectrum_scoring: bool,
    use_float64_scoring: bool,
    use_float64_projections: bool | None,
) -> None:
    """Refuse a coarse pass that RELION's exact coarse operands cannot score.

    Pass 1 projects with the supplied RELION projector by texture interpolation,
    scores the half spectrum in float32 and runs RELION's CUDA preprocessing and
    translation, as RELION's GPU coarse kernels do. The generic dense scorer that
    served the other configurations was removed on 2026-10-02.
    """

    missing = []
    if not use_relion_projector:
        missing.append("the supplied RELION projector (relion_projector_half)")
    if not coarse_texture_interp:
        missing.append("texture interpolation of that projector")
    if not half_spectrum_scoring:
        missing.append("half-spectrum scoring")
    if use_float64_scoring or use_float64_projections:
        missing.append("float32 scoring and projections")
    if missing:
        raise ValueError("pass 1 scores RELION's exact coarse operands and needs " + ", ".join(missing))


def _coarse_max_posterior_for_host(batch_weights, actual_batch_size):
    """Publish active row maxima without specializing on fringe batch sizes.

    Rows are independent. Reducing the physical table before slicing the
    compact host vector keeps padded rows out of the published statistics.
    """
    return np.asarray(jnp.max(batch_weights, axis=1), dtype=np.float32)[
        :actual_batch_size
    ]


@jax.jit
def _any_over_leading_rows(mask, n_rows):
    """``jnp.any(mask[:n_rows], axis=0)`` with a runtime row count.

    The count is an operand, so one program serves every batch tail.
    """
    active = jnp.arange(mask.shape[0]) < n_rows
    return jnp.any(mask & active.reshape((-1,) + (1,) * (mask.ndim - 1)), axis=0)


def _coarse_significance_support_audit_enabled(
    *,
    default: bool = False,
) -> bool:
    """Resolve exact, diagnostic-only coarse-support hashing."""

    return parse_env_strict_flag(_COARSE_SIGNIFICANCE_SUPPORT_AUDIT_ENV, default=default)


def _coarse_pad_final_image_batch_enabled(*, default: bool = True) -> bool:
    """Whether every coarse image batch is padded to the requested batch size.

    A half set is rarely an exact multiple of ``image_batch_size``, so its last
    batch has its own image extent and every program in the coarse pass,
    significance and image preprocessing is traced and compiled a second time
    for it. Padding repeats image row zero up to ``image_batch_size``; the
    repeated rows are discarded from every science output, so the retained
    values are bit-for-bit those of the unpadded batch (the per-image
    computations are independent, and the batch-level reductions are maxima
    over rows that row zero already contributes).

    On by default since the K=1 resident flip (qualified with the resident flag
    set on the fast tier and the four flip datasets); ``=0`` turns it off. VDAM
    requests the same padding explicitly through ``pad_final_image_batch``.
    """

    return parse_env_strict_flag(
        _COARSE_PAD_FINAL_IMAGE_BATCH_ENV,
        default=default,
    )


def _coarse_significance_support_audit_ids_enabled() -> bool:
    """Whether a support audit also retains its exact selected IDs."""
    return parse_env_strict_flag(_COARSE_SIGNIFICANCE_SUPPORT_AUDIT_IDS_ENV)




def _capture_offset_free_and_absolute_float32_scores(scores, log_score_offset):
    """Capture native score margins before adding a large common offset."""

    offset_free = np.asarray(scores, dtype=np.float32)
    absolute = (
        np.asarray(scores, dtype=np.float64) + np.asarray(log_score_offset, dtype=np.float64)
    ).astype(np.float32)
    return offset_free, absolute


def _global_pass1_relion_projector_texture_enabled() -> bool:
    """Whether dense/global pass-1 significance uses texture arithmetic.

    Coarse significance defaults to RELION's texture projector.  Set the
    environment flag to false to force the manual/JAX diagnostic fallback.
    """
    return parse_env_strict_flag(
        _GLOBAL_PASS1_RELION_PROJECTOR_TEXTURE_ENV,
        default=True,
    )


def _dense_projection_scale(image_shape) -> float:
    """Match the dense E-step projection scaling used by the shared helper."""

    token = (os.environ.get("RELAX_DENSE_MEANS_SCALE") or "-N2").strip()
    n = int(image_shape[0])
    scale = {"-N2": -(n**2), "N2": float(n**2)}.get(token)
    if scale is None:
        raise ValueError(f"Unsupported RELAX_DENSE_MEANS_SCALE={token!r}")
    return scale


@lru_cache(maxsize=8)
def _pass1_batch_constants(batch_size: int, enable_x64: bool):
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


def _pass1_initial_state(batch_constants, n_classes: int):
    """The running pass-1 reductions of one image batch before its first block (``_coarse_pass1_blocks``).

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
    :func:`_relion_coarse_gaussian_gemm_scores_jit`, with ``initial_diff2``) or
    ``"normalized_cc"`` (RELION's coarse CC, :func:`_relion_coarse_normalized_cc_gemm_scores_jit`;
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
        scores = _relion_coarse_gaussian_gemm_scores_jit(
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
        scores = _relion_coarse_normalized_cc_gemm_scores_jit(
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
    class_max, class_sum = _update_logsumexp(class_max, class_sum, scores)
    global_max, global_sum = _update_logsumexp(global_max, global_sum, scores)
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


def _class_block_state(state, class_index: int):
    """``state`` (:func:`_pass1_initial_state`) narrowed to class ``class_index``'s entries."""

    global_terms, (class_max, class_sum), best, class_poses, raw_score_max = state
    return (
        global_terms,
        (class_max[class_index], class_sum[class_index]),
        best,
        tuple(values[class_index] for values in class_poses),
        raw_score_max,
    )


def _merge_class_block_state(state, block_state, class_index: int):
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
def _coarse_pass1_blocks(
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
    without a rotation prior). ``state`` is :func:`_pass1_initial_state`: ``((global
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
            _class_block_state(state, class_index),
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
        state = _merge_class_block_state(state, block_state, class_index)
        values.append(block_values)
        dumps.append(block_dump)
    return state, tuple(values), None if dump_rows is None else tuple(dumps)


@partial(jax.jit, static_argnames=("rows", "block_rows") + _PASS1_STATIC)
def _coarse_pass1_block(
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


@nvtx.annotate("kclass.adaptive.pass1_significance", color="orange", domain=NVTX_DOMAIN_EM)
def _prepare_coarse_relion_projector(
    projector, *, use_float64_scoring, use_float64_projections,
):
    """Upload at the explicit consumer precision, retaining the source slab.

    None preserves legacy direct callers. Double scoring/projection diagnostics
    retain complex128 projection; production K1/K4 explicitly request float32.
    """
    dtype = None
    if use_float64_projections is not None:
        dtype = jnp.complex128 if (use_float64_scoring or use_float64_projections) else jnp.complex64
    return jnp.asarray(projector, dtype=dtype)


def _class_stacked_coarse_relion_projector(
    projector, n_classes, *, use_float64_scoring, use_float64_projections,
):
    """Upload a coarse PPref as ``(n_classes, z, y, x_half)``.

    K=1 callers may pass one ``(z, y, x_half)`` slab, such as the host-compacted
    firstiter-CC projector; its class axis is added on the host before upload.
    """
    if np.ndim(projector) == 3 and int(n_classes) == 1:
        projector = projector[None]
    projector = _prepare_coarse_relion_projector(
        projector, use_float64_scoring=use_float64_scoring,
        use_float64_projections=use_float64_projections,
    )
    if projector.ndim != 4 or int(projector.shape[0]) != int(n_classes):
        raise ValueError(
            "relion_projector_half must have shape "
            f"({n_classes}, z, y, x_half), or (z, y, x_half) for K=1; got {projector.shape}",
        )
    return projector


def _compute_k_class_significance_batched(
    experiment_dataset,
    noise_variance,
    rotations,
    translations,
    *,
    class_log_priors,
    adaptive_fraction,
    max_significants,
    image_batch_size,
    rotation_block_size,
    current_size,
    score_with_masked_images=False,
    rotation_log_prior=None,
    translation_log_prior=None,
    image_corrections=None,
    scale_corrections=None,
    image_pre_shifts=None,
    half_spectrum_scoring=False,
    projection_padding_factor=1,
    square_window=False,
    window_at_box=False,
    use_float64_scoring=False,
    use_float64_projections: bool | None = None,
    relion_projector_half=None,
    relion_projector_r_max=None,
    relion_projector_texture_interp: bool | None = None,
    score_mode: str = "gaussian",
    collect_significance: bool = True,
    return_relion_f32_normalization: bool = False,
    return_class_best: bool = False,
    return_class_second: bool = False,
    debug_iteration: int | None = None,
    coarse_healpix_order: int | None = None,
    coarse_rotation_ids=None,
    translation_phase_source=None,
    symmetry_label: str = "C1",
    relion_f32_coarse_tie_ulps: int = 0,
    pad_final_image_batch: bool = False,
    stable_fourier_window_shapes: bool = False,
    relion_translation_angle_scale: float = 1.0,
    optics_group_ids=None,
    tree_rescore_max_margin: float | None = None,
    firstiter_cc_support: str = "relion",
    nyquist_column_counting: str = "relion",
):
    """Find significant samples from one posterior over ``class x rotation x translation``.

    ``noise_variance`` is one shared spectrum or ``[G, P]`` rows of G optics groups;
    with rows, ``optics_group_ids`` gives each image's group and every image scores
    with its own group's spectrum (:mod:`relax.helpers.optics_noise`).

    ``firstiter_cc_support="gaussian"`` weights a normalized-CC pass, and its image power, on
    the Gaussian support of the current size; ``nyquist_column_counting="once"`` drops the
    redundant members of the full-size Nyquist column from the Gaussian weights and from the
    image power (docs/math/relion_consistency_options.md).
    """

    if return_class_second and not return_class_best:
        raise ValueError("return_class_second requires return_class_best")
    if return_relion_f32_normalization and (
        not collect_significance or score_mode != "gaussian" or use_float64_scoring
    ):
        raise ValueError("RELION float32 normalization requires Gaussian float32 significance")

    from recovar.reconstruction import noise as noise_utils

    from relax.helpers.fourier_window import make_fourier_window_spec, relion_fftw_order_for_square_score_window
    from relax.helpers.half_spectrum import make_scoring_half_image_weights, redundant_nyquist_column_pixels
    from relax.helpers.image_shifts import apply_relion_integer_pre_shifts, tiled_half_image_phase_factors
    from relax.helpers.oversampling import find_significant_rotations as _find_sig
    from relax.helpers.preprocessing import (
        prepare_batch_preprocess_operands,
    )
    from relax.helpers.projection import (
        compute_relion_projector_projections_block as _compute_relion_projector_projections_block,
    )
    from relax.scoring.scoring import (
        _relion_coarse_normalized_cc_rescore,
    )

    if score_mode not in {"gaussian", "normalized_cc"}:
        raise ValueError(f"score_mode must be 'gaussian' or 'normalized_cc', got {score_mode!r}")
    # VDAM asks for the padded tail batch explicitly; the global coarse pass
    # opts in through the environment while the bitwise equality is qualified.
    pad_final_image_batch = bool(pad_final_image_batch) or _coarse_pad_final_image_batch_enabled()
    # RELION's pdf_orientation/pdf_offset priors and score/evidence/Pmax
    # outputs are RFLOAT (double), never narrowed -- derive from the
    # caller's own use_float64_scoring instead of hardcoding float32.
    score_real_dtype = np.float64 if use_float64_scoring else np.float32
    # The class count is the length of the class priors; relion_projector_half carries the same count.
    class_log_priors_np = np.asarray(class_log_priors, dtype=np.float64).reshape(-1)
    n_classes = int(class_log_priors_np.shape[0])

    translations_source = np.asarray(
        translations if translation_phase_source is None else translation_phase_source,
    )
    if translations_source.shape != translations.shape:
        raise ValueError(
            "translation_phase_source must match translations: "
            f"{translations_source.shape} != {translations.shape}",
        )
    n_rot = int(rotations.shape[0])
    n_trans = int(translations.shape[0])
    n_images = int(experiment_dataset.n_units)
    input_image_batch_size = operator.index(image_batch_size)
    if input_image_batch_size <= 0:
        raise ValueError("image_batch_size must be positive")
    image_batch_size = input_image_batch_size
    image_shape = experiment_dataset.image_shape
    volume_shape = experiment_dataset.volume_shape
    n_half = int(image_shape[0] * (image_shape[1] // 2 + 1))
    if coarse_rotation_ids is not None:
        coarse_rotation_ids = np.asarray(coarse_rotation_ids, dtype=np.int64).reshape(-1)
        if coarse_rotation_ids.shape != (n_rot,):
            raise ValueError(
                f"coarse_rotation_ids must have shape ({n_rot},), got {coarse_rotation_ids.shape}",
            )
    if coarse_healpix_order is None:
        coarse_healpix_order = _infer_relion_coarse_healpix_order(n_rot, **({"symmetry_label": symmetry_label} if symmetry_label != "C1" else {}))
    elif int(coarse_healpix_order) < 0:
        raise ValueError(f"coarse_healpix_order must be non-negative, got {coarse_healpix_order}")

    use_relion_projector = relion_projector_half is not None
    if use_relion_projector and relion_projector_r_max is None:
        raise ValueError("relion_projector_r_max is required when relion_projector_half is provided")
    if use_relion_projector:
        relion_projector_half = _class_stacked_coarse_relion_projector(
            relion_projector_half, n_classes, use_float64_scoring=use_float64_scoring,
            use_float64_projections=use_float64_projections,
        )

    cc_gaussian_support = score_mode == "normalized_cc" and firstiter_cc_support == "gaussian"
    half_weights = make_scoring_half_image_weights(
        image_shape,
        relion_half_sum=half_spectrum_scoring,
        exclude_relion_redundant_x0=score_mode != "normalized_cc",
        nyquist_column_counting=nyquist_column_counting,
        firstiter_cc_support_size=(
            (image_shape[0] if current_size is None else current_size) if cc_gaussian_support else None
        ),
    )
    window_spec_kwargs = {}
    if score_mode == "normalized_cc":
        window_spec_kwargs = {
            "score_square": True,
            "score_include_dc": True,
        }
    window_spec = make_fourier_window_spec(
        image_shape,
        current_size,
        n_half,
        square=square_window,
        include_recon_window=False,
        # RELION's radial window at the box too (Gaussian scoring; the normalized-CC score
        # keeps its rectangular window).
        window_at_box=bool(window_at_box) and score_mode != "normalized_cc",
        **window_spec_kwargs,
    )
    use_window = window_spec.use_window
    score_size = int(image_shape[0]) if current_size is None else int(current_size)
    # RELION's coarse kernel projects and shifts the rows beyond maxR at ``i - window`` inside
    # the model sphere only for a window between 2 r_max and about 2 s r_max (an optics group on
    # a coarser grid: its rotations carry 1 / s); the exact coarse operands then shift them there.
    coarse_kernel_window = None
    if use_relion_projector and score_size // 2 > int(relion_projector_r_max):
        from relax.helpers.optics_scale import coarse_rows_wrap_inside

        rotation_scale = 1.0 / float(np.linalg.norm(np.asarray(rotations, dtype=np.float64).reshape(-1, 3, 3)[0, 0]))
        if coarse_rows_wrap_inside(score_size, int(relion_projector_r_max), rotation_scale):
            coarse_kernel_window = score_size
    window_indices = window_spec.score_indices
    coarse_texture_interp = (
        _global_pass1_relion_projector_texture_enabled()
        if relion_projector_texture_interp is None
        else bool(relion_projector_texture_interp)
    )
    if tree_rescore_max_margin is not None and not (
        np.isfinite(tree_rescore_max_margin) and tree_rescore_max_margin >= 0.0
    ):
        raise ValueError(f"tree_rescore_max_margin must be a finite non-negative float, got {tree_rescore_max_margin!r}")
    # The margin is a run option, while only iteration 1 uses normalized CC.
    # Later Gaussian iterations must remain unaffected.
    tree_rescore_enabled = (
        tree_rescore_max_margin is not None and score_mode == "normalized_cc"
    )
    # Pass 1 scores RELION's exact coarse operands only: the Gaussian passes with the coarse GEMMs
    # (_relion_coarse_gaussian_gemm_scores_jit), the --firstiter_cc passes with RELION's coarse CC
    # (_relion_coarse_normalized_cc_gemm_scores_jit). The generic dense scorer was removed on
    # 2026-10-02; like pass 2, pass 1 needs a CUDA GPU and RELION's CUDA image preprocessing.
    _require_exact_pass1_operands(
        use_relion_projector=use_relion_projector,
        coarse_texture_interp=coarse_texture_interp,
        half_spectrum_scoring=half_spectrum_scoring,
        use_float64_scoring=use_float64_scoring,
        use_float64_projections=use_float64_projections,
    )
    exact_gaussian = score_mode == "gaussian"
    coarse_gaussian_gemm_projection_cache_requested = exact_gaussian and (
        _coarse_gaussian_gemm_projection_cache_enabled(default=True)
    )
    # An explicit cache request fails closed; the default cache quietly stands
    # down wherever its contract or memory budget does not hold.
    coarse_gaussian_gemm_projection_cache_explicit = (
        _COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_ENV in os.environ
    )
    if coarse_gaussian_gemm_projection_cache_requested:
        try:
            _validate_coarse_gaussian_gemm_projection_cache_request(
                n_rotations=n_rot,
                # The dtype of the class stack: indexing a device array for it dispatched a slice per pass.
                relion_projector_dtype=(
                    relion_projector_half.dtype
                    if hasattr(relion_projector_half, "dtype")
                    else relion_projector_half[0].dtype
                ),
            )
        except (ValueError, TypeError) as reason:
            if coarse_gaussian_gemm_projection_cache_explicit:
                raise
            logger.info("coarse GEMM projection cache off: %s", reason)
            coarse_gaussian_gemm_projection_cache_requested = False
    relion_f32_coarse_support_enabled = exact_gaussian and _k1_relion_f32_coarse_support_enabled(default=True)
    # Compact the coarse support mask on the device instead of pulling it
    # (ticket T13) whenever the ids are collected; every dense-mask
    # diagnostic keeps the host pull, checked per batch.
    coarse_significance_device_enabled = bool(collect_significance)
    # Per class: each class's slice of the joint support mask is compacted on
    # its own, exactly as the host path encodes each class's slice.
    device_significance_counts = [[] for _ in range(int(n_classes))]
    device_significance_polarity = [[] for _ in range(int(n_classes))]
    device_significance_ids = [[] for _ in range(int(n_classes))]
    device_significance_starts = [[] for _ in range(int(n_classes))]
    if coarse_significance_device_enabled:
        from relax.sparse_pass2.resident_significance import (
            compact_batch_significance_classes,
        )
    coarse_gaussian_full_to_compact = None
    coarse_gaussian_full_to_compact_np = None
    coarse_gaussian_score_indices = None
    coarse_gaussian_score_indices_np = None
    coarse_gaussian_score_active_mask = None
    coarse_gaussian_window_positions = None
    coarse_gaussian_powerclass = None
    coarse_gaussian_projector_full = None
    coarse_gaussian_gemm_resource_estimate = None
    coarse_gaussian_gemm_projection_cache_plan = None
    coarse_gaussian_square_layout = None
    if exact_gaussian:
        if n_trans > 128:
            raise ValueError(f"the coarse GEMM scorer supports at most 128 translations, got {n_trans}")
        from relax.sparse_pass2.sparse_pass2_scoring import (
            _relion_cuda_powerclass_highres_xi2_half,
        )

        if not _custom_cuda_ready():
            raise RuntimeError("pass 1 scores RELION's exact coarse operands and needs the custom CUDA backend")
        active_score_indices_np = (
            np.arange(n_half, dtype=np.int32)
            if window_spec.score_indices_np is None
            else np.asarray(window_spec.score_indices_np, dtype=np.int32)
        )
        coarse_gaussian_square_layout = plan_coarse_gaussian_square_layout(
            image_shape,
            score_size,
            active_score_indices_np,
            stable_fourier_window_shapes=bool(stable_fourier_window_shapes),
        )
        square_score_indices_np = coarse_gaussian_square_layout.score_indices_np
        square_score_count = coarse_gaussian_square_layout.physical_square_count
        coarse_gaussian_projector_output_size = (
            coarse_gaussian_square_layout.physical_current_size
        )
        coarse_gaussian_score_indices_np = np.asarray(
            square_score_indices_np,
            dtype=np.int32,
        )
        coarse_gaussian_score_indices = jnp.asarray(
            coarse_gaussian_score_indices_np,
            dtype=jnp.int32,
        )
        coarse_gaussian_score_active_mask = jnp.asarray(
            coarse_gaussian_square_layout.score_active_mask_np,
            dtype=jnp.bool_,
        )
        coarse_gaussian_window_positions = jnp.asarray(
            compact_projection_window_positions(
                square_score_indices_np,
                active_score_indices_np,
            ),
            dtype=jnp.int32,
        )
        coarse_gaussian_full_to_compact_np = np.asarray(
            coarse_gaussian_square_layout.full_to_compact_np,
            dtype=np.int32,
        )
        coarse_gaussian_full_to_compact = jnp.asarray(
            coarse_gaussian_full_to_compact_np,
            dtype=jnp.int32,
        )
        coarse_gaussian_powerclass = _relion_cuda_powerclass_highres_xi2_half
        coarse_gaussian_gemm_transient_budget = (
            _coarse_gaussian_gemm_projected_transient_budget_bytes()
        )
        # The plain GEMM splits the rotation axis until the projector
        # transient fits.
        fitted_block_size = _coarse_gaussian_gemm_fit_rotation_block_size(
            int(rotation_block_size),
            image_shape=image_shape,
            compact_pixel_count=int(square_score_count),
            budget_bytes=coarse_gaussian_gemm_transient_budget,
        )
        if fitted_block_size != int(rotation_block_size):
            logger.info(
                "coarse GEMM rotation block %d -> %d rows to fit the %d-byte "
                "projector transient budget",
                int(rotation_block_size),
                fitted_block_size,
                coarse_gaussian_gemm_transient_budget,
            )
            rotation_block_size = fitted_block_size
        coarse_gaussian_gemm_transient_budget = max(
            coarse_gaussian_gemm_transient_budget,
            int(rotation_block_size)
            * _coarse_gaussian_gemm_projection_row_bytes(
                image_shape=image_shape,
                compact_pixel_count=int(square_score_count),
            ),
        )
        coarse_gaussian_gemm_resource_estimate = _coarse_gaussian_gemm_resources(
            rotation_block_size=int(rotation_block_size),
            image_shape=image_shape,
            compact_pixel_count=int(square_score_count),
            budget_bytes=coarse_gaussian_gemm_transient_budget,
        )
        if coarse_gaussian_gemm_projection_cache_requested:
            coarse_gaussian_gemm_projection_cache_plan = (
                _plan_coarse_gaussian_gemm_projection_cache(
                    n_classes=n_classes,
                    n_rotations=n_rot,
                    compact_pixel_count=int(square_score_count),
                    image_shape=image_shape,
                    budget_bytes=(
                        _coarse_gaussian_gemm_projection_cache_budget_bytes()
                    ),
                )
            )
            if not (
                coarse_gaussian_gemm_projection_cache_plan.admitted
                or coarse_gaussian_gemm_projection_cache_explicit
            ):
                logger.info(
                    "coarse GEMM projection cache off: %s",
                    coarse_gaussian_gemm_projection_cache_plan.admission_reason,
                )
                coarse_gaussian_gemm_projection_cache_plan = None
        if coarse_gaussian_gemm_projection_cache_plan is not None:
            # Cached projections need no per-block projector transient, so the
            # GEMM block grows to what its own temporaries allow (usually every
            # rotation): each image batch runs one score, prior and reduction
            # program per class and block instead of one per 5,000 rows.
            rotation_block_size = _coarse_gaussian_gemm_cached_block_rows(
                n_rot,
                image_batch_size=int(image_batch_size),
                n_translations=int(n_trans),
                compact_pixel_count=int(square_score_count),
                budget_bytes=_coarse_gaussian_gemm_projected_transient_budget_bytes(),
            )
        logger.info(
            "Coarse pass on RELION's exact operands (coarse GEMMs): classes=%d rotations=%d "
            "current_size=%d physical_size=%d square_pixels=%d image_lanes=%d translations=%d stable_shapes=%s",
            n_classes,
            n_rot,
            score_size,
            coarse_gaussian_projector_output_size,
            square_score_count,
            int(image_batch_size),
            n_trans,
            bool(stable_fourier_window_shapes),
        )
    tree_rescore_fftw_order = None
    tree_rescore_translation_angles = None
    if tree_rescore_enabled:
        if n_classes != 1:
            raise ValueError(
                "the coarse-tree top-2 rescore (tree_rescore_max_margin) currently "
                "supports K=1 only",
            )
        if not return_class_best:
            raise ValueError(
                "the coarse-tree top-2 rescore (tree_rescore_max_margin) requires "
                "return_class_best=True",
            )
        if not use_relion_projector or not coarse_texture_interp:
            raise ValueError(
                "the coarse-tree top-2 rescore (tree_rescore_max_margin) requires "
                "the supplied RELION projector with texture interpolation",
            )
        if not half_spectrum_scoring:
            raise ValueError(
                "the coarse-tree top-2 rescore (tree_rescore_max_margin) requires "
                "half-spectrum scoring",
            )
        from recovar import cuda_backproject

        from relax.helpers.projection import relion_projector_half_to_texture_full
        from relax.relion.relion_ctf import _relion_exact_ctf_half_from_source_star
        from relax.sparse_pass2.sparse_pass2_bucket_io import _relion_translation_angles_f32

        if (
            jax.default_backend() != "gpu"
            or not cuda_backproject.custom_cuda_requested()
            or not cuda_backproject.cuda_available()
        ):
            raise RuntimeError(
                "the coarse-tree top-2 rescore (tree_rescore_max_margin) requires "
                "the custom CUDA backend",
            )
        coarse_gaussian_projector_full = jnp.asarray(
            relion_projector_half_to_texture_full(relion_projector_half[0])
            * jnp.asarray(_dense_projection_scale(image_shape), dtype=jnp.float32),
            dtype=jnp.complex64,
        )
        score_indices_np = (
            np.arange(n_half, dtype=np.int32)
            if window_spec.score_indices_np is None
            else window_spec.score_indices_np
        )
        tree_rescore_fftw_order = jnp.asarray(
            relion_fftw_order_for_square_score_window(
                image_shape,
                score_size,
                score_indices_np,
            ),
            dtype=jnp.int32,
        )
        tree_rescore_translation_angles = jnp.asarray(
            _relion_translation_angles_f32(
                translations_source,
                image_shape,
                angle_scale=relion_translation_angle_scale,
            ),
            dtype=jnp.float32,
        )
        logger.warning(
            "RELION coarse-tree top-2 rescore: max_margin=%g current_size=%d",
            tree_rescore_max_margin,
            score_size,
        )
    # --firstiter_cc on RELION's exact coarse operands: the tree rescore's per-image
    # FFT, RFLOAT CTF and corr_img operands, translated with RELION's sincosf for
    # every translation and scored by the coarse GEMMs
    # (_relion_coarse_normalized_cc_gemm_scores_jit).
    exact_cc_enabled = score_mode == "normalized_cc"
    exact_cc_score_indices = None
    exact_cc_translation_angles = None
    if exact_cc_enabled:
        from relax.relion.relion_ctf import _relion_exact_ctf_half_from_source_star
        from relax.sparse_pass2.sparse_pass2_bucket_io import _relion_translation_angles_f32

        exact_cc_score_indices = jnp.asarray(
            np.arange(n_half) if window_spec.score_indices_np is None else window_spec.score_indices_np,
            dtype=jnp.int32,
        )
        exact_cc_translation_angles = jnp.asarray(
            _relion_translation_angles_f32(
                translations_source,
                image_shape,
                angle_scale=relion_translation_angle_scale,
            ),
            dtype=jnp.float32,
        )
        logger.info(
            "RELION normalized-CC coarse pass on the exact operands: classes=%d current_size=%d "
            "score_pixels=%d translations=%d",
            n_classes,
            score_size,
            int(exact_cc_score_indices.shape[0]),
            n_trans,
        )
    track_class_second = return_class_second or tree_rescore_enabled
    if use_window:
        half_weights_windowed = window_spec.score_values(half_weights)

    n_blocks = (n_rot + rotation_block_size - 1) // rotation_block_size
    n_rot_padded = n_blocks * rotation_block_size
    if n_rot_padded > n_rot:
        pad_size = n_rot_padded - n_rot
        rotations_padded = np.concatenate(
            [
                rotations,
                np.tile(np.eye(3, dtype=np.asarray(rotations).dtype), (pad_size, 1, 1)),
            ],
            axis=0,
        )
    else:
        rotations_padded = rotations

    rotation_log_prior_padded = None
    if rotation_log_prior is not None:
        prior = np.asarray(rotation_log_prior, dtype=score_real_dtype)
        if prior.ndim == 1:
            if prior.shape != (n_rot,):
                raise ValueError(f"rotation_log_prior must have shape ({n_rot},), got {prior.shape}")
            prior = np.broadcast_to(prior[None, :], (n_classes, n_rot)).copy()
        elif prior.shape != (n_classes, n_rot):
            raise ValueError(
                f"rotation_log_prior must have shape ({n_rot},) or ({n_classes}, {n_rot}), got {prior.shape}",
            )
        if n_rot_padded > n_rot:
            rotation_log_prior_padded = np.pad(
                prior,
                ((0, 0), (0, n_rot_padded - n_rot)),
                mode="constant",
            )
        else:
            rotation_log_prior_padded = prior

    if translation_log_prior is not None:
        translation_log_prior = np.asarray(translation_log_prior, dtype=score_real_dtype)
        if translation_log_prior.ndim == 1:
            if translation_log_prior.shape != (n_trans,):
                raise ValueError(
                    f"translation_log_prior must have shape ({n_trans},), got {translation_log_prior.shape}"
                )
        elif translation_log_prior.ndim == 2:
            if translation_log_prior.shape != (n_images, n_trans):
                raise ValueError(
                    "translation_log_prior must have shape "
                    f"({n_images}, {n_trans}) when image-specific, got {translation_log_prior.shape}",
                )
        else:
            raise ValueError(f"translation_log_prior must be 1D or 2D, got {translation_log_prior.ndim} dimensions")

    noise_variance_half = noise_utils.to_batched_half_pixel_noise(noise_variance, image_shape).squeeze()
    if noise_variance_half.ndim == 2 and optics_group_ids is None:
        raise ValueError("a per-optics-group noise table needs optics_group_ids")
    # Each batch gathers its images' group spectra on the host: an eager device gather would
    # compile a program for every batch size (1339 compiles in a several-shape VDAM run).
    noise_table_host = np.asarray(noise_variance_half) if noise_variance_half.ndim == 2 else None
    image_groups_host = None if noise_table_host is None else np.asarray(optics_group_ids, dtype=np.int32)
    coarse_gaussian_shifted_corrected = None
    coarse_gaussian_unshifted_corrected = None
    coarse_gaussian_translation_angles = None
    coarse_gaussian_pixel_weight = None
    coarse_gaussian_initial_diff2 = None

    # The texture projector naturally produces a centered current-size crop.
    # Ask it only for the rows consumed by the scorer instead of scattering the
    # crop into a full image and immediately gathering the same rows again.
    # This is an exact index remapping and avoids a large transient scatter for
    # global rotation blocks.
    projector_compact_indices_np = None
    projector_output_size = None
    if use_relion_projector and coarse_texture_interp:
        if exact_gaussian:
            projector_compact_indices_np = coarse_gaussian_score_indices_np
            projector_output_size = coarse_gaussian_projector_output_size
        elif use_window:
            projector_compact_indices_np = window_spec.score_indices_np
            projector_output_size = score_size
    projector_returns_compact = projector_compact_indices_np is not None

    coarse_rotated_radius = _coarse_rotated_radius_enabled(
        default=bool(use_relion_projector and coarse_texture_interp and projector_returns_compact),
    )
    if coarse_rotated_radius and not (
        use_relion_projector and coarse_texture_interp and projector_returns_compact
    ):
        raise ValueError("rotated coarse radius requires the compact RELION texture projector")

    def _project_relion_compact_score_rows(
        class_index,
        rots_b,
        *,
        return_abs2: bool,
    ):
        """Project the exact compact rows shared by direct and GEMM scoring."""

        projected, projected_abs2 = _compute_relion_projector_projections_block(
            relion_projector_half[class_index],
            rots_b,
            image_shape,
            r_max=int(relion_projector_r_max),
            padding_factor=int(projection_padding_factor),
            return_abs2=return_abs2,
            centered_rows=True,
            dense_scale=True,
            projector_output_size=int(projector_output_size),
            # Keep the already-host-resident table on the host so validation
            # cannot materialize its JAX mirror once per score block.
            pixel_indices=projector_compact_indices_np,
            relion_texture_interp=True,
            relion_kernel="coarse",
            # Certificate and exact scorer share the canonical rotated
            # float32 radius. Explicit legacy diagnostics may retain the
            # source-pixel disk without changing projection storage.
            mask_current_image_disk=not coarse_rotated_radius,
            image_r_max=(
                jnp.asarray(score_size // 2, dtype=jnp.int32)
                if coarse_rotated_radius else None
            ),
            current_image_mask_size=(
                jnp.asarray(score_size, dtype=jnp.int32)
                if stable_fourier_window_shapes
                else None
            ),
        )
        return projected, projected_abs2

    def _project_block(class_index, rots_b):
        if projector_returns_compact:
            return _project_relion_compact_score_rows(class_index, rots_b, return_abs2=True)
        projector_kwargs = {}
        if current_size is not None:
            projector_kwargs["projector_output_size"] = int(current_size)
        return _compute_relion_projector_projections_block(
            relion_projector_half[class_index],
            rots_b,
            image_shape,
            r_max=int(relion_projector_r_max),
            padding_factor=int(projection_padding_factor),
            centered_rows=True,
            dense_scale=True,
            relion_texture_interp=True,
            relion_kernel="coarse",
            **projector_kwargs,
        )

    # The references and rotation blocks are fixed for the whole pass, so a
    # block's projections are the same for every image batch; they are kept
    # (up to a byte cap) instead of being recomputed per batch and per scoring
    # sweep, which at K4 100k/256 was 11 s of the pass per iteration.
    projection_memo: dict = {}
    projection_memo_bytes = [0]

    def _project_block_once(class_index, rots_b, *, rotation_start):
        key = (int(class_index), int(rotation_start), int(rots_b.shape[0]))
        cached = projection_memo.get(key)
        if cached is not None:
            return cached
        projected = _project_block(class_index, rots_b)
        block_bytes = sum(int(value.size) * int(value.dtype.itemsize) for value in projected)
        if projection_memo_bytes[0] + block_bytes <= _PASS1_PROJECTION_MEMO_MAX_BYTES:
            projection_memo[key] = projected
            projection_memo_bytes[0] += block_bytes
        return projected

    coarse_gaussian_gemm_projection_cache = None
    if coarse_gaussian_gemm_projection_cache_plan is not None:

        def _project_coarse_gemm_cache_build_block(table_index, start, stop):
            projected_reference, projected_reference_abs2 = (
                _project_relion_compact_score_rows(
                    table_index,
                    rotations[start:stop],
                    return_abs2=False,
                )
            )
            if projected_reference_abs2 is not None:
                raise RuntimeError(
                    "coarse GEMM cache build unexpectedly materialized abs2",
                )
            return projected_reference

        coarse_gaussian_gemm_projection_cache = (
            build_projection_cache(
                coarse_gaussian_gemm_projection_cache_plan,
                _project_coarse_gemm_cache_build_block,
            )
        )
        logger.warning(
            "Coarse GEMM C64 projection cache built: "
            "shape=%s chunks=%d conservative_peak_bytes=%d budget_bytes=%d",
            coarse_gaussian_gemm_projection_cache_plan.cache_shape,
            coarse_gaussian_gemm_projection_cache_plan.chunk_count_per_table,
            coarse_gaussian_gemm_projection_cache_plan.predicted_peak_bytes,
            coarse_gaussian_gemm_projection_cache_plan.budget_bytes,
        )

    # The class and rotation prior terms of the pass-1 program, built at its first batch.
    pass1_prior_terms = None
    exact_rotation_prior = None

    sig_rot_any = np.zeros((n_classes, n_rot), dtype=bool)
    n_sig_all = np.empty(n_images, dtype=np.int32)
    cutoff_count_all = np.empty(n_images, dtype=np.int32)
    hard_assignment = np.empty(n_images, dtype=np.int32)
    class_assignment = np.empty(n_images, dtype=np.int32)
    significant_sample_indices = [[None] * n_images for _ in range(n_classes)] if collect_significance else None
    normalization_log_z = np.empty(n_images, dtype=np.float64)
    normalization_log_evidence = np.empty(n_images, dtype=np.float64)
    log_evidence = np.empty(n_images, dtype=score_real_dtype)
    best_log_score = np.empty(n_images, dtype=score_real_dtype)
    max_posterior = np.empty(n_images, dtype=score_real_dtype)
    relion_f32_sum_weight = (
        np.empty(n_images, dtype=np.float32)
        if (relion_f32_coarse_support_enabled or return_relion_f32_normalization) and collect_significance
        else None
    )
    relion_f32_max_posterior = (
        np.empty(n_images, dtype=np.float32) if return_relion_f32_normalization else None
    )
    class_log_evidence = np.empty((n_classes, n_images), dtype=np.float64)
    class_best_log_score = (
        np.empty((n_classes, n_images), dtype=score_real_dtype) if return_class_best else None
    )
    class_second_best_log_score = (
        np.empty((n_classes, n_images), dtype=score_real_dtype) if return_class_second else None
    )
    # Diagnostic-only native scores before the large, class-common image
    # normalization offset.  The offset is useful for absolute log evidence,
    # but adding it before a float32 cast can erase class and pose margins.
    class_best_offset_free_log_score = (
        np.empty((n_classes, n_images), dtype=score_real_dtype) if return_class_best else None
    )
    class_second_best_offset_free_log_score = (
        np.empty((n_classes, n_images), dtype=score_real_dtype) if return_class_second else None
    )
    class_hard_assignment = (
        np.empty((n_classes, n_images), dtype=np.int32) if return_class_best else None
    )
    class_second_hard_assignment = (
        np.empty((n_classes, n_images), dtype=np.int32) if return_class_second else None
    )
    tree_rescore_examined = 0
    tree_rescore_ambiguous = 0
    tree_rescore_winner_changes = 0
    tree_rescore_exact_ties = 0

    start_idx = 0
    image_indices = np.arange(n_images)
    from relax.helpers.batch_fetch import iter_indexed_batches, prefetched_batches

    _coarse_batch_starts = []
    _coarse_loop_t0 = time.time()
    def _publish_batch(batch):
        """Read one batch's pass-1 outputs back and store them in the per-image results.

        The batch loop calls this for a batch once the next batch's operands are on the device, before
        that batch's score program: the device scores this batch while the host prepares the next (a
        dump batch publishes at once). ``batch`` holds the batch's device outputs and its place in the
        image order.
        """

        nonlocal sig_rot_any
        if collect_significance:
            if relion_f32_coarse_support_enabled:
                relion_f32_sum_weight[batch.start_idx:batch.end_idx] = np.asarray(
                    batch.sum_weight,
                    dtype=np.float32,
                )[:batch.actual_batch_size]
                batch_pmax_host = np.asarray(batch.pmax, dtype=np.float32)[:batch.actual_batch_size]
                if return_relion_f32_normalization:
                    relion_f32_max_posterior[batch.start_idx:batch.end_idx] = batch_pmax_host
            device_significance_batch = (
                coarse_significance_device_enabled
                and not batch.debug_dump_enabled
            )
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
                    n_classes=n_classes,
                    actual_batch_size=batch.actual_batch_size,
                    n_coarse_rot=n_rot,
                    n_coarse_trans=n_trans,
                    batch_n_sig=batch.n_sig if n_classes == 1 else None,
                )
                for class_index, (
                    batch_device_counts,
                    batch_device_polarity,
                    batch_device_ids,
                    _batch_device_rot_any,
                ) in enumerate(class_results):
                    device_significance_counts[class_index].append(batch_device_counts)
                    device_significance_polarity[class_index].append(batch_device_polarity)
                    device_significance_ids[class_index].append(batch_device_ids)
                    device_significance_starts[class_index].append(int(batch.start_idx))
            else:
                batch_sig_mask_np = np.array(batch.sig_mask, dtype=bool, copy=True)
            sig_rot_any |= np.asarray(
                _any_over_leading_rows(batch.sig_rot_mask, batch.actual_batch_size),
                dtype=bool,
            ).reshape(n_classes, n_rot)
            n_sig_all[batch.start_idx:batch.end_idx] = np.asarray(batch.n_sig, dtype=np.int32)[:batch.actual_batch_size]
            cutoff_count_all[batch.start_idx:batch.end_idx] = np.asarray(
                batch.cutoff_count,
                dtype=np.int32,
            )[:batch.actual_batch_size]
        else:
            batch_sig_mask_np = None
            n_sig_all[batch.start_idx:batch.end_idx] = 0
            cutoff_count_all[batch.start_idx:batch.end_idx] = 0

        hard_assignment[batch.start_idx:batch.end_idx] = np.asarray(
            batch.best_argmax,
            dtype=np.int32,
        )[:batch.actual_batch_size]
        class_assignment[batch.start_idx:batch.end_idx] = np.asarray(
            batch.best_class,
            dtype=np.int32,
        )[:batch.actual_batch_size]

        # The exact scorers score without an image-energy offset.
        log_score_offset = np.zeros(batch.batch_size, dtype=np.float64)
        global_log_z_np = np.asarray(batch.global_log_z, dtype=np.float64)
        best_score_np = np.asarray(batch.best_score, dtype=np.float64)
        output_slice = slice(0, batch.actual_batch_size)
        normalization_log_z[batch.start_idx:batch.end_idx] = global_log_z_np[output_slice]
        normalization_log_evidence[batch.start_idx:batch.end_idx] = (
            global_log_z_np[output_slice] + log_score_offset[output_slice]
        )
        log_evidence[batch.start_idx:batch.end_idx] = normalization_log_evidence[batch.start_idx:batch.end_idx].astype(score_real_dtype)
        best_log_score[batch.start_idx:batch.end_idx] = (
            best_score_np[output_slice] + log_score_offset[output_slice]
        ).astype(score_real_dtype)
        if relion_f32_coarse_support_enabled and collect_significance:
            max_posterior[batch.start_idx:batch.end_idx] = batch_pmax_host
        else:
            max_posterior[batch.start_idx:batch.end_idx] = np.exp(
                best_score_np[output_slice] - global_log_z_np[output_slice]
            ).astype(score_real_dtype)
        for class_index, class_log_z in enumerate(batch.class_log_z_values):
            class_log_evidence[class_index, batch.start_idx:batch.end_idx] = (
                np.asarray(class_log_z, dtype=np.float64)[output_slice]
                + log_score_offset[output_slice]
            )
        if return_class_best:
            for class_index in range(n_classes):
                offset_free, absolute = _capture_offset_free_and_absolute_float32_scores(
                    batch.class_best_scores[class_index],
                    log_score_offset,
                )
                class_best_offset_free_log_score[class_index, batch.start_idx:batch.end_idx] = offset_free[output_slice]
                class_best_log_score[class_index, batch.start_idx:batch.end_idx] = absolute[output_slice]
                class_hard_assignment[class_index, batch.start_idx:batch.end_idx] = np.asarray(
                    batch.class_best_argmaxes[class_index][output_slice],
                    dtype=np.int32,
                )
        if return_class_second:
            for class_index in range(n_classes):
                offset_free, absolute = _capture_offset_free_and_absolute_float32_scores(
                    batch.class_second_best_scores[class_index],
                    log_score_offset,
                )
                class_second_best_offset_free_log_score[class_index, batch.start_idx:batch.end_idx] = offset_free[output_slice]
                class_second_best_log_score[class_index, batch.start_idx:batch.end_idx] = absolute[output_slice]
                class_second_hard_assignment[class_index, batch.start_idx:batch.end_idx] = np.asarray(
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
                experiment_dataset=experiment_dataset,
                indices=batch.indices,
                n_classes=n_classes,
                rotations=rotations,
                translations=translations,
                # ``batch_weights`` is the class-major concatenation of each class's
                # weights on every route.
                class_weight_mats=[
                    np.asarray(
                        batch.weights.reshape(
                            batch.batch_size,
                            n_classes,
                            n_rot * n_trans,
                        )[:, class_index, :],
                        dtype=np.float64,
                    )
                    for class_index in range(n_classes)
                ],
                batch_sig_mask=batch_sig_mask_np,
                batch_n_sig=np.asarray(batch.n_sig, dtype=np.int64),
                hard_assignment_batch=np.asarray(batch.best_argmax, dtype=np.int64),
                class_assignment_batch=np.asarray(batch.best_class, dtype=np.int64),
                global_log_z=global_log_z_np,
                class_log_z_values=batch.class_log_z_values,
                best_score=best_score_np,
                max_posterior=max_posterior[batch.start_idx:batch.end_idx],
                rotation_log_prior_padded=rotation_log_prior_padded,
                batch_translation_log_prior=batch.translation_log_prior,
                class_log_priors=class_log_priors_np,
                current_size=current_size,
                adaptive_fraction=adaptive_fraction,
                max_significants=max_significants,
                target_local_positions=target_local_positions_for_dump,
                target_scores_pre_prior_per_class=target_scores_pre_prior_per_class,
                target_scores_with_prior_per_class=target_scores_with_prior_per_class,
                # RELION's exact coarse operands: the Gaussian GEMM's, or the CC pass's.
                coarse_gaussian_shifted_corrected=(
                    batch.exact_cc_shifted if exact_cc_enabled else batch.coarse_gaussian_shifted_corrected
                ),
                coarse_gaussian_unshifted_corrected=(
                    batch.exact_cc_operands.windowed_unshifted
                    if exact_cc_enabled
                    else batch.coarse_gaussian_unshifted_corrected
                ),
                coarse_gaussian_pixel_weight=(
                    batch.exact_cc_pixel_weight if exact_cc_enabled else batch.coarse_gaussian_pixel_weight
                ),
                coarse_gaussian_initial_diff2=None if exact_cc_enabled else batch.coarse_gaussian_initial_diff2,
                coarse_gaussian_score_indices=(
                    exact_cc_score_indices if exact_cc_enabled else coarse_gaussian_score_indices
                ),
                translation_phase_source=translations_source,
                relion_projector_half=relion_projector_half,
                relion_projector_r_max=relion_projector_r_max,
                projection_padding_factor=projection_padding_factor,
                relion_f32_sum_weight=(
                    batch.sum_weight if relion_f32_coarse_support_enabled else None
                ),
                relion_f32_significant_weight=(
                    batch.significant_weight
                    if relion_f32_coarse_support_enabled
                    else None
                ),
                relion_f32_cutoff_count=(
                    batch.cutoff_count if relion_f32_coarse_support_enabled else None
                ),
                score_capture_mode=score_capture_mode,
                debug_iteration=debug_iteration,
            )

        if collect_significance and not device_significance_batch:
            samples_per_class = n_rot * n_trans
            for local_idx, global_idx in enumerate(batch.indices):
                for class_index in range(n_classes):
                    c0 = class_index * samples_per_class
                    c1 = c0 + samples_per_class
                    mask = batch_sig_mask_np[local_idx, c0:c1]
                    significant_sample_indices[class_index][global_idx] = compact_significant_sample_indices_from_mask(
                        mask,
                    )

    pending_batch = None
    from relax.cuda.kernels import deferred_relion_preprocess_checks

    # The image preprocess kernel's finite check is read at the end of the loop: read at each call, it
    # waits for the previous batch's score program inside the next batch's preparation.
    with deferred_relion_preprocess_checks("pass 1") as preprocess_checks, prefetched_batches(
        iter_indexed_batches(experiment_dataset, image_indices, image_batch_size)
    ) as batches:
        for batch_data, _, _, ctf_params, _, _, indices in batches:
            _coarse_batch_starts.append(time.time())
            actual_batch_size = len(indices)
            end_idx = start_idx + actual_batch_size
            preprocess_checks.at(
                f"batch {len(_coarse_batch_starts) - 1} (images {start_idx}-{end_idx - 1} of the pass, "
                f"dataset images {int(indices[0])}-{int(indices[-1])})"
            )
            # The outputs a batch has only on some routes; _publish_batch receives every name.
            batch_pmax = batch_weights = batch_sig_mask = batch_sig_rot_mask = batch_n_sig = batch_cutoff_count = None
            _batch_sum_weight = _batch_significant_weight = None
            exact_cc_operands = exact_cc_pixel_weight = exact_cc_shifted = None
            (
                relion_cuda_preprocess,
                integer_pre_shifts,
                batch_corr_np,
                batch_scale_np,
                relion_preprocess_kwargs,
            ) = prepare_batch_preprocess_operands(
                experiment_dataset,
                batch_data,
                indices,
                image_corrections=image_corrections,
                scale_corrections=scale_corrections,
                image_pre_shifts=image_pre_shifts,
                dtype=score_real_dtype,
            )
            if pad_final_image_batch and actual_batch_size < int(image_batch_size):
                (
                    batch_data,
                    ctf_params,
                    integer_pre_shifts,
                    batch_corr_np,
                    batch_scale_np,
                    relion_preprocess_kwargs,
                ) = _pad_significance_preprocess_inputs(
                    batch_data,
                    ctf_params,
                    integer_pre_shifts,
                    batch_corr_np,
                    batch_scale_np,
                    relion_preprocess_kwargs,
                    target_size=int(image_batch_size),
                )
            batch_size = int(batch_data.shape[0])
            # The batch's dataset images, repeat-padded as batch_data is.
            batch_image_indices = _repeat_pad_batch_axis(np.asarray(indices), batch_size)
            # Each image's own optics-group spectrum; the one shared spectrum otherwise.
            batch_noise_half = noise_variance_half
            if noise_table_host is not None:
                batch_groups = image_groups_host[np.asarray(indices, dtype=np.int64)]
                batch_noise_half = jnp.asarray(_repeat_pad_batch_axis(noise_table_host[batch_groups], batch_size))
            real_space_pre_shift_applied = integer_pre_shifts is not None
            if real_space_pre_shift_applied and not relion_cuda_preprocess:
                batch_data = apply_relion_integer_pre_shifts(batch_data, integer_pre_shifts)
            batch_data = jnp.asarray(batch_data)
            if translation_log_prior is None:
                batch_translation_log_prior = None
            elif translation_log_prior.ndim == 1:
                batch_translation_log_prior = jnp.asarray(translation_log_prior)
            else:
                batch_translation_log_prior_np = np.asarray(translation_log_prior[start_idx:end_idx])
                if batch_size > actual_batch_size:
                    batch_translation_log_prior_np = _repeat_pad_batch_axis(
                        batch_translation_log_prior_np,
                        batch_size,
                    )
                batch_translation_log_prior = jnp.asarray(batch_translation_log_prior_np)

            if not relion_cuda_preprocess or relion_preprocess_kwargs is None:
                raise ValueError("pass 1 scores RELION's exact coarse operands and needs RELION's CUDA image preprocessing")
            if exact_cc_enabled:
                from relax.cuda import kernels as em_cuda_kernels

                exact_cc_processed = _process_relion_exact_coarse_half_image(
                    experiment_dataset,
                    batch_data,
                    score_with_masked_images,
                    relion_preprocess_kwargs=relion_preprocess_kwargs,
                    image_indices=batch_image_indices,
                )
                exact_cc_phase_factors = None
                if image_pre_shifts is not None and not real_space_pre_shift_applied:
                    exact_cc_shifts = np.asarray(image_pre_shifts)[np.asarray(indices)]
                    exact_cc_phase_factors = tiled_half_image_phase_factors(
                        image_shape,
                        jnp.asarray(_repeat_pad_batch_axis(exact_cc_shifts, batch_size)),
                        1,
                    )
                # The padded rows of a short last batch repeat its first image's CTF row.
                exact_cc_operands = assemble_relion_cc_coarse_operands(
                    exact_cc_processed,
                    _relion_exact_ctf_half_from_source_star(
                        experiment_dataset,
                        _repeat_pad_batch_axis(np.asarray(indices), batch_size),
                        image_shape,
                    ),
                    _relion_cc_inverse_power_from_processed(
                        exact_cc_processed,
                        window_indices if use_window else None,
                        (half_weights_windowed if use_window else half_weights) if cc_gaussian_support else None,
                    ),
                    jnp.asarray(batch_scale_np, dtype=jnp.float32),
                    phase_factors=exact_cc_phase_factors,
                    window_indices=window_indices if use_window else None,
                    scale_corrections_enabled=scale_corrections is not None,
                )
                exact_cc_shifted = em_cuda_kernels.relion_translate_score_f32(
                    exact_cc_operands.windowed_unshifted,
                    exact_cc_translation_angles,
                    exact_cc_score_indices,
                    image_shape,
                ).reshape(batch_size, n_trans, -1)
                exact_cc_pixel_weight = exact_cc_operands.windowed_corr_img * (
                    half_weights_windowed if use_window else half_weights
                )
                if tree_rescore_enabled:
                    # The bounded top-two rescore uses these same exact CUDA
                    # operands. Reuse the per-image FFT/CTF assembly instead of
                    # preparing a second copy before the GEMM coarse pass.
                    tree_rescore_unshifted_data = exact_cc_operands.windowed_unshifted
                    tree_rescore_corr_img_data = exact_cc_operands.windowed_corr_img

            if exact_gaussian:
                processed_direct = _process_relion_exact_coarse_half_image(
                    experiment_dataset,
                    batch_data,
                    score_with_masked_images,
                    relion_preprocess_kwargs=relion_preprocess_kwargs,
                    image_indices=batch_image_indices,
                )
                if nyquist_column_counting != "relion":
                    # The score weights are zero on these pixels; zeroing them here also takes them
                    # out of powerClass's high-shell image power (the diff2 constant).
                    processed_direct = jnp.where(
                        jnp.asarray(redundant_nyquist_column_pixels(image_shape))[None, :],
                        jnp.zeros((), dtype=processed_direct.dtype),
                        processed_direct,
                    )
                exact_operands = _assemble_relion_exact_coarse_gaussian_operands(
                    experiment_dataset,
                    processed_direct,
                    indices,
                    use_float64_scoring=use_float64_scoring,
                    batch_scale_np=batch_scale_np,
                    actual_batch_size=actual_batch_size,
                    batch_size=batch_size,
                    score_indices=coarse_gaussian_score_indices,
                    score_indices_np=coarse_gaussian_score_indices_np,
                    score_active_mask=coarse_gaussian_score_active_mask,
                    translations_source=translations_source,
                    relion_translation_angle_scale=relion_translation_angle_scale,
                    image_shape=image_shape,
                    noise_variance_half=batch_noise_half,
                    scale_corrections_enabled=scale_corrections is not None,
                    half_weights=half_weights,
                    powerclass=coarse_gaussian_powerclass,
                    current_size=(
                        coarse_gaussian_square_layout.physical_current_size
                        if stable_fourier_window_shapes
                        else current_size
                    ),
                    runtime_current_size=(
                        jnp.asarray(score_size, dtype=jnp.int32) if stable_fourier_window_shapes else None
                    ),
                    coarse_kernel_window=coarse_kernel_window,
                    coarse_kernel_r_max=None if coarse_kernel_window is None else int(relion_projector_r_max),
                )
                coarse_gaussian_shifted_corrected = exact_operands.shifted_corrected
                coarse_gaussian_pixel_weight = exact_operands.pixel_weight
                coarse_gaussian_unshifted_corrected = exact_operands.unshifted_corrected
                coarse_gaussian_initial_diff2 = exact_operands.initial_diff2
                coarse_gaussian_translation_angles = exact_operands.translation_angles

            # Identify per-batch dump target rows so we can record raw scores
            # (pre-prior) for each target image inside the per-class block loop.
            # This enables direct diff against RELION's exp_Mweight_diff2
            # without needing the full (batch, n_classes, n_rot*n_trans) cache.
            debug_dump_enabled = collect_significance and _significance_debug_dump_matches(
                current_size=current_size,
                debug_iteration=debug_iteration,
            )
            dump_target_local_positions = None
            if debug_dump_enabled:
                _dump_targets = parse_env_int_set("RELAX_SIGNIFICANCE_DUMP_ORIGINAL_INDICES")
                if _dump_targets:
                    _local_for_dump = np.asarray(indices, dtype=np.int64)
                    _orig = original_image_indices(experiment_dataset, _local_for_dump)
                    _positions = np.flatnonzero(np.isin(_orig, np.fromiter(_dump_targets, dtype=np.int64)))
                    if _positions.size:
                        dump_target_local_positions = _positions.astype(np.int64)
            # Per-class collectors for raw (pre-prior) score blocks at target rows.
            # Shape after concat per class: (n_targets, n_rot, n_trans)
            dump_target_pre_prior_blocks_per_class = (
                [[] for _ in range(n_classes)] if dump_target_local_positions is not None else None
            )
            dump_target_with_prior_blocks_per_class = (
                [[] for _ in range(n_classes)] if dump_target_local_positions is not None else None
            )
            neg_inf_f, zeros_f64, zeros_i32 = _pass1_batch_constants(batch_size, bool(jax.config.jax_enable_x64))
            global_max = neg_inf_f
            global_sum = zeros_f64
            class_max_values = []
            class_sum_values = []
            best_score_batch = neg_inf_f
            best_argmax_batch = zeros_i32
            best_class_batch = zeros_i32
            class_best_scores = [neg_inf_f] * n_classes if return_class_best else None
            class_best_argmaxes = [zeros_i32] * n_classes if return_class_best else None
            class_second_best_scores = [neg_inf_f] * n_classes if track_class_second else None
            class_second_best_argmaxes = [zeros_i32] * n_classes if track_class_second else None
            # RELION's CUDA coarse kernel forms ``pdf_orientation + pdf_offset +
            # min_diff2 - diff2`` left to right (cuda_kernel_weights_exponent_coarse).
            # Adding the priors to the absolute scores and the min_diff2 offset
            # afterwards can tie poses that RELION separates by one ULP, so the
            # support pass keeps the pre-prior scores.
            relion_exact_coarse_weight_order = bool(relion_f32_coarse_support_enabled and n_classes == 1)
            if not relion_f32_coarse_support_enabled:
                relion_raw_score_max = None
            else:
                relion_raw_score_max = jnp.full(
                    batch_size,
                    -jnp.inf,
                    dtype=jnp.float32,
                )

            # Pass 1 of the coarse GEMM scorers is one program per batch
            # (_coarse_pass1_blocks): over the cached projections in one call, or
            # one call per class and rotation block on its projection when the
            # cache does not fit and for --firstiter_cc. The program also returns
            # the RELAX_SIGNIFICANCE_DUMP_* targets' scores and the class runner-up.
            batched_support_values = None
            if pass1_prior_terms is None:
                pass1_prior_terms = tuple(
                    tuple(
                        (
                            jnp.asarray(class_log_priors_np[class_index], dtype=jnp.float32),
                            None
                            if rotation_log_prior_padded is None
                            else jnp.asarray(
                                rotation_log_prior_padded[class_index, r0 : r0 + rotation_block_size]
                            ),
                        )
                        for r0 in range(0, n_rot_padded, rotation_block_size)
                    )
                    for class_index in range(n_classes)
                )
            pass1_blocks = tuple(
                (class_index, r0, min(rotation_block_size, n_rot - r0), int(rotation_block_size))
                for class_index in range(n_classes)
                for r0 in range(0, n_rot, rotation_block_size)
            )
            if exact_cc_enabled:
                pass1_operands = (exact_cc_shifted, jnp.asarray(exact_cc_pixel_weight, dtype=jnp.float32), None)
            else:
                pass1_operands = (
                    jnp.asarray(coarse_gaussian_shifted_corrected, dtype=jnp.complex64),
                    jnp.asarray(coarse_gaussian_pixel_weight, dtype=jnp.float32),
                    jnp.asarray(coarse_gaussian_initial_diff2, dtype=jnp.float32),
                )
            pass1_static = dict(
                n_trans=int(n_trans),
                image_shape=tuple(int(value) for value in image_shape),
                volume_shape=tuple(int(value) for value in volume_shape),
                float64=_coarse_gemm_float64_requested(),
                score_kind="normalized_cc" if exact_cc_enabled else "gaussian",
                exact_weight_order=relion_exact_coarse_weight_order,
                return_class_best=bool(return_class_best),
                track_class_second=bool(track_class_second),
                return_values=bool(collect_significance),
            )
            pass1_dump_rows = (
                None
                if dump_target_local_positions is None
                else jnp.asarray(dump_target_local_positions, dtype=jnp.int32)
            )
            if pending_batch is not None:
                _publish_batch(pending_batch)
                pending_batch = None
            pass1_state = _pass1_initial_state((neg_inf_f, zeros_f64, zeros_i32), n_classes)
            if coarse_gaussian_gemm_projection_cache is not None and not exact_cc_enabled:
                pass1_state, pass1_values, pass1_dumps = _coarse_pass1_blocks(
                    pass1_state,
                    coarse_gaussian_gemm_projection_cache,
                    *pass1_operands,
                    actual_batch_size,
                    tuple(
                        pass1_prior_terms[class_index][r0 // rotation_block_size]
                        for class_index, r0, _, _ in pass1_blocks
                    ),
                    batch_translation_log_prior,
                    pass1_dump_rows,
                    blocks=pass1_blocks,
                    **pass1_static,
                )
            else:
                pass1_values = []
                pass1_dumps = []
                for class_index, r0, rows, block_rows in pass1_blocks:
                    rots_b = rotations_padded[r0 : r0 + block_rows]
                    if exact_cc_enabled:
                        reference, _ = _project_block_once(class_index, rots_b, rotation_start=r0)
                    else:
                        reference, _ = _project_relion_compact_score_rows(class_index, rots_b, return_abs2=True)
                    block_state, block_values, block_dump = _coarse_pass1_block(
                        _class_block_state(pass1_state, class_index),
                        reference,
                        *pass1_operands,
                        actual_batch_size,
                        pass1_prior_terms[class_index][r0 // rotation_block_size],
                        batch_translation_log_prior,
                        jnp.int32(class_index),
                        jnp.int32(r0),
                        pass1_dump_rows,
                        rows=rows,
                        block_rows=block_rows,
                        **pass1_static,
                    )
                    pass1_state = _merge_class_block_state(pass1_state, block_state, class_index)
                    pass1_values.append(block_values)
                    pass1_dumps.append(block_dump)
            if collect_significance:
                batched_support_values = jnp.concatenate(pass1_values, axis=1)
            if pass1_dump_rows is not None:
                for (class_index, _, _, _), (pre_prior, with_prior) in zip(pass1_blocks, pass1_dumps, strict=True):
                    dump_target_pre_prior_blocks_per_class[class_index].append(np.asarray(pre_prior, dtype=np.float64))
                    dump_target_with_prior_blocks_per_class[class_index].append(
                        np.asarray(with_prior, dtype=np.float64)
                    )
            (
                (global_max, global_sum),
                (class_max_tuple, class_sum_tuple),
                (best_score_batch, best_argmax_batch, best_class_batch),
                (class_best_tuple, class_best_argmax_tuple, class_second_tuple, class_second_argmax_tuple),
                pass1_raw_score_max,
            ) = pass1_state
            if relion_raw_score_max is not None:
                relion_raw_score_max = pass1_raw_score_max
            class_max_values = list(class_max_tuple)
            class_sum_values = list(class_sum_tuple)
            if return_class_best:
                class_best_scores = list(class_best_tuple)
                class_best_argmaxes = list(class_best_argmax_tuple)
            if track_class_second:
                class_second_best_scores = list(class_second_tuple)
                class_second_best_argmaxes = list(class_second_argmax_tuple)

            if tree_rescore_enabled:
                tree_score_dtype = np.float32
                best_scores_np = np.asarray(class_best_scores[0], dtype=tree_score_dtype)
                second_scores_np = np.asarray(class_second_best_scores[0], dtype=tree_score_dtype)
                score_margins = best_scores_np - second_scores_np
                ambiguous_rows = np.flatnonzero(
                    np.isfinite(score_margins) & (score_margins <= tree_rescore_max_margin)
                ).astype(np.int32)
                tree_rescore_examined += int(batch_size)
                tree_rescore_ambiguous += int(ambiguous_rows.size)
                if ambiguous_rows.size:
                    best_pose_np = np.asarray(class_best_argmaxes[0], dtype=np.int32)[ambiguous_rows]
                    second_pose_np = np.asarray(class_second_best_argmaxes[0], dtype=np.int32)[
                        ambiguous_rows
                    ]
                    candidate_pose_ids = np.sort(
                        np.stack([best_pose_np, second_pose_np], axis=1),
                        axis=1,
                    )
                    candidate_rotation_ids = candidate_pose_ids // n_trans
                    candidate_translation_ids = candidate_pose_ids % n_trans
                    candidate_rotations = jnp.asarray(
                        rotations[candidate_rotation_ids.reshape(-1)],
                        dtype=jnp.float32,
                    ).reshape(
                        ambiguous_rows.size,
                        2,
                        3,
                        3,
                    )
                    unshifted_candidates = jnp.broadcast_to(
                        tree_rescore_unshifted_data[
                            jnp.asarray(ambiguous_rows, dtype=jnp.int32), None, :
                        ],
                        (
                            ambiguous_rows.size,
                            2,
                            tree_rescore_unshifted_data.shape[-1],
                        ),
                    )
                    candidate_translation_angles = tree_rescore_translation_angles[
                        jnp.asarray(candidate_translation_ids, dtype=jnp.int32)
                    ]
                    score_weight_candidates = jnp.broadcast_to(
                        tree_rescore_corr_img_data[
                            jnp.asarray(ambiguous_rows, dtype=jnp.int32), None, :
                        ],
                        unshifted_candidates.shape,
                    )
                    rescored_candidates = _relion_coarse_normalized_cc_rescore(
                        unshifted_candidates,
                        score_weight_candidates,
                        None,
                        half_weights_windowed if use_window else half_weights,
                        tree_rescore_fftw_order,
                        projector_full=coarse_gaussian_projector_full,
                        rotation_matrices=candidate_rotations,
                        translation_angles=candidate_translation_angles,
                        current_size=score_size,
                        padding_factor=projection_padding_factor,
                        projector_max_r=relion_projector_r_max,
                        numerator_weight_candidates=score_weight_candidates,
                    )
                    rescored_scores_np = np.asarray(rescored_candidates, dtype=tree_score_dtype)
                    rescored_winner_slot, exact_ties = _select_relion_coarse_rescore_winner_slots(
                        rescored_scores_np,
                        candidate_pose_ids,
                        n_trans=n_trans,
                        healpix_order=coarse_healpix_order,
                        coarse_rotation_ids=coarse_rotation_ids,
                        score_dtype=tree_score_dtype,
                        **({"symmetry_label": symmetry_label} if symmetry_label != "C1" else {}),
                    )
                    _maybe_dump_tree_rescore_batch(
                        experiment_dataset=experiment_dataset,
                        indices=indices,
                        ambiguous_rows=ambiguous_rows,
                        candidate_pose_ids=candidate_pose_ids,
                        original_best_pose=best_pose_np,
                        original_best_score=best_scores_np[ambiguous_rows],
                        original_second_pose=second_pose_np,
                        original_second_score=second_scores_np[ambiguous_rows],
                        rescored_scores=rescored_scores_np,
                        rescored_winner_slot=rescored_winner_slot,
                        shifted_candidates=unshifted_candidates,
                        score_weight_candidates=score_weight_candidates,
                        numerator_weight_candidates=score_weight_candidates,
                        rotation_matrices=candidate_rotations,
                        translation_angles=candidate_translation_angles,
                        n_trans=n_trans,
                        half_weights=(
                            half_weights_windowed if use_window else half_weights
                        ),
                        packed_to_compact=tree_rescore_fftw_order,
                        projector_full=coarse_gaussian_projector_full,
                        current_size=score_size,
                        padding_factor=projection_padding_factor,
                        projector_max_r=relion_projector_r_max,
                        debug_iteration=debug_iteration,
                    )
                    tree_rescore_exact_ties += exact_ties
                    row_ids = np.arange(ambiguous_rows.size, dtype=np.int32)
                    rescored_runner_slot = 1 - rescored_winner_slot
                    rescored_winner_pose = candidate_pose_ids[row_ids, rescored_winner_slot]
                    rescored_runner_pose = candidate_pose_ids[row_ids, rescored_runner_slot]
                    rescored_winner_score = rescored_scores_np[row_ids, rescored_winner_slot]
                    rescored_runner_score = rescored_scores_np[row_ids, rescored_runner_slot]
                    tree_rescore_winner_changes += int(
                        np.count_nonzero(rescored_winner_pose != best_pose_np)
                    )
                    applied_rows = np.arange(ambiguous_rows.size, dtype=np.int32)
                    if applied_rows.size:
                        rows_jax = jnp.asarray(ambiguous_rows[applied_rows], dtype=jnp.int32)
                        best_argmax_batch = best_argmax_batch.at[rows_jax].set(
                            rescored_winner_pose[applied_rows]
                        )
                        best_score_batch = best_score_batch.at[rows_jax].set(
                            rescored_winner_score[applied_rows]
                        )
                        class_best_argmaxes[0] = class_best_argmaxes[0].at[rows_jax].set(
                            rescored_winner_pose[applied_rows]
                        )
                        class_best_scores[0] = class_best_scores[0].at[rows_jax].set(
                            rescored_winner_score[applied_rows]
                        )
                        class_second_best_argmaxes[0] = class_second_best_argmaxes[0].at[
                            rows_jax
                        ].set(rescored_runner_pose[applied_rows])
                        class_second_best_scores[0] = class_second_best_scores[0].at[
                            rows_jax
                        ].set(rescored_runner_score[applied_rows])

            global_log_z = global_max + jnp.log(global_sum)
            class_log_z_values = [
                class_max + jnp.log(class_sum) for class_max, class_sum in zip(class_max_values, class_sum_values)
            ]

            normalization_score_mats = []
            if collect_significance:
                batch_values = (
                    batched_support_values
                    if relion_f32_coarse_support_enabled
                    else jnp.exp(batched_support_values - global_log_z[:, None])
                )
                if return_relion_f32_normalization and not relion_f32_coarse_support_enabled:
                    # The program's values are the with-prior scores here.
                    normalization_score_mats.append(batched_support_values)
                if relion_f32_coarse_support_enabled:
                    # The float32 posterior and what the batch publishes from it are one program
                    # (coarse_support_posterior). K=1 hands it the pre-prior scores and the priors:
                    # RELION's log-weight order (relion_exact_coarse_weight_order above).
                    if relion_exact_coarse_weight_order and exact_rotation_prior is None:
                        exact_rotation_prior = (
                            jnp.zeros(n_rot, dtype=jnp.float32)
                            if rotation_log_prior_padded is None
                            else jnp.asarray(rotation_log_prior_padded[0, :n_rot], dtype=jnp.float32)
                        ) + jnp.asarray(class_log_priors_np[0], dtype=jnp.float32)
                    support = coarse_support_posterior(
                        batch_values,
                        relion_raw_score_max,
                        exact_rotation_prior,
                        (
                            None
                            if not relion_exact_coarse_weight_order
                            else np.zeros(n_trans, dtype=np.float32)
                            if batch_translation_log_prior is None
                            else batch_translation_log_prior
                        ),
                        exact_weight_order=relion_exact_coarse_weight_order,
                        n_trans=int(n_trans),
                        adaptive_fraction=float(adaptive_fraction),
                        max_significants=max_significants,
                        tie_score_ulps=int(relion_f32_coarse_tie_ulps),
                    )
                    batch_weights = support["weights"]
                    batch_sig_mask = support["mask"]
                    batch_n_sig = support["n_significant"]
                    batch_cutoff_count = support["cutoff_count"]
                    _batch_sum_weight = support["sum_weight"]
                    _batch_significant_weight = support["significant_weight"]
                    batch_sig_rot_mask = support["rotation_mask"]
                    batch_pmax = support["pmax"]
                    if relion_exact_coarse_weight_order:
                        # RELION publishes the coarse winner from these weights.
                        best_argmax_batch = support["winner"]
                        best_class_batch = zeros_i32
                else:
                    batch_weights = batch_values
                    if return_relion_f32_normalization:
                        from relax.sparse_pass2.sparse_pass2_posterior import _relion_f32_fine_posterior

                        # Retain the existing coarse selector and all of its outputs.
                        # The symbolic fine pass needs the numeric maximum-shifted
                        # denominator, not exp(logZ) or a second normalized support.
                        # See docs/math/zero_oversampling.md.
                        normalization_probs, _, _, _, normalization_sum, _ = _relion_f32_fine_posterior(
                            jnp.concatenate(normalization_score_mats, axis=1),
                            adaptive_fraction=adaptive_fraction,
                            keep_all=True,
                        )
                        relion_f32_sum_weight[start_idx:end_idx] = np.asarray(
                            normalization_sum, dtype=np.float32,
                        )[:actual_batch_size]
                        relion_f32_max_posterior[start_idx:end_idx] = (
                            _coarse_max_posterior_for_host(normalization_probs, actual_batch_size)
                        )
                    (
                        batch_sig_mask,
                        batch_sig_rot_mask,
                        batch_n_sig,
                        batch_cutoff_count,
                    ) = _find_sig(
                        batch_weights,
                        n_classes * n_rot,
                        n_trans,
                        adaptive_fraction=adaptive_fraction,
                        max_significants=max_significants,
                        return_cutoff_count=True,
                    )

            # A batch on the float32 support route publishes while the next batch's operands are on the
            # device and before that batch's score program (the call above the program): the device
            # scores this batch while the host prepares the next. Only a dump reads the large score and
            # operand arrays, and a dump batch publishes at once, so a waiting batch does not hold them.
            defer_publish = collect_significance and relion_f32_coarse_support_enabled and not debug_dump_enabled
            batch_outputs = BatchOutputs(
                start_idx=start_idx,
                end_idx=end_idx,
                actual_batch_size=actual_batch_size,
                batch_size=batch_size,
                indices=indices,
                pmax=batch_pmax,
                weights=None if defer_publish else batch_weights,
                sig_mask=batch_sig_mask,
                sig_rot_mask=batch_sig_rot_mask,
                n_sig=batch_n_sig,
                cutoff_count=batch_cutoff_count,
                sum_weight=_batch_sum_weight,
                significant_weight=None if defer_publish else _batch_significant_weight,
                best_argmax=best_argmax_batch,
                best_class=best_class_batch,
                best_score=best_score_batch,
                global_log_z=global_log_z,
                class_log_z_values=class_log_z_values,
                class_best_scores=class_best_scores,
                class_best_argmaxes=class_best_argmaxes,
                class_second_best_scores=class_second_best_scores,
                class_second_best_argmaxes=class_second_best_argmaxes,
                debug_dump_enabled=debug_dump_enabled,
                dump_target_local_positions=dump_target_local_positions,
                dump_target_pre_prior_blocks_per_class=None if defer_publish else dump_target_pre_prior_blocks_per_class,
                dump_target_with_prior_blocks_per_class=None if defer_publish else dump_target_with_prior_blocks_per_class,
                translation_log_prior=None if defer_publish else batch_translation_log_prior,
                coarse_gaussian_shifted_corrected=None if defer_publish else coarse_gaussian_shifted_corrected,
                coarse_gaussian_unshifted_corrected=None if defer_publish else coarse_gaussian_unshifted_corrected,
                coarse_gaussian_pixel_weight=None if defer_publish else coarse_gaussian_pixel_weight,
                coarse_gaussian_initial_diff2=None if defer_publish else coarse_gaussian_initial_diff2,
                exact_cc_operands=None if defer_publish else exact_cc_operands,
                exact_cc_pixel_weight=None if defer_publish else exact_cc_pixel_weight,
                exact_cc_shifted=None if defer_publish else exact_cc_shifted,
            )
            if defer_publish:
                pending_batch = batch_outputs
            else:
                _publish_batch(batch_outputs)
            start_idx = end_idx
        if pending_batch is not None:
            _publish_batch(pending_batch)

    if _coarse_batch_starts:
        _loop_end = time.time()
        _loop_s = _loop_end - _coarse_loop_t0
        _coarse_batch_walls = [
            b - a for a, b in zip(_coarse_batch_starts, _coarse_batch_starts[1:])
        ] + [_loop_end - _coarse_batch_starts[-1]]
        _tot = sum(_coarse_batch_walls)
        _srt = sorted(_coarse_batch_walls)
        logger.info(
            "K-class coarse pass-1 batch timing: batches=%d images=%d loop=%.2fs "
            "covered=%.2fs uncovered=%.2fs mean=%.3fs median=%.3fs max=%.3fs",
            len(_coarse_batch_walls),
            int(n_images),
            _loop_s,
            _tot,
            _loop_s - _tot,
            _tot / len(_coarse_batch_walls),
            _srt[len(_srt) // 2],
            _srt[-1],
        )

    if any(device_significance_counts):
        from relax.sparse_pass2.resident_significance import (
            DeviceCompactedSignificantSamples,
            build_coarse_significance_csr,
            host_support_rows,
        )

        for class_index in range(n_classes):
            covered = int(sum(int(counts.size) for counts in device_significance_counts[class_index]))
            if covered != n_images:
                # Some batches kept the host mask (a score dump): publish the
                # compacted batches as host rows too.
                for start, counts, polarity, ids in zip(
                    device_significance_starts[class_index],
                    device_significance_counts[class_index],
                    device_significance_polarity[class_index],
                    device_significance_ids[class_index],
                    strict=True,
                ):
                    batch_rows = host_support_rows(
                        build_coarse_significance_csr(
                            n_images=int(counts.size),
                            n_coarse_rot=n_rot,
                            n_coarse_trans=n_trans,
                            n_significant_per_batch=[counts],
                            store_excluded_per_batch=[polarity],
                            ids_per_batch=[ids],
                        )
                    )
                    for offset, row in enumerate(batch_rows):
                        significant_sample_indices[class_index][start + offset] = row
                continue
            coarse_significance_csr = build_coarse_significance_csr(
                n_images=n_images,
                n_coarse_rot=n_rot,
                n_coarse_trans=n_trans,
                n_significant_per_batch=device_significance_counts[class_index],
                store_excluded_per_batch=device_significance_polarity[class_index],
                ids_per_batch=device_significance_ids[class_index],
            )
            significant_sample_indices[class_index] = DeviceCompactedSignificantSamples(
                host_support_rows(coarse_significance_csr),
                csr=coarse_significance_csr,
            )
            logger.info(
                "Coarse significance compacted on the device (class %d): %d images, %d ids "
                "(%.2f MB) instead of a %.2f GB support mask",
                class_index,
                n_images,
                int(coarse_significance_csr.ids.size),
                coarse_significance_csr.ids.nbytes / 1e6,
                float(n_images) * float(n_rot) * float(n_trans) / 1e9,
            )

    full_stats = {
        "normalization_log_z": normalization_log_z,
        "normalization_log_evidence": normalization_log_evidence,
        "log_evidence_per_image": log_evidence,
        "best_log_score_per_image": best_log_score,
        "max_posterior_per_image": max_posterior,
        "class_log_evidence_per_image": class_log_evidence,
        "class_assignments": class_assignment,
        # RELION serializes the cutoff rank before inclusive threshold ties
        # expand the pass-2/M-step support represented by ``n_sig_all``.
        "significant_cutoff_counts": cutoff_count_all,
        "executed_coarse_backend": (
            "exact_cc_gemm" if exact_cc_enabled else "gemm_macro"
        ),
    }
    if coarse_gaussian_square_layout is not None:
        full_stats["coarse_gaussian_square_layout"] = coarse_square_layout_metadata(
            coarse_gaussian_square_layout,
            stable_fourier_window_shapes=stable_fourier_window_shapes,
        )
    if coarse_gaussian_gemm_resource_estimate is not None:
        full_stats["coarse_gaussian_gemm_resources"] = {
            field: int(value)
            for field, value in coarse_gaussian_gemm_resource_estimate._asdict().items()
        }
    if coarse_gaussian_gemm_projection_cache_plan is not None:
        full_stats["coarse_gaussian_gemm_projection_cache"] = (
            _coarse_gaussian_gemm_projection_cache_stats(
                coarse_gaussian_gemm_projection_cache_plan,
                enabled=coarse_gaussian_gemm_projection_cache is not None,
            )
        )
    if _coarse_significance_support_audit_enabled():
        if significant_sample_indices is None:
            raise RuntimeError(
                f"{_COARSE_SIGNIFICANCE_SUPPORT_AUDIT_ENV}=1 requires "
                "collect_significance=True",
            )
        full_stats["coarse_significance_support_audit"] = (
            _build_coarse_significance_support_audit(
                significant_sample_indices,
                samples_per_class=n_rot * n_trans,
                include_ids=_coarse_significance_support_audit_ids_enabled(),
            )
        )
    if relion_f32_sum_weight is not None:
        # RELION's oversampling-zero second pass deliberately reuses this
        # coarse, maximum-shifted float32 denominator numerically.  It is not
        # interchangeable with a log-evidence value because the fine pass
        # independently shifts its own maximum to 50 before division.
        full_stats["relion_f32_sum_weight"] = relion_f32_sum_weight
    if relion_f32_max_posterior is not None:
        full_stats["relion_f32_max_posterior"] = relion_f32_max_posterior
    if return_class_best:
        full_stats["class_best_log_score_per_image"] = class_best_log_score
        full_stats["class_best_offset_free_log_score_per_image"] = class_best_offset_free_log_score
        full_stats["class_hard_assignments"] = class_hard_assignment
    if return_class_second:
        full_stats["class_second_best_log_score_per_image"] = class_second_best_log_score
        full_stats["class_second_hard_assignments"] = class_second_hard_assignment
        full_stats["class_second_best_offset_free_log_score_per_image"] = class_second_best_offset_free_log_score
    if tree_rescore_enabled:
        full_stats["firstiter_cc_tree_top2_rescore"] = {
            "max_margin": float(tree_rescore_max_margin),
            "examined_images": int(tree_rescore_examined),
            "ambiguous_images": int(tree_rescore_ambiguous),
            "exact_score_ties": int(tree_rescore_exact_ties),
            "winner_changes": int(tree_rescore_winner_changes),
        }
        logger.warning(
            "RELION coarse-tree top-2 rescore complete: "
            "examined=%d ambiguous=%d exact_ties=%d winner_changes=%d",
            tree_rescore_examined,
            tree_rescore_ambiguous,
            tree_rescore_exact_ties,
            tree_rescore_winner_changes,
        )
    return sig_rot_any, n_sig_all, hard_assignment, class_assignment, significant_sample_indices, full_stats
