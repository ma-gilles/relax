"""The first-iteration tree top-two rescore of pass 1 (``--firstiter_cc`` with ``tree_rescore_max_margin``).

RELION's coarse tree can pick a winner whose normalized CC is within float32 noise of the runner-up's. For each image
whose best and second-best coarse poses differ by at most ``max_margin``, the two candidates are scored again by
RELION's direct texture projection (``relion_coarse_normalized_cc_rescore``), and the better one becomes the winner.
Only K=1 on a CUDA backend runs it.

:func:`require_tree_rescore_call` refuses a call that cannot run it, :func:`plan_tree_rescore` builds what is fixed for the pass; :func:`rescore_ambiguous_images`
rescoring one batch returns the arrays it replaced (the caller assigns them) and what it counted.
"""

import logging
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from relax.diagnostics.coarse_gaussian_diagnostics import maybe_dump_tree_rescore_batch
from relax.relion.relion_coarse_operands import select_relion_coarse_rescore_winner_slots

logger = logging.getLogger(__name__)

_DENSE_MEANS_SCALE_ENV = "RELAX_DENSE_MEANS_SCALE"


def _dense_projection_scale(image_shape) -> float:
    """Match the dense E-step projection scaling used by the shared helper."""

    import os

    token = (os.environ.get(_DENSE_MEANS_SCALE_ENV) or "-N2").strip()
    n = int(image_shape[0])
    scale = {"-N2": -(n**2), "N2": float(n**2)}.get(token)
    if scale is None:
        raise ValueError(f"Unsupported {_DENSE_MEANS_SCALE_ENV}={token!r}")
    return scale


@dataclass(frozen=True)
class TreeRescoreGeometry:
    """The pass's coarse geometry that the rescore and its dump read, fixed for the pass.

    ``half_weights`` are the score weights of the rows scored (windowed when the pass is), ``rotations`` ``[R, 3, 3]``
    the coarse rotations, ``n_trans`` the number of coarse translations, ``score_size`` the current size, and
    ``padding_factor`` and ``projector_max_r`` the projection's padding factor and maximum radius. The coarse HEALPix
    order, ``coarse_rotation_ids`` and the symmetry label order exact ties.
    """

    half_weights: Any
    rotations: Any
    n_trans: int
    score_size: int
    padding_factor: int
    projector_max_r: int
    coarse_healpix_order: int
    coarse_rotation_ids: Any
    symmetry_label: str


@dataclass(frozen=True)
class TreeRescorePlan:
    """What the rescore of one pass 1 needs, fixed before its first batch.

    ``projector_full`` is the complex64 texture projector of the one class, scaled as the dense E-step scales its
    references; ``fftw_order`` maps the packed score rows to the compact ones, and ``translation_angles`` ``[T]`` are
    RELION's phase angles of every coarse translation.
    """

    max_margin: float
    projector_full: jax.Array
    fftw_order: jax.Array
    translation_angles: jax.Array
    geometry: TreeRescoreGeometry


class TreeRescoreState(NamedTuple):
    """The six per-image arrays of a batch that the rescore may replace (device arrays, one row per scored image).

    ``best_*`` is the joint winner of the batch (K=1: the same pose as ``class_best_*``); ``class_second_*`` the
    class's runner-up. ``*_argmax`` are pose ids ``rotation * T + translation``, ``*_score`` the normalized CC.
    """

    best_argmax: Any
    best_score: Any
    class_best_argmax: Any
    class_best_score: Any
    class_second_argmax: Any
    class_second_score: Any


class TreeRescoreBatch(NamedTuple):
    """One batch after the rescore: its state (the arrays it replaced; the same objects if no image was ambiguous),
    the images it found ambiguous, the exact score ties among their candidates, and the winners it changed."""

    state: TreeRescoreState
    ambiguous_images: int
    exact_ties: int
    winner_changes: int


@dataclass(frozen=True)
class TreeRescoreTotals:
    """What the rescore has counted over the batches of a pass so far: images examined, ambiguous, exact score ties
    among the candidates, and winners changed."""

    examined: int = 0
    ambiguous: int = 0
    exact_ties: int = 0
    winner_changes: int = 0

    def after_batch(self, batch_size: int, rescored: TreeRescoreBatch) -> "TreeRescoreTotals":
        """The totals with one more batch of ``batch_size`` rows, whose rescore returned ``rescored``."""

        return TreeRescoreTotals(
            examined=self.examined + int(batch_size),
            ambiguous=self.ambiguous + rescored.ambiguous_images,
            exact_ties=self.exact_ties + rescored.exact_ties,
            winner_changes=self.winner_changes + rescored.winner_changes,
        )


def tree_rescore_report(totals: TreeRescoreTotals, max_margin: float) -> dict:
    """The pass's report of the rescore (the ``firstiter_cc_tree_top2_rescore`` entry of ``full_stats``)."""

    return {
        "max_margin": float(max_margin),
        "examined_images": int(totals.examined),
        "ambiguous_images": int(totals.ambiguous),
        "exact_score_ties": int(totals.exact_ties),
        "winner_changes": int(totals.winner_changes),
    }


def log_tree_rescore_totals(totals: TreeRescoreTotals) -> None:
    """Log the totals of a pass's rescore."""

    logger.warning(
        "RELION coarse-tree top-2 rescore complete: "
        "examined=%d ambiguous=%d exact_ties=%d winner_changes=%d",
        totals.examined,
        totals.ambiguous,
        totals.exact_ties,
        totals.winner_changes,
    )


def require_tree_rescore_call(*, n_classes: int, return_class_best: bool) -> None:
    """Refuse, naming ``tree_rescore_max_margin``, a call that cannot run the rescore.

    The rescore is K=1 and returns the class best. The supplied RELION projector with texture interpolation and
    half-spectrum scoring, which it also needs, are required of every pass 1 before it is planned
    (``_require_exact_pass1_operands``).
    """

    if n_classes != 1:
        raise ValueError(
            "the coarse-tree top-2 rescore (tree_rescore_max_margin) currently supports K=1 only",
        )
    if not return_class_best:
        raise ValueError(
            "the coarse-tree top-2 rescore (tree_rescore_max_margin) requires return_class_best=True",
        )


def plan_tree_rescore(
    *,
    max_margin: float,
    relion_projector_half,
    image_shape,
    n_half: int,
    score_indices_np,
    translation_angles,
    geometry: TreeRescoreGeometry,
) -> TreeRescorePlan:
    """Build the rescore's plan once :func:`require_tree_rescore_call` has accepted the call.

    Refuses a backend that is not the custom CUDA one. ``score_indices_np`` are the scored rows of the window
    (``None``: every half-spectrum row). ``translation_angles`` is RELION's float32 ``[T, 2]`` table of the
    translations on the device, the one the pass's normalized-CC operands translate with.
    """

    from recovar import cuda_backproject

    from relax.helpers.fourier_window import relion_fftw_order_for_square_score_window
    from relax.helpers.projection import relion_projector_half_to_texture_full

    if (
        jax.default_backend() != "gpu"
        or not cuda_backproject.custom_cuda_requested()
        or not cuda_backproject.cuda_available()
    ):
        raise RuntimeError(
            "the coarse-tree top-2 rescore (tree_rescore_max_margin) requires the custom CUDA backend",
        )
    projector_full = jnp.asarray(
        relion_projector_half_to_texture_full(relion_projector_half[0])
        * jnp.asarray(_dense_projection_scale(image_shape), dtype=jnp.float32),
        dtype=jnp.complex64,
    )
    score_indices_np = np.arange(n_half, dtype=np.int32) if score_indices_np is None else score_indices_np
    fftw_order = jnp.asarray(
        relion_fftw_order_for_square_score_window(
            image_shape,
            geometry.score_size,
            score_indices_np,
        ),
        dtype=jnp.int32,
    )
    logger.warning(
        "RELION coarse-tree top-2 rescore: max_margin=%g current_size=%d",
        max_margin,
        geometry.score_size,
    )
    return TreeRescorePlan(
        max_margin=max_margin,
        projector_full=projector_full,
        fftw_order=fftw_order,
        translation_angles=translation_angles,
        geometry=geometry,
    )


def rescore_ambiguous_images(
    plan: TreeRescorePlan,
    state: TreeRescoreState,
    unshifted,
    corr_img,
    *,
    experiment_dataset,
    indices,
    debug_iteration,
) -> TreeRescoreBatch:
    """Rescore the images of one batch whose best and runner-up scores are within ``plan.max_margin``.

    ``unshifted`` and ``corr_img`` ``[B, P]`` are the batch's CC operands (the image Fourier rows and their weights, from
    the normalized-CC operand assembly). For each ambiguous row the two candidate poses are scored again by the direct
    texture projection and the better one replaces the row's best (and the other its runner-up) in the returned state;
    the rows that are not ambiguous, and every array when none is, are untouched. ``experiment_dataset``, ``indices``
    and ``debug_iteration`` only name the candidates of a ``RELAX_SIGNIFICANCE_DUMP_*`` dump.
    """

    from relax.scoring.scoring import relion_coarse_normalized_cc_rescore

    tree_score_dtype = np.float32
    best_scores_np = np.asarray(state.class_best_score, dtype=tree_score_dtype)
    second_scores_np = np.asarray(state.class_second_score, dtype=tree_score_dtype)
    score_margins = best_scores_np - second_scores_np
    ambiguous_rows = np.flatnonzero(np.isfinite(score_margins) & (score_margins <= plan.max_margin)).astype(np.int32)
    if not ambiguous_rows.size:
        return TreeRescoreBatch(state, ambiguous_images=0, exact_ties=0, winner_changes=0)
    best_pose_np = np.asarray(state.class_best_argmax, dtype=np.int32)[ambiguous_rows]
    second_pose_np = np.asarray(state.class_second_argmax, dtype=np.int32)[ambiguous_rows]
    candidate_pose_ids = np.sort(
        np.stack([best_pose_np, second_pose_np], axis=1),
        axis=1,
    )
    candidate_rotation_ids = candidate_pose_ids // plan.geometry.n_trans
    candidate_translation_ids = candidate_pose_ids % plan.geometry.n_trans
    candidate_rotations = jnp.asarray(
        plan.geometry.rotations[candidate_rotation_ids.reshape(-1)],
        dtype=jnp.float32,
    ).reshape(
        ambiguous_rows.size,
        2,
        3,
        3,
    )
    unshifted_candidates = jnp.broadcast_to(
        unshifted[jnp.asarray(ambiguous_rows, dtype=jnp.int32), None, :],
        (
            ambiguous_rows.size,
            2,
            unshifted.shape[-1],
        ),
    )
    candidate_translation_angles = plan.translation_angles[jnp.asarray(candidate_translation_ids, dtype=jnp.int32)]
    score_weight_candidates = jnp.broadcast_to(
        corr_img[jnp.asarray(ambiguous_rows, dtype=jnp.int32), None, :],
        unshifted_candidates.shape,
    )
    rescored_candidates = relion_coarse_normalized_cc_rescore(
        unshifted_candidates,
        score_weight_candidates,
        None,
        plan.geometry.half_weights,
        plan.fftw_order,
        projector_full=plan.projector_full,
        rotation_matrices=candidate_rotations,
        translation_angles=candidate_translation_angles,
        current_size=plan.geometry.score_size,
        padding_factor=plan.geometry.padding_factor,
        projector_max_r=plan.geometry.projector_max_r,
        numerator_weight_candidates=score_weight_candidates,
    )
    rescored_scores_np = np.asarray(rescored_candidates, dtype=tree_score_dtype)
    rescored_winner_slot, exact_ties = select_relion_coarse_rescore_winner_slots(
        rescored_scores_np,
        candidate_pose_ids,
        n_trans=plan.geometry.n_trans,
        healpix_order=plan.geometry.coarse_healpix_order,
        coarse_rotation_ids=plan.geometry.coarse_rotation_ids,
        score_dtype=tree_score_dtype,
        **({"symmetry_label": plan.geometry.symmetry_label} if plan.geometry.symmetry_label != "C1" else {}),
    )
    maybe_dump_tree_rescore_batch(
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
        n_trans=plan.geometry.n_trans,
        half_weights=plan.geometry.half_weights,
        packed_to_compact=plan.fftw_order,
        projector_full=plan.projector_full,
        current_size=plan.geometry.score_size,
        padding_factor=plan.geometry.padding_factor,
        projector_max_r=plan.geometry.projector_max_r,
        debug_iteration=debug_iteration,
    )
    row_ids = np.arange(ambiguous_rows.size, dtype=np.int32)
    rescored_runner_slot = 1 - rescored_winner_slot
    rescored_winner_pose = candidate_pose_ids[row_ids, rescored_winner_slot]
    rescored_runner_pose = candidate_pose_ids[row_ids, rescored_runner_slot]
    rescored_winner_score = rescored_scores_np[row_ids, rescored_winner_slot]
    rescored_runner_score = rescored_scores_np[row_ids, rescored_runner_slot]
    winner_changes = int(np.count_nonzero(rescored_winner_pose != best_pose_np))
    applied_rows = np.arange(ambiguous_rows.size, dtype=np.int32)
    rows_jax = jnp.asarray(ambiguous_rows[applied_rows], dtype=jnp.int32)
    replaced = TreeRescoreState(
        best_argmax=state.best_argmax.at[rows_jax].set(rescored_winner_pose[applied_rows]),
        best_score=state.best_score.at[rows_jax].set(rescored_winner_score[applied_rows]),
        class_best_argmax=state.class_best_argmax.at[rows_jax].set(rescored_winner_pose[applied_rows]),
        class_best_score=state.class_best_score.at[rows_jax].set(rescored_winner_score[applied_rows]),
        class_second_argmax=state.class_second_argmax.at[rows_jax].set(rescored_runner_pose[applied_rows]),
        class_second_score=state.class_second_score.at[rows_jax].set(rescored_runner_score[applied_rows]),
    )
    return TreeRescoreBatch(
        replaced,
        ambiguous_images=int(ambiguous_rows.size),
        exact_ties=exact_ties,
        winner_changes=winner_changes,
    )
