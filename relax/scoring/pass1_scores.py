"""The plan and the results of pass 1's score program: its blocks, priors and static arguments, and one batch's scores.

:func:`relax.scoring.pass1_program.run_score_program` runs the program from a :class:`ScoreProgramPlan` and returns
:class:`BatchScores`.
"""

import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple

import jax.numpy as jnp

from relax.helpers.projection_cache import build_projection_cache
from relax.scoring.coarse_projector import CoarseProjector
from relax.scoring.pass1_priors import RotationBlocks
from relax.scoring.pass1_request import Pass1Request
from relax.scoring.pass1_results import PassShape
from relax.scoring.pass1_route import RoutePlan

logger = logging.getLogger(__name__)


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
class ProgramStatics:
    """What the score program is specialised on: the static arguments of its jitted kernels, one hashable record.

    The program is compiled once per distinct record (and block shape). ``n_trans`` translations per rotation score
    images of ``image_shape`` against a volume of ``volume_shape``; ``float64`` is the diagnostic precision.
    ``score_kind`` is ``"gaussian"`` or ``"normalized_cc"``; ``exact_weight_order`` asks for the pre-prior support
    values (RELION's log-weight order) and ``return_values`` for the support values at all. ``return_class_best`` and
    ``track_class_second`` ask for each class's best pose and its runner-up.
    """

    n_trans: int
    image_shape: tuple
    volume_shape: tuple
    float64: bool
    score_kind: str
    exact_weight_order: bool
    return_class_best: bool
    track_class_second: bool = False
    return_values: bool = True


@dataclass(frozen=True)
class ScoreProgramPlan:
    """What every batch's score program runs over, fixed for the pass.

    ``blocks`` and ``prior_terms`` are aligned (:func:`score_blocks`, :func:`block_prior_terms`). ``rotations_padded``
    are the rotations padded with identities to whole blocks. ``projector`` yields a block's reference rows; with a
    ``projection_cache`` (the cached C64 table of the Gaussian GEMM) one program reads every block from it, otherwise
    each block is projected and scored in its own call. ``statics`` is what the kernels are specialised on.
    """

    blocks: tuple
    prior_terms: tuple
    rotations_padded: Any
    projector: CoarseProjector
    projection_cache: Any
    n_classes: int
    statics: ProgramStatics


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


def plan_score_program(
    request: Pass1Request,
    shape: PassShape,
    route: RoutePlan,
    relion_projector_half,
    rotation_blocks: RotationBlocks,
    class_log_priors,
    *,
    exact_weight_order: bool,
    float64: bool,
    rotated_radius: bool,
) -> ScoreProgramPlan:
    """Plan the score program of a pass over ``shape`` on ``route``: its blocks, priors, projector and projection cache.

    ``rotation_blocks`` are the rotations and rotation prior padded to whole blocks of ``route.rotation_block_size`` and
    ``class_log_priors`` the classes' prior (``[K]``). The projector is built from ``relion_projector_half``, on the
    route's compact rows; ``rotated_radius`` asks it for the canonical rotated clipping. ``exact_weight_order`` asks for
    the pre-prior support values and ``float64`` for the diagnostic precision. Builds the projection cache the route
    planned (none when it planned none).
    """

    coarse_projector = CoarseProjector(
        relion_projector_half=relion_projector_half,
        relion_projector_r_max=request.relion_projector_r_max,
        image_shape=shape.image_shape,
        projection_padding_factor=request.projection_padding_factor,
        current_size=request.current_size,
        score_size=shape.score_size,
        stable_fourier_window_shapes=request.stable_fourier_window_shapes,
        compact=route.compact_rows,
        rotated_radius=rotated_radius,
    )

    projection_cache = None
    if route.projection_cache_plan is not None:
        projection_cache = build_projection_cache(
            route.projection_cache_plan,
            partial(coarse_projector.cache_block, request.rotations),
        )
        logger.warning(
            "Coarse GEMM C64 projection cache built: "
            "shape=%s chunks=%d conservative_peak_bytes=%d budget_bytes=%d",
            route.projection_cache_plan.cache_shape,
            route.projection_cache_plan.chunk_count_per_table,
            route.projection_cache_plan.predicted_peak_bytes,
            route.projection_cache_plan.budget_bytes,
        )

    blocks = score_blocks(shape.n_classes, shape.n_rot, route.rotation_block_size)
    return ScoreProgramPlan(
        blocks=blocks,
        prior_terms=block_prior_terms(
            blocks, class_log_priors, rotation_blocks.rotation_log_prior_padded, route.rotation_block_size
        ),
        rotations_padded=rotation_blocks.rotations_padded,
        projector=coarse_projector,
        projection_cache=projection_cache,
        n_classes=shape.n_classes,
        statics=ProgramStatics(
            n_trans=int(shape.n_trans),
            image_shape=tuple(int(value) for value in shape.image_shape),
            volume_shape=tuple(int(value) for value in request.experiment_dataset.volume_shape),
            float64=float64,
            score_kind=route.score_kind,
            exact_weight_order=exact_weight_order,
            return_class_best=bool(request.return_class_best),
            track_class_second=bool(request.return_class_second or route.tree_rescore_plan is not None),
            return_values=bool(request.collect_significance),
        ),
    )
