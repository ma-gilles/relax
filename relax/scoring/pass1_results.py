"""What pass 1 hands from one stage to the next, as records."""

from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np


@dataclass(frozen=True)
class PassShape:
    """The sizes of one pass 1, fixed from the call's arguments: what its stage plans are drawn up over.

    ``n_classes`` classes, ``n_rot`` coarse rotations and ``n_trans`` translations make the pose grid. ``image_shape``
    is the dataset's image shape, ``n_half`` its half-spectrum pixel count, and ``score_size`` the current size the
    pass scores at (the image size when the call gives none).
    """

    n_classes: int
    n_rot: int
    n_trans: int
    n_half: int
    image_shape: Any
    score_size: int


@dataclass(frozen=True)
class OutputPlan:
    """What one pass 1 returns and how many rows it has: fixed from the call's arguments before the first batch.

    ``n_images`` images are scored against ``n_classes * n_rot * n_trans`` poses. ``collect_significance`` asks for the
    significant samples; ``relion_f32_coarse_support_enabled`` is the float32 support route (RELION's coarse posterior
    on the device, which also yields ``sum_weight`` and the per-image maximum), ``return_relion_f32_normalization`` asks
    for that normalization on every route; ``return_class_best`` and ``return_class_second`` ask for each class's best
    and runner-up pose. ``score_real_dtype`` is the dtype of the published scores (float64 only for the diagnostic
    float64 scoring).
    """

    n_classes: int
    n_rot: int
    n_trans: int
    n_images: int
    score_real_dtype: Any
    collect_significance: bool
    relion_f32_coarse_support_enabled: bool
    return_relion_f32_normalization: bool
    return_class_best: bool
    return_class_second: bool


@dataclass(frozen=True)
class ScoreDumpContext:
    """What the score dump of a pass reads besides the batch: the call's own inputs, fixed for the pass.

    ``RELAX_SIGNIFICANCE_DUMP_*`` writes the scores and operands of chosen images
    (:func:`relax.diagnostics.coarse_gaussian_diagnostics._maybe_dump_k_class_significance_batch`). The dump only
    observes; nothing here steers the pass. ``score_indices`` are the scored rows of the half spectrum of the pass's
    own score (the Gaussian GEMM's square rows, or the normalized CC's), on the device.
    """

    experiment_dataset: Any
    rotations: Any
    translations: Any
    translations_source: Any
    class_log_priors: np.ndarray  # float64 [K]
    rotation_log_prior_padded: Any
    current_size: int | None
    adaptive_fraction: float
    max_significants: int
    debug_iteration: int | None
    relion_projector_half: Any
    relion_projector_r_max: int | None
    projection_padding_factor: int
    score_indices: Any


@dataclass(frozen=True)
class BatchOutputs:
    """One image batch's outputs of pass 1, from the end of its score program until ``publish_batch`` reads it back.

    ``batch_size`` rows were scored (``actual_batch_size`` of them are images of the pass: a short last batch is
    repeat-padded); the pass's images ``start_idx:end_idx`` are ``indices``. Arrays stay on the device; a field
    that a route does not produce is ``None``.

    ``pmax`` .. ``cutoff_count`` and ``sum_weight`` are the float32 support's results (the pass's ``collect_significance``
    route); ``normalization_sum_weight`` and ``normalization_max_posterior`` are the generic route's RELION float32
    normalization when asked (host arrays of the real rows, written to the pass's results as they are); ``best_*`` is the
    joint class-and-pose winner; ``class_*`` hold one array per class.

    The last six fields are read only to dump the batch (``debug_dump_enabled``). A batch that waits for its
    read-back while the device scores the next one (the float32 support route without a dump) leaves them ``None``,
    so that it does not hold the batch's score and operand arrays. ``operands`` are the batch's
    :class:`relax.scoring.pass1_operands.ScoreOperands`.
    """

    start_idx: int
    end_idx: int
    actual_batch_size: int
    batch_size: int
    indices: Any
    pmax: Any
    weights: Any
    sig_mask: Any
    sig_rot_mask: Any
    n_sig: Any
    cutoff_count: Any
    sum_weight: Any
    significant_weight: Any
    normalization_sum_weight: Any
    normalization_max_posterior: Any
    best_argmax: Any
    best_class: Any
    best_score: Any
    global_log_z: Any
    class_log_z_values: list
    class_best_scores: list | None
    class_best_argmaxes: list | None
    class_second_best_scores: list | None
    class_second_best_argmaxes: list | None
    debug_dump_enabled: bool
    dump_target_local_positions: Any
    dump_target_pre_prior_blocks_per_class: list | None
    dump_target_with_prior_blocks_per_class: list | None
    translation_log_prior: Any
    operands: Any


@dataclass
class Pass1Outputs:
    """The per-image results of one pass 1: arrays allocated once for the pass and filled batch by batch.

    ``publish_batch`` writes rows ``start_idx:end_idx`` of every array in place; nothing else writes them. Rows are
    images in the order of the pass; ``K`` is the number of classes, ``R`` the number of coarse rotations. Arrays are
    ``np.empty`` until their rows are published. The class arrays and the float32 arrays exist only on the routes
    that return them (the other fields are ``None``).

    ``class_best_offset_free_log_score`` and its runner-up twin are diagnostic native scores before the large,
    class-common image normalization offset: the offset is useful for absolute log evidence, but adding it before a
    float32 cast can erase class and pose margins.

    ``device_significance_*`` hold, per class and per batch, the support compacted on the device (counts, polarity,
    ids, and the first image of the batch); they are read once, after the loop, to build the class's CSR.
    """

    sig_rot_any: np.ndarray  # bool [K, R]: rotations with a significant sample for some image
    n_sig_all: np.ndarray  # int32 [N]: significant samples, ties included
    cutoff_count_all: np.ndarray  # int32 [N]: RELION's cutoff rank, before ties
    hard_assignment: np.ndarray  # int32 [N]: best joint (rotation * T + translation) pose
    class_assignment: np.ndarray  # int32 [N]: best class
    significant_sample_indices: list | None  # [K][N] per-image encodings; None without collect_significance
    normalization_log_z: np.ndarray  # float64 [N]
    normalization_log_evidence: np.ndarray  # float64 [N]
    log_evidence: np.ndarray  # score dtype [N]
    best_log_score: np.ndarray  # score dtype [N]
    max_posterior: np.ndarray  # score dtype [N]
    relion_f32_sum_weight: np.ndarray | None  # float32 [N]
    relion_f32_max_posterior: np.ndarray | None  # float32 [N]
    class_log_evidence: np.ndarray  # float64 [K, N]
    class_best_log_score: np.ndarray | None  # score dtype [K, N]
    class_second_best_log_score: np.ndarray | None
    class_best_offset_free_log_score: np.ndarray | None
    class_second_best_offset_free_log_score: np.ndarray | None
    class_hard_assignment: np.ndarray | None  # int32 [K, N]
    class_second_hard_assignment: np.ndarray | None
    device_significance_counts: list
    device_significance_polarity: list
    device_significance_ids: list
    device_significance_starts: list

    @classmethod
    def allocate(cls, plan: OutputPlan) -> "Pass1Outputs":
        """The arrays of a pass over ``plan.n_images`` images, for the routes the call asks for."""

        return cls(
            sig_rot_any=np.zeros((plan.n_classes, plan.n_rot), dtype=bool),
            n_sig_all=np.empty(plan.n_images, dtype=np.int32),
            cutoff_count_all=np.empty(plan.n_images, dtype=np.int32),
            hard_assignment=np.empty(plan.n_images, dtype=np.int32),
            class_assignment=np.empty(plan.n_images, dtype=np.int32),
            significant_sample_indices=[[None] * plan.n_images for _ in range(plan.n_classes)]
            if plan.collect_significance
            else None,
            normalization_log_z=np.empty(plan.n_images, dtype=np.float64),
            normalization_log_evidence=np.empty(plan.n_images, dtype=np.float64),
            log_evidence=np.empty(plan.n_images, dtype=plan.score_real_dtype),
            best_log_score=np.empty(plan.n_images, dtype=plan.score_real_dtype),
            max_posterior=np.empty(plan.n_images, dtype=plan.score_real_dtype),
            relion_f32_sum_weight=(
                np.empty(plan.n_images, dtype=np.float32)
                if (plan.relion_f32_coarse_support_enabled or plan.return_relion_f32_normalization)
                and plan.collect_significance
                else None
            ),
            relion_f32_max_posterior=np.empty(plan.n_images, dtype=np.float32)
            if plan.return_relion_f32_normalization
            else None,
            class_log_evidence=np.empty((plan.n_classes, plan.n_images), dtype=np.float64),
            class_best_log_score=(
                np.empty((plan.n_classes, plan.n_images), dtype=plan.score_real_dtype)
                if plan.return_class_best
                else None
            ),
            class_second_best_log_score=(
                np.empty((plan.n_classes, plan.n_images), dtype=plan.score_real_dtype)
                if plan.return_class_second
                else None
            ),
            class_best_offset_free_log_score=(
                np.empty((plan.n_classes, plan.n_images), dtype=plan.score_real_dtype)
                if plan.return_class_best
                else None
            ),
            class_second_best_offset_free_log_score=(
                np.empty((plan.n_classes, plan.n_images), dtype=plan.score_real_dtype)
                if plan.return_class_second
                else None
            ),
            class_hard_assignment=np.empty((plan.n_classes, plan.n_images), dtype=np.int32)
            if plan.return_class_best
            else None,
            class_second_hard_assignment=(
                np.empty((plan.n_classes, plan.n_images), dtype=np.int32) if plan.return_class_second else None
            ),
            device_significance_counts=[[] for _ in range(plan.n_classes)],
            device_significance_polarity=[[] for _ in range(plan.n_classes)],
            device_significance_ids=[[] for _ in range(plan.n_classes)],
            device_significance_starts=[[] for _ in range(plan.n_classes)],
        )


class Pass1Result(NamedTuple):
    """What pass 1 returns to its callers; the six positions of the tuple it returned before, now named.

    ``sig_rot_any`` is the bool ``[K, R]`` of rotations with a significant sample for some image, ``n_sig_all`` the
    significant samples per image (ties included), ``hard_assignment`` and ``class_assignment`` the best joint pose and
    class per image, and ``significant_sample_indices`` the per-class supports (``None`` without
    ``collect_significance``). ``full_stats`` is the dict of the pass's statistics and reports
    (:func:`relax.scoring.pass1_assembly.build_full_stats`).
    """

    sig_rot_any: np.ndarray
    n_sig_all: np.ndarray
    hard_assignment: np.ndarray
    class_assignment: np.ndarray
    significant_sample_indices: Any
    full_stats: dict
