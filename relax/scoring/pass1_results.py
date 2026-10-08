"""What pass 1 hands from one stage to the next, as records."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BatchOutputs:
    """One image batch's outputs of pass 1, from the end of its score program until its host read-back.

    ``batch_size`` rows were scored (``actual_batch_size`` of them are images of the pass: a short last batch is
    repeat-padded); the pass's images ``start_idx:end_idx`` are ``indices``. Arrays stay on the device; a field
    that a route does not produce is ``None``.

    ``pmax`` .. ``cutoff_count`` and ``sum_weight`` are the float32 support's results (the pass's ``collect_significance``
    route); ``best_*`` is the joint class-and-pose winner; ``class_*`` hold one array per class.

    The last twelve fields are read only to dump the batch (``debug_dump_enabled``). A batch that waits for its
    read-back while the device scores the next one (the float32 support route without a dump) leaves them ``None``,
    so that it does not hold the batch's score and operand arrays.
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
    coarse_gaussian_shifted_corrected: Any
    coarse_gaussian_unshifted_corrected: Any
    coarse_gaussian_pixel_weight: Any
    coarse_gaussian_initial_diff2: Any
    exact_cc_operands: Any
    exact_cc_pixel_weight: Any
    exact_cc_shifted: Any
