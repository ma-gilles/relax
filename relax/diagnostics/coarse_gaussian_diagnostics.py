"""Dumps of the K-class significance pass.

The tree-rescore and K-class significance batch dumps (``RELAX_SIGNIFICANCE_DUMP_*``)
and the env-gated stop after a dump. None of this changes production arithmetic.
"""

import logging
import os

import numpy as np

from relax.helpers.batch_fetch import original_image_indices
from relax.helpers.env_flags import parse_env_int_set

logger = logging.getLogger(__name__)


_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET_ENV = (
    "RELAX_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET"
)


class SignificanceDumpComplete(RuntimeError):
    """Raised after an explicitly targeted coarse-significance dump is durable."""

    def __init__(self, *, dump_path: str):
        self.dump_path = str(dump_path)
        super().__init__(
            "requested RECOVAR coarse-significance target was written "
            f"(dump_path={self.dump_path})"
        )


def _maybe_stop_after_significance_dump(
    dump_path: str,
    *,
    dump_dir: str,
    target_original_indices: set[int],
    current_size: int | None,
    debug_iteration: int | None,
) -> None:
    """Stop an explicit diagnostic only after its complete target set exists."""

    if os.environ.get(_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET_ENV) != "1":
        return
    if not os.path.isfile(dump_path):
        raise RuntimeError(
            "RECOVAR significance stop target is missing its dump file: "
            f"{dump_path}"
        )
    target_iteration = os.environ.get("RELAX_SIGNIFICANCE_DUMP_ITERATION")
    iteration_suffix = (
        ""
        if not target_iteration
        else f"_it{int(debug_iteration):03d}"
    )
    current_size_label = -1 if current_size is None else int(current_size)
    expected_paths = [
        os.path.join(
            dump_dir,
            f"significance_orig{int(original_index):06d}{iteration_suffix}_cs"
            f"{current_size_label:03d}.npz",
        )
        for original_index in sorted(target_original_indices)
    ]
    missing_paths = [path for path in expected_paths if not os.path.isfile(path)]
    if missing_paths:
        logger.info(
            "RECOVAR coarse-significance stop target progress: %d/%d files written",
            len(expected_paths) - len(missing_paths),
            len(expected_paths),
        )
        return
    raise SignificanceDumpComplete(dump_path=dump_path)


def _significance_debug_dump_matches(*, current_size, debug_iteration) -> bool:
    """Return whether significance capture applies at this scoring boundary."""

    if not os.environ.get("RELAX_SIGNIFICANCE_DUMP_DIR"):
        return False
    if not parse_env_int_set("RELAX_SIGNIFICANCE_DUMP_ORIGINAL_INDICES"):
        return False
    target_current_size = os.environ.get("RELAX_SIGNIFICANCE_DUMP_CURRENT_SIZE")
    if target_current_size and (
        current_size is None or int(current_size) != int(target_current_size)
    ):
        return False
    target_iteration = os.environ.get("RELAX_SIGNIFICANCE_DUMP_ITERATION")
    if target_iteration and (
        debug_iteration is None or int(debug_iteration) != int(target_iteration)
    ):
        return False
    return True


def _maybe_dump_tree_rescore_batch(
    *,
    experiment_dataset,
    indices,
    ambiguous_rows,
    candidate_pose_ids,
    original_best_pose,
    original_best_score,
    original_second_pose,
    original_second_score,
    rescored_scores,
    rescored_winner_slot,
    shifted_candidates,
    score_weight_candidates,
    numerator_weight_candidates,
    rotation_matrices,
    translation_angles,
    n_trans,
    half_weights,
    packed_to_compact,
    projector_full,
    current_size,
    padding_factor,
    projector_max_r,
    debug_iteration,
):
    """Persist exact bounded-rescore operands for selected pass-1 particles."""

    if not _significance_debug_dump_matches(
        current_size=current_size,
        debug_iteration=debug_iteration,
    ):
        return
    target_original_indices = parse_env_int_set(
        "RELAX_SIGNIFICANCE_DUMP_ORIGINAL_INDICES"
    )
    batch_original_indices = original_image_indices(experiment_dataset, indices)
    ambiguous_original_indices = batch_original_indices[
        np.asarray(ambiguous_rows, dtype=np.int64)
    ]
    dump_dir = os.environ["RELAX_SIGNIFICANCE_DUMP_DIR"]
    os.makedirs(dump_dir, exist_ok=True)
    candidate_pose_ids = np.asarray(candidate_pose_ids, dtype=np.int32)
    original_best_pose = np.asarray(original_best_pose, dtype=np.int32)
    original_best_score = np.asarray(original_best_score, dtype=np.float32)
    original_second_pose = np.asarray(original_second_pose, dtype=np.int32)
    original_second_score = np.asarray(original_second_score, dtype=np.float32)
    rescored_scores = np.asarray(rescored_scores, dtype=np.float32)
    rescored_winner_slot = np.asarray(rescored_winner_slot, dtype=np.int32)
    for row, original_index in enumerate(ambiguous_original_indices):
        if int(original_index) not in target_original_indices:
            continue
        original_scores_by_candidate = np.where(
            candidate_pose_ids[row] == original_best_pose[row],
            original_best_score[row],
            original_second_score[row],
        ).astype(np.float32, copy=False)
        out_path = os.path.join(
            dump_dir,
            f"tree_rescore_orig{int(original_index):06d}_it"
            f"{int(debug_iteration):03d}_cs{int(current_size):03d}.npz",
        )
        np.savez_compressed(
            out_path,
            original_index=np.int64(original_index),
            candidate_pose_ids=candidate_pose_ids[row],
            candidate_rotation_ids=(candidate_pose_ids[row] // int(n_trans)),
            candidate_translation_ids=(candidate_pose_ids[row] % int(n_trans)),
            original_best_pose=original_best_pose[row],
            original_second_pose=original_second_pose[row],
            original_scores_by_candidate=original_scores_by_candidate,
            direct_texture_scores=rescored_scores[row],
            direct_texture_winner_slot=rescored_winner_slot[row],
            image_candidates=np.asarray(shifted_candidates[row], dtype=np.complex64),
            image_candidates_are_unshifted=np.asarray(True, dtype=np.bool_),
            translation_angles=np.asarray(translation_angles[row], dtype=np.float32),
            score_weight_candidates=np.asarray(
                score_weight_candidates[row], dtype=np.float32
            ),
            numerator_weight_candidates=np.asarray(
                numerator_weight_candidates[row], dtype=np.float32
            ),
            rotation_matrices=np.asarray(rotation_matrices[row], dtype=np.float32),
            half_weights=np.asarray(half_weights, dtype=np.float32),
            packed_to_compact=np.asarray(packed_to_compact, dtype=np.int32),
            projector_full=np.asarray(projector_full, dtype=np.complex64),
            current_size=np.int64(current_size),
            padding_factor=np.int64(padding_factor),
            projector_max_r=np.int64(projector_max_r),
        )


def _maybe_dump_k_class_significance_batch(
    *,
    experiment_dataset,
    indices,
    n_classes: int,
    rotations,
    translations,
    class_weight_mats,
    batch_sig_mask,
    batch_n_sig,
    hard_assignment_batch,
    class_assignment_batch,
    global_log_z,
    class_log_z_values,
    best_score,
    max_posterior,
    rotation_log_prior_padded,
    batch_translation_log_prior,
    class_log_priors,
    current_size,
    adaptive_fraction,
    max_significants,
    target_local_positions=None,
    target_scores_pre_prior_per_class=None,
    target_scores_with_prior_per_class=None,
    coarse_gaussian_shifted_corrected=None,
    coarse_gaussian_unshifted_corrected=None,
    coarse_gaussian_pixel_weight=None,
    coarse_gaussian_initial_diff2=None,
    coarse_gaussian_score_indices=None,
    translation_phase_source=None,
    relion_projector_half=None,
    relion_projector_r_max=None,
    projection_padding_factor=None,
    relion_f32_sum_weight=None,
    relion_f32_significant_weight=None,
    relion_f32_cutoff_count=None,
    score_capture_mode="intrusive_per_block_host_materialization",
    debug_iteration=None,
):
    """Env-gated debug dump for the K-class significance pass.

    File naming matches the single-class dump so existing diff tooling works.
    The payload extends the K=1 schema with per-class fields and an explicit
    ``n_classes`` scalar so the user can decode the joint candidate space.
    """

    if not _significance_debug_dump_matches(
        current_size=current_size,
        debug_iteration=debug_iteration,
    ):
        return
    dump_dir = os.environ["RELAX_SIGNIFICANCE_DUMP_DIR"]
    target_original_indices = parse_env_int_set("RELAX_SIGNIFICANCE_DUMP_ORIGINAL_INDICES")
    target_iteration = os.environ.get("RELAX_SIGNIFICANCE_DUMP_ITERATION")

    local_indices = np.asarray(indices, dtype=np.int64)
    original_indices = original_image_indices(experiment_dataset, local_indices)

    os.makedirs(dump_dir, exist_ok=True)
    n_rot = int(rotations.shape[0])
    n_trans = int(translations.shape[0])

    weights_per_class = np.stack(
        [np.asarray(mat, dtype=np.float64) for mat in class_weight_mats],
        axis=1,
    )
    sig_mask_array = np.asarray(batch_sig_mask, dtype=bool)
    sig_mask_full = sig_mask_array.reshape(
        sig_mask_array.shape[0],
        n_classes,
        n_rot * n_trans,
    )[: local_indices.shape[0]]
    class_log_z_stack = np.stack(
        [np.asarray(class_log_z, dtype=np.float64) for class_log_z in class_log_z_values],
        axis=1,
    )

    flat_indices = np.arange(n_classes * n_rot * n_trans, dtype=np.int32)
    class_indices_flat = (flat_indices // (n_rot * n_trans)).astype(np.int32)
    rot_indices_flat = ((flat_indices % (n_rot * n_trans)) // n_trans).astype(np.int32)
    trans_indices_flat = (flat_indices % n_trans).astype(np.int32)

    # Build a map from local_pos to dump-target index (row in
    # target_scores_pre_prior_per_class[c]) so we can pick the right
    # per-class raw-score slab for each saved particle.
    target_pos_to_dump_row = None
    if target_local_positions is not None:
        target_pos_to_dump_row = {int(p): row for row, p in enumerate(np.asarray(target_local_positions).tolist())}

    projector_half_per_class = None
    if relion_projector_half is not None:
        projector_half_per_class = np.stack(
            [np.asarray(value, dtype=np.complex64) for value in relion_projector_half],
            axis=0,
        )
        if projector_half_per_class.shape[0] != n_classes:
            raise ValueError(
                "RELION projector dump class count differs from significance class count: "
                f"{projector_half_per_class.shape[0]} != {n_classes}",
            )

    for local_pos, original_idx in enumerate(original_indices):
        if int(original_idx) not in target_original_indices:
            continue
        weights_full = weights_per_class[local_pos].reshape(-1)
        sig_mask = sig_mask_full[local_pos].reshape(-1)
        sig_indices = np.flatnonzero(sig_mask).astype(np.int32)
        trans_prior = None
        if batch_translation_log_prior is not None:
            prior_arr = np.asarray(batch_translation_log_prior)
            trans_prior = prior_arr if prior_arr.ndim == 1 else prior_arr[local_pos]
        rot_prior_arr = (
            np.asarray(rotation_log_prior_padded, dtype=np.float64)[:, :n_rot]
            if rotation_log_prior_padded is not None
            else None
        )

        # Per-class raw scores (pre-prior and with-prior) for this image,
        # if the engine collected them. Shape per class: (n_rot, n_trans).
        scores_pre_prior_per_class = None
        scores_with_prior_per_class = None
        if target_pos_to_dump_row is not None and target_scores_pre_prior_per_class is not None:
            dump_row = target_pos_to_dump_row.get(int(local_pos))
            if dump_row is not None:
                scores_pre_prior_per_class = np.stack(
                    [np.asarray(arr[dump_row], dtype=np.float64) for arr in target_scores_pre_prior_per_class],
                    axis=0,
                )
                scores_with_prior_per_class = np.stack(
                    [np.asarray(arr[dump_row], dtype=np.float64) for arr in target_scores_with_prior_per_class],
                    axis=0,
                )

        iteration_suffix = "" if not target_iteration else f"_it{int(debug_iteration):03d}"
        out_path = os.path.join(
            dump_dir,
            f"significance_orig{int(original_idx):06d}{iteration_suffix}_cs"
            f"{(-1 if current_size is None else int(current_size)):03d}.npz",
        )
        save_kwargs = dict(
            original_index=np.int64(original_idx),
            local_index=np.int64(local_indices[local_pos]),
            debug_iteration=np.int64(-1 if debug_iteration is None else int(debug_iteration)),
            one_based_iteration=np.int64(-1 if debug_iteration is None else int(debug_iteration)),
            current_size=np.int64(-1 if current_size is None else int(current_size)),
            adaptive_fraction=np.float64(adaptive_fraction),
            max_significants=np.int64(max_significants),
            n_classes=np.int64(n_classes),
            n_rot=np.int64(n_rot),
            n_trans=np.int64(n_trans),
            weights_full=weights_full,
            weights_per_class=weights_per_class[local_pos],
            significant_mask=sig_mask,
            significant_indices=sig_indices,
            n_significant=np.int64(batch_n_sig[local_pos]),
            hard_assignment=np.int64(hard_assignment_batch[local_pos]),
            class_assignment=np.int64(class_assignment_batch[local_pos]),
            normalization_log_z=np.float64(global_log_z[local_pos]),
            class_log_z=class_log_z_stack[local_pos],
            best_score=np.float64(best_score[local_pos]),
            max_posterior=np.float64(max_posterior[local_pos]),
            rotations=np.asarray(rotations),
            translations=np.asarray(translations),
            class_indices=class_indices_flat,
            rot_indices=rot_indices_flat,
            trans_indices=trans_indices_flat,
            class_log_priors=np.asarray(class_log_priors, dtype=np.float64),
            rotation_log_prior=(rot_prior_arr if rot_prior_arr is not None else np.empty((0,), dtype=np.float64)),
            translation_log_prior=(
                np.asarray(trans_prior, dtype=np.float64)
                if trans_prior is not None
                else np.empty((0,), dtype=np.float64)
            ),
            coarse_gaussian_unshifted_corrected=(
                np.asarray(coarse_gaussian_unshifted_corrected[local_pos])
                if coarse_gaussian_unshifted_corrected is not None
                else np.empty((0,), dtype=np.complex64)
            ),
            coarse_gaussian_shifted_corrected=(
                np.asarray(coarse_gaussian_shifted_corrected[local_pos], dtype=np.complex64)
                if coarse_gaussian_shifted_corrected is not None
                else np.empty((0,), dtype=np.complex64)
            ),
            coarse_gaussian_pixel_weight=(
                np.asarray(coarse_gaussian_pixel_weight[local_pos])
                if coarse_gaussian_pixel_weight is not None
                else np.empty((0,), dtype=np.float32)
            ),
            coarse_gaussian_initial_diff2=(
                np.asarray(coarse_gaussian_initial_diff2[local_pos])
                if coarse_gaussian_initial_diff2 is not None
                else np.empty((0,), dtype=np.float32)
            ),
            coarse_gaussian_score_indices=(
                np.asarray(coarse_gaussian_score_indices, dtype=np.int32)
                if coarse_gaussian_score_indices is not None
                else np.empty((0,), dtype=np.int32)
            ),
            translation_phase_source=(
                np.asarray(translation_phase_source)
                if translation_phase_source is not None
                else np.empty((0, 2), dtype=np.float64)
            ),
            relion_projector_half_per_class=(
                projector_half_per_class
                if projector_half_per_class is not None
                else np.empty((0,), dtype=np.complex64)
            ),
            relion_projector_r_max=np.int64(
                -1 if relion_projector_r_max is None else int(relion_projector_r_max)
            ),
            projection_padding_factor=np.int64(
                -1 if projection_padding_factor is None else int(projection_padding_factor)
            ),
            relion_f32_sum_weight=(
                np.float32(np.asarray(relion_f32_sum_weight)[local_pos])
                if relion_f32_sum_weight is not None
                else np.float32(np.nan)
            ),
            relion_f32_significant_weight=(
                np.float32(np.asarray(relion_f32_significant_weight)[local_pos])
                if relion_f32_significant_weight is not None
                else np.float32(np.nan)
            ),
            relion_f32_cutoff_count=(
                np.int32(np.asarray(relion_f32_cutoff_count)[local_pos])
                if relion_f32_cutoff_count is not None
                else np.int32(-1)
            ),
            score_capture_mode=np.asarray(str(score_capture_mode)),
        )
        if scores_pre_prior_per_class is not None:
            # Per-class raw recovar score (= -0.5 * residual in
            # `_e_step_block_scores`; differs from RELION's diff2 by the
            # per-image Xi2/2 constant which cancels in relative pose
            # comparisons). Shape (n_classes, n_rot, n_trans).
            save_kwargs["scores_pre_prior_per_class"] = scores_pre_prior_per_class
            save_kwargs["scores_with_prior_per_class"] = scores_with_prior_per_class
        np.savez_compressed(out_path, **save_kwargs)
        _maybe_stop_after_significance_dump(
            out_path,
            dump_dir=dump_dir,
            target_original_indices=target_original_indices,
            current_size=current_size,
            debug_iteration=debug_iteration,
        )


