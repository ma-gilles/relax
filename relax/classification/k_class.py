"""K-class EM orchestration on the adaptive sparse pass and the exact-local single-volume engines."""

from __future__ import annotations

import logging
import time

import jax.numpy as jnp
import numpy as np
from recovar.utils.nvtx_shim import nvtx

import relax.runtime.engine_tuning
from relax.classification.given_pose_classes import given_pose_cc_classes, given_pose_class_supports
from relax.classification.k_class_firstiter import (
    run_dense_k_class_joint_firstiter_score_probe,
    run_sparse_firstiter_global_winner_subset_pass2,
)
from relax.classification.k_class_inputs import (
    as_class_means,
    dataset_image_count,
    projection_float64_from_kwargs,
    resolve_class_log_priors,
    seed_iteration_first_class,
    seed_iteration_supports,
    translation_angle_scale_kwargs,
)
from relax.classification.k_class_results import (
    KClassEMResult,
    with_coarse_significance_diagnostics,
)
from relax.classification.k_class_sparse_pass2 import (
    infer_healpix_order_from_rotation_count,
    run_sparse_k_class_adaptive_pass2,
)
from relax.classification.probes import KClassProbe, NoProbe
from relax.fine_pass.significant_samples import ComplementSignificantSampleIndices

logger = logging.getLogger(__name__)
NVTX_DOMAIN_EM = "recovar_em"
_RELION_X_HALF_BP_FUSED_ATOMICS_ENV = "RELAX_RELION_X_HALF_BP_FUSED_ATOMICS"
_LOCAL_HOST_RESULT_PUBLICATION_ENV = "RELAX_EXACT_LOCAL_HOST_RESULT_PUBLICATION"


def _validate_bpref_device_signature_sparse_route(
    *,
    active: bool,
    n_classes: int,
) -> None:
    """Fail closed when scoped BPref capture is active with no class."""

    if bool(active) and int(n_classes) < 1:
        raise RuntimeError("active BPref device signature scope requires at least one class")


def _fine_support_stats(
    sig_sample_indices_by_class,
    *,
    n_rot_coarse: int,
    n_trans_coarse: int,
    rot_parent_map: np.ndarray,
    trans_parent_map: np.ndarray,
    n_rot_fine: int,
    n_trans_fine: int,
) -> dict[str, float]:
    if n_rot_coarse <= 0 or n_trans_coarse <= 0 or n_rot_fine <= 0 or n_trans_fine <= 0:
        raise ValueError("rotation and translation grid sizes must be positive")
    rot_parent_map_np = np.asarray(rot_parent_map, dtype=np.int64)
    if rot_parent_map_np.shape != (n_rot_fine,):
        raise ValueError(
            f"rot_parent_map must have shape ({n_rot_fine},), got {rot_parent_map_np.shape}",
        )
    if np.any(rot_parent_map_np < 0) or int(rot_parent_map_np.max(initial=-1)) >= n_rot_coarse:
        raise ValueError("rot_parent_map values must be in [0, n_rot_coarse)")
    trans_parent_map_np = np.asarray(trans_parent_map, dtype=np.int64)
    if trans_parent_map_np.shape != (n_trans_fine,):
        raise ValueError(
            f"trans_parent_map must have shape ({n_trans_fine},), got {trans_parent_map_np.shape}",
        )
    if np.any(trans_parent_map_np < 0) or int(trans_parent_map_np.max(initial=-1)) >= n_trans_coarse:
        raise ValueError("trans_parent_map values must be in [0, n_trans_coarse)")
    fine_children_per_coarse = np.bincount(rot_parent_map_np, minlength=n_rot_coarse)
    fine_trans_children_per_coarse = np.bincount(trans_parent_map_np, minlength=n_trans_coarse)

    rotation_counts = []
    pose_counts = []
    for samples_by_image in sig_sample_indices_by_class:
        for samples in samples_by_image:
            if samples is None:
                rotation_counts.append(n_rot_fine)
                pose_counts.append(n_rot_fine * n_trans_fine)
                continue
            if isinstance(samples, ComplementSignificantSampleIndices):
                excluded = np.asarray(samples.excluded_indices, dtype=np.int64).reshape(-1)
                if excluded.size == 0:
                    rotation_counts.append(n_rot_fine)
                    pose_counts.append(n_rot_fine * n_trans_fine)
                    continue
                if np.any(excluded < 0) or int(excluded.max(initial=-1)) >= n_rot_coarse * n_trans_coarse:
                    raise ValueError("excluded sample ids must index the coarse rotation/translation grid")
                excluded_rot_ids = excluded // n_trans_coarse
                excluded_trans_ids = excluded % n_trans_coarse
                excluded_per_rot = np.bincount(excluded_rot_ids, minlength=n_rot_coarse)
                significant_coarse_rot = np.flatnonzero(excluded_per_rot < n_trans_coarse)
                excluded_fine_pose_count = int(
                    np.sum(
                        fine_children_per_coarse[excluded_rot_ids]
                        * fine_trans_children_per_coarse[excluded_trans_ids],
                        dtype=np.int64,
                    ),
                )
                rotation_counts.append(int(fine_children_per_coarse[significant_coarse_rot].sum()))
                pose_counts.append(int(n_rot_fine * n_trans_fine) - excluded_fine_pose_count)
                continue
            sample_ids = np.asarray(samples, dtype=np.int64).ravel()
            if sample_ids.size == 0:
                rotation_counts.append(0)
                pose_counts.append(0)
                continue
            sample_ids = np.unique(sample_ids)
            coarse_rot_ids = sample_ids // n_trans_coarse
            coarse_trans_ids = sample_ids % n_trans_coarse
            if np.any(coarse_rot_ids < 0) or int(coarse_rot_ids.max(initial=-1)) >= n_rot_coarse:
                raise ValueError("significant sample ids must index the coarse rotation/translation grid")
            rotation_counts.append(int(fine_children_per_coarse[np.unique(coarse_rot_ids)].sum()))
            pose_counts.append(
                int(
                    np.sum(
                        fine_children_per_coarse[coarse_rot_ids]
                        * fine_trans_children_per_coarse[coarse_trans_ids],
                    ),
                ),
            )

    rotation_counts_np = np.asarray(rotation_counts, dtype=np.float64)
    pose_counts_np = np.asarray(pose_counts, dtype=np.float64)
    if rotation_counts_np.size == 0:
        rotation_counts_np = np.asarray([0.0], dtype=np.float64)
        pose_counts_np = np.asarray([0.0], dtype=np.float64)
    rotation_median = float(np.median(rotation_counts_np))
    rotation_mean = float(np.mean(rotation_counts_np))
    rotation_max = float(np.max(rotation_counts_np))
    pose_median = float(np.median(pose_counts_np))
    pose_mean = float(np.mean(pose_counts_np))
    pose_max = float(np.max(pose_counts_np))
    n_fine_poses = float(n_rot_fine * n_trans_fine)
    return {
        "entries": float(rotation_counts_np.size),
        "rotation_median": rotation_median,
        "rotation_mean": rotation_mean,
        "rotation_max": rotation_max,
        "rotation_median_fraction": rotation_median / float(n_rot_fine),
        "rotation_mean_fraction": rotation_mean / float(n_rot_fine),
        "rotation_max_fraction": rotation_max / float(n_rot_fine),
        "pose_median": pose_median,
        "pose_mean": pose_mean,
        "pose_max": pose_max,
        "pose_total": int(np.sum(pose_counts, dtype=np.int64)),
        "pose_median_fraction": pose_median / n_fine_poses,
        "pose_mean_fraction": pose_mean / n_fine_poses,
        "pose_max_fraction": pose_max / n_fine_poses,
    }


def _supports_coarse_parents(supports_by_class, n_images, n_coarse_rot, n_coarse_trans):
    """The coarse parents any class's pass-2 rows descend from, or None for every parent."""

    from relax.fine_pass.resident_significance import significant_coarse_parents

    parents = []
    for support in supports_by_class:
        class_parents = significant_coarse_parents(
            support, n_images=n_images, n_coarse_rot=n_coarse_rot, n_coarse_trans=n_coarse_trans
        )
        if class_parents is None:
            return None
        parents.append(class_parents)
    return np.unique(np.concatenate(parents)) if parents else None


def run_dense_k_class_em_adaptive(
    experiment_dataset,
    means,
    mean_variance,
    noise_variance,
    coarse_rotations,
    coarse_translations,
    fine_rotations,
    fine_translations,
    rot_parent_map,
    trans_parent_map,
    disc_type: str,
    *,
    class_log_priors=None,
    accumulate_noise: bool = False,
    adaptive_fraction: float = 0.999,
    max_significants: int = -1,
    significance_image_batch_size: int | None = None,
    significance_rotation_block_size: int | None = None,
    significance_pad_final_image_batch: bool = False,
    coarse_current_size: int | None = None,
    fine_current_size: int | None = None,
    coarse_healpix_order: int | None = None,
    coarse_rotation_ids=None,
    oversampling_order: int | None = None,
    coarse_translation_log_prior=None,
    coarse_rotation_log_prior=None,
    coarse_class_rotation_log_prior=None,
    relion_fine_mstep_prune: bool = False,
    firstiter_cc_pass2_only_best_coarse: bool = False,
    coarse_relion_projector_texture_interp: bool | None = None,
    relion_projector_half=None,
    relion_projector_r_max: int | None = None,
    fine_mstep_rotations_override=None,
    return_best_pose_details: bool = False,
    bpref_device_signature_active: bool = False,
    debug_iteration: int | None = None,
    pass2_use_float64_scoring: bool | None = None,
    pass2_use_float64_projections: bool | None = None,
    coarse_translation_phase_source=None,
    coarse_engine: str = "auto",
    image_seed_classes=None,
    fill_fine_rows=None,
    given_supports=None,
    given_image_translations=None,
    zero_oversampling_coarse_state_for_classes: bool = False,
    probe: KClassProbe | None = None,
    engine_tuning: relax.runtime.engine_tuning.EngineTuning,
    scoring_overrides,
    **engine_kwargs,
) -> KClassEMResult:
    """K-class adaptive 2-pass EM: coarse pass-1 significance + sparse fine pass 2.

    Mirrors RELION's adaptive 2-pass logic in
    ``ml_optimiser.cpp::expectationOneParticle`` (line 5022).  Pass-1 evaluates
    the coarse grid and keeps, per particle, the coarse samples retaining
    ``adaptive_fraction`` of the posterior mass.  Pass-2 runs on the
    device-resident sparse engine over the oversampled children of those
    samples only. ``sparse_pass2=False`` (the removed dense pass 2) is refused.

    Parameters
    ----------
    coarse_rotations, coarse_translations : np.ndarray
        Pass-1 coarse pose grids.
    fine_rotations, fine_translations : np.ndarray
        Pass-2 fine (oversampled) pose grids.
    fine_mstep_rotations_override : np.ndarray or None
        Optional pass-2 rotations used only for M-step backprojection. Score
        projections, posterior selection, and reported best poses continue to
        use ``fine_rotations``. Supported by sparse pass 2 only.
    fill_fine_rows : callable or None
        For a deferred fine grid (:class:`relax.sampling.oversampling.DeferredFineRows`):
        called with the significant coarse parents before the sparse pass 2, or
        with None (every parent) before the full-grid or firstiter-CC routes read the fine rows.
    rot_parent_map : np.ndarray of int, shape (n_rot_fine,)
        Index into ``coarse_rotations`` for each fine rotation.
    trans_parent_map : np.ndarray of int, shape (n_trans_fine,)
        Index into ``coarse_translations`` for each fine translation.
    coarse_current_size, fine_current_size : int or None
        Per-pass Fourier window radii.  Pass-1 typically uses a smaller
        ``coarse_current_size`` per RELION's ``image_coarse_size`` semantics.
        When ``None``, both passes use the same ``current_size``.
    given_supports : list of np.ndarray or None
        Each image's coarse samples, given by the caller instead of found by pass 1
        (RELION ``--skip_align``: one sample per image, its stored pose,
        :func:`relax.refinement.given_poses.given_pose_grids`). Pass 1 is skipped
        and pass 2 scores every class on these samples, so the posterior is over the
        classes alone.
    zero_oversampling_coarse_state_for_classes : bool
        At ``oversampling_order`` 0 with K > 1, hand pass 2 the coarse pass's sum of weights, Pmax and
        winner (class and pose), as RELION's GPU code keeps them (acc_ml_optimiser_impl.h:1984-1988,
        docs/math/zero_oversampling.md; relax#73). K = 1 always does. Off, K > 1 normalizes pass 2 on the
        retained samples.
    given_image_translations : np.ndarray or None
        ``[n_images, 2]`` with ``given_supports``: each image's own translation sample
        (pixels), applied to its images by pass 2 (``image_translations`` of the resident
        pass); the translation grids are then the one zero translation.
    probe : KClassProbe or None
        The diagnostics that observe the pass (:mod:`relax.classification.probes`: score dumps, the support audit,
        the first-iteration winner summary); None is the :class:`~relax.classification.probes.NoProbe`, which
        observes nothing. No result depends on it.
    engine_tuning : EngineTuning
        How the engines execute (``relax.runtime.engine_tuning``; the command reads it from the environment).
    scoring_overrides : ScoringOverrides
        The diagnostic overrides of pass 1 that change values (``relax.runtime.scoring_overrides``; the command
        reads them from the environment).
    coarse_healpix_order, oversampling_order : int or None
        RELION sampling metadata for sparse pass-2 diagnostics.  When omitted,
        the values are inferred from exact HEALPix grid sizes for compatibility
        with older callers.
    coarse_*_log_prior : optional priors used only at pass-1.  ``engine_kwargs``
        carries the priors used at pass-2.
    significance_pad_final_image_batch : bool
        Pad the last coarse image batch to the full batch size, so a caller whose
        image count changes every iteration (VDAM's subsets) reuses one pass-1
        executable. Splitting only the image axis leaves every score unchanged.
    coarse_relion_projector_texture_interp : bool or None
        Explicitly select the supplied-PPref coarse projector.  ``None``
        is RELION texture interpolation, the strict-parity default.
    """
    probe = NoProbe() if probe is None else probe
    # How the resident pass 2 executes (EngineTuning); it travels with the pass-2 keyword arguments.
    engine_kwargs["engine_tuning"] = engine_tuning
    # The run's value-changing overrides travel with it too (the resident pass 2 reads its fine execution order).
    engine_kwargs["scoring_overrides"] = scoring_overrides
    if "relion_exact_coarse" in engine_kwargs:
        raise TypeError(
            "relion_exact_coarse was removed on 2026-10-02: pass 1 always scores RELION's exact coarse operands"
        )
    # Lazy import to avoid the formatter stripping a top-level name that is
    # only referenced inside this function.
    from relax.sampling.symmetry import canonicalize_rotational_symmetry
    from relax.scoring.significance import compute_k_class_significance_batched

    if not engine_kwargs.pop("sparse_pass2", True):
        raise RuntimeError(
            "the dense adaptive pass 2 was removed on 2026-10-03 (with RELAX_K1_DENSE_PASS2 and "
            "RELAX_K_CLASS_DENSE_PASS2): pass 2 runs only on the device-resident sparse engine; "
            "do not pass sparse_pass2=False"
        )
    symmetry_label = canonicalize_rotational_symmetry(
        engine_kwargs.get("symmetry_label", "C1")
    )
    non_c1_symmetry = symmetry_label != "C1"
    if non_c1_symmetry and coarse_engine != "gemm_dense" and not bool(
        engine_kwargs.get("mstep_relion_x_half", False)
    ):
        raise RuntimeError(
            f"{symmetry_label} adaptive reconstruction requires RELION x-half BPref "
            "accumulation; full/native-half M-step routes are unsupported"
        )
    if non_c1_symmetry and oversampling_order is None:
        raise ValueError(
            f"{symmetry_label} adaptive K-class refinement requires explicit "
            "oversampling_order; symmetry-boundary children do not form a "
            "complete reduced fine grid, so the order cannot be inferred from "
            "their count"
        )

    overall_t0 = time.time()
    if relion_projector_half is not None:
        # Keep the supplied PPref available to fine pass 2 after consuming it
        # as an explicit coarse-pass argument above.
        engine_kwargs["relion_projector_half"] = relion_projector_half
        engine_kwargs["relion_projector_r_max"] = relion_projector_r_max
    logger.info(
        "Adaptive K-class coarse projector: supplied_ppref=%s texture_interp=%s",
        relion_projector_half is not None,
        coarse_relion_projector_texture_interp,
    )
    means_array = as_class_means(means)
    n_classes = int(means_array.shape[0])
    log_priors = resolve_class_log_priors(n_classes, class_log_priors)
    # Per-optics-group noise rows score every class (coarse significance, then the
    # device-resident pass 2); images on another grid than the reference carry their
    # scale in the rotations and fill each class's backprojector at the reference size.

    coarse_rotations_np = np.asarray(coarse_rotations)
    coarse_translations_np = np.asarray(coarse_translations)
    fine_rotations_np = np.asarray(fine_rotations)
    fine_mstep_rotations_np = (
        None
        if fine_mstep_rotations_override is None
        else np.asarray(fine_mstep_rotations_override)
    )
    fine_translations_source_np = np.asarray(fine_translations)
    fine_translations_np = np.asarray(fine_translations_source_np)
    rot_parent_map_np = np.asarray(rot_parent_map, dtype=np.int64)
    trans_parent_map_np = np.asarray(trans_parent_map, dtype=np.int64)

    n_rot_coarse = int(coarse_rotations_np.shape[0])
    n_trans_coarse = int(coarse_translations_np.shape[0])
    n_rot_fine = int(fine_rotations_np.shape[0])
    n_trans_fine = int(fine_translations_np.shape[0])
    # RELION constructs oversampled translations in host RFLOAT (double in
    # the deployed build) and rounds only the CUDA translation angle to
    # float32. Preserve that source precision for K=1 sparse pass 2 while
    # retaining the established float32 pose/prior/output representation.
    sparse_fine_translations_np = (
        fine_translations_source_np if n_classes == 1 else fine_translations_np
    )

    if fine_mstep_rotations_np is not None and fine_mstep_rotations_np.shape != fine_rotations_np.shape:
        raise ValueError(
            "fine_mstep_rotations_override must match fine_rotations shape: "
            f"{fine_mstep_rotations_np.shape} vs {fine_rotations_np.shape}",
        )

    def _resolved_coarse_healpix_order() -> int:
        if coarse_healpix_order is not None:
            return int(coarse_healpix_order)
        return infer_healpix_order_from_rotation_count(n_rot_coarse, symmetry_label)

    def _resolved_oversampling_order() -> int:
        if oversampling_order is not None:
            return max(0, int(oversampling_order))
        return max(
            0,
            infer_healpix_order_from_rotation_count(n_rot_fine, symmetry_label) - _resolved_coarse_healpix_order(),
        )

    if rot_parent_map_np.shape != (n_rot_fine,):
        raise ValueError(
            f"rot_parent_map must have shape ({n_rot_fine},), got {rot_parent_map_np.shape}",
        )
    if trans_parent_map_np.shape != (n_trans_fine,):
        raise ValueError(
            f"trans_parent_map must have shape ({n_trans_fine},), got {trans_parent_map_np.shape}",
        )
    if int(rot_parent_map_np.max(initial=-1)) >= n_rot_coarse:
        raise ValueError("rot_parent_map values must be < n_rot_coarse")
    if int(trans_parent_map_np.max(initial=-1)) >= n_trans_coarse:
        raise ValueError("trans_parent_map values must be < n_trans_coarse")
    n_images = dataset_image_count(experiment_dataset)
    if coarse_engine not in {"auto", "gemm_hybrid", "gemm_dense"}:
        raise ValueError(f"unknown coarse engine {coarse_engine!r}")
    if coarse_engine == "gemm_hybrid":
        if relion_projector_half is None:
            raise ValueError("gemm_hybrid requires the RELION projector")
        if pass2_use_float64_scoring or pass2_use_float64_projections or engine_kwargs.get("use_float64_scoring"):
            raise ValueError("gemm_hybrid requires float32 production arithmetic")
        engine_kwargs["mstep_relion_x_half"] = True
    if fill_fine_rows is not None and (coarse_engine == "gemm_dense" or firstiter_cc_pass2_only_best_coarse):
        # Only the sparse pass 2 reads just the significant parents' rows.
        fill_fine_rows(None)
        fill_fine_rows = None
    if coarse_engine == "gemm_dense":
        logger.warning(
            "gemm_dense is experimental: it evaluates the full pose grid without pruning; "
            "RELION-parity quality and end-to-end speed are not qualified"
        )
        if bool(pass2_use_float64_scoring) or bool(pass2_use_float64_projections) or bool(engine_kwargs.get("use_float64_scoring")):
            raise ValueError("gemm_dense is a float32 production engine")
        complete_support = [[None] * n_images for _ in range(n_classes)]
        dense_kwargs = dict(engine_kwargs)
        if "current_size" not in dense_kwargs and fine_current_size is not None:
            dense_kwargs["current_size"] = fine_current_size
        # The selected full-grid engine computes its own joint normalizer and
        # accumulates native x-half BPref volumes. These legacy coarse fields
        # describe the caller's adaptive route, which this branch bypasses.
        for field in (
            "relion_f32_normalization_sum_weight",
            "relion_coarse_hard_assignment",
            "relion_coarse_max_posterior",
            "relion_coarse_class_assignment",
        ):
            dense_kwargs.pop(field, None)
        dense_kwargs["mstep_relion_x_half"] = True
        dense_kwargs["dense_gemm_full_grid"] = True
        dense_kwargs["relion_firstiter_score_mode"] = (
            "normalized_cc" if firstiter_cc_pass2_only_best_coarse else dense_kwargs.get("relion_firstiter_score_mode", "gaussian")
        )
        dense_result = run_sparse_k_class_adaptive_pass2(
            experiment_dataset, means_array, mean_variance, noise_variance,
            coarse_rotations_np, coarse_translations_np, fine_rotations_np,
            fine_mstep_rotations_np, rot_parent_map_np, sparse_fine_translations_np,
            trans_parent_map_np, complete_support, disc_type,
            class_log_priors=log_priors, accumulate_noise=accumulate_noise,
            return_best_pose_details=return_best_pose_details,
            coarse_healpix_order=_resolved_coarse_healpix_order(),
            oversampling_order=_resolved_oversampling_order(),
            random_perturbation=0.0, engine_kwargs=dense_kwargs,
        )
        from relax.runtime.engine_record import record_coarse_engine_call

        score_mode = dense_kwargs["relion_firstiter_score_mode"]
        record_coarse_engine_call(
            requested="gemm_dense", resolved="gemm_dense", strategy="dense", fine_engine=None,
            images=n_images, classes=n_classes, rotations=n_rot_fine, translations=n_trans_fine,
            coarse_candidates_per_image=n_classes * n_rot_coarse * n_trans_coarse,
            fine_candidates_per_image=n_classes * n_rot_fine * n_trans_fine,
            evaluated_fine_candidates_total=n_images * n_classes * n_rot_fine * n_trans_fine,
            selected_fine_candidates_total=(n_images if score_mode == "normalized_cc"
                                            else n_images * n_classes * n_rot_fine * n_trans_fine),
            current_size=int(fine_current_size or engine_kwargs.get("current_size") or experiment_dataset.image_shape[0]),
            reconstruction_current_size=int(engine_kwargs.get("reconstruction_current_size") or fine_current_size
                                            or engine_kwargs.get("current_size") or experiment_dataset.image_shape[0]),
            score_mode=score_mode,
            posterior_policy="cc_winner" if score_mode == "normalized_cc" else "gaussian",
            pruned=False,
            precision={"score": "float32", "projection": "float32", "mstep": "float32"},
        )
        return dense_result._replace(significant_counts=jnp.full(
            (n_images,), n_classes * n_rot_coarse * n_trans_coarse, jnp.int32
        ))
    image_batch_size = int(engine_kwargs.get("image_batch_size", 500))
    rotation_block_size = int(engine_kwargs.get("rotation_block_size", 5000))
    sig_ibs = int(significance_image_batch_size or image_batch_size)
    sig_rbs = int(significance_rotation_block_size or rotation_block_size)

    # Pass-1 priors fall back to pass-2 priors when not supplied separately.
    if coarse_translation_log_prior is None:
        coarse_translation_log_prior = engine_kwargs.get("translation_log_prior")
    if coarse_rotation_log_prior is None:
        coarse_rotation_log_prior = engine_kwargs.get("rotation_log_prior")
    if coarse_class_rotation_log_prior is None:
        coarse_class_rotation_log_prior = engine_kwargs.get("class_rotation_log_prior")

    pass1_rotation_prior = (
        coarse_class_rotation_log_prior if coarse_class_rotation_log_prior is not None else coarse_rotation_log_prior
    )

    coarse_class_assignments = None
    reuse_zero_oversampling_coarse_state = bool(
        (
            n_classes == 1
            # Classes take RELION's route on request; a seed iteration and given poses have no joint pass 1.
            or (zero_oversampling_coarse_state_for_classes and image_seed_classes is None and given_supports is None)
        )
        and _resolved_oversampling_order() == 0
        and engine_kwargs.get("mstep_relion_x_half", False)
        and not firstiter_cc_pass2_only_best_coarse
        and not engine_kwargs.get("relion_firstiter_winner_take_all", False)
        and engine_kwargs.get("relion_firstiter_score_mode", "gaussian") == "gaussian"
        and not engine_kwargs.get("use_float64_scoring", False)
        and not (pass2_use_float64_scoring if pass2_use_float64_scoring is not None
                 else engine_kwargs.get("use_float64_scoring", False))
    )
    significant_counts_for_result = None
    # RELION's seed iteration (Class3D from one reference): pass 1 scores the first copy of the reference and
    # each image's support goes to its random class (k_class_inputs.seed_iteration_supports).
    significance_means, significance_log_priors = means_array, log_priors
    significance_rotation_prior, significance_projector_half = pass1_rotation_prior, relion_projector_half
    if image_seed_classes is not None:
        # With --firstiter_cc RELION scores its first iteration against class 0 alone (every image's seed is 0)
        # and seeds the random classes at iteration 2 (ml_optimiser.cpp:4626, :4880-4898).
        if n_classes == 1:
            raise ValueError("a seed iteration is a K-class pass with coarse significance")
        significance_means, class_prior = seed_iteration_first_class(
            means_array, pass1_rotation_prior if np.ndim(pass1_rotation_prior) == 2 else None
        )
        if class_prior is not None:
            significance_rotation_prior = class_prior
        significance_log_priors = log_priors[:1]
        if relion_projector_half is not None:
            significance_projector_half = seed_iteration_first_class(relion_projector_half)[0]
    coarse_significance_support_audit = None
    coarse_actual_backend = None
    pass1_t0 = time.time()
    if given_supports is not None:
        coarse_result = None
        sig_sample_indices_by_class, significant_counts_for_result, given_classes = given_pose_class_supports(
            given_supports, n_classes=n_classes, n_images=n_images, image_seed_classes=image_seed_classes,
            firstiter_cc=firstiter_cc_pass2_only_best_coarse, n_coarse_rot=n_rot_coarse, n_coarse_trans=n_trans_coarse,
        )
        if firstiter_cc_pass2_only_best_coarse:
            coarse_class_assignments = given_classes
    elif firstiter_cc_pass2_only_best_coarse:
        # RELION firstiter_cc branch: restrict pass-2 to children of each
        # class's per-class coarse-best pose, then gate by the global winning
        # class. This is the production default-GUI parity path for iter 1.
        coarse_probe_kwargs = dict(engine_kwargs)
        coarse_probe_kwargs.pop("rotation_translation_mask", None)
        coarse_probe_kwargs.pop("class_rotation_translation_mask", None)
        # The coarse probe is score-only.  RELION's separate model-coordinate
        # cutoff is consumed by fine-pass BPref, not by this scorer wrapper.
        coarse_probe_kwargs.pop("reconstruction_current_size", None)
        coarse_probe_kwargs.pop("wsum_current_size", None)
        coarse_probe_kwargs["image_batch_size"] = sig_ibs
        coarse_probe_kwargs["rotation_block_size"] = sig_rbs
        coarse_probe_kwargs["relion_firstiter_score_mode"] = "normalized_cc"
        coarse_probe_kwargs["relion_firstiter_winner_take_all"] = True
        coarse_probe_kwargs["coarse_relion_projector_texture_interp"] = (
            coarse_relion_projector_texture_interp
        )
        coarse_probe_kwargs["current_size"] = (
            coarse_current_size if coarse_current_size is not None else fine_current_size
        )
        coarse_probe_kwargs["debug_iteration"] = debug_iteration
        coarse_probe_kwargs["coarse_healpix_order"] = _resolved_coarse_healpix_order()
        coarse_probe_kwargs["coarse_rotation_ids"] = coarse_rotation_ids
        if n_classes == 1 and coarse_translation_phase_source is not None:
            # RELION builds CUDA translation phases from host RFLOAT
            # coordinates. Keep the established float32 pose/prior grid, but
            # do not derive strict K=1 score phases from that rounded copy.
            coarse_probe_kwargs["translation_phase_source"] = (
                coarse_translation_phase_source
            )
        probe_means = means_array
        if image_seed_classes is not None:
            probe_means = significance_means
            if coarse_probe_kwargs.get("relion_projector_half") is not None:
                coarse_probe_kwargs["relion_projector_half"] = significance_projector_half
            class_rotation_prior = coarse_probe_kwargs.get("class_rotation_log_prior")
            if class_rotation_prior is not None and np.ndim(class_rotation_prior) == 2:
                coarse_probe_kwargs["class_rotation_log_prior"] = seed_iteration_first_class(
                    means_array, class_rotation_prior
                )[1]
        with probe.coarse_score_dump_label():
            with nvtx.annotate("kclass.adaptive.coarse_probe", color="yellow", domain=NVTX_DOMAIN_EM):
                coarse_result = run_dense_k_class_joint_firstiter_score_probe(
                    experiment_dataset,
                    probe_means,
                    noise_variance,
                    coarse_rotations_np,
                    coarse_translations_np,
                    engine_kwargs=coarse_probe_kwargs,
                    probe=probe,
                    scoring_overrides=scoring_overrides,
                )
        coarse_actual_backend = coarse_result.coarse_score_backend
        # ``per_class_hard_assignments[k, i]`` is class k's best coarse pose
        # (independently scored per class). For each class, restrict pass-2
        # to that single pose's children.
        coarse_per_class_assn = np.asarray(coarse_result.per_class_hard_assignments, dtype=np.int64)
        # Preserve the K-class assignment from the coarse diagnostic probe.
        coarse_class_assignments = np.asarray(
            coarse_result.class_assignments,
            dtype=np.int32,
        )
        sig_sample_indices_by_class = [
            [np.array([int(coarse_per_class_assn[k, i])], dtype=np.int32) for i in range(n_images)]
            for k in range(int(coarse_per_class_assn.shape[0]))
        ]
        if image_seed_classes is not None:
            sig_sample_indices_by_class = seed_iteration_supports(
                sig_sample_indices_by_class[0], image_seed_classes, n_classes
            )
            coarse_class_assignments = np.asarray(image_seed_classes, dtype=np.int32)
            # Each image's coarse evidence belongs to its class; a class it is not scored against has none.
            probe_evidence = np.asarray(coarse_result.class_log_evidence)[0]
            scored = np.arange(n_classes)[:, None] == coarse_class_assignments[None, :]
            coarse_result = coarse_result._replace(
                class_log_evidence=np.where(scored, probe_evidence[None, :], -np.inf).astype(probe_evidence.dtype),
                class_assignments=coarse_class_assignments,
            )
        # RELION one-hot encodes the joint class/pose coarse posterior in
        # firstiter_cc and serializes one retained sample, even though the
        # per-class child lists above remain convenient for pass-2 routing.
        significant_counts_for_result = np.ones(n_images, dtype=np.int32)
    else:
        sig_kwargs = dict(
            adaptive_fraction=adaptive_fraction,
            max_significants=max_significants,
            image_batch_size=sig_ibs,
            rotation_block_size=sig_rbs,
            current_size=(coarse_current_size if coarse_current_size is not None else fine_current_size),
            image_current_size=fine_current_size,
            score_with_masked_images=engine_kwargs.get("score_with_masked_images", True),
            rotation_log_prior=significance_rotation_prior,
            translation_log_prior=coarse_translation_log_prior,
            image_corrections=engine_kwargs.get("image_corrections"),
            scale_corrections=engine_kwargs.get("scale_corrections"),
            image_pre_shifts=engine_kwargs.get("image_pre_shifts"),
            half_spectrum_scoring=engine_kwargs.get("half_spectrum_scoring", False),
            projection_padding_factor=engine_kwargs.get("projection_padding_factor", 1),
            square_window=engine_kwargs.get("square_window", False),
            window_at_box=bool(engine_kwargs.get("window_at_box", False)),
            use_float64_scoring=engine_kwargs.get("use_float64_scoring", False),
            use_float64_projections=projection_float64_from_kwargs(engine_kwargs),
            score_mode=engine_kwargs.get("relion_firstiter_score_mode", "gaussian"),
            relion_projector_half=significance_projector_half,
            relion_projector_r_max=relion_projector_r_max,
            relion_projector_texture_interp=coarse_relion_projector_texture_interp,
            debug_iteration=debug_iteration,
            probe=probe,
            translation_phase_source=coarse_translation_phase_source,
            tree_rescore_max_margin=engine_kwargs.get("firstiter_cc_tree_rescore_max_margin"),
            optics_group_ids=engine_kwargs.get("optics_group_ids"),
            pad_final_image_batch=bool(significance_pad_final_image_batch),
            firstiter_cc_support=engine_kwargs.get("firstiter_cc_support", "relion"),
            nyquist_column_counting=engine_kwargs.get("nyquist_column_counting", "relion"),
            # Pass 1 takes the same quantized physical windows as pass 2, so a new current size
            # reuses its class's programs instead of compiling new ones.
            stable_fourier_window_shapes=engine_tuning.stable_windows,
        )
        if reuse_zero_oversampling_coarse_state:
            sig_kwargs["return_relion_f32_normalization"] = True

        with nvtx.annotate("kclass.adaptive.significance", color="orange", domain=NVTX_DOMAIN_EM):
            pass1_result = compute_k_class_significance_batched(
                experiment_dataset,
                noise_variance,
                coarse_rotations_np,
                coarse_translations_np,
                class_log_priors=significance_log_priors,
                engine_tuning=engine_tuning,
                scoring_overrides=scoring_overrides,
                **sig_kwargs,
                **({"symmetry_label": engine_kwargs["symmetry_label"]} if engine_kwargs.get("symmetry_label", "C1") != "C1" else {}),
                **translation_angle_scale_kwargs(engine_kwargs),
            )
        sig_sample_indices_by_class = pass1_result.significant_sample_indices
        if image_seed_classes is not None:
            sig_sample_indices_by_class = seed_iteration_supports(
                sig_sample_indices_by_class[0], image_seed_classes, n_classes
            )
        significant_counts_for_result = np.asarray(
            pass1_result.stats.significant_cutoff_counts,
            dtype=np.int32,
        )
        coarse_actual_backend = pass1_result.stats.executed_coarse_backend
        coarse_significance_support_audit = pass1_result.stats.coarse_significance_support_audit
    if coarse_engine == "gemm_hybrid" and coarse_actual_backend not in {"gemm_macro", "exact_cc_gemm"}:
        raise RuntimeError(f"gemm_hybrid selected but coarse scorer executed {coarse_actual_backend!r}")
    pass1_s = time.time() - pass1_t0

    def _with_significant_counts(result: KClassEMResult) -> KClassEMResult:
        if significant_counts_for_result is not None:
            result = result._replace(
                significant_counts=jnp.asarray(
                    significant_counts_for_result,
                    dtype=jnp.int32,
                ),
            )
        result = with_coarse_significance_diagnostics(
            result,
            support_audit=coarse_significance_support_audit,
        )
        if coarse_engine == "gemm_hybrid":
            from relax.runtime.engine_record import record_coarse_engine_call

            support = sig_sample_indices_by_class
            if firstiter_cc_pass2_only_best_coarse:
                winners = np.asarray(coarse_class_assignments, dtype=np.int32)
                support = [
                    [samples if int(winners[i]) == class_id else np.empty(0, np.int32)
                     for i, samples in enumerate(by_image)]
                    for class_id, by_image in enumerate(sig_sample_indices_by_class)
                ]
            evaluated = _fine_support_stats(
                support,
                n_rot_coarse=n_rot_coarse, n_trans_coarse=n_trans_coarse,
                rot_parent_map=rot_parent_map_np, trans_parent_map=trans_parent_map_np,
                n_rot_fine=n_rot_fine, n_trans_fine=n_trans_fine,
            )["pose_total"]
            mode = "normalized_cc" if firstiter_cc_pass2_only_best_coarse else "gaussian"
            record_coarse_engine_call(
                requested="gemm_hybrid", resolved="gemm_hybrid", strategy="hybrid",
                fine_engine="resident", images=n_images, classes=n_classes,
                rotations=n_rot_coarse, translations=n_trans_coarse,
                coarse_candidates_per_image=n_classes * n_rot_coarse * n_trans_coarse,
                fine_candidates_per_image=n_classes * n_rot_fine * n_trans_fine,
                evaluated_fine_candidates_total=evaluated,
                # The resident fine posterior owns a further adaptive M-step
                # cutoff and currently does not publish its retained cell count.
                selected_fine_candidates_total=None,
                current_size=int(coarse_current_size or fine_current_size
                                 or engine_kwargs.get("current_size") or experiment_dataset.image_shape[0]),
                reconstruction_current_size=int(engine_kwargs.get("reconstruction_current_size")
                                                or fine_current_size or engine_kwargs.get("current_size")
                                                or experiment_dataset.image_shape[0]),
                score_mode=mode,
                posterior_policy="cc_winner" if mode == "normalized_cc" else "gaussian",
                pruned=True,
                precision={"score": "float32", "projection": "float32", "mstep": "float32"},
            )
        return result

    mask_t0 = time.time()
    pass2_kwargs = dict(engine_kwargs)
    if reuse_zero_oversampling_coarse_state:
        pass2_kwargs["relion_f32_normalization_sum_weight"] = pass1_result.stats.relion_f32_sum_weight
        pass2_kwargs["relion_coarse_max_posterior"] = pass1_result.stats.relion_f32_max_posterior
        pass2_kwargs["relion_coarse_hard_assignment"] = pass1_result.hard_assignment
        if n_classes > 1:
            pass2_kwargs["relion_coarse_class_assignment"] = pass1_result.class_assignment
    if pass2_use_float64_scoring is not None:
        pass2_kwargs["use_float64_scoring"] = bool(pass2_use_float64_scoring)
    if pass2_use_float64_projections is not None:
        pass2_kwargs["use_float64_projections"] = bool(pass2_use_float64_projections)
    pass2_kwargs["relion_fine_mstep_prune"] = bool(relion_fine_mstep_prune)
    if given_image_translations is not None:
        if given_supports is None:
            raise ValueError("given image translations go with the given supports")
        pass2_kwargs["image_translations"] = np.asarray(given_image_translations, dtype=np.float64)
    # Pass 2 reads each image's coarse significant samples, not a fine-grid mask.
    pass2_kwargs.pop("rotation_translation_mask", None)
    if firstiter_cc_pass2_only_best_coarse:
        pass2_kwargs.pop("rotation_log_prior", None)
        pass2_kwargs.pop("class_rotation_log_prior", None)
        pass2_kwargs.pop("translation_log_prior", None)
    if "current_size" not in pass2_kwargs and fine_current_size is not None:
        pass2_kwargs["current_size"] = fine_current_size

    # The diagnostic dump and arm the command read (EngineTuning.resident_diagnostics names the set ones).
    device_signature_configured = "RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR" in engine_tuning.resident_diagnostics
    fused_atomic_env_enabled = _RELION_X_HALF_BP_FUSED_ATOMICS_ENV in engine_tuning.resident_diagnostics
    fused_atomic_diagnostic_requested = bool(
        fused_atomic_env_enabled
        and (bpref_device_signature_active or not device_signature_configured)
    )
    firstiter_fused_atomic_supported = (
        firstiter_cc_pass2_only_best_coarse
        and coarse_class_assignments is not None
        and hasattr(experiment_dataset, "subset")
    )
    later_soft_particle_fused_supported = (
        bool(bpref_device_signature_active)
        and device_signature_configured
        and not firstiter_cc_pass2_only_best_coarse
        and bool(pass2_kwargs.get("mstep_relion_x_half", False))
    )
    fused_atomic_diagnostic_supported = bool(
        firstiter_fused_atomic_supported or later_soft_particle_fused_supported
    )
    if fused_atomic_diagnostic_requested and not fused_atomic_diagnostic_supported:
        raise RuntimeError(
            "RELAX_RELION_X_HALF_BP_FUSED_ATOMICS is qualified only for the sparse "
            "first-iteration global-winner subset or explicitly scoped later "
            "soft-posterior pass 2"
        )
    if bpref_device_signature_active:
        if not device_signature_configured:
            raise RuntimeError("active BPref device signature scope requires a device dump directory")
        _validate_bpref_device_signature_sparse_route(
            active=True,
            n_classes=n_classes,
        )
        if not later_soft_particle_fused_supported and not firstiter_fused_atomic_supported:
            raise RuntimeError(
                "active BPref device signature scope requires supported sparse RELION x-half topology"
            )

    if firstiter_cc_pass2_only_best_coarse:
        if not hasattr(experiment_dataset, "subset"):
            raise NotImplementedError(
                "the --firstiter_cc fine pass scores each coarse winner class over its image subset and needs a "
                "dataset with subset(); the dense global-winner fallback was removed on 2026-10-03"
            )
        pass2_t0 = time.time()

        def _winner_subset_pass2(class_assignments, given_pose_evidence=None):
            return run_sparse_firstiter_global_winner_subset_pass2(
                experiment_dataset,
                means_array,
                mean_variance,
                noise_variance,
                coarse_translations_np,
                fine_rotations_np,
                fine_mstep_rotations_np,
                sparse_fine_translations_np,
                rot_parent_map_np,
                trans_parent_map_np,
                sig_sample_indices_by_class,
                disc_type,
                coarse_result=coarse_result,
                coarse_class_assignments=class_assignments,
                n_rot_coarse=n_rot_coarse,
                n_fine_trans=n_trans_fine,
                healpix_order=_resolved_coarse_healpix_order(),
                oversampling_order=_resolved_oversampling_order(),
                accumulate_noise=accumulate_noise,
                return_best_pose_details=return_best_pose_details,
                pass2_kwargs=pass2_kwargs,
                given_pose_evidence=given_pose_evidence,
            )

        if coarse_class_assignments is None:
            # Given poses from one reference per class: no coarse probe named the winners.
            coarse_class_assignments = given_pose_cc_classes(_winner_subset_pass2, n_classes, n_images)
        result = _winner_subset_pass2(coarse_class_assignments)
        pass2_s = time.time() - pass2_t0
        logger.info(
            "Adaptive K-class EM profile: classes=%d images=%d coarse=(rot=%d,trans=%d) sparse_firstiter_fine=(rot=%d,trans=%d) pass1=%.1fs mask=%.1fs pass2=%.1fs total=%.1fs",
            n_classes,
            n_images,
            n_rot_coarse,
            n_trans_coarse,
            n_rot_fine,
            n_trans_fine,
            pass1_s,
            time.time() - mask_t0,
            pass2_s,
            time.time() - overall_t0,
        )
        return _with_significant_counts(result)

    if fill_fine_rows is not None:
        # The sparse pass 2 reads only its significant parents' rows.
        fill_fine_rows(_supports_coarse_parents(sig_sample_indices_by_class, n_images, n_rot_coarse, n_trans_coarse))
    result = run_sparse_k_class_adaptive_pass2(
        experiment_dataset,
        means_array,
        mean_variance,
        noise_variance,
        coarse_rotations_np,
        coarse_translations_np,
        fine_rotations_np,
        fine_mstep_rotations_np,
        rot_parent_map_np,
        sparse_fine_translations_np,
        trans_parent_map_np,
        sig_sample_indices_by_class,
        disc_type,
        class_log_priors=log_priors,
        accumulate_noise=accumulate_noise,
        return_best_pose_details=return_best_pose_details,
        coarse_healpix_order=_resolved_coarse_healpix_order(),
        oversampling_order=_resolved_oversampling_order(),
        random_perturbation=0.0,
        engine_kwargs=pass2_kwargs,
    )
    logger.info(
        "Adaptive K-class EM profile: classes=%d images=%d coarse=(rot=%d,trans=%d) sparse_fine=(rot=%d,trans=%d) pass1=%.1fs mask=%.1fs total=%.1fs",
        n_classes,
        n_images,
        n_rot_coarse,
        n_trans_coarse,
        n_rot_fine,
        n_trans_fine,
        pass1_s,
        time.time() - mask_t0,
        time.time() - overall_t0,
    )
    return _with_significant_counts(result)
