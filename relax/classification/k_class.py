"""K-class EM orchestration on the adaptive sparse pass and the exact-local single-volume engines."""

from __future__ import annotations

import logging
import os
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from recovar.utils.nvtx_shim import nvtx

from relax.classification.k_class_inputs import (
    _as_class_means,
    _class_log_priors,
    _select_class_value,
    _select_projector_half_for_class,
    _select_required_class_value,
    seed_iteration_first_class,
    seed_iteration_supports,
)
from relax.classification.k_class_results import (
    KClassEMResult,
    _assemble_result,
    _expand_subset_noise_stats,
    _expand_subset_pose_details,
    _full_stats_from_subset,
    _zero_subset_noise_stats,
)
from relax.diagnostics.coarse_score_diagnostics import (
    _with_coarse_significance_diagnostics,
)
from relax.diagnostics.local_debug import score_dump_label
from relax.helpers.env_flags import parse_env_flag
from relax.helpers.half_volume_mstep import relion_backprojector_volume_shape
from relax.helpers.scale_groups import prepare_scale_correction_groups
from relax.helpers.types import RelionStats, make_relion_stats
from relax.scoring.significant_samples import ComplementSignificantSampleIndices

logger = logging.getLogger(__name__)
NVTX_DOMAIN_EM = "recovar_em"
_RELION_X_HALF_BP_FUSED_ATOMICS_ENV = "RELAX_RELION_X_HALF_BP_FUSED_ATOMICS"
_LOCAL_HOST_RESULT_PUBLICATION_ENV = "RELAX_EXACT_LOCAL_HOST_RESULT_PUBLICATION"


class _DenseKClassScoreProbeResult(NamedTuple):
    """Joint ``--firstiter_cc`` coarse probe output: per-class evidence, best poses and global class winners."""

    class_log_evidence: np.ndarray
    per_class_hard_assignments: np.ndarray
    per_class_stats: tuple[RelionStats, ...]
    class_assignments: np.ndarray
    coarse_score_backend: str | None = None


def _apply_bpref_particle_order_policy(
    common: dict,
    engine_kwargs: dict,
    *,
    n_classes: int,
) -> bool:
    """Forward the guarded fresh-K=1 BPref order into a sparse pass."""

    preserve = bool(engine_kwargs.get("preserve_bpref_particle_order", False))
    if preserve and int(n_classes) != 1:
        raise ValueError("RELION BPref particle-order preservation is K=1-only")
    if preserve:
        common["preserve_bpref_particle_order"] = True
    return preserve


def _validate_bpref_device_signature_sparse_route(
    *,
    active: bool,
    n_classes: int,
) -> None:
    """Fail closed unless scoped BPref capture uses a sparse pass-2 route."""

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
    n_rot_coarse = int(n_rot_coarse)
    n_trans_coarse = int(n_trans_coarse)
    n_rot_fine = int(n_rot_fine)
    n_trans_fine = int(n_trans_fine)
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

### M- THIS SEEMS TO USE THE DATASET OBJECT OF RECOVAR - IS THIS OUTDATED - SHOULD WE REMOVE IT FOR BETTER EFFICIENCY/READABILITY? IT WASN'T MADE FOR THIS SO WE SHOULD THINK ABOUT JUST STRIPPING IT OUT/CLEANING IT

def _dataset_image_count(experiment_dataset, fallback: int | None = None) -> int:
    if hasattr(experiment_dataset, "n_units"):
        return int(experiment_dataset.n_units)
    if hasattr(experiment_dataset, "n_images"):
        return int(experiment_dataset.n_images)
    if fallback is not None:
        return int(fallback)
    raise AttributeError("experiment_dataset must expose n_units or n_images")

## THIS SEEMS TO INDICATE POOR PLANNING/SETTING UP IF WE HAVE TO DO THIS - REPLACE BY A NICER CLASS OR SOMETHING?
def _infer_healpix_order_from_rotation_count(
    n_rot: int,
    symmetry_label: str = "C1",
) -> int:
    from relax.sampling import rotation_grid_size

    n_rot = int(n_rot)
    for order in range(16):
        try:
            grid_size = rotation_grid_size(order, symmetry_label)
        except ValueError:
            continue
        if grid_size == n_rot:
            return order
    raise ValueError(
        f"Cannot infer RELION {symmetry_label} HEALPix order from {n_rot} rotations"
    )


def _rotation_prior_with_class_log_prior(
    rotation_log_prior, class_log_prior: float, n_rot: int, *, dtype: np.dtype = np.float32
):
    if rotation_log_prior is None:
        return np.full(int(n_rot), float(class_log_prior), dtype=dtype)
    prior = np.asarray(rotation_log_prior, dtype=dtype)
    return prior + np.asarray(float(class_log_prior), dtype=dtype)


def _sparse_pose_ids_to_fine_grid(hard_assignment, best_rotation_ids, n_fine_trans: int) -> np.ndarray:
    trans_ids = np.asarray(hard_assignment, dtype=np.int64) % int(n_fine_trans)
    rot_ids = np.asarray(best_rotation_ids, dtype=np.int64)
    return (rot_ids * int(n_fine_trans) + trans_ids).astype(np.int32, copy=False)


def _run_sparse_k_class_adaptive_pass2(
    experiment_dataset,
    means_array,
    mean_variance,
    noise_variance,
    coarse_rotations_np,
    coarse_translations_np,
    fine_rotations_np,
    fine_mstep_rotations_np,
    rot_parent_map_np,
    fine_translations_np,
    trans_parent_map_np,
    sig_sample_indices_by_class,
    disc_type: str,
    *,
    class_log_priors,
    accumulate_noise: bool,
    return_best_pose_details: bool,
    coarse_healpix_order: int | None = None,
    oversampling_order: int,
    random_perturbation: float,
    engine_kwargs: dict,
) -> KClassEMResult:
    """Run K-class adaptive pass-2 over RELION significant sparse support."""

    from relax.sparse_pass2.dispatch import compute_pass2_stats_sparse

    n_classes = int(means_array.shape[0])
    n_rot_coarse = int(coarse_rotations_np.shape[0])
    n_fine_trans = int(fine_translations_np.shape[0])
    symmetry_label = engine_kwargs.get("symmetry_label", "C1")
    healpix_order = (
        int(coarse_healpix_order)
        if coarse_healpix_order is not None
        else _infer_healpix_order_from_rotation_count(n_rot_coarse, symmetry_label)
    )
    base_engine_kwargs = dict(engine_kwargs)
    relion_projector_half_by_class = base_engine_kwargs.get("relion_projector_half")
    relion_projector_r_max = base_engine_kwargs.get("relion_projector_r_max")
    use_k1_fine_diff2_ffi = False
    if n_classes == 1:
        from recovar import cuda_backproject

        use_k1_fine_diff2_ffi = cuda_backproject.cuda_available()
    source_faithful_spectrum_norm = bool(
        base_engine_kwargs.get("source_faithful_spectrum_norm", False)
    )
    if source_faithful_spectrum_norm and n_classes != 1:
        raise ValueError("source-faithful powerClass normalization is K=1-only")

    def _class_rotation_prior(class_index: int):
        class_prior = base_engine_kwargs.get("class_rotation_log_prior")
        if class_prior is not None:
            rot_prior = _select_required_class_value(
                class_prior,
                class_index,
                n_classes,
                "class_rotation_log_prior",
            )
        else:
            rot_prior = base_engine_kwargs.get("rotation_log_prior")
        return _rotation_prior_with_class_log_prior(
            rot_prior,
            float(class_log_priors[class_index]),
            n_rot_coarse,
            dtype=(np.float64 if base_engine_kwargs.get("use_float64_scoring") else np.float32),
        )

    common = dict(
        nside_level=healpix_order,
        disc_type=disc_type,
        oversampling_order=int(oversampling_order),
        current_size=base_engine_kwargs.get("current_size"),
        reconstruction_current_size=base_engine_kwargs.get("reconstruction_current_size"),
        translation_step=None,
        score_with_masked_images=bool(base_engine_kwargs.get("score_with_masked_images", False)),
        return_stats=True,
        translation_log_prior=base_engine_kwargs.get("translation_log_prior"),
        half_spectrum_scoring=bool(base_engine_kwargs.get("half_spectrum_scoring", False)),
        projection_padding_factor=int(base_engine_kwargs.get("projection_padding_factor", 1)),
        projection_mask_current_image_disk=bool(
            base_engine_kwargs.get("projection_mask_current_image_disk", False)
        ),
        reconstruction_padding_factor=int(base_engine_kwargs.get("reconstruction_padding_factor", 1)),
        image_corrections=base_engine_kwargs.get("image_corrections"),
        scale_corrections=base_engine_kwargs.get("scale_corrections"),
        group_ids=base_engine_kwargs.get("group_ids"),
        optics_group_ids=base_engine_kwargs.get("optics_group_ids"),
        reconstruction_volume_current_size=base_engine_kwargs.get("reconstruction_volume_current_size"),
        reconstruction_image_radius=base_engine_kwargs.get("reconstruction_image_radius"),
        scale_correction_group_count=base_engine_kwargs.get("scale_correction_group_count"),
        scale_correction_data_vs_prior=base_engine_kwargs.get("scale_correction_data_vs_prior"),
        image_pre_shifts=base_engine_kwargs.get("image_pre_shifts"),
        use_float64_scoring=bool(base_engine_kwargs.get("use_float64_scoring", False)),
        translation_prior_centers=base_engine_kwargs.get("translation_prior_centers"),
        do_gridding_correction=bool(base_engine_kwargs.get("do_gridding_correction", False)),
        square_window=bool(base_engine_kwargs.get("square_window", False)),
        window_at_box=bool(base_engine_kwargs.get("window_at_box", False)),
        relion_half_volume_mstep=bool(base_engine_kwargs.get("relion_half_volume_mstep", False)),
        relion_x_half_mstep=bool(base_engine_kwargs.get("mstep_relion_x_half", False)),
        mstep_subtract_ctf_projection=bool(
            base_engine_kwargs.get("mstep_subtract_ctf_projection", False)
        ),
        adaptive_fraction=float(base_engine_kwargs.get("adaptive_fraction", 0.999)),
        relion_fine_mstep_prune=bool(base_engine_kwargs.get("relion_fine_mstep_prune", False)) and n_classes == 1,
        relion_firstiter_score_mode=base_engine_kwargs.get(
            "relion_firstiter_score_mode",
            "gaussian",
        ),
        relion_exact_fine_gaussian=bool(
            base_engine_kwargs.get("relion_exact_fine_gaussian", True)
        ),
        # The exact rectangular/pair CUDA reduction is qualified for K=1.
        # Preserve the existing K>1 scorer until its independent boundary is
        # localized.
        relion_fine_diff2_fused_ffi=use_k1_fine_diff2_ffi,
        # RELION's float32 exp/sort/scan significance path is now qualified
        # for K=1.  Keep K>1 byte-preserving until its separate posterior
        # boundary is diagnosed.
        relion_f32_fine_posterior=n_classes == 1,
        # The exact normalized-CC tree is a deliberately K=1-scoped parity
        # candidate. Keep the K>1 route byte-preserving until K=1 closes.
        relion_exact_fine_normalized_cc=n_classes == 1,
        dense_gemm_full_grid=bool(base_engine_kwargs.get("dense_gemm_full_grid", False)),
        relion_firstiter_winner_take_all=bool(
            base_engine_kwargs.get("relion_firstiter_winner_take_all", False)
        ),
        random_perturbation=float(random_perturbation),
        fine_rotations_override=fine_rotations_np,
        fine_mstep_rotations_override=fine_mstep_rotations_np,
        fine_rotation_parent_override=rot_parent_map_np,
        fine_translations_override=fine_translations_np,
        fine_translation_parent_override=trans_parent_map_np,
        bpref_device_signature_active=bool(
            base_engine_kwargs.get("bpref_device_signature_active", False)
        ),
        source_faithful_spectrum_norm=source_faithful_spectrum_norm,
        nyquist_column_counting=base_engine_kwargs.get("nyquist_column_counting", "relion"),
        firstiter_cc_support=base_engine_kwargs.get("firstiter_cc_support", "relion"),
        **({"symmetry_label": base_engine_kwargs["symmetry_label"]}
           if base_engine_kwargs.get("symmetry_label", "C1") != "C1" else {}),
        **_translation_angle_scale_kwargs(base_engine_kwargs),
    )
    # VDAM's pseudo-halfset accumulator slots (class + K * group): the resident engine only.
    reconstruction_groups = base_engine_kwargs.get("reconstruction_group_ids") is not None
    if reconstruction_groups:
        common["reconstruction_group_ids"] = base_engine_kwargs["reconstruction_group_ids"]
        common["reconstruction_group_count"] = int(base_engine_kwargs["reconstruction_group_count"])
    if n_classes == 1 and base_engine_kwargs.get("relion_f32_normalization_sum_weight") is not None:
        common["relion_f32_normalization_sum_weight"] = base_engine_kwargs["relion_f32_normalization_sum_weight"]
        common["relion_coarse_hard_assignment"] = base_engine_kwargs.get("relion_coarse_hard_assignment")
        common["relion_coarse_max_posterior"] = base_engine_kwargs.get("relion_coarse_max_posterior")
    preserve_bpref_particle_order = _apply_bpref_particle_order_policy(
        common,
        base_engine_kwargs,
        n_classes=n_classes,
    )
    # Images on another grid fill the backprojector at the reference model size.
    mstep_current_size = (
        common["reconstruction_volume_current_size"]
        if common.get("reconstruction_volume_current_size") is not None
        else common["current_size"]
        if common["reconstruction_current_size"] is None
        else common["reconstruction_current_size"]
    )
    mstep_accumulator_shape = (
        relion_backprojector_volume_shape(
            experiment_dataset.volume_shape,
            common["reconstruction_padding_factor"],
            current_size=mstep_current_size,
        )
        if common["relion_x_half_mstep"]
        else None
    )

    common["return_source_eulers"] = bool(return_best_pose_details)
    common["fine_source_eulers_override"] = base_engine_kwargs.get("fine_source_eulers_override")
    if n_classes > 1:
        # RELION's Class3D fine pass: every class in one resident sweep.
        return _run_resident_k_class_pass2(
            experiment_dataset,
            means_array,
            noise_variance,
            coarse_translations_np,
            sig_sample_indices_by_class,
            common=common,
            engine_kwargs=base_engine_kwargs,
            class_rotation_priors=[_class_rotation_prior(k) for k in range(n_classes)],
            relion_projector_half_by_class=relion_projector_half_by_class,
            relion_projector_r_max=relion_projector_r_max,
            accumulate_noise=accumulate_noise,
            mstep_accumulator_shape=mstep_accumulator_shape,
        )

    # K=1: one resident pass. RELION's shared unweighted power_img high-shell term
    # belongs to this only class.
    common["include_unweighted_norm_high_shell"] = True
    del preserve_bpref_particle_order
    mstep_t0 = time.time()
    result = compute_pass2_stats_sparse(
        experiment_dataset,
        means_array[0],
        _select_class_value(mean_variance, 0, n_classes),
        noise_variance,
        coarse_translations_np,
        sig_sample_indices_by_class[0],
        rotation_log_prior=_class_rotation_prior(0),
        accumulate_noise=accumulate_noise,
        # No other classes: the degenerate cross-class normalizer of the production K=1 pass.
        **({} if common["dense_gemm_full_grid"] else {
            "normalization_other_score_log_z": np.full(
                _dataset_image_count(experiment_dataset), -np.inf, dtype=np.float64,
            ),
        }),
        normalization_score_mode=(
            None if common["dense_gemm_full_grid"] else common["relion_firstiter_score_mode"]
        ),
        return_score_log_z=True,
        relion_projector_half=_select_projector_half_for_class(relion_projector_half_by_class, 0, n_classes),
        relion_projector_r_max=relion_projector_r_max,
        **common,
    )
    mstep_s = time.time() - mstep_t0
    logger.info(
        "Sparse adaptive K=1 pass2 profile: images=%d mstep=%.1fs",
        _dataset_image_count(experiment_dataset),
        mstep_s,
    )
    return single_class_pass2_em_result(
        result,
        n_fine_trans=n_fine_trans,
        accumulate_noise=accumulate_noise,
        return_best_pose_details=return_best_pose_details,
        profile_summary={
            "sparse_adaptive_probe_s": np.float64(0.0),
            "sparse_adaptive_mstep_s": np.float64(mstep_s),
        },
        mstep_full_half_axis=0 if common["relion_x_half_mstep"] else None,
        mstep_accumulator_shape=mstep_accumulator_shape,
    )


def single_class_pass2_em_result(
    result,
    *,
    n_fine_trans: int,
    accumulate_noise: bool,
    return_best_pose_details: bool,
    mstep_full_half_axis,
    mstep_accumulator_shape,
    profile_summary=None,
) -> KClassEMResult:
    """The K-class result of one K=1 resident pass 2 (``SparsePass2Output``; SPA images or tomo particles)."""

    return _assemble_result(
        class_log_evidence=np.asarray(result.relion_stats.log_evidence_per_image, dtype=np.float64)[None],
        new_means=None,
        Ft_y=[_as_host_accumulator(result.Ft_y)],
        Ft_ctf=[_as_host_accumulator(result.Ft_ctf)],
        per_class_hard_assignments=_sparse_pose_ids_to_fine_grid(
            result.hard_assignment, result.best_rotation_indices, n_fine_trans
        )[None],
        per_class_stats=(result.relion_stats,),
        noise_stats=None if not accumulate_noise else (result.noise_stats,),
        per_class_best_pose_eulers_deg=[result.source_eulers] if return_best_pose_details else None,
        per_class_best_pose_rotations=[result.best_rotations] if return_best_pose_details else None,
        per_class_best_pose_translations=[result.best_translations] if return_best_pose_details else None,
        per_class_best_pose_rotation_ids=[result.best_rotation_indices] if return_best_pose_details else None,
        profile_summary=profile_summary,
        host_accumulators=True,
        mstep_full_half_axis=mstep_full_half_axis,
        mstep_accumulator_shape=mstep_accumulator_shape,
    )


def _as_host_accumulator(value):
    """Copy a full-volume accumulator off GPU before retaining it."""

    return np.asarray(jax.device_get(value))


def _resident_production_arithmetic(options: dict) -> dict:
    """RELION's pass-2 arithmetic, the K=1 production set, for a K>1 pass on the resident engine.

    RELION's E-step arithmetic does not depend on the class count: its float32
    fine posterior and pruned M-step, the literal normalized-CC reduction of
    ``--firstiter_cc``, its powerClass spectrum and exact BPref operands, and
    the atomic Wavg triplet of the preserved BPref order
    (``source_faithful_spectrum_norm``, ``preserve_bpref_particle_order``). The
    compact K>1 routes keep their historical arithmetic until they are deleted.
    """

    options = dict(options)
    options.update(
        source_faithful_spectrum_norm=True,
        preserve_bpref_particle_order=True,
        relion_f32_fine_posterior=True,
        relion_fine_mstep_prune=True,
        relion_fine_diff2_fused_ffi=True,
        relion_exact_fine_normalized_cc=True,
    )
    return options


def _run_resident_k_class_pass2(
    experiment_dataset,
    means_array,
    noise_variance,
    coarse_translations_np,
    sig_sample_indices_by_class,
    *,
    common: dict,
    engine_kwargs: dict,
    class_rotation_priors,
    relion_projector_half_by_class,
    relion_projector_r_max,
    accumulate_noise: bool,
    mstep_accumulator_shape,
) -> KClassEMResult:
    """RELION's Class3D fine pass on the device-resident engine.

    Runs with the K=1 production arithmetic (:func:`_resident_production_arithmetic`). A
    configuration the resident checks refuse (``ResidentConfigurationUnsupported``) is an
    error: the resident engine is relax's one pass-2 engine. The pass records its engine
    (``engine_record``).
    """

    from relax.sparse_pass2.engine_record import record_pass_engine
    from relax.sparse_pass2.resident_pass2 import compute_k_class_pass2_stats_resident

    if engine_kwargs.get("normalization_log_evidence") is not None:
        raise NotImplementedError(
            "an externally supplied normalization is not part of the resident K-class pass"
        )
    n_classes = int(means_array.shape[0])
    options = _resident_production_arithmetic(common)
    if not options.get("dense_gemm_full_grid", False):
        options.pop("relion_exact_fine_normalized_cc")
    options.update(
        relion_projector_half=relion_projector_half_by_class,
        relion_projector_r_max=relion_projector_r_max,
        accumulate_noise=accumulate_noise,
    )
    t0 = time.time()
    output = compute_k_class_pass2_stats_resident(
        experiment_dataset,
        means_array,
        noise_variance,
        coarse_translations_np,
        sig_sample_indices_by_class,
        options.pop("nside_level"),
        options.pop("disc_type"),
        rotation_log_priors_by_class=class_rotation_priors,
        **options,
    )
    record_pass_engine("global", "gemm_dense" if options.get("dense_gemm_full_grid", False) else "resident")
    logger.info(
        "Resident K-class pass2: classes=%d images=%d total=%.1fs",
        n_classes,
        _dataset_image_count(experiment_dataset),
        time.time() - t0,
    )
    return _class_segmented_em_result(
        output,
        n_classes=n_classes,
        class_posterior_sums_from_noise=True,
        return_profile=False,
        host_accumulators=True,
        mstep_full_half_axis=0 if common["relion_x_half_mstep"] else None,
        mstep_accumulator_shape=mstep_accumulator_shape,
    )


def _class_segmented_em_result(
    output,
    *,
    n_classes: int,
    class_posterior_sums_from_noise: bool,
    return_profile: bool,
    **accumulator_layout,
) -> KClassEMResult:
    """The K-class result of one engine call that scored every class jointly.

    ``output`` is the class-segmented result of the exact-local engine or of the
    resident pass 2 (``ResidentKClassPass2Output``, the same field names).
    ``accumulator_layout`` is passed to ``_assemble_result`` as is.
    """

    # Follow the existing K-class convention: the class log evidence is float64 for
    # the responsibility algebra, while every published per-image statistic keeps the
    # engine's scoring precision, which is what the per-class route returns and what
    # downstream consumers read.
    class_log_evidence = np.asarray(output.class_log_evidence_per_image, dtype=np.float64)
    class_best_log_score = np.asarray(output.class_best_log_score_per_image)
    joint_log_evidence = np.asarray(output.stats.log_evidence_per_image)
    # Every class's Pmax is measured against the joint normalizer, which is what the
    # per-class M-step calls do when they are given the joint log evidence.
    #
    # The normalizer is now published at the normalization dtype, which is wider than
    # the scoring dtype by default. Pmax is a scoring quantity, so the operand is cast
    # to the winning score's dtype before the subtraction, reproducing the precision
    # this subtraction and exponential had when the normalizer was itself narrowed.
    # Casting only the final result would leave the subtraction and the exponential at
    # the wider precision and silently change them.
    joint_log_evidence_for_posterior = joint_log_evidence.astype(
        class_best_log_score.dtype, copy=False,
    )
    with np.errstate(over="ignore"):
        class_max_posterior = np.exp(
            class_best_log_score - joint_log_evidence_for_posterior[None, :]
        )
    class_rotation_posterior_sums = np.asarray(output.class_rotation_posterior_sums, dtype=np.float64)
    # Every per-class M-step call in the per-class route is given the joint log
    # evidence as its normalizer, so each class's RelionStats reports that joint
    # value; the class's own evidence travels separately as class_log_evidence.
    # Keep both conventions rather than moving a per-class value into the field
    # that names the normalizer actually used.
    per_class_stats = tuple(
        make_relion_stats(
            log_evidence_per_image=joint_log_evidence,
            best_log_score_per_image=class_best_log_score[class_index],
            max_posterior_per_image=class_max_posterior[class_index],
            rotation_posterior_sums=class_rotation_posterior_sums[class_index],
        )
        for class_index in range(n_classes)
    )
    return _assemble_result(
        class_log_evidence=class_log_evidence,
        # One engine call scored every class jointly, so its joint max posterior is
        # authoritative; see _assemble_result for why it is not rebuilt downstream.
        joint_max_posterior_per_image=np.asarray(output.stats.max_posterior_per_image),
        new_means=None,
        Ft_y=[output.Ft_y[class_index] for class_index in range(n_classes)],
        Ft_ctf=[output.Ft_ctf[class_index] for class_index in range(n_classes)],
        per_class_hard_assignments=np.asarray(output.per_class_hard_assignments, dtype=np.int64),
        per_class_stats=per_class_stats,
        noise_stats=None,
        aggregate_noise_stats_override=output.noise_stats,
        class_posterior_sums_override=(
            np.asarray(output.class_reconstruction_posterior_sums, dtype=np.float64)
            if class_posterior_sums_from_noise
            else None
        ),
        per_class_best_pose_rotations=(
            None if output.per_class_best_pose_rotations is None else list(output.per_class_best_pose_rotations)
        ),
        per_class_best_pose_translations=(
            None
            if output.per_class_best_pose_translations is None
            else list(output.per_class_best_pose_translations)
        ),
        per_class_best_pose_rotation_ids=(
            None
            if output.per_class_best_pose_rotation_ids is None
            else list(output.per_class_best_pose_rotation_ids)
        ),
        # Canonical source Eulers are published to STAR metadata and read back by
        # local search, so they are carried from the winning row, never rebuilt from
        # its rotation matrix.
        per_class_best_pose_eulers_deg=(
            None
            if output.per_class_best_pose_eulers_deg is None
            else list(output.per_class_best_pose_eulers_deg)
        ),
        profile_summary=output.profile if return_profile else None,
        uncast_log_evidence_per_image=output.uncast_log_evidence_per_image,
        **accumulator_layout,
    )


def _run_dense_k_class_joint_firstiter_score_probe(
    experiment_dataset,
    means_array,
    noise_variance,
    rotations,
    translations,
    disc_type: str,
    *,
    engine_kwargs: dict,
) -> _DenseKClassScoreProbeResult:
    """Score RELION firstiter-CC K-class coarse poses in one shared pass."""

    from relax.diagnostics.coarse_gaussian_diagnostics import _significance_debug_dump_matches
    from relax.helpers.projection import compact_relion_projector_half_for_centered_indices
    from relax.scoring.significance import (
        _compute_k_class_significance_batched,
        _global_pass1_relion_projector_texture_enabled,
    )

    means_array = _as_class_means(means_array)
    n_classes = int(means_array.shape[0])
    n_rot = int(np.asarray(rotations).shape[0])
    n_images = _dataset_image_count(experiment_dataset)

    # Keep the full Projector::data host slab for fine scoring, but transfer
    # only the centered support consumed by this coarse normalized-CC probe.
    # At box 800, staging the full PPref plus its CUDA texture can otherwise
    # exhaust an 80-GB device before the first particle is scored.
    score_projector_half = engine_kwargs.get("relion_projector_half")
    score_projector_r_max = engine_kwargs.get("relion_projector_r_max")
    score_texture_interp = engine_kwargs.get(
        "coarse_relion_projector_texture_interp",
        False,
    )
    if score_texture_interp is None:
        score_texture_interp = _global_pass1_relion_projector_texture_enabled()
    if score_projector_half is not None and n_classes == 1 and score_texture_interp:
        from relax.helpers.fourier_window import make_fourier_window_spec

        image_shape = tuple(int(value) for value in experiment_dataset.image_shape)
        n_half = image_shape[0] * (image_shape[1] // 2 + 1)
        score_window = make_fourier_window_spec(
            image_shape,
            engine_kwargs.get("current_size"),
            n_half,
            square=bool(engine_kwargs.get("square_window", False)),
            include_recon_window=False,
            score_square=True,
            score_include_dc=True,
        )
        if score_window.score_indices_np is not None:
            score_projector_half = _select_projector_half_for_class(
                score_projector_half,
                0,
                1,
            )
            original_projector_shape = tuple(int(value) for value in score_projector_half.shape)
            original_projector_r_max = int(score_projector_r_max)
            score_projector_half, score_projector_r_max = (
                compact_relion_projector_half_for_centered_indices(
                    score_projector_half,
                    score_window.score_indices_np,
                    image_shape,
                    r_max=int(score_projector_r_max),
                    padding_factor=int(engine_kwargs.get("projection_padding_factor", 1)),
                )
            )
            logger.info(
                "RELION firstiter-CC coarse PPref host compaction: "
                "r_max=%d->%d shape=%s->%s allocated=%.4f GiB",
                original_projector_r_max,
                int(score_projector_r_max),
                original_projector_shape,
                tuple(int(value) for value in score_projector_half.shape),
                float(score_projector_half.nbytes / 2**30),
            )

    # RELION's iter-1 firstiter_cc path performs WTA on raw normalized-CC
    # scores before the non-firstiter prior-weighting branch is reached.
    firstiter_class_log_priors = np.zeros(n_classes, dtype=np.float64)
    full_stats = _compute_k_class_significance_batched(
        experiment_dataset,
        means_array,
        noise_variance,
        rotations,
        translations,
        disc_type,
        class_log_priors=firstiter_class_log_priors,
        adaptive_fraction=1.0,
        max_significants=1,
        image_batch_size=int(engine_kwargs.get("image_batch_size", 500)),
        rotation_block_size=int(engine_kwargs.get("rotation_block_size", 5000)),
        current_size=engine_kwargs.get("current_size"),
        score_with_masked_images=bool(engine_kwargs.get("score_with_masked_images", False)),
        rotation_log_prior=None,
        translation_log_prior=None,
        image_corrections=engine_kwargs.get("image_corrections"),
        scale_corrections=engine_kwargs.get("scale_corrections"),
        image_pre_shifts=engine_kwargs.get("image_pre_shifts"),
        # The normalized CC does not weight by the noise, but the scorer reads each image's row.
        optics_group_ids=engine_kwargs.get("optics_group_ids"),
        half_spectrum_scoring=bool(engine_kwargs.get("half_spectrum_scoring", False)),
        projection_padding_factor=int(engine_kwargs.get("projection_padding_factor", 1)),
        do_gridding_correction=bool(engine_kwargs.get("do_gridding_correction", False)),
        square_window=bool(engine_kwargs.get("square_window", False)),
        use_float64_scoring=bool(engine_kwargs.get("use_float64_scoring", False)),
        use_float64_projections=_projection_float64_from_kwargs(engine_kwargs),
        relion_projector_half=score_projector_half,
        relion_projector_r_max=score_projector_r_max,
        relion_projector_texture_interp=engine_kwargs.get(
            "coarse_relion_projector_texture_interp",
            False,
        ),
        score_mode="normalized_cc",
        firstiter_cc_support=engine_kwargs.get("firstiter_cc_support", "relion"),
        nyquist_column_counting=engine_kwargs.get("nyquist_column_counting", "relion"),
        tree_rescore_max_margin=engine_kwargs.get("firstiter_cc_tree_rescore_max_margin"),
        collect_significance=_significance_debug_dump_matches(
            current_size=engine_kwargs.get("current_size"),
            debug_iteration=engine_kwargs.get("debug_iteration"),
        ),
        return_class_best=True,
        return_class_second=bool(os.environ.get("RELAX_GLOBAL_WINNER_SUMMARY_PATH", "").strip()),
        debug_iteration=engine_kwargs.get("debug_iteration"),
        coarse_healpix_order=engine_kwargs.get("coarse_healpix_order"),
        coarse_rotation_ids=engine_kwargs.get("coarse_rotation_ids"),
        translation_phase_source=engine_kwargs.get("translation_phase_source"),
        **({"symmetry_label": engine_kwargs["symmetry_label"]} if engine_kwargs.get("symmetry_label", "C1") != "C1" else {}),
        **_translation_angle_scale_kwargs(engine_kwargs),
    )[-1]
    from relax.diagnostics.global_winner_summary import maybe_dump_global_winner_summary

    maybe_dump_global_winner_summary(
        experiment_dataset=experiment_dataset,
        full_stats=full_stats,
        n_classes=n_classes,
        n_rotations=n_rot,
        n_translations=int(np.asarray(translations).shape[0]),
        iteration=engine_kwargs.get("debug_iteration"),
    )
    class_log_evidence = np.asarray(full_stats["class_log_evidence_per_image"], dtype=np.float64)
    per_class_hard = np.asarray(full_stats["class_hard_assignments"], dtype=np.int32)
    score_dtype = _score_dtype_from_kwargs(engine_kwargs)
    class_best_log_score = np.asarray(full_stats["class_best_log_score_per_image"], dtype=score_dtype)
    class_assignments = np.asarray(full_stats["class_assignments"], dtype=np.int32)
    per_class_stats = tuple(
        make_relion_stats(
            log_evidence_per_image=np.asarray(class_log_evidence[class_index], dtype=score_dtype),
            best_log_score_per_image=np.asarray(class_best_log_score[class_index], dtype=score_dtype),
            max_posterior_per_image=np.ones(n_images, dtype=score_dtype),
            rotation_posterior_sums=np.zeros(n_rot, dtype=score_dtype),
        )
        for class_index in range(n_classes)
    )

    return _DenseKClassScoreProbeResult(
        class_log_evidence=class_log_evidence,
        per_class_hard_assignments=per_class_hard,
        per_class_stats=per_class_stats,
        class_assignments=class_assignments,
        coarse_score_backend=full_stats.get("executed_coarse_backend"),
    )


# The image-axis inputs a firstiter-CC subset pass forwards for each winner class.
_SUBSET_PASS_CLASS_KWARGS = ("image_corrections", "scale_corrections", "image_pre_shifts", "translation_prior_centers")

_IMAGE_AXIS_ENGINE_KWARGS = (
    "image_corrections",
    "scale_corrections",
    "group_ids",
    "optics_group_ids",
    "image_pre_shifts",
    "translation_prior_centers",
    "translation_log_prior",
    "rotation_log_prior",
    "normalization_log_evidence",
    "relion_f32_normalization_sum_weight",
)


def _subset_image_axis_engine_kwargs(kwargs: dict, image_indices: np.ndarray, n_images: int) -> dict:
    """Slice image-axis kwargs when running a class-specific dataset subset."""

    out = dict(kwargs)
    image_indices = np.asarray(image_indices, dtype=np.int64)
    for name in _IMAGE_AXIS_ENGINE_KWARGS:
        value = out.get(name)
        if value is None:
            continue
        array = np.asarray(value)
        if array.ndim > 0 and int(array.shape[0]) == int(n_images):
            out[name] = array[image_indices]
    return out


def _translation_angle_scale_kwargs(kwargs: dict) -> dict:
    """Forward a non-unit RELION model/optics translation-angle scale (K=1 only)."""

    scale = float(kwargs.get("relion_translation_angle_scale", 1.0))
    return {} if scale == 1.0 else {"relion_translation_angle_scale": scale}


def _projection_float64_from_kwargs(kwargs: dict) -> bool:
    """Whether projections run in double; production projects in float32."""

    return bool(kwargs.get("use_float64_projections", False))


def _pose_dtype_from_kwargs(kwargs: dict):
    """Host pose dtype: float64 when scoring or projections run in double, else float32."""

    return np.float64 if kwargs.get("use_float64_scoring", False) or _projection_float64_from_kwargs(kwargs) else np.float32


def _score_dtype_from_kwargs(kwargs: dict):
    """Host score dtype: float64 when scoring runs in double, else float32."""

    return np.float64 if kwargs.get("use_float64_scoring", False) else np.float32


def _full_group_count_from_kwargs(kwargs: dict) -> int | None:
    _, group_count = prepare_scale_correction_groups(
        kwargs.get("group_ids"), kwargs.get("scale_correction_group_count"),
    )
    return group_count or None


class _PerClassSubsetResults:
    """Per-class outputs of a firstiter-CC global-winner subset pass, in class order.

    Every image belongs to exactly one winning class, so each class contributes
    accumulators, assignments, statistics, noise and best poses over its own
    image subset, expanded back to the full image axis. A class without images
    contributes zero accumulators, ``-inf`` best scores and zero posteriors.
    ``host_accumulators`` says whether the appended M-step accumulators are moved
    to the host; the sparse route hosts engine outputs and keeps its empty-class
    zeros on the device.
    """

    def __init__(self, *, n_images, accumulate_noise, return_best_pose_details, full_group_count, pose_dtype, score_dtype):
        self.Ft_y = []
        self.Ft_ctf = []
        self.hard_assignments = []
        self.per_class_stats = []
        self.per_class_noise = [] if accumulate_noise else None
        self.best_pose_rotations = [] if return_best_pose_details else None
        self.best_pose_translations = [] if return_best_pose_details else None
        self.best_pose_rotation_ids = [] if return_best_pose_details else None
        self.subset_counts = []
        self.n_images = int(n_images)
        self.full_group_count = full_group_count
        self.pose_dtype = pose_dtype
        self.score_dtype = score_dtype
        self.return_best_pose_details = bool(return_best_pose_details)

    def append_empty_class(self, *, mean, class_log_evidence, noise_variance, class_index, n_classes, n_rot, host_accumulators):
        zero = jnp.zeros_like(mean)
        zero_ctf = jnp.zeros_like(jnp.real(mean))
        self.Ft_y.append(_as_host_accumulator(zero) if host_accumulators else zero)
        self.Ft_ctf.append(_as_host_accumulator(zero_ctf) if host_accumulators else zero_ctf)
        self.hard_assignments.append(np.zeros(self.n_images, dtype=np.int32))
        self.per_class_stats.append(
            make_relion_stats(
                log_evidence_per_image=np.asarray(class_log_evidence, dtype=self.score_dtype),
                best_log_score_per_image=np.full(self.n_images, -np.inf, dtype=self.score_dtype),
                max_posterior_per_image=np.zeros(self.n_images, dtype=self.score_dtype),
                rotation_posterior_sums=np.zeros(n_rot, dtype=self.score_dtype),
            ),
        )
        if self.per_class_noise is not None:
            self.per_class_noise.append(
                _zero_subset_noise_stats(
                    noise_variance,
                    n_images=self.n_images,
                    full_group_count=self.full_group_count,
                ),
            )
        if self.return_best_pose_details:
            self.best_pose_rotations.append(np.zeros((self.n_images, 3, 3), dtype=self.pose_dtype))
            self.best_pose_translations.append(np.zeros((self.n_images, 2), dtype=self.pose_dtype))
            self.best_pose_rotation_ids.append(np.zeros(self.n_images, dtype=np.int32))

    def append_class(self, *, image_indices, Ft_y, Ft_ctf, hard_full, stats_subset, class_log_evidence, noise, best_pose, host_accumulators):
        self.Ft_y.append(_as_host_accumulator(Ft_y) if host_accumulators else Ft_y)
        self.Ft_ctf.append(_as_host_accumulator(Ft_ctf) if host_accumulators else Ft_ctf)
        self.hard_assignments.append(hard_full)
        self.per_class_stats.append(
            _full_stats_from_subset(
                stats_subset,
                image_indices,
                self.n_images,
                class_log_evidence=class_log_evidence,
            ),
        )
        if self.per_class_noise is not None:
            self.per_class_noise.append(
                _expand_subset_noise_stats(
                    noise,
                    image_indices,
                    self.n_images,
                    full_group_count=self.full_group_count,
                ),
            )
        if self.return_best_pose_details:
            best_rots, best_trans, best_rot_ids = best_pose
            best_rots_full, best_trans_full, best_rot_ids_full = _expand_subset_pose_details(
                best_rots, best_trans, best_rot_ids, image_indices, self.n_images
            )
            self.best_pose_rotations.append(best_rots_full)
            self.best_pose_translations.append(best_trans_full)
            self.best_pose_rotation_ids.append(best_rot_ids_full)

    def assemble(self, class_log_evidence, *, profile_summary, **assemble_kwargs):
        """The winner-take-all K-class result of a subset pass; the subset counts are its class posterior sums."""

        if any(self.subset_counts):
            # A class without images adds zeros in the scored classes' layout (x-half BPref accumulators, noise per
            # shell), not the full-volume, per-pixel placeholders: a seed iteration's CC pass puts every image in
            # class 1.
            scored = next(k for k, n in enumerate(self.subset_counts) if n)
            for k, n in enumerate(self.subset_counts):
                if n:
                    continue
                self.Ft_y[k] = np.zeros_like(np.asarray(self.Ft_y[scored]))
                self.Ft_ctf[k] = np.zeros_like(np.asarray(self.Ft_ctf[scored]))
                if self.per_class_noise is not None:
                    self.per_class_noise[k] = _zero_noise_stats_like(self.per_class_noise[scored])
        return _assemble_result(
            class_log_evidence=class_log_evidence,
            new_means=None,
            Ft_y=self.Ft_y,
            Ft_ctf=self.Ft_ctf,
            per_class_hard_assignments=np.stack(self.hard_assignments, axis=0),
            per_class_stats=tuple(self.per_class_stats),
            noise_stats=None if self.per_class_noise is None else tuple(self.per_class_noise),
            per_class_best_pose_rotations=self.best_pose_rotations,
            per_class_best_pose_translations=self.best_pose_translations,
            per_class_best_pose_rotation_ids=self.best_pose_rotation_ids,
            class_posterior_sums_override=np.asarray(self.subset_counts, dtype=np.float64),
            firstiter_winner_take_all=True,
            profile_summary=profile_summary,
            **assemble_kwargs,
        )


def _zero_noise_stats_like(stats):
    """``stats`` with every sum zero, in its layout."""

    return stats._replace(
        **{
            name: (None if value is None else 0.0 if np.ndim(value) == 0 else np.zeros_like(np.asarray(value)))
            for name, value in stats._asdict().items()
        }
    )


def _run_sparse_firstiter_global_winner_subset_pass2(
    experiment_dataset,
    means_array,
    mean_variance,
    noise_variance,
    coarse_translations_np,
    fine_rotations_np,
    fine_mstep_rotations_np,
    fine_translations_np,
    rot_parent_map_np: np.ndarray,
    trans_parent_map_np: np.ndarray,
    sig_sample_indices_by_class,
    disc_type: str,
    *,
    coarse_result: _DenseKClassScoreProbeResult,
    coarse_class_assignments: np.ndarray,
    n_rot_coarse: int,
    n_fine_trans: int,
    healpix_order: int,
    oversampling_order: int,
    accumulate_noise: bool,
    return_best_pose_details: bool,
    pass2_kwargs: dict,
) -> KClassEMResult:
    """Sparse RELION firstiter_cc fine pass over global-winner image subsets."""

    from relax.sparse_pass2.dispatch import compute_pass2_stats_sparse

    n_classes = int(means_array.shape[0])
    n_images = int(coarse_class_assignments.shape[0])
    relion_projector_half_by_class = pass2_kwargs.get("relion_projector_half")
    relion_projector_r_max = pass2_kwargs.get("relion_projector_r_max")
    source_faithful_spectrum_norm = bool(
        pass2_kwargs.get("source_faithful_spectrum_norm", False)
    )
    if source_faithful_spectrum_norm and n_classes != 1:
        raise ValueError("source-faithful powerClass normalization is K=1-only")
    score_dtype = _score_dtype_from_kwargs(pass2_kwargs)
    pose_dtype = _pose_dtype_from_kwargs(pass2_kwargs)

    def _class_rotation_prior(class_index: int):
        del class_index
        return np.zeros(int(n_rot_coarse), dtype=score_dtype)

    common = dict(
        nside_level=int(healpix_order),
        disc_type=disc_type,
        oversampling_order=int(oversampling_order),
        current_size=pass2_kwargs.get("current_size"),
        reconstruction_current_size=pass2_kwargs.get("reconstruction_current_size"),
        translation_step=None,
        score_with_masked_images=bool(pass2_kwargs.get("score_with_masked_images", False)),
        return_stats=True,
        half_spectrum_scoring=bool(pass2_kwargs.get("half_spectrum_scoring", False)),
        projection_padding_factor=int(pass2_kwargs.get("projection_padding_factor", 1)),
        reconstruction_padding_factor=int(pass2_kwargs.get("reconstruction_padding_factor", 1)),
        use_float64_scoring=bool(pass2_kwargs.get("use_float64_scoring", False)),
        do_gridding_correction=bool(pass2_kwargs.get("do_gridding_correction", False)),
        square_window=bool(pass2_kwargs.get("square_window", False)),
        random_perturbation=0.0,
        fine_rotations_override=fine_rotations_np,
        fine_mstep_rotations_override=fine_mstep_rotations_np,
        fine_rotation_parent_override=rot_parent_map_np,
        fine_translations_override=fine_translations_np,
        fine_translation_parent_override=trans_parent_map_np,
        relion_half_volume_mstep=bool(pass2_kwargs.get("relion_half_volume_mstep", False)),
        relion_x_half_mstep=bool(pass2_kwargs.get("mstep_relion_x_half", False)),
        relion_firstiter_score_mode="normalized_cc",
        relion_firstiter_winner_take_all=True,
        # K=1 production firstiter-CC must use the literal RELION fine
        # numerator/denominator reduction.  Previously only the coarse probe
        # enabled this path, leaving the actual pass-2 M-step on the folded
        # algebraic shortcut.  Both are algebraically equivalent, but the
        # literal route follows RELION's reduction contract directly.
        relion_exact_fine_normalized_cc=n_classes == 1,
        bpref_device_signature_active=bool(
            pass2_kwargs.get("bpref_device_signature_active", False)
        ),
        source_faithful_spectrum_norm=source_faithful_spectrum_norm,
        nyquist_column_counting=pass2_kwargs.get("nyquist_column_counting", "relion"),
        firstiter_cc_support=pass2_kwargs.get("firstiter_cc_support", "relion"),
        # Images on another grid (K=1 shape classes): the reference-model M-step size and
        # image radius, as the Gaussian route passes them.
        reconstruction_volume_current_size=pass2_kwargs.get("reconstruction_volume_current_size"),
        reconstruction_image_radius=pass2_kwargs.get("reconstruction_image_radius"),
        **({"symmetry_label": pass2_kwargs["symmetry_label"]}
           if pass2_kwargs.get("symmetry_label", "C1") != "C1" else {}),
        **_translation_angle_scale_kwargs(pass2_kwargs),
    )
    _apply_bpref_particle_order_policy(
        common,
        pass2_kwargs,
        n_classes=n_classes,
    )
    resident_arithmetic = n_classes > 1
    if resident_arithmetic:
        # Each image's fine pass is a K=1 pass inside its coarse winner class,
        # so on the resident engine it is the K=1 production pass.
        common = _resident_production_arithmetic(common)
    # Images on another grid fill the backprojector at the reference model size.
    mstep_current_size = (
        common["reconstruction_volume_current_size"]
        if common.get("reconstruction_volume_current_size") is not None
        else common["current_size"]
        if common["reconstruction_current_size"] is None
        else common["reconstruction_current_size"]
    )
    mstep_accumulator_shape = (
        relion_backprojector_volume_shape(
            experiment_dataset.volume_shape,
            common["reconstruction_padding_factor"],
            current_size=mstep_current_size,
        )
        if common["relion_x_half_mstep"]
        else None
    )

    results = _PerClassSubsetResults(
        n_images=n_images,
        accumulate_noise=accumulate_noise,
        return_best_pose_details=return_best_pose_details,
        full_group_count=_full_group_count_from_kwargs(pass2_kwargs),
        pose_dtype=pose_dtype,
        score_dtype=score_dtype,
    )
    t0 = time.time()
    for class_index in range(n_classes):
        image_indices = np.nonzero(coarse_class_assignments == class_index)[0].astype(np.int64, copy=False)
        results.subset_counts.append(int(image_indices.size))
        if image_indices.size == 0:
            results.append_empty_class(
                mean=means_array[class_index],
                class_log_evidence=coarse_result.class_log_evidence[class_index],
                noise_variance=noise_variance,
                class_index=class_index,
                n_classes=n_classes,
                n_rot=n_rot_coarse,
                host_accumulators=False,
            )
            continue

        subset_dataset = experiment_dataset.subset(image_indices)
        subset_sig = [sig_sample_indices_by_class[class_index][int(i)] for i in image_indices]
        class_kwargs = _subset_image_axis_engine_kwargs(
            {name: pass2_kwargs.get(name) for name in _SUBSET_PASS_CLASS_KWARGS}, image_indices, n_images
        )
        scale_groups = {"group_ids": None, "scale_correction_group_count": None}
        if resident_arithmetic:
            # The K=1 production pass of each winner class carries the scale groups as the
            # K=1 route does. RELION accumulates the scale sums in the firstiter_cc iteration
            # and does not apply them (ml_optimiser.cpp:6190).
            scale_groups = _subset_image_axis_engine_kwargs(
                {"group_ids": pass2_kwargs.get("group_ids")}, image_indices, n_images
            ) | {"scale_correction_group_count": pass2_kwargs.get("scale_correction_group_count")}

        result = compute_pass2_stats_sparse(
            subset_dataset,
            means_array[class_index],
            _select_class_value(mean_variance, class_index, n_classes),
            noise_variance,
            coarse_translations_np,
            subset_sig,
            rotation_log_prior=_class_rotation_prior(class_index),
            translation_log_prior=None,
            accumulate_noise=accumulate_noise,
            image_corrections=class_kwargs.get("image_corrections"),
            scale_corrections=class_kwargs.get("scale_corrections"),
            **scale_groups,
            image_pre_shifts=class_kwargs.get("image_pre_shifts"),
            translation_prior_centers=class_kwargs.get("translation_prior_centers"),
            # Each image's noise row.
            optics_group_ids=_subset_image_axis_engine_kwargs(
                {"optics_group_ids": pass2_kwargs.get("optics_group_ids")}, image_indices, n_images
            )["optics_group_ids"],
            relion_projector_half=_select_projector_half_for_class(
                relion_projector_half_by_class,
                class_index,
                n_classes,
            ),
            relion_projector_r_max=relion_projector_r_max,
            bpref_class_index=class_index,
            **common,
        )
        hard_full = np.zeros(n_images, dtype=np.int32)
        hard_full[image_indices] = _sparse_pose_ids_to_fine_grid(
            result.hard_assignment, result.best_rotation_indices, n_fine_trans
        )
        results.append_class(
            image_indices=image_indices,
            Ft_y=result.Ft_y,
            Ft_ctf=result.Ft_ctf,
            hard_full=hard_full,
            stats_subset=result.relion_stats,
            class_log_evidence=coarse_result.class_log_evidence[class_index],
            noise=result.noise_stats,
            best_pose=(
                (result.best_rotations, result.best_translations, result.best_rotation_indices)
                if return_best_pose_details
                else None
            ),
            host_accumulators=True,
        )

    logger.info(
        "Sparse firstiter-CC global-winner subset pass2: classes=%d images=%d subset_counts=%s total=%.1fs",
        n_classes,
        n_images,
        results.subset_counts,
        time.time() - t0,
    )
    return results.assemble(
        coarse_result.class_log_evidence,
        profile_summary={"sparse_firstiter_subset_pass2_s": np.float64(time.time() - t0)},
        host_accumulators=True,
        mstep_full_half_axis=0 if common["relion_x_half_mstep"] else None,
        mstep_accumulator_shape=mstep_accumulator_shape,
    )


def _supports_coarse_parents(supports_by_class, n_images, n_coarse_rot, n_coarse_trans):
    """The coarse parents any class's pass-2 rows descend from, or None for every parent."""

    from relax.sparse_pass2.resident_significance import significant_coarse_parents

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
        For a deferred fine grid (:class:`relax.helpers.oversampling.DeferredFineRows`):
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
        defers to ``RELAX_RELION_GLOBAL_PASS1_PROJECTOR_TEXTURE_INTERP``;
        the strict-parity default is RELION texture interpolation.
    """
    if "relion_exact_coarse" in engine_kwargs:
        raise TypeError(
            "relion_exact_coarse was removed on 2026-10-02: pass 1 always scores RELION's exact coarse operands"
        )
    # Lazy import to avoid the formatter stripping a top-level name that is
    # only referenced inside this function.
    from relax.scoring.significance import _compute_k_class_significance_batched
    from relax.symmetry import canonicalize_rotational_symmetry

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
    means_array = _as_class_means(means)
    n_classes = int(means_array.shape[0])
    log_priors = _class_log_priors(n_classes, class_log_priors)
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
        return _infer_healpix_order_from_rotation_count(n_rot_coarse, symmetry_label)

    def _resolved_oversampling_order() -> int:
        if oversampling_order is not None:
            return max(0, int(oversampling_order))
        return max(
            0,
            _infer_healpix_order_from_rotation_count(n_rot_fine, symmetry_label) - _resolved_coarse_healpix_order(),
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
    n_images = _dataset_image_count(experiment_dataset)
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
        ):
            dense_kwargs.pop(field, None)
        dense_kwargs["mstep_relion_x_half"] = True
        dense_kwargs["dense_gemm_full_grid"] = True
        dense_kwargs["relion_firstiter_score_mode"] = (
            "normalized_cc" if firstiter_cc_pass2_only_best_coarse else dense_kwargs.get("relion_firstiter_score_mode", "gaussian")
        )
        dense_result = _run_sparse_k_class_adaptive_pass2(
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
        from relax.sparse_pass2.engine_record import record_coarse_engine_call

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
        n_classes == 1
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
    exact_coarse_operand_assembly = None
    coarse_actual_backend = None
    pass1_t0 = time.time()
    if firstiter_cc_pass2_only_best_coarse:
        # RELION firstiter_cc branch: restrict pass-2 to children of each
        # class's per-class coarse-best pose, then gate by the global winning
        # class. This is the production default-GUI parity path for iter 1.
        coarse_probe_kwargs = dict(engine_kwargs)
        coarse_probe_kwargs.pop("rotation_translation_mask", None)
        coarse_probe_kwargs.pop("class_rotation_translation_mask", None)
        # The coarse probe is score-only.  RELION's separate model-coordinate
        # cutoff is consumed by fine-pass BPref, not by this scorer wrapper.
        coarse_probe_kwargs.pop("reconstruction_current_size", None)
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
        with score_dump_label("coarse"):
            with nvtx.annotate("kclass.adaptive.coarse_probe", color="yellow", domain=NVTX_DOMAIN_EM):
                coarse_result = _run_dense_k_class_joint_firstiter_score_probe(
                    experiment_dataset,
                    probe_means,
                    noise_variance,
                    coarse_rotations_np,
                    coarse_translations_np,
                    disc_type,
                    engine_kwargs=coarse_probe_kwargs,
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
            score_with_masked_images=engine_kwargs.get("score_with_masked_images", True),
            rotation_log_prior=significance_rotation_prior,
            translation_log_prior=coarse_translation_log_prior,
            image_corrections=engine_kwargs.get("image_corrections"),
            scale_corrections=engine_kwargs.get("scale_corrections"),
            image_pre_shifts=engine_kwargs.get("image_pre_shifts"),
            half_spectrum_scoring=engine_kwargs.get("half_spectrum_scoring", False),
            projection_padding_factor=engine_kwargs.get("projection_padding_factor", 1),
            do_gridding_correction=engine_kwargs.get("do_gridding_correction", False),
            square_window=engine_kwargs.get("square_window", False),
            window_at_box=bool(engine_kwargs.get("window_at_box", False)),
            use_float64_scoring=engine_kwargs.get("use_float64_scoring", False),
            use_float64_projections=_projection_float64_from_kwargs(engine_kwargs),
            score_mode=engine_kwargs.get("relion_firstiter_score_mode", "gaussian"),
            relion_projector_half=significance_projector_half,
            relion_projector_r_max=relion_projector_r_max,
            relion_projector_texture_interp=coarse_relion_projector_texture_interp,
            debug_iteration=debug_iteration,
            translation_phase_source=coarse_translation_phase_source,
            tree_rescore_max_margin=engine_kwargs.get("firstiter_cc_tree_rescore_max_margin"),
            optics_group_ids=engine_kwargs.get("optics_group_ids"),
            pad_final_image_batch=bool(significance_pad_final_image_batch),
            firstiter_cc_support=engine_kwargs.get("firstiter_cc_support", "relion"),
            nyquist_column_counting=engine_kwargs.get("nyquist_column_counting", "relion"),
        )
        if reuse_zero_oversampling_coarse_state:
            sig_kwargs["return_relion_f32_normalization"] = True

        with nvtx.annotate("kclass.adaptive.significance", color="orange", domain=NVTX_DOMAIN_EM):
            (
                _sig_rot_any_by_class,
                _n_sig_per_image,
                _coarse_hard_assignment,
                _coarse_class_assignment,
                sig_sample_indices_by_class,
                _full_coarse_stats,
            ) = _compute_k_class_significance_batched(
                experiment_dataset,
                significance_means,
                noise_variance,
                coarse_rotations_np,
                coarse_translations_np,
                disc_type,
                class_log_priors=significance_log_priors,
                **sig_kwargs,
                **({"symmetry_label": engine_kwargs["symmetry_label"]} if engine_kwargs.get("symmetry_label", "C1") != "C1" else {}),
                **_translation_angle_scale_kwargs(engine_kwargs),
            )
        if image_seed_classes is not None:
            sig_sample_indices_by_class = seed_iteration_supports(
                sig_sample_indices_by_class[0], image_seed_classes, n_classes
            )
        if _full_coarse_stats is None or "significant_cutoff_counts" not in _full_coarse_stats:
            raise RuntimeError("K-class significance did not return RELION cutoff-rank counts")
        significant_counts_for_result = np.asarray(
            _full_coarse_stats["significant_cutoff_counts"],
            dtype=np.int32,
        )
        coarse_actual_backend = _full_coarse_stats.get("executed_coarse_backend")
        coarse_significance_support_audit = _full_coarse_stats.get(
            "coarse_significance_support_audit",
        )
        exact_coarse_operand_assembly = _full_coarse_stats.get(
            "exact_coarse_operand_assembly",
        )
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
        result = _with_coarse_significance_diagnostics(
            result,
            support_audit=coarse_significance_support_audit,
            exact_coarse_operand_assembly=exact_coarse_operand_assembly,
        )
        if coarse_engine == "gemm_hybrid":
            from relax.sparse_pass2.engine_record import record_coarse_engine_call

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
        pass2_kwargs["relion_f32_normalization_sum_weight"] = _full_coarse_stats["relion_f32_sum_weight"]
        pass2_kwargs["relion_coarse_max_posterior"] = _full_coarse_stats["relion_f32_max_posterior"]
        pass2_kwargs["relion_coarse_hard_assignment"] = _coarse_hard_assignment
    if pass2_use_float64_scoring is not None:
        pass2_kwargs["use_float64_scoring"] = bool(pass2_use_float64_scoring)
    if pass2_use_float64_projections is not None:
        pass2_kwargs["use_float64_projections"] = bool(pass2_use_float64_projections)
    pass2_kwargs["relion_fine_mstep_prune"] = bool(relion_fine_mstep_prune)
    # Pass 2 reads each image's coarse significant samples, not a fine-grid mask.
    pass2_kwargs.pop("rotation_translation_mask", None)
    if firstiter_cc_pass2_only_best_coarse:
        pass2_kwargs.pop("rotation_log_prior", None)
        pass2_kwargs.pop("class_rotation_log_prior", None)
        pass2_kwargs.pop("translation_log_prior", None)
    if "current_size" not in pass2_kwargs and fine_current_size is not None:
        pass2_kwargs["current_size"] = fine_current_size

    device_signature_configured = bool(
        os.environ.get("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR", "").strip()
    )
    fused_atomic_env_enabled = parse_env_flag(_RELION_X_HALF_BP_FUSED_ATOMICS_ENV)
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
        pass2_kwargs["bpref_device_signature_active"] = True

    if firstiter_cc_pass2_only_best_coarse:
        if not hasattr(experiment_dataset, "subset"):
            raise NotImplementedError(
                "the --firstiter_cc fine pass scores each coarse winner class over its image subset and needs a "
                "dataset with subset(); the dense global-winner fallback was removed on 2026-10-03"
            )
        pass2_t0 = time.time()
        result = _run_sparse_firstiter_global_winner_subset_pass2(
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
            coarse_class_assignments=coarse_class_assignments,
            n_rot_coarse=n_rot_coarse,
            n_fine_trans=n_trans_fine,
            healpix_order=_resolved_coarse_healpix_order(),
            oversampling_order=_resolved_oversampling_order(),
            accumulate_noise=accumulate_noise,
            return_best_pose_details=return_best_pose_details,
            pass2_kwargs=pass2_kwargs,
        )
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
    result = _run_sparse_k_class_adaptive_pass2(
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
