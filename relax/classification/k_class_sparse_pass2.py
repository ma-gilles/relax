"""The sparse (or resident) adaptive fine pass 2 of the dense K-class engine.

``run_sparse_k_class_adaptive_pass2`` runs pass 2 per class over the coarse supports, on the sparse route or
the resident one (``_run_resident_k_class_pass2``), and returns the K-class result;
``class_segmented_em_result`` and ``single_class_pass2_em_result`` turn a pass-2 output into it. The
first-iteration subset pass (``k_class_firstiter``) shares the pose-id, accumulator and arithmetic helpers.
The engine entry (``k_class.run_dense_k_class_em_adaptive``) selects the pass. Imports nothing from
``k_class``.
"""

from __future__ import annotations

import logging
import time

import jax
import numpy as np

from relax.classification.k_class_inputs import (
    dataset_image_count,
    select_projector_half_for_class,
    select_required_class_value,
    translation_angle_scale_kwargs,
)
from relax.classification.k_class_results import KClassEMResult, assemble_result
from relax.reconstruction.half_volume_mstep import relion_backprojector_volume_shape
from relax.types import make_relion_stats

logger = logging.getLogger("relax.classification.k_class")  # the engine's logger before the split: log rows keep it


def check_bpref_particle_order_policy(engine_kwargs: dict, *, n_classes: int) -> None:
    """Refuse RELION's preserved BPref particle order on a K>1 sparse pass (it is K=1-only)."""

    if bool(engine_kwargs.get("preserve_bpref_particle_order", False)) and int(n_classes) != 1:
        raise ValueError("RELION BPref particle-order preservation is K=1-only")


def infer_healpix_order_from_rotation_count(
    n_rot: int,
    symmetry_label: str = "C1",
) -> int:
    from relax.sampling.grids import rotation_grid_size

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


def rotation_prior_with_class_log_prior(
    rotation_log_prior, class_log_prior: float, n_rot: int, *, dtype: np.dtype = np.float32
):
    if rotation_log_prior is None:
        return np.full(int(n_rot), float(class_log_prior), dtype=dtype)
    prior = np.asarray(rotation_log_prior, dtype=dtype)
    return prior + np.asarray(float(class_log_prior), dtype=dtype)


def sparse_pose_ids_to_fine_grid(hard_assignment, best_rotation_ids, n_fine_trans: int) -> np.ndarray:
    trans_ids = np.asarray(hard_assignment, dtype=np.int64) % int(n_fine_trans)
    rot_ids = np.asarray(best_rotation_ids, dtype=np.int64)
    return (rot_ids * int(n_fine_trans) + trans_ids).astype(np.int32, copy=False)


def run_sparse_k_class_adaptive_pass2(
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

    from relax.fine_pass.dispatch import compute_pass2_stats_sparse

    n_classes = int(means_array.shape[0])
    n_rot_coarse = int(coarse_rotations_np.shape[0])
    n_fine_trans = int(fine_translations_np.shape[0])
    symmetry_label = engine_kwargs.get("symmetry_label", "C1")
    healpix_order = (
        int(coarse_healpix_order)
        if coarse_healpix_order is not None
        else infer_healpix_order_from_rotation_count(n_rot_coarse, symmetry_label)
    )
    base_engine_kwargs = dict(engine_kwargs)
    relion_projector_half_by_class = base_engine_kwargs.get("relion_projector_half")
    relion_projector_r_max = base_engine_kwargs.get("relion_projector_r_max")
    source_faithful_spectrum_norm = bool(
        base_engine_kwargs.get("source_faithful_spectrum_norm", False)
    )
    if source_faithful_spectrum_norm and n_classes != 1:
        raise ValueError("source-faithful powerClass normalization is K=1-only")

    def _class_rotation_prior(class_index: int):
        class_prior = base_engine_kwargs.get("class_rotation_log_prior")
        if class_prior is not None:
            rot_prior = select_required_class_value(
                class_prior,
                class_index,
                n_classes,
                "class_rotation_log_prior",
            )
        else:
            rot_prior = base_engine_kwargs.get("rotation_log_prior")
        return rotation_prior_with_class_log_prior(
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
        image_translations=base_engine_kwargs.get("image_translations"),
        use_float64_scoring=bool(base_engine_kwargs.get("use_float64_scoring", False)),
        translation_prior_centers=base_engine_kwargs.get("translation_prior_centers"),
        do_gridding_correction=bool(base_engine_kwargs.get("do_gridding_correction", False)),
        square_window=bool(base_engine_kwargs.get("square_window", False)),
        window_at_box=bool(base_engine_kwargs.get("window_at_box", False)),
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
        # RELION's float32 exp/sort/scan significance path is now qualified
        # for K=1.  Keep K>1 byte-preserving until its separate posterior
        # boundary is diagnosed.
        relion_f32_fine_posterior=n_classes == 1,
        # The exact normalized-CC tree is a deliberately K=1-scoped parity
        # candidate. Keep the K>1 route byte-preserving until K=1 closes.
        relion_exact_fine_normalized_cc=n_classes == 1,
        full_grid_engine=_full_grid_engine(base_engine_kwargs),
        relion_firstiter_winner_take_all=bool(
            base_engine_kwargs.get("relion_firstiter_winner_take_all", False)
        ),
        random_perturbation=float(random_perturbation),
        fine_rotations_override=fine_rotations_np,
        fine_mstep_rotations_override=fine_mstep_rotations_np,
        fine_rotation_parent_override=rot_parent_map_np,
        fine_translations_override=fine_translations_np,
        fine_translation_parent_override=trans_parent_map_np,
        source_faithful_spectrum_norm=source_faithful_spectrum_norm,
        nyquist_column_counting=base_engine_kwargs.get("nyquist_column_counting", "relion"),
        firstiter_cc_support=base_engine_kwargs.get("firstiter_cc_support", "relion"),
        engine_tuning=base_engine_kwargs["engine_tuning"],
        scoring_overrides=base_engine_kwargs["scoring_overrides"],
        **({"symmetry_label": base_engine_kwargs["symmetry_label"]}
           if base_engine_kwargs.get("symmetry_label", "C1") != "C1" else {}),
        **translation_angle_scale_kwargs(base_engine_kwargs),
        **({"keep_physical_bpref": True} if base_engine_kwargs.get("keep_physical_bpref") else {}),
    )
    # VDAM's pseudo-halfset accumulator slots (class + K * group): the resident engine only.
    reconstruction_groups = base_engine_kwargs.get("reconstruction_group_ids") is not None
    if reconstruction_groups:
        common["reconstruction_group_ids"] = base_engine_kwargs["reconstruction_group_ids"]
        common["reconstruction_group_count"] = int(base_engine_kwargs["reconstruction_group_count"])
    if base_engine_kwargs.get("wsum_current_size") is not None:
        # --strict_highres_exp: the weighted sums' image size, above the E-step's (the resident engine).
        common["wsum_current_size"] = int(base_engine_kwargs["wsum_current_size"])
    if base_engine_kwargs.get("relion_f32_normalization_sum_weight") is not None:
        common["relion_f32_normalization_sum_weight"] = base_engine_kwargs["relion_f32_normalization_sum_weight"]
        common["relion_coarse_hard_assignment"] = base_engine_kwargs.get("relion_coarse_hard_assignment")
        common["relion_coarse_max_posterior"] = base_engine_kwargs.get("relion_coarse_max_posterior")
        if n_classes > 1:
            common["relion_coarse_class_assignment"] = base_engine_kwargs["relion_coarse_class_assignment"]
    check_bpref_particle_order_policy(base_engine_kwargs, n_classes=n_classes)
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
    mstep_t0 = time.time()
    result = compute_pass2_stats_sparse(
        experiment_dataset,
        means_array[0],
        noise_variance,
        coarse_translations_np,
        sig_sample_indices_by_class[0],
        rotation_log_prior=_class_rotation_prior(0),
        accumulate_noise=accumulate_noise,
        # No other classes: the degenerate cross-class normalizer of the production K=1 pass.
        **({} if common["full_grid_engine"] is not None else {
            "normalization_other_score_log_z": np.full(
                dataset_image_count(experiment_dataset), -np.inf, dtype=np.float64,
            ),
        }),
        normalization_score_mode=(
            None if common["full_grid_engine"] is not None else common["relion_firstiter_score_mode"]
        ),
        return_score_log_z=True,
        relion_projector_half=select_projector_half_for_class(relion_projector_half_by_class, 0, n_classes),
        relion_projector_r_max=relion_projector_r_max,
        **common,
    )
    mstep_s = time.time() - mstep_t0
    logger.info(
        "Sparse adaptive K=1 pass2 profile: images=%d mstep=%.1fs",
        dataset_image_count(experiment_dataset),
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

    return assemble_result(
        class_log_evidence=np.asarray(result.relion_stats.log_evidence_per_image, dtype=np.float64)[None],
        Ft_y=[as_host_accumulator(result.Ft_y)],
        Ft_ctf=[as_host_accumulator(result.Ft_ctf)],
        per_class_hard_assignments=sparse_pose_ids_to_fine_grid(
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


def as_host_accumulator(value):
    """Copy a full-volume accumulator off GPU before retaining it."""

    return np.asarray(jax.device_get(value))


def _full_grid_engine(engine_kwargs: dict):
    """The full-grid engine the fine pass hands its resolved operands to, or None (the resident tables).

    ``engine_kwargs["dense_gemm_full_grid"]`` selects the dense GEMM engine (``coarse_engine="gemm_dense"``).
    """

    if not engine_kwargs.get("dense_gemm_full_grid", False):
        return None
    from relax.dense import gemm_coarse_engine

    return gemm_coarse_engine.run_dense_gemm_full_grid


def resident_production_arithmetic(options: dict) -> dict:
    """RELION's pass-2 arithmetic, the K=1 production set, for a K>1 pass on the resident engine.

    RELION's E-step arithmetic does not depend on the class count: its float32
    fine posterior and pruned M-step, the literal normalized-CC reduction of
    ``--firstiter_cc``, its powerClass spectrum and exact BPref operands, and
    the atomic Wavg triplet of the preserved BPref order
    (``source_faithful_spectrum_norm``). The
    compact K>1 routes keep their historical arithmetic until they are deleted.
    """

    options = dict(options)
    options.update(
        source_faithful_spectrum_norm=True,
        relion_f32_fine_posterior=True,
        relion_fine_mstep_prune=True,
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

    Runs with the K=1 production arithmetic (:func:`resident_production_arithmetic`). A
    configuration the resident checks refuse (``ResidentConfigurationUnsupported``) is an
    error: the resident engine is relax's one pass-2 engine. The pass records its engine
    (``engine_record``).
    """

    from relax.fine_pass.resident_pass2 import compute_k_class_pass2_stats_resident
    from relax.runtime.engine_record import record_pass_engine

    if engine_kwargs.get("normalization_log_evidence") is not None:
        raise NotImplementedError(
            "an externally supplied normalization is not part of the resident K-class pass"
        )
    n_classes = int(means_array.shape[0])
    options = resident_production_arithmetic(common)
    if options.get("full_grid_engine") is None:
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
    record_pass_engine("global", "gemm_dense" if options.get("full_grid_engine") is not None else "resident")
    logger.info(
        "Resident K-class pass2: classes=%d images=%d total=%.1fs",
        n_classes,
        dataset_image_count(experiment_dataset),
        time.time() - t0,
    )
    return class_segmented_em_result(
        output,
        n_classes=n_classes,
        host_accumulators=True,
        mstep_full_half_axis=0 if common["relion_x_half_mstep"] else None,
        mstep_accumulator_shape=mstep_accumulator_shape,
        # Zero oversampling: the particle's class is the coarse winner's (docs/math/zero_oversampling.md).
        class_assignments_override=output.coarse_class_assignments,
    )


def class_segmented_em_result(
    output,
    *,
    n_classes: int,
    **accumulator_layout,
) -> KClassEMResult:
    """The K-class result of one engine call that scored every class jointly.

    ``output`` is the class-segmented result of the exact-local engine or of the
    resident pass 2 (``ResidentKClassPass2Output``, the same field names).
    ``accumulator_layout`` is passed to ``assemble_result`` as is.
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
    return assemble_result(
        class_log_evidence=class_log_evidence,
        # One engine call scored every class jointly, so its joint max posterior is
        # authoritative; see assemble_result for why it is not rebuilt downstream.
        joint_max_posterior_per_image=np.asarray(output.stats.max_posterior_per_image),
        Ft_y=[output.Ft_y[class_index] for class_index in range(n_classes)],
        Ft_ctf=[output.Ft_ctf[class_index] for class_index in range(n_classes)],
        per_class_hard_assignments=np.asarray(output.per_class_hard_assignments, dtype=np.int64),
        per_class_stats=per_class_stats,
        noise_stats=None,
        aggregate_noise_stats_override=output.noise_stats,
        class_posterior_sums_override=np.asarray(output.class_reconstruction_posterior_sums, dtype=np.float64),
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
        uncast_log_evidence_per_image=output.uncast_log_evidence_per_image,
        **accumulator_layout,
    )
