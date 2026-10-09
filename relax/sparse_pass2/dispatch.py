"""Route a sparse pass 2 to the device-resident engine."""

import logging

from relax.sparse_pass2.engine_record import record_pass_engine

# Preserve the category consumed by existing run-log collectors.
logger = logging.getLogger("relax.helpers.oversampling")


def compute_pass2_stats_sparse(
    experiment_dataset,
    volume,
    mean_variance,
    noise_variance,
    translations,
    significant_sample_indices,
    nside_level,
    disc_type,
    oversampling_order=1,
    current_size=None,
    reconstruction_current_size=None,
    translation_step=None,
    *,
    rotation_log_prior=None,
    score_with_masked_images=False,
    return_stats=False,
    translation_log_prior=None,
    accumulate_noise=False,
    half_spectrum_scoring=False,
    projection_padding_factor=1,
    projection_mask_current_image_disk=False,
    reconstruction_padding_factor=1,
    image_corrections=None,
    scale_corrections=None,
    group_ids=None,
    scale_correction_group_count=None,
    scale_correction_data_vs_prior=None,
    image_pre_shifts=None,
    image_translations=None,
    use_float64_scoring=False,
    do_gridding_correction=False,
    square_window=False,
    window_at_box=False,
    random_perturbation=0.0,
    translation_prior_centers=None,
    normalization_log_z=None,
    relion_f32_normalization_sum_weight=None,
    relion_coarse_hard_assignment=None,
    relion_coarse_max_posterior=None,
    normalization_other_score_log_z=None,
    normalization_score_mode=None,
    return_score_log_z=False,
    return_score_log_z_only=False,
    fine_source_eulers_override=None,
    return_source_eulers=False,
    fine_rotations_override=None,
    fine_mstep_rotations_override=None,
    fine_rotation_parent_override=None,
    fine_translations_override=None,
    fine_translation_parent_override=None,
    relion_half_volume_mstep=False,
    relion_x_half_mstep=False,
    mstep_subtract_ctf_projection=False,
    relion_fine_mstep_prune=False,
    relion_firstiter_score_mode="gaussian",
    relion_firstiter_winner_take_all=False,
    relion_exact_fine_gaussian=True,
    relion_fine_diff2_fused_ffi=False,
    relion_f32_fine_posterior=False,
    relion_exact_fine_normalized_cc=False,
    relion_projector_half=None,
    relion_projector_r_max=None,
    adaptive_fraction=0.999,
    bpref_device_signature_active: bool = False,
    bpref_class_index: int = 0,
    include_unweighted_norm_high_shell: bool = True,
    preserve_bpref_particle_order: bool = False,
    source_faithful_spectrum_norm: bool = False,
    symmetry_label: str = "C1",
    keep_physical_bpref: bool = False,
    relion_translation_angle_scale: float = 1.0,
    optics_group_ids=None,
    reconstruction_volume_current_size=None,
    reconstruction_image_radius=None,
    reconstruction_group_ids=None,
    reconstruction_group_count=None,
    dense_gemm_full_grid: bool = False,
    nyquist_column_counting: str = "relion",
    firstiter_cc_support: str = "relion",
):
    """Exact sparse pass 2 over per-image significant coarse samples.

    This matches RELION's pass-2 structure more closely than the dense-union
    approximation: each image only evaluates oversampled children of the
    coarse (rotation, translation) pairs that survived pass 1.

    Implementation note
    -------------------
    The device-resident driver
    (:func:`relax.sparse_pass2.resident_pass2.compute_pass2_stats_resident`) runs every
    pass; a configuration its checks refuse is an error. The per-image reference pass 2
    (a dense ``run_em`` per image) was removed on 2026-10-03.

    ``relion_exact_fine_gaussian`` selects the float32 scorer that follows
    RELION's fine-search diff2/minimum ordering. Float64 diagnostics retain
    the historical algebraic scorer so they do not silently downcast.
    """
    from relax.symmetry import canonicalize_rotational_symmetry

    symmetry_label = canonicalize_rotational_symmetry(symmetry_label)
    has_external_score_normalization = (
        normalization_log_z is not None or normalization_other_score_log_z is not None
    )
    if has_external_score_normalization and normalization_score_mode is None:
        raise ValueError(
            "external sparse pass-2 score normalization requires normalization_score_mode; "
            "Gaussian logZ is absolute while normalized-CC logZ is centered"
        )
    if normalization_score_mode is not None and not has_external_score_normalization:
        raise ValueError("normalization_score_mode requires an external score normalization")
    if (
        normalization_score_mode is not None
        and normalization_score_mode != relion_firstiter_score_mode
    ):
        raise ValueError(
            "external score normalization mode does not match this pass: "
            f"external={normalization_score_mode!r}, pass={relion_firstiter_score_mode!r}"
        )
    from relax.sparse_pass2.resident_pass2 import compute_pass2_stats_resident

    # The device-resident pass 2 is relax's one pass-2 engine: a configuration its
    # checks refuse (ResidentConfigurationUnsupported) is an error, not a fallback.
    record_pass_engine("global", "gemm_dense" if dense_gemm_full_grid else "resident")
    return compute_pass2_stats_resident(
        experiment_dataset,
        volume,
        noise_variance,
        translations,
        significant_sample_indices,
        nside_level,
        disc_type,
        oversampling_order=oversampling_order,
        current_size=current_size,
        reconstruction_current_size=reconstruction_current_size,
        translation_step=translation_step,
        rotation_log_prior=rotation_log_prior,
        score_with_masked_images=score_with_masked_images,
        return_stats=return_stats,
        translation_log_prior=translation_log_prior,
        accumulate_noise=accumulate_noise,
        half_spectrum_scoring=half_spectrum_scoring,
        projection_padding_factor=projection_padding_factor,
        projection_mask_current_image_disk=projection_mask_current_image_disk,
        reconstruction_padding_factor=reconstruction_padding_factor,
        image_corrections=image_corrections,
        scale_corrections=scale_corrections,
        group_ids=group_ids,
        scale_correction_group_count=scale_correction_group_count,
        scale_correction_data_vs_prior=scale_correction_data_vs_prior,
        image_pre_shifts=image_pre_shifts,
        image_translations=image_translations,
        use_float64_scoring=use_float64_scoring,
        translation_prior_centers=translation_prior_centers,
        do_gridding_correction=do_gridding_correction,
        square_window=square_window,
        window_at_box=window_at_box,
        random_perturbation=random_perturbation,
        normalization_log_z=normalization_log_z,
        relion_f32_normalization_sum_weight=relion_f32_normalization_sum_weight,
        relion_coarse_hard_assignment=relion_coarse_hard_assignment,
        relion_coarse_max_posterior=relion_coarse_max_posterior,
        normalization_other_score_log_z=normalization_other_score_log_z,
        normalization_score_mode=normalization_score_mode,
        return_score_log_z=return_score_log_z,
        return_score_log_z_only=return_score_log_z_only,
        fine_source_eulers_override=fine_source_eulers_override,
        return_source_eulers=return_source_eulers,
        fine_rotations_override=fine_rotations_override,
        fine_mstep_rotations_override=fine_mstep_rotations_override,
        fine_rotation_parent_override=fine_rotation_parent_override,
        fine_translations_override=fine_translations_override,
        fine_translation_parent_override=fine_translation_parent_override,
        relion_half_volume_mstep=relion_half_volume_mstep,
        relion_x_half_mstep=relion_x_half_mstep,
        mstep_subtract_ctf_projection=mstep_subtract_ctf_projection,
        relion_fine_mstep_prune=relion_fine_mstep_prune,
        relion_firstiter_score_mode=relion_firstiter_score_mode,
        relion_firstiter_winner_take_all=relion_firstiter_winner_take_all,
        relion_exact_fine_gaussian=relion_exact_fine_gaussian,
        relion_fine_diff2_fused_ffi=relion_fine_diff2_fused_ffi,
        relion_f32_fine_posterior=relion_f32_fine_posterior,
        relion_exact_fine_normalized_cc=relion_exact_fine_normalized_cc,
        relion_projector_half=relion_projector_half,
        relion_projector_r_max=relion_projector_r_max,
        adaptive_fraction=adaptive_fraction,
        dense_gemm_full_grid=dense_gemm_full_grid,
        bpref_device_signature_active=bpref_device_signature_active,
        bpref_class_index=bpref_class_index,
        include_unweighted_norm_high_shell=include_unweighted_norm_high_shell,
        preserve_bpref_particle_order=preserve_bpref_particle_order,
        source_faithful_spectrum_norm=source_faithful_spectrum_norm,
        nyquist_column_counting=nyquist_column_counting,
        firstiter_cc_support=firstiter_cc_support,
        **({"symmetry_label": symmetry_label} if symmetry_label != "C1" else {}),
        **({"keep_physical_bpref": True} if keep_physical_bpref else {}),
        **(
            {"relion_translation_angle_scale": float(relion_translation_angle_scale)}
            if float(relion_translation_angle_scale) != 1.0
            else {}
        ),
        **({"optics_group_ids": optics_group_ids} if optics_group_ids is not None else {}),
        **(
            {
                "reconstruction_group_ids": reconstruction_group_ids,
                "reconstruction_group_count": reconstruction_group_count,
            }
            if reconstruction_group_ids is not None or reconstruction_group_count is not None
            else {}
        ),
        **(
            {"reconstruction_volume_current_size": int(reconstruction_volume_current_size)}
            if reconstruction_volume_current_size is not None
            else {}
        ),
        **(
            {"reconstruction_image_radius": float(reconstruction_image_radius)}
            if reconstruction_image_radius is not None
            else {}
        ),
    )
