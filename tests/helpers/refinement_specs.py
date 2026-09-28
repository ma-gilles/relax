"""Concise builders for production refinement owner arguments in unit tests."""

from relax.refinement import half_scoring, local_search_iteration, mean_helpers

_LOCAL_ITERATION_POSITIONAL = (
    "experiment_dataset",
    "mean",
    "noise_variance",
    "prior_rotations",
    "rotation_grid_rotations",
    "healpix_order",
    "sigma_rot",
    "sigma_psi",
    "translations",
    "prior_translations",
    "sigma_offset_angstrom",
    "disc_type",
    "image_batch_size",
    "rotation_block_size",
    "current_size",
)


def mean_reconstruction_owners(means, **values):
    """Build the five owners of the production reconstruction boundary."""

    values = dict(values)
    owners = (
        mean_helpers.MeanReconstructionData(means=means),
        mean_helpers.MeanAccumulatorState(
            Ft_y_0=values.pop("Ft_y_0"),
            Ft_y_1=values.pop("Ft_y_1"),
            Ft_ctf_0=values.pop("Ft_ctf_0"),
            Ft_ctf_1=values.pop("Ft_ctf_1"),
            Ft_y_combined=values.pop("Ft_y_combined"),
            Ft_ctf_combined=values.pop("Ft_ctf_combined"),
            retained_Ft_y_0_device=values.pop("retained_Ft_y_0_device", None),
        ),
        mean_helpers.MeanPriorSpec(
            mean_signal_variance=values.pop("mean_signal_variance"),
            mean_signal_variance_shells=values.pop("mean_signal_variance_shells"),
            mean_signal_variance_per_half=values.pop("mean_signal_variance_per_half"),
            mean_signal_variance_shells_per_half=values.pop(
                "mean_signal_variance_shells_per_half", None
            ),
            tau2_fudge=values.pop("tau2_fudge"),
            relion_minres_map=values.pop("relion_minres_map"),
        ),
        mean_helpers.MeanGeometrySpec(
            current_size=values.pop("cs"),
            grid_size=values.pop("grid_size"),
            cryo=values.pop("cryo"),
            volume_shape=values.pop("volume_shape"),
            padding_factor=values.pop("padding_factor"),
            projection_padding_factor=values.pop("projection_padding_factor"),
            accumulator_volume_shape=values.pop("accumulator_volume_shape", None),
        ),
        mean_helpers.MeanPostprocessPolicy(
            n_classes=values.pop("n_classes"),
            iteration=values.pop("iteration"),
            particle_diameter_ang=values.pop("particle_diameter_ang"),
            relion_firstiter_cc_this_iter=values.pop(
                "relion_firstiter_cc_this_iter"
            ),
            relion_firstiter_ini_high_angstrom=values.pop(
                "relion_firstiter_ini_high_angstrom"
            ),
            relion_width_mask_edge=values.pop("relion_width_mask_edge"),
            relion_fmask_edge=values.pop("relion_fmask_edge"),
        ),
    )
    assert not values, f"unmapped mean reconstruction values: {sorted(values)}"
    return owners


def local_iteration_owners(*args, **values):
    """Build the six owners of the production local-iteration boundary."""

    if len(args) > len(_LOCAL_ITERATION_POSITIONAL):
        raise TypeError(f"expected at most {len(_LOCAL_ITERATION_POSITIONAL)} positional values")
    values = dict(values)
    for name, value in zip(_LOCAL_ITERATION_POSITIONAL, args):
        if name in values:
            raise TypeError(f"multiple values for {name}")
        values[name] = value
    owners = (
        local_search_iteration.LocalSearchData(
            experiment_dataset=values.pop("experiment_dataset"),
            mean=values.pop("mean"),
            noise_variance=values.pop("noise_variance"),
            image_corrections=values.pop("image_corrections", None),
            scale_corrections=values.pop("scale_corrections", None),
            group_ids=values.pop("group_ids", None),
            scale_correction_group_count=values.pop("scale_correction_group_count", None),
            scale_correction_data_vs_prior=values.pop("scale_correction_data_vs_prior", None),
            image_pre_shifts=values.pop("image_pre_shifts", None),
            optics_group_ids=values.pop("optics_group_ids", None),
        ),
        local_search_iteration.LocalSearchGridSpec(
            prior_rotations=values.pop("prior_rotations"),
            rotation_grid_rotations=values.pop("rotation_grid_rotations"),
            healpix_order=values.pop("healpix_order"),
            sigma_rot=values.pop("sigma_rot"),
            sigma_psi=values.pop("sigma_psi"),
            translations=values.pop("translations"),
            prior_translations=values.pop("prior_translations"),
            sigma_offset_angstrom=values.pop("sigma_offset_angstrom"),
            translation_prior_reference_translations=values.pop("translation_prior_reference_translations", None),
            pass2_layout=values.pop("pass2_layout", None),
            translation_prior_centers=values.pop("translation_prior_centers", None),
            rotation_grid_random_perturbation=values.pop("rotation_grid_random_perturbation", 0.0),
            rotation_grid_angular_sampling_deg=values.pop("rotation_grid_angular_sampling_deg", None),
            local_parent_oversampling_order=values.pop("local_parent_oversampling_order", 0),
            rotation_grid_mstep_rotations=values.pop("rotation_grid_mstep_rotations", None),
            generate_relion_mstep_rotations=values.pop("generate_relion_mstep_rotations", False),
            symmetry=values.pop("symmetry", "C1"),
        ),
        local_search_iteration.LocalSearchBatchPolicy(
            image_batch_size=values.pop("image_batch_size"),
            rotation_block_size=values.pop("rotation_block_size"),
            batch_size_planner=values.pop("batch_size_planner", None),
        ),
        local_search_iteration.LocalSearchKernelPolicy(
            disc_type=values.pop("disc_type"),
            current_size=values.pop("current_size"),
            reconstruction_current_size=values.pop("reconstruction_current_size", None),
            accumulate_noise=values.pop("accumulate_noise", False),
            projection_padding_factor=values.pop("projection_padding_factor", 1),
            reconstruction_padding_factor=values.pop("reconstruction_padding_factor", 1),
            use_float64_scoring=values.pop("use_float64_scoring", False),
            use_float64_projections=values.pop("use_float64_projections", False),
            do_gridding_correction=values.pop("do_gridding_correction", False),
            square_window=values.pop("square_window", False),
            half_spectrum_scoring=values.pop("half_spectrum_scoring", False),
            relion_exact_score_translation=values.pop("relion_exact_score_translation", False),
            projection_relion_texture_interp=values.pop("projection_relion_texture_interp", False),
            projection_relion_acc_double_floorf_quirk=values.pop("projection_relion_acc_double_floorf_quirk", False),
            projection_relion_kernel=values.pop("projection_relion_kernel", "fine"),
            relion_projector_half=values.pop("relion_projector_half", None),
            relion_projector_r_max=values.pop("relion_projector_r_max", None),
            source_faithful_spectrum_norm=values.pop("source_faithful_spectrum_norm", False),
            relion_translation_angle_scale=values.pop("relion_translation_angle_scale", 1.0),
            projection_scale=values.pop("projection_scale", 1.0),
            reconstruction_volume_current_size=values.pop("reconstruction_volume_current_size", None),
            reconstruction_image_radius=values.pop("reconstruction_image_radius", None),
        ),
        local_search_iteration.LocalSearchSupportPolicy(
            mstep_relion_x_half=values.pop("mstep_relion_x_half", False),
            disable_adjoint_y=values.pop("disable_adjoint_y", False),
            disable_adjoint_ctf=values.pop("disable_adjoint_ctf", False),
            adaptive_fraction=values.pop("adaptive_fraction", 0.999),
            max_significants=values.pop("max_significants", -1),
            reconstruct_significant_only=values.pop("reconstruct_significant_only", True),
            return_best_pose_details=values.pop("return_best_pose_details", False),
            normalization_log_evidence=values.pop("normalization_log_evidence", None),
            return_reconstruction_sample_indices=values.pop("return_reconstruction_sample_indices", False),
            apply_max_significants_to_support=values.pop("apply_max_significants_to_support", False),
            stats_use_reconstruction_probs=values.pop("stats_use_reconstruction_probs", False),
            score_only=values.pop("score_only", False),
        ),
        local_search_iteration.LocalSearchDiagnosticPolicy(
            return_profile=values.pop("return_profile", False),
            debug_iteration=values.pop("debug_iteration", None),
            debug_pass_label=values.pop("debug_pass_label", None),
        ),
    )
    assert not values, f"unmapped local iteration owner values: {sorted(values)}"
    return owners


def local_half_owners(**values):
    """Build the seven explicit exact-local owners from concise test values."""

    owners = (
        half_scoring.LocalHalfData(
            k=values.pop("k"),
            experiment_dataset=values.pop("experiment_dataset"),
            means_k=values.pop("means_k"),
            noise_variance_k=values.pop("noise_variance_k"),
            previous_best_rotation_eulers_k=values.pop("previous_best_rotation_eulers_k"),
            image_corrections_k=values.pop("image_corrections_k"),
            scale_corrections_k=values.pop("scale_corrections_k"),
            outputs=values.pop("outputs"),
            group_ids_k=values.pop("group_ids_k", None),
            group_count_k=values.pop("group_count_k", None),
            scale_correction_data_vs_prior=values.pop("scale_correction_data_vs_prior", None),
            optics_group_ids_k=values.pop("optics_group_ids_k", None),
        ),
        half_scoring.LocalSamplingSpec(
            local_search_rotations=values.pop("local_search_rotations"),
            local_search_mstep_rotations=values.pop("local_search_mstep_rotations", None),
            local_search_order=values.pop("local_search_order"),
            sigma_rot=values.pop("sigma_rot"),
            sigma_psi=values.pop("sigma_psi"),
            current_translations=values.pop("current_translations"),
            base_translations=values.pop("base_translations"),
            disc_type=values.pop("disc_type"),
            cs_for_engine=values.pop("cs_for_engine"),
            model_current_size_for_engine=values.pop("model_current_size_for_engine", None),
            local_pass1_current_size=values.pop("local_pass1_current_size"),
            local_search_random_perturbation=values.pop("local_search_random_perturbation"),
            local_search_angular_sampling_deg=values.pop("local_search_angular_sampling_deg"),
            local_parent_oversampling_order=values.pop("local_parent_oversampling_order"),
            symmetry=values.pop("symmetry", "C1"),
        ),
        half_scoring.LocalPriorSpec(
            trans_prior_center=values.pop("trans_prior_center"),
            trans_prior_center_for_engine=values.pop("trans_prior_center_for_engine"),
            current_sigma_offset_angstrom=values.pop("current_sigma_offset_angstrom"),
            translation_search_base=values.pop("translation_search_base"),
            local_search_translation_prior_mode=values.pop("local_search_translation_prior_mode"),
            replay_prior_translations=values.pop("replay_prior_translations"),
        ),
        half_scoring.LocalBatchPolicy(
            max_significants=values.pop("max_significants"),
            safe_batch_sizes=values.pop("safe_batch_sizes"),
        ),
        half_scoring.LocalExecutionPolicy(
            disable_adjoint_y=values.pop("disable_adjoint_y"),
            disable_adjoint_ctf=values.pop("disable_adjoint_ctf"),
            relion_projector_half=values.pop("relion_projector_half", None),
            relion_projector_r_max=values.pop("relion_projector_r_max", None),
            source_faithful_spectrum_norm=values.pop("source_faithful_spectrum_norm", False),
            relion_translation_angle_scale=values.pop("relion_translation_angle_scale", 1.0),
        ),
        half_scoring.LocalDiagnosticPolicy(
            iteration=values.pop("iteration"),
            debug_iteration=values.pop("debug_iteration", None),
            save_intermediates_dir=values.pop("save_intermediates_dir"),
            collect_local_search_profile=values.pop("collect_local_search_profile"),
            diagnostic_score_only=values.pop("diagnostic_score_only"),
            local_profile_history=values.pop("local_profile_history"),
            bpref_device_signature_active=values.pop("bpref_device_signature_active", False),
        ),
        half_scoring.LocalOpticsSpec(
            noise_radial_k=values.pop("noise_radial_k", None),
            coarse_sizing=values.pop("coarse_sizing", None),
            class_translation_overrides=values.pop("class_translation_overrides", None),
            projection_scale=values.pop("projection_scale", 1.0),
            reference_current_size=values.pop("reference_current_size", None),
        ),
    )
    assert not values, f"unmapped local owner values: {sorted(values)}"
    return owners
