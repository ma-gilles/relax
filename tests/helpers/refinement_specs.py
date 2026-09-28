"""Concise builders for production refinement specifications in unit tests."""

from relax.refinement import half_scoring


def local_half_spec(**values):
    """Group legacy-sized test fixtures into the exact-local production spec."""

    spec = half_scoring.LocalHalfScoringSpec(
        half=half_scoring.LocalHalfData(
            k=values.pop("k"),
            experiment_dataset=values.pop("experiment_dataset"),
            means_k=values.pop("means_k"),
            noise_variance_k=values.pop("noise_variance_k"),
            previous_best_rotation_eulers_k=values.pop(
                "previous_best_rotation_eulers_k"
            ),
            image_corrections_k=values.pop("image_corrections_k"),
            scale_corrections_k=values.pop("scale_corrections_k"),
            outputs=values.pop("outputs"),
            group_ids_k=values.pop("group_ids_k", None),
            group_count_k=values.pop("group_count_k", None),
            scale_correction_data_vs_prior=values.pop(
                "scale_correction_data_vs_prior", None
            ),
            optics_group_ids_k=values.pop("optics_group_ids_k", None),
        ),
        sampling=half_scoring.LocalSamplingSpec(
            local_search_rotations=values.pop("local_search_rotations"),
            local_search_mstep_rotations=values.pop(
                "local_search_mstep_rotations", None
            ),
            local_search_order=values.pop("local_search_order"),
            sigma_rot=values.pop("sigma_rot"),
            sigma_psi=values.pop("sigma_psi"),
            current_translations=values.pop("current_translations"),
            base_translations=values.pop("base_translations"),
            disc_type=values.pop("disc_type"),
            cs_for_engine=values.pop("cs_for_engine"),
            model_current_size_for_engine=values.pop(
                "model_current_size_for_engine", None
            ),
            local_pass1_current_size=values.pop("local_pass1_current_size"),
            local_search_random_perturbation=values.pop(
                "local_search_random_perturbation"
            ),
            local_search_angular_sampling_deg=values.pop(
                "local_search_angular_sampling_deg"
            ),
            local_parent_oversampling_order=values.pop(
                "local_parent_oversampling_order"
            ),
            symmetry=values.pop("symmetry", "C1"),
        ),
        priors=half_scoring.LocalPriorSpec(
            trans_prior_center=values.pop("trans_prior_center"),
            trans_prior_center_for_engine=values.pop(
                "trans_prior_center_for_engine"
            ),
            current_sigma_offset_angstrom=values.pop(
                "current_sigma_offset_angstrom"
            ),
            translation_search_base=values.pop("translation_search_base"),
            local_search_translation_prior_mode=values.pop(
                "local_search_translation_prior_mode"
            ),
            replay_prior_translations=values.pop("replay_prior_translations"),
        ),
        batching=half_scoring.LocalBatchPolicy(
            max_significants=values.pop("max_significants"),
            safe_batch_sizes=values.pop("safe_batch_sizes"),
        ),
        execution=half_scoring.LocalExecutionPolicy(
            disable_adjoint_y=values.pop("disable_adjoint_y"),
            disable_adjoint_ctf=values.pop("disable_adjoint_ctf"),
            relion_projector_half=values.pop("relion_projector_half", None),
            relion_projector_r_max=values.pop("relion_projector_r_max", None),
            source_faithful_spectrum_norm=values.pop(
                "source_faithful_spectrum_norm", False
            ),
            relion_translation_angle_scale=values.pop(
                "relion_translation_angle_scale", 1.0
            ),
        ),
        diagnostics=half_scoring.LocalDiagnosticPolicy(
            iteration=values.pop("iteration"),
            debug_iteration=values.pop("debug_iteration", None),
            save_intermediates_dir=values.pop("save_intermediates_dir"),
            collect_local_search_profile=values.pop(
                "collect_local_search_profile"
            ),
            diagnostic_score_only=values.pop("diagnostic_score_only"),
            local_profile_history=values.pop("local_profile_history"),
            bpref_device_signature_active=values.pop(
                "bpref_device_signature_active", False
            ),
        ),
        optics=half_scoring.LocalOpticsSpec(
            noise_radial_k=values.pop("noise_radial_k", None),
            coarse_sizing=values.pop("coarse_sizing", None),
            class_translation_overrides=values.pop(
                "class_translation_overrides", None
            ),
            projection_scale=values.pop("projection_scale", 1.0),
            reference_current_size=values.pop("reference_current_size", None),
        ),
    )
    assert not values, f"unmapped local spec values: {sorted(values)}"
    return spec
