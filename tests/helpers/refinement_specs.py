"""Concise builders for production refinement owner arguments in unit tests."""

import dataclasses

from relax.dense import scoring_policy
from relax.refinement import (
    half_inputs,
    half_scoring,
    local_half,
    local_sampling,
    local_search_iteration,
    optics_shapes,
)
from relax.refinement.half_inputs import HalfSet
from relax.refinement.ports import NoProbe
from relax.refinement.projector_preparation import PreparedProjector
from relax.refinement.refinement_options import LocalAdaptivePass2Support, ScoringVariants
from relax.sparse_pass2 import local_search_records

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
    "current_size",
)


def local_iteration_owners(*args, **values):
    """Build the four owners of the production local-iteration boundary."""

    if len(args) > len(_LOCAL_ITERATION_POSITIONAL):
        raise TypeError(f"expected at most {len(_LOCAL_ITERATION_POSITIONAL)} positional values")
    values = dict(values)
    for name, value in zip(_LOCAL_ITERATION_POSITIONAL, args):
        if name in values:
            raise TypeError(f"multiple values for {name}")
        values[name] = value
    owners = (
        local_search_records.LocalSearchData(
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
            n_classes=values.pop("n_classes", 1),
        ),
        local_search_records.LocalSearchKernelPolicy(
            disc_type=values.pop("disc_type"),
            current_size=values.pop("current_size"),
            reconstruction_current_size=values.pop("reconstruction_current_size", None),
            accumulate_noise=values.pop("accumulate_noise", False),
            projection_padding_factor=values.pop("projection_padding_factor", 1),
            reconstruction_padding_factor=values.pop("reconstruction_padding_factor", 1),
            use_float64_scoring=values.pop("use_float64_scoring", False),
            use_float64_projections=values.pop("use_float64_projections", False),
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
            nyquist_column_counting=values.pop("nyquist_column_counting", "relion"),
            wsum_current_size=values.pop("wsum_current_size", None),
            firstiter_cc=values.pop("firstiter_cc", False),
        ),
        local_search_records.LocalSearchSupportPolicy(
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
            return_profile=values.pop("return_profile", False),
        ),
    )
    assert not values, f"unmapped local iteration owner values: {sorted(values)}"
    return owners


def local_iteration_keywords(fake):
    """Adapt a keyword-style fake to the four-owner local-iteration boundary.

    Every owner field carries its former keyword name, so the fake receives the
    same names it read before the owners existed.
    """

    def call_with_keywords(*owners):
        values = {}
        for owner in owners:
            for field in dataclasses.fields(owner):
                values[field.name] = getattr(owner, field.name)
        return fake(**values)

    return call_with_keywords


def local_half_owners(**values):
    """Build the seven explicit exact-local owners from concise test values."""

    projector_data = values.pop("relion_projector_half", None)
    projector_r_max = values.pop("relion_projector_r_max", None)
    projector = None if projector_data is None else PreparedProjector(data=projector_data, r_max=projector_r_max)
    owners = (
        half_inputs.HalfScoringData(
            particles=HalfSet(
                index=values.pop("k"),
                dataset=values.pop("experiment_dataset"),
                rotation_eulers=values.pop("previous_best_rotation_eulers_k"),
                optics_group_ids=values.pop("optics_group_ids_k", None),
                image_corrections=values.pop("image_corrections_k"),
                scale_corrections=values.pop("scale_corrections_k"),
            ),
            reference=values.pop("means_k"),
            projector=projector,
            noise_variance=values.pop("noise_variance_k"),
            scale_group_ids=values.pop("group_ids_k", None),
            scale_group_count=values.pop("group_count_k", None),
            scale_correction_data_vs_prior=values.pop("scale_correction_data_vs_prior", None),
        ),
        local_sampling.LocalSampling(
            search=local_sampling.LocalSearchSettings(
                healpix_order=values.pop("local_search_order"),
                oversampling_order=values.pop("local_parent_oversampling_order"),
                sigma_rot=values.pop("sigma_rot"),
                sigma_psi=values.pop("sigma_psi"),
                symmetry=values.pop("symmetry", "C1"),
            ),
            rotations=values.pop("local_search_rotations"),
            mstep_rotations=values.pop("local_search_mstep_rotations", None),
            translations=values.pop("current_translations"),
            base_translations=values.pop("base_translations"),
            image_window_size=values.pop("cs_for_engine"),
            model_support_size=values.pop("model_current_size_for_engine", None),
            coarse_image_window_size=values.pop("local_pass1_current_size"),
            perturbation=values.pop("local_search_random_perturbation"),
            angular_step_deg=values.pop("local_search_angular_sampling_deg"),
        ),
        local_half.LocalPriorSpec(
            trans_prior_center=values.pop("trans_prior_center"),
            trans_prior_center_for_engine=values.pop("trans_prior_center_for_engine"),
            current_sigma_offset_angstrom=values.pop("current_sigma_offset_angstrom"),
            translation_search_base=values.pop("translation_search_base"),
            replay_prior_translations=values.pop("replay_prior_translations"),
        ),
        local_half.LocalBatchPolicy(
            max_significants=values.pop("max_significants"),
        ),
        local_half.LocalExecutionPolicy(
            disc_type=values.pop("disc_type"),
            source_faithful_spectrum_norm=values.pop("source_faithful_spectrum_norm", False),
            relion_translation_angle_scale=values.pop("relion_translation_angle_scale", 1.0),
            nyquist_column_counting=values.pop("nyquist_column_counting", "relion"),
            # The run's ScoringVariants, read from the environment as the options would be, unless given.
            relion_x_half_mstep=values.pop("relion_x_half_mstep", None)
            if "relion_x_half_mstep" in values
            else ScoringVariants.from_environ().k1_relion_x_half_mstep,
            adaptive_pass2=LocalAdaptivePass2Support(
                full_parent=values.pop("adaptive_pass2_full_parent", False),
                rotation_only=values.pop("adaptive_pass2_rotation_only", False),
                denominator_mode=values.pop("adaptive_pass2_denominator_mode", None),
            ),
            # The process precision, and pass 2's for the debug iteration (iteration + 1 by default).
            precision=scoring_policy.DENSE_PRECISION,
            fine_precision=scoring_policy.local_precision(
                scoring_policy.DENSE_PRECISION,
                values["iteration"] + 1 if values.get("debug_iteration") is None else values["debug_iteration"],
                pass_index=2,
            ),
            score_only=values.pop("diagnostic_score_only"),
            firstiter_cc=values.pop("firstiter_cc", False),
        ),
        local_half.LocalDiagnosticPolicy(
            iteration=values.pop("iteration"),
            debug_iteration=values.pop("debug_iteration", None),
            collect_local_search_profile=values.pop("collect_local_search_profile"),
            local_profile_history=values.pop("local_profile_history"),
            bpref_device_signature_active=values.pop("bpref_device_signature_active", False),
            probe=values.pop("probe", NoProbe()),
        ),
        optics_shapes.OpticsSpec(
            noise_radial_k=values.pop("noise_radial_k", None),
            coarse_sizing=values.pop("coarse_sizing", None),
            class_translations=values.pop("class_translations", None),
            projection_scale=values.pop("projection_scale", 1.0),
            reference_current_size=values.pop("reference_current_size", None),
        ),
    )
    assert not values, f"unmapped local owner values: {sorted(values)}"
    return owners


# RELION's values of the dense policy fields a test does not set: the production builders
# (expectation.score_numbered_half, finalization.run_final_all_data) pass every field.
def dense_sampling_spec(**fields):
    return half_scoring.DenseSamplingSpec(**{"coarse_engine": "auto", "symmetry": "C1", **fields})


def dense_batch_policy(**fields):
    unset = dict.fromkeys(
        (
            "significance_safe_batch_sizes",
            "k_class_image_batch_size_override",
            "k_class_rotation_block_size_override",
            "significance_image_batch_size_override",
            "significance_rotation_block_size_override",
            "class_batch_overrides",
        )
    )
    return half_scoring.DenseBatchPolicy(**{**unset, **fields})


def dense_variant_policy(**fields):
    return half_scoring.DenseVariantPolicy(
        **{
            "coarse_window_size": None,
            "fine_window_size": None,
            "skip_align": False,
            **fields,
        }
    )


def dense_execution_policy(**fields):
    return half_scoring.DenseExecutionPolicy(
        **{
            "return_best_pose_details": True,
            "bpref_device_signature_active": False,
            "debug_iteration": None,
            "diagnostic_float64_pass2": False,
            "preserve_bpref_particle_order": False,
            "source_faithful_spectrum_norm": False,
            "relion_translation_angle_scale": 1.0,
            "firstiter_cc_tree_rescore_max_margin": None,
            "firstiter_cc_support": "relion",
            "nyquist_column_counting": "relion",
            **fields,
        }
    )
