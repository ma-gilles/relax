"""Select the supported sparse pass-2 engine without changing admission policy."""

import functools
import logging

import numpy as np

from relax.reference.sparse_pass2 import _compute_pass2_stats_sparse_perimage_reference
from relax.sparse_pass2.engine_record import record_pass_engine, warn_deprecated_engine
from relax.sparse_pass2.sparse_pass2_policy import resident_refusal_reason

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
    disable_adjoint_y=False,
    disable_adjoint_ctf=False,
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
    use_perimage_reference=False,
    bpref_device_signature_active: bool = False,
    bpref_class_index: int = 0,
    include_unweighted_norm_high_shell: bool = True,
    preserve_bpref_particle_order: bool = False,
    source_faithful_spectrum_norm: bool = False,
    symmetry_label: str = "C1",
    relion_translation_angle_scale: float = 1.0,
    optics_group_ids=None,
    reconstruction_volume_current_size=None,
    reconstruction_image_radius=None,
    reconstruction_group_ids=None,
    reconstruction_group_count=None,
):
    """Exact sparse pass 2 over per-image significant coarse samples.

    This matches RELION's pass-2 structure more closely than the dense-union
    approximation: each image only evaluates oversampled children of the
    coarse (rotation, translation) pairs that survived pass 1.

    Implementation note
    -------------------
    By default the bucketed batched implementation in
    :mod:`recovar.em.sparse_pass2.sparse_pass2_bucketed`
    is used: images are grouped by their oversampled rotation count
    (quantized) and evaluated as one GPU call per bucket, which keeps
    the number of distinct XLA shapes bounded across iterations.

    The original per-image Python loop is preserved as
    :func:`_compute_pass2_stats_sparse_perimage_reference` for testing
    parity; it can be selected by setting ``use_perimage_reference=True``.
    The two paths must produce identical outputs (modulo float rounding).

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
    if use_perimage_reference and (return_score_log_z or return_score_log_z_only):
        raise NotImplementedError("score-logZ returns are only implemented for the bucketed sparse pass-2 path")
    if (
        use_perimage_reference
        and relion_exact_fine_gaussian
        and relion_firstiter_score_mode == "gaussian"
        and not use_float64_scoring
    ):
        raise NotImplementedError(
            "exact RELION fine Gaussian scoring requires the bucketed sparse pass-2 path"
        )
    if use_perimage_reference and group_ids is not None:
        logger.warning(
            "Sparse per-image reference pass-2 does not accumulate native group-scale correction stats; "
            "use the bucketed sparse pass-2 path for native scale updates."
        )
    if use_perimage_reference and fine_mstep_rotations_override is not None:
        raise NotImplementedError(
            "fine_mstep_rotations_override is only implemented for the bucketed sparse pass-2 path",
        )
    full_grid_reference = (
        symmetry_label == "C1"
        and all(samples is None for samples in significant_sample_indices)
        and not return_score_log_z
        and not return_score_log_z_only
        and normalization_log_z is None
        and relion_f32_normalization_sum_weight is None
        and normalization_other_score_log_z is None
        and normalization_score_mode is None
        and group_ids is None
        and scale_correction_group_count is None
        and scale_correction_data_vs_prior is None
        and fine_rotations_override is None
        and fine_mstep_rotations_override is None
        and fine_rotation_parent_override is None
        and fine_translations_override is None
        and fine_translation_parent_override is None
        and not relion_x_half_mstep
        and not relion_fine_mstep_prune
        and relion_firstiter_score_mode == "gaussian"
        and not relion_firstiter_winner_take_all
        and include_unweighted_norm_high_shell
        and not preserve_bpref_particle_order
        and reconstruction_current_size is None
        and not (
            relion_exact_fine_gaussian
            and not use_float64_scoring
        )
    )
    if not use_perimage_reference and not full_grid_reference:
        from relax.sparse_pass2.resident_pass2 import (
            RESIDENT_PASS2_ENV,
            compute_pass2_stats_resident,
            resident_pass2_out_of_scope_reason,
            resident_pass2_requested,
        )
        from relax.sparse_pass2.sparse_pass2_bucketed import compute_pass2_stats_sparse_bucketed
        from relax.sparse_pass2.sparse_pass2_policy import resident_engine_selection

        # The device-resident K=1 driver is the default (RELAX_SPARSE_PASS2_RESIDENT
        # unset). Under an explicit =1 it raises a named NotImplementedError on
        # any configuration mismatch rather than falling back, so a measured
        # comparison always knows which engine produced a result; with the
        # default a mismatch runs the compact engine and logs why
        # (_resident_with_compact_default). The scoring route it was never
        # scoped to cover is different (the unordered Wavg arithmetic of subset
        # replays; see
        # resident_pass2_out_of_scope_reason), so those passes go to the
        # compact engine and the log says which and why.
        sparse_pass2_impl = compute_pass2_stats_sparse_bucketed
        compact_reason = f"{RESIDENT_PASS2_ENV}=0"
        if resident_pass2_requested():
            out_of_scope = resident_pass2_out_of_scope_reason(
                accumulate_noise=accumulate_noise,
                scale_groups_available=group_ids is not None,
                preserve_bpref_particle_order=preserve_bpref_particle_order,
                source_faithful_spectrum_norm=source_faithful_spectrum_norm,
            )
            if out_of_scope is None:
                sparse_pass2_impl = compute_pass2_stats_resident
            else:
                compact_reason = f"out of resident scope: {out_of_scope}"
                logger.info(
                    "Device-resident sparse pass 2 is enabled but does not cover %s; "
                    "this pass runs on the compact engine",
                    out_of_scope,
                )
        texture = None
        if sparse_pass2_impl is compute_pass2_stats_sparse_bucketed and not use_float64_scoring:
            texture = _open_persistent_relion_projector_texture(
                relion_projector_half,
                relion_projector_r_max=relion_projector_r_max,
                projection_padding_factor=projection_padding_factor,
            )
        from relax.diagnostics import resident_shadow

        def open_compact_texture():
            return _open_persistent_relion_projector_texture(
                relion_projector_half,
                relion_projector_r_max=relion_projector_r_max,
                projection_padding_factor=projection_padding_factor,
            )

        if (
            sparse_pass2_impl is compute_pass2_stats_resident
            and resident_shadow.shadow_dir() is None
            and resident_engine_selection(RESIDENT_PASS2_ENV) == "default"
        ):
            sparse_pass2_impl = functools.partial(
                _resident_with_compact_default,
                compute_pass2_stats_resident,
                compute_pass2_stats_sparse_bucketed,
                open_compact_texture,
            )
        if sparse_pass2_impl is compute_pass2_stats_resident and resident_shadow.shadow_dir() is not None:
            sparse_pass2_impl = functools.partial(
                resident_shadow.run_resident_with_compact_shadow,
                compute_pass2_stats_resident,
                compute_pass2_stats_sparse_bucketed,
                open_compact_texture,
            )
        # The default wrapper records its own outcome; the other routes are fixed here.
        if sparse_pass2_impl is compute_pass2_stats_sparse_bucketed:
            record_pass_engine("global", "compact", compact_reason)
        elif not isinstance(sparse_pass2_impl, functools.partial) or (
            sparse_pass2_impl.func is not _resident_with_compact_default
        ):
            record_pass_engine("global", "resident")
        return _call_with_persistent_texture_cleanup(
            texture, sparse_pass2_impl,
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
            disable_adjoint_y=disable_adjoint_y,
            disable_adjoint_ctf=disable_adjoint_ctf,
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
            relion_projector_half=relion_projector_half if texture is None else None,
            **({"relion_projector_texture": texture} if texture is not None else {}),
            relion_projector_r_max=relion_projector_r_max,
            adaptive_fraction=adaptive_fraction,
            bpref_device_signature_active=bpref_device_signature_active,
            bpref_class_index=bpref_class_index,
            include_unweighted_norm_high_shell=include_unweighted_norm_high_shell,
            preserve_bpref_particle_order=preserve_bpref_particle_order,
            source_faithful_spectrum_norm=source_faithful_spectrum_norm,
            **({"symmetry_label": symmetry_label} if symmetry_label != "C1" else {}),
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

    if any(value is not None for value in (
        optics_group_ids, reconstruction_volume_current_size, reconstruction_image_radius,
    )):
        raise NotImplementedError(
            "per-optics-group noise and images on another grid run on the device-resident "
            "sparse pass 2 only"
        )
    if any(value is not None for value in (
        relion_f32_normalization_sum_weight, relion_coarse_hard_assignment, relion_coarse_max_posterior,
    )):
        raise NotImplementedError("coarse float32 normalization requires bucketed sparse pass 2")
    if float(relion_translation_angle_scale) != 1.0:
        raise NotImplementedError(
            "RELION model/optics translation-angle scaling requires the bucketed sparse pass-2 path"
        )
    if relion_projector_half is not None:
        raise NotImplementedError("RELION projector sparse pass-2 requires the bucketed implementation")
    if reconstruction_current_size is not None:
        raise NotImplementedError(
            "separate score/reconstruction current sizes require the bucketed sparse pass-2 path",
        )
    if window_at_box and (current_size is None or int(current_size) >= int(experiment_dataset.image_shape[0])):
        raise NotImplementedError("RELION's window at the full box requires the bucketed sparse pass-2 path")

    # DEPRECATED route: to be removed once the resident engine covers a full-grid C1 pass
    # without significance supports; see em_status 'One engine' TODO.
    warn_deprecated_engine(
        "per_image_reference",
        "global",
        "use_perimage_reference=True"
        if use_perimage_reference
        else "a full-grid C1 pass without significance supports takes the per-image reference",
    )
    return _compute_pass2_stats_sparse_perimage_reference(
        experiment_dataset,
        volume,
        mean_variance,
        noise_variance,
        translations,
        significant_sample_indices,
        nside_level,
        disc_type,
        oversampling_order=oversampling_order,
        current_size=current_size,
        translation_step=translation_step,
        rotation_log_prior=rotation_log_prior,
        score_with_masked_images=score_with_masked_images,
        return_stats=return_stats,
        translation_log_prior=translation_log_prior,
        accumulate_noise=accumulate_noise,
        half_spectrum_scoring=half_spectrum_scoring,
        projection_padding_factor=projection_padding_factor,
        reconstruction_padding_factor=reconstruction_padding_factor,
        image_corrections=image_corrections,
        scale_corrections=scale_corrections,
        image_pre_shifts=image_pre_shifts,
        translation_prior_centers=translation_prior_centers,
        use_float64_scoring=use_float64_scoring,
        do_gridding_correction=do_gridding_correction,
        square_window=square_window,
        random_perturbation=random_perturbation,
        normalization_log_z=normalization_log_z,
        normalization_other_score_log_z=normalization_other_score_log_z,
        disable_adjoint_y=disable_adjoint_y,
        disable_adjoint_ctf=disable_adjoint_ctf,
        relion_half_volume_mstep=relion_half_volume_mstep,
        relion_firstiter_score_mode=relion_firstiter_score_mode,
        relion_firstiter_winner_take_all=relion_firstiter_winner_take_all,
        **({"symmetry_label": symmetry_label} if symmetry_label != "C1" else {}),
    )



def _open_persistent_relion_projector_texture(
    relion_projector_half,
    *,
    relion_projector_r_max,
    projection_padding_factor,
    relion_texture_interp=None,
    log_label="Sparse pass-2",
):
    """Upload an eligible host ``PPref`` slab without staging it through JAX.

    This is deliberately a narrow fast path for fine sparse pass-2.  Other
    inputs retain the established transient texture/JAX behavior.
    """

    from relax.helpers.projection import _host_relion_projector_texture_enabled

    if not _host_relion_projector_texture_enabled(
        relion_projector_half, r_max=relion_projector_r_max,
        padding_factor=projection_padding_factor, allow_float32_cast=True,
        enabled=relion_texture_interp,
    ):
        return None

    from relax.cuda.kernels import RelionPersistentHalfTextureF32

    # Match the bucketed float32 consumer cast, before any device upload.
    relion_projector_half = np.asarray(relion_projector_half, dtype=np.complex64)

    logger.info(
        "%s persistent RELION projector texture: shape=%s host=%.2f GiB",
        str(log_label),
        tuple(relion_projector_half.shape),
        relion_projector_half.nbytes / float(1024**3),
    )
    return RelionPersistentHalfTextureF32(
        relion_projector_half,
        padding_factor=int(projection_padding_factor),
        projector_max_r=int(relion_projector_r_max),
        projector_scale=1.0,
    )



def _resident_with_compact_default(resident_impl, compact_impl, open_texture, *args, **kwargs):
    """The resident default: the resident driver where it covers the pass, compact elsewhere.

    DEPRECATED: to be removed once the resident engine covers subset and focused replays, the
    K-class fused and 2K-1 fallbacks, CPU-only execution and its memory refusals; see em_status
    'One engine' TODO.

    The resident configuration checks run before any device work, so a pass they
    refuse (:class:`ResidentConfigurationUnsupported`) runs on the compact engine,
    with its own persistent texture, and the log names the reason. Any other error
    propagates.
    """

    from relax.sparse_pass2.sparse_pass2_policy import ResidentConfigurationUnsupported

    try:
        result = resident_impl(*args, **kwargs)
    except ResidentConfigurationUnsupported as exc:
        logger.info(
            "Device-resident sparse pass 2 (the K=1 default) does not cover this pass; "
            "it runs on the compact engine: %s",
            exc,
        )
        record_pass_engine("global", "compact", resident_refusal_reason(exc))
    else:
        record_pass_engine("global", "resident")
        return result
    compact_kwargs = dict(kwargs)
    texture = open_texture()
    if texture is not None:
        compact_kwargs["relion_projector_half"] = None
        compact_kwargs["relion_projector_texture"] = texture
    return _call_with_persistent_texture_cleanup(texture, compact_impl, *args, **compact_kwargs)


def _call_with_persistent_texture_cleanup(texture, callback, *args, **kwargs):
    """Run ``callback`` and close an optional texture on every exit path."""

    try:
        return callback(*args, **kwargs)
    finally:
        if texture is not None:
            texture.close()

