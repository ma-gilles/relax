"""The resident pass 2's resolved state, handed to a full-grid engine instead of building candidate tables.

:func:`relax.fine_pass.resident_pass2._resident_pass2` fills this record just before it would build the sparse
candidate tables; a full-grid engine (``relax.dense.gemm_coarse_engine.run_dense_gemm_full_grid``) scores the
complete expanded grid from it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DenseGemmPreparedState:
    """State already resolved by the resident driver before sparse table build."""

    dataset: object
    class_volumes: tuple
    class_projector_halves: tuple
    class_rotation_priors: tuple
    fine_rotations: object
    fine_mstep_rotations: object
    fine_rotation_parent: object
    fine_translations_source: object
    fine_translations: object
    fine_translation_parent: object
    fine_translation_prior: object
    translation_prior_centers: object
    noise_variance_half: object
    n_coarse_rot: int
    nside_level: int
    oversampling_order: int
    random_perturbation: float
    relion_parent_execution_order: bool
    current_size: int
    mstep_current_size: int
    n_half: int
    window_spec_kwargs: dict
    disc_type: str
    projector_r_max: int | None
    projection_padding_factor: int
    reconstruction_padding_factor: int
    reconstruction_volume_current_size: int | None
    reconstruction_image_radius: float | None
    # How the engines execute (EngineTuning, from the resident driver's caller).
    engine_tuning: object
    image_corrections: object
    scale_corrections: object
    image_pre_shifts: object
    group_ids: object
    scale_correction_group_count: int | None
    scale_correction_data_vs_prior: object
    optics_group_ids: object
    reconstruction_group_ids: object
    reconstruction_group_count: int | None
    score_with_masked_images: bool
    half_spectrum_scoring: bool
    square_window: bool
    accumulate_noise: bool
    use_exact_relion_gaussian: bool
    source_faithful_spectrum_norm: bool
    relion_wavg_atomic_direct_noise: bool
    relion_wavg_atomic_scale_aa: bool
    accumulate_scale: bool
    relion_exact_bpref_operands: bool
    relion_native_fine_units: bool
    relion_firstiter_score_mode: str
    mstep_subtract_ctf_projection: bool
    include_unweighted_norm_high_shell: bool
    relion_translation_angle_scale: float
    precision_policy: object
    symmetry_label: str
