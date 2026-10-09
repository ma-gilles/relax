"""The records one local-search pass hands the device-resident local engine.

:func:`relax.sparse_pass2.resident_local_pass2.compute_local_search_resident` takes the images and
per-image corrections (:class:`LocalSearchData`), the kernel, window, projector and optics choices
(:class:`LocalSearchKernelPolicy`) and the posterior-support and returned-detail choices
(:class:`LocalSearchSupportPolicy`). ``relax/refinement/half_scoring.py`` builds them once per pass and
``relax/refinement/local_search_iteration.py`` passes them on; they live beside the engine so that it
does not import from the refinement controller. The records are frozen; the arrays they hold are not
copied and are not mutated by the engine.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class LocalSearchData:
    """Images, reference and per-image corrections for one local pass."""

    experiment_dataset: object
    mean: object
    noise_variance: object
    image_corrections: object | None = None
    scale_corrections: object | None = None
    group_ids: object | None = None
    scale_correction_group_count: object | None = None
    scale_correction_data_vs_prior: object | None = None
    image_pre_shifts: object | None = None
    optics_group_ids: object | None = None


@dataclass(frozen=True, kw_only=True)
class LocalSearchKernelPolicy:
    """Numerical kernel, Fourier window, projector and optics choices."""

    disc_type: str
    current_size: int | None
    reconstruction_current_size: int | None
    accumulate_noise: bool = False
    projection_padding_factor: int
    reconstruction_padding_factor: int
    use_float64_scoring: bool = False
    use_float64_projections: bool = False
    square_window: bool
    half_spectrum_scoring: bool
    relion_exact_score_translation: bool = False
    projection_relion_texture_interp: bool | None = False
    projection_relion_acc_double_floorf_quirk: bool = False
    projection_relion_kernel: str = "fine"
    relion_projector_half: object | None
    relion_projector_r_max: int | None
    source_faithful_spectrum_norm: bool
    relion_translation_angle_scale: float
    projection_scale: float
    reconstruction_volume_current_size: int | None
    reconstruction_image_radius: float | None
    # RelionConsistencyOptions.nyquist_column_counting: how the per-image sums count the Hermitian
    # pairs of the full-size Nyquist column.
    nyquist_column_counting: str
    # --strict_highres_exp: the fine pass's weighted-sum image size (None: current_size).
    wsum_current_size: int | None
    # RELION's --firstiter_cc iteration of a search local from its first iteration (--sigma_ang):
    # normalized CC and winner-take-all in the parent probe and the fine pass.
    firstiter_cc: bool


@dataclass(frozen=True, kw_only=True)
class LocalSearchSupportPolicy:
    """Posterior support, accumulation and returned-detail choices."""

    mstep_relion_x_half: bool = False
    disable_adjoint_y: bool
    disable_adjoint_ctf: bool
    adaptive_fraction: float
    max_significants: int | None
    reconstruct_significant_only: bool = True
    return_best_pose_details: bool = False
    normalization_log_evidence: object | None = None
    return_reconstruction_sample_indices: bool = False
    apply_max_significants_to_support: bool = False
    stats_use_reconstruction_probs: bool = False
    score_only: bool = False
    return_profile: bool

    @property
    def applied_max_significants(self) -> int | None:
        """RELION's ``maximum_significants`` cap on this pass's support: ``max_significants`` when the pass
        applies it (``apply_max_significants_to_support``, the pass-1 parent probe), else -1 (no cap)."""

        return self.max_significants if self.apply_max_significants_to_support else -1
