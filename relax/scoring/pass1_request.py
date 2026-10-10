"""Everything one pass 1 is asked to do: the call's inputs, as the caller gave them."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, kw_only=True)
class Pass1Request:
    """The inputs of one coarse significance pass (:func:`relax.scoring.significance._compute_k_class_significance_batched`).

    The pass finds significant samples from one posterior over ``class x rotation x translation``: the poses that carry
    ``adaptive_fraction`` of it, at most ``max_significants``. ``noise_variance`` is one shared spectrum or ``[G, P]`` rows
    of G optics groups; with rows, ``optics_group_ids`` gives each image's group and every image scores with its own
    group's spectrum (:mod:`relax.relion.optics_noise`).

    ``firstiter_cc_support="gaussian"`` weights a normalized-CC pass, and its image power, on the Gaussian support of the
    current size; ``nyquist_column_counting="once"`` drops the redundant members of the full-size Nyquist column from the
    Gaussian weights and from the image power (docs/math/relion_consistency_options.md).
    """

    experiment_dataset: Any = field(kw_only=False)
    noise_variance: Any = field(kw_only=False)
    rotations: Any = field(kw_only=False)
    translations: Any = field(kw_only=False)
    class_log_priors: Any
    adaptive_fraction: float
    max_significants: int
    image_batch_size: int
    rotation_block_size: int
    current_size: int | None
    score_with_masked_images: bool = False
    rotation_log_prior: Any = None
    translation_log_prior: Any = None
    image_corrections: Any = None
    scale_corrections: Any = None
    image_pre_shifts: Any = None
    half_spectrum_scoring: bool = False
    projection_padding_factor: int = 1
    square_window: bool = False
    window_at_box: bool = False
    use_float64_scoring: bool = False
    use_float64_projections: bool | None = None
    relion_projector_half: Any = None
    relion_projector_r_max: int | None = None
    relion_projector_texture_interp: bool | None = None
    score_mode: str = "gaussian"
    collect_significance: bool = True
    return_relion_f32_normalization: bool = False
    return_class_best: bool = False
    return_class_second: bool = False
    debug_iteration: int | None = None
    coarse_healpix_order: int | None = None
    coarse_rotation_ids: Any = None
    translation_phase_source: Any = None
    symmetry_label: str = "C1"
    relion_f32_coarse_tie_ulps: int = 0
    pad_final_image_batch: bool = False
    stable_fourier_window_shapes: bool = False
    relion_translation_angle_scale: float = 1.0
    optics_group_ids: Any = None
    tree_rescore_max_margin: float | None = None
    firstiter_cc_support: str = "relion"
    nyquist_column_counting: str = "relion"
