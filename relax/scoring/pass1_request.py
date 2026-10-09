"""Everything one pass 1 is asked to do: the call's inputs, as the caller gave them."""

import operator
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True, kw_only=True)
class Pass1Request:
    """The inputs of one coarse significance pass (:func:`relax.scoring.significance._compute_k_class_significance_batched`).

    The pass finds significant samples from one posterior over ``class x rotation x translation``: the poses that carry
    ``adaptive_fraction`` of it, at most ``max_significants``. ``noise_variance`` is one shared spectrum or ``[G, P]`` rows
    of G optics groups; with rows, ``optics_group_ids`` gives each image's group and every image scores with its own
    group's spectrum (:mod:`relax.helpers.optics_noise`).

    ``firstiter_cc_support="gaussian"`` weights a normalized-CC pass, and its image power, on the Gaussian support of the
    current size; ``nyquist_column_counting="once"`` drops the redundant members of the full-size Nyquist column from the
    Gaussian weights and from the image power (docs/math/relion_consistency_options.md).

    Building the request refuses what its own fields contradict, with a ``ValueError`` (a non-integer ``image_batch_size``
    raises the ``TypeError`` of ``operator.index``): a runner-up without the best, RELION's float32 normalization
    without a collecting Gaussian float32 pass, an unknown ``score_mode``, a non-positive ``image_batch_size``, a
    ``translation_phase_source`` or ``coarse_rotation_ids`` of another shape than the translations or rotations, a negative
    ``coarse_healpix_order``, a projector without its ``relion_projector_r_max`` and a negative or non-finite
    ``tree_rescore_max_margin``. What also needs the planner's resources (the projector's class count, the scoring
    window, the CUDA backend) is refused by :func:`relax.scoring.pass1_plan.plan_pass1`.
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

    def __post_init__(self):
        if self.return_class_second and not self.return_class_best:
            raise ValueError("return_class_second requires return_class_best")
        if self.return_relion_f32_normalization and (
            not self.collect_significance or self.score_mode != "gaussian" or self.use_float64_scoring
        ):
            raise ValueError("RELION float32 normalization requires Gaussian float32 significance")
        if self.score_mode not in {"gaussian", "normalized_cc"}:
            raise ValueError(f"score_mode must be 'gaussian' or 'normalized_cc', got {self.score_mode!r}")
        if self.translation_phase_source is not None:
            source_shape = np.asarray(self.translation_phase_source).shape
            if source_shape != self.translations.shape:
                raise ValueError(
                    "translation_phase_source must match translations: "
                    f"{source_shape} != {self.translations.shape}",
                )
        if operator.index(self.image_batch_size) <= 0:
            raise ValueError("image_batch_size must be positive")
        if self.coarse_rotation_ids is not None:
            ids_shape = np.asarray(self.coarse_rotation_ids, dtype=np.int64).reshape(-1).shape
            if ids_shape != (int(self.rotations.shape[0]),):
                raise ValueError(
                    f"coarse_rotation_ids must have shape ({int(self.rotations.shape[0])},), got {ids_shape}",
                )
        if self.coarse_healpix_order is not None and int(self.coarse_healpix_order) < 0:
            raise ValueError(f"coarse_healpix_order must be non-negative, got {self.coarse_healpix_order}")
        if self.relion_projector_half is not None and self.relion_projector_r_max is None:
            raise ValueError("relion_projector_r_max is required when relion_projector_half is provided")
        if self.tree_rescore_max_margin is not None and not (
            np.isfinite(self.tree_rescore_max_margin) and self.tree_rescore_max_margin >= 0.0
        ):
            raise ValueError(
                f"tree_rescore_max_margin must be a finite non-negative float, got {self.tree_rescore_max_margin!r}"
            )
