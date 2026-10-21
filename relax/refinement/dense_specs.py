"""The records through which the controller calls the dense scoring engine for one half.

``DenseSamplingSpec`` (the dense sampling grids, oversampling and reconstruction sizes, and the adaptive trial
grids built from them), ``DensePriorSpec`` (direction, translation and class priors), ``DenseBatchPolicy``
(batch sizes and route overrides), ``DenseVariantPolicy`` (the route selections) and ``DenseExecutionPolicy``
(interpolation, precision, diagnostics and engine tuning). All are frozen. The definitions moved verbatim from
``relax.refinement.dense_half``, whose routes read them.
"""

from dataclasses import dataclass

from relax.classification.probes import KClassProbe
from relax.numerics.precision import DensePrecisionPolicy
from relax.runtime.engine_tuning import EngineTuning
from relax.runtime.scoring_overrides import ScoringOverrides
from relax.sampling.oversampling import AdaptivePass2Grids, prepare_adaptive_pass2_grids


@dataclass(frozen=True, kw_only=True)
class DenseSamplingSpec:
    """Resolved dense sampling grids, oversampling and reconstruction sizes."""

    effective_rotations: object
    current_translations: object
    base_translations: object
    current_healpix_order: int
    oversampling_order: int
    translation_step: float
    random_perturbation: float
    image_window_size: int | None
    coarse_engine: str
    model_support_size: int | None = None
    # --strict_highres_exp: the weighted sums' image size, above image_window_size (None: image_window_size).
    wsum_current_size: int | None = None
    coarse_angular_step_deg: float | None = None
    coarse_rotation_ids: object | None = None
    coarse_scoring_rotations: object | None = None
    symmetry: str
    # RELION's device-built pass-1 source of ``effective_rotations`` / ``coarse_scoring_rotations``
    # (None: host-built rows); images on another grid or magnified rebuild their rows from it.
    effective_device_source: object | None = None
    coarse_scoring_device_source: object | None = None

    def rotation_source(self, adaptive_oversampling: int, symmetry: str, *, coarse_scoring: bool = False) -> dict:
        """Provenance of :meth:`pass2_grids` rows for :func:`project_pass2_rotations`; ``coarse_scoring``
        when the scored coarse rows are ``coarse_scoring_rotations``."""
        return {
            "coarse_healpix_order": self.current_healpix_order,
            "adaptive_oversampling": adaptive_oversampling,
            "random_perturbation": self.random_perturbation,
            "coarse_rotation_ids": self.coarse_rotation_ids,
            "symmetry": symmetry,
            "coarse_device_source": (
                self.coarse_scoring_device_source if coarse_scoring else self.effective_device_source
            ),
            "grid_device_source": self.effective_device_source,
        }

    def pass2_grids(self, *, adaptive_oversampling: int, symmetry: str) -> AdaptivePass2Grids:
        """The adaptive pass-1 and pass-2 trial grids of these sampling settings."""
        return prepare_adaptive_pass2_grids(
            self.effective_rotations,
            self.current_translations,
            self.base_translations,
            healpix_order=self.current_healpix_order,
            adaptive_oversampling=adaptive_oversampling,
            translation_step=self.translation_step,
            random_perturbation=self.random_perturbation,
            coarse_rotation_ids=self.coarse_rotation_ids,
            symmetry=symmetry,
        )


@dataclass(frozen=True, kw_only=True)
class DensePriorSpec:
    """Direction, translation and class priors for one dense half."""

    rotation_log_prior_k: object
    class_rotation_log_prior_k: object
    translation_log_prior: object
    translation_search_base: object
    trans_prior_center_for_engine: object
    class_log_priors: object


@dataclass(frozen=True, kw_only=True)
class DenseBatchPolicy:
    """Existing batch planners and route-specific overrides."""

    safe_batch_sizes: object
    max_significants: int | None
    significance_safe_batch_sizes: object | None
    k_class_image_batch_size_override: int | None
    k_class_rotation_block_size_override: int | None
    significance_image_batch_size_override: int | None
    significance_rotation_block_size_override: int | None
    # Multi-shape halves: each shape class's own batch sizes (expectation_batches.ShapeClassBatchSizes).
    class_batch_overrides: tuple | None


@dataclass(frozen=True, kw_only=True)
class DenseVariantPolicy:
    """Explicit dense, adaptive and first-iteration route selections.

    ``coarse_window_size`` and ``fine_window_size`` are the adaptive passes' image windows on every
    iteration (None: the engine's current size; the fine one is None on a non-adaptive pass).
    """

    score_mode: str
    winner_take_all: bool
    k_class_enabled: bool
    firstiter_cc: bool
    coarse_window_size: int | None
    fine_window_size: int | None
    # RELION --skip_align: classify at each particle's stored pose (relax.refinement.given_poses).
    skip_align: bool


@dataclass(frozen=True, kw_only=True)
class DenseExecutionPolicy:
    """Interpolation, ablation and dense diagnostic settings."""

    disc_type: str
    return_best_pose_details: bool
    bpref_device_signature_active: bool
    debug_iteration: int | None
    diagnostic_float64_pass2: bool
    preserve_bpref_particle_order: bool
    source_faithful_spectrum_norm: bool
    relion_translation_angle_scale: float
    # The K=1 --firstiter_cc coarse-tree top-2 rescore margin (RelionParityOptions); None is off.
    firstiter_cc_tree_rescore_max_margin: float | None
    # RelionConsistencyOptions: the support of the first-iteration normalized CC, and how the
    # per-image sums count the Hermitian pairs of the full-size Nyquist column.
    firstiter_cc_support: str
    nyquist_column_counting: str
    # ScoringVariants.relion_x_half_mstep for this half's class count: RELION's x-half M-step accumulators.
    relion_x_half_mstep: bool
    # The run's dense precision (RefinementOptions.precision): engine precision and the pose dtype.
    precision: DensePrecisionPolicy
    # The K-class E-step's diagnostics (relax.classification.probes.KClassProbe): they observe, never steer.
    k_class_probe: KClassProbe
    # The observer asks for an adaptive K=1 half's fine pose grid as RELION Euler angles (its expectation probe).
    keep_pose_rotation_eulers: bool
    # How the engines execute (RefinementEnvironment.engine, read by the command).
    engine_tuning: EngineTuning
    # The diagnostic overrides that change values (RefinementEnvironment.scoring, read by the command).
    scoring_overrides: ScoringOverrides
