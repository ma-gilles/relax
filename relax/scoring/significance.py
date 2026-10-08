"""Coarse significance pruning for adaptive class and pose searches.

The shared scorer serves K=1, multiclass refinement and initial-model searches.
It selects per-image significant samples from one class/rotation/translation
posterior, with optional diagnostic capture of the coarse scoring boundary.
"""

import logging
import operator
import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from recovar.utils.nvtx_shim import nvtx

from relax.diagnostics.coarse_gaussian_diagnostics import (
    _significance_debug_dump_matches,
)
from relax.diagnostics.coarse_score_diagnostics import (
    _build_coarse_significance_support_audit,
)
from relax.helpers.batch_fetch import original_image_indices
from relax.helpers.env_flags import (
    parse_env_int_set,
    parse_env_strict_flag,
)
from relax.helpers.projection_cache import build_projection_cache
from relax.relion.relion_coarse_operands import (
    _infer_relion_coarse_healpix_order,
    _k1_relion_f32_coarse_support_enabled,
)
from relax.scoring.coarse_gaussian_gemm import (
    _coarse_gaussian_gemm_projection_cache_stats,
)
from relax.scoring.coarse_projector import CoarseProjector, CompactRows
from relax.scoring.coarse_publication import coarse_square_layout_metadata, coarse_support_posterior
from relax.scoring.gaussian_plan import plan_coarse_gaussian
from relax.scoring.pass1_assembly import log_batch_timing, significant_samples_after_loop
from relax.scoring.pass1_batch import BatchInputPlan, prepare_batch_inputs
from relax.scoring.pass1_operands import CcOperandPlan, GaussianOperandPlan
from relax.scoring.pass1_program import (
    _class_block_state,
    _coarse_pass1_block,
    _coarse_pass1_blocks,
    _merge_class_block_state,
    _pass1_batch_constants,
    _pass1_initial_state,
)
from relax.scoring.pass1_publish import publish_batch
from relax.scoring.pass1_results import BatchOutputs, OutputPlan, Pass1Outputs, PassShape, ScoreDumpContext
from relax.scoring.scoring import (
    _coarse_gemm_float64_requested,
)
from relax.scoring.tree_rescore import (
    TreeRescoreGeometry,
    TreeRescoreState,
    plan_tree_rescore,
    require_tree_rescore_call,
    rescore_ambiguous_images,
)

_GLOBAL_PASS1_RELION_PROJECTOR_TEXTURE_ENV = "RELAX_RELION_GLOBAL_PASS1_PROJECTOR_TEXTURE_INTERP"
_COARSE_SIGNIFICANCE_SUPPORT_AUDIT_ENV = (
    "RECOVAR_COARSE_SIGNIFICANCE_SUPPORT_AUDIT"
)
_COARSE_SIGNIFICANCE_SUPPORT_AUDIT_IDS_ENV = (
    "RECOVAR_COARSE_SIGNIFICANCE_SUPPORT_AUDIT_IDS"
)
_COARSE_PAD_FINAL_IMAGE_BATCH_ENV = (
    "RELAX_COARSE_PAD_FINAL_IMAGE_BATCH"
)
NVTX_DOMAIN_EM = "recovar_em"
logger = logging.getLogger(__name__)



def _coarse_rotated_radius_enabled(*, default: bool = False) -> bool:
    """Select canonical clipping when the active projector supports it."""
    token = os.environ.get("RELAX_K1_COARSE_ROTATED_RADIUS", "1" if default else "0")
    if token not in {"0", "1"}:
        raise ValueError("RELAX_K1_COARSE_ROTATED_RADIUS must be 0 or 1")
    return token == "1"


def _require_exact_pass1_operands(
    *,
    use_relion_projector: bool,
    coarse_texture_interp: bool,
    half_spectrum_scoring: bool,
    use_float64_scoring: bool,
    use_float64_projections: bool | None,
) -> None:
    """Refuse a coarse pass that RELION's exact coarse operands cannot score.

    Pass 1 projects with the supplied RELION projector by texture interpolation,
    scores the half spectrum in float32 and runs RELION's CUDA preprocessing and
    translation, as RELION's GPU coarse kernels do. The generic dense scorer that
    served the other configurations was removed on 2026-10-02.
    """

    missing = []
    if not use_relion_projector:
        missing.append("the supplied RELION projector (relion_projector_half)")
    if not coarse_texture_interp:
        missing.append("texture interpolation of that projector")
    if not half_spectrum_scoring:
        missing.append("half-spectrum scoring")
    if use_float64_scoring or use_float64_projections:
        missing.append("float32 scoring and projections")
    if missing:
        raise ValueError("pass 1 scores RELION's exact coarse operands and needs " + ", ".join(missing))


def _coarse_max_posterior_for_host(batch_weights, actual_batch_size):
    """Publish active row maxima without specializing on fringe batch sizes.

    Rows are independent. Reducing the physical table before slicing the
    compact host vector keeps padded rows out of the published statistics.
    """
    return np.asarray(jnp.max(batch_weights, axis=1), dtype=np.float32)[
        :actual_batch_size
    ]


def _coarse_significance_support_audit_enabled(
    *,
    default: bool = False,
) -> bool:
    """Resolve exact, diagnostic-only coarse-support hashing."""

    return parse_env_strict_flag(_COARSE_SIGNIFICANCE_SUPPORT_AUDIT_ENV, default=default)


def _coarse_pad_final_image_batch_enabled(*, default: bool = True) -> bool:
    """Whether every coarse image batch is padded to the requested batch size.

    A half set is rarely an exact multiple of ``image_batch_size``, so its last
    batch has its own image extent and every program in the coarse pass,
    significance and image preprocessing is traced and compiled a second time
    for it. Padding repeats image row zero up to ``image_batch_size``; the
    repeated rows are discarded from every science output, so the retained
    values are bit-for-bit those of the unpadded batch (the per-image
    computations are independent, and the batch-level reductions are maxima
    over rows that row zero already contributes).

    On by default since the K=1 resident flip (qualified with the resident flag
    set on the fast tier and the four flip datasets); ``=0`` turns it off. VDAM
    requests the same padding explicitly through ``pad_final_image_batch``.
    """

    return parse_env_strict_flag(
        _COARSE_PAD_FINAL_IMAGE_BATCH_ENV,
        default=default,
    )


def _coarse_significance_support_audit_ids_enabled() -> bool:
    """Whether a support audit also retains its exact selected IDs."""
    return parse_env_strict_flag(_COARSE_SIGNIFICANCE_SUPPORT_AUDIT_IDS_ENV)




def _global_pass1_relion_projector_texture_enabled() -> bool:
    """Whether dense/global pass-1 significance uses texture arithmetic.

    Coarse significance defaults to RELION's texture projector.  Set the
    environment flag to false to force the manual/JAX diagnostic fallback.
    """
    return parse_env_strict_flag(
        _GLOBAL_PASS1_RELION_PROJECTOR_TEXTURE_ENV,
        default=True,
    )


@nvtx.annotate("kclass.adaptive.pass1_significance", color="orange", domain=NVTX_DOMAIN_EM)
def _prepare_coarse_relion_projector(
    projector, *, use_float64_scoring, use_float64_projections,
):
    """Upload at the explicit consumer precision, retaining the source slab.

    None preserves legacy direct callers. Double scoring/projection diagnostics
    retain complex128 projection; production K1/K4 explicitly request float32.
    """
    dtype = None
    if use_float64_projections is not None:
        dtype = jnp.complex128 if (use_float64_scoring or use_float64_projections) else jnp.complex64
    return jnp.asarray(projector, dtype=dtype)


def _class_stacked_coarse_relion_projector(
    projector, n_classes, *, use_float64_scoring, use_float64_projections,
):
    """Upload a coarse PPref as ``(n_classes, z, y, x_half)``.

    K=1 callers may pass one ``(z, y, x_half)`` slab, such as the host-compacted
    firstiter-CC projector; its class axis is added on the host before upload.
    """
    if np.ndim(projector) == 3 and int(n_classes) == 1:
        projector = projector[None]
    projector = _prepare_coarse_relion_projector(
        projector, use_float64_scoring=use_float64_scoring,
        use_float64_projections=use_float64_projections,
    )
    if projector.ndim != 4 or int(projector.shape[0]) != int(n_classes):
        raise ValueError(
            "relion_projector_half must have shape "
            f"({n_classes}, z, y, x_half), or (z, y, x_half) for K=1; got {projector.shape}",
        )
    return projector


def _compute_k_class_significance_batched(
    experiment_dataset,
    noise_variance,
    rotations,
    translations,
    *,
    class_log_priors,
    adaptive_fraction,
    max_significants,
    image_batch_size,
    rotation_block_size,
    current_size,
    score_with_masked_images=False,
    rotation_log_prior=None,
    translation_log_prior=None,
    image_corrections=None,
    scale_corrections=None,
    image_pre_shifts=None,
    half_spectrum_scoring=False,
    projection_padding_factor=1,
    square_window=False,
    window_at_box=False,
    use_float64_scoring=False,
    use_float64_projections: bool | None = None,
    relion_projector_half=None,
    relion_projector_r_max=None,
    relion_projector_texture_interp: bool | None = None,
    score_mode: str = "gaussian",
    collect_significance: bool = True,
    return_relion_f32_normalization: bool = False,
    return_class_best: bool = False,
    return_class_second: bool = False,
    debug_iteration: int | None = None,
    coarse_healpix_order: int | None = None,
    coarse_rotation_ids=None,
    translation_phase_source=None,
    symmetry_label: str = "C1",
    relion_f32_coarse_tie_ulps: int = 0,
    pad_final_image_batch: bool = False,
    stable_fourier_window_shapes: bool = False,
    relion_translation_angle_scale: float = 1.0,
    optics_group_ids=None,
    tree_rescore_max_margin: float | None = None,
    firstiter_cc_support: str = "relion",
    nyquist_column_counting: str = "relion",
):
    """Find significant samples from one posterior over ``class x rotation x translation``.

    ``noise_variance`` is one shared spectrum or ``[G, P]`` rows of G optics groups;
    with rows, ``optics_group_ids`` gives each image's group and every image scores
    with its own group's spectrum (:mod:`relax.helpers.optics_noise`).

    ``firstiter_cc_support="gaussian"`` weights a normalized-CC pass, and its image power, on
    the Gaussian support of the current size; ``nyquist_column_counting="once"`` drops the
    redundant members of the full-size Nyquist column from the Gaussian weights and from the
    image power (docs/math/relion_consistency_options.md).
    """

    if return_class_second and not return_class_best:
        raise ValueError("return_class_second requires return_class_best")
    if return_relion_f32_normalization and (
        not collect_significance or score_mode != "gaussian" or use_float64_scoring
    ):
        raise ValueError("RELION float32 normalization requires Gaussian float32 significance")

    from recovar.reconstruction import noise as noise_utils

    from relax.helpers.fourier_window import make_fourier_window_spec
    from relax.helpers.half_spectrum import make_scoring_half_image_weights
    from relax.helpers.oversampling import find_significant_rotations as _find_sig

    if score_mode not in {"gaussian", "normalized_cc"}:
        raise ValueError(f"score_mode must be 'gaussian' or 'normalized_cc', got {score_mode!r}")
    # VDAM asks for the padded tail batch explicitly; the global coarse pass
    # opts in through the environment while the bitwise equality is qualified.
    pad_final_image_batch = bool(pad_final_image_batch) or _coarse_pad_final_image_batch_enabled()
    # RELION's pdf_orientation/pdf_offset priors and score/evidence/Pmax
    # outputs are RFLOAT (double), never narrowed -- derive from the
    # caller's own use_float64_scoring instead of hardcoding float32.
    score_real_dtype = np.float64 if use_float64_scoring else np.float32
    # The class count is the length of the class priors; relion_projector_half carries the same count.
    class_log_priors_np = np.asarray(class_log_priors, dtype=np.float64).reshape(-1)
    n_classes = int(class_log_priors_np.shape[0])

    translations_source = np.asarray(
        translations if translation_phase_source is None else translation_phase_source,
    )
    if translations_source.shape != translations.shape:
        raise ValueError(
            "translation_phase_source must match translations: "
            f"{translations_source.shape} != {translations.shape}",
        )
    n_rot = int(rotations.shape[0])
    n_trans = int(translations.shape[0])
    n_images = int(experiment_dataset.n_units)
    input_image_batch_size = operator.index(image_batch_size)
    if input_image_batch_size <= 0:
        raise ValueError("image_batch_size must be positive")
    image_batch_size = input_image_batch_size
    image_shape = experiment_dataset.image_shape
    volume_shape = experiment_dataset.volume_shape
    n_half = int(image_shape[0] * (image_shape[1] // 2 + 1))
    if coarse_rotation_ids is not None:
        coarse_rotation_ids = np.asarray(coarse_rotation_ids, dtype=np.int64).reshape(-1)
        if coarse_rotation_ids.shape != (n_rot,):
            raise ValueError(
                f"coarse_rotation_ids must have shape ({n_rot},), got {coarse_rotation_ids.shape}",
            )
    if coarse_healpix_order is None:
        coarse_healpix_order = _infer_relion_coarse_healpix_order(n_rot, **({"symmetry_label": symmetry_label} if symmetry_label != "C1" else {}))
    elif int(coarse_healpix_order) < 0:
        raise ValueError(f"coarse_healpix_order must be non-negative, got {coarse_healpix_order}")

    use_relion_projector = relion_projector_half is not None
    if use_relion_projector and relion_projector_r_max is None:
        raise ValueError("relion_projector_r_max is required when relion_projector_half is provided")
    if use_relion_projector:
        relion_projector_half = _class_stacked_coarse_relion_projector(
            relion_projector_half, n_classes, use_float64_scoring=use_float64_scoring,
            use_float64_projections=use_float64_projections,
        )

    cc_gaussian_support = score_mode == "normalized_cc" and firstiter_cc_support == "gaussian"
    half_weights = make_scoring_half_image_weights(
        image_shape,
        relion_half_sum=half_spectrum_scoring,
        exclude_relion_redundant_x0=score_mode != "normalized_cc",
        nyquist_column_counting=nyquist_column_counting,
        firstiter_cc_support_size=(
            (image_shape[0] if current_size is None else current_size) if cc_gaussian_support else None
        ),
    )
    window_spec_kwargs = {}
    if score_mode == "normalized_cc":
        window_spec_kwargs = {
            "score_square": True,
            "score_include_dc": True,
        }
    window_spec = make_fourier_window_spec(
        image_shape,
        current_size,
        n_half,
        square=square_window,
        include_recon_window=False,
        # RELION's radial window at the box too (Gaussian scoring; the normalized-CC score
        # keeps its rectangular window).
        window_at_box=bool(window_at_box) and score_mode != "normalized_cc",
        **window_spec_kwargs,
    )
    use_window = window_spec.use_window
    score_size = int(image_shape[0]) if current_size is None else int(current_size)
    # RELION's coarse kernel projects and shifts the rows beyond maxR at ``i - window`` inside
    # the model sphere only for a window between 2 r_max and about 2 s r_max (an optics group on
    # a coarser grid: its rotations carry 1 / s); the exact coarse operands then shift them there.
    coarse_kernel_window = None
    if use_relion_projector and score_size // 2 > int(relion_projector_r_max):
        from relax.helpers.optics_scale import coarse_rows_wrap_inside

        rotation_scale = 1.0 / float(np.linalg.norm(np.asarray(rotations, dtype=np.float64).reshape(-1, 3, 3)[0, 0]))
        if coarse_rows_wrap_inside(score_size, int(relion_projector_r_max), rotation_scale):
            coarse_kernel_window = score_size
    window_indices = window_spec.score_indices
    coarse_texture_interp = (
        _global_pass1_relion_projector_texture_enabled()
        if relion_projector_texture_interp is None
        else bool(relion_projector_texture_interp)
    )
    if tree_rescore_max_margin is not None and not (
        np.isfinite(tree_rescore_max_margin) and tree_rescore_max_margin >= 0.0
    ):
        raise ValueError(f"tree_rescore_max_margin must be a finite non-negative float, got {tree_rescore_max_margin!r}")
    # The margin is a run option, while only iteration 1 uses normalized CC.
    # Later Gaussian iterations must remain unaffected.
    tree_rescore_enabled = (
        tree_rescore_max_margin is not None and score_mode == "normalized_cc"
    )
    # Pass 1 scores RELION's exact coarse operands only: the Gaussian passes with the coarse GEMMs
    # (_relion_coarse_gaussian_gemm_scores_jit), the --firstiter_cc passes with RELION's coarse CC
    # (_relion_coarse_normalized_cc_gemm_scores_jit). The generic dense scorer was removed on
    # 2026-10-02; like pass 2, pass 1 needs a CUDA GPU and RELION's CUDA image preprocessing.
    _require_exact_pass1_operands(
        use_relion_projector=use_relion_projector,
        coarse_texture_interp=coarse_texture_interp,
        half_spectrum_scoring=half_spectrum_scoring,
        use_float64_scoring=use_float64_scoring,
        use_float64_projections=use_float64_projections,
    )
    exact_gaussian = score_mode == "gaussian"
    pass_shape = PassShape(
        n_classes=n_classes,
        n_rot=n_rot,
        n_trans=n_trans,
        n_half=n_half,
        image_shape=image_shape,
        score_size=score_size,
    )
    gaussian_plan = None
    if exact_gaussian:
        gaussian_plan = plan_coarse_gaussian(
            pass_shape,
            window_spec.score_indices_np,
            relion_projector_half,
            rotation_block_size=rotation_block_size,
            image_batch_size=image_batch_size,
            stable_fourier_window_shapes=stable_fourier_window_shapes,
        )
        rotation_block_size = gaussian_plan.rotation_block_size
    relion_f32_coarse_support_enabled = exact_gaussian and _k1_relion_f32_coarse_support_enabled(default=True)
    # --firstiter_cc on RELION's exact coarse operands: the tree rescore's per-image
    # FFT, RFLOAT CTF and corr_img operands, translated with RELION's sincosf for
    # every translation and scored by the coarse GEMMs
    # (_relion_coarse_normalized_cc_gemm_scores_jit).
    exact_cc_enabled = score_mode == "normalized_cc"
    exact_cc_score_indices = None
    exact_cc_translation_angles = None
    if exact_cc_enabled:
        from relax.sparse_pass2.sparse_pass2_bucket_io import _relion_translation_angles_f32

        exact_cc_score_indices = jnp.asarray(
            np.arange(n_half) if window_spec.score_indices_np is None else window_spec.score_indices_np,
            dtype=jnp.int32,
        )
        exact_cc_translation_angles = jnp.asarray(
            _relion_translation_angles_f32(
                translations_source,
                image_shape,
                angle_scale=relion_translation_angle_scale,
            ),
            dtype=jnp.float32,
        )
        logger.info(
            "RELION normalized-CC coarse pass on the exact operands: classes=%d current_size=%d "
            "score_pixels=%d translations=%d",
            n_classes,
            score_size,
            int(exact_cc_score_indices.shape[0]),
            n_trans,
        )
    track_class_second = return_class_second or tree_rescore_enabled
    score_half_weights = window_spec.score_values(half_weights) if use_window else half_weights
    tree_rescore_plan = None
    if tree_rescore_enabled:
        require_tree_rescore_call(
            n_classes=n_classes,
            return_class_best=return_class_best,
            use_relion_projector=use_relion_projector,
            coarse_texture_interp=coarse_texture_interp,
            half_spectrum_scoring=half_spectrum_scoring,
        )
        tree_rescore_plan = plan_tree_rescore(
            max_margin=tree_rescore_max_margin,
            relion_projector_half=relion_projector_half,
            image_shape=image_shape,
            n_half=n_half,
            score_indices_np=window_spec.score_indices_np,
            translations_source=translations_source,
            relion_translation_angle_scale=relion_translation_angle_scale,
            geometry=TreeRescoreGeometry(
                half_weights=score_half_weights,
                rotations=rotations,
                n_trans=n_trans,
                score_size=score_size,
                padding_factor=projection_padding_factor,
                projector_max_r=relion_projector_r_max,
                coarse_healpix_order=coarse_healpix_order,
                coarse_rotation_ids=coarse_rotation_ids,
                symmetry_label=symmetry_label,
            ),
        )

    if exact_cc_enabled:
        operand_plan = CcOperandPlan(
            experiment_dataset=experiment_dataset,
            image_shape=image_shape,
            image_pre_shifts=image_pre_shifts,
            window_indices=window_indices if use_window else None,
            score_indices=exact_cc_score_indices,
            score_half_weights=score_half_weights,
            support_power_weights=score_half_weights if cc_gaussian_support else None,
            translation_angles=exact_cc_translation_angles,
            n_trans=n_trans,
            score_with_masked_images=score_with_masked_images,
            scale_corrections_enabled=scale_corrections is not None,
        )
    else:
        operand_plan = GaussianOperandPlan(
            experiment_dataset=experiment_dataset,
            gaussian_plan=gaussian_plan,
            image_shape=image_shape,
            half_weights=half_weights,
            translations_source=translations_source,
            relion_translation_angle_scale=relion_translation_angle_scale,
            score_with_masked_images=score_with_masked_images,
            nyquist_column_counting=nyquist_column_counting,
            scale_corrections_enabled=scale_corrections is not None,
            use_float64_scoring=use_float64_scoring,
            stable_fourier_window_shapes=stable_fourier_window_shapes,
            score_size=score_size,
            current_size=current_size,
            coarse_kernel_window=coarse_kernel_window,
            coarse_kernel_r_max=None if coarse_kernel_window is None else int(relion_projector_r_max),
        )

    n_blocks = (n_rot + rotation_block_size - 1) // rotation_block_size
    n_rot_padded = n_blocks * rotation_block_size
    if n_rot_padded > n_rot:
        pad_size = n_rot_padded - n_rot
        rotations_padded = np.concatenate(
            [
                rotations,
                np.tile(np.eye(3, dtype=np.asarray(rotations).dtype), (pad_size, 1, 1)),
            ],
            axis=0,
        )
    else:
        rotations_padded = rotations

    rotation_log_prior_padded = None
    if rotation_log_prior is not None:
        prior = np.asarray(rotation_log_prior, dtype=score_real_dtype)
        if prior.ndim == 1:
            if prior.shape != (n_rot,):
                raise ValueError(f"rotation_log_prior must have shape ({n_rot},), got {prior.shape}")
            prior = np.broadcast_to(prior[None, :], (n_classes, n_rot)).copy()
        elif prior.shape != (n_classes, n_rot):
            raise ValueError(
                f"rotation_log_prior must have shape ({n_rot},) or ({n_classes}, {n_rot}), got {prior.shape}",
            )
        if n_rot_padded > n_rot:
            rotation_log_prior_padded = np.pad(
                prior,
                ((0, 0), (0, n_rot_padded - n_rot)),
                mode="constant",
            )
        else:
            rotation_log_prior_padded = prior

    if translation_log_prior is not None:
        translation_log_prior = np.asarray(translation_log_prior, dtype=score_real_dtype)
        if translation_log_prior.ndim == 1:
            if translation_log_prior.shape != (n_trans,):
                raise ValueError(
                    f"translation_log_prior must have shape ({n_trans},), got {translation_log_prior.shape}"
                )
        elif translation_log_prior.ndim == 2:
            if translation_log_prior.shape != (n_images, n_trans):
                raise ValueError(
                    "translation_log_prior must have shape "
                    f"({n_images}, {n_trans}) when image-specific, got {translation_log_prior.shape}",
                )
        else:
            raise ValueError(f"translation_log_prior must be 1D or 2D, got {translation_log_prior.ndim} dimensions")

    noise_variance_half = noise_utils.to_batched_half_pixel_noise(noise_variance, image_shape).squeeze()
    if noise_variance_half.ndim == 2 and optics_group_ids is None:
        raise ValueError("a per-optics-group noise table needs optics_group_ids")
    # Each batch gathers its images' group spectra on the host: an eager device gather would
    # compile a program for every batch size (1339 compiles in a several-shape VDAM run).
    noise_table_host = np.asarray(noise_variance_half) if noise_variance_half.ndim == 2 else None
    image_groups_host = None if noise_table_host is None else np.asarray(optics_group_ids, dtype=np.int32)

    # The texture projector naturally produces a centered current-size crop.
    # Ask it only for the rows consumed by the scorer instead of scattering the
    # crop into a full image and immediately gathering the same rows again.
    # This is an exact index remapping and avoids a large transient scatter for
    # global rotation blocks.
    projector_compact_rows = None
    if use_relion_projector and coarse_texture_interp:
        if exact_gaussian:
            projector_compact_rows = CompactRows(gaussian_plan.score_indices_np, gaussian_plan.projector_output_size)
        elif use_window:
            projector_compact_rows = CompactRows(window_spec.score_indices_np, score_size)
    projector_returns_compact = projector_compact_rows is not None

    coarse_rotated_radius = _coarse_rotated_radius_enabled(
        default=bool(use_relion_projector and coarse_texture_interp and projector_returns_compact),
    )
    if coarse_rotated_radius and not (
        use_relion_projector and coarse_texture_interp and projector_returns_compact
    ):
        raise ValueError("rotated coarse radius requires the compact RELION texture projector")

    coarse_projector = CoarseProjector(
        relion_projector_half=relion_projector_half,
        relion_projector_r_max=relion_projector_r_max,
        image_shape=image_shape,
        projection_padding_factor=projection_padding_factor,
        current_size=current_size,
        score_size=score_size,
        stable_fourier_window_shapes=stable_fourier_window_shapes,
        compact=projector_compact_rows,
        rotated_radius=coarse_rotated_radius,
    )

    coarse_gaussian_gemm_projection_cache = None
    if exact_gaussian and gaussian_plan.projection_cache_plan is not None:

        coarse_gaussian_gemm_projection_cache = (
            build_projection_cache(
                gaussian_plan.projection_cache_plan,
                partial(coarse_projector.cache_block, rotations),
            )
        )
        logger.warning(
            "Coarse GEMM C64 projection cache built: "
            "shape=%s chunks=%d conservative_peak_bytes=%d budget_bytes=%d",
            gaussian_plan.projection_cache_plan.cache_shape,
            gaussian_plan.projection_cache_plan.chunk_count_per_table,
            gaussian_plan.projection_cache_plan.predicted_peak_bytes,
            gaussian_plan.projection_cache_plan.budget_bytes,
        )

    # The class and rotation prior terms of the pass-1 program, built at its first batch.
    pass1_prior_terms = None
    exact_rotation_prior = None

    output_plan = OutputPlan(
        n_classes=n_classes,
        n_rot=n_rot,
        n_trans=n_trans,
        n_images=n_images,
        score_real_dtype=score_real_dtype,
        collect_significance=collect_significance,
        relion_f32_coarse_support_enabled=relion_f32_coarse_support_enabled,
        return_relion_f32_normalization=return_relion_f32_normalization,
        return_class_best=return_class_best,
        return_class_second=return_class_second,
    )
    outputs = Pass1Outputs.allocate(output_plan)

    tree_rescore_examined = 0
    tree_rescore_ambiguous = 0
    tree_rescore_winner_changes = 0
    tree_rescore_exact_ties = 0

    start_idx = 0
    image_indices = np.arange(n_images)
    from relax.helpers.batch_fetch import iter_indexed_batches, prefetched_batches

    _coarse_batch_starts = []
    _coarse_loop_t0 = time.time()
    dump_context = ScoreDumpContext(
        experiment_dataset=experiment_dataset,
        rotations=rotations,
        translations=translations,
        translations_source=translations_source,
        class_log_priors=class_log_priors_np,
        rotation_log_prior_padded=rotation_log_prior_padded,
        current_size=current_size,
        adaptive_fraction=adaptive_fraction,
        max_significants=max_significants,
        debug_iteration=debug_iteration,
        relion_projector_half=relion_projector_half,
        relion_projector_r_max=relion_projector_r_max,
        projection_padding_factor=projection_padding_factor,
        score_indices=operand_plan.score_indices,
    )
    batch_input_plan = BatchInputPlan(
        experiment_dataset=experiment_dataset,
        image_corrections=image_corrections,
        scale_corrections=scale_corrections,
        image_pre_shifts=image_pre_shifts,
        score_real_dtype=score_real_dtype,
        pad_final_image_batch=pad_final_image_batch,
        image_batch_size=image_batch_size,
        noise_variance_half=noise_variance_half,
        noise_table_host=noise_table_host,
        image_groups_host=image_groups_host,
        translation_log_prior=translation_log_prior,
    )
    pending_batch = None
    from relax.cuda.kernels import deferred_relion_preprocess_checks

    # The image preprocess kernel's finite check is read at the end of the loop: read at each call, it
    # waits for the previous batch's score program inside the next batch's preparation.
    with deferred_relion_preprocess_checks("pass 1") as preprocess_checks, prefetched_batches(
        iter_indexed_batches(experiment_dataset, image_indices, image_batch_size)
    ) as batches:
        for batch_data, _, _, _, _, _, indices in batches:
            _coarse_batch_starts.append(time.time())
            actual_batch_size = len(indices)
            end_idx = start_idx + actual_batch_size
            preprocess_checks.at(
                f"batch {len(_coarse_batch_starts) - 1} (images {start_idx}-{end_idx - 1} of the pass, "
                f"dataset images {int(indices[0])}-{int(indices[-1])})"
            )
            # The outputs a batch has only on some routes; publish_batch receives every name.
            batch_pmax = batch_weights = batch_sig_mask = batch_sig_rot_mask = batch_n_sig = batch_cutoff_count = None
            _batch_sum_weight = _batch_significant_weight = None
            batch_inputs = prepare_batch_inputs(
                batch_input_plan,
                batch_data,
                indices,
                start_idx=start_idx,
                end_idx=end_idx,
            )
            operands = operand_plan.prepare(batch_inputs, indices)

            # Identify per-batch dump target rows so we can record raw scores
            # (pre-prior) for each target image inside the per-class block loop.
            # This enables direct diff against RELION's exp_Mweight_diff2
            # without needing the full (batch, n_classes, n_rot*n_trans) cache.
            debug_dump_enabled = collect_significance and _significance_debug_dump_matches(
                current_size=current_size,
                debug_iteration=debug_iteration,
            )
            dump_target_local_positions = None
            if debug_dump_enabled:
                _dump_targets = parse_env_int_set("RELAX_SIGNIFICANCE_DUMP_ORIGINAL_INDICES")
                if _dump_targets:
                    _local_for_dump = np.asarray(indices, dtype=np.int64)
                    _orig = original_image_indices(experiment_dataset, _local_for_dump)
                    _positions = np.flatnonzero(np.isin(_orig, np.fromiter(_dump_targets, dtype=np.int64)))
                    if _positions.size:
                        dump_target_local_positions = _positions.astype(np.int64)
            # Per-class collectors for raw (pre-prior) score blocks at target rows.
            # Shape after concat per class: (n_targets, n_rot, n_trans)
            dump_target_pre_prior_blocks_per_class = (
                [[] for _ in range(n_classes)] if dump_target_local_positions is not None else None
            )
            dump_target_with_prior_blocks_per_class = (
                [[] for _ in range(n_classes)] if dump_target_local_positions is not None else None
            )
            neg_inf_f, zeros_f64, zeros_i32 = _pass1_batch_constants(batch_inputs.batch_size, bool(jax.config.jax_enable_x64))
            # RELION's CUDA coarse kernel forms ``pdf_orientation + pdf_offset +
            # min_diff2 - diff2`` left to right (cuda_kernel_weights_exponent_coarse).
            # Adding the priors to the absolute scores and the min_diff2 offset
            # afterwards can tie poses that RELION separates by one ULP, so the
            # support pass keeps the pre-prior scores.
            relion_exact_coarse_weight_order = bool(relion_f32_coarse_support_enabled and n_classes == 1)

            # Pass 1 of the coarse GEMM scorers is one program per batch
            # (_coarse_pass1_blocks): over the cached projections in one call, or
            # one call per class and rotation block on its projection when the
            # cache does not fit and for --firstiter_cc. The program also returns
            # the RELAX_SIGNIFICANCE_DUMP_* targets' scores and the class runner-up.
            batched_support_values = None
            if pass1_prior_terms is None:
                pass1_prior_terms = tuple(
                    tuple(
                        (
                            jnp.asarray(class_log_priors_np[class_index], dtype=jnp.float32),
                            None
                            if rotation_log_prior_padded is None
                            else jnp.asarray(
                                rotation_log_prior_padded[class_index, r0 : r0 + rotation_block_size]
                            ),
                        )
                        for r0 in range(0, n_rot_padded, rotation_block_size)
                    )
                    for class_index in range(n_classes)
                )
            pass1_blocks = tuple(
                (class_index, r0, min(rotation_block_size, n_rot - r0), int(rotation_block_size))
                for class_index in range(n_classes)
                for r0 in range(0, n_rot, rotation_block_size)
            )
            program_inputs = operands.program_inputs()
            pass1_static = dict(
                n_trans=int(n_trans),
                image_shape=tuple(int(value) for value in image_shape),
                volume_shape=tuple(int(value) for value in volume_shape),
                float64=_coarse_gemm_float64_requested(),
                score_kind="normalized_cc" if exact_cc_enabled else "gaussian",
                exact_weight_order=relion_exact_coarse_weight_order,
                return_class_best=bool(return_class_best),
                track_class_second=bool(track_class_second),
                return_values=bool(collect_significance),
            )
            pass1_dump_rows = (
                None
                if dump_target_local_positions is None
                else jnp.asarray(dump_target_local_positions, dtype=jnp.int32)
            )
            if pending_batch is not None:
                publish_batch(pending_batch, outputs, output_plan, dump_context)
                pending_batch = None
            pass1_state = _pass1_initial_state((neg_inf_f, zeros_f64, zeros_i32), n_classes)
            if coarse_gaussian_gemm_projection_cache is not None and not exact_cc_enabled:
                pass1_state, pass1_values, pass1_dumps = _coarse_pass1_blocks(
                    pass1_state,
                    coarse_gaussian_gemm_projection_cache,
                    *program_inputs,
                    actual_batch_size,
                    tuple(
                        pass1_prior_terms[class_index][r0 // rotation_block_size]
                        for class_index, r0, _, _ in pass1_blocks
                    ),
                    batch_inputs.translation_log_prior,
                    pass1_dump_rows,
                    blocks=pass1_blocks,
                    **pass1_static,
                )
            else:
                pass1_values = []
                pass1_dumps = []
                for class_index, r0, rows, block_rows in pass1_blocks:
                    rots_b = rotations_padded[r0 : r0 + block_rows]
                    if exact_cc_enabled:
                        reference, _ = coarse_projector.block_once(class_index, rots_b, rotation_start=r0)
                    else:
                        reference, _ = coarse_projector.compact_rows(class_index, rots_b, return_abs2=True)
                    block_state, block_values, block_dump = _coarse_pass1_block(
                        _class_block_state(pass1_state, class_index),
                        reference,
                        *program_inputs,
                        actual_batch_size,
                        pass1_prior_terms[class_index][r0 // rotation_block_size],
                        batch_inputs.translation_log_prior,
                        jnp.int32(class_index),
                        jnp.int32(r0),
                        pass1_dump_rows,
                        rows=rows,
                        block_rows=block_rows,
                        **pass1_static,
                    )
                    pass1_state = _merge_class_block_state(pass1_state, block_state, class_index)
                    pass1_values.append(block_values)
                    pass1_dumps.append(block_dump)
            if collect_significance:
                batched_support_values = jnp.concatenate(pass1_values, axis=1)
            if pass1_dump_rows is not None:
                for (class_index, _, _, _), (pre_prior, with_prior) in zip(pass1_blocks, pass1_dumps, strict=True):
                    dump_target_pre_prior_blocks_per_class[class_index].append(np.asarray(pre_prior, dtype=np.float64))
                    dump_target_with_prior_blocks_per_class[class_index].append(
                        np.asarray(with_prior, dtype=np.float64)
                    )
            (
                (global_max, global_sum),
                (class_max_tuple, class_sum_tuple),
                (best_score_batch, best_argmax_batch, best_class_batch),
                (class_best_tuple, class_best_argmax_tuple, class_second_tuple, class_second_argmax_tuple),
                pass1_raw_score_max,
            ) = pass1_state
            relion_raw_score_max = pass1_raw_score_max if relion_f32_coarse_support_enabled else None
            class_max_values = list(class_max_tuple)
            class_sum_values = list(class_sum_tuple)
            class_best_scores = list(class_best_tuple) if return_class_best else None
            class_best_argmaxes = list(class_best_argmax_tuple) if return_class_best else None
            class_second_best_scores = list(class_second_tuple) if track_class_second else None
            class_second_best_argmaxes = list(class_second_argmax_tuple) if track_class_second else None

            if tree_rescore_enabled:
                # The bounded top-two rescore uses the batch's exact CUDA CC operands (the per-image FFT/CTF
                # assembly), not a second copy.
                rescored = rescore_ambiguous_images(
                    tree_rescore_plan,
                    TreeRescoreState(
                        best_argmax=best_argmax_batch,
                        best_score=best_score_batch,
                        class_best_argmax=class_best_argmaxes[0],
                        class_best_score=class_best_scores[0],
                        class_second_argmax=class_second_best_argmaxes[0],
                        class_second_score=class_second_best_scores[0],
                    ),
                    operands.unshifted,
                    operands.corr_img,
                    experiment_dataset=experiment_dataset,
                    indices=indices,
                    debug_iteration=debug_iteration,
                )
                tree_rescore_examined += int(batch_inputs.batch_size)
                tree_rescore_ambiguous += rescored.ambiguous_images
                tree_rescore_exact_ties += rescored.exact_ties
                tree_rescore_winner_changes += rescored.winner_changes
                best_argmax_batch = rescored.state.best_argmax
                best_score_batch = rescored.state.best_score
                class_best_argmaxes[0] = rescored.state.class_best_argmax
                class_best_scores[0] = rescored.state.class_best_score
                class_second_best_argmaxes[0] = rescored.state.class_second_argmax
                class_second_best_scores[0] = rescored.state.class_second_score

            global_log_z = global_max + jnp.log(global_sum)
            class_log_z_values = [
                class_max + jnp.log(class_sum) for class_max, class_sum in zip(class_max_values, class_sum_values)
            ]

            normalization_score_mats = []
            if collect_significance:
                batch_values = (
                    batched_support_values
                    if relion_f32_coarse_support_enabled
                    else jnp.exp(batched_support_values - global_log_z[:, None])
                )
                if return_relion_f32_normalization and not relion_f32_coarse_support_enabled:
                    # The program's values are the with-prior scores here.
                    normalization_score_mats.append(batched_support_values)
                if relion_f32_coarse_support_enabled:
                    # The float32 posterior and what the batch publishes from it are one program
                    # (coarse_support_posterior). K=1 hands it the pre-prior scores and the priors:
                    # RELION's log-weight order (relion_exact_coarse_weight_order above).
                    if relion_exact_coarse_weight_order and exact_rotation_prior is None:
                        exact_rotation_prior = (
                            jnp.zeros(n_rot, dtype=jnp.float32)
                            if rotation_log_prior_padded is None
                            else jnp.asarray(rotation_log_prior_padded[0, :n_rot], dtype=jnp.float32)
                        ) + jnp.asarray(class_log_priors_np[0], dtype=jnp.float32)
                    support = coarse_support_posterior(
                        batch_values,
                        relion_raw_score_max,
                        exact_rotation_prior,
                        (
                            None
                            if not relion_exact_coarse_weight_order
                            else np.zeros(n_trans, dtype=np.float32)
                            if batch_inputs.translation_log_prior is None
                            else batch_inputs.translation_log_prior
                        ),
                        exact_weight_order=relion_exact_coarse_weight_order,
                        n_trans=int(n_trans),
                        adaptive_fraction=float(adaptive_fraction),
                        max_significants=max_significants,
                        tie_score_ulps=int(relion_f32_coarse_tie_ulps),
                    )
                    batch_weights = support["weights"]
                    batch_sig_mask = support["mask"]
                    batch_n_sig = support["n_significant"]
                    batch_cutoff_count = support["cutoff_count"]
                    _batch_sum_weight = support["sum_weight"]
                    _batch_significant_weight = support["significant_weight"]
                    batch_sig_rot_mask = support["rotation_mask"]
                    batch_pmax = support["pmax"]
                    if relion_exact_coarse_weight_order:
                        # RELION publishes the coarse winner from these weights (class 0: one class).
                        best_argmax_batch = support["winner"]
                else:
                    batch_weights = batch_values
                    if return_relion_f32_normalization:
                        from relax.sparse_pass2.sparse_pass2_posterior import _relion_f32_fine_posterior

                        # Retain the existing coarse selector and all of its outputs.
                        # The symbolic fine pass needs the numeric maximum-shifted
                        # denominator, not exp(logZ) or a second normalized support.
                        # See docs/math/zero_oversampling.md.
                        normalization_probs, _, _, _, normalization_sum, _ = _relion_f32_fine_posterior(
                            jnp.concatenate(normalization_score_mats, axis=1),
                            adaptive_fraction=adaptive_fraction,
                            keep_all=True,
                        )
                        outputs.relion_f32_sum_weight[start_idx:end_idx] = np.asarray(
                            normalization_sum, dtype=np.float32,
                        )[:actual_batch_size]
                        outputs.relion_f32_max_posterior[start_idx:end_idx] = (
                            _coarse_max_posterior_for_host(normalization_probs, actual_batch_size)
                        )
                    (
                        batch_sig_mask,
                        batch_sig_rot_mask,
                        batch_n_sig,
                        batch_cutoff_count,
                    ) = _find_sig(
                        batch_weights,
                        n_classes * n_rot,
                        n_trans,
                        adaptive_fraction=adaptive_fraction,
                        max_significants=max_significants,
                        return_cutoff_count=True,
                    )

            # A batch on the float32 support route publishes while the next batch's operands are on the
            # device and before that batch's score program (the call above the program): the device
            # scores this batch while the host prepares the next. Only a dump reads the large score and
            # operand arrays, and a dump batch publishes at once, so a waiting batch does not hold them.
            defer_publish = collect_significance and relion_f32_coarse_support_enabled and not debug_dump_enabled
            batch_outputs = BatchOutputs(
                start_idx=start_idx,
                end_idx=end_idx,
                actual_batch_size=actual_batch_size,
                batch_size=batch_inputs.batch_size,
                indices=indices,
                pmax=batch_pmax,
                weights=None if defer_publish else batch_weights,
                sig_mask=batch_sig_mask,
                sig_rot_mask=batch_sig_rot_mask,
                n_sig=batch_n_sig,
                cutoff_count=batch_cutoff_count,
                sum_weight=_batch_sum_weight,
                significant_weight=None if defer_publish else _batch_significant_weight,
                best_argmax=best_argmax_batch,
                best_class=best_class_batch,
                best_score=best_score_batch,
                global_log_z=global_log_z,
                class_log_z_values=class_log_z_values,
                class_best_scores=class_best_scores,
                class_best_argmaxes=class_best_argmaxes,
                class_second_best_scores=class_second_best_scores,
                class_second_best_argmaxes=class_second_best_argmaxes,
                debug_dump_enabled=debug_dump_enabled,
                dump_target_local_positions=dump_target_local_positions,
                dump_target_pre_prior_blocks_per_class=None if defer_publish else dump_target_pre_prior_blocks_per_class,
                dump_target_with_prior_blocks_per_class=None if defer_publish else dump_target_with_prior_blocks_per_class,
                translation_log_prior=None if defer_publish else batch_inputs.translation_log_prior,
                operands=None if defer_publish else operands,
            )
            if defer_publish:
                pending_batch = batch_outputs
            else:
                publish_batch(batch_outputs, outputs, output_plan, dump_context)
            start_idx = end_idx
        if pending_batch is not None:
            publish_batch(pending_batch, outputs, output_plan, dump_context)

    log_batch_timing(_coarse_batch_starts, _coarse_loop_t0, n_images)

    significant_sample_indices = significant_samples_after_loop(outputs, output_plan)

    full_stats = {
        "normalization_log_z": outputs.normalization_log_z,
        "normalization_log_evidence": outputs.normalization_log_evidence,
        "log_evidence_per_image": outputs.log_evidence,
        "best_log_score_per_image": outputs.best_log_score,
        "max_posterior_per_image": outputs.max_posterior,
        "class_log_evidence_per_image": outputs.class_log_evidence,
        "class_assignments": outputs.class_assignment,
        # RELION serializes the cutoff rank before inclusive threshold ties
        # expand the pass-2/M-step support represented by ``n_sig_all``.
        "significant_cutoff_counts": outputs.cutoff_count_all,
        "executed_coarse_backend": (
            "exact_cc_gemm" if exact_cc_enabled else "gemm_macro"
        ),
    }
    if gaussian_plan is not None:
        full_stats["coarse_gaussian_square_layout"] = coarse_square_layout_metadata(
            gaussian_plan.square_layout,
            stable_fourier_window_shapes=stable_fourier_window_shapes,
        )
    if gaussian_plan is not None:
        full_stats["coarse_gaussian_gemm_resources"] = {
            field: int(value)
            for field, value in gaussian_plan.resource_estimate._asdict().items()
        }
    if gaussian_plan is not None and gaussian_plan.projection_cache_plan is not None:
        full_stats["coarse_gaussian_gemm_projection_cache"] = (
            _coarse_gaussian_gemm_projection_cache_stats(
                gaussian_plan.projection_cache_plan,
                enabled=coarse_gaussian_gemm_projection_cache is not None,
            )
        )
    if _coarse_significance_support_audit_enabled():
        if significant_sample_indices is None:
            raise RuntimeError(
                f"{_COARSE_SIGNIFICANCE_SUPPORT_AUDIT_ENV}=1 requires "
                "collect_significance=True",
            )
        full_stats["coarse_significance_support_audit"] = (
            _build_coarse_significance_support_audit(
                significant_sample_indices,
                samples_per_class=n_rot * n_trans,
                include_ids=_coarse_significance_support_audit_ids_enabled(),
            )
        )
    if outputs.relion_f32_sum_weight is not None:
        # RELION's oversampling-zero second pass deliberately reuses this
        # coarse, maximum-shifted float32 denominator numerically.  It is not
        # interchangeable with a log-evidence value because the fine pass
        # independently shifts its own maximum to 50 before division.
        full_stats["relion_f32_sum_weight"] = outputs.relion_f32_sum_weight
    if outputs.relion_f32_max_posterior is not None:
        full_stats["relion_f32_max_posterior"] = outputs.relion_f32_max_posterior
    if return_class_best:
        full_stats["class_best_log_score_per_image"] = outputs.class_best_log_score
        full_stats["class_best_offset_free_log_score_per_image"] = outputs.class_best_offset_free_log_score
        full_stats["class_hard_assignments"] = outputs.class_hard_assignment
    if return_class_second:
        full_stats["class_second_best_log_score_per_image"] = outputs.class_second_best_log_score
        full_stats["class_second_hard_assignments"] = outputs.class_second_hard_assignment
        full_stats["class_second_best_offset_free_log_score_per_image"] = outputs.class_second_best_offset_free_log_score
    if tree_rescore_enabled:
        full_stats["firstiter_cc_tree_top2_rescore"] = {
            "max_margin": float(tree_rescore_max_margin),
            "examined_images": int(tree_rescore_examined),
            "ambiguous_images": int(tree_rescore_ambiguous),
            "exact_score_ties": int(tree_rescore_exact_ties),
            "winner_changes": int(tree_rescore_winner_changes),
        }
        logger.warning(
            "RELION coarse-tree top-2 rescore complete: "
            "examined=%d ambiguous=%d exact_ties=%d winner_changes=%d",
            tree_rescore_examined,
            tree_rescore_ambiguous,
            tree_rescore_exact_ties,
            tree_rescore_winner_changes,
        )
    return (
        outputs.sig_rot_any,
        outputs.n_sig_all,
        outputs.hard_assignment,
        outputs.class_assignment,
        significant_sample_indices,
        full_stats,
    )
