"""Planning one pass 1 (coarse significance) before its first batch: the refusals, the route and every stage's plan.

:func:`plan_pass1` turns a :class:`~relax.scoring.pass1_request.Pass1Request` into a :class:`Pass1Plan`: it refuses a
request pass 1 cannot score, decides once whether the pass scores with the Gaussian GEMM or the normalized CC, builds
the projector (and the Gaussian route's projection cache) and each stage's plan record. Nothing is scored here.
"""

import logging
import operator
import os
from dataclasses import dataclass
from functools import partial
from typing import Any

import jax.numpy as jnp
import numpy as np
from recovar.utils.nvtx_shim import nvtx

from relax.helpers.env_flags import (
    parse_env_strict_flag,
)
from relax.helpers.projection_cache import build_projection_cache
from relax.relion.relion_coarse_operands import (
    infer_relion_coarse_healpix_order,
    k1_relion_f32_coarse_support_enabled,
)
from relax.scoring.coarse_projector import CoarseProjector, CompactRows
from relax.scoring.gaussian_plan import coarse_gaussian_report, plan_coarse_gaussian
from relax.scoring.pass1_batch import BatchInputPlan, resolve_noise_tables
from relax.scoring.pass1_operands import CcOperandPlan, GaussianOperandPlan
from relax.scoring.pass1_priors import plan_rotation_blocks, validated_translation_log_prior
from relax.scoring.pass1_request import Pass1Request
from relax.scoring.pass1_results import (
    OutputPlan,
    PassShape,
    ScoreDumpContext,
)
from relax.scoring.pass1_scores import ScoreProgramPlan, block_prior_terms, score_blocks
from relax.scoring.pass1_support import (
    SupportPlan,
    exact_order_rotation_prior,
)
from relax.scoring.pass1_window import coarse_kernel_window, plan_scoring_window
from relax.scoring.scoring import (
    coarse_gemm_float64_requested,
)
from relax.scoring.tree_rescore import (
    TreeRescoreGeometry,
    plan_tree_rescore,
    require_tree_rescore_call,
)

_GLOBAL_PASS1_RELION_PROJECTOR_TEXTURE_ENV = "RELAX_RELION_GLOBAL_PASS1_PROJECTOR_TEXTURE_INTERP"
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


def global_pass1_relion_projector_texture_enabled() -> bool:
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


@dataclass(frozen=True)
class Pass1Plan:
    """A planned pass 1: what :func:`relax.scoring.significance.run_pass1` runs its batches from.

    ``experiment_dataset`` yields the ``n_images`` images in batches of ``image_batch_size``; ``current_size``,
    ``debug_iteration`` and ``collect_significance`` are the request's. ``relion_f32_coarse_support_enabled`` is the
    support route (float32 on the device, or the generic one). The stage plans: ``batch_input_plan`` (preprocessing),
    ``operand_plan`` (the route's score operands), ``score_program_plan``, ``support_plan``, ``tree_rescore_plan``
    (``None`` unless the pass rescores) with ``tree_rescore_max_margin`` its margin, ``output_plan`` (the results and how
    they are allocated and published) and ``dump_context`` (the score dump). ``executed_backend`` names the scorer and
    ``gaussian_report`` is the Gaussian route's entries of ``full_stats`` (empty for the normalized CC).
    """

    experiment_dataset: Any
    n_images: int
    image_batch_size: int
    current_size: Any
    debug_iteration: Any
    collect_significance: bool
    relion_f32_coarse_support_enabled: bool
    batch_input_plan: Any
    operand_plan: Any
    score_program_plan: Any
    support_plan: Any
    tree_rescore_plan: Any
    tree_rescore_max_margin: Any
    output_plan: Any
    dump_context: Any
    executed_backend: str
    gaussian_report: dict

    @property
    def tree_rescore_enabled(self) -> bool:
        """Whether the pass rescores the ambiguous winners of its normalized-CC scores."""

        return self.tree_rescore_plan is not None


def plan_pass1(request: Pass1Request) -> Pass1Plan:
    """Refuse a request pass 1 cannot score, and plan the pass: see the module docstring."""

    coarse_healpix_order = request.coarse_healpix_order
    coarse_rotation_ids = request.coarse_rotation_ids
    image_batch_size = request.image_batch_size
    pad_final_image_batch = request.pad_final_image_batch
    relion_projector_half = request.relion_projector_half
    rotation_block_size = request.rotation_block_size
    translation_log_prior = request.translation_log_prior

    if request.return_class_second and not request.return_class_best:
        raise ValueError("return_class_second requires return_class_best")
    if request.return_relion_f32_normalization and (
        not request.collect_significance or request.score_mode != "gaussian" or request.use_float64_scoring
    ):
        raise ValueError("RELION float32 normalization requires Gaussian float32 significance")



    if request.score_mode not in {"gaussian", "normalized_cc"}:
        raise ValueError(f"score_mode must be 'gaussian' or 'normalized_cc', got {request.score_mode!r}")
    # VDAM asks for the padded tail batch explicitly; the global coarse pass
    # opts in through the environment while the bitwise equality is qualified.
    pad_final_image_batch = bool(pad_final_image_batch) or _coarse_pad_final_image_batch_enabled()
    # RELION's pdf_orientation/pdf_offset priors and score/evidence/Pmax
    # outputs are RFLOAT (double), never narrowed -- derive from the
    # caller's own use_float64_scoring instead of hardcoding float32.
    score_real_dtype = np.float64 if request.use_float64_scoring else np.float32
    # The class count is the length of the class priors; relion_projector_half carries the same count.
    class_log_priors_np = np.asarray(request.class_log_priors, dtype=np.float64).reshape(-1)
    n_classes = int(class_log_priors_np.shape[0])

    translations_source = np.asarray(
        request.translations if request.translation_phase_source is None else request.translation_phase_source,
    )
    if translations_source.shape != request.translations.shape:
        raise ValueError(
            "translation_phase_source must match translations: "
            f"{translations_source.shape} != {request.translations.shape}",
        )
    n_rot = int(request.rotations.shape[0])
    n_trans = int(request.translations.shape[0])
    n_images = int(request.experiment_dataset.n_units)
    input_image_batch_size = operator.index(image_batch_size)
    if input_image_batch_size <= 0:
        raise ValueError("image_batch_size must be positive")
    image_batch_size = input_image_batch_size
    image_shape = request.experiment_dataset.image_shape
    volume_shape = request.experiment_dataset.volume_shape
    n_half = int(image_shape[0] * (image_shape[1] // 2 + 1))
    if coarse_rotation_ids is not None:
        coarse_rotation_ids = np.asarray(coarse_rotation_ids, dtype=np.int64).reshape(-1)
        if coarse_rotation_ids.shape != (n_rot,):
            raise ValueError(
                f"coarse_rotation_ids must have shape ({n_rot},), got {coarse_rotation_ids.shape}",
            )
    if coarse_healpix_order is None:
        coarse_healpix_order = infer_relion_coarse_healpix_order(n_rot, **({"symmetry_label": request.symmetry_label} if request.symmetry_label != "C1" else {}))
    elif int(coarse_healpix_order) < 0:
        raise ValueError(f"coarse_healpix_order must be non-negative, got {coarse_healpix_order}")

    use_relion_projector = relion_projector_half is not None
    if use_relion_projector and request.relion_projector_r_max is None:
        raise ValueError("relion_projector_r_max is required when relion_projector_half is provided")
    if use_relion_projector:
        relion_projector_half = _class_stacked_coarse_relion_projector(
            relion_projector_half, n_classes, use_float64_scoring=request.use_float64_scoring,
            use_float64_projections=request.use_float64_projections,
        )

    window = plan_scoring_window(
        image_shape,
        n_half,
        request.current_size,
        score_mode=request.score_mode,
        half_spectrum_scoring=request.half_spectrum_scoring,
        square_window=request.square_window,
        window_at_box=request.window_at_box,
        nyquist_column_counting=request.nyquist_column_counting,
        firstiter_cc_support=request.firstiter_cc_support,
    )
    score_size = window.score_size
    window_spec = window.window_spec
    kernel_window = (
        coarse_kernel_window(score_size, request.relion_projector_r_max, request.rotations) if use_relion_projector else None
    )
    coarse_texture_interp = (
        global_pass1_relion_projector_texture_enabled()
        if request.relion_projector_texture_interp is None
        else bool(request.relion_projector_texture_interp)
    )
    if request.tree_rescore_max_margin is not None and not (
        np.isfinite(request.tree_rescore_max_margin) and request.tree_rescore_max_margin >= 0.0
    ):
        raise ValueError(
            f"tree_rescore_max_margin must be a finite non-negative float, got {request.tree_rescore_max_margin!r}"
        )
    # The margin is a run option, while only iteration 1 uses normalized CC.
    # Later Gaussian iterations must remain unaffected.
    tree_rescore_enabled = (
        request.tree_rescore_max_margin is not None and request.score_mode == "normalized_cc"
    )
    # Pass 1 scores RELION's exact coarse operands only: the Gaussian passes with the coarse GEMMs
    # (relion_coarse_gaussian_gemm_scores_jit), the --firstiter_cc passes with RELION's coarse CC
    # (relion_coarse_normalized_cc_gemm_scores_jit). The generic dense scorer was removed on
    # 2026-10-02; like pass 2, pass 1 needs a CUDA GPU and RELION's CUDA image preprocessing.
    _require_exact_pass1_operands(
        use_relion_projector=use_relion_projector,
        coarse_texture_interp=coarse_texture_interp,
        half_spectrum_scoring=request.half_spectrum_scoring,
        use_float64_scoring=request.use_float64_scoring,
        use_float64_projections=request.use_float64_projections,
    )
    exact_gaussian = request.score_mode == "gaussian"
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
            stable_fourier_window_shapes=request.stable_fourier_window_shapes,
        )
        rotation_block_size = gaussian_plan.rotation_block_size
    relion_f32_coarse_support_enabled = exact_gaussian and k1_relion_f32_coarse_support_enabled(default=True)
    # --firstiter_cc on RELION's exact coarse operands: the tree rescore's per-image
    # FFT, RFLOAT CTF and corr_img operands, translated with RELION's sincosf for
    # every translation and scored by the coarse GEMMs
    # (relion_coarse_normalized_cc_gemm_scores_jit).
    exact_cc_enabled = request.score_mode == "normalized_cc"
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
                angle_scale=request.relion_translation_angle_scale,
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
    track_class_second = request.return_class_second or tree_rescore_enabled
    tree_rescore_plan = None
    if tree_rescore_enabled:
        require_tree_rescore_call(
            n_classes=n_classes,
            return_class_best=request.return_class_best,
            use_relion_projector=use_relion_projector,
            coarse_texture_interp=coarse_texture_interp,
            half_spectrum_scoring=request.half_spectrum_scoring,
        )
        tree_rescore_plan = plan_tree_rescore(
            max_margin=request.tree_rescore_max_margin,
            relion_projector_half=relion_projector_half,
            image_shape=image_shape,
            n_half=n_half,
            score_indices_np=window_spec.score_indices_np,
            translations_source=translations_source,
            relion_translation_angle_scale=request.relion_translation_angle_scale,
            geometry=TreeRescoreGeometry(
                half_weights=window.score_half_weights,
                rotations=request.rotations,
                n_trans=n_trans,
                score_size=score_size,
                padding_factor=request.projection_padding_factor,
                projector_max_r=request.relion_projector_r_max,
                coarse_healpix_order=coarse_healpix_order,
                coarse_rotation_ids=coarse_rotation_ids,
                symmetry_label=request.symmetry_label,
            ),
        )

    if exact_cc_enabled:
        operand_plan = CcOperandPlan(
            experiment_dataset=request.experiment_dataset,
            image_shape=image_shape,
            image_pre_shifts=request.image_pre_shifts,
            window_indices=window.window_indices,
            score_indices=exact_cc_score_indices,
            score_half_weights=window.score_half_weights,
            support_power_weights=window.score_half_weights if window.cc_gaussian_support else None,
            translation_angles=exact_cc_translation_angles,
            n_trans=n_trans,
            score_with_masked_images=request.score_with_masked_images,
            scale_corrections_enabled=request.scale_corrections is not None,
        )
    else:
        operand_plan = GaussianOperandPlan(
            experiment_dataset=request.experiment_dataset,
            gaussian_plan=gaussian_plan,
            image_shape=image_shape,
            half_weights=window.half_weights,
            translations_source=translations_source,
            relion_translation_angle_scale=request.relion_translation_angle_scale,
            score_with_masked_images=request.score_with_masked_images,
            nyquist_column_counting=request.nyquist_column_counting,
            scale_corrections_enabled=request.scale_corrections is not None,
            use_float64_scoring=request.use_float64_scoring,
            stable_fourier_window_shapes=request.stable_fourier_window_shapes,
            score_size=score_size,
            current_size=request.current_size,
            coarse_kernel_window=kernel_window,
            coarse_kernel_r_max=None if kernel_window is None else int(request.relion_projector_r_max),
        )

    rotation_blocks = plan_rotation_blocks(
        request.rotations,
        request.rotation_log_prior,
        n_classes=n_classes,
        rotation_block_size=rotation_block_size,
        score_real_dtype=score_real_dtype,
    )
    rotations_padded = rotation_blocks.rotations_padded
    rotation_log_prior_padded = rotation_blocks.rotation_log_prior_padded
    translation_log_prior = validated_translation_log_prior(
        translation_log_prior,
        n_images=n_images,
        n_trans=n_trans,
        score_real_dtype=score_real_dtype,
    )

    noise_tables = resolve_noise_tables(request.noise_variance, image_shape, request.optics_group_ids)

    # The texture projector naturally produces a centered current-size crop.
    # Ask it only for the rows consumed by the scorer instead of scattering the
    # crop into a full image and immediately gathering the same rows again.
    # This is an exact index remapping and avoids a large transient scatter for
    # global rotation blocks.
    projector_compact_rows = None
    if use_relion_projector and coarse_texture_interp:
        if exact_gaussian:
            projector_compact_rows = CompactRows(gaussian_plan.score_indices_np, gaussian_plan.projector_output_size)
        elif window.use_window:
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
        relion_projector_r_max=request.relion_projector_r_max,
        image_shape=image_shape,
        projection_padding_factor=request.projection_padding_factor,
        current_size=request.current_size,
        score_size=score_size,
        stable_fourier_window_shapes=request.stable_fourier_window_shapes,
        compact=projector_compact_rows,
        rotated_radius=coarse_rotated_radius,
    )

    coarse_gaussian_gemm_projection_cache = None
    if exact_gaussian and gaussian_plan.projection_cache_plan is not None:

        coarse_gaussian_gemm_projection_cache = (
            build_projection_cache(
                gaussian_plan.projection_cache_plan,
                partial(coarse_projector.cache_block, request.rotations),
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

    # RELION's CUDA coarse kernel forms ``pdf_orientation + pdf_offset +
    # min_diff2 - diff2`` left to right (cuda_kernel_weights_exponent_coarse).
    # Adding the priors to the absolute scores and the min_diff2 offset
    # afterwards can tie poses that RELION separates by one ULP, so the
    # support pass keeps the pre-prior scores.
    relion_exact_coarse_weight_order = bool(relion_f32_coarse_support_enabled and n_classes == 1)
    pass1_blocks = score_blocks(n_classes, n_rot, rotation_block_size)
    score_program_plan = ScoreProgramPlan(
        blocks=pass1_blocks,
        prior_terms=block_prior_terms(pass1_blocks, class_log_priors_np, rotation_log_prior_padded, rotation_block_size),
        rotations_padded=rotations_padded,
        projector=coarse_projector,
        projection_cache=coarse_gaussian_gemm_projection_cache,
        n_classes=n_classes,
        n_trans=int(n_trans),
        image_shape=tuple(int(value) for value in image_shape),
        volume_shape=tuple(int(value) for value in volume_shape),
        float64=coarse_gemm_float64_requested(),
        score_kind="normalized_cc" if exact_cc_enabled else "gaussian",
        exact_weight_order=relion_exact_coarse_weight_order,
        return_class_best=bool(request.return_class_best),
        track_class_second=bool(track_class_second),
        return_values=bool(request.collect_significance),
    )
    support_plan = SupportPlan(
        n_classes=n_classes,
        n_rot=n_rot,
        n_trans=int(n_trans),
        adaptive_fraction=request.adaptive_fraction,
        max_significants=request.max_significants,
        tie_score_ulps=int(request.relion_f32_coarse_tie_ulps),
        return_relion_f32_normalization=request.return_relion_f32_normalization,
        exact_weight_order=relion_exact_coarse_weight_order,
        exact_rotation_prior=(
            exact_order_rotation_prior(class_log_priors_np, rotation_log_prior_padded, n_rot)
            if relion_exact_coarse_weight_order
            else None
        ),
    )

    output_plan = OutputPlan(
        n_classes=n_classes,
        n_rot=n_rot,
        n_trans=n_trans,
        n_images=n_images,
        score_real_dtype=score_real_dtype,
        collect_significance=request.collect_significance,
        relion_f32_coarse_support_enabled=relion_f32_coarse_support_enabled,
        return_relion_f32_normalization=request.return_relion_f32_normalization,
        return_class_best=request.return_class_best,
        return_class_second=request.return_class_second,
    )

    dump_context = ScoreDumpContext(
        experiment_dataset=request.experiment_dataset,
        rotations=request.rotations,
        translations=request.translations,
        translations_source=translations_source,
        class_log_priors=class_log_priors_np,
        rotation_log_prior_padded=rotation_log_prior_padded,
        current_size=request.current_size,
        adaptive_fraction=request.adaptive_fraction,
        max_significants=request.max_significants,
        debug_iteration=request.debug_iteration,
        relion_projector_half=relion_projector_half,
        relion_projector_r_max=request.relion_projector_r_max,
        projection_padding_factor=request.projection_padding_factor,
        score_indices=operand_plan.score_indices,
    )
    batch_input_plan = BatchInputPlan(
        experiment_dataset=request.experiment_dataset,
        image_corrections=request.image_corrections,
        scale_corrections=request.scale_corrections,
        image_pre_shifts=request.image_pre_shifts,
        score_real_dtype=score_real_dtype,
        pad_final_image_batch=pad_final_image_batch,
        image_batch_size=image_batch_size,
        noise=noise_tables,
        translation_log_prior=translation_log_prior,
    )
    return Pass1Plan(
        experiment_dataset=request.experiment_dataset,
        n_images=n_images,
        image_batch_size=image_batch_size,
        current_size=request.current_size,
        debug_iteration=request.debug_iteration,
        collect_significance=request.collect_significance,
        relion_f32_coarse_support_enabled=relion_f32_coarse_support_enabled,
        batch_input_plan=batch_input_plan,
        operand_plan=operand_plan,
        score_program_plan=score_program_plan,
        support_plan=support_plan,
        tree_rescore_plan=tree_rescore_plan,
        tree_rescore_max_margin=request.tree_rescore_max_margin,
        output_plan=output_plan,
        dump_context=dump_context,
        executed_backend="exact_cc_gemm" if exact_cc_enabled else "gemm_macro",
        gaussian_report=(
            coarse_gaussian_report(
                gaussian_plan, stable_fourier_window_shapes=request.stable_fourier_window_shapes
            )
            if exact_gaussian
            else {}
        ),
    )
