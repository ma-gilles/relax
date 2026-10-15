"""Scoring window, RELION flags and per-pass budgets of the sparse bucketed pass 2.

The scoring-window setup shared by both pass-2 entry points, the RELION
behaviour flags they resolve once per pass, the half-spectrum scoring
weights, the projection budget, the two-dimensional fine translation prior,
the shared K-class noise variance and the projected-reference subtraction
applied to sparse M-step sums.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from recovar.core.configs import ForwardModelConfig

from relax.fourier.fourier_window import centered_half_indices_to_fftw_half_indices, make_fourier_window_spec
from relax.fourier.half_spectrum import make_scoring_half_image_weights
from relax.numerics.precision import DensePrecisionPolicy
from relax.runtime.memory_budget import (
    device_memory_limit_bytes,
    max_projected_rotations_per_call_for_pass,
    projection_budget_pixels_for_pass,
    projection_cache_budget_complex_dtype,
)
from relax.sampling.translation_prior import expand_fine_translation_prior

logger = logging.getLogger(__name__)


class _SparsePass2WindowSetup(NamedTuple):
    """Forward-model configuration and Fourier windows of one sparse pass-2 scorer."""

    config: object
    window_spec: object
    use_window: bool
    window_indices_np: object
    window_indices: object
    recon_window_indices: object
    relion_x_half_recon_indices: object
    windowed_prepare: bool
    n_windowed: int
    n_recon_windowed: int


def sparse_pass2_window_setup(
    experiment_dataset,
    *,
    disc_type,
    image_shape,
    current_size,
    n_half,
    mstep_current_size,
    square_window,
    window_spec_kwargs,
    use_relion_x_half_mstep,
    log_label,
    window_spec_override=None,
) -> _SparsePass2WindowSetup:
    """Build the forward model and score/reconstruction windows of a sparse pass 2.

    The RELION x-half M-step addresses its reconstruction window in FFTW half
    order, so the centred reconstruction indices are converted when that
    layout is active. Windowed prepare is logged once per pass with the
    caller's label. The single-class and fused K-class sparse scorers share
    this setup. A supplied physical window spec retains the same conversion
    and prepare policy while the caller owns its logical support masks.
    """

    config = ForwardModelConfig.from_dataset(
        experiment_dataset,
        disc_type=disc_type,
        process_fn=experiment_dataset.process_images,
    )
    window_spec = window_spec_override
    if window_spec is None:
        window_spec = make_fourier_window_spec(
            image_shape,
            current_size,
            n_half,
            reconstruction_current_size=mstep_current_size,
            square=square_window,
            include_recon_window=True,
            **window_spec_kwargs,
        )
    use_window = window_spec.use_window
    recon_window_indices = window_spec.recon_indices
    relion_x_half_recon_indices = None
    if use_relion_x_half_mstep:
        centered_recon_indices = (
            recon_window_indices
            if recon_window_indices is not None
            else jnp.arange(int(n_half), dtype=jnp.int32)
        )
        relion_x_half_recon_indices = centered_half_indices_to_fftw_half_indices(
            image_shape,
            centered_recon_indices,
        )
    # A windowed pass materializes only its active Fourier rows.
    windowed_prepare = bool(use_window)
    n_windowed = window_spec.n_score
    n_recon_windowed = window_spec.n_recon
    if windowed_prepare:
        logger.info(
            "%s windowed prepare (score_pixels=%d recon_pixels=%d full_half_pixels=%d)",
            log_label,
            int(n_windowed),
            int(n_recon_windowed),
            int(n_half),
        )
    return _SparsePass2WindowSetup(
        config,
        window_spec,
        use_window,
        window_spec.score_indices_np,
        window_spec.score_indices,
        recon_window_indices,
        relion_x_half_recon_indices,
        windowed_prepare,
        n_windowed,
        n_recon_windowed,
    )


def pass2_relion_flags(*, relion_exact_fine_gaussian, relion_firstiter_score_mode, relion_f32_fine_posterior, f32_fine_posterior_override):
    """The two RELION fine-scoring flags of a pass 2: exact Gaussian scoring and the float32 fine posterior.

    The float32 fine posterior is on when the caller's arithmetic asks for it (``relion_f32_fine_posterior``) or the
    run's ``ScoringOverrides.f32_fine_posterior`` (``f32_fine_posterior_override``) forces it.
    """

    use_exact_relion_gaussian = bool(
        relion_exact_fine_gaussian
        and relion_firstiter_score_mode == "gaussian"
    )
    use_relion_f32_fine_posterior = bool(relion_f32_fine_posterior or f32_fine_posterior_override)
    return use_exact_relion_gaussian, use_relion_f32_fine_posterior


def pass2_half_weights(
    image_shape,
    window_spec,
    *,
    half_spectrum_scoring: bool,
    relion_firstiter_score_mode,
    use_float64_scoring: bool,
    nyquist_column_counting: str = "relion",
    firstiter_cc_support: str = "relion",
    current_size=None,
):
    """Scoring half-image weights of a pass 2, full and windowed, in the scoring precision.

    ``nyquist_column_counting`` and ``firstiter_cc_support`` are the consistency options of
    :func:`relax.fourier.half_spectrum.make_scoring_half_image_weights`; a normalized-CC pass with
    ``firstiter_cc_support="gaussian"`` is weighted on the Gaussian support of ``current_size``.
    """

    normalized_cc = relion_firstiter_score_mode == "normalized_cc"
    half_weights = make_scoring_half_image_weights(
        image_shape,
        relion_half_sum=half_spectrum_scoring,
        exclude_relion_redundant_x0=not normalized_cc,
        nyquist_column_counting=nyquist_column_counting,
        firstiter_cc_support_size=(
            (image_shape[0] if current_size is None else current_size)
            if normalized_cc and firstiter_cc_support == "gaussian"
            else None
        ),
    )
    half_weights_windowed = window_spec.score_values(half_weights)
    if use_float64_scoring:
        half_weights = half_weights.astype(jnp.float64)
        half_weights_windowed = window_spec.score_values(half_weights)
    return half_weights, half_weights_windowed


def pass2_projection_budget(
    mean_dtype,
    precision_policy: DensePrecisionPolicy,
    *,
    n_half: int,
    use_relion_projector: bool,
    budget_window_spec,
    device_memory_bytes,
    include_abs2: bool,
    max_projected_rotations_override: int | None,
    projection_max_bytes: int | None,
):
    """Projection cache dtype, budget pixels and rotations per projection call of a pass 2
    (``max_projected_rotations_override``: ``EngineTuning.max_projected_rotations``; ``projection_max_bytes``:
    ``EngineTuning.projection_max_bytes``).

    ``include_abs2`` says whether the projection call also materializes the
    squared magnitudes; the single-volume route skips them for score-only
    passes, the fused K-class route only under a window.
    """

    projection_complex_dtype = projection_cache_budget_complex_dtype(
        mean_dtype,
        precision_policy.score_complex_dtype,
        use_relion_projector=use_relion_projector,
    )
    projection_budget_pixels = projection_budget_pixels_for_pass(
        n_half,
        use_window=budget_window_spec.use_window,
        use_relion_projector=use_relion_projector,
    )
    max_projected_rotations_per_projection_call = max_projected_rotations_per_call_for_pass(
        device_memory_bytes=device_memory_bytes,
        n_projection_pixels=projection_budget_pixels,
        projection_complex_dtype=projection_complex_dtype,
        include_abs2=include_abs2,
        override=max_projected_rotations_override,
        projection_max_bytes=projection_max_bytes,
    )
    return projection_complex_dtype, projection_budget_pixels, max_projected_rotations_per_projection_call


class _Pass2WindowSetup(NamedTuple):
    """Window, memory and precision setup shared by both bucketed pass-2 entry points."""

    mstep_current_size: int | None
    n_half: int
    window_spec_kwargs: dict
    budget_window_spec: object
    device_memory_bytes: int | None
    precision_policy: DensePrecisionPolicy


def pass2_window_setup(
    image_shape,
    *,
    current_size,
    reconstruction_current_size,
    half_spectrum_scoring: bool,
    square_window: bool,
    relion_firstiter_score_mode,
    use_exact_relion_gaussian: bool,
    use_float64_scoring: bool,
    device_memory_override: int | None,
    window_at_box: bool = False,
    reference_sphere_clip: bool = False,
) -> _Pass2WindowSetup:
    """Resolve the M-step current size, the score/recon window and the precision policy of a pass 2.

    ``device_memory_override`` (``EngineTuning.device_memory_bytes``) replaces the device memory probe.

    ``reference_sphere_clip`` marks images on another grid than the reference, whose M-step
    adjoint clips at the reference sphere (``relax.projection.adjoint.ReferenceSphereClip``). Their
    reconstruction window then keeps RELION's rounded support (``ires < image_current_size / 2 + 1``)
    without the exact image-radius cut: RELION's backprojector bounds only the rotated reference
    radius (BP.cuh:322), which the clip reproduces. At the class's box the exact cut would drop
    its outer ring (radius N/2 to N/2 + 1/2) that RELION backprojects.
    """

    H, W = image_shape
    mstep_current_size = (
        current_size
        if reconstruction_current_size is None
        else int(reconstruction_current_size)
    )
    if window_at_box and mstep_current_size is None:
        mstep_current_size = int(W)
    if (
        use_exact_relion_gaussian
        and current_size is not None
        and int(current_size) < int(W)
        and (not half_spectrum_scoring or square_window)
    ):
        raise NotImplementedError(
            "exact RELION fine Gaussian high-resolution scoring requires "
            "half_spectrum_scoring=True and square_window=False"
        )
    n_half = H * (W // 2 + 1)
    window_spec_kwargs = {}
    if relion_firstiter_score_mode == "normalized_cc":
        window_spec_kwargs = {
            "score_square": True,
            "score_include_dc": True,
        }
    if window_at_box:
        window_spec_kwargs = {**window_spec_kwargs, "window_at_box": True}
    if reference_sphere_clip:
        window_spec_kwargs = {**window_spec_kwargs, "recon_exact_radius": False}
    budget_window_spec = make_fourier_window_spec(
        image_shape,
        current_size,
        n_half,
        reconstruction_current_size=mstep_current_size,
        square=square_window,
        include_recon_window=True,
        **window_spec_kwargs,
    )
    return _Pass2WindowSetup(
        mstep_current_size=mstep_current_size,
        n_half=n_half,
        window_spec_kwargs=window_spec_kwargs,
        budget_window_spec=budget_window_spec,
        device_memory_bytes=device_memory_limit_bytes(device_memory_override),
        precision_policy=DensePrecisionPolicy(use_float64_scoring=use_float64_scoring),
    )


def fine_translation_log_prior_2d(translation_log_prior, fine_translation_parent, *, n_images, n_fine_trans, dtype):
    """Expand the coarse translation log-prior onto the fine translation grid, or ``None`` without a prior."""

    if translation_log_prior is None:
        return None
    translation_log_prior_np = np.asarray(translation_log_prior, dtype=dtype)
    return expand_fine_translation_prior(
        translation_log_prior_np,
        fine_translation_parent,
        n_images=n_images,
        n_fine_trans=n_fine_trans,
        dtype=dtype,
    )


