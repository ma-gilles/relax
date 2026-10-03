"""Test-only score diagnostics, retained without changing their arithmetic.

The float64 JAX lane replay is diagnostic, not a production precision policy.
Repeat/scale summaries cannot qualify numerical noise or scientific acceptance
by themselves. Runtime scoring does not import this module.
"""

import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def _relion_coarse_normalized_cc_rescore_f64(
    shifted_candidates,
    score_weight_candidates,
    projection_candidates,
    half_weights,
    fftw_order,
):
    """Rescore coarse candidates with RELION double-ACC lane arithmetic."""

    shifted = jnp.asarray(shifted_candidates, dtype=jnp.complex128)[..., fftw_order]
    score_weight = jnp.asarray(score_weight_candidates, dtype=jnp.float64)[..., fftw_order]
    projection = jnp.asarray(projection_candidates, dtype=jnp.complex128)[..., fftw_order]
    weights = jnp.asarray(half_weights, dtype=jnp.float64)[fftw_order]
    numerator_pixels = jnp.real(jnp.conj(shifted) * projection) * weights
    norm_pixels = score_weight * (jnp.abs(projection) ** 2) * weights

    def reduce_lanes(values):
        n_pixels = int(values.shape[-1])
        n_passes = (n_pixels + 127) // 128
        padded = jnp.pad(
            values,
            [(0, 0)] * (values.ndim - 1) + [(0, n_passes * 128 - n_pixels)],
        )
        passes = padded.reshape(values.shape[:-1] + (n_passes, 128))
        lanes = jnp.zeros(values.shape[:-1] + (128,), dtype=jnp.float64)
        for pass_index in range(n_passes):
            lanes = lanes + passes[..., pass_index, :]
        for stride in (64, 32, 16, 8, 4, 2, 1):
            lanes = lanes[..., :stride] + lanes[..., stride : 2 * stride]
        return lanes[..., 0]

    numerator = reduce_lanes(numerator_pixels)
    norm = reduce_lanes(norm_pixels)
    contribution = numerator / (
        jnp.asarray(128.0, dtype=jnp.float64)
        * jnp.sqrt(jnp.maximum(norm, jnp.asarray(1e-30, dtype=jnp.float64)))
    )

    def add_once(_, accumulated):
        return accumulated + contribution

    return jax.lax.fori_loop(0, 128, add_once, jnp.zeros_like(contribution))


def _coarse_gaussian_repeat_spread_diagnostics(score_delta_repeats):
    """Report repeat-to-repeat drift without defining an acceptance epsilon.

    ``score_delta_repeats`` has layout
    ``[repeat,image,class,rotation,translation]`` and should be assembled from
    immutable same-hardware paired captures.  The caller must compare this raw
    spread with native/direct repeats, scale panels, discrete outcomes, and
    final quality; this helper cannot promote the GEMM path by itself.
    """

    deltas = np.asarray(score_delta_repeats, dtype=np.float64)
    if deltas.ndim != 5 or deltas.shape[0] < 2:
        raise ValueError(
            "coarse GEMM repeat diagnostics require at least two "
            "[repeat,image,class,rotation,translation] score-delta surfaces",
        )
    flattened = deltas.reshape(deltas.shape[0], deltas.shape[1], -1)
    signed_mean_per_repeat_image = np.mean(flattened, axis=2)
    max_abs_per_repeat_image = np.max(np.abs(flattened), axis=2)
    return {
        "repeat_count": np.asarray(deltas.shape[0], dtype=np.int64),
        "signed_mean_delta_per_repeat_image": signed_mean_per_repeat_image,
        "max_abs_delta_per_repeat_image": max_abs_per_repeat_image,
        "signed_mean_repeat_spread_per_image": np.ptp(
            signed_mean_per_repeat_image,
            axis=0,
        ),
        "max_abs_repeat_spread_per_image": np.ptp(
            max_abs_per_repeat_image,
            axis=0,
        ),
        "elementwise_delta_repeat_spread": np.ptp(deltas, axis=0),
        "qualification_status": np.asarray(
            "NO_GO_raw_repeat_spread_requires_native_scale_discrete_quality_runtime_context"
        ),
    }


def _coarse_gaussian_scale_panel_diagnostics(
    operand_scales,
    score_deltas_by_scale,
    *,
    precision_bits: int,
):
    """Classify strictly growing multiscale cancellation drift as NO-GO.

    This is an ordering test over raw observations, not an epsilon threshold.
    A non-growing panel remains unqualified until same-hardware repeats,
    discrete outcomes, final quality, and clean runtime all pass.
    """

    scales = np.asarray(operand_scales, dtype=np.float64).reshape(-1)
    deltas = np.asarray(score_deltas_by_scale, dtype=np.float64)
    if scales.size < 2 or deltas.ndim != 5 or deltas.shape[0] != scales.size:
        raise ValueError(
            "coarse GEMM scale diagnostics require matching scale and "
            "[scale,image,class,rotation,translation] arrays with at least two scales",
        )
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0.0) or np.any(np.diff(scales) <= 0.0):
        raise ValueError("coarse GEMM operand scales must be finite, positive, and increasing")
    if int(precision_bits) not in (32, 64):
        raise ValueError("coarse GEMM precision_bits must be 32 or 64")
    flattened = deltas.reshape(scales.size, -1)
    max_abs_by_scale = np.max(np.abs(flattened), axis=1)
    signed_mean_by_scale = np.mean(flattened, axis=1)
    growing_steps = max_abs_by_scale[1:] > max_abs_by_scale[:-1]
    scale_amplified = bool(np.any(growing_steps))
    return {
        "operand_scales": scales,
        "precision_bits": np.asarray(precision_bits, dtype=np.int64),
        "signed_mean_delta_by_scale": signed_mean_by_scale,
        "max_abs_delta_by_scale": max_abs_by_scale,
        "scale_growth_steps": growing_steps,
        "scale_amplified": np.asarray(scale_amplified),
        "qualification_status": np.asarray(
            (
                "NO_GO_scale-amplified_drift"
                if scale_amplified
                else "NO_GO_unqualified_non-growing_scale_panel"
            )
        ),
    }


def _coarse_gaussian_direct_macro_diagnostics(
    direct_scores,
    macro_scores,
    *,
    direct_support=None,
    macro_support=None,
):
    """Summarize paired direct-square and expanded-GEMM score surfaces.

    Inputs use the public coarse layout ``[image, class, rotation,
    translation]``.  This helper deliberately reports raw deltas, signed bias,
    winner margins, and exact discrete differences without defining a
    promotion tolerance.  Repeated H100 artifacts can therefore establish a
    native/repeat envelope without silently turning one observed delta into an
    acceptance threshold.
    """

    direct = np.asarray(direct_scores)
    macro = np.asarray(macro_scores)
    if direct.shape != macro.shape or direct.ndim != 4:
        raise ValueError(
            "paired coarse score diagnostics require equal "
            "[image, class, rotation, translation] arrays, got "
            f"{direct.shape} and {macro.shape}",
        )
    if direct.dtype != macro.dtype or direct.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError(
            "paired coarse score diagnostics require one shared float32 or "
            f"float64 dtype, got {direct.dtype} and {macro.dtype}",
        )
    direct_flat = direct.reshape(direct.shape[0], -1)
    macro_flat = macro.reshape(macro.shape[0], -1)
    delta = macro_flat.astype(np.float64) - direct_flat.astype(np.float64)
    absolute_delta = np.abs(delta)
    relative_delta_to_direct = np.full(delta.shape, np.nan, dtype=np.float64)
    np.divide(
        delta,
        np.abs(direct_flat.astype(np.float64)),
        out=relative_delta_to_direct,
        where=direct_flat != 0,
    )

    def ordered_float_bits(values):
        values = np.asarray(values).copy()
        values[values == 0] = 0
        if values.dtype == np.float32:
            unsigned = values.view(np.uint32)
            sign_mask = np.uint32(1 << 31)
            ordered = np.where(
                (unsigned & sign_mask) != 0,
                ~unsigned,
                unsigned ^ sign_mask,
            ).astype(np.uint64)
        else:
            unsigned = values.view(np.uint64)
            sign_mask = np.uint64(1 << 63)
            ordered = np.where(
                (unsigned & sign_mask) != 0,
                ~unsigned,
                unsigned ^ sign_mask,
            )
        return ordered

    direct_ordered = ordered_float_bits(direct_flat)
    macro_ordered = ordered_float_bits(macro_flat)
    ulp_delta = np.maximum(direct_ordered, macro_ordered) - np.minimum(
        direct_ordered,
        macro_ordered,
    )
    nonfinite = ~np.isfinite(direct_flat) | ~np.isfinite(macro_flat)
    ulp_delta = np.where(nonfinite, np.iinfo(np.uint64).max, ulp_delta)

    def winner_and_margin(values):
        winner = np.argmax(values, axis=1).astype(np.int64)
        if values.shape[1] == 1:
            margin = np.full(values.shape[0], np.inf, dtype=np.float64)
        else:
            largest_two = np.partition(values.astype(np.float64), -2, axis=1)[:, -2:]
            margin = largest_two[:, 1] - largest_two[:, 0]
        return winner, margin

    direct_winner, direct_margin = winner_and_margin(direct_flat)
    macro_winner, macro_margin = winner_and_margin(macro_flat)
    diagnostics = {
        "score_delta": delta.reshape(direct.shape),
        "absolute_score_delta": absolute_delta.reshape(direct.shape),
        "relative_score_delta_to_direct": relative_delta_to_direct.reshape(direct.shape),
        "ulp_score_delta": ulp_delta.reshape(direct.shape),
        "score_precision_bits": np.asarray(direct.dtype.itemsize * 8, dtype=np.int64),
        "direct_max_abs_score_per_image": np.max(
            np.abs(direct_flat.astype(np.float64)),
            axis=1,
        ),
        "macro_max_abs_score_per_image": np.max(
            np.abs(macro_flat.astype(np.float64)),
            axis=1,
        ),
        "signed_mean_delta_per_image": np.mean(delta, axis=1),
        "rms_delta_per_image": np.sqrt(np.mean(delta * delta, axis=1)),
        "max_abs_delta_per_image": np.max(np.abs(delta), axis=1),
        "positive_delta_count_per_image": np.count_nonzero(delta > 0.0, axis=1),
        "negative_delta_count_per_image": np.count_nonzero(delta < 0.0, axis=1),
        "zero_delta_count_per_image": np.count_nonzero(delta == 0.0, axis=1),
        "exact_zero_direct_nonzero_macro_per_image": np.all(
            direct_flat == 0.0,
            axis=1,
        )
        & np.any(macro_flat != 0.0, axis=1),
        "direct_argmax": direct_winner,
        "macro_argmax": macro_winner,
        "argmax_equal": direct_winner == macro_winner,
        "direct_winner_margin": direct_margin,
        "macro_winner_margin": macro_margin,
    }
    if (direct_support is None) != (macro_support is None):
        raise ValueError("direct_support and macro_support must be supplied together")
    if direct_support is not None:
        direct_support = np.asarray(direct_support, dtype=bool)
        macro_support = np.asarray(macro_support, dtype=bool)
        if direct_support.shape != direct_flat.shape or macro_support.shape != direct_flat.shape:
            raise ValueError(
                "coarse support diagnostics must match flattened score surfaces, got "
                f"{direct_support.shape}, {macro_support.shape}, and {direct_flat.shape}",
            )
        diagnostics.update(
            direct_support=direct_support.reshape(direct.shape),
            macro_support=macro_support.reshape(direct.shape),
            support_equal=np.all(direct_support == macro_support, axis=1),
            support_symmetric_difference_count=np.count_nonzero(
                direct_support != macro_support,
                axis=1,
            ).astype(np.int64),
        )
    return diagnostics
