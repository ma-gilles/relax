"""Host-only coarse-score diagnostics and qualification reporting.

Keep NumPy evidence summaries separate from JAX score/M-step kernels.
These reports do not establish scientific or performance acceptance by themselves.
"""

import hashlib
import operator

import numpy as np

from relax.scoring.significant_samples import significant_sample_ids


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


def _coarse_gaussian_qualification_decision(
    *,
    exact_arithmetic_equivalent: bool,
    repeat_stable: bool | None,
    unbiased_non_directional: bool | None,
    bounded_non_growing: bool | None,
    discrete_choices_equal: bool | None,
    final_basin_quality_equal: bool | None,
    material_runtime_win: bool | None,
    scale_amplified: bool | None,
    negative_implied_diff2: bool,
    nonfinite_scores: bool,
    exact_zero_cancellation_drift: bool,
):
    """Apply the coarse GEMM promotion policy without a numeric tolerance.

    Mathematically equivalent reduction-order noise may pass without bitwise
    score equality only after independent evidence establishes repeat
    stability, non-directional bias, bounded/non-growing scale and iteration
    behavior, exact discrete decisions, unchanged final basin/quality, and a
    material clean-run speedup.  Cancellation evidence, non-finite scores, or
    a negative implied diff2 is an unconditional NO-GO.  ``None`` means the
    corresponding empirical gate has not run and therefore cannot promote.
    """

    failures = []
    pending = []
    if not exact_arithmetic_equivalent:
        failures.append("not_exact-arithmetic-equivalent")
    if scale_amplified is True:
        failures.append("scale-amplified_drift")
    elif scale_amplified is None:
        pending.append("multiscale_growth")
    if negative_implied_diff2:
        failures.append("negative_implied_diff2")
    if nonfinite_scores:
        failures.append("nonfinite_scores")
    if exact_zero_cancellation_drift:
        failures.append("exact-zero_cancellation_drift")
    for name, value in (
        ("repeat_stability", repeat_stable),
        ("unbiased_non-directional", unbiased_non_directional),
        ("bounded_non-growing", bounded_non_growing),
        ("discrete_choices", discrete_choices_equal),
        ("final_basin_quality", final_basin_quality_equal),
        ("material_runtime_win", material_runtime_win),
    ):
        if value is None:
            pending.append(name)
        elif not value:
            failures.append(name)
    if failures:
        status = "NO_GO"
    elif pending:
        status = "NO_GO_UNQUALIFIED"
    else:
        status = "GO_STABLE_BOUNDED_MATHEMATICALLY_EQUIVALENT_NOISE"
    return {
        "status": status,
        "failure_reasons": tuple(failures),
        "pending_gates": tuple(pending),
        "requires_bitwise_score_identity": False,
        "requires_exact_discrete_identity": True,
    }


def _build_coarse_significance_support_audit(
    significant_sample_indices,
    *,
    samples_per_class: int,
    include_ids: bool = False,
) -> dict:
    """Hash every ordered class/image coarse support without changing it.

    The canonical byte stream for one row is four little-endian int64 header
    values ``(class, image, samples_per_class, selected_count)`` followed by
    the strictly increasing selected sample IDs as little-endian int64. Each
    length-prefixed row enters the aggregate digest in class-major/image-major
    order. Per-row digests and counts make any mismatch localizable while the
    aggregate digest provides a compact direct/hybrid equality gate.
    """

    try:
        total_size = operator.index(samples_per_class)
    except TypeError as error:
        raise ValueError("samples_per_class must be an integer") from error
    if total_size <= 0:
        raise ValueError("samples_per_class must be positive")
    if not isinstance(significant_sample_indices, (tuple, list)) or not significant_sample_indices:
        raise ValueError("support audit requires at least one class")
    n_images = None
    aggregate = hashlib.sha256()
    per_class_image_sha256: list[list[str]] = []
    per_class_counts: list[list[int]] = []
    per_class_ids: list[list[list[int]]] = []
    for class_index, rows in enumerate(significant_sample_indices):
        if not isinstance(rows, (tuple, list)):
            raise TypeError("support audit class rows must be a sequence")
        if n_images is None:
            n_images = len(rows)
            if n_images <= 0:
                raise ValueError("support audit requires at least one image")
        elif len(rows) != n_images:
            raise ValueError("support audit classes must cover the same images")
        row_digests = []
        row_counts = []
        row_ids: list[list[int]] = []
        for image_index, samples in enumerate(rows):
            ids = np.asarray(
                significant_sample_ids(samples, total_size),
                dtype=np.int64,
            ).reshape(-1)
            if (
                np.any(ids < 0)
                or np.any(ids >= total_size)
                or (ids.size > 1 and np.any(np.diff(ids) <= 0))
            ):
                raise ValueError(
                    "support audit requires unique, strictly increasing in-range IDs",
                )
            header = np.asarray(
                (class_index, image_index, total_size, ids.size),
                dtype="<i8",
            )
            ids_le = np.ascontiguousarray(ids.astype("<i8", copy=False))
            row_bytes = header.tobytes(order="C") + ids_le.tobytes(order="C")
            row_digests.append(hashlib.sha256(row_bytes).hexdigest())
            row_counts.append(int(ids.size))
            if include_ids:
                row_ids.append([int(value) for value in ids])
            aggregate.update(
                np.asarray((len(row_bytes),), dtype="<u8").tobytes(order="C"),
            )
            aggregate.update(row_bytes)
        per_class_image_sha256.append(row_digests)
        per_class_counts.append(row_counts)
        if include_ids:
            per_class_ids.append(row_ids)

    counts = np.ascontiguousarray(np.asarray(per_class_counts, dtype="<i8"))
    result = {
        "schema": "recovar.coarse_significance_support_audit.v2",
        "classification": "diagnostic_only",
        "canonical_encoding": (
            "class-major/image-major; uint64 row-byte-length; "
            "int64-le header(class,image,total,count); int64-le sorted IDs"
        ),
        "n_classes": len(per_class_counts),
        "n_images": int(n_images),
        "samples_per_class": total_size,
        "selected_count_sum": int(np.sum(counts, dtype=np.int64)),
        "selected_count_min": int(np.min(counts)),
        "selected_count_max": int(np.max(counts)),
        "per_class_image_selected_counts": per_class_counts,
        "per_class_image_selected_counts_sha256": hashlib.sha256(
            counts.tobytes(order="C"),
        ).hexdigest(),
        "support_ids_included": bool(include_ids),
        "per_class_image_support_sha256": per_class_image_sha256,
        "aggregate_support_sha256": aggregate.hexdigest(),
    }
    if include_ids:
        # Explicit opt-in avoids retaining a potentially enormous full-support
        # diagnostic in other geometries. GF46 iteration 181 has only ~5,900
        # selected IDs across all 1,000 images.
        result["per_class_image_support_ids"] = per_class_ids
    return result


def _with_coarse_significance_diagnostics(
    result,
    *,
    support_audit: dict | None,
    exact_coarse_operand_assembly: dict | None = None,
):
    """Propagate exact coarse-support telemetry to InitialModel."""

    additions = {
        key: dict(value)
        for key, value in (
            ("coarse_significance_support_audit", support_audit),
            (
                "exact_coarse_operand_assembly",
                exact_coarse_operand_assembly,
            ),
        )
        if value is not None
    }
    if not additions:
        return result
    profile_summary = dict(result.profile_summary or {})
    profile_summary.update(additions)
    return result._replace(profile_summary=profile_summary)


