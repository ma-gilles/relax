"""Host-only coarse-significance support audit.

Keep NumPy evidence summaries separate from JAX score/M-step kernels.
These reports do not establish scientific or performance acceptance by themselves.
"""

import hashlib
import operator

import numpy as np

from relax.scoring.significant_samples import significant_sample_ids


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


