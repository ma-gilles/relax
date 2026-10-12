"""Padded static-shape buckets over a ``LocalHypothesisLayout``: plans and materialized bucket arrays.

Script library: moved from relax/local_search/layout.py, where no relax command reached it; its users are
``scripts/lib/local_dataset.py`` and tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from relax.fine_pass.rotation_buckets import _local_search_engine_rotation_block_size, exact_bucket_rotation_size
from relax.local_search.layout import LocalHypothesisLayout
from relax.runtime.batch_planning import plan_consecutive_padded_batches

# Global bucket-size unification pads every image to the halfset's largest local
# neighborhood. That buys one compiled shape per layout, which is worth a lot while
# the subsets are small, and costs a lot once the subset is the whole dataset.
#
# Measured on K=4 100k/256 InitialModel (H100, seed 29, exclusive nodes,
# em_work/codex/vdam_k4_unify_20260918): at the final all-data iteration unification
# turned 54.1 M real padded rows into 204.8 M and pass 2 took 799.7 s instead of
# 384.0 s. Over the cheap subset iterations the same setting was worth ~250 s the
# other way, because each extra bucket shape is another XLA program. Unifying only
# while the padding it adds stays bounded keeps both ends.
#
# The bound is on rows the unification would add. 15.4 M was neutral in that run and
# 150.7 M was a large loss, so the default sits between them with margin.
EXACT_LOCAL_UNIFY_MAX_PADDED_ROWS = 32_000_000


EXACT_LOCAL_BUCKET_MIN_QUANTUM = 256


DEFAULT_LOCAL_IMAGE_CAPACITY_LADDER = (16, 32, 64, 128, 256)


def resolve_local_image_capacity_ladder(explicit=None) -> tuple[int, ...]:
    """Resolve the opt-in image-axis capacity ladder; ``()`` means off.

    The exact local engine's images-per-bucket capacity is
    ``min(image_batch_size, max_hypotheses_per_microbatch // bucket_rotations)``.
    ``image_batch_size`` comes from a memory estimate that moves by a few images
    between iterations, and it is the leading axis of every per-bucket program,
    so a one-image change recompiles the whole bucket program set. Snapping the
    capacity to a fixed ladder makes that axis stable across iterations.

    Accepted values: ``None``, ``False``/``""``/``"0"``/``"off"`` (ladder off),
    ``True``/``"1"``/``"on"``/``"auto"`` (the default ladder) or an explicit
    sequence or comma-separated string of positive capacities.
    """

    source = "local_image_capacity_ladder"
    raw = explicit
    if raw is None or raw is False:
        return ()
    if raw is True:
        return DEFAULT_LOCAL_IMAGE_CAPACITY_LADDER
    if isinstance(raw, str):
        token = raw.strip().lower()
        if token in {"", "0", "off", "false", "no", "none"}:
            return ()
        if token in {"1", "on", "true", "yes", "auto", "default"}:
            return DEFAULT_LOCAL_IMAGE_CAPACITY_LADDER
        raw = [part for part in token.replace(" ", "").split(",") if part]
    try:
        ladder = tuple(sorted({int(value) for value in raw}))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{source} must be a comma-separated list of positive image capacities"
        ) from exc
    if not ladder:
        return ()
    if ladder[0] < 1:
        raise ValueError(f"{source} capacities must be positive")
    return ladder


def _ladder_image_capacity(max_images: int, ladder: tuple[int, ...]) -> int:
    """Snap an images-per-bucket capacity DOWN to the ladder.

    Snapping down, never up: ``max_images`` is already the planner's memory
    bound (the ``image_batch_size`` estimate and the hypothesis cap), so a
    larger capacity would plan a bucket the caller's own estimate refused. A
    capacity below the ladder's minimum is left alone for the same reason --
    there is no smaller ladder rung to fall back to that keeps the bucket
    non-empty, and raising it would break the bound.
    """

    if not ladder:
        return max_images
    rungs = [rung for rung in ladder if rung <= max_images]
    if not rungs:
        return max_images
    return int(rungs[-1])


def _exact_local_large_bucket_quantum(rotation_block_size: int, explicit: int | None = None) -> int:
    """Return the large-neighborhood bucket quantum for exact local search."""

    if explicit is not None:
        return max(1, int(explicit))
    engine_cap = int(_local_search_engine_rotation_block_size(rotation_block_size))
    return max(EXACT_LOCAL_BUCKET_MIN_QUANTUM, engine_cap)


@dataclass(frozen=True)
class LocalBucketSpec:
    """Static-shape padded execution batch for the exact local engine."""

    image_indices: np.ndarray
    bucket_image_count: int
    bucket_rotation_count: int
    actual_rotation_counts: np.ndarray
    local_rotation_ids: np.ndarray
    local_rotations: np.ndarray
    local_rotation_log_prior: np.ndarray
    local_rotation_mask: np.ndarray
    translation_log_prior: np.ndarray
    local_rotation_posterior_ids: np.ndarray | None = None
    local_sample_mask: np.ndarray | None = None
    local_mstep_rotations: np.ndarray | None = None
    local_source_eulers: np.ndarray | None = None


@dataclass(frozen=True)
class LocalBucketPlan:
    """Image order and padded capacities, without per-candidate array storage."""

    image_indices: np.ndarray
    bucket_image_count: int
    bucket_rotation_count: int
    actual_rotation_counts: np.ndarray


def plan_local_hypothesis_buckets(
    layout: LocalHypothesisLayout,
    image_batch_size: int,
    rotation_block_size: int,
    *,
    max_hypotheses_per_microbatch: int = 32768,
    unify_bucket_sizes: bool | None = None,
    large_bucket_quantum: int | None = None,
    preserve_image_order: bool = False,
    exact_local_bucket_radix: int | None = None,
    consecutive_mixed_bucket_size: int | None = None,
    image_capacity_ladder=None,
) -> list[LocalBucketPlan]:
    """Plan static bucket shapes and image order without allocating candidate arrays."""

    image_batch_size = int(max(1, image_batch_size))
    max_hypotheses_per_microbatch = int(max(1, max_hypotheses_per_microbatch))
    image_capacity_ladder = resolve_local_image_capacity_ladder(image_capacity_ladder)
    rotations_dtype = np.asarray(layout.rotations_flat).dtype
    mstep_rotations_flat = (
        np.asarray(layout.rotations_flat, dtype=rotations_dtype)
        if layout.mstep_rotations_flat is None
        else np.asarray(layout.mstep_rotations_flat)
    )
    if mstep_rotations_flat.shape != np.asarray(layout.rotations_flat).shape:
        raise ValueError(
            "mstep_rotations_flat must match rotations_flat shape: "
            f"{mstep_rotations_flat.shape} vs {np.asarray(layout.rotations_flat).shape}",
        )
    resolved_large_bucket_quantum = _exact_local_large_bucket_quantum(rotation_block_size, large_bucket_quantum)
    bucket_sizes = np.asarray(
        [
            exact_bucket_rotation_size(
                int(count),
                rotation_block_size,
                large_bucket_quantum=resolved_large_bucket_quantum,
                exact_local_bucket_radix=exact_local_bucket_radix,
            )
            for count in layout.rotation_counts
        ],
        dtype=np.int32,
    )
    # ``RELAX_LOCAL_BUCKET_UNIFY=1`` forces all images to share a single
    # rotation-count bucket class so the JIT only compiles one shape per
    # layout. At 50k/256 K=1 this collapses ~13 unique shapes (per-image
    # significant rotation counts vary widely across iters) into 1, removing
    # per-bucket JIT compilation overhead. Memory cost: smaller-significance
    # images carry extra rotation padding.
    if unify_bucket_sizes is None:
        unify_bucket_sizes = os.environ.get("RELAX_LOCAL_BUCKET_UNIFY", "").lower() in {"1", "true", "yes", "on"}
    if consecutive_mixed_bucket_size is not None:
        if consecutive_mixed_bucket_size <= 0:
            raise ValueError("consecutive_mixed_bucket_size must be positive")
        if not preserve_image_order:
            raise ValueError("consecutive mixed buckets require preserved image order")
        if bool(unify_bucket_sizes):
            raise ValueError("consecutive mixed buckets cannot use run-global bucket unification")
    if bucket_sizes.size and bool(unify_bucket_sizes):
        unified_size = int(bucket_sizes.max())
        added_rows = unified_size * int(bucket_sizes.size) - int(bucket_sizes.sum(dtype=np.int64))
        # Past this bound the rows unification would add cost more than the extra
        # compiled shapes it avoids, and each image keeps its own size.
        if added_rows <= EXACT_LOCAL_UNIFY_MAX_PADDED_ROWS:
            bucket_sizes = np.full_like(bucket_sizes, unified_size)
    processing_order = (
        np.arange(layout.n_images, dtype=np.int32)
        if preserve_image_order
        else np.lexsort((layout.rotation_counts, bucket_sizes)).astype(np.int32)
    )

    if processing_order.size == 0:
        return []

    planned_groups: list[tuple[np.ndarray, int, int]] = []
    if consecutive_mixed_bucket_size is not None:
        for plan in plan_consecutive_padded_batches(
            bucket_sizes,
            processing_order=processing_order,
            target_items_per_batch=consecutive_mixed_bucket_size,
            max_items_per_batch=image_batch_size,
            max_padded_values_per_batch=max_hypotheses_per_microbatch,
            item_alignment=3,
        ):
            planned_groups.append(
                (
                    np.asarray(plan.item_indices, dtype=np.int32),
                    int(plan.padded_size),
                    int(plan.padded_item_capacity),
                )
            )
    elif preserve_image_order:
        boundaries = np.flatnonzero(
            np.r_[True, bucket_sizes[processing_order][1:] != bucket_sizes[processing_order][:-1], True]
        )
        bucket_groups = [
            processing_order[start:stop] for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)
        ]
    else:
        bucket_groups = [
            processing_order[bucket_sizes[processing_order] == bucket_size]
            for bucket_size in np.unique(bucket_sizes[processing_order])
        ]
    if consecutive_mixed_bucket_size is None:
        for bucket_images in bucket_groups:
            bucket_size = int(bucket_sizes[int(bucket_images[0])])
            max_images = max(1, min(image_batch_size, max_hypotheses_per_microbatch // int(bucket_size)))
            if preserve_image_order and max_images >= 3:
                # RELION's InitialModel default processes pools of three particles.
                # Keep static-shape FFI boundaries on pool boundaries so a new call
                # never changes which physical particles may update BPref together.
                max_images = max(3, (max_images // 3) * 3)
            else:
                # Only when the pool rule is not in force: ladder rungs are not
                # multiples of three, so snapping here would move an InitialModel
                # FFI boundary off a particle pool.
                max_images = _ladder_image_capacity(max_images, image_capacity_ladder)
            # Every group of this rotation class keeps the SAME capacity, the
            # remainder group included. Giving the remainder its own smaller rung
            # saves a little padding and costs a whole extra compiled program per
            # rotation class, which measured 7.96 s -> 16.26 s of local-engine
            # compile at the 10k/256 order-4 state.
            for start in range(0, bucket_images.shape[0], max_images):
                planned_groups.append(
                    (
                        np.asarray(bucket_images[start : start + max_images], dtype=np.int32),
                        bucket_size,
                        max_images,
                    )
                )

    return [
        LocalBucketPlan(indices, max_images, size, layout.rotation_counts[indices].astype(np.int32, copy=False))
        for indices, size, max_images in planned_groups
    ]


def _materialize_local_bucket(layout: LocalHypothesisLayout, plan: LocalBucketPlan) -> LocalBucketSpec:
    """Construct one bucket with the layout dtypes and planned physical order."""

    image_indices = plan.image_indices
    bucket_size = plan.bucket_rotation_count
    max_images = plan.bucket_image_count
    rotations_dtype = np.asarray(layout.rotations_flat).dtype
    mstep_rotations_flat = (
        np.asarray(layout.rotations_flat, dtype=rotations_dtype)
        if layout.mstep_rotations_flat is None
        else np.asarray(layout.mstep_rotations_flat)
    )
    actual_counts = plan.actual_rotation_counts
    batch_size = int(image_indices.shape[0])
    padded_rotations = np.broadcast_to(
        np.eye(3, dtype=rotations_dtype),
        (batch_size, int(bucket_size), 3, 3),
    ).copy()
    padded_mstep_rotations = (
        None
        if layout.mstep_rotations_flat is None
        else np.broadcast_to(
            np.eye(3, dtype=mstep_rotations_flat.dtype),
            (batch_size, int(bucket_size), 3, 3),
        ).copy()
    )
    padded_rotation_ids = np.full((batch_size, int(bucket_size)), -1, dtype=np.int64)
    padded_log_prior = np.full(
        (batch_size, int(bucket_size)), -1e30, dtype=np.asarray(layout.rotation_log_priors_flat).dtype
    )
    padded_mask = np.zeros((batch_size, int(bucket_size)), dtype=bool)
    padded_posterior_ids = (
        None
        if layout.rotation_posterior_ids_flat is None
        else np.full((batch_size, int(bucket_size)), -1, dtype=np.int32)
    )
    padded_sample_mask = (
        None
        if layout.sample_mask_bits is None
        else np.zeros(
            (batch_size, int(bucket_size), int(layout.translation_grid.shape[0])),
            dtype=bool,
        )
    )

    padded_source_eulers = (
        None if layout.source_eulers_flat is None else np.zeros((batch_size, int(bucket_size), 3), dtype=np.float64)
    )

    for row, image_idx in enumerate(image_indices.tolist()):
        start_off = int(layout.rotation_offsets[image_idx])
        end_off = int(layout.rotation_offsets[image_idx + 1])
        count = end_off - start_off
        padded_rotations[row, :count] = layout.rotations_flat[start_off:end_off]
        if padded_source_eulers is not None:
            padded_source_eulers[row, :count] = layout.source_eulers_flat[start_off:end_off]
        if padded_mstep_rotations is not None:
            padded_mstep_rotations[row, :count] = mstep_rotations_flat[start_off:end_off]
        padded_rotation_ids[row, :count] = layout.rotation_ids_flat[start_off:end_off]
        padded_log_prior[row, :count] = layout.rotation_log_priors_flat[start_off:end_off]
        padded_mask[row, :count] = True
        if padded_posterior_ids is not None:
            padded_posterior_ids[row, :count] = layout.rotation_posterior_ids_flat[start_off:end_off]
        if padded_sample_mask is not None:
            padded_sample_mask[row, :count, :] = layout.sample_mask_rows(start_off, end_off)

    return LocalBucketSpec(
        image_indices=image_indices,
        bucket_image_count=int(max_images),
        bucket_rotation_count=int(bucket_size),
        actual_rotation_counts=actual_counts,
        local_rotation_ids=padded_rotation_ids,
        local_rotations=padded_rotations,
        local_source_eulers=padded_source_eulers,
        local_rotation_log_prior=padded_log_prior,
        local_rotation_mask=padded_mask,
        translation_log_prior=np.asarray(layout.translation_log_priors[image_indices]),
        local_mstep_rotations=padded_mstep_rotations,
        local_rotation_posterior_ids=padded_posterior_ids,
        local_sample_mask=padded_sample_mask,
    )


def bucket_local_hypothesis_layout(
    layout: LocalHypothesisLayout,
    image_batch_size: int,
    rotation_block_size: int,
    *,
    max_hypotheses_per_microbatch: int = 32768,
    unify_bucket_sizes: bool | None = None,
    large_bucket_quantum: int | None = None,
    preserve_image_order: bool = False,
    exact_local_bucket_radix: int | None = None,
    consecutive_mixed_bucket_size: int | None = None,
    image_capacity_ladder=None,
) -> list[LocalBucketSpec]:
    """Materialize all local buckets; use plans for bounded-memory iteration."""

    plans = plan_local_hypothesis_buckets(
        layout,
        image_batch_size,
        rotation_block_size,
        max_hypotheses_per_microbatch=max_hypotheses_per_microbatch,
        unify_bucket_sizes=unify_bucket_sizes,
        large_bucket_quantum=large_bucket_quantum,
        preserve_image_order=preserve_image_order,
        exact_local_bucket_radix=exact_local_bucket_radix,
        consecutive_mixed_bucket_size=consecutive_mixed_bucket_size,
        image_capacity_ladder=image_capacity_ladder,
    )
    return [_materialize_local_bucket(layout, plan) for plan in plans]
