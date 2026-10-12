"""Padded shape classes of the bucketed pass-2 chain and the local bucket planner.

Script library: moved from relax/runtime/shape_buckets.py, relax/fine_pass/rotation_buckets.py and
relax/runtime/batch_planning.py, where no relax command reached them; their users are
``scripts/lib/bucketed_pass2.py``, ``scripts/lib/local_bucket_layout.py`` and tests. The padded rotation-row
counts are compile-friendly sizes for a per-image support, so nearby support sizes share compiled programs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from relax.runtime.shape_buckets import round_up_to_multiple


def power_bucket(
    value: int,
    *,
    base: int = 2,
    minimum: int = 1,
    maximum: int | None = None,
) -> int:
    """Return a power-of-``base`` padded bucket for low-cardinality shapes."""

    if base < 2:
        raise ValueError(f"base must be at least 2, got {base}")
    if value <= 0:
        return max(1, minimum)
    target = max(value, minimum)
    bucket = 1
    while bucket < target:
        bucket *= base
    if maximum is not None:
        bucket = min(bucket, int(maximum))
    return max(bucket, value, minimum)


def power_of_two_bucket(value: int, *, minimum: int = 1, maximum: int | None = None) -> int:
    """Return a power-of-two padded bucket for low-cardinality shape classes."""

    return power_bucket(value, base=2, minimum=minimum, maximum=maximum)


def coarse_bucket(value: int, *, small_power2_max: int, large_multiple: int, minimum: int = 1) -> int:
    """Bucket small values by power-of-two and large values by coarse multiples."""

    if value <= int(small_power2_max):
        return power_of_two_bucket(value, minimum=minimum, maximum=small_power2_max)
    return round_up_to_multiple(value, large_multiple)


EXACT_LOCAL_BUCKET_RADIX_ENV = "RELAX_EXACT_LOCAL_BUCKET_RADIX"


def _resolve_exact_local_bucket_radix(explicit: int | None = None) -> int:
    """Resolve and validate the exact-local small-bucket radix."""

    source = "exact_local_bucket_radix"
    raw_value = explicit
    if raw_value is None:
        source = EXACT_LOCAL_BUCKET_RADIX_ENV
        raw_value = os.environ.get(EXACT_LOCAL_BUCKET_RADIX_ENV, "2")
    try:
        bucket_radix = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} must be an integer at least 2") from exc
    if bucket_radix < 2:
        raise ValueError(f"{source} must be at least 2")
    return bucket_radix


def exact_bucket_rotation_size(
    local_rotation_count: int,
    rotation_block_size: int,
    *,
    large_bucket_quantum: int | None = None,
    exact_local_bucket_radix: int | None = None,
) -> int:
    """Return a compile-friendly padded size for one exact local neighborhood.

    The exact local engine cannot safely cap the bucket size below the true
    per-image neighborhood cardinality. Use power-of-two style padding for
    smaller neighborhoods. For larger exact neighborhoods, round up to a
    coarse fixed quantum so nearby local-support
    sizes reuse the same compiled shapes instead of each exact count generating
    its own XLA program.
    """
    if local_rotation_count <= 0:
        return 1
    engine_cap = int(_local_search_engine_rotation_block_size(rotation_block_size))
    if local_rotation_count <= engine_cap:
        bucket_radix = _resolve_exact_local_bucket_radix(exact_local_bucket_radix)
        if bucket_radix != 2:
            return int(
                power_bucket(
                    local_rotation_count,
                    base=bucket_radix,
                    minimum=16,
                    maximum=engine_cap,
                ),
            )
        return int(
            coarse_bucket(
                local_rotation_count,
                small_power2_max=engine_cap,
                large_multiple=engine_cap,
                minimum=16,
            ),
        )
    # ``RELAX_LOCAL_BUCKET_QUANTUM`` lets callers override the large-bucket
    # quantization. The default is deliberately coarser than the exact-local
    # engine cap: outlier-heavy/local-search tails otherwise generate hundreds
    # of near-duplicate XLA shapes. Hypothesis/tile caps still chunk each
    # bucket, so this changes padding/shape reuse rather than the candidate set.
    env_quantum = os.environ.get("RELAX_LOCAL_BUCKET_QUANTUM", "")
    if env_quantum:
        large_bucket_quantum = max(1, int(env_quantum))
    elif large_bucket_quantum is None:
        large_bucket_quantum = max(4096, engine_cap)
    else:
        large_bucket_quantum = max(1, int(large_bucket_quantum))
    return int(
        coarse_bucket(
            local_rotation_count,
            small_power2_max=engine_cap,
            large_multiple=large_bucket_quantum,
            minimum=16,
        ),
    )


def _local_search_engine_rotation_block_size(rotation_block_size: int) -> int:
    """Cap the exact local-search engine block size.

    Local search already reduces the candidate set per image from the full
    HEALPix grid down to a few thousand rotations. Reusing the dense-search
    5k rotation tile size here creates oversized XLA kernels whose compile
    time dominates the first local-search iteration. A 1k cap keeps the
    candidate set exact while making the compiled score kernels much smaller.
    """
    return int(max(64, min(int(rotation_block_size), 1024)))


@dataclass(frozen=True)
class _ConsecutivePaddedBatch:
    """One physical-order batch and its static padded shape."""

    item_indices: np.ndarray
    padded_size: int
    padded_item_capacity: int


def plan_consecutive_padded_batches(
    padded_sizes,
    *,
    processing_order=None,
    target_items_per_batch: int,
    max_items_per_batch: int,
    max_padded_values_per_batch: int,
    values_per_padded_size: int = 1,
    item_alignment: int = 1,
) -> list[_ConsecutivePaddedBatch]:
    """Plan bounded mixed-size batches without changing physical item order.

    Each batch uses the largest padded size among its consecutive items.  The
    planner greedily admits aligned item groups until either the target item
    count or padded-work cap would be exceeded.  ``padded_item_capacity`` is
    the stable image-axis shape for a tail batch with the same padded size.

    This is the shared shape policy used around the mature sparse EM pass and
    InitialModel's exact-local pass.  It changes padding and executable reuse,
    never candidate membership or processing order.
    """

    padded_sizes = np.asarray(padded_sizes, dtype=np.int64).reshape(-1)
    n_items = int(padded_sizes.size)
    if np.any(padded_sizes <= 0):
        raise ValueError("padded_sizes must contain only positive values")
    if processing_order is None:
        processing_order = np.arange(n_items, dtype=np.int64)
    else:
        processing_order = np.asarray(processing_order, dtype=np.int64).reshape(-1)
    if processing_order.shape != (n_items,):
        raise ValueError(
            f"processing_order must have shape ({n_items},), got {processing_order.shape}",
        )
    if not np.array_equal(np.sort(processing_order), np.arange(n_items, dtype=np.int64)):
        raise ValueError("processing_order must be a permutation of item indices")

    if target_items_per_batch <= 0 or max_items_per_batch <= 0:
        raise ValueError("batch item limits must be positive")
    if max_padded_values_per_batch <= 0 or values_per_padded_size <= 0:
        raise ValueError("padded-work limits must be positive")
    if item_alignment <= 0:
        raise ValueError("item_alignment must be positive")
    if n_items == 0:
        return []

    item_limit = min(target_items_per_batch, max_items_per_batch)
    if item_alignment > 1 and n_items > item_limit and item_limit < item_alignment:
        raise ValueError(
            "batch item limits are too small to preserve the requested item alignment: "
            f"limit={item_limit}, alignment={item_alignment}",
        )
    effective_alignment = item_alignment
    plans: list[_ConsecutivePaddedBatch] = []
    start = 0
    while start < n_items:
        stop = start
        batch_padded_size = 0
        while stop < n_items:
            next_stop = min(n_items, stop + effective_alignment)
            next_count = next_stop - start
            next_padded_size = max(
                batch_padded_size,
                int(np.max(padded_sizes[processing_order[stop:next_stop]], initial=1)),
            )
            next_work = next_count * next_padded_size * values_per_padded_size
            if stop == start and next_work > max_padded_values_per_batch and effective_alignment > 1:
                raise ValueError(
                    "padded-work cap is too small for one aligned item group: "
                    f"required={next_work}, cap={max_padded_values_per_batch}, "
                    f"alignment={effective_alignment}",
                )
            if stop > start and (
                next_count > item_limit
                or next_work > max_padded_values_per_batch
            ):
                break
            stop = next_stop
            batch_padded_size = next_padded_size
            if next_count >= item_limit:
                break

        item_indices = np.asarray(processing_order[start:stop], dtype=np.int64)
        capacity_by_work = max(
            1,
            max_padded_values_per_batch
            // max(1, batch_padded_size * values_per_padded_size),
        )
        padded_item_capacity = min(item_limit, capacity_by_work)
        if padded_item_capacity >= effective_alignment:
            padded_item_capacity = (
                padded_item_capacity // effective_alignment
            ) * effective_alignment
        padded_item_capacity = max(int(item_indices.size), padded_item_capacity)
        plans.append(
            _ConsecutivePaddedBatch(
                item_indices=item_indices,
                padded_size=int(batch_padded_size),
                padded_item_capacity=int(padded_item_capacity),
            )
        )
        start = stop

    return plans
