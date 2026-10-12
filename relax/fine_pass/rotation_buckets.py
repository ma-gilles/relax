"""The padded rotation-row counts of exact local and pass-2 buckets: compile-friendly sizes for a per-image support.

Local search (``relax.local_search.layout``) and pass 2's bucket assembly (``relax.fine_pass.bucket_arrays``) both
pad each image's rotation support to these sizes, so nearby support sizes share compiled programs.
"""

from __future__ import annotations

import os

from relax.runtime.shape_buckets import coarse_bucket, power_bucket

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
