"""The bucketed pass-2 chain of ``scripts/run_k_class_parity.py --relion-bpref-mstep``.

Groups each image's pass-2 support into padded buckets that share a rotation count, stacks the per-image rows into
bucket arrays, normalizes bucket scores with a full-grid log-Z and forms the M-step numerator and denominator sums.

Script library: moved from relax/fine_pass (bucket_arrays, bucket_io, local_backprojection, posterior), where no
relax command reached it; relax's pass 2 is the resident engine (relax/fine_pass/resident_pass2.py).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from relax.fine_pass.local_backprojection import (
    compute_local_ctf_sums_from_probs_sum_t,
    compute_local_weighted_sums,
    relion_x_half_sequential_translation_reduction_enabled,
)
from relax.runtime.env_flags import parse_env_binary_flag, parse_env_flag
from relax.runtime.shape_buckets import pow2_ceil, pow2_floor
from scripts.lib.bucket_shapes import exact_bucket_rotation_size, plan_consecutive_padded_batches, power_of_two_bucket

_LARGE_BUCKET_POW2_ENV = "RELAX_SPARSE_PASS2_LARGE_BUCKET_POW2"
_LARGE_BUCKET_POW2_THRESHOLD = 1024


def _pass2_bucket_rotation_size(count: int, rotation_block_size_for_quantization: int) -> int:
    """Padded rotation rows for one pass-2 image.

    The shared quantiser pads small supports to powers of two and large ones to
    multiples of a ~4096 quantum, which at HEALPix order 3 produced 17-31
    distinct large sizes per half (12288, 20480, 24576, 28672, ..., 217088);
    each distinct size compiles its own set of bucket programs.  With
    ``RELAX_SPARSE_PASS2_LARGE_BUCKET_POW2=1`` (default off, measurement
    knob) sizes above the engine cap are rounded up to a power of two instead,
    bounding the large sizes to about eight and costing at most 2x padding on
    those rows.  Padded rows carry zero posterior mass, so this changes shape
    reuse and atomic interleaving, not the candidate set.
    """

    size = int(exact_bucket_rotation_size(int(count), rotation_block_size_for_quantization))
    if size > _LARGE_BUCKET_POW2_THRESHOLD and parse_env_flag(_LARGE_BUCKET_POW2_ENV, default=False):
        return int(power_of_two_bucket(size))
    return size


_DEFAULT_MAX_HYPOTHESES_PER_MICROBATCH = 1_000_000
_DEFAULT_TAIL_BUCKET_COALESCE_MAX_INFLATION = 2.0
_DEFAULT_TAIL_BUCKET_COALESCE_MIN_BUCKET_SIZE = 4096


def _split_run_into_chunks(run, max_per_chunk):
    """Split one support-size run into consecutive bounded views."""
    cap = max(1, int(max_per_chunk))
    return [run[start:stop] for start, stop in bucket_chunk_bounds(len(run), cap)]


def _bucket_pass2_inputs(
    per_image_inputs,
    n_fine_trans,
    rotation_block_size_for_quantization=5000,
    max_hypotheses_per_microbatch=_DEFAULT_MAX_HYPOTHESES_PER_MICROBATCH,
    max_images_per_microbatch=2048,
    small_bucket_coalesce_size=None,
    tail_bucket_coalesce_max_images=None,
    tail_bucket_coalesce_max_inflation=None,
    tail_bucket_coalesce_min_bucket_size=None,
    processing_order_override=None,
    processing_order_chunk_size=1,
    processing_order_group_by_bucket_size=False,
    processing_order_batch_consecutive_bucket_sizes=False,
):
    """Group images into buckets that share a padded rotation count.

    Return bucket specifications: a padded rotation count and the image indices
    assigned to that bucket. Array builders materialize the selected rows later.

    To avoid OOM when one bucket is very large
    (``bucket_size * n_images_in_bucket * n_fine_trans`` is the (B, R, T)
    score tensor footprint), we split each per-quantization-size group
    into chunks of at most ``max_hypotheses_per_microbatch /
    (bucket_size * n_fine_trans)`` images.
    """
    n_images = len(per_image_inputs["oversampled_rots"])
    rotation_counts = np.array(
        [rots.shape[0] for rots in per_image_inputs["oversampled_rots"]],
        dtype=np.int64,
    )
    if n_images == 0:
        return []

    bucket_sizes = np.array(
        [_pass2_bucket_rotation_size(int(count), rotation_block_size_for_quantization) for count in rotation_counts],
        dtype=np.int64,
    )
    if small_bucket_coalesce_size is not None:
        bucket_sizes = _coalesce_small_bucket_sizes(bucket_sizes, small_bucket_coalesce_size)
    bucket_sizes = _coalesce_tail_bucket_sizes(
        bucket_sizes,
        max_images=tail_bucket_coalesce_max_images,
        max_inflation=tail_bucket_coalesce_max_inflation,
        min_bucket_size=tail_bucket_coalesce_min_bucket_size,
        max_images_per_microbatch=max_images_per_microbatch,
    )

    if processing_order_override is not None:
        processing_order = np.asarray(processing_order_override, dtype=np.int64).reshape(-1)
        if processing_order.shape != (n_images,):
            raise ValueError(
                "processing_order_override must have shape "
                f"({n_images},), got {processing_order.shape}",
            )
        if not np.array_equal(np.sort(processing_order), np.arange(n_images, dtype=np.int64)):
            raise ValueError("processing_order_override must be a permutation of image indices")
        if not processing_order_group_by_bucket_size:
            if processing_order_batch_consecutive_bucket_sizes:
                buckets = []
                run_start = 0
                while run_start < n_images:
                    run_bucket_size = int(bucket_sizes[processing_order[run_start]])
                    run_end = run_start + 1
                    while (
                        run_end < n_images
                        and int(bucket_sizes[processing_order[run_end]]) == run_bucket_size
                    ):
                        run_end += 1
                    cap_by_hypotheses = max(
                        1,
                        int(max_hypotheses_per_microbatch)
                        // max(1, run_bucket_size * int(n_fine_trans)),
                    )
                    max_per_chunk = max(
                        1,
                        min(int(max_images_per_microbatch), cap_by_hypotheses),
                    )
                    for chunk in _split_run_into_chunks(
                        processing_order[run_start:run_end],
                        max_per_chunk,
                    ):
                        buckets.append(
                            {
                                "bucket_size": run_bucket_size,
                                "image_indices": np.asarray(chunk, dtype=np.int64),
                            }
                        )
                    run_start = run_end
                return buckets
            plans = plan_consecutive_padded_batches(
                bucket_sizes,
                processing_order=processing_order,
                target_items_per_batch=int(processing_order_chunk_size),
                max_items_per_batch=int(max_images_per_microbatch),
                max_padded_values_per_batch=int(max_hypotheses_per_microbatch),
                values_per_padded_size=int(n_fine_trans),
            )
            return [
                {
                    "bucket_size": int(plan.padded_size),
                    "image_indices": np.asarray(plan.item_indices, dtype=np.int64),
                }
                for plan in plans
            ]
    else:
        # Group by bucket size, smaller buckets first, then by rotation count (RECOVAR's bucket order); the
        # stable sort keeps RELION order within each equal padded-size bucket.
        processing_order = np.lexsort((rotation_counts, bucket_sizes)).astype(np.int64)

    unique_bucket_sizes = np.unique(bucket_sizes[processing_order])
    buckets = []
    for bucket_size in unique_bucket_sizes.tolist():
        bucket_image_indices = processing_order[bucket_sizes[processing_order] == bucket_size]
        # Chunk by max_hypotheses_per_microbatch and max_images_per_microbatch
        cap_by_hypotheses = max(
            1,
            int(max_hypotheses_per_microbatch) // max(1, bucket_size * int(n_fine_trans)),
        )
        max_per_chunk = max(1, min(int(max_images_per_microbatch), cap_by_hypotheses))
        for chunk in _split_run_into_chunks(
            bucket_image_indices,
            max_per_chunk,
        ):
            buckets.append(
                {
                    "bucket_size": bucket_size,
                    "image_indices": np.asarray(chunk, dtype=np.int64),
                }
            )
    return buckets


def _coalesce_small_bucket_sizes(bucket_sizes, small_bucket_coalesce_size):
    bucket_sizes = np.asarray(bucket_sizes, dtype=np.int64)
    coalesce_size = int(small_bucket_coalesce_size)
    if coalesce_size <= 1:
        return bucket_sizes
    small_or_target_mask = bucket_sizes <= coalesce_size
    if np.unique(bucket_sizes[small_or_target_mask]).size <= 1:
        return bucket_sizes
    return np.where(bucket_sizes < coalesce_size, coalesce_size, bucket_sizes)


def _coalesce_tail_bucket_sizes(
    bucket_sizes,
    *,
    max_images,
    max_inflation,
    min_bucket_size,
    max_images_per_microbatch,
):
    """Merge tiny adjacent groups within image-count and padding-inflation caps.

    Bucket builders enforce class/translation hypothesis budgets when chunking
    the resulting groups. Coalescing does not change those execution budgets.
    """

    bucket_sizes = np.asarray(bucket_sizes, dtype=np.int64)
    if bucket_sizes.size == 0 or max_images is None:
        return bucket_sizes
    if max_images <= 1:
        return bucket_sizes
    max_inflation = (
        _DEFAULT_TAIL_BUCKET_COALESCE_MAX_INFLATION
        if max_inflation is None
        else float(max_inflation)
    )
    min_bucket_size = (
        _DEFAULT_TAIL_BUCKET_COALESCE_MIN_BUCKET_SIZE
        if min_bucket_size is None
        else int(min_bucket_size)
    )
    if max_inflation < 1.0:
        return bucket_sizes

    unique_sizes, inverse, counts = np.unique(
        bucket_sizes,
        return_inverse=True,
        return_counts=True,
    )
    if unique_sizes.size <= 1:
        return bucket_sizes

    assigned_sizes = unique_sizes.copy()
    group_count = unique_sizes.size
    i = 0
    while i < group_count:
        size_i = int(unique_sizes[i])
        count_i = int(counts[i])
        if size_i < min_bucket_size or count_i > max_images:
            i += 1
            continue

        best_j = i
        total_images = 0
        total_rows = 0
        for j in range(i, group_count):
            size_j = int(unique_sizes[j])
            count_j = int(counts[j])
            if size_j < min_bucket_size or count_j > max_images:
                break
            total_images += count_j
            total_rows += size_j * count_j
            if total_images > max_images or total_images > int(max_images_per_microbatch):
                break
            target_size = size_j
            padded_rows = target_size * total_images
            if total_rows <= 0:
                continue
            inflation = float(padded_rows) / float(total_rows)
            # The bucket builder chunks each coalesced size by
            # max_hypotheses_per_microbatch before execution.  Applying the
            # same cap here prevents adjacent tiny high-tail groups from
            # sharing one padded size even when every eventual chunk remains
            # within the score-tensor budget.
            if inflation <= max_inflation:
                best_j = j

        if best_j > i:
            assigned_sizes[i : best_j + 1] = unique_sizes[best_j]
            i = best_j + 1
        else:
            i += 1

    if np.array_equal(assigned_sizes, unique_sizes):
        return bucket_sizes
    return assigned_sizes[inverse]


ROTATIONS_BY_INDEX_ENV = "RELAX_SPARSE_KCLASS_ROTATIONS_BY_INDEX"
_ROTATION_TABLE_DEVICE_CACHE: dict = {}


def rotations_by_index_enabled() -> bool:
    """Gather the padded bucket rotations from a device-resident fine-grid table.

    With ``RELAX_SPARSE_KCLASS_BUCKET_ROTATIONS_DEVICE`` the host still
    concatenates every image's (rows, 3, 3) float32 rows per class-chunk and
    uploads them (13.6 s of device_put plus ~7 s of host concatenation in the
    100k/256 K=4 iteration 2, job 13834297).  When the per-image inputs carry
    the fine-grid table (paired RELION override), the rows are gathers of that
    table by ``oversampled_rot_indices``; this flag uploads the table once per
    distinct content and gathers on the device from the padded rotation index
    array the builder already forms.  Values are copied, so bit-identical.
    """

    return parse_env_binary_flag(ROTATIONS_BY_INDEX_ENV)


def _rotation_table_device(table, key):
    """Device copy of a fine-grid rotation table, keyed by content (bounded cache)."""

    import jax.numpy as jnp

    cached = _ROTATION_TABLE_DEVICE_CACHE.get(key)
    if cached is None:
        if len(_ROTATION_TABLE_DEVICE_CACHE) >= 8:
            _ROTATION_TABLE_DEVICE_CACHE.clear()
        cached = jnp.asarray(np.ascontiguousarray(table))
        _ROTATION_TABLE_DEVICE_CACHE[key] = cached
    return cached


def _padded_rotations_from_table_impl(table, rotation_indices, counts, fill, *, rows):
    import jax.numpy as jnp

    gathered = jnp.take(table, jnp.clip(rotation_indices, 0, table.shape[0] - 1), axis=0)
    valid = jnp.arange(rows, dtype=jnp.int32)[None, :] < counts[:, None]
    return jnp.where(valid[:, :, None, None], gathered, fill[None, None])


def padded_rotations_from_table_device(table, table_key, rotation_indices, counts, rows, fill):
    """``(images, rows, 3, 3)`` device rotations gathered from a resident table.

    ``rotation_indices`` is the host ``(images, rows)`` int64 padded index array
    (zeros beyond ``counts``); padded slots receive ``fill`` (identity).  Equals
    :func:`padded_rows_from_flat_device` on the concatenated per-image rows.
    """
    import jax
    import jax.numpy as jnp

    fn = padded_rotations_from_table_device.__dict__.get("_compiled")
    if fn is None:
        fn = jax.jit(_padded_rotations_from_table_impl, static_argnames=("rows",))
        padded_rotations_from_table_device.__dict__["_compiled"] = fn
    table_dev = _rotation_table_device(table, table_key)
    return fn(
        table_dev,
        jnp.asarray(np.asarray(rotation_indices, dtype=np.int32)),
        jnp.asarray(np.asarray(counts, dtype=np.int32)),
        jnp.asarray(np.asarray(fill, dtype=table.dtype)),
        rows=int(rows),
    )


def _flat_rows_quantum(n_rows: int) -> int:
    n_rows = max(1, int(n_rows))
    if n_rows <= 4096:
        return pow2_ceil(n_rows)
    return ((n_rows + 4095) // 4096) * 4096


def _padded_rows_from_flat_impl(flat, starts, counts, fill, *, rows):
    """``out[b, r] = flat[starts[b] + r]`` for ``r < counts[b]``, else ``fill``."""
    import jax.numpy as jnp

    slots = jnp.arange(int(rows), dtype=jnp.int32)[None, :]
    src = jnp.clip(starts[:, None] + slots, 0, int(flat.shape[0]) - 1)
    valid = slots < counts[:, None]
    gathered = jnp.take(flat, src, axis=0)
    return jnp.where(valid.reshape(valid.shape + (1,) * (flat.ndim - 1)), gathered, fill)


def padded_rows_from_flat_device(per_row_values, counts, rows, fill):
    """Stack per-image row blocks into a device ``(images, rows, ...)`` array.

    ``per_row_values`` is a sequence of ``(count_i, ...)`` host arrays (one per
    real image), ``counts`` the ``(images,)`` int vector including capacity rows
    (zero), ``fill`` the value for padding slots (identity for rotations). One
    upload of the concatenated real rows, padded to a size ladder so the jitted
    padder compiles once per ``(images, rows, ladder step)``.
    """
    import jax
    import jax.numpy as jnp

    counts = np.asarray(counts, dtype=np.int32)
    real = [np.asarray(v) for v in per_row_values]
    if real:
        flat = np.concatenate(real, axis=0)
    else:
        flat = np.zeros((0,) + np.asarray(fill).shape, dtype=np.asarray(fill).dtype)
    n_flat = int(flat.shape[0])
    n_pad = _flat_rows_quantum(n_flat)
    if n_pad > n_flat:
        tail = np.empty((n_pad - n_flat,) + flat.shape[1:], dtype=flat.dtype)
        tail[...] = fill
        flat = np.concatenate([flat, tail], axis=0)
    starts = (np.cumsum(counts, dtype=np.int64) - counts).astype(np.int32)
    fn = padded_rows_from_flat_device.__dict__.get("_compiled")
    if fn is None:
        fn = jax.jit(_padded_rows_from_flat_impl, static_argnames=("rows",))
        padded_rows_from_flat_device.__dict__["_compiled"] = fn
    return fn(
        jnp.asarray(flat),
        jnp.asarray(starts),
        jnp.asarray(counts),
        jnp.asarray(np.asarray(fill, dtype=flat.dtype)),
        rows=int(rows),
    )


def _build_bucket_arrays(
    bucket,
    per_image_inputs,
    n_fine_trans,
    *,
    include_dense_score_fields: bool = True,
    capacity_rows=None,
    device_rotations: bool = False,
):
    """Stack/pad per-image arrays into batched bucket tensors."""
    bucket_size = int(bucket["bucket_size"])
    image_indices = np.asarray(bucket["image_indices"], dtype=np.int64)
    n_real = int(image_indices.shape[0])
    batch = n_real if capacity_rows is None else max(n_real, int(capacity_rows))
    padded_image_indices = image_indices
    if batch > n_real:
        padded_image_indices = np.concatenate([image_indices, np.repeat(image_indices[-1:], batch - n_real)])

    # padded_rotations: identity-fill — projection of identity is harmless
    # because we mask via candidate_mask=False everywhere for padded rows.
    rotation_dtype = np.result_type(
        *(np.asarray(per_image_inputs["oversampled_rots"][int(image_idx)]).dtype for image_idx in image_indices)
    )
    padded_rotations = (
        None
        if device_rotations
        else np.broadcast_to(
            np.eye(3, dtype=rotation_dtype),
            (batch, bucket_size, 3, 3),
        ).copy()
    )
    separate_mstep_rotations = any(
        per_image_inputs["oversampled_mstep_rots"][int(image_idx)]
        is not per_image_inputs["oversampled_rots"][int(image_idx)]
        for image_idx in image_indices.tolist()
    )
    mstep_rotation_dtype = np.result_type(
        *(np.asarray(per_image_inputs["oversampled_mstep_rots"][int(image_idx)]).dtype for image_idx in image_indices)
    )
    padded_mstep_rotations = (
        (
            None
            if device_rotations
            else np.broadcast_to(
                np.eye(3, dtype=mstep_rotation_dtype),
                (batch, bucket_size, 3, 3),
            ).copy()
        )
        if separate_mstep_rotations
        else padded_rotations
    )
    padded_log_prior = np.full(
        (batch, bucket_size),
        -1e30,
        dtype=np.result_type(
            *(np.asarray(per_image_inputs["log_prior"][int(image_idx)]).dtype for image_idx in image_indices)
        ),
    )
    padded_candidate_mask = (
        np.zeros((batch, bucket_size, n_fine_trans), dtype=bool) if include_dense_score_fields else None
    )
    padded_parent_map = np.full((batch, bucket_size), -1, dtype=np.int32) if include_dense_score_fields else None
    padded_rotation_indices = np.zeros((batch, bucket_size), dtype=np.int64)
    actual_counts = np.zeros(batch, dtype=np.int32)
    for row, image_idx in enumerate(image_indices.tolist()):
        rots = per_image_inputs["oversampled_rots"][image_idx]
        cnt = int(rots.shape[0])
        actual_counts[row] = cnt
        if not device_rotations:
            padded_rotations[row, :cnt] = rots
            if separate_mstep_rotations:
                padded_mstep_rotations[row, :cnt] = per_image_inputs["oversampled_mstep_rots"][image_idx]
        padded_log_prior[row, :cnt] = per_image_inputs["log_prior"][image_idx]
        if include_dense_score_fields:
            padded_candidate_mask[row, :cnt, :] = np.asarray(per_image_inputs["candidate_mask"][image_idx], dtype=bool)
            padded_parent_map[row, :cnt] = per_image_inputs["parent_map"][image_idx]
        padded_rotation_indices[row, :cnt] = per_image_inputs["oversampled_rot_indices"][image_idx]

    rotation_table = per_image_inputs.get("rotation_table") if isinstance(per_image_inputs, dict) else None
    if (
        device_rotations
        and rotation_table is not None
        and np.dtype(rotation_table.dtype) == np.dtype(rotation_dtype)
        and rotations_by_index_enabled()
    ):
        padded_rotations = padded_rotations_from_table_device(
            rotation_table,
            per_image_inputs["rotation_table_key"],
            padded_rotation_indices,
            actual_counts,
            bucket_size,
            np.eye(3, dtype=rotation_dtype),
        )
        mstep_table = per_image_inputs.get("mstep_rotation_table")
        if not separate_mstep_rotations:
            padded_mstep_rotations = padded_rotations
        elif mstep_table is not None and np.dtype(mstep_table.dtype) == np.dtype(mstep_rotation_dtype):
            padded_mstep_rotations = padded_rotations_from_table_device(
                mstep_table,
                per_image_inputs["mstep_rotation_table_key"],
                padded_rotation_indices,
                actual_counts,
                bucket_size,
                np.eye(3, dtype=mstep_rotation_dtype),
            )
        else:
            padded_mstep_rotations = padded_rows_from_flat_device(
                [np.asarray(per_image_inputs["oversampled_mstep_rots"][i], dtype=mstep_rotation_dtype) for i in padded_image_indices[:n_real].tolist()],
                actual_counts,
                bucket_size,
                np.eye(3, dtype=mstep_rotation_dtype),
            )
    elif device_rotations:
        real_images = padded_image_indices[:n_real].tolist()
        padded_rotations = padded_rows_from_flat_device(
            [np.asarray(per_image_inputs["oversampled_rots"][i], dtype=rotation_dtype) for i in real_images],
            actual_counts,
            bucket_size,
            np.eye(3, dtype=rotation_dtype),
        )
        padded_mstep_rotations = (
            padded_rows_from_flat_device(
                [np.asarray(per_image_inputs["oversampled_mstep_rots"][i], dtype=mstep_rotation_dtype) for i in real_images],
                actual_counts,
                bucket_size,
                np.eye(3, dtype=mstep_rotation_dtype),
            )
            if separate_mstep_rotations
            else padded_rotations
        )

    return {
        "image_indices": padded_image_indices,
        "bucket_size": bucket_size,
        "actual_counts": actual_counts,
        "rotations": padded_rotations,
        "mstep_rotations": padded_mstep_rotations,
        "rotation_indices": padded_rotation_indices,
        "log_prior": padded_log_prior,
        "candidate_mask": padded_candidate_mask,
        "parent_map": padded_parent_map,
    }


LADDER_CHUNKS_ENV = "RELAX_SPARSE_PASS2_LADDER_CHUNKS"
LADDER_CHUNK_FLOOR = 16


def ladder_chunks_enabled() -> bool:
    """Return whether bucket image lists are split into power-of-two chunks.

    Fused K-class pass 2 compiles one XLA program per helper and bucket shape.
    Without the ladder each rotation/pair bucket holds however many images
    fall into it (a different count every iteration), so every bucket of every
    iteration is a new shape. With the ladder the image axis takes only
    power-of-two sizes at or above :data:`LADDER_CHUNK_FLOOR` plus one
    remainder below the floor, so bucket shapes repeat across iterations.
    Per-image results are unchanged; only the grouping of per-bucket
    reductions differs.
    """

    return parse_env_binary_flag(LADDER_CHUNKS_ENV)


def bucket_chunk_bounds(n_images: int, max_per_chunk: int, *, ladder: bool | None = None):
    """Return ``(start, stop)`` chunk bounds for one bucket's image list.

    Default: consecutive chunks of ``max_per_chunk``. Ladder: greedy powers of
    two no larger than ``max_per_chunk`` while at least
    :data:`LADDER_CHUNK_FLOOR` images remain, then one remainder chunk.
    """

    max_per_chunk = max(1, int(max_per_chunk))
    if ladder is None:
        ladder = ladder_chunks_enabled()
    if not ladder:
        return [(start, min(start + max_per_chunk, n_images)) for start in range(0, n_images, max_per_chunk)]
    largest_power = pow2_floor(max_per_chunk)
    bounds = []
    start = 0
    while n_images - start >= LADDER_CHUNK_FLOOR:
        size = min(largest_power, pow2_floor(n_images - start))
        bounds.append((start, start + size))
        start += size
    # The remainder must still respect the caller's cap: it is derived from
    # gather/prepare/dense-M-step byte budgets, so emitting the tail as one
    # chunk would overshoot them (a cap of 1 image would yield a chunk of 15).
    while start < n_images:
        stop = min(start + max_per_chunk, n_images)
        bounds.append((start, stop))
        start = stop
    return bounds


def _reorder_to_indices(image_indices_returned, requested_image_indices, *arrays):
    """Reorder per-image arrays so they match the order returned by the dataset."""
    if np.array_equal(image_indices_returned, requested_image_indices):
        return arrays
    position = {int(idx): pos for pos, idx in enumerate(np.asarray(requested_image_indices).tolist())}
    order = np.array([position[int(idx)] for idx in np.asarray(image_indices_returned).tolist()], dtype=np.int64)
    return tuple(arr[order] for arr in arrays)


@jax.jit
def _normalize_pass2_bucket_with_log_z(scores, log_z):
    """Normalize sparse candidate scores with a precomputed full-grid log-Z."""
    scores = jnp.where(jnp.isfinite(scores), scores, -jnp.inf)
    flat = scores.reshape(scores.shape[0], -1)
    best_log_score = jnp.max(flat, axis=1)
    has_finite_score = jnp.isfinite(best_log_score) & jnp.isfinite(log_z)
    safe_log_z = jnp.where(has_finite_score, log_z, 0.0)
    probs = jnp.exp(scores - safe_log_z[:, None, None])
    probs = jnp.where(has_finite_score[:, None, None] & jnp.isfinite(probs), probs, 0.0)
    best_argmax = jnp.where(has_finite_score, jnp.argmax(flat, axis=1), 0)
    max_posterior = jnp.exp(best_log_score - safe_log_z)
    max_posterior = jnp.where(has_finite_score & jnp.isfinite(max_posterior), max_posterior, 0.0)
    best_log_score = jnp.where(has_finite_score, best_log_score, -jnp.inf)
    return safe_log_z, probs, best_log_score, best_argmax, max_posterior


@jax.jit
def compute_local_ctf_sums(probs, ctf2_over_nv):
    """Compute weighted CTF^2/noise sums for one exact local bucket."""

    probs_sum_t = jnp.sum(probs, axis=-1)  # (B, R)
    return compute_local_ctf_sums_from_probs_sum_t(probs_sum_t, ctf2_over_nv)


@jax.jit
def compute_relion_sequential_mstep_sums(probs, shifted, ctf2_over_nv):
    """Reduce translations in RELION order without changing precision.

    RELION accumulates this loop in ``XFLOAT``.  That is float32 in its normal
    GPU build and float64 in a double-precision build, so the RECOVAR parity
    path must follow the operand precision rather than unconditionally model
    the former.
    """

    real_dtype = jnp.result_type(probs, ctf2_over_nv, jnp.real(shifted))
    complex_dtype = jnp.complex128 if real_dtype == jnp.dtype(jnp.float64) else jnp.complex64
    probs = jnp.asarray(probs, dtype=real_dtype)
    shifted = jnp.asarray(shifted, dtype=complex_dtype)
    ctf2_over_nv = jnp.asarray(ctf2_over_nv, dtype=real_dtype)
    batch, n_rot, n_trans = probs.shape
    n_pixels = shifted.shape[-1]
    numerator0 = jnp.zeros((batch, n_rot, n_pixels), dtype=complex_dtype)
    denominator0 = jnp.zeros((batch, n_rot, n_pixels), dtype=real_dtype)

    def add_translation(trans_idx, carry):
        numerator, denominator = carry
        weight = probs[:, :, trans_idx, None]
        numerator = numerator + weight * shifted[:, None, trans_idx, :]
        denominator = denominator + weight * ctf2_over_nv[:, None, :]
        return numerator, denominator

    return jax.lax.fori_loop(0, n_trans, add_translation, (numerator0, denominator0))


def compute_local_mstep_sums(
    probs,
    shifted,
    ctf2_over_nv,
    *,
    relion_x_half: bool,
    default_probs_sum_t=None,
    sequential_translation_reduction: bool | None = None,
):
    """Compute numerator/denominator sums, optionally using the x-half diagnostic."""

    use_sequential_reduction = (
        relion_x_half_sequential_translation_reduction_enabled()
        if sequential_translation_reduction is None
        else bool(sequential_translation_reduction)
    )
    if relion_x_half and use_sequential_reduction:
        return compute_relion_sequential_mstep_sums(probs, shifted, ctf2_over_nv)
    denominator = (
        compute_local_ctf_sums(probs, ctf2_over_nv)
        if default_probs_sum_t is None
        else compute_local_ctf_sums_from_probs_sum_t(default_probs_sum_t, ctf2_over_nv)
    )
    return compute_local_weighted_sums(probs, shifted), denominator


def flatten_bucket_rotations(rotations):
    """Flatten a bucket's per-image rotations into one row-major batch.

    This is a pure reshape and is deliberately not jitted: bucket rotation
    blocks arrive as host arrays with a different ``(images, rotations)`` shape
    for almost every bucket chunk, and a jitted reshape paid one compilation
    cache lookup (key hashing plus a persistent-cache file read of ~85 ms)
    per new shape, ~30% of a warm hp3 pass 2 (job 14051847 sampler).  Host
    arrays are reshaped on the host; device arrays use the eager reshape.
    """

    if isinstance(rotations, np.ndarray):
        return rotations.reshape(rotations.shape[0] * rotations.shape[1], 3, 3)
    return jnp.reshape(rotations, (rotations.shape[0] * rotations.shape[1], 3, 3))
