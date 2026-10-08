"""Host planning and array assembly for rectangular and compact sparse buckets.

Callers select execution policies and budgets. These helpers group supplied
support into bounded buckets, expand parent support and gather/pad rows. They
preserve particle order, precision, scoring/M-step aliases and inert padding;
they neither execute scoring nor choose scientific or device policies.
"""

from __future__ import annotations

import logging

import numpy as np

from relax.helpers.batch_planning import _plan_consecutive_padded_batches
from relax.helpers.env_flags import parse_env_binary_flag, parse_env_flag
from relax.helpers.shape_buckets import pow2_ceil, pow2_floor, power_of_two_bucket
from relax.local.local_layout import _exact_bucket_rotation_size

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

    size = int(_exact_bucket_rotation_size(int(count), rotation_block_size_for_quantization))
    if size > _LARGE_BUCKET_POW2_THRESHOLD and parse_env_flag(_LARGE_BUCKET_POW2_ENV, default=False):
        return int(power_of_two_bucket(size))
    return size
from relax.scoring.compact_candidates import (
    SparseCandidateMask,
    _candidate_mask_to_dense,
)
from relax.scoring.significant_samples import ComplementSignificantSampleIndices

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
            plans = _plan_consecutive_padded_batches(
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
        # Group by bucket size, smaller buckets first. The secondary rotation
        # count key is historical RECOVAR behavior; an explicit order keeps
        # RELION order stable within each equal padded-size bucket.
        processing_order = np.lexsort((rotation_counts, bucket_sizes)).astype(np.int64)

    unique_bucket_sizes = np.unique(bucket_sizes[processing_order])
    buckets = []
    for bucket_size in unique_bucket_sizes:
        bucket_size = int(bucket_size)
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
    max_images = int(max_images)
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


logger = logging.getLogger(__name__)


VECTORIZED_HYPOTHESIS_PREP_ENV = "RELAX_SPARSE_PASS2_VECTORIZED_HYPOTHESIS_PREP"


def vectorized_hypothesis_prep_enabled() -> bool:
    """Prepare the per-image pass-2 hypotheses for all images at once.

    ``_prepare_per_image_pass2_inputs`` loops over every image in Python
    (np.unique, boolean child masks, searchsorted, per-image gathers); at
    100k/256 K=4 that is 34 s of host time per iteration (job 13832892,
    "hypothesis_prep=34.06s").  The vectorized path builds the same per-image
    arrays from flat concatenations and hands out views, for the paired
    RELION fine-grid override with a parent-major fine grid.  Images that need
    another branch (no samples, complement masks, empty sets) keep the loop.
    """

    return parse_env_binary_flag(VECTORIZED_HYPOTHESIS_PREP_ENV)


def _fine_children_ranges(fine_parent_np, n_coarse_rot):
    """``(child_start, child_count)`` per coarse rotation, or ``None`` if not parent-major."""

    fine_parent_np = np.asarray(fine_parent_np, dtype=np.int64)
    if fine_parent_np.ndim != 1 or (fine_parent_np.size > 1 and np.any(np.diff(fine_parent_np) < 0)):
        return None
    child_count = np.bincount(fine_parent_np, minlength=int(n_coarse_rot)).astype(np.int64)
    child_start = np.concatenate(([0], np.cumsum(child_count)[:-1])).astype(np.int64)
    return child_start, child_count


def _prepare_coarse_images_vectorized(
    sample_lists,
    *,
    n_coarse_rot,
    n_coarse_trans,
    n_fine_trans,
    fine_translation_parent,
    rotation_log_prior_np,
    fine_rotations_np,
    fine_mstep_rotations_np,
    fine_source_eulers,
    child_start,
    child_count,
    dtype,
):
    """Vectorized equivalent of the per-image ``coarse`` branch of the loop.

    Every returned per-image array is a view into one flat array and equals
    the loop's result element for element (values, dtypes and order); the
    unique coarse rotations are ascending, the fine rows are the ascending
    children of those rotations (``np.flatnonzero`` order for a parent-major
    grid) and ``parent_map`` is the rank of each row's parent.
    """

    n_images = len(sample_lists)
    lengths = np.fromiter((int(np.asarray(s).size) for s in sample_lists), dtype=np.int64, count=n_images)
    flat_sig = (
        np.concatenate([np.asarray(s, dtype=np.int32).reshape(-1) for s in sample_lists]).astype(np.int64)
        if n_images
        else np.zeros(0, dtype=np.int64)
    )
    sig_img = np.repeat(np.arange(n_images, dtype=np.int64), lengths)
    # The loop marks a boolean (coarse rotation, coarse translation) table, so a
    # significant index listed twice for one image counts once; dedupe here so
    # the per-pair sample sums below give the loop's ``count`` (lead review
    # em_clean_vectorized_hypothesis_duplicate_20260913).
    total_cells = int(n_coarse_rot) * int(n_coarse_trans)
    cell_keys = sig_img * total_cells + flat_sig
    if cell_keys.size > 1 and not bool(np.all(np.diff(cell_keys) > 0)):
        # Only sort when some image lists an index twice or out of order; the
        # significance pass emits sorted unique indices, so this is the rare path.
        cell_keys = np.unique(cell_keys)
        sig_img = cell_keys // total_cells
        flat_sig = cell_keys % total_cells
    coarse_rot = flat_sig // int(n_coarse_trans)
    coarse_trans = flat_sig % int(n_coarse_trans)
    pair_key, sig_pair = np.unique(sig_img * int(n_coarse_rot) + coarse_rot, return_inverse=True)
    sig_pair = np.asarray(sig_pair).reshape(-1)
    pair_img = pair_key // int(n_coarse_rot)
    pair_urot = pair_key % int(n_coarse_rot)
    n_pairs = int(pair_key.shape[0])
    pairs_per_image = np.bincount(pair_img, minlength=n_images).astype(np.int64)
    pair_starts = np.concatenate(([0], np.cumsum(pairs_per_image)[:-1])).astype(np.int64)
    pair_rank = np.arange(n_pairs, dtype=np.int64) - np.repeat(pair_starts, pairs_per_image)

    rows_per_pair = child_count[pair_urot]
    n_rows = int(rows_per_pair.sum())
    row_pair = np.repeat(np.arange(n_pairs, dtype=np.int64), rows_per_pair)
    row_starts = np.concatenate(([0], np.cumsum(rows_per_pair)[:-1])).astype(np.int64)
    row_offset = np.arange(n_rows, dtype=np.int64) - np.repeat(row_starts, rows_per_pair)
    flat_rot_indices = child_start[pair_urot][row_pair] + row_offset
    flat_parent_map = pair_rank[row_pair].astype(np.int32)
    rows_per_image = np.bincount(pair_img, weights=rows_per_pair, minlength=n_images).astype(np.int64)

    # np.take on a contiguous first axis is the fastest host gather for these
    # (rows, 3, 3) / (rows, 3) tables; the per-image loop gathers the same rows.
    flat_rots = np.take(np.asarray(fine_rotations_np, dtype=dtype), flat_rot_indices, axis=0)
    flat_mstep_rots = (
        None
        if fine_mstep_rotations_np is None
        else np.take(np.asarray(fine_mstep_rotations_np, dtype=dtype), flat_rot_indices, axis=0)
    )
    flat_eulers = None if fine_source_eulers is None else np.take(np.asarray(fine_source_eulers), flat_rot_indices, axis=0)
    if rotation_log_prior_np is not None:
        flat_log_prior = np.take(np.asarray(rotation_log_prior_np, dtype=dtype), pair_urot[row_pair])
    else:
        flat_log_prior = np.zeros(n_rows, dtype=dtype)

    coarse_valid_flat = np.zeros((n_pairs, int(n_coarse_trans)), dtype=bool)
    coarse_valid_flat[sig_pair, coarse_trans] = True
    ftp = np.asarray(fine_translation_parent, dtype=np.int64).reshape(-1)
    fine_children_per_coarse_trans = np.bincount(ftp, minlength=int(n_coarse_trans)).astype(np.int64)
    # Each significant sample is one distinct (coarse rotation, coarse translation)
    # cell of its image, so the fine-translation children of a pair's valid cells
    # sum over its samples; exact in float64 for these small integers.
    valid_fine_per_pair = np.rint(
        np.bincount(sig_pair, weights=fine_children_per_coarse_trans[coarse_trans], minlength=n_pairs)
    ).astype(np.int64)
    counts_per_image = np.rint(
        np.bincount(pair_img, weights=rows_per_pair * valid_fine_per_pair, minlength=n_images)
    ).astype(np.int64)

    row_bounds = np.concatenate(([0], np.cumsum(rows_per_image))).astype(np.int64)
    pair_bounds = np.concatenate(([0], np.cumsum(pairs_per_image))).astype(np.int64)
    unique_rot_flat = pair_urot.astype(np.int32)
    out = {
        "source_eulers": [],
        "oversampled_rots": [],
        "oversampled_mstep_rots": [],
        "parent_map": [],
        "oversampled_rot_indices": [],
        "unique_rot": [],
        "log_prior": [],
        "candidate_mask": [],
    }
    # Flat coarse tables and row parents for the device index builder
    # (RELAX_SPARSE_KCLASS_RESIDENT_HYPOTHESIS_TABLES): one upload per class
    # and iteration instead of a host (images, cR, cT)/(images, rows) build and
    # upload per class-chunk. ``token`` identifies this content for the cache.
    import uuid

    out["_resident"] = {
        "coarse_valid_flat": coarse_valid_flat,
        "parent_map_flat": flat_parent_map,
        "rot_indices_flat": flat_rot_indices.astype(np.int32, copy=False),
        "pair_bounds": pair_bounds,
        "row_bounds": row_bounds,
        "n_coarse_trans": int(n_coarse_trans),
        "token": uuid.uuid4().hex,
    }
    for i in range(n_images):
        r0, r1 = int(row_bounds[i]), int(row_bounds[i + 1])
        p0, p1 = int(pair_bounds[i]), int(pair_bounds[i + 1])
        rots = flat_rots[r0:r1]
        parent_map = flat_parent_map[r0:r1]
        out["source_eulers"].append(None if flat_eulers is None else flat_eulers[r0:r1])
        out["oversampled_rots"].append(rots)
        out["oversampled_mstep_rots"].append(rots if flat_mstep_rots is None else flat_mstep_rots[r0:r1])
        out["parent_map"].append(parent_map)
        out["oversampled_rot_indices"].append(flat_rot_indices[r0:r1])
        out["unique_rot"].append(unique_rot_flat[p0:p1])
        out["log_prior"].append(flat_log_prior[r0:r1])
        out["candidate_mask"].append(
            SparseCandidateMask(
                mode="coarse",
                n_rows=r1 - r0,
                n_fine_trans=n_fine_trans,
                parent_map=parent_map,
                coarse_valid=coarse_valid_flat[p0:p1],
                fine_translation_parent=fine_translation_parent,
                count=int(counts_per_image[i]),
            )
        )
    return out


def _rotation_table_key(table) -> tuple:
    """Content key for the device table cache: shape, dtype and a SHA-1 of the bytes."""

    import hashlib

    table = np.ascontiguousarray(table)
    return (table.shape, str(table.dtype), hashlib.sha1(table.view(np.uint8)).hexdigest())


def relion_parent_execution_key(parent_ids, *, n_coarse_rot: int, nside_level: int) -> np.ndarray:
    """RELION's execution key for coarse rotation ids.

    RECOVAR's coarse grid is psi-slow and direction-fast
    (``id = psi * n_directions + direction``); RELION executes parents
    direction-major (``direction * n_psi + psi``). The direction count comes
    from the grid itself, ``n_coarse_rot / n_psi``, so a symmetry-reduced grid
    keeps its own direction count. The host pass-2 preparation and the resident
    CSR tables both order rows with this key.
    """

    from relax.sampling import rotation_grid_n_in_planes

    n_psi = int(rotation_grid_n_in_planes(nside_level))
    n_coarse_rot = int(n_coarse_rot)
    if n_coarse_rot <= 0 or n_coarse_rot % n_psi:
        raise ValueError(
            f"RELION parent execution order needs whole psi rows: {n_coarse_rot} coarse "
            f"rotations with {n_psi} psi angles at healpix level {int(nside_level)}"
        )
    n_directions = n_coarse_rot // n_psi
    parent_ids = np.asarray(parent_ids, dtype=np.int64).reshape(-1)
    if parent_ids.size and (int(parent_ids.min()) < 0 or int(parent_ids.max()) >= n_coarse_rot):
        raise ValueError("RELION parent execution key is outside the coarse grid")
    return (parent_ids % n_directions) * n_psi + parent_ids // n_directions


def _prepare_per_image_pass2_inputs(
    significant_sample_indices,
    n_coarse_rot,
    n_coarse_trans,
    nside_level,
    oversampling_order,
    n_fine_trans,
    fine_translation_parent,
    rotation_log_prior,
    random_perturbation,
    fine_source_eulers_override=None,
    fine_rotations_override=None,
    fine_mstep_rotations_override=None,
    fine_rotation_parent_override=None,
    relion_parent_execution_order=False,
    dtype: np.dtype = np.float32,
    symmetry_label: str = "C1",
    coarse_rotation_ids=None,
    per_image_rotation_log_prior=None,
):
    """Compute per-image oversampled rotations / parent maps / candidate masks.

    Builds each image's oversampled children exactly as the per-image reference
    pass 2 did (removed on 2026-10-03), so the batched path is a strict per-image
    equivalent.

    ``dtype`` controls the precision of every fine/oversampled rotation
    matrix this function builds or accepts: the RELION-supplied fine
    rotation override (``fine_rotations_override`` /
    ``fine_mstep_rotations_override``) *and* the standard
    ``get_oversampled_rotation_grid_from_samples`` grid built when no
    override is supplied. RELION's own fine-search rotation matrices stay
    ``RFLOAT`` (double) end to end in a double-precision build; pass
    ``precision_policy.score_real_dtype`` from the caller so this matches
    ``use_float64_scoring`` instead of always narrowing to float32.

    ``coarse_rotation_ids`` makes the coarse grid a compact subset of the HEALPix grid at
    ``nside_level`` (coarse id ``c`` is grid rotation ``coarse_rotation_ids[c]``; subtomogram
    local searches, relax.refinement.tomo_half); RELION's parent execution order is then the
    grid's. ``per_image_rotation_log_prior`` replaces ``rotation_log_prior`` by one prior per
    image over that image's significant coarse rotations in ascending order (RELION's local
    orientation prior is the particle's own).
    """
    from relax.sampling import (
        get_oversampled_rotation_grid_from_samples,
        rotation_grid_size,
    )

    if per_image_rotation_log_prior is not None and rotation_log_prior is not None:
        raise ValueError("a pass takes one shared rotation prior or one per image, not both")
    if coarse_rotation_ids is not None:
        coarse_rotation_ids = np.asarray(coarse_rotation_ids, dtype=np.int64).reshape(-1)
        if coarse_rotation_ids.shape != (int(n_coarse_rot),):
            raise ValueError("coarse_rotation_ids must give one grid rotation per coarse rotation")
        grid_rotation_count = int(rotation_grid_size(nside_level, symmetry_label))
        if coarse_rotation_ids.size and (
            int(coarse_rotation_ids.min()) < 0 or int(coarse_rotation_ids.max()) >= grid_rotation_count
        ):
            raise ValueError("coarse_rotation_ids must index the HEALPix grid")
    from relax.symmetry import canonicalize_rotational_symmetry

    symmetry_label = canonicalize_rotational_symmetry(symmetry_label)
    # A compact coarse grid (coarse_rotation_ids) is checked against the symmetric grid above.
    if symmetry_label != "C1" and coarse_rotation_ids is None:
        expected_coarse_rot = rotation_grid_size(nside_level, symmetry_label)
        if int(n_coarse_rot) != int(expected_coarse_rot):
            raise ValueError(
                f"{symmetry_label} sparse pass-2 coarse rotation count mismatch: "
                f"{n_coarse_rot} != {expected_coarse_rot}"
            )

    n_images = len(significant_sample_indices)
    per_image_source_eulers = []
    per_image_oversampled_rots = []
    per_image_oversampled_mstep_rots = []
    per_image_parent_map = []
    per_image_oversampled_rot_indices = []
    per_image_unique_rot = []
    per_image_log_prior = []
    per_image_candidate_mask = []
    per_image_lists = {
        "source_eulers": per_image_source_eulers,
        "oversampled_rots": per_image_oversampled_rots,
        "oversampled_mstep_rots": per_image_oversampled_mstep_rots,
        "parent_map": per_image_parent_map,
        "oversampled_rot_indices": per_image_oversampled_rot_indices,
        "unique_rot": per_image_unique_rot,
        "log_prior": per_image_log_prior,
        "candidate_mask": per_image_candidate_mask,
    }
    full_unique_rot = np.arange(n_coarse_rot, dtype=np.int32)
    full_support_rotation_cache = None
    full_support_log_prior_cache = None
    full_support_candidate_mask_cache = None

    if rotation_log_prior is not None:
        rotation_log_prior_np = np.asarray(rotation_log_prior, dtype=dtype)
    else:
        rotation_log_prior_np = None

    fine_rotations_np = None
    fine_mstep_rotations_np = None
    fine_parent_np = None
    if fine_rotations_override is None and fine_rotation_parent_override is None:
        pass
    elif fine_rotations_override is not None and fine_rotation_parent_override is not None:
        fine_rotations_np = np.asarray(fine_rotations_override, dtype=dtype)
        fine_parent_np = np.asarray(fine_rotation_parent_override, dtype=np.int64)
        if fine_parent_np.ndim != 1:
            raise ValueError("fine_rotation_parent_override must be a 1D array")
        if fine_rotations_np.shape[0] != fine_parent_np.shape[0]:
            raise ValueError(
                "fine_rotations_override and fine_rotation_parent_override disagree on rotation count: "
                f"{fine_rotations_np.shape[0]} vs {fine_parent_np.shape[0]}",
            )
        if int(fine_parent_np.min(initial=0)) < 0 or int(fine_parent_np.max(initial=-1)) >= int(n_coarse_rot):
            raise ValueError("fine_rotation_parent_override values must be in [0, n_coarse_rot)")
    else:
        raise ValueError("fine_rotations_override and fine_rotation_parent_override must be provided together")

    if fine_mstep_rotations_override is not None:
        if fine_rotations_np is None:
            raise ValueError("fine_mstep_rotations_override requires fine_rotations_override")
        fine_mstep_rotations_np = np.asarray(fine_mstep_rotations_override, dtype=dtype)
        if fine_mstep_rotations_np.shape != fine_rotations_np.shape:
            raise ValueError(
                "fine_mstep_rotations_override must match fine_rotations_override shape: "
                f"{fine_mstep_rotations_np.shape} vs {fine_rotations_np.shape}",
            )

    fine_source_eulers = None
    if fine_source_eulers_override is not None:
        fine_source_eulers = np.asarray(fine_source_eulers_override)
        if (
            fine_rotations_np is None
            or fine_source_eulers.dtype != np.float64
            or fine_source_eulers.shape != (len(fine_rotations_np), 3)
            or not np.all(np.isfinite(fine_source_eulers))
        ):
            raise ValueError(
                "fine source Euler metadata must match the supplied rotation rows as finite float64 triples"
            )

    def _reorder_children(rotations, parent_map, rotation_indices, source_eulers, parent_ids):
        if not relion_parent_execution_order:
            return rotations, parent_map, rotation_indices, source_eulers
        parent_ids = np.asarray(parent_ids, dtype=np.int64).reshape(-1)
        if parent_ids.shape != np.asarray(parent_map).shape:
            raise ValueError("RELION parent execution keys must match fine rotations")
        if coarse_rotation_ids is None:
            relion_parent_key = relion_parent_execution_key(
                parent_ids, n_coarse_rot=n_coarse_rot, nside_level=nside_level
            )
        else:
            relion_parent_key = relion_parent_execution_key(
                coarse_rotation_ids[parent_ids], n_coarse_rot=grid_rotation_count, nside_level=nside_level
            )
        order = np.argsort(relion_parent_key, kind="stable")
        return (
            np.asarray(rotations)[order],
            np.asarray(parent_map)[order],
            np.asarray(rotation_indices)[order],
            None if source_eulers is None else source_eulers[order],
        )

    vectorized_indices = []
    if (
        fine_rotations_np is not None
        and not relion_parent_execution_order
        and per_image_rotation_log_prior is None
        and vectorized_hypothesis_prep_enabled()
    ):
        children = _fine_children_ranges(fine_parent_np, n_coarse_rot)
        if children is not None:
            vectorized_indices = [
                image_idx
                for image_idx, sig_samples in enumerate(significant_sample_indices)
                if sig_samples is not None
                and not isinstance(sig_samples, ComplementSignificantSampleIndices)
                and np.asarray(sig_samples).size > 0
            ]
    vectorized = None
    if vectorized_indices:
        vectorized = _prepare_coarse_images_vectorized(
            [significant_sample_indices[image_idx] for image_idx in vectorized_indices],
            n_coarse_rot=n_coarse_rot,
            n_coarse_trans=n_coarse_trans,
            n_fine_trans=n_fine_trans,
            fine_translation_parent=fine_translation_parent,
            rotation_log_prior_np=rotation_log_prior_np,
            fine_rotations_np=fine_rotations_np,
            fine_mstep_rotations_np=fine_mstep_rotations_np,
            fine_source_eulers=fine_source_eulers,
            child_start=children[0],
            child_count=children[1],
            dtype=dtype,
        )
    vectorized_set = set(vectorized_indices)
    vectorized_position = {image_idx: position for position, image_idx in enumerate(vectorized_indices)}
    resident_hypothesis = None
    if vectorized is not None:
        positions = np.full(n_images, -1, dtype=np.int64)
        positions[np.asarray(vectorized_indices, dtype=np.int64)] = np.arange(len(vectorized_indices), dtype=np.int64)
        resident_hypothesis = dict(vectorized.pop("_resident"), positions=positions)

    for image_idx, sig_samples in enumerate(significant_sample_indices):
        if image_idx in vectorized_set:
            position = vectorized_position[image_idx]
            for key, values in vectorized.items():
                per_image_lists[key].append(values[position])
            continue
        coarse_excluded = None
        if sig_samples is None:
            unique_rot = full_unique_rot
            use_full_candidate_mask = True
            use_full_rotation_support = True
            coarse_rot = unique_rot
            coarse_trans = None
        elif isinstance(sig_samples, ComplementSignificantSampleIndices):
            if int(sig_samples.total_size) != int(n_coarse_rot * n_coarse_trans):
                raise ValueError(
                    "Complement significant sample mask total_size does not match coarse pose grid: "
                    f"{int(sig_samples.total_size)} vs {int(n_coarse_rot * n_coarse_trans)}",
                )
            unique_rot = full_unique_rot
            use_full_candidate_mask = False
            use_full_rotation_support = True
            coarse_rot = unique_rot
            coarse_trans = None
            coarse_excluded = np.asarray(sig_samples.excluded_indices, dtype=np.int32).reshape(-1)
        else:
            use_full_candidate_mask = False
            use_full_rotation_support = False
            sig_samples = np.asarray(sig_samples, dtype=np.int32).reshape(-1)
            if sig_samples.size == 0:
                coarse_rot = np.empty(0, dtype=np.int32)
                coarse_trans = np.empty(0, dtype=np.int32)
                unique_rot = np.array([0], dtype=np.int32)
            else:
                coarse_rot = sig_samples // n_coarse_trans
                coarse_trans = sig_samples % n_coarse_trans
                unique_rot = np.unique(coarse_rot)

        if unique_rot.size == 0:
            raise ValueError(f"Image {image_idx} has no significant coarse samples for sparse pass 2")

        if use_full_rotation_support:
            if full_support_rotation_cache is None:
                if fine_rotations_override is None and fine_rotation_parent_override is None:
                    full_rots, full_parent_map, full_rot_indices, full_eulers = (
                        get_oversampled_rotation_grid_from_samples(
                            full_unique_rot,
                            nside_level,
                            oversampling_order=oversampling_order,
                            random_perturbation=random_perturbation,
                            return_rotation_indices=True,
                            return_source_eulers=True,
                            dtype=dtype,
                            **({} if symmetry_label == "C1" else {"symmetry": symmetry_label}),
                        )
                    )
                    full_support_rotation_cache = (
                        np.asarray(full_rots, dtype=dtype),
                        np.asarray(full_parent_map, dtype=np.int32),
                        np.asarray(full_rot_indices, dtype=np.int64),
                        full_eulers,
                    )
                else:  # Paired overrides were validated before entering the image loop.
                    full_support_rotation_cache = (
                        fine_rotations_np,
                        fine_parent_np.astype(np.int32, copy=False),
                        np.arange(fine_rotations_np.shape[0], dtype=np.int64),
                        fine_source_eulers,
                    )
                full_support_rotation_cache = _reorder_children(
                    *full_support_rotation_cache,
                    parent_ids=full_unique_rot[full_support_rotation_cache[1]],
                )
            oversampled_rots, parent_map, oversampled_rot_indices, source_eulers = full_support_rotation_cache
        elif fine_rotations_override is None and fine_rotation_parent_override is None:
            oversampled_rots, parent_map, oversampled_rot_indices, source_eulers = (
                get_oversampled_rotation_grid_from_samples(
                    unique_rot,
                    nside_level,
                    oversampling_order=oversampling_order,
                    random_perturbation=random_perturbation,
                    return_rotation_indices=True,
                    return_source_eulers=True,
                    dtype=dtype,
                    **({} if symmetry_label == "C1" else {"symmetry": symmetry_label}),
                )
            )
            oversampled_rots = np.asarray(oversampled_rots, dtype=dtype)
            parent_map = np.asarray(parent_map, dtype=np.int32)
            oversampled_rot_indices = np.asarray(oversampled_rot_indices, dtype=np.int64)
            oversampled_rots, parent_map, oversampled_rot_indices, source_eulers = _reorder_children(
                oversampled_rots,
                parent_map,
                oversampled_rot_indices,
                source_eulers,
                unique_rot[parent_map],
            )
        else:  # Paired overrides were validated before entering the image loop.
            selected_parent = np.zeros(n_coarse_rot, dtype=bool)
            selected_parent[unique_rot] = True
            child_mask = selected_parent[fine_parent_np]
            oversampled_rot_indices = np.flatnonzero(child_mask).astype(np.int64)
            oversampled_rots = fine_rotations_np[oversampled_rot_indices]
            parent_map = np.searchsorted(unique_rot, fine_parent_np[oversampled_rot_indices]).astype(np.int32)
            oversampled_rots, parent_map, oversampled_rot_indices, source_eulers = _reorder_children(
                oversampled_rots,
                parent_map,
                oversampled_rot_indices,
                None if fine_source_eulers is None else fine_source_eulers[oversampled_rot_indices],
                fine_parent_np[oversampled_rot_indices],
            )

        oversampled_mstep_rots = (
            oversampled_rots if fine_mstep_rotations_np is None else fine_mstep_rotations_np[oversampled_rot_indices]
        )

        if per_image_rotation_log_prior is not None:
            if use_full_rotation_support:
                raise ValueError("a per-image rotation prior needs each image's explicit coarse support")
            image_prior = np.asarray(per_image_rotation_log_prior[image_idx], dtype=dtype).reshape(-1)
            if image_prior.shape != (unique_rot.size,):
                raise ValueError(
                    f"image {image_idx}'s rotation prior covers {image_prior.size} rotations, "
                    f"its support {unique_rot.size}"
                )
            local_rotation_log_prior = image_prior[parent_map]
        elif use_full_rotation_support and full_support_log_prior_cache is not None:
            local_rotation_log_prior = full_support_log_prior_cache
        elif rotation_log_prior_np is not None:
            local_rotation_log_prior = rotation_log_prior_np[unique_rot][parent_map]
            if use_full_rotation_support:
                local_rotation_log_prior = local_rotation_log_prior.astype(dtype, copy=False)
        else:
            local_rotation_log_prior = np.zeros(oversampled_rots.shape[0], dtype=dtype)
        if use_full_rotation_support:
            full_support_log_prior_cache = local_rotation_log_prior

        if use_full_candidate_mask:
            if full_support_candidate_mask_cache is None:
                full_support_candidate_mask_cache = SparseCandidateMask(
                    mode="full",
                    n_rows=oversampled_rots.shape[0],
                    n_fine_trans=n_fine_trans,
                    count=int(oversampled_rots.shape[0]) * int(n_fine_trans),
                )
            candidate_mask = full_support_candidate_mask_cache
        elif coarse_excluded is not None:
            excluded = np.unique(coarse_excluded.astype(np.int64, copy=False))
            if excluded.size and (
                int(excluded.min(initial=0)) < 0 or int(excluded.max(initial=-1)) >= int(n_coarse_rot * n_coarse_trans)
            ):
                raise ValueError("Complement significant sample exclusions must index the coarse pose grid")
            excluded_rot = excluded // int(n_coarse_trans)
            excluded_trans = excluded % int(n_coarse_trans)
            fine_rot_children = np.bincount(np.asarray(parent_map, dtype=np.int64), minlength=int(n_coarse_rot))
            fine_trans_children = np.bincount(
                np.asarray(fine_translation_parent, dtype=np.int64),
                minlength=int(n_coarse_trans),
            )
            excluded_fine_count = int(
                np.sum(fine_rot_children[excluded_rot] * fine_trans_children[excluded_trans], dtype=np.int64),
            )
            candidate_mask = SparseCandidateMask(
                mode="coarse_exclude",
                n_rows=oversampled_rots.shape[0],
                n_fine_trans=n_fine_trans,
                parent_map=parent_map,
                coarse_excluded=excluded.astype(np.int32, copy=False),
                fine_translation_parent=fine_translation_parent,
                count=int(oversampled_rots.shape[0]) * int(n_fine_trans) - excluded_fine_count,
            )
        elif coarse_trans.size == 0:
            candidate_mask = SparseCandidateMask(
                mode="empty",
                n_rows=oversampled_rots.shape[0],
                n_fine_trans=n_fine_trans,
                count=0,
            )
        else:
            coarse_valid = np.zeros((unique_rot.size, n_coarse_trans), dtype=bool)
            coarse_valid[np.searchsorted(unique_rot, coarse_rot), coarse_trans] = True
            # count = sum over fine rows of the parent coarse row's fine-translation
            # count; O(cR*T + R) instead of expanding the (R, T) table per image.
            fine_per_coarse_row = coarse_valid[:, fine_translation_parent].sum(axis=1, dtype=np.int64)
            candidate_mask = SparseCandidateMask(
                mode="coarse",
                n_rows=oversampled_rots.shape[0],
                n_fine_trans=n_fine_trans,
                parent_map=parent_map,
                coarse_valid=coarse_valid,
                fine_translation_parent=fine_translation_parent,
                count=int(fine_per_coarse_row[parent_map].sum()),
            )

        per_image_source_eulers.append(source_eulers)
        per_image_oversampled_rots.append(oversampled_rots)
        per_image_oversampled_mstep_rots.append(oversampled_mstep_rots)
        per_image_parent_map.append(parent_map)
        per_image_oversampled_rot_indices.append(oversampled_rot_indices)
        per_image_unique_rot.append(unique_rot)
        per_image_log_prior.append(local_rotation_log_prior.astype(dtype, copy=False))
        per_image_candidate_mask.append(candidate_mask)

    assert len(per_image_oversampled_rots) == n_images
    # Global fine-grid tables (paired override only): every image's rows are
    # gathers of these by ``oversampled_rot_indices``, so the device can gather
    # them from one resident copy instead of receiving the rows per chunk.
    rotation_table = None if fine_rotations_np is None else np.asarray(fine_rotations_np, dtype=dtype)
    mstep_rotation_table = (
        None if fine_mstep_rotations_np is None else np.asarray(fine_mstep_rotations_np, dtype=dtype)
    )
    return {
        "source_eulers": per_image_source_eulers,
        "oversampled_rots": per_image_oversampled_rots,
        "oversampled_mstep_rots": per_image_oversampled_mstep_rots,
        "parent_map": per_image_parent_map,
        "oversampled_rot_indices": per_image_oversampled_rot_indices,
        "unique_rot": per_image_unique_rot,
        "log_prior": per_image_log_prior,
        "candidate_mask": per_image_candidate_mask,
        "rotation_table": rotation_table,
        "rotation_table_key": None if rotation_table is None else _rotation_table_key(rotation_table),
        "mstep_rotation_table": mstep_rotation_table,
        "mstep_rotation_table_key": None if mstep_rotation_table is None else _rotation_table_key(mstep_rotation_table),
        "resident_hypothesis": resident_hypothesis,
        # One coarse-row capacity for the whole pass: the device index builder
        # otherwise compiled a family per chunk-wise coarse-row count (census job
        # 13837258: 28 compiles, 18.7 s). Padded coarse rows are inert.
        "coarse_rows_capacity": pow2_ceil(
            max((int(np.asarray(u).shape[0]) for u in per_image_unique_rot), default=1), minimum=64
        ),
    }


BUCKET_ROTATIONS_DEVICE_ENV = "RELAX_SPARSE_KCLASS_BUCKET_ROTATIONS_DEVICE"
ROTATIONS_BY_INDEX_ENV = "RELAX_SPARSE_KCLASS_ROTATIONS_BY_INDEX"
RESIDENT_HYPOTHESIS_TABLES_ENV = "RELAX_SPARSE_KCLASS_RESIDENT_HYPOTHESIS_TABLES"
_ROTATION_TABLE_DEVICE_CACHE: dict = {}
_RESIDENT_ROW_INDEX_CACHE: dict = {}


def resident_hypothesis_tables_enabled() -> bool:
    """Feed the device index builder from device-resident flat hypothesis tables.

    With the vectorized hypothesis preparation the coarse validity table and
    the row parent map of every image already exist as one flat array per
    class; the device index builder otherwise rebuilt a padded
    ``(images, cR, cT)``/``(images, rows)`` pair on the host and uploaded it for
    every class-chunk (7.7 s of device_put plus the host fill in the 100k/256
    K=4 iteration 2, job 13834297).  Chunks whose images all took the
    vectorized path gather those tables on the device instead; values are
    identical, so the pair index arrays are bit-identical.
    """

    return parse_env_binary_flag(RESIDENT_HYPOTHESIS_TABLES_ENV)


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


def _resident_row_indices_device(resident):
    """Device copy of one class's flat per-row global rotation indices, keyed by prep token."""

    import jax.numpy as jnp

    key = resident["token"]
    cached = _RESIDENT_ROW_INDEX_CACHE.get(key)
    if cached is None:
        if len(_RESIDENT_ROW_INDEX_CACHE) >= 8:
            _RESIDENT_ROW_INDEX_CACHE.clear()
        cached = (
            jnp.asarray(np.asarray(resident["rot_indices_flat"], dtype=np.int32)),
            jnp.asarray(np.asarray(resident["row_bounds"], dtype=np.int32)),
        )
        _RESIDENT_ROW_INDEX_CACHE[key] = cached
    return cached


def _padded_rotations_from_resident_impl(table, rot_flat, row_bounds, positions, counts, fill, *, rows):
    import jax.numpy as jnp

    safe = jnp.clip(positions, 0, row_bounds.shape[0] - 2)
    start = row_bounds[safe]
    offsets = jnp.arange(rows, dtype=jnp.int32)[None, :]
    valid = offsets < counts[:, None]
    source = jnp.clip(start[:, None] + offsets, 0, rot_flat.shape[0] - 1)
    gathered = jnp.take(table, jnp.where(valid, jnp.take(rot_flat, source), 0), axis=0)
    return jnp.where(valid[:, :, None, None], gathered, fill[None, None])


def padded_rotations_from_resident_device(table, table_key, resident, positions, counts, rows, fill):
    """``(images, rows, 3, 3)`` rotations gathered entirely on the device.

    The per-row global rotation indices live in one resident int32 table per class and
    iteration, so a chunk uploads only its per-image positions and counts instead of an
    ``(images, rows)`` index array. At 100k/256 K=4 that index upload was ~17 s of the
    iteration (job 13838987 stack samples: padded_rotations_from_table_device). Requires
    every image of the chunk to have come from the vectorized preparation; the values are
    the same table entries, so the result is bit-identical to the index-array gather.
    """
    import jax
    import jax.numpy as jnp

    fn = padded_rotations_from_resident_device.__dict__.get("_compiled")
    if fn is None:
        fn = jax.jit(_padded_rotations_from_resident_impl, static_argnames=("rows",))
        padded_rotations_from_resident_device.__dict__["_compiled"] = fn
    rot_flat, row_bounds = _resident_row_indices_device(resident)
    return fn(
        _rotation_table_device(table, table_key),
        rot_flat,
        row_bounds,
        jnp.asarray(np.asarray(positions, dtype=np.int32)),
        jnp.asarray(np.asarray(counts, dtype=np.int32)),
        jnp.asarray(np.asarray(fill, dtype=table.dtype)),
        rows=int(rows),
    )


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
            padded_candidate_mask[row, :cnt, :] = _candidate_mask_to_dense(
                per_image_inputs["candidate_mask"][image_idx]
            )
            padded_parent_map[row, :cnt] = per_image_inputs["parent_map"][image_idx]
        padded_rotation_indices[row, :cnt] = per_image_inputs["oversampled_rot_indices"][image_idx]

    rotation_table = per_image_inputs.get("rotation_table") if isinstance(per_image_inputs, dict) else None
    resident_rows = None
    if isinstance(per_image_inputs, dict) and resident_hypothesis_tables_enabled():
        candidate = per_image_inputs.get("resident_hypothesis")
        if candidate is not None and candidate.get("rot_indices_flat") is not None:
            row_positions = np.asarray(candidate["positions"], dtype=np.int64)[padded_image_indices[:n_real]]
            if np.all(row_positions >= 0):
                resident_rows = (candidate, np.concatenate(
                    [row_positions, np.full(batch - n_real, -1, dtype=np.int64)]
                ) if batch > n_real else row_positions)
    if (
        device_rotations
        and rotation_table is not None
        and np.dtype(rotation_table.dtype) == np.dtype(rotation_dtype)
        and rotations_by_index_enabled()
    ):
        if resident_rows is not None:
            padded_rotations = padded_rotations_from_resident_device(
                rotation_table,
                per_image_inputs["rotation_table_key"],
                resident_rows[0],
                resident_rows[1],
                actual_counts,
                bucket_size,
                np.eye(3, dtype=rotation_dtype),
            )
        else:
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


IMAGE_CAPACITY_ENV = "RELAX_SPARSE_PASS2_IMAGE_CAPACITY"
IMAGE_CAPACITY_FLOOR = 16
IMAGE_CAPACITY_MAX_GROWTH_ENV = "RELAX_SPARSE_PASS2_IMAGE_CAPACITY_MAX_GROWTH"
DEFAULT_IMAGE_CAPACITY_MAX_GROWTH = 2.0


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

    n_images = int(n_images)
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
