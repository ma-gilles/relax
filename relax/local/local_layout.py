"""Per-image local hypothesis layout and bucketization helpers."""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass, replace
from typing import NamedTuple

import numpy as np
from recovar import utils

from relax.healpix_sampling import euler_angles_to_matrix
from relax.helpers.batch_planning import (
    plan_consecutive_padded_batches,
)
from relax.helpers.orientation_priors import make_relion_translation_log_prior
from relax.helpers.shape_buckets import coarse_bucket, power_bucket
from relax.sampling import (
    _compute_oversampled_rotation_grid,
    _relion_mstep_rotations_from_eulers,
    _wrapped_abs_diff_deg,
    apply_relion_rotation_perturbation_to_eulers,
    build_local_search_grid_metadata,
    get_local_rotation_grid_fast,
    get_oversampled_rotation_grid_from_samples,
    get_oversampled_translation_grid,
    infer_translation_step,
    rotation_grid_n_in_planes,
    rotation_grid_size,
    rotation_indices_to_relion_eulers,
    unique_nonnegative_ids,
)
from relax.scoring.significant_samples import significant_sample_ids
from relax.symmetry import canonicalize_rotational_symmetry, rotational_operators

EXACT_LOCAL_BUCKET_QUANTUM_ENV = "RELAX_EXACT_LOCAL_BUCKET_QUANTUM"

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
EXACT_LOCAL_UNIFY_MAX_PADDED_ROWS_ENV = "RELAX_EXACT_LOCAL_UNIFY_MAX_PADDED_ROWS"

# When the bound above drops unification, each image keeps its own size, which at
# K=4 100k/256 produced 7 new bucket shapes at the all-data iteration and 67.4 s of
# XLA compilation (14 compiles of run_local_bucket_big_jit at ~4.74 s each). Capping
# the number of distinct sizes trades a little of the padding win back for fewer
# compiled shapes. 0 means uncapped, which is the measured 1426.1 s behavior.
EXACT_LOCAL_MAX_BUCKET_SIZE_CLASSES = 0
EXACT_LOCAL_MAX_BUCKET_SIZE_CLASSES_ENV = "RELAX_EXACT_LOCAL_MAX_BUCKET_SIZE_CLASSES"
EXACT_LOCAL_BUCKET_RADIX_ENV = "RELAX_EXACT_LOCAL_BUCKET_RADIX"
EXACT_LOCAL_BUCKET_MIN_QUANTUM = 256

LOCAL_IMAGE_CAPACITY_LADDER_ENV = "RELAX_LOCAL_IMAGE_CAPACITY_LADDER"
DEFAULT_LOCAL_IMAGE_CAPACITY_LADDER = (16, 32, 64, 128, 256)


def resolve_local_image_capacity_ladder(explicit=None) -> tuple[int, ...]:
    """Resolve the opt-in image-axis capacity ladder; ``()`` means off.

    The exact local engine's images-per-bucket capacity is
    ``min(image_batch_size, max_hypotheses_per_microbatch // bucket_rotations)``.
    ``image_batch_size`` comes from a memory estimate that moves by a few images
    between iterations, and it is the leading axis of every per-bucket program,
    so a one-image change recompiles the whole bucket program set. Snapping the
    capacity to a fixed ladder makes that axis stable across iterations.

    ``explicit`` beats the environment. Accepted values: ``None`` (consult the
    environment), ``False``/``""``/``"0"``/``"off"`` (ladder off), ``True``/
    ``"1"``/``"on"``/``"auto"`` (the default ladder) or an explicit sequence or
    comma-separated string of positive capacities.

    The bucket planners below treat ``None`` as off, because they are also called by
    ``recovar/em/ppca_refinement/local_dataset.py``, a pipeline with its own
    validation that an EM-scoped environment variable must not re-bucket.
    """

    source = "local_image_capacity_ladder"
    raw = explicit
    if raw is None:
        source = LOCAL_IMAGE_CAPACITY_LADDER_ENV
        raw = os.environ.get(LOCAL_IMAGE_CAPACITY_LADDER_ENV, "")
    if raw is False:
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


def _planner_image_capacity_ladder(explicit) -> tuple[int, ...]:
    """Resolve a planner's ladder argument; ``None`` means off, not "ask the env"."""

    if explicit is None:
        return ()
    return resolve_local_image_capacity_ladder(explicit)


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


def _nonnegative_count_override(raw: str, env_name: str, default: int) -> int:
    """``raw`` (an environment value) as a non-negative count; blank is ``default``, a non-integer raises."""

    token = raw.strip()
    if not token:
        return int(default)
    value = int(token)
    if value < 0:
        raise ValueError(f"{env_name} must be non-negative")
    return value


def _exact_local_max_bucket_size_classes() -> int:
    """Cap on distinct bucket sizes when unification is dropped; 0 means uncapped."""

    return _nonnegative_count_override(
        os.environ.get(EXACT_LOCAL_MAX_BUCKET_SIZE_CLASSES_ENV, ""),
        EXACT_LOCAL_MAX_BUCKET_SIZE_CLASSES_ENV,
        EXACT_LOCAL_MAX_BUCKET_SIZE_CLASSES,
    )


def _cap_bucket_size_classes(sizes: np.ndarray) -> np.ndarray:
    """Round sizes up onto at most N distinct values, never below the true need."""

    max_classes = _exact_local_max_bucket_size_classes()
    distinct = np.unique(sizes)
    if max_classes <= 0 or distinct.size <= max_classes:
        return sizes
    # Keep representative ceilings spread over the observed ladder, always including
    # the largest so no neighborhood is truncated, then round every size up onto them.
    picks = np.unique(np.linspace(0, distinct.size - 1, max_classes).round().astype(int))
    kept = distinct[picks]
    kept[-1] = distinct[-1]
    return kept[np.clip(np.searchsorted(kept, sizes, side="left"), 0, kept.size - 1)]


def _exact_local_unify_max_padded_rows() -> int:
    """Rows that global bucket-size unification may add before it is dropped.

    See ``EXACT_LOCAL_UNIFY_MAX_PADDED_ROWS`` for the measurement behind the default.
    """

    return _nonnegative_count_override(
        os.environ.get(EXACT_LOCAL_UNIFY_MAX_PADDED_ROWS_ENV, ""),
        EXACT_LOCAL_UNIFY_MAX_PADDED_ROWS_ENV,
        EXACT_LOCAL_UNIFY_MAX_PADDED_ROWS,
    )


def _exact_local_large_bucket_quantum(rotation_block_size: int, explicit: int | None = None) -> int:
    """Return the large-neighborhood bucket quantum for exact local search."""

    if explicit is not None:
        return max(1, int(explicit))
    env_quantum = os.environ.get(EXACT_LOCAL_BUCKET_QUANTUM_ENV, "")
    if env_quantum:
        return max(1, int(env_quantum))
    engine_cap = int(_local_search_engine_rotation_block_size(rotation_block_size))
    return max(EXACT_LOCAL_BUCKET_MIN_QUANTUM, engine_cap)


@dataclass(frozen=True)
class LocalHypothesisLayout:
    """Flat per-image local hypothesis storage."""

    n_global_rotations: int
    n_pixels: int
    n_psi: int
    rotation_offsets: np.ndarray
    rotation_ids_flat: np.ndarray
    rotations_flat: np.ndarray
    rotation_log_priors_flat: np.ndarray
    rotation_counts: np.ndarray
    translation_grid: np.ndarray
    translation_log_priors: np.ndarray
    rotation_posterior_ids_flat: np.ndarray | None = None
    sample_mask_bits: np.ndarray | None = None  # uint8, translation bits packed little-endian
    mstep_rotations_flat: np.ndarray | None = None
    source_eulers_flat: np.ndarray | None = None
    # The rows are oversampled children of parents (RELION's oversampled orientations): rows and M-step rows
    # are the host inverse matrices of ``source_eulers_flat``, and ``rotation_ids_flat`` index the fine grid.
    oversampled_rows: bool = False
    # Other rows were built from ``rotation_ids_flat`` on the local grid of this (healpix_order,
    # random_perturbation, angular_sampling_deg): the builder's arguments, which a caller's grid need not repeat.
    id_rows_source: tuple[int, float, float | None] | None = None
    symmetry: str = "C1"
    # Class3D (K>1) local searches: every image's rows repeated once per class, class-major within the
    # image (:func:`expand_local_layout_classes`); ``row_class_flat`` is each row's class. None: K=1.
    row_class_flat: np.ndarray | None = None
    n_classes: int = 1

    def sample_mask_rows(self, start=0, stop=None) -> np.ndarray | None:
        """Expand only the requested rotation rows to the kernel's boolean mask."""
        if self.sample_mask_bits is None:
            return None
        return np.unpackbits(
            self.sample_mask_bits[start:stop], axis=1,
            count=int(self.translation_grid.shape[0]), bitorder="little",
        ).view(np.bool_)

    @property
    def n_images(self) -> int:
        return int(self.rotation_counts.shape[0])

    @property
    def total_local_rotations(self) -> int:
        return int(self.rotation_ids_flat.shape[0])


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


def _resolve_prior_rotations(prior_rotations: np.ndarray, healpix_order: int, grid_metadata):
    """Return float64 RELION eulers and rotation matrices for local-support construction.

    RELION selects local supports from the priors in double (RFLOAT, ``selectOrientationsWithNonZeroPriorProbability``,
    healpix_sampling.cpp:710-790, via ``Euler_angles2direction``), also in its GPU build
    (acc_ml_optimiser_impl.h:611), so the eulers stay float64 and the matrices are ``Euler_angles2matrix``.
    """

    prior_rotations = np.asarray(prior_rotations)
    if prior_rotations.ndim == 0:
        prior_rotations = prior_rotations.reshape(1)

    if prior_rotations.ndim == 1:
        if "eulers_full" in grid_metadata:
            prior_eulers = np.asarray(grid_metadata["eulers_full"], dtype=np.float64)[prior_rotations.astype(np.int64)]
        else:
            prior_eulers = rotation_indices_to_relion_eulers(
                prior_rotations.astype(np.int64),
                healpix_order,
                symmetry=str(grid_metadata.get("symmetry", "C1")),
            )
        prior_eulers = np.asarray(prior_eulers, dtype=np.float64).reshape(-1, 3)
        return prior_eulers, euler_angles_to_matrix(prior_eulers)
    if prior_rotations.ndim == 2 and prior_rotations.shape[-1] == 3:
        prior_eulers = np.asarray(prior_rotations, dtype=np.float64).reshape(-1, 3)
        return prior_eulers, euler_angles_to_matrix(prior_eulers)
    prior_rotation_mats = np.asarray(prior_rotations, dtype=np.float64).reshape(-1, 3, 3)
    prior_eulers = np.asarray(utils.R_to_relion(prior_rotation_mats, degrees=True), dtype=np.float64)
    return prior_eulers, prior_rotation_mats


def _local_selector_chunk_size(n_images: int, n_pixels: int, n_psi: int, use_direction: bool, use_psi: bool) -> int:
    explicit = os.environ.get("RELAX_LOCAL_SELECTOR_CHUNK_SIZE", "")
    if explicit:
        return max(1, min(int(n_images), int(explicit)))

    max_elements = int(os.environ.get("RELAX_LOCAL_SELECTOR_MAX_ELEMENTS", "16000000"))
    per_image_elements = 0
    if use_direction:
        per_image_elements += int(n_pixels)
    if use_psi:
        per_image_elements += int(n_psi)
    if per_image_elements <= 0:
        return max(1, int(n_images))
    return max(1, min(int(n_images), max_elements // per_image_elements))


def _segment_starts(counts: np.ndarray) -> np.ndarray:
    starts = np.zeros(counts.shape[0], dtype=np.int64)
    if counts.shape[0]:
        starts[1:] = np.cumsum(counts, dtype=np.int64)[:-1]
    return starts


def _dense_direction_dots(prior_dir_vecs, dir_vecs, rows):
    return prior_dir_vecs[rows] @ dir_vecs.T


class _SparseCandidates(NamedTuple):
    """Candidate ``(image, point)`` pairs of a chunk, image-major with ascending points, and their values."""

    n_images: int
    image: np.ndarray
    point: np.ndarray
    values: np.ndarray


def _direction_cone_candidates(prior_dir_vecs, dir_vecs, dir_z_order, dir_z_sorted, cone_rad, cos_prefilter):
    """The grid directions within ``cone_rad`` of each prior direction, with their dot products.

    A direction within the cone has a polar angle within ``cone_rad`` of the
    prior's, so only the grid's latitude band (a contiguous run of the grid
    sorted by z) is tested: about 6% of an order-5 grid at the 10097 local
    cones instead of every direction. The dot products of the band are formed
    elementwise and kept where ``dot >= cos_prefilter``, the dense prefilter's
    test; a pairwise sum can differ from a BLAS dot product in the last bit.
    """

    n_images = int(prior_dir_vecs.shape[0])
    polar = np.arccos(np.clip(prior_dir_vecs[:, 2], -1.0, 1.0))
    z_high = np.cos(np.clip(polar - cone_rad, 0.0, np.pi)) + 1e-12
    z_low = np.cos(np.clip(polar + cone_rad, 0.0, np.pi)) - 1e-12
    lo = np.searchsorted(dir_z_sorted, z_low, side="left")
    hi = np.searchsorted(dir_z_sorted, z_high, side="right")
    band = (hi - lo).astype(np.int64)
    image = np.repeat(np.arange(n_images, dtype=np.int64), band)
    position = lo.astype(np.int64)[image] + np.arange(image.shape[0], dtype=np.int64) - np.repeat(
        _segment_starts(band), band
    )
    point = dir_z_order[position]
    prior = prior_dir_vecs[image]
    grid = dir_vecs[point]
    dots = prior[:, 0] * grid[:, 0] + prior[:, 1] * grid[:, 1] + prior[:, 2] * grid[:, 2]
    inside = dots >= cos_prefilter
    image, point, dots = image[inside], point[inside], dots[inside]
    order = np.argsort(image * np.int64(dir_vecs.shape[0]) + point, kind="stable")
    return _SparseCandidates(n_images, image[order], point[order], dots[order])


def _chunk_local_supports(values, candidates, to_distance, cutoff_deg, sigma_deg):
    """Each image's kept grid points and their normalized log Gaussian prior, for a chunk of images.

    The chunk form of the per-image statements: a point is kept when its
    distance (``to_distance`` of ``values``, or ``values`` itself) is below
    ``cutoff_deg``, among the ``candidates`` prefilter when given; an image
    with no kept point keeps its nearest point with log prior 0. Returns
    ``(image, point, log_prior)`` flat arrays, image-major with ascending point
    ids, which is the order the per-image loop produced. The per-image
    normalization is a segmented sum, so the log priors can differ from the
    loop's pairwise sum in the last float64 bit before their float32 rounding.
    """

    if isinstance(candidates, _SparseCandidates):
        # Candidate (image, point) pairs already listed, image-major with
        # ascending points, with their values; ``values`` gives the dense
        # values of an image that keeps no candidate.
        n_images = candidates.n_images
        image, point = candidates.image, candidates.point
        distance = to_distance(candidates.values)
    elif candidates is None:
        n_images = int(values.shape[0])
        image = np.repeat(np.arange(n_images, dtype=np.int64), values.shape[1])
        point = np.tile(np.arange(values.shape[1], dtype=np.int64), n_images)
        distance = np.asarray(values, dtype=np.float64).reshape(-1)
    else:
        n_images = int(values.shape[0])
        image, point = np.nonzero(candidates)
        distance = to_distance(values[image, point])
    inside = distance < cutoff_deg
    image, point, distance = image[inside], point[inside], distance[inside]
    n_kept = np.bincount(image, minlength=n_images)
    # float32 as _normalized_log_weights returns it; the outer product adds it
    # to the other axis's prior in whichever dtype the two promote to.
    log_prior = np.zeros(distance.shape[0], dtype=np.float32)
    kept = n_kept > 0
    if distance.size:
        weights = np.exp(-0.5 * (np.asarray(distance, dtype=np.float64) / float(sigma_deg)) ** 2)
        starts = _segment_starts(n_kept[kept])
        totals = np.add.reduceat(weights, starts)
        per_point_total = np.repeat(totals, n_kept[kept])
        normalized = weights / per_point_total
        bad = ~np.isfinite(totals) | (totals <= 0.0)
        if np.any(bad):
            bad_points = np.repeat(bad, n_kept[kept])
            normalized[bad_points] = 1.0 / np.repeat(n_kept[kept], n_kept[kept])[bad_points]
        log_prior = np.log(np.clip(normalized, np.finfo(np.float32).tiny, None)).astype(np.float32)
    empty = np.flatnonzero(~kept)
    if empty.size:
        empty_values = values(empty) if callable(values) else values[empty]
        distances = empty_values if to_distance is None else to_distance(empty_values)
        image = np.concatenate([image, empty])
        point = np.concatenate([point, np.argmin(distances, axis=1)])
        log_prior = np.concatenate([log_prior, np.zeros(empty.size, dtype=np.float32)])
        order = np.argsort(image, kind="stable")
        image, point, log_prior = image[order], point[order], log_prior[order]
    return image, point.astype(np.int64), log_prior


def _chunk_uniform_supports(n_images: int, n_points: int, *, dtype):
    """Every grid point for every image with a uniform log prior, as ``(image, point, log_prior)``."""

    image = np.repeat(np.arange(n_images, dtype=np.int64), n_points)
    point = np.tile(np.arange(n_points, dtype=np.int64), n_images)
    log_prior = np.full(image.shape[0], -np.log(max(n_points, 1)), dtype=dtype)
    return image, point, log_prior


def _chunk_outer_supports(n_images, psi_supports, dir_supports, *, n_pixels: int, dtype):
    """Each image's psi x direction products, psi-major, as the per-image loop built them.

    Returns the flat rotation ids ``psi * n_pixels + direction``, their log
    priors ``psi_log_prior + direction_log_prior`` and the per-image counts.
    """

    psi_image, psi_ids, psi_log_prior = psi_supports
    dir_image, dir_ids, dir_log_prior = dir_supports
    n_psi = np.bincount(psi_image, minlength=n_images).astype(np.int64)
    n_dir = np.bincount(dir_image, minlength=n_images).astype(np.int64)
    counts = n_psi * n_dir
    total = int(counts.sum())
    image = np.repeat(np.arange(n_images, dtype=np.int64), counts)
    local = np.arange(total, dtype=np.int64) - np.repeat(_segment_starts(counts), counts)
    image_dirs = n_dir[image]
    psi_row = _segment_starts(n_psi)[image] + local // image_dirs
    dir_row = _segment_starts(n_dir)[image] + local % image_dirs
    ids = psi_ids[psi_row] * np.int64(n_pixels) + dir_ids[dir_row]
    log_prior = (psi_log_prior[psi_row] + dir_log_prior[dir_row]).astype(dtype)
    return ids, log_prior, counts.astype(np.int32)


def _build_factorized_local_entries(
    prior_rotations: np.ndarray,
    healpix_order: int,
    sigma_rot: float,
    sigma_psi: float,
    grid_metadata,
    *,
    dtype: np.dtype = np.float32,
):
    """Build exact per-image local supports for factorized HEALPix x psi grids."""

    prior_eulers, prior_rotation_mats = _resolve_prior_rotations(prior_rotations, healpix_order, grid_metadata)
    dir_vecs = np.asarray(grid_metadata["dir_vecs"], dtype=np.float64)
    psi_deg_grid = np.asarray(grid_metadata["psi_deg"], dtype=np.float64)
    n_pixels = int(grid_metadata["n_pixels"])
    symmetry = canonicalize_rotational_symmetry(str(grid_metadata.get("symmetry", "C1")))
    symmetry_operators = None if symmetry == "C1" else rotational_operators(symmetry, dtype=np.float64)

    prior_dir_vecs = np.asarray(prior_rotation_mats[:, 2, :], dtype=np.float64)
    prior_dir_norm = np.linalg.norm(prior_dir_vecs, axis=1, keepdims=True)
    prior_dir_norm = np.where(prior_dir_norm > 0.0, prior_dir_norm, 1.0)
    prior_dir_vecs = prior_dir_vecs / prior_dir_norm
    prior_psi_deg = np.mod(np.asarray(prior_eulers[:, 2], dtype=np.float64), 360.0)

    sigma_rot_deg = float(np.rad2deg(sigma_rot))
    sigma_psi_deg = float(np.rad2deg(sigma_psi))
    # RELION widens the direction cone with max(sigma_rot, sigma_tilt).
    # In this SPA path sigma_tilt == sigma_rot; sigma_psi only controls
    # in-plane support and must not widen directions.
    biggest_sigma_deg = sigma_rot_deg
    cutoff_dir_deg = 3.0 * biggest_sigma_deg
    cutoff_psi_deg = 3.0 * sigma_psi_deg
    # cos of the cone half-angle widened by 1e-4 relative: every direction the
    # exact degree test keeps has a larger dot product (see the chunk loop).
    widened_cutoff_rad = np.deg2rad(cutoff_dir_deg) * (1.0 + 1e-4) + 1e-9
    cos_prefilter = -np.inf if widened_cutoff_rad >= np.pi else float(np.cos(widened_cutoff_rad))

    rotation_ids_parts: list[np.ndarray] = []
    log_prior_parts: list[np.ndarray] = []
    n_images = int(prior_eulers.shape[0])
    counts = np.zeros(n_images, dtype=np.int32)
    offsets = np.zeros(n_images + 1, dtype=np.int64)
    chunk_size = _local_selector_chunk_size(
        n_images,
        n_pixels,
        int(grid_metadata["n_psi"]),
        sigma_rot_deg > 0.0,
        sigma_psi_deg > 0.0,
    )
    running_offset = 0
    # Without symmetry the direction cone is searched in the grid's latitude
    # band (_direction_cone_candidates) instead of against every direction.
    banded = symmetry_operators is None and np.isfinite(cos_prefilter)
    if banded:
        dir_z_order = np.argsort(dir_vecs[:, 2], kind="stable").astype(np.int64)
        dir_z_sorted = dir_vecs[dir_z_order, 2]

    for chunk_start in range(0, n_images, chunk_size):
        chunk_stop = min(n_images, chunk_start + chunk_size)
        if sigma_rot_deg > 0.0 and banded:
            chunk_dirs = prior_dir_vecs[chunk_start:chunk_stop]
            candidate_chunk = _direction_cone_candidates(
                chunk_dirs, dir_vecs, dir_z_order, dir_z_sorted, widened_cutoff_rad, cos_prefilter
            )
            # An image that keeps no candidate falls back to its nearest
            # direction, which needs its dense row.
            dots = functools.partial(_dense_direction_dots, chunk_dirs, dir_vecs)
        elif sigma_rot_deg > 0.0:
            if symmetry_operators is None:
                dots = prior_dir_vecs[chunk_start:chunk_stop] @ dir_vecs.T
            else:
                dots = np.max(
                    np.einsum(
                        "di,sij,bj->bsd",
                        dir_vecs,
                        symmetry_operators,
                        prior_dir_vecs[chunk_start:chunk_stop],
                        optimize=True,
                    ),
                    axis=1,
                )
            # Only directions inside the cone need their angle. A cosine
            # prefilter with a margin keeps a superset of them, the exact test
            # below runs on the same arccos values as before, and the
            # arccos over every direction of the grid is skipped.
            candidate_chunk = dots >= cos_prefilter
        else:
            dots = None

        if sigma_psi_deg > 0.0:
            diffpsi_chunk = _wrapped_abs_diff_deg(
                psi_deg_grid[None, :],
                prior_psi_deg[chunk_start:chunk_stop, None],
            )
        else:
            diffpsi_chunk = None

        n_chunk = chunk_stop - chunk_start
        if sigma_rot_deg > 0.0:
            dir_image, dir_ids, dir_log_prior = _chunk_local_supports(
                dots,
                candidate_chunk,
                lambda values: np.rad2deg(np.arccos(np.clip(values, -1.0, 1.0))),
                cutoff_dir_deg,
                biggest_sigma_deg,
            )
        else:
            dir_image, dir_ids, dir_log_prior = _chunk_uniform_supports(n_chunk, n_pixels, dtype=dtype)
        if sigma_psi_deg > 0.0:
            psi_image, psi_ids, psi_log_prior = _chunk_local_supports(
                diffpsi_chunk, None, None, cutoff_psi_deg, sigma_psi_deg
            )
        else:
            psi_image, psi_ids, psi_log_prior = _chunk_uniform_supports(
                n_chunk, int(grid_metadata["n_psi"]), dtype=dtype
            )
        local_ids, local_log_prior, chunk_counts = _chunk_outer_supports(
            n_chunk,
            (psi_image, psi_ids, psi_log_prior),
            (dir_image, dir_ids, dir_log_prior),
            n_pixels=n_pixels,
            dtype=dtype,
        )
        counts[chunk_start:chunk_stop] = chunk_counts
        offsets[chunk_start + 1 : chunk_stop + 1] = running_offset + np.cumsum(chunk_counts, dtype=np.int64)
        running_offset = int(offsets[chunk_stop])
        rotation_ids_parts.append(local_ids)
        log_prior_parts.append(local_log_prior)

    rotation_ids_flat = _flat_parts(rotation_ids_parts, empty_shape=0, dtype=np.int64, cast=np.int64)
    rotation_log_priors_flat = _flat_parts(log_prior_parts, empty_shape=0, dtype=dtype)
    return offsets, counts, rotation_ids_flat, rotation_log_priors_flat


def _build_parent_expanded_local_entries(
    prior_rotations: np.ndarray,
    fine_healpix_order: int,
    sigma_rot: float,
    sigma_psi: float,
    *,
    oversampling_order: int,
    rotation_log_prior: np.ndarray | None = None,
    random_perturbation: float = 0.0,
    generate_relion_mstep_rotations: bool = False,
    dtype: np.dtype = np.float32,
    symmetry: str = "C1",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Build RELION-style local support by expanding selected coarse parents.

    RELION local search first calls
    ``selectOrientationsWithNonZeroPriorProbability`` on the current coarse
    sampling object, then ``getOrientations`` expands those selected coarse
    direction/psi parents into oversampled children. The child orientations
    inherit the parent prior; the prior is not redistributed over children.
    """

    if oversampling_order <= 0:
        raise ValueError("oversampling_order must be positive for parent-expanded local support")
    parent_order = fine_healpix_order - oversampling_order
    if parent_order < 0:
        raise ValueError(
            "fine_healpix_order must be >= oversampling_order for parent-expanded local support; "
            f"got fine_healpix_order={fine_healpix_order}, oversampling_order={oversampling_order}"
        )

    symmetry = canonicalize_rotational_symmetry(symmetry)
    parent_metadata = build_local_search_grid_metadata(parent_order, symmetry=symmetry)
    rotation_log_prior_np = None
    if rotation_log_prior is not None:
        rotation_log_prior_np = np.asarray(rotation_log_prior, dtype=dtype)
        expected_parent_size = rotation_grid_size(parent_order, symmetry)
        if rotation_log_prior_np.shape[0] != expected_parent_size:
            raise ValueError(
                "rotation_log_prior must have one value per parent-grid rotation "
                f"({expected_parent_size}) for parent-expanded local search; got {rotation_log_prior_np.shape}"
            )
    parent_offsets, parent_counts, parent_ids_flat, parent_log_priors_flat = _build_factorized_local_entries(
        prior_rotations,
        parent_order,
        sigma_rot,
        sigma_psi,
        parent_metadata,
        dtype=dtype,
    )

    n_images = int(parent_counts.shape[0])
    offsets = np.zeros(n_images + 1, dtype=np.int64)
    counts = np.zeros(n_images, dtype=np.int32)
    rotation_ids_parts: list[np.ndarray] = []
    log_prior_parts: list[np.ndarray] = []
    rotations_parts: list[np.ndarray] = []
    mstep_rotations_parts: list[np.ndarray] = []
    source_eulers_parts = []
    posterior_ids_parts: list[np.ndarray] = []
    running_offset = 0

    for image_idx in range(n_images):
        start = int(parent_offsets[image_idx])
        stop = int(parent_offsets[image_idx + 1])
        parent_ids = np.asarray(parent_ids_flat[start:stop], dtype=np.int64)
        parent_log_prior = np.asarray(parent_log_priors_flat[start:stop], dtype=dtype)
        if rotation_log_prior_np is not None:
            parent_log_prior = parent_log_prior + rotation_log_prior_np[parent_ids]
        oversampled = get_oversampled_rotation_grid_from_samples(
            parent_ids,
            parent_order,
            oversampling_order=oversampling_order,
            random_perturbation=float(random_perturbation),
            return_rotation_indices=True,
            return_source_eulers=True,
            return_mstep_rotations=bool(generate_relion_mstep_rotations),
            rotation_index_order="recovar",
            dtype=dtype,
            symmetry=symmetry,
        )
        source_eulers_parts.append(oversampled[-1])
        child_rotations, parent_map, child_ids = oversampled[:3]
        child_mstep_rotations = oversampled[3] if bool(generate_relion_mstep_rotations) else None
        parent_map = np.asarray(parent_map, dtype=np.int64)
        child_ids = np.asarray(child_ids, dtype=np.int64)
        child_log_prior = parent_log_prior[parent_map].astype(dtype, copy=False)

        counts[image_idx] = int(child_ids.shape[0])
        running_offset += int(child_ids.shape[0])
        offsets[image_idx + 1] = running_offset
        rotation_ids_parts.append(child_ids)
        log_prior_parts.append(child_log_prior)
        rotations_parts.append(np.asarray(child_rotations, dtype=dtype))
        if child_mstep_rotations is not None:
            mstep_rotations_parts.append(np.asarray(child_mstep_rotations, dtype=dtype))
        if symmetry != "C1":
            # Non-C1 child ids deliberately refer to the unreduced fine
            # HEALPix lattice: an oversampled child can cross the parent ASU
            # boundary and therefore need not have a row in the reduced fine
            # grid.  Posterior bookkeeping must not use those ids.  Aggregate
            # each child back to its exact reduced coarse parent, matching the
            # adaptive-local pass-2 layout and avoiding an arbitrary nearest
            # reduced-grid remap.
            posterior_ids_parts.append(parent_ids[parent_map].astype(np.int64, copy=False))

    rotation_ids_flat = _flat_parts(rotation_ids_parts, empty_shape=0, dtype=np.int64, cast=np.int64)
    rotation_log_priors_flat = _flat_parts(log_prior_parts, empty_shape=0, dtype=dtype)
    rotations_flat = _flat_parts(rotations_parts, empty_shape=(0, 3, 3), dtype=dtype)
    mstep_rotations_flat = (
        np.concatenate(mstep_rotations_parts, axis=0)
        if mstep_rotations_parts
        else (np.zeros((0, 3, 3), dtype=dtype) if generate_relion_mstep_rotations else None)
    )
    source_eulers_flat = (
        np.concatenate(source_eulers_parts)
        if source_eulers_parts and all(x is not None for x in source_eulers_parts)
        else None
    )
    posterior_ids_flat = (
        np.concatenate(posterior_ids_parts, axis=0) if posterior_ids_parts else None
    )
    return (
        offsets,
        counts,
        rotation_ids_flat,
        rotation_log_priors_flat,
        rotations_flat,
        mstep_rotations_flat,
        source_eulers_flat,
        posterior_ids_flat,
    )


def _rotation_eulers_from_grid_metadata(
    rotation_ids: np.ndarray,
    grid_metadata,
    *,
    dtype=np.float32,
) -> np.ndarray:
    """Return canonical RELION Euler angles for selected rotation ids only."""

    rotation_ids = np.asarray(rotation_ids, dtype=np.int64).reshape(-1)
    if rotation_ids.size == 0:
        return np.zeros((0, 3), dtype=dtype)
    if "eulers_full" in grid_metadata:
        return np.asarray(grid_metadata["eulers_full"], dtype=dtype)[rotation_ids]
    if str(grid_metadata["mode"]) != "factorized":
        raise ValueError("Selected rotation eulers require factorized metadata or eulers_full")
    n_pixels = int(grid_metadata["n_pixels"])
    pixel_idx = rotation_ids % n_pixels
    psi_idx = rotation_ids // n_pixels
    return np.stack(
        [
            np.asarray(grid_metadata["rot_deg"], dtype=dtype)[pixel_idx],
            np.asarray(grid_metadata["tilt_deg"], dtype=dtype)[pixel_idx],
            np.asarray(grid_metadata["psi_deg"], dtype=dtype)[psi_idx],
        ],
        axis=1,
    ).astype(dtype, copy=False)


def selected_rotation_matrices(
    rotation_ids: np.ndarray,
    rotation_grid_rotations: np.ndarray | None,
    grid_metadata,
    *,
    random_perturbation: float = 0.0,
    angular_sampling_deg: float | None = None,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Build matrices for selected local ids without materializing the full grid."""

    rotation_ids = np.asarray(rotation_ids, dtype=np.int64).reshape(-1)
    if rotation_ids.size == 0:
        return np.zeros((0, 3, 3), dtype=dtype)
    if rotation_grid_rotations is not None:
        return np.asarray(rotation_grid_rotations, dtype=dtype).reshape(-1, 3, 3)[rotation_ids]
    unique_ids, inverse = unique_nonnegative_ids(rotation_ids)
    # Stage eulers at the requested dtype before apply_relion_rotation_perturbation_to_eulers
    # re-derives float64 internally regardless; truncating to float32 here
    # first (the previous unconditional default) would discard precision
    # that a dtype=float64 caller asked to keep, even though it leaves the
    # existing float32 default path's output bit-for-bit unchanged. Matches
    # the mstep sibling below, which already does this.
    selected_eulers = _rotation_eulers_from_grid_metadata(unique_ids, grid_metadata, dtype=dtype)
    if abs(float(random_perturbation)) > 1e-12:
        if angular_sampling_deg is None:
            raise ValueError("angular_sampling_deg is required when random_perturbation is nonzero")
        rotations, _ = apply_relion_rotation_perturbation_to_eulers(
            selected_eulers,
            float(random_perturbation),
            float(angular_sampling_deg),
            dtype=dtype,
        )
    else:
        # Preserve RELION's accelerated-path handoff: host RFLOAT inverse
        # matrices are cast to XFLOAT before scoring on the device (a no-op
        # cast under ACC_DOUBLE_PRECISION, when dtype=float64).
        rotations, _ = apply_relion_rotation_perturbation_to_eulers(
            selected_eulers,
            0.0,
            0.0,
            dtype=dtype,
        )
    return rotations.astype(dtype, copy=False)[inverse]


def _selected_mstep_rotation_matrices(
    rotation_ids: np.ndarray,
    rotation_grid_mstep_rotations: np.ndarray | None,
    grid_metadata,
    *,
    random_perturbation: float = 0.0,
    angular_sampling_deg: float | None = None,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Build RELION host-path adjoint matrices for selected local ids."""

    rotation_ids = np.asarray(rotation_ids, dtype=np.int64).reshape(-1)
    if rotation_ids.size == 0:
        return np.zeros((0, 3, 3), dtype=dtype)
    if rotation_grid_mstep_rotations is not None:
        return np.asarray(rotation_grid_mstep_rotations, dtype=dtype).reshape(-1, 3, 3)[rotation_ids]
    unique_ids, inverse = unique_nonnegative_ids(rotation_ids)
    selected_eulers = _rotation_eulers_from_grid_metadata(unique_ids, grid_metadata, dtype=np.float64)
    if angular_sampling_deg is None:
        if abs(float(random_perturbation)) > 1e-12:
            raise ValueError("angular_sampling_deg is required when random_perturbation is nonzero")
        angular_sampling_deg = 0.0
    mstep_rotations, _ = apply_relion_rotation_perturbation_to_eulers(
        selected_eulers,
        float(random_perturbation),
        float(angular_sampling_deg),
        dtype=dtype,
    )
    return np.asarray(mstep_rotations, dtype=dtype)[inverse]


def local_layout_host_rotations(layout: LocalHypothesisLayout, *, mstep: bool) -> np.ndarray:
    """A layout's scoring (or, with ``mstep``, M-step) rows as float64 matrices, by the rule that built them.

    The rows of images on another grid or magnified are composed from these and cast once
    (:func:`relax.sampling.project_rows`). Oversampled children are the host inverse matrices of their
    source Euler rows (:func:`relax.sampling.get_oversampled_rotation_grid_from_samples`); other rows are
    rebuilt from ``rotation_ids_flat`` with the grid and perturbation the layout was built with
    (``id_rows_source``), not the caller's.
    """

    if layout.oversampled_rows:
        if layout.source_eulers_flat is None:
            raise ValueError("oversampled local rows carry no source Euler rows to rebuild them from")
        return _relion_mstep_rotations_from_eulers(np.asarray(layout.source_eulers_flat), dtype=np.float64)
    if layout.id_rows_source is None:
        raise ValueError("local rows record no builder arguments to rebuild them from")
    healpix_order, random_perturbation, angular_sampling_deg = layout.id_rows_source
    builder = _selected_mstep_rotation_matrices if mstep else selected_rotation_matrices
    return builder(
        layout.rotation_ids_flat,
        None,
        build_local_search_grid_metadata(healpix_order, symmetry=layout.symmetry),
        random_perturbation=random_perturbation,
        angular_sampling_deg=angular_sampling_deg,
        dtype=np.float64,
    )


def _flat_parts(parts, *, empty_shape, dtype, cast=None):
    """Concatenate per-image layout parts along axis 0, or the typed empty array when no image contributed.

    ``cast`` recasts the concatenation (the rotation ids are carried as int64).
    """

    if not parts:
        return np.zeros(empty_shape, dtype=dtype)
    flat = np.concatenate(parts, axis=0)
    return flat if cast is None else flat.astype(cast, copy=False)


def build_local_hypothesis_layout(
    prior_rotations: np.ndarray,
    rotation_grid_rotations: np.ndarray | None,
    sigma_rot: float,
    sigma_psi: float,
    healpix_order: int,
    translations: np.ndarray,
    prior_translations: np.ndarray,
    sigma_offset_angstrom: float,
    offset_range_pixels: float | None,
    voxel_size: float,
    *,
    grid_metadata,
    translation_prior_reference_translations: np.ndarray | None = None,
    rotation_log_prior: np.ndarray | None = None,
    rotation_grid_random_perturbation: float = 0.0,
    rotation_grid_angular_sampling_deg: float | None = None,
    local_parent_oversampling_order: int = 0,
    rotation_grid_mstep_rotations: np.ndarray | None = None,
    generate_relion_mstep_rotations: bool = False,
    dtype: np.dtype = np.float32,
) -> LocalHypothesisLayout:
    """Build exact per-image local neighborhoods and translation priors.

    ``dtype`` controls the precision of every rotation/translation/prior
    array built here (default float32, matching RELION's accelerated-GPU
    single-precision path). Pass ``np.float64`` to keep this local-search
    hypothesis grid genuinely double precision end to end; the caller is
    responsible for deriving this from ``use_float64_scoring`` /
    ``use_float64_projections`` so the default stays unchanged.
    """

    # The priors keep their own precision: local supports are selected in double (_resolve_prior_rotations).
    prior_rotations = np.asarray(prior_rotations)
    symmetry = canonicalize_rotational_symmetry(str(grid_metadata.get("symmetry", "C1")))
    if rotation_grid_rotations is not None:
        rotation_grid_rotations = np.asarray(rotation_grid_rotations, dtype=dtype).reshape(-1, 3, 3)
    if rotation_grid_mstep_rotations is not None:
        rotation_grid_mstep_rotations = np.asarray(rotation_grid_mstep_rotations, dtype=dtype).reshape(-1, 3, 3)
        expected_rotation_count = (
            int(rotation_grid_rotations.shape[0])
            if rotation_grid_rotations is not None
            else int(grid_metadata["n_pixels"]) * int(grid_metadata["n_psi"])
        )
        if int(rotation_grid_mstep_rotations.shape[0]) != expected_rotation_count:
            raise ValueError(
                "rotation_grid_mstep_rotations must match the scoring grid size: "
                f"{rotation_grid_mstep_rotations.shape[0]} vs {expected_rotation_count}",
            )
    generate_relion_mstep_rotations = bool(
        generate_relion_mstep_rotations or rotation_grid_mstep_rotations is not None
    )
    translations = np.asarray(translations, dtype=dtype)
    prior_translations = np.asarray(prior_translations, dtype=dtype).reshape(-1, translations.shape[1])
    rotation_log_prior_np = None if rotation_log_prior is None else np.asarray(rotation_log_prior, dtype=dtype)

    source_eulers_flat = None
    rotations_flat_override = None
    mstep_rotations_flat_override = None
    rotation_posterior_ids_flat_override = None
    if int(local_parent_oversampling_order) > 0:
        (
            offsets,
            counts,
            rotation_ids_flat,
            rotation_log_priors_flat,
            rotations_flat_override,
            mstep_rotations_flat_override,
            source_eulers_flat,
            rotation_posterior_ids_flat_override,
        ) = _build_parent_expanded_local_entries(
            prior_rotations,
            healpix_order,
            sigma_rot,
            sigma_psi,
            oversampling_order=int(local_parent_oversampling_order),
            rotation_log_prior=rotation_log_prior_np,
            random_perturbation=float(rotation_grid_random_perturbation),
            generate_relion_mstep_rotations=generate_relion_mstep_rotations,
            dtype=dtype,
            symmetry=symmetry,
        )
    elif str(grid_metadata["mode"]) == "factorized":
        offsets, counts, rotation_ids_flat, rotation_log_priors_flat = _build_factorized_local_entries(
            prior_rotations,
            healpix_order,
            sigma_rot,
            sigma_psi,
            grid_metadata,
            dtype=dtype,
        )
    else:
        n_images = int(prior_rotations.shape[0])
        offsets = np.zeros(n_images + 1, dtype=np.int64)
        counts = np.zeros(n_images, dtype=np.int32)
        rotation_ids_parts: list[np.ndarray] = []
        log_prior_parts: list[np.ndarray] = []

        for image_idx in range(n_images):
            local_ids, local_log_prior = get_local_rotation_grid_fast(
                prior_rotations[image_idx : image_idx + 1],
                sigma_rot,
                sigma_psi,
                healpix_order,
                sigma_cutoff=3.0,
                per_image=True,
                grid_metadata=grid_metadata,
            )
            local_ids = np.asarray(local_ids, dtype=np.int64).reshape(-1)
            local_log_prior = np.asarray(local_log_prior[0], dtype=dtype).reshape(-1)
            counts[image_idx] = int(local_ids.shape[0])
            offsets[image_idx + 1] = offsets[image_idx] + local_ids.shape[0]
            rotation_ids_parts.append(local_ids)
            log_prior_parts.append(local_log_prior)

        rotation_ids_flat = _flat_parts(rotation_ids_parts, empty_shape=0, dtype=np.int64, cast=np.int64)
        rotation_log_priors_flat = _flat_parts(log_prior_parts, empty_shape=0, dtype=dtype)
    if rotation_log_prior_np is not None and int(local_parent_oversampling_order) <= 0:
        if rotation_log_prior_np.shape[0] != int(grid_metadata["n_pixels"]) * int(grid_metadata["n_psi"]):
            raise ValueError(
                "rotation_log_prior must have one value per local-grid rotation "
                f"({int(grid_metadata['n_pixels']) * int(grid_metadata['n_psi'])}); "
                f"got {rotation_log_prior_np.shape}"
            )
        rotation_log_priors_flat = (
            rotation_log_priors_flat + rotation_log_prior_np[np.asarray(rotation_ids_flat, dtype=np.int64)]
        )
    rotations_flat = (
        rotations_flat_override
        if rotations_flat_override is not None
        else selected_rotation_matrices(
            rotation_ids_flat,
            rotation_grid_rotations,
            grid_metadata,
            random_perturbation=rotation_grid_random_perturbation,
            angular_sampling_deg=rotation_grid_angular_sampling_deg,
            dtype=dtype,
        )
    )
    mstep_rotations_flat = None
    if generate_relion_mstep_rotations:
        mstep_rotations_flat = (
            mstep_rotations_flat_override
            if mstep_rotations_flat_override is not None
            else _selected_mstep_rotation_matrices(
                rotation_ids_flat,
                rotation_grid_mstep_rotations,
                grid_metadata,
                random_perturbation=rotation_grid_random_perturbation,
                angular_sampling_deg=rotation_grid_angular_sampling_deg,
                dtype=dtype,
            )
        )
    translation_grid = translations
    translation_parent = None
    if int(local_parent_oversampling_order) > 0:
        translation_grid, translation_parent = get_oversampled_translation_grid(
            translations,
            infer_translation_step(translations),
            oversampling_order=int(local_parent_oversampling_order),
        )
        translation_grid = np.asarray(translation_grid, dtype=dtype)
        translation_parent = np.asarray(translation_parent, dtype=np.int32)

    reference_translations = (
        np.asarray(translation_prior_reference_translations, dtype=dtype)
        if translation_prior_reference_translations is not None
        else translations
    )

    coarse_translation_log_priors = make_relion_translation_log_prior(
        reference_translations,
        voxel_size,
        sigma_offset_angstrom,
        prior_translations,
        offset_range_pixels=offset_range_pixels,
        dtype=dtype,
    ).astype(dtype, copy=False)
    if translation_parent is None:
        translation_log_priors = coarse_translation_log_priors
    else:
        translation_log_priors = _fine_translation_log_prior(
            coarse_translation_log_priors,
            translation_parent,
            int(prior_translations.shape[0]),
            int(translation_grid.shape[0]),
            dtype=dtype,
        )

    if rotation_posterior_ids_flat_override is not None:
        parent_order = int(healpix_order) - int(local_parent_oversampling_order)
        n_global_rotations = rotation_grid_size(parent_order, symmetry)
    elif rotation_grid_rotations is not None:
        n_global_rotations = int(rotation_grid_rotations.shape[0])
    else:
        n_global_rotations = int(grid_metadata["n_pixels"]) * int(grid_metadata["n_psi"])

    if (
        source_eulers_flat is None
        and int(local_parent_oversampling_order) == 0
        and rotation_grid_rotations is None
        and str(grid_metadata["mode"]) == "factorized"
    ):
        source_eulers_flat = get_oversampled_rotation_grid_from_samples(
            rotation_ids_flat,
            healpix_order,
            oversampling_order=0,
            random_perturbation=float(rotation_grid_random_perturbation),
            return_source_eulers=True,
            dtype=dtype,
            symmetry=symmetry,
        )[-1]
    if rotation_posterior_ids_flat_override is not None:
        posterior_ids = np.asarray(rotation_posterior_ids_flat_override, dtype=np.int64)
        if posterior_ids.shape != np.asarray(rotation_ids_flat).shape:
            raise RuntimeError(
                "parent-expanded local posterior ids must match the scored rotation rows"
            )
        if np.any(posterior_ids < 0) or int(posterior_ids.max(initial=-1)) >= n_global_rotations:
            raise RuntimeError(
                f"parent-expanded {symmetry} posterior ids must index the reduced "
                f"parent grid of size {n_global_rotations}"
            )

    return LocalHypothesisLayout(
        n_global_rotations=n_global_rotations,
        n_pixels=int(grid_metadata["n_pixels"]),
        n_psi=int(grid_metadata["n_psi"]),
        rotation_offsets=offsets,
        rotation_ids_flat=rotation_ids_flat,
        rotations_flat=rotations_flat,
        rotation_log_priors_flat=rotation_log_priors_flat,
        rotation_counts=counts,
        translation_grid=translation_grid,
        translation_log_priors=np.asarray(translation_log_priors, dtype=dtype),
        mstep_rotations_flat=mstep_rotations_flat,
        source_eulers_flat=source_eulers_flat,
        oversampled_rows=int(local_parent_oversampling_order) > 0,
        id_rows_source=(
            None
            if int(local_parent_oversampling_order) > 0
            else (
                int(healpix_order),
                float(rotation_grid_random_perturbation),
                None if rotation_grid_angular_sampling_deg is None else float(rotation_grid_angular_sampling_deg),
            )
        ),
        rotation_posterior_ids_flat=rotation_posterior_ids_flat_override,
        symmetry=symmetry,
    )


def expand_local_layout_classes(layout: LocalHypothesisLayout, n_classes: int) -> LocalHypothesisLayout:
    """The layout with every image's rows repeated for each of ``n_classes`` classes, class-major per image.

    RELION's Class3D local search centres one search per particle on its previous best angles and scores
    every class at the same orientations (ml_optimiser.cpp, getFourierTransformsAndCtfs and
    getAllSquaredDifferences loop over classes inside one particle). A row's prior, rotation, mask and ids
    are its K=1 row's; ``row_class_flat`` says which class's reference it is scored against.
    """

    if n_classes < 2:
        raise ValueError(f"a class expansion needs at least two classes, got {n_classes}")
    if layout.n_classes != 1:
        raise ValueError("the layout is already expanded over classes")
    offsets = np.asarray(layout.rotation_offsets, dtype=np.int64)
    counts = np.diff(offsets)
    n_images = int(counts.shape[0])
    # Source row of every expanded row: image i's rows once per class, in class order.
    image_of = np.repeat(np.arange(n_images, dtype=np.int64), counts * n_classes)
    within = np.arange(int(counts.sum()) * n_classes, dtype=np.int64) - np.repeat(
        offsets[:-1] * n_classes, counts * n_classes
    )
    image_counts = counts[image_of]
    source = offsets[:-1][image_of] + within % np.maximum(image_counts, 1)
    row_class = (within // np.maximum(image_counts, 1)).astype(np.int32)

    def take(values):
        return None if values is None else np.asarray(values)[source]

    new_counts = (counts * n_classes).astype(np.asarray(layout.rotation_counts).dtype)
    new_offsets = np.zeros(n_images + 1, dtype=np.int64)
    new_offsets[1:] = np.cumsum(new_counts, dtype=np.int64)
    return replace(
        layout,
        rotation_offsets=new_offsets,
        rotation_ids_flat=take(layout.rotation_ids_flat),
        rotations_flat=take(layout.rotations_flat),
        rotation_log_priors_flat=take(layout.rotation_log_priors_flat),
        rotation_counts=new_counts,
        rotation_posterior_ids_flat=take(layout.rotation_posterior_ids_flat),
        sample_mask_bits=take(layout.sample_mask_bits),
        mstep_rotations_flat=take(layout.mstep_rotations_flat),
        source_eulers_flat=take(layout.source_eulers_flat),
        row_class_flat=row_class,
        n_classes=n_classes,
    )


def restrict_local_layout_classes(layout: LocalHypothesisLayout, image_classes) -> LocalHypothesisLayout:
    """A class-expanded layout keeping, for each image, only the rows of its given class.

    RELION's first Class3D iteration from one reference scores each particle against one random class
    only (do_generate_seeds: exp_iclass_min = exp_iclass_max), with or without local searches.
    """

    if layout.n_classes == 1:
        raise ValueError("only a class-expanded layout has classes to restrict")
    image_classes = np.asarray(image_classes, dtype=np.int64).reshape(-1)
    offsets = np.asarray(layout.rotation_offsets, dtype=np.int64)
    counts = np.diff(offsets)
    if image_classes.shape != counts.shape:
        raise ValueError(f"one class per image is needed, got {image_classes.shape} for {counts.shape[0]} images")
    row_image = np.repeat(np.arange(counts.shape[0], dtype=np.int64), counts)
    keep = np.asarray(layout.row_class_flat, dtype=np.int64) == image_classes[row_image]
    new_counts = np.bincount(row_image[keep], minlength=counts.shape[0])
    if np.any(new_counts == 0):
        raise ValueError(f"image {int(np.flatnonzero(new_counts == 0)[0])} has no row of its seed class")

    def take(values):
        return None if values is None else np.asarray(values)[keep]

    new_offsets = np.zeros(counts.shape[0] + 1, dtype=np.int64)
    new_offsets[1:] = np.cumsum(new_counts, dtype=np.int64)
    return replace(
        layout,
        rotation_offsets=new_offsets,
        rotation_ids_flat=take(layout.rotation_ids_flat),
        rotations_flat=take(layout.rotations_flat),
        rotation_log_priors_flat=take(layout.rotation_log_priors_flat),
        rotation_counts=new_counts.astype(np.asarray(layout.rotation_counts).dtype),
        rotation_posterior_ids_flat=take(layout.rotation_posterior_ids_flat),
        sample_mask_bits=take(layout.sample_mask_bits),
        mstep_rotations_flat=take(layout.mstep_rotations_flat),
        source_eulers_flat=take(layout.source_eulers_flat),
        row_class_flat=take(layout.row_class_flat).astype(np.int32),
    )


def build_local_adaptive_pass2_hypothesis_layout(
    parent_layout: LocalHypothesisLayout,
    significant_sample_indices,
    parent_healpix_order: int,
    *,
    oversampling_order: int,
    random_perturbation: float = 0.0,
    translation_step: float | None = None,
    dtype: np.dtype = np.float32,
    symmetry: str | None = None,
) -> LocalHypothesisLayout:
    """Expand local adaptive parent support while preserving significant pairs.

    RELION's adaptive local pass 2 first expands significant coarse orientation
    parents, then only scores fine translation children for coarse
    ``(orientation, translation)`` pairs that survived pass 1. ``parent_layout``
    carries the image-specific local Gaussian priors from that coarse pass.

    A class-expanded parent layout (``n_classes > 1``) has significant samples
    ``(class * n_global_rotations + rotation) * n_trans + t``, so each class keeps
    its own surviving parents; the children carry their parent's class.
    """

    if oversampling_order <= 0:
        raise ValueError("oversampling_order must be positive for adaptive local pass 2")
    fine_healpix_order = parent_healpix_order + oversampling_order
    symmetry = canonicalize_rotational_symmetry(
        parent_layout.symmetry if symmetry is None else symmetry
    )
    if len(significant_sample_indices) != parent_layout.n_images:
        raise ValueError(
            "significant_sample_indices must have one entry per image; "
            f"got {len(significant_sample_indices)} for {parent_layout.n_images} images",
        )

    coarse_translations = np.asarray(parent_layout.translation_grid, dtype=dtype)
    n_coarse_trans = int(coarse_translations.shape[0])
    if translation_step is None:
        translation_step = infer_translation_step(coarse_translations)
    fine_translations, fine_translation_parent = get_oversampled_translation_grid(
        coarse_translations,
        float(translation_step),
        oversampling_order=oversampling_order,
    )
    fine_translations = np.asarray(fine_translations, dtype=dtype)
    fine_translation_parent = np.asarray(fine_translation_parent, dtype=np.int32)
    n_fine_trans = int(fine_translations.shape[0])

    n_real_global = int(parent_layout.n_global_rotations)
    # Class-expanded layouts number a (class, rotation) parent class * n_global + rotation.
    n_parent_global = parent_layout.n_classes * n_real_global
    parent_offsets = np.asarray(parent_layout.rotation_offsets, dtype=np.int64)
    parent_ids_flat = np.asarray(parent_layout.rotation_ids_flat, dtype=np.int64)
    if parent_layout.n_classes > 1:
        parent_ids_flat = np.asarray(parent_layout.row_class_flat, dtype=np.int64) * n_real_global + parent_ids_flat
    parent_log_prior_flat = np.asarray(parent_layout.rotation_log_priors_flat, dtype=dtype)
    parent_counts = np.diff(parent_offsets)
    if parent_layout.n_images and np.any(parent_counts == 0):
        image_idx = int(np.flatnonzero(parent_counts == 0)[0])
        raise ValueError(f"Image {image_idx} has no local parent rotations for adaptive pass 2")

    # Each image's parent rotations, flat and image-major: the sorted unique
    # rotations of its significant samples, or every local parent (in the
    # parent layout's order) when it has none. One pass over all images; the
    # per-image statements are unchanged.
    sig_arrays = [
        np.zeros(0, dtype=np.int64) if samples is None else np.asarray(samples, dtype=np.int64).reshape(-1)
        for samples in significant_sample_indices
    ]
    sig_counts = np.fromiter((a.size for a in sig_arrays), dtype=np.int64, count=parent_layout.n_images)
    full_images = sig_counts == 0
    sig_flat = np.concatenate(sig_arrays) if sig_arrays else np.zeros(0, dtype=np.int64)
    sig_image = np.repeat(np.arange(parent_layout.n_images, dtype=np.int64), sig_counts)
    sig_rot = sig_flat // n_coarse_trans
    sig_trans = sig_flat % n_coarse_trans
    key_scale = np.int64(max(n_parent_global, 1)) + np.int64(1)
    sparse_keys = np.unique(sig_image * key_scale + sig_rot)
    sparse_image = sparse_keys // key_scale
    sparse_rot = sparse_keys % key_scale
    full_parent_rows = np.flatnonzero(np.repeat(full_images, parent_counts))
    full_image = np.repeat(np.arange(parent_layout.n_images, dtype=np.int64), parent_counts)[full_parent_rows]
    unit_image = np.concatenate([sparse_image, full_image])
    unit_rot = np.concatenate([sparse_rot, parent_ids_flat[full_parent_rows]])
    order = np.argsort(unit_image, kind="stable")
    unit_image, unit_rot = unit_image[order], unit_rot[order]
    bad_rot = (unit_rot < 0) | (unit_rot >= n_parent_global)
    if np.any(bad_rot):
        image_idx = int(unit_image[np.flatnonzero(bad_rot)[0]])
        raise ValueError(f"Image {image_idx} has significant rotation ids outside the parent grid")

    # Each unit's log prior from its image's local parents (a later duplicate
    # id wins, as in the id lookup of tests/unit/test_refine_relion_mode.py).
    parent_image = np.repeat(np.arange(parent_layout.n_images, dtype=np.int64), parent_counts)
    parent_keys = parent_image * key_scale + parent_ids_flat
    parent_order = np.argsort(parent_keys, kind="stable")
    sorted_parent_keys = parent_keys[parent_order]
    unit_keys = unit_image * key_scale + unit_rot
    position = np.searchsorted(sorted_parent_keys, unit_keys, side="right") - 1
    matched = position >= 0
    matched[matched] = sorted_parent_keys[position[matched]] == unit_keys[matched]
    if not np.all(matched):
        image_idx = int(unit_image[np.flatnonzero(~matched)[0]])
        missing = unit_rot[(unit_image == image_idx) & ~matched]
        raise ValueError(
            f"Image {image_idx} has significant rotations outside its local parent support: {missing[:8].tolist()}"
        )
    unit_log_prior = parent_log_prior_flat[parent_order[position]]
    unit_class = (unit_rot // n_real_global).astype(np.int32)
    unit_rot = unit_rot % n_real_global

    # Every image's oversampled children in one call: each parent's children
    # depend on that parent only, so the rows are the per-image calls' rows.
    oversampled_rots, parent_map, oversampled_rot_indices, oversampled_mstep_rots, source_eulers = (
        _compute_oversampled_rotation_grid(
            unit_rot,
            parent_healpix_order,
            oversampling_order=oversampling_order,
            random_perturbation=float(random_perturbation),
            return_rotation_indices=True,
            return_mstep_rotations=True,
            return_source_eulers=True,
            rotation_index_order="recovar",
            dtype=dtype,
            symmetry=symmetry,
        )
    )
    rotations_flat = np.asarray(oversampled_rots, dtype=dtype).reshape(-1, 3, 3)
    mstep_rotations_flat = np.asarray(oversampled_mstep_rots, dtype=dtype).reshape(-1, 3, 3)
    parent_map = np.asarray(parent_map, dtype=np.int64)
    rotation_ids_flat = np.asarray(oversampled_rot_indices, dtype=np.int32).astype(np.int64)
    posterior_ids_flat = unit_rot[parent_map].astype(np.int32)
    rotation_log_priors_flat = unit_log_prior[parent_map].astype(dtype, copy=False)
    child_image = unit_image[parent_map]
    counts = np.bincount(child_image, minlength=parent_layout.n_images).astype(np.int32)
    offsets = np.zeros(parent_layout.n_images + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts, dtype=np.int64)

    if np.all(full_images):
        # ``None`` is the exact-local engine's compact representation of full
        # per-rotation/per-translation support. Avoid materializing massive
        # all-ones masks for RELION full-parent local pass 2.
        sample_mask_bits = None if parent_layout.n_images else np.zeros((0, (n_fine_trans + 7) // 8), dtype=np.uint8)
    else:
        coarse_mask = np.repeat(full_images[unit_image][:, None], n_coarse_trans, axis=1)
        sig_unit = np.searchsorted(unit_keys, sig_image * key_scale + sig_rot)
        coarse_mask[sig_unit, sig_trans] = True
        sample_mask_bits = np.packbits(
            coarse_mask[parent_map][:, fine_translation_parent], axis=1, bitorder="little"
        )

    fine_metadata = build_local_search_grid_metadata(fine_healpix_order, symmetry=symmetry)
    return LocalHypothesisLayout(
        oversampled_rows=True,
        n_global_rotations=rotation_grid_size(parent_healpix_order, symmetry),
        n_pixels=int(fine_metadata["n_pixels"]),
        n_psi=int(fine_metadata["n_psi"]),
        rotation_offsets=offsets,
        rotation_ids_flat=rotation_ids_flat,
        rotations_flat=rotations_flat,
        source_eulers_flat=(
            np.empty((0, 3), dtype=np.float64)
            if not parent_layout.n_images
            else (None if source_eulers is None else np.asarray(source_eulers))
        ),
        rotation_log_priors_flat=rotation_log_priors_flat,
        rotation_counts=counts,
        translation_grid=fine_translations,
        translation_log_priors=np.asarray(parent_layout.translation_log_priors, dtype=dtype)[
            :, fine_translation_parent
        ],
        rotation_posterior_ids_flat=posterior_ids_flat,
        sample_mask_bits=sample_mask_bits,
        mstep_rotations_flat=mstep_rotations_flat,
        symmetry=symmetry,
        row_class_flat=None if parent_layout.n_classes == 1 else unit_class[parent_map],
        n_classes=parent_layout.n_classes,
    )


def _positions_in_sorted_unique_ids(sorted_ids: np.ndarray, query_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map query ids to row positions in a sorted unique id array."""

    sorted_np = np.asarray(sorted_ids, dtype=np.int64).reshape(-1)
    query_np = np.asarray(query_ids, dtype=np.int64).reshape(-1)
    if query_np.size == 0:
        return np.zeros(0, dtype=np.int64), np.ones(0, dtype=bool)
    pos = np.searchsorted(sorted_np, query_np)
    valid = pos < sorted_np.size
    matched = np.zeros(query_np.shape, dtype=bool)
    if np.any(valid):
        matched[valid] = sorted_np[pos[valid]] == query_np[valid]
    return pos.astype(np.int64, copy=False), matched


def _fine_translation_log_prior(
    translation_log_prior: np.ndarray | None,
    fine_translation_parent: np.ndarray,
    n_images: int,
    n_fine_translations: int,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    if translation_log_prior is None:
        return np.zeros((n_images, n_fine_translations), dtype=dtype)
    translation_log_prior_np = np.asarray(translation_log_prior, dtype=dtype)
    if translation_log_prior_np.ndim == 1:
        fine = translation_log_prior_np[fine_translation_parent]
        return np.broadcast_to(fine[None, :], (n_images, n_fine_translations)).astype(dtype, copy=False)
    if translation_log_prior_np.ndim == 2:
        if translation_log_prior_np.shape[0] != n_images:
            raise ValueError(
                "translation_log_prior must have one row per image when 2D; "
                f"got {translation_log_prior_np.shape[0]} rows for {n_images} images",
            )
        return translation_log_prior_np[:, fine_translation_parent].astype(dtype, copy=False)
    raise ValueError(f"translation_log_prior must be 1D or 2D, got {translation_log_prior_np.ndim} dimensions")


def _pass2_translation_log_prior(
    translation_log_prior: np.ndarray | None,
    fine_translation_log_prior: np.ndarray | None,
    fine_translation_parent: np.ndarray,
    n_images: int,
    n_fine_translations: int,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    if fine_translation_log_prior is None:
        return _fine_translation_log_prior(
            translation_log_prior,
            fine_translation_parent,
            n_images,
            n_fine_translations,
            dtype=dtype,
        )
    if translation_log_prior is not None:
        raise ValueError("translation_log_prior and fine_translation_log_prior are mutually exclusive")

    prior_np = np.asarray(fine_translation_log_prior, dtype=dtype)
    if prior_np.ndim == 1:
        if prior_np.shape[0] != n_fine_translations:
            raise ValueError(
                "fine_translation_log_prior must have one value per fine translation; "
                f"got {prior_np.shape[0]} values for {n_fine_translations} translations",
            )
        return np.broadcast_to(prior_np[None, :], (n_images, n_fine_translations)).astype(dtype, copy=False)
    if prior_np.ndim == 2:
        if prior_np.shape != (n_images, n_fine_translations):
            raise ValueError(
                f"fine_translation_log_prior must have shape ({n_images}, {n_fine_translations}); got {prior_np.shape}",
            )
        return prior_np.astype(dtype, copy=False)
    raise ValueError(f"fine_translation_log_prior must be 1D or 2D, got {prior_np.ndim} dimensions")


def build_pass2_hypothesis_layout(
    significant_sample_indices,
    n_coarse_rotations: int,
    n_coarse_translations: int,
    nside_level: int,
    translations: np.ndarray,
    *,
    oversampling_order: int,
    translation_step: float | None = None,
    rotation_log_prior: np.ndarray | None = None,
    translation_log_prior: np.ndarray | None = None,
    fine_translation_log_prior: np.ndarray | None = None,
    random_perturbation: float = 0.0,
    rotation_index_order: str = "recovar",
    allow_empty: bool = False,
    dtype: np.dtype = np.float32,
    symmetry: str = "C1",
) -> LocalHypothesisLayout:
    """Build exact-local layout for RELION adaptive pass-2 hypotheses.

    Pass 2 is not a Gaussian local search around one previous best pose. RELION
    oversamples the coarse ``(rotation, translation)`` samples that survived
    pass 1. The exact local engine can score the same structure if each image
    carries its own oversampled rotations plus a sparse ``(R, T)`` mask.
    """

    symmetry = canonicalize_rotational_symmetry(symmetry)
    dtype = np.dtype(dtype)
    translations_np = np.asarray(translations, dtype=dtype)
    if translation_step is None:
        translation_step = infer_translation_step(translations_np)
    fine_translations, fine_translation_parent = get_oversampled_translation_grid(
        translations_np,
        float(translation_step),
        oversampling_order=oversampling_order,
    )
    fine_translations = np.asarray(fine_translations, dtype=dtype)
    fine_translation_parent = np.asarray(fine_translation_parent, dtype=np.int32)
    n_fine_translations = int(fine_translations.shape[0])
    n_images = len(significant_sample_indices)
    rotation_log_prior_np = None if rotation_log_prior is None else np.asarray(rotation_log_prior, dtype=dtype)

    coarse_rows = []
    for image_idx, sig_samples in enumerate(significant_sample_indices):
        if sig_samples is None:
            unique_rot = np.arange(n_coarse_rotations, dtype=np.int32)
            coarse_rot = unique_rot
            coarse_trans = None
            use_full_candidate_mask = True
        else:
            sig_samples = significant_sample_ids(
                sig_samples, int(n_coarse_rotations) * int(n_coarse_translations),
            )
            if sig_samples.size == 0:
                if not allow_empty:
                    raise ValueError(f"Image {image_idx} has no significant coarse samples for sparse pass 2")
                unique_rot = np.zeros(1, dtype=np.int64)
                coarse_rot = unique_rot
                coarse_trans = np.zeros(0, dtype=np.int64)
            else:
                coarse_rot = sig_samples // int(n_coarse_translations)
                coarse_trans = sig_samples % int(n_coarse_translations)
                unique_rot = np.unique(coarse_rot).astype(np.int64, copy=False)
            use_full_candidate_mask = False

        if np.any(unique_rot < 0) or np.any(unique_rot >= int(n_coarse_rotations)):
            raise ValueError(f"Image {image_idx} has significant rotation ids outside the coarse grid")

        coarse_rows.append((unique_rot, coarse_rot, coarse_trans, use_full_candidate_mask))

    # Child orientations depend on the coarse sample and iteration perturbation,
    # not on the image. Generate only the requested union once, then gather each
    # image's rows in the same parent/child order as independent generation.
    shared_parent_ids = (
        np.unique(np.concatenate([row[0] for row in coarse_rows]))
        if coarse_rows else np.zeros(0, dtype=np.int64)
    )
    shared_rotations, shared_parent_map, shared_rotation_ids, shared_eulers = (
        get_oversampled_rotation_grid_from_samples(
            shared_parent_ids,
            int(nside_level),
            oversampling_order=oversampling_order,
            random_perturbation=random_perturbation,
            return_rotation_indices=True,
            return_source_eulers=True,
            rotation_index_order=rotation_index_order,
            dtype=dtype,
            symmetry=symmetry,
        )
    )
    children_per_parent = 8 ** int(oversampling_order)
    if not np.array_equal(
        shared_parent_map,
        np.repeat(np.arange(shared_parent_ids.size), children_per_parent),
    ):
        raise RuntimeError("Pass-2 oversampling must retain contiguous children per parent")
    child_offsets = np.arange(children_per_parent, dtype=np.int64)

    # Allocate the final layout once; retaining every image's arrays until
    # concatenation otherwise doubles the largest host allocations.
    counts = np.asarray([row[0].size * children_per_parent for row in coarse_rows], dtype=np.int32)
    offsets = np.zeros(n_images + 1, dtype=np.int64)
    np.cumsum(counts, dtype=np.int64, out=offsets[1:])
    n_rows = int(offsets[-1])
    rotations_flat = np.empty_like(shared_rotations, shape=(n_rows, 3, 3), dtype=dtype)
    rotation_ids_flat = np.empty(n_rows, dtype=np.int64)
    posterior_ids_flat = np.empty(n_rows, dtype=np.int32)
    rotation_log_priors_flat = np.empty(n_rows, dtype=dtype)
    sample_mask_bits = np.empty((n_rows, (n_fine_translations + 7) // 8), dtype=np.uint8)
    source_eulers_flat = np.empty_like(shared_eulers, shape=(n_rows, 3)) if shared_eulers is not None else None
    if not n_images:
        source_eulers_flat = np.empty((0, 3), dtype=np.float64)

    for image_idx, (unique_rot, coarse_rot, coarse_trans, use_full_candidate_mask) in enumerate(coarse_rows):
        shared_positions = np.searchsorted(shared_parent_ids, unique_rot)
        rows = (shared_positions[:, None] * children_per_parent + child_offsets).reshape(-1)
        oversampled_rots = np.asarray(shared_rotations[rows], dtype=dtype)
        oversampled_rot_indices = np.asarray(shared_rotation_ids[rows], dtype=np.int64)
        parent_map = np.repeat(np.arange(unique_rot.size, dtype=np.int32), children_per_parent)
        coarse_parent_ids = unique_rot[parent_map].astype(np.int32, copy=False)

        if rotation_log_prior_np is None:
            local_rotation_log_prior = np.zeros(oversampled_rots.shape[0], dtype=dtype)
        else:
            local_rotation_log_prior = rotation_log_prior_np[unique_rot][parent_map].astype(dtype, copy=False)

        if use_full_candidate_mask:
            sample_mask = np.ones((oversampled_rots.shape[0], n_fine_translations), dtype=bool)
        else:
            sample_mask = np.zeros((oversampled_rots.shape[0], n_fine_translations), dtype=bool)
            if coarse_trans.size:
                # Vectorized replacement of the inner-image Python loop over
                # unique_rot. Build a (n_unique_rot, n_coarse_trans) mask of
                # significant (rot, trans) pairs, then expand by parent_map
                # and fine_translation_parent. At 50k/256 K=1 this cuts pass2
                # layout-build time from ~10 s/iter to <1 s.
                significance_mask_coarse = np.zeros(
                    (unique_rot.shape[0], int(n_coarse_translations)),
                    dtype=bool,
                )
                local_idx_per_sample, matched_coarse_rot = _positions_in_sorted_unique_ids(unique_rot, coarse_rot)
                if not np.all(matched_coarse_rot):
                    raise ValueError(f"Image {image_idx} has significant samples outside unique coarse rotations")
                significance_mask_coarse[local_idx_per_sample, coarse_trans] = True
                sample_mask = significance_mask_coarse[parent_map][:, fine_translation_parent]

        if not np.any(sample_mask) and not allow_empty:
            raise ValueError(f"Image {image_idx} has no valid sparse pass-2 candidates after oversampling")

        target = slice(offsets[image_idx], offsets[image_idx + 1])
        rotations_flat[target] = oversampled_rots
        rotation_ids_flat[target] = oversampled_rot_indices
        posterior_ids_flat[target] = coarse_parent_ids
        rotation_log_priors_flat[target] = local_rotation_log_prior
        sample_mask_bits[target] = np.packbits(sample_mask, axis=1, bitorder="little")
        if source_eulers_flat is not None:
            source_eulers_flat[target] = shared_eulers[rows]

    n_pixels = (12 * (2 ** int(nside_level)) ** 2 if symmetry == "C1" else int(
        build_local_search_grid_metadata(int(nside_level), symmetry=symmetry)["n_pixels"]
    ))

    return LocalHypothesisLayout(
        oversampled_rows=True,
        n_global_rotations=int(n_coarse_rotations),
        n_pixels=int(n_pixels),
        n_psi=int(rotation_grid_n_in_planes(int(nside_level))),
        rotation_offsets=offsets,
        rotation_ids_flat=rotation_ids_flat,
        rotations_flat=rotations_flat,
        source_eulers_flat=source_eulers_flat,
        rotation_log_priors_flat=rotation_log_priors_flat,
        rotation_counts=counts,
        translation_grid=fine_translations,
        translation_log_priors=_pass2_translation_log_prior(
            translation_log_prior,
            fine_translation_log_prior,
            fine_translation_parent,
            n_images,
            n_fine_translations,
            dtype=dtype,
        ),
        rotation_posterior_ids_flat=posterior_ids_flat,
        sample_mask_bits=sample_mask_bits,
        symmetry=symmetry,
    )


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
    image_capacity_ladder = _planner_image_capacity_ladder(image_capacity_ladder)
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
        if added_rows <= _exact_local_unify_max_padded_rows():
            bucket_sizes = np.full_like(bucket_sizes, unified_size)
        else:
            # Past this point the rows unification would add cost more than the extra
            # compiled shapes it avoids; optionally bound how many shapes that creates.
            bucket_sizes = _cap_bucket_size_classes(bucket_sizes)
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


def _local_search_engine_rotation_block_size(rotation_block_size: int) -> int:
    """Cap the exact local-search engine block size.

    Local search already reduces the candidate set per image from the full
    HEALPix grid down to a few thousand rotations. Reusing the dense-search
    5k rotation tile size here creates oversized XLA kernels whose compile
    time dominates the first local-search iteration. A 1k cap keeps the
    candidate set exact while making the compiled score kernels much smaller.
    """
    return int(max(64, min(int(rotation_block_size), 1024)))
