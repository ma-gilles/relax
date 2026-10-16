"""The local direction supports: which fine rotations each image's local search keeps.

``build_factorized_local_entries`` selects, in chunks of images, each image's prior directions inside the
search cone (``_direction_cone_candidates``, banded by the cone), weights them by the Gaussian direction prior
(or the tilt prior) and factors them with the in-plane rotations; ``build_parent_expanded_local_entries``
expands those parents into their oversampled children. ``flat_parts`` concatenates the per-image parts. The
layout builders (``layout.build_local_hypothesis_layout``) call them; this module imports nothing from
``layout``.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import numpy as np
from recovar import utils

from relax.sampling.grids import (
    build_local_search_grid_metadata,
    rotation_grid_size,
    rotation_indices_to_relion_eulers,
    wrapped_abs_diff_deg,
)
from relax.sampling.healpix import euler_angles_to_matrix
from relax.sampling.oversampled_grids import get_oversampled_rotation_grid_from_samples
from relax.sampling.symmetry import canonicalize_rotational_symmetry, rotational_operators


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


def _local_selector_chunk_size(
    n_images: int, n_pixels: int, n_psi: int, use_direction: bool, use_psi: bool, *, engine_tuning,
) -> int:
    """Images per selector chunk: ``engine_tuning.local_selector_chunk_size`` when given, else as many as keep a
    chunk within ``engine_tuning.local_selector_max_elements``."""
    if engine_tuning.local_selector_chunk_size is not None:
        return max(1, min(int(n_images), int(engine_tuning.local_selector_chunk_size)))

    max_elements = int(engine_tuning.local_selector_max_elements)
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


def _tilt_prior_direction_supports(prior_tilt_deg, grid_tilt_deg, mate_tilt_deg, cutoff_deg, sigma_deg):
    """Each image's directions under a prior on the tilt angle alone, as ``(image, point, log_prior)``.

    RELION's ``sigma_rot == 0 < sigma_tilt`` branch of
    ``HealpixSampling::selectOrientationsWithNonZeroPriorProbability`` (healpix_sampling.cpp:841-879): a
    direction's distance from the prior is ``|tilt - prior_tilt|`` (``|d - 360|`` above 180), the smallest
    over the direction itself (its grid tilt, ``grid_tilt_deg``) and its symmetry mates
    (``mate_tilt_deg[m]``, the tilt ``Euler_direction2angles`` gives mate ``m``); directions below
    ``sigma_cutoff * sigma_tilt`` are kept with ``gaussian1D(distance, sigma_tilt)``, normalized to sum 1.

    An image that keeps no direction keeps one with prior 1 (healpix_sampling.cpp:937-952). RELION names
    it ``best_idir``, tracked with ``if (best_diffang < best_ang) {best_idir = idir; best_ang = diffang;}``:
    the stored ``best_ang`` is the distance of the direction's last symmetry mate, not its smallest, so
    with symmetry it is not always the nearest direction. That scan is reproduced as written.
    """

    prior_tilt_deg = np.asarray(prior_tilt_deg, dtype=np.float64).reshape(-1)

    def wrapped(tilt):
        diff = np.abs(np.asarray(tilt, dtype=np.float64)[None, :] - prior_tilt_deg[:, None])
        return np.where(diff > 180.0, np.abs(diff - 360.0), diff)

    own = wrapped(grid_tilt_deg)
    best = own
    last = own
    for mate in mate_tilt_deg:
        last = wrapped(mate)
        best = np.minimum(best, last)
    image, point, log_prior = _chunk_local_supports(best, None, None, cutoff_deg, sigma_deg)
    kept = np.bincount(image, minlength=prior_tilt_deg.shape[0])
    # _chunk_local_supports gave an image without a kept direction its nearest one; RELION's scan differs.
    for row in np.flatnonzero((kept == 1) & ~(best < cutoff_deg).any(axis=1)):
        best_idir, best_ang = -1, 9999.0
        for idir in range(best.shape[1]):
            if best[row, idir] < best_ang:
                best_idir, best_ang = idir, last[row, idir]
        point[np.flatnonzero(image == row)[0]] = best_idir
    return image, point, log_prior


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


def build_factorized_local_entries(
    prior_rotations: np.ndarray,
    healpix_order: int,
    sigma_rot: float,
    sigma_psi: float,
    grid_metadata,
    *,
    dtype: np.dtype = np.float32,
    sigma_tilt_deg: float,
    prior_tilt_deg: np.ndarray | None = None,
    engine_tuning,
):
    """Build exact per-image local supports for factorized HEALPix x psi grids.

    ``sigma_rot`` is the width of the prior on the direction (RELION's ``sigma_rot == sigma_tilt``) and
    ``sigma_psi`` of the in-plane angle; a zero width keeps every point of that axis with a uniform prior.
    ``sigma_tilt_deg`` (with ``sigma_rot == 0``) is RELION's prior on the tilt angle alone, centred on
    ``prior_tilt_deg`` per image (:func:`_tilt_prior_direction_supports`; ``--sigma_tilt``,
    docs/math/relion_refinement_algorithm.md, "A prior on the tilt angle alone"). It is in degrees, the value the user gave: RELION compares
    distances in degrees with ``3 * sigma_tilt``, HEALPix rings sit on that boundary for round widths,
    and a radian round trip moves the boundary by one ulp.
    """

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
    if sigma_tilt_deg > 0.0:
        if sigma_rot_deg > 0.0:
            # RELION's cone for both widths uses max(sigma_rot, sigma_tilt); this module has them equal.
            raise NotImplementedError("a tilt prior with a separate rot prior is not implemented")
        if prior_tilt_deg is None or np.shape(prior_tilt_deg) != (int(prior_eulers.shape[0]),):
            raise ValueError("a tilt prior needs one prior tilt angle per image")
        grid_tilt_deg = np.asarray(grid_metadata["tilt_deg"], dtype=np.float64)
        # RELION's symmetry mates of a direction d are (d^T R)^T for the non-identity operators, in SymList
        # order; Euler_direction2angles gives a mate the tilt acos(z).
        mate_tilt_deg = (
            []
            if symmetry_operators is None
            else [
                np.rad2deg(np.arccos(np.clip(dir_vecs @ operator[:, 2], -1.0, 1.0)))
                for operator in symmetry_operators[1:]
            ]
        )
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
        sigma_rot_deg > 0.0 or sigma_tilt_deg > 0.0,
        sigma_psi_deg > 0.0,
        engine_tuning=engine_tuning,
    )
    if sigma_tilt_deg > 0.0:
        # The tilt distances are held once per symmetry mate.
        chunk_size = max(1, chunk_size // (len(mate_tilt_deg) + 2))
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
            diffpsi_chunk = wrapped_abs_diff_deg(
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
        elif sigma_tilt_deg > 0.0:
            dir_image, dir_ids, dir_log_prior = _tilt_prior_direction_supports(
                np.asarray(prior_tilt_deg, dtype=np.float64)[chunk_start:chunk_stop],
                grid_tilt_deg,
                mate_tilt_deg,
                3.0 * sigma_tilt_deg,
                sigma_tilt_deg,
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

    rotation_ids_flat = flat_parts(rotation_ids_parts, empty_shape=0, dtype=np.int64, cast=np.int64)
    rotation_log_priors_flat = flat_parts(log_prior_parts, empty_shape=0, dtype=dtype)
    return offsets, counts, rotation_ids_flat, rotation_log_priors_flat


def build_parent_expanded_local_entries(
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
    sigma_tilt_deg: float,
    prior_tilt_deg: np.ndarray | None = None,
    engine_tuning,
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
    parent_offsets, parent_counts, parent_ids_flat, parent_log_priors_flat = build_factorized_local_entries(
        prior_rotations,
        parent_order,
        sigma_rot,
        sigma_psi,
        parent_metadata,
        dtype=dtype,
        sigma_tilt_deg=sigma_tilt_deg,
        prior_tilt_deg=prior_tilt_deg,
        engine_tuning=engine_tuning,
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

    rotation_ids_flat = flat_parts(rotation_ids_parts, empty_shape=0, dtype=np.int64, cast=np.int64)
    rotation_log_priors_flat = flat_parts(log_prior_parts, empty_shape=0, dtype=dtype)
    rotations_flat = flat_parts(rotations_parts, empty_shape=(0, 3, 3), dtype=dtype)
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


def flat_parts(parts, *, empty_shape, dtype, cast=None):
    """Concatenate per-image layout parts along axis 0, or the typed empty array when no image contributed.

    ``cast`` recasts the concatenation (the rotation ids are carried as int64).
    """

    if not parts:
        return np.zeros(empty_shape, dtype=dtype)
    flat = np.concatenate(parts, axis=0)
    return flat if cast is None else flat.astype(cast, copy=False)
