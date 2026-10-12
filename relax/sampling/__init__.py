import functools

import healpy as hp
import numpy as np
from recovar import utils

from relax.sampling import grids, healpix
from relax.sampling.symmetry import canonicalize_rotational_symmetry


def _normalized_log_weights(diff_deg: np.ndarray, sigma_deg: float) -> np.ndarray:
    """Return log Gaussian weights normalized to sum to one."""
    if sigma_deg <= 0.0:
        out = np.full(diff_deg.shape, -np.log(max(diff_deg.size, 1)), dtype=np.float64)
        return out.astype(np.float32)

    weights = np.exp(-0.5 * (np.asarray(diff_deg, dtype=np.float64) / float(sigma_deg)) ** 2)
    total = float(weights.sum())
    if total <= 0.0 or not np.isfinite(total):
        weights.fill(1.0 / max(weights.size, 1))
    else:
        weights /= total
    return np.log(np.clip(weights, np.finfo(np.float32).tiny, None)).astype(np.float32)


@functools.lru_cache(maxsize=None)
def _relion_nested_child_offsets(oversampling_order: int) -> np.ndarray:
    """Fine NEST offsets inside one parent pixel in RELION's enumeration order.

    HEALPix NEST child pixels are bit-interleaved ``x/y`` offsets. Repeated
    ``4 * parent + child`` subdivision yields depth-first offset order; RELION's
    ``HealPixOver`` instead scans the fine ``xyf`` cell with ``y`` as the outer
    loop and ``x`` as the inner loop. The two orders are identical for one
    oversampling level but differ for two or more levels.
    """
    if oversampling_order < 0:
        raise ValueError("oversampling_order must be non-negative")

    factor = 1 << oversampling_order
    offsets = np.empty(factor * factor, dtype=np.int64)
    pos = 0
    for y in range(factor):
        for x in range(factor):
            nested = 0
            for bit in range(oversampling_order):
                nested |= ((x >> bit) & 1) << (2 * bit)
                nested |= ((y >> bit) & 1) << (2 * bit + 1)
            offsets[pos] = nested
            pos += 1
    return offsets


_OVERSAMPLED_GRID_MEMO: dict = {}


def unique_nonnegative_ids(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``np.unique(ids, return_inverse=True)`` for non-negative integer ids, without a sort.

    A presence table over ``[0, max]`` gives the same sorted distinct ids and
    the same inverse in linear time when the ids are dense (grid rotation ids);
    sparse ids fall back to ``np.unique``.
    """

    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    if ids.size == 0:
        return ids.copy(), np.zeros(0, dtype=np.int64)
    top = int(ids.max())
    if int(ids.min()) < 0 or top >= 4 * ids.size + 1024:
        return np.unique(ids, return_inverse=True)
    present = np.zeros(top + 1, dtype=bool)
    present[ids] = True
    distinct = np.flatnonzero(present)
    rank = np.cumsum(present, dtype=np.int64) - 1
    return distinct, rank[ids]


def get_oversampled_rotation_grid_from_samples(
    parent_rotation_indices,
    parent_nside_level,
    oversampling_order=1,
    *,
    random_perturbation=0.0,
    return_rotation_indices=False,
    return_mstep_rotations=False,
    return_source_eulers=False,
    rotation_index_order: str = "recovar",
    dtype: np.dtype = np.float32,
    symmetry: str = "C1",
):
    """Oversampled child orientations of coarse samples (see ``_compute_oversampled_rotation_grid``).

    One VDAM iteration asks for the same full fine grid more than once (the
    pass-2 grids and the source Euler rows); the most recent grid is kept, as
    read-only arrays, and served again for identical arguments.
    """

    parents = np.ascontiguousarray(np.asarray(parent_rotation_indices, dtype=np.int64))
    key = (
        parents.shape,
        parents.tobytes(),
        int(parent_nside_level),
        int(oversampling_order),
        float(random_perturbation),
        str(rotation_index_order),
        np.dtype(dtype).str,
        str(symmetry),
    )
    cached = _OVERSAMPLED_GRID_MEMO.get(key)
    if cached is None:
        cached = _compute_oversampled_rotation_grid(
            parents,
            parent_nside_level,
            oversampling_order,
            random_perturbation=random_perturbation,
            return_rotation_indices=True,
            return_mstep_rotations=True,
            return_source_eulers=True,
            rotation_index_order=rotation_index_order,
            dtype=dtype,
            symmetry=symmetry,
        )
        for value in cached:
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
        _OVERSAMPLED_GRID_MEMO.clear()
        _OVERSAMPLED_GRID_MEMO[key] = cached
    matrices, parent_map, rotation_indices, mstep_rotations, source_eulers = cached
    outputs = [matrices, parent_map]
    if return_rotation_indices:
        outputs.append(rotation_indices)
    if return_mstep_rotations:
        outputs.append(mstep_rotations)
    if return_source_eulers:
        outputs.append(source_eulers)
    return tuple(outputs)


def _compute_oversampled_rotation_grid(
    parent_rotation_indices,
    parent_nside_level,
    oversampling_order=1,
    **kwargs,
):
    """:func:`_compute_oversampled_rotation_grid_rows`, computed once per distinct parent.

    Every output row depends on its parent sample alone and the rows are
    parent-major, so a parent list with many repeats (the local rotations of
    every image, 13M rows over a 2.4M-sample grid on the full EMPIAR-10097
    dataset) expands its distinct parents and gathers their rows; the returned
    rows are the same rows, and ``parent_map`` still indexes the given list.
    """

    parents = np.asarray(parent_rotation_indices, dtype=np.int64).reshape(-1)
    if parents.size < 1024 or int(parents.min()) < 0:
        return _compute_oversampled_rotation_grid_rows(parents, parent_nside_level, oversampling_order, **kwargs)
    distinct, inverse = unique_nonnegative_ids(parents)
    if 2 * distinct.size > parents.size:
        return _compute_oversampled_rotation_grid_rows(parents, parent_nside_level, oversampling_order, **kwargs)
    outputs = _compute_oversampled_rotation_grid_rows(distinct, parent_nside_level, oversampling_order, **kwargs)
    per_parent = 8 ** int(oversampling_order)
    rows = (inverse.reshape(-1)[:, None] * per_parent + np.arange(per_parent, dtype=np.int64)[None, :]).reshape(-1)
    parent_map = np.repeat(np.arange(parents.size, dtype=np.int64), per_parent)
    return tuple(
        parent_map if position == 1 else (None if value is None else value[rows])
        for position, value in enumerate(outputs)
    )


def _compute_oversampled_rotation_grid_rows(
    parent_rotation_indices,
    parent_nside_level,
    oversampling_order=1,
    *,
    random_perturbation=0.0,
    return_rotation_indices=False,
    return_mstep_rotations=False,
    return_source_eulers=False,
    rotation_index_order: str = "recovar",
    dtype: np.dtype = np.float32,
    symmetry: str = "C1",
):
    """Generate oversampled child orientations from coarse sample indices.

    RELION oversamples each coarse orientation sample, not just the HEALPix
    direction. For a single oversampling level in 3D, each coarse sample
    expands into 4 child directions and 2 child in-plane angles, yielding
    ``8`` child orientations per parent sample.

    ``return_source_eulers=True`` appends the unmodified RFLOAT Euler rows of
    RELION's ``getOrientations`` (:func:`relax.sampling.healpix.oversampled_orientations`). Matrix outputs
    and legacy tuple layouts are unchanged. These rows are host metadata only.

    Parameters
    ----------
    parent_rotation_indices : array-like of int
        Indices into the coarse RELION-parity rotation grid. Each index
        corresponds to a specific ``(healpix_pixel, psi_index)`` sample with
        ``healpix_pixel`` interpreted in RELION's NEST ordering.
    parent_nside_level : int
        HEALPix level of the coarse grid.
    oversampling_order : int
        Number of oversampling levels.
    random_perturbation : float
        RELION's per-iteration perturbation instance. When nonzero, the child
        orientations are right-multiplied by the same perturbation rotation as
        RELION's ``getOrientations``.

    Returns
    -------
    matrices : np.ndarray, shape (n_children, 3, 3)
        Oversampled child rotation matrices.
    parent_map : np.ndarray, shape (n_children,)
        Index into ``parent_rotation_indices`` for each child orientation.
    child_rotation_indices : np.ndarray, shape (n_children,), optional
        Nearest full-grid indices of the child orientations on the fine grid.
        RELION's oversampled psi children are midpoints inside the parent bin,
        so for 3D they are generally not exact rows of the global fine grid.
        For non-C1 symmetry these identifiers use the unreduced fine NEST
        direction grid because oversampled children can cross an asymmetric-
        unit boundary and therefore need not be rows of the reduced grid.
        Only returned when ``return_rotation_indices=True``.
    mstep_rotations : np.ndarray, shape (n_children, 3, 3), optional
        RELION host-path rotations used by weighted-sum backprojection. These
        are formed from float64 Euler rows with RELION's explicit
        ``Matrix2D::inv()`` operation order before any public float32 Euler
        truncation. Only returned when ``return_mstep_rotations=True``. When
        both optional returns are requested, this is the fourth result after
        ``child_rotation_indices``.
    dtype : rotation-matrix precision for both ``matrices`` and
        ``mstep_rotations`` (default float32, matching RELION's
        single-precision ACC build). Pass ``np.float64`` to match
        ``ACC_DOUBLE_PRECISION``.
    """
    # Source Euler metadata is host RFLOAT, independent of device matrix dtype.
    # It must never be reconstructed from a rounded matrix or nearest-grid ID.
    parent_rotation_indices = np.asarray(parent_rotation_indices, dtype=np.int64)
    if parent_rotation_indices.size == 0:
        empty_rot = np.empty((0, 3, 3), dtype=dtype)
        empty_map = np.empty((0,), dtype=np.int64)
        outputs = [empty_rot, empty_map]
        if return_rotation_indices:
            outputs.append(empty_map.copy())
        if return_mstep_rotations:
            outputs.append(empty_rot.copy())
        if return_source_eulers:
            outputs.append(np.empty((0, 3), dtype=np.float64))
        return tuple(outputs)

    if rotation_index_order == "relion_hidden":
        rotation_index_order = "relion"
    if rotation_index_order not in {"recovar", "relion"}:
        raise ValueError(
            "rotation_index_order must be 'recovar', 'relion', or "
            f"'relion_hidden', got {rotation_index_order!r}"
        )

    symmetry = canonicalize_rotational_symmetry(symmetry)
    coarse_metadata = (
        None if symmetry == "C1"
        else grids.get_relion_grid_metadata(int(parent_nside_level), symmetry)
    )
    coarse_n_pixels = (
        hp.nside2npix(2**parent_nside_level) if coarse_metadata is None
        else int(coarse_metadata["n_pixels"])
    )
    coarse_n_in_planes = grids.rotation_grid_n_in_planes(parent_nside_level)
    if rotation_index_order == "recovar":
        parent_directions = parent_rotation_indices % coarse_n_pixels
        parent_psi = parent_rotation_indices // coarse_n_pixels
    else:
        parent_directions = parent_rotation_indices // coarse_n_in_planes
        parent_psi = parent_rotation_indices % coarse_n_in_planes

    if np.any(parent_directions < 0) or np.any(parent_directions >= coarse_n_pixels):
        raise ValueError(
            "parent rotation index contains a direction outside the "
            f"{symmetry} asymmetric-unit grid of size {coarse_n_pixels}"
        )
    if np.any(parent_psi < 0) or np.any(parent_psi >= coarse_n_in_planes):
        raise ValueError(
            "parent rotation index contains an in-plane sample outside the "
            f"grid of size {coarse_n_in_planes}"
        )
    parent_pixels = (
        parent_directions if coarse_metadata is None
        else np.asarray(coarse_metadata["directions_ipix"], dtype=np.int64)[parent_directions]
    )

    current_pixels = parent_pixels.copy()
    parent_map = np.arange(len(parent_rotation_indices), dtype=np.int64)
    if oversampling_order > 0:
        offsets = _relion_nested_child_offsets(oversampling_order)
        current_pixels = (
            current_pixels.astype(np.int64, copy=False)[:, None] * (4**oversampling_order) + offsets[None, :]
        ).reshape(-1)
        parent_map = np.repeat(parent_map, offsets.size)

    psi_factor = 2**oversampling_order
    coarse_psi_step = 2.0 * np.pi / coarse_n_in_planes
    fine_nside_level = parent_nside_level + oversampling_order
    fine_nside = 2**fine_nside_level
    fine_n_pixels = hp.nside2npix(fine_nside)
    fine_n_in_planes = grids.rotation_grid_n_in_planes(fine_nside_level)
    fine_psi_step = 2.0 * np.pi / fine_n_in_planes

    current_parent_psi = parent_psi[parent_map]
    # Match RELION's pushbackOversampledPsiAngles(): oversampled psi samples
    # are midpoints inside the parent psi bin, not rows of the fine global grid.
    psi_child_angles = (
        current_parent_psi[:, None] * coarse_psi_step
        - 0.5 * coarse_psi_step
        + (0.5 + np.arange(psi_factor, dtype=np.float64)[None, :]) * (coarse_psi_step / psi_factor)
    )
    nearest_child_psi = (
        np.floor(np.mod(psi_child_angles, 2.0 * np.pi) / fine_psi_step + 0.5).astype(np.int64) % fine_n_in_planes
    )

    child_pixels = np.repeat(current_pixels, psi_factor)
    if rotation_index_order == "recovar":
        child_rotation_indices = nearest_child_psi.reshape(-1) * fine_n_pixels + child_pixels
    else:
        child_rotation_indices = child_pixels * fine_n_in_planes + nearest_child_psi.reshape(-1)

    euler_angles = healpix.oversampled_orientations(
        int(parent_nside_level),
        int(oversampling_order),
        parent_directions,
        parent_psi,
        float(random_perturbation),
        symmetry,
    )
    matrices = grids.relion_mstep_rotations_from_eulers(euler_angles, dtype=dtype)
    mstep_rotations = matrices if return_mstep_rotations else None
    parent_map = np.repeat(parent_map, psi_factor)

    outputs = [matrices, parent_map]
    if return_rotation_indices:
        outputs.append(child_rotation_indices.astype(np.int64))
    if return_mstep_rotations:
        outputs.append(mstep_rotations)
    if return_source_eulers:
        outputs.append(euler_angles.copy())
    return tuple(outputs)


def infer_translation_step(translations: np.ndarray) -> float:
    """Infer spacing from an existing grid without narrowing its precision."""
    unique_vals = np.unique(np.asarray(translations))
    diffs = np.diff(np.sort(unique_vals))
    diffs = diffs[diffs > 1e-6]
    return float(diffs.min()) if diffs.size else 1.0


def get_oversampled_translation_grid(parent_translations, pixel_offset, oversampling_order=1):
    """Generate a finer translation grid by subdividing each parent cell.

    Each parent translation cell is subdivided into ``(2**oversampling_order)**D``
    child cells (``D`` = 2 or 3 dimensions), with centers evenly spaced within the
    parent cell.

    Args:
        parent_translations: float (N, D) parent translation grid points.
        pixel_offset: float, step size between parent grid points (pixels).
        oversampling_order: int, number of subdivision levels (default 1).

    Returns:
        fine_translations: float (N * (2**oversampling_order)**D, D), parent-major.
        parent_map: int (N * (2**oversampling_order)**D,) index into parent_translations.
    """
    parent_translations = np.asarray(parent_translations)
    n_parents, n_dims = parent_translations.shape
    n_subdiv = 2**oversampling_order  # per dimension
    fine_offset = pixel_offset / n_subdiv

    half_offsets = np.linspace(
        -pixel_offset / 2 + fine_offset / 2,
        pixel_offset / 2 - fine_offset / 2,
        n_subdiv,
    )
    grids = np.meshgrid(*([half_offsets] * n_dims), indexing="ij")
    child_offsets = np.stack([g.ravel() for g in grids], axis=-1)  # (n_subdiv**D, D)
    n_children = n_subdiv**n_dims

    fine_translations = (parent_translations[:, None, :] + child_offsets[None, :, :]).reshape(-1, n_dims)
    parent_map = np.repeat(np.arange(n_parents), n_children)
    return fine_translations, parent_map


# ---------------------------------------------------------------------------
# Variable-order rotation grid
# ---------------------------------------------------------------------------


def get_oversampled_relion_hidden_rotation_grid_from_samples(
    parent_rotation_indices,
    parent_nside_level,
    oversampling_order=1,
    *,
    random_perturbation=0.0,
    return_rotation_indices=False,
    symmetry: str = "C1",
):
    """Generate RELION hidden-order oversampled children from coarse samples."""

    return get_oversampled_rotation_grid_from_samples(
        parent_rotation_indices,
        parent_nside_level,
        oversampling_order=oversampling_order,
        random_perturbation=random_perturbation,
        return_rotation_indices=return_rotation_indices,
        rotation_index_order="relion",
        symmetry=symmetry,
    )


# ---------------------------------------------------------------------------
# Local angular search
# ---------------------------------------------------------------------------


def get_local_rotation_grid_fast(
    prior_rotation_indices,
    sigma_rot,
    sigma_psi,
    healpix_order,
    sigma_cutoff=3.0,
    *,
    per_image=False,
    grid_metadata=None,
):
    """RELION-style local rotation selection on a reduced HEALPix x psi grid.

    This mirrors the non-helical SPA path in
    ``HealpixSampling::selectOrientationsWithNonZeroPriorProbability`` for
    the common auto-refine case used on this branch:

    - factored direction and psi priors
    - ``sigma_tilt = sigma_rot``
    - no bimodal psi search
    - no multi-body extra priors

    The selected set is the Cartesian product of:

    - directions with ``diffang < sigma_cutoff * max(sigma_rot, sigma_tilt)``
    - psi angles with ``diffpsi < sigma_cutoff * sigma_psi``

    and the per-rotation log-prior is
    ``log(direction_prior) + log(psi_prior)`` with both factors normalized
    exactly as RELION does before the product.

    Parameters
    ----------
    prior_rotation_indices : np.ndarray
        Either full-grid rotation indices of shape ``(n_priors,)``,
        explicit RELION Euler angles of shape ``(n_priors, 3)``, or exact
        prior rotation matrices of shape ``(n_priors, 3, 3)``.
    sigma_rot : float
        Gaussian prior sigma for rotation, **radians**. Used as the
        cone radius scale and the log-prior denominator.
    sigma_psi : float
        Gaussian prior sigma for in-plane angle, **radians**. Combined
        Independent in-plane prior sigma; it must not widen the direction
        cone. RELION's direction `biggest_sigma` is `max(sigma_rot,
        sigma_tilt)`, and this SPA path uses `sigma_tilt == sigma_rot`.
    healpix_order : int
        HEALPix order (nside = 2^order) of the rotation grid.
    sigma_cutoff : float
        Include grid points within ``sigma_cutoff * sigma_rot``
        SO(3) distance of at least one prior (default 3.0).
    per_image : bool
        When True, return a per-image log-prior of shape
        ``(n_priors, n_selected)``. When False, collapse to the max over
        priors per grid index, shape ``(n_selected,)``.

    Returns
    -------
    selected_indices : np.ndarray, shape (n_selected,), dtype int
        Sorted indices into the full rotation grid.
    rotation_log_prior : np.ndarray
        Per-image (or aggregated) Gaussian log-prior over the selected
        union, with out-of-cone entries set to ``-1e30``.
    """
    prior_rotation_indices = np.asarray(prior_rotation_indices)
    grid_metadata = grids.build_local_search_grid_metadata(healpix_order) if grid_metadata is None else grid_metadata
    mode = str(grid_metadata["mode"])
    symmetry = canonicalize_rotational_symmetry(str(grid_metadata.get("symmetry", "C1")))
    if symmetry == "C1":
        symmetry_operators = None
    else:
        from relax.sampling.symmetry import rotational_operators

        symmetry_operators = rotational_operators(symmetry, dtype=np.float64)
    n_pixels = int(grid_metadata["n_pixels"])
    n_psi = int(grid_metadata["n_psi"])
    n_total = int(n_pixels * n_psi)

    if prior_rotation_indices.ndim == 0:
        prior_rotation_indices = prior_rotation_indices.reshape(1)

    if prior_rotation_indices.ndim == 1:
        if "eulers_full" in grid_metadata:
            prior_eulers = np.asarray(grid_metadata["eulers_full"], dtype=np.float32)[
                prior_rotation_indices.astype(np.int64)
            ]
        else:
            prior_eulers = grids.rotation_indices_to_relion_eulers(
                prior_rotation_indices.astype(np.int64),
                healpix_order,
                symmetry=symmetry,
            )
        prior_rot_deg = prior_eulers[:, 0]
        prior_psi_deg = prior_eulers[:, 2]
        prior_rotations = utils.R_from_relion(prior_eulers, degrees=True)
    elif prior_rotation_indices.ndim == 2 and prior_rotation_indices.shape[-1] == 3:
        prior_eulers = np.asarray(prior_rotation_indices, dtype=np.float64).reshape(-1, 3)
        prior_rot_deg = prior_eulers[:, 0]
        prior_psi_deg = prior_eulers[:, 2]
        prior_rotations = utils.R_from_relion(prior_eulers, degrees=True)
    else:
        prior_rotations = np.asarray(prior_rotation_indices, dtype=np.float64).reshape(-1, 3, 3)
        prior_eulers = utils.R_to_relion(prior_rotations, degrees=True)
        prior_rot_deg = prior_eulers[:, 0]
        prior_psi_deg = prior_eulers[:, 2]

    prior_dir_vecs = np.asarray(prior_rotations[:, 2, :], dtype=np.float64)
    prior_dir_norm = np.linalg.norm(prior_dir_vecs, axis=1, keepdims=True)
    prior_dir_norm = np.where(prior_dir_norm > 0.0, prior_dir_norm, 1.0)
    prior_dir_vecs = prior_dir_vecs / prior_dir_norm

    n_priors = int(np.asarray(prior_rot_deg).reshape(-1).shape[0])
    sigma_rot_deg = float(np.rad2deg(sigma_rot))
    sigma_psi_deg = float(np.rad2deg(sigma_psi))
    biggest_sigma_deg = sigma_rot_deg

    selected_union = set()
    prior_entries: list[tuple[np.ndarray, np.ndarray]] = []

    if mode == "factorized":
        dir_vecs = np.asarray(grid_metadata["dir_vecs"], dtype=np.float64)
        psi_deg_grid = np.asarray(grid_metadata["psi_deg"], dtype=np.float64)

        for i in range(n_priors):
            if sigma_rot_deg > 0.0:
                if symmetry_operators is None:
                    dots = dir_vecs @ prior_dir_vecs[i]
                else:
                    # RELION compares the prior direction against every
                    # ``(d^T R)^T`` symmetry mate of the sampled direction.
                    dots = np.max(
                        np.einsum(
                            "di,sij,j->sd",
                            dir_vecs,
                            symmetry_operators,
                            prior_dir_vecs[i],
                            optimize=True,
                        ),
                        axis=0,
                    )
                dots = np.clip(dots, -1.0, 1.0)
                diffang = np.rad2deg(np.arccos(dots))
                dir_mask = diffang < float(sigma_cutoff) * biggest_sigma_deg
                dir_indices = np.flatnonzero(dir_mask).astype(np.int64)
                if dir_indices.size == 0:
                    dir_indices = np.array([int(np.argmin(diffang))], dtype=np.int64)
                    dir_log_prior = np.zeros(1, dtype=np.float32)
                else:
                    dir_log_prior = _normalized_log_weights(diffang[dir_indices], biggest_sigma_deg)
            else:
                dir_indices = np.arange(n_pixels, dtype=np.int64)
                dir_log_prior = np.full(
                    n_pixels,
                    -np.log(max(n_pixels, 1)),
                    dtype=np.float32,
                )

            if sigma_psi_deg > 0.0:
                wrapped_prior_psi = float(np.mod(prior_psi_deg[i], 360.0))
                diffpsi = grids.wrapped_abs_diff_deg(psi_deg_grid, wrapped_prior_psi)
                psi_mask = diffpsi < float(sigma_cutoff) * sigma_psi_deg
                psi_indices = np.flatnonzero(psi_mask).astype(np.int64)
                if psi_indices.size == 0:
                    psi_indices = np.array([int(np.argmin(diffpsi))], dtype=np.int64)
                    psi_log_prior = np.zeros(1, dtype=np.float32)
                else:
                    psi_log_prior = _normalized_log_weights(diffpsi[psi_indices], sigma_psi_deg)
            else:
                psi_indices = np.arange(n_psi, dtype=np.int64)
                psi_log_prior = np.full(
                    n_psi,
                    -np.log(max(n_psi, 1)),
                    dtype=np.float32,
                )

            flat_indices = (psi_indices[:, None] * n_pixels + dir_indices[None, :]).reshape(-1)
            flat_log_prior = (psi_log_prior[:, None] + dir_log_prior[None, :]).reshape(-1).astype(np.float32)
            prior_entries.append((flat_indices.astype(np.int64), flat_log_prior))
            selected_union.update(flat_indices.tolist())
    else:
        dir_vecs_full = np.asarray(grid_metadata["dir_vecs_full"], dtype=np.float64)
        psi_deg_full = np.asarray(grid_metadata["psi_deg_full"], dtype=np.float64)
        sigma_rot_scale = max(biggest_sigma_deg, np.finfo(np.float64).tiny)
        sigma_psi_scale = max(sigma_psi_deg, np.finfo(np.float64).tiny)
        log_prior_full = np.full((n_priors, n_total), -1e30, dtype=np.float32)
        wrapped_prior_psi_deg = np.mod(np.asarray(prior_psi_deg, dtype=np.float64), 360.0)
        cutoff_dir_deg = float(sigma_cutoff) * biggest_sigma_deg
        cutoff_psi_deg = float(sigma_cutoff) * sigma_psi_deg
        block_size = 8

        for start in range(0, n_priors, block_size):
            stop = min(start + block_size, n_priors)
            block = slice(start, stop)

            if sigma_rot_deg > 0.0:
                if symmetry_operators is None:
                    dots = np.asarray(prior_dir_vecs[block], dtype=np.float64) @ dir_vecs_full.T
                else:
                    dots = np.max(
                        np.einsum(
                            "di,sij,bj->bsd",
                            dir_vecs_full,
                            symmetry_operators,
                            np.asarray(prior_dir_vecs[block], dtype=np.float64),
                            optimize=True,
                        ),
                        axis=1,
                    )
                dots = np.clip(dots, -1.0, 1.0)
                diffang = np.rad2deg(np.arccos(dots))
                joint_mask = diffang < cutoff_dir_deg
            else:
                diffang = np.zeros((stop - start, n_total), dtype=np.float64)
                joint_mask = np.ones((stop - start, n_total), dtype=bool)

            if sigma_psi_deg > 0.0:
                diffpsi = grids.wrapped_abs_diff_deg(psi_deg_full[None, :], wrapped_prior_psi_deg[block, None])
                joint_mask &= diffpsi < cutoff_psi_deg
            else:
                diffpsi = np.zeros((stop - start, n_total), dtype=np.float64)

            row_has_support = np.any(joint_mask, axis=1)
            if np.any(row_has_support):
                joint_logw = np.zeros((stop - start, n_total), dtype=np.float64)
                if sigma_rot_deg > 0.0:
                    joint_logw += -0.5 * (diffang / sigma_rot_scale) ** 2
                if sigma_psi_deg > 0.0:
                    joint_logw += -0.5 * (diffpsi / sigma_psi_scale) ** 2

                support_mask = joint_mask[row_has_support]
                support_logw = joint_logw[row_has_support]
                support_logw = np.where(support_mask, support_logw, -np.inf)
                max_logw = np.max(support_logw, axis=1, keepdims=True)
                weights = np.exp(support_logw - max_logw)
                sums = np.sum(weights, axis=1, keepdims=True)
                normalized = np.where(
                    support_mask,
                    support_logw - (max_logw + np.log(sums)),
                    -1e30,
                ).astype(np.float32)
                block_prior = log_prior_full[block]
                block_prior[row_has_support] = normalized
                log_prior_full[block] = block_prior

            if np.any(~row_has_support):
                fallback_rows = np.flatnonzero(~row_has_support)
                joint_cost = np.zeros((fallback_rows.shape[0], n_total), dtype=np.float64)
                if sigma_rot_deg > 0.0:
                    joint_cost += (diffang[fallback_rows] / sigma_rot_scale) ** 2
                if sigma_psi_deg > 0.0:
                    joint_cost += (diffpsi[fallback_rows] / sigma_psi_scale) ** 2
                fallback_indices = np.argmin(joint_cost, axis=1)
                block_prior = log_prior_full[block]
                block_prior[fallback_rows, fallback_indices] = 0.0
                log_prior_full[block] = block_prior

        selected_mask = np.any(log_prior_full > -1e29, axis=0)
        selected_indices = np.flatnonzero(selected_mask).astype(np.int64)
        if selected_indices.size == 0:
            selected_indices = np.arange(n_total, dtype=np.int64)
            log_prior = np.full((n_priors, n_total), -np.log(max(n_total, 1)), dtype=np.float32)
        else:
            log_prior = log_prior_full[:, selected_indices]
        if per_image:
            return selected_indices, log_prior
        return selected_indices, np.max(log_prior, axis=0).astype(np.float32)

    selected_indices = np.array(sorted(selected_union), dtype=np.int64)
    if selected_indices.size == 0:
        selected_indices = np.arange(n_total, dtype=np.int64)

    index_to_pos = {int(idx): pos for pos, idx in enumerate(selected_indices.tolist())}
    log_prior = np.full((n_priors, selected_indices.shape[0]), -1e30, dtype=np.float32)

    for i, (flat_indices, flat_log_prior) in enumerate(prior_entries):
        positions = np.array([index_to_pos[int(idx)] for idx in flat_indices], dtype=np.int64)
        log_prior[i, positions] = flat_log_prior

    if per_image:
        return selected_indices, log_prior
    return selected_indices, np.max(log_prior, axis=0).astype(np.float32)
