"""RELION's rotation and translation grids.

HEALPix orientation grids and their cached per-order geometry, the translation grids, the M-step source
rotations, and the grid records :class:`RotationGrid` and :class:`TrialGrid`.
"""

import functools
import os
from dataclasses import dataclass
from typing import NamedTuple

import healpy as hp
import jax.numpy as jnp
import numpy as np
from recovar import utils

from relax.sampling import healpix
from relax.sampling.symmetry import canonicalize_rotational_symmetry

# Cached per-order geometry used by the exact RELION local-search selector.
# For the RELION-parity grid, the flattened index is ``psi_idx * n_pixels +
# pixel_idx`` with ``pixel_idx`` following RELION's NEST-ordered HEALPix
# enumeration.
_GRID_METADATA_CACHE: "dict[tuple[int, str], dict[str, np.ndarray]]" = {}


def get_relion_grid_metadata(
    healpix_order: int,
    symmetry: str = "C1",
) -> dict[str, np.ndarray]:
    """Return cached RELION HEALPix geometry for one order and point group.

    ``directions_ipix`` preserves the original NEST HEALPix pixel for every
    retained asymmetric-unit direction.  The public flattened direction
    component is the compact row index into this retained table.
    """
    symmetry = canonicalize_rotational_symmetry(symmetry)
    cache_key = (healpix_order, symmetry)
    cached = _GRID_METADATA_CACHE.get(cache_key)
    if cached is not None:
        return cached

    source = healpix.healpix_sampling(healpix_order, symmetry)
    directions = np.column_stack([source["rot"], source["tilt"]])
    directions_ipix = np.asarray(source["directions_ipix"], dtype=np.int64)
    psi_deg = np.asarray(source["psi"], dtype=np.float64)
    n_psi = int(psi_deg.shape[0])

    n_pixels = int(directions.shape[0])
    if n_pixels < 1:
        raise ValueError(
            f"RELION retained no {symmetry} asymmetric-unit directions at "
            f"healpix_order={healpix_order}; increase the initial HEALPix order"
        )
    # Use the actual matrix view directions of the RELION grid rather than a
    # closed-form HEALPix angle formula. This keeps the local-search selector
    # aligned with the trial rotations that are actually scored.
    rot_deg = np.asarray(directions[:, 0], dtype=np.float64)
    tilt_deg = np.asarray(directions[:, 1], dtype=np.float64)
    # RELION's viewing direction is the third ROW of the rotation matrix,
    # not the third column.
    dir_eulers = np.column_stack([rot_deg, tilt_deg, np.zeros(n_pixels, dtype=np.float64)])
    dir_rotations = utils.R_from_relion(dir_eulers, degrees=True)
    dir_vecs = np.asarray(dir_rotations[:, 2, :], dtype=np.float64)
    dir_norm = np.linalg.norm(dir_vecs, axis=1, keepdims=True)
    dir_norm = np.where(dir_norm > 0.0, dir_norm, 1.0)
    dir_vecs = dir_vecs / dir_norm

    cached = {
        "rot_deg": rot_deg,
        "tilt_deg": tilt_deg,
        "dir_vecs": dir_vecs,
        "psi_deg": psi_deg,
        "directions_ipix": directions_ipix,
        "n_pixels": np.asarray(n_pixels, dtype=np.int64),
        "n_psi": np.asarray(n_psi, dtype=np.int64),
        "symmetry": symmetry,
    }
    _GRID_METADATA_CACHE[cache_key] = cached
    return cached


def wrapped_abs_diff_deg(values_deg: np.ndarray, ref_deg: np.ndarray | float) -> np.ndarray:
    """Circular absolute difference in degrees, wrapped to [0, 180]."""
    diff = np.abs(np.asarray(values_deg, dtype=np.float64) - np.asarray(ref_deg, dtype=np.float64))
    return np.where(diff > 180.0, np.abs(diff - 360.0), diff)


def build_local_search_grid_metadata(
    healpix_order: int,
    grid_eulers: np.ndarray | None = None,
    *,
    grid_rotations: np.ndarray | None = None,
    symmetry: str = "C1",
) -> dict[str, np.ndarray]:
    """Prepare local-search metadata for either the canonical or a custom grid.

    When ``grid_eulers`` is provided, it must be a full-grid table in the same
    flattened index order as ``get_relion_rotation_grid_eulers`` / the live
    trial grid. If the grid still factorizes into independent direction and psi
    axes, return factorized metadata. Otherwise, fall back to full per-rotation
    metadata.
    """
    symmetry = canonicalize_rotational_symmetry(symmetry)
    if grid_eulers is None:
        meta = get_relion_grid_metadata(healpix_order, symmetry)
        return {
            "mode": "factorized",
            "rot_deg": np.asarray(meta["rot_deg"], dtype=np.float64),
            "tilt_deg": np.asarray(meta["tilt_deg"], dtype=np.float64),
            "dir_vecs": np.asarray(meta["dir_vecs"], dtype=np.float64),
            "psi_deg": np.asarray(meta["psi_deg"], dtype=np.float64),
            "directions_ipix": np.asarray(meta["directions_ipix"], dtype=np.int64),
            "n_pixels": np.asarray(meta["n_pixels"], dtype=np.int64),
            "n_psi": np.asarray(meta["n_psi"], dtype=np.int64),
            "symmetry": symmetry,
        }

    n_pixels = int(get_relion_grid_metadata(healpix_order, symmetry)["n_pixels"])
    n_psi = rotation_grid_n_in_planes(healpix_order)
    expected = n_pixels * n_psi
    if grid_rotations is not None:
        grid_rotations_full = np.asarray(grid_rotations, dtype=np.float32).reshape(-1, 3, 3)
        if grid_rotations_full.shape[0] != expected:
            raise ValueError(
                f"grid_rotations must have shape ({expected}, 3, 3) for healpix_order={healpix_order}, "
                f"got {grid_rotations_full.shape}",
            )
    else:
        grid_rotations_full = None

    grid_eulers = np.asarray(grid_eulers, dtype=np.float32).reshape(-1, 3)
    if grid_eulers.shape[0] != expected:
        raise ValueError(
            f"grid_eulers must have shape ({expected}, 3) for healpix_order={healpix_order}, got {grid_eulers.shape}",
        )

    grid_3d = grid_eulers.reshape(n_psi, n_pixels, 3)
    psi_deg_full = np.mod(grid_eulers[:, 2].astype(np.float64), 360.0)
    if grid_rotations_full is None:
        grid_rotations_full = utils.R_from_relion(grid_eulers, degrees=True).astype(np.float32)
    dir_vecs_full = np.asarray(grid_rotations_full[:, 2, :], dtype=np.float32)
    dir_norm = np.linalg.norm(dir_vecs_full, axis=1, keepdims=True)
    dir_norm = np.where(dir_norm > 0.0, dir_norm, 1.0)
    dir_vecs_full = dir_vecs_full / dir_norm

    dir_vecs_3d = dir_vecs_full.reshape(n_psi, n_pixels, 3)
    psi_3d = psi_deg_full.reshape(n_psi, n_pixels)

    factorized_dirs = np.allclose(dir_vecs_3d, dir_vecs_3d[0:1], rtol=1e-6, atol=1e-6)
    psi_ref = psi_3d[:, :1]
    factorized_psi = float(np.max(wrapped_abs_diff_deg(psi_3d, psi_ref))) < 1e-4

    if factorized_dirs and factorized_psi:
        return {
            "mode": "factorized",
            "rot_deg": np.asarray(grid_3d[0, :, 0], dtype=np.float32),
            "tilt_deg": np.asarray(grid_3d[0, :, 1], dtype=np.float32),
            "dir_vecs": np.asarray(dir_vecs_3d[0], dtype=np.float32),
            "psi_deg": np.asarray(psi_3d[:, 0], dtype=np.float32),
            "n_pixels": np.asarray(n_pixels, dtype=np.int64),
            "n_psi": np.asarray(n_psi, dtype=np.int64),
            "eulers_full": np.asarray(grid_eulers, dtype=np.float32),
            "symmetry": symmetry,
        }

    return {
        "mode": "full",
        "dir_vecs_full": np.asarray(dir_vecs_full, dtype=np.float32),
        "psi_deg_full": np.asarray(psi_deg_full, dtype=np.float32),
        "n_pixels": np.asarray(n_pixels, dtype=np.int64),
        "n_psi": np.asarray(n_psi, dtype=np.int64),
        "eulers_full": np.asarray(grid_eulers, dtype=np.float32),
        "symmetry": symmetry,
    }


def rotation_grid_n_in_planes(order: int) -> int:
    """Number of in-plane angles used by the RELION-style HEALPix grid."""
    angle_res = 360.0 / (6.0 * 2**order)
    return int(np.round(360.0 / angle_res))


def rotation_grid_size(order: int, symmetry: str = "C1") -> int:
    """Total rotations in RELION's symmetry-reduced HEALPix x psi grid."""
    symmetry = canonicalize_rotational_symmetry(symmetry)
    if symmetry == "C1":
        nside = 2**order
        return hp.nside2npix(nside) * rotation_grid_n_in_planes(order)
    metadata = get_relion_grid_metadata(int(order), symmetry)
    return int(metadata["n_pixels"]) * int(metadata["n_psi"])


def _split_rotation_indices(
    indices,
    healpix_order,
    *,
    rotation_index_order: str = "recovar",
    symmetry: str = "C1",
):
    """Split full-grid rotation indices into HEALPix pixel and psi components."""
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    symmetry = canonicalize_rotational_symmetry(symmetry)
    n_pixels = (
        hp.nside2npix(2**healpix_order) if symmetry == "C1"
        else int(get_relion_grid_metadata(int(healpix_order), symmetry)["n_pixels"])
    )
    if rotation_index_order == "recovar":
        pixel_idx = indices % n_pixels
        psi_idx = indices // n_pixels
    elif rotation_index_order == "relion":
        n_psi = rotation_grid_n_in_planes(healpix_order)
        pixel_idx = indices // n_psi
        psi_idx = indices % n_psi
    else:
        raise ValueError(f"rotation_index_order must be 'recovar' or 'relion', got {rotation_index_order!r}")
    return pixel_idx, psi_idx


def rotation_direction_ids(healpix_order: int, *, rotation_index_order: str, symmetry: str = "C1") -> np.ndarray:
    """The HEALPix direction (pixel) of every rotation of the order's grid, ``(n_rotations,)`` int64.

    ``rotation_index_order`` is the grid's: RELION's hidden-variable order (``"relion"``) keeps each
    direction's in-plane angles together, RECOVAR's (``"recovar"``) runs over the directions within each
    in-plane angle.
    """
    indices = np.arange(rotation_grid_size(healpix_order, symmetry), dtype=np.int64)
    return _split_rotation_indices(
        indices, healpix_order, rotation_index_order=rotation_index_order, symmetry=symmetry
    )[0]


def get_rotation_grid(nside_level, n_in_planes=None, matrices=False):

    #  * order	Npix	Theta-sampling
    #  * 0		12		58.6
    #  * 1		48		29.3
    #  * 2		192		14.7
    #  * 3		768		7.33
    #  * 4		3072	3.66
    #  * 5		12288	1.83
    #  * 6		49152	0.92
    #  * 7		196608	0.46
    #  * 8		786432	0.23

    nside = 2**nside_level
    m = hp.nside2npix(nside)
    z = hp.pix2ang(nside, np.arange(m))

    if n_in_planes is None:
        angle_res = 360 / (6 * 2**nside_level)
        n_in_planes = np.round(360 / angle_res).astype(int)

    in_angle_angles = np.linspace(0, 2 * np.pi, n_in_planes, endpoint=False)
    angles = np.meshgrid(np.arange(m), in_angle_angles)
    theta = z[0][angles[0]]
    phi = z[1][angles[0]]
    # RELION convention: rot=phi (azimuth), tilt=theta (polar), psi=in-plane.
    angles = np.stack([phi, theta, angles[1]], axis=-1)
    angles = angles.reshape(-1, 3)
    angles = angles / (2 * np.pi) * 360
    if matrices:
        angles = utils.R_from_relion(angles)
    return angles


def get_translation_grid(max_pixel, pixel_offset):
    gridded_max_pixel = (max_pixel // pixel_offset) * pixel_offset
    xrange = np.arange(-gridded_max_pixel, gridded_max_pixel + 1, pixel_offset)
    # Match RELION's HealpixSampling::setTranslations loop order exactly:
    # x is the outer loop, y is the inner loop.  The ordering is scientifically
    # irrelevant but important for source-level parity of itrans diagnostics.
    x, y = np.meshgrid(xrange, xrange, indexing="ij")
    grid = np.stack([x.flatten(), y.flatten()], axis=1)
    norm_res = np.linalg.norm(grid, axis=1) <= max_pixel + 0.001
    grid = grid[norm_res]
    return grid


def get_relion_translation_grid(
    max_pixel,
    pixel_offset,
    *,
    source_units_per_pixel=1.0,
):
    """Return RELION's non-helical 2D translation grid in pixel units.

    RELION enumerates integer step indices through
    ``CEIL(offset_range / offset_step)`` and then applies its squared-radius
    cutoff.  The ceil is important when rounded STAR values convert to a
    ratio just below an integer (for example, 4.25 / 1.416667 pixels): using
    floor division silently drops the outer axial translation samples.

    ``offset_range`` and ``offset_step`` live in Angstroms in RELION, and its
    ``+0.001`` squared-radius tolerance is therefore in Angstrom squared.  Our
    search grids live in pixels, so callers must supply the Angstroms-per-pixel
    conversion through ``source_units_per_pixel``.  Applying the unscaled
    tolerance after converting to pixels can incorrectly admit boundary rows.
    """
    max_pixel = float(max_pixel)
    pixel_offset = float(pixel_offset)
    source_units_per_pixel = float(source_units_per_pixel)
    if not np.isfinite(max_pixel) or max_pixel < 0.0:
        raise ValueError(f"max_pixel must be finite and nonnegative, got {max_pixel}")
    if not np.isfinite(pixel_offset) or pixel_offset <= 0.0:
        raise ValueError(f"pixel_offset must be finite and positive, got {pixel_offset}")
    if not np.isfinite(source_units_per_pixel) or source_units_per_pixel <= 0.0:
        raise ValueError(
            "source_units_per_pixel must be finite and positive, got "
            f"{source_units_per_pixel}"
        )

    max_index = int(np.ceil(max_pixel / pixel_offset))
    indices = np.arange(-max_index, max_index + 1, dtype=np.int64)
    x_index, y_index = np.meshgrid(indices, indices, indexing="ij")
    grid = np.stack(
        [x_index.reshape(-1), y_index.reshape(-1)],
        axis=1,
    ).astype(np.float64)
    grid *= pixel_offset
    squared_radius = np.sum(grid * grid, axis=1)
    squared_tolerance_pixels = 0.001 / (source_units_per_pixel * source_units_per_pixel)
    return grid[squared_radius < max_pixel * max_pixel + squared_tolerance_pixels]


def get_relion_translation_grid_3d(offset_range, offset_step):
    """RELION's 3D translation grid (subtomograms, ``is_3d_trans``) in Angstrom, ``[T, 3]`` float64.

    ``HealpixSampling::setTranslations`` (healpix_sampling.cpp:330, 395-418) enumerates
    ``x`` outer, ``y`` middle and ``z`` inner over ``-CEIL(range/step)..CEIL(range/step)``
    and keeps ``x^2 + y^2 + z^2 <= range^2``, in Angstrom and without the 2D grid's
    ``+0.001`` tolerance. The comparison stays in Angstrom so the axial points at exactly
    ``range`` are kept as RELION keeps them; pixels come from
    :func:`relion_translations_in_pixel_3d`.
    """
    offset_range = float(offset_range)
    offset_step = float(offset_step)
    if not np.isfinite(offset_range) or offset_range < 0.0:
        raise ValueError(f"offset_range must be finite and nonnegative, got {offset_range}")
    if not np.isfinite(offset_step) or offset_step <= 0.0:
        raise ValueError(f"offset_step must be finite and positive, got {offset_step}")
    max_index = int(np.ceil(offset_range / offset_step))
    indices = np.arange(-max_index, max_index + 1, dtype=np.int64)
    x_index, y_index, z_index = np.meshgrid(indices, indices, indices, indexing="ij")
    grid = np.stack([x_index.reshape(-1), y_index.reshape(-1), z_index.reshape(-1)], axis=1) * offset_step
    return grid[np.sum(grid * grid, axis=1) <= offset_range * offset_range]


def relion_translations_in_pixel_3d(
    translations_angst, offset_step, *, oversampling_order, pixel_size, random_perturbation=0.0
):
    """RELION's oversampled 3D translations in pixels and each one's parent, ``([T * 8^os, 3], [T * 8^os])``.

    ``HealpixSampling::getTranslationsInPixel`` (healpix_sampling.cpp:1741-1790,
    1810-1826): every parent is split into ``2^os`` children per axis at
    ``t - step/2 + (0.5 + k) step / 2^os``, enumerated ``x`` outer, ``y`` middle, ``z``
    inner, each divided by the optics group's pixel size; the iteration's perturbation
    adds ``random_perturbation * step / pixel_size`` to every axis.
    """
    translations_angst = np.asarray(translations_angst, dtype=np.float64)
    if translations_angst.ndim != 2 or translations_angst.shape[1] != 3:
        raise ValueError(f"translations_angst must have shape (T, 3), got {translations_angst.shape}")
    offset_step = float(offset_step)
    pixel_size = float(pixel_size)
    if oversampling_order < 0:
        raise ValueError("oversampling_order must be non-negative")
    n_sub = int(round(2.0**oversampling_order))
    if oversampling_order == 0:
        fine = translations_angst / pixel_size
        n_children = 1
    else:
        # RELION's association: (t - step/2) + (0.5 + k) * step / n, then / pixel_size.
        steps = (0.5 + np.arange(n_sub)) * offset_step / n_sub
        cx, cy, cz = np.meshgrid(steps, steps, steps, indexing="ij")
        children = np.stack([cx.reshape(-1), cy.reshape(-1), cz.reshape(-1)], axis=1)
        fine = ((translations_angst - 0.5 * offset_step)[:, None, :] + children[None, :, :]).reshape(-1, 3)
        fine = fine / pixel_size
        n_children = children.shape[0]
    if abs(float(random_perturbation)) > 0.0:
        fine = fine + float(random_perturbation) * offset_step / pixel_size  # RELION's association
    parent = np.repeat(np.arange(translations_angst.shape[0]), n_children)
    return fine, parent


_K1_RELION_EXACT_TRANSLATION_GRID_ENV = "RELAX_K1_RELION_EXACT_TRANSLATION_GRID"


def _k1_relion_exact_translation_grid_enabled(environ=None):
    """Return the production-on K=1 grid policy with a diagnostic opt-out."""
    env = os.environ if environ is None else environ
    raw = str(env.get(_K1_RELION_EXACT_TRANSLATION_GRID_ENV, "")).strip().lower()
    if raw in {"", "1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{_K1_RELION_EXACT_TRANSLATION_GRID_ENV} must be a boolean value, got {raw!r}"
    )


def translation_grid_for_class_count(max_pixel, pixel_offset, *, n_classes, source_units_per_pixel=1.0):
    """RELION's translation grid (``HealpixSampling::setTranslations``) for every class count.

    ``RELAX_K1_RELION_EXACT_TRANSLATION_GRID=0`` keeps the floor-division grid for K=1 as a diagnostic
    control arm.
    """
    if int(n_classes) == 1 and not _k1_relion_exact_translation_grid_enabled():
        return get_translation_grid(max_pixel, pixel_offset)
    return get_relion_translation_grid(max_pixel, pixel_offset, source_units_per_pixel=source_units_per_pixel)


def rotation_indices_to_relion_eulers(
    indices,
    healpix_order,
    *,
    rotation_index_order: str = "recovar",
    symmetry: str = "C1",
):
    """Convert ring-order full-grid indices to RELION Euler angles."""
    meta = get_relion_grid_metadata(int(healpix_order), symmetry)
    pixel_idx, psi_idx = _split_rotation_indices(
        indices,
        healpix_order,
        rotation_index_order=rotation_index_order,
        symmetry=symmetry,
    )
    return np.stack(
        [
            np.asarray(meta["rot_deg"], dtype=np.float32)[pixel_idx],
            np.asarray(meta["tilt_deg"], dtype=np.float32)[pixel_idx],
            np.asarray(meta["psi_deg"], dtype=np.float32)[psi_idx],
        ],
        axis=-1,
    ).astype(np.float32)


def relion_angular_sampling_deg(healpix_order, adaptive_oversampling=0):
    """RELION's getAngularSampling() for 3D: 360 / (6 * 2^(order+adaptive_oversampling)).

    Ref: healpix_sampling.cpp:1589-1598.
    """
    order = int(healpix_order) + int(adaptive_oversampling)
    return 360.0 / (6 * 2**order)


def relion_mstep_rotations_from_eulers(
    eulers_deg: np.ndarray,
    *,
    dtype: np.dtype = np.float32,
    left_matrices: np.ndarray | None = None,
) -> np.ndarray:
    """Return RECOVAR-frame host-inverse rotations from RELION Euler rows.

    RELION's accelerated scorer and M-step both use host
    ``generateEulerMatrices(..., inverse=true)``. It regenerates the Euler
    matrix in RFLOAT precision and calls the explicit 3x3 special case of
    ``Matrix2D::inv()``. Although an ideal rotation's inverse is its transpose,
    substituting a transpose changes float32 boundary decisions after the
    final cast. Preserve the cofactor, determinant, and division operation
    order from ``matrix2d.h`` here.

    The RELION inverse matrix is transposed once on return because RECOVAR's
    projection/backprojection rotation convention is the transpose of
    RELION's row-major Euler matrix convention.

    ``dtype`` controls only the final cast (default float32, matching
    RELION's single-precision ACC build, where this host-computed RFLOAT
    matrix is cast to XFLOAT before use on the device). The Euler matrix keeps
    RELION's ``DEG2RAD`` operation order, so host-libm trig rounding matches
    RELION's; that rounding can decide the strict radius predicate on an exact
    outer-shell pixel in an ACC double-precision run. Under
    ``ACC_DOUBLE_PRECISION`` the final cast is a no-op -- pass ``np.float64`` to match.

    ``left_matrices`` ``[N, 3, 3]`` is ``generateEulerMatrices``' ``L`` for each row: a
    tilt image's ``Aproj`` times its optics scale (acc_ml_optimiser_impl.h:1709-1734,
    3212-3234). The inverse is then ``inv(L A)`` (acc_helper_functions_impl.h:248-255).
    """
    eulers = np.asarray(eulers_deg, dtype=np.float64).reshape(-1, 3)
    return relion_mstep_rotations_from_euler_matrices(
        healpix.euler_angles_to_matrix(eulers), dtype=dtype, left_matrices=left_matrices
    )


def relion_mstep_rotations_from_euler_matrices(
    matrix: np.ndarray,
    *,
    dtype: np.dtype = np.float32,
    left_matrices: np.ndarray | None = None,
) -> np.ndarray:
    """:func:`relion_mstep_rotations_from_eulers` from the rows' RFLOAT Euler matrices ``[N, 3, 3]``.

    A caller whose rows repeat a rotation under different ``left_matrices`` (a subtomogram's tilt images)
    builds each rotation's Euler matrix once and passes it gathered per row.
    """
    matrix = np.asarray(matrix, dtype=np.float64).reshape(-1, 3, 3)
    if left_matrices is not None:
        left_matrices = np.asarray(left_matrices, dtype=np.float64)
        if left_matrices.shape != (matrix.shape[0], 3, 3):
            raise ValueError(f"left_matrices must have shape {(matrix.shape[0], 3, 3)}, got {left_matrices.shape}")
    if left_matrices is not None:
        # Matrix2D operator*: each entry sums over k in order.
        matrix = sum(left_matrices[:, :, k, None] * matrix[:, None, k, :] for k in range(3))
    inverse = np.empty_like(matrix)

    inverse[:, 0, 0] = matrix[:, 2, 2] * matrix[:, 1, 1] - matrix[:, 2, 1] * matrix[:, 1, 2]
    inverse[:, 0, 1] = -(matrix[:, 2, 2] * matrix[:, 0, 1] - matrix[:, 2, 1] * matrix[:, 0, 2])
    inverse[:, 0, 2] = matrix[:, 1, 2] * matrix[:, 0, 1] - matrix[:, 1, 1] * matrix[:, 0, 2]
    inverse[:, 1, 0] = -(matrix[:, 2, 2] * matrix[:, 1, 0] - matrix[:, 2, 0] * matrix[:, 1, 2])
    inverse[:, 1, 1] = matrix[:, 2, 2] * matrix[:, 0, 0] - matrix[:, 2, 0] * matrix[:, 0, 2]
    inverse[:, 1, 2] = -(matrix[:, 1, 2] * matrix[:, 0, 0] - matrix[:, 1, 0] * matrix[:, 0, 2])
    inverse[:, 2, 0] = matrix[:, 2, 1] * matrix[:, 1, 0] - matrix[:, 2, 0] * matrix[:, 1, 1]
    inverse[:, 2, 1] = -(matrix[:, 2, 1] * matrix[:, 0, 0] - matrix[:, 2, 0] * matrix[:, 0, 1])
    inverse[:, 2, 2] = matrix[:, 1, 1] * matrix[:, 0, 0] - matrix[:, 1, 0] * matrix[:, 0, 1]

    determinant = (
        matrix[:, 0, 0] * inverse[:, 0, 0] + matrix[:, 1, 0] * inverse[:, 0, 1] + matrix[:, 2, 0] * inverse[:, 0, 2]
    )
    inverse /= determinant[:, None, None]
    return np.swapaxes(inverse, 1, 2).astype(dtype)


@functools.lru_cache(maxsize=8)
def _relion_order_rotation_grid(order: int, symmetry: str) -> np.ndarray:
    # VDAM asks for the same unperturbed grid every iteration; read-only, callers get copies.
    R = utils.R_from_relion(healpix.coarse_orientations(order, symmetry), degrees=True)
    R.setflags(write=False)
    return R


def get_relion_rotation_grid(
    order,
    *,
    rotation_index_order: str = "recovar",
    symmetry: str = "C1",
):
    """Generate the exact RELION HEALPix rotation grid (:func:`relax.sampling.healpix.coarse_orientations`).

    Returns rotation matrices in recovar's frame that correspond to exactly
    the same set of orientations RELION uses at the given healpix_order.

    RELION orders its grid as (direction-slow, psi-fast). By default this
    function reorders to recovar's convention (psi-slow, direction-fast) so
    that ``index % n_pixels`` gives the HEALPix pixel index. Pass
    ``rotation_index_order="relion"`` to preserve RELION's native flattened
    order for source-level InitialModel parity.
    """
    symmetry = canonicalize_rotational_symmetry(symmetry)
    R = _relion_order_rotation_grid(int(order), symmetry)
    if rotation_index_order == "relion":
        return R.copy()
    if rotation_index_order != "recovar":
        raise ValueError(f"rotation_index_order must be 'recovar' or 'relion', got {rotation_index_order!r}")
    n_dir = int(get_relion_grid_metadata(int(order), symmetry)["n_pixels"])
    n_psi = R.shape[0] // n_dir
    reordered = R.reshape(n_dir, n_psi, 3, 3).transpose(1, 0, 2, 3).reshape(-1, 3, 3)
    return reordered if reordered.flags.writeable else reordered.copy()


def get_relion_hidden_rotation_grid(
    order: int,
    *,
    matrices: bool = True,
    symmetry: str = "C1",
) -> np.ndarray:
    """Return RELION's native hidden-variable rotation order."""

    if matrices:
        return get_relion_rotation_grid(
            order,
            rotation_index_order="relion",
            symmetry=symmetry,
        )
    return get_relion_rotation_grid_eulers(
        order,
        rotation_index_order="relion",
        symmetry=symmetry,
    )


def get_relion_rotation_grid_eulers(
    order,
    *,
    rotation_index_order: str = "recovar",
    symmetry: str = "C1",
):
    """Return RELION Euler angles in the same index order as get_relion_rotation_grid."""
    return get_relion_rotation_grid_eulers_float64(
        order,
        rotation_index_order=rotation_index_order,
        symmetry=symmetry,
    ).astype(np.float32)


def get_relion_rotation_grid_eulers_float64(
    order,
    *,
    rotation_index_order: str = "recovar",
    symmetry: str = "C1",
):
    """Return source-precision RELION Euler rows without public float32 truncation."""

    symmetry = canonicalize_rotational_symmetry(symmetry)
    relion_euler = healpix.coarse_orientations(int(order), symmetry)
    if rotation_index_order == "relion":
        return relion_euler
    if rotation_index_order != "recovar":
        raise ValueError(f"rotation_index_order must be 'recovar' or 'relion', got {rotation_index_order!r}")
    n_dir = int(get_relion_grid_metadata(int(order), symmetry)["n_pixels"])
    n_psi = relion_euler.shape[0] // n_dir
    return relion_euler.reshape(n_dir, n_psi, 3).transpose(1, 0, 2).reshape(-1, 3)


@dataclass(frozen=True, eq=False)
class RotationGrid:
    """Unperturbed scoring matrices/Euler rows with their order and point group."""

    rotations: np.ndarray
    rotation_eulers: np.ndarray
    healpix_order: int
    symmetry: str


class TrialGrid(NamedTuple):
    """Scoring orientations/translations and optional exact M-step matrices."""

    rotations: np.ndarray
    rotation_eulers: np.ndarray
    mstep_rotations: np.ndarray | None
    translations: jnp.ndarray


def relion_mstep_source_eulers(rotation_eulers, healpix_order, *, use_grid_eulers: bool = False, symmetry: str = "C1"):
    """Euler angles that seed the exact RELION M-step rotations of a scoring grid.

    RELION derives its M-step matrices from the sampling grid's native RFLOAT
    angles. A sealed captured grid supplies its own angles; otherwise RELION's
    canonical grid at ``healpix_order`` for the point group ``symmetry`` is
    used, unless its row count differs from the scoring grid (capped orders,
    subsets), in which case the scoring grid's own angles are used.
    """

    if use_grid_eulers:
        return np.asarray(rotation_eulers, dtype=np.float64)
    symmetry = canonicalize_rotational_symmetry(symmetry)
    # Compare row counts before building the canonical grid: a final local search at a high
    # order (subtomograms reach order 9, 9.7e9 rows) would otherwise build it only to discard it.
    if int(rotation_grid_size(healpix_order, symmetry=symmetry)) != int(np.shape(rotation_eulers)[0]):
        return np.asarray(rotation_eulers, dtype=np.float64)
    return get_relion_rotation_grid_eulers_float64(healpix_order, symmetry=symmetry)


def relion_base_translation_grid(translation_range, translation_step, *, n_classes, voxel_size):
    """Unperturbed RELION translation grid in pixels as a host float64 array.

    ``voxel_size`` (Angstrom) supplies the source units of the K=1 exact
    enumeration; a non-positive value falls back to pixel units.  The grid is
    kept in host double precision so every SamplingPerturbation starts from
    the unrounded coordinates (see ``perturbed_trial_grid``).
    """

    return translation_grid_for_class_count(
        translation_range,
        translation_step,
        n_classes=n_classes,
        source_units_per_pixel=(voxel_size if voxel_size > 0 else 1.0),
    ).astype(np.float64, copy=False)


def relion_scoring_rotation_grid(healpix_order: int, *, dtype: np.dtype = np.float32, symmetry: str = "C1") -> RotationGrid:
    """Return scoring matrices and working Euler rows in the requested precision.

    Construct matrices from canonical host-double angles before the final cast.
    Working Euler rows use scoring precision because subsequent perturbations
    operate on them. Float64 output supports ACC_DOUBLE_PRECISION diagnostics.
    """
    order = int(healpix_order)
    source_eulers = get_relion_rotation_grid_eulers_float64(order, symmetry=symmetry)
    eulers = source_eulers.astype(dtype)
    # RELION's accelerated expectation path constructs inverse projector
    # matrices on the host in RFLOAT precision, casts to XFLOAT, then copies
    # them to the device.  Preserve source Euler precision until that cast.
    rotations = relion_mstep_rotations_from_eulers(source_eulers, dtype=dtype)
    return RotationGrid(
        rotations=rotations, rotation_eulers=eulers,
        healpix_order=order, symmetry=symmetry,
    )


def get_rotation_grid_at_order(order, n_in_planes=None, matrices=True):
    """Generate HEALPix rotation grid at the specified order.

    Thin wrapper around :func:`get_rotation_grid` that makes the order
    parameter explicit and defaults to returning rotation matrices.

    Parameters
    ----------
    order : int
        HEALPix order (nside = 2^order).
    n_in_planes : int or None
        Number of in-plane rotation angles.  If None, derived from the
        HEALPix angular step (matching RELION convention).
    matrices : bool
        If True, return (N, 3, 3) rotation matrices.
        If False, return (N, 3) Euler angles in degrees.

    Returns
    -------
    np.ndarray
        Rotation matrices (N, 3, 3) or Euler angles (N, 3).
    """
    return get_rotation_grid(order, n_in_planes=n_in_planes, matrices=matrices)
