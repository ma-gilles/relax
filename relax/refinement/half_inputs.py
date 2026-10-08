"""Particle-half ownership, pose interpretation and correction inputs."""

import logging
from dataclasses import dataclass

import numpy as np
from recovar import utils

from relax.helpers.convergence import concatenate_pose_stacks_or_none

logger = logging.getLogger(__name__)


def optional_half_arrays(values, *, dtype=None):
    """Return optional per-half arrays, preserving precision by default.

    Float32 sources remain float32; higher-precision state is not narrowed
    unless the caller supplies an explicit dtype.
    """
    if values is None:
        return [None, None]
    return [
        np.asarray(values[0], dtype=dtype) if values[0] is not None else None,
        np.asarray(values[1], dtype=dtype) if values[1] is not None else None,
    ]


def _sigma_offset_for_half(
    current_sigma_offset_angstrom, current_sigma_offset_angstrom_per_half, half_index: int
) -> float:
    if current_sigma_offset_angstrom_per_half is None:
        return float(current_sigma_offset_angstrom)
    return float(current_sigma_offset_angstrom_per_half[half_index])


def _optional_group_count_half_pair(values):
    """Return an optional explicit group cardinality for each half-set."""
    if values is None:
        return [None, None]
    arr = np.asarray(values).reshape(-1)
    if arr.size == 1:
        arr = np.repeat(arr, 2)
    if arr.size != 2:
        raise ValueError(
            f"init_group_count must be a scalar or contain exactly two values; got shape {np.asarray(values).shape}"
        )
    counts = []
    for value in arr:
        if value is None:
            counts.append(None)
            continue
        count = int(value)
        if count < 0 or float(value) != float(count):
            raise ValueError(f"init_group_count values must be non-negative integers, got {value!r}")
        counts.append(count)
    return counts


def _logged_half_arrays(values, *, label: str):
    """Normalize per-half correction arrays and log summary statistics."""
    per_half = optional_half_arrays(values)
    for k, arr in enumerate(per_half):
        if arr is None:
            continue
        if arr.size:
            logger.info(
                "RELION mode: %s half-%d: mean=%.4f, std=%.4f, min=%.4f, max=%.4f (%d images)",
                label,
                k + 1,
                arr.mean(),
                arr.std(),
                arr.min(),
                arr.max(),
                len(arr),
            )
        else:
            logger.info("RELION mode: %s half-%d: empty", label, k + 1)
    return per_half


@dataclass(kw_only=True)
class HalfSet:
    """Persistent particles, poses and corrections for one independent half."""

    index: int
    dataset: object
    optics_group_ids: object | None = None
    rotation_eulers: object | None = None
    translations: object | None = None
    image_corrections: object | None = None
    scale_corrections: object | None = None
    group_ids: object | None = None
    group_count: int | None = None

    def centre_absent_poses(self, *, offset_dims: int):
        """Set absent angles and offsets to zeros, in place: RELION reads absent angles and origins as 0
        (exp_model.cpp:1103-1134) and searches locally around them (ml_optimiser.cpp:978-983); it never falls
        back to a global search. ``offset_dims`` is 2, or 3 for subtomograms."""
        n_particles = int(self.dataset.n_units)
        if self.rotation_eulers is None:
            self.rotation_eulers = np.zeros((n_particles, 3), dtype=np.float64)
        if self.translations is None:
            self.translations = np.zeros((n_particles, offset_dims), dtype=np.float64)

    def require_local_search_poses(self):
        if self.rotation_eulers is None or self.translations is None:
            raise ValueError(f"Local search requires orientations and translations for half {self.index + 1}")


def best_rotation_matrices(halves, *, dtype) -> list:
    """Each half's rotation matrices of its current best angles (None without angles; an empty half an empty
    ``(0, 3, 3)`` stack), in ``dtype``."""

    return [
        None
        if half.rotation_eulers is None
        else np.zeros((0, 3, 3), dtype=dtype)
        if len(half.rotation_eulers) == 0
        else np.asarray(utils.R_from_relion(np.asarray(half.rotation_eulers), degrees=True), dtype=dtype)
        for half in halves
    ]


@dataclass(frozen=True, eq=False)
class ParticlePoses:
    """One half's selected orientations and relative/absolute shifts in pixels."""

    rotations: np.ndarray
    eulers_deg: np.ndarray
    relative_translations_pixels: np.ndarray
    translations_pixels: np.ndarray


def resolve_particle_poses(
    assignments,
    translation_grid,
    *,
    previous_translations,
    own_pixel_factors,
    best_pose_rotations,
    best_pose_rotation_eulers,
    best_pose_translations,
    pose_rotations,
    local_sampling,
    dtype,
) -> ParticlePoses:
    """Resolve explicit engine poses or grid IDs into persistent particle poses.

    Supplied Euler metadata stays float64; deferred local matrices stay float32.
    ``own_pixel_factors`` are the images' reference -> own pixel factors
    (:func:`relax.refinement.optics_shapes.image_translation_factors`; None on one grid): the
    previous offset is rounded in each image's own pixels, as RELION does.
    See ``docs/math/relion_refinement_algorithm.md``, section 3.
    """
    from recovar import utils

    from relax.local.local_layout import _selected_rotation_matrices
    from relax.relion.relion_metadata import _relion_metadata_translations
    from relax.sampling import build_local_search_grid_metadata

    if best_pose_rotations is not None:
        rotations = np.asarray(best_pose_rotations, dtype=dtype)
        eulers = (
            np.asarray(best_pose_rotation_eulers, dtype=np.float64)
            if best_pose_rotation_eulers is not None
            else utils.R_to_relion(rotations, degrees=True).astype(dtype)
        )
        translations = np.asarray(best_pose_translations, dtype=dtype)
    else:
        rotation_ids = assignments // translation_grid.shape[0]
        translation_ids = assignments % translation_grid.shape[0]
        if local_sampling is not None:
            if local_sampling.rotations is None:
                grid_metadata = build_local_search_grid_metadata(
                    local_sampling.search.healpix_order, symmetry=local_sampling.search.symmetry,
                )
                rotations = _selected_rotation_matrices(
                    rotation_ids,
                    None,
                    grid_metadata,
                    random_perturbation=local_sampling.perturbation,
                    angular_sampling_deg=local_sampling.angular_step_deg,
                )
                eulers = utils.R_to_relion(np.asarray(rotations), degrees=True).astype(dtype)
            else:
                rotations = np.asarray(local_sampling.rotations, dtype=dtype)[rotation_ids]
                if local_sampling.rotation_eulers is not None:
                    eulers = np.asarray(local_sampling.rotation_eulers, dtype=dtype)[rotation_ids]
                else:
                    eulers = utils.R_to_relion(np.asarray(rotations), degrees=True).astype(dtype)
        else:
            rotations = np.asarray(pose_rotations, dtype=dtype)[rotation_ids]
            eulers = utils.R_to_relion(np.asarray(rotations), degrees=True).astype(dtype)
        translations = np.asarray(translation_grid)[translation_ids]
    return ParticlePoses(
        rotations=rotations,
        eulers_deg=eulers,
        relative_translations_pixels=translations,
        translations_pixels=_relion_metadata_translations(
            previous_translations, translations, own_pixel_factors=own_pixel_factors, dtype=dtype,
        ),
    )


@dataclass(frozen=True, eq=False)
class ParticlePoseUpdate:
    """Previous pose snapshots and resolved poses for one numbered iteration."""

    previous_rotations: list[np.ndarray | None]
    previous_translations_pixels: list[np.ndarray | None]
    current: tuple[ParticlePoses, ParticlePoses]


@dataclass(frozen=True, eq=False)
class PoseComparison:
    """Half-ordered pose stacks consumed by convergence-change measurements."""

    current_rotations: np.ndarray | None
    previous_rotations: np.ndarray | None
    current_translations_pixels: np.ndarray | None
    previous_translations_pixels: np.ndarray | None


def prepare_particle_pose_update(
    scores,
    halves,
    translation_grid,
    *,
    previous_rotations,
    local_sampling,
    dtype,
) -> ParticlePoseUpdate:
    """Snapshot prior poses and resolve engine outputs without installing state.

    Snapshot both halves before resolving either. Canonical Euler metadata,
    relative search shifts and absolute particle offsets retain their existing
    precision and frames. See ``docs/math/relion_refinement_algorithm.md``,
    section 3. The controller owns publication and diagnostic ordering.
    """
    from relax.refinement.optics_shapes import image_translation_factors

    previous_rotations_snapshot = [
        np.asarray(rot).copy() if rot is not None else None for rot in previous_rotations
    ]
    previous_translations_snapshot = [
        np.asarray(half.translations).copy() if half.translations is not None else None
        for half in halves
    ]
    current = tuple(
        resolve_particle_poses(
            scores.hard_assignments[k],
            translation_grid,
            previous_translations=previous_translations_snapshot[k],
            own_pixel_factors=image_translation_factors(halves[k].dataset),
            best_pose_rotations=scores.best_pose_rotations[k],
            best_pose_rotation_eulers=scores.best_pose_rotation_eulers[k],
            best_pose_translations=scores.best_pose_translations[k],
            pose_rotations=scores.pose_rotations[k],
            local_sampling=local_sampling,
            dtype=dtype,
        )
        for k in range(2)
    )
    return ParticlePoseUpdate(
        previous_rotations=previous_rotations_snapshot,
        previous_translations_pixels=previous_translations_snapshot,
        current=current,
    )


def prepare_pose_comparison(
    pose_update: ParticlePoseUpdate,
    *,
    translation_dimension: int,
    dtype,
    log,
) -> PoseComparison:
    """Prepare aligned current/previous pose stacks in physical half order.

    Missing or malformed stacks retain the existing convergence primitive's
    refusal and logging behavior. Publication has already happened in the
    controller; this operation neither updates poses nor decides convergence.
    """
    current_rotations = concatenate_pose_stacks_or_none(
        [poses.rotations for poses in pose_update.current],
        trailing_shape=(3, 3),
        label="current rotation",
        dtype=dtype,
        logger=log,
    )
    previous_rotations = concatenate_pose_stacks_or_none(
        pose_update.previous_rotations,
        trailing_shape=(3, 3),
        label="previous rotation",
        dtype=dtype,
        logger=log,
    )
    current_translations_pixels = concatenate_pose_stacks_or_none(
        [poses.translations_pixels for poses in pose_update.current],
        trailing_shape=(translation_dimension,),
        label="current translation",
        dtype=dtype,
        logger=log,
    )
    previous_translations_pixels = concatenate_pose_stacks_or_none(
        pose_update.previous_translations_pixels,
        trailing_shape=(translation_dimension,),
        label="previous translation",
        dtype=dtype,
        logger=log,
    )

    return PoseComparison(
        current_rotations=current_rotations,
        previous_rotations=previous_rotations,
        current_translations_pixels=current_translations_pixels,
        previous_translations_pixels=previous_translations_pixels,
    )


def initialize_halfsets(
    datasets,
    *,
    optics_group_ids,
    previous_best_translations,
    previous_best_rotation_eulers,
    image_corrections,
    scale_corrections,
    group_ids,
    group_count,
) -> tuple[HalfSet, HalfSet]:
    """Normalize input arrays once and attach each to its particle half."""
    translations = optional_half_arrays(previous_best_translations)
    rotation_eulers = optional_half_arrays(previous_best_rotation_eulers)
    images = _logged_half_arrays(image_corrections, label="image_corrections")
    scales = _logged_half_arrays(scale_corrections, label="scale_corrections")
    groups = optional_half_arrays(group_ids, dtype=np.int64)
    counts = _optional_group_count_half_pair(group_count)
    return tuple(
        HalfSet(
            index=k,
            dataset=datasets[k],
            optics_group_ids=optics_group_ids[k],
            translations=translations[k],
            rotation_eulers=rotation_eulers[k],
            image_corrections=images[k],
            scale_corrections=scales[k],
            group_ids=groups[k],
            group_count=counts[k],
        )
        for k in range(2)
    )


def normalize_sigma_offset_per_half(values):
    """Return a strict two-element float list for half-specific sigma offsets."""
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size != 2:
        raise ValueError(
            f"translation_sigma_angstrom_per_half must contain exactly two values; got shape {np.asarray(values).shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("translation_sigma_angstrom_per_half must be finite")
    return [float(arr[0]), float(arr[1])]


def as_sigma_offset_half_pair(values):
    """Return a scalar or explicit pair as a strict two-half sigma list."""

    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        arr = np.repeat(arr, 2)
    return normalize_sigma_offset_per_half(arr)


def _mean_sigma_offset_per_half(values):
    per_half = normalize_sigma_offset_per_half(values)
    if per_half is None:
        return None
    return float(0.5 * (per_half[0] + per_half[1]))


@dataclass(frozen=True)
class SigmaOffset:
    """The translation prior width in Angstrom: the value the halves share and each half's own.

    A producer supplies both; the shared value is not always the mean of the two.
    """

    shared_angstrom: object
    per_half_angstrom: object


def copy_optional_float_pair(values):
    """Two widths as Python floats, for a record that outlives the run's arrays (None stays None)."""

    if values is None:
        return None
    return [float(values[0]), float(values[1])]


def sigma_offset_from_halves(per_half) -> SigmaOffset:
    """Each half's width with their mean as the shared value."""

    return SigmaOffset(_mean_sigma_offset_per_half(per_half), per_half)


def configure_half_image_preprocessing(
    experiment_datasets,
    *,
    pixel_size_angstrom,
    particle_diameter_angstrom: float | None,
    width_mask_edge_px: float,
    fourier_backend: str,
    source_faithful_spectrum_norm: bool,
    log,
) -> None:
    """Configure half image backends and masks before refinement state is built.

    Shape classes use their own pixels; SPA and tilt-image masks keep the
    reference pixel scalar used by the existing refinement path.
    """
    from relax.helpers.batch_planning import _image_backend
    from relax.refinement.optics_shapes import MultiShapeHalf
    from relax.refinement.tomo_half import TomoHalf

    multi_shape_halves = isinstance(experiment_datasets[0], MultiShapeHalf)
    # A half of several image shapes sets up each shape class's images, masked with
    # the class's own pixel size as RELION does.
    image_datasets = [
        dataset
        for half in experiment_datasets
        for dataset in (
            [c.dataset for c in half.classes]
            if isinstance(half, MultiShapeHalf)
            else [half.images] if isinstance(half, TomoHalf) else [half]
        )
    ]
    for ds in image_datasets:
        backend = _image_backend(ds)
        if backend is None:
            continue
        mask_pixel_size = ds.voxel_size if multi_shape_halves else pixel_size_angstrom
        if hasattr(backend, "set_relion_fourier_backend"):
            from relax.cuda import (
                kernels as _em_cuda_kernels,  # noqa: F401  (registers the relion_cuda preprocessor, relax split seam S2)
            )

            backend.set_relion_fourier_backend(fourier_backend)
        if source_faithful_spectrum_norm and getattr(backend, "relion_fourier_backend", None) not in (None, "relion_cuda"):
            # The fresh K=1 defaults score from RELION's CUDA image preprocessing;
            # fail here instead of inside the first sparse pass 2.
            raise ValueError(
                "fresh K=1 refinement defaults (source-faithful powerClass normalization and "
                "exact RELION BPref operands) require RELION CUDA image preprocessing; pass "
                "--image-fourier-backend relion_cuda or disable the fresh particle order"
            )
        if particle_diameter_angstrom is not None and particle_diameter_angstrom > 0:
            backend.set_relion_image_mask(
                pixel_size=mask_pixel_size,
                particle_diameter_ang=particle_diameter_angstrom,
                width_mask_edge_px=width_mask_edge_px,
            )
            log.info(
                "RELION mode: image mask radius=%.1f px (particle_diameter=%.1f A, edge=%g px)",
                particle_diameter_angstrom / (2.0 * mask_pixel_size),
                particle_diameter_angstrom,
                width_mask_edge_px,
            )
