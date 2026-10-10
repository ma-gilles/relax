"""Particle-half ownership and a half's inputs to one expectation (poses: ``particle_poses.py``)."""

import logging
from dataclasses import dataclass, replace
from typing import NamedTuple

import numpy as np

from relax.refinement.projector_preparation import PreparedProjector

logger = logging.getLogger(__name__)


class HalfPair(NamedTuple):
    """One value for each half-set. What builds the pair says whether the halves share one value
    (``HalfPair.shared``) or each has its own: an array's shape never decides it."""

    half1: object
    half2: object

    @classmethod
    def shared(cls, value) -> "HalfPair":
        """Both halves refer to the one ``value``."""
        return cls(value, value)

    def map(self, convert) -> "HalfPair":
        """``convert`` of each half's value; halves sharing one value share its one result."""
        first = convert(self.half1)
        return HalfPair(first, first if self.half2 is self.half1 else convert(self.half2))


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
    from relax.helpers.batch_planning import image_backend
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
        backend = image_backend(ds)
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


@dataclass(frozen=True, kw_only=True)
class HalfScoringData:
    """Persistent particle half and the model operands for one expectation."""

    particles: HalfSet
    reference: object
    noise_variance: object
    noise_radial: object | None = None
    mean_variance: object | None = None
    projector: PreparedProjector | None = None
    scale_group_ids: object | None = None
    scale_group_count: int | None = None
    scale_correction_data_vs_prior: object | None = None
    image_seed_classes: object | None = None


def local_search_centre_half(half, angle_priors, state):
    """The half whose ``rotation_eulers`` centre this iteration's local angular searches.

    relion_refine centres a local search on the particle's current angles, except while an orientational
    prior from ``--sigma_ang`` is on below ``--auto_local_healpix_order`` in auto-refine: there each angle
    with an ``rlnAngle*Prior`` is centred on that prior, every iteration, and the others on the current angle
    (``getFourierTransformsAndCtfs``; non-helical). ``angle_priors`` is the half's ``[N, 3]`` priors in
    degrees, NaN where absent, or None. Reads from ``state``: ``do_local_search``, ``auto_sampling``,
    ``healpix_order`` and ``auto_local_healpix_order``.
    See ``docs/math/relion_refinement_algorithm.md#iteration-convergence-policy`` (local searches from the start).
    """

    if (
        angle_priors is None
        or half.rotation_eulers is None
        or not (state.do_local_search and state.auto_sampling)
        or state.healpix_order >= state.auto_local_healpix_order
    ):
        return half
    current = np.asarray(half.rotation_eulers)
    centres = np.where(np.isnan(angle_priors), current, angle_priors).astype(current.dtype)
    return replace(half, rotation_eulers=centres)
