"""RELION image-size scheduling and resolution helpers.

Owns current-size bootstrapping, coarse image sizes, FSC/data-vs-prior
boundaries, and first-iteration resolution and tau2-reporting rules. The
iteration controller selects when to apply these rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from relax.helpers.convergence import healpix_angular_step
from relax.helpers.fourier_window import quantize_current_size
from relax.reconstruction.regularization_relion import (
    compute_current_size_relion,
    fsc_to_relion_ssnr,
    resolution_from_data_vs_prior,
)

if TYPE_CHECKING:
    from relax.helpers.convergence import RefinementState
    from relax.refinement.refinement_options import RefinementOptions


@dataclass(frozen=True, kw_only=True)
class ImageGeometry:
    """Physical input-image grid; independent of Fourier windows and model support."""

    image_shape: tuple[int, int]
    pixel_size_angstrom: float

    def __post_init__(self):
        pixel_size = float(self.pixel_size_angstrom)
        if not np.isfinite(pixel_size) or pixel_size <= 0:
            raise ValueError(f"Particle pixel size must be finite and positive, got {pixel_size}")
        object.__setattr__(self, "pixel_size_angstrom", pixel_size)

    @property
    def box_size(self) -> int:
        return self.image_shape[0]


def relion_local_pass1_current_size(
    *,
    pre_update_healpix_order: int,
    pixel_size: float,
    box_size: int,
    particle_diameter: float | None,
    current_size: int | None,
) -> int | None:
    """Return RELION's local pass-1 size for the current expectation.

    RELION computes ``image_coarse_size`` before ``updateAngularSampling``.
    When an expectation advances the sampling order, its parent hypotheses
    therefore use the incoming order for Fourier sizing even though the
    updated order controls the parent grid and fine-child expansion.
    """

    coarse_size = relion_coarse_image_size(
        healpix_angular_step(int(pre_update_healpix_order)),
        pixel_size,
        box_size,
        particle_diameter=particle_diameter,
        current_size=current_size,
    )
    return coarse_size if coarse_size < int(box_size) else None

def relion_expectation_coarse_size_order(
    *,
    state_healpix_order: int,
    replay_saved_healpix_order: int | None,
) -> int:
    """Choose the pre-update sampling order at an expectation boundary.

    In strict replay, RECOVAR's live refinement state may have advanced after
    the preceding M-step.  RELION instead enters the next expectation from
    the preceding numbered sampling STAR and only then updates angular
    sampling, so the saved replay order is authoritative for Fourier sizing.
    """

    if replay_saved_healpix_order is not None:
        return int(replay_saved_healpix_order)
    return int(state_healpix_order)


def shell_index_to_resolution_angstrom(shell_index, box_size, voxel_size):
    """Convert a Fourier shell index into a real-space resolution in Angstrom."""
    if voxel_size <= 0:
        return float(shell_index)
    if shell_index <= 0:
        return float("inf")
    return float(box_size) * float(voxel_size) / shell_index


def relion_optics_image_current_sizes(
    model_current_size,
    *,
    model_box_size,
    model_pixel_size,
    optics_image_sizes,
    optics_pixel_sizes,
):
    """Return RELION's per-optics particle-image Fourier window sizes.

    RELION keeps ``mymodel.current_size`` in reference-map coordinates, but
    remaps the particle window separately for each optics group::

        remap = (image_pixel_size * image_size) /
                (model_pixel_size * model_box_size)
        image_current_size = 2 * ceil(0.5 * remap * model_current_size)

    The distinction matters even when the nominal fields print identically in
    STAR files.  For example, an MRC voxel size stored as float32 may be
    1.4166666269 while the optics STAR stores 1.416667.  At model size 56 the
    upward ``ceil`` then produces a 58-pixel particle window, while the
    Projector and BackProjector retain radius 28 from the 56-pixel model size.
    """

    image_sizes = np.asarray(optics_image_sizes, dtype=np.int64).reshape(-1)
    pixel_sizes = np.asarray(optics_pixel_sizes, dtype=np.float64).reshape(-1)
    if image_sizes.shape != pixel_sizes.shape or image_sizes.size == 0:
        raise ValueError(
            "optics image sizes and pixel sizes must be non-empty arrays with identical shapes",
        )
    if model_current_size <= 0 or model_box_size <= 0 or model_pixel_size <= 0.0:
        raise ValueError("model current size, original size, and pixel size must be positive")
    if np.any(image_sizes <= 0) or np.any(pixel_sizes <= 0.0):
        raise ValueError("optics image sizes and pixel sizes must be positive")

    remap_sizes = (
        pixel_sizes * image_sizes.astype(np.float64)
    ) / (float(model_pixel_size) * float(model_box_size))
    current_sizes = 2 * np.ceil(0.5 * remap_sizes * float(model_current_size))
    return np.minimum(current_sizes.astype(np.int64), image_sizes)


# Default threshold for when adaptive pass 2 is skipped.
def compute_coarse_image_size(
    angular_step_deg,
    pixel_size,
    box_size,
    particle_diameter=None,
):
    """Compute the coarse image size for pass 1 of adaptive oversampling.

    RELION formula (expectation.cpp line 5760):
        rotated_distance = (angular_step / 360) * pi * particle_diameter
        coarse_resolution = rotated_distance / 1.2       (3D)
        image_coarse_size = 2 * ceil(pixel_size * ori_size / coarse_resolution)

    Parameters
    ----------
    angular_step_deg : float
        Effective angular step in degrees (after oversampling).
    pixel_size : float
        Pixel size in Angstrom.
    box_size : int
        Original image box size in pixels.
    particle_diameter : float or None
        Particle diameter in Angstrom.  If None, use box_size * pixel_size.

    Returns
    -------
    coarse_size : int
        Coarse image size (diameter in pixels), clamped to [8, box_size].
    """
    if particle_diameter is None:
        particle_diameter = box_size * pixel_size

    rotated_distance = (angular_step_deg / 360.0) * np.pi * particle_diameter
    coarse_resolution = rotated_distance / 1.2  # keepsafe_factor for 3D

    if coarse_resolution <= 0:
        return box_size

    coarse_size = int(2 * np.ceil(pixel_size * box_size / coarse_resolution))
    coarse_size = max(8, min(coarse_size, box_size))
    return coarse_size


def clamp_relion_coarse_image_size(coarse_size, current_size, box_size):
    """Clamp pass-1 image size the way RELION does.

    RELION computes ``image_coarse_size`` from the angular step and particle
    diameter, then clamps it to ``image_current_size`` rather than forcing a
    smaller fallback. See ``ml_optimiser.cpp`` around the
    ``image_coarse_size = XMIPP_MIN(image_current_size, image_coarse_size)``
    update.
    """
    coarse_size = int(coarse_size)
    if coarse_size % 2 != 0:
        coarse_size += 1
    coarse_size = max(8, min(coarse_size, int(box_size)))
    if current_size is None:
        return coarse_size
    return min(int(current_size), coarse_size)


def relion_coarse_image_size(
    angular_step_deg, pixel_size, box_size, *, particle_diameter, current_size, clamp_box_size=None
):
    """RELION's adaptive pass-1 image size: :func:`compute_coarse_image_size` clamped by
    :func:`clamp_relion_coarse_image_size` to ``current_size`` and ``clamp_box_size`` (``box_size`` when None;
    the numbered planner computes at the first optics group's box and clamps at the model box).
    """
    coarse_size = compute_coarse_image_size(angular_step_deg, pixel_size, box_size, particle_diameter=particle_diameter)
    return clamp_relion_coarse_image_size(
        coarse_size, current_size, box_size if clamp_box_size is None else clamp_box_size
    )


def bootstrap_current_size_relion(init_current_size: int, box_size: int, incr_size: int = 10) -> int:
    """Match RELION's first expectation-time current_size growth step.

    RELION seeds the initial resolution from ``--ini_high`` and then immediately
    calls ``updateImageSizeAndResolutionPointers()`` before the first E-step.
    At startup ``ave_Pmax == 0`` and ``has_high_fsc_at_limit == false``, so the
    first current_size is the initial resolution shell plus ``incr_size``.
    """
    init_shell = max(0, int(np.ceil(init_current_size / 2.0)))
    raw_cs = compute_current_size_relion(
        init_shell,
        box_size,
        ave_Pmax=0.0,
        has_high_fsc_at_limit=False,
        incr_size=incr_size,
    )
    return quantize_current_size(raw_cs, box_size=box_size)


def bootstrap_current_size_from_ini_high_relion(
    box_size: int,
    voxel_size: float,
    ini_high_angstrom: float | None,
    incr_size: int = 10,
) -> int | None:
    """Bootstrap RELION's first current_size directly from ``--ini_high``."""
    if ini_high_angstrom is None or float(ini_high_angstrom) <= 0.0:
        return None
    init_shell = max(1, int(np.round(float(box_size) * float(voxel_size) / float(ini_high_angstrom))))
    return bootstrap_current_size_relion(2 * init_shell, box_size=box_size, incr_size=incr_size)


def zero_shells_past_current_size(shell_curve, *, current_size, box_size, dtype=np.float32):
    """Zero the shells of an FSC or data-vs-prior curve beyond RELION's inclusive current-size boundary.

    BackProjector includes radii ``R <= current_size / 2``.  The boundary
    shell therefore remains part of RELION's full-array threshold scan;
    only shells starting at ``current_size // 2 + 1`` are unavailable.
    The shell axis is the last one (``(n_shells,)`` or ``(n_classes, n_shells)``).
    """
    truncated = np.asarray(shell_curve, dtype=dtype).copy()
    if int(current_size) < int(box_size):
        first_unavailable_shell = min(truncated.shape[-1], int(current_size) // 2 + 1)
        truncated[..., first_unavailable_shell:] = 0.0
    return truncated


def class_resolution_shells(data_vs_prior, *, box_size):
    """Each class's ``updateCurrentResolution`` shell, in class order.

    ``data_vs_prior`` is ``(n_classes, n_shells)``, already truncated to the
    current size. Class3D scans each class without the split-half
    high-resolution recheck; RELION's current resolution is the maximum.
    """
    return [
        resolution_from_data_vs_prior(dvp_class, box_size=box_size, allow_high_res_recovery=False)
        for dvp_class in np.asarray(data_vs_prior)
    ]


def k1_current_resolution_shell(data_vs_prior, *, current_size, box_size, dtype=np.float32):
    """Return RELION's ``updateCurrentResolution`` shell from a K=1 DVP curve.

    Mirrors ``MlOptimiser::updateCurrentResolution`` (relion/src/
    ml_optimiser.cpp:6753): the last shell before data_vs_prior drops below
    1, with split-half auto-refine's high-resolution recheck.
    ``data_vs_prior`` is ``(n_shells,)``. Shells beyond ``current_size`` are
    unavailable and zeroed first. See docs/math/relion_refinement_algorithm.md
    section 7 for the split-half and final all-data curves.
    """
    dvp = zero_shells_past_current_size(data_vs_prior, current_size=current_size, box_size=box_size, dtype=dtype)
    return resolution_from_data_vs_prior(dvp, box_size=box_size, allow_high_res_recovery=True)


def class_current_resolution_shell(data_vs_prior, *, current_size, box_size, dtype=np.float32):
    """Return RELION's ``updateCurrentResolution`` shell from the classes' DVP curves.

    ``data_vs_prior`` is ``(n_classes, n_shells)``. Shells beyond
    ``current_size`` are zeroed first; the result is the maximum over classes
    of each class's shell (``class_resolution_shells``, no recheck).
    """
    dvp = zero_shells_past_current_size(data_vs_prior, current_size=current_size, box_size=box_size, dtype=dtype)
    return max(class_resolution_shells(dvp, box_size=box_size))


def initialize_resolution_from_fsc(
    state: RefinementState, options: RefinementOptions, *, box_size: int, voxel_size: float, dtype
) -> None:
    """Seed current/previous resolution from a caller-provided initial FSC curve."""
    schedule = options.schedule
    fsc = np.asarray(schedule.init_fsc, dtype=dtype).copy()
    previous_current_size = int(schedule.init_current_size)
    if previous_current_size < box_size:
        fsc[min(len(fsc), previous_current_size // 2) :] = 0.0
    data_vs_prior = np.asarray(fsc_to_relion_ssnr(fsc, tau2_fudge=options.parity.tau2_fudge))
    resolution_shell = resolution_from_data_vs_prior(data_vs_prior, box_size=box_size, allow_high_res_recovery=True)
    resolution_angstrom = shell_index_to_resolution_angstrom(resolution_shell, box_size, voxel_size)
    if np.isfinite(resolution_angstrom) and resolution_angstrom > 0.0:
        state.current_resolution = float(resolution_angstrom)
        state.previous_resolution = float(resolution_angstrom)


def initialize_resolution_from_firstiter_ini_high(
    state: RefinementState, options: RefinementOptions, *, box_size: int, voxel_size: float
) -> None:
    """Seed current/previous resolution from the first-iteration ini_high lowpass."""
    initialize_resolution_from_ini_high(
        state,
        options.parity.relion_firstiter_ini_high_angstrom,
        box_size=box_size,
        voxel_size=voxel_size,
    )


def initialize_resolution_from_ini_high(
    state: RefinementState, ini_high_angstrom: float, *, box_size: int, voxel_size: float
) -> None:
    """Seed current/previous resolution with RELION's start-up ``--ini_high`` shell.

    ``MlOptimiserMpi::iterate`` calls ``updateCurrentResolution`` before iteration 1
    (ml_optimiser_mpi.cpp:4028); with ``ini_high > 0`` at ``iter == 0`` it sets
    ``current_resolution`` to the shell ``ROUND(ori_size pixel_size / ini_high)``
    (ml_optimiser.cpp:6768-6770). Iteration 1 then joins the half accumulators up to
    ``max(low_resol_join_halves, 1 / current_resolution)`` (ml_optimiser_mpi.cpp:3280)
    and compares its own resolution with it for the stall counter (ml_optimiser.cpp:6835).
    """
    pixel_size = float(voxel_size if voxel_size > 0 else 1.0)
    shell = _firstiter_cc_ini_high_resolution_shell(box_size, pixel_size, ini_high_angstrom)
    resolution_angstrom = shell_index_to_resolution_angstrom(shell, box_size, pixel_size)
    state.current_resolution = float(resolution_angstrom)
    state.previous_resolution = float(resolution_angstrom)


def _firstiter_cc_ini_high_resolution_shell(box_size, voxel_size, ini_high_angstrom):
    """RELION's firstiter_cc current-resolution shell from ``--ini_high``."""
    px = float(voxel_size if voxel_size > 0 else 1.0)
    shell = int(np.floor(int(box_size) * px / float(ini_high_angstrom) + 0.5))
    return max(1, min(int(box_size) // 2, shell))


def firstiter_cc_scheduling_resolution_shell(
    resolution_shell,
    *,
    emulate_relion_firstiter_cc,
    ini_high_angstrom,
    relion_iteration,
    box_size,
    voxel_size,
):
    """Apply RELION's iter-1 ``--firstiter_cc`` current-size rule.

    ``MlOptimiser::updateCurrentResolution`` uses the ``--ini_high`` shell at
    physical iteration 1 regardless of whether refinement has one or multiple
    classes. Keep this override independent of the K=1/K-class data-vs-prior
    calculation so both paths schedule iteration 2 identically.
    """

    if (
        emulate_relion_firstiter_cc
        and ini_high_angstrom is not None
        and int(relion_iteration) == 1
    ):
        return _firstiter_cc_ini_high_resolution_shell(
            box_size,
            voxel_size,
            ini_high_angstrom,
        )
    return int(resolution_shell)

@dataclass(frozen=True)
class ResolutionEstimate:
    """Post-reconstruction curve and the observed/first-iteration scheduling shells."""

    data_vs_prior: np.ndarray
    observed_shell: int
    scheduling_shell: float


def estimate_k1_iteration_resolution(
    data_vs_prior,
    *,
    current_size,
    box_size,
    voxel_size,
    emulate_relion_firstiter_cc,
    ini_high_angstrom,
    relion_iteration,
    dtype,
) -> ResolutionEstimate:
    """Estimate the K1 resolution from the curve the split-half reconstruction used.

    ``data_vs_prior`` is the half-1 ``ssnr_shells`` of the iteration's prior
    estimate. The shell scan applies RELION's split-half high-resolution
    recheck. Keep the observed shell separate from RELION's first-iteration
    ``ini_high`` scheduling override.
    See ``docs/math/relion_refinement_algorithm.md#6-sampling-transitions-and-convergence``.
    """
    data_vs_prior = zero_shells_past_current_size(
        data_vs_prior,
        current_size=current_size,
        box_size=box_size,
        dtype=dtype,
    )
    observed_shell = resolution_from_data_vs_prior(data_vs_prior, box_size=box_size, allow_high_res_recovery=True)
    scheduling_shell = float(
        firstiter_cc_scheduling_resolution_shell(
            observed_shell,
            emulate_relion_firstiter_cc=emulate_relion_firstiter_cc,
            ini_high_angstrom=ini_high_angstrom,
            relion_iteration=relion_iteration,
            box_size=box_size,
            voxel_size=voxel_size,
        )
    )
    return ResolutionEstimate(data_vs_prior, observed_shell, scheduling_shell)


def estimate_class_iteration_resolution(
    class_data_vs_prior,
    *,
    current_size,
    box_size,
    voxel_size,
    emulate_relion_firstiter_cc,
    ini_high_angstrom,
    relion_iteration,
    dtype,
) -> ResolutionEstimate:
    """Estimate the Class3D resolution from the recorded ``(n_classes, n_shells)`` curves.

    The observed shell is the maximum over classes, without the split-half
    recheck. Keep it separate from RELION's first-iteration ``ini_high``
    scheduling override.
    See ``docs/math/relion_refinement_algorithm.md#6-sampling-transitions-and-convergence``.
    """
    data_vs_prior = zero_shells_past_current_size(
        class_data_vs_prior,
        current_size=current_size,
        box_size=box_size,
        dtype=dtype,
    )
    observed_shell = max(class_resolution_shells(data_vs_prior, box_size=box_size))
    scheduling_shell = float(
        firstiter_cc_scheduling_resolution_shell(
            observed_shell,
            emulate_relion_firstiter_cc=emulate_relion_firstiter_cc,
            ini_high_angstrom=ini_high_angstrom,
            relion_iteration=relion_iteration,
            box_size=box_size,
            voxel_size=voxel_size,
        )
    )
    return ResolutionEstimate(data_vs_prior, observed_shell, scheduling_shell)



def firstiter_cc_ini_high_tau2_taper(
    n_shells,
    box_size,
    voxel_size,
    ini_high_angstrom,
    *,
    filter_edgewidth,
):
    """RELION's squared post-firstiter ``ini_high`` taper for tau2 state."""

    if ini_high_angstrom is None or float(ini_high_angstrom) <= 0.0:
        return np.ones(int(n_shells), dtype=np.float64)
    edge = float(filter_edgewidth)
    radius = float(box_size) * float(voxel_size) / float(ini_high_angstrom) - edge / 2.0
    radius_p = radius + edge
    shells = np.arange(int(n_shells), dtype=np.float64)
    taper = np.ones(int(n_shells), dtype=np.float64)
    taper[shells > radius_p] = 0.0
    transition = (shells >= radius) & (shells <= radius_p)
    taper[transition] = 0.5 - 0.5 * np.cos(np.pi * (radius_p - shells[transition]) / edge)
    return taper * taper


def _firstiter_cc_ini_high_tapered(
    values,
    box_size,
    voxel_size,
    ini_high_angstrom,
    *,
    filter_edgewidth,
):
    """``values`` times :func:`firstiter_cc_ini_high_tau2_taper` along its last (shell) axis.

    Class3D's per-class tau2 and data_vs_prior curves ``[K, n_shells]``
    (ml_optimiser.cpp:6389-6420); the result keeps ``values``' dtype.
    """

    values = np.asarray(values)
    taper = firstiter_cc_ini_high_tau2_taper(
        int(values.shape[-1]),
        box_size,
        voxel_size,
        ini_high_angstrom,
        filter_edgewidth=filter_edgewidth,
    )
    return values * taper.astype(values.dtype)
