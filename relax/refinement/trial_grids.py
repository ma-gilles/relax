"""The trial grids of a pass: the coarse rotation and translation grids a numbered iteration scores
(``CoarseGrids``), their refresh when the sampling changes, the pass-1 rotations of the adaptive route, and the
final expectation's sampling and grids (``FinalSampling``; docs/math/relion_refinement_algorithm.md, section 7)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from relax import sampling
from relax.helpers.convergence import (
    RefinementState,
    _exhaustive_grid_order_for_state,
    native_final_perturbation_healpix_order,
)
from relax.helpers.resolution import ImageGeometry
from relax.refinement.ports import InputSource
from relax.sampling import relion_adaptive_pass1_rotations

if TYPE_CHECKING:
    from relax.refinement.iteration_planning import FirstIterationPolicy
    from relax.refinement.refinement_options import RefinementOptions

logger = logging.getLogger("relax.refinement.iteration_loop")


@dataclass(frozen=True)
class CoarseGrids:
    """The exhaustive coarse trial grid: built at start-up, rebuilt by ``refresh_coarse_grids``.

    ``translations`` is the device grid the expectation scores; the numbered loop replaces it with the
    iteration's perturbed copy, while ``base_translations`` keeps the unperturbed host coordinates.
    """

    rotation_grid: sampling.RotationGrid
    base_translations: np.ndarray
    translations: jnp.ndarray


def build_initial_coarse_grids(
    healpix_order,
    translations,
    *,
    translation_range,
    translation_step,
    n_classes,
    voxel_size,
    symmetry="C1",
    dtype,
) -> CoarseGrids:
    """Pair RELION's canonical initial rotations with its translation grid, in the scoring ``dtype``."""

    rotation_grid = sampling.relion_scoring_rotation_grid(
        healpix_order,
        dtype=dtype,
        symmetry=symmetry,
    )
    if translations is None:
        translations = sampling.relion_base_translation_grid(
            translation_range,
            translation_step,
            n_classes=n_classes,
            voxel_size=voxel_size,
        )
    base_translations = np.asarray(translations, dtype=np.float64)
    return CoarseGrids(
        rotation_grid=rotation_grid,
        base_translations=base_translations,
        translations=jnp.asarray(translations, dtype=dtype),
    )


def refresh_coarse_grids(
    grids: CoarseGrids,
    state: RefinementState,
    options: RefinementOptions,
    *,
    voxel_size,
    dtype,
    log: logging.Logger,
) -> CoarseGrids:
    """The exhaustive coarse grids of ``state``'s sampling, rebuilt where they changed.

    A new HEALPix order rebuilds the rotation grid (up to the exhaustive-grid cap) and the translation
    grid (a replayed range or step at the same order is the input source's, ``InputSource.coarse_grids``).
    ``grids.translations`` may be a perturbed copy; a rebuild replaces it with the base grid. Reads from ``state``: ``healpix_order`` (and the fields the
    exhaustive-grid cap reads), ``translation_range`` and ``translation_step``; from ``options``:
    ``k_class.n_classes`` and ``symmetry.point_group``.
    """
    symmetry = options.symmetry.point_group
    current_rotation_grid = grids.rotation_grid
    base_translations = grids.base_translations
    current_translations = grids.translations
    if state.healpix_order != current_rotation_grid.healpix_order:
        new_order = _exhaustive_grid_order_for_state(state)
        if new_order != current_rotation_grid.healpix_order:
            log.info(
                "Regenerating rotation grid: order %d -> %d",
                current_rotation_grid.healpix_order,
                new_order,
            )
            current_rotation_grid = sampling.relion_scoring_rotation_grid(
                new_order, dtype=dtype,
                symmetry=symmetry,
            )
        else:
            log.info(
                "Angular step refined to order %d (exhaustive grid stays at order %d — local search handles finer sampling)",
                state.healpix_order,
                current_rotation_grid.healpix_order,
            )

        base_translations = sampling.relion_base_translation_grid(
            state.translation_range,
            state.translation_step,
            n_classes=options.k_class.n_classes,
            voxel_size=voxel_size,
        )
        current_translations = jnp.asarray(base_translations, dtype=dtype)
        log.info(
            "New grid: %d rotations, %d translations (range=%.1f, step=%.1f)",
            current_rotation_grid.rotations.shape[0],
            current_translations.shape[0],
            state.translation_range,
            state.translation_step,
        )
    return CoarseGrids(current_rotation_grid, base_translations, current_translations)


def iteration_trial_grid(
    grids: CoarseGrids,
    state: RefinementState,
    options: RefinementOptions,
    random_perturbation: float,
    *,
    perturbation_order: int | None,
    sealed_grid: bool,
    dtype,
) -> sampling.TrialGrid:
    """This iteration's trial grid: the coarse grid, under RELION's sampling perturbation where one applies.

    ``perturbation_order`` is the HEALPix order whose angular sampling scales the perturbation, or None when
    none applies: then the coarse rotations and the grid's current translations are the trial grid, without
    exact M-step rotations. Reads from ``grids``: the rotation grid (rotations, Euler rows),
    ``base_translations`` (the unperturbed host translations) and ``translations``; ``state.translation_step``;
    from ``options``: ``symmetry.point_group``. A sealed grid (``sealed_grid``) keeps its own Euler rows.
    """
    rotation_grid = grids.rotation_grid
    rotation_eulers = np.asarray(
        rotation_grid.rotation_eulers,
        dtype=dtype,
    )
    if perturbation_order is None:
        return sampling.TrialGrid(
            rotations=rotation_grid.rotations,
            rotation_eulers=rotation_eulers,
            mstep_rotations=None,
            translations=grids.translations,
        )
    angsamp_deg = sampling.relion_angular_sampling_deg(perturbation_order, adaptive_oversampling=0)
    return sampling.perturbed_trial_grid(
        rotation_eulers=rotation_eulers,
        mstep_source_eulers=sampling.relion_mstep_source_eulers(
            rotation_eulers,
            perturbation_order,
            use_grid_eulers=sealed_grid,
            symmetry=options.symmetry.point_group,
        ),
        base_translations=grids.base_translations,
        translation_step=float(state.translation_step),
        random_perturbation=random_perturbation,
        angular_sampling_deg=angsamp_deg,
        dtype=dtype,
    )


def builds_coarse_pass1_rotations(
    state: RefinementState, options: RefinementOptions, first_iteration: FirstIterationPolicy, *, use_local: bool
) -> bool:
    """Whether a global search scores pass 1 with RELION's device-built coarse rotations
    (``coarse_pass1_rotations``): with adaptive oversampling, and at OS0 for a K=1 Gaussian iteration in single
    precision. RELION's coarse device geometry also applies at OS0; it stays separate from the host fine/M-step
    geometry (docs/math/zero_coarse_geometry.md).
    """
    return not use_local and (
        int(state.adaptive_oversampling) > 0
        or (
            int(state.adaptive_oversampling) == 0
            and options.k_class.n_classes == 1
            and first_iteration.score_mode == "gaussian"
            and not first_iteration.winner_take_all
            and not options.precision.use_float64_scoring
        )
    )


def coarse_pass1_rotations(
    rotation_grid: sampling.RotationGrid,
    random_perturbation: float,
    options: RefinementOptions,
    *,
    perturbation_order: int | None,
    dtype,
    log: logging.Logger,
) -> tuple[object, sampling.DevicePass1Source] | None:
    """RELION's device-built rotations for the pass-1 coarse scorer and their source, or None where the host grid serves.

    The source is the unperturbed coarse Euler rows of ``rotation_grid``, in the scoring ``dtype`` and
    then widened to float64. The perturbation applies only when the trial grid is perturbed
    (``perturbation_order``, the order whose angular step scales it; None: unperturbed, at the grid's order).
    Images on another grid or magnified rebuild their rows from the source
    (:func:`relax.sampling.relion_device_projection_rotations`).
    """
    source_eulers = np.asarray(np.asarray(rotation_grid.rotation_eulers, dtype=dtype), dtype=np.float64)
    adaptive_pass1_order = perturbation_order if perturbation_order is not None else int(rotation_grid.healpix_order)
    source = sampling.DevicePass1Source(
        source_eulers_deg=source_eulers,
        random_perturbation=random_perturbation if perturbation_order is not None else 0.0,
        angular_sampling_deg=sampling.relion_angular_sampling_deg(adaptive_pass1_order, adaptive_oversampling=0),
        use_float64=options.precision.use_float64_scoring,
    )
    adaptive_pass1_rotations = relion_adaptive_pass1_rotations(
        source.source_eulers_deg,
        source.random_perturbation,
        source.angular_sampling_deg,
        use_float64=source.use_float64,
    )
    if adaptive_pass1_rotations is None:
        return None
    log.info(
        "RELION adaptive pass 1: using %s-built coarse scorer rotations; fine/M-step rotations remain host-generated",
        "double-precision CUDA" if source.use_float64 else "CUDA",
    )
    return adaptive_pass1_rotations, source


@dataclass(frozen=True, kw_only=True)
class FinalSamplingSettings:
    """Final sampling in image pixels; None perturbation means no application."""

    relion_iteration: int
    grid_order: int
    perturbation_order: int
    translation_range: float
    translation_step: float
    pixel_size_angstrom: float
    perturbation_factor: float
    perturbation: float | None
    sampling_star: str | None = None
    sampling_star_source: str | None = None

    @property
    def random_perturbation(self) -> float:
        """Numeric perturbation for engines and the existing output format."""
        return 0.0 if self.perturbation is None else self.perturbation


@dataclass(frozen=True, kw_only=True)
class FinalSampling:
    """Scoring grids and their resolved sampling settings."""

    settings: FinalSamplingSettings
    base_rotations: np.ndarray
    base_translations: jnp.ndarray
    grid: sampling.TrialGrid


def advance_final_perturbation(previous_perturbation: float, options: RefinementOptions, *, relion_iteration: int, rng):
    """The final pass's own perturbation: the native advance where the run perturbs, else None (none applied).

    Reads ``parity.perturb_factor`` and ``perturb_seed``.
    """
    parity = options.parity
    if not parity.perturb_factor > 0:
        return None
    perturbation, seed = sampling.advance_relion_perturbation_for_iteration(
        previous_perturbation,
        perturb_factor=parity.perturb_factor,
        perturb_seed=parity.perturb_seed,
        relion_iteration=relion_iteration,
        rng=rng,
    )
    if seed is not None:
        logger.info(
            "Perturbation advance: final all-data relion_iter=%d seed=%d rp=%+.5f",
            relion_iteration, seed, perturbation,
        )
    else:
        logger.info(
            "Perturbation advance: final all-data relion_iter=%d rp=%+.5f",
            relion_iteration, perturbation,
        )
    return perturbation


def native_final_sampling_settings(
    state: RefinementState,
    image_geometry: ImageGeometry,
    options: RefinementOptions,
    *,
    grid_order: int,
    relion_iteration: int,
    perturbation: float | None,
) -> FinalSamplingSettings:
    """The final pass's sampling from the run's own state: ``state``'s translation range and step, the
    perturbation order of its last sampling, and ``perturbation`` (None: none applied)."""
    return FinalSamplingSettings(
        relion_iteration=relion_iteration,
        grid_order=grid_order,
        perturbation_order=native_final_perturbation_healpix_order(state, grid_order),
        translation_range=state.translation_range,
        translation_step=state.translation_step,
        pixel_size_angstrom=image_geometry.pixel_size_angstrom,
        perturbation_factor=options.parity.perturb_factor,
        perturbation=perturbation,
    )


def prepare_final_sampling(
    state: RefinementState,
    image_geometry: ImageGeometry,
    options: RefinementOptions,
    *,
    previous_rotation_grid: sampling.RotationGrid,
    numbered_iteration_count: int,
    source: InputSource,
    previous_perturbation: float,
    rng,
    dtype,
) -> FinalSampling:
    """Resolve the final pass's sampling (the input source's, or the run's own) and return arrays in scoring
    precision.

    Final-pass base translations are rounded to device precision before
    perturbation, unlike the host-double bases used by numbered iterations.
    ``numbered_iteration_count`` is the number of numbered iterations this run completed. Reads
    ``options.k_class.n_classes``, ``schedule.init_relion_iteration`` and the fields
    ``native_final_sampling_settings`` and ``advance_final_perturbation`` name.
    """
    grid_order = _exhaustive_grid_order_for_state(state)
    symmetry = previous_rotation_grid.symmetry
    if grid_order == previous_rotation_grid.healpix_order:
        rotation_grid = previous_rotation_grid
    else:
        rotation_grid = sampling.relion_scoring_rotation_grid(
            grid_order, dtype=dtype, symmetry=symmetry,
        )
    rotations = rotation_grid.rotations
    eulers = np.asarray(rotation_grid.rotation_eulers, dtype=dtype)
    relion_iteration = options.schedule.init_relion_iteration + numbered_iteration_count + 1
    settings = source.final_sampling_settings(
        state, image_geometry, grid_order=grid_order, relion_iteration=relion_iteration,
        native=lambda: native_final_sampling_settings(
            state, image_geometry, options, grid_order=grid_order, relion_iteration=relion_iteration,
            perturbation=advance_final_perturbation(
                previous_perturbation, options, relion_iteration=relion_iteration, rng=rng,
            ),
        ),
    )
    base_translations = jnp.asarray(
        sampling.relion_base_translation_grid(
            settings.translation_range, settings.translation_step,
            n_classes=options.k_class.n_classes, voxel_size=settings.pixel_size_angstrom,
        ),
        dtype=dtype,
    )
    if settings.perturbation is None:
        grid = sampling.TrialGrid(rotations, eulers, None, base_translations)
    else:
        grid = sampling.perturbed_trial_grid(
            rotation_eulers=eulers,
            mstep_source_eulers=sampling.relion_mstep_source_eulers(
                eulers, settings.perturbation_order, symmetry=symmetry,
            ),
            base_translations=base_translations,
            translation_step=settings.translation_step,
            random_perturbation=settings.perturbation,
            angular_sampling_deg=sampling.relion_angular_sampling_deg(
                settings.perturbation_order, adaptive_oversampling=0,
            ),
            dtype=dtype,
        )
    return FinalSampling(
        settings=settings,
        base_rotations=rotations,
        base_translations=base_translations,
        grid=grid,
    )
