"""Resolve final-expectation sampling and materialize its scoring grids.

See docs/math/relion_refinement_algorithm.md, section 7.
"""

import logging
from dataclasses import dataclass

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
from relax.refinement.refinement_options import RefinementOptions

logger = logging.getLogger("relax.refinement.iteration_loop")


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
