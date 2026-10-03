"""Resolve final-expectation sampling and materialize its scoring grids.

See docs/math/relion_refinement_algorithm.md, section 7.
"""

import logging
import os
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from relax import sampling
from relax.diagnostics.relion_replay import (
    _perturbation_restart_state_iteration,
    _resolve_replay_random_perturbation,
    select_final_sampling_star,
)
from relax.helpers.convergence import (
    RefinementState,
    _exhaustive_grid_order_for_state,
    _native_final_perturbation_healpix_order,
)
from relax.helpers.resolution import ImageGeometry
from relax.refinement.refinement_options import RelionParityOptions
from relax.relion.relion_metadata import read_relion_sampling_metadata

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


def _log_replayed_translation_grid_change(settings, *, replay_dir, replay_prefix, n_classes):
    if settings.sampling_star_source != "final":
        return
    numbered_path = os.path.join(
        replay_dir, f"{replay_prefix}_it{settings.relion_iteration - 1:03d}_sampling.star",
    )
    if not os.path.exists(numbered_path):
        return
    numbered = read_relion_sampling_metadata(numbered_path)
    numbered_range = numbered["offset_range"] / settings.pixel_size_angstrom
    numbered_step = numbered["offset_step"] / settings.pixel_size_angstrom
    numbered_grid = sampling._relion_base_translation_grid(
        numbered_range, numbered_step,
        n_classes=n_classes, voxel_size=settings.pixel_size_angstrom,
    ).astype(np.float32)
    final_grid = sampling._relion_base_translation_grid(
        settings.translation_range, settings.translation_step,
        n_classes=n_classes, voxel_size=settings.pixel_size_angstrom,
    ).astype(np.float32)
    if numbered_grid.shape != final_grid.shape or not np.allclose(
        numbered_grid, final_grid, rtol=0.0, atol=1e-6,
    ):
        logger.info(
            "RELION final all-data sampling grid differs from last numbered sampling: "
            "numbered n=%d range=%.9g step=%.9g hp=%d; final n=%d range=%.9g step=%.9g hp=%d",
            numbered_grid.shape[0], numbered_range, numbered_step, numbered["healpix_order"],
            final_grid.shape[0], settings.translation_range, settings.translation_step,
            settings.perturbation_order,
        )


def _resolve_final_sampling_settings(
    state: RefinementState,
    image_geometry: ImageGeometry,
    *,
    grid_order: int,
    last_numbered_iteration: int,
    parity: RelionParityOptions,
    active_replay_dir: str | None,
    require_final_state: bool,
    previous_perturbation: float,
    rng,
    n_classes: int,
) -> FinalSamplingSettings:
    relion_iteration = last_numbered_iteration + 1
    replay_dir = (
        parity.final_sampling_replay_relion_dir
        if parity.final_sampling_replay_relion_dir is not None
        else active_replay_dir
    )
    if replay_dir is not None:
        star, source, candidates = select_final_sampling_star(
            replay_dir, parity.perturb_replay_relion_prefix,
            final_iteration=relion_iteration,
            previous_iteration=last_numbered_iteration,
            require_final_state=require_final_state,
        )
        if star is not None:
            metadata = read_relion_sampling_metadata(star)
            replay_iteration = last_numbered_iteration if source == "last-numbered" else relion_iteration
            perturbation, perturbation_source = _resolve_replay_random_perturbation(
                star_value=metadata["random_perturbation"],
                perturbation_factor=metadata["perturbation_factor"],
                relion_iteration=replay_iteration,
                replay_dir=str(replay_dir),
                replay_prefix=parity.perturb_replay_relion_prefix,
                explicit_seed=parity.perturb_seed,
                precision_mode=parity.perturb_replay_precision,
                restart_state_iteration=_perturbation_restart_state_iteration(
                    parity.perturb_replay_restart_state_iterations, replay_iteration,
                ),
            )
            pixel_size = image_geometry.pixel_size_angstrom
            settings = FinalSamplingSettings(
                relion_iteration=relion_iteration,
                grid_order=grid_order,
                perturbation_order=metadata["healpix_order"],
                translation_range=metadata["offset_range"] / pixel_size,
                translation_step=metadata["offset_step"] / pixel_size,
                pixel_size_angstrom=pixel_size,
                perturbation_factor=metadata["perturbation_factor"],
                perturbation=perturbation,
                sampling_star=star,
                sampling_star_source=source,
            )
            _log_replayed_translation_grid_change(
                settings, replay_dir=replay_dir,
                replay_prefix=parity.perturb_replay_relion_prefix, n_classes=n_classes,
            )
            logger.info(
                "Perturbation replay: final all-data relion_iter=%d rp=%+.12g pf=%.3f "
                "relion_hp_order=%d offset_range=%.3f px offset_step=%.3f px source=%s/%s",
                replay_iteration, settings.random_perturbation, settings.perturbation_factor,
                settings.perturbation_order, settings.translation_range, settings.translation_step,
                source, perturbation_source,
            )
            return settings
        logger.info(
            "Perturbation replay: final all-data sampling STAR missing for relion_iter=%d (%s); "
            "leaving final trial grid unperturbed",
            relion_iteration, ", ".join(path for path, _source in candidates),
        )
        # A final-only replay directory historically applies a zero perturbation
        # when native perturbation is enabled, without advancing its RNG.
        perturbation = 0.0 if active_replay_dir is None and parity.perturb_factor > 0 else None
    elif parity.perturb_factor > 0:
        perturbation, seed = sampling._advance_relion_perturbation(
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
    else:
        perturbation = None
    return FinalSamplingSettings(
        relion_iteration=relion_iteration,
        grid_order=grid_order,
        perturbation_order=_native_final_perturbation_healpix_order(state, grid_order),
        translation_range=state.translation_range,
        translation_step=state.translation_step,
        pixel_size_angstrom=image_geometry.pixel_size_angstrom,
        perturbation_factor=parity.perturb_factor,
        perturbation=perturbation,
    )


def prepare_final_sampling(
    state: RefinementState,
    image_geometry: ImageGeometry,
    *,
    previous_rotation_grid: sampling.RotationGrid,
    last_numbered_iteration: int,
    parity: RelionParityOptions,
    active_replay_dir: str | None,
    require_final_state: bool,
    previous_perturbation: float,
    rng,
    n_classes: int,
    dtype=np.float32,
) -> FinalSampling:
    """Resolve native/replayed sampling and return arrays in scoring precision.

    Final-pass base translations are rounded to device precision before
    perturbation, unlike the host-double bases used by numbered iterations.
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
    settings = _resolve_final_sampling_settings(
        state, image_geometry,
        grid_order=grid_order,
        last_numbered_iteration=last_numbered_iteration,
        parity=parity,
        active_replay_dir=active_replay_dir,
        require_final_state=require_final_state,
        previous_perturbation=previous_perturbation,
        rng=rng,
        n_classes=n_classes,
    )
    base_translations = jnp.asarray(
        sampling._relion_base_translation_grid(
            settings.translation_range, settings.translation_step,
            n_classes=n_classes, voxel_size=settings.pixel_size_angstrom,
        ),
        dtype=dtype,
    )
    if settings.perturbation is None:
        grid = sampling.TrialGrid(rotations, eulers, None, base_translations)
    else:
        grid = sampling._perturbed_trial_grid(
            rotation_eulers=eulers,
            mstep_source_eulers=sampling._relion_mstep_source_eulers(
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
