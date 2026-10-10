"""InitialModel (VDAM) E-step for optics groups on several image shapes (RELION S3b).

RELION scores each particle on its own optics group's grid against the one reference
(``ml_optimiser.cpp:6085``, the scale difference of ``getFourierTransformsAndCtfs``).
:func:`run_by_shape_class` is that orchestration around the adaptive route of
:mod:`relax.vdam.adaptive_estep`: one engine call per image shape with the shape's own
translation grids, pre-shifts and offset prior, merged into one result on the reference
grid, and the particles' new offsets in reference pixels.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from relax.classification.k_class import run_dense_k_class_em_adaptive
from relax.helpers.convergence import healpix_angular_step
from relax.refinement import shape_class_scoring
from relax.relion.optics_aberrations import reported_rotations
from relax.sampling.orientation_priors import relion_round_away_from_zero, relion_sigma_offset_prior_center
from relax.vdam import native_sampling
from relax.vdam.estep_common import ENGINE_DISC_TYPE, InitialModelEstepConfig
from relax.vdam.state import InitialModelState


def _unscaled_poses(result, scale: float):
    """The engine's best pose matrices without the projection scale: poses are reported unscaled."""

    def unscale(rotations):
        return None if rotations is None else reported_rotations(rotations, scale)

    per_class = result.per_class_best_pose_rotations
    return result._replace(
        best_pose_rotations=unscale(result.best_pose_rotations),
        per_class_best_pose_rotations=None if per_class is None else tuple(unscale(r) for r in per_class),
    )


def run_by_shape_class(
    half,
    state: InitialModelState,
    config: InitialModelEstepConfig,
    means,
    mean_variance,
    route,
    class_route,
    image_indices: np.ndarray,
    *,
    engine_call: dict[str, Any],
    route_kwargs: dict[str, Any],
):
    """The adaptive route once per image shape of the subset, merged (RELION S3b).

    Each shape class scores its images on its own grid (``shape_class_scoring.shape_class_engine_inputs``:
    projection scale, remapped sizes, noise rows, reference-grid backprojector). Its translation
    grids are rebuilt in its pixels from the Angstrom grid, and its pre-shifts, prior centers and
    coarse ``pdf_offset`` from each particle's offset in its own pixels, rounded there
    (ml_optimiser.cpp:6085). Returns the merged result and the subset's new offsets in
    reference pixels: the rounded old offset plus the winning translation, in the image's pixels.

    ``route`` is the iteration's reference-grid route (``adaptive_estep.AdaptiveRouteGrids``) and
    ``class_route(factor)`` rebuilds it with the translations scaled to a class's pixels;
    ``engine_call`` and ``route_kwargs`` are the single-shape engine call's keywords.
    """

    estep_sampling = config.sampling
    shape_translations = estep_sampling.multi_shape_translations
    coarse_sizing = (
        healpix_angular_step(estep_sampling.pass1_healpix_order),
        estep_sampling.particle_diameter_ang,
    )

    offsets_ref = shape_translations.offsets_px[image_indices]
    coarse_prior_ref = np.asarray(shape_translations.coarse_prior_translations, dtype=np.float64)
    sigma_angstrom = shape_translations.sigma_angstrom
    noise_radial = np.asarray(state.sigma2_noise, dtype=np.float64) * float(state.box_size) ** 4
    kwargs = {
        name: value
        for name, value in route_kwargs.items()
        if name not in ("image_pre_shifts", "translation_prior_centers", "translation_log_prior")
    }
    current_size = route_kwargs.get("current_size")
    if route.fill_fine_rows is not None:
        # Each class projects its own copy of the fine rotations: fill every deferred row first.
        route.fill_fine_rows(None)
    new_offsets = np.empty_like(offsets_ref)
    results = []
    for shape_class in half.classes:
        factor = float(shape_class.translation_factor)
        grids = class_route(factor).grids
        inputs = shape_class_scoring.shape_class_engine_inputs(
            shape_class,
            half,
            noise_radial=noise_radial,
            coarse_rotations=route.pass1_rotations,
            fine_rotations=route.grids.fine_rotations,
            fine_mstep_rotations=route.grids.fine_mstep_rotations,
            coarse_translations=route.grids.coarse_translations,
            fine_translations=route.grids.fine_translations,
            coarse_current_size=engine_call["coarse_current_size"],
            fine_current_size=engine_call["fine_current_size"],
            reference_current_size=current_size,
            engine_kwargs=kwargs,
            rotation_source={
                "coarse_healpix_order": int(engine_call["coarse_healpix_order"]),
                "adaptive_oversampling": int(engine_call["oversampling_order"]),
                "random_perturbation": estep_sampling.random_perturbation,
                "coarse_device_source": route.pass1_device_source,
            },
            coarse_sizing=coarse_sizing,
        )
        offsets = offsets_ref[shape_class.image_indices] * factor
        pre_shifts = relion_round_away_from_zero(offsets)
        coarse_prior = native_sampling.sampling_translation_log_prior(
            coarse_prior_ref * factor,
            voxel_size=float(shape_class.pixel_size),
            sigma_angstrom=sigma_angstrom,
            old_offsets=pre_shifts,
        )
        class_result = run_dense_k_class_em_adaptive(
            inputs.dataset,
            means,
            mean_variance,
            inputs.noise_variance,
            inputs.coarse_rotations,
            grids.coarse_translations,
            inputs.fine_rotations,
            grids.fine_translations,
            route.grids.rotation_parent_map,
            grids.translation_parent_map,
            ENGINE_DISC_TYPE,
            **{
                **engine_call,
                "fill_fine_rows": None,
                "coarse_current_size": inputs.coarse_current_size,
                "fine_current_size": inputs.fine_current_size,
                "coarse_translation_log_prior": coarse_prior,
                "fine_mstep_rotations_override": inputs.fine_mstep_rotations,
                "coarse_translation_phase_source": grids.coarse_translation_phase_source,
            },
            **{
                **inputs.engine_kwargs,
                "image_pre_shifts": pre_shifts,
                "translation_prior_centers": relion_sigma_offset_prior_center(offsets),
                "translation_log_prior": coarse_prior,
            },
        )
        if shape_class.scale != 1.0:
            class_result = _unscaled_poses(class_result, shape_class.scale)
        fine_translations = np.asarray(grids.fine_translations, dtype=np.float64)
        winners = np.mod(np.asarray(class_result.pose_assignments, dtype=np.int64), fine_translations.shape[0])
        new_offsets[shape_class.image_indices] = (
            np.asarray(pre_shifts, dtype=np.float64) + fine_translations[winners, :2]
        ) / factor
        results.append(class_result)
    merged = shape_class_scoring.merge_k_class_engine_results(results, half.classes, half.n_units, int(half.image_shape[0]))
    return merged, new_offsets
