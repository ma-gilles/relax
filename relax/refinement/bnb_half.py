"""The branch-and-bound pose search of one K=1 global-search half (``--pose_search bnb``).

:func:`bnb_applies` decides once per half whether the search runs. :func:`search_half` builds the search levels from
the iteration's sampling (coarser HEALPix orders under the same perturbation, then the iteration's own coarse grid)
and runs :func:`relax.scoring.bnb_search.run_search`; the controller takes its window centres as the half's previous
offsets, so RELION's pre-shift, the offset prior and the reported offsets all follow them.
:func:`window_supports` then finds pass 2's supports on the RELION window around those centres, in place of pass 1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from relax.refinement.refinement_options import BnBSearchOptions, RefinementOptions
from relax.scoring.bnb_search import (
    BnBLevel,
    BnBPruning,
    BnBSearchResult,
    children_table,
    coarse_level_prior,
    offset_log_prior,
    run_search,
    translation_grid,
    window_significance,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HalfBnB:
    """One half's search: its result, its last level (the iteration's coarse grid) and the pass-1 fields it used."""

    search: BnBSearchResult
    final_level: BnBLevel
    request_fields: dict
    image_current_size: int | None
    image_batch_size: int


def bnb_applies(
    options: RefinementOptions,
    *,
    use_local: bool,
    tomo_halves: bool,
    k_class_enabled: bool,
    use_adaptive: bool,
    multi_shape_halves: bool,
    score_mode: str,
    winner_take_all: bool,
) -> bool:
    """Whether this half runs the branch-and-bound search: ``--pose_search bnb`` on a global, adaptive, Gaussian
    K=1 single-particle search with one image shape. Refuses a configuration the search does not implement."""

    if options.adaptive.pose_search != "bnb" or use_local:
        return False
    if tomo_halves or k_class_enabled or multi_shape_halves:
        raise NotImplementedError("--pose_search bnb runs K=1 single-particle refinement with one image shape")
    if options.symmetry.point_group != "C1":
        raise NotImplementedError("--pose_search bnb builds its rotation levels for C1 only")
    if options.consistency.offset_prior_units != "consistent":
        raise ValueError("--pose_search bnb scores a global offset prior and needs --offset_prior_units consistent")
    return bool(use_adaptive and score_mode == "gaussian" and not winner_take_all)


def _coarse_size(order: int, image_geometry, options: RefinementOptions, current_size: int | None) -> int:
    """RELION's pass-1 image size for HEALPix ``order`` (its angular step's adequately sampled resolution)."""

    from relax.reconstruction.resolution import relion_coarse_image_size
    from relax.sampling.healpix import angular_sampling_deg

    return int(
        relion_coarse_image_size(
            angular_sampling_deg(int(order)),
            float(image_geometry.pixel_size_angstrom),
            int(image_geometry.box_size),
            particle_diameter=options.schedule.particle_diameter_ang,
            current_size=current_size,
            clamp_box_size=int(image_geometry.box_size),
        )
    )


def _perturbed_grid(order: int, random_perturbation: float) -> np.ndarray:
    """RELION's rotation grid of ``order`` under the iteration's perturbation (scaled by that order's step)."""

    from relax.sampling.grids import relion_scoring_rotation_grid
    from relax.sampling.healpix import angular_sampling_deg
    from relax.sampling.perturbation import apply_relion_rotation_perturbation_to_eulers

    grid = relion_scoring_rotation_grid(int(order))
    rotations, _ = apply_relion_rotation_perturbation_to_eulers(
        grid.rotation_eulers, random_perturbation, angular_sampling_deg(int(order))
    )
    return np.asarray(rotations, dtype=np.float32)


def _levels(
    settings: BnBSearchOptions,
    sampling,
    rotation_log_prior,
    image_geometry,
    options: RefinementOptions,
    *,
    sigma_offset_angstrom: float,
    window_radius: float,
) -> list[BnBLevel]:
    """The search levels, coarse to fine: ``settings.levels`` coarser HEALPix orders (none below
    ``settings.min_order``), then the iteration's coarse grid, each at RELION's coarse size of its order; with
    ``settings.half_band``, the first order is scored at half that size first."""

    from relax.sampling.healpix import angular_sampling_deg

    order = int(sampling.current_healpix_order)
    orders = sorted({max(min(int(settings.min_order), order), order - k) for k in range(int(settings.levels) + 1)})
    box = int(image_geometry.box_size)
    pixel = float(image_geometry.pixel_size_angstrom)
    current_size = sampling.image_window_size
    sigma_px = float(sigma_offset_angstrom) / pixel
    radius = min(max(float(settings.translation_sigmas) * sigma_px, float(window_radius)), 0.3 * box)
    grids = {o: _perturbed_grid(o, sampling.random_perturbation) for o in orders[:-1]}
    grids[order] = np.asarray(sampling.effective_rotations, dtype=np.float32)
    schedule = [(o, _coarse_size(o, image_geometry, options, current_size)) for o in orders]
    if settings.half_band:
        schedule.insert(0, (orders[0], max(8, 2 * (schedule[0][1] // 4))))
    prior = None
    if rotation_log_prior is not None:
        prior = np.asarray(rotation_log_prior, dtype=np.float32).reshape(-1)
        if prior.size != grids[order].shape[0]:
            raise ValueError(f"rotation prior of {prior.size} rows for a grid of {grids[order].shape[0]} rotations")
    children = [None] * len(schedule)
    for k in range(1, len(schedule)):
        parent_order, child_order = schedule[k - 1][0], schedule[k][0]
        radius_deg = 0.0 if parent_order == child_order else settings.child_overlap * angular_sampling_deg(parent_order)
        children[k] = children_table(grids[parent_order], grids[child_order], radius_deg=radius_deg)
    priors = [None] * len(schedule)
    priors[-1] = prior
    for k in range(len(schedule) - 1, 0, -1):
        priors[k - 1] = coarse_level_prior(priors[k], children[k])
    levels = []
    for k, (o, size) in enumerate(schedule):
        step = float(np.clip(box / (2.0 * size), settings.min_step_px, settings.max_step_px))
        translations = translation_grid(radius, step)
        levels.append(
            BnBLevel(
                order=o,
                rotations=grids[o],
                rotation_log_prior=priors[k],
                children=children[k],
                size=size,
                translations=translations,
                translation_log_prior=offset_log_prior(
                    translations, pixel_size=pixel, sigma_offset_angstrom=sigma_offset_angstrom
                ),
            )
        )
    return levels


def pass1_request_fields(
    particles, half_data, options: RefinementOptions, *, image_batch_size: int, relion_translation_angle_scale: float
) -> dict:
    """The pass-1 request fields every search level shares; the search grid is absolute (no pre-shift)."""

    from relax.relion.geometry import PROJECTION_PADDING_FACTOR
    from relax.steps.scoring_policy import RELION_FOURIER_WINDOW_SQUARE

    if half_data.projector is None:
        raise ValueError("--pose_search bnb scores with RELION's coarse projector and needs the prepared projector")
    return dict(
        experiment_dataset=particles.dataset,
        noise_variance=half_data.noise_variance,
        class_log_priors=np.zeros(1, dtype=np.float64),
        adaptive_fraction=0.999,
        max_significants=-1,
        image_batch_size=int(image_batch_size),
        rotation_block_size=4096,
        score_with_masked_images=True,
        image_corrections=particles.image_corrections,
        scale_corrections=particles.scale_corrections,
        image_pre_shifts=None,
        half_spectrum_scoring=True,
        projection_padding_factor=PROJECTION_PADDING_FACTOR,
        square_window=RELION_FOURIER_WINDOW_SQUARE,
        use_float64_scoring=False,
        use_float64_projections=False,
        relion_projector_half=half_data.projector.data,
        relion_projector_r_max=half_data.projector.r_max,
        collect_significance=False,
        optics_group_ids=particles.optics_group_ids,
        nyquist_column_counting=options.consistency.nyquist_column_counting,
        relion_translation_angle_scale=float(relion_translation_angle_scale),
        engine_tuning=options.environment.engine,
        scoring_overrides=options.environment.scoring,
    )


def search_half(
    half,
    sampling,
    options: RefinementOptions,
    image_geometry,
    *,
    window_radius: float,
    relion_translation_angle_scale: float,
) -> HalfBnB:
    """Run the search on ``half`` (``NumberedHalfInputs``) at the iteration's ``sampling``.

    ``window_radius`` (pixels) is the RELION translation window the centres are averaged over and the least
    search radius.
    """

    settings = options.adaptive.bnb
    particles = half.data.particles
    levels = _levels(
        settings,
        sampling,
        half.direction_priors.rotation_log_prior,
        image_geometry,
        options,
        sigma_offset_angstrom=float(half.sigma_offset_angstrom),
        window_radius=window_radius,
    )
    fields = pass1_request_fields(
        particles, half.data, options, image_batch_size=settings.image_batch_size,
        relion_translation_angle_scale=relion_translation_angle_scale,
    )
    result = run_search(
        levels,
        fields,
        pruning=BnBPruning(
            mass_fraction=settings.mass_fraction,
            margin=settings.margin,
            max_kept=settings.max_kept,
            min_kept=settings.min_kept,
        ),
        image_current_size=sampling.image_window_size,
        centre_radius=max(float(window_radius), 1.0),
        image_batch_size=settings.image_batch_size,
    )
    previous = particles.translations
    if previous is not None:
        moved = np.linalg.norm(result.centres - np.asarray(previous, dtype=np.float64), axis=1)
        logger.info(
            "BnB half %d: window centres moved from the previous offsets by median %.2f px, 90%% %.2f px, max %.2f px; "
            "offset spread median %.2f px",
            particles.index + 1, float(np.median(moved)), float(np.quantile(moved, 0.9)), float(moved.max()),
            float(np.median(result.translation_spread)),
        )
    return HalfBnB(
        search=result,
        final_level=levels[-1],
        request_fields=fields,
        image_current_size=sampling.image_window_size,
        image_batch_size=settings.image_batch_size,
    )


def recentred_window(bnb: HalfBnB, sampling, options: RefinementOptions, *, voxel_size: float, dtype):
    """The half's sampling with RELION's translation window widened, when needed, to hold the offset posterior
    (``BnBSearchOptions.window_quantile``), and the window's perturbed translations.

    A widened window keeps RELION's step unless the radius would span more than ``window_steps`` steps; it never
    narrows RELION's window and never reaches past the search radius.
    """

    from dataclasses import replace

    import jax.numpy as jnp

    from relax.sampling.grids import relion_base_translation_grid
    from relax.sampling.perturbation import apply_relion_translation_perturbation

    settings = options.adaptive.bnb
    relion_radius = float(np.max(np.linalg.norm(np.asarray(sampling.base_translations), axis=1), initial=0.0))
    if not settings.window_quantile > 0.0:
        return sampling, sampling.current_translations
    search_radius = float(np.max(np.linalg.norm(bnb.final_level.translations, axis=1), initial=0.0))
    cover = float(np.quantile(bnb.search.hold_radius, settings.window_quantile))
    radius = min(cover, search_radius)
    if not radius > relion_radius + 1e-6:
        return sampling, sampling.current_translations
    step = max(float(sampling.translation_step), radius / float(settings.window_steps))
    base = relion_base_translation_grid(radius, step, n_classes=1, voxel_size=voxel_size)
    current = jnp.asarray(
        apply_relion_translation_perturbation(base, float(sampling.random_perturbation), step), dtype=dtype
    )
    logger.info(
        "BnB window: radius %.2f -> %.2f px, step %.2f -> %.2f px (%d translations), to hold 99%% of the offset "
        "posterior of %.0f%% of the particles",
        relion_radius, radius, float(sampling.translation_step), step, int(base.shape[0]),
        100.0 * settings.window_quantile,
    )
    return replace(sampling, base_translations=base, current_translations=current, translation_step=step), current


def window_supports(bnb: HalfBnB, sampling, translation_log_prior, *, adaptive_fraction: float, max_significants: int):
    """Pass 2's coarse supports of the half: each image's significant samples over its candidates and the RELION
    window around its centre (``sampling``'s coarse translations under the perturbation), as the compacted CSR."""

    from relax.classification.given_pose_classes import given_support_csr
    from relax.sampling.perturbation import apply_relion_translation_perturbation

    window = apply_relion_translation_perturbation(
        np.asarray(sampling.base_translations, dtype=np.float64),
        float(sampling.random_perturbation),
        float(sampling.translation_step),
    )
    fields = dict(bnb.request_fields)
    fields["image_pre_shifts"] = bnb.search.centres
    supports = window_significance(
        bnb.final_level,
        fields,
        bnb.search.candidates,
        window,
        np.asarray(translation_log_prior, dtype=np.float32),
        image_current_size=bnb.image_current_size,
        adaptive_fraction=float(adaptive_fraction),
        max_significants=int(max_significants),
        image_batch_size=bnb.image_batch_size,
    )
    csr = given_support_csr(
        supports, n_coarse_rot=int(bnb.final_level.rotations.shape[0]), n_coarse_trans=int(window.shape[0])
    )
    return csr, np.asarray([row.size for row in supports], dtype=np.int32)
