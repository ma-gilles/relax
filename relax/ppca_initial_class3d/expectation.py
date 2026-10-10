"""Mixture E-step over physical-particle tiles with joint class/pose normalization."""

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np
from recovar.reconstruction.noise import make_radial_noise

from relax import sampling
from relax.ppca_initial_class3d.diagnostics import PosteriorDiagnostics
from relax.ppca_initial_class3d.stream import (
    accumulate_mixture_full_row_tile,
    accumulate_mixture_oversampled_tile,
    plan_mixture_tile_images,
    prepare_mixture_full_row_stream,
    prepare_mixture_oversampled_stream,
)
from relax.ppca_initial_model.iteration_loop import pass1_image_size
from relax.ppca_initial_model.tomo import TiltParticles, load_tilt_tile, tilt_tiles
from relax.ppca_refinement.config import GeometryConfig, ScheduleConfig, ScoringConfig
from relax.ppca_refinement.full_row_stream import prepare_full_row_stream
from relax.ppca_refinement.oversampled_stream import plan_job_chunk, prepare_oversampled_stream, relion_child_grids


@dataclass(frozen=True)
class PoseGrid:
    rotations: np.ndarray
    eulers: np.ndarray
    translations: np.ndarray
    direction_ids: np.ndarray
    order: int = 0  # coarse HEALPix order of the rotations (the parents' order for a child grid)
    translation_step: float = 0.0  # coarse shift step in pixels (the parents' step for a child grid)


def oversampled_update(config, iteration) -> bool:
    """Whether the update at ``iteration`` runs RELION's two passes: from ``oversampling_start`` on, or from the
    second stage's first update when unset (the first stage's posteriors are flat and its grid is cheap, as the
    single-model controller notes); earlier updates score the stage's full grid."""
    if config.oversampling_start is None:
        start = config.stages[1][0] if len(config.stages) > 1 else config.stages[0][0]
    else:
        start = config.oversampling_start
    return bool(config.oversampling) and iteration >= start


def pose_grid(dataset, config, iteration, *, children=False, sampling_state=None):
    """Canonical Euler metadata, never reconstructed from rounded matrices.

    ``children`` gives the stage's order-1 child grid (parent-major, the grid an oversampled update's pass 2
    indexes), with the sampler's own Euler rows of the children. With ``sampling_state`` (auto-sampling), its
    coarse order, offset step and range replace the stage's order and the configured shift grid.
    """
    if sampling_state is None:
        hp, step, shift_range = config.stage(iteration)[1], config.shift_step, config.shift_range
    else:
        hp, step, shift_range = (int(sampling_state.healpix_order), float(sampling_state.offset_step_px),
                                 float(sampling_state.offset_range_px))
    translations = (
        sampling.get_relion_translation_grid_3d(shift_range, step)
        if isinstance(dataset, TiltParticles)
        else sampling.get_relion_translation_grid(max_pixel=shift_range, pixel_offset=step)
    ).astype(np.float32)
    if children:
        rotations, translations = relion_child_grids(hp, translations, step)
        n_coarse = sampling.rotation_grid_size(hp)
        eulers = sampling.get_oversampled_rotation_grid_from_samples(
            np.arange(n_coarse), hp, 1, return_source_eulers=True, rotation_index_order="relion"
        )[-1]
        eulers = np.asarray(eulers, np.float64)
        if eulers.shape[0] != rotations.shape[0]:
            raise ValueError("The child grid's Euler rows do not match its rotations")
    else:
        rotations = sampling.get_relion_hidden_rotation_grid(hp, matrices=True).astype(np.float32)
        eulers = sampling.get_relion_hidden_rotation_grid(hp, matrices=False)
    _, directions = np.unique(eulers[:, :2], axis=0, return_inverse=True)
    return PoseGrid(np.asarray(rotations, np.float32), eulers, np.asarray(translations, np.float32),
                    directions.reshape(-1), int(hp), float(step))


@dataclass(frozen=True)
class ComponentStatistics:
    lhs_tri: object
    residual_gradient: object


@dataclass
class BatchStatistics:
    components: list[ComponentStatistics | None]
    class_mass: np.ndarray
    rotation_mass: np.ndarray
    offset_second: float = 0.0
    log_likelihood: float = 0.0
    count: int = 0
    residual_num: object = None
    residual_den: object = None
    posterior: object = None
    pmax_sum: float = 0.0
    # Per class: the tiles' significant-sample counts, held mass and cap flags per particle (oversampled updates).
    oversampling: list = field(default_factory=list)
    membership_tiles: list = field(default_factory=list)


def mixture_tiles(dataset, state, config, groups, iteration, grid, *, moments, diameter_ang=None, candidates=None):
    """Yield (group index, noise group, joint tile result), one noise owner at a time.

    Releasing each owner's K streams avoids storing K full models for every
    tomography optics/noise group. All components share image/CTF/noise operands.
    ``grid`` is the update's coarse pose grid (:func:`pose_grid`). An oversampled update
    (:func:`oversampled_update`) scores pass 1 on RELION's coarse image window of the
    grid's angular step (``diameter_ang`` sizes it; None takes the box) and pass 2 on the
    children of each particle's significant samples, per class. ``candidates(tile)`` gives a
    tile's per-particle candidate coarse rows (local searches,
    :func:`relax.ppca_initial_class3d.local_search.tile_candidates`); None scores every pose.
    """
    tomo = isinstance(dataset, TiltParticles)
    radius, hp = config.stage(iteration)[0], int(grid.order)
    pass2_size = 2 * min(radius, dataset.grid_size // 2 - 1)
    geometry = GeometryConfig(current_size=pass2_size, q=config.q, volume_domain="fourier_half")
    oversampled = oversampled_update(config, iteration)
    pass1_size = pass1_image_size(hp, dataset, pass2_size, diameter_ang) if oversampled else pass2_size
    pass1_geometry = GeometryConfig(current_size=pass1_size, q=config.q, volume_domain="fourier_half")
    schedule = ScheduleConfig(image_batch_size=config.image_batch_size, rotation_block_size=config.rotation_block_size)
    scoring = ScoringConfig(relion_texture_interp=False, full_real_observation=True)
    angular = (np.asarray(state.direction_prior, np.float32)
               if state.direction_order == hp and state.direction_prior is not None
               else np.full(len(grid.rotations), 1 / len(grid.rotations), np.float32))
    log_rotation = np.log(angular, where=angular > 0, out=np.full_like(angular, -np.inf))
    log_translation = -jnp.sum(jnp.asarray(grid.translations)**2, axis=-1) / (2 * state.offset_variance)
    log_translation -= jnp.log(jnp.sum(jnp.exp(log_translation)))
    noise_rows = np.asarray(state.noise) if tomo else np.asarray(state.noise)[None]
    owners = dataset.particle_noise_group if tomo else np.zeros(dataset.n_images, np.int32)
    tile_size, job_chunk = None, None
    for noise_group in np.unique(owners[np.concatenate(groups)]):
        nv = np.asarray(make_radial_noise(noise_rows[noise_group], dataset.image_shape), np.float32)

        # The engine restricts a tile image's poses through its coarse parents. Local searches give every
        # coarse pose its own parent so a particle's candidates select rows; a global search keeps the single
        # parent of the single-model controller, whose tile mask is then one flag per image.
        n_rotations, n_translations = len(grid.rotations), len(grid.translations)
        if candidates is None:
            parents = dict(rotation_parent=np.zeros(n_rotations, np.int32),
                           translation_parent=np.zeros(n_translations, np.int32),
                           n_coarse_rotations=1, n_coarse_translations=1)
        else:
            parents = dict(rotation_parent=np.arange(n_rotations, dtype=np.int32),
                           translation_parent=np.arange(n_translations, dtype=np.int32),
                           n_coarse_rotations=n_rotations, n_coarse_translations=n_translations)

        def class_streams(stream_geometry):
            return [
                prepare_full_row_stream(
                    dataset, model[:, 0], model[:, 1:], noise_variance=nv,
                    rotations=grid.rotations, translations=grid.translations,
                    rotation_log_prior=log_rotation, translation_log_prior=np.asarray(log_translation),
                    **parents,
                    geometry=stream_geometry, schedule=schedule, scoring=scoring,
                    gemm_precision=config.gemm_precision, tile_loader=load_tilt_tile if tomo else None,
                    # Export all conditional embeddings, including low-probability class tails.
                    pass2_mass_floor=config.pass2_mass_floor if moments else 0.0,
                ) for model in state.theta
            ]

        streams = class_streams(pass1_geometry)
        mixture = prepare_mixture_full_row_stream(streams, state.class_prior)
        if tile_size is None:
            tile_size = plan_mixture_tile_images(mixture, config.image_batch_size,
                                                retained_groups=len(groups) if moments else 0)
        streams = [s._replace(tile_images=tile_size) for s in streams]
        if oversampled:
            pass2_streams = class_streams(geometry) if pass1_size != pass2_size else streams
            fine_rotations, fine_translations = relion_child_grids(hp, grid.translations, grid.translation_step)
            # RELION's cap, at most the number of coarse poses (small test grids have fewer).
            max_significant = min(config.max_significant, len(grid.rotations) * len(grid.translations))
            ostreams = [
                prepare_oversampled_stream(
                    coarse, fine_rotations, fine_translations, pass2=pass2._replace(tile_images=tile_size),
                    adaptive_fraction=config.target_mass, max_significant=max_significant,
                ) for coarse, pass2 in zip(streams, pass2_streams)
            ]
            if job_chunk is None:
                job_chunk = plan_job_chunk(ostreams[0], tile_size)
            ostreams = [o._replace(job_chunk=job_chunk) for o in ostreams]
            mixture = prepare_mixture_oversampled_stream(ostreams, state.class_prior)
            accumulate = accumulate_mixture_oversampled_tile
        else:
            mixture = mixture._replace(streams=tuple(streams))
            accumulate = accumulate_mixture_full_row_tile
        for group, indices in enumerate(groups):
            ids = np.asarray(indices)[owners[indices] == noise_group]
            tiles = (tilt_tiles(dataset, ids, tile_size) if tomo else
                     [ids[start:start + tile_size] for start in range(0, len(ids), tile_size)])
            for tile in tiles:
                if len(tile):
                    support = [None] * len(tile) if candidates is None else candidates(tile)
                    yield group, int(noise_group), accumulate(mixture, tile, support, moments=moments)
        del mixture, streams


def _tile_poses(tile, grid):
    """The marginal-MAP class's conditional MAP pose of each tile particle on ``grid``: Euler degrees and the
    shift in pixels padded to three (membership pose memory)."""
    probabilities = np.asarray(tile.class_probabilities)
    rows, labels = np.arange(probabilities.shape[0]), probabilities.argmax(axis=1)
    eulers = np.asarray(grid.eulers, np.float32)[np.asarray(tile.best_rotation_idx)[rows, labels]]
    shifts = np.asarray(grid.translations, np.float32)[np.asarray(tile.best_translation_idx)[rows, labels]]
    padded = np.zeros((shifts.shape[0], 3), np.float32)
    padded[:, : shifts.shape[1]] = shifts
    return eulers, padded


def expectation_groups(dataset, state, config, groups, iteration, grid, *, diameter_ang=None, candidates=None,
                       child_grid=None):
    """Reduce each tile immediately: never retain a volume accumulator per tile.

    Each ``membership_tiles`` entry holds a tile's particle ids, class probabilities and MAP poses (Euler
    degrees, shifts in pixels) on the grid its poses index: ``child_grid`` for an oversampled update (its pass 2
    indexes the children), else ``grid``; an oversampled update without ``child_grid`` records no poses.
    """
    oversampled = oversampled_update(config, iteration)
    pose_source = (child_grid if oversampled else grid)
    results = [BatchStatistics([None] * config.n_classes, np.zeros(config.n_classes, np.float32),
                               np.zeros(len(grid.rotations), np.float32),
                               posterior=[PosteriorDiagnostics() for _ in range(config.n_classes)],
                               oversampling=[{"counts": [], "mass": [], "capped": []} for _ in range(config.n_classes)])
               for _ in groups]
    tomo = isinstance(dataset, TiltParticles)
    for group, noise_group, tile in mixture_tiles(
        dataset, state, config, groups, iteration, grid, moments=True, diameter_ang=diameter_ang,
        candidates=candidates,
    ):
        result = results[group]
        # Already computed on pass 1; retain only small host arrays, not GPU
        # statistics, pose grids or latents. Never run another inference pass.
        eulers, shifts = _tile_poses(tile, pose_source) if pose_source is not None else (None, None)
        result.membership_tiles.append((np.asarray(tile.original_image_ids).copy(),
                                        np.asarray(tile.class_probabilities).copy(), eulers, shifts))
        result.count += len(tile.original_image_ids)
        result.class_mass += tile.component_mass
        result.pmax_sum += float(np.sum(tile.max_posterior_per_image, dtype=np.float64))
        result.log_likelihood += tile.log_likelihood
        if result.residual_num is None:
            shape = ((dataset.n_noise_groups,) + tile.residual_num.shape) if tomo else tile.residual_num.shape
            result.residual_num = jnp.zeros(shape, jnp.float32)
            result.residual_den = jnp.zeros(shape, jnp.float32)
        if tomo:
            result.residual_num = result.residual_num.at[noise_group].add(tile.residual_num)
            result.residual_den = result.residual_den.at[noise_group].add(tile.residual_den)
        else:
            result.residual_num += tile.residual_num
            result.residual_den += tile.residual_den
        for k, component in enumerate(tile.statistics):
            result.posterior[k].add(component)
            if "significant_samples_per_image" in component.diagnostics:
                for key, name in (("counts", "significant_samples_per_image"), ("mass", "significant_mass_per_image"),
                                  ("capped", "significant_capped_per_image")):
                    result.oversampling[k][key].append(component.diagnostics[name])
            previous = result.components[k]
            result.components[k] = ComponentStatistics(
                component.lhs_tri if previous is None else previous.lhs_tri + component.lhs_tri,
                component.residual_gradient if previous is None else previous.residual_gradient + component.residual_gradient,
            )
            result.rotation_mass += component.diagnostics["rotation_mass"]
            result.offset_second += component.diagnostics["offset_second_sum_px2"]
    for result, ids in zip(results, groups, strict=True):
        if result.count != len(ids) or any(c is None for c in result.components):
            raise ValueError("Mixture E-step failed to cover a complete nonempty particle group")
        if not np.isclose(result.class_mass.sum(), len(ids), rtol=1e-5):
            raise ValueError("Mixture responsibilities must sum to the physical-particle count")
        actual_ids = np.concatenate([item[0] for item in result.membership_tiles])
        expected_ids = dataset.original_image_indices_from_local(ids)
        if (len(np.unique(actual_ids)) != len(ids)
                or not np.array_equal(np.sort(actual_ids), np.sort(expected_ids))):
            raise ValueError("Mixture memberships do not cover the selected physical particles exactly once")
    return results
