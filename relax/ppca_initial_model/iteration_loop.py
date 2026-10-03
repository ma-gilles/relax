"""Pose-free PPCA controller; algorithm sections 9–10 and plan package E.

Every update scores the full pose grid of its stage (no oversampling; section 14 of
``docs/math/vdam_ppca_algorithm.md`` records why) on the device-resident stream
(:mod:`relax.ppca_refinement.full_row_stream`), or, with ``stream_coarse_recompute``
off, on the host-mask dense engine that serves as its float32 reference (q <= 2).
"""

import dataclasses
import functools
import json
import logging
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from recovar.core import fourier_transform_utils as ftu
from recovar.ppca.pose_accumulators import AugmentedPPCAStats
from recovar.ppca.triangular import unpack_tri_to_full
from recovar.reconstruction.noise import make_radial_noise

from relax import sampling
from relax.ppca_initial_model import checkpoint
from relax.ppca_initial_model.initialization import bandlimit_and_mask, initialize, support_mask
from relax.ppca_initial_model.noise import update_noise
from relax.ppca_initial_model.sgd_update import metric_trace, momentum_step
from relax.ppca_initial_model.state import State
from relax.ppca_initial_model.tomo import TiltParticles, initialize_tilts, load_tilt_tile, tilt_tiles
from relax.ppca_initial_model.update import coupled_direction, empty_moments, metric_floor, stochastic_update
from relax.ppca_refinement.config import GeometryConfig, ScheduleConfig, ScoringConfig, SparsePass2Config
from relax.ppca_refinement.dense_dataset import (
    DensePPCAEmbeddings,
    accumulate_dense_ppca_statistics,
    compute_dense_ppca_embeddings,
)
from relax.ppca_refinement.full_row_stream import (
    TracePPCAStats,
    accumulate_full_row_tiles,
    full_row_tile_embeddings,
    plan_tile_images,
    prepare_full_row_stream,
    resolve_gemm_precision,
)
from relax.ppca_refinement.residual_statistics import full_float32

logger = logging.getLogger(__name__)

def _json(value):
    if isinstance(value, (np.ndarray, jnp.ndarray)):
        return np.asarray(value).tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


@functools.lru_cache(maxsize=4)
def _rotation_grid(hp):
    """The HEALPix order's float32 rotation matrices (read-only); HP4 takes about 0.3 s to build."""
    rotations = sampling.get_relion_hidden_rotation_grid(hp, matrices=True).astype(np.float32)
    rotations.flags.writeable = False
    return rotations


@functools.lru_cache(maxsize=4)
def _direction_ids(hp):
    """Index of each rotation's (rot, tilt) direction on the order-``hp`` grid (read-only; a sort of every rotation)."""
    eulers = sampling.get_relion_hidden_rotation_grid(hp, matrices=False)
    _, direction_ids = np.unique(eulers[:, :2], axis=0, return_inverse=True)
    direction_ids = direction_ids.reshape(-1)
    direction_ids.flags.writeable = False
    return direction_ids


def _merge_statistics(parts):
    trace_only = all(isinstance(s, TracePPCAStats) for s in parts)
    fields = dict(
        # The streamed engine produces no RHS volume (only lhs_tri and the residual gradient are read).
        rhs=None if any(s.rhs is None for s in parts) else sum(s.rhs for s in parts),
        lhs_tri=None if trace_only else sum(s.lhs_tri for s in parts),
        residual_gradient=sum(s.residual_gradient for s in parts),
        residual_num=sum(s.residual_num for s in parts),
        residual_den=sum(s.residual_den for s in parts),
        embeddings=jnp.concatenate([s.embeddings for s in parts]),
        original_image_ids=np.concatenate([s.original_image_ids for s in parts]),
        n_images=sum(s.n_images for s in parts),
        log_likelihood=sum(s.log_likelihood for s in parts),
        diagnostics={
            "offset_second_sum_px2": sum(s.diagnostics["offset_second_sum_px2"] for s in parts),
            **{
                key: sum(s.diagnostics[key] * s.n_images for s in parts) / sum(s.n_images for s in parts)
                for key in ("latent_covariance_trace_mean", "pose_entropy_mean", "pmax_mean")
            },
        },
    )
    if trace_only:
        return TracePPCAStats(metric_trace=sum(s.metric_trace for s in parts), **fields)
    return AugmentedPPCAStats(**fields)


def _curvature_trace(stats, p):
    """Momentum SGD's per-frequency metric trace summed over the halves."""
    if all(isinstance(s, TracePPCAStats) for s in stats):
        return sum(s.metric_trace for s in stats)
    return metric_trace(sum(s.lhs_tri for s in stats), p)


def _select_halves(rng, order, count, balanced):
    """Draw one fresh permutation, optionally stratified by stable pseudo-half ID."""
    shuffled = rng.permutation(order)
    if balanced:
        if count % 2:
            raise ValueError("Balanced stochastic batch requires an even effective image count")
        half_count = count // 2
        halves = [shuffled[shuffled % 2 == half][:half_count] for half in range(2)]
        if any(len(ids) != half_count for ids in halves):
            raise ValueError("Fixed stochastic batch exceeds an even/odd pseudo-half population")
        return np.concatenate(halves), halves
    selected = shuffled[:count]
    return selected, [selected[selected % 2 == half] for half in range(2)]


def _gemm_precision_used(config):
    """The stream GEMM precision of this run's updates; the host-mask engine multiplies in fp32."""
    if not config.stream_coarse_recompute:
        return "fp32"
    return resolve_gemm_precision(config.gemm_precision, jax.local_devices()[0])


def _streams_groups(config):
    return config.stream_coarse_recompute


def expectation(dataset, state, config, ids, iteration, *, embeddings_only=False):
    """Posterior statistics (or embeddings) of the images ``ids`` at ``iteration``."""
    if _streams_groups(config):
        return _expectation(dataset, state, config, [ids], iteration, embeddings_only=embeddings_only)[0]
    return _expectation(dataset, state, config, ids, iteration, embeddings_only=embeddings_only)


def expectation_groups(dataset, state, config, groups, iteration, *, embeddings_only=False):
    """:func:`expectation` of each id group (the pseudo-halves), one result per group.

    The streamed coarse-recompute path prepares the model once for all groups
    and pipelines their image tiles; every other path runs group by group.
    """
    if not _streams_groups(config):
        return [expectation(dataset, state, config, ids, iteration, embeddings_only=embeddings_only) for ids in groups]
    return _expectation(dataset, state, config, groups, iteration, embeddings_only=embeddings_only)


@full_float32
def _expectation(dataset, state, config, ids, iteration, *, embeddings_only=False):
    radius, hp = config.stage(iteration)
    # full-box default projector excludes unpaired Nyquist, as existing PPCA.
    radius = min(radius, dataset.grid_size // 2 - 1)
    geometry = GeometryConfig(current_size=2 * radius, q=config.q, volume_domain="fourier_half")
    schedule = ScheduleConfig(image_batch_size=config.image_batch_size, rotation_block_size=config.rotation_block_size)
    scoring = ScoringConfig(relion_texture_interp=False, full_real_observation=True)
    rotations = _rotation_grid(hp)
    canonical_eulers = sampling.get_relion_hidden_rotation_grid(hp, matrices=False)
    rotation_prior = (
        np.asarray(state.direction_prior, np.float32)
        if state.direction_order == hp and state.direction_prior is not None
        else np.full(len(rotations), 1 / len(rotations), np.float32)
    )
    rotation_log_prior = np.log(rotation_prior, where=rotation_prior > 0, out=np.full_like(rotation_prior, -np.inf))
    tilts = isinstance(dataset, TiltParticles)
    if tilts and not _streams_groups(config):
        raise ValueError("Subtomogram PPCA needs the streamed full-grid engine (stream_coarse_recompute)")
    translations = (
        # Subtomogram particles have one 3D shift each (section 16.1), on RELION's 3D grid in pixels.
        sampling.get_relion_translation_grid_3d(config.shift_range, config.shift_step)
        if tilts
        else sampling.get_relion_translation_grid(max_pixel=config.shift_range, pixel_offset=config.shift_step)
    ).astype(np.float32)
    prior = -jnp.sum(jnp.asarray(translations) ** 2, axis=-1) / (2 * state.offset_variance)
    prior = prior - jnp.log(jnp.sum(jnp.exp(prior)))
    # Subtomogram particles keep one noise spectrum per noise group (rows of ``state.noise``, section 16.6).
    noise_rows = np.asarray(state.noise) if tilts else np.asarray(state.noise)[None]
    nvs = [np.asarray(make_radial_noise(row, dataset.image_shape), np.float32) for row in noise_rows]
    nv = nvs[0]
    common = dict(noise_variance=nv, geometry=geometry, schedule=schedule, scoring=scoring, image_indices=ids)
    mu, W = state.theta[:, 0], state.theta[:, 1:]
    if config.stream_coarse_recompute:
        # One artificial coarse parent represents the full coarse pose grid.
        # It keeps the shared full-row mask Bx1x1 instead of BxRxT. Each noise group has its
        # own stream (its noise enters the operands); single particles have one group.
        streams = [
            prepare_full_row_stream(
                dataset,
                mu,
                W,
                noise_variance=group_nv,
                rotations=rotations,
                translations=translations,
                rotation_log_prior=rotation_log_prior,
                translation_log_prior=np.asarray(prior),
                rotation_parent=np.zeros(len(rotations), np.int32),
                translation_parent=np.zeros(len(translations), np.int32),
                n_coarse_rotations=1,
                n_coarse_translations=1,
                geometry=geometry,
                schedule=schedule,
                scoring=scoring,
                # Momentum SGD reads only the metric trace (sgd_update.momentum_step).
                metric_trace_only=config.optimizer == "momentum_sgd",
                gemm_precision=config.gemm_precision,
                tile_loader=load_tilt_tile if tilts else None,
            )
            for group_nv in nvs
        ]
        # Here ``ids`` is the list of id groups; each group is cut into image tiles, of one tilt
        # group and one noise group each for subtomogram particles. ``image_batch_size`` particles
        # per tile at most, fewer when a tile (with all its particles' tilts) would not fit the device.
        tile_size = plan_tile_images(
            streams[0],
            config.image_batch_size,
            n_frames=max(len(frames) for frames in dataset.group_frames) if tilts else 1,
        )
        tiles = [
            (group, int(dataset.particle_noise_group[tile[0]]) if tilts else 0, np.asarray(tile))
            for group, ids_group in enumerate(ids)
            for tile in (
                tilt_tiles(dataset, ids_group, tile_size)
                if tilts
                else [ids_group[begin : begin + tile_size] for begin in range(0, len(ids_group), tile_size)]
            )
        ]
        parts = [None] * len(tiles)
        for noise_group, stream in enumerate(streams):
            members = [k for k, (_, owner, _) in enumerate(tiles) if owner == noise_group]
            items = [(tiles[k][2], [None] * len(tiles[k][2])) for k in members]
            if embeddings_only:
                done = [full_row_tile_embeddings(stream, *item) for item in items]
            else:
                done = accumulate_full_row_tiles(stream, items) if items else []
            for k, part in zip(members, done):
                parts[k] = part
        noise_owner = [owner for _, owner, _ in tiles]
        tiles = [(group, tile_ids) for group, _, tile_ids in tiles]
        results = []
        for group in range(len(ids)):
            group_parts = [part for (owner, _), part in zip(tiles, parts) if owner == group]
            if embeddings_only:
                results.append(DensePPCAEmbeddings(
                    jnp.concatenate([part.embeddings for part in group_parts]),
                    np.concatenate([part.original_image_ids for part in group_parts]),
                    sum(part.n_images for part in group_parts),
                ))
                continue
            stats = _merge_statistics(group_parts)
            if tilts:
                # Noise sums per noise group, (G, S); a group without images in this id group adds zeros.
                owners = [owner for (half, _), owner in zip(tiles, noise_owner) if half == group]
                stats = dataclasses.replace(
                    stats,
                    **{
                        name: jnp.stack([
                            sum(
                                (getattr(part, name) for part, owner in zip(group_parts, owners) if owner == g),
                                jnp.zeros_like(getattr(group_parts[0], name)),
                            )
                            for g in range(len(streams))
                        ])
                        for name in ("residual_num", "residual_den")
                    },
                )
            stats.diagnostics.update(
                {
                    "rotation_mass": sum(np.asarray(part.diagnostics["rotation_mass"]) for part in group_parts),
                    "coarse_omitted_mass_bound": 0.0,
                    "canonical_euler_count": len(canonical_eulers),
                    "engine": "full_row_coarse_recompute",
                    "tile_images": tile_size,
                    "scored_image_rows": sum(part.diagnostics["scored_image_rows"] for part in group_parts),
                    "supported_image_rows": sum(part.diagnostics["supported_image_rows"] for part in group_parts),
                }
            )
            results.append(stats)
        return results
    function = compute_dense_ppca_embeddings if embeddings_only else accumulate_dense_ppca_statistics
    options = {} if embeddings_only else {"sparse_pass2": SparsePass2Config(enabled=False), "collect_residuals": True}
    stats = function(
        dataset,
        mu,
        W,
        rotations=rotations,
        translations=translations,
        translation_log_prior=np.asarray(prior),
        rotation_log_prior=rotation_log_prior,
        **options,
        **common,
    )
    if not embeddings_only:
        stats.diagnostics.update({"coarse_omitted_mass_bound": 0.0, "canonical_euler_count": len(canonical_eulers)})
    return stats


def run(dataset, config, output, identity, diameter_ang, *, resume=None, stop_after=None,
        stop_file=None, log_direction_prior=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    resumed_precision = None
    resumed_source = None
    if resume:
        state = checkpoint.load(resume, config, identity)
        saved_source = checkpoint.saved_source(resume)
        if saved_source != checkpoint.canonical(identity).get("source"):
            # Resuming under other code is allowed; the first update records where the state came from.
            resumed_source = saved_source
            logger.warning("resumed a checkpoint written by other source: %s", json.dumps(saved_source)[:200])
        saved = checkpoint.saved_gemm_precision(resume)
        if saved != config.gemm_precision:
            resumed_precision = saved
            logger.warning(
                "resumed %s checkpoint under %s (%s)", saved, config.gemm_precision, _gemm_precision_used(config)
            )
    else:
        initializer = initialize_tilts if isinstance(dataset, TiltParticles) else initialize
        theta, noise, info = initializer(dataset, seed=config.seed, diameter_ang=diameter_ang, q=config.q)
        rng = np.random.default_rng(config.seed + 1)
        order = rng.permutation(dataset.n_images)
        state = State(
            theta,
            empty_moments(theta) if config.optimizer == "vdam" else None,
            noise,
            0,
            order,
            rng.bit_generator.state,
            100.0 / dataset.voxel_size**2,
            0,
            info,
            sgd_momentum=jnp.zeros_like(theta) if config.optimizer == "momentum_sgd" else None,
        )
        checkpoint.save(output / "checkpoint_0000.npz", state, config, identity)
    if state.theta.ndim != 2 or state.theta.shape[1] != config.q + 1:
        raise ValueError("PPCA model rank differs from the configured q")
    if state.theta.dtype != jnp.complex64 or state.noise.dtype != jnp.float32:
        raise TypeError("InitialModel requires production float32 model and noise")
    if dataset.n_images < 4:
        raise ValueError("Both pseudo-halfsets need particles")
    mask = support_mask(dataset.grid_size, diameter_ang / dataset.voxel_size)
    shells = np.asarray(ftu.get_grid_of_radial_distances_real(dataset.volume_shape), np.int32).reshape(-1)
    sgd_radii = (
        np.asarray(ftu.get_grid_of_radial_distances_real(dataset.volume_shape, rounded=False)).reshape(-1)
        if config.optimizer == "momentum_sgd"
        else None
    )
    end = config.iterations if stop_after is None else min(config.iterations, stop_after)
    for iteration in range(state.iteration + 1, end + 1):
        started = time.monotonic()
        count, step, fudge = config.schedule(iteration, dataset.n_images)
        rng = np.random.default_rng()
        rng.bit_generator.state = state.rng_state
        # Persistent random order; each iteration takes a fresh uniformly selected batch.
        # Stable pseudo-half identity, independent of poses and labels.
        selected, halves = _select_halves(rng, state.order, count, config.balanced_stochastic_halves)
        if any(len(ids) == 0 for ids in halves):
            raise ValueError("Selected batch has an empty pseudo-halfset")
        stats = expectation_groups(dataset, state, config, halves, iteration)
        if config.optimizer == "momentum_sgd":
            radius = min(config.stage(iteration)[0], dataset.grid_size // 2 - 1)
            _, proposed_momentum, diagnostics = momentum_step(
                state.theta,
                state.sgd_momentum,
                sum(result.residual_gradient for result in stats),
                _curvature_trace(stats, config.q + 1),
                sgd_radii <= radius,
                learning_rate=config.sgd_learning_rate,
                floor=metric_floor(dataset.grid_size),
            )
            momentum = bandlimit_and_mask(proposed_momentum, dataset.volume_shape, radius, mask)
            theta = bandlimit_and_mask(state.theta + proposed_momentum, dataset.volume_shape, radius, mask)
            moments = None
            metric_info = []
            diagnostics["velocity_l2_after_support"] = float(jnp.linalg.norm(momentum))
        else:
            directions = []
            metric_info = []
            coverage = []
            for result in stats:
                direction, info = coupled_direction(
                    result.lhs_tri, result.residual_gradient, floor=metric_floor(dataset.grid_size)
                )
                directions.append(direction)
                metric_info.append(info)
                coverage.append(jnp.trace(unpack_tri_to_full(result.lhs_tri, config.q + 1), axis1=-2, axis2=-1) > 0)
            theta, moments, diagnostics = stochastic_update(
                state.theta,
                state.moments,
                jnp.stack(directions),
                jnp.stack(coverage),
                shells,
                step=step,
                fudge=fudge,
                image_size=dataset.grid_size,
            )
            radius = min(config.stage(iteration)[0], dataset.grid_size // 2 - 1)
            theta = bandlimit_and_mask(theta, dataset.volume_shape, radius, mask)
            momentum = None
        numerator = sum(s.residual_num for s in stats)
        denominator = sum(s.residual_den for s in stats)
        # Noise rows (one per noise group for subtomograms) padded to the statistics' shell count.
        current = np.asarray(state.noise)
        noise_shells = np.asarray(numerator).shape[-1]
        padding = [(0, 0)] * (current.ndim - 1) + [(0, max(0, noise_shells - current.shape[-1]))]
        previous = np.pad(current, padding, mode="edge")[..., :noise_shells]
        try:
            noise = update_noise(previous, numerator, denominator, full_data=count == dataset.n_images)
        except ValueError:
            checkpoint.save(output / f"failure_before_{iteration:04d}.npz", state, config, identity)
            np.savez(
                output / f"failure_noise_{iteration:04d}.npz",
                previous=previous,
                numerator=np.asarray(numerator),
                denominator=np.asarray(denominator),
                selected=selected,
            )
            raise
        # Mean squared offset per dimension: 2D in-plane shifts, 3D subtomogram shifts.
        dimensions = 3 if isinstance(dataset, TiltParticles) else 2
        offset = sum(s.diagnostics["offset_second_sum_px2"] for s in stats) / (dimensions * count)
        beta = 0 if count == dataset.n_images else 0.9
        offset_variance = max(2.0 / dataset.voxel_size**2, beta * state.offset_variance + (1 - beta) * offset)
        if (
            theta.dtype != jnp.complex64
            or (moments is not None and (moments.first.dtype != jnp.complex64 or moments.second.dtype != jnp.float32))
            or (momentum is not None and momentum.dtype != jnp.complex64)
        ):
            raise TypeError("Non-float32 production model/moments")
        if not np.all(np.isfinite(np.asarray(theta))):
            raise ValueError("Nonfinite PPCA model")
        hp = config.stage(iteration)[1]
        direction_ids = _direction_ids(hp)
        masses = sum(np.asarray(result.diagnostics["rotation_mass"]) for result in stats)
        direction_mass = np.bincount(direction_ids, weights=masses)
        multiplicity = np.bincount(direction_ids)
        estimated = (direction_mass[direction_ids] / multiplicity[direction_ids] / count).astype(np.float32)
        old_prior = (
            state.direction_prior
            if state.direction_order == hp and state.direction_prior is not None
            else np.full(len(direction_ids), 1 / len(direction_ids), np.float32)
        )
        direction_prior = (beta * old_prior + (1 - beta) * estimated).astype(np.float32)
        state = State(
            theta,
            moments,
            noise,
            iteration,
            state.order,
            rng.bit_generator.state,
            offset_variance,
            radius,
            state.initialization,
            direction_prior,
            hp,
            momentum,
        )
        diagnostics.update(
            {
                "iteration": iteration,
                "particle_ids": [s.original_image_ids for s in stats],
                "half_counts": [s.n_images for s in stats],
                "radius": radius,
                "healpix_order": config.stage(iteration)[1],
                "gemm_precision": _gemm_precision_used(config),
                **({"resumed_from_gemm_precision": resumed_precision} if resumed_precision else {}),
                **({"resumed_from_source": resumed_source} if resumed_source is not None else {}),
                "step": config.sgd_learning_rate if config.optimizer == "momentum_sgd" else step,
                "fudge": fudge,
                "noise": noise,
                "metric": metric_info,
                "loading_singular_values": np.linalg.svd(np.asarray(theta[:, 1:]), compute_uv=False),
                "mean_power": float(np.sum(np.abs(np.asarray(theta[:, 0])) ** 2)),
                "loading_power": float(np.sum(np.abs(np.asarray(theta[:, 1:])) ** 2)),
                "posterior": [
                    {
                        key: s.diagnostics.get(key)
                        for key in (
                            "latent_covariance_trace_mean",
                            "pose_entropy_mean",
                            "pmax_mean",
                            "coarse_omitted_mass_bound",
                            "tile_images",
                        )
                    }
                    for s in stats
                ],
                "log_likelihood": sum(s.log_likelihood for s in stats),
                "likelihood_scope": "minibatch/support/noise dependent; omitted observation constants",
                "direction_prior": direction_prior,
                "offset_variance_px2": offset_variance,
                "dtype": str(theta.dtype),
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        if config.optimizer == "momentum_sgd":
            diagnostics.update(
                optimizer="momentum_sgd",
                sgd_learning_rate=config.sgd_learning_rate,
                vdam_scheduled_step_unused=step,
                vdam_scheduled_fudge_unused=fudge,
            )
        if not log_direction_prior:
            # The full angular prior is checkpointed in State. Large stochastic
            # sweeps can omit its per-update JSON duplicate without changing it.
            diagnostics.pop("direction_prior")
        with open(output / "iterations.jsonl", "a") as stream:
            stream.write(json.dumps(diagnostics, default=_json) + "\n")
        if iteration % config.checkpoint_interval == 0 or iteration == end:
            checkpoint.save(output / f"checkpoint_{iteration:04d}.npz", state, config, identity)
        if stop_file is not None and Path(stop_file).exists():
            break
    if state.iteration == config.iterations and not config.skip_final_embeddings:
        ids = np.arange(dataset.n_images)
        final = expectation(dataset, state, config, ids, config.iterations, embeddings_only=True)
        if not np.array_equal(np.sort(final.original_image_ids), ids):
            raise ValueError("Final embedding does not cover every particle exactly once")
        np.savez(output / "embeddings.npz", particle_ids=final.original_image_ids, z=np.asarray(final.embeddings))
    return state
