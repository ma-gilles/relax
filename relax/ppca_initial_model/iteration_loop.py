"""Pose-free PPCA controller; algorithm sections 9–10 and plan package E.

Global coarse PPCA scores are recomputed every iteration. Fine candidates use
the maintained LocalHypothesisLayout, without any extra fine pruning. Per-image
rows run through the shared host-mask dense statistics path; streamed full rows
run through the device-resident full-row engine
(:mod:`relax.ppca_refinement.full_row_stream`), recorded as ``fine_engine``.
"""

import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from recovar.core import fourier_transform_utils as ftu
from recovar.ppca.pose_accumulators import AugmentedPPCAStats
from recovar.ppca.triangular import unpack_tri_to_full
from recovar.reconstruction.noise import make_radial_noise

from relax import sampling
from relax.local.local_layout import build_pass2_hypothesis_layout
from relax.ppca_initial_model import checkpoint
from relax.ppca_initial_model.initialization import bandlimit_and_mask, initialize, support_mask
from relax.ppca_initial_model.noise import update_noise
from relax.ppca_initial_model.state import State
from relax.ppca_initial_model.update import coupled_direction, empty_moments, metric_floor, stochastic_update
from relax.ppca_refinement.config import GeometryConfig, ScheduleConfig, ScoringConfig, SparsePass2Config
from relax.ppca_refinement.dense_dataset import (
    accumulate_dense_ppca_statistics,
    compute_dense_ppca_adaptive_significance,
    compute_dense_ppca_embeddings,
)
from relax.ppca_refinement.full_row_stream import (
    FULL_ROW_ENGINE,
    accumulate_full_row_tile,
    full_row_tile_embeddings,
    prepare_full_row_stream,
)
from relax.ppca_refinement.residual_statistics import full_float32


def _json(value):
    if isinstance(value, (np.ndarray, jnp.ndarray)):
        return np.asarray(value).tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _merge_statistics(parts):
    return AugmentedPPCAStats(
        rhs=sum(s.rhs for s in parts),
        lhs_tri=sum(s.lhs_tri for s in parts),
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


@full_float32
def expectation(dataset, state, config, ids, iteration, *, embeddings_only=False):
    radius, hp = config.stage(iteration)
    # full-box default projector excludes unpaired Nyquist, as existing PPCA.
    radius = min(radius, dataset.grid_size // 2 - 1)
    geometry = GeometryConfig(current_size=2 * radius, q=config.q, volume_domain="fourier_half")
    schedule = ScheduleConfig(image_batch_size=config.image_batch_size, rotation_block_size=config.rotation_block_size)
    scoring = ScoringConfig(relion_texture_interp=False, full_real_observation=True)
    rotations = sampling.get_relion_hidden_rotation_grid(hp, matrices=True).astype(np.float32)
    canonical_eulers = sampling.get_relion_hidden_rotation_grid(hp, matrices=False)
    rotation_prior = (
        np.asarray(state.direction_prior, np.float32)
        if state.direction_order == hp and state.direction_prior is not None
        else np.full(len(rotations), 1 / len(rotations), np.float32)
    )
    rotation_log_prior = np.log(rotation_prior, where=rotation_prior > 0, out=np.full_like(rotation_prior, -np.inf))
    translations = sampling.get_relion_translation_grid(
        max_pixel=config.shift_range, pixel_offset=config.shift_step
    ).astype(np.float32)
    prior = -jnp.sum(jnp.asarray(translations) ** 2, axis=-1) / (2 * state.offset_variance)
    prior = prior - jnp.log(jnp.sum(jnp.exp(prior)))
    nv = np.asarray(make_radial_noise(np.asarray(state.noise), dataset.image_shape), np.float32)
    common = dict(noise_variance=nv, geometry=geometry, schedule=schedule, scoring=scoring, image_indices=ids)
    mu, W = state.theta[:, 0], state.theta[:, 1:]
    if config.oversampling == 0:
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
    coarse = compute_dense_ppca_adaptive_significance(
        dataset,
        mu,
        W,
        rotations=rotations,
        translations=translations,
        translation_log_prior=np.asarray(prior),
        rotation_log_prior=rotation_log_prior,
        adaptive_fraction=config.target_mass,
        max_significants=-1,
        **common,
    )
    children_per_parent = 8 ** config.oversampling
    full_rotation_count = len(rotations) * children_per_parent
    stream_full = bool(config.stream_full_fine_rows)
    if stream_full:
        # This opt-in route is exact only when every image retained every
        # coarse orientation. The per-image translation support can still vary.
        for image, significant in enumerate(coarse.significant_sample_indices):
            if significant is not None and np.unique(np.asarray(significant) // len(translations)).size != len(rotations):
                raise ValueError(f"Image {image} lacks a full fine rotation row; streaming full rows cannot change support")
    layout = build_pass2_hypothesis_layout(
        [None] if stream_full else coarse.significant_sample_indices,
        len(rotations),
        len(translations),
        hp,
        translations,
        oversampling_order=config.oversampling,
        translation_step=config.shift_step,
        rotation_log_prior=rotation_log_prior,
        rotation_index_order="relion",
        allow_empty=False,
    )
    if stream_full:
        fine_grid, fine_translation_parent = sampling.get_oversampled_translation_grid(
            translations, config.shift_step, oversampling_order=config.oversampling,
        )
        if not np.array_equal(np.asarray(fine_grid, np.float32), layout.translation_grid):
            raise RuntimeError("Streamed fine translation grid differs from the exact local layout")
        if not np.array_equal(
            layout.rotation_posterior_ids_flat, np.repeat(np.arange(len(rotations)), children_per_parent)
        ):
            raise RuntimeError("Streamed fine rotation rows must keep contiguous children per coarse parent")
    stats = None
    coarse_mass = np.zeros(len(rotations), np.float32)
    fine_prior = -np.sum(layout.translation_grid**2, axis=-1) / (2 * state.offset_variance)
    fine_prior = fine_prior - np.log(np.sum(np.exp(fine_prior)))
    if stream_full:
        # One upload of the model, fine grids and priors; each tile then
        # expands its coarse support to fine poses on the device.
        stream = prepare_full_row_stream(
            dataset,
            mu,
            W,
            noise_variance=nv,
            rotations=layout.rotations_flat,
            translations=layout.translation_grid,
            rotation_log_prior=layout.rotation_log_priors_flat,
            translation_log_prior=fine_prior.astype(np.float32),
            rotation_parent=layout.rotation_posterior_ids_flat,
            translation_parent=fine_translation_parent,
            n_coarse_rotations=len(rotations),
            n_coarse_translations=len(translations),
            geometry=geometry,
            schedule=schedule,
            scoring=scoring,
        )
    tile_limit = min(config.fine_image_tile_size, config.image_batch_size)
    row = 0
    while row < len(ids):
        if stream_full:
            begin, end = 0, full_rotation_count
            tile_end = min(row + tile_limit, len(ids))
        else:
            begin = int(layout.rotation_offsets[row])
            end = begin + int(layout.rotation_counts[row])
            tile_end = row + 1
        if not stream_full and tile_limit > 1 and end - begin == full_rotation_count:
            # The dense engine shares projections within an image batch. Group
            # only identical full-grid rotation rows; the per-image translation
            # masks still exclude every coarse-rejected pose exactly.
            while tile_end < min(row + tile_limit, len(ids)):
                other_begin = int(layout.rotation_offsets[tile_end])
                other_end = other_begin + int(layout.rotation_counts[tile_end])
                if other_end - other_begin != full_rotation_count or not (
                    np.array_equal(layout.rotation_ids_flat[begin:end], layout.rotation_ids_flat[other_begin:other_end])
                    and np.array_equal(layout.rotations_flat[begin:end], layout.rotations_flat[other_begin:other_end])
                    and np.array_equal(layout.rotation_log_priors_flat[begin:end], layout.rotation_log_priors_flat[other_begin:other_end])
                    and np.array_equal(layout.rotation_posterior_ids_flat[begin:end], layout.rotation_posterior_ids_flat[other_begin:other_end])
                ):
                    break
                tile_end += 1
        if stream_full:
            tile_ids = np.asarray(ids[row:tile_end])
            significant = coarse.significant_sample_indices[row:tile_end]
            # A multi-image tile factors the latent Gram once per image/rotation.
            part = (
                full_row_tile_embeddings(stream, tile_ids, significant)
                if embeddings_only
                else accumulate_full_row_tile(stream, tile_ids, significant, factor_once=tile_end - row > 1)
            )
        else:
            if tile_end == row + 1:
                mask = layout.sample_mask_rows(begin, end)
            else:
                mask = np.stack([
                    layout.sample_mask_rows(int(layout.rotation_offsets[index]), int(layout.rotation_offsets[index + 1]))
                    for index in range(row, tile_end)
                ])
            function = compute_dense_ppca_embeddings if embeddings_only else accumulate_dense_ppca_statistics
            options = {} if embeddings_only else {"sparse_pass2": SparsePass2Config(enabled=False), "collect_residuals": True}
            if not embeddings_only and tile_end - row > 1:
                # Score the real multi-image tile with one latent factorization
                # per image/rotation and aggregate its posterior before adjoint.
                options["factor_once_score"] = True
            part = function(
                dataset,
                mu,
                W,
                rotations=layout.rotations_flat[begin:end],
                translations=layout.translation_grid,
                rotation_translation_mask=mask,
                rotation_log_prior=layout.rotation_log_priors_flat[begin:end],
                translation_log_prior=fine_prior.astype(np.float32),
                noise_variance=nv,
                geometry=geometry,
                schedule=schedule,
                scoring=scoring,
                image_indices=np.asarray(ids[row:tile_end]),
                **options,
            )
        if embeddings_only:
            stats = part if stats is None else type(part)(
                jnp.concatenate([stats.embeddings, part.embeddings]),
                np.concatenate([stats.original_image_ids, part.original_image_ids]),
                stats.n_images + part.n_images,
            )
        else:
            np.add.at(coarse_mass, layout.rotation_posterior_ids_flat[begin:end], part.diagnostics["rotation_mass"])
            stats = part if stats is None else _merge_statistics([stats, part])
        row = tile_end
    if embeddings_only:
        return stats
    if config.fine_image_tile_size > 1 and len(ids) > 1:
        # The reference path merges per-particle results, retaining only these
        # four aggregate diagnostics. A single tiled call must expose the same
        # controller-facing result rather than its dense-engine internals.
        summary_keys = ("offset_second_sum_px2", "latent_covariance_trace_mean", "pose_entropy_mean", "pmax_mean")
        summary = {key: stats.diagnostics[key] for key in summary_keys}
        stats.diagnostics.clear()
        stats.diagnostics.update(summary)
    stats.diagnostics.update(
        {
            "coarse_omitted_mass_bound": 1 - config.target_mass,
            "fine_pruning": False,
            "rotation_mass": coarse_mass,
            "fine_rotation_count": full_rotation_count * len(ids) if stream_full else layout.total_local_rotations,
            "fine_engine": FULL_ROW_ENGINE if stream_full else "dense_host_mask",
            "canonical_euler_count": len(canonical_eulers),
        }
    )
    return stats


def run(dataset, config, output, identity, diameter_ang, *, resume=None, stop_after=None, stop_file=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if resume:
        state = checkpoint.load(resume, config, identity)
    else:
        theta, noise, info = initialize(dataset, seed=config.seed, diameter_ang=diameter_ang)
        rng = np.random.default_rng(config.seed + 1)
        order = rng.permutation(dataset.n_images)
        state = State(
            theta,
            empty_moments(theta),
            noise,
            0,
            order,
            rng.bit_generator.state,
            100.0 / dataset.voxel_size**2,
            0,
            info,
        )
        checkpoint.save(output / "checkpoint_0000.npz", state, config, identity)
    if dataset.n_images < 4:
        raise ValueError("Both pseudo-halfsets need particles")
    mask = support_mask(dataset.grid_size, diameter_ang / dataset.voxel_size)
    shells = np.asarray(ftu.get_grid_of_radial_distances_real(dataset.volume_shape), np.int32).reshape(-1)
    end = config.iterations if stop_after is None else min(config.iterations, stop_after)
    for iteration in range(state.iteration + 1, end + 1):
        started = time.monotonic()
        count, step, fudge = config.schedule(iteration, dataset.n_images)
        rng = np.random.default_rng()
        rng.bit_generator.state = state.rng_state
        # Persistent random order; each iteration takes a fresh uniformly selected batch.
        selected = rng.permutation(state.order)[:count]
        # Stable pseudo-half identity, independent of poses and labels.
        halves = [selected[selected % 2 == half] for half in range(2)]
        if any(len(ids) == 0 for ids in halves):
            raise ValueError("Selected batch has an empty pseudo-halfset")
        stats = [expectation(dataset, state, config, ids, iteration) for ids in halves]
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
        numerator = sum(s.residual_num for s in stats)
        denominator = sum(s.residual_den for s in stats)
        previous = np.pad(np.asarray(state.noise), (0, max(0, len(numerator) - len(state.noise))), mode="edge")[
            : len(numerator)
        ]
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
        offset = sum(s.diagnostics["offset_second_sum_px2"] for s in stats) / (2 * count)
        beta = 0 if count == dataset.n_images else 0.9
        offset_variance = max(2.0 / dataset.voxel_size**2, beta * state.offset_variance + (1 - beta) * offset)
        if theta.dtype != jnp.complex64 or moments.first.dtype != jnp.complex64 or moments.second.dtype != jnp.float32:
            raise TypeError("Non-float32 production model/moments")
        if not np.all(np.isfinite(np.asarray(theta))):
            raise ValueError("Nonfinite PPCA model")
        hp = config.stage(iteration)[1]
        eulers = sampling.get_relion_hidden_rotation_grid(hp, matrices=False)
        _, direction_ids = np.unique(eulers[:, :2], axis=0, return_inverse=True)
        masses = sum(np.asarray(result.diagnostics["rotation_mass"]) for result in stats)
        direction_mass = np.bincount(direction_ids, weights=masses)
        multiplicity = np.bincount(direction_ids)
        estimated = (direction_mass[direction_ids] / multiplicity[direction_ids] / count).astype(np.float32)
        old_prior = (
            state.direction_prior
            if state.direction_order == hp and state.direction_prior is not None
            else np.full(len(eulers), 1 / len(eulers), np.float32)
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
        )
        diagnostics.update(
            {
                "iteration": iteration,
                "particle_ids": [s.original_image_ids for s in stats],
                "half_counts": [s.n_images for s in stats],
                "radius": radius,
                "healpix_order": config.stage(iteration)[1],
                "step": step,
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
                            "fine_rotation_count",
                            "fine_pruning",
                            "fine_engine",
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
        with open(output / "iterations.jsonl", "a") as stream:
            stream.write(json.dumps(diagnostics, default=_json) + "\n")
        if iteration % config.checkpoint_interval == 0 or iteration == end:
            checkpoint.save(output / f"checkpoint_{iteration:04d}.npz", state, config, identity)
        if stop_file is not None and Path(stop_file).exists():
            break
    if state.iteration == config.iterations:
        ids = np.arange(dataset.n_images)
        final = expectation(dataset, state, config, ids, config.iterations, embeddings_only=True)
        if not np.array_equal(np.sort(final.original_image_ids), ids):
            raise ValueError("Final embedding does not cover every particle exactly once")
        np.savez(output / "embeddings.npz", particle_ids=final.original_image_ids, z=np.asarray(final.embeddings))
    return state
