"""Joint mixture-PPCA InitialModel, with VDAM updates and shared observation noise.

Pseudo-halves are optimizer histories, not independent gold-standard maps.
"""

import dataclasses
import json
import logging
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from recovar.core import fourier_transform_utils as ftu
from recovar.ppca.triangular import unpack_tri_to_full

from relax.ppca_initial_class3d import checkpoint
from relax.ppca_initial_class3d.auto_sampling import (
    STALLED_UPDATES,
    SamplingState,
    expected_errors,
    resolution_shell,
    track_resolution,
    update_due,
    update_sampling,
)
from relax.ppca_initial_class3d.diagnostics import MetricFailure, model_summary
from relax.ppca_initial_class3d.expectation import expectation_groups, oversampled_update, pose_grid
from relax.ppca_initial_class3d.local_search import LocalGrid, relion_local_sigma_deg, tile_candidates
from relax.ppca_initial_class3d.membership import Membership
from relax.ppca_initial_class3d.outputs import export_assignments, export_models
from relax.ppca_initial_class3d.state import initialize_state, validate_state
from relax.ppca_initial_model.checkpoint import saved_source
from relax.ppca_initial_model.initialization import bandlimit_and_mask, support_mask
from relax.ppca_initial_model.iteration_loop import _select_halves as select_halves
from relax.ppca_initial_model.noise import update_noise
from relax.ppca_initial_model.tomo import TiltParticles
from relax.ppca_initial_model.update import coupled_direction, metric_floor, stochastic_update

logger = logging.getLogger(__name__)


def update_class_prior(previous, counts, *, pseudocount, full_data):
    """Dirichlet-smoothed physical-particle occupancy; same EMA timing as noise."""
    counts = np.asarray(counts, np.float32)
    if np.any(counts < 0) or not np.all(np.isfinite(counts)) or counts.sum() <= 0:
        raise ValueError("Invalid mixture class occupancy")
    estimated = (counts + np.float32(pseudocount)) / (counts.sum() + np.float32(len(counts) * pseudocount))
    beta = np.float32(0 if full_data else 0.9)
    prior = beta * np.asarray(previous, np.float32) + (1 - beta) * estimated
    return (prior / prior.sum()).astype(np.float32)


def select_tomogram_batch(rng, dataset, count):
    """Draw ``count`` particles as whole tilt groups (tomograms) in a fresh random order, the last group cut at
    random, so the stream's tiles hold a tomogram's particles; pseudo-halves by particle parity, as
    :func:`relax.ppca_initial_model.iteration_loop._select_halves`."""
    groups = np.asarray(dataset.particle_group)
    chosen, total = [], 0
    for group in rng.permutation(len(dataset.group_frames)):
        members = rng.permutation(np.flatnonzero(groups == group))
        chosen.append(members)
        total += members.size
        if total >= count:
            break
    selected = np.concatenate(chosen)[:count]
    return selected, [selected[selected % 2 == half] for half in range(2)]


def local_search_policy(config, state, grid):
    """Whether an update at ``grid`` searches locally around the stored poses, and with which width (degrees)
    and shift range (pixels, None: the whole grid): ``local_search_start`` or the auto-sampling state."""
    sampling = state.sampling
    local = ((config.local_search_start is not None and state.iteration + 1 >= config.local_search_start)
             or (sampling is not None and sampling.local_searches))
    if not local:
        return False, None, None
    if config.local_search_sigma_deg is not None:
        sigma = float(config.local_search_sigma_deg)
    elif sampling is not None and sampling.sigma_deg > 0:
        sigma = float(sampling.sigma_deg)
    else:
        sigma = relion_local_sigma_deg(grid.order)
    return True, sigma, None if sampling is None else float(sampling.offset_range_px)


def candidate_source(dataset, membership, local_grid, *, sigma_deg, offset_range_px, counts):
    """The ``candidates`` callable of :func:`expectation_groups`; ``counts`` collects the rows per particle."""

    def candidates(tile):
        rows = tile_candidates(local_grid, membership, dataset.original_image_indices_from_local(np.asarray(tile)),
                               sigma_deg=sigma_deg, offset_range_px=offset_range_px)
        counts.extend(-1 if r is None else int(r.size) for r in rows)
        return rows

    return candidates


def _oversampling_record(stats):
    """The batch's significant samples per class: median count per particle, the share of particles the cap
    stopped short of the adaptive fraction, and their mean held mass."""
    record = {}
    for k in range(len(stats[0].posterior)):
        counts = np.concatenate([np.asarray(c) for s in stats for c in s.oversampling[k]["counts"]])
        capped = np.concatenate([np.asarray(c) for s in stats for c in s.oversampling[k]["capped"]])
        mass = np.concatenate([np.asarray(c) for s in stats for c in s.oversampling[k]["mass"]])
        record[f"class{k}"] = {
            "samples_median": float(np.median(counts)) if counts.size else 0.0,
            "capped_fraction": float(capped.mean()) if capped.size else 0.0,
            "capped_mass_mean": float(mass[capped].mean()) if capped.any() else None,
        }
    return record


def _json(value):
    if isinstance(value, (np.ndarray, jnp.ndarray)):
        return np.asarray(value).tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def update_models(state, stats, config, dataset, *, radius, support, shells, mask, step, fudge):
    """Use responsibility-weighted normal equations without scaling twice by occupancy."""
    models, histories, metrics, diagnostics = [], [], [], []
    for k, model in enumerate(state.theta):
        directions, coverage, class_metrics = [], [], []
        for half_index, half in enumerate(stats):
            component = half.components[k]
            try:
                direction, info = coupled_direction(component.lhs_tri, component.residual_gradient, support,
                                                    floor=metric_floor(dataset.grid_size))
            except ValueError as error:
                raise MetricFailure(k, half_index, component, str(error)) from error
            directions.append(direction)
            covered = jnp.trace(unpack_tri_to_full(component.lhs_tri, config.q + 1), axis1=-2, axis2=-1) > 0
            coverage.append(covered & jnp.asarray(support))
            class_metrics.append(info)
        updated, moments, update_info = stochastic_update(
            model, state.moments[k], jnp.stack(directions), jnp.stack(coverage), shells,
            step=step, fudge=fudge, image_size=dataset.grid_size,
        )
        models.append(bandlimit_and_mask(updated, dataset.volume_shape, radius, mask))
        histories.append(moments)
        metrics.append(class_metrics)
        diagnostics.append({"class_index": k, **update_info, **model_summary(models[-1])})
    return jnp.stack(models), tuple(histories), metrics, diagnostics


def run(dataset, config, output, identity, diameter_ang, *, resume=None, stop_after=None, stop_file=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if dataset.n_images < 4:
        raise ValueError("Mixture InitialModel needs at least four physical particles for pseudo-halves")
    if not np.isfinite(diameter_ang) or diameter_ang <= 0:
        raise ValueError("Particle diameter must be finite and positive")
    if resume:
        state = checkpoint.load(resume, config, identity)
        if saved_source(resume) != identity.get("source"):
            logger.warning("Resuming mixture checkpoint under different source; prior source: %s", saved_source(resume))
    else:
        if list(output.glob("checkpoint_*.npz")):
            raise ValueError("Output has checkpoints; use --resume or a new directory")
        state = initialize_state(dataset, config, diameter_ang)
        state.membership = Membership.empty(
            dataset.original_image_indices_from_local(np.arange(dataset.n_images)), config.n_classes
        )
        checkpoint.save(output / "checkpoint_0000.npz", state, config, identity)
    validate_state(state, config)
    frequency_count = int(np.prod(ftu.volume_shape_to_half_volume_shape(dataset.volume_shape)))
    if state.theta.shape[1] != frequency_count:
        raise ValueError("Mixture checkpoint volume geometry differs from the dataset")
    if isinstance(dataset, TiltParticles):
        if state.noise.ndim != 2 or state.noise.shape[0] != dataset.n_noise_groups:
            raise ValueError("Mixture checkpoint tomography noise groups differ from the dataset")
    elif state.noise.ndim != 1:
        raise ValueError("Single-particle mixture expects one radial noise spectrum")
    if not np.array_equal(np.sort(state.order), np.arange(dataset.n_images)):
        raise ValueError("Checkpoint particle ordering does not match the dataset")
    original_ids = np.sort(dataset.original_image_indices_from_local(np.arange(dataset.n_images)))
    if state.membership is None:
        logger.warning("Checkpoint has no membership history; earlier labels remain unknown until re-evaluated")
        state.membership = Membership.empty(original_ids, config.n_classes)
    elif not np.array_equal(state.membership.particle_ids, original_ids):
        raise ValueError("Checkpoint membership particle IDs differ from the dataset")
    if config.auto_sampling and state.sampling is None:
        state.sampling = SamplingState.initial(config)
    mask = support_mask(dataset.grid_size, diameter_ang / dataset.voxel_size)
    shells = np.asarray(ftu.get_grid_of_radial_distances_real(dataset.volume_shape), np.int32).reshape(-1)
    radii = np.asarray(ftu.get_grid_of_radial_distances_real(dataset.volume_shape, rounded=False)).reshape(-1)
    end = config.iterations if stop_after is None else min(config.iterations, stop_after)
    grid, child_grid, local_grid, grid_key = None, None, None, None
    for iteration in range(state.iteration + 1, end + 1):
        started = time.monotonic()
        count, scheduled_step, fudge = config.schedule(iteration, dataset.n_images)
        step_factor = config.step_factor(iteration, dataset.n_images)
        step = scheduled_step * step_factor
        rng = np.random.default_rng()
        rng.bit_generator.state = state.rng_state
        if config.tomogram_batches and isinstance(dataset, TiltParticles):
            selected, halves = select_tomogram_batch(rng, dataset, count)
        else:
            selected, halves = select_halves(rng, state.order, count, config.balanced_stochastic_halves)
        if any(len(ids) == 0 for ids in halves):
            raise ValueError("Selected mixture minibatch has an empty pseudo-halfset")
        radius = min(config.stage(iteration)[0], dataset.grid_size // 2 - 1)
        key = (config.stage(iteration)[1], None) if state.sampling is None else state.sampling.grid_key()
        if key != grid_key:
            grid = pose_grid(dataset, config, iteration, sampling_state=state.sampling)
            child_grid = (pose_grid(dataset, config, iteration, children=True, sampling_state=state.sampling)
                          if config.oversampling else None)
            local_grid, grid_key = LocalGrid.of(grid), key
        hp = grid.order
        if state.direction_order == hp and state.direction_prior is not None:
            if len(state.direction_prior) != len(grid.rotations):
                raise ValueError("Mixture checkpoint angular prior does not match its pose grid")
        local, sigma_deg, local_range = local_search_policy(config, state, grid)
        candidate_counts = []
        candidates = (candidate_source(dataset, state.membership, local_grid, sigma_deg=sigma_deg,
                                       offset_range_px=local_range, counts=candidate_counts) if local else None)
        stats = expectation_groups(dataset, state, config, halves, iteration, grid, diameter_ang=diameter_ang,
                                   candidates=candidates,
                                   child_grid=child_grid if oversampled_update(config, iteration) else None)
        numerator = sum(s.residual_num for s in stats)
        denominator = sum(s.residual_den for s in stats)
        try:
            theta, moments, metrics, class_diagnostics = update_models(
                state, stats, config, dataset, radius=radius, support=radii <= radius, shells=shells,
                mask=mask, step=step, fudge=fudge,
            )
        except MetricFailure as error:
            checkpoint.save(output / f"failure_before_{iteration:04d}.npz", state, config, identity)
            np.savez(output / f"failure_metric_{iteration:04d}_class{error.class_index:03d}_half{error.half_index}.npz",
                     lhs_tri=np.asarray(error.component.lhs_tri),
                     residual_gradient=np.asarray(error.component.residual_gradient), selected=selected,
                     half_particle_ids=halves[error.half_index], support=radii <= radius)
            raise
        try:
            previous = np.asarray(state.noise)
            noise_shells = numerator.shape[-1]
            padding = [(0, 0)] * (previous.ndim - 1) + [(0, max(0, noise_shells - previous.shape[-1]))]
            previous = np.pad(previous, padding, mode="edge")[..., :noise_shells]
            noise = update_noise(previous, numerator, denominator, full_data=count == dataset.n_images)
        except ValueError:
            checkpoint.save(output / f"failure_before_{iteration:04d}.npz", state, config, identity)
            np.savez(output / f"failure_noise_{iteration:04d}.npz", numerator=np.asarray(numerator),
                     denominator=np.asarray(denominator), selected=selected)
            raise
        class_mass = sum(s.class_mass for s in stats)
        class_prior = update_class_prior(state.class_prior, class_mass, pseudocount=config.class_pseudocount,
                                         full_data=count == dataset.n_images)
        beta = 0 if count == dataset.n_images else 0.9
        dimensions = 3 if isinstance(dataset, TiltParticles) else 2
        offset = sum(s.offset_second for s in stats) / (dimensions * count)
        offset_variance = max(2.0 / dataset.voxel_size**2, beta * state.offset_variance + (1 - beta) * offset)
        mass = sum(s.rotation_mass for s in stats)
        directions = grid.direction_ids
        direction_mass = np.bincount(directions, weights=mass)
        multiplicity = np.bincount(directions)
        estimate = (direction_mass[directions] / multiplicity[directions] / count).astype(np.float32)
        old = (state.direction_prior if state.direction_order == hp and state.direction_prior is not None
               else np.full(len(directions), 1 / len(directions), np.float32))
        direction_prior = (beta * old + (1 - beta) * estimate).astype(np.float32)
        state = dataclasses.replace(
            state, theta=theta, moments=moments, noise=noise, class_prior=class_prior, iteration=iteration,
            rng_state=rng.bit_generator.state, offset_variance=offset_variance, radius=radius,
            direction_prior=direction_prior, direction_order=hp,
        )
        offset_change_sq, offset_count = 0.0, 0
        for half in stats:
            for ids, probabilities, eulers, shifts in half.membership_tiles:
                change, known = state.membership.update(ids, probabilities, iteration=iteration,
                                                        model_iteration=iteration - 1, eulers_deg=eulers,
                                                        translations_px=shifts)
                offset_change_sq, offset_count = offset_change_sq + change, offset_count + known
        sampling_update = None
        if state.sampling is not None:
            # RELION's running changes of the optimal offsets, resolution tracking and, every accuracy_interval
            # updates, its sampling decision (auto_sampling.py).
            sampling = track_resolution(state.sampling, resolution_shell(class_diagnostics, radius=radius))
            if offset_count:
                sampling = dataclasses.replace(
                    sampling, offset_changes_px=float(np.sqrt(offset_change_sq / (2.0 * offset_count))))
            if update_due(iteration, config):
                if sampling.updates_without_resolution_gain >= STALLED_UPDATES:
                    acc_rot, acc_trans, per_class = expected_errors(
                        dataset, state, config, state.membership, radius=radius,
                        rng=np.random.default_rng(int(config.seed) + 7919 * iteration))
                    sampling, sampling_update = update_sampling(
                        sampling, config, acc_rot_deg=acc_rot, acc_trans_px=acc_trans, iteration=iteration)
                    sampling_update["acc_rot_deg"], sampling_update["acc_trans_px"] = acc_rot, acc_trans
                    sampling_update["per_class"] = per_class
                    logger.info("Auto-sampling: accuracy angles %.3f degrees, offsets %.3f px -> order %d, offset "
                                "step %.3f px, range %.3f px, local searches %s", acc_rot, acc_trans,
                                sampling.healpix_order, sampling.offset_step_px, sampling.offset_range_px,
                                sampling.local_searches)
                else:
                    sampling, sampling_update = update_sampling(sampling, config)
            state = dataclasses.replace(state, sampling=sampling)
        validate_state(state, config)
        record = dict(
            iteration=iteration, radius=radius, healpix_order=hp, particle_ids=selected,
            half_counts=[len(ids) for ids in halves], class_soft_counts=class_mass, class_prior=class_prior,
            log_likelihood=sum(s.log_likelihood for s in stats),
            likelihood_scope="minibatch/support/noise dependent; common Gaussian normalization omitted",
            noise=noise, offset_variance_px2=offset_variance, step=step, fudge=fudge,
            vdam_step_factor=step_factor, direction_prior=direction_prior,
            class_diagnostics=class_diagnostics,
            pmax_mean=sum(s.pmax_sum for s in stats) / count,
            posterior=[[p.record() for p in s.posterior] for s in stats],
            pass2_mass_floor=config.pass2_mass_floor,
            gemm_precision=sorted({p for s in stats for item in s.posterior for p in item.precisions}),
            metric=metrics, dtype=str(theta.dtype), elapsed_seconds=time.monotonic() - started,
            engine="mixture_full_row_adaptive_oversampling" if oversampled_update(config, iteration)
            else "mixture_full_row_device_resident",
            batch_draw="tomogram" if config.tomogram_batches and isinstance(dataset, TiltParticles) else "random",
            local_searches=bool(local), local_search_sigma_deg=sigma_deg,
            candidate_rows_mean=(float(np.mean([c for c in candidate_counts if c >= 0]))
                                 if any(c >= 0 for c in candidate_counts) else None),
            coarse_healpix_order=int(hp), coarse_shift_step_px=float(grid.translation_step),
            n_coarse_poses=int(len(grid.rotations) * len(grid.translations)),
            sampling=None if state.sampling is None else state.sampling.to_json(),
            sampling_update=sampling_update,
            **({"oversampling": _oversampling_record(stats)} if oversampled_update(config, iteration) else {}),
        )
        with open(output / "iterations.jsonl", "a") as stream:
            stream.write(json.dumps(record, default=_json) + "\n")
        logger.info("Mixture update %d/%d: class mass %s; %.1f s", iteration, config.iterations,
                    np.round(class_mass, 2), record["elapsed_seconds"])
        if iteration % config.checkpoint_interval == 0 or iteration == end:
            checkpoint.save(output / f"checkpoint_{iteration:04d}.npz", state, config, identity)
        if stop_file is not None and Path(stop_file).exists():
            checkpoint.save(output / f"checkpoint_{iteration:04d}.npz", state, config, identity)
            break
    if state.iteration == config.iterations:
        export_models(output, state, dataset)
        if not config.skip_final_embeddings:
            final = pose_grid(dataset, config, config.iterations, sampling_state=state.sampling)
            final_children = (pose_grid(dataset, config, config.iterations, children=True, sampling_state=state.sampling)
                              if oversampled_update(config, config.iterations) else None)
            # The final inference pass searches as the last update did: locally around every stored pose.
            last = dataclasses.replace(state, iteration=config.iterations - 1)
            local, sigma_deg, local_range = local_search_policy(config, last, final)
            candidates = (candidate_source(dataset, state.membership, LocalGrid.of(final), sigma_deg=sigma_deg,
                                           offset_range_px=local_range, counts=[]) if local else None)
            export_assignments(output, dataset, state, config, final, diameter_ang=diameter_ang,
                               child_grid=final_children, candidates=candidates)
    return state
