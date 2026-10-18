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
    data_vs_prior_shell,
    expected_errors,
    relion_radius,
    track_resolution,
    update_due,
    update_sampling,
)
from relax.ppca_initial_class3d.diagnostics import MetricFailure, model_summary
from relax.ppca_initial_class3d.expectation import expectation_groups, mixture_tiles, oversampled_update, pose_grid
from relax.ppca_initial_class3d.local_search import LocalGrid, relion_local_sigma_deg, tile_candidates
from relax.ppca_initial_class3d.membership import Membership
from relax.ppca_initial_class3d.outputs import export_assignments, export_models
from relax.ppca_initial_class3d.pc_policy import (
    center_latents,
    class_active_pcs,
    class_pc_radii,
    close_inactive,
    close_tested_pc,
    evidence_gains,
    keep_band,
    keep_tested_pc,
    latent_moments,
    limit_pc_band,
    open_pcs,
    pc_band_test_updates,
    pc_test_updates,
    project_rigid_motions,
    truncate_pc_band,
    uncouple_mean,
    variance_shares,
)
from relax.ppca_initial_class3d.state import initialize_state, validate_state
from relax.ppca_initial_model.checkpoint import saved_source
from relax.ppca_initial_model.initialization import bandlimit_and_mask, support_mask
from relax.ppca_initial_model.iteration_loop import _select_halves as select_halves
from relax.ppca_initial_model.noise import update_noise
from relax.ppca_initial_model.tomo import TiltParticles
from relax.ppca_initial_model.update import Moments, coupled_direction, metric_floor, stochastic_update

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


def closed_pc_evidence(dataset, state, config, groups, iteration, grid, theta, *, radius, diameter_ang, candidates):
    """Each particle's log-evidence under ``theta`` (pc_policy's PC test: the update's model with the tested PC
    closed), scored as the update's E-step scored the batch, without its moments."""
    closed = dataclasses.replace(state, theta=theta)
    values = {}
    for _, _, tile in mixture_tiles(dataset, closed, config, groups, iteration, grid, radius=radius, moments=False,
                                    diameter_ang=diameter_ang, candidates=candidates):
        values.update(zip(np.asarray(tile.original_image_ids).tolist(),
                          np.asarray(tile.image_log_likelihood, np.float64).tolist()))
    return values


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


def update_models(state, stats, config, dataset, *, radius, support, shells, mask, step, fudge, fresh=(),
                  uncoupled=()):
    """Use responsibility-weighted normal equations without scaling twice by occupancy.

    ``fresh[k]`` lists class ``k``'s loading channels (1..q) opened in this update (pc_policy): their VDAM first
    moments start at this update's directions, as every channel's do in the run's first update. ``uncoupled[k]``
    (None or a frequency mask) marks where class ``k``'s loadings are held at zero (pc_policy's PC band): there
    its mean is solved alone.
    """
    models, histories, metrics, diagnostics = [], [], [], []
    fresh = [list(f) for f in fresh] if fresh else [[] for _ in range(len(state.theta))]
    uncoupled = list(uncoupled) if uncoupled else [None] * len(state.theta)
    for k, model in enumerate(state.theta):
        directions, coverage, class_metrics = [], [], []
        for half_index, half in enumerate(stats):
            component = half.components[k]
            lhs = (component.lhs_tri if uncoupled[k] is None
                   else uncouple_mean(component.lhs_tri, config.q, uncoupled[k]))
            try:
                direction, info = coupled_direction(lhs, component.residual_gradient, support,
                                                    floor=metric_floor(dataset.grid_size))
            except ValueError as error:
                raise MetricFailure(k, half_index, component, str(error)) from error
            directions.append(direction)
            covered = jnp.trace(unpack_tri_to_full(component.lhs_tri, config.q + 1), axis1=-2, axis2=-1) > 0
            coverage.append(covered & jnp.asarray(support))
            class_metrics.append(info)
        history = state.moments[k]
        if fresh[k]:
            history = Moments(
                jnp.asarray(history.first).at[..., fresh[k]].set(jnp.stack(directions)[..., fresh[k]]),
                history.second, history.initialized)
        updated, moments, update_info = stochastic_update(
            model, history, jnp.stack(directions), jnp.stack(coverage), shells,
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
        # RELION's current image size (auto_sampling.relion_radius) or the stage's radius.
        radius = (relion_radius(state.resolution_shell, dataset.grid_size) if config.adaptive_radius
                  else min(config.stage(iteration)[0], dataset.grid_size // 2 - 1))
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
        # Gradual PCs (pc_policy): closed loadings stay at zero; a PC opening in this update starts before the E-step.
        gradual = config.pc_start_fraction is not None
        if gradual and state.pc_caps is None:
            state.pc_caps = [int(config.q)] * config.n_classes
        caps = state.pc_caps if gradual else [int(config.q)] * config.n_classes
        active = class_active_pcs(config, iteration, caps)
        opened = class_active_pcs(config, iteration - 1, caps) if gradual else active
        if gradual:
            theta, moments = close_inactive(state.theta, state.moments, opened)
            theta = open_pcs(theta, config, previously_active=opened, active=active,
                             volume_shape=dataset.volume_shape, radius=radius, mask=mask)
            state = dataclasses.replace(state, theta=theta, moments=moments)
            for k, (before, after) in enumerate(zip(opened, active)):
                if after > before:
                    logger.info("Gradual PCs: update %d opens PC %d of class %d", iteration, after, k)
        # PC band (pc_policy): each class's loadings stay within its radius; a class with untested new shells may
        # be tested at the end of the stage.
        band = config.pc_band_test
        if band and state.pc_band_caps is None:
            state.pc_band_caps = [None] * config.n_classes
        pc_radii = class_pc_radii(state.pc_band_caps, config.n_classes, radius)
        local, sigma_deg, local_range = local_search_policy(config, state, grid)
        candidate_counts = []
        candidates = (candidate_source(dataset, state.membership, local_grid, sigma_deg=sigma_deg,
                                       offset_range_px=local_range, counts=candidate_counts) if local else None)
        stats = expectation_groups(dataset, state, config, halves, iteration, grid, radius=radius,
                                   diameter_ang=diameter_ang,
                                   candidates=candidates,
                                   child_grid=child_grid if oversampled_update(config, iteration) else None)
        latent_mean, latent_second = latent_moments(stats, int(config.q), radii <= radius)
        tested = pc_test_updates(config).get(iteration)
        tested_classes = [k for k in range(len(active)) if active[k] == tested]
        if tested_classes:
            # The PC test (pc_policy): the same batch, grid and candidate poses, scored with the tested PC closed.
            without = closed_pc_evidence(
                dataset, state, config, halves, iteration, grid,
                close_tested_pc(state.theta, tested_classes, tested, latent_mean), radius=radius,
                diameter_ang=diameter_ang,
                candidates=(candidate_source(dataset, state.membership, local_grid, sigma_deg=sigma_deg,
                                             offset_range_px=local_range, counts=[]) if local else None))
        band_test = pc_band_test_updates(config, dataset.grid_size).get(iteration)
        band_classes = ([k for k in range(config.n_classes) if active[k] > 0 and pc_radii[k] is None]
                        if band_test is not None else [])
        if band_classes:
            # The band test (pc_policy): the same batch, scored with these classes' loadings cut at the previous
            # stage's radius.
            band_without = closed_pc_evidence(
                dataset, state, config, halves, iteration, grid,
                truncate_pc_band(state.theta, band_classes, band_test[0], radii, latent_mean), radius=radius,
                diameter_ang=diameter_ang,
                candidates=(candidate_source(dataset, state.membership, local_grid, sigma_deg=sigma_deg,
                                             offset_range_px=local_range, counts=[]) if local else None))
        numerator = sum(s.residual_num for s in stats)
        denominator = sum(s.residual_den for s in stats)
        try:
            theta, moments, metrics, class_diagnostics = update_models(
                state, stats, config, dataset, radius=radius, support=radii <= radius, shells=shells,
                mask=mask, step=step, fudge=fudge,
                fresh=[range(1 + before, 1 + after) for before, after in zip(opened, active)],
                uncoupled=[None if r is None else radii > r for r in pc_radii],
            )
        except MetricFailure as error:
            checkpoint.save(output / f"failure_before_{iteration:04d}.npz", state, config, identity)
            np.savez(output / f"failure_metric_{iteration:04d}_class{error.class_index:03d}_half{error.half_index}.npz",
                     lhs_tri=np.asarray(error.component.lhs_tri),
                     residual_gradient=np.asarray(error.component.residual_gradient), selected=selected,
                     half_particle_ids=halves[error.half_index], support=radii <= radius)
            raise
        pc_record = {"active_pcs": list(active)}
        pc_record["latent_centers"] = [latent_mean[k, :active[k]] for k in range(len(active))]
        pc_record["variance_shares"] = [variance_shares(theta[k], latent_second[k], active[k])
                                        for k in range(len(active))]
        if gradual:
            theta, moments = close_inactive(theta, moments, active)
            if tested_classes:
                # The end of PC j's phase: a class keeps PC j only if it raised the batch's log-evidence enough;
                # otherwise it closes it again and opens no further PC (pc_policy).
                gains, mass = evidence_gains(stats, without, config.n_classes)
                if state.pc_first_gain is None:
                    state.pc_first_gain = [None] * config.n_classes
                tests, closing = [], []
                for k in tested_classes:
                    if tested == 1:
                        state.pc_first_gain[k] = float(gains[k])
                    keep = keep_tested_pc(tested, gains[k], state.pc_first_gain[k], mass[k])
                    tests.append({"class": k, "pc": tested, "gain": float(gains[k]),
                                  "pc1_gain": state.pc_first_gain[k], "class_mass": float(mass[k]),
                                  "kept": bool(keep)})
                    if not keep:
                        closing.append(k)
                        state.pc_caps[k] = tested - 1
                        active[k] = tested - 1
                    logger.info("Gradual PCs: class %d %s PC %d: log-evidence gain %.4g per particle, class mass %.1f",
                                k, "keeps" if keep else "closes", tested, gains[k], mass[k])
                # A closed PC leaves its share of the batch's average structure in the mean, as in the test.
                theta = close_tested_pc(theta, closing, tested, latent_mean)
                theta, moments = close_inactive(theta, moments, active)
                pc_record["pc_tests"] = tests
                pc_record["active_pcs_after_tests"] = list(active)
        if band_classes:
            # The end of a stage: a class keeps its loadings' new shells only if they raised the batch's
            # log-evidence; otherwise its loadings stay at the previous radius for the rest of the run (pc_policy).
            gains, mass = evidence_gains(stats, band_without, config.n_classes)
            band_tests, cut = [], []
            for k in band_classes:
                keep = keep_band(gains[k], mass[k])
                band_tests.append({"class": k, "from_radius": band_test[0], "to_radius": band_test[1],
                                   "gain": float(gains[k]), "class_mass": float(mass[k]), "kept": bool(keep)})
                if not keep:
                    cut.append(k)
                    state.pc_band_caps[k] = band_test[0]
                logger.info("PC band: class %d %s its loadings' shells %d-%d: log-evidence gain %.4g per particle, "
                            "class mass %.1f", k, "keeps" if keep else "cuts", band_test[0], band_test[1], gains[k],
                            mass[k])
            theta = truncate_pc_band(theta, cut, band_test[0], radii, latent_mean)
            pc_radii = class_pc_radii(state.pc_band_caps, config.n_classes, radius)
            pc_record["pc_band_tests"] = band_tests
        if config.center_latents:
            centers = np.zeros((len(active), int(config.q)), np.float32)
            for k in range(len(active)):
                centers[k, :active[k]] = latent_mean[k, :active[k]]
            theta = center_latents(theta, centers)
        if config.project_rigid_motions:
            theta, removed = project_rigid_motions(theta, active, volume_shape=dataset.volume_shape, radius=radius,
                                                   mask=mask)
            pc_record["rigid_motion_removed_fraction"] = removed
        if band:
            theta, moments = limit_pc_band(theta, moments, pc_radii, radii)
            pc_record["pc_radius"] = [radius if r is None else r for r in pc_radii]
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
        # RELION's current resolution: the E-step's references (the models before this update) and the first
        # pseudo-half's weight.
        shell = data_vs_prior_shell(state.theta, latent_mean,
                                    [jnp.asarray(c.lhs_tri)[:, 0] for c in stats[0].components], shells,
                                    fudge=fudge, radius=radius)
        state = dataclasses.replace(
            state, theta=theta, moments=moments, noise=noise, class_prior=class_prior, iteration=iteration,
            rng_state=rng.bit_generator.state, offset_variance=offset_variance, radius=radius,
            direction_prior=direction_prior, direction_order=hp,
        )
        if config.adaptive_radius:
            state = dataclasses.replace(state, resolution_shell=shell)
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
            sampling = track_resolution(state.sampling, shell)
            if offset_count:
                sampling = dataclasses.replace(
                    sampling, offset_changes_px=float(np.sqrt(offset_change_sq / (2.0 * offset_count))))
            if update_due(iteration, config):
                if sampling.updates_without_resolution_gain >= STALLED_UPDATES:
                    acc_rot, acc_trans, per_class = expected_errors(
                        dataset, state, config, state.membership, radius=radius,
                        rng=np.random.default_rng(int(config.seed) + 7919 * iteration), centers=latent_mean)
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
            resolution_shell=shell, coarse_healpix_order=int(hp), coarse_shift_step_px=float(grid.translation_step),
            n_coarse_poses=int(len(grid.rotations) * len(grid.translations)),
            sampling=None if state.sampling is None else state.sampling.to_json(),
            sampling_update=sampling_update, **pc_record,
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
