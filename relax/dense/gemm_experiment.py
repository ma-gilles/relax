"""Opt-in K1 dense resident GEMM EM experiment.

One compiled batch holds its images while it streams fixed rotation and
translation tiles.  Exact mode scores the complete grid twice.  Lagged mode
scores it once against a previous-iteration normalizer, after an exact
bootstrap.  The caller keeps the reference and accumulators on device across
batches and swaps the two ID-indexed normalizer tables only after an iteration.
See ``docs/math/dense_gemm_experiment.md`` for the objective and update equations.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from relax.dense.gemm_experiment_kernels import (
    cc_score_tile,
    class_score_tile,
    empty_normalizer_table,
    merge_normalizers,
    normalizer_logz,
    score_model_power,
    score_tile,
    tile_normalizer,
    weighted_denominator_slices,
    weighted_numerator_slices,
)


class DenseGemmBatch(NamedTuple):
    """Fixed-capacity, already-preprocessed image batch and stable IDs."""

    score_image: jax.Array  # [B, Ps], unshifted corrected image, complex64
    score_weight: jax.Array  # [B, Ps], exact coarse pixel weight, float32
    initial_diff2: jax.Array  # [B], absolute out-of-window score term
    rec_image: jax.Array  # [B, Pr], native M-step image, complex64
    rec_weight: jax.Array  # [B, Pr], native M-step CTF²/noise, float32
    rec_raw_image: jax.Array  # [B, Pr], unweighted BPref image for native translation control
    rec_weighted_ctf: jax.Array  # [B, Pr], BPref CTF/noise applied after native translation
    rotation_prior: jax.Array  # [B, padded R]
    translation_prior: jax.Array  # [B, padded T]
    particle_ids: jax.Array  # [B], padding uses table's final sentinel index
    valid_images: jax.Array  # [B], bool


class DenseGemmGrid(NamedTuple):
    """Padded grid; padding never contributes posterior mass."""

    score_rotations: jax.Array  # [padded R, 3, 3]
    backprojection_rotations: jax.Array  # [padded R, 3, 3]
    score_phase: jax.Array  # [padded T, Ps]
    rec_phase: jax.Array  # [padded T, Pr]
    valid_rotations: jax.Array  # [padded R]
    valid_translations: jax.Array  # [padded T]


class DenseGemmBatchResult(NamedTuple):
    numerator: jax.Array
    denominator: jax.Array
    next_logz_table: jax.Array
    next_pair_table: jax.Array
    batch_logz: jax.Array
    batch_mass: jax.Array
    invalid_normalizer: jax.Array
    invalid_weight: jax.Array


class DenseGemmIterationResult(NamedTuple):
    """Device-resident accumulators and ID-indexed diagnostics after all batches."""

    numerator: jax.Array
    denominator: jax.Array
    next_logz_table: jax.Array
    next_pair_table: jax.Array
    mass_table: jax.Array
    invalid_normalizer_table: jax.Array
    invalid_weight_table: jax.Array


class JointDenseGemmBatchResult(NamedTuple):
    """Exact full-grid class/pose results for one fixed image batch.

    The class axis of ``numerator`` and ``denominator`` precedes the optional
    reconstruction-group axis. All score fields retain the padded image axis;
    callers publish only rows where ``batch.valid_images`` is true.
    """

    numerator: jax.Array  # [K, G, V]
    denominator: jax.Array  # [K, G, V]
    joint_normalizer: jax.Array  # [B, 2], (maximum, log shifted sum)
    class_normalizers: jax.Array  # [K, B, 2]
    class_best_scores: jax.Array  # [K, B]
    class_best_pose_ids: jax.Array  # [K, B], rotation * n_translations + translation
    rotation_posterior_sums: jax.Array  # [K, R], padded rotations have zero mass
    class_posterior_sums: jax.Array  # [K]
    class_translation_marginals: jax.Array  # [K, B, T], padded translations zero
    image_posterior_mass: jax.Array  # [B]
    invalid_normalizer: jax.Array  # [B]
    invalid_weight: jax.Array  # [B]
    statistics_state: object = None  # optional bounded resident row-stat carry


@dataclass(frozen=True)
class DenseGemmTileConfig:
    image_capacity: int
    rotation_tile: int
    translation_tile: int
    translation_side: str
    mode: str

    def __post_init__(self):
        if min(self.image_capacity, self.rotation_tile, self.translation_tile) <= 0:
            raise ValueError("B, Q and U must be positive")
        if self.translation_side not in {"image", "projection"}:
            raise ValueError("translation_side must be 'image' or 'projection'")
        if self.mode not in {"exact", "lagged"}:
            raise ValueError("mode must be 'exact' or 'lagged'")


def make_batch_program(
    config: DenseGemmTileConfig,
    project: Callable[[jax.Array, jax.Array], jax.Array],
    backproject: Callable[[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], tuple[jax.Array, jax.Array]],
):
    """Compile one resident batch with device-side candidate loops.

    ``project(reference, score_rotations)`` returns complex64 score pixels.
    ``backproject(y, w, summed_y, summed_w, bp_rotations)`` updates flat
    complex64/float32 volume accumulators.  Both callbacks are traced into
    the same JAX program; the production adapter below calls existing CUDA
    projector and fused x-half backprojection targets.
    """
    qsize = config.rotation_tile
    usize = config.translation_tile
    side = config.translation_side
    exact = config.mode == "exact"

    @partial(jax.jit, donate_argnames=("numerator", "denominator", "next_pair_table"))
    def run(
        reference, numerator, denominator, old_pair_table, next_pair_table, batch: DenseGemmBatch, grid: DenseGemmGrid
    ):
        nrot = grid.score_rotations.shape[0] // qsize
        ntrans = grid.score_phase.shape[0] // usize
        old_pair = old_pair_table[batch.particle_ids]

        def score_for(projection, model_power, rot_start, trans_start):
            phase = jax.lax.dynamic_slice_in_dim(grid.score_phase, trans_start, usize, axis=0)
            rprior = jax.lax.dynamic_slice_in_dim(batch.rotation_prior, rot_start, qsize, axis=1)
            tprior = jax.lax.dynamic_slice_in_dim(batch.translation_prior, trans_start, usize, axis=1)
            valid_rot = jax.lax.dynamic_slice_in_dim(grid.valid_rotations, rot_start, qsize, axis=0)
            valid_trans = jax.lax.dynamic_slice_in_dim(grid.valid_translations, trans_start, usize, axis=0)
            return score_tile(
                projection,
                batch.score_image,
                batch.score_weight,
                batch.initial_diff2,
                phase,
                rprior,
                tprior,
                batch.valid_images,
                valid_rot,
                valid_trans,
                translation_side=side,
                model_power=model_power,
            )

        def logz_rotation(rot_index, pair):
            rot_start = rot_index * qsize
            rotations = jax.lax.dynamic_slice_in_dim(grid.score_rotations, rot_start, qsize, axis=0)
            projection = project(reference, rotations)
            model_power = score_model_power(projection, batch.score_weight)

            def logz_translation(trans_index, current):
                scores = score_for(projection, model_power, rot_start, trans_index * usize)
                return merge_normalizers(current, tile_normalizer(scores))

            return jax.lax.fori_loop(0, ntrans, logz_translation, pair)

        if exact:
            normalizer = jax.lax.fori_loop(0, nrot, logz_rotation, empty_normalizer_table(config.image_capacity))
        else:
            normalizer = old_pair

        initial_pair = empty_normalizer_table(config.image_capacity)
        initial_mass = jnp.zeros((config.image_capacity,), dtype=jnp.float32)

        def reconstruction_rotation(rot_index, state):
            y_volume, w_volume, new_pair, mass = state
            rot_start = rot_index * qsize
            rotations = jax.lax.dynamic_slice_in_dim(grid.score_rotations, rot_start, qsize, axis=0)
            bp_rotations = jax.lax.dynamic_slice_in_dim(grid.backprojection_rotations, rot_start, qsize, axis=0)
            projection = project(reference, rotations)
            model_power = score_model_power(projection, batch.score_weight)
            y_slices = jnp.zeros((qsize, batch.rec_image.shape[1]), dtype=jnp.complex64)
            rotation_mass = jnp.zeros((config.image_capacity, qsize), dtype=jnp.float32)

            def reconstruction_translation(trans_index, tile_state):
                y_sum, summed_rotation_mass, pair, q_mass = tile_state
                trans_start = trans_index * usize
                scores = score_for(projection, model_power, rot_start, trans_start)
                if not exact:
                    pair = merge_normalizers(pair, tile_normalizer(scores))
                weights = jnp.exp(
                    (scores - normalizer[:, 0, None, None]) - normalizer[:, 1, None, None]
                )
                weights = jnp.where(batch.valid_images[:, None, None], weights, 0.0)
                rec_phase = jax.lax.dynamic_slice_in_dim(grid.rec_phase, trans_start, usize, axis=0)
                dy = weighted_numerator_slices(weights, batch.rec_image, rec_phase, translation_side=side)
                tile_rotation_mass = jnp.sum(weights, axis=2)
                return (
                    y_sum + dy,
                    summed_rotation_mass + tile_rotation_mass,
                    pair,
                    q_mass + jnp.sum(tile_rotation_mass, axis=1),
                )

            y_slices, rotation_mass, new_pair, mass = jax.lax.fori_loop(
                0,
                ntrans,
                reconstruction_translation,
                (y_slices, rotation_mass, new_pair, mass),
            )
            w_slices = weighted_denominator_slices(rotation_mass, batch.rec_weight)
            y_volume, w_volume = backproject(y_volume, w_volume, y_slices, w_slices, bp_rotations)
            return y_volume, w_volume, new_pair, mass

        numerator, denominator, calculated_pair, mass = jax.lax.fori_loop(
            0,
            nrot,
            reconstruction_rotation,
            (numerator, denominator, initial_pair, initial_mass),
        )
        batch_pair = normalizer if exact else calculated_pair
        batch_logz = normalizer_logz(batch_pair)
        invalid_normalizer = batch.valid_images & (
            ~jnp.isfinite(batch_pair).all(axis=1) | (~jnp.isfinite(old_pair).all(axis=1) if not exact else False)
        )
        invalid_weight = batch.valid_images & (~jnp.isfinite(mass) | (mass <= 0))
        # The sentinel occupies the final table row; padded lanes only write
        # there.  Real particle IDs are unique within a batch.
        next_pair_table = next_pair_table.at[batch.particle_ids].set(batch_pair)
        next_logz_table = normalizer_logz(next_pair_table)
        return DenseGemmBatchResult(
            numerator,
            denominator,
            next_logz_table,
            next_pair_table,
            batch_logz,
            mass,
            invalid_normalizer,
            invalid_weight,
        )

    return run


def make_joint_k_batch_program(
    config: DenseGemmTileConfig,
    project: Callable[[jax.Array, jax.Array], jax.Array],
    backproject: Callable[[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], tuple[jax.Array, jax.Array]],
    *,
    score_mode: str = "gaussian",
    project_reconstruction: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
    mstep_subtract_ctf_projection: bool = False,
    statistics_step: Callable | None = None,
    statistics_finish_class: Callable | None = None,
    class_batch_scores: bool = False,
    cache_scores: bool = False,
    score_cache_max_bytes: int = 512 << 20,
):
    """Compile an exact dense K-class E/M batch with one joint normalizer.

    The class-specific rotation prior must already contain ``log(pdf_class)``.
    ``reconstruction_groups`` is a per-image 0-based group id; for VDAM its
    two values select pseudo-halfset BackProjectors without duplicating the
    E-step. By default the full grid is scored twice and no candidate is pruned.
    ``class_batch_scores`` combines K projections in first-sweep score GEMMs.
    ``cache_scores`` also retains those float32 scores, subject to a byte cap,
    and reuses them in the unchanged class-ordered reconstruction sweep.
    """
    if config.mode != "exact":
        raise ValueError("the production K-class route requires exact two-sweep normalization")
    if score_mode not in {"gaussian", "normalized_cc"}:
        raise ValueError(f"unsupported dense GEMM score mode {score_mode!r}")
    if (mstep_subtract_ctf_projection or statistics_step is not None) and project_reconstruction is None:
        raise ValueError("gradient BPref and row statistics require a reconstruction-window projector")
    if (statistics_step is None) != (statistics_finish_class is None):
        raise ValueError("statistics_step and statistics_finish_class must be supplied together")
    if cache_scores and score_cache_max_bytes <= 0:
        raise ValueError("score_cache_max_bytes must be positive")
    qsize = config.rotation_tile
    usize = config.translation_tile
    side = config.translation_side

    @partial(jax.jit, donate_argnames=("numerator", "denominator"))
    def run(
        references,
        numerator,
        denominator,
        batch: DenseGemmBatch,
        grid: DenseGemmGrid,
        class_rotation_prior,
        reconstruction_groups,
        statistics_state=None,
        statistics_operands=None,
    ):
        n_classes = references.shape[0]
        n_groups = numerator.shape[1]
        nrot = grid.score_rotations.shape[0] // qsize
        ntrans = grid.score_phase.shape[0] // usize
        if cache_scores:
            cache_bytes = (n_classes * config.image_capacity
                           * grid.score_rotations.shape[0] * grid.score_phase.shape[0] * 4)
            if cache_bytes > score_cache_max_bytes:
                raise MemoryError(
                    f"dense GEMM score cache needs {cache_bytes} bytes, "
                    f"above its {score_cache_max_bytes}-byte cap"
                )
        real_translations = jnp.sum(grid.valid_translations).astype(jnp.int32)
        group_mask = (reconstruction_groups[:, None] == jnp.arange(n_groups)[None, :]) & batch.valid_images[:, None]
        class_pairs = jnp.broadcast_to(
            empty_normalizer_table(config.image_capacity)[None, :, :], (n_classes, config.image_capacity, 2)
        )
        class_best = jnp.full((n_classes, config.image_capacity), -jnp.inf, jnp.float32)
        class_pose = jnp.full((n_classes, config.image_capacity), -1, jnp.int32)

        def score_for(projection, model_power, class_id, rot_start, trans_start):
            phase = jax.lax.dynamic_slice_in_dim(grid.score_phase, trans_start, usize, axis=0)
            prior = jax.lax.dynamic_slice_in_dim(class_rotation_prior[class_id], rot_start, qsize, axis=1)
            tprior = jax.lax.dynamic_slice_in_dim(batch.translation_prior, trans_start, usize, axis=1)
            valid_rot = jax.lax.dynamic_slice_in_dim(grid.valid_rotations, rot_start, qsize, axis=0)
            valid_trans = jax.lax.dynamic_slice_in_dim(grid.valid_translations, trans_start, usize, axis=0)
            if score_mode == "normalized_cc":
                return cc_score_tile(
                    projection, batch.score_image, batch.score_weight, phase,
                    batch.valid_images, valid_rot, valid_trans,
                    translation_side=side, model_power=model_power,
                )
            return score_tile(
                projection, batch.score_image, batch.score_weight, batch.initial_diff2,
                phase, prior, tprior, batch.valid_images, valid_rot, valid_trans,
                translation_side=side, model_power=model_power,
            )

        def first_class(class_id, state):
            pairs, best, pose = state

            def first_rotation(rot_index, rotation_state):
                current_pair, current_best, current_pose = rotation_state
                rot_start = rot_index * qsize
                rotations = jax.lax.dynamic_slice_in_dim(grid.score_rotations, rot_start, qsize, axis=0)
                projection = project(references[class_id], rotations)
                model_power = score_model_power(projection, batch.score_weight)

                def first_translation(trans_index, translation_state):
                    pair, best_score, best_pose = translation_state
                    trans_start = trans_index * usize
                    scores = score_for(projection, model_power, class_id, rot_start, trans_start)
                    pair = merge_normalizers(pair, tile_normalizer(scores))
                    flat = scores.reshape(config.image_capacity, qsize * usize)
                    winner = jnp.argmax(flat, axis=1)
                    maximum = jnp.take_along_axis(flat, winner[:, None], axis=1)[:, 0]
                    pose_id = ((rot_start + winner // usize) * real_translations + trans_start + winner % usize).astype(jnp.int32)
                    # The loop visits Q rotations for one U-translation tile
                    # before advancing to the next tile.  That differs from
                    # canonical class/rotation/translation order, so a tied
                    # score must explicitly keep the smaller global pose ID.
                    better = (maximum > best_score) | (
                        jnp.isfinite(maximum)
                        & (maximum == best_score)
                        & ((best_pose < 0) | (pose_id < best_pose))
                    )
                    return pair, jnp.where(better, maximum, best_score), jnp.where(better, pose_id, best_pose)

                return jax.lax.fori_loop(
                    0, ntrans, first_translation, (current_pair, current_best, current_pose)
                )

            class_pair, class_maximum, class_winner = jax.lax.fori_loop(
                0, nrot, first_rotation, (pairs[class_id], best[class_id], pose[class_id])
            )
            return (
                pairs.at[class_id].set(class_pair),
                best.at[class_id].set(class_maximum),
                pose.at[class_id].set(class_winner),
            )

        if class_batch_scores or cache_scores:
            def first_rotation_batched(rot_index, rotation_state):
                pairs, best, pose, cached = rotation_state
                rot_start = rot_index * qsize
                rotations = jax.lax.dynamic_slice_in_dim(grid.score_rotations, rot_start, qsize, axis=0)
                _, projections = jax.lax.scan(
                    lambda unused, reference: (unused, project(reference, rotations)),
                    None, references,
                )
                k, q, p = projections.shape
                model_power = score_model_power(projections.reshape(k * q, p), batch.score_weight)
                prior = jax.lax.dynamic_slice_in_dim(class_rotation_prior, rot_start, qsize, axis=2)
                valid_rot = jax.lax.dynamic_slice_in_dim(grid.valid_rotations, rot_start, qsize, axis=0)

                def first_translation_batched(trans_index, translation_state):
                    current_pairs, current_best, current_pose, current_cache = translation_state
                    trans_start = trans_index * usize
                    phase = jax.lax.dynamic_slice_in_dim(grid.score_phase, trans_start, usize, axis=0)
                    tprior = jax.lax.dynamic_slice_in_dim(batch.translation_prior, trans_start, usize, axis=1)
                    valid_trans = jax.lax.dynamic_slice_in_dim(grid.valid_translations, trans_start, usize, axis=0)
                    scores = class_score_tile(
                        projections, batch.score_image, batch.score_weight, batch.initial_diff2,
                        phase, prior, tprior, batch.valid_images, valid_rot, valid_trans,
                        translation_side=side, score_mode=score_mode, model_power=model_power,
                    )
                    if cache_scores:
                        current_cache = jax.lax.dynamic_update_slice(
                            current_cache, scores, (0, 0, rot_start, trans_start)
                        )
                    current_pairs = jax.vmap(merge_normalizers)(
                        current_pairs, jax.vmap(tile_normalizer)(scores)
                    )
                    flat = scores.reshape(n_classes, config.image_capacity, qsize * usize)
                    winner = jnp.argmax(flat, axis=2)
                    maximum = jnp.take_along_axis(flat, winner[:, :, None], axis=2)[:, :, 0]
                    pose_id = ((rot_start + winner // usize) * real_translations
                               + trans_start + winner % usize).astype(jnp.int32)
                    better = (maximum > current_best) | (
                        jnp.isfinite(maximum)
                        & (maximum == current_best)
                        & ((current_pose < 0) | (pose_id < current_pose))
                    )
                    return (
                        current_pairs,
                        jnp.where(better, maximum, current_best),
                        jnp.where(better, pose_id, current_pose),
                        current_cache,
                    )

                return jax.lax.fori_loop(
                    0, ntrans, first_translation_batched, (pairs, best, pose, cached)
                )

            score_cache = (
                jnp.full((n_classes, config.image_capacity, grid.score_rotations.shape[0],
                          grid.score_phase.shape[0]), -jnp.inf, jnp.float32)
                if cache_scores else jnp.empty((0,), jnp.float32)
            )
            class_pairs, class_best, class_pose, score_cache = jax.lax.fori_loop(
                0, nrot, first_rotation_batched, (class_pairs, class_best, class_pose, score_cache)
            )
        else:
            class_pairs, class_best, class_pose = jax.lax.fori_loop(
                0, n_classes, first_class, (class_pairs, class_best, class_pose)
            )
        joint_pair = jax.lax.fori_loop(
            0, n_classes, lambda k, pair: merge_normalizers(pair, class_pairs[k]),
            empty_normalizer_table(config.image_capacity),
        )
        safe_max = jnp.where(jnp.isfinite(joint_pair[:, 0]), joint_pair[:, 0], 0.0)
        safe_logsum = jnp.where(jnp.isfinite(joint_pair[:, 1]), joint_pair[:, 1], 0.0)
        joint_best_class = jnp.argmax(class_best, axis=0).astype(jnp.int32)
        joint_best_pose = jnp.take_along_axis(class_pose, joint_best_class[None, :], axis=0)[0]
        rotation_sums = jnp.zeros((n_classes, grid.score_rotations.shape[0]), jnp.float64)
        class_mass = jnp.zeros((n_classes,), jnp.float64)
        class_translation = jnp.zeros((n_classes, config.image_capacity, grid.score_phase.shape[0]), jnp.float32)
        image_mass = jnp.zeros((config.image_capacity,), jnp.float32)

        def second_class(class_id, state):
            y_by_class, w_by_class, rotation_totals, class_totals, translation_totals, image_totals, stat_state = state

            def second_rotation(rot_index, rotation_state):
                y_all, w_all, rot_all, class_all, trans_all, image_all, block_stats = rotation_state
                rot_start = rot_index * qsize
                rotations = jax.lax.dynamic_slice_in_dim(grid.score_rotations, rot_start, qsize, axis=0)
                bp_rotations = jax.lax.dynamic_slice_in_dim(grid.backprojection_rotations, rot_start, qsize, axis=0)
                projection = None if cache_scores else project(references[class_id], rotations)
                rec_projection = (
                    project_reconstruction(references[class_id], rotations)
                    if project_reconstruction is not None else None
                )
                model_power = None if cache_scores else score_model_power(projection, batch.score_weight)
                y_slices = jnp.zeros((n_groups, qsize, batch.rec_image.shape[1]), jnp.complex64)
                rotation_mass = jnp.zeros((n_groups, config.image_capacity, qsize), jnp.float32)

                def second_translation(trans_index, tile_state):
                    y_sum, mass_sum, trans_total, image_total, tile_stats = tile_state
                    trans_start = trans_index * usize
                    scores = (
                        jax.lax.dynamic_slice(
                            score_cache, (class_id, 0, rot_start, trans_start),
                            (1, config.image_capacity, qsize, usize),
                        )[0]
                        if cache_scores else score_for(projection, model_power, class_id, rot_start, trans_start)
                    )
                    if score_mode == "normalized_cc":
                        rot_ids = rot_start + jnp.arange(qsize, dtype=jnp.int32)
                        trans_ids = trans_start + jnp.arange(usize, dtype=jnp.int32)
                        tile_pose = rot_ids[:, None] * real_translations + trans_ids[None, :]
                        weights = (
                            (joint_best_class == class_id)[:, None, None]
                            & (tile_pose[None, :, :] == joint_best_pose[:, None, None])
                            & jnp.isfinite(scores)
                        ).astype(jnp.float32)
                    else:
                        weights = jnp.exp((scores - safe_max[:, None, None]) - safe_logsum[:, None, None])
                    weights = jnp.where(batch.valid_images[:, None, None], weights, 0.0)
                    phase = jax.lax.dynamic_slice_in_dim(grid.rec_phase, trans_start, usize, axis=0)

                    def add_group(group_id, group_state):
                        group_y, group_mass = group_state
                        q = jnp.where(group_mask[:, group_id, None, None], weights, 0.0)
                        dy = weighted_numerator_slices(q, batch.rec_image, phase, translation_side=side)
                        return (
                            group_y.at[group_id].add(dy),
                            group_mass.at[group_id].add(jnp.sum(q, axis=2)),
                        )

                    y_sum, mass_sum = jax.lax.fori_loop(0, n_groups, add_group, (y_sum, mass_sum))
                    class_trans = trans_total[class_id]
                    previous = jax.lax.dynamic_slice_in_dim(class_trans, trans_start, usize, axis=1)
                    class_trans = jax.lax.dynamic_update_slice_in_dim(
                        class_trans, previous + jnp.sum(weights, axis=1), trans_start, axis=1
                    )
                    trans_total = trans_total.at[class_id].set(class_trans)
                    if statistics_step is not None:
                        tile_stats = statistics_step(
                            tile_stats, statistics_operands, weights, rec_projection,
                            class_id, rot_start, trans_start, batch, grid,
                        )
                    return y_sum, mass_sum, trans_total, image_total + jnp.sum(weights, axis=(1, 2)), tile_stats

                y_slices, rotation_mass, trans_all, image_all, block_stats = jax.lax.fori_loop(
                    0, ntrans, second_translation, (y_slices, rotation_mass, trans_all, image_all, block_stats)
                )
                # RELION's residual uses the projected reference at the
                # scored fine rotation. Its BPref matrix may differ.

                def backproject_group(group_id, volumes):
                    y_volumes, w_volumes = volumes
                    w_slices = weighted_denominator_slices(rotation_mass[group_id], batch.rec_weight)
                    group_y_slices = y_slices[group_id]
                    if mstep_subtract_ctf_projection:
                        group_y_slices = group_y_slices - rec_projection * w_slices
                    new_y, new_w = backproject(
                        y_volumes[class_id, group_id], w_volumes[class_id, group_id],
                        group_y_slices, w_slices, bp_rotations,
                    )
                    return y_volumes.at[class_id, group_id].set(new_y), w_volumes.at[class_id, group_id].set(new_w)

                y_all, w_all = jax.lax.fori_loop(0, n_groups, backproject_group, (y_all, w_all))
                per_rotation = jnp.sum(rotation_mass, axis=(0, 1)).astype(jnp.float64)
                old_rotation = jax.lax.dynamic_slice(rot_all, (class_id, rot_start), (1, qsize))
                rot_all = jax.lax.dynamic_update_slice(
                    rot_all, old_rotation + per_rotation[None, :], (class_id, rot_start)
                )
                class_all = class_all.at[class_id].add(jnp.sum(per_rotation))
                return y_all, w_all, rot_all, class_all, trans_all, image_all, block_stats

            class_output = jax.lax.fori_loop(
                0, nrot, second_rotation,
                (y_by_class, w_by_class, rotation_totals, class_totals, translation_totals, image_totals, stat_state),
            )
            if statistics_finish_class is not None:
                class_output = (*class_output[:-1], statistics_finish_class(
                    class_output[-1], statistics_operands, class_id
                ))
            return class_output

        numerator, denominator, rotation_sums, class_mass, class_translation, image_mass, statistics_state = jax.lax.fori_loop(
            0, n_classes, second_class,
            (numerator, denominator, rotation_sums, class_mass, class_translation, image_mass, statistics_state),
        )
        invalid_pair = batch.valid_images & (~jnp.isfinite(joint_pair).all(axis=1))
        invalid_mass = batch.valid_images & (~jnp.isfinite(image_mass) | (image_mass <= 0))
        return JointDenseGemmBatchResult(
            numerator, denominator, joint_pair, class_pairs, class_best, class_pose,
            rotation_sums, class_mass, class_translation, image_mass, invalid_pair, invalid_mass, statistics_state,
        )

    return run


def run_resident_iteration(program, reference, batches, grid, old_pair_table, *, volume_size: int):
    """Dispatch fixed-capacity batches with one immutable reference.

    Images and metadata are staged before each compiled batch call.  The
    normalizers, accumulators and diagnostics stay on device; this function
    makes no scalar reads or score-dependent Python decisions.  The caller
    checks failure flags and swaps old/new tables at the iteration boundary.
    """
    if volume_size <= 0 or old_pair_table.ndim != 2 or old_pair_table.shape[1] != 2 or old_pair_table.shape[0] < 2:
        raise ValueError("volume_size and ID-indexed normalizer table must be positive")
    table_size = old_pair_table.shape[0]
    numerator = jnp.zeros((volume_size,), dtype=jnp.complex64)
    denominator = jnp.zeros((volume_size,), dtype=jnp.float32)
    next_table = empty_normalizer_table(table_size)
    mass_table = jnp.zeros((table_size,), dtype=jnp.float32)
    invalid_normalizer = jnp.zeros((table_size,), dtype=jnp.bool_)
    invalid_weight = jnp.zeros((table_size,), dtype=jnp.bool_)
    for batch in batches:
        result = program(reference, numerator, denominator, old_pair_table, next_table, batch, grid)
        numerator, denominator, next_table = result.numerator, result.denominator, result.next_pair_table
        mass_table = mass_table.at[batch.particle_ids].set(result.batch_mass)
        invalid_normalizer = invalid_normalizer.at[batch.particle_ids].set(result.invalid_normalizer)
        invalid_weight = invalid_weight.at[batch.particle_ids].set(result.invalid_weight)
    return DenseGemmIterationResult(
        numerator, denominator, normalizer_logz(next_table), next_table, mass_table, invalid_normalizer, invalid_weight
    )


def pad_grid(score_rotations, backprojection_rotations, score_phase, rec_phase, *, rotation_tile, translation_tile):
    """Pad fixed grid capacities without adding candidate mass."""
    nr, nt = len(score_rotations), len(score_phase)
    if nr <= 0 or nt <= 0:
        raise ValueError("the dense grid must have rotations and translations")
    if len(backprojection_rotations) != nr or len(rec_phase) != nt:
        raise ValueError("score and reconstruction grid counts must match")
    rpad = (-nr) % rotation_tile
    tpad = (-nt) % translation_tile
    return DenseGemmGrid(
        jnp.pad(jnp.asarray(score_rotations), ((0, rpad), (0, 0), (0, 0)), mode="edge"),
        jnp.pad(jnp.asarray(backprojection_rotations), ((0, rpad), (0, 0), (0, 0)), mode="edge"),
        jnp.pad(jnp.asarray(score_phase), ((0, tpad), (0, 0)), mode="edge"),
        jnp.pad(jnp.asarray(rec_phase), ((0, tpad), (0, 0)), mode="edge"),
        jnp.arange(nr + rpad) < nr,
        jnp.arange(nt + tpad) < nt,
    )


def pad_batch(
    score_image,
    score_weight,
    initial_diff2,
    rec_image,
    rec_weight,
    rotation_prior,
    translation_prior,
    particle_ids,
    *,
    image_capacity,
    grid: DenseGemmGrid,
    sentinel_id: int,
    rec_raw_image=None,
    rec_weighted_ctf=None,
):
    """Create one fixed batch; input IDs must name distinct real particles."""
    count = len(particle_ids)
    if count <= 0 or count > image_capacity:
        raise ValueError("batch count must be within its fixed image capacity")
    ids_host = np.asarray(particle_ids)
    if (
        ids_host.ndim != 1
        or len(np.unique(ids_host)) != count
        or np.any(ids_host < 0)
        or np.any(ids_host >= sentinel_id)
    ):
        raise ValueError("real particle IDs must be distinct and within the normalizer table")
    # The joint-class program supplies its class prior separately.  A None
    # placeholder avoids constructing an unused image-by-rotation table.
    rcount = 0 if rotation_prior is None else np.shape(rotation_prior)[1]
    tcount = np.shape(translation_prior)[1]
    if rcount > grid.score_rotations.shape[0] or tcount > grid.score_phase.shape[0]:
        raise ValueError("prior exceeds padded grid")
    padding = image_capacity - count
    if (rec_raw_image is None) != (rec_weighted_ctf is None):
        raise ValueError("raw BPref image and weighted CTF must be supplied together")
    if rec_raw_image is None:
        rec_raw_image = rec_image
        rec_weighted_ctf = np.ones(np.shape(rec_weight), dtype=np.float32)

    def pad_rows(value, dtype):
        array = jnp.asarray(value, dtype=dtype)
        if array.shape[0] != count:
            raise ValueError("batch arrays must have matching image counts")
        return jnp.pad(array, ((0, padding),) + ((0, 0),) * (array.ndim - 1))

    rprior = (jnp.zeros((1, 1), jnp.float32) if rotation_prior is None else jnp.pad(
        jnp.asarray(rotation_prior, jnp.float32), ((0, padding), (0, grid.score_rotations.shape[0] - rcount))
    ))
    tprior = jnp.pad(
        jnp.asarray(translation_prior, jnp.float32), ((0, padding), (0, grid.score_phase.shape[0] - tcount))
    )
    return DenseGemmBatch(
        pad_rows(score_image, jnp.complex64),
        pad_rows(score_weight, jnp.float32),
        pad_rows(initial_diff2, jnp.float32),
        pad_rows(rec_image, jnp.complex64),
        pad_rows(rec_weight, jnp.float32),
        pad_rows(rec_raw_image, jnp.complex64),
        pad_rows(rec_weighted_ctf, jnp.float32),
        rprior,
        tprior,
        jnp.pad(jnp.asarray(particle_ids, jnp.int32), ((0, padding),), constant_values=sentinel_id),
        jnp.arange(image_capacity) < count,
    )


def native_relion_callbacks(
    *,
    image_shape,
    score_indices,
    rec_indices,
    r_max,
    projector_output_size,
    volume_shape,
    backprojection_backend="fused",
    backprojection_r_max=None,
    capacity_texture=None,
    padding_factor=1,
):
    """Adapt the existing CUDA RELION projector and fused x-half adjoint."""
    from relax.cuda.kernels import relion_fused_x_half_backproject_indexed
    from relax.helpers.projection import _relion_projector_texture_enabled, compute_relion_projector_projections_block

    if backprojection_backend not in {"fused", "separate"}:
        raise ValueError("backprojection_backend must be 'fused' or 'separate'")

    score_indices = jnp.asarray(score_indices, jnp.int32)
    rec_indices = jnp.asarray(rec_indices, jnp.int32)

    def project(reference, rotations):
        if not _relion_projector_texture_enabled(reference, r_max=r_max, padding_factor=padding_factor, enabled=True):
            raise RuntimeError("dense GEMM experiment requires the native RELION CUDA texture projector")
        projected, _ = compute_relion_projector_projections_block(
            reference,
            rotations,
            image_shape,
            r_max=r_max,
            padding_factor=padding_factor,
            centered_rows=True,
            dense_scale=True,
            return_abs2=False,
            projector_output_size=projector_output_size,
            pixel_indices=score_indices,
            relion_texture_interp=True,
            capacity_texture=capacity_texture,
        )
        return projected.astype(jnp.complex64)

    def backproject(y_volume, w_volume, y_slices, w_slices, rotations):
        from relax.helpers.adjoint import ReferenceSphereClip, adjoint_slice_volume_windowed

        if isinstance(backprojection_r_max, ReferenceSphereClip):
            # The fused CUDA kernel has only a scalar radius.  The canonical
            # adjoint owns the image-grid clip and reference-grid upsampling.
            def one(slices, volume):
                return adjoint_slice_volume_windowed(
                    slices, rec_indices, rotations, volume, image_shape,
                    volume_shape, "linear_interp", True, True,
                    backprojection_r_max, True,
                )
            return one(y_slices, y_volume), one(w_slices, w_volume)
        if backprojection_backend == "separate":
            from recovar.cuda_backproject import backproject_indexed

            kwargs = dict(
                image_shape=image_shape,
                volume_shape=volume_shape,
                order=1,
                half_volume=True,
                half_image=True,
                max_r=float(r_max if backprojection_r_max is None else backprojection_r_max),
                relion_x_half=True,
            )
            return (
                backproject_indexed(y_volume, y_slices, rec_indices, rotations, **kwargs),
                backproject_indexed(w_volume, w_slices, rec_indices, rotations, **kwargs),
            )
        return relion_fused_x_half_backproject_indexed(
            y_volume,
            w_volume,
            y_slices,
            w_slices,
            rec_indices,
            rotations,
            image_shape,
            volume_shape,
            float(r_max if backprojection_r_max is None else backprojection_r_max),
        )

    return project, backproject


def native_phase_table(translation_angles, pixel_indices, image_shape):
    """Generate the exact CUDA float32 image-translation phase at chosen pixels."""
    from relax.cuda.kernels import relion_translate_score_f32

    indices = jnp.asarray(pixel_indices, dtype=jnp.int32)
    ones = jnp.ones((1, len(indices)), dtype=jnp.complex64)
    return relion_translate_score_f32(
        ones,
        jnp.asarray(translation_angles, jnp.float32),
        indices,
        image_shape,
    ).reshape(len(translation_angles), len(indices))


def leading_tile_bytes(*, image_capacity, rotation_tile, translation_tile, score_pixels, rec_pixels, side):
    """Explicit lower bound for live tile arrays; compiler/workspace is extra."""
    b, q, u, ps, pr = image_capacity, rotation_tile, translation_tile, score_pixels, rec_pixels
    if side not in {"image", "projection"}:
        raise ValueError(f"unknown translation side {side!r}")
    return {
        "translation_expansion": 8 * (b if side == "image" else q) * u * ps,
        "score_and_posterior": 2 * 4 * b * q * u,
        "numerator_slices": 8 * q * pr,
        "denominator_slices": 4 * q * pr,
        "projection_reconstruction_temporary": 8 * q * u * pr if side == "projection" else 0,
    }
