"""Single-class dense GEMM batch program and its helpers, used by the dense GEMM benchmark scripts.

Moved out of ``relax/dense/gemm_kernels.py``: no relax module calls them (the
production engine uses ``make_joint_k_batch_program``). The scripts
``benchmark_dense_gemm_em.py`` and ``benchmark_dense_gemm_trajectory.py`` and their
tests import them from here.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from relax.dense.gemm_kernels import (
    DenseGemmGrid,
    DenseGemmTileConfig,
    empty_normalizer_table,
    merge_normalizers,
    pad_batch,
    score_model_power,
    score_tile,
    tile_normalizer,
    weighted_denominator_slices,
    weighted_numerator_slices,
)


class ExperimentBatch(NamedTuple):
    """``DenseGemmBatch`` plus the per-image rotation prior the single-class programs read.

    The production joint-class program takes its class rotation prior
    separately, so only these experiment programs carry it on the batch.
    """

    score_image: jax.Array  # [B, Ps], unshifted corrected image, complex64
    score_weight: jax.Array  # [B, Ps], exact coarse pixel weight, float32
    initial_diff2: jax.Array  # [B], absolute out-of-window score term
    rec_image: jax.Array  # [B, Pr], native M-step image, complex64
    rec_weight: jax.Array  # [B, Pr], native M-step CTF²/noise, float32
    rec_raw_image: jax.Array  # [B, Pr], unweighted BPref image for native translation control
    rec_weighted_ctf: jax.Array  # [B, Pr], BPref CTF/noise applied after native translation
    translation_prior: jax.Array  # [B, padded T]
    particle_ids: jax.Array  # [B], padding uses table's final sentinel index
    valid_images: jax.Array  # [B], bool
    rotation_prior: jax.Array  # [B, padded R]


def pad_experiment_batch(
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
) -> ExperimentBatch:
    """Pad one fixed batch as ``relax.dense.gemm_kernels.pad_batch`` does, plus its rotation prior.

    The rotation prior is zero-padded to ``image_capacity`` rows and to the
    grid's padded rotation count.
    """
    rcount = np.shape(rotation_prior)[1]
    if rcount > grid.score_rotations.shape[0]:
        raise ValueError("prior exceeds padded grid")
    batch = pad_batch(
        score_image, score_weight, initial_diff2, rec_image, rec_weight, translation_prior, particle_ids,
        image_capacity=image_capacity, grid=grid, sentinel_id=sentinel_id,
        rec_raw_image=rec_raw_image, rec_weighted_ctf=rec_weighted_ctf,
    )
    padding = image_capacity - len(particle_ids)
    rprior = jnp.pad(
        jnp.asarray(rotation_prior, jnp.float32), ((0, padding), (0, grid.score_rotations.shape[0] - rcount))
    )
    return ExperimentBatch(**batch._asdict(), rotation_prior=rprior)


def weighted_slices(q, rec_image, rec_weight, rec_phase, *, translation_side: str):
    """Reduce all images/translations to one native-compatible slice per rotation."""
    numerator = weighted_numerator_slices(q, rec_image, rec_phase, translation_side=translation_side)
    denominator = weighted_denominator_slices(jnp.sum(q, axis=2), rec_weight)
    return numerator, denominator


def normalizer_logz(pair):
    """Collapsed absolute logZ is for reporting, never posterior weights."""
    return pair[..., 0] + pair[..., 1]


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
        reference, numerator, denominator, old_pair_table, next_pair_table, batch: ExperimentBatch, grid: DenseGemmGrid
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
