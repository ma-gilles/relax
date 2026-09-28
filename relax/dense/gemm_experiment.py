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
    rcount, tcount = np.shape(rotation_prior)[1], np.shape(translation_prior)[1]
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

    rprior = jnp.pad(
        jnp.asarray(rotation_prior, jnp.float32), ((0, padding), (0, grid.score_rotations.shape[0] - rcount))
    )
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
        if backprojection_backend == "separate":
            from recovar.cuda_backproject import backproject_indexed

            kwargs = dict(
                image_shape=image_shape,
                volume_shape=volume_shape,
                order=1,
                half_volume=True,
                half_image=True,
                max_r=float(r_max),
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
            float(r_max),
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
